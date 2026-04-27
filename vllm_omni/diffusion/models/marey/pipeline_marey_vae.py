# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Marey VAE decoding stage.

Stage 2 of the staged Marey pipeline. Consumes the denoised latent tensor
produced by the DiT stage (delivered via ``req.prompts[0][
"additional_information"]["latents"]``) and returns the decoded video.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable

import torch
import yaml
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.marey.vae_loader import (
    build_marey_vae_sp_hooks,
    load_vae,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)


def _load_yaml_config(model_path: str) -> dict:
    config_path = os.path.join(model_path, "config.yaml")
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")


def _resolve_vae_config(od_config: OmniDiffusionConfig) -> dict:
    config = _load_yaml_config(od_config.model)
    vae_cfg = config.get("vae", {}) or {}
    cp_path = vae_cfg.get("cp_path", "")
    if cp_path and not os.path.isabs(cp_path):
        vae_cfg["cp_path"] = os.path.join(od_config.model, cp_path)
    if od_config.model_config and "vae" in od_config.model_config:
        vae_cfg.update(od_config.model_config["vae"])
    return vae_cfg


def get_marey_vae_post_process_func(od_config: OmniDiffusionConfig):
    """Video post-processing: decoded frames → numpy/tensor/etc."""
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


class MareyVaePipeline(nn.Module):
    """Stage-2 Marey VAE decoder pipeline."""

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)

        vae_cfg = _resolve_vae_config(od_config)

        self.vae, self.vae_init_error = load_vae(vae_cfg, "cpu", self.dtype)
        if self.vae is None:
            raise RuntimeError(f"MareyVaePipeline failed to load VAE: {self.vae_init_error}")
        self.vae.to(self.device)
        self.sp_shard, self.sp_gather = build_marey_vae_sp_hooks()

    def _dummy_latent(self, height: int, width: int, num_frames: int) -> torch.Tensor:
        t_ds = self.vae.model.get_downsample_factors(0)[0]
        s_ds = self.vae.model.get_downsample_factors(0)[1]
        channels = int(getattr(self.vae.model, "latent_embed_dim", 16))
        lat_t = max(1, (num_frames + t_ds - 1) // t_ds)
        lat_h = max(1, (height + s_ds - 1) // s_ds)
        lat_w = max(1, (width + s_ds - 1) // s_ds)
        return torch.zeros((1, channels, lat_t, lat_h, lat_w), device=self.device, dtype=self.dtype), height, width, num_frames

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        logger.info("[marey-timing] stage=vae forward start request_id=%s", req.request_id)
        _t_start = time.perf_counter()
        try:
            return self._forward_impl(req)
        finally:
            _elapsed = time.perf_counter() - _t_start
            logger.info("[marey-timing] stage=vae forward end request_id=%s elapsed=%.3fs", req.request_id, _elapsed)

    def _forward_impl(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        vae_outer_start_time = time.perf_counter()
        torch.cuda.empty_cache()
        logger.info(f'Time after empty cache: {time.perf_counter() - vae_outer_start_time:.4f} seconds')
        if len(req.prompts) != 1:
            raise ValueError("MareyVaePipeline only supports a single prompt per request.")

        raw_prompt = req.prompts[0]
        if isinstance(raw_prompt, str):
            raise ValueError(
                "MareyVaePipeline expects a dict prompt carrying latents in "
                "additional_information['latents']; got a raw string."
            )
        add_info = raw_prompt.get("additional_information") or {}

        # ``additional_information`` is the sole inter-stage channel — populated
        # by ``stage_input_processors.marey.diffusion2vae`` from Stage 1's
        # ``DiffusionOutput.custom_output``. The defaults below only apply to
        # warmup / dummy runs, where the diffusion engine's ``_dummy_run``
        # sends an ``OmniTextPrompt`` with no ``additional_information``.
        if "latents" not in add_info:
            logger.info("No latents in request, building dummy latent")
            z, height, width, num_frames_req = self._dummy_latent(720, 1280, 128)
        else:
            z = add_info["latents"]
            height = add_info["height"]
            width = add_info["width"]
            num_frames_req = add_info["num_frames"]

        logger.info(f'Time before to(device): {time.perf_counter() - vae_outer_start_time:.4f} seconds')

        z = z.to(device=self.device, dtype=self.dtype)

        logger.info(f'Time after to(device): {time.perf_counter() - vae_outer_start_time:.4f} seconds')

        sp = req.sampling_params
        output_type = sp.output_type or self.od_config.output_type or "np"
        if output_type == "latent":
            return DiffusionOutput(output=z)

        vae_t_ds = self.vae.model.get_downsample_factors(0)[0]
        num_latent_t = z.shape[2]
        num_pixel_frames = num_latent_t * vae_t_ds
        chunk = self.vae.cfg.frame_chunk_len
        if num_latent_t <= 1:
            num_pixel_frames = 1
        elif chunk is not None and num_pixel_frames % chunk != 0:
            num_pixel_frames = (num_pixel_frames // chunk) * chunk

        
        vae_inner_start_time = time.perf_counter()
        logger.info(f'Time before decode: {time.perf_counter() - vae_outer_start_time:.4f} seconds')
        with torch.no_grad():
            output = self.vae.decode(
                z,
                num_frames=num_pixel_frames,
                spatial_size=(int(height), int(width)),
                sp_shard=self.sp_shard,
                sp_gather=self.sp_gather,
            )
        torch.cuda.synchronize()
        vae_inner_end_time = time.perf_counter()
        logger.info(f'VAE inner decode completed in {vae_inner_end_time - vae_inner_start_time:.4f} seconds')
        if isinstance(output, tuple):
            output = output[0]
        return DiffusionOutput(output=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # VAE weights are loaded inside load_vae() via
        # wlam.models.vae.two_stage_vae.TwoStageVAE.load_from_checkpoint —
        # the ``weights`` iterator is irrelevant here. Return all parameter
        # names so the framework considers the load successful.
        _ = list(weights)
        return {name for name, _ in self.named_parameters()}
