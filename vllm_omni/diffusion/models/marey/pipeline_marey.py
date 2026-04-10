# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Marey video generation pipeline for vllm_omni.

Orchestrates:
    - Text encoding (UL2-style T5 + optional CLIP vector conditioning)
    - Rectified flow diffusion sampling (via FlowUniPCMultistepScheduler)
    - VAE decoding
    - CFG parallel support
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, T5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.marey.marey_transformer import MareyTransformer
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / model construction helpers
# ---------------------------------------------------------------------------


def load_transformer_config(
    model_path: str,
    subfolder: str = "transformer",
    local_files_only: bool = True,
) -> dict:
    """Load transformer config from model directory."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=model_path, filename=f"{subfolder}/config.json")
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(config: dict) -> MareyTransformer:
    """Build a MareyTransformer from a config dict."""
    field_map = {
        "in_channels": "in_channels",
        "out_channels": "out_channels",
        "hidden_size": "hidden_size",
        "depth": "depth",
        "depth_single_blocks": "depth_single_blocks",
        "num_heads": "num_heads",
        "mlp_ratio": "mlp_ratio",
        "patch_size": "patch_size",
        "caption_channels": "caption_channels",
        "model_max_length": "model_max_length",
        "vector_cond_channels": "vector_cond_channels",
        "qk_norm": "qk_norm",
        "rope_channels_ratio": "rope_channels_ratio",
        "rope_dim": "rope_dim",
        "add_pos_embed_at_every_block": "add_pos_embed_at_every_block",
        "input_sq_size": "input_sq_size",
        "class_dropout_prob": "class_dropout_prob",
        "num_kv_heads": "num_kv_heads",
        "learned_pe": "learned_pe",
    }
    kwargs = {}
    for config_key, param_name in field_map.items():
        if config_key in config:
            val = config[config_key]
            if config_key == "patch_size" and isinstance(val, list):
                val = tuple(val)
            kwargs[param_name] = val
    return MareyTransformer(**kwargs)


# ---------------------------------------------------------------------------
# Post / pre process functions
# ---------------------------------------------------------------------------


def get_marey_post_process_func(od_config: OmniDiffusionConfig):
    """Return a function that converts decoded latents to output frames."""
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_marey_pre_process_func(od_config: OmniDiffusionConfig):
    """Optional pre-process for I2V: load and resize input image."""
    import numpy as np
    import PIL.Image
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}
            if raw_image is None:
                continue
            image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image
            if request.sampling_params.height is None or request.sampling_params.width is None:
                max_area = 720 * 1280
                aspect_ratio = image.height / image.width
                mod_value = 16
                height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
                if request.sampling_params.height is None:
                    request.sampling_params.height = height
                if request.sampling_params.width is None:
                    request.sampling_params.width = width
            image = image.resize(
                (request.sampling_params.width, request.sampling_params.height),
                PIL.Image.Resampling.LANCZOS,
            )
            prompt["multi_modal_data"]["image"] = image
            prompt["additional_information"]["preprocessed_image"] = video_processor.preprocess(
                image, height=request.sampling_params.height, width=request.sampling_params.width
            )
            request.prompts[i] = prompt
        return request

    return pre_process_func


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class MareyPipeline(nn.Module, CFGParallelMixin, ProgressBarMixin):
    """Marey MMDiT video generation pipeline.

    Follows the Wan22Pipeline pattern:
        __init__  -> load text encoder, VAE, transformer, scheduler
        forward   -> encode prompt, prepare latents, denoise loop, decode
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Weight sources for DiffusersPipelineLoader
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Text encoder
        text_encoder_subfolder = "text_encoder"
        if local_files_only and not os.path.exists(os.path.join(model, text_encoder_subfolder)):
            text_encoder_subfolder = "text_encoder_1"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            model, subfolder=text_encoder_subfolder, torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)

        # VAE
        self._load_vae(model, dtype, local_files_only)

        # Transformer (weights loaded later via load_weights)
        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_transformer_from_config(transformer_config)

        # Scheduler: rectified flow matching
        flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 3.0
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )

        self.vae_scale_factor_temporal = getattr(self.vae.config, "scale_factor_temporal", 4) if self.vae else 4
        self.vae_scale_factor_spatial = getattr(self.vae.config, "scale_factor_spatial", 8) if self.vae else 8

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

    def _load_vae(self, model: str, dtype: torch.dtype, local_files_only: bool) -> None:
        """Load VAE, trying diffusers first, then falling back."""
        try:
            from diffusers import AutoencoderKLWan

            self.vae = AutoencoderKLWan.from_pretrained(
                model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_files_only
            ).to(self.device)
        except Exception:
            try:
                from diffusers import AutoencoderKL

                self.vae = AutoencoderKL.from_pretrained(
                    model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_files_only
                ).to(self.device)
            except Exception:
                logger.warning("Could not load VAE from model directory. VAE decoding will not be available.")
                self.vae = None

    # -- Properties ----------------------------------------------------------

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    # -- Forward -------------------------------------------------------------

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        frame_num: int = 81,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        # Extract parameters from request
        if len(req.prompts) > 1:
            raise ValueError("This model only supports a single prompt per request.")
        if len(req.prompts) == 1:
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames if req.sampling_params.num_frames else frame_num

        patch_size = self.transformer.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        self._guidance_scale = guidance_scale

        device = self.device
        dtype = self.transformer.dtype

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        # Text encoding
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                max_sequence_length=req.sampling_params.max_sequence_length or 512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        # Timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # Prepare latents
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=req.sampling_params.latents,
        )

        # Prepare conditioning tensors
        height_tensor = torch.tensor([height], device=device, dtype=dtype)
        width_tensor = torch.tensor([width], device=device, dtype=dtype)
        fps_value = req.sampling_params.fps if hasattr(req.sampling_params, "fps") and req.sampling_params.fps else 24
        fps_tensor = torch.tensor([fps_value], device=device, dtype=dtype)

        # Denoising loop
        with self.progress_bar(total=len(timesteps)) as pbar:
            for t in timesteps:
                self._current_timestep = t
                latent_model_input = latents.to(dtype)
                timestep = t.expand(latents.shape[0])

                do_true_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None
                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_hidden_states_mask": None,
                    "height": height_tensor,
                    "width": width_tensor,
                    "fps": fps_tensor,
                    "return_dict": False,
                }
                if do_true_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "encoder_hidden_states_mask": None,
                        "height": height_tensor,
                        "width": width_tensor,
                        "fps": fps_tensor,
                        "return_dict": False,
                    }
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                pbar.update()

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None

        # VAE decode
        if output_type == "latent" or self.vae is None:
            output = latents
        else:
            latents_for_decode = latents.to(self.vae.dtype)
            if hasattr(self.vae.config, "latents_mean"):
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents_for_decode.device, latents_for_decode.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents_for_decode.device, latents_for_decode.dtype)
                latents_for_decode = latents_for_decode / latents_std + latents_mean
            output = self.vae.decode(latents_for_decode, return_dict=False)[0]

        return DiffusionOutput(output=output)

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        return self.transformer(**kwargs)[0]

    # -- Text encoding -------------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # Trim to actual lengths then re-pad
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )

        return prompt_embeds, negative_prompt_embeds

    # -- Latent preparation --------------------------------------------------

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # -- Weight loading ------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
