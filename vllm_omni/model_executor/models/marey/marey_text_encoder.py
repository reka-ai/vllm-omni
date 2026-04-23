# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Marey stage-0 text encoder model.

Packages the three Marey text encoders (UL2, CLIP, ByT5) into a single
vLLM-compatible ``nn.Module``. Runs under ``OmniGenerationScheduler`` with
``requires_raw_input_tokens = True`` — vLLM's token embedding path is
bypassed. The raw prompt (and negative prompt, for CFG) is passed through
``runtime_additional_information``; the model does its own tokenization
with three tokenizers and returns the six encoder tensors via
``OmniOutput.multimodal_outputs``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from typing import Any

import torch
import yaml
from torch import nn
from vllm.config import VllmConfig

from vllm_omni.diffusion.models.marey.text_encoders import (
    DEFAULT_NEGATIVE_PROMPT,
    MareyTextEncoderBundle,
    TextEncoderConfig,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)


def _load_te_config_from_model_dir(model_dir: str) -> dict:
    """Read text_encoder section from the Marey config.yaml."""
    cfg_path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        logger.warning("Marey config.yaml not found at %s; using TextEncoderConfig defaults", cfg_path)
        return {}
    with open(cfg_path) as f:
        full = yaml.safe_load(f) or {}
    return full.get("text_encoder", {}) or {}


def _shard_ul2_encoder(ul2_model: nn.Module, dtype: torch.dtype) -> None:
    """FSDP-shard UL2's encoder across the stage's worker group.

    Stage 0 does not initialise vllm_omni's ``_FS`` group, so we cannot
    use ``apply_hsdp_to_model`` directly. Instead we build a device mesh
    over the default torch.distributed world and call ``shard_model``.
    Each worker must already have its local CUDA device selected (vLLM
    does this in ``init_worker_distributed_environment``).
    """
    from torch.distributed import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from transformers.models.t5.modeling_t5 import T5Block

    from vllm_omni.diffusion.distributed.hsdp import shard_model

    world_size = torch.distributed.get_world_size()
    mesh = init_device_mesh(
        "cuda",
        mesh_shape=(1, world_size),
        mesh_dim_names=("replicate", "shard"),
    )
    logger.info("Mesh shape: %s, param_dtype: %s", mesh.shape, dtype)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )

    def _is_t5_block(name: str, module: nn.Module) -> bool:
        return isinstance(module, T5Block)

    shard_model(
        ul2_model.encoder,
        reshard_after_forward=True,
        mp_policy=mp_policy,
        mesh=mesh,
        hsdp_shard_conditions=[_is_t5_block],
    )
    logger.info(
        "FSDP-sharded UL2 encoder across %d ranks (world_size=%d)",
        world_size,
        world_size,
    )


class MareyTextEncoder(nn.Module):
    """Stage-0 Marey text encoder (UL2 + CLIP + ByT5).

    Produces encoder tensors for both the positive prompt and, when CFG is
    active, the negative prompt in a single forward pass. Outputs flow back
    through ``OmniOutput.multimodal_outputs`` and are picked up by
    ``stage_input_processors.marey.text2diffusion`` to build the next
    stage's prompt.
    """

    input_modalities = "text"
    have_multimodal_outputs = True
    requires_raw_input_tokens = True
    enable_update_additional_information = True
    has_preprocess = False
    has_postprocess = False

    # Per-request keys emitted in OmniOutput.multimodal_outputs. The
    # diffusion worker's multimodal output handler expects each dict value
    # to be a plain Tensor (or list[Tensor] with len == num_reqs), not
    # nested dicts — so we flatten the encoder result into one key per
    # tensor and let the stage input processor reassemble them.
    _MM_KEYS = (
        "prompt_embeds_ul2",
        "prompt_embeds_byt5",
        "prompt_masks_ul2",
        "prompt_masks_byt5",
        "vector_cond",
        "neg_prompt_embeds_ul2",
        "neg_prompt_embeds_byt5",
        "neg_prompt_masks_ul2",
        "neg_prompt_masks_byt5",
        "neg_vector_cond",
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        te_dict = _load_te_config_from_model_dir(self.model_path)
        self.te_cfg = TextEncoderConfig.from_te_dict(te_dict)

        dtype = getattr(vllm_config.model_config, "dtype", None) or torch.bfloat16
        self.dtype = dtype

        # Build the bundle eagerly. vLLM wraps model construction in
        # ``set_current_vllm_config(...)`` which any future TP layer would
        # need. Weights land on CPU (``device_map="cpu"``); we either shard
        # UL2 across all workers in this stage via FSDP or fall back to a
        # plain ``.to(device)``.
        device = vllm_config.device_config.device
        logger.info("Loading Marey text encoders (ul2/clip/byt5) on %s", device)
        self._bundle = MareyTextEncoderBundle(self.te_cfg, dtype=dtype)

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            _shard_ul2_encoder(self._bundle.ul2_model, dtype=dtype)
            # CLIP + ByT5 stay replicated per worker (small, not worth
            # sharding). Move them to the local GPU explicitly.
            # self._bundle.clip_model.to(device)
            self._bundle.clip_model.to(device)
            _shard_ul2_encoder(self._bundle.byt5_model, dtype=dtype)
            # self._bundle.byt5_model.to(device)
        else:
            self._bundle.to_device(device)

        self._loaded = True

    def _ensure_loaded(self) -> MareyTextEncoderBundle:
        return self._bundle

    # -- vLLM model interface ------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # Stage-0 ignores token embeddings. Keep a stable dummy embedding so
        # the vLLM runner can still micro-batch input_ids positions.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Bundle weights are loaded eagerly in __init__ from HF. Consume
        # anything vLLM's loader feeds us and report every bundle parameter
        # as initialised so the framework's coverage check passes.
        _ = list(weights)
        return {name for name, _ in self.named_parameters()}

    # -- Encoding forward ----------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        logger.info("[marey-timing] stage=text_encoder forward start")
        _t_start = time.perf_counter()
        try:
            return self._forward_impl(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                runtime_additional_information=runtime_additional_information,
                **kwargs,
            )
        finally:
            _elapsed = time.perf_counter() - _t_start
            logger.info("[marey-timing] stage=text_encoder forward end elapsed=%.3fs", _elapsed)

    def _forward_impl(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        bundle = self._ensure_loaded()
        device = self.vllm_config.device_config.device

        additional_infos: list[dict[str, Any]] = list(runtime_additional_information or [])
        if not additional_infos:
            additional_infos = [{}]

        # One list per key, filled positionally per request. The diffusion
        # worker expects multimodal_outputs[key] to be a Tensor or
        # list[Tensor] with len == num_reqs.
        by_key: dict[str, list[torch.Tensor]] = {k: [] for k in self._MM_KEYS}

        for info in additional_infos:
            prompt_text = info.get("prompt_text") or info.get("prompt") or ""
            neg_text = info.get("negative_prompt_text") or info.get("negative_prompt")

            # vLLM's profile / dummy-forward path calls us with an empty
            # runtime_additional_information entry. Emit zero placeholders
            # sized from the encoder configs — the worker only requires a
            # Tensor per key per request, and the downstream stage input
            # processor skips unfinished/dummy requests anyway.
            if not isinstance(prompt_text, str) or not prompt_text:
                self._append_dummy(by_key, device)
                continue

            # Reference Marey passes quote_override="" during inference so
            # ByT5 encodes an empty string when no quotes are detected.
            pos_embeds, pos_masks, pos_vec = bundle.encode_prompt(
                prompt_text, device=device, dtype=self.dtype, quote_override="",
            )
            # Always encode negative too; stage 1 decides whether to apply
            # CFG based on its own guidance_scale (avoids optional tensors
            # in the multimodal_outputs contract).
            neg_prompt = neg_text if (isinstance(neg_text, str) and neg_text) else DEFAULT_NEGATIVE_PROMPT
            neg_embeds, neg_masks, neg_vec = bundle.encode_prompt(
                neg_prompt, device=device, dtype=self.dtype, quote_override="",
            )

            by_key["prompt_embeds_ul2"].append(pos_embeds[0].cpu().contiguous())
            by_key["prompt_embeds_byt5"].append(pos_embeds[1].cpu().contiguous())
            by_key["prompt_masks_ul2"].append(pos_masks[0].cpu().contiguous())
            by_key["prompt_masks_byt5"].append(pos_masks[1].cpu().contiguous())
            by_key["vector_cond"].append(pos_vec.cpu().contiguous())
            by_key["neg_prompt_embeds_ul2"].append(neg_embeds[0].cpu().contiguous())
            by_key["neg_prompt_embeds_byt5"].append(neg_embeds[1].cpu().contiguous())
            by_key["neg_prompt_masks_ul2"].append(neg_masks[0].cpu().contiguous())
            by_key["neg_prompt_masks_byt5"].append(neg_masks[1].cpu().contiguous())
            by_key["neg_vector_cond"].append(neg_vec.cpu().contiguous())

        # ``multimodal_outputs`` is a dict of per-request lists; downstream
        # output processing iterates positionally across requests.
        torch.cuda.empty_cache()
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs=by_key,
        )

    def _append_dummy(self, by_key: dict[str, list[torch.Tensor]], device: torch.device) -> None:
        """Append zero-shaped placeholder tensors for a dummy/profile request."""
        cfg = self.te_cfg
        ul2_dim = 4096  # Matches google/ul2 d_model; exact value is irrelevant for dummy.
        byt5_dim = 1536
        clip_dim = 768
        dt = self.dtype
        zeros_ul2 = torch.zeros((1, cfg.ul2_max_length, ul2_dim), device=device, dtype=dt).cpu().contiguous()
        zeros_byt5 = torch.zeros((1, cfg.byt5_max_length, byt5_dim), device=device, dtype=dt).cpu().contiguous()
        ones_ul2_mask = torch.ones((1, cfg.ul2_max_length), device=device, dtype=torch.bool).cpu().contiguous()
        ones_byt5_mask = torch.ones((1, cfg.byt5_max_length), device=device, dtype=torch.bool).cpu().contiguous()
        zeros_clip = torch.zeros((1, clip_dim), device=device, dtype=dt).cpu().contiguous()
        by_key["prompt_embeds_ul2"].append(zeros_ul2)
        by_key["prompt_embeds_byt5"].append(zeros_byt5)
        by_key["prompt_masks_ul2"].append(ones_ul2_mask)
        by_key["prompt_masks_byt5"].append(ones_byt5_mask)
        by_key["vector_cond"].append(zeros_clip)
        by_key["neg_prompt_embeds_ul2"].append(zeros_ul2.clone())
        by_key["neg_prompt_embeds_byt5"].append(zeros_byt5.clone())
        by_key["neg_prompt_masks_ul2"].append(ones_ul2_mask.clone())
        by_key["neg_prompt_masks_byt5"].append(ones_byt5_mask.clone())
        by_key["neg_vector_cond"].append(zeros_clip.clone())

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        raise TypeError(
            f"MareyTextEncoder expected an OmniOutput from forward(), got {type(model_outputs)}"
        )
