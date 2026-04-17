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

    _MM_KEY = "marey_text_encoder_out"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        te_dict = _load_te_config_from_model_dir(self.model_path)
        self.te_cfg = TextEncoderConfig.from_te_dict(te_dict)

        dtype = getattr(vllm_config.model_config, "dtype", None) or torch.bfloat16
        self.dtype = dtype

        self._bundle: MareyTextEncoderBundle | None = None
        self._loaded = False

    # -- Lazy encoder load ---------------------------------------------------

    def _ensure_loaded(self) -> MareyTextEncoderBundle:
        if self._bundle is not None:
            return self._bundle
        device = self.vllm_config.device_config.device
        logger.info("Loading Marey text encoders (ul2/clip/byt5) on %s", device)
        bundle = MareyTextEncoderBundle(self.te_cfg, dtype=self.dtype)
        bundle.to_device(device)
        self._bundle = bundle
        self._loaded = True
        return bundle

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
        # Encoder weights load from HF on the first forward pass.
        _ = list(weights)
        return set()

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
        bundle = self._ensure_loaded()
        device = self.vllm_config.device_config.device

        additional_infos: list[dict[str, Any]] = list(runtime_additional_information or [])
        if not additional_infos:
            additional_infos = [{}]

        results: list[dict[str, Any]] = []
        for info in additional_infos:
            prompt_text = info.get("prompt_text") or info.get("prompt") or ""
            neg_text = info.get("negative_prompt_text") or info.get("negative_prompt")
            guidance_scale = info.get("guidance_scale")

            if not isinstance(prompt_text, str) or not prompt_text:
                raise ValueError(
                    "MareyTextEncoder requires a non-empty 'prompt_text' in "
                    "runtime_additional_information for each request."
                )

            # Reference Marey passes quote_override="" during inference so
            # ByT5 encodes an empty string when no quotes are detected.
            pos_embeds, pos_masks, pos_vec = bundle.encode_prompt(
                prompt_text, device=device, dtype=self.dtype, quote_override="",
            )

            use_cfg = False
            if guidance_scale is not None:
                try:
                    use_cfg = float(guidance_scale) > 1.0
                except Exception:
                    use_cfg = False
            if not use_cfg and neg_text is not None:
                # If a negative prompt was explicitly provided, always encode
                # it — the downstream DiT stage decides whether to apply CFG
                # based on its own guidance_scale logic.
                use_cfg = True

            neg_embeds = neg_masks = neg_vec = None
            if use_cfg:
                neg_prompt = neg_text if (isinstance(neg_text, str) and neg_text) else DEFAULT_NEGATIVE_PROMPT
                neg_embeds, neg_masks, neg_vec = bundle.encode_prompt(
                    neg_prompt, device=device, dtype=self.dtype, quote_override="",
                )

            results.append({
                "prompt_embeds":     [t.cpu() for t in pos_embeds],
                "prompt_masks":      [t.cpu() for t in pos_masks],
                "vector_cond":       pos_vec.cpu(),
                "neg_prompt_embeds": [t.cpu() for t in neg_embeds] if neg_embeds is not None else None,
                "neg_prompt_masks":  [t.cpu() for t in neg_masks] if neg_masks is not None else None,
                "neg_vector_cond":   neg_vec.cpu() if neg_vec is not None else None,
            })

        # ``multimodal_outputs`` is a dict of per-request lists; downstream
        # output processing iterates positionally across requests.
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={self._MM_KEY: results},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        raise TypeError(
            f"MareyTextEncoder expected an OmniOutput from forward(), got {type(model_outputs)}"
        )
