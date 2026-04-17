# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage input processors for the staged Marey pipeline.

Two hooks, both invoked by the orchestrator when the *next* stage is a
diffusion stage (``_forward_to_next_stage`` in ``engine/orchestrator.py``):

- ``text2diffusion``:  stage 0 (LLM/text encoders) → stage 1 (DiT).
  Extracts the six encoder tensors from the stage-0 ``OmniOutput``
  multimodal payload and packs them into a dict prompt that
  ``MareyDitPipeline.forward`` consumes.

- ``diffusion2vae``:  stage 1 (DiT) → stage 2 (VAE).  Extracts the final
  denoised latent (plus spatial/temporal sizing) from the stage-1
  ``OmniRequestOutput`` and packs it into the prompt that
  ``MareyVaePipeline.forward`` consumes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Keep in sync with MareyTextEncoder._MM_KEY
_TEXT_ENC_MM_KEY = "marey_text_encoder_out"


def _get_source_outputs(stage_list: list[Any], engine_input_source: list[int]) -> list[Any]:
    if not engine_input_source:
        raise ValueError("engine_input_source is empty; cannot locate upstream stage")
    source_stage_id = engine_input_source[0]
    source = stage_list[source_stage_id].engine_outputs
    if source is None:
        raise RuntimeError(f"No engine_outputs on upstream stage {source_stage_id}")
    return list(source)


def _pick_prompt_text(prompt: Any) -> str | None:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return prompt.get("prompt")
    return getattr(prompt, "prompt", None)


def _pick_negative_prompt(prompt: Any) -> str | None:
    if isinstance(prompt, dict):
        return prompt.get("negative_prompt")
    return getattr(prompt, "negative_prompt", None)


# ---------------------------------------------------------------------------
# Stage 0 → Stage 1: text encoders → DiT
# ---------------------------------------------------------------------------


def text2diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Build the DiT stage prompt from the text-encoder stage output.

    The MareyTextEncoder returns an ``OmniOutput`` whose
    ``multimodal_outputs`` is ``{"marey_text_encoder_out": [dict_per_req]}``.
    vLLM's output processor unwraps the per-request dict onto each request's
    ``output.multimodal_output``.
    """
    source_outputs = _get_source_outputs(stage_list, engine_input_source)

    built: list[dict[str, Any]] = []
    for source_output in source_outputs:
        if not getattr(source_output, "finished", True):
            continue

        output = source_output.outputs[0]
        mm = getattr(output, "multimodal_output", None)
        if mm is None:
            raise RuntimeError(
                f"Missing multimodal_output on stage-0 request {getattr(source_output, 'request_id', '?')}"
            )

        enc_out = mm.get(_TEXT_ENC_MM_KEY) if isinstance(mm, dict) else None
        if enc_out is None:
            # Fallback: some runners may have already unwrapped by key.
            enc_out = mm
        if not isinstance(enc_out, dict):
            raise RuntimeError(
                f"Unexpected multimodal_output shape for stage-0 output: {type(enc_out)}"
            )

        diffusion_prompt: dict[str, Any] = {
            "prompt": _pick_prompt_text(prompt) or "",
            "additional_information": {
                "prompt_embeds":     enc_out.get("prompt_embeds"),
                "prompt_masks":      enc_out.get("prompt_masks"),
                "vector_cond":       enc_out.get("vector_cond"),
                "neg_prompt_embeds": enc_out.get("neg_prompt_embeds"),
                "neg_prompt_masks":  enc_out.get("neg_prompt_masks"),
                "neg_vector_cond":   enc_out.get("neg_vector_cond"),
            },
        }

        built.append(diffusion_prompt)

    if not built:
        raise RuntimeError("text2diffusion produced no outputs — stage 0 finished with no valid requests")
    return built


# ---------------------------------------------------------------------------
# Stage 1 → Stage 2: DiT → VAE
# ---------------------------------------------------------------------------


def diffusion2vae(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Build the VAE stage prompt from the DiT stage output.

    Stage 1 stores the final latent tensor on ``OmniRequestOutput.latents``
    and size metadata on ``_custom_output`` (populated from
    ``DiffusionOutput.custom_output`` by the diffusion engine).
    """
    source_outputs = _get_source_outputs(stage_list, engine_input_source)

    built: list[dict[str, Any]] = []
    for source_output in source_outputs:
        latents = getattr(source_output, "latents", None)
        if latents is None:
            # Some diffusion stage wrappers stash the tensor on `output`
            # instead; try that.
            latents = getattr(source_output, "output", None)
        if latents is None:
            raise RuntimeError(
                f"Stage-1 output for req {getattr(source_output, 'request_id', '?')} "
                "has no latents tensor."
            )

        custom = getattr(source_output, "_custom_output", None) or {}
        height = custom.get("height")
        width = custom.get("width")
        num_frames = custom.get("num_frames")
        fps = custom.get("fps")

        vae_prompt: dict[str, Any] = {
            "prompt": _pick_prompt_text(prompt) or "",
            "additional_information": {
                "latents": latents,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "fps": fps,
            },
        }

        built.append(vae_prompt)

    if not built:
        raise RuntimeError("diffusion2vae produced no outputs — stage 1 finished with no valid requests")
    return built
