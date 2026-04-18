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

    MareyTextEncoder emits a flat dict of per-request tensors under
    ``multimodal_outputs`` (one key per tensor, e.g. ``prompt_embeds_ul2``).
    The vLLM output processor forwards the single active request's
    payload on ``output.multimodal_output``; here we reassemble the
    list/tensor contract that ``MareyDitPipeline.forward`` expects.
    """
    source_outputs = _get_source_outputs(stage_list, engine_input_source)

    built: list[dict[str, Any]] = []
    for source_output in source_outputs:
        if not getattr(source_output, "finished", True):
            continue

        output = source_output.outputs[0]
        mm = getattr(output, "multimodal_output", None)
        if not isinstance(mm, dict):
            raise RuntimeError(
                f"Missing/unexpected multimodal_output on stage-0 request "
                f"{getattr(source_output, 'request_id', '?')}: type={type(mm)}"
            )

        def _get(k: str):
            v = mm.get(k)
            # Some worker paths may have boxed the value inside a 1-length list.
            if isinstance(v, list) and len(v) == 1:
                return v[0]
            return v

        diffusion_prompt: dict[str, Any] = {
            "prompt": _pick_prompt_text(prompt) or "",
            "additional_information": {
                "prompt_embeds":     [_get("prompt_embeds_ul2"), _get("prompt_embeds_byt5")],
                "prompt_masks":      [_get("prompt_masks_ul2"), _get("prompt_masks_byt5")],
                "vector_cond":       _get("vector_cond"),
                "neg_prompt_embeds": [_get("neg_prompt_embeds_ul2"), _get("neg_prompt_embeds_byt5")],
                "neg_prompt_masks":  [_get("neg_prompt_masks_ul2"), _get("neg_prompt_masks_byt5")],
                "neg_vector_cond":   _get("neg_vector_cond"),
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
        # Stage 1 stashes the denoised latent + size metadata in
        # ``custom_output`` (populated from ``DiffusionOutput.custom_output``
        # by the diffusion engine). ``OmniRequestOutput.latents`` stays
        # unset because the engine routes ``DiffusionOutput.output`` into
        # ``.images`` for diffusion stages.
        custom = getattr(source_output, "_custom_output", None) or {}
        latents = custom.get("latents")
        if latents is None:
            # Fall back to whatever the engine left on the request output.
            latents = getattr(source_output, "latents", None)
        if latents is None:
            images = getattr(source_output, "images", None)
            if isinstance(images, list) and images:
                latents = images[0]
        if latents is None:
            raise RuntimeError(
                f"Stage-1 output for req {getattr(source_output, 'request_id', '?')} "
                "has no latents tensor."
            )

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
