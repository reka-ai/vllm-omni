from __future__ import annotations

from typing import Any


def _get_source_outputs(stage_list: list[Any], engine_input_source: list[int]) -> list[Any]:
    if not engine_input_source:
        raise ValueError("engine_input_source is empty")
    source_stage_id = engine_input_source[0]
    source = stage_list[source_stage_id].engine_outputs
    if source is None:
        raise RuntimeError(f"No engine_outputs on upstream stage {source_stage_id}")
    return list(source)


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return str(prompt.get("prompt", ""))
    return str(getattr(prompt, "prompt", ""))


def _prompt_info(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, dict):
        info = prompt.get("additional_information", {})
        if not isinstance(info, dict):
            raise ValueError("WLAM prompt additional_information must be a dict")
        return dict(info)
    return {}


def ar2diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    source_outputs = _get_source_outputs(stage_list, engine_input_source)
    built: list[dict[str, Any]] = []
    info = _prompt_info(prompt)
    for source_output in source_outputs:
        if not getattr(source_output, "finished", True):
            continue
        built.append({"prompt": _prompt_text(prompt), "additional_information": info})
    if not built:
        raise RuntimeError("ar2diffusion produced no outputs")
    return built


def diffusion2vae(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    source_outputs = _get_source_outputs(stage_list, engine_input_source)
    built: list[dict[str, Any]] = []
    for source_output in source_outputs:
        custom = source_output._custom_output or {}
        latents = custom.get("latents")
        if latents is None:
            raise RuntimeError(f"Stage output {getattr(source_output, 'request_id', '?')} has no latents")
        built.append(
            {
                "prompt": _prompt_text(prompt),
                "additional_information": {
                    "latents": latents,
                    "height": custom.get("height"),
                    "width": custom.get("width"),
                    "num_frames": custom.get("num_frames"),
                    "fps": custom.get("fps"),
                },
            }
        )
    if not built:
        raise RuntimeError("diffusion2vae produced no outputs")
    return built
