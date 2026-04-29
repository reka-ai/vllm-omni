from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest


def _additional_information(req: OmniDiffusionRequest) -> dict[str, Any]:
    if not req.prompts:
        return {}
    prompt = req.prompts[0]
    if isinstance(prompt, dict):
        info = prompt.get("additional_information", {})
        if not isinstance(info, dict):
            raise ValueError("WLAM VAE prompt additional_information must be a dict")
        return info
    return {}


class WLAMVaePipeline(nn.Module):
    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        super().__init__()
        self.od_config = od_config

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        info = _additional_information(req)
        latents = info.get("latents")
        if latents is None:
            raise ValueError("WLAM VAE stage requires latents from the diffusion stage")
        if not isinstance(latents, torch.Tensor):
            latents = torch.tensor(latents)
        return DiffusionOutput(
            output=latents,
            custom_output={
                "latents": latents,
                "height": info.get("height"),
                "width": info.get("width"),
                "num_frames": info.get("num_frames"),
                "fps": info.get("fps"),
            },
        )
