from __future__ import annotations

import glob
import os
from types import SimpleNamespace
from typing import Any

import safetensors.torch
import torch
import yaml
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.models.wlam.common import WLAMModelArgs
from vllm_omni.model_executor.models.wlam.rope import timestep_embedding

from .transformer import WLAMDiffusionTransformer


def _torch_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype.removeprefix("torch."))
    raise TypeError(f"Unsupported dtype for WLAM diffusion: {dtype!r}")


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_model_config(od_config: OmniDiffusionConfig) -> Any:
    cfg: dict[str, Any] = {}
    if od_config.model:
        yaml_path = os.path.join(od_config.model, "config.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                loaded = yaml.safe_load(f) or {}
            cfg = loaded.get("model", loaded)
        else:
            try:
                from transformers import AutoConfig

                return AutoConfig.from_pretrained(
                    od_config.model,
                    trust_remote_code=od_config.trust_remote_code,
                )
            except Exception:
                cfg = {}
    if od_config.model_config:
        _deep_update(cfg, od_config.model_config)
    return _to_namespace(cfg)


def _load_safetensors(module: nn.Module, model_path: str | None) -> None:
    if not model_path:
        return
    paths = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not paths:
        return
    state: dict[str, torch.Tensor] = {}
    for path in paths:
        state.update(safetensors.torch.load_file(path, device="cpu"))
    module.load_state_dict(state, strict=False)


def _additional_information(req: OmniDiffusionRequest) -> dict[str, Any]:
    if not req.prompts:
        return {}
    prompt = req.prompts[0]
    if isinstance(prompt, dict):
        info = prompt.get("additional_information", {})
        if not isinstance(info, dict):
            raise ValueError("WLAM diffusion prompt additional_information must be a dict")
        return info
    return {}


class WLAMDiffusionPipeline(nn.Module):
    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = _torch_dtype(od_config.dtype)
        self.args = WLAMModelArgs.from_hf_config(_load_model_config(od_config))
        self.transformer = WLAMDiffusionTransformer(self.args).to(device=self.device, dtype=self.dtype)
        _load_safetensors(self, od_config.model)

    @property
    def guidance_scale(self) -> float:
        return 1.0

    def _target_latents(
        self,
        req: OmniDiffusionRequest,
        info: dict[str, Any],
    ) -> tuple[torch.Tensor, tuple[int, ...] | None]:
        latents = info.get("target_latents", req.sampling_params.latents)
        if latents is not None:
            latents = latents if isinstance(latents, torch.Tensor) else torch.tensor(latents)
            original_shape = tuple(latents.shape)
            latents = latents.to(device=self.device, dtype=self.dtype)
            if latents.ndim == 2:
                latents = latents.unsqueeze(0)
            if latents.ndim > 3:
                latents = latents.reshape(latents.shape[0], -1, latents.shape[-1])
            if latents.shape[-1] != self.args.visual_latent_dim:
                raise ValueError(
                    f"target_latents last dim must be {self.args.visual_latent_dim}, got {latents.shape[-1]}"
                )
            return latents, original_shape

        shape = info.get("target_latent_shape", req.sampling_params.raw_latent_shape)
        if shape is None:
            n_tokens = info.get("num_target_tokens", req.sampling_params.n_tokens)
            if n_tokens is None:
                raise ValueError("WLAM diffusion requires target_latents, target_latent_shape, or n_tokens")
            shape = (1, int(n_tokens), self.args.visual_latent_dim)
        if isinstance(shape, torch.Tensor):
            shape = tuple(int(x) for x in shape.flatten().tolist())
        else:
            shape = tuple(int(x) for x in shape)
        if len(shape) == 2:
            shape = (1, *shape)
        if shape[-1] != self.args.visual_latent_dim:
            raise ValueError(f"target_latent_shape last dim must be {self.args.visual_latent_dim}, got {shape[-1]}")

        generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.sampling_params.seed)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        return latents, shape

    def _position_ids(
        self,
        info: dict[str, Any],
        num_tokens: int,
    ) -> torch.Tensor:
        position_ids = info.get("target_mrope_position_ids")
        if position_ids is not None:
            position_ids = position_ids if isinstance(position_ids, torch.Tensor) else torch.tensor(position_ids)
            return position_ids.to(device=self.device, dtype=torch.long).reshape(num_tokens, -1)
        n_axes = len(self.args.mrope_section or [2, 2, 31, 31])
        return torch.arange(num_tokens, device=self.device, dtype=torch.long).reshape(-1, 1).expand(-1, n_axes)

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        info = _additional_information(req)
        past_key_values = req.sampling_params.past_key_values
        if past_key_values is None:
            raise ValueError("WLAM diffusion requires past_key_values from the AR stage")

        latents, original_shape = self._target_latents(req, info)
        position_ids = self._position_ids(info, latents.shape[1])
        num_steps = int(req.sampling_params.num_inference_steps or info.get("num_inference_steps", 50))
        timesteps = req.sampling_params.timesteps
        if timesteps is None:
            timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device, dtype=torch.float32)[:-1]
        else:
            timesteps = timesteps.to(device=self.device, dtype=torch.float32).flatten()
            num_steps = int(timesteps.numel())

        dt = 1.0 / max(num_steps, 1)
        for t in timesteps:
            t_batch = t.expand(latents.shape[0])
            t_emb = timestep_embedding(t_batch * 1000.0, self.args.hidden_size).to(device=self.device, dtype=self.dtype)
            v_pred = self.transformer(
                latents,
                position_ids,
                t_emb,
                past_key_values=past_key_values,
            )
            latents = latents - dt * v_pred

        output = latents
        if original_shape is not None and len(original_shape) > 0:
            output = latents.reshape(original_shape)

        custom_output = {
            "latents": output,
            "height": info.get("height", req.sampling_params.height),
            "width": info.get("width", req.sampling_params.width),
            "num_frames": info.get("num_frames", req.sampling_params.num_frames),
            "fps": info.get("fps", req.sampling_params.fps),
        }
        return DiffusionOutput(output=output, custom_output=custom_output)
