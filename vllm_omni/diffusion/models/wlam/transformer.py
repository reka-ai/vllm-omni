from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.model_executor.models.wlam.common_layers import (
    WLAMAdaLN,
    WLAMDiffusionHead,
    WLAMSwiGLUMLP,
    as_tensor,
    gate_residual_update,
    t2i_modulate,
)
from vllm_omni.model_executor.models.wlam.common import WLAMModelArgs
from vllm_omni.model_executor.models.wlam.conditioning import WLAMSinusoidalEmbedder
from vllm_omni.model_executor.models.wlam.rope import WLAMMultimodalRotaryEmbedding


def _as_tensor(x: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
    return as_tensor(x)


def _cache_tensor(cache: Any, layer_idx: int, attr: str) -> torch.Tensor:
    if not hasattr(cache, attr):
        raise ValueError(f"past_key_values is missing {attr}")
    value = getattr(cache, attr)[layer_idx]
    if value is None:
        raise ValueError(f"past_key_values.{attr}[{layer_idx}] is empty")
    return value


class WLAMDiffusionAttention(nn.Module):
    def __init__(self, args: WLAMModelArgs, *, layer_idx: int, prefix: str = "") -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if args.num_attention_heads % tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by tensor parallel size")
        if args.num_key_value_heads >= tp_size and args.num_key_value_heads % tp_size != 0:
            raise ValueError("num_key_value_heads must be divisible by tensor parallel size")

        self.args = args
        self.layer_idx = layer_idx
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_size = (args.num_attention_heads // tp_size) * self.head_dim
        self.kv_size = max(1, args.num_key_value_heads // tp_size) * self.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=args.hidden_size,
            head_size=self.head_dim,
            total_num_heads=args.num_attention_heads,
            total_num_kv_heads=args.num_key_value_heads,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.qkv_proj_diffusion",
        )
        self.o_proj = RowParallelLinear(
            input_size=args.num_attention_heads * self.head_dim,
            output_size=args.hidden_size,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.o_proj_diffusion",
        )
        self.rope = WLAMMultimodalRotaryEmbedding(
            self.head_dim,
            args.mrope_section or [2, 2, 31, 31],
            base=args.rope_theta,
            mode=args.mrope_freq_mode,
        )
        self.attn = Attention(
            num_heads=self.qkv_proj.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.scale,
            num_kv_heads=self.qkv_proj.num_kv_heads,
            prefix=f"{prefix}.attn",
        )

    def _split_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.unflatten(-1, (self.qkv_proj.num_heads, self.head_dim))
        k = k.unflatten(-1, (self.qkv_proj.num_kv_heads, self.head_dim))
        v = v.unflatten(-1, (self.qkv_proj.num_kv_heads, self.head_dim))
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any | None,
        *,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        q, k, v = self._split_qkv(hidden_states)
        rope_position_ids = position_ids.repeat(hidden_states.shape[0], 1)
        q, k = self.rope.apply(
            q.reshape(-1, q.shape[-2], q.shape[-1]),
            k.reshape(-1, k.shape[-2], k.shape[-1]),
            rope_position_ids,
        )
        q = q.reshape(hidden_states.shape[0], hidden_states.shape[1], -1, self.head_dim)
        k = k.reshape(hidden_states.shape[0], hidden_states.shape[1], -1, self.head_dim)

        if past_key_values is not None:
            ctx_k = _cache_tensor(past_key_values, self.layer_idx, "key_cache").to(device=k.device, dtype=k.dtype)
            ctx_v = _cache_tensor(past_key_values, self.layer_idx, "value_cache").to(device=v.device, dtype=v.dtype)
            if ctx_k.ndim == 3:
                ctx_k = ctx_k.unsqueeze(0).expand(hidden_states.shape[0], -1, -1, -1)
                ctx_v = ctx_v.unsqueeze(0).expand(hidden_states.shape[0], -1, -1, -1)
            k = torch.cat([ctx_k, k], dim=1)
            v = torch.cat([ctx_v, v], dim=1)

        out = self.attn(q, k, v, attn_metadata=attn_metadata).flatten(2, 3)
        return _as_tensor(self.o_proj(out))


class WLAMDiffusionLayer(nn.Module):
    def __init__(self, args: WLAMModelArgs, *, layer_idx: int, prefix: str = "") -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = WLAMDiffusionAttention(args, layer_idx=layer_idx, prefix=f"{prefix}.self_attn")
        self.mlp = WLAMSwiGLUMLP(
            args.hidden_size,
            args.diffusion_intermediate_size or args.intermediate_size,
            prefix=f"{prefix}.mlp_diffusion",
        )
        self.modulation = WLAMAdaLN(args.hidden_size) if args.use_time_modulation else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any | None,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        modulation = None
        if self.modulation is not None:
            if condition is None:
                raise ValueError("condition is required when use_time_modulation=True")
            modulation = self.modulation(condition)

        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        if modulation is not None:
            normed = t2i_modulate(normed, modulation.shift_msa, modulation.scale_msa)
        attn_out = self.self_attn(normed, position_ids, past_key_values)
        if modulation is not None:
            attn_out = gate_residual_update(attn_out, modulation.gate_msa)
        hidden_states = residual + attn_out

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        if modulation is not None:
            normed = t2i_modulate(normed, modulation.shift_mlp, modulation.scale_mlp)
        mlp_out = self.mlp(normed)
        if modulation is not None:
            mlp_out = gate_residual_update(mlp_out, modulation.gate_mlp)
        return residual + mlp_out


class WLAMDiffusionTransformer(nn.Module):
    def __init__(self, args: WLAMModelArgs) -> None:
        super().__init__()
        self.args = args
        self.latent_projection = nn.Sequential(
            nn.Linear(args.visual_latent_dim, args.hidden_size),
            nn.GELU(),
            nn.Linear(args.hidden_size, args.hidden_size),
        )
        self.time_embedder = WLAMSinusoidalEmbedder(args.hidden_size)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [
                WLAMDiffusionLayer(
                    self.args,
                    layer_idx=i,
                    prefix=f"transformer.layers.{i}",
                )
                for i in range(args.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.diffusion_extra_layers = nn.ModuleList(
            [
                WLAMDiffusionLayer(
                    self.args,
                    layer_idx=args.num_hidden_layers + i,
                    prefix=f"transformer.diffusion_extra_layers.{i}",
                )
                for i in range(args.num_diffusion_extra_layers)
            ]
        )
        self.diffusion_extra_norm = (
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            if args.num_diffusion_extra_layers > 0
            else None
        )
        self.diffusion_head = WLAMDiffusionHead(
            args.hidden_size,
            args.visual_latent_dim,
            use_adaLN=args.diffusion_head_adaLN,
            eps=args.rms_norm_eps,
        )

    def forward(
        self,
        target_latents: torch.Tensor,
        position_ids: torch.Tensor,
        timestep: torch.Tensor,
        *,
        past_key_values: Any | None,
    ) -> torch.Tensor:
        hidden_states = self.latent_projection(target_latents)
        condition = self.time_projection(self.time_embedder(timestep, dtype=target_latents.dtype))
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, past_key_values, condition)
        hidden_states = self.norm(hidden_states)
        if self.diffusion_extra_layers:
            for layer in self.diffusion_extra_layers:
                hidden_states = layer(hidden_states, position_ids, None, condition)
            if self.diffusion_extra_norm is None:
                raise ValueError("diffusion_extra_norm is required when diffusion_extra_layers are present")
            hidden_states = self.diffusion_extra_norm(hidden_states)
        return self.diffusion_head(hidden_states, condition)
