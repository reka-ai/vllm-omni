from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .common import WLAMModelArgs
from .common_layers import WLAMAdaLN, WLAMSwiGLUMLP, as_tensor, gate_residual_update, t2i_modulate
from .conditioning import (
    WLAMCondition,
    WLAMTimeEmbedMode,
    WLAMSinusoidalEmbedder,
    combine_adaln_conditions,
    embed_scalar_condition,
)
from .layout import WLAMTokenLayout, WLAMTokenType
from .rope import WLAMMultimodalRotaryEmbedding


def _as_tensor(x: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
    return as_tensor(x)


def _compact_or_full(
    tensor: torch.Tensor | None,
    idx: torch.Tensor,
    *,
    full_len: int,
    name: str,
) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"{name} is required for {idx.numel()} token(s)")
    if tensor.shape[0] == full_len:
        return tensor[idx]
    if tensor.shape[0] == idx.numel():
        return tensor
    raise ValueError(
        f"{name} must have first dim {full_len} or {idx.numel()}, got {tuple(tensor.shape)}"
    )


class WLAMARAttention(nn.Module):
    def __init__(
        self,
        args: WLAMModelArgs,
        *,
        config: Any | None = None,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if args.num_attention_heads % tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by tensor parallel size")
        if args.num_key_value_heads >= tp_size and args.num_key_value_heads % tp_size != 0:
            raise ValueError("num_key_value_heads must be divisible by tensor parallel size")

        self.args = args
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
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=args.num_attention_heads * self.head_dim,
            output_size=args.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if args.use_dual_attn_weights:
            self.qkv_proj_diffusion = QKVParallelLinear(
                hidden_size=args.hidden_size,
                head_size=self.head_dim,
                total_num_heads=args.num_attention_heads,
                total_num_kv_heads=args.num_key_value_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj_diffusion",
            )
            self.o_proj_diffusion = RowParallelLinear(
                input_size=args.num_attention_heads * self.head_dim,
                output_size=args.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj_diffusion",
            )

        self.rope = WLAMMultimodalRotaryEmbedding(
            self.head_dim,
            args.mrope_section or [2, 2, 31, 31],
            base=args.rope_theta,
            mode=args.mrope_freq_mode,
        )
        self.attn = Attention(
            self.qkv_proj.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.qkv_proj.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def _project_qkv(self, hidden_states: torch.Tensor, layout: WLAMTokenLayout) -> torch.Tensor:
        if not self.args.use_dual_attn_weights or not layout.has_diffusion:
            return _as_tensor(self.qkv_proj(hidden_states))
        vlm_h, diffusion_h = layout.gather(hidden_states)
        vlm_qkv = _as_tensor(self.qkv_proj(vlm_h))
        diffusion_qkv = _as_tensor(self.qkv_proj_diffusion(diffusion_h))
        return layout.scatter(vlm_qkv, diffusion_qkv, out_dim=vlm_qkv.shape[-1])

    def _project_o(self, attn_out: torch.Tensor, layout: WLAMTokenLayout) -> torch.Tensor:
        if not self.args.use_dual_attn_weights or not layout.has_diffusion:
            return _as_tensor(self.o_proj(attn_out))
        vlm_out = _as_tensor(self.o_proj(attn_out[layout.vlm_idx]))
        diffusion_out = _as_tensor(self.o_proj_diffusion(attn_out[layout.diffusion_idx]))
        return layout.scatter(vlm_out, diffusion_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layout: WLAMTokenLayout,
    ) -> torch.Tensor:
        qkv = self._project_qkv(hidden_states, layout)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(q.shape[0], -1, self.head_dim)
        k = k.view(k.shape[0], -1, self.head_dim)
        q, k = self.rope.apply(q, k, position_ids)
        q = q.reshape(q.shape[0], -1)
        k = k.reshape(k.shape[0], -1)
        attn_out = self.attn(q, k, v)
        return self._project_o(attn_out, layout)


class WLAMARDecoderLayer(nn.Module):
    def __init__(
        self,
        args: WLAMModelArgs,
        *,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.args = args
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if args.use_dual_attn_weights:
            self.input_layernorm_diffusion = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if args.use_dual_mlp_weights:
            self.post_attention_layernorm_diffusion = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.mlp_diffusion = WLAMSwiGLUMLP(
                args.hidden_size,
                args.diffusion_intermediate_size or args.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp_diffusion",
            )
        self.modulation_diffusion = WLAMAdaLN(args.hidden_size) if args.use_time_modulation else None
        self.self_attn = WLAMARAttention(
            args,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = WLAMSwiGLUMLP(
            args.hidden_size,
            args.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layout: WLAMTokenLayout,
        adaln_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_idx = layout.diffusion_target_idx
        modulation = None
        if adaln_condition is not None and target_idx.numel() > 0:
            if self.modulation_diffusion is None:
                raise ValueError("adaln_condition was provided but use_time_modulation=False")
            modulation = self.modulation_diffusion(adaln_condition)

        residual = hidden_states
        if self.args.use_dual_attn_weights and layout.has_diffusion:
            normed = layout.apply(
                hidden_states,
                self.input_layernorm,
                self.input_layernorm_diffusion,
            )
        else:
            normed = self.input_layernorm(hidden_states)
        if modulation is not None:
            normed = normed.index_put(
                (target_idx,),
                t2i_modulate(normed[target_idx], modulation.shift_msa, modulation.scale_msa),
            )
        attn_out = self.self_attn(normed, position_ids, layout)
        if modulation is not None:
            attn_out = gate_residual_update(attn_out, modulation.gate_msa, target_idx)
        hidden_states = residual + attn_out

        residual = hidden_states
        if self.args.use_dual_mlp_weights and layout.has_diffusion:
            normed = layout.apply(
                hidden_states,
                self.post_attention_layernorm,
                self.post_attention_layernorm_diffusion,
            )
            mlp_out = layout.apply(normed, self.mlp, self.mlp_diffusion)
        else:
            normed = self.post_attention_layernorm(hidden_states)
            mlp_out = self.mlp(normed)
        if modulation is not None:
            normed = normed.index_put(
                (target_idx,),
                t2i_modulate(normed[target_idx], modulation.shift_mlp, modulation.scale_mlp),
            )
            if self.args.use_dual_mlp_weights and layout.has_diffusion:
                mlp_out = layout.apply(normed, self.mlp, self.mlp_diffusion)
            else:
                mlp_out = self.mlp(normed)
            mlp_out = gate_residual_update(mlp_out, modulation.gate_mlp, target_idx)
        return residual + mlp_out


class WLAMARModel(nn.Module):
    def __init__(
        self,
        args: WLAMModelArgs,
        *,
        config: Any | None = None,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = VocabParallelEmbedding(
            args.vocab_size,
            args.hidden_size,
            org_num_embeddings=args.vocab_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.language_projection = nn.Sequential(
            nn.Linear(args.vision_hidden_size, args.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(args.hidden_size, args.hidden_size, bias=True),
        )
        self.vision_model = None
        vision_config = getattr(config, "vision_config", None) if config is not None else None
        if vision_config is not None:
            from transformers import ConvNextV2Model

            if not hasattr(vision_config, "num_stages"):
                vision_config.num_stages = 4
            self.vision_model = ConvNextV2Model(vision_config)
        pool_side = int(args.num_query_tokens**0.5)
        if pool_side * pool_side != args.num_query_tokens:
            raise ValueError("num_query_tokens must be a perfect square for ConvNeXT pooling")
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(pool_side)
        self.diffusion_projection = nn.Sequential(
            nn.Linear(args.visual_latent_dim, args.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(args.hidden_size, args.hidden_size, bias=True),
        )
        self.diffusion_time_embedder = WLAMSinusoidalEmbedder(args.hidden_size)
        self.aspect_ratio_embedder = (
            WLAMSinusoidalEmbedder(args.hidden_size) if args.use_aspect_ratio_cond else None
        )
        self.diffusion_modulation_proj = (
            nn.Sequential(nn.SiLU(), nn.Linear(args.hidden_size, args.hidden_size, bias=True), nn.SiLU())
            if args.time_embed_mode == WLAMTimeEmbedMode.ADALN.value
            else None
        )
        self.layers = nn.ModuleList(
            [
                WLAMARDecoderLayer(
                    args,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(args.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _runtime_tensor(
        self,
        runtime: list[dict[str, Any]] | None,
        key: str,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not runtime:
            return None
        vals = [info[key] for info in runtime if isinstance(info, dict) and key in info]
        if not vals:
            return None
        vals = [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in vals]
        return torch.cat([v.to(device=device) for v in vals], dim=0)

    def _encode_convnext_pixels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.vision_model is None:
            raise ValueError("pixel_values were provided, but this WLAM config has no vision_config")
        if pixel_values.ndim > 4:
            pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
        if pixel_values.ndim != 4:
            raise ValueError(f"pixel_values must be [N, C, H, W], got {tuple(pixel_values.shape)}")
        dtype = next(self.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(device=next(self.vision_model.parameters()).device, dtype=dtype)
        features = self.vision_model(pixel_values).last_hidden_state
        features = self.adaptive_pooling(features)
        return features.flatten(2).transpose(1, 2).reshape(-1, features.shape[1])

    def _build_position_ids(
        self,
        positions: torch.Tensor,
        runtime: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        runtime_pos = self._runtime_tensor(runtime, "mrope_position_ids", device=positions.device)
        if runtime_pos is not None:
            return runtime_pos.long()
        n_axes = len(self.args.mrope_section or [2, 2, 31, 31])
        return positions.long().reshape(-1, 1).expand(-1, n_axes)

    def _build_token_type_ids(
        self,
        input_ids: torch.Tensor,
        runtime: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        runtime_types = self._runtime_tensor(runtime, "token_type_ids", device=input_ids.device)
        if runtime_types is not None:
            return runtime_types.long()
        return torch.full_like(input_ids, int(WLAMTokenType.TEXT))

    def _runtime_or_kwarg_tensor(
        self,
        kwargs: dict[str, Any],
        runtime: list[dict[str, Any]] | None,
        key: str,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        value = kwargs.get(key)
        if value is None:
            value = self._runtime_tensor(runtime, key, device=device)
        if value is None:
            return None
        return value if isinstance(value, torch.Tensor) else torch.tensor(value, device=device)

    def _scalar_condition(
        self,
        kwargs: dict[str, Any],
        runtime: list[dict[str, Any]] | None,
        key: str,
        instance_key: str,
        embedder: WLAMSinusoidalEmbedder,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> WLAMCondition | None:
        value = self._runtime_or_kwarg_tensor(kwargs, runtime, key, device=device)
        if value is None:
            return None
        instance_ids = self._runtime_or_kwarg_tensor(kwargs, runtime, instance_key, device=device)
        if instance_ids is not None:
            instance_ids = instance_ids.long()
        return embed_scalar_condition(
            value.to(device=device, dtype=dtype),
            embedder,
            instance_ids,
            dtype,
            expand_instance_eager=self.args.expand_instance_eager,
        )

    def _build_adaln_condition(
        self,
        layout: WLAMTokenLayout,
        runtime: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if layout.diffusion_target_idx.numel() == 0:
            return None
        timestep = self._scalar_condition(
            kwargs,
            runtime,
            "timestep",
            "time_instance_ids",
            self.diffusion_time_embedder,
            device=device,
            dtype=dtype,
        )
        if timestep is None:
            raise ValueError("timestep is required when time_embed_mode=adaLN and diffusion targets are present")

        conditions = [timestep]
        if self.args.use_aspect_ratio_cond and self.args.metadata_embed_mode == WLAMTimeEmbedMode.ADALN.value:
            if self.aspect_ratio_embedder is None:
                raise ValueError("aspect ratio conditioning requested without aspect_ratio_embedder")
            aspect = self._scalar_condition(
                kwargs,
                runtime,
                "aspect_ratio",
                "aspect_ratio_instance_ids",
                self.aspect_ratio_embedder,
                device=device,
                dtype=dtype,
            )
            if aspect is None:
                raise ValueError("aspect_ratio is required when metadata_embed_mode=adaLN")
            conditions.append(aspect)

        condition = combine_adaln_conditions(
            conditions,
            expand_instance_eager=self.args.expand_instance_eager,
        )
        gathered = condition.gather_at(layout.diffusion_target_idx)
        if self.diffusion_modulation_proj is not None:
            if condition.granularity.name == "GLOBAL":
                projected = self.diffusion_modulation_proj(condition.embedding)
                return projected.squeeze(0)
            projected = self.diffusion_modulation_proj(condition.embedding)
            condition = WLAMCondition(projected, condition.granularity, condition.instance_ids)
            gathered = condition.gather_at(layout.diffusion_target_idx)
        return gathered

    def _build_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        layout: WLAMTokenLayout,
        runtime: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype
        hidden_states = inputs_embeds.clone() if inputs_embeds is not None else self.embed_tokens(input_ids)

        convnext = kwargs.get("convnext_embeds")
        if convnext is None:
            convnext = self._runtime_tensor(runtime, "convnext_embeds", device=device)
        if convnext is None:
            pixel_values = kwargs.get("pixel_values")
            if pixel_values is None:
                pixel_values = self._runtime_tensor(runtime, "pixel_values", device=device)
            if pixel_values is not None:
                convnext = self._encode_convnext_pixels(pixel_values)
        if layout.has_convnext:
            convnext = _compact_or_full(
                convnext,
                layout.convnext_idx,
                full_len=layout.n_total,
                name="convnext_embeds",
            ).to(device=device, dtype=dtype)
            hidden_states[layout.convnext_idx] = self.language_projection(convnext)

        vae_latents = kwargs.get("vae_latents")
        if vae_latents is None:
            vae_latents = self._runtime_tensor(runtime, "vae_latents", device=device)
        if layout.diffusion_latent_idx.numel() > 0:
            vae_latents = _compact_or_full(
                vae_latents,
                layout.diffusion_latent_idx,
                full_len=layout.n_total,
                name="vae_latents",
            ).to(device=device, dtype=dtype)
            hidden_states[layout.diffusion_latent_idx] = self.diffusion_projection(vae_latents)

        if layout.time_embed_idx.numel() > 0:
            if self.args.time_embed_mode != WLAMTimeEmbedMode.IN_CONTEXT.value:
                raise ValueError("DIFFUSION_TIME tokens require time_embed_mode=in_context")
            time_cond = self._scalar_condition(
                kwargs,
                runtime,
                "timestep",
                "time_instance_ids",
                self.diffusion_time_embedder,
                device=device,
                dtype=dtype,
            )
            if time_cond is None:
                raise ValueError("timestep is required for DIFFUSION_TIME tokens")
            hidden_states[layout.time_embed_idx] = time_cond.gather_at(layout.time_embed_idx).to(dtype=dtype)

        if self.args.use_aspect_ratio_cond and self.args.metadata_embed_mode == WLAMTimeEmbedMode.IN_CONTEXT.value:
            if self.aspect_ratio_embedder is None:
                raise ValueError("aspect ratio conditioning requested without aspect_ratio_embedder")
            aspect_idx = layout.metadata_embed_indices.get(int(WLAMTokenType.ASPECT_RATIO))
            if aspect_idx is not None and aspect_idx.numel() > 0:
                aspect_cond = self._scalar_condition(
                    kwargs,
                    runtime,
                    "aspect_ratio",
                    "aspect_ratio_instance_ids",
                    self.aspect_ratio_embedder,
                    device=device,
                    dtype=dtype,
                )
                if aspect_cond is None:
                    raise ValueError("aspect_ratio is required for ASPECT_RATIO tokens")
                hidden_states[aspect_idx] = aspect_cond.gather_at(aspect_idx).to(dtype=dtype)

        if self.args.metadata_embed_mode == WLAMTimeEmbedMode.ADDITION.value:
            if self.args.use_aspect_ratio_cond and self.aspect_ratio_embedder is not None:
                aspect_cond = self._scalar_condition(
                    kwargs,
                    runtime,
                    "aspect_ratio",
                    "aspect_ratio_instance_ids",
                    self.aspect_ratio_embedder,
                    device=device,
                    dtype=dtype,
                )
                if aspect_cond is None:
                    raise ValueError("aspect_ratio is required when metadata_embed_mode=addition")
                hidden_states = aspect_cond.add_to_target_at(hidden_states, layout.diffusion_latent_idx)

        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        runtime = model_intermediate_buffer or runtime_additional_information
        token_type_ids = self._build_token_type_ids(input_ids, runtime)
        position_ids = self._build_position_ids(positions, runtime)
        metadata_token_types = (
            [int(WLAMTokenType.ASPECT_RATIO)]
            if self.args.use_aspect_ratio_cond and self.args.metadata_embed_mode == WLAMTimeEmbedMode.IN_CONTEXT.value
            else None
        )
        layout = WLAMTokenLayout.from_token_type_ids(
            token_type_ids,
            metadata_token_types=metadata_token_types,
        )
        hidden_states = self._build_embeddings(input_ids, inputs_embeds, layout, runtime, kwargs)
        adaln_condition = None
        if self.args.time_embed_mode == WLAMTimeEmbedMode.ADALN.value:
            adaln_condition = self._build_adaln_condition(
                layout,
                runtime,
                kwargs,
                device=input_ids.device,
                dtype=hidden_states.dtype,
            )
        elif self.args.time_embed_mode == WLAMTimeEmbedMode.ADDITION.value:
            time_cond = self._scalar_condition(
                kwargs,
                runtime,
                "timestep",
                "time_instance_ids",
                self.diffusion_time_embedder,
                device=input_ids.device,
                dtype=hidden_states.dtype,
            )
            if time_cond is None and layout.diffusion_target_idx.numel() > 0:
                raise ValueError("timestep is required when time_embed_mode=addition and diffusion targets are present")
            if time_cond is not None:
                hidden_states = time_cond.add_to_target_at(hidden_states, layout.diffusion_target_idx)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, layout, adaln_condition)
        return self.norm(hidden_states)


class WLAMForConditionalGeneration(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "qkv_proj_diffusion": ["q_proj_diffusion", "k_proj_diffusion", "v_proj_diffusion"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.args = WLAMModelArgs.from_hf_config(self.config)
        self.model = WLAMARModel(
            self.args,
            config=self.config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )
        self.lm_head = ParallelLMHead(
            self.args.vocab_size,
            self.args.hidden_size,
            org_num_embeddings=self.args.vocab_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
        )
        self.logits_processor = LogitsProcessor(self.args.vocab_size)
        self.make_empty_intermediate_tensors = lambda *args, **kwargs: None

    def get_language_model(self) -> nn.Module:
        return self

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is not None:
            raise ValueError("WLAM expects ConvNeXT/VAE tensors through runtime metadata, not multimodal_embeddings")
        return self.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput | IntermediateTensors:
        if intermediate_tensors is not None:
            raise NotImplementedError("Pipeline parallel intermediate_tensors are not implemented for WLAM yet")
        hidden_states = self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return OmniOutput(text_hidden_states=hidden_states, multimodal_outputs={}, intermediate_tensors=None)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_kv_transfer_metadata(
        self,
        req_id: str,
        *,
        num_computed_tokens: int | None = None,
    ) -> dict[str, Any]:
        return {"context_seq_len": num_computed_tokens} if num_computed_tokens is not None else {}
