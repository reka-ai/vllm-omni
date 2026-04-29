from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WLAMModelArgs:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-5
    rope_theta: int = 10000
    visual_latent_dim: int = 16
    vision_hidden_size: int = 2816
    num_query_tokens: int = 64
    mrope_section: list[int] | None = None
    use_dual_attn_weights: bool = True
    use_dual_mlp_weights: bool = True
    use_time_modulation: bool = True

    @classmethod
    def from_hf_config(cls, config: Any) -> "WLAMModelArgs":
        text_config = getattr(config, "text_config", config)
        vision_config = getattr(config, "vision_config", None)
        hidden_size = int(
            getattr(text_config, "hidden_size", getattr(config, "hidden_size", getattr(config, "d_model", 4096)))
        )
        num_heads = int(getattr(text_config, "num_attention_heads", getattr(config, "num_attention_heads", 16)))
        intermediate_size = int(
            getattr(
                text_config,
                "intermediate_size",
                getattr(config, "intermediate_size", getattr(config, "ffn_hidden_size", 5440)),
            )
        )
        vision_hidden_size = getattr(config, "vision_hidden_size", None)
        if vision_hidden_size is None and vision_config is not None:
            vision_hidden_size = getattr(vision_config, "hidden_size", None)
            vision_hidden_size = getattr(vision_config, "output_dim", vision_hidden_size)
        vision_encoder = getattr(config, "vision_encoder", None)
        if vision_hidden_size is None and vision_encoder is not None:
            vision_hidden_size = getattr(vision_encoder, "output_dim", None)
        return cls(
            vocab_size=int(getattr(text_config, "vocab_size", getattr(config, "vocab_size", 100352))),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=int(
                getattr(text_config, "num_hidden_layers", getattr(config, "num_hidden_layers", getattr(config, "num_layers", 16)))
            ),
            num_attention_heads=num_heads,
            num_key_value_heads=int(
                getattr(
                    text_config,
                    "num_key_value_heads",
                    getattr(text_config, "num_query_groups", getattr(config, "num_query_groups", num_heads)),
                )
            ),
            rms_norm_eps=float(
                getattr(text_config, "rms_norm_eps", getattr(text_config, "layer_norm_eps", 1e-5))
            ),
            rope_theta=int(getattr(text_config, "rotary_emb_base", getattr(config, "rotary_base", 10000))),
            visual_latent_dim=int(getattr(config, "visual_latent_dim", 16)),
            vision_hidden_size=int(vision_hidden_size or 2816),
            num_query_tokens=int(getattr(config, "num_query_tokens", 64)),
            mrope_section=list(getattr(config, "mrope_section", [2, 2, 31, 31])),
            use_dual_attn_weights=bool(getattr(config, "use_dual_attn_weights", True)),
            use_dual_mlp_weights=bool(getattr(config, "use_dual_mlp_weights", True)),
            use_time_modulation=bool(getattr(config, "use_time_modulation", True)),
        )
