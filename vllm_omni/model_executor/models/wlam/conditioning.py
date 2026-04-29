from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from .rope import timestep_embedding


class WLAMTimeEmbedMode(str, Enum):
    ADALN = "adaLN"
    IN_CONTEXT = "in_context"
    ADDITION = "addition"


class WLAMConditionGranularity(str, Enum):
    GLOBAL = "global"
    PER_INSTANCE = "per_instance"
    PER_TOKEN = "per_token"


@dataclass
class WLAMCondition:
    embedding: torch.Tensor
    granularity: WLAMConditionGranularity
    instance_ids: torch.Tensor | None = None

    def gather_at(self, idx: torch.Tensor) -> torch.Tensor:
        if self.granularity == WLAMConditionGranularity.GLOBAL:
            return self.embedding.squeeze(0)
        if self.granularity == WLAMConditionGranularity.PER_INSTANCE:
            if self.instance_ids is None:
                raise ValueError("PER_INSTANCE condition requires instance_ids")
            return self.embedding[self.instance_ids[idx]]
        return self.embedding[idx]

    def add_to_target_at(self, target: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return target.index_put((idx,), target[idx] + self.gather_at(idx))


class WLAMSinusoidalEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, value: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        value = value.reshape(-1)
        emb = timestep_embedding(value, self.frequency_embedding_size).to(dtype=dtype)
        return self.mlp(emb)


def embed_scalar_condition(
    value: torch.Tensor,
    embedder: WLAMSinusoidalEmbedder,
    instance_ids: torch.Tensor | None,
    dtype: torch.dtype,
    *,
    expand_instance_eager: bool = False,
) -> WLAMCondition:
    if instance_ids is not None:
        embedding = embedder(value.reshape(-1), dtype=dtype)
        if expand_instance_eager:
            return WLAMCondition(
                embedding[instance_ids],
                WLAMConditionGranularity.PER_TOKEN,
            )
        return WLAMCondition(
            embedding,
            WLAMConditionGranularity.PER_INSTANCE,
            instance_ids,
        )
    embedding = embedder(value.reshape(-1)[:1], dtype=dtype)
    return WLAMCondition(embedding, WLAMConditionGranularity.GLOBAL)


def combine_adaln_conditions(
    conditions: list[WLAMCondition],
    *,
    expand_instance_eager: bool = False,
) -> WLAMCondition:
    if not conditions:
        raise ValueError("combine_adaln_conditions requires at least one condition")
    if len(conditions) == 1:
        return conditions[0]
    if expand_instance_eager:
        seq_len = next(
            c.embedding.shape[0]
            for c in conditions
            if c.granularity == WLAMConditionGranularity.PER_TOKEN
        )
        combined = None
        for cond in conditions:
            emb = cond.embedding
            if cond.granularity == WLAMConditionGranularity.GLOBAL:
                emb = emb.expand(seq_len, -1)
            elif cond.granularity == WLAMConditionGranularity.PER_INSTANCE:
                if cond.instance_ids is None:
                    raise ValueError("PER_INSTANCE condition requires instance_ids")
                emb = emb[cond.instance_ids]
            combined = emb if combined is None else combined + emb
        return WLAMCondition(combined, WLAMConditionGranularity.PER_TOKEN)

    global_sum = None
    per_instance = []
    for cond in conditions:
        if cond.granularity == WLAMConditionGranularity.GLOBAL:
            emb = cond.embedding.squeeze(0)
            global_sum = emb if global_sum is None else global_sum + emb
        elif cond.granularity == WLAMConditionGranularity.PER_INSTANCE:
            if cond.instance_ids is None:
                raise ValueError("PER_INSTANCE condition requires instance_ids")
            per_instance.append(cond)
        else:
            per_instance.append(
                WLAMCondition(
                    cond.embedding,
                    WLAMConditionGranularity.PER_INSTANCE,
                    torch.arange(cond.embedding.shape[0], device=cond.embedding.device),
                )
            )

    if not per_instance:
        if global_sum is None:
            raise ValueError("No condition embeddings to combine")
        return WLAMCondition(global_sum.unsqueeze(0), WLAMConditionGranularity.GLOBAL)

    stacked = torch.stack([c.instance_ids for c in per_instance], dim=0)  # type: ignore[list-item]
    unique_combos, merged_ids = torch.unique(stacked, dim=1, return_inverse=True)
    combined = torch.zeros(
        unique_combos.shape[1],
        per_instance[0].embedding.shape[-1],
        dtype=per_instance[0].embedding.dtype,
        device=per_instance[0].embedding.device,
    )
    for i, cond in enumerate(per_instance):
        combined = combined + cond.embedding[unique_combos[i]]
    if global_sum is not None:
        combined = combined + global_sum
    return WLAMCondition(combined, WLAMConditionGranularity.PER_INSTANCE, merged_ids)
