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


@dataclass
class WLAMCondition:
    embedding: torch.Tensor
    instance_ids: torch.Tensor | None = None

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        if self.instance_ids is None:
            return self.embedding
        return self.embedding[self.instance_ids[idx]]


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
