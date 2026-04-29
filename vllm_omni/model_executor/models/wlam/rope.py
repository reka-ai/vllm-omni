from __future__ import annotations

import math

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class WLAMMultimodalRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        mrope_section: list[int],
        *,
        base: int = 10000,
    ) -> None:
        super().__init__()
        if len(mrope_section) not in (3, 4):
            raise ValueError(f"mrope_section must have 3 or 4 entries, got {mrope_section}")
        if sum(mrope_section) > head_dim // 2:
            raise ValueError(
                f"sum(mrope_section)={sum(mrope_section)} exceeds head_dim//2={head_dim // 2}"
            )
        self.head_dim = head_dim
        self.mrope_section = list(mrope_section)
        self.base = base

    def _axis_freqs(
        self,
        positions: torch.Tensor,
        slots: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if slots == 0:
            empty = positions.new_empty((*positions.shape, 0), dtype=torch.float32)
            return empty, empty
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, slots, device=positions.device, dtype=torch.float32) / max(slots, 1))
        )
        freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def cos_sin(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape[-1] != len(self.mrope_section):
            raise ValueError(
                f"position_ids must be [N, {len(self.mrope_section)}], got {tuple(position_ids.shape)}"
            )

        cos_parts: list[torch.Tensor] = []
        sin_parts: list[torch.Tensor] = []
        for axis, slots in enumerate(self.mrope_section):
            c, s = self._axis_freqs(position_ids[:, axis], slots)
            cos_parts.append(c)
            sin_parts.append(s)

        used = 2 * sum(self.mrope_section)
        if used < self.head_dim:
            pad = self.head_dim - used
            cos_parts.append(torch.ones(position_ids.shape[0], pad, device=position_ids.device))
            sin_parts.append(torch.zeros(position_ids.shape[0], pad, device=position_ids.device))

        return torch.cat(cos_parts, dim=-1).to(dtype), torch.cat(sin_parts, dim=-1).to(dtype)

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin(position_ids, dtype=q.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
