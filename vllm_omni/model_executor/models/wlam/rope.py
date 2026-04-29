from __future__ import annotations

import math
from enum import Enum

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
        mode: str = "standard",
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
        self.mode = _normalize_mode(mode)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if self.mode == "interleaved":
            n = len(self.mrope_section)
            half_dim = head_dim // 2
            for axis, slots in enumerate(self.mrope_section):
                available = (half_dim - axis - 1) // n + 1
                if available < slots:
                    raise ValueError(
                        f"interleaved mrope section {axis} needs {slots} frequencies, got {available}"
                    )

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

        inv_freq = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        freqs_parts: list[torch.Tensor] = []
        offset = 0
        n_axes = len(self.mrope_section)
        for axis, slots in enumerate(self.mrope_section):
            pos = position_ids[:, axis].float().unsqueeze(-1)
            if self.mode == "standard":
                freq_slice = inv_freq[offset : offset + slots]
                offset += slots
            elif self.mode == "restart":
                freq_slice = inv_freq[:slots]
            elif self.mode == "interleaved":
                freq_slice = inv_freq[axis::n_axes][:slots]
            else:
                raise ValueError(f"Unknown mrope mode {self.mode!r}")
            freqs_parts.append(pos * freq_slice.unsqueeze(0))

        freqs = torch.cat(freqs_parts, dim=-1)
        half_dim = self.head_dim // 2
        if freqs.shape[-1] < half_dim:
            pad = torch.zeros(
                position_ids.shape[0],
                half_dim - freqs.shape[-1],
                device=position_ids.device,
                dtype=torch.float32,
            )
            freqs = torch.cat([freqs, pad], dim=-1)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

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


def _normalize_mode(mode: str | Enum) -> str:
    value = mode.value if isinstance(mode, Enum) else str(mode)
    value = value.split(".")[-1].lower()
    if value not in {"standard", "restart", "interleaved"}:
        raise ValueError(f"Unsupported mrope frequency mode {mode!r}")
    return value
