# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared helpers for the in-tree Marey VAE."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def cast_tuple(t: Any, length: int = 1) -> tuple[Any, ...]:
    return t if isinstance(t, tuple) else ((t,) * length)


def is_odd(n: int) -> bool:
    return n % 2 == 1


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


def pad_at_dim(t: torch.Tensor, pad: tuple[int, int], dim: int = -1) -> torch.Tensor:
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), mode="constant")
