#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two final pre-VAE latent tensors saved by Phase 0 inference runs.

Expects each input to be a `.pt` file containing a single `torch.Tensor`
of shape (B, 4, T_latent, H_latent, W_latent). Prints a small report and
exits non-zero if the divergence exceeds the given tolerance.

Usage:
    python compare_final_latents.py A.pt B.pt [--rel-tol 1e-6]
"""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="First latent file (.pt)")
    p.add_argument("b", help="Second latent file (.pt)")
    p.add_argument(
        "--rel-tol",
        type=float,
        default=1e-6,
        help="Pass if max_abs_diff < rel_tol * max_abs(a). Default 1e-6 (effectively bit-equality for bf16).",
    )
    args = p.parse_args()

    a = torch.load(args.a, map_location="cpu")
    b = torch.load(args.b, map_location="cpu")

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        print(f"FAIL: one of the inputs is not a tensor (a={type(a)}, b={type(b)})")
        return 2

    print(f"A: {args.a}  shape={tuple(a.shape)}  dtype={a.dtype}  device={a.device}")
    print(f"B: {args.b}  shape={tuple(b.shape)}  dtype={b.dtype}  device={b.device}")

    if a.shape != b.shape:
        print(f"FAIL: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        return 1

    a_f = a.to(torch.float64).flatten()
    b_f = b.to(torch.float64).flatten()
    diff = (a_f - b_f).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    a_mag_max = a_f.abs().max().item()
    a_mag_mean = a_f.abs().mean().item()
    rel_max = max_abs / a_mag_max if a_mag_max > 0 else float("nan")
    rel_mean = mean_abs / a_mag_mean if a_mag_mean > 0 else float("nan")

    a_norm = a_f.norm()
    b_norm = b_f.norm()
    cos = (a_f @ b_f / (a_norm * b_norm)).item() if a_norm > 0 and b_norm > 0 else float("nan")

    print(f"max_abs_diff:        {max_abs:.6e}")
    print(f"mean_abs_diff:       {mean_abs:.6e}")
    print(f"max_abs(a):          {a_mag_max:.6e}")
    print(f"mean_abs(a):         {a_mag_mean:.6e}")
    print(f"max_abs_diff/max(a): {rel_max:.6e}    (relative to peak)")
    print(f"mean_abs_diff/mean(a): {rel_mean:.6e}  (relative to magnitude)")
    print(f"cosine_similarity:   {cos:.10f}")

    threshold = args.rel_tol * a_mag_max
    print(f"\nTolerance: rel_tol={args.rel_tol:.1e} -> max_abs_diff must be < {threshold:.6e}")
    if max_abs < threshold:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
