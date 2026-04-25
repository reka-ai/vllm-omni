#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 dump-correctness checker.

Runs four independent assertions against a Phase 1 reference dump dir and
prints a PASS/FAIL table:

  1. Cross-reference identity
       encode_cond_seq_cond_0.pt must be byte-identical to
       step0_cond_encoder_hidden_states_0.pt (same Python object flows from
       text_encoder.encode() into the transformer's encoder_hidden_states on
       step 0). Same check for the uncond side, if an uncond step is present.
       A mismatch means B2 kwarg-mapping is wrong, B3 labeling is swapped,
       or something clones the tensor between capture points.

  2. Reproducibility (requires --repro-dir)
       Every tensor in <dump> must byte-match the corresponding tensor in
       <repro_dir>. Catches stateful bugs in the wrappers (shared counters,
       missed resets).

  3. Shape + count sanity
       Expected files for 30B distilled 33-step: z_initial, step_noise_0..31,
       timesteps, encode_{cond,uncond}_*, 33x step<i>_cond_*,
       M x step<i>_uncond_* (M = 33 - skip_uncond positions). All
       hidden_states / v_pred tensors same shape as z_initial.

  4. Value distribution sanity
       z_initial mean ~0, std ~1. timesteps monotonically non-increasing
       from ~0.999 to ~0.001. v_pred tensors non-trivial (non-zero,
       non-NaN).

Usage:
    python check_phase1_dump.py <dump_dir>
    python check_phase1_dump.py <dump_dir> --repro-dir <repro_dir>
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import torch


def _list_dump(path: str) -> dict[str, str]:
    out = {}
    if not os.path.isdir(path):
        raise SystemExit(f"Dump dir not found: {path}")
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".pt"):
            out[fname[:-3]] = os.path.join(path, fname)
    return out


def _load(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def _check_cross_reference_identity(files: dict[str, str]) -> Tuple[bool, list[str]]:
    """(1) encode_*_seq_cond_0 == step0_*_encoder_hidden_states_0 byte-for-byte."""
    msgs: list[str] = []
    all_ok = True
    for label in ("cond", "uncond"):
        encode_key = f"encode_{label}_seq_cond_0"
        step_key = f"step0_{label}_encoder_hidden_states_0"
        if encode_key not in files:
            msgs.append(f"  {label}: {encode_key}.pt absent (skip if uncond_skip)")
            continue
        if step_key not in files:
            msgs.append(f"  {label}: {step_key}.pt absent (skip if uncond_skip)")
            continue
        a = _load(files[encode_key])
        b = _load(files[step_key])
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            msgs.append(f"  {label}: non-tensor .pt file")
            all_ok = False
            continue
        if a.shape != b.shape:
            msgs.append(f"  {label}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
            all_ok = False
            continue
        diff = (a.float() - b.float()).abs().max().item()
        if diff == 0.0:
            msgs.append(f"  {label}: PASS ({encode_key} == {step_key}, max_abs=0)")
        else:
            msgs.append(f"  {label}: FAIL max_abs_diff={diff:.6e}")
            all_ok = False
    return all_ok, msgs


def _check_reproducibility(files_a: dict[str, str], files_b: dict[str, str]) -> Tuple[bool, list[str]]:
    """(2) every tensor byte-matches between two runs with same seed."""
    msgs: list[str] = []
    all_ok = True
    common = sorted(set(files_a) & set(files_b))
    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))
    msgs.append(f"  common: {len(common)}, only_a: {len(only_a)}, only_b: {len(only_b)}")
    if only_a:
        msgs.append(f"  only in A (first 5): {only_a[:5]}")
        all_ok = False
    if only_b:
        msgs.append(f"  only in B (first 5): {only_b[:5]}")
        all_ok = False

    n_checked = 0
    worst_diff = 0.0
    worst_name = ""
    for name in common:
        a = _load(files_a[name])
        b = _load(files_b[name])
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            continue
        if a.shape != b.shape:
            msgs.append(f"  {name}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
            all_ok = False
            continue
        d = (a.float() - b.float()).abs().max().item()
        if d > worst_diff:
            worst_diff = d
            worst_name = name
        n_checked += 1

    msgs.append(f"  checked {n_checked} tensors; worst max_abs_diff={worst_diff:.6e} @ {worst_name or '(none)'}")
    if worst_diff != 0.0:
        all_ok = False
    return all_ok, msgs


def _check_shape_count_sanity(files: dict[str, str]) -> Tuple[bool, list[str]]:
    """(3) expected counts and consistent shapes for 30B distilled 33-step."""
    msgs: list[str] = []
    all_ok = True

    # Required singletons
    required = ["z_initial", "timesteps"]
    for name in required:
        if name not in files:
            msgs.append(f"  missing: {name}.pt")
            all_ok = False
    # Per-step noise: expect 32 files for 33 sampling steps (i < len-1)
    noise_keys = [k for k in files if k.startswith("step_noise_")]
    msgs.append(f"  step_noise_*: {len(noise_keys)} (expected 32)")
    if len(noise_keys) != 32:
        all_ok = False
    # Encode keys
    enc_cond = [k for k in files if k.startswith("encode_cond_")]
    enc_uncond = [k for k in files if k.startswith("encode_uncond_")]
    msgs.append(f"  encode_cond_*: {len(enc_cond)}, encode_uncond_*: {len(enc_uncond)}")
    if len(enc_cond) == 0:
        all_ok = False
    # Per-step cond/uncond
    step_cond = sorted({int(k.split("_")[0][4:]) for k in files if k.startswith("step") and "_cond_" in k and "_noise" not in k and k.startswith("step") and not k.startswith("step_noise_")})
    step_uncond = sorted({int(k.split("_")[0][4:]) for k in files if k.startswith("step") and "_uncond_" in k and not k.startswith("step_noise_")})
    msgs.append(f"  step<i>_cond_*: {len(step_cond)} distinct i (expected 33); uncond: {len(step_uncond)}")
    if len(step_cond) != 33:
        all_ok = False

    # Shape consistency: z_initial, step_noise_*, step<i>_*_hidden_states
    if "z_initial" in files:
        z_shape = tuple(_load(files["z_initial"]).shape)
        msgs.append(f"  z_initial shape: {z_shape}")
        for k in noise_keys[:3]:  # spot check
            if tuple(_load(files[k]).shape) != z_shape:
                msgs.append(f"  {k} shape != z_initial")
                all_ok = False
        for i in (0, 15, 32):
            key = f"step{i}_cond_hidden_states"
            if key in files and tuple(_load(files[key]).shape) != z_shape:
                msgs.append(f"  {key} shape != z_initial")
                all_ok = False

    return all_ok, msgs


def _check_value_distribution(files: dict[str, str]) -> Tuple[bool, list[str]]:
    """(4) basic sanity on known-distribution tensors."""
    msgs: list[str] = []
    all_ok = True

    if "z_initial" in files:
        z = _load(files["z_initial"]).float()
        m = z.mean().item()
        s = z.std().item()
        msgs.append(f"  z_initial: mean={m:.4f} (expect ~0), std={s:.4f} (expect ~1)")
        if not (-0.05 < m < 0.05):
            msgs.append("    ! z_initial mean out of [-0.05, 0.05]")
            all_ok = False
        if not (0.9 < s < 1.1):
            msgs.append("    ! z_initial std out of [0.9, 1.1]")
            all_ok = False

    if "timesteps" in files:
        ts = _load(files["timesteps"]).float().flatten()
        msgs.append(f"  timesteps: shape={tuple(ts.shape)} first={ts[0].item():.4f} last={ts[-1].item():.4f}")
        diffs = ts[1:] - ts[:-1]
        if (diffs > 1e-6).any().item():
            msgs.append(f"    ! timesteps not monotonically non-increasing (max increase={diffs.max().item():.6f})")
            all_ok = False

    # v_pred spot check
    for i in (0, 16, 32):
        key = f"step{i}_cond_v_pred"
        if key in files:
            v = _load(files[key]).float()
            max_abs = v.abs().max().item()
            has_nan = torch.isnan(v).any().item()
            if has_nan:
                msgs.append(f"  {key}: NaN present")
                all_ok = False
            elif max_abs == 0.0:
                msgs.append(f"  {key}: all-zero")
                all_ok = False
            else:
                msgs.append(f"  {key}: max_abs={max_abs:.4f} (ok)")
    return all_ok, msgs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump_dir")
    p.add_argument("--repro-dir", default=None,
                   help="Second dump dir from the same seed/CLI; required for check (2).")
    args = p.parse_args()

    files = _list_dump(args.dump_dir)
    print(f"Dump dir: {args.dump_dir}  ({len(files)} .pt files)")

    results: list[tuple[str, bool, list[str]]] = []

    ok1, m1 = _check_cross_reference_identity(files)
    results.append(("1. Cross-reference identity", ok1, m1))

    if args.repro_dir:
        repro = _list_dump(args.repro_dir)
        ok2, m2 = _check_reproducibility(files, repro)
        results.append((f"2. Reproducibility vs {args.repro_dir}", ok2, m2))
    else:
        results.append(("2. Reproducibility (SKIPPED — pass --repro-dir)", True, ["  (skipped)"]))

    ok3, m3 = _check_shape_count_sanity(files)
    results.append(("3. Shape + count sanity", ok3, m3))

    ok4, m4 = _check_value_distribution(files)
    results.append(("4. Value distribution sanity", ok4, m4))

    print()
    print("=" * 70)
    all_ok = True
    for name, ok, msgs in results:
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}")
        for m in msgs:
            print(m)
        print()
        if not ok:
            all_ok = False
    print("=" * 70)
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
