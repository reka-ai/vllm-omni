#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diff two dump directories produced by ``DumpMixin`` (vllm-omni side) and/or
``marey_reference_inference_dump.py`` (moonvalley_ai side).

Schema-agnostic: walks both dirs, intersects basenames, reports per-tensor
metrics. Tensors only present on one side are listed as "only in A/B".

Per-tensor metrics:
    max_abs_diff       max(|a - b|)
    mean_abs_diff      mean(|a - b|)
    relative_avg_diff  mean(|a - b|) / mean(|a|) (relative to A's magnitude)
    cosine_sim         cosine similarity over flattened tensors

Acceptance for this tool itself: diffing two identical dumps reports zero
divergence on every tensor (verified inline by --self-test).

Acceptance envelope from the 30B Notion correctness doc (per-tensor):
    text encoders / extra features      rel diff < 1%
    v_pred step 0                       ~0.1% rel avg diff
    v_pred final step                   ≤ ~20% rel avg diff (SP slop)

Usage:
    python compare_dumps.py --a ./dump_ref_30b/ref_run --b ./dump_vllm_30b/<req>
    python compare_dumps.py --self-test ./dump_ref_30b/ref_run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Diff metrics
# ---------------------------------------------------------------------------


@dataclass
class TensorDiff:
    name: str
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float
    relative_avg_diff: float
    cosine_sim: float

    @property
    def shape_match(self) -> bool:
        return self.shape_a == self.shape_b


def _diff_tensors(a: torch.Tensor, b: torch.Tensor, name: str) -> TensorDiff:
    shape_a = tuple(a.shape)
    shape_b = tuple(b.shape)
    if shape_a != shape_b:
        return TensorDiff(
            name=name,
            shape_a=shape_a,
            shape_b=shape_b,
            max_abs_diff=float("nan"),
            mean_abs_diff=float("nan"),
            relative_avg_diff=float("nan"),
            cosine_sim=float("nan"),
        )

    a_f = a.detach().to(torch.float64).flatten()
    b_f = b.detach().to(torch.float64).flatten()
    diff = (a_f - b_f).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    a_mag = a_f.abs().mean().item()
    rel = mean_abs / a_mag if a_mag > 0 else float("nan")

    a_norm = a_f.norm()
    b_norm = b_f.norm()
    if a_norm > 0 and b_norm > 0:
        cos = (a_f @ b_f / (a_norm * b_norm)).item()
    else:
        cos = float("nan")

    return TensorDiff(
        name=name,
        shape_a=shape_a,
        shape_b=shape_b,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        relative_avg_diff=rel,
        cosine_sim=cos,
    )


# ---------------------------------------------------------------------------
# Grouping by step index
# ---------------------------------------------------------------------------


_STEP_I_PAT = re.compile(r"^step(\d+)_(cond|uncond)_(.+)$")


def _step_index(name: str) -> int | None:
    """Extract a step index from a tensor basename, if present.

    Recognised patterns:
        step_noise_<i>             -> i
        step<i>_<label>_<field>    -> i
    """
    if name.startswith("step_noise_"):
        try:
            return int(name[len("step_noise_"):])
        except ValueError:
            return None
    m = _STEP_I_PAT.match(name)
    if m:
        return int(m.group(1))
    return None


def _category(name: str) -> str:
    if name == "z_initial":
        return "z_initial"
    if name == "z_final":
        return "z_final"
    if name == "timesteps":
        return "timesteps_schedule"
    if name.startswith("step_noise_"):
        return "step_noise"
    # Text encoder outputs: encode_<label>_<field>[_<j>]
    if name.startswith("encode_"):
        if "_seq_cond" in name:
            return "text_seq_cond"
        if "_seq_mask" in name:
            return "text_seq_mask"
        if "_vector_cond" in name:
            return "text_vector_cond"
        return "text_misc"
    # Per-step transformer I/O: step<i>_<label>_<field>
    m = _STEP_I_PAT.match(name)
    if m:
        tail = m.group(3)
        if tail == "hidden_states":
            return "transformer_hidden_states"
        if tail == "timestep":
            return "transformer_timestep"
        if tail == "v_pred" or tail == "model_out":
            return "transformer_v_pred"
        if tail.startswith("encoder_hidden_states"):
            return "transformer_encoder_hidden_states"
        if tail.startswith("vector_cond"):
            return "transformer_vector_cond"
        # Normalize extra_features: mv wrapper dumps as kw_<k>,
        # vllm-omni DumpMixin dumps as extra_<k>. Same bucket.
        if tail.startswith("extra_") or tail.startswith("kw_"):
            return "transformer_extra"
        return "transformer_misc"
    return "other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_KW_EXTRA_PAT = re.compile(r"^(step\d+_(?:cond|uncond)_)kw_(.+)$")


def _canonical_name(name: str) -> str:
    """Canonicalize across mv / vllm-omni naming for cross-codebase diff.

    - ``step<i>_<label>_kw_<key>`` (mv fallthrough) → ``step<i>_<label>_extra_<key>``
      (vllm-omni DumpMixin's name for ``extra_features`` dict entries).
    - ``latents`` (mv's ``--save-latents`` output) → ``z_final`` (vllm-omni's
      DumpMixin name for the pre-VAE final latent).
    """
    if name == "latents":
        return "z_final"
    m = _KW_EXTRA_PAT.match(name)
    if m:
        return f"{m.group(1)}extra_{m.group(2)}"
    return name


def _list_dump(path: str) -> dict[str, str]:
    """Map canonical basename (without .pt) → full path for every .pt file."""
    out = {}
    if not os.path.isdir(path):
        raise SystemExit(f"Dump dir not found: {path}")
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".pt"):
            out[_canonical_name(fname[:-3])] = os.path.join(path, fname)
    return out


def _print_table(diffs: list[TensorDiff], header: str) -> None:
    if not diffs:
        return
    print(f"\n{header}")
    print(f"  {'name':<48}  {'shape':<22}  {'max_abs':>10}  {'mean_abs':>10}  {'rel':>10}  {'cos_sim':>10}")
    for d in diffs:
        shape_str = (
            f"{d.shape_a}" if d.shape_match else f"!! {d.shape_a} vs {d.shape_b}"
        )
        print(
            f"  {d.name:<48}  {shape_str:<22}  "
            f"{d.max_abs_diff:>10.3e}  {d.mean_abs_diff:>10.3e}  "
            f"{d.relative_avg_diff:>10.3e}  {d.cosine_sim:>10.6f}"
        )


def _summary_by_step(diffs: list[TensorDiff], category: str) -> None:
    by_step: dict[int, list[TensorDiff]] = defaultdict(list)
    no_step: list[TensorDiff] = []
    for d in diffs:
        s = _step_index(d.name)
        if s is None:
            no_step.append(d)
        else:
            by_step[s].append(d)

    if not by_step:
        return
    print(f"\n  {category} per-step relative_avg_diff (mean across tensors):")
    print(f"    {'step':>5}  {'mean_rel':>10}  {'max_rel':>10}  {'n_tensors':>10}")
    for step in sorted(by_step.keys()):
        ds = by_step[step]
        rels = [d.relative_avg_diff for d in ds if d.shape_match and not (d.relative_avg_diff != d.relative_avg_diff)]
        if not rels:
            continue
        print(f"    {step:>5}  {sum(rels) / len(rels):>10.3e}  {max(rels):>10.3e}  {len(rels):>10}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=str, help="First dump dir (e.g. moonvalley_ai reference)")
    p.add_argument("--b", type=str, help="Second dump dir (e.g. vllm-omni)")
    p.add_argument("--self-test", type=str, default=None, metavar="DIR",
                   help="Diff a dump dir against itself; expects all-zero diffs.")
    p.add_argument("--list-categories", type=str, default=None, metavar="DIR",
                   help="List category counts for a single dump dir; no diff.")
    p.add_argument("--show-all", action="store_true",
                   help="Print every tensor (default: only summary + worst-N per category)")
    p.add_argument("--worst-n", type=int, default=5,
                   help="Per-category, show the N tensors with largest relative diff (default 5)")
    args = p.parse_args()

    if args.list_categories:
        files = _list_dump(args.list_categories)
        by_cat: dict[str, int] = defaultdict(int)
        examples: dict[str, list[str]] = defaultdict(list)
        for name in files:
            cat = _category(name)
            by_cat[cat] += 1
            if len(examples[cat]) < 2:
                examples[cat].append(name)
        print(f"Categories in {args.list_categories}  ({len(files)} tensors):")
        print(f"  {'category':<35}  {'count':>6}   examples")
        for cat in sorted(by_cat, key=lambda c: (-by_cat[c], c)):
            ex = ", ".join(examples[cat])
            print(f"  {cat:<35}  {by_cat[cat]:>6}   {ex}")
        if by_cat.get("other", 0) > 0:
            print(f"\n!! {by_cat['other']} tensor(s) fell into 'other' — classifier needs a new branch.")
            return 1
        return 0

    if args.self_test:
        a_path = b_path = args.self_test
    else:
        if not args.a or not args.b:
            p.error("--a and --b are required (or use --self-test / --list-categories)")
        a_path, b_path = args.a, args.b

    a_files = _list_dump(a_path)
    b_files = _list_dump(b_path)

    common = sorted(set(a_files) & set(b_files))
    only_a = sorted(set(a_files) - set(b_files))
    only_b = sorted(set(b_files) - set(a_files))

    print(f"A: {a_path}  ({len(a_files)} files)")
    print(f"B: {b_path}  ({len(b_files)} files)")
    print(f"Common: {len(common)}, only in A: {len(only_a)}, only in B: {len(only_b)}")

    if only_a:
        print("\nOnly in A (first 10):")
        for n in only_a[:10]:
            print(f"  {n}")
    if only_b:
        print("\nOnly in B (first 10):")
        for n in only_b[:10]:
            print(f"  {n}")

    # Diff every common tensor
    by_category: dict[str, list[TensorDiff]] = defaultdict(list)
    shape_mismatches: list[TensorDiff] = []
    for name in common:
        a = torch.load(a_files[name], map_location="cpu")
        b = torch.load(b_files[name], map_location="cpu")
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            print(f"  skipping non-tensor: {name}")
            continue
        d = _diff_tensors(a, b, name)
        if not d.shape_match:
            shape_mismatches.append(d)
        by_category[_category(name)].append(d)

    if shape_mismatches:
        print("\n!! Shape mismatches:")
        for d in shape_mismatches:
            print(f"  {d.name}  A={d.shape_a}  B={d.shape_b}")

    # Per-category summary
    print("\n=== Per-category summary ===")
    print(f"  {'category':<26}  {'n':>4}  {'mean_rel':>10}  {'max_rel':>10}  {'min_cos':>10}")
    for cat in sorted(by_category):
        ds = [d for d in by_category[cat] if d.shape_match]
        if not ds:
            continue
        rels = [d.relative_avg_diff for d in ds if not (d.relative_avg_diff != d.relative_avg_diff)]
        coss = [d.cosine_sim for d in ds if not (d.cosine_sim != d.cosine_sim)]
        mean_rel = sum(rels) / len(rels) if rels else float("nan")
        max_rel = max(rels) if rels else float("nan")
        min_cos = min(coss) if coss else float("nan")
        print(f"  {cat:<26}  {len(ds):>4}  {mean_rel:>10.3e}  {max_rel:>10.3e}  {min_cos:>10.6f}")

    # Per-step trajectory for v_pred and step_noise (the most informative)
    for cat in ("transformer_v_pred", "step_noise", "transformer_hidden_states"):
        if cat in by_category:
            _summary_by_step(by_category[cat], cat)

    # Worst-N per category
    for cat in sorted(by_category):
        ds = [d for d in by_category[cat] if d.shape_match]
        if not ds:
            continue
        ds.sort(key=lambda d: d.relative_avg_diff if d.relative_avg_diff == d.relative_avg_diff else -1, reverse=True)
        if args.show_all:
            _print_table(ds, f"{cat} (all):")
        else:
            _print_table(ds[: args.worst_n], f"{cat} (worst-{args.worst_n} by rel diff):")

    if args.self_test:
        # Verify everything zeroed out
        bad = [d for d in (sum(by_category.values(), [])) if d.shape_match and d.max_abs_diff > 0]
        if bad:
            print(f"\n!! Self-test FAILED: {len(bad)} tensors have nonzero diff:")
            for d in bad[:5]:
                print(f"  {d.name}  max_abs={d.max_abs_diff}")
            return 1
        print("\nSelf-test PASSED: all tensors diff to zero.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
