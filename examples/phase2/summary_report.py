#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 dump-comparison summary report.

Compares a vllm-omni Phase 2 dump dir against the Phase 1 moonvalley reference
and writes a durable markdown report. Designed to be run after every Phase 2
inference (L1 / L2 / L3 / base) so progress across levels is diffable.

Usage:
    python examples/phase2/summary_report.py \\
        --run /mnt/localdisk/vllm_omni_storage/phase2/vllm_runA/vllm_runA \\
        [--ref /mnt/localdisk/vllm_omni_storage/phase1/ref_30b] \\
        [--out <run>/report.md] \\
        [--tag vllm_runA]

Sections:
  1. Header — run tag, timestamps, dir sizes, env (git commit)
  2. End-to-end — z_final vs latents.pt (max_abs, rel, cosine)
  3. Per-category summary — one row per category, ordered worst-first
  4. Per-step trajectory — v_pred + hidden_states, cond vs uncond columns
  5. Worst offenders — top-3 per category
  6. Schema parity — only_a / only_b lists (truncated)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

# Reuse categorization / diff helpers from compare_dumps.py
_THIS = os.path.dirname(os.path.abspath(__file__))
_COMPARE = os.path.join(_THIS, "..", "offline_inference", "marey")
sys.path.insert(0, _COMPARE)
from compare_dumps import (  # noqa: E402
    _category,
    _diff_tensors,
    _list_dump,
    _step_index,
    TensorDiff,
)


@dataclass
class StepDiff:
    step: int = -1
    cond: TensorDiff | None = None
    uncond: TensorDiff | None = None


def _cmd(args: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "<unavailable>"


def _split_label(name: str) -> tuple[int | None, str | None, str | None]:
    """Return (step, label, tail) for a step<i>_<label>_<tail> name; else (None, None, None)."""
    if not name.startswith("step") or name.startswith("step_noise_"):
        return None, None, None
    try:
        _, _rest = name.split("step", 1)
        step_str, after = _rest.split("_", 1)
        step = int(step_str)
    except (ValueError, IndexError):
        return None, None, None
    parts = after.split("_", 1)
    if len(parts) != 2:
        return None, None, None
    label, tail = parts
    if label not in ("cond", "uncond", "unknown"):
        return None, None, None
    return step, label, tail


def _load_pair(a_path: str, b_path: str, name: str) -> TensorDiff | None:
    a = torch.load(a_path, map_location="cpu")
    b = torch.load(b_path, map_location="cpu")
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None
    return _diff_tensors(a, b, name)


def _fmt(x: float, spec: str = ".3e") -> str:
    if x != x:  # NaN
        return "   nan"
    return format(x, spec)


def _write_header(f, tag: str, ref_dir: str, run_dir: str,
                  a_files: dict, b_files: dict) -> None:
    now = _dt.datetime.now().isoformat(timespec="seconds")
    repo = _THIS
    while repo and repo != "/" and not os.path.isdir(os.path.join(repo, ".git")):
        repo = os.path.dirname(repo)
    commit = _cmd(["git", "rev-parse", "HEAD"], cwd=repo) if repo else "<unavailable>"
    branch = _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo) if repo else "<unavailable>"
    f.write(f"# Phase 2 summary — {tag}\n\n")
    f.write(f"- Generated: {now}\n")
    f.write(f"- Repo: `{repo}` @ `{branch}` ({commit[:10]})\n")
    f.write(f"- Ref dir: `{ref_dir}` ({len(a_files)} .pt files)\n")
    f.write(f"- Run dir: `{run_dir}` ({len(b_files)} .pt files)\n\n")


def _write_end_to_end(f, a_files: dict, b_files: dict) -> None:
    f.write("## 1. End-to-end (z_final vs ref/latents.pt)\n\n")
    if "z_final" not in a_files or "z_final" not in b_files:
        f.write("_z_final missing on one side — skipped._\n\n")
        return
    d = _load_pair(a_files["z_final"], b_files["z_final"], "z_final")
    if d is None:
        f.write("_z_final failed to load — skipped._\n\n")
        return
    f.write(f"- shape: `{d.shape_a}`\n")
    f.write(f"- max_abs_diff: `{_fmt(d.max_abs_diff)}`\n")
    f.write(f"- mean_abs_diff: `{_fmt(d.mean_abs_diff)}`\n")
    f.write(f"- relative_avg_diff: `{_fmt(d.relative_avg_diff)}`\n")
    f.write(f"- cosine_sim: `{_fmt(d.cosine_sim, '.6f')}`\n\n")


def _write_category_summary(f, diffs_by_cat: dict) -> None:
    f.write("## 2. Per-category summary (worst-first by mean_rel)\n\n")
    f.write("| category | n | mean_rel | max_rel | min_cos |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    rows: list[tuple[str, int, float, float, float]] = []
    for cat, ds in diffs_by_cat.items():
        ds = [d for d in ds if d.shape_match]
        if not ds:
            continue
        rels = [d.relative_avg_diff for d in ds if d.relative_avg_diff == d.relative_avg_diff]
        coss = [d.cosine_sim for d in ds if d.cosine_sim == d.cosine_sim]
        mean_rel = sum(rels) / len(rels) if rels else float("nan")
        max_rel = max(rels) if rels else float("nan")
        min_cos = min(coss) if coss else float("nan")
        rows.append((cat, len(ds), mean_rel, max_rel, min_cos))
    rows.sort(key=lambda r: (-r[2] if r[2] == r[2] else 0))
    for cat, n, mean_rel, max_rel, min_cos in rows:
        f.write(f"| `{cat}` | {n} | {_fmt(mean_rel)} | {_fmt(max_rel)} | {_fmt(min_cos, '.6f')} |\n")
    f.write("\n")


def _write_per_step(f, common: list[str], a_files: dict, b_files: dict) -> None:
    """Per-step trajectory for v_pred and hidden_states, cond vs uncond."""
    v_pred: dict[int, StepDiff] = defaultdict(StepDiff)
    hs: dict[int, StepDiff] = defaultdict(StepDiff)
    for name in common:
        step, label, tail = _split_label(name)
        if step is None or label is None:
            continue
        if tail not in ("v_pred", "hidden_states"):
            continue
        d = _load_pair(a_files[name], b_files[name], name)
        if d is None:
            continue
        bucket = v_pred if tail == "v_pred" else hs
        entry = bucket[step]
        entry.step = step
        if label == "cond":
            entry.cond = d
        elif label == "uncond":
            entry.uncond = d

    def _emit(title: str, bucket: dict[int, StepDiff]) -> None:
        if not bucket:
            return
        f.write(f"### {title}\n\n")
        f.write("| step | cond_rel | cond_max_abs | uncond_rel | uncond_max_abs |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for step in sorted(bucket):
            e = bucket[step]
            c_rel = _fmt(e.cond.relative_avg_diff) if e.cond else "—"
            c_max = _fmt(e.cond.max_abs_diff) if e.cond else "—"
            u_rel = _fmt(e.uncond.relative_avg_diff) if e.uncond else "—"
            u_max = _fmt(e.uncond.max_abs_diff) if e.uncond else "—"
            f.write(f"| {step} | {c_rel} | {c_max} | {u_rel} | {u_max} |\n")
        f.write("\n")

    f.write("## 3. Per-step trajectory\n\n")
    _emit("Transformer v_pred", v_pred)
    _emit("Transformer hidden_states (input)", hs)


def _write_worst_offenders(f, diffs_by_cat: dict, top_k: int = 3) -> None:
    f.write(f"## 4. Worst offenders per category (top-{top_k} by rel)\n\n")
    for cat in sorted(diffs_by_cat):
        ds = [d for d in diffs_by_cat[cat] if d.shape_match]
        if not ds:
            continue
        ds.sort(key=lambda d: d.relative_avg_diff if d.relative_avg_diff == d.relative_avg_diff else -1, reverse=True)
        f.write(f"### `{cat}`\n\n")
        f.write("| name | shape | max_abs | mean_abs | rel | cos_sim |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for d in ds[:top_k]:
            f.write(f"| `{d.name}` | `{d.shape_a}` | {_fmt(d.max_abs_diff)} | {_fmt(d.mean_abs_diff)} | {_fmt(d.relative_avg_diff)} | {_fmt(d.cosine_sim, '.6f')} |\n")
        f.write("\n")


def _write_phase2_findings(f) -> None:
    """Static section embedded in every report — the verification checklist
    and root-cause analysis from the Phase 2 investigation. See
    examples/phase2/PHASE2_FINDINGS.md for the full report.
    """
    f.write("## 5. Phase 2 findings — known divergence sources\n\n")
    f.write("The numbers above sit on top of a known noise floor. The investigation\n")
    f.write("(see `examples/phase2/PHASE2_FINDINGS.md`) ruled out the following as\n")
    f.write("sources of the residual ~0.84% per-step `transformer_v_pred` divergence:\n\n")
    f.write("| Hypothesis | Verdict | Evidence |\n")
    f.write("|---|---|---|\n")
    f.write("| SwiGLU gate/up swap | ✅ ruled out | `_MLP_WEIGHT_MAP` remap aligns silu to fc1_g on both sides |\n")
    f.write("| `LlamaRMSNorm` eps / fp32 promotion | ✅ ruled out | Implementations line-by-line identical (eps=1e-6, fp32 cast) |\n")
    f.write("| `apply_rope` / `apply_rotary_emb` | ✅ ruled out | Implementations line-by-line identical |\n")
    f.write("| `t2i_modulate` | ✅ ruled out | Both compute `x * (1 + scale) + shift` |\n")
    f.write("| Modulation Linear path | ✅ ruled out | Both use `nn.Linear(h, 6h)` with identical chunk order (use_block_v2=true) |\n")
    f.write("| Block forward order | ✅ ruled out | norm-mod-attn-gate-norm-mod-mlp-gate identical on both sides |\n")
    f.write("| Attention masking semantics | ✅ ruled out | Both use mask-via-zeroing + fully-dense flash_attn |\n")
    f.write("| Weight loading completeness | ✅ ruled out | Zero `Skipping weight` warnings; all checkpoint keys accepted |\n")
    f.write("| **ulysses degree** (SP collective bf16 noise) | ✅ **ruled out** | ul=4 vs ul=8 produced bit-essentially-identical v_pred (mean_rel changed by 0.0018% absolute) |\n")
    f.write("| Per-rank SP shard alignment | ⚠️ benign permutation | Rank-0 dumps differ in *order* (cos_sim 0.999998 on sorted token norms) but hold the same content set; full-tensor v_pred is unaffected |\n")
    f.write("| **flash_attn_3 build mismatch** | 🎯 **most likely cause** | vllm-omni venv ships PyPI's `fa3_fwd 0.0.2`; mv ships internal S3's `flash_attn_3 3.0.0b1`. Independent FA3 forks; can't be cross-installed due to torch C++ ABI lock |\n")
    f.write("\n")
    f.write("The FA3 mismatch is the only remaining numerical primitive that could\n")
    f.write("produce the observed signature: **deterministic** (same value across runs)\n")
    f.write("AND **invariant to ulysses degree** (kernel-level, not collective-level).\n")
    f.write("Direct verification requires building `flash_attn_3` from source against\n")
    f.write("vllm-omni's torch 2.10 + cu129; not done in this Phase 2.\n\n")


def _write_schema_parity(f, a_files: dict, b_files: dict) -> None:
    only_a = sorted(set(a_files) - set(b_files))
    only_b = sorted(set(b_files) - set(a_files))
    f.write("## 6. Schema parity\n\n")
    f.write(f"- common: {len(set(a_files) & set(b_files))}\n")
    f.write(f"- only in ref: {len(only_a)}\n")
    f.write(f"- only in run: {len(only_b)}\n\n")
    if only_a:
        f.write(f"<details><summary>only in ref ({len(only_a)})</summary>\n\n```\n")
        for n in only_a[:50]:
            f.write(f"{n}\n")
        if len(only_a) > 50:
            f.write(f"... {len(only_a) - 50} more\n")
        f.write("```\n</details>\n\n")
    if only_b:
        f.write(f"<details><summary>only in run ({len(only_b)})</summary>\n\n```\n")
        for n in only_b[:50]:
            f.write(f"{n}\n")
        if len(only_b) > 50:
            f.write(f"... {len(only_b) - 50} more\n")
        f.write("```\n</details>\n\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Phase 2 dump dir (the actual per-request subdir)")
    p.add_argument("--ref", default="/mnt/localdisk/vllm_omni_storage/phase1/ref_30b",
                   help="Phase 1 reference dump dir (default: ref_30b)")
    p.add_argument("--out", default=None,
                   help="Output markdown file (default: <run>/report.md)")
    p.add_argument("--tag", default=None,
                   help="Short label for the report header (default: basename of --run)")
    args = p.parse_args()

    run_dir = os.path.abspath(args.run)
    ref_dir = os.path.abspath(args.ref)
    out_path = args.out or os.path.join(run_dir, "report.md")
    tag = args.tag or os.path.basename(run_dir.rstrip("/"))

    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run dir not found: {run_dir}")
    if not os.path.isdir(ref_dir):
        raise SystemExit(f"Ref dir not found: {ref_dir}")

    a_files = _list_dump(ref_dir)
    b_files = _list_dump(run_dir)
    common = sorted(set(a_files) & set(b_files))

    # Per-tensor diff, grouped by category
    diffs_by_cat: dict[str, list[TensorDiff]] = defaultdict(list)
    for name in common:
        d = _load_pair(a_files[name], b_files[name], name)
        if d is not None:
            diffs_by_cat[_category(name)].append(d)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        _write_header(f, tag, ref_dir, run_dir, a_files, b_files)
        _write_end_to_end(f, a_files, b_files)
        _write_category_summary(f, diffs_by_cat)
        _write_per_step(f, common, a_files, b_files)
        _write_worst_offenders(f, diffs_by_cat)
        _write_phase2_findings(f)
        _write_schema_parity(f, a_files, b_files)

    print(f"[summary] wrote {out_path}")
    # Print a terse one-line summary to stdout too, convenient for CI.
    vp = diffs_by_cat.get("transformer_v_pred", [])
    vp_rels = [d.relative_avg_diff for d in vp if d.shape_match and d.relative_avg_diff == d.relative_avg_diff]
    vp_mean = sum(vp_rels) / len(vp_rels) if vp_rels else float("nan")
    if "z_final" in a_files and "z_final" in b_files:
        zf = _load_pair(a_files["z_final"], b_files["z_final"], "z_final")
        zf_rel = zf.relative_avg_diff if zf else float("nan")
    else:
        zf_rel = float("nan")
    print(f"[summary] tag={tag}  v_pred_mean_rel={_fmt(vp_mean)}  z_final_rel={_fmt(zf_rel)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
