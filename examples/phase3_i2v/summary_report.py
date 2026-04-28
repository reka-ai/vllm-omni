#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 (I2V) dump-comparison summary report.

Compares a vllm-omni Phase 3 I2V dump dir against the moonvalley_ai reference
and writes a durable markdown report. Mirrors examples/phase2/summary_report.py
but adds I2V-specific sections covering cond_images / cond_frames /
cond_offsets and the transformer-internal I2V intermediates (t0_emb,
x_after_concat, x_t_mask, x_pre_slice).

Usage:
    python examples/phase3_i2v/summary_report.py \\
        --run <phase3_root>/<tag>/<auto_req_id> \\
        --ref <phase3_root>/ref_single \\
        --tag vllm_l1_single \\
        --out <run>/report.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass

import torch

# Reuse helpers from compare_dumps.py.
_THIS = os.path.dirname(os.path.abspath(__file__))
_COMPARE = os.path.join(_THIS, "..", "offline_inference", "marey")
sys.path.insert(0, _COMPARE)
from compare_dumps import (  # noqa: E402
    _category,
    _diff_tensors,
    _list_dump,
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
    """Return (step, label, tail) for ``step<i>_<label>_<tail>``; else (None, None, None)."""
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
    if x != x:
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
    f.write(f"# Phase 3 I2V summary — {tag}\n\n")
    f.write(f"- Generated: {now}\n")
    f.write(f"- Repo: `{repo}` @ `{branch}` ({commit[:10]})\n")
    f.write(f"- Ref dir: `{ref_dir}` ({len(a_files)} .pt files)\n")
    f.write(f"- Run dir: `{run_dir}` ({len(b_files)} .pt files)\n\n")


def _write_i2v_static(f, a_files: dict, b_files: dict) -> None:
    """Once-per-request I2V tensors: cond_images, cond_frames, cond_offsets."""
    f.write("## 1. I2V conditioning (once per request)\n\n")
    rows: list[tuple[str, str]] = []
    for name in ("cond_images", "cond_frames", "cond_offsets"):
        if name not in a_files or name not in b_files:
            rows.append((name, "_(missing on one side)_"))
            continue
        d = _load_pair(a_files[name], b_files[name], name)
        if d is None:
            rows.append((name, "_(load failed)_"))
            continue
        if name == "cond_offsets":
            # int tensor — report exact-equality + value comparison.
            a = torch.load(a_files[name], map_location="cpu")
            b = torch.load(b_files[name], map_location="cpu")
            equal = bool(torch.equal(a.to(torch.int64), b.to(torch.int64)))
            rows.append((
                name,
                f"shape `{d.shape_a}`, equal={equal}, ref={a.tolist()}, run={b.tolist()}",
            ))
        else:
            rows.append((
                name,
                f"shape `{d.shape_a}`  max_abs `{_fmt(d.max_abs_diff)}`  "
                f"rel `{_fmt(d.relative_avg_diff)}`  cos `{_fmt(d.cosine_sim, '.6f')}`",
            ))
    for name, summary in rows:
        f.write(f"- **`{name}`** — {summary}\n")
    f.write("\n")


def _write_end_to_end(f, a_files: dict, b_files: dict) -> None:
    f.write("## 2. End-to-end (z_final vs ref/latents.pt)\n\n")
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
    f.write("## 3. Per-category summary (worst-first by mean_rel)\n\n")
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
    """Per-step trajectory for v_pred, hidden_states, and I2V intermediates."""
    buckets: dict[str, dict[int, StepDiff]] = {
        "v_pred": defaultdict(StepDiff),
        "hidden_states": defaultdict(StepDiff),
        "t0_emb": defaultdict(StepDiff),
        "x_after_concat": defaultdict(StepDiff),
        "x_t_mask": defaultdict(StepDiff),
        "x_pre_slice": defaultdict(StepDiff),
    }
    for name in common:
        step, label, tail = _split_label(name)
        if step is None or label is None or tail not in buckets:
            continue
        d = _load_pair(a_files[name], b_files[name], name)
        if d is None:
            continue
        entry = buckets[tail][step]
        entry.step = step
        if label == "cond":
            entry.cond = d
        elif label == "uncond":
            entry.uncond = d

    def _emit(title: str, bucket: dict[int, StepDiff]) -> None:
        if not bucket:
            return
        f.write(f"### {title}\n\n")
        f.write("| step | cond_rel | cond_max_abs | cond_cos | uncond_rel | uncond_max_abs | uncond_cos |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for step in sorted(bucket):
            e = bucket[step]
            c_rel = _fmt(e.cond.relative_avg_diff) if e.cond else "—"
            c_max = _fmt(e.cond.max_abs_diff) if e.cond else "—"
            c_cos = _fmt(e.cond.cosine_sim, ".6f") if e.cond else "—"
            u_rel = _fmt(e.uncond.relative_avg_diff) if e.uncond else "—"
            u_max = _fmt(e.uncond.max_abs_diff) if e.uncond else "—"
            u_cos = _fmt(e.uncond.cosine_sim, ".6f") if e.uncond else "—"
            f.write(f"| {step} | {c_rel} | {c_max} | {c_cos} | {u_rel} | {u_max} | {u_cos} |\n")
        f.write("\n")

    f.write("## 4. Per-step trajectory\n\n")
    _emit("Transformer v_pred", buckets["v_pred"])
    _emit("Transformer hidden_states (input)", buckets["hidden_states"])
    _emit("I2V t0_emb (post-t_block)", buckets["t0_emb"])
    _emit("I2V x_after_concat (pre-SP-shard)", buckets["x_after_concat"])
    _emit("I2V x_t_mask (pre-SP-shard)", buckets["x_t_mask"])
    _emit("I2V x_pre_slice (post-final_layer post-gather)", buckets["x_pre_slice"])


def _write_worst_offenders(f, diffs_by_cat: dict, top_k: int = 3) -> None:
    f.write(f"## 5. Worst offenders per category (top-{top_k} by rel)\n\n")
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
    p.add_argument("--run", required=True, help="Phase 3 dump dir (the actual per-request subdir)")
    p.add_argument("--ref", required=True, help="Phase 3 reference dump dir (mv side)")
    p.add_argument("--out", default=None, help="Output markdown file (default: <run>/report.md)")
    p.add_argument("--tag", default=None, help="Short label for the report header")
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

    diffs_by_cat: dict[str, list[TensorDiff]] = defaultdict(list)
    for name in common:
        d = _load_pair(a_files[name], b_files[name], name)
        if d is not None:
            diffs_by_cat[_category(name)].append(d)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        _write_header(f, tag, ref_dir, run_dir, a_files, b_files)
        _write_i2v_static(f, a_files, b_files)
        _write_end_to_end(f, a_files, b_files)
        _write_category_summary(f, diffs_by_cat)
        _write_per_step(f, common, a_files, b_files)
        _write_worst_offenders(f, diffs_by_cat)
        _write_schema_parity(f, a_files, b_files)

    print(f"[summary] wrote {out_path}")
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
