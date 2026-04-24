# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Diff two Marey diffusion-pipeline dumps.

See :mod:`vllm_omni.diffusion.models.marey._dumper` for the dump layout.
Invoke via::

    python -m vllm_omni.diffusion.models.marey.compare_dumps \\
        --a /dumps/old/<request_id> \\
        --b /dumps/new/<request_id>

Exits non-zero when any tensor pair fails ``torch.allclose`` at the chosen
``--atol``/``--rtol``, or when a file is present in only one side.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


_MAX_FOR_QUANTILE = 16_000_000


@dataclass
class DiffRow:
    rel_path: str
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    dtype_a: str
    dtype_b: str
    numel: int
    max_abs: float
    mean_abs: float
    p99_abs: float
    max_rel: float
    mean_rel: float
    is_close: bool
    note: str = ""


def _pt_files(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in root.rglob("*.pt")}


def _load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)


def _tensor_stats(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> DiffRow:
    shape_a = tuple(a.shape) if a.ndim > 0 else ()
    shape_b = tuple(b.shape) if b.ndim > 0 else ()
    if shape_a != shape_b:
        return DiffRow(
            rel_path="",
            shape_a=shape_a,
            shape_b=shape_b,
            dtype_a=str(a.dtype),
            dtype_b=str(b.dtype),
            numel=a.numel(),
            max_abs=float("inf"),
            mean_abs=float("inf"),
            p99_abs=float("inf"),
            max_rel=float("inf"),
            mean_rel=float("inf"),
            is_close=False,
            note="shape mismatch",
        )
    af = a.detach().float()
    bf = b.detach().float()
    diff = (af - bf).abs()
    flat = diff.flatten()
    if flat.numel() > _MAX_FOR_QUANTILE:
        idx = torch.randperm(flat.numel())[:_MAX_FOR_QUANTILE]
        flat_sampled = flat[idx]
    else:
        flat_sampled = flat
    p99 = float(flat_sampled.quantile(0.99)) if flat_sampled.numel() > 0 else 0.0
    # Relative diff: |a-b| / mean(|a|, |b|); ignore positions where both are 0.
    denom = (af.abs() + bf.abs()) * 0.5
    mask = denom > 0
    if mask.any():
        rel = diff[mask] / denom[mask]
        max_rel = float(rel.max())
        mean_rel = float(rel.mean())
    else:
        max_rel = 0.0
        mean_rel = 0.0
    close = torch.allclose(af, bf, atol=atol, rtol=rtol)
    return DiffRow(
        rel_path="",
        shape_a=shape_a,
        shape_b=shape_b,
        dtype_a=str(a.dtype),
        dtype_b=str(b.dtype),
        numel=a.numel(),
        max_abs=float(diff.max()) if diff.numel() > 0 else 0.0,
        mean_abs=float(diff.mean()) if diff.numel() > 0 else 0.0,
        p99_abs=p99,
        max_rel=max_rel,
        mean_rel=mean_rel,
        is_close=bool(close),
    )


def compare_dumps(
    dir_a: Path,
    dir_b: Path,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> tuple[list[DiffRow], list[str], list[str]]:
    """Compare two dump directories.

    Returns ``(rows, only_in_a, only_in_b)``. Rows are sorted by ``max_abs``
    descending. ``only_in_*`` are relative paths present in only one side.
    """
    files_a = _pt_files(dir_a)
    files_b = _pt_files(dir_b)
    common = sorted(files_a.keys() & files_b.keys())
    only_in_a = sorted(set(files_a) - set(files_b))
    only_in_b = sorted(set(files_b) - set(files_a))

    rows: list[DiffRow] = []
    for rel in common:
        try:
            a = _load(files_a[rel])
            b = _load(files_b[rel])
        except Exception as e:
            rows.append(
                DiffRow(
                    rel_path=rel,
                    shape_a=(),
                    shape_b=(),
                    dtype_a="?",
                    dtype_b="?",
                    numel=0,
                    max_abs=float("inf"),
                    mean_abs=float("inf"),
                    p99_abs=float("inf"),
                    max_rel=float("inf"),
                    mean_rel=float("inf"),
                    is_close=False,
                    note=f"load error: {type(e).__name__}: {e}",
                )
            )
            continue
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            rows.append(
                DiffRow(
                    rel_path=rel,
                    shape_a=(),
                    shape_b=(),
                    dtype_a=type(a).__name__,
                    dtype_b=type(b).__name__,
                    numel=0,
                    max_abs=float("inf"),
                    mean_abs=float("inf"),
                    p99_abs=float("inf"),
                    max_rel=float("inf"),
                    mean_rel=float("inf"),
                    is_close=False,
                    note="non-tensor payload",
                )
            )
            continue
        row = _tensor_stats(a, b, atol=atol, rtol=rtol)
        row.rel_path = rel
        rows.append(row)

    rows.sort(key=lambda r: r.max_abs, reverse=True)
    return rows, only_in_a, only_in_b


def _format_table(rows: list[DiffRow]) -> str:
    if not rows:
        return "(no common tensors)\n"
    header = (
        f"{'close':<6}{'max_abs':>12}{'mean_abs':>12}{'p99_abs':>12}"
        f"{'max_rel':>12}{'mean_rel':>12}  {'shape':<24}{'dtype a/b':<20}path"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        close_mark = "OK" if r.is_close else "FAIL"
        shape_s = "x".join(str(d) for d in r.shape_a) or "scalar"
        dtype_s = f"{r.dtype_a.replace('torch.', '')}/{r.dtype_b.replace('torch.', '')}"
        line = (
            f"{close_mark:<6}"
            f"{r.max_abs:12.4g}"
            f"{r.mean_abs:12.4g}"
            f"{r.p99_abs:12.4g}"
            f"{r.max_rel:12.4g}"
            f"{r.mean_rel:12.4g}"
            f"  {shape_s:<24}{dtype_s:<20}{r.rel_path}"
        )
        if r.note:
            line += f"  [{r.note}]"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _format_summary(rows: list[DiffRow]) -> str:
    if not rows:
        return "(no common tensors)\n"
    passed = sum(1 for r in rows if r.is_close)
    failed = len(rows) - passed
    worst = rows[0]
    by_category: dict[str, list[DiffRow]] = {}
    for r in rows:
        head = r.rel_path.split("/", 1)[0] if "/" in r.rel_path else "(root)"
        by_category.setdefault(head, []).append(r)
    cat_lines = []
    for cat, cat_rows in sorted(by_category.items()):
        max_abs = max((r.max_abs for r in cat_rows), default=0.0)
        max_rel = max((r.max_rel for r in cat_rows), default=0.0)
        fail = sum(1 for r in cat_rows if not r.is_close)
        cat_lines.append(
            f"  {cat:<24} n={len(cat_rows):<5} fail={fail:<4} "
            f"max_abs={max_abs:.4g} max_rel={max_rel:.4g}"
        )
    return (
        f"{passed} close, {failed} fail (of {len(rows)} tensors)\n"
        f"worst: max_abs={worst.max_abs:.4g} max_rel={worst.max_rel:.4g} at {worst.rel_path}\n"
        "by category:\n"
        + "\n".join(cat_lines)
        + "\n"
    )


def _compare_meta(dir_a: Path, dir_b: Path) -> str:
    lines = []
    for label, root in (("A", dir_a), ("B", dir_b)):
        p = root / "meta.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                lines.append(f"[{label}] meta.json: {data}")
            except Exception as e:
                lines.append(f"[{label}] meta.json unreadable: {e}")
    return "\n".join(lines) + ("\n" if lines else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="First dump directory (a single request).")
    ap.add_argument("--b", type=Path, required=True, help="Second dump directory (a single request).")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="Write the report to this file too.")
    args = ap.parse_args(argv)

    if not args.a.is_dir():
        print(f"--a is not a directory: {args.a}", file=sys.stderr)
        return 2
    if not args.b.is_dir():
        print(f"--b is not a directory: {args.b}", file=sys.stderr)
        return 2

    rows, only_a, only_b = compare_dumps(args.a, args.b, atol=args.atol, rtol=args.rtol)

    parts = [_compare_meta(args.a, args.b)]
    if args.summary_only:
        parts.append(_format_summary(rows))
    else:
        parts.append(_format_table(rows))
        parts.append("\n" + _format_summary(rows))
    if only_a:
        parts.append("\nOnly in A:\n" + "\n".join(f"  {p}" for p in only_a) + "\n")
    if only_b:
        parts.append("\nOnly in B:\n" + "\n".join(f"  {p}" for p in only_b) + "\n")
    report = "".join(parts)
    sys.stdout.write(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)

    any_failure = any(not r.is_close for r in rows) or bool(only_a or only_b)
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
