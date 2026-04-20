#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run moonvalley_ai's marey_inference.py CLI with dump/load instrumentation.

Symmetric to ``vllm_omni/diffusion/debug/dump.py`` (the vllm-omni-side
``DumpMixin``): captures every tensor on the diffusion path so the run can be
diffed against a vllm-omni run with ``compare_dumps.py``.

Activation env vars (same names + on-disk layout as ``DumpMixin``):

  - ``MAREY_DUMP_DIR``           — root dir; this run dumps to ``<dir>/ref_run/``
  - ``MAREY_LOAD_INITIAL_NOISE`` — path to a .pt file with the initial noise
  - ``MAREY_LOAD_STEP_NOISE_DIR``— dir containing ``step_noise_<i>.pt`` files
  - ``MAREY_DUMP_REQUEST_ID``    — override the per-request subdir name (default ``ref_run``)

Without any of those set, this script behaves identically to invoking
moonvalley_ai's marey_inference.py directly.

Required env var: ``MOONVALLEY_AI_PATH`` (path to the moonvalley_ai checkout)

Usage:
    torchrun --nproc_per_node=8 marey_reference_inference_dump.py infer ...args

Pass-through: every CLI arg after the script name is forwarded to
moonvalley_ai's typer app. See the ref-inference shell wrappers for the
canonical 30B and 7B flag sets.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Resolve moonvalley_ai paths and import the CLI
# ---------------------------------------------------------------------------

MOONVALLEY = os.environ.get("MOONVALLEY_AI_PATH")
if not MOONVALLEY:
    raise RuntimeError("MOONVALLEY_AI_PATH must be set to the moonvalley_ai checkout")

for p in (
    MOONVALLEY,
    os.path.join(MOONVALLEY, "open_sora"),
    os.path.join(MOONVALLEY, "inference-service"),
):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Dump/load configuration (read once, applied for the duration of the process)
# ---------------------------------------------------------------------------

DUMP_DIR_ROOT = os.environ.get("MAREY_DUMP_DIR") or None
LOAD_INITIAL_PATH = os.environ.get("MAREY_LOAD_INITIAL_NOISE") or None
LOAD_STEP_DIR = os.environ.get("MAREY_LOAD_STEP_NOISE_DIR") or None
REQUEST_ID = os.environ.get("MAREY_DUMP_REQUEST_ID", "ref_run")
ACTIVE = bool(DUMP_DIR_ROOT or LOAD_INITIAL_PATH or LOAD_STEP_DIR)


def _maybe_dump(dump_dir: str | None, name: str, tensor: torch.Tensor) -> None:
    if not dump_dir:
        return
    path = os.path.join(dump_dir, f"{name}.pt")
    torch.save(tensor.detach().to("cpu"), path)


# ---------------------------------------------------------------------------
# Monkey-patch RFLOW.sample to intercept noise + wrap the model callable
# ---------------------------------------------------------------------------

if ACTIVE:
    logger.warning(
        "marey_reference_inference_dump active: dump_dir=%s load_initial=%s load_step_dir=%s req_id=%s",
        DUMP_DIR_ROOT,
        LOAD_INITIAL_PATH,
        LOAD_STEP_DIR,
        REQUEST_ID,
    )

    # Local import: must come *before* marey_inference.py is imported below.
    from opensora.schedulers.rf import RFLOW

    _real_randn_like = torch.randn_like
    _original_sample = RFLOW.sample

    # Only rank 0 writes dumps. After RFLOW.sample's per-rank shards are
    # all-gathered inside the model, the model() return value is the same on
    # every rank, so 8 writers would just overwrite the same data 8x.
    try:
        import torch.distributed as _dist
        _rank = _dist.get_rank() if _dist.is_available() and _dist.is_initialized() else 0
    except Exception:
        _rank = 0
    _dump_enabled = _rank == 0

    def _patched_sample(self: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        # Per-process dump dir (single CLI invocation = one dump dir).
        if DUMP_DIR_ROOT and _dump_enabled:
            dump_dir = os.path.join(DUMP_DIR_ROOT, REQUEST_ID)
            os.makedirs(dump_dir, exist_ok=True)
            logger.warning("(rank 0) Dumping moonvalley_ai reference tensors to %s", dump_dir)
        else:
            dump_dir = None

        # State for semantic step + cond/uncond naming.
        # Within a step, model() is called 1 or 2 times; per RFLOW.sample's
        # get_guided_prediction the order is uncond-first then cond. If only
        # one call (uncond skipped), it's the cond call.
        # We buffer calls during a step and emit them with semantic labels
        # at the next step boundary (next torch.randn_like call) or at
        # sample() exit (for the final step).
        noise_call_idx = [0]   # 0 = initial, 1..N = per-step (idx - 1 = step_idx)
        current_step = [0]     # which denoising step the buffered calls belong to
        within_step_buffer: list[tuple[torch.Tensor, torch.Tensor, dict, Any]] = []

        def _flush_step_buffer() -> None:
            n = len(within_step_buffer)
            if n == 0 or dump_dir is None:
                within_step_buffer.clear()
                return
            if n == 1:
                labels = ["cond"]
            elif n == 2:
                labels = ["uncond", "cond"]
            else:
                labels = [f"call{j}" for j in range(n)]
            for j, (z, t, mkwargs, out) in enumerate(within_step_buffer):
                prefix = f"step{current_step[0]}_{labels[j]}"
                _maybe_dump(dump_dir, f"{prefix}_hidden_states", z)
                _maybe_dump(dump_dir, f"{prefix}_timestep", t)
                for k, v in mkwargs.items():
                    if isinstance(v, torch.Tensor):
                        _maybe_dump(dump_dir, f"{prefix}_kw_{k}", v)
                    elif isinstance(v, (list, tuple)):
                        for jj, t_ in enumerate(v):
                            if isinstance(t_, torch.Tensor):
                                _maybe_dump(dump_dir, f"{prefix}_kw_{k}_{jj}", t_)
                    elif isinstance(v, dict):
                        for sk, t_ in v.items():
                            if isinstance(t_, torch.Tensor):
                                _maybe_dump(dump_dir, f"{prefix}_kw_{k}_{sk}", t_)
                if isinstance(out, torch.Tensor):
                    _maybe_dump(dump_dir, f"{prefix}_v_pred", out)
            within_step_buffer.clear()

        def _hook_randn_like(input_tensor: torch.Tensor, *a: Any, **kw: Any) -> torch.Tensor:
            idx = noise_call_idx[0]
            if idx == 0:
                # Initial noise (sampled before any denoising step runs).
                if LOAD_INITIAL_PATH:
                    out = torch.load(LOAD_INITIAL_PATH, map_location=input_tensor.device)
                    out = out.to(device=input_tensor.device, dtype=input_tensor.dtype)
                    assert tuple(out.shape) == tuple(input_tensor.shape), (
                        f"Loaded initial noise shape {tuple(out.shape)} != "
                        f"expected {tuple(input_tensor.shape)} (file: {LOAD_INITIAL_PATH})"
                    )
                else:
                    out = _real_randn_like(input_tensor, *a, **kw)
                _maybe_dump(dump_dir, "z_initial", out)
            else:
                # Per-step DDPM noise. By this point all transformer calls for
                # `current_step` have completed; flush them with semantic
                # labels before advancing.
                _flush_step_buffer()
                step_idx = idx - 1
                if LOAD_STEP_DIR:
                    path = os.path.join(LOAD_STEP_DIR, f"step_noise_{step_idx}.pt")
                    out = torch.load(path, map_location=input_tensor.device)
                    out = out.to(device=input_tensor.device, dtype=input_tensor.dtype)
                    assert tuple(out.shape) == tuple(input_tensor.shape), (
                        f"Loaded step noise {step_idx} shape {tuple(out.shape)} != "
                        f"z shape {tuple(input_tensor.shape)} (file: {path})"
                    )
                else:
                    out = _real_randn_like(input_tensor, *a, **kw)
                _maybe_dump(dump_dir, f"step_noise_{step_idx}", out)
                current_step[0] = step_idx + 1  # next iteration's buffer belongs here
            noise_call_idx[0] += 1
            return out

        # Wrap the model callable so every transformer call (cond + uncond per
        # step) is buffered and emitted with semantic labels.
        original_model = model

        def _wrapped_model(z: torch.Tensor, t: torch.Tensor, **mkwargs: Any) -> torch.Tensor:
            out = original_model(z, t, **mkwargs)
            within_step_buffer.append((z, t, mkwargs, out))
            return out

        # Patch torch.randn_like for the duration of sample() only.
        torch.randn_like = _hook_randn_like  # type: ignore[assignment]
        try:
            result = _original_sample(self, _wrapped_model, *args, **kwargs)
        finally:
            torch.randn_like = _real_randn_like  # type: ignore[assignment]
            # The final denoising step doesn't trigger a torch.randn_like call
            # (rflow only samples noise for `i < len(timesteps) - 1`), so its
            # transformer calls are still buffered. Flush them now.
            _flush_step_buffer()
            logger.warning(
                "Done. noise_calls=%d, dump_dir=%s",
                noise_call_idx[0],
                dump_dir,
            )
        return result

    RFLOW.sample = _patched_sample  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Hand off to marey_inference's typer CLI (uses sys.argv as-is)
# ---------------------------------------------------------------------------

import marey_inference  # noqa: E402  (must come after monkey-patch above)

if __name__ == "__main__":
    marey_inference.app()
