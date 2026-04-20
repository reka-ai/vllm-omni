# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-step tensor dumping for the Marey diffusion pipeline.

Enabled by the ``MAREY_DUMP_DIR`` env var. When set, every call to
:meth:`MareyDitPipeline.forward` writes a per-request subdirectory with the
initial noise, per-step inputs/outputs, and final latent. Meant for
debugging numerical drift (e.g. comparing a TP=1 run against a TP=N run),
not for production use.

When ``MAREY_DUMP_DIR`` is unset, the :class:`DumpWriter` returned by
:func:`create_writer` is a ``_NullDumpWriter`` whose methods are no-ops —
the pipeline carries a dumper unconditionally so the hooks stay in one
place without a per-call ``if env_var`` branch.

Under tensor parallelism only rank 0 writes; all ranks share the same
post-all-reduce tensors so duplicating them adds I/O without new signal.

Layout
------
::

    $MAREY_DUMP_DIR/<request_id>/
        meta.json
        inputs/prompt_embeds_*.pt, prompt_masks_*.pt, vector_cond.pt
        inputs/neg_prompt_embeds_*.pt, neg_prompt_masks_*.pt, neg_vector_cond.pt
        initial_noise.pt
        timesteps.pt
        guidance_schedule.pt                   # CFG runs only
        step_000/
            z_in.pt, timestep.pt, pred_cond.pt, pred_uncond.pt (if CFG this step),
            gs.pt, v_pred.pt, x0.pt, noise.pt (omitted on last step)
        step_NNN/...
        final_z.pt
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


_ENV_DIR = "MAREY_DUMP_DIR"
_ENV_FP32 = "MAREY_DUMP_FLOAT32"


def _is_rank_zero() -> bool:
    """Return True on rank 0 or when vllm's parallel state isn't initialised."""
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except Exception:
        return True


def _prepare_tensor(t: Any, cast_fp32: bool) -> torch.Tensor:
    """Normalise ``t`` to a CPU tensor. Accepts tensors, lists, or scalars.

    ``_create_flow_timesteps`` returns a plain Python list of 0-d tensors, for
    instance — we coerce to a 1-d tensor so the dump has a single ``.pt`` file
    regardless of upstream shape.
    """
    if isinstance(t, torch.Tensor):
        out = t.detach().cpu()
    elif isinstance(t, list) and t and all(isinstance(x, torch.Tensor) for x in t):
        out = torch.stack([x.detach().cpu() for x in t])
    else:
        out = torch.as_tensor(t)
    if cast_fp32 and out.is_floating_point() and out.dtype != torch.float32:
        out = out.float()
    return out


class _NullDumpWriter:
    """Stub writer used when ``MAREY_DUMP_DIR`` is unset. Every method is a no-op."""

    enabled = False

    def write_inputs(self, *args: Any, **kwargs: Any) -> None: ...
    def write_initial(self, *args: Any, **kwargs: Any) -> None: ...
    def begin_step(self, *args: Any, **kwargs: Any) -> None: ...
    def write_step_tensor(self, *args: Any, **kwargs: Any) -> None: ...
    def write_final(self, *args: Any, **kwargs: Any) -> None: ...


class DumpWriter:
    """Writes tensors for a single ``MareyDitPipeline.forward`` call."""

    enabled = True

    def __init__(self, root: Path, request_id: str, cast_fp32: bool) -> None:
        self.root = root / request_id
        self.cast_fp32 = cast_fp32
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "inputs").mkdir(exist_ok=True)
        self._step_dir: Path | None = None

    def _save(self, path: Path, tensor: torch.Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_prepare_tensor(tensor, self.cast_fp32), path)

    def write_inputs(
        self,
        *,
        prompt_embeds: list[torch.Tensor] | torch.Tensor | None,
        prompt_masks: list[torch.Tensor] | torch.Tensor | None,
        vector_cond: torch.Tensor | None,
        neg_prompt_embeds: list[torch.Tensor] | torch.Tensor | None,
        neg_prompt_masks: list[torch.Tensor] | torch.Tensor | None,
        neg_vector_cond: torch.Tensor | None,
        meta: dict[str, Any],
    ) -> None:
        self._save_embed_list(prompt_embeds, "prompt_embeds")
        self._save_mask_list(prompt_masks, "prompt_masks")
        if vector_cond is not None:
            self._save(self.root / "inputs" / "vector_cond.pt", vector_cond)
        self._save_embed_list(neg_prompt_embeds, "neg_prompt_embeds")
        self._save_mask_list(neg_prompt_masks, "neg_prompt_masks")
        if neg_vector_cond is not None:
            self._save(self.root / "inputs" / "neg_vector_cond.pt", neg_vector_cond)
        (self.root / "meta.json").write_text(
            json.dumps(meta, default=_json_default, indent=2) + "\n"
        )

    def _save_embed_list(
        self,
        embeds: list[torch.Tensor] | torch.Tensor | None,
        base_name: str,
    ) -> None:
        if embeds is None:
            return
        if isinstance(embeds, torch.Tensor):
            self._save(self.root / "inputs" / f"{base_name}.pt", embeds)
            return
        for i, t in enumerate(embeds):
            if t is not None:
                self._save(self.root / "inputs" / f"{base_name}_{i}.pt", t)

    def _save_mask_list(
        self,
        masks: list[torch.Tensor] | torch.Tensor | None,
        base_name: str,
    ) -> None:
        if masks is None:
            return
        if isinstance(masks, torch.Tensor):
            self._save(self.root / "inputs" / f"{base_name}.pt", masks)
            return
        for i, m in enumerate(masks):
            if m is not None:
                self._save(self.root / "inputs" / f"{base_name}_{i}.pt", m)

    def write_initial(
        self,
        *,
        initial_noise: torch.Tensor,
        timesteps: torch.Tensor | list[torch.Tensor],
        guidance_schedule: list[float] | None,
    ) -> None:
        self._save(self.root / "initial_noise.pt", initial_noise)
        self._save(self.root / "timesteps.pt", timesteps)
        if guidance_schedule is not None:
            gs_t = torch.tensor(guidance_schedule, dtype=torch.float32)
            self._save(self.root / "guidance_schedule.pt", gs_t)

    def begin_step(self, step_index: int) -> None:
        self._step_dir = self.root / f"step_{step_index:04d}"
        self._step_dir.mkdir(parents=True, exist_ok=True)

    def write_step_tensor(self, name: str, tensor: torch.Tensor | float | int) -> None:
        """Write a per-step tensor; scalars are wrapped to 0-d tensors.

        Must be called between ``begin_step`` and the next ``begin_step``/
        ``write_final``.
        """
        if self._step_dir is None:
            raise RuntimeError("write_step_tensor called before begin_step")
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        self._save(self._step_dir / f"{name}.pt", tensor)

    def write_final(self, z: torch.Tensor) -> None:
        self._save(self.root / "final_z.pt", z)
        self._step_dir = None


def _json_default(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def create_writer(request_id: str | None) -> DumpWriter | _NullDumpWriter:
    """Return a :class:`DumpWriter` if dumping is enabled, else a null stub."""
    dump_dir = os.environ.get(_ENV_DIR)
    if not dump_dir:
        return _NullDumpWriter()
    if not _is_rank_zero():
        return _NullDumpWriter()
    cast_fp32 = os.environ.get(_ENV_FP32, "0") not in ("", "0", "false", "False")
    rid = request_id or f"warmup_{int(time.time() * 1000)}"
    root = Path(dump_dir).expanduser().resolve()
    try:
        writer = DumpWriter(root=root, request_id=rid, cast_fp32=cast_fp32)
    except OSError as e:
        logger.warning("MAREY_DUMP_DIR=%s unusable (%s); skipping dump", dump_dir, e)
        return _NullDumpWriter()
    logger.info("Dumping Marey diffusion tensors to %s", writer.root)
    return writer
