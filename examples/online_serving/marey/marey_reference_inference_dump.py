#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run moonvalley_ai's marey_inference.py CLI with dump/load instrumentation.

Symmetric to ``vllm_omni/diffusion/debug/dump.py`` (the vllm-omni-side
``DumpMixin``): captures every tensor on the diffusion path so the run can be
diffed against a vllm-omni run with ``compare_dumps.py``.

Activation env vars (same names + on-disk layout as ``DumpMixin``):

  - ``MAREY_DUMP_DIR``               — root dir; this run dumps to ``<dir>/ref_run/``
  - ``MAREY_LOAD_INITIAL_NOISE``     — path to a .pt file with the initial noise
  - ``MAREY_LOAD_STEP_NOISE_DIR``    — dir containing ``step_noise_<i>.pt`` files
  - ``MAREY_LOAD_COND_FRAMES_PATH``  — path to ``cond_frames.pt`` (I2V); substitutes
                                       the VAE-encoded conditioning latent that
                                       ``MareyInference._frame_conditions`` returned
  - ``MAREY_LOAD_COND_OFFSETS_PATH`` — path to ``cond_offsets.pt`` (I2V); substitutes
                                       the latent-time offsets
  - ``MAREY_DUMP_BLOCKS_AT_STEPS``   — comma-separated step indices for per-block dumps
                                       and I2V transformer-internal intermediates
                                       (``t0_emb``, ``x_after_concat``, ``x_t_mask``,
                                       ``x_pre_slice``)
  - ``MAREY_DUMP_REQUEST_ID``        — override the per-request subdir name (default ``ref_run``)

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
LOAD_COND_FRAMES_PATH = os.environ.get("MAREY_LOAD_COND_FRAMES_PATH") or None
LOAD_COND_OFFSETS_PATH = os.environ.get("MAREY_LOAD_COND_OFFSETS_PATH") or None
REQUEST_ID = os.environ.get("MAREY_DUMP_REQUEST_ID", "ref_run")
_DUMP_BLOCKS_AT_STEPS_ENV = os.environ.get("MAREY_DUMP_BLOCKS_AT_STEPS", "")
DUMP_BLOCKS_AT_STEPS: set[int] = {
    int(s) for s in _DUMP_BLOCKS_AT_STEPS_ENV.split(",") if s.strip().isdigit()
}
ACTIVE = bool(
    DUMP_DIR_ROOT
    or LOAD_INITIAL_PATH
    or LOAD_STEP_DIR
    or LOAD_COND_FRAMES_PATH
    or LOAD_COND_OFFSETS_PATH
)


def _maybe_dump(dump_dir: str | None, name: str, tensor: torch.Tensor) -> None:
    if not dump_dir:
        return
    path = os.path.join(dump_dir, f"{name}.pt")
    torch.save(tensor.detach().to("cpu"), path)


def _dump_textcond(dump_dir: str | None, label: str, tc: Any) -> None:
    """Dump a TextCond / UnstructuredTextCond using the DumpMixin schema.

    Handles both shapes:
      - TextCond: seq_cond is a Tensor, seq_cond_mask is a Tensor,
        vector_cond is an Optional[Tensor].
      - UnstructuredTextCond: seq_cond is List[Tensor], seq_cond_mask is a
        Tensor, vector_cond is Optional[List[Tensor]].

    Filenames match vllm-omni's DumpMixin output (encode_{label}_...).
    """
    if tc is None or dump_dir is None:
        return
    seq_cond = getattr(tc, "seq_cond", None)
    seq_cond_mask = getattr(tc, "seq_cond_mask", None)
    vector_cond = getattr(tc, "vector_cond", None)

    if isinstance(seq_cond, (list, tuple)):
        for j, t in enumerate(seq_cond):
            if isinstance(t, torch.Tensor):
                _maybe_dump(dump_dir, f"encode_{label}_seq_cond_{j}", t)
    elif isinstance(seq_cond, torch.Tensor):
        _maybe_dump(dump_dir, f"encode_{label}_seq_cond_0", seq_cond)

    if isinstance(seq_cond_mask, (list, tuple)):
        for j, t in enumerate(seq_cond_mask):
            if isinstance(t, torch.Tensor):
                _maybe_dump(dump_dir, f"encode_{label}_seq_mask_{j}", t)
    elif isinstance(seq_cond_mask, torch.Tensor):
        _maybe_dump(dump_dir, f"encode_{label}_seq_mask_0", seq_cond_mask)

    if isinstance(vector_cond, (list, tuple)):
        for j, t in enumerate(vector_cond):
            if isinstance(t, torch.Tensor):
                _maybe_dump(dump_dir, f"encode_{label}_vector_cond_{j}", t)
    elif isinstance(vector_cond, torch.Tensor):
        _maybe_dump(dump_dir, f"encode_{label}_vector_cond", vector_cond)


def _dump_transformer_text_cond(dump_dir: str | None, prefix: str, tc: Any) -> None:
    """Dump the text_cond argument that flows into model() using DumpMixin
    transformer-input schema (step<i>_<label>_encoder_hidden_states_<j>, etc.).
    """
    if tc is None or dump_dir is None:
        return
    seq_cond = getattr(tc, "seq_cond", None)
    vector_cond = getattr(tc, "vector_cond", None)

    if isinstance(seq_cond, (list, tuple)):
        for j, t in enumerate(seq_cond):
            if isinstance(t, torch.Tensor):
                _maybe_dump(dump_dir, f"{prefix}_encoder_hidden_states_{j}", t)
    elif isinstance(seq_cond, torch.Tensor):
        _maybe_dump(dump_dir, f"{prefix}_encoder_hidden_states_0", seq_cond)

    if isinstance(vector_cond, (list, tuple)):
        for j, t in enumerate(vector_cond):
            if isinstance(t, torch.Tensor):
                _maybe_dump(dump_dir, f"{prefix}_vector_cond_{j}", t)
    elif isinstance(vector_cond, torch.Tensor):
        _maybe_dump(dump_dir, f"{prefix}_vector_cond", vector_cond)


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

    def _patched_sample(self: Any, *args: Any, **kwargs: Any) -> Any:
        # marey_inference.py calls scheduler.sample(**scheduler_params), so
        # all sample() params arrive as kwargs. We need text_encoder (for
        # Step 1.3's encode/null wrap) and the model (for the per-call
        # buffer). Extract both defensively.
        if "model" in kwargs:
            model = kwargs.pop("model")
        elif len(args) > 0:
            model = args[0]
            args = args[1:]
        else:
            raise RuntimeError("_patched_sample: could not find 'model' arg")

        if "text_encoder" in kwargs:
            text_encoder = kwargs.pop("text_encoder")
        elif len(args) > 0:
            text_encoder = args[0]
            args = args[1:]
        else:
            raise RuntimeError("_patched_sample: could not find 'text_encoder' arg")

        # Per-process dump dir (single CLI invocation = one dump dir).
        if DUMP_DIR_ROOT and _dump_enabled:
            dump_dir = os.path.join(DUMP_DIR_ROOT, REQUEST_ID)
            os.makedirs(dump_dir, exist_ok=True)
            logger.warning("(rank 0) Dumping moonvalley_ai reference tensors to %s", dump_dir)
        else:
            dump_dir = None

        # State machine:
        #   - noise_call_idx  counts torch.randn_like() calls. idx 0 = initial
        #     latent noise; idx 1..N = per-step DDPM noise (idx - 1 = step_idx).
        #   - current_step    tracks which denoising step the buffered model()
        #     calls belong to. Advances when torch.randn_like fires (step
        #     boundary) or at sample() exit (final step).
        #   - within_step_buffer holds (z, t, mkwargs, out) tuples for each
        #     model() call in the current step; flushed with semantic labels.
        #   - cond_id / uncond_id cache the id() of the TextCond objects
        #     returned by text_encoder.encode() / text_encoder.null(). Used to
        #     label model() calls by tensor identity (B3) instead of by order.
        noise_call_idx = [0]
        current_step = [0]
        within_step_buffer: list[tuple[torch.Tensor, torch.Tensor, dict, Any]] = []
        cond_id = [None]
        uncond_id = [None]
        encode_call_count = [0]
        per_step_first_timesteps: list[torch.Tensor] = []
        # Per-call label stash for the block hooks (set by _wrapped_model
        # before each original_model call, read by block forward hooks).
        current_call_label: list[str | None] = [None]
        # Env-gated per-block dumping (mirror of vllm-omni's
        # MAREY_DUMP_BLOCKS_AT_STEPS). Only dump block outputs at these steps.
        _dump_blocks_env = os.environ.get("MAREY_DUMP_BLOCKS_AT_STEPS", "")
        dump_blocks_at_steps: set[int] = {
            int(s) for s in _dump_blocks_env.split(",") if s.strip().isdigit()
        }
        block_hooks_registered = [False]

        # -- Step 1.3: wrap text_encoder.encode and .null to dump outputs
        # and cache object ids for identity-based labeling (B3).
        _orig_te_encode = text_encoder.encode
        _orig_te_null = text_encoder.null

        def _wrapped_encode(*a: Any, **kw: Any) -> Any:
            out = _orig_te_encode(*a, **kw)
            # First encode() in RFLOW.sample is for the positive prompts
            # (the cond path). If use_negative_prompts is True, a second
            # encode() follows for the negatives (the uncond path).
            idx = encode_call_count[0]
            label = "cond" if idx == 0 else "uncond"
            _dump_textcond(dump_dir, label, out)
            if label == "cond":
                cond_id[0] = id(out)
            else:
                uncond_id[0] = id(out)
            encode_call_count[0] += 1
            return out

        def _wrapped_null(text_cond: Any) -> Any:
            # Called when use_negative_prompts=False: moonvalley synthesizes an
            # empty/null TextCond from the positive one. This is the uncond
            # path's TextCond object.
            out = _orig_te_null(text_cond)
            _dump_textcond(dump_dir, "uncond", out)
            uncond_id[0] = id(out)
            return out

        text_encoder.encode = _wrapped_encode  # type: ignore[assignment]
        text_encoder.null = _wrapped_null  # type: ignore[assignment]

        def _label_by_identity(mkwargs: dict) -> str | None:
            """Return 'cond' / 'uncond' if mkwargs['text_cond'] matches one of
            the cached ids, else None (caller falls back to order-based).
            """
            tc = mkwargs.get("text_cond")
            if tc is None:
                return None
            tc_id = id(tc)
            if cond_id[0] is not None and tc_id == cond_id[0]:
                return "cond"
            if uncond_id[0] is not None and tc_id == uncond_id[0]:
                return "uncond"
            return None

        def _flush_step_buffer() -> None:
            n = len(within_step_buffer)
            if n == 0 or dump_dir is None:
                within_step_buffer.clear()
                return

            # B3: label by tensor identity where possible. Fall back to
            # order-based (uncond, cond) with a warning if identity fails.
            labels: list[str] = []
            identity_failed = False
            for (_z, _t, mkwargs, _out) in within_step_buffer:
                lbl = _label_by_identity(mkwargs)
                if lbl is None:
                    identity_failed = True
                    break
                labels.append(lbl)
            if identity_failed:
                logger.warning(
                    "Identity-based cond/uncond labeling failed at step %d "
                    "(n=%d). Falling back to order-based labels. cond_id=%s "
                    "uncond_id=%s mkwargs text_cond ids=%s",
                    current_step[0], n, cond_id[0], uncond_id[0],
                    [id(mk.get("text_cond")) for (_, _, mk, _) in within_step_buffer],
                )
                if n == 1:
                    labels = ["cond"]
                elif n == 2:
                    # moonvalley RFLOW.sample has two model() sites with
                    # opposite orders; the cmp branch docs one as uncond-first.
                    labels = ["uncond", "cond"]
                else:
                    labels = [f"call{j}" for j in range(n)]

            # Capture the first call's timestep for the full-schedule dump.
            first_t = within_step_buffer[0][1]
            if isinstance(first_t, torch.Tensor):
                per_step_first_timesteps.append(first_t.detach().to("cpu"))

            for j, (z, t, mkwargs, out) in enumerate(within_step_buffer):
                prefix = f"step{current_step[0]}_{labels[j]}"
                _maybe_dump(dump_dir, f"{prefix}_hidden_states", z)
                _maybe_dump(dump_dir, f"{prefix}_timestep", t)
                # B2: emit text_cond with the vllm-omni transformer-input
                # schema (encoder_hidden_states_<j>, vector_cond). Other
                # kwargs keep the kw_ prefix — if they're ever needed for
                # diffing, the rename can be added here.
                for k, v in mkwargs.items():
                    if k == "text_cond":
                        _dump_transformer_text_cond(dump_dir, prefix, v)
                    elif isinstance(v, torch.Tensor):
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

        def _register_block_hooks_once() -> None:
            """Register forward hooks on each model.blocks[i] on first model call.

            Delayed because ``original_model`` here is the *user-supplied*
            model arg to RFLOW.sample, which on moonvalley is a deepspeed-
            wrapped nn.Module. Access .blocks via .module if wrapped.
            """
            if block_hooks_registered[0] or not dump_blocks_at_steps or dump_dir is None:
                return
            # Unwrap any deepspeed/ds wrapper to find the actual nn.Module
            # with .blocks. Fall back to iterating attributes.
            cand = original_model
            for attr in ("module", "model"):
                if hasattr(cand, "blocks"):
                    break
                if hasattr(cand, attr):
                    cand = getattr(cand, attr)
            if not hasattr(cand, "blocks"):
                logger.warning(
                    "Block dump requested but model has no .blocks attr; skipping"
                )
                block_hooks_registered[0] = True
                return
            logger.warning(
                "Reference wrapper per-block dump enabled at steps %s (%d blocks)",
                sorted(dump_blocks_at_steps), len(cand.blocks),
            )

            # I2V transformer-internal dumps via method wrapping. Same
            # step/label gating as block hooks. Mirrors vllm-omni's
            # ``DumpMixin._dump_i2v_intermediate``.
            def _i2v_dump(name: str, tensor: Any) -> None:
                if not isinstance(tensor, torch.Tensor):
                    return
                if current_step[0] not in dump_blocks_at_steps:
                    return
                label = current_call_label[0] or "unknown"
                _maybe_dump(dump_dir, f"step{current_step[0]}_{label}_{name}", tensor)

            # (a) ``mix_time_and_embeddings`` is called twice per forward:
            # first with the real timestep (→ t_emb), then with zeros
            # (→ t0_emb) when I2V is active. Detect zero-timestep input
            # and dump the post-``t_block`` value (the FIRST tuple element,
            # which mv calls ``t0_emb`` in dit3d.py:1008).
            if hasattr(cand, "mix_time_and_embeddings"):
                _orig_mtae = cand.mix_time_and_embeddings

                def _wrap_mtae(timestep: Any, dtype: Any, extra_emb: Any) -> Any:
                    out = _orig_mtae(timestep, dtype, extra_emb)
                    is_zero = (
                        isinstance(timestep, torch.Tensor)
                        and timestep.numel() > 0
                        and bool((timestep == 0).all())
                    )
                    if is_zero and isinstance(out, tuple) and len(out) >= 1:
                        _i2v_dump("t0_emb", out[0])
                    return out

                cand.mix_time_and_embeddings = _wrap_mtae  # type: ignore[assignment]

            # (b) ``split_sp_variables`` receives the pre-shard ``sp_vars``
            # dict containing ``x`` (post-concat) and ``x_t_mask``. Capture
            # both pre-shard so they line up with vllm-omni's pre-shard dumps.
            if hasattr(cand, "split_sp_variables"):
                _orig_split_sp = cand.split_sp_variables

                def _wrap_split_sp(sp_vars: dict, *args: Any, **kwargs: Any) -> Any:
                    if isinstance(sp_vars, dict):
                        if "x" in sp_vars and isinstance(sp_vars["x"], torch.Tensor):
                            _i2v_dump("x_after_concat", sp_vars["x"])
                        x_t_mask = sp_vars.get("x_t_mask")
                        if isinstance(x_t_mask, torch.Tensor):
                            _i2v_dump("x_t_mask", x_t_mask)
                    return _orig_split_sp(sp_vars, *args, **kwargs)

                cand.split_sp_variables = _wrap_split_sp  # type: ignore[assignment]

            # (c) ``gather_sequence`` is called after ``final_layer`` and
            # before the I2V slice. Capture the gathered post-final-layer
            # value (first element of the returned tuple) as x_pre_slice.
            if hasattr(cand, "gather_sequence"):
                _orig_gather = cand.gather_sequence

                def _wrap_gather(*args: Any, **kwargs: Any) -> Any:
                    out = _orig_gather(*args, **kwargs)
                    if isinstance(out, tuple) and len(out) >= 1 and isinstance(out[0], torch.Tensor):
                        _i2v_dump("x_pre_slice", out[0])
                    return out

                cand.gather_sequence = _wrap_gather  # type: ignore[assignment]

            for block_idx, block in enumerate(cand.blocks):
                def _make_post_hook(bi: int):
                    def _hook(module, args, kwargs, output):
                        del module, args, kwargs
                        if current_step[0] not in dump_blocks_at_steps:
                            return
                        label = current_call_label[0] or "unknown"
                        prefix = f"step{current_step[0]}_{label}_block{bi}"
                        if isinstance(output, tuple) and len(output) >= 2:
                            if isinstance(output[0], torch.Tensor):
                                _maybe_dump(dump_dir, f"{prefix}_x_out", output[0])
                            if isinstance(output[1], torch.Tensor):
                                _maybe_dump(dump_dir, f"{prefix}_y_out", output[1])
                        elif isinstance(output, torch.Tensor):
                            _maybe_dump(dump_dir, f"{prefix}_out", output)
                    return _hook

                def _make_pre_hook(bi: int):
                    def _hook(module, args, kwargs):
                        del module
                        if current_step[0] not in dump_blocks_at_steps:
                            return
                        label = current_call_label[0] or "unknown"
                        prefix = f"step{current_step[0]}_{label}_block{bi}"
                        # mv FluxBlock signature: forward(self, x, y, t_x, t_y, ...)
                        x_in = args[0] if len(args) > 0 else kwargs.get("x")
                        y_in = args[1] if len(args) > 1 else kwargs.get("y")
                        if isinstance(x_in, torch.Tensor):
                            _maybe_dump(dump_dir, f"{prefix}_x_in", x_in)
                        if isinstance(y_in, torch.Tensor):
                            _maybe_dump(dump_dir, f"{prefix}_y_in", y_in)
                    return _hook

                block.register_forward_pre_hook(_make_pre_hook(block_idx), with_kwargs=True)
                block.register_forward_hook(_make_post_hook(block_idx), with_kwargs=True)
            block_hooks_registered[0] = True

        def _wrapped_model(z: torch.Tensor, t: torch.Tensor, **mkwargs: Any) -> torch.Tensor:
            _register_block_hooks_once()
            # Resolve cond/uncond label BEFORE the forward so block hooks know
            # what to tag their dumps with. Falls back to "unknown" if identity
            # matching fails (e.g. first call before ids are cached).
            current_call_label[0] = _label_by_identity(mkwargs) or "unknown"
            try:
                out = original_model(z, t, **mkwargs)
            finally:
                current_call_label[0] = None
            within_step_buffer.append((z, t, mkwargs, out))
            return out

        # Patch torch.randn_like for the duration of sample() only.
        torch.randn_like = _hook_randn_like  # type: ignore[assignment]
        try:
            # Call shape: sample(self, model, text_encoder, *rest, **kwargs).
            # Pass our wrapped model and the text_encoder (now patched) as
            # the two leading positional args; everything else flows through.
            result = _original_sample(
                self, _wrapped_model, text_encoder, *args, **kwargs
            )
        finally:
            torch.randn_like = _real_randn_like  # type: ignore[assignment]
            text_encoder.encode = _orig_te_encode  # type: ignore[assignment]
            text_encoder.null = _orig_te_null  # type: ignore[assignment]
            # The final denoising step doesn't trigger a torch.randn_like call
            # (rflow only samples noise for `i < len(timesteps) - 1`), so its
            # transformer calls are still buffered. Flush them now.
            _flush_step_buffer()

            # Step 1.2: dump the full timestep schedule actually used for
            # denoising (one entry per denoising step, taken from the first
            # model() call of each step).
            if dump_dir is not None and per_step_first_timesteps:
                try:
                    timesteps_tensor = torch.stack(
                        [t.reshape(-1)[0] if t.dim() > 0 else t
                         for t in per_step_first_timesteps]
                    )
                    _maybe_dump(dump_dir, "timesteps", timesteps_tensor)
                except Exception as e:
                    logger.warning("timesteps stack/dump failed: %s", e)

            # B4: assert the patched randn_like fired the expected number of
            # times. For distilled 30B with num_sampling_steps=33 we expect
            # 1 (initial) + 32 (per-step for i < final) = 33. If off, some
            # non-sampler code consumed the patch.
            expected_calls_hint = 1 + len(per_step_first_timesteps) - 1 \
                if per_step_first_timesteps else None
            if expected_calls_hint is not None \
                    and noise_call_idx[0] != expected_calls_hint:
                logger.warning(
                    "B4: noise_call_idx=%d != expected=%d (steps=%d). "
                    "Some non-sampler code likely consumed torch.randn_like "
                    "while patched — per-step noise indexing may be wrong.",
                    noise_call_idx[0], expected_calls_hint,
                    len(per_step_first_timesteps),
                )

            logger.warning(
                "Done. noise_calls=%d, dumped_steps=%d, dump_dir=%s",
                noise_call_idx[0],
                len(per_step_first_timesteps),
                dump_dir,
            )
        return result

    RFLOW.sample = _patched_sample  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Hand off to marey_inference's typer CLI (uses sys.argv as-is)
# ---------------------------------------------------------------------------

import marey_inference  # noqa: E402  (must come after monkey-patch above)


# ---------------------------------------------------------------------------
# I2V: monkey-patch ``MareyInference._frame_conditions`` to dump and optionally
# inject the VAE-encoded conditioning latents (cond_frames) and latent-time
# offsets (cond_offsets). Runs BEFORE ``RFLOW.sample`` (the patched call), so
# the dump_dir is reconstructed from env directly.
# ---------------------------------------------------------------------------

if ACTIVE and hasattr(marey_inference, "MareyInference"):
    _orig_frame_conditions = marey_inference.MareyInference._frame_conditions

    # Rank-0 only writes; loading happens on every rank (each rank needs the
    # full cond_frames tensor, same policy as initial noise loading).
    try:
        import torch.distributed as _dist_fc
        _rank_fc = _dist_fc.get_rank() if _dist_fc.is_available() and _dist_fc.is_initialized() else 0
    except Exception:
        _rank_fc = 0
    _dump_enabled_fc = _rank_fc == 0

    def _patched_frame_conditions(
        self: Any,
        frame_conditions: dict,
        device: torch.device,
        output_dimensions: tuple,
        high_precision_input: bool = False,
    ) -> dict:
        result = _orig_frame_conditions(
            self, frame_conditions, device, output_dimensions, high_precision_input
        )
        cond_frames = result.get("cond_frames")
        cond_offsets = result.get("cond_offsets")

        if DUMP_DIR_ROOT and _dump_enabled_fc:
            dump_dir = os.path.join(DUMP_DIR_ROOT, REQUEST_ID)
            os.makedirs(dump_dir, exist_ok=True)
            if isinstance(cond_frames, torch.Tensor):
                _maybe_dump(dump_dir, "cond_frames", cond_frames)
            if isinstance(cond_offsets, torch.Tensor):
                _maybe_dump(dump_dir, "cond_offsets", cond_offsets)
            logger.warning(
                "(rank 0) _frame_conditions dumped: cond_frames.shape=%s "
                "cond_offsets=%s",
                tuple(cond_frames.shape) if isinstance(cond_frames, torch.Tensor) else None,
                cond_offsets.tolist() if isinstance(cond_offsets, torch.Tensor) else None,
            )

        # Optional injection from a reference dump (e.g. vllm-omni's). Every
        # rank loads independently because every rank needs the full tensor.
        if LOAD_COND_FRAMES_PATH and isinstance(cond_frames, torch.Tensor):
            if os.path.exists(LOAD_COND_FRAMES_PATH):
                loaded = torch.load(LOAD_COND_FRAMES_PATH, map_location=cond_frames.device)
                if tuple(loaded.shape) == tuple(cond_frames.shape):
                    result["cond_frames"] = loaded.to(
                        device=cond_frames.device, dtype=cond_frames.dtype
                    )
                    logger.warning(
                        "_frame_conditions loaded cond_frames from %s",
                        LOAD_COND_FRAMES_PATH,
                    )
                else:
                    logger.warning(
                        "_frame_conditions SKIPPING cond_frames load: shape %s != ref %s",
                        tuple(loaded.shape), tuple(cond_frames.shape),
                    )
        if LOAD_COND_OFFSETS_PATH and isinstance(cond_offsets, torch.Tensor):
            if os.path.exists(LOAD_COND_OFFSETS_PATH):
                loaded = torch.load(LOAD_COND_OFFSETS_PATH, map_location=cond_offsets.device)
                if tuple(loaded.shape) == tuple(cond_offsets.shape):
                    # Preserve int64 dtype.
                    result["cond_offsets"] = loaded.to(device=cond_offsets.device)
                    logger.warning(
                        "_frame_conditions loaded cond_offsets from %s (values %s)",
                        LOAD_COND_OFFSETS_PATH, loaded.tolist(),
                    )
                else:
                    logger.warning(
                        "_frame_conditions SKIPPING cond_offsets load: shape %s != ref %s",
                        tuple(loaded.shape), tuple(cond_offsets.shape),
                    )

        return result

    marey_inference.MareyInference._frame_conditions = _patched_frame_conditions  # type: ignore[assignment]


if __name__ == "__main__":
    marey_inference.app()
