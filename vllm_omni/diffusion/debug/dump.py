# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump/load instrumentation mixin for diffusion pipelines.

Used to verify a vllm-omni pipeline against a reference inference (e.g.
moonvalley_ai's marey_inference.py) by capturing every tensor on the
diffusion path and optionally loading reference noise instead of sampling.

The mixin assumes the pipeline class follows the convention introduced in
``pipeline_marey.py`` (P1 refactor):

  - ``self._sample_initial_noise(shape, generator, device, dtype) -> Tensor``
  - ``self._sample_step_noise(z, generator, step_idx) -> Tensor``
  - ``self.encode_prompt(prompt, device, dtype, **kwargs) -> (seq, masks, vec)``
  - ``self.transformer`` is an ``nn.Module``

Activation: any of these env vars set will engage the mixin:

  - ``MAREY_DUMP_DIR``           — root dir; per-request subdir created on each forward()
  - ``MAREY_LOAD_INITIAL_NOISE`` — path to a .pt file with the initial noise tensor
  - ``MAREY_LOAD_STEP_NOISE_DIR``— dir containing ``step_noise_<i>.pt`` files

If none are set, the mixin is inert (no overhead, no behavior change).

Naming convention for dumped tensors (same on both sides — ref and vllm-omni):

  z_initial.pt
  step_noise_<i>.pt          (i is the loop index where noise was sampled)
  encode_cond_seq_cond_<j>.pt, encode_cond_seq_mask_<j>.pt, encode_cond_vector_cond.pt
  encode_uncond_*            (mirror, present only on the side that wraps encode_prompt)
  step<i>_<cond|uncond>_hidden_states.pt
  step<i>_<cond|uncond>_timestep.pt
  step<i>_<cond|uncond>_encoder_hidden_states_<k>.pt
  step<i>_<cond|uncond>_vector_cond.pt
  step<i>_<cond|uncond>_extra_<key>.pt
  step<i>_<cond|uncond>_v_pred.pt

Cond/uncond labelling on the vllm-omni side is detected by tensor identity
(same Python object as the cached ``encode_prompt`` cond/uncond text emb).
The reference-side wrapper uses an order-based heuristic (it knows
moonvalley's call order is uncond-first, then cond) — see
``examples/online_serving/marey/marey_reference_inference_dump.py``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


class DumpMixin:
    """Mixin that adds dump/load hooks to a diffusion pipeline.

    Subclass usage::

        from vllm_omni.diffusion.models.marey.pipeline_marey import MareyPipeline
        class DumpMareyPipeline(DumpMixin, MareyPipeline):
            pass
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._dump_dir_root: str | None = os.environ.get("MAREY_DUMP_DIR") or None
        self._load_initial_noise_path: str | None = (
            os.environ.get("MAREY_LOAD_INITIAL_NOISE") or None
        )
        self._load_step_noise_dir: str | None = (
            os.environ.get("MAREY_LOAD_STEP_NOISE_DIR") or None
        )

        # Only rank 0 dumps. After SP all-gather every rank holds the same
        # tensors, so dumping from all 8 produces 8 copies. Loading still
        # happens on every rank because each rank needs the full noise tensor.
        try:
            import torch.distributed as dist
            self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except Exception:
            self._rank = 0
        self._dump_enabled = self._rank == 0

        self._active = bool(
            self._dump_dir_root
            or self._load_initial_noise_path
            or self._load_step_noise_dir
        )
        self._current_dump_dir: str | None = None

        # State for semantic cond/uncond labelling.
        self._current_step: int = 0
        # id() of the first (UL2) tensor in the cond / uncond text-embed list,
        # captured during the two encode_prompt calls. Used by the transformer
        # hook to label each call as cond or uncond.
        self._cond_text_id: int | None = None
        self._uncond_text_id: int | None = None
        self._encode_prompt_call_count: int = 0

        if not self._active:
            return

        logger.warning(
            "DumpMixin active: dump_dir=%s load_initial=%s load_step_dir=%s",
            self._dump_dir_root,
            self._load_initial_noise_path,
            self._load_step_noise_dir,
        )

        # Wrap encode_prompt so we capture text encoder outputs without
        # touching the production pipeline's forward() body.
        self._original_encode_prompt = self.encode_prompt
        self.encode_prompt = self._wrapped_encode_prompt  # type: ignore[method-assign]

        # Capture every transformer forward call (cond + uncond per step).
        if hasattr(self, "transformer") and self.transformer is not None:
            self.transformer.register_forward_hook(
                self._transformer_forward_hook, with_kwargs=True
            )

    # -- forward() wrapper: set up per-request dump dir ---------------------

    def forward(self, req: Any) -> Any:  # type: ignore[override]
        if self._active and self._dump_dir_root and self._dump_enabled:
            req_id = self._get_request_id(req)
            self._current_dump_dir = os.path.join(self._dump_dir_root, req_id)
            os.makedirs(self._current_dump_dir, exist_ok=True)
            logger.warning("DumpMixin (rank 0) dumping to %s", self._current_dump_dir)
        else:
            self._current_dump_dir = None

        # Reset per-request state.
        self._current_step = 0
        self._cond_text_id = None
        self._uncond_text_id = None
        self._encode_prompt_call_count = 0
        try:
            return super().forward(req)  # type: ignore[misc]
        finally:
            self._current_dump_dir = None

    @staticmethod
    def _get_request_id(req: Any) -> str:
        ids = getattr(req, "request_ids", None)
        if ids:
            return str(ids[0])
        return f"req_{int(time.time() * 1e6)}"

    # -- noise-sampling overrides (the P1 contract) -------------------------

    def _sample_initial_noise(
        self,
        shape: tuple[int, ...],
        generator: torch.Generator | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._load_initial_noise_path:
            loaded = torch.load(self._load_initial_noise_path, map_location=device)
            if tuple(loaded.shape) == tuple(shape):
                tensor = loaded.to(device=device, dtype=dtype)
                logger.warning(
                    "DumpMixin loaded initial noise from %s (shape %s)",
                    self._load_initial_noise_path,
                    tuple(shape),
                )
            else:
                # Shape mismatch usually means this is a startup warmup/profile
                # pass with dummy inputs, not the real user request.
                logger.warning(
                    "DumpMixin SKIPPING initial-noise load: requested shape %s != "
                    "loaded shape %s (likely a warmup/profile pass)",
                    tuple(shape),
                    tuple(loaded.shape),
                )
                tensor = super()._sample_initial_noise(shape, generator, device, dtype)  # type: ignore[misc]
        else:
            tensor = super()._sample_initial_noise(shape, generator, device, dtype)  # type: ignore[misc]
        self._dump("z_initial", tensor)
        return tensor

    def _sample_step_noise(
        self,
        z: torch.Tensor,
        generator: torch.Generator | None,
        step_idx: int,
    ) -> torch.Tensor:
        if self._load_step_noise_dir:
            path = os.path.join(self._load_step_noise_dir, f"step_noise_{step_idx}.pt")
            if os.path.exists(path):
                loaded = torch.load(path, map_location=z.device).to(
                    device=z.device, dtype=z.dtype
                )
                if tuple(loaded.shape) == tuple(z.shape):
                    tensor = loaded
                else:
                    logger.warning(
                        "DumpMixin SKIPPING step_noise_%d load: shape %s != z %s "
                        "(likely warmup/profile pass)",
                        step_idx,
                        tuple(loaded.shape),
                        tuple(z.shape),
                    )
                    tensor = super()._sample_step_noise(z, generator, step_idx=step_idx)  # type: ignore[misc]
            else:
                tensor = super()._sample_step_noise(z, generator, step_idx=step_idx)  # type: ignore[misc]
        else:
            tensor = super()._sample_step_noise(z, generator, step_idx=step_idx)  # type: ignore[misc]
        self._dump(f"step_noise_{step_idx}", tensor)
        # _sample_step_noise is invoked at the END of step `step_idx` (just
        # before transitioning to the next step's denoising). Advance our
        # step counter so the next transformer-hook calls are tagged as the
        # next step.
        self._current_step = step_idx + 1
        return tensor

    # -- encode_prompt wrapper: capture text encoder outputs ----------------

    def _wrapped_encode_prompt(self, *args: Any, **kwargs: Any) -> Any:
        result = self._original_encode_prompt(*args, **kwargs)
        # First encode_prompt call is the positive prompt (cond), second is
        # the negative prompt (uncond). pipeline_marey.py:forward() always
        # calls in this order.
        idx = self._encode_prompt_call_count
        label = "cond" if idx == 0 else ("uncond" if idx == 1 else f"call{idx}")

        try:
            seq_cond, seq_cond_masks, vector_cond = result
        except (ValueError, TypeError) as e:
            logger.warning("DumpMixin encode_prompt unpack failed: %s", e)
            self._encode_prompt_call_count += 1
            return result

        # Cache identity of the first sequence tensor so the transformer
        # hook can later identify cond vs uncond by tensor identity.
        first_seq = seq_cond[0] if isinstance(seq_cond, (list, tuple)) else seq_cond
        if first_seq is not None:
            if label == "cond":
                self._cond_text_id = id(first_seq)
            elif label == "uncond":
                self._uncond_text_id = id(first_seq)

        if self._current_dump_dir:
            if isinstance(seq_cond, (list, tuple)):
                for j, t in enumerate(seq_cond):
                    if t is not None:
                        self._dump(f"encode_{label}_seq_cond_{j}", t)
            elif seq_cond is not None:
                self._dump(f"encode_{label}_seq_cond", seq_cond)
            if isinstance(seq_cond_masks, (list, tuple)):
                for j, t in enumerate(seq_cond_masks):
                    if t is not None:
                        self._dump(f"encode_{label}_seq_mask_{j}", t)
            elif seq_cond_masks is not None:
                self._dump(f"encode_{label}_seq_mask", seq_cond_masks)
            if vector_cond is not None:
                self._dump(f"encode_{label}_vector_cond", vector_cond)

        self._encode_prompt_call_count += 1
        return result

    # -- transformer forward hook: capture per-call IO ----------------------

    def _identify_call(self, encoder_hidden_states: Any) -> str:
        """Return ``"cond"``/``"uncond"`` based on the text-embed identity.

        Falls back to ``"unknown"`` if the encoder_hidden_states tensor doesn't
        match either cached identity (e.g. warmup pass with dummy inputs).
        """
        if encoder_hidden_states is None:
            return "unknown"
        first = (
            encoder_hidden_states[0]
            if isinstance(encoder_hidden_states, (list, tuple))
            else encoder_hidden_states
        )
        if first is None:
            return "unknown"
        ehs_id = id(first)
        if self._cond_text_id is not None and ehs_id == self._cond_text_id:
            return "cond"
        if self._uncond_text_id is not None and ehs_id == self._uncond_text_id:
            return "uncond"
        return "unknown"

    def _transformer_forward_hook(
        self,
        module: torch.nn.Module,
        args: tuple,
        kwargs: dict,
        output: Any,
    ) -> None:
        del module
        if not self._current_dump_dir:
            return

        ehs = kwargs.get("encoder_hidden_states")
        label = self._identify_call(ehs)
        prefix = f"step{self._current_step}_{label}"

        # Inputs
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is not None:
            self._dump(f"{prefix}_hidden_states", hs)
        for k in ("timestep", "vector_cond"):
            v = kwargs.get(k)
            if v is not None:
                self._dump(f"{prefix}_{k}", v)
        if isinstance(ehs, (list, tuple)):
            for j, t in enumerate(ehs):
                if t is not None:
                    self._dump(f"{prefix}_encoder_hidden_states_{j}", t)
        elif ehs is not None:
            self._dump(f"{prefix}_encoder_hidden_states", ehs)
        ef = kwargs.get("extra_features")
        if isinstance(ef, dict):
            for k, t in ef.items():
                if t is not None:
                    self._dump(f"{prefix}_extra_{k}", t)
        # Output (v_pred). The pipeline calls with return_dict=False, so output
        # is a tuple whose first element is the predicted velocity.
        v_pred = output[0] if isinstance(output, tuple) else getattr(output, "sample", output)
        if v_pred is not None:
            self._dump(f"{prefix}_v_pred", v_pred)

    # -- dump helper --------------------------------------------------------

    def _dump(self, name: str, tensor: torch.Tensor) -> None:
        if not self._current_dump_dir or not self._dump_enabled:
            return
        path = os.path.join(self._current_dump_dir, f"{name}.pt")
        # Detach + move to CPU so the saved file is portable, even if the
        # producer was on GPU.
        torch.save(tensor.detach().to("cpu"), path)
