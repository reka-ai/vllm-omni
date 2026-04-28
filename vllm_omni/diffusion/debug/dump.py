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

  - ``MAREY_DUMP_DIR``                   — root dir; per-request subdir created on each forward()
  - ``MAREY_LOAD_INITIAL_NOISE``         — path to a .pt file with the initial noise tensor
  - ``MAREY_LOAD_STEP_NOISE_DIR``        — dir containing ``step_noise_<i>.pt`` files
  - ``MAREY_LOAD_TEXT_EMBEDS_DIR``       — dir containing ``encode_{cond,uncond}_*.pt`` from a
                                           reference run; substitutes text-encoder outputs
  - ``MAREY_LOAD_TRANSFORMER_INPUTS_DIR``— dir containing ``step<i>_<label>_*.pt`` from a
                                           reference run; substitutes every per-step transformer
                                           input (hidden_states, timestep, encoder_hidden_states,
                                           vector_cond, extra_features) via a forward pre-hook
  - ``MAREY_LOAD_COND_FRAMES_PATH``      — path to ``cond_frames.pt`` from a reference run;
                                           substitutes the I2V VAE-encoded conditioning latent
                                           that ``MareyPipeline._encode_cond_frames`` returned.
  - ``MAREY_LOAD_COND_OFFSETS_PATH``     — path to ``cond_offsets.pt`` from a reference run;
                                           substitutes the I2V latent-time offsets.
  - ``MAREY_LOAD_PIPELINE_LATENTS_DIR``  — dir containing ``step<i>_cond_hidden_states.pt`` from a
                                           reference run; substitutes the pipeline's local
                                           ``z`` variable at the START of each denoising
                                           iteration (the value used both by the transformer
                                           call AND by the inline DDPM update). Defines the L0
                                           tier — strictest possible isolation: only the LAST
                                           iteration's transformer drift can propagate to
                                           ``z_final`` because every prior step's drift gets
                                           overwritten on the next iteration boundary.
  - ``MAREY_DUMP_BLOCKS_AT_STEPS``       — comma-separated step indices at which to dump every
                                           DiT block's (x, y) output as
                                           ``step<i>_<label>_block<b>_{x,y}_out.pt`` AND the
                                           transformer-internal I2V intermediates
                                           (``t0_emb``, ``x_after_concat``, ``x_t_mask``,
                                           ``x_pre_slice``). Default unset. Use ``"0"`` for
                                           step-0-only localization of transformer divergence.

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

I2V (image-to-video) tensors. (1)(2)(3) are once-per-request; (4)(5) are per-step
and gated by ``MAREY_DUMP_BLOCKS_AT_STEPS`` to keep file count bounded.

  cond_images.pt                           — (1, C=3, T_kf, H, W) in [-1, 1], pre-VAE
  cond_frames.pt                           — (1, C_vae, T_lat, H_lat, W_lat), post-VAE
  cond_offsets.pt                          — int64 (T_kf,), latent-time offsets
  step<i>_<label>_t0_emb.pt                — (B, hidden), t_embedder(zeros)+vec_cond
  step<i>_<label>_x_after_concat.pt        — pre-SP-shard concat result (full tensor)
  step<i>_<label>_x_t_mask.pt              — pre-SP-shard mask (bool)
  step<i>_<label>_x_pre_slice.pt           — post-final_layer post-all-gather, pre-slice

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
        self._load_text_embeds_dir: str | None = (
            os.environ.get("MAREY_LOAD_TEXT_EMBEDS_DIR") or None
        )
        self._load_transformer_inputs_dir: str | None = (
            os.environ.get("MAREY_LOAD_TRANSFORMER_INPUTS_DIR") or None
        )
        self._load_cond_frames_path: str | None = (
            os.environ.get("MAREY_LOAD_COND_FRAMES_PATH") or None
        )
        self._load_cond_offsets_path: str | None = (
            os.environ.get("MAREY_LOAD_COND_OFFSETS_PATH") or None
        )
        self._load_pipeline_latents_dir: str | None = (
            os.environ.get("MAREY_LOAD_PIPELINE_LATENTS_DIR") or None
        )
        _dump_blocks_env = os.environ.get("MAREY_DUMP_BLOCKS_AT_STEPS", "")
        self._dump_blocks_at_steps: set[int] = {
            int(s) for s in _dump_blocks_env.split(",") if s.strip().isdigit()
        }

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
            or self._load_text_embeds_dir
            or self._load_transformer_inputs_dir
            or self._load_cond_frames_path
            or self._load_cond_offsets_path
            or self._load_pipeline_latents_dir
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
        # Label resolved by the pre-hook; consumed by the post-hook of the
        # same call. At L1, the pre-hook replaces encoder_hidden_states so the
        # post-hook can no longer match by id() — this cache bridges them.
        self._pending_call_label: str | None = None

        if not self._active:
            return

        logger.warning(
            "DumpMixin active: dump_dir=%s load_initial=%s load_step_dir=%s "
            "load_text_embeds_dir=%s load_transformer_inputs_dir=%s "
            "load_cond_frames_path=%s load_cond_offsets_path=%s",
            self._dump_dir_root,
            self._load_initial_noise_path,
            self._load_step_noise_dir,
            self._load_text_embeds_dir,
            self._load_transformer_inputs_dir,
            self._load_cond_frames_path,
            self._load_cond_offsets_path,
        )

        # Wrap encode_prompt so we capture text encoder outputs without
        # touching the production pipeline's forward() body.
        self._original_encode_prompt = self.encode_prompt
        self.encode_prompt = self._wrapped_encode_prompt  # type: ignore[method-assign]

        # Capture every transformer forward call (cond + uncond per step).
        # Pre-hook is always registered so the label stash works even when the
        # MAREY_LOAD_TRANSFORMER_INPUTS_DIR env var isn't set (post-hook reads
        # the stash first, falls back to id-matching on the current ehs).
        if hasattr(self, "transformer") and self.transformer is not None:
            # Attach self to the transformer so its I2V branch can call back
            # to ``_dump_i2v_intermediate`` for tensors that are local to
            # ``MareyTransformer.forward()`` (t0_emb, x_after_concat,
            # x_t_mask, x_pre_slice). MUST bypass ``nn.Module.__setattr__``
            # via the instance ``__dict__``: a plain assignment would store
            # ``MareyPipeline`` (itself an ``nn.Module``) as a child module
            # of ``self.transformer``, creating a pipeline ↔ transformer
            # cycle that breaks recursive ``module.train()`` / ``eval()``.
            self.transformer.__dict__["_dump_pipeline"] = self

            self.transformer.register_forward_pre_hook(
                self._transformer_forward_pre_hook, with_kwargs=True
            )
            self.transformer.register_forward_hook(
                self._transformer_forward_hook, with_kwargs=True
            )
            # Optional per-DiT-block output capture at specified steps.
            # Block input/output is (x, y); dump as
            # step<i>_<label>_block<b>_{x,y}_{in,out}.pt.
            if self._dump_blocks_at_steps and hasattr(self.transformer, "blocks"):
                logger.warning(
                    "DumpMixin per-block dump enabled at steps %s",
                    sorted(self._dump_blocks_at_steps),
                )
                for block_idx, block in enumerate(self.transformer.blocks):
                    block.register_forward_pre_hook(
                        self._make_block_pre_hook(block_idx), with_kwargs=True
                    )
                    block.register_forward_hook(
                        self._make_block_hook(block_idx), with_kwargs=True
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
        self._pending_call_label = None
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

        # Substitute reference text-encoder outputs BEFORE id-caching so
        # the identity that flows into the transformer matches what we cache.
        if self._load_text_embeds_dir:
            seq_cond, seq_cond_masks, vector_cond = self._load_text_embeds(
                label, seq_cond, seq_cond_masks, vector_cond
            )
            result = (seq_cond, seq_cond_masks, vector_cond)

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

    def _load_text_embeds(
        self,
        label: str,
        seq_cond: Any,
        seq_cond_masks: Any,
        vector_cond: Any,
    ) -> tuple[Any, Any, Any]:
        """Substitute reference ``encode_<label>_*`` tensors into the return."""
        assert self._load_text_embeds_dir
        base = self._load_text_embeds_dir

        def _load_if_exists(path: str, ref: torch.Tensor) -> torch.Tensor:
            if not os.path.exists(path):
                logger.warning("DumpMixin text-embed: %s absent, keeping vllm-omni value", path)
                return ref
            loaded = torch.load(path, map_location=ref.device).to(
                device=ref.device, dtype=ref.dtype
            )
            if tuple(loaded.shape) != tuple(ref.shape):
                logger.warning(
                    "DumpMixin SKIPPING text-embed %s: shape %s != ref %s",
                    path, tuple(loaded.shape), tuple(ref.shape),
                )
                return ref
            return loaded

        if isinstance(seq_cond, (list, tuple)):
            new_seq = type(seq_cond)(
                _load_if_exists(
                    os.path.join(base, f"encode_{label}_seq_cond_{j}.pt"), t
                ) if t is not None else t
                for j, t in enumerate(seq_cond)
            )
        elif seq_cond is not None:
            new_seq = _load_if_exists(
                os.path.join(base, f"encode_{label}_seq_cond.pt"), seq_cond
            )
        else:
            new_seq = seq_cond

        if isinstance(seq_cond_masks, (list, tuple)):
            new_masks = type(seq_cond_masks)(
                _load_if_exists(
                    os.path.join(base, f"encode_{label}_seq_mask_{j}.pt"), t
                ) if t is not None else t
                for j, t in enumerate(seq_cond_masks)
            )
        elif seq_cond_masks is not None:
            new_masks = _load_if_exists(
                os.path.join(base, f"encode_{label}_seq_mask.pt"), seq_cond_masks
            )
        else:
            new_masks = seq_cond_masks

        if vector_cond is not None:
            # mv emits vector_cond_0; vllm-omni emits vector_cond — try both.
            vc_path = os.path.join(base, f"encode_{label}_vector_cond_0.pt")
            if not os.path.exists(vc_path):
                vc_path = os.path.join(base, f"encode_{label}_vector_cond.pt")
            new_vc = _load_if_exists(vc_path, vector_cond)
        else:
            new_vc = vector_cond

        return new_seq, new_masks, new_vc

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

    def _transformer_forward_pre_hook(
        self,
        module: torch.nn.Module,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        """Substitute every per-step transformer input from a reference dump.

        Engaged only when ``MAREY_LOAD_TRANSFORMER_INPUTS_DIR`` is set. Runs
        BEFORE the forward call; the post-hook then sees (and dumps) the
        injected tensors, which is desirable — per-step dumps should reflect
        what the transformer actually consumed.

        Also stashes the resolved cond/uncond label into
        ``self._pending_call_label`` so the post-hook dumps with the correct
        label even though it now sees the replaced ``encoder_hidden_states``.
        """
        del module
        ehs = kwargs.get("encoder_hidden_states")
        label = self._identify_call(ehs)
        # Stash so the post-hook can reuse it regardless of whether we inject.
        self._pending_call_label = label

        if not self._load_transformer_inputs_dir:
            return args, kwargs
        if label == "unknown":
            # Warmup/profile pass — don't inject.
            return args, kwargs

        base = self._load_transformer_inputs_dir
        prefix = f"step{self._current_step}_{label}"
        new_kwargs = dict(kwargs)
        new_args = list(args)

        def _load_and_replace(path: str, ref: torch.Tensor) -> torch.Tensor:
            if not os.path.exists(path):
                return ref
            loaded = torch.load(path, map_location=ref.device).to(
                device=ref.device, dtype=ref.dtype
            )
            if tuple(loaded.shape) != tuple(ref.shape):
                logger.warning(
                    "DumpMixin SKIPPING transformer-input %s: shape %s != ref %s",
                    path, tuple(loaded.shape), tuple(ref.shape),
                )
                return ref
            return loaded

        hs_ref = new_args[0] if new_args else new_kwargs.get("hidden_states")
        if hs_ref is not None:
            hs_new = _load_and_replace(
                os.path.join(base, f"{prefix}_hidden_states.pt"), hs_ref
            )
            if new_args:
                new_args[0] = hs_new
            else:
                new_kwargs["hidden_states"] = hs_new

        ts_ref = new_kwargs.get("timestep")
        if ts_ref is not None:
            new_kwargs["timestep"] = _load_and_replace(
                os.path.join(base, f"{prefix}_timestep.pt"), ts_ref
            )

        vc_ref = new_kwargs.get("vector_cond")
        if vc_ref is not None:
            # mv: vector_cond_0.pt; vllm-omni: vector_cond.pt.
            vc_path = os.path.join(base, f"{prefix}_vector_cond_0.pt")
            if not os.path.exists(vc_path):
                vc_path = os.path.join(base, f"{prefix}_vector_cond.pt")
            new_kwargs["vector_cond"] = _load_and_replace(vc_path, vc_ref)

        ehs_ref = new_kwargs.get("encoder_hidden_states")
        if isinstance(ehs_ref, (list, tuple)):
            new_kwargs["encoder_hidden_states"] = type(ehs_ref)(
                _load_and_replace(
                    os.path.join(base, f"{prefix}_encoder_hidden_states_{j}.pt"), t
                ) if t is not None else t
                for j, t in enumerate(ehs_ref)
            )
        elif ehs_ref is not None:
            new_kwargs["encoder_hidden_states"] = _load_and_replace(
                os.path.join(base, f"{prefix}_encoder_hidden_states.pt"), ehs_ref
            )

        ef_ref = new_kwargs.get("extra_features")
        if isinstance(ef_ref, dict):
            new_ef = {}
            for k, v in ef_ref.items():
                if v is None:
                    new_ef[k] = v
                    continue
                # mv names as kw_<k>, vllm-omni names as extra_<k>.
                mv_path = os.path.join(base, f"{prefix}_kw_{k}.pt")
                path = mv_path if os.path.exists(mv_path) else os.path.join(
                    base, f"{prefix}_extra_{k}.pt"
                )
                new_ef[k] = _load_and_replace(path, v)
            new_kwargs["extra_features"] = new_ef

        # height/width/fps/num_frames are separate kwargs on vllm-omni's
        # transformer but appear as extra_features entries on mv (dumped as
        # kw_<name> via fallthrough). Inject same way.
        for k in ("height", "width", "fps", "num_frames"):
            v = new_kwargs.get(k)
            if v is None:
                continue
            mv_path = os.path.join(base, f"{prefix}_kw_{k}.pt")
            path = mv_path if os.path.exists(mv_path) else os.path.join(
                base, f"{prefix}_extra_{k}.pt"
            )
            new_kwargs[k] = _load_and_replace(path, v)

        return tuple(new_args), new_kwargs

    def _transformer_forward_hook(
        self,
        module: torch.nn.Module,
        args: tuple,
        kwargs: dict,
        output: Any,
    ) -> None:
        del module
        # Pre-hook stashed the label (it sees the ORIGINAL encoder_hidden_states
        # before any injection replaces it). Consume + clear. Fall back to
        # id-matching on current kwargs only if the stash is missing (shouldn't
        # happen unless someone skips the pre-hook path).
        label = self._pending_call_label
        self._pending_call_label = None
        if label is None:
            label = self._identify_call(kwargs.get("encoder_hidden_states"))

        if not self._current_dump_dir:
            return

        prefix = f"step{self._current_step}_{label}"

        # Inputs
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is not None:
            self._dump(f"{prefix}_hidden_states", hs)
        for k in ("timestep", "vector_cond"):
            v = kwargs.get(k)
            if v is not None:
                self._dump(f"{prefix}_{k}", v)
        ehs = kwargs.get("encoder_hidden_states")
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
        # vllm-omni passes height/width/fps as separate transformer kwargs
        # (not inside extra_features). mv dumps them as kw_<name> via its
        # generic fallthrough. Dump them under the same `extra_` bucket so
        # the canonical_name normalization in compare_dumps.py pairs them.
        for k in ("height", "width", "fps", "num_frames"):
            v = kwargs.get(k)
            if v is not None:
                self._dump(f"{prefix}_extra_{k}", v)
        # Output (v_pred). The pipeline calls with return_dict=False, so output
        # is a tuple whose first element is the predicted velocity.
        v_pred = output[0] if isinstance(output, tuple) else getattr(output, "sample", output)
        if v_pred is not None:
            self._dump(f"{prefix}_v_pred", v_pred)

    # -- per-DiT-block output hook (optional, step-gated) -------------------

    def _make_block_hook(self, block_idx: int):
        """Factory for a forward hook that dumps a single block's (x, y) output
        at the steps listed in ``MAREY_DUMP_BLOCKS_AT_STEPS``.

        The label (cond/uncond) is read from ``self._pending_call_label`` which
        was stashed by the transformer pre-hook earlier in the same forward.
        """
        def _hook(module, args, kwargs, output):
            del module, args, kwargs
            if not self._current_dump_dir:
                return
            if self._current_step not in self._dump_blocks_at_steps:
                return
            label = self._pending_call_label or "unknown"
            prefix = f"step{self._current_step}_{label}_block{block_idx}"
            if isinstance(output, tuple) and len(output) >= 2:
                if output[0] is not None:
                    self._dump(f"{prefix}_x_out", output[0])
                if output[1] is not None:
                    self._dump(f"{prefix}_y_out", output[1])
            elif isinstance(output, torch.Tensor):
                self._dump(f"{prefix}_out", output)
        return _hook

    def _make_block_pre_hook(self, block_idx: int):
        """Factory for a forward pre-hook that dumps the block's INPUT (x, y)
        at the steps listed in ``MAREY_DUMP_BLOCKS_AT_STEPS``. Useful to
        check per-rank shard alignment between codebases.
        """
        def _hook(module, args, kwargs):
            del module
            if not self._current_dump_dir:
                return
            if self._current_step not in self._dump_blocks_at_steps:
                return
            label = self._pending_call_label or "unknown"
            prefix = f"step{self._current_step}_{label}_block{block_idx}"
            # MareyFluxBlock signature: forward(self, x, y, t_x, t_y, ...).
            # Args order: (x, y, t_x, t_y, ...). May come as args or kwargs.
            x_in = args[0] if len(args) > 0 else kwargs.get("x")
            y_in = args[1] if len(args) > 1 else kwargs.get("y")
            if x_in is not None:
                self._dump(f"{prefix}_x_in", x_in)
            if y_in is not None:
                self._dump(f"{prefix}_y_in", y_in)
        return _hook

    # -- I2V hooks (called by the pipeline / transformer) -------------------

    def _dump_cond_images(self, cond_images: torch.Tensor) -> None:
        """Capture the pre-VAE conditioning image stack as ``cond_images.pt``.

        Called once per request from ``MareyPipeline.forward()`` right before
        ``_encode_cond_frames``. Shape: ``(1, C=3, T_kf, H, W)`` in ``[-1, 1]``.
        """
        if self._current_dump_dir:
            self._dump("cond_images", cond_images)

    def _dump_cond_frames(
        self, cond_frames: torch.Tensor, cond_offsets: torch.Tensor
    ) -> None:
        """Capture the I2V VAE-encoded conditioning latent + latent offsets.

        Called once per request from ``MareyPipeline.forward()`` right after
        ``_encode_cond_frames`` returns. Shapes: ``cond_frames`` is
        ``(1, C_vae, T_lat, H_lat, W_lat)``; ``cond_offsets`` is int64 ``(T_kf,)``.
        """
        if self._current_dump_dir:
            self._dump("cond_frames", cond_frames)
            self._dump("cond_offsets", cond_offsets)

    def _maybe_load_cond_frames(self, default: torch.Tensor) -> torch.Tensor:
        """Substitute ``cond_frames`` from ``MAREY_LOAD_COND_FRAMES_PATH`` if set.

        Returns ``default`` unchanged when the env var is unset, the file is
        missing, or shapes don't match (warmup/profile pass).
        """
        if not self._load_cond_frames_path or default is None:
            return default
        if not os.path.exists(self._load_cond_frames_path):
            logger.warning(
                "DumpMixin cond_frames load: %s absent, keeping vllm-omni value",
                self._load_cond_frames_path,
            )
            return default
        loaded = torch.load(self._load_cond_frames_path, map_location=default.device)
        if tuple(loaded.shape) != tuple(default.shape):
            logger.warning(
                "DumpMixin SKIPPING cond_frames load: shape %s != ref %s",
                tuple(loaded.shape), tuple(default.shape),
            )
            return default
        loaded = loaded.to(device=default.device, dtype=default.dtype)
        logger.warning(
            "DumpMixin loaded cond_frames from %s (shape %s)",
            self._load_cond_frames_path, tuple(loaded.shape),
        )
        return loaded

    def _maybe_load_cond_offsets(self, default: torch.Tensor) -> torch.Tensor:
        """Substitute ``cond_offsets`` from ``MAREY_LOAD_COND_OFFSETS_PATH`` if set."""
        if not self._load_cond_offsets_path or default is None:
            return default
        if not os.path.exists(self._load_cond_offsets_path):
            logger.warning(
                "DumpMixin cond_offsets load: %s absent, keeping vllm-omni value",
                self._load_cond_offsets_path,
            )
            return default
        loaded = torch.load(self._load_cond_offsets_path, map_location=default.device)
        if tuple(loaded.shape) != tuple(default.shape):
            logger.warning(
                "DumpMixin SKIPPING cond_offsets load: shape %s != ref %s",
                tuple(loaded.shape), tuple(default.shape),
            )
            return default
        # Preserve ref dtype (int64) — don't coerce to default's dtype.
        loaded = loaded.to(device=default.device)
        logger.warning(
            "DumpMixin loaded cond_offsets from %s (values %s)",
            self._load_cond_offsets_path, loaded.tolist(),
        )
        return loaded

    def _maybe_load_pipeline_latents(
        self, default: torch.Tensor, step_idx: int
    ) -> torch.Tensor:
        """L0 hook: substitute the pipeline's ``z`` variable at the start of
        iteration ``step_idx`` with mv's ``step<step_idx>_cond_hidden_states.pt``.

        Called from ``MareyPipeline.forward()`` at the top of the denoising
        loop body. With the pre-hook already substituting the transformer's
        kwarg, this additionally resets the LOCAL ``z`` so the inline DDPM
        update (``x0 = z - sigma * v_pred``) also operates on mv's state —
        eliminating the L1 cross-step accumulation that compounds per-step
        v_pred drift through the scheduler over all 33 steps.

        No-op when ``MAREY_LOAD_PIPELINE_LATENTS_DIR`` is unset, when the
        per-step file is missing (e.g. warmup pass), or when shapes mismatch.
        """
        if not self._load_pipeline_latents_dir:
            return default
        path = os.path.join(
            self._load_pipeline_latents_dir, f"step{step_idx}_cond_hidden_states.pt"
        )
        if not os.path.exists(path):
            return default
        loaded = torch.load(path, map_location=default.device)
        if tuple(loaded.shape) != tuple(default.shape):
            logger.warning(
                "DumpMixin SKIPPING pipeline_latents step %d load: shape %s != ref %s",
                step_idx, tuple(loaded.shape), tuple(default.shape),
            )
            return default
        return loaded.to(device=default.device, dtype=default.dtype)

    def _dump_i2v_intermediate(self, name: str, tensor: torch.Tensor) -> None:
        """Capture an I2V transformer-internal tensor.

        Called from ``MareyTransformer.forward()`` for ``t0_emb``,
        ``x_after_concat``, ``x_t_mask``, and ``x_pre_slice``. Uses
        ``_pending_call_label`` (cond/uncond, set by the pre-hook) and
        ``_current_step`` to construct the filename. Gated by step in
        ``MAREY_DUMP_BLOCKS_AT_STEPS`` to keep file count bounded.
        """
        if not self._current_dump_dir:
            return
        if self._current_step not in self._dump_blocks_at_steps:
            return
        label = self._pending_call_label or "unknown"
        self._dump(f"step{self._current_step}_{label}_{name}", tensor)

    # -- final-latent hook (called by pipeline just before VAE decode) ------

    def _dump_final_latent(self, z: torch.Tensor) -> None:
        """Capture the final pre-VAE latent as ``z_final.pt``."""
        if self._current_dump_dir:
            self._dump("z_final", z)

    def _dump_timesteps(self, timesteps: Any) -> None:
        """Capture the 1-D sampling schedule as ``timesteps.pt``.

        Accepts a single 1-D tensor or a list of scalar/1-element tensors
        (vllm-omni builds timesteps as the latter).
        """
        if not self._current_dump_dir:
            return
        if isinstance(timesteps, (list, tuple)):
            t = torch.cat([x.flatten() for x in timesteps])
        else:
            t = timesteps
        self._dump("timesteps", t)

    # -- dump helper --------------------------------------------------------

    def _dump(self, name: str, tensor: torch.Tensor) -> None:
        if not self._current_dump_dir or not self._dump_enabled:
            return
        path = os.path.join(self._current_dump_dir, f"{name}.pt")
        # Detach + move to CPU so the saved file is portable, even if the
        # producer was on GPU.
        torch.save(tensor.detach().to("cpu"), path)
