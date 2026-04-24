# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Loader for the Marey spatiotemporal VAE, backed by ``wlam.models.vae``.

Re-uses wlam's :class:`TwoStageVAEInference` wrapper (chunking, scaling/bias,
overlap-and-drop decode, per-stage ``max_batch_size``) and injects
sequence-parallel sharding via the ``sp_shard`` / ``sp_gather`` hooks wlam
exposes inside its ``_encode`` / ``_decode``.

We only depend on ``wlam.models.vae.*`` — its imports (``torch``, ``einops``,
``diffusers``) are already in vllm-omni, so this is a light dependency. If
``wlam`` isn't installed, we fall back to ``$WLAM_SRC`` or a sibling-repo
path.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _ensure_wlam_importable() -> None:
    """Ensure ``wlam.models.vae`` is importable.

    Tries, in order: a regular import, ``$WLAM_SRC``, and a sibling-repo
    fallback (``<vllm-omni-parent>/wlam/src``).
    """
    try:
        import wlam.models.vae.vae_inference  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    candidates: list[Path] = []
    env_src = os.environ.get("WLAM_SRC")
    if env_src:
        candidates.append(Path(env_src).expanduser())
    here = Path(__file__).resolve()
    candidates.append(here.parents[6] / "wlam" / "src")

    for src in candidates:
        if src.is_dir() and (src / "wlam" / "models" / "vae").is_dir():
            sys.path.insert(0, str(src))
            logger.info("Added wlam source path for Marey VAE: %s", src)
            return

    raise ModuleNotFoundError(
        "Could not import `wlam.models.vae`. Install the wlam package, set "
        "the WLAM_SRC env var to the path of wlam's `src/` directory, or "
        "check out the wlam repo as a sibling of wlam-inference."
    )


def _build_sp_hooks() -> tuple[
    Callable[[torch.Tensor], torch.Tensor] | None,
    Callable[[torch.Tensor], torch.Tensor] | None,
]:
    """Build ``(sp_shard, sp_gather)`` callables over vllm-omni's SP group.

    Each pair shares a closure so the pad_size produced by the shard call
    can be consumed by the matching gather.
    """
    from vllm_omni.diffusion.distributed.parallel_state import (
        get_sequence_parallel_world_size,
    )
    if get_sequence_parallel_world_size() <= 1:
        return None, None

    from vllm_omni.diffusion.distributed.sp_sharding import (
        sp_gather as _sp_gather_primitive,
        sp_shard_with_padding,
    )

    state = {"pad_size": 0}

    def sp_shard(x: torch.Tensor) -> torch.Tensor:
        sharded, pad = sp_shard_with_padding(x, dim=0)
        state["pad_size"] = pad
        return sharded

    def sp_gather(x: torch.Tensor) -> torch.Tensor:
        y = _sp_gather_primitive(x, dim=0)
        pad = state["pad_size"]
        logging.info(f"sp_gather: pad: {pad}")
        if pad > 0:
            y = y.narrow(0, 0, y.size(0) - pad)
        return y

    return sp_shard, sp_gather


class MareyWlamVAE(nn.Module):
    """Thin adapter over wlam's :class:`TwoStageVAEInference`.

    Surfaces the handful of attributes the Marey pipeline reads
    (``latent_embed_dim``, ``downsample_factors``, ``frame_chunk_len``) and
    wires SP sharding through wlam's ``sp_shard`` / ``sp_gather`` hooks.
    """

    def __init__(
        self,
        inference: nn.Module,  # wlam.models.vae.vae_inference.TwoStageVAEInference
        *,
        enable_sequence_parallelism: bool = False,
    ) -> None:
        super().__init__()
        self.inference = inference
        self.enable_sequence_parallelism = enable_sequence_parallelism

    # ---- surface used by the Marey pipeline --------------------------------

    @property
    def model(self) -> nn.Module:
        return self.inference.model

    @property
    def latent_embed_dim(self) -> int:
        return int(self.model.latent_embed_dim)

    @property
    def downsample_factors(self) -> tuple[int, int, int]:
        print(f"MareyWlamVAE downsample_factors: {self.model.get_downsample_factors(0)}")
        return tuple(self.model.get_downsample_factors(0))  # type: ignore[return-value]

    @property
    def frame_chunk_len(self) -> int:
        return int(self.inference.cfg.frame_chunk_len)

    # ---- SP hooks ----------------------------------------------------------

    def _sp_hooks(
        self,
    ) -> tuple[
        Callable[[torch.Tensor], torch.Tensor] | None,
        Callable[[torch.Tensor], torch.Tensor] | None,
    ]:
        if not self.enable_sequence_parallelism:
            return None, None
        return _build_sp_hooks()

    # ---- encode / decode (pass-through with SP hooks) ----------------------

    def encode(
        self,
        x: torch.Tensor,
        compression: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        sp_shard, sp_gather = self._sp_hooks()
        return self.inference.encode(
            x, compression=compression, sp_shard=sp_shard, sp_gather=sp_gather
        )

    def decode(
        self,
        z: torch.Tensor,
        num_frames: int | None = None,
        spatial_size: tuple[int, int] | None = None,
        expansion: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        sp_shard, sp_gather = self._sp_hooks()
        return self.inference.decode(
            z,
            num_frames=num_frames,
            spatial_size=spatial_size,
            expansion=expansion,
            sp_shard=sp_shard,
            sp_gather=sp_gather,
        )


def load_vae(
    vae_config: dict[str, Any],
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[nn.Module | None, str | None]:
    """Load the Marey spatiotemporal VAE using ``wlam.models.vae``.

    Returns ``(vae_module, error_msg)``. ``error_msg`` is None on success.
    """
    vae_path = vae_config.get("cp_path", "")
    if not os.path.exists(vae_path):
        return None, f"VAE checkpoint not found at {vae_path}."

    try:
        _ensure_wlam_importable()
        from wlam.models.vae.two_stage_vae import TwoStageVAE
        from wlam.models.vae.vae_inference import TwoStageVAEInferenceConfig

        from vllm_omni.diffusion.distributed.parallel_state import (
            get_sequence_parallel_world_size,
        )
        enable_sequence_parallelism = get_sequence_parallel_world_size() > 1
        logger.info(f"[VAE] enable_sequence_parallelism: {enable_sequence_parallelism}")

        # Pre-load on CPU so we can honour the Marey config's non-strict flag
        # (``TwoStageVAEInferenceConfig`` would otherwise call load with
        # ``strict=True`` when given a path).

        decode_chunking_strategy = (
            "overlap-and-drop"
            if vae_config.get("extra_context_and_drop_strategy", False)
            else "basic"
        )

        max_batch_size = 16
        inference = TwoStageVAEInferenceConfig(
            checkpoint=vae_path,
            frame_chunk_len=vae_config["frame_chunk_len"],
            decode_chunking_strategy=decode_chunking_strategy,
            scaling_factor=vae_config.get("scaling_factor", 1.0),
            bias_factor=vae_config.get("bias_factor", 0.0),
            max_batch_size=max_batch_size,
            torch_compile_kwargs={
                'dynamic': False,
                'fullgraph': True,
                'mode':'default',
            }
        ).make()

        vae = MareyWlamVAE(
            inference,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        vae = vae.to(device, dtype).eval()
        logger.info(
            "Loaded wlam Marey VAE (latent_embed_dim=%s, downsample=%s) from %s. Frame chunk len: %s, max batch size: %s, decode chunking strategy: %s",
            vae.latent_embed_dim,
            vae.downsample_factors,
            vae_path,
            vae.frame_chunk_len,
            max_batch_size,
            decode_chunking_strategy,
        )
        return vae, None
    except Exception as e:
        raise e
        return None, f"Could not load Marey VAE from {vae_path}: {type(e).__name__}: {e}"
