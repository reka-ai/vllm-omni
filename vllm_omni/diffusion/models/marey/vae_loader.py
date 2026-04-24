# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Loader for the Marey spatiotemporal VAE, backed by ``wlam.models.vae``.

Returns a :class:`wlam.models.vae.vae_inference.TwoStageVAEInference` directly
(chunking, scaling/bias, overlap-and-drop decode, per-stage ``max_batch_size``).
Sequence-parallel sharding is plumbed by the consumer via the ``sp_shard`` /
``sp_gather`` hooks that wlam's ``_encode`` / ``_decode`` expose — see
:func:`build_marey_vae_sp_hooks`.

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


def build_marey_vae_sp_hooks() -> tuple[
    Callable[[torch.Tensor], torch.Tensor] | None,
    Callable[[torch.Tensor], torch.Tensor] | None,
]:
    """Build ``(sp_shard, sp_gather)`` callables over vllm-omni's SP group.

    Returns ``(None, None)`` when SP is disabled. The returned pair shares a
    closure so the pad_size produced by ``sp_shard`` is consumed by the
    matching ``sp_gather``.
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


def load_vae(
    vae_config: dict[str, Any],
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[nn.Module | None, str | None]:
    """Load the Marey spatiotemporal VAE using ``wlam.models.vae``.

    Returns ``(vae_module, error_msg)``. ``error_msg`` is None on success.
    On success ``vae_module`` is a ``TwoStageVAEInference`` moved to
    ``device`` / ``dtype`` and set to ``eval()``.
    """
    vae_path = vae_config.get("cp_path", "")
    if not os.path.exists(vae_path):
        return None, f"VAE checkpoint not found at {vae_path}."

    _ensure_wlam_importable()
    from wlam.models.vae.vae_inference import TwoStageVAEInferenceConfig

    decode_chunking_strategy = (
        "overlap-and-drop"
        if vae_config.get("extra_context_and_drop_strategy", False)
        else "basic"
    )

    max_batch_size = 8
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
            'mode': 'default',
        },
    ).make()

    inference = inference.to(device, dtype).eval()
    logger.info(
        "Loaded wlam Marey VAE (latent_embed_dim=%s, downsample=%s) from %s. "
        "Frame chunk len: %s, max batch size: %s, decode chunking strategy: %s",
        inference.model.latent_embed_dim,
        inference.model.get_downsample_factors(0),
        vae_path,
        inference.cfg.frame_chunk_len,
        max_batch_size,
        decode_chunking_strategy,
    )
    return inference, None
