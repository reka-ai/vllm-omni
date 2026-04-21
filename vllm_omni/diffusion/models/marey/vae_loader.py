# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""In-tree loader for the Marey spatiotemporal VAE.

Builds :class:`SpatioTemporalVAETokenizer` from a Lightning checkpoint. All
opensora/moonvalley_ai import-path gymnastics have been removed — the VAE
implementation now lives under ``vllm_omni/diffusion/models/marey/vae/``.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

from vllm_omni.diffusion.models.marey.vae.tokenizer import SpatioTemporalVAETokenizer

logger = logging.getLogger(__name__)


def load_vae(
    vae_config: dict,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[nn.Module | None, str | None]:
    """Load the Marey spatiotemporal VAE from a Lightning checkpoint.

    Returns ``(vae_module, error_msg)``. ``error_msg`` is None on success.
    """
    vae_path = vae_config.get("cp_path", "")
    if not os.path.exists(vae_path):
        return None, f"VAE checkpoint not found at {vae_path}."

    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_sequence_parallel_world_size,
        )
        enable_sequence_parallelism = get_sequence_parallel_world_size() > 1
        logger.info(f"[VAE] enable_sequence_parallelism: {enable_sequence_parallelism}")
        vae = SpatioTemporalVAETokenizer.from_checkpoint(
            vae_path,
            strict_loading=vae_config.get("strict_loading", False),
            map_location="cpu",
            scaling_factor=vae_config.get("scaling_factor", 1.0),
            bias_factor=vae_config.get("bias_factor", 0.0),
            frame_chunk_len=vae_config.get("frame_chunk_len"),
            max_batch_size=vae_config.get("max_batch_size"),
            reuse_as_spatial_vae=vae_config.get("reuse_as_spatial_vae", False),
            extra_context_and_drop_strategy=vae_config.get("extra_context_and_drop_strategy", False),
            enable_sequence_parallelism=enable_sequence_parallelism,
            enable_vae_slicing=vae_config.get("enable_vae_slicing", True),
        )
        vae = vae.to(device, dtype).eval()
        logger.info(
            "Loaded in-tree Marey VAE (out_channels=%s, downsample=%s) from %s",
            vae.out_channels,
            vae.downsample_factors,
            vae_path,
        )
        return vae, None
    except Exception as e:
        return None, f"Could not load Marey VAE from {vae_path}: {type(e).__name__}: {e}"
