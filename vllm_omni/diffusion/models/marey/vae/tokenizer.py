# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tokenizer wrapper around :class:`TwoStageVAE`.

Public surface mirrors ``PretrainedSpatioTemporalVAETokenizer`` from
opensora — ``encode``, ``decode``, ``forward``, plus ``out_channels``,
``downsample_factors``, ``frame_chunk_len``, etc. — so the Marey pipeline
doesn't need to change.

Sequence parallelism uses vllm-omni's own ``sp_shard_with_padding`` /
``sp_gather`` primitives in place of opensora's DeepSpeed-based
``communications.split`` / ``communications.gather``. Sharding happens on
the batch-of-chunks dimension (``dim=0``) after per-chunk rearrangement,
before per-rank ``max_batch_size`` sub-batching — preserving opensora's
ordering so ``max_batch_size`` remains a per-rank memory cap.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from .common import ceildiv
from .two_stage import TwoStageVAE

logger = logging.getLogger(__name__)


def _extra_context_and_drop_wrapper(fn, win_size: int, extra_frames_in: int, extra_frames_out: int | None = None):
    """Sliding-window decode with boundary drop — 1:1 port of opensora.

    Each window of ``win_size`` latent frames overlaps the previous by
    ``2 * extra_frames_in`` so that internal frames always see neighbouring
    context; the overlapping ``extra_frames_out`` pixel frames at each
    boundary are then dropped. The first and last windows contribute their
    leading/trailing ``extra_frames_out`` since they have no left/right
    context to drop.
    """
    extra_frames_out = extra_frames_in if extra_frames_out is None else extra_frames_out

    def apply(x, **kwargs):
        overlap_size = 2 * extra_frames_in
        x_wins = x.unfold(dimension=-3, size=win_size, step=win_size - overlap_size)
        x_wins = rearrange(x_wins, "B C num_wins ... win_size -> (B num_wins) C win_size ...")
        y_wins = fn(x_wins, **kwargs)
        y_wins = rearrange(
            y_wins, "(B num_wins) C win_size_out ... -> B C num_wins win_size_out ...", B=x.shape[0]
        )
        win_size_out = y_wins.shape[3]
        y_mids = y_wins[:, :, :, extra_frames_out : win_size_out - extra_frames_out]
        y_mids_flat = rearrange(y_mids, "B C N L ... -> B C (N L) ...")
        y_prefix = y_wins[:, :, 0, :extra_frames_out]
        y_postfix = y_wins[:, :, -1, win_size_out - extra_frames_out :]
        return torch.cat([y_prefix, y_mids_flat, y_postfix], dim=2)

    return apply


class SpatioTemporalVAETokenizer(nn.Module):
    """Inference-only tokenizer wrapping a :class:`TwoStageVAE`."""

    def __init__(
        self,
        two_stage_vae: TwoStageVAE,
        *,
        scaling_factor: float | list[float] = 1.0,
        bias_factor: float | list[float] = 0.0,
        frame_chunk_len: int | None = None,
        max_batch_size: int | None = None,
        reuse_as_spatial_vae: bool = False,
        spatial_vae_kwargs: dict[str, Any] | None = None,
        extra_context_and_drop_strategy: bool = False,
        enable_sequence_parallelism: bool = False,
        enable_vae_slicing: bool = True,
        default_skip_n_blocks: int | None = None,
    ) -> None:
        super().__init__()
        self.two_stage_vae = two_stage_vae

        if np.isscalar(scaling_factor):
            scaling_factor = [scaling_factor]
        if np.isscalar(bias_factor):
            bias_factor = [bias_factor]
        # (C, 1, 1, 1) layout so broadcasts align with (B, C, T, H, W).
        scaling = torch.as_tensor(scaling_factor)[:, None, None, None]
        bias = torch.as_tensor(bias_factor)[:, None, None, None]
        self.register_buffer("scaling_factor", scaling)
        self.register_buffer("bias_factor", bias)

        self.out_channels = two_stage_vae.latent_embed_dim
        self.frame_chunk_len = frame_chunk_len
        self.max_batch_size = max_batch_size

        if enable_vae_slicing:
            two_stage_vae.spatial_vae.module.enable_slicing()

        self.reuse_as_spatial_vae = reuse_as_spatial_vae
        if not reuse_as_spatial_vae and spatial_vae_kwargs is not None:
            # Nested spatial-only tokenizer path (unused by Marey config).
            # Constructed explicitly by the caller when they want to reuse
            # the encode pipeline for images only.
            self.spatial_vae = SpatioTemporalVAETokenizer(**spatial_vae_kwargs)
            if enable_vae_slicing:
                self.spatial_vae.two_stage_vae.spatial_vae.module.enable_slicing()

        self.extra_context_and_drop_strategy = extra_context_and_drop_strategy
        self.enable_sequence_parallelism = enable_sequence_parallelism

        if default_skip_n_blocks is None:
            default_skip_n_blocks = self.two_stage_vae.default_skip_n_blocks
        if isinstance(default_skip_n_blocks, (tuple, list)):
            self.default_encode_skip_n_blocks, self.default_decode_skip_n_blocks = default_skip_n_blocks
        else:
            assert isinstance(default_skip_n_blocks, int)
            self.default_encode_skip_n_blocks = default_skip_n_blocks
            self.default_decode_skip_n_blocks = default_skip_n_blocks

    # ---- properties mirroring opensora ------------------------------------

    @property
    def downsample_factors(self) -> tuple[int, int, int]:
        return self.two_stage_vae.downsample_factors

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self.two_stage_vae.patch_size

    @property
    def time_downsample_factor(self) -> int:
        return self.downsample_factors[0]

    @property
    def latent_frame_chunk_len(self) -> int | None:
        if self.frame_chunk_len is None:
            return None
        return self.frame_chunk_len // self.downsample_factors[0]

    @property
    def micro_frame_size(self) -> int | None:
        return self.frame_chunk_len

    @property
    def device(self) -> torch.device:
        return self.two_stage_vae.device

    @property
    def dtype(self) -> torch.dtype:
        return self.two_stage_vae.dtype

    def get_latent_size(self, input_size: list[int | None]) -> list[int | None]:
        return self.two_stage_vae.get_latent_size(input_size)

    # ---- SP helpers --------------------------------------------------------

    def _sp_world_size(self) -> int:
        if not self.enable_sequence_parallelism:
            return 1
        # Imported lazily: parallel_state initialisation is a runtime concern
        # and we don't want to require distributed init at import time.
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_sequence_parallel_world_size,
        )
        return get_sequence_parallel_world_size()

    def _sp_shard_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        if self._sp_world_size() <= 1:
            return x, 0
        from vllm_omni.diffusion.distributed.sp_sharding import sp_shard_with_padding
        return sp_shard_with_padding(x, dim=0)

    def _sp_gather_batch(self, x: torch.Tensor, pad_size: int) -> torch.Tensor:
        if self._sp_world_size() <= 1:
            return x
        from vllm_omni.diffusion.distributed.sp_sharding import sp_gather
        y = sp_gather(x, dim=0)
        if pad_size > 0:
            y = y.narrow(0, 0, y.size(0) - pad_size)
        return y

    # ---- encode / decode ---------------------------------------------------

    @torch.inference_mode()
    def encode(self, x: torch.Tensor, skip_first_n_down_blocks: int | None = None) -> torch.Tensor:
        if skip_first_n_down_blocks is None:
            skip_first_n_down_blocks = self.default_encode_skip_n_blocks
        B, num_frames = x.shape[0], x.shape[-3]

        if num_frames == 1:
            num_chunks = frames_per_chunk = latent_frames_per_chunk = 1
        else:
            if not self.frame_chunk_len or self.frame_chunk_len > num_frames:
                frames_per_chunk = num_frames
            else:
                frames_per_chunk = self.frame_chunk_len
            if num_frames % frames_per_chunk != 0:
                raise ValueError(f"{num_frames=} must be divisible by {self.frame_chunk_len=}")
            latent_frames_per_chunk = frames_per_chunk // self.time_downsample_factor
            num_chunks = num_frames // frames_per_chunk

        x = rearrange(
            x,
            "B C (num_chunks frames_per_chunk) H W -> (B num_chunks) C frames_per_chunk H W",
            B=B,
            frames_per_chunk=frames_per_chunk,
            num_chunks=num_chunks,
        )

        x, pad_size = self._sp_shard_batch(x)
        z = self._encode_batch_of_chunks(x, skip_first_n_down_blocks=skip_first_n_down_blocks)
        z = self._sp_gather_batch(z, pad_size)

        z = rearrange(
            z,
            "(B num_chunks) C frames_per_chunk H W -> B C (num_chunks frames_per_chunk) H W",
            B=B,
            frames_per_chunk=latent_frames_per_chunk,
            num_chunks=num_chunks,
        )
        return z

    def _encode_batch_of_chunks(self, x: torch.Tensor, skip_first_n_down_blocks: int = 0) -> torch.Tensor:
        total_batch_size, _, frames_per_chunk, _, _ = x.shape
        if self.max_batch_size is None:
            max_batch_size = total_batch_size
        else:
            max_batch_size = self.max_batch_size
            if frames_per_chunk == 1 and self.frame_chunk_len is not None:
                max_batch_size *= self.frame_chunk_len // self.time_downsample_factor

        num_batches = ceildiv(total_batch_size, max_batch_size) if total_batch_size > 0 else 1
        if total_batch_size == 0:
            # This shouldn't happen now that sp_shard_with_padding pads to
            # sp_size, but keep a dummy-pass fallback for local-rank-less
            # callers that use raw sharding.
            x_chunks = [torch.zeros((1, *x.shape[1:]), device=x.device, dtype=x.dtype)]
        else:
            x_chunks = list(torch.chunk(x, num_batches, dim=0))

        zs = []
        for x_chunk in x_chunks:
            first_stage = self.two_stage_vae.spatial_vae.encode(
                x_chunk, skip_first_n_down_blocks=skip_first_n_down_blocks
            )
            if frames_per_chunk == 1:
                first_stage = self.two_stage_vae.maybe_replicate_single_frame(first_stage)
            second_stage = self.two_stage_vae.temporal_vae.encode(first_stage).mean
            zs.append(second_stage)

        z = torch.cat(zs, dim=0)
        if total_batch_size == 0:
            z = z[:0]
        assert z.shape[0] == total_batch_size, f"{z.shape=} {total_batch_size=}"
        return (z + self.bias_factor) * self.scaling_factor

    @torch.inference_mode()
    def decode(
        self,
        z: torch.Tensor,
        num_frames: int,
        spatial_size: tuple[int, int] | None = None,
        skip_last_n_up_blocks: int | None = None,
    ) -> torch.Tensor:
        if skip_last_n_up_blocks is None:
            skip_last_n_up_blocks = self.default_decode_skip_n_blocks
        z = z / self.scaling_factor - self.bias_factor

        if num_frames == 1:
            x = self._decode_step(z, num_frames=1, spatial_size=spatial_size, skip_last_n_up_blocks=skip_last_n_up_blocks)
            assert x.shape[-3] == 1
            return x

        frame_chunk_len = self.frame_chunk_len if self.frame_chunk_len is not None else num_frames
        assert num_frames % frame_chunk_len == 0, f"{num_frames=} must be divisible by {frame_chunk_len=}"

        ctx_latent_frames_in = 1 if self.extra_context_and_drop_strategy else 0
        ctx_video_frames_out = ctx_latent_frames_in * self.time_downsample_factor
        decode_wrapper = _extra_context_and_drop_wrapper(
            fn=self._decode_step,
            win_size=self.latent_frame_chunk_len,
            extra_frames_in=ctx_latent_frames_in,
            extra_frames_out=ctx_video_frames_out,
        )
        return decode_wrapper(
            z,
            num_frames=frame_chunk_len,
            spatial_size=spatial_size,
            skip_last_n_up_blocks=skip_last_n_up_blocks,
        )

    def _decode_step(
        self,
        z: torch.Tensor,
        num_frames: int,
        spatial_size: tuple[int, int] | None,
        skip_last_n_up_blocks: int = 0,
    ) -> torch.Tensor:
        orig_batch_size = z.shape[0]

        if spatial_size is not None:
            ds_h, ds_w = self.two_stage_vae.spatial_vae.downsample_factors[1:]
            ds_h //= 2**skip_last_n_up_blocks
            ds_w //= 2**skip_last_n_up_blocks
            assert spatial_size[0] % ds_h == 0, "output height not divisible by spatial downsample"
            assert spatial_size[1] % ds_w == 0, "output width not divisible by spatial downsample"
            stage_two_spatial_size_out = (spatial_size[0] // ds_h, spatial_size[1] // ds_w)
        else:
            stage_two_spatial_size_out = None

        z, pad_size = self._sp_shard_batch(z)

        local_batch_size = z.shape[0]
        if local_batch_size == 0:
            z_chunks = [torch.zeros((1, *z.shape[1:]), device=z.device, dtype=z.dtype)]
        else:
            max_batch_size = orig_batch_size if self.max_batch_size is None else self.max_batch_size
            num_batches = ceildiv(local_batch_size, max_batch_size)
            z_chunks = list(torch.chunk(z, num_batches, dim=0))

        xs = []
        for z_chunk in z_chunks:
            second_stage_recon = self.two_stage_vae.temporal_vae.decode(
                z_chunk,
                num_frames=num_frames,
                spatial_size=stage_two_spatial_size_out,
            )
            xs.append(
                self.two_stage_vae.spatial_vae.decode(
                    second_stage_recon, skip_last_n_up_blocks=skip_last_n_up_blocks
                )
            )
        y = torch.cat(xs, dim=0)
        if local_batch_size == 0:
            y = y[:0]

        y = self._sp_gather_batch(y, pad_size)
        return y

    def encode_images(self, x: torch.Tensor) -> torch.Tensor:
        if self.reuse_as_spatial_vae:
            return self.encode(x)
        return self.spatial_vae.encode(x)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_frames = x.shape[-3]
        if num_frames == 1:
            first_stage = self.two_stage_vae.spatial_vae.encode(x)
            first_stage = self.two_stage_vae.maybe_replicate_single_frame(first_stage)
            second_stage_recon, _, _ = self.two_stage_vae.temporal_vae(first_stage, sample_posterior=False)
            second_stage_recon = second_stage_recon[:, :, -1:]
            return self.two_stage_vae.spatial_vae.decode(second_stage_recon)

        frame_chunk_len = self.frame_chunk_len if self.frame_chunk_len is not None else num_frames
        assert num_frames % frame_chunk_len == 0
        x_chunks = torch.split(x, frame_chunk_len, dim=-3)

        x_recons = []
        for x_chunk in x_chunks:
            first_stage = self.two_stage_vae.spatial_vae.encode(x_chunk)
            second_stage_recon, _, _ = self.two_stage_vae.temporal_vae(first_stage, sample_posterior=False)
            x_recons.append(self.two_stage_vae.spatial_vae.decode(second_stage_recon))
        return torch.cat(x_recons, dim=-3)

    # ---- constructors ------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        cp_path: str,
        *,
        strict_loading: bool = False,
        map_location: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "SpatioTemporalVAETokenizer":
        """Build a tokenizer from a Lightning VAE checkpoint."""
        two_stage = TwoStageVAE.from_lightning_checkpoint(
            cp_path, strict=strict_loading, map_location=map_location
        )
        return cls(two_stage, **kwargs)
