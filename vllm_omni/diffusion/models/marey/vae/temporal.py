# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Temporal (3D) VAE — port of opensora ``vae_temporal``.

Drops ``causal`` convolutions (the Marey checkpoint uses ``conv_type='std'``)
and drops the optional ``embed_dim`` quant_conv wrappers (the checkpoint was
trained without them). Submodule naming mirrors opensora so the Lightning
checkpoint's ``temporal_vae.encoder.*`` / ``temporal_vae.decoder.*`` keys
load as-is — in particular ``StdConv3d`` keeps its ``.conv`` nested attribute
so state-dict entries like ``conv_in.conv.weight`` line up.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .common import cast_tuple, is_odd, pad_at_dim
from .distributions import DiagonalGaussianDistribution


class StdConv3d(nn.Module):
    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int | tuple[int, int, int],
        strides: int | tuple[int, int, int] = 1,
        pad_mode: str = "zeros",
        padding: int | tuple[int, int, int] | str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        t_k, h_k, w_k = kernel_size
        if padding is None:
            assert is_odd(t_k) and is_odd(h_k) and is_odd(w_k)
            padding = (t_k // 2, h_k // 2, w_k // 2)
        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=strides,
            padding_mode=pad_mode,
            padding=padding,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _get_conv_builder(conv_type: str) -> Callable[..., nn.Module]:
    if conv_type in ("std-3d", "std", "vanilla"):
        return StdConv3d
    raise ValueError(f"invalid convolution type `{conv_type}` (causal was dropped intentionally)")


def _get_activation_fn(activation: str) -> type[nn.Module]:
    if activation in ("swish", "silu"):
        return nn.SiLU
    if activation == "relu":
        return nn.ReLU
    raise ValueError(f"unsupported activation `{activation}`")


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        conv_fn: Callable[..., nn.Module],
        main_kernel_size: tuple[int, int, int] = (3, 3, 3),
        activation_fn: type[nn.Module] = nn.SiLU,
        use_conv_shortcut: bool = False,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.activate = activation_fn()
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = conv_fn(in_channels, filters, kernel_size=main_kernel_size, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, filters)
        self.conv2 = conv_fn(filters, filters, kernel_size=main_kernel_size, bias=False)
        if in_channels != filters:
            shortcut_kernel = main_kernel_size if use_conv_shortcut else (1, 1, 1)
            self.conv3 = conv_fn(in_channels, filters, kernel_size=shortcut_kernel, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.in_channels != self.filters:
            residual = self.conv3(residual)
        return x + residual


class Encoder(nn.Module):
    """Temporal VAE encoder."""

    def __init__(
        self,
        in_out_channels: int = 4,
        latent_embed_dim: int = 512,
        filters: int = 128,
        main_kernel_size: tuple[int, int, int] = (3, 3, 3),
        num_res_blocks: int = 4,
        conv_type: str = "std",
        channel_multipliers: tuple[int, ...] = (1, 2, 2, 4),
        temporal_downsample: tuple[bool, ...] = (False, True, True),
        spatial_downsample: tuple[bool, ...] = (False, False, False),
        num_groups: int = 32,
        activation_fn: str = "swish",
    ) -> None:
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim

        self.activation_fn = _get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = _get_conv_builder(conv_type)
        block_args = dict(
            main_kernel_size=main_kernel_size,
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        self.conv_in = self.conv_fn(in_out_channels, filters, kernel_size=main_kernel_size, bias=False)

        self.block_res_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        prev_filters = filters
        for i in range(self.num_blocks):
            filters_i = self.filters * self.channel_multipliers[i]
            block_items = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters_i, **block_args))
                prev_filters = filters_i
            self.block_res_blocks.append(block_items)

            if i < self.num_blocks - 1:
                t_down = self.temporal_downsample[i]
                s_down = self.spatial_downsample[i]
                if t_down or s_down:
                    strides = (2 if t_down else 1, 2 if s_down else 1, 2 if s_down else 1)
                    self.conv_blocks.append(
                        self.conv_fn(prev_filters, filters_i, kernel_size=main_kernel_size, strides=strides)
                    )
                    prev_filters = filters_i
                else:
                    self.conv_blocks.append(nn.Identity(prev_filters))
                    prev_filters = filters_i

        self.res_blocks = nn.ModuleList()
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(prev_filters, filters_i, **block_args))
            prev_filters = filters_i

        self.norm1 = nn.GroupNorm(self.num_groups, prev_filters)
        self.conv2 = self.conv_fn(prev_filters, self.embedding_dim, kernel_size=(1, 1, 1), padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for i in range(self.num_blocks):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i < self.num_blocks - 1:
                x = self.conv_blocks[i](x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """Temporal VAE decoder."""

    def __init__(
        self,
        in_out_channels: int = 4,
        latent_embed_dim: int = 512,
        filters: int = 128,
        main_kernel_size: tuple[int, int, int] = (3, 3, 3),
        num_res_blocks: int = 4,
        conv_type: str = "std",
        channel_multipliers: tuple[int, ...] = (1, 2, 2, 4),
        temporal_downsample: tuple[bool, ...] = (False, True, True),
        spatial_downsample: tuple[bool, ...] = (False, False, False),
        temporal_upsample_mode: str = "depth-to-time",
        upsample_interp_mode: str = "nearest-exact",
        num_groups: int = 32,
        activation_fn: str = "swish",
    ) -> None:
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim
        self.s_stride = 1  # hardcoded in opensora; decoder spatial upsample uses interpolation
        self.temporal_upsample_mode = temporal_upsample_mode
        self.upsample_interp_mode = upsample_interp_mode

        self.activation_fn = _get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = _get_conv_builder(conv_type)
        block_args = dict(
            main_kernel_size=main_kernel_size,
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        filters_last = self.filters * self.channel_multipliers[-1]
        prev_filters = filters_last

        self.conv1 = self.conv_fn(self.embedding_dim, filters_last, kernel_size=main_kernel_size, bias=True)

        self.res_blocks = nn.ModuleList()
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(filters_last, filters_last, **block_args))

        self.block_res_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        for i in reversed(range(self.num_blocks)):
            filters_i = self.filters * self.channel_multipliers[i]
            block_items = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters_i, **block_args))
                prev_filters = filters_i
            self.block_res_blocks.insert(0, block_items)

            if i > 0:
                if self.temporal_downsample[i - 1]:
                    t_stride, _ = self._temporal_upsample(i)
                    # depth-to-time: expand channel dim by t_stride (and historically by s_stride^2=1).
                    self.conv_blocks.insert(
                        0,
                        self.conv_fn(
                            prev_filters,
                            prev_filters * t_stride * self.s_stride * self.s_stride,
                            kernel_size=main_kernel_size,
                        ),
                    )
                else:
                    self.conv_blocks.insert(0, nn.Identity(prev_filters))

        self.norm1 = nn.GroupNorm(self.num_groups, prev_filters)
        self.conv_out = self.conv_fn(filters_i, in_out_channels, 3)

    def _temporal_upsample(self, block_i: int) -> tuple[int, int]:
        if self.temporal_upsample_mode == "depth-to-time":
            t_stride = 2 if self.temporal_downsample[block_i - 1] else 1
            t_interp = 1
        elif self.temporal_upsample_mode == "interpolate":
            t_stride = 1
            t_interp = 2 if self.temporal_downsample[block_i - 1] else 1
        else:
            raise ValueError(f"invalid temporal_upsample_mode {self.temporal_upsample_mode}")
        return t_stride, t_interp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)
        for i in reversed(range(self.num_blocks)):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i > 0:
                t_stride, t_interp = self._temporal_upsample(i)
                x = self.conv_blocks[i - 1](x)
                x = rearrange(
                    x,
                    "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
                    ts=t_stride,
                    hs=self.s_stride,
                    ws=self.s_stride,
                )
                spatial_interp = (2, 2) if self.spatial_downsample[i - 1] else (1, 1)
                interp_scales = (t_interp,) + spatial_interp
                if interp_scales != (1, 1, 1):
                    x = F.interpolate(x, scale_factor=interp_scales, mode=self.upsample_interp_mode)
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv_out(x)
        return x


class VAE_Temporal(nn.Module):
    """Temporal VAE used as the second stage of the two-stage VAE."""

    def __init__(
        self,
        in_out_channels: int = 4,
        latent_embed_dim: int = 4,
        filters: int = 128,
        main_kernel_size: tuple[int, int, int] = (3, 3, 3),
        num_res_blocks: int = 4,
        conv_type: str = "std",
        channel_multipliers: tuple[int, ...] = (1, 2, 2, 4),
        temporal_downsample: tuple[bool, ...] = (True, True, False),
        spatial_downsample: tuple[bool, ...] = (False, False, False),
        num_groups: int = 32,
        activation_fn: str = "swish",
        temporal_upsample_mode: str = "depth-to-time",
        upsample_interp_mode: str = "nearest-exact",
    ) -> None:
        super().__init__()

        self.time_downsample_factor = 2 ** sum(temporal_downsample)
        self.spatial_downsample_factor = 2 ** sum(spatial_downsample)
        self.downsample_factors = (
            self.time_downsample_factor,
            self.spatial_downsample_factor,
            self.spatial_downsample_factor,
        )
        self.out_channels = in_out_channels
        self.latent_embed_dim = latent_embed_dim

        self.encoder = Encoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim * 2,
            filters=filters,
            main_kernel_size=main_kernel_size,
            num_res_blocks=num_res_blocks,
            conv_type=conv_type,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            spatial_downsample=spatial_downsample,
            num_groups=num_groups,
            activation_fn=activation_fn,
        )
        self.decoder = Decoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim,
            filters=filters,
            main_kernel_size=main_kernel_size,
            num_res_blocks=num_res_blocks,
            conv_type=conv_type,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            spatial_downsample=spatial_downsample,
            num_groups=num_groups,
            activation_fn=activation_fn,
            temporal_upsample_mode=temporal_upsample_mode,
            upsample_interp_mode=upsample_interp_mode,
        )

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self.downsample_factors

    def get_time_padding(self, num_frames: int) -> int:
        if num_frames % self.time_downsample_factor == 0:
            return 0
        return self.time_downsample_factor - num_frames % self.time_downsample_factor

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        time_padding = self.get_time_padding(x.shape[2])
        x = pad_at_dim(x, (time_padding, 0), dim=2)
        moments = self.encoder(x)
        moments = moments.to(x.dtype)
        return DiagonalGaussianDistribution(moments)

    def decode(
        self,
        z: torch.Tensor,
        num_frames: int | None = None,
        spatial_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        x = self.decoder(z)

        if spatial_size is not None:
            h, w = spatial_size
            if h % self.spatial_downsample_factor > 1 or w % self.spatial_downsample_factor > 1:
                raise NotImplementedError(
                    "spatial_size not evenly divisible by temporal-VAE spatial downsample factor."
                )
        else:
            h, w = x.shape[-2:]
        frame_offset = 0 if num_frames is None else self.get_time_padding(num_frames)
        return x[:, :, frame_offset:, :h, :w]

    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> tuple[torch.Tensor, DiagonalGaussianDistribution, torch.Tensor]:
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        recon = self.decode(z, num_frames=x.shape[-3], spatial_size=x.shape[-2:])
        return recon, posterior, z
