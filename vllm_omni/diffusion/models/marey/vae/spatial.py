# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Spatial (2D) VAE encoder and decoder.

Thin port of ``moonvalley_ai/open_sora/opensora/models/vae/custom_autoencoder_kl.py``
that keeps only what Marey uses at inference time:

* ``double_z=True`` encoder (outputs ``2*latent_channels`` for mean+logvar),
* ``skip_first_n_down_blocks`` / ``skip_last_n_up_blocks`` hooks,
* optional ``decoder_block_out_channels`` for asymmetric decoder widths.

No ``ConfigMixin``, no attention-processor machinery, no tiling/slicing —
the tokenizer handles batching. Weight-key layout matches the diffusers
AutoencoderKL so Marey's Lightning checkpoint loads without per-layer renames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_block_types: tuple[str, ...],
        block_out_channels: tuple[int, ...],
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()
        assert len(down_block_types) == len(block_out_channels)

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            self.down_blocks.append(
                get_down_block(
                    down_block_type,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    add_downsample=not is_final_block,
                    resnet_eps=1e-6,
                    downsample_padding=0,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    attention_head_dim=output_channel,
                    temb_channels=None,
                )
            )

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, skip_first_n_down_blocks: int = 0) -> torch.Tensor:
        x = self.conv_in(x)
        for down_block in self.down_blocks[skip_first_n_down_blocks:]:
            x = down_block(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class SpatialDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_block_types: tuple[str, ...],
        block_out_channels: tuple[int, ...],
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()
        assert len(up_block_types) == len(block_out_channels)

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        self.up_blocks = nn.ModuleList()
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            self.up_blocks.append(
                get_up_block(
                    up_block_type,
                    num_layers=layers_per_block + 1,
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    prev_output_channel=None,
                    add_upsample=not is_final_block,
                    resnet_eps=1e-6,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    attention_head_dim=output_channel,
                    temb_channels=None,
                    resnet_time_scale_shift="default",
                )
            )

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, skip_last_n_up_blocks: int = 0) -> torch.Tensor:
        x = self.conv_in(x)

        # Skip the N blocks before the final one; always keep the final up_block
        # so the terminal resolution / channel count is preserved.
        if skip_last_n_up_blocks > 0:
            rest_up_blocks = self.up_blocks[:-1]
            final_up_blocks = list(rest_up_blocks[:-skip_last_n_up_blocks]) + [self.up_blocks[-1]]
        else:
            final_up_blocks = self.up_blocks

        x = self.mid_block(x, None)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        x = x.to(upscale_dtype)
        for up_block in final_up_blocks:
            x = up_block(x, None)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x
