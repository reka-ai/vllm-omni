# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TwoStageVAE assembly + Lightning-checkpoint loader.

The Marey VAE checkpoint is a PyTorch Lightning ``.ckpt`` with:

* ``hyper_parameters.cfg.spatial_vae`` → opensora ``VideoAutoencoderKL`` config
  (``type``, ``cfg`` with down-block spec, channel schedule, etc.).
* ``hyper_parameters.cfg.temporal_vae`` → opensora ``VAE_Temporal`` config.
* ``state_dict`` with prefixes ``spatial_vae.module.{encoder,decoder}.*``,
  ``temporal_vae.{encoder,decoder}.*``, plus ``fid.*``/``vae_loss_fn.*`` which
  are training-only and discarded here.

We preserve the original weight-key layout so ``load_state_dict`` is a
direct drop-in — no per-layer renaming.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

from .distributions import DiagonalGaussianDistribution
from .spatial import SpatialDecoder, SpatialEncoder
from .temporal import VAE_Temporal

logger = logging.getLogger(__name__)


_DROP_STATE_DICT_PREFIXES = ("fid.", "vae_loss_fn.", "discriminator.")


class _SpatialVAEModule(nn.Module):
    """Matches the checkpoint path ``spatial_vae.module.{encoder,decoder}``."""

    def __init__(self, encoder: SpatialEncoder, decoder: SpatialDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def enable_slicing(self) -> None:
        # Tokenizer-level chunking subsumes the diffusers "enable_slicing" mode
        # for Marey; kept as a no-op so the tokenizer API stays identical.
        pass


class _SpatialVAE(nn.Module):
    """Port of opensora ``VideoAutoencoderKL`` restricted to what Marey uses.

    Wraps a ``_SpatialVAEModule`` under ``.module`` to match the checkpoint's
    ``spatial_vae.module.encoder.*`` / ``spatial_vae.module.decoder.*`` keys.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        down_block_types: tuple[str, ...],
        up_block_types: tuple[str, ...],
        block_out_channels: tuple[int, ...],
        decoder_block_out_channels: tuple[int, ...] | None = None,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention: bool = True,
        scaling_factor: float = 1.0,
        posterior_sample: str = "mean",
    ) -> None:
        super().__init__()
        encoder = SpatialEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )
        decoder_block_out_channels = decoder_block_out_channels or block_out_channels
        decoder = SpatialDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.module = _SpatialVAEModule(encoder, decoder)

        self.out_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.posterior_sample = posterior_sample
        # Each down block except the last halves the spatial resolution.
        spatial_down = 2 ** (len(block_out_channels) - 1)
        self.downsample_factors: tuple[int, int, int] = (1, spatial_down, spatial_down)

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self.downsample_factors

    def _posterior_sample(self, distribution: DiagonalGaussianDistribution) -> torch.Tensor:
        if self.posterior_sample == "mean":
            return distribution.mean
        if self.posterior_sample == "sample":
            return distribution.sample()
        if self.posterior_sample == "hiddens":
            return distribution.parameters
        raise ValueError(f"invalid posterior_sample `{self.posterior_sample}`")

    def encode(self, x: torch.Tensor, skip_first_n_down_blocks: int = 0) -> torch.Tensor:
        # x: (B, C, T, H, W) → fold time into batch for a 2D pass
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        h = self.module.encoder(x, skip_first_n_down_blocks=skip_first_n_down_blocks)
        z = self._posterior_sample(DiagonalGaussianDistribution(h)) * self.scaling_factor
        return rearrange(z, "(B T) C H W -> B C T H W", B=B)

    def decode(self, x: torch.Tensor, skip_last_n_up_blocks: int = 0, **_: Any) -> torch.Tensor:
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.module.decoder(x / self.scaling_factor, skip_last_n_up_blocks=skip_last_n_up_blocks)
        return rearrange(x, "(B T) C H W -> B C T H W", B=B)

    def get_latent_size(self, input_size: list[int | None]) -> list[int | None]:
        latent_size = []
        for size_i, down in zip(input_size, self.downsample_factors):
            if size_i is None:
                latent_size.append(None)
                continue
            assert size_i % down == 0, f"input {size_i} not divisible by downsample {down}"
            latent_size.append(size_i // down)
        return latent_size


def _build_spatial_vae(cfg: dict[str, Any]) -> _SpatialVAE:
    if cfg.get("type") != "VideoAutoencoderKL":
        raise NotImplementedError(
            f"spatial VAE type `{cfg.get('type')}` is not supported by the in-tree Marey VAE port."
        )
    inner = cfg.get("cfg", {}) or {}
    return _SpatialVAE(
        in_channels=inner["in_channels"],
        out_channels=inner["out_channels"],
        latent_channels=inner["latent_channels"],
        down_block_types=tuple(inner["down_block_types"]),
        up_block_types=tuple(inner["up_block_types"]),
        block_out_channels=tuple(inner["block_out_channels"]),
        decoder_block_out_channels=(
            tuple(inner["decoder_block_out_channels"]) if inner.get("decoder_block_out_channels") else None
        ),
        layers_per_block=inner.get("layers_per_block", 2),
        norm_num_groups=inner.get("norm_num_groups", 32),
        act_fn=inner.get("act_fn", "silu"),
        mid_block_add_attention=inner.get("mid_block_add_attention", True),
        scaling_factor=inner.get("scaling_factor", 1.0),
    )


def _build_temporal_vae(cfg: dict[str, Any]) -> VAE_Temporal:
    vae_type = cfg.get("type")
    if vae_type not in ("VAE_Temporal", "VAE_Temporal_SD", "VAE_Temporal_SD3"):
        raise NotImplementedError(f"temporal VAE type `{vae_type}` is not supported.")

    # ``VAE_Temporal_SD`` carried fixed hyper-parameters in opensora; they're
    # the same numbers opensora embeds, so we inline them rather than ship a
    # registry lookup.
    if vae_type == "VAE_Temporal_SD":
        defaults = dict(
            in_out_channels=4,
            latent_embed_dim=4,
            filters=128,
            num_res_blocks=4,
            channel_multipliers=(1, 2, 2, 4),
            temporal_downsample=(False, True, True),
            spatial_downsample=(False, False, False),
        )
    else:
        defaults = {}

    kwargs: dict[str, Any] = {**defaults}
    for k, v in cfg.items():
        if k in ("type", "from_pretrained"):
            continue
        if k == "channel_multipliers" or k == "temporal_downsample" or k == "spatial_downsample":
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v
    return VAE_Temporal(**kwargs)


class TwoStageVAE(nn.Module):
    """Spatial + temporal VAE wrapper matching opensora's ``SpatioTemporalVAE``."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.spatial_vae = _build_spatial_vae(cfg["spatial_vae"])
        self.temporal_vae = _build_temporal_vae(cfg["temporal_vae"])

        temporal_cfg = cfg["temporal_vae"]
        self.latent_embed_dim = temporal_cfg.get("latent_embed_dim", self.temporal_vae.latent_embed_dim)
        self.replicate_single_frames = cfg.get("replicate_single_frames", True)
        self.default_skip_n_blocks = cfg.get("default_skip_n_blocks", 0)

    @property
    def downsample_factors(self) -> tuple[int, int, int]:
        return tuple(
            s * t for s, t in zip(self.spatial_vae.downsample_factors, self.temporal_vae.downsample_factors)
        )  # type: ignore[return-value]

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self.downsample_factors

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def maybe_replicate_single_frame(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] == 1 and self.replicate_single_frames:
            return x.tile((1, 1, self.downsample_factors[0], 1, 1))
        return x

    def get_latent_size(self, input_size: list[int | None]) -> list[int | None]:
        return self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(input_size))

    @classmethod
    def from_lightning_checkpoint(
        cls,
        path: str,
        *,
        strict: bool = False,
        map_location: str | torch.device = "cpu",
    ) -> "TwoStageVAE":
        """Load ``cls`` from a Lightning-format checkpoint.

        Strips training-only state_dict entries (``fid.*``, ``vae_loss_fn.*``,
        ``discriminator.*``). ``strict=False`` is the sane default here because
        the checkpoint carries loss-module buffers that we don't need.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        try:
            cfg = ckpt["hyper_parameters"]["cfg"]
            state_dict: dict[str, torch.Tensor] = ckpt["state_dict"]
        except KeyError as e:
            raise ValueError(f"{path} is not a Lightning-format VAE checkpoint: missing {e!r}") from e

        model = cls(cfg)

        cleaned = {k: v for k, v in state_dict.items() if not k.startswith(_DROP_STATE_DICT_PREFIXES)}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(f"strict load failed: missing={missing}, unexpected={unexpected}")
        if missing:
            logger.warning("TwoStageVAE load: %d missing keys (first 5: %s)", len(missing), missing[:5])
        if unexpected:
            logger.warning(
                "TwoStageVAE load: %d unexpected keys (first 5: %s)", len(unexpected), unexpected[:5]
            )
        return model
