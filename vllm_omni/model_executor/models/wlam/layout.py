from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

import torch


class WLAMTokenType(IntEnum):
    PAD = -1
    BEGIN_TEXT_SEQ = 0
    TEXT_CONTEXT = 1
    TEXT = 2
    BEGIN_IMAGE_SEQ = 3
    IMAGE_CONTEXT = 5
    IMAGE = 6
    BEGIN_VIDEO_SEQ = 7
    VIDEO_CONTEXT = 8
    VIDEO = 9
    DIFFUSION_TIME = 13
    BEGIN_VLM_IMAGE_SEQ = 14
    VLM_IMAGE_CONTEXT = 15
    VLM_IMAGE = 16
    ASPECT_RATIO = 17


@dataclass
class WLAMTokenLayout:
    vlm_idx: torch.Tensor
    diffusion_idx: torch.Tensor
    diffusion_target_idx: torch.Tensor
    text_target_idx: torch.Tensor
    text_embed_idx: torch.Tensor
    convnext_idx: torch.Tensor
    diffusion_latent_idx: torch.Tensor
    time_embed_idx: torch.Tensor
    n_total: int
    metadata_embed_indices: dict[int, torch.Tensor] = field(default_factory=dict)

    @property
    def has_diffusion(self) -> bool:
        return self.diffusion_idx.numel() > 0

    @property
    def has_convnext(self) -> bool:
        return self.convnext_idx.numel() > 0

    @staticmethod
    def from_token_type_ids(
        token_type_ids: torch.Tensor,
        *,
        metadata_token_types: list[int] | None = None,
    ) -> "WLAMTokenLayout":
        flat = token_type_ids.reshape(-1)

        is_diffusion_ctx = (
            (flat == int(WLAMTokenType.IMAGE_CONTEXT))
            | (flat == int(WLAMTokenType.VIDEO_CONTEXT))
            | (flat == int(WLAMTokenType.DIFFUSION_TIME))
        )
        metadata_indices: dict[int, torch.Tensor] = {}
        if metadata_token_types:
            for token_type in metadata_token_types:
                mask = flat == int(token_type)
                is_diffusion_ctx = is_diffusion_ctx | mask
                metadata_indices[int(token_type)] = mask.nonzero(as_tuple=True)[0]

        is_diffusion_target = (
            (flat == int(WLAMTokenType.IMAGE))
            | (flat == int(WLAMTokenType.VIDEO))
        )
        is_diffusion = is_diffusion_ctx | is_diffusion_target
        is_text = (
            (flat == int(WLAMTokenType.TEXT))
            | (flat == int(WLAMTokenType.TEXT_CONTEXT))
            | (flat == int(WLAMTokenType.BEGIN_TEXT_SEQ))
        )
        is_convnext = (
            (flat == int(WLAMTokenType.VLM_IMAGE))
            | (flat == int(WLAMTokenType.VLM_IMAGE_CONTEXT))
        )
        is_latent = (
            (flat == int(WLAMTokenType.IMAGE))
            | (flat == int(WLAMTokenType.IMAGE_CONTEXT))
            | (flat == int(WLAMTokenType.VIDEO))
            | (flat == int(WLAMTokenType.VIDEO_CONTEXT))
        )

        return WLAMTokenLayout(
            vlm_idx=(~is_diffusion).nonzero(as_tuple=True)[0],
            diffusion_idx=is_diffusion.nonzero(as_tuple=True)[0],
            diffusion_target_idx=is_diffusion_target.nonzero(as_tuple=True)[0],
            text_target_idx=(flat == int(WLAMTokenType.TEXT)).nonzero(as_tuple=True)[0],
            text_embed_idx=is_text.nonzero(as_tuple=True)[0],
            convnext_idx=is_convnext.nonzero(as_tuple=True)[0],
            diffusion_latent_idx=is_latent.nonzero(as_tuple=True)[0],
            time_embed_idx=(flat == int(WLAMTokenType.DIFFUSION_TIME)).nonzero(as_tuple=True)[0],
            n_total=flat.shape[0],
            metadata_embed_indices=metadata_indices,
        )

    def gather(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states[self.vlm_idx], hidden_states[self.diffusion_idx]

    def scatter(
        self,
        vlm_out: torch.Tensor,
        diffusion_out: torch.Tensor,
        *,
        out_dim: int | None = None,
    ) -> torch.Tensor:
        dim = int(out_dim or vlm_out.shape[-1])
        out = torch.empty(
            self.n_total,
            dim,
            dtype=vlm_out.dtype,
            device=vlm_out.device,
        )
        out[self.vlm_idx] = vlm_out
        out[self.diffusion_idx] = diffusion_out
        return out

    def apply(
        self,
        hidden_states: torch.Tensor,
        vlm_fn: Callable[[torch.Tensor], torch.Tensor],
        diffusion_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        out_dim: int | None = None,
    ) -> torch.Tensor:
        vlm_h, diffusion_h = self.gather(hidden_states)
        return self.scatter(vlm_fn(vlm_h), diffusion_fn(diffusion_h), out_dim=out_dim)
