# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Marey text encoder bundle: UL2 + CLIP + ByT5.

Split out of pipeline_marey.py so both the legacy monolithic pipeline and the
new staged `MareyTextEncoder` vLLM model can share the same loading and
encoding logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel


DEFAULT_NEGATIVE_PROMPT = (
    "<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, bright, "
    "vignette, artifacts, still, noise, texture, scanlines, videogame, 360 camera, "
    "VR, transition, flare, saturation, distorted, warped, wide angle, contrast, "
    "saturated, vibrant, glowing, cross dissolve, texture, videogame, saturation, "
    "cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, blown out, "
    "horrible, blurry, worst quality, bad, transition, dissolve, cross-dissolve, melt, "
    "fade in, fade out, wobbly, weird, low quality, plastic, stock footage, video camera, "
    "boring, static"
)


def extract_quotes(text: str) -> str:
    """Extract text between quotes for ByT5 encoding.
    Returns the full prompt if no quotes are found."""
    matches = re.findall(r'["\u201c\u201d](.*?)["\u201c\u201d]', text)
    return " ".join(matches) if matches else text


@dataclass
class TextEncoderConfig:
    ul2_pretrained: str = "google/ul2"
    clip_pretrained: str = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    byt5_pretrained: str = "google/byt5-large"
    ul2_max_length: int = 300
    clip_max_length: int = 77
    byt5_max_length: int = 70

    @classmethod
    def from_te_dict(cls, te_cfg: dict) -> "TextEncoderConfig":
        return cls(
            ul2_pretrained=te_cfg.get("ul2_pretrained", cls.ul2_pretrained),
            clip_pretrained=te_cfg.get("clip_pretrained", cls.clip_pretrained),
            byt5_pretrained=te_cfg.get("byt5_pretrained", cls.byt5_pretrained),
            ul2_max_length=te_cfg.get("ul2_max_length", cls.ul2_max_length),
            clip_max_length=te_cfg.get("clip_max_length", cls.clip_max_length),
            byt5_max_length=te_cfg.get("byt5_max_length", cls.byt5_max_length),
        )


class MareyTextEncoderBundle(nn.Module):
    """UL2 + CLIP + ByT5 encoder bundle.

    Holds the three tokenizer/encoder pairs and exposes a single
    :meth:`encode_prompt` entry point that matches the reference Marey
    encoding order (UL2 → CLIP → ByT5).
    """

    def __init__(self, cfg: TextEncoderConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype

        self.ul2_tokenizer = AutoTokenizer.from_pretrained(cfg.ul2_pretrained)
        self.ul2_model = T5EncoderModel.from_pretrained(
            cfg.ul2_pretrained, torch_dtype=dtype, device_map="cpu"
        ).eval()

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_pretrained)
        self.clip_model = CLIPTextModel.from_pretrained(
            cfg.clip_pretrained, torch_dtype=dtype, device_map="cpu"
        ).eval()

        self.byt5_tokenizer = AutoTokenizer.from_pretrained(cfg.byt5_pretrained)
        self.byt5_model = T5EncoderModel.from_pretrained(
            cfg.byt5_pretrained, torch_dtype=dtype, device_map="cpu"
        ).eval()

    @property
    def ul2_hidden_size(self) -> int:
        return int(self.ul2_model.config.d_model)

    @property
    def byt5_hidden_size(self) -> int:
        return int(self.byt5_model.config.d_model)

    @property
    def clip_hidden_size(self) -> int:
        return int(self.clip_model.config.hidden_size)

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        dtype: torch.dtype | None = None,
        quote_override: str | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Encode a single prompt with all three encoders.

        Returns:
            ``(seq_cond, seq_cond_masks, vector_cond)`` where:
              - ``seq_cond`` is ``[ul2_seq, byt5_seq]``,
              - ``seq_cond_masks`` is ``[ul2_mask, byt5_mask]`` (bool),
              - ``vector_cond`` is the CLIP pooled embedding.
        """
        out_dtype = dtype if dtype is not None else self.dtype

        # UL2
        ul2_inputs = self.ul2_tokenizer(
            prompt,
            padding="max_length",
            max_length=self.cfg.ul2_max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ul2_mask = ul2_inputs["attention_mask"].to(device, torch.bool)
        ul2_output = self.ul2_model(
            input_ids=ul2_inputs.input_ids.to(device),
            attention_mask=ul2_inputs.attention_mask.to(device),
        )
        ul2_seq = ul2_output.last_hidden_state.to(out_dtype)

        # CLIP (vector conditioning)
        clip_inputs = self.clip_tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.cfg.clip_max_length,
            truncation=True,
            return_tensors="pt",
        )
        clip_output = self.clip_model(input_ids=clip_inputs["input_ids"].to(device))
        vector_cond = clip_output.pooler_output.to(out_dtype)

        # ByT5 (glyph/quote channel)
        quote_text = quote_override if quote_override is not None else extract_quotes(prompt)
        byt5_inputs = self.byt5_tokenizer(
            quote_text,
            padding="max_length",
            max_length=self.cfg.byt5_max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        byt5_mask = byt5_inputs["attention_mask"].to(device, torch.bool)
        byt5_output = self.byt5_model(
            input_ids=byt5_inputs.input_ids.to(device),
            attention_mask=byt5_inputs.attention_mask.to(device),
        )
        byt5_seq = byt5_output.last_hidden_state.to(out_dtype)

        return [ul2_seq, byt5_seq], [ul2_mask, byt5_mask], vector_cond

    def to_device(self, device: torch.device) -> None:
        self.ul2_model.to(device)
        self.clip_model.to(device)
        self.byt5_model.to(device)

    def to_cpu(self) -> None:
        self.ul2_model.to("cpu")
        self.clip_model.to("cpu")
        self.byt5_model.to("cpu")
