import pytest
import torch

from vllm_omni.diffusion.models.marey.marey_transformer import MareyFluxAttention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_marey_attention_split_handles_joint_output() -> None:
    y = torch.randn(1, 370, 16)
    out = torch.randn(1, 1024 + 370, 16)

    x_out, y_out, has_text_output = MareyFluxAttention._split_attention_outputs(out, y, 1024, 370)

    assert x_out.shape == (1, 1024, 16)
    assert y_out.shape == (1, 370, 16)
    assert has_text_output is True


def test_marey_attention_split_handles_visual_only_output() -> None:
    y = torch.randn(1, 370, 16)
    out = torch.randn(1, 1024, 16)

    x_out, y_out, has_text_output = MareyFluxAttention._split_attention_outputs(out, y, 1024, 370)

    assert x_out.shape == (1, 1024, 16)
    assert torch.equal(y_out, torch.zeros_like(y))
    assert has_text_output is False


def test_marey_attention_split_rejects_unexpected_length() -> None:
    y = torch.randn(1, 370, 16)
    out = torch.randn(1, 42, 16)

    with pytest.raises(RuntimeError, match="Unexpected Marey attention output length 42"):
        MareyFluxAttention._split_attention_outputs(out, y, 1024, 370)
