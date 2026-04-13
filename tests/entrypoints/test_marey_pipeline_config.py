import importlib

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("vllm_omni.entrypoints.omni_diffusion", "OmniDiffusion"),
        ("vllm_omni.entrypoints.async_omni_diffusion", "AsyncOmniDiffusion"),
    ],
)
def test_marey_explicit_model_class_skips_hf_config_lookup(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
) -> None:
    (tmp_path / "config.yaml").write_text("name: marey\n")

    module = importlib.import_module(module_name)
    captured: dict[str, object] = {}

    def _unexpected_hf_lookup(*args, **kwargs):
        raise AssertionError("custom Marey config path should bypass HF config lookups")

    def _fake_make_engine(od_config):
        captured["od_config"] = od_config
        return object()

    monkeypatch.setattr(module, "get_hf_file_to_dict", _unexpected_hf_lookup)
    monkeypatch.setattr(module.DiffusionEngine, "make_engine", _fake_make_engine)

    model_cls = getattr(module, class_name)
    model_cls(model=str(tmp_path), model_class_name="MareyPipeline")

    od_config = captured["od_config"]
    assert od_config.model_class_name == "MareyPipeline"
    assert od_config.tf_model_config.to_dict() == {}


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("vllm_omni.entrypoints.omni_diffusion", "OmniDiffusion"),
        ("vllm_omni.entrypoints.async_omni_diffusion", "AsyncOmniDiffusion"),
    ],
)
def test_marey_requires_local_config_yaml(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
) -> None:
    module = importlib.import_module(module_name)

    def _unexpected_hf_lookup(*args, **kwargs):
        raise AssertionError("missing Marey config should fail before HF config lookups")

    monkeypatch.setattr(module, "get_hf_file_to_dict", _unexpected_hf_lookup)

    model_cls = getattr(module, class_name)
    with pytest.raises(ValueError, match="MareyPipeline requires a local checkpoint directory containing 'config.yaml'"):
        model_cls(model=str(tmp_path), model_class_name="MareyPipeline")


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("vllm_omni.entrypoints.omni_diffusion", "OmniDiffusion"),
        ("vllm_omni.entrypoints.async_omni_diffusion", "AsyncOmniDiffusion"),
    ],
)
def test_marey_model_index_skips_transformer_config_lookup(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
) -> None:
    (tmp_path / "config.yaml").write_text("name: marey\n")

    module = importlib.import_module(module_name)
    captured: dict[str, object] = {}

    def _fake_hf_lookup(filename: str, model: str, *args, **kwargs):
        if filename == "model_index.json":
            return {"_class_name": "MareyPipeline", "_diffusers_version": "0.0"}
        raise AssertionError(f"unexpected HF config lookup for {filename}")

    def _fake_make_engine(od_config):
        captured["od_config"] = od_config
        return object()

    monkeypatch.setattr(module, "get_hf_file_to_dict", _fake_hf_lookup)
    monkeypatch.setattr(module.DiffusionEngine, "make_engine", _fake_make_engine)

    model_cls = getattr(module, class_name)
    model_cls(model=str(tmp_path))

    od_config = captured["od_config"]
    assert od_config.model_class_name == "MareyPipeline"
    assert od_config.tf_model_config.to_dict() == {}
