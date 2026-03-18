"""Minimal comparison: OpenSora reference forward pass vs vLLM-omni forward pass.

Loads the same weights into both models and runs a single forward pass with
identical inputs, then compares the outputs to identify any discrepancies.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import safetensors.torch
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = "/app/wlam/models/checkpoints/marey/distilled-0001/config.yaml"
WEIGHTS_PATH = (
    "/app/wlam/models/checkpoints/marey/distilled-0001/"
    "epoch0-global_step7000_distilled/ema_inference_ckpt.safetensors"
)


def setup_opensora_imports():
    moonvalley_dir = str(REPO_ROOT / "moonvalley_ai")
    opensora_dir = str(REPO_ROOT / "moonvalley_ai" / "open_sora")
    if moonvalley_dir not in sys.path:
        sys.path.insert(0, moonvalley_dir)
    if opensora_dir not in sys.path:
        sys.path.insert(0, opensora_dir)

    for mod_name in (
        "opensora.datasets",
        "opensora.datasets.utils",
        "opensora.datasets.video_transforms",
        "opensora.datasets.datasets",
    ):
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = []
            stub.__package__ = mod_name
            sys.modules[mod_name] = stub


def load_opensora_model(config, device, dtype):
    """Build and load the OpenSora FluxControl model."""
    setup_opensora_imports()
    from opensora.models.stdit.flux_control import FluxControl, FluxControlConfig

    model_cfg = config.get("model", {})
    te_cfg = config.get("text_encoder", {})

    caption_channels = [
        te_cfg.get("ul2_hidden_size", 4096),
    ]
    byt5_hidden_size = te_cfg.get("byt5_hidden_size", None)
    if byt5_hidden_size:
        caption_channels.append(byt5_hidden_size)
    else:
        caption_channels.append(1536)

    vector_cond_channels = [te_cfg.get("clip_hidden_size", 768)]

    flux_config = FluxControlConfig(
        in_channels=16,
        out_channels=model_cfg.get("out_channels", 32),
        hidden_size=model_cfg.get("hidden_size", 5120),
        depth=model_cfg.get("depth", 42),
        depth_single_blocks=model_cfg.get("depth_single_blocks", 28),
        num_heads=model_cfg.get("num_heads", 40),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        patch_size=tuple(model_cfg.get("patch_size", [1, 2, 2])),
        caption_channels=caption_channels,
        model_max_length=[te_cfg.get("ul2_max_length", 300), te_cfg.get("byt5_max_length", 70)],
        vector_cond_channels=vector_cond_channels,
        qk_norm=model_cfg.get("qk_norm", True),
        rope_channels_ratio=model_cfg.get("rope_channels_ratio", 0.5),
        rope_dim=model_cfg.get("rope_dim", -1),
        add_pos_embed_at_every_block=model_cfg.get("add_pos_embed_at_every_block", True),
        class_dropout_prob=0.0,
        learned_pe=model_cfg.get("learned_pe", True),
        use_block_v2=True,
        mlp_type="swiglu",
        sequence_camera_condition=model_cfg.get("sequence_camera_condition", False),
        camera_dim=model_cfg.get("camera_dim", 0),
    )

    ref_model = FluxControl(flux_config)
    sd = safetensors.torch.load_file(WEIGHTS_PATH)
    missing, unexpected = ref_model.load_state_dict(sd, strict=False)
    print(f"OpenSora model loaded: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print(f"  Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected keys (first 10): {unexpected[:10]}")
    ref_model = ref_model.to(device, dtype).eval()
    return ref_model


def load_vllm_model(config, device, dtype):
    """Build and load our vLLM-omni MareyTransformer."""
    sys.path.insert(0, str(REPO_ROOT))

    from vllm_omni.diffusion.models.marey.marey_transformer import MareyTransformer

    model_cfg = config.get("model", {})
    te_cfg = config.get("text_encoder", {})

    caption_channels = [4096, 1536]
    vector_cond_channels = 768

    ul2_max_length = te_cfg.get("ul2_max_length", 300)
    byt5_max_length = te_cfg.get("byt5_max_length", 70)
    model_max_length = [ul2_max_length, byt5_max_length]

    our_model = MareyTransformer(
        in_channels=16,
        out_channels=model_cfg.get("out_channels", 32),
        hidden_size=model_cfg.get("hidden_size", 5120),
        depth=model_cfg.get("depth", 42),
        depth_single_blocks=model_cfg.get("depth_single_blocks", 28),
        num_heads=model_cfg.get("num_heads", 40),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        patch_size=tuple(model_cfg.get("patch_size", [1, 2, 2])),
        caption_channels=caption_channels,
        model_max_length=model_max_length,
        vector_cond_channels=vector_cond_channels,
        qk_norm=model_cfg.get("qk_norm", True),
        rope_channels_ratio=model_cfg.get("rope_channels_ratio", 0.5),
        rope_dim=model_cfg.get("rope_dim", -1),
        add_pos_embed_at_every_block=model_cfg.get("add_pos_embed_at_every_block", True),
        class_dropout_prob=0.0,
        learned_pe=model_cfg.get("learned_pe", True),
    )

    remap = [
        ("extra_features_embedders.fps.mlp.", "fps_embedder.mlp."),
        ("y_embedder.vector_embedding_0", "y_embedder.vector_embedding"),
    ]
    sd = safetensors.torch.load_file(WEIGHTS_PATH)
    remapped = {}
    for k, v in sd.items():
        rk = k
        for old, new in remap:
            if old in rk:
                rk = rk.replace(old, new)
                break
        remapped[rk] = v

    loaded = our_model.load_weights(remapped.items())
    print(f"vLLM model loaded: {len(loaded)} weights loaded")
    our_model = our_model.to(device, dtype).eval()
    return our_model


def compare():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Use a TINY resolution so it fits in memory alongside both models
    B, C, T, H_lat, W_lat = 1, 16, 2, 8, 8
    timestep_val = 500.0

    z = torch.randn(B, C, T, H_lat, W_lat, device=device, dtype=dtype)
    timestep = torch.tensor([timestep_val], device=device, dtype=dtype)
    ul2_seq = torch.randn(B, 300, 4096, device=device, dtype=dtype) * 0.1
    byt5_seq = torch.randn(B, 70, 1536, device=device, dtype=dtype) * 0.1
    clip_vec = torch.randn(B, 768, device=device, dtype=dtype) * 0.1
    height_t = torch.tensor([128.0], device=device, dtype=dtype)
    width_t = torch.tensor([128.0], device=device, dtype=dtype)
    fps_t = torch.tensor([24.0], device=device, dtype=dtype)

    print("\n=== Loading OpenSora reference model ===")
    ref_model = load_opensora_model(config, device, dtype)

    print("\n=== Running OpenSora forward pass ===")
    with torch.no_grad():
        from opensora.models.text_encoder import TextCond
        text_cond = TextCond(
            seq_cond=[ul2_seq, byt5_seq],
            vector_cond=[clip_vec],
            seq_cond_mask=None,
        )
        ref_out = ref_model(
            z, timestep,
            text_cond=text_cond,
            height=height_t,
            width=width_t,
            fps=fps_t,
        )
    if isinstance(ref_out, tuple):
        ref_out = ref_out[0]
    ref_out = ref_out.float()
    print(f"OpenSora output: shape={list(ref_out.shape)} "
          f"mean={ref_out.mean():.6f} std={ref_out.std():.6f}")

    # Free reference model
    del ref_model
    torch.cuda.empty_cache()

    print("\n=== Loading vLLM-omni model ===")
    our_model = load_vllm_model(config, device, dtype)

    print("\n=== Running vLLM-omni forward pass ===")
    with torch.no_grad():
        our_out = our_model(
            hidden_states=z,
            timestep=timestep,
            encoder_hidden_states=[ul2_seq, byt5_seq],
            vector_cond=clip_vec,
            height=height_t,
            width=width_t,
            fps=fps_t,
            return_dict=False,
        )[0]
    our_out = our_out.float()
    print(f"vLLM output:    shape={list(our_out.shape)} "
          f"mean={our_out.mean():.6f} std={our_out.std():.6f}")

    # Compare
    if ref_out.shape == our_out.shape:
        diff = (ref_out - our_out).abs()
        print(f"\n=== Comparison ===")
        print(f"Max abs diff: {diff.max():.6f}")
        print(f"Mean abs diff: {diff.mean():.6f}")
        print(f"Relative diff (mean): {(diff / (ref_out.abs() + 1e-8)).mean():.6f}")
        cos = torch.nn.functional.cosine_similarity(
            ref_out.flatten(), our_out.flatten(), dim=0
        )
        print(f"Cosine similarity: {cos:.6f}")
    else:
        print(f"Shape mismatch: ref={list(ref_out.shape)} vs ours={list(our_out.shape)}")
        print("Comparing first 16 channels...")
        r = ref_out[:, :16]
        o = our_out[:, :16]
        diff = (r - o).abs()
        print(f"Max abs diff: {diff.max():.6f}")
        print(f"Mean abs diff: {diff.mean():.6f}")


if __name__ == "__main__":
    compare()
