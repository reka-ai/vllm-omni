"""Compare full forward pass using a small model (2 blocks) between
OpenSora and vLLM-omni implementations.
"""
from __future__ import annotations

import os
import socket
import sys
import types
from pathlib import Path

import safetensors.torch as st
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = "/app/wlam/models/checkpoints/marey/distilled-0001/config.yaml"
WEIGHTS_PATH = (
    "/app/wlam/models/checkpoints/marey/distilled-0001/"
    "epoch0-global_step7000_distilled/ema_inference_ckpt.safetensors"
)

SMALL_DEPTH = 2
SMALL_SINGLE = 1


def init_tp1():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from vllm_omni.diffusion.distributed import parallel_state as dist_state
    if not dist_state.model_parallel_is_initialized():
        dist_state.init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method="env://",
        )
        dist_state.initialize_model_parallel(tensor_parallel_size=1)


def setup_opensora_imports():
    moonvalley_dir = str(REPO_ROOT / "moonvalley_ai")
    opensora_dir = str(REPO_ROOT / "moonvalley_ai" / "open_sora")
    for d in [moonvalley_dir, opensora_dir]:
        if d not in sys.path:
            sys.path.insert(0, d)
    for mod_name in (
        "opensora.datasets", "opensora.datasets.utils",
        "opensora.datasets.video_transforms", "opensora.datasets.datasets",
    ):
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = []
            stub.__package__ = mod_name
            sys.modules[mod_name] = stub


def load_subset_weights(block_limit=SMALL_DEPTH):
    """Load only the weights needed for the small model."""
    sd = st.load_file(WEIGHTS_PATH)
    subset = {}
    for k, v in sd.items():
        if k.startswith("blocks."):
            block_idx = int(k.split(".")[1])
            if block_idx >= block_limit:
                continue
        # Skip extra_features and camera (not used in small model)
        if k.startswith("extra_features_embedders.") and not k.startswith("extra_features_embedders.fps."):
            continue
        if k.startswith("camera_embedder."):
            continue
        if k == "rope.freqs":
            continue
        subset[k] = v
    del sd
    return subset


def build_ref_model(config, sd, device, dtype):
    """Build OpenSora FluxControl with reduced depth."""
    setup_opensora_imports()
    from opensora.models.stdit.flux_control import FluxControlV2, FluxControlConfig

    model_cfg = config.get("model", {})
    te_cfg = config.get("text_encoder", {})

    flux_config = FluxControlConfig(
        in_channels=16,
        out_channels=model_cfg.get("out_channels", 32),
        hidden_size=model_cfg.get("hidden_size", 5120),
        depth=SMALL_DEPTH,
        depth_single_blocks=SMALL_SINGLE,
        num_heads=model_cfg.get("num_heads", 40),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        patch_size=tuple(model_cfg.get("patch_size", [1, 2, 2])),
        caption_channels=[4096, 1536],
        model_max_length=[
            te_cfg.get("ul2_max_length", 300),
            te_cfg.get("byt5_max_length", 70),
        ],
        vector_cond_channels=[768],
        qk_norm=model_cfg.get("qk_norm", True),
        rope_channels_ratio=model_cfg.get("rope_channels_ratio", 0.5),
        rope_dim=model_cfg.get("rope_dim", -1),
        add_pos_embed_at_every_block=model_cfg.get("add_pos_embed_at_every_block", True),
        class_dropout_prob=0.0,
        learned_pe=model_cfg.get("learned_pe", True),
        use_block_v2=True,
        mlp_type="swiglu",
        sequence_camera_condition=False,
        camera_dim=0,
        extra_cross_attn_blocks=None,
        extra_cross_attn_mode=None,
        extra_cross_attn_variables=None,
    )
    model = FluxControlV2(flux_config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Ref model: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing and len(missing) <= 20:
        print(f"  Missing: {missing}")
    return model.to(device, dtype).eval()


def build_our_model(config, sd, device, dtype):
    """Build vLLM-omni MareyTransformer with reduced depth."""
    sys.path.insert(0, str(REPO_ROOT))
    from vllm_omni.diffusion.models.marey.marey_transformer import MareyTransformer

    model_cfg = config.get("model", {})
    te_cfg = config.get("text_encoder", {})

    model = MareyTransformer(
        in_channels=16,
        out_channels=model_cfg.get("out_channels", 32),
        hidden_size=model_cfg.get("hidden_size", 5120),
        depth=SMALL_DEPTH,
        depth_single_blocks=SMALL_SINGLE,
        num_heads=model_cfg.get("num_heads", 40),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        patch_size=tuple(model_cfg.get("patch_size", [1, 2, 2])),
        caption_channels=[4096, 1536],
        model_max_length=[
            te_cfg.get("ul2_max_length", 300),
            te_cfg.get("byt5_max_length", 70),
        ],
        vector_cond_channels=768,
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
    remapped = {}
    for k, v in sd.items():
        rk = k
        for old, new in remap:
            if old in rk:
                rk = rk.replace(old, new)
                break
        remapped[rk] = v

    loaded = model.load_weights(remapped.items())
    print(f"Our model: {len(loaded)} weights loaded")
    return model.to(device, dtype).eval()


def compare():
    init_tp1()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    sd = load_subset_weights()
    print(f"Loaded {len(sd)} weight tensors for {SMALL_DEPTH}-block model")

    # Inputs
    B, C, T, H_lat, W_lat = 1, 16, 2, 8, 8
    z = torch.randn(B, C, T, H_lat, W_lat, device=device, dtype=dtype)
    timestep = torch.tensor([500.0], device=device, dtype=dtype)
    ul2_seq = torch.randn(B, 300, 4096, device=device, dtype=dtype) * 0.1
    byt5_seq = torch.randn(B, 70, 1536, device=device, dtype=dtype) * 0.1
    clip_vec = torch.randn(B, 768, device=device, dtype=dtype) * 0.1
    height_t = torch.tensor([128.0], device=device, dtype=dtype)
    width_t = torch.tensor([128.0], device=device, dtype=dtype)
    fps_t = torch.tensor([24.0], device=device, dtype=dtype)

    # === Build and run reference model ===
    print("\n--- OpenSora Reference Model ---")
    ref_model = build_ref_model(config, sd, device, dtype)
    with torch.no_grad():
        from opensora.models.text_encoder.classes import UnstructuredTextCond
        seq_mask = torch.ones(B, 300 + 70, device=device, dtype=torch.int64)
        text_cond = UnstructuredTextCond(
            seq_cond=[ul2_seq, byt5_seq],
            vector_cond=[clip_vec],
            seq_cond_mask=seq_mask,
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
    ref_out_f = ref_out.float()
    print(f"Output shape: {list(ref_out.shape)}")
    print(f"Output mean={ref_out_f.mean():.6f} std={ref_out_f.std():.6f}")
    print(f"  ch0-15 mean={ref_out_f[:,:16].mean():.6f} std={ref_out_f[:,:16].std():.6f}")
    print(f"  ch16-31 mean={ref_out_f[:,16:].mean():.6f} std={ref_out_f[:,16:].std():.6f}")

    del ref_model
    torch.cuda.empty_cache()

    # === Build and run our model ===
    print("\n--- vLLM-omni Model ---")
    our_model = build_our_model(config, sd, device, dtype)
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
    our_out_f = our_out.float()
    print(f"Output shape: {list(our_out.shape)}")
    print(f"Output mean={our_out_f.mean():.6f} std={our_out_f.std():.6f}")
    print(f"  ch0-15 mean={our_out_f[:,:16].mean():.6f} std={our_out_f[:,:16].std():.6f}")
    if our_out.shape[1] > 16:
        print(f"  ch16-31 mean={our_out_f[:,16:].mean():.6f} std={our_out_f[:,16:].std():.6f}")

    del our_model
    torch.cuda.empty_cache()

    # === Compare ===
    print("\n--- Comparison ---")
    if ref_out.shape != our_out.shape:
        print(f"Shape mismatch: ref={list(ref_out.shape)} ours={list(our_out.shape)}")
        min_ch = min(ref_out.shape[1], our_out.shape[1])
        ref_cmp = ref_out_f[:, :min_ch]
        our_cmp = our_out_f[:, :min_ch]
    else:
        ref_cmp = ref_out_f
        our_cmp = our_out_f

    diff = (ref_cmp - our_cmp).abs()
    print(f"Max abs diff: {diff.max():.6f}")
    print(f"Mean abs diff: {diff.mean():.6f}")
    rel_diff = diff / (ref_cmp.abs() + 1e-8)
    print(f"Mean relative diff: {rel_diff.mean():.6f}")
    cos = torch.nn.functional.cosine_similarity(
        ref_cmp.flatten(), our_cmp.flatten(), dim=0,
    )
    print(f"Cosine similarity: {cos:.6f}")

    # Per-channel comparison
    for ch in range(min(ref_cmp.shape[1], 4)):
        r = ref_cmp[0, ch].flatten()
        o = our_cmp[0, ch].flatten()
        ch_cos = torch.nn.functional.cosine_similarity(r, o, dim=0)
        print(f"  ch{ch} cosine: {ch_cos:.6f}")


if __name__ == "__main__":
    compare()
