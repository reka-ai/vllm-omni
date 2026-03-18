"""Compare specific components between OpenSora and vLLM-omni implementations.

Uses raw checkpoint weights to manually compute outputs for each component
and compare them. Does NOT require loading the full 30B model.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import safetensors.torch as st
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = "/app/wlam/models/checkpoints/marey/distilled-0001/config.yaml"
WEIGHTS_PATH = (
    "/app/wlam/models/checkpoints/marey/distilled-0001/"
    "epoch0-global_step7000_distilled/ema_inference_ckpt.safetensors"
)


def compare_y_embedder():
    """Compare the CaptionEmbedder (y_embedder) output."""
    print("=" * 60)
    print("Comparing y_embedder (CaptionEmbedder)")
    print("=" * 60)

    sd = st.load_file(WEIGHTS_PATH)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Shared test inputs
    B = 1
    ul2_seq = torch.randn(B, 300, 4096, device=device, dtype=dtype) * 0.1
    byt5_seq = torch.randn(B, 70, 1536, device=device, dtype=dtype) * 0.1
    clip_vec = torch.randn(B, 768, device=device, dtype=dtype) * 0.1

    # === Manual computation (simulating OpenSora's CaptionEmbedder) ===
    # y_proj: SeqProjector with 2 projections
    # Projection 0 (UL2): Linear(4096, 5120) + LayerNorm(5120)
    w_ul2 = sd["y_embedder.y_proj.projections.0.0.weight"].to(device)
    b_ul2 = sd["y_embedder.y_proj.projections.0.0.bias"].to(device)
    ln_w_ul2 = sd["y_embedder.y_proj.projections.0.1.weight"].to(device)
    ln_b_ul2 = sd["y_embedder.y_proj.projections.0.1.bias"].to(device)

    w_byt5 = sd["y_embedder.y_proj.projections.1.0.weight"].to(device)
    b_byt5 = sd["y_embedder.y_proj.projections.1.0.bias"].to(device)
    ln_w_byt5 = sd["y_embedder.y_proj.projections.1.1.weight"].to(device)
    ln_b_byt5 = sd["y_embedder.y_proj.projections.1.1.bias"].to(device)

    ul2_proj = F.linear(ul2_seq, w_ul2, b_ul2)
    ul2_proj = F.layer_norm(ul2_proj, [5120], ln_w_ul2, ln_b_ul2)
    byt5_proj = F.linear(byt5_seq, w_byt5, b_byt5)
    byt5_proj = F.layer_norm(byt5_proj, [5120], ln_w_byt5, ln_b_byt5)
    ref_seq_cond = torch.cat([ul2_proj, byt5_proj], dim=1)  # [1, 370, 5120]

    # vector_proj: SeqProjector with 1 projection
    w_clip = sd["y_embedder.vector_proj.projections.0.0.weight"].to(device)
    b_clip = sd["y_embedder.vector_proj.projections.0.0.bias"].to(device)
    ln_w_clip = sd["y_embedder.vector_proj.projections.0.1.weight"].to(device)
    ln_b_clip = sd["y_embedder.vector_proj.projections.0.1.bias"].to(device)
    ref_vec_cond = F.linear(clip_vec, w_clip, b_clip)
    ref_vec_cond = F.layer_norm(ref_vec_cond, [5120], ln_w_clip, ln_b_clip)

    print(f"Reference seq_cond: shape={list(ref_seq_cond.shape)} "
          f"mean={ref_seq_cond.float().mean():.6f} std={ref_seq_cond.float().std():.6f}")
    print(f"Reference vec_cond: shape={list(ref_vec_cond.shape)} "
          f"mean={ref_vec_cond.float().mean():.6f} std={ref_vec_cond.float().std():.6f}")

    # === Our CaptionEmbedder ===
    sys.path.insert(0, str(REPO_ROOT))
    from vllm_omni.diffusion.models.marey.marey_transformer import CaptionEmbedder

    y_embedder = CaptionEmbedder(
        in_channels=[4096, 1536],
        hidden_size=5120,
        uncond_prob=0.0,
        token_num=[300, 70],
        vector_in_channels=768,
    )

    # Load weights manually
    y_keys = {k: v for k, v in sd.items() if k.startswith("y_embedder.")}
    remap = {"y_embedder.vector_embedding_0": "y_embedder.vector_embedding"}
    buf = dict(y_embedder.named_buffers())
    par = dict(y_embedder.named_parameters())
    loaded = 0
    for k, v in y_keys.items():
        rk = remap.get(k, k)
        short = rk.replace("y_embedder.", "")
        if short in par:
            par[short].data.copy_(v)
            loaded += 1
        elif short in buf:
            buf[short].copy_(v)
            loaded += 1
    print(f"Loaded {loaded} y_embedder weights")

    y_embedder = y_embedder.to(device, dtype).eval()

    with torch.no_grad():
        our_seq, our_mask, our_vec = y_embedder(
            [ul2_seq, byt5_seq], None, clip_vec, train=False,
        )

    print(f"Our seq_cond:     shape={list(our_seq.shape)} "
          f"mean={our_seq.float().mean():.6f} std={our_seq.float().std():.6f}")
    print(f"Our vec_cond:     shape={list(our_vec.shape)} "
          f"mean={our_vec.float().mean():.6f} std={our_vec.float().std():.6f}")

    # Compare
    seq_diff = (ref_seq_cond.float() - our_seq.float()).abs()
    vec_diff = (ref_vec_cond.float() - our_vec.float()).abs()
    print(f"\nseq_cond diff: max={seq_diff.max():.8f} mean={seq_diff.mean():.8f}")
    print(f"vec_cond diff: max={vec_diff.max():.8f} mean={vec_diff.mean():.8f}")

    del sd
    torch.cuda.empty_cache()
    return ref_seq_cond, our_seq, ref_vec_cond, our_vec


def compare_timestep_and_modulation():
    """Compare timestep embedding, t_block, and first block's modulation."""
    print("\n" + "=" * 60)
    print("Comparing timestep embedding + t_block + modulation")
    print("=" * 60)

    sd = st.load_file(WEIGHTS_PATH)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    hidden_size = 5120
    vec_cond = torch.randn(1, hidden_size, device=device, dtype=dtype) * 0.5
    timestep = torch.tensor([500.0], device=device, dtype=dtype)

    # === Manual t_embedder computation ===
    freq_dim = 256
    half = freq_dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, dtype=torch.float32) / half)
    args = timestep.float()[:, None] * freqs[None].to(device)
    t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)

    w0 = sd["t_embedder.mlp.0.weight"].to(device)
    b0 = sd["t_embedder.mlp.0.bias"].to(device)
    w2 = sd["t_embedder.mlp.2.weight"].to(device)
    b2 = sd["t_embedder.mlp.2.bias"].to(device)

    t_emb = F.linear(t_freq, w0, b0)
    t_emb = F.silu(t_emb)
    t_emb = F.linear(t_emb, w2, b2)
    t_emb = t_emb + vec_cond  # t + extra_emb

    # === t_block ===
    tw0 = sd["t_block.1.weight"].to(device)
    tb0 = sd["t_block.1.bias"].to(device)
    t_mlp = F.silu(t_emb)
    t_mlp = F.linear(t_mlp, tw0, tb0)
    t_mlp = F.silu(t_mlp)

    print(f"Reference t_emb: mean={t_emb.float().mean():.6f} std={t_emb.float().std():.6f}")
    print(f"Reference t_mlp: mean={t_mlp.float().mean():.6f} std={t_mlp.float().std():.6f}")

    # === Our implementation ===
    sys.path.insert(0, str(REPO_ROOT))
    from vllm_omni.diffusion.models.marey.marey_transformer import TimestepEmbedder

    t_embedder = TimestepEmbedder(hidden_size)
    t_embedder.mlp[0].weight.data.copy_(w0)
    t_embedder.mlp[0].bias.data.copy_(b0)
    t_embedder.mlp[2].weight.data.copy_(w2)
    t_embedder.mlp[2].bias.data.copy_(b2)
    t_embedder = t_embedder.to(device, dtype).eval()

    t_block = nn.Sequential(
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size, bias=True),
        nn.SiLU(),
    )
    t_block[1].weight.data.copy_(tw0)
    t_block[1].bias.data.copy_(tb0)
    t_block = t_block.to(device, dtype).eval()

    with torch.no_grad():
        our_t_emb = t_embedder(timestep, dtype=dtype)
        our_t_emb = our_t_emb + vec_cond
        our_t_mlp = t_block(our_t_emb)

    print(f"Our t_emb:       mean={our_t_emb.float().mean():.6f} std={our_t_emb.float().std():.6f}")
    print(f"Our t_mlp:       mean={our_t_mlp.float().mean():.6f} std={our_t_mlp.float().std():.6f}")

    t_diff = (t_emb.float() - our_t_emb.float()).abs()
    mlp_diff = (t_mlp.float() - our_t_mlp.float()).abs()
    print(f"\nt_emb diff:  max={t_diff.max():.8f} mean={t_diff.mean():.8f}")
    print(f"t_mlp diff:  max={mlp_diff.max():.8f} mean={mlp_diff.mean():.8f}")

    # === Block 0 modulation ===
    print("\n--- Block 0 modulation_x ---")
    mod_w = sd["blocks.0.modulation_x.weight"].to(device)
    mod_b = sd["blocks.0.modulation_x.bias"].to(device)
    ref_mod = F.linear(t_mlp, mod_w, mod_b)
    ref_chunks = ref_mod.reshape(-1, 6, hidden_size).chunk(6, dim=1)
    ref_shift, ref_scale, ref_gate = ref_chunks[0], ref_chunks[1], ref_chunks[2]
    print(f"Reference shift_msa: mean={ref_shift.float().mean():.6f} std={ref_shift.float().std():.6f}")
    print(f"Reference scale_msa: mean={ref_scale.float().mean():.6f} std={ref_scale.float().std():.6f}")
    print(f"Reference gate_msa:  mean={ref_gate.float().mean():.6f} std={ref_gate.float().std():.6f}")

    our_mod = F.linear(our_t_mlp, mod_w, mod_b)
    our_mod = our_mod.unsqueeze(1)
    our_chunks = our_mod.chunk(6, dim=-1)
    our_shift, our_scale, our_gate = our_chunks[0], our_chunks[1], our_chunks[2]
    print(f"Our shift_msa:      mean={our_shift.float().mean():.6f} std={our_shift.float().std():.6f}")
    print(f"Our scale_msa:      mean={our_scale.float().mean():.6f} std={our_scale.float().std():.6f}")
    print(f"Our gate_msa:       mean={our_gate.float().mean():.6f} std={our_gate.float().std():.6f}")

    mod_diff = (ref_shift.float().squeeze(1) - our_shift.float().squeeze(1)).abs()
    print(f"shift diff: max={mod_diff.max():.8f}")

    del sd
    torch.cuda.empty_cache()


def init_tp1():
    """Initialize TP=1 for single-GPU testing."""
    import socket
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


def compare_single_block_forward():
    """Compare a single block's full forward pass."""
    print("\n" + "=" * 60)
    print("Comparing Block 0 full forward pass")
    print("=" * 60)

    init_tp1()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    hidden_size = 5120
    num_heads = 40
    mlp_ratio = 4.0

    # Create matching inputs
    B, N_x, N_y = 1, 64, 370
    x = torch.randn(B, N_x, hidden_size, device=device, dtype=dtype) * 0.1
    y = torch.randn(B, N_y, hidden_size, device=device, dtype=dtype) * 0.1
    t_x = torch.randn(B, hidden_size, device=device, dtype=dtype) * 0.5
    t_y = t_x.clone()
    temporal_pos = torch.arange(N_x, device=device, dtype=dtype).unsqueeze(0)

    # === Load our MareyFluxBlock ===
    sys.path.insert(0, str(REPO_ROOT))
    from vllm_omni.diffusion.models.marey.marey_transformer import MareyFluxBlock

    our_block = MareyFluxBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        share_weights=False,
        rope_channels_ratio=0.5,
        rope_dim=-1,
        qk_norm=True,
        add_spatial_pos_emb=True,
    )

    sd = st.load_file(WEIGHTS_PATH)
    block_keys = {k.replace("blocks.0.", ""): v for k, v in sd.items()
                  if k.startswith("blocks.0.")}
    del sd

    # Load weights using the MLP remap
    MLP_REMAP = {
        ".mlp_x.fc1_x.": ".mlp_x.w1.",
        ".mlp_x.fc1_g.": ".mlp_x.w2.",
        ".mlp_x.fc2.": ".mlp_x.w3.",
        ".mlp_y.fc1_x.": ".mlp_y.w1.",
        ".mlp_y.fc1_g.": ".mlp_y.w2.",
        ".mlp_y.fc2.": ".mlp_y.w3.",
    }

    params = dict(our_block.named_parameters())
    buffers = dict(our_block.named_buffers())
    loaded = 0
    skipped = []
    for name, weight in block_keys.items():
        mapped = name
        for old, new in MLP_REMAP.items():
            if old in mapped:
                mapped = mapped.replace(old, new)
                break
        if mapped in params:
            from vllm.model_executor.model_loader.weight_utils import default_weight_loader
            loader = getattr(params[mapped], "weight_loader", default_weight_loader)
            loader(params[mapped], weight)
            loaded += 1
        elif mapped in buffers:
            buffers[mapped].copy_(weight)
            loaded += 1
        else:
            skipped.append(name)

    print(f"Block 0: loaded {loaded} weights, skipped {len(skipped)}")
    if skipped:
        print(f"  Skipped: {skipped}")

    our_block = our_block.to(device, dtype).eval()

    with torch.no_grad():
        x_out, y_out = our_block(
            x, y, t_x=t_x, t_y=t_y,
            temporal_pos=temporal_pos,
        )

    print(f"Our block output:")
    print(f"  x: mean={x_out.float().mean():.6f} std={x_out.float().std():.6f}")
    print(f"  y: mean={y_out.float().mean():.6f} std={y_out.float().std():.6f}")

    # Check if output differs significantly from input (model is doing something)
    x_change = (x_out - x).float().norm() / x.float().norm()
    y_change = (y_out - y).float().norm() / y.float().norm()
    print(f"  x relative change: {x_change:.4f}")
    print(f"  y relative change: {y_change:.4f}")

    del our_block
    torch.cuda.empty_cache()


if __name__ == "__main__":
    compare_y_embedder()
    compare_timestep_and_modulation()
    compare_single_block_forward()
