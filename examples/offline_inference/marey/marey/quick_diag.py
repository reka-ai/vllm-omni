"""Minimal single-GPU diagnostic: load vllm_omni model, run one forward pass, print stats."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import yaml
import safetensors.torch

CONFIG = "/app/wlam/models/checkpoints/marey/distilled-0001/config.yaml"
WEIGHTS = (
    "/app/wlam/models/checkpoints/marey/distilled-0001/"
    "epoch0-global_step7000_distilled/ema_inference_ckpt.safetensors"
)


def main():
    from text_to_video import init_distributed
    init_distributed(tp_size=1)

    with open(CONFIG) as f:
        config = yaml.safe_load(f)

    from text_to_video import build_transformer, load_transformer_weights
    vae_cfg = config.get("vae", {})
    in_channels = len(vae_cfg.get("scaling_factor", [0]*16))

    device = torch.device("cuda:0")
    transformer = build_transformer(config, in_channels, caption_channels=4096, vector_cond_channels=768)
    load_transformer_weights(transformer, WEIGHTS, device=device, dtype=torch.bfloat16)
    transformer = transformer.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in transformer.parameters()):,}")

    torch.manual_seed(42)
    B, C, T, H, W = 1, in_channels, 1, 64, 64
    z = torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
    seq_cond = torch.randn(B, 10, 4096, device=device, dtype=torch.bfloat16)
    vec_cond = torch.randn(B, 768, device=device, dtype=torch.bfloat16)
    fps = torch.tensor([24], device=device, dtype=torch.bfloat16)
    height_t = torch.tensor([H * 16], device=device)
    width_t = torch.tensor([W * 16], device=device)

    print(f"\nInput: z std={z.float().std().item():.4f}, timestep={timestep.item():.1f}")

    # Hook to capture intermediate values
    hook_data = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            hook_data[name] = out.float().detach()
        return hook_fn

    # Register hooks on key layers
    transformer.x_embedder.register_forward_hook(make_hook("x_embedder"))
    transformer.t_embedder.register_forward_hook(make_hook("t_embedder"))
    transformer.y_embedder.register_forward_hook(make_hook("y_embedder"))
    transformer.t_block.register_forward_hook(make_hook("t_block"))
    transformer.blocks[0].register_forward_hook(make_hook("block_0"))
    transformer.blocks[0].attn.register_forward_hook(make_hook("block_0.attn"))
    transformer.blocks[0].mlp_x.register_forward_hook(make_hook("block_0.mlp_x"))
    transformer.blocks[1].register_forward_hook(make_hook("block_1"))
    mid = len(transformer.blocks) // 2
    transformer.blocks[mid].register_forward_hook(make_hook(f"block_{mid}"))
    transformer.blocks[-1].register_forward_hook(make_hook("block_last"))
    transformer.final_layer.register_forward_hook(make_hook("final_layer"))

    with torch.no_grad():
        out = transformer(
            hidden_states=z,
            timestep=timestep,
            encoder_hidden_states=seq_cond,
            vector_cond=vec_cond,
            height=height_t,
            width=width_t,
            fps=fps,
            return_dict=False,
        )[0]

    print(f"\nOutput shape: {list(out.shape)}")
    out_f = out.float()
    print(f"Output: mean={out_f.mean().item():.4f} std={out_f.std().item():.4f} "
          f"abs_max={out_f.abs().max().item():.4f}")
    if out.shape[1] > C:
        lo = out_f[:, :C]
        hi = out_f[:, C:]
        print(f"  lo16 (velocity): std={lo.std().item():.4f}")
        print(f"  hi16 (unused):   std={hi.std().item():.4f}")

    print("\n--- Hook data ---")
    for name, tensor in hook_data.items():
        if tensor.dim() >= 2:
            print(f"  {name}: shape={list(tensor.shape)} std={tensor.std().item():.4f} "
                  f"abs_max={tensor.abs().max().item():.4f}")
        else:
            print(f"  {name}: shape={list(tensor.shape)} std={tensor.std().item():.4f}")


if __name__ == "__main__":
    main()
