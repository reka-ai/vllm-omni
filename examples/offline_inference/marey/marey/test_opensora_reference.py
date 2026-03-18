"""
Run OpenSora FluxControlV2 reference model directly with the same checkpoint
to verify it produces good video output.

Usage:
    CUDA_VISIBLE_DEVICES=1 python test_opensora_reference.py
"""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MOONVALLEY_DIR = str(REPO_ROOT / "moonvalley_ai")
OPENSORA_DIR = str(REPO_ROOT / "moonvalley_ai" / "open_sora")

if MOONVALLEY_DIR not in sys.path:
    sys.path.insert(0, MOONVALLEY_DIR)
if OPENSORA_DIR not in sys.path:
    sys.path.insert(0, OPENSORA_DIR)

DEFAULT_CONFIG = "/app/wlam/models/checkpoints/marey/distilled-0001/config.yaml"
DEFAULT_WEIGHTS = (
    "/app/wlam/models/checkpoints/marey/distilled-0001/"
    "epoch0-global_step7000_distilled/ema_inference_ckpt.safetensors"
)
DEFAULT_VAE = "/app/wlam/models/checkpoints/marey/vae/epoch_4_step2819000.ckpt"

DEFAULT_NEGATIVE_PROMPT = (
    "Detailed description: <ungraded>, <timelapse>, <scene cut>, <timelapse> "
    "blurry, distortion, low quality, low resolution, jpeg artifacts, artifacts, "
    "logo, sign, watermark, overlay text, text, mark, "
)


def load_yaml_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def setup_opensora_modules():
    """Stub out heavy OpenSora modules not needed for inference."""
    import types
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


def encode_text(prompt, config, device, dtype):
    """Encode text using UL2, CLIP, ByT5 - matches our text_to_video.py."""
    from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor, T5EncoderModel

    te_cfg = config.get("text_encoder", {})

    # UL2
    ul2_name = te_cfg.get("ul2_pretrained", "google/ul2")
    ul2_max_len = te_cfg.get("ul2_max_length", 300)
    print(f"Loading UL2: {ul2_name}")
    ul2_tok = AutoTokenizer.from_pretrained(ul2_name)
    ul2_model = AutoModel.from_pretrained(ul2_name, torch_dtype=dtype).encoder.to(device).eval()
    inputs = ul2_tok(prompt, padding="max_length", max_length=ul2_max_len, truncation=True, return_tensors="pt")
    with torch.no_grad():
        ul2_seq = ul2_model(input_ids=inputs.input_ids.to(device)).last_hidden_state.to(dtype)
    del ul2_model
    torch.cuda.empty_cache()

    # CLIP
    clip_name = te_cfg.get("clip_pretrained", "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
    clip_max_len = te_cfg.get("clip_max_length", 77)
    print(f"Loading CLIP: {clip_name}")
    clip_model = CLIPModel.from_pretrained(clip_name, torch_dtype=dtype).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_name)
    clip_inputs = clip_processor(text=[prompt], padding="max_length", max_length=clip_max_len, truncation=True, return_tensors="pt")
    with torch.no_grad():
        clip_out = clip_model.get_text_features(**{k: v.to(device) for k, v in clip_inputs.items()})
        vector_cond = clip_out.to(dtype)
    del clip_model
    torch.cuda.empty_cache()

    # ByT5
    byt5_name = te_cfg.get("byt5_pretrained", "google/byt5-large")
    byt5_max_len = te_cfg.get("byt5_max_length", 70)
    print(f"Loading ByT5: {byt5_name}")
    byt5_tok = AutoTokenizer.from_pretrained(byt5_name)
    byt5_model = T5EncoderModel.from_pretrained(byt5_name, torch_dtype=dtype).to(device).eval()
    import re
    matches = re.findall(r'["\u201c\u201d](.*?)["\u201c\u201d]', prompt)
    quote_text = " ".join(matches) if matches else prompt
    byt5_inputs = byt5_tok(quote_text, padding="max_length", max_length=byt5_max_len, truncation=True, return_tensors="pt")
    with torch.no_grad():
        byt5_seq = byt5_model(input_ids=byt5_inputs.input_ids.to(device)).last_hidden_state.to(dtype)
    del byt5_model
    torch.cuda.empty_cache()

    return [ul2_seq, byt5_seq], vector_cond


def build_opensora_model(config, device, dtype):
    """Build and load the OpenSora FluxControlV2 model."""
    from opensora.models.stdit.flux_control import FluxControlV2, FluxControlV2Config

    model_cfg = config.get("model", {})

    flux_config = FluxControlV2Config(
        depth=model_cfg.get("depth", 42),
        hidden_size=model_cfg.get("hidden_size", 5120),
        num_heads=model_cfg.get("num_heads", 40),
        patch_size=model_cfg.get("patch_size", [1, 2, 2]),
        in_channels=16,
        out_channels=model_cfg.get("out_channels", 32),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        depth_single_blocks=model_cfg.get("depth_single_blocks", 28),
        qk_norm=model_cfg.get("qk_norm", True),
        rope_channels_ratio=model_cfg.get("rope_channels_ratio", 0.5),
        rope_dim=model_cfg.get("rope_dim", -1),
        add_pos_embed_at_every_block=model_cfg.get("add_pos_embed_at_every_block", True),
        input_sq_size=model_cfg.get("input_sq_size", 512),
        class_dropout_prob=0.0,
        learned_pe=model_cfg.get("learned_pe", True),
        mlp_type=model_cfg.get("mlp_type", "swiglu"),
        caption_channels=model_cfg.get("caption_channels", [4096, 1536]),
        model_max_length=model_cfg.get("model_max_length", [300, 70]),
        vector_cond_channels=model_cfg.get("vector_cond_channels", 768),
        extra_features_embedders=model_cfg.get("extra_features_embedders", None),
        use_block_v2=model_cfg.get("use_block_v2", True),
        flash_attn_version=None,  # Use SDPA fallback
        sequence_camera_condition=model_cfg.get("sequence_camera_condition", False),
        camera_dim=model_cfg.get("camera_dim", 0),
        img_fps=model_cfg.get("img_fps", 29.97003),
    )

    print(f"Building FluxControlV2 with depth={flux_config.depth}, hidden={flux_config.hidden_size}")
    model = FluxControlV2(flux_config)

    print(f"Loading weights from {DEFAULT_WEIGHTS}")
    sd = safetensors.torch.load_file(DEFAULT_WEIGHTS)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing {len(missing)} keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"Unexpected {len(unexpected)} keys (first 10): {unexpected[:10]}")

    model = model.to(device, dtype).eval()
    del sd
    torch.cuda.empty_cache()

    return model


def create_timesteps(num_steps, teacher_steps=100, tmin=0.001, tmax=1.0, device=None):
    sigmas = torch.linspace(tmax, tmin, teacher_steps)
    timesteps_all = sigmas * 1000
    stride = max(1, teacher_steps // num_steps)
    return [timesteps_all[i * stride].unsqueeze(0).to(device) for i in range(num_steps)]


@torch.inference_mode()
def generate_with_opensora(
    model, timesteps, seq_cond, vector_cond, neg_seq_cond, neg_vector_cond,
    num_frames, height, width, guidance_scale, device, dtype, generator,
    extra_features,
):
    """Run denoising loop using OpenSora model directly."""
    from opensora.models.stdit.flux_control import UnstructuredTextCond

    vae_ds = (4, 16, 16)
    if num_frames % vae_ds[0] != 1:
        num_frames = num_frames // vae_ds[0] * vae_ds[0] + 1
    num_latent_frames = (num_frames - 1) // vae_ds[0] + 1
    latent_h = math.ceil(height / vae_ds[1])
    latent_w = math.ceil(width / vae_ds[2])

    shape = (1, 16, num_latent_frames, latent_h, latent_w)
    z = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(device)
    print(f"Latent shape: {shape}, z std: {z.std().item():.4f}")

    height_t = torch.tensor([height], device=device, dtype=dtype)
    width_t = torch.tensor([width], device=device, dtype=dtype)

    do_cfg = guidance_scale > 1.0 and neg_seq_cond is not None

    def make_text_cond(sc, vc):
        return UnstructuredTextCond(seq_cond=sc, vector_cond=vc, seq_cond_mask=None)

    def model_forward(z_in, t_in, text_cond, extra_feats):
        raw = model(
            x=z_in.to(dtype),
            timestep=t_in.to(dtype),
            text_cond=text_cond,
            height=height_t,
            width=width_t,
            **extra_feats,
        )
        if raw.shape[1] != 16:
            raw = raw[:, :16]
        return raw

    cond_text = make_text_cond(seq_cond, vector_cond)
    uncond_text = make_text_cond(neg_seq_cond, neg_vector_cond) if do_cfg else None

    print(f"Starting denoising: {len(timesteps)} steps, CFG={'on' if do_cfg else 'off'}, scale={guidance_scale}")

    for i, t in enumerate(timesteps):
        t_input = t.expand(1)

        pred_cond = model_forward(z, t_input, cond_text, extra_features)

        if do_cfg:
            pred_uncond = model_forward(z, t_input, uncond_text, extra_features)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            v_pred = pred_cond

        sigma_t = (t / 1000).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x0 = z - sigma_t * v_pred

        if i < len(timesteps) - 1:
            sigma_s = (timesteps[i + 1] / 1000).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = 1.0 - sigma_t
            alpha_s = 1.0 - sigma_s
            alpha_ts = alpha_t / alpha_s
            alpha_ts_sq = alpha_ts ** 2
            sigma_s_div_t_sq = (sigma_s / sigma_t) ** 2
            sigma_ts_div_t_sq = 1.0 - alpha_ts_sq * sigma_s_div_t_sq

            mean = alpha_ts * sigma_s_div_t_sq * z + alpha_s * sigma_ts_div_t_sq * x0
            variance = sigma_ts_div_t_sq * sigma_s ** 2
            noise = torch.randn_like(z)
            z = mean + torch.sqrt(variance) * noise
        else:
            z = x0

        if (i + 1) % 10 == 0 or i == 0 or i == len(timesteps) - 1:
            print(f"  Step {i+1}/{len(timesteps)}: sigma={sigma_t.item():.4f} "
                  f"v_pred std={v_pred.float().std().item():.4f} "
                  f"x0 std={x0.float().std().item():.4f} "
                  f"z std={z.float().std().item():.4f}")

    return z


def load_vae(vae_config, device, dtype):
    from opensora.models.vae.vae_adapters import PretrainedSpatioTemporalVAETokenizer
    vae = PretrainedSpatioTemporalVAETokenizer(
        cp_path=vae_config.get("cp_path", DEFAULT_VAE),
        strict_loading=False,
        extra_kwargs={"no_losses": True},
        scaling_factor=vae_config.get("scaling_factor", 1.0),
        bias_factor=vae_config.get("bias_factor", 0.0),
        frame_chunk_len=vae_config.get("frame_chunk_len"),
        max_batch_size=vae_config.get("max_batch_size"),
        reuse_as_spatial_vae=vae_config.get("reuse_as_spatial_vae", False),
        extra_context_and_drop_strategy=vae_config.get("extra_context_and_drop_strategy", False),
    )
    return vae.to(device, dtype).eval()


def save_video(frames, output_path, fps=24):
    from diffusers.utils import export_to_video
    if frames.ndim == 5:
        frames = frames[0]
    if frames.ndim == 4 and frames.shape[-1] not in (1, 3, 4):
        frames = np.transpose(frames, (0, 2, 3, 1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(list(frames), output_path, fps=fps)
    print(f"Saved video to {output_path}")


def main():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    seed = 42
    prompt = "A serene lakeside sunrise with mist over the water."
    height, width = 720, 1280
    num_frames = 33
    num_steps = 100
    guidance_scale = 7.5

    config = load_yaml_config(DEFAULT_CONFIG)
    setup_opensora_modules()

    # Encode text
    print("\n=== Encoding text ===")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    seq_cond, vector_cond = encode_text(prompt, config, device, dtype)
    neg_seq_cond, neg_vector_cond = encode_text(DEFAULT_NEGATIVE_PROMPT, config, device, dtype)
    print(f"seq_cond shapes: {[s.shape for s in seq_cond]}, vec: {vector_cond.shape}")

    # Build model
    print("\n=== Building OpenSora FluxControlV2 ===")
    t0 = time.perf_counter()
    model = build_opensora_model(config, device, dtype)
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s")
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Parameters: {num_params:.1f}B")

    # Build extra features
    model_cfg = config.get("model", {})
    extra_features = {"fps": torch.tensor([24.0], device=device, dtype=dtype)}

    # Add extra conditioning features that the model expects
    extra_cfg = model_cfg.get("extra_features_embedders", {})
    for feat, params in extra_cfg.items():
        if feat == "fps":
            continue
        ftype = params.get("type", "")
        if ftype == "SizeEmbedder":
            if feat == "ar":
                extra_features[feat] = torch.tensor([float(height) / float(width)], device=device, dtype=dtype)
            else:
                extra_features[feat] = torch.tensor([1.0], device=device, dtype=dtype)
        elif ftype == "LabelEmbedder":
            eval_val = params.get("eval_value", 0)
            extra_features[feat] = torch.tensor([eval_val], device=device).long()
        elif ftype == "OrderedEmbedder":
            eval_val = params.get("eval_value", -1)
            extra_features[feat] = torch.tensor([eval_val], device=device).long()

    print(f"Extra features: {list(extra_features.keys())}")

    # Timesteps
    sched_cfg = config.get("scheduler", {})
    tmin = sched_cfg.get("tmin", 0.001)
    tmax = sched_cfg.get("tmax", 1.0)
    teacher_steps = sched_cfg.get("num_sampling_steps", 100)
    timesteps = create_timesteps(num_steps, teacher_steps, tmin, tmax, device)
    print(f"Timesteps: {len(timesteps)} steps, first={timesteps[0].item():.1f}, last={timesteps[-1].item():.1f}")

    # Generate
    print("\n=== Generating video with OpenSora reference ===")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = generate_with_opensora(
        model, timesteps, seq_cond, vector_cond, neg_seq_cond, neg_vector_cond,
        num_frames, height, width, guidance_scale, device, dtype, generator,
        extra_features,
    )

    lf = latents.float()
    print(f"\nFinal latents: shape={list(latents.shape)} "
          f"mean={lf.mean().item():.4f} std={lf.std().item():.4f}")

    # Decode and save
    del model
    torch.cuda.empty_cache()

    vae_cfg = config.get("vae", {})
    print("\n=== Loading VAE ===")
    vae = load_vae(vae_cfg, device, dtype)

    vae_t_ds = vae.downsample_factors[0]
    num_latent_t = latents.shape[2]
    num_pixel_frames = num_latent_t * vae_t_ds
    chunk = vae.frame_chunk_len
    if chunk is not None and num_pixel_frames % chunk != 0:
        num_pixel_frames = (num_pixel_frames // chunk) * chunk

    with torch.no_grad():
        video = vae.decode(latents.to(dtype), num_frames=num_pixel_frames, spatial_size=(height, width))
    if isinstance(video, tuple):
        video = video[0]

    vf = video.float().cpu()
    print(f"VAE output: shape={list(vf.shape)} mean={vf.mean().item():.4f} std={vf.std().item():.4f}")

    if vf.dim() == 5 and vf.shape[1] in (3, 4):
        vf = vf[0].permute(1, 2, 3, 0)
    vf = vf.clamp(-1, 1) * 0.5 + 0.5
    save_video(vf.numpy(), "opensora_reference_output.mp4", fps=24)


if __name__ == "__main__":
    main()
