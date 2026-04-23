# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Marey DiT denoising stage.

Stage 1 of the staged Marey pipeline. Consumes pre-computed encoder
embeddings (positive + optional negative) from the upstream text-encoder
stage via ``req.prompts[0]["additional_information"]`` and produces final
denoised latents. Classifier-free guidance is handled in-loop (cond +
uncond forward passes per step).
"""

from __future__ import annotations

import glob
import logging
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path

import safetensors.torch
import torch
import yaml
from diffusers.utils.torch_utils import randn_tensor
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.marey._dumper import create_writer
from vllm_omni.diffusion.models.marey.marey_transformer import MareyTransformer
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)


# Checkpoint key remapping (training → inference naming)
_CHECKPOINT_KEY_REMAP = [
    ("y_embedder.vector_embedding_0", "y_embedder.vector_embedding"),
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_yaml_config(model_path: str) -> dict:
    config_path = os.path.join(model_path, "config.yaml")
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_config(od_config: OmniDiffusionConfig) -> dict:
    """Load config.yaml from od_config.model and apply model_config overrides."""
    config = _load_yaml_config(od_config.model)
    if od_config.model_config:
        _deep_update(config, od_config.model_config)

    vae_cfg = config.get("vae")
    if isinstance(vae_cfg, dict):
        cp_path = vae_cfg.get("cp_path", "")
        if cp_path and not os.path.isabs(cp_path):
            vae_cfg["cp_path"] = os.path.join(od_config.model, cp_path)

    return config


def _resolve_encoder_dims(te_cfg: dict) -> tuple[list[int], int]:
    """Return ``(caption_channels, vector_cond_channels)`` from HF configs.

    ``caption_channels = [ul2_d_model, byt5_d_model]`` and
    ``vector_cond_channels = clip_hidden_size``. The config.yaml may override
    the pretrained names; hidden sizes are pulled from ``AutoConfig`` so
    encoder weights are not materialized.
    """
    from transformers import AutoConfig

    ul2_name = te_cfg.get("ul2_pretrained", "google/ul2")
    byt5_name = te_cfg.get("byt5_pretrained", "google/byt5-large")
    clip_name = te_cfg.get("clip_pretrained", "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")

    ul2_dim = int(AutoConfig.from_pretrained(ul2_name).d_model)
    byt5_dim = int(AutoConfig.from_pretrained(byt5_name).d_model)
    clip_cfg = AutoConfig.from_pretrained(clip_name)
    clip_text_cfg = getattr(clip_cfg, "text_config", clip_cfg)
    clip_dim = int(clip_text_cfg.hidden_size)

    return [ul2_dim, byt5_dim], clip_dim


# ---------------------------------------------------------------------------
# Transformer construction
# ---------------------------------------------------------------------------


def _create_transformer_from_config(
    model_cfg: dict,
    te_cfg: dict,
    in_channels: int,
    caption_channels: list[int],
    vector_cond_channels: int | None,
) -> MareyTransformer:
    depth = model_cfg.get("depth", 42)
    num_heads = model_cfg.get("num_heads", 40)
    head_dim = model_cfg.get("head_dim", 128)
    hidden_size = model_cfg.get("hidden_size", num_heads * head_dim)
    patch_size = tuple(model_cfg.get("patch_size", [1, 2, 2]))
    depth_single_blocks = model_cfg.get("depth_single_blocks", 28)
    mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
    rope_dim = model_cfg.get("rope_dim", -1)
    rope_channels_ratio = model_cfg.get("rope_channels_ratio", 0.5)
    qk_norm = model_cfg.get("qk_norm", True)
    add_pos_embed_at_every_block = model_cfg.get("add_pos_embed_at_every_block", True)
    learned_pe = model_cfg.get("learned_pe", True)

    ul2_max_length = te_cfg.get("ul2_max_length", 300)
    byt5_max_length = te_cfg.get("byt5_max_length", 0)
    model_max_length = [ul2_max_length, byt5_max_length] if byt5_max_length > 0 else ul2_max_length

    out_channels = model_cfg.get("out_channels", 2 * in_channels)
    extra_features_config = model_cfg.get("extra_features_embedders", None)
    camera_dim = model_cfg.get("camera_dim") if model_cfg.get("sequence_camera_condition", False) else None

    return MareyTransformer(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=hidden_size,
        depth=depth,
        depth_single_blocks=depth_single_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        patch_size=patch_size,
        caption_channels=caption_channels,
        model_max_length=model_max_length,
        vector_cond_channels=vector_cond_channels,
        qk_norm=qk_norm,
        rope_channels_ratio=rope_channels_ratio,
        rope_dim=rope_dim,
        add_pos_embed_at_every_block=add_pos_embed_at_every_block,
        class_dropout_prob=0.0,
        learned_pe=learned_pe,
        extra_features_config=extra_features_config,
        camera_dim=camera_dim,
    )


def _build_extra_features(
    model_cfg: dict,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    extra_cfg = model_cfg.get("extra_features_embedders", {})
    features: dict[str, torch.Tensor] = {}
    for feat, params in extra_cfg.items():
        if feat == "fps":
            continue
        ftype = params.get("type", "")
        if ftype == "SizeEmbedder":
            val = float(height) / float(width) if feat == "ar" else 1.0
            features[feat] = torch.tensor([val], device=device, dtype=dtype)
        elif ftype in ("LabelEmbedder", "OrderedEmbedder"):
            eval_val = params.get("eval_value", -1)
            features[feat] = torch.tensor([eval_val], device=device, dtype=torch.long)
    return features


# ---------------------------------------------------------------------------
# Timestep / guidance schedule
# ---------------------------------------------------------------------------


def _create_flow_timesteps(
    num_steps: int,
    shift: float | None = None,
    num_train_timesteps: int = 1000,
    tmin: float = 0.001,
    tmax: float = 1.0,
    teacher_steps: int = 100,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Rectified-flow timesteps for distilled inference (matches reference RFLOW)."""
    nt = num_train_timesteps
    sigmas_f64 = torch.linspace(tmax, tmin, teacher_steps).tolist()
    timesteps_all = [torch.tensor([t * nt], device=device) for t in sigmas_f64]

    if shift is not None and shift > 0:
        for i, t in enumerate(timesteps_all):
            t_norm = t / nt
            timesteps_all[i] = (shift * t_norm / (1.0 + (shift - 1.0) * t_norm)) * nt

    stride = max(1, teacher_steps // num_steps)
    return [timesteps_all[i * stride] for i in range(num_steps)]


def _build_guidance_schedule(
    num_steps: int,
    guidance_scale: float,
    warmup_steps: int = 4,
    cooldown_steps: int = 18,
    guidance_every_n_steps: int = 2,
) -> list[float]:
    """Oscillating guidance schedule matching the reference RFLOW scheduler."""
    cooldown_start = num_steps - cooldown_steps

    schedule = []
    for i in range(num_steps):
        if i < warmup_steps:
            schedule.append(guidance_scale)
        elif i >= cooldown_start:
            schedule.append(1.0)
        else:
            middle_idx = i - warmup_steps
            if middle_idx > 0 and (middle_idx - 1) % guidance_every_n_steps == 0:
                schedule.append(guidance_scale)
            else:
                schedule.append(1.0)
    return schedule


# ---------------------------------------------------------------------------
# Post-process (DiT stage — passthrough; VAE stage handles frame post-proc)
# ---------------------------------------------------------------------------


def get_marey_post_process_func(od_config: OmniDiffusionConfig):
    """DiT-stage post-process is a passthrough.

    Stage 1's output is the final denoised latent tensor, which is consumed
    by stage 2 (VAE). Applying the video processor here would treat the
    latent as pixel frames and corrupt it.
    """

    def post_process_func(latent: torch.Tensor, output_type: str = "latent"):
        return latent

    return post_process_func


# ---------------------------------------------------------------------------
# DiT stage pipeline
# ---------------------------------------------------------------------------


class MareyDitPipeline(nn.Module, ProgressBarMixin):
    """Stage-1 Marey denoising pipeline (DiT only).

    Expects upstream text-encoder stage outputs in
    ``req.prompts[0]["additional_information"]``:
        - ``prompt_embeds``:     list[Tensor]  (UL2, ByT5 seq embeds)
        - ``prompt_masks``:      list[Tensor]  (UL2, ByT5 attention masks)
        - ``vector_cond``:       Tensor        (CLIP pooled)
        - ``neg_prompt_embeds``: list[Tensor] | None
        - ``neg_prompt_masks``:  list[Tensor] | None
        - ``neg_vector_cond``:   Tensor | None
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        self.config = _load_config(od_config)
        te_cfg = self.config.get("text_encoder", {})
        vae_cfg = self.config.get("vae", {})
        model_cfg = self.config.get("model", {})
        sched_cfg = self.config.get("scheduler", {})

        # VAE downsample factors are baked into latent shape math; we do NOT
        # load a VAE in this stage.
        vae_ds = tuple(vae_cfg.get("downsample_factors", (4, 16, 16)))
        self.vae_scale_factor_temporal = vae_ds[0]
        self.vae_scale_factor_spatial = vae_ds[1]

        in_channels = len(vae_cfg.get("scaling_factor", [0] * 16))
        # Caption channels match the actual encoder hidden sizes. Stage 1
        # does not load the encoders; pull dimensions from their HF configs
        # (weights are not materialized). The text_encoder section of the
        # Marey config.yaml may override the pretrained names but does not
        # carry hidden sizes, so we look them up here.
        caption_channels, vector_cond_channels = _resolve_encoder_dims(te_cfg)

        # Cached encoder dims — used to fabricate zero embeddings during the
        # diffusion engine's dummy/warmup run (which arrives with an empty
        # `additional_information`).
        self._caption_channels = caption_channels
        self._vector_cond_channels = vector_cond_channels
        self._ul2_max_length = int(te_cfg.get("ul2_max_length", 300))
        self._byt5_max_length = int(te_cfg.get("byt5_max_length", 70))

        self.transformer = _create_transformer_from_config(
            model_cfg, te_cfg, in_channels, caption_channels, vector_cond_channels,
        )

        # Scheduler config
        self.num_train_timesteps = 1000
        self.sched_tmin = sched_cfg.get("tmin", 0.001)
        self.sched_tmax = sched_cfg.get("tmax", 1.0)
        self.sched_teacher_steps = sched_cfg.get("num_sampling_steps", 100)
        self.flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 3.0

        # Guidance defaults
        self.skip_uncond = True  # distilled models skip uncond when scale=1.0
        self.default_clip_value = 10.0
        self.default_warmup_steps = 4
        self.default_cooldown_steps = 18
        self.default_guidance_every_n_steps = 2

        self._guidance_scale: float | None = None
        self._num_timesteps: int | None = None
        self._current_timestep: torch.Tensor | None = None

    # -- Properties ----------------------------------------------------------

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    # -- Dummy encoder tensors (warmup/profile) ------------------------------

    def _dummy_encoder_tensors(self) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        device = self.device
        dtype = self.transformer.dtype
        ul2_dim, byt5_dim = self._caption_channels
        lengths = (self._ul2_max_length, self._byt5_max_length)
        dims = (ul2_dim, byt5_dim)
        embeds = [torch.zeros((1, L, D), device=device, dtype=dtype) for L, D in zip(lengths, dims)]
        masks = [torch.ones((1, L), device=device, dtype=torch.bool) for L in lengths]
        vec = torch.zeros((1, self._vector_cond_channels), device=device, dtype=dtype)
        return embeds, masks, vec

    # -- Latent preparation --------------------------------------------------

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        time_pad = 0 if (num_frames % self.vae_scale_factor_temporal == 0) else (
            self.vae_scale_factor_temporal - num_frames % self.vae_scale_factor_temporal
        )
        num_latent_frames = (num_frames + time_pad) // self.vae_scale_factor_temporal
        latent_h = math.ceil(height / self.vae_scale_factor_spatial)
        latent_w = math.ceil(width / self.vae_scale_factor_spatial)
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_h, latent_w)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # -- Forward (DDPM flow-matching loop) -----------------------------------

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        logger.info("[marey-timing] stage=dit forward start request_id=%s", req.request_id)
        _t_start = time.perf_counter()
        try:
            return self._forward_impl(req)
        finally:
            _elapsed = time.perf_counter() - _t_start
            logger.info("[marey-timing] stage=dit forward end request_id=%s elapsed=%.3fs", req.request_id, _elapsed)

    def _forward_impl(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("MareyDitPipeline only supports a single prompt per request.")

        raw_prompt = req.prompts[0]
        if isinstance(raw_prompt, str):
            raise ValueError(
                "MareyDitPipeline expects a dict prompt carrying pre-computed "
                "encoder embeddings in additional_information; got a raw string."
            )
        add_info = raw_prompt.get("additional_information") or {}

        prompt_embeds = add_info.get("prompt_embeds")
        prompt_masks = add_info.get("prompt_masks")
        vector_cond = add_info.get("vector_cond")
        # The diffusion engine's warmup issues a dummy OmniTextPrompt with no
        # `additional_information`. Fabricate zero-filled encoder tensors of
        # the expected shape so the warmup executes the full transformer path.
        if prompt_embeds is None or vector_cond is None:
            prompt_embeds, prompt_masks, vector_cond = self._dummy_encoder_tensors()
        negative_prompt_embeds = add_info.get("neg_prompt_embeds")
        negative_prompt_masks = add_info.get("neg_prompt_masks")
        negative_vector_cond = add_info.get("neg_vector_cond")

        sp = req.sampling_params
        height = sp.height or 720
        width = sp.width or 1280
        num_frames = sp.num_frames if sp.num_frames else 33
        num_steps = sp.num_inference_steps or 100
        guidance_scale = sp.guidance_scale if sp.guidance_scale_provided else 7.5
        self._guidance_scale = guidance_scale

        device = self.device
        dtype = self.transformer.dtype
        logger.info(f"MareyTransformer dtype: {dtype}")

        generator = sp.generator
        if generator is None and sp.seed is not None:
            generator = torch.Generator(device=device).manual_seed(sp.seed)
        elif generator is None:
            generator = torch.Generator(device=device).manual_seed(0)

        # Move embeddings to the right device/dtype (they may have arrived
        # from another process on CPU).
        def _to_device_list(xs):
            if xs is None:
                return None
            return [x.to(device=device, dtype=dtype) if x.dtype.is_floating_point else x.to(device=device) for x in xs]

        prompt_embeds = _to_device_list(prompt_embeds)
        prompt_masks = _to_device_list(prompt_masks)
        vector_cond = vector_cond.to(device=device, dtype=dtype)
        negative_prompt_embeds = _to_device_list(negative_prompt_embeds)
        negative_prompt_masks = _to_device_list(negative_prompt_masks)
        if negative_vector_cond is not None:
            negative_vector_cond = negative_vector_cond.to(device=device, dtype=dtype)

        use_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

        dumper = create_writer(req.request_id)

        # Timesteps
        timesteps = _create_flow_timesteps(
            num_steps=num_steps,
            shift=self.flow_shift if self.flow_shift > 0 else None,
            num_train_timesteps=self.num_train_timesteps,
            tmin=self.sched_tmin,
            tmax=self.sched_tmax,
            teacher_steps=self.sched_teacher_steps,
            device=device,
        )
        self._num_timesteps = len(timesteps)

        guidance_schedule: list[float] | None = None
        if use_cfg:
            guidance_schedule = _build_guidance_schedule(
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                warmup_steps=self.default_warmup_steps,
                cooldown_steps=self.default_cooldown_steps,
                guidance_every_n_steps=self.default_guidance_every_n_steps,
            )

        clip_value = self.default_clip_value

        # Latents
        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=sp.latents,
        )

        if dumper.enabled:
            dumper.write_inputs(
                prompt_embeds=prompt_embeds,
                prompt_masks=prompt_masks,
                vector_cond=vector_cond,
                neg_prompt_embeds=negative_prompt_embeds,
                neg_prompt_masks=negative_prompt_masks,
                neg_vector_cond=negative_vector_cond,
                meta={
                    "request_id": req.request_id,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "num_steps": num_steps,
                    "guidance_scale": guidance_scale,
                    "use_cfg": use_cfg,
                    "fps": sp.fps,
                    "seed": sp.seed,
                    "flow_shift": self.flow_shift,
                    "num_train_timesteps": self.num_train_timesteps,
                    "dtype": str(dtype),
                },
            )
            dumper.write_initial(
                initial_noise=latents,
                timesteps=timesteps,
                guidance_schedule=guidance_schedule,
            )

        # Extra features + conditioning tensors
        model_cfg = self.config.get("model", {})
        extra_features = _build_extra_features(model_cfg, height, width, device, dtype)
        uncond_extra_features = extra_features  # quality guidance disabled for now

        height_t = torch.tensor([height], device=device, dtype=dtype)
        width_t = torch.tensor([width], device=device, dtype=dtype)
        fps_value = sp.fps if sp.fps else 24
        fps_t = torch.tensor([float(fps_value)], device=device, dtype=dtype)

        in_channels = self.transformer.in_channels

        def _model_forward(z_in, t_in, text_emb, vec_cond, text_mask=None, ef=None):
            raw = self.transformer(
                hidden_states=z_in.to(dtype),
                timestep=t_in,
                encoder_hidden_states=text_emb,
                encoder_hidden_states_mask=text_mask,
                vector_cond=vec_cond,
                height=height_t,
                width=width_t,
                fps=fps_t,
                extra_features=ef if ef is not None else extra_features,
                return_dict=False,
            )[0]
            if raw.shape[1] != in_channels:
                raw = raw[:, :in_channels]
            return raw

        # DDPM flow-matching denoising loop
        z = latents
        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                self._current_timestep = t
                t_input = t.expand(1)

                gs_i = guidance_schedule[i] if guidance_schedule is not None else guidance_scale
                use_uncond = use_cfg and ((not self.skip_uncond) or (gs_i > 1.0))

                if dumper.enabled:
                    dumper.begin_step(i)
                    dumper.write_step_tensor("z_in", z)
                    dumper.write_step_tensor("timestep", t_input)
                    dumper.write_step_tensor("gs", float(gs_i))

                pred_cond = _model_forward(
                    z, t_input, prompt_embeds, vector_cond, prompt_masks,
                    ef=extra_features,
                )
                if dumper.enabled:
                    dumper.write_step_tensor("pred_cond", pred_cond)

                if use_uncond:
                    pred_uncond = _model_forward(
                        z, t_input, negative_prompt_embeds, negative_vector_cond,
                        negative_prompt_masks,
                        ef=uncond_extra_features,
                    )
                    if dumper.enabled:
                        dumper.write_step_tensor("pred_uncond", pred_uncond)
                    v_pred = pred_uncond + gs_i * (pred_cond - pred_uncond)
                else:
                    v_pred = pred_cond

                if dumper.enabled:
                    dumper.write_step_tensor("v_pred", v_pred)

                sigma_t = (t / self.num_train_timesteps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x0 = z - sigma_t * v_pred
                if clip_value is not None and clip_value > 0:
                    x0 = torch.clamp(x0, -clip_value, clip_value)

                if dumper.enabled:
                    dumper.write_step_tensor("x0", x0)

                if i < len(timesteps) - 1:
                    sigma_s = (timesteps[i + 1] / self.num_train_timesteps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    alpha_t = 1.0 - sigma_t
                    alpha_s = 1.0 - sigma_s
                    alpha_ts = alpha_t / alpha_s
                    alpha_ts_sq = alpha_ts ** 2
                    sigma_s_div_t_sq = (sigma_s / sigma_t) ** 2
                    sigma_ts_div_t_sq = 1.0 - alpha_ts_sq * sigma_s_div_t_sq

                    mean = alpha_ts * sigma_s_div_t_sq * z + alpha_s * sigma_ts_div_t_sq * x0
                    variance = sigma_ts_div_t_sq * sigma_s ** 2

                    noise = torch.randn_like(z, generator=generator)
                    if dumper.enabled:
                        dumper.write_step_tensor("noise", noise)
                    z = mean + torch.sqrt(variance) * noise
                else:
                    z = x0

                pbar.update()

        self._current_timestep = None

        if dumper.enabled:
            dumper.write_final(z)

        # Stage output: final latents. The diffusion engine stores
        # ``DiffusionOutput.output`` on the downstream ``OmniRequestOutput``
        # as ``.images``, but ``.latents`` stays unset. Stash the latent
        # in ``custom_output`` so ``stage_input_processors.marey.diffusion2vae``
        # reads it from ``source_output._custom_output`` with the other
        # size metadata.
        z = z.to("cpu")
        torch.cuda.empty_cache()
        return DiffusionOutput(
            output=z,
            custom_output={
                "latents": z,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "fps": fps_value,
            },
        )

    # -- Weight loading ------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load transformer weights from the ema_inference_ckpt.safetensors file.

        The ``weights`` iterator is ignored — we locate the real checkpoint
        under ``od_config.model`` directly, matching the reference offline
        inference loader.
        """
        model_dir = self.od_config.model
        patterns = [
            os.path.join(model_dir, "epoch0-*", "ema_inference_ckpt.safetensors"),
            os.path.join(model_dir, "**", "ema_inference_ckpt.safetensors"),
        ]
        ckpt_path = None
        for pat in patterns:
            matches = sorted(glob.glob(pat, recursive=True))
            if matches:
                ckpt_path = matches[0]
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Cannot find ema_inference_ckpt.safetensors under {model_dir}"
            )

        logger.info("Loading transformer weights from %s", ckpt_path)
        state_dict = safetensors.torch.load_file(ckpt_path)

        def _remap_items():
            for name, tensor in state_dict.items():
                r = name
                for old, new in _CHECKPOINT_KEY_REMAP:
                    if old in r:
                        r = r.replace(old, new)
                        break
                yield r, tensor

        self.transformer.load_weights(_remap_items())

        del state_dict
        torch.cuda.empty_cache()

        return {name for name, _ in self.named_parameters()}
