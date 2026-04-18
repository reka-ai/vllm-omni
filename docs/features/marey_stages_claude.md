# Marey Staged Pipeline — Implementation & Debugging Guide

Status: implemented, not yet smoke-tested. This document is a self-contained
handoff for the first debugging session.

---

## 1. Motivation

The legacy Marey pipeline (`vllm_omni/diffusion/models/marey/pipeline_marey.py`,
pre-refactor) was a single monolithic `nn.Module` that ran text encoding
(UL2 + CLIP + ByT5), the rectified-flow denoising loop, and opensora VAE
decoding inside one `forward()` call, shuffling models between CPU and GPU
between phases. It also imported a `marey_serving_runtime` module that does
not exist in the tree, so it isn't currently loadable via `DiffusionEngine`.

The new implementation splits Marey into three vllm-omni stages running in
separate processes, each owning its own GPU memory, and removes I2V
preprocessing entirely (text-to-video only).

---

## 2. Topology

| Stage | `stage_type` | Model class                                           | Devices        | Role                                       |
|-------|--------------|-------------------------------------------------------|----------------|--------------------------------------------|
| 0     | `llm`        | `MareyTextEncoder` (vLLM model, non-AR)               | GPU 0          | UL2 + CLIP + ByT5 forward for pos + neg    |
| 1     | `diffusion`  | `MareyDitPipeline`                                    | GPUs 1-4 (SP)  | DDPM flow-matching loop w/ in-loop CFG     |
| 2     | `diffusion`  | `MareyVaePipeline`                                    | GPU 5          | opensora VAE decode → video frames         |

Sequence parallelism on the DiT uses `parallel_config.ulysses_degree: 4`
(matches the current single-stage `--ulysses-degree` usage).

### Data flow

```
user request
   │
   ▼
stage 0: MareyTextEncoder (vLLM LLM stage, OmniGenerationScheduler)
   │  OmniOutput(multimodal_outputs={"marey_text_encoder_out": [dict, …]})
   │  dict keys: prompt_embeds, prompt_masks, vector_cond,
   │             neg_prompt_embeds, neg_prompt_masks, neg_vector_cond
   ▼
stage 1: MareyDitPipeline (DiffusionEngine)
   │  reads embeddings from req.prompts[0]["additional_information"]
   │  runs denoising loop with per-step cond/uncond forwards
   │  DiffusionOutput(output=z, custom_output={height, width, num_frames, fps})
   ▼
stage 2: MareyVaePipeline (DiffusionEngine)
   │  reads latents from req.prompts[0]["additional_information"]["latents"]
   │  runs vae.decode
   │  DiffusionOutput(output=video)
   ▼
user receives video (post-processed by get_marey_vae_post_process_func)
```

**Stage-to-stage transfer is in-memory through the orchestrator.** No
connectors are used for baseline Marey. Connectors would only be needed for:
KV-cache transfer (future autoregressive text refinement), CFG companions
(Marey doesn't need these — CFG is in-loop in stage 1), or async_chunk
streaming (not applicable).

---

## 3. File map

### New files

| Path | Purpose |
|------|---------|
| `vllm_omni/diffusion/models/marey/text_encoders.py` | `TextEncoderConfig`, `MareyTextEncoderBundle`, `DEFAULT_NEGATIVE_PROMPT`, `extract_quotes()`. Pure PyTorch + HF — no vLLM deps. Callable from either the stage-0 vLLM adapter or standalone tests. |
| `vllm_omni/diffusion/models/marey/vae_loader.py` | `load_vae()`, `setup_opensora_imports()`, `resolve_moonvalley_dir()`, `opensora_logging_guard()`. Extracted from the old monolithic pipeline. |
| `vllm_omni/diffusion/models/marey/pipeline_marey_vae.py` | `MareyVaePipeline` + `get_marey_vae_post_process_func`. Stage 2. |
| `vllm_omni/model_executor/models/marey/__init__.py` | Exports `MareyTextEncoder`. |
| `vllm_omni/model_executor/models/marey/marey_text_encoder.py` | `MareyTextEncoder` vLLM model. Stage 0. |
| `vllm_omni/model_executor/stage_input_processors/marey.py` | `text2diffusion` (stage 0→1), `diffusion2vae` (stage 1→2). |
| `vllm_omni/model_executor/stage_configs/marey.yaml` | 3-stage config with GPU 0 / 1,2,3,4 / 5 allocation. |

### Modified files

| Path | Change |
|------|--------|
| `vllm_omni/diffusion/models/marey/pipeline_marey.py` | **Rewritten** as `MareyDitPipeline` (stage 1 only). Drops text encoders, VAE loading, I2V path, and the non-existent `marey_serving_runtime` import. |
| `vllm_omni/diffusion/models/marey/__init__.py` | Exports the two new pipeline classes + post-process funcs; removed `MareyPipeline` and `get_marey_pre_process_func`. |
| `vllm_omni/diffusion/registry.py` | Added `MareyDitPipeline` and `MareyVaePipeline` to `_DIFFUSION_MODELS`; added both to `_DIFFUSION_POST_PROCESS_FUNCS`. |
| `vllm_omni/model_executor/models/registry.py` | Added `MareyTextEncoder` to `_OMNI_MODELS`. |

---

## 4. Key contracts between stages

### Stage 0 `MareyTextEncoder.forward` → stage 1

`MareyTextEncoder` returns an `OmniOutput` with:
```python
OmniOutput(
    text_hidden_states=None,
    multimodal_outputs={"marey_text_encoder_out": [<dict per request>, ...]},
)
```
Each per-request dict:
```python
{
    "prompt_embeds":     [ul2_seq_tensor, byt5_seq_tensor],   # 2 tensors, CPU
    "prompt_masks":      [ul2_mask, byt5_mask],               # bool
    "vector_cond":       clip_pooled_tensor,                  # CPU
    "neg_prompt_embeds": [...] | None,
    "neg_prompt_masks":  [...] | None,
    "neg_vector_cond":   ... | None,
}
```
All tensors are on **CPU** before return (they'll be re-serialized for IPC
to stage 1 and moved back to GPU inside the DiT pipeline's forward).

`MareyTextEncoder` reads its inputs from
`runtime_additional_information[i]["prompt_text"]` and (optional)
`["negative_prompt_text"]`. The model **ignores** `input_ids` — it has
`requires_raw_input_tokens = True` and `embed_input_ids` returns a
zero-filled dummy embedding just to satisfy vLLM's runner.

### Stage 1 `MareyDitPipeline.forward` → stage 2

Expects `req.prompts[0]` to be a dict with:
```python
{
    "prompt": <string, optional, for logging>,
    "additional_information": {
        "prompt_embeds":     [ul2_seq, byt5_seq],
        "prompt_masks":      [ul2_mask, byt5_mask],
        "vector_cond":       clip_pooled,
        "neg_prompt_embeds": [...] | None,
        "neg_prompt_masks":  [...] | None,
        "neg_vector_cond":   ... | None,
    },
}
```
Returns:
```python
DiffusionOutput(
    output=z,                                              # final latent
    custom_output={"height": H, "width": W, "num_frames": F, "fps": fps},
)
```

### Stage 2 `MareyVaePipeline.forward`

Expects `req.prompts[0]` to be a dict with:
```python
{
    "prompt": <string, optional>,
    "additional_information": {
        "latents":     <tensor z>,
        "height":      H,
        "width":       W,
        "num_frames":  F,
        "fps":         fps,
    },
}
```
Returns `DiffusionOutput(output=video_tensor)`. Final post-processing to np /
pt is done by `get_marey_vae_post_process_func` (registered in
`_DIFFUSION_POST_PROCESS_FUNCS`).

---

## 5. Stage YAML (exact content)

At `vllm_omni/model_executor/stage_configs/marey.yaml`:

```yaml
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_stage: text_encoder
      model_arch: MareyTextEncoder
      hf_overrides:
        architectures: [MareyTextEncoder]
      worker_type: generation
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      engine_output_type: latent
      enforce_eager: true
      trust_remote_code: true
      distributed_executor_backend: "mp"
      enable_prefix_caching: false
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.15
      max_num_batched_tokens: 512
      max_model_len: 1024
    is_comprehension: true
    final_output: false
    default_sampling_params:
      temperature: 0.0
      top_k: 1
      max_tokens: 1
      detokenize: false
      seed: 42

  - stage_id: 1
    stage_type: diffusion
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.marey.text2diffusion
    runtime:
      devices: "1,2,3,4"
      max_batch_size: 1
    engine_args:
      model_stage: dit
      model_class_name: MareyDitPipeline
      enforce_eager: true
      trust_remote_code: true
      distributed_executor_backend: "mp"
      parallel_config:
        ulysses_degree: 4
    engine_input_source: [0]
    final_output: false
    is_comprehension: false
    default_sampling_params:
      seed: 42
      guidance_scale: 7.5
      num_inference_steps: 100

  - stage_id: 2
    stage_type: diffusion
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.marey.diffusion2vae
    runtime:
      devices: "5"
      max_batch_size: 1
    engine_args:
      model_stage: vae
      model_class_name: MareyVaePipeline
      enforce_eager: true
      trust_remote_code: true
      distributed_executor_backend: "mp"
    engine_input_source: [1]
    final_output: true
    final_output_type: video
    is_comprehension: false
    default_sampling_params:
      seed: 42

runtime:
  enabled: true
  defaults:
    window_size: -1
    max_inflight: 1
  edges:
    - { from: 0, to: 1, window_size: -1 }
    - { from: 1, to: 2, window_size: -1 }
```

The shared-memory connector stub from the plan is commented out in the
actual file so we don't pay for unused IPC.

---

## 6. Run command

```bash
uv run --project "${VLLM_OMNI_PROJECT}" vllm-omni serve "$MODEL" --omni \
    --port "$PORT" \
    --stage-configs-path "${VLLM_OMNI_PROJECT}/vllm_omni/model_executor/stage_configs/marey.yaml" \
    --flow-shift "$FLOW_SHIFT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
```

Differences from the old monolithic command:
- `--model-class-name MareyPipeline` removed (per-stage `model_class_name` /
  `model_arch` are in the YAML).
- `--ulysses-degree` removed (YAML sets it on stage 1 only). Override per-stage
  with `--stage-1-ulysses-degree N` if needed.

For separate-terminal launches (one stage per terminal), use bagel's pattern:
```
vllm-omni serve "$MODEL" --omni --stage-configs-path ... \
    --stage-id {0,1,2} --headless -oma <addr> -omp <port>
```
Reference: `examples/online_serving/bagel/run_server_stage_cli.sh`.

---

## 7. Expected iteration points on first run

Three places most likely to need small adjustments on first live run. Each is
~5 lines to fix once observed.

### (a) Stage-0 prompt injection shim

`MareyTextEncoder.forward` reads:
```python
runtime_additional_information[i]["prompt_text"]
runtime_additional_information[i]["negative_prompt_text"]   # optional
```
The original user-facing prompt for Marey is an `OmniTextPrompt` with
`prompt`, `negative_prompt` keys. vLLM's LLM entry-point doesn't typically
forward those through to `runtime_additional_information` — we need a shim.

**Where to look**: `vllm_omni/entrypoints/async_omni.py` and
`vllm_omni/entrypoints/openai/api_server.py` for how a stage-0 LLM request
gets built from the user prompt. Look for where `additional_information` is
populated. Compare with how `Qwen3TTSCode2Wav` receives `left_context_size`
via the same channel (see `qwen3_tts_code2wav.py:232-244` — it reads from
`runtime_additional_information[i]["left_context_size"]`).

If `prompt_text` isn't arriving, the fallback in `MareyTextEncoder.forward`
also tries `info.get("prompt")`. A minimal fix is to add a one-line entry
in whatever request-builder maps user prompts to stage-0 inputs.

**Quick debug**: add this at the top of `MareyTextEncoder.forward`:
```python
logger.warning("MareyTextEncoder forward: rai=%s keys=%s", type(runtime_additional_information), [list(d.keys()) for d in (runtime_additional_information or [])])
```

### (b) `multimodal_output` unwrapping

`text2diffusion` tries two shapes:
```python
enc_out = mm.get(_TEXT_ENC_MM_KEY) if isinstance(mm, dict) else None
if enc_out is None:
    enc_out = mm   # fallback: already-unwrapped
```
The vLLM output processor may or may not unwrap the list-per-request at a
level below our stage-input-processor. Whichever branch works is right; the
other can be removed once we see a live output shape.

**Quick debug** in `text2diffusion`:
```python
logger.warning("text2diffusion source_output=%s mm=%s", type(source_output), type(mm))
logger.warning("mm keys: %s", list(mm.keys()) if isinstance(mm, dict) else None)
```

### (c) Stage-1 → stage-2 latent access

`diffusion2vae` tries:
```python
latents = getattr(source_output, "latents", None)
if latents is None:
    latents = getattr(source_output, "output", None)
```
`OmniRequestOutput` (see `vllm_omni/outputs.py:57-148`) has a `.latents`
field populated by `from_diffusion(latents=...)`. The diffusion engine may or
may not route `DiffusionOutput.output` into `OmniRequestOutput.latents` —
we'll know once we see the error.

If neither works, the fix is either:
- set `DiffusionOutput(output=..., custom_output={"latents": ...})` in
  `MareyDitPipeline.forward`, or
- look at `vllm_omni/diffusion/diffusion_engine.py` to see how
  `DiffusionOutput` is mapped into `OmniRequestOutput` and add the routing.

**Quick debug** in `diffusion2vae`:
```python
logger.warning("diffusion2vae source_output fields: type=%s dir=%s",
               type(source_output), [a for a in dir(source_output) if not a.startswith("_")])
```

---

## 8. Debugging playbook

### Startup failures

- **"No pipeline mapping for model_type"** — the model's HF config
  `model_type` isn't in `PIPELINE_MODELS`. The staged path goes via
  `--stage-configs-path`, not `PIPELINE_MODELS`, so this shouldn't fire. If
  it does, verify the CLI flag is actually reaching `StageConfigFactory`.

- **"Model class MareyDitPipeline not found in diffusion model registry"** —
  `vllm_omni/diffusion/registry.py:_DIFFUSION_MODELS` edit didn't land, or a
  stale `.pyc` is shadowing it. Check `_DIFFUSION_MODELS` contains both
  `MareyDitPipeline` and `MareyVaePipeline`.

- **"Model arch MareyTextEncoder not found"** — the vLLM model registry edit
  didn't land. Check `vllm_omni/model_executor/models/registry.py:_OMNI_MODELS`.

- **Opensora import errors on stage 2** — `MOONVALLEY_AI_PATH` env var, or
  ensure `moonvalley_ai/` sits as a sibling to the vllm-omni repo root. See
  `vae_loader.py:resolve_moonvalley_dir()` for the 5-candidate search order.

- **Stage 1 can't find `ema_inference_ckpt.safetensors`** —
  `MareyDitPipeline.load_weights` searches
  `{model_dir}/epoch0-*/ema_inference_ckpt.safetensors` and
  `{model_dir}/**/ema_inference_ckpt.safetensors`. Check `od_config.model`
  points at the right directory.

### Runtime failures (follow the three iteration points above)

### Correctness checks

Once a request runs end-to-end:
- Compare output against the monolithic baseline with the same seed, prompt,
  `guidance_scale`, and `num_inference_steps`. Bit-exactness is unlikely
  (different process boundaries → different random state for the noise), but
  quality should match.
- CFG on (`guidance_scale > 1.0`): stage 0 runs both encoders; stage 1 does
  cond + uncond per step; stage 2 sees a single final latent.
- CFG off (`guidance_scale == 1.0`): stage 0 skips negative encoding; stage 1
  skips uncond forward entirely.

---

## 9. Design decisions & rationale

| Decision | Why |
|---|---|
| Stage 0 is `stage_type: llm` (not `diffusion`) | User preference; gives the KV-transfer machinery for free when a future autoregressive text-refinement stage is added. `OmniGenerationScheduler` + `requires_raw_input_tokens=True` lets a non-AR encoder fit this slot cleanly (same pattern as `Qwen3TTSCode2Wav`). |
| Two files for text-encoder code (bundle + vLLM adapter) | Keeps pure-PyTorch encoder glue reusable / testable without vLLM. Matches repo convention: `diffusion/models/` for pure model code, `model_executor/models/` for vLLM LLM-model adapters. |
| CFG in stage 1 (not orchestrator-level companions like bagel) | Marey's CFG only needs separate encoder **embeddings**, not separate KV caches. One stage-0 run produces both embedding sets; no companion prompt expansion needed. |
| No connectors in the YAML | Stage-to-stage data flows in-memory through the orchestrator (see `engine/orchestrator.py:521-583`). Connectors are only for KV transfer, async_chunk, or CFG companions — none apply. Shared-memory connector stub left commented for future KV-transfer upgrade. |
| Text-to-video only (no I2V) | User requirement. All `multi_modal_data` / `preprocessed_image` plumbing is removed. |
| SP on stage 1 via `parallel_config.ulysses_degree: 4` | Matches current single-stage deployment (`--ulysses-degree 4` across 4 GPUs). `_apply_sequence_parallel_if_enabled` at `vllm_omni/diffusion/registry.py:268` picks up `_sp_plan` on `MareyTransformer` unchanged. |
| Checkpoint loader stays on stage 1 | Only the DiT needs the `ema_inference_ckpt.safetensors` weights. Stage 0 encoders load from HF on first forward. Stage 2 VAE loads via opensora `PretrainedSpatioTemporalVAETokenizer.cp_path`. |

---

## 10. Reference patterns in-repo

When debugging, the closest existing analogs:

- **Non-AR LLM stage**: `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py` — same scheduler (`OmniGenerationScheduler`), same flags (`requires_raw_input_tokens`, `have_multimodal_outputs`), same `OmniOutput.multimodal_outputs` return path.
- **LLM → diffusion with CFG**: `vllm_omni/model_executor/stage_configs/bagel.yaml` + `vllm_omni/diffusion/models/bagel/pipeline_bagel.py` — our structural analog, but bagel does CFG via orchestrator companions; we don't.
- **Stage input processor**: `vllm_omni/model_executor/stage_input_processors/cosyvoice3.py:65` (`text2flow`) and `…/qwen3_tts.py:22` (`talker2code2wav`) — same signature `(stage_list, engine_input_source, prompt, requires_multimodal_data)`.
- **Diffusion pipeline interface**: `vllm_omni/diffusion/models/z_image/pipeline_z_image.py` is a simple reference for a stage-runnable diffusion `nn.Module` with `__init__(*, od_config, prefix)` and `forward(req) -> DiffusionOutput`.

---

## 11. Known non-goals

- Async-chunk streaming
- Image-to-video (I2V)
- Autoregressive text refinement (architecture is ready; not wired)
- Multi-node / RDMA deployment

---

## 12. First-run debugging session (log)

This section records the issues hit on the first live run and the fixes
landed on branch `aitor/marey_stages` (commits after `b8f47c3b Marey
stages implementation`). Each sub-section states the symptom, the root
cause, and the change.

Test harness:
- Server: `VLLM_OMNI_STORAGE_PATH=/mnt/localdisk/vllm_omni_storage HF_HOME=/mnt/localdisk/vllm_omni_hf_cache/ MODEL=/app/hf_checkpoints/marey-distilled-0100/ MOONVALLEY_AI_PATH=/home/aormazabal/wlam/wlam-inference/moonvalley_ai/ bash examples/online_serving/marey/run_server.sh`
- Request: `bash examples/online_serving/marey/run_curl_text_to_video.sh`

Outcome after the fixes below: request completes end-to-end,
`status: completed`, ~205 s inference (33 steps at 1920×1080, 128
frames), valid MP4 that visually matches the prompt.

### 12.1 Stage 1 DiT: caption-channel mismatch on weight load

**Symptom** (stage 1 init, during weight load):

```
RuntimeError: The size of tensor a (1472) must match the size of tensor b (1536) at non-singleton dimension 1
```

**Cause.** `MareyDitPipeline.__init__` read ByT5 hidden size from the
stage config with a hardcoded fallback of `1472` when the key wasn't
present in `config.yaml`. The actual ByT5-large `d_model` is `1536`, so
the `y_embedder` (caption embedder) was constructed with the wrong
input dim and the checkpoint tensors didn't match at load time. UL2
(4096) and CLIP (768) defaults happened to be right; ByT5 wasn't.

**Fix.** Replaced the hardcoded fallbacks in
`vllm_omni/diffusion/models/marey/pipeline_marey.py` with
`_resolve_encoder_dims(te_cfg)`, which pulls `d_model` /
`text_config.hidden_size` from HF `AutoConfig` for the configured
pretrained IDs. No extra weight I/O — only the JSON config is read.

### 12.2 Stage 1 DiT: dummy-run crash (no additional_information)

**Symptom** (stage 1 init, profile/warmup):

```
ValueError: MareyDitPipeline requires prompt_embeds and vector_cond in additional_information (populated by the text-encoder stage).
```

**Cause.** `DiffusionEngine._dummy_run` (see
`diffusion/diffusion_engine.py:402-447`) warms the pipeline with a
synthetic `OmniTextPrompt("dummy run")` that has no
`additional_information`. The strict check in `MareyDitPipeline.forward`
rejected this before any GPU work.

**Fix.** Added `_dummy_encoder_tensors()` (in `pipeline_marey.py`)
that fabricates zero tensors sized from the cached encoder dims. When
`prompt_embeds`/`vector_cond` are missing, `forward` now falls through
to the fabricated stand-ins so the full transformer path executes on
a representative shape. Cached `_caption_channels`, `_vector_cond_channels`,
`_ul2_max_length`, `_byt5_max_length` on the pipeline for that helper.

### 12.3 Stage 2 VAE: dummy-run crash (no latent)

**Symptom.** Same pattern as 12.2 but from the VAE stage: the engine
sends a dummy prompt, `MareyVaePipeline.forward` requires
`additional_information["latents"]`, raises.

**Fix.** Added `_dummy_latent(height, width, num_frames)` to
`pipeline_marey_vae.py` and fell back to it when the prompt's
`latents` is missing. Shapes come from
`(self.vae_scale_factor_temporal, self.vae_scale_factor_spatial,
 vae.out_channels)`.

### 12.4 Video endpoint rejects multi-stage pipelines

**Symptom** (runtime, first request):

```
Video generation only supports diffusion stages, found 'llm' stage.
```

**Cause.** `OmniOpenAIServingVideo._run_generation` walked the stage
list and raised 503 on any non-diffusion stage. The Marey pipeline has
an `llm`-typed stage 0.

**Fix** (`entrypoints/openai/serving_video.py`): allow stages of type
`llm` or `diffusion`, but keep the constraint that the final stage must
be diffusion (it produces frames).

### 12.5 Prompt text not reaching the text encoder

**Symptom.** `MareyTextEncoder.forward` raised
`"requires a non-empty 'prompt_text'"` on real requests because
nothing was populating `runtime_additional_information[i]["prompt_text"]`.

**Cause.** The video entrypoint built a plain
`OmniTextPrompt(prompt=...)` with no `additional_information`. vLLM's
`_upgrade_to_omni_request` only copies user-supplied
`additional_information`, so the stage-0 runtime had `None` for
`prompt_text`.

**Fix** (`serving_video.py`): before dispatching, shim
`additional_information["prompt_text"]` (and `negative_prompt_text` if
provided) onto the prompt dict. Keys are ignored by models that don't
consume them.

### 12.6 Stage 0 LLM: ModelConfig validation — no config.json

**Symptom** (stage-0 init):

```
ValueError: Invalid repository ID or local directory specified:
'/app/hf_checkpoints/marey-distilled-0100/'.
...
For Hugging Face models: ensure the presence of a 'config.json'.
```

**Cause.** vLLM's `ModelConfig` constructor needs an HF `config.json`
at `self.model`. The Marey checkpoint dir ships a `config.yaml` and
`model_index.json`; no `config.json`.

**Fix attempt → cascade.** Pointed stage-0 `hf_config_path` at a
repo-shipped stub. That cleared the "no config.json" error but the
stub's invented `model_type` tripped Transformers' registry:

```
ValueError: ... has model type `marey_text_encoder` but Transformers does not recognize this architecture.
```

**Final fix** (`vllm_omni/model_executor/stage_configs/marey.yaml`):
`hf_config_path: google/ul2`. The UL2 config is already cached because
the Marey text encoder bundle uses it for real, and it is a valid
Transformers config. `hf_overrides.architectures: [MareyTextEncoder]`
routes instantiation to our registered vLLM model, so UL2's own
architecture is never built. The stub directory at
`vllm_omni/model_executor/models/marey/te_stub/` was left in the tree
but is now unused — can be removed.

### 12.7 Stage 0 LLM: max_model_len > UL2 n_positions

**Symptom.**

```
ValueError: User-specified max_model_len (1024) is greater than the derived max_model_len (n_positions=512.0 ...)
```

**Cause.** Using UL2's config inherits `n_positions=512`. vLLM refuses
to exceed this without `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.

**Fix.** Lowered stage-0 `max_model_len` from 1024 to 512 in
`marey.yaml`. vLLM's tokenized prompt path is irrelevant here — the
encoder does its own tokenization with the real UL2/CLIP/ByT5
tokenizers — but vLLM still validates the limit.

### 12.8 Stage 0 LLM: encoder-decoder mm-budget assertion

**Symptom** (stage-0 core init, inside the scheduler):

```
AssertionError: Encoder-decoder models are expected to implement the multimodal interface with at most one modality.
```

**Cause.** With UL2's config in use, vLLM tags the stage as
encoder-decoder and asserts an `mm_budget` must be present with ≤ 1
modality. MareyTextEncoder isn't a vLLM cross-attention model and
doesn't declare a multimodal input budget.

**Fix** (`marey.yaml`): override the config to look decoder-only:

```yaml
hf_overrides:
  architectures: [MareyTextEncoder]
  is_encoder_decoder: false
  model_type: gpt2
```

The fields are never used by `MareyTextEncoder.forward` — they only
steer vLLM's scheduler/runner heuristics away from the T5 / enc-dec
paths.

### 12.9 OmegaConf env var for `VLLM_OMNI_PROJECT`

**Symptom.** First try at 12.6 used a repo-internal stub referenced
via `${oc.env:VLLM_OMNI_PROJECT,<fallback>}`. OmegaConf substitution
runs inside the Python process, which didn't see the env var because
`run_server.sh` didn't export it.

**Fix.** Added `VLLM_OMNI_PROJECT="${VLLM_OMNI_PROJECT}"` to the
`env_args` list in `examples/online_serving/marey/run_server.sh`.
Kept the export even after switching to `hf_config_path: google/ul2`
in case future per-stage configs reference repo-local paths.

### 12.10 Params-type mismatch on stage 0

**Symptom** (first real request):

```
TypeError: params must be either SamplingParams or PoolingParams, but got OmniDiffusionSamplingParams
```

**Cause.** `serving_video._run_generation` built
`sampling_params_list = [gen_params for _ in stage_configs]` — the user's
`OmniDiffusionSamplingParams` went to every stage, including the llm
stage. vLLM's input processor rejects that type.

**Fix** (`serving_video.py`): build a per-stage list — diffusion stages
get the user-supplied `gen_params`, llm stages get the YAML default
from `engine_client.default_sampling_params_list[idx]` (which is a
real `SamplingParams` for llm stages by construction in
`engine/stage_init_utils.py:302`).

### 12.11 Stage 0 → Stage 1: multimodal_outputs shape contract

**Symptom** (worker, inside the stage-0 output handler):

```
AttributeError: 'dict' object has no attribute 'detach'
File ".../gpu_generation_model_runner.py", line 394, in sample_tokens
  mm_payload[key] = out[i].detach().to("cpu").contiguous()
```

**Cause.** `MareyTextEncoder.forward` returned
`OmniOutput(multimodal_outputs={"marey_text_encoder_out": [dict_per_req]})`.
The worker's mm handler expects `dict[str, Tensor | list[Tensor]]` and
calls `.detach()` on each element; nested dicts break the contract.

**Fix.** Two changes, one on each side of the boundary:

1. `model_executor/models/marey/marey_text_encoder.py` — rewrote
   `forward` to emit a **flat** dict of per-key, per-request tensors.
   Keys: `prompt_embeds_ul2`, `prompt_embeds_byt5`,
   `prompt_masks_ul2`, `prompt_masks_byt5`, `vector_cond`, and
   the corresponding `neg_*`. Each value is a `list[Tensor]` of length
   `num_reqs`. Always encodes the negative prompt (with
   `DEFAULT_NEGATIVE_PROMPT` if none supplied) so the shape contract is
   fixed — stage 1 decides whether to apply CFG based on
   `guidance_scale`. Added `_append_dummy(...)` for the profile/dummy
   path (no prompt text) so every key still gets a tensor.
2. `model_executor/stage_input_processors/marey.py::text2diffusion` —
   reassembles the flat keys back into the
   `[[ul2, byt5], [ul2_mask, byt5_mask], vector_cond, ...]` shapes that
   `MareyDitPipeline.forward` expects in
   `prompt.additional_information`.

### 12.12 Stage 1 post-process corrupts the latent

**Cause** (discovered while tracing the stage-1 → stage-2 hand-off). The
DiT stage's `get_marey_post_process_func` was wired in the registry
(`diffusion/registry.py:_DIFFUSION_POST_PROCESS_FUNCS`) and applied
`VideoProcessor.postprocess_video` to `DiffusionOutput.output`, which
for stage 1 is the raw denoised latent — not pixel frames. That would
mangle the tensor before it ever reached the VAE.

**Fix** (`pipeline_marey.py`): turned `get_marey_post_process_func`
into a passthrough (returns the latent unchanged regardless of
`output_type`). Frame post-processing happens only in
`get_marey_vae_post_process_func` after stage 2.

### 12.13 Stage 1 → Stage 2: latent isn't on `.latents`

**Cause** (follow-up to 12.12). `DiffusionEngine` packs
`DiffusionOutput.output` into `OmniRequestOutput.images` (see
`diffusion_engine.py:237`), not `.latents`. The original
`diffusion2vae` tried `source_output.latents` then `source_output.output`
— both `None`.

**Fix.** Two-sided:

- `pipeline_marey.py::MareyDitPipeline.forward` now also stashes the
  latent in `DiffusionOutput.custom_output["latents"]`. The diffusion
  engine forwards `custom_output` to `OmniRequestOutput._custom_output`,
  which is accessible to downstream stage input processors.
- `stage_input_processors/marey.py::diffusion2vae` reads
  `source_output._custom_output["latents"]` first, with fallbacks to
  `.latents` and `.images[0]` just in case a different engine path
  routes it differently.

### 12.14 Orchestrator: diffusion → diffusion forwarding

**Symptom** (first end-to-end request, mid-pipeline):

```
AttributeError: 'StageDiffusionClient' object has no attribute 'set_engine_outputs'
File ".../engine/orchestrator.py", line 538, in _forward_to_next_stage
  self.stage_clients[stage_id].set_engine_outputs([output])
```

**Cause.** The orchestrator stashes a completed stage's output on the
client so the next stage's input processor can read it via
`stage_list[src].engine_outputs`. `StageEngineCoreClient` implements
`set_engine_outputs(...)` and an `engine_outputs` attribute;
`StageDiffusionClient` didn't. Fine for pipelines where only the last
stage was diffusion; broken once a diffusion stage feeds another
diffusion stage (here: DiT → VAE).

**Fix** (`diffusion/stage_diffusion_client.py`): added
`engine_outputs: Any = None` to client init and a
`set_engine_outputs(...)` method that mirrors `StageEngineCoreClient`'s.

### 12.15 Summary of changed files

Code files touched across the debugging session (commit
`7b2a0c1e Debugging` and follow-ups):

- `vllm_omni/model_executor/stage_configs/marey.yaml` —
  `hf_config_path`, `tokenizer`, `hf_overrides`, `max_model_len: 512`.
- `vllm_omni/model_executor/models/marey/marey_text_encoder.py` —
  flat multimodal_outputs contract + dummy fallback.
- `vllm_omni/model_executor/models/marey/te_stub/config.json` —
  created during debug, no longer used after switching to
  `google/ul2`; safe to delete.
- `vllm_omni/model_executor/stage_input_processors/marey.py` —
  `text2diffusion` reassembly; `diffusion2vae` custom_output read.
- `vllm_omni/diffusion/models/marey/pipeline_marey.py` —
  `_resolve_encoder_dims`, `_dummy_encoder_tensors`, passthrough
  post-process, latent in `custom_output`.
- `vllm_omni/diffusion/models/marey/pipeline_marey_vae.py` —
  `_dummy_latent`.
- `vllm_omni/diffusion/stage_diffusion_client.py` —
  `set_engine_outputs` + `engine_outputs`.
- `vllm_omni/entrypoints/openai/serving_video.py` — allow llm+diffusion
  mix, prompt-text shim, per-stage sampling params.
- `examples/online_serving/marey/run_server.sh` —
  export `VLLM_OMNI_PROJECT`.

### 12.16 Still open / worth a follow-up

- **CFG correctness not yet compared to the monolithic baseline.** The
  curl run uses `guidance_scale=3.5`. Stage 0 always encodes negative
  now (see 12.11) and stage 1's CFG path was exercised during the
  successful request. But bit-exact comparison against the old
  monolithic pipeline hasn't been done.
- **`te_stub/` directory is dead code** — from the initial attempt at
  12.6 before pivoting to `google/ul2`. Safe to delete along with the
  comments referencing it.
- **Video bitrate is much higher than the baseline's** (~22 Mbps vs
  ~1.5 Mbps on the reference mp4 at the same prompt). That's the
  `preset=ultrafast` encode setting in the API server, not a pipeline
  issue.
- **Stage-0 output uses UL2 as a fake config.** If vLLM changes
  enc-dec handling or validates more config fields, the stage-0 boot
  path may regress. A repo-local real HF config (sized to 512) would
  be more hermetic but requires trust_remote_code or a registered
  `model_type`.
