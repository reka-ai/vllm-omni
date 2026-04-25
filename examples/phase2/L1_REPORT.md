# L1 — vllm-omni vs moonvalley parity, full-injection mode

**Purpose:** isolate the **transformer forward math** as the only variable.

When everything else (text embeddings, initial noise, per-step DDPM noise, **and every per-step transformer input**) is injected from moonvalley's reference dump, the only remaining computation that vllm-omni does on its own is the transformer's forward pass. Any divergence at the transformer output (`v_pred`) is therefore directly attributable to numerical differences inside the transformer — kernel choice, reduction order, etc.

L1 answers the question: **at the kernel/numerical-primitive level, how close is vllm-omni's transformer to moonvalley's?**

---

## Setup

| | Value |
|---|---|
| Branch | `marey-serving-comparison` (vllm-omni), `marey-serving-comparison` (moonvalley_ai) |
| Model | 30B distilled Marey, `/app/hf_checkpoints/marey-distilled-0100` |
| Resolution | 1920×1080, 128 frames, 24 fps |
| Steps | 33 (distilled) |
| Seed | 42 |
| ulysses_degree | 8 |
| Ref dump | `/mnt/localdisk/vllm_omni_storage/phase1/ref_30b/` |
| Run dump | `/mnt/localdisk/vllm_omni_storage/phase2/vllm_runA/` |

**L1 injection set** (from `examples/phase2/run_vllm_omni_dump.sh`, `LEVEL=L1`):

```
MAREY_LOAD_INITIAL_NOISE          = ref_30b/z_initial.pt
MAREY_LOAD_STEP_NOISE_DIR         = ref_30b/
MAREY_LOAD_TEXT_EMBEDS_DIR        = ref_30b/
MAREY_LOAD_TRANSFORMER_INPUTS_DIR = ref_30b/        # the L1-defining one
```

### Exact reproduce — full pipeline

The L1 comparison requires **two inferences**: mv produces the reference dump, vllm-omni runs against it.

#### 1. moonvalley reference inference (one-time, produces `ref_30b/`)

Wrapper: `examples/phase1/run_moonvalley_dump.sh`. Drives mv via `marey_reference_inference_dump.py` which monkey-patches `RFLOW.sample` and `text_encoder.{encode,null}` for tensor capture.

One-liner:
```bash
bash /home/yizhu/code/vllm-omni/examples/phase1/run_moonvalley_dump.sh ref_30b
```

What that wrapper does (env + CLI):
```bash
# Workaround: flash_attn_3 wheels need GLIBCXX_3.4.32+ (Ubuntu 22.04 system libstdc++ tops out at 3.4.30).
export LD_PRELOAD=/home/yizhu/miniconda3/lib/libstdc++.so.6

export MAREY_DUMP_DIR=/mnt/localdisk/vllm_omni_storage/phase1
export MAREY_DUMP_REQUEST_ID=ref_30b
export MOONVALLEY_AI_PATH=/home/yizhu/code/moonvalley_ai_master

# moonvalley resolves vae.cp_path relative to CWD; cd to its repo root.
cd /home/yizhu/code/moonvalley_ai_master
ln -sf /app/hf_checkpoints/marey-distilled-0100/vae.ckpt ./vae.ckpt   # symlink for relative path resolution

PYTHONPATH=/home/yizhu/code/moonvalley_ai_master/inference-service:/home/yizhu/code/moonvalley_ai_master:/home/yizhu/code/moonvalley_ai_master/open_sora \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/yizhu/code/moonvalley_ai_master/inference-service/.venv/bin/torchrun --nproc_per_node=8 \
    /home/yizhu/code/vllm-omni/examples/online_serving/marey/marey_reference_inference_dump.py infer \
    --num-seq-parallel-splits 8 \
    --offload-diffusion --offload-vae --offload-text-encoder \
    --model-folder      /app/hf_checkpoints/marey-distilled-0100 \
    --checkpoint-folder /app/hf_checkpoints/marey-distilled-0100 \
    --watermarker-path  /app/wlam/models/checkpoints/marey/videoseal/y_256b_img.jit \
    --height 1080 --width 1920 --num-frames 128 --fps 24 \
    --steps 33 --guidance-scale 3.5 --disable-caching \
    --use-negative-prompts \
    --negative-prompt "<see runner — long string>" \
    --use-distilled-steps --shift-value 3.0 \
    --use-guidance-schedule --add-quality-guidance --clip-value 10.0 \
    --seed 42 --warmup-steps 4 --cooldown-steps 18 \
    --save-latents \
    --output /mnt/localdisk/vllm_omni_storage/phase1/ref_30b/output.mp4 \
    "<eagle prompt — see runner>"
```

Notes:
- **Drop** `--use-timestep-transform`. On the comparison branch it's a switch-pair (passing it sets True); on main it's a single flag (passing it toggles default-True → False). Same argv produces different results across branches. The runtime default is True on both, so don't pass it.
- **mv-side fix (2026-04-25):** `_setup_scheduler_config` originally dropped `--use-distilled-steps` (didn't propagate to scheduler's OmegaConf), so the scheduler ran `linspace(tmax, tmin, num_steps)` ending at ~3 instead of the documented stride-3 path ending at ~88. Fix: one `OmegaConf.update(self.model_cfg, "scheduler.use_distilled_steps", params.get("use_distilled_steps", False))` line added at `inference-service/marey_inference.py:_setup_scheduler_config`. After fix, mv's `timesteps` ends at 88.30 and matches vllm-omni's documented schedule byte-for-byte (`timesteps_schedule` rel = 0.000%). Documented in `~/.claude/projects/-home-yizhu-code-vllm-omni/memory/project_mv_distilled_steps_bug.md`.
- Output dir contains 1033 `.pt` tensor files + `latents.pt` + `output.mp4` + `run.log`.

#### 2. vllm-omni L1 inference (against `ref_30b/`)

Wrapper: `examples/phase2/run_vllm_omni_dump.sh`. Spawns the vllm-omni server, fires the curl client, reaps the server, runs `summary_report.py`.

One-liner:
```bash
cd /home/yizhu/code/vllm-omni
LEVEL=L1 bash examples/phase2/run_vllm_omni_dump.sh vllm_runA
```

What that wrapper does (env + CLI):
```bash
# Pipeline + injection
export MAREY_PIPELINE_CLASS=DumpMareyPipeline
export MAREY_DUMP_DIR=/mnt/localdisk/vllm_omni_storage/phase2/vllm_runA
export MAREY_LOAD_INITIAL_NOISE=/mnt/localdisk/vllm_omni_storage/phase1/ref_30b/z_initial.pt
export MAREY_LOAD_STEP_NOISE_DIR=/mnt/localdisk/vllm_omni_storage/phase1/ref_30b
export MAREY_LOAD_TEXT_EMBEDS_DIR=/mnt/localdisk/vllm_omni_storage/phase1/ref_30b
export MAREY_LOAD_TRANSFORMER_INPUTS_DIR=/mnt/localdisk/vllm_omni_storage/phase1/ref_30b

# Generic env (paths + storage)
export HF_HOME=/mnt/localdisk/vllm_omni_hf_cache
export VLLM_OMNI_STORAGE_PATH=/mnt/localdisk/vllm_omni_storage
export MODEL=/app/hf_checkpoints/marey-distilled-0100/
export MOONVALLEY_AI_PATH=/home/yizhu/code/moonvalley_ai_master

# Server (one process, 8 GPUs via ulysses)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
env "PYTORCH_CUDA_ALLOC_CONF=..." "MOONVALLEY_AI_PATH=..." "HF_HOME=..." "VLLM_OMNI_STORAGE_PATH=..." \
    "MAREY_DUMP_DIR=..." "MAREY_LOAD_INITIAL_NOISE=..." "MAREY_LOAD_STEP_NOISE_DIR=..." \
    "MAREY_LOAD_TEXT_EMBEDS_DIR=..." "MAREY_LOAD_TRANSFORMER_INPUTS_DIR=..." \
    /home/yizhu/code/vllm-omni/.venv/bin/python -m vllm_omni.entrypoints.cli.main serve \
        /app/hf_checkpoints/marey-distilled-0100/ --omni \
        --port 8098 \
        --model-class-name DumpMareyPipeline \
        --flow-shift 3.0 \
        --gpu-memory-utilization 0.98 \
        --ulysses-degree 8 \
    >"$SERVER_LOG" 2>&1 &

# (wait for "Application startup complete" in server log, then:)

# Client — submits the canonical eagle prompt with seed=42
SEED=42 OUTPUT_PATH=/mnt/localdisk/vllm_omni_storage/phase2/vllm_runA/output.mp4 \
    bash /home/yizhu/code/vllm-omni/examples/online_serving/marey/run_curl_text_to_video.sh
```

The client (`run_curl_text_to_video.sh`) submits the same eagle prompt mv used (encoded in the script as `prompt=...`), the same `seed=42`, then polls the `/v1/videos/{video_id}` status until ready and downloads the mp4.

Notes:
- The runner's pkill pattern is `vllm_omni.entrypoints.cli.main serve` (underscore, matching the actual command line). Earlier `vllm-omni serve` (hyphen) never matched — fixed bug.
- After inference, the runner symlinks `vllm_runA/vllm_runA → vllm_runA/video_gen_<id>` (the actual dump dir). It picks the subdir with the most `.pt` files so the warmup-pass dir doesn't shadow it.
- Auto-runs `python examples/phase2/summary_report.py --run <dump_subdir>` at the end, producing `report.md`.

#### 3. Diff
```bash
# After both runs complete:
python /home/yizhu/code/vllm-omni/examples/phase2/summary_report.py \
    --ref /mnt/localdisk/vllm_omni_storage/phase1/ref_30b \
    --run /mnt/localdisk/vllm_omni_storage/phase2/vllm_runA/vllm_runA \
    --tag vllm_runA \
    --out /mnt/localdisk/vllm_omni_storage/phase2/vllm_runA/report.md
```

---

## Results

### End-to-end

| Metric | Value | Meaning |
|---|---:|---|
| `z_final` shape | `(1, 16, 32, 68, 120)` = 4.18M elements | full pre-VAE latent |
| max_abs_diff | 8.722e-01 | element-wise max abs deviation |
| mean_abs_diff | 3.802e-02 | element-wise mean abs deviation |
| **relative_avg_diff** | **4.889%** | mean_abs_diff / mean(\|z_mv\|) |
| **cosine_sim** | **0.998886** | angle ≈ 2.7° between vectors in 4.2M-dim space |

### Per-category

| Category | n | mean_rel | max_rel | min_cos | What it represents |
|---|---:|---:|---:|---:|---|
| `z_final` | 1 | 4.889% | 4.889% | 0.998886 | end-to-end pre-VAE latent |
| `transformer_v_pred` | 42 | **0.811%** | 1.771% | 0.999844 | transformer output, per (step, label) |
| `transformer_hidden_states` | 42 | 0.134% | 0.143% | 0.999999 | injected z_t after fp32→bf16 cast |
| `text_seq_cond` | 4 | **0.000%** | 0.000% | 1.000000 | injected UL2/ByT5 embeds (byte-identical) |
| `transformer_encoder_hidden_states` | 84 | **0.000%** | 0.000% | 1.000000 | injected per-step text inputs |
| `transformer_extra` | 336 | **0.000%** | 0.000% | 1.000000 | injected aesthetics/ar/fps/etc. |
| `transformer_timestep` | 42 | **0.000%** | 0.000% | 1.000000 | injected per-step timestep scalar |
| `step_noise` | 32 | **0.000%** | 0.000% | 1.000000 | injected per-step DDPM noise |
| `timesteps_schedule` | 1 | **0.000%** | 0.000% | 1.000000 | full 33-step schedule (after timestep-fix) |
| `z_initial` | 1 | **0.000%** | 0.000% | 1.000000 | injected initial noise |

**Reading:** every injected input is byte-identical to mv's reference. The 0.134% on `transformer_hidden_states` is pure float32→bfloat16 quantization (mv stores f32 on disk, vllm-omni loads f32 and casts to bf16 for the transformer). The single non-trivial category is `transformer_v_pred`: **0.811% mean relative deviation across 42 transformer calls**, with cosine_sim 0.9998+ everywhere (so the velocity directions are nearly identical, just with small magnitude noise).

### Per-step `v_pred` divergence trajectory

```
step  cond_rel   uncond_rel    note
  0   0.844%      0.782%       baseline (z is injected initial noise on both sides)
  1   0.662%      0.825%
  2   0.723%      0.695%
  3   0.832%      0.752%
  ...
 14   0.711%      —            (cooldown begins here; uncond no longer called)
 15   0.735%      —
 ...
 27   0.799%      —
 28   0.861%      —            (cooldown bias starts to grow)
 29   0.910%      —
 30   1.007%      —
 31   1.272%      —
 32   1.771%      —            (final step, biggest per-step deviation)
```

Pre-cooldown (steps 0–14, with both cond+uncond): rel diff stays **flat at 0.7–0.9%**. Post-cooldown (steps 15+, cond-only): mostly flat, then grows in the last 5 steps. The growth in the last 5 steps reflects that vllm-omni's `v_pred` magnitudes are larger near the end of denoising (`mean(|v_pred|)` shrinks as `t→0`), so the same absolute noise floor produces a larger relative number. The cosine_sim stays >0.9998 throughout.

---

## Verification checklist — what was ruled out

The 0.811% per-step deviation could in principle come from any of: a swapped weight, a different LayerNorm impl, a different RoPE formula, an attention-masking semantics mismatch, an SP collective bug, or a kernel-implementation difference. We verified each:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **SwiGLU gate/up swap** | ✅ ruled out | `_MLP_WEIGHT_MAP` in `marey_transformer.py:1246` remaps `fc1_x→w1` and `fc1_g→w2`. vllm-omni computes `silu(w2(x)) * w1(x)` = `silu(fc1_g(x)) * fc1_x(x)`, which matches timm's `SwiGLU.forward` exactly. |
| **`LlamaRMSNorm` eps + fp32 promotion** | ✅ ruled out | `marey_transformer.py:115-127` and `moonvalley_ai/.../layers/norms.py:4-19` are line-by-line identical. Both use `eps=1e-6`, both cast to fp32 before computing variance. |
| **`apply_rope` / `apply_rotary_emb`** | ✅ ruled out | Implementations in `marey_transformer.py:79-106` and `moonvalley_ai/.../layers/rope.py` are line-by-line identical, including the float32 promotion of `fraction` and `positions`. |
| **`t2i_modulate(x, shift, scale)`** | ✅ ruled out | Both compute `x * (1 + scale) + shift`. Identical 1-line function. |
| **Modulation Linear path** | ✅ ruled out | 30B config has `use_block_v2: true` → mv takes the `nn.Linear(h, 6h, bias=True)` path. vllm-omni uses the same layer. Chunk-into-6 ordering is the same on both sides (both interpret the linear output as `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]`). |
| **Block forward order** | ✅ ruled out | `pre_norm → modulate → attn → gate-residual → pre_norm → modulate → mlp → gate-residual` on both sides. Compared `MareyFluxBlock.forward` to mv's `FluxBlock.forward → pre_attention → post_attention`. Same structure. |
| **Attention masking semantics** | ✅ ruled out | mv's `flux.py:908-930` `rearrange_tokens_for_attention` reduces to `y * mask` (it doesn't actually rearrange — the function is misnamed). mv's `_sdpa_attention` and `_flash_attention` take only `(q, k, v)` — no mask. vllm-omni's `MareyTransformer.forward:1175` does `y = seq_cond * mask` and runs fully-dense attention. **Both sides use mask-via-V-zeroing + dense attention**, so attention pattern is equivalent (both have ~18-55% padding "ghost weight" via softmax; same on both sides). |
| **Weight loading completeness** | ✅ ruled out | Server log shows zero `Skipping weight` warnings. All 1100 checkpoint keys mapped to model params/buffers via the `y_embedder.vector_embedding_0 → y_embedder.vector_embedding` remap and the SwiGLU `_MLP_WEIGHT_MAP`. |
| **ulysses_degree (SP collective bf16 noise)** | ✅ **ruled out by direct measurement** | Ran the same L1 setup at `ulysses_degree=4`. Result: z_final 4.8966% vs ul=8's 4.8964% (Δ = 0.0001% absolute), v_pred mean_rel 0.844% vs 0.842% (Δ = 0.002%). If SP collective reduction-order in bf16 were the source, halving the SP degree would change the noise pattern. Bit-essentially-identical means the divergence is **invariant to ulysses degree** — i.e., not from SP. (Measured prior to mv-side timestep fix; numbers are within run-to-run variance of the post-fix L1.) |
| Per-rank SP shard alignment | ⚠️ benign permutation | Block-0 INPUT rank-0 dumps gave cos_sim 0.0096 (orthogonal). But sorted token L2-norms gave cos_sim **0.999998** — same set of tokens, different sequence order. Each codebase shards visual tokens differently across SP ranks; this is a layout-only difference, not a value error. The full-tensor `v_pred` (post all-gather) is unaffected. |

---

## Most likely cause — the only remaining numerical primitive

**Different `flash_attn_3` builds** in the two venvs:

| Side | Package | Source |
|---|---|---|
| **vllm-omni** | `fa3_fwd 0.0.2` | PyPI `fa3_fwd-0.0.2-cp39-abi3-manylinux_2_24_x86_64.whl` |
| **moonvalley** | `flash_attn_3 3.0.0b1` | `mv-packages.s3.amazonaws.com/python/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl` |

Both expose `flash_attn_func` and `flash_attn_varlen_func` with the same signatures, so vllm-omni's `fa.py` fallback chain (`fa3_fwd_interface` → `flash_attn_interface` → FA2) imports whichever is installed. They are **independent FA3 implementations** — different upstreams, different kernel scheduling, different bf16 reduction order in the softmax.

**Why this matches the L1 signature exactly:**

- **Deterministic** divergence (same value across multiple runs) — kernel-level, not stochastic SP collective noise.
- **Invariant to ulysses degree** (verified at ul=4 vs ul=8) — kernel-level, not collective-level.
- **Architectural components verified identical** — only remaining numerical primitive that could differ.

**Why we couldn't directly verify with a swap test:**

The two wheels are torch-ABI-locked. Cross-installing fails:

```
flash_attn_3 3.0.0b1 on vllm-omni's torch 2.10:
    ImportError: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib

fa3_fwd 0.0.2 on mv's torch 2.7.1:
    ImportError: undefined symbol: aoti_torch_create_device_guard
```

Definitive verification requires building `flash_attn_3` from source against vllm-omni's torch 2.10 + cu129. ~30-60 min CUDA build, deferred.

---

## Bugs found and fixed during the L1 investigation

| Bug | Symptom | Fix |
|---|---|---|
| mv `_setup_scheduler_config` dropped `--use-distilled-steps` | `timesteps_schedule` rel 3.04%; mv's last step 2.99 (linspace) vs vllm-omni's documented 88.3 (stride-3); corrupted `sigma_t` in mv's DDPM math at every step | One-line fix in `inference-service/marey_inference.py:_setup_scheduler_config`: `OmegaConf.update(self.model_cfg, "scheduler.use_distilled_steps", params.get("use_distilled_steps", False))`. After fix, both sides run the documented stride-3 distilled schedule (last step ≈ 88.30). vllm-omni's `_create_flow_timesteps` was reverted to its original documented implementation. |
| Post-hook label collision | All per-step dumps labeled `unknown`; cond+uncond writes collided onto same filename | Pre-hook stashes resolved label in `_pending_call_label`; post-hook consumes it (id-match was failing because pre-hook had already replaced `encoder_hidden_states`). |
| Runner cleanup pattern | Old vllm-omni server processes left running between runs | `pkill -f` pattern was `vllm-omni serve` (hyphen); actual command uses `vllm_omni` (underscore). Fixed pattern. |
| Symlink to warmup dir | Runner symlinked `<tag>` to the warmup-pass req dir (~23 files) instead of the real-inference dir (~600 files) | Pick the subdir with the most `.pt` files. |
| `_dump_timesteps` list-vs-tensor | Inference crashed: `'list' object has no attribute 'detach'` | `_create_flow_timesteps` returns a list of scalars; `_dump_timesteps` now `torch.cat`s them. |

---

## Conclusion

**vllm-omni's transformer is computationally equivalent to moonvalley's at the architecture level.** All structural components — modulation, RoPE, RMSNorm, SwiGLU, attention masking, weight layout, scheduler math — were verified identical or computationally equivalent.

The residual **0.811% per-step `transformer_v_pred` divergence** at L1 is most plausibly caused by the two codebases shipping **different `flash_attn_3` builds** (`fa3_fwd 0.0.2` vs `flash_attn_3 3.0.0b1`). This explains:

- Why the divergence is deterministic.
- Why it's invariant to ulysses degree.
- Why no architectural inspection found a bug.

The cosine_sim is 0.9998+ on every transformer output — the divergence is mostly a small magnitude offset, not a directional error. End-to-end, `z_final` is within **4.89% rel-avg / cosine 0.9989** of mv's reference, which corresponds to an angle of ≈2.7° between the two latent vectors.

For applications that can tolerate this noise level, vllm-omni passes L1 and ships as-is. For bit-identical parity with mv (relevant only if downstream consumers are extremely sensitive to per-pixel differences), the path forward is to install `flash_attn_3` matching mv's build, against vllm-omni's torch ABI.

---

## Related artifacts

| Path | Purpose |
|---|---|
| `examples/phase2/PHASE2_FINDINGS.md` | Combined Phase 2 (L1 + L2) summary with bugs-fixed table and full investigation record |
| `examples/phase2/L2_REPORT.md` | (TBD) standalone L2 narrative — scheduler-recurrence compounding |
| `/mnt/localdisk/vllm_omni_storage/phase2/vllm_runA/report.md` | Auto-generated per-run L1 metrics report |
| `/mnt/localdisk/vllm_omni_storage/phase2/vllm_runA_ul4/report.md` | Same setup at `ulysses_degree=4` (used to rule out SP) |
| `~/.claude/projects/-home-yizhu-code-vllm-omni/memory/project_phase2_fa3_mismatch.md` | Cross-session memory pointer for the FA3 finding |
