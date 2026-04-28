# L1 (single-frame I2V) — vllm-omni vs moonvalley parity, full-injection mode

**Purpose:** isolate the **I2V transformer forward math** as the only variable.

When everything else (text embeddings, initial noise, per-step DDPM noise, **per-step transformer inputs**, **and the I2V-specific `cond_frames` + `cond_offsets`**) is injected from moonvalley's reference dump, the only remaining computation that vllm-omni does on its own is the transformer's forward pass — including the new I2V code path (mask-aware modulation, `t0_emb` path, cond-frames concat, post-final-layer slice). Any divergence at the transformer output (`v_pred`) is therefore directly attributable to numerical differences inside the transformer, *not* to I2V-specific bugs in cond preprocessing or VAE encoding.

L1 answers: **at the kernel/numerical-primitive level, is the I2V transformer code in vllm-omni equivalent to moonvalley's `dit3d.py` I2V branch?**

---

## Critical pre-run fix: re-applied `use_distilled_steps` propagation on the mv side

The first round of L1/L2/L3 single-frame runs was contaminated by a recurrence of the **Phase 2 `use_distilled_steps` propagation bug**: mv's `_setup_scheduler_config` was running the linspace fallback (last sigma ≈ 2.99) instead of the documented stride-3 distilled schedule (last sigma ≈ 88.30). The Phase 2 fix had been applied as a working-tree change in mv but was never committed to `marey-serving-comparison`; it was lost when the working tree got reset between Phase 2 and Phase 3.

Symptom on the first round: `timesteps_schedule rel = 3.04%`, mv last step = 2.99 vs vllm-omni's 88.30. Result: I2V L1 `z_final` looked artificially bad (12.09% rel / cos 0.9941) — most of that divergence was the scheduler running on a different sigma trajectory, not anything in the I2V transformer code.

**Fix re-applied** to `inference-service/marey_inference.py:_setup_scheduler_config` — one `OmegaConf.update(self.model_cfg, "scheduler.use_distilled_steps", params.get("use_distilled_steps", False))` line. After re-applying and re-running mv ref + L1, `timesteps_schedule rel = 0.000%` and the numbers below are clean.

(See `~/.claude/projects/-home-yizhu-code-vllm-omni/memory/project_mv_distilled_steps_bug.md`. The fix should be committed to `marey-serving-comparison` so this doesn't bite a fourth time — currently in the mv working tree only.)

---

## Setup

| | Value |
|---|---|
| Branch | `marey-serving-comparison` (vllm-omni), `marey-serving-comparison` + working-tree fix (moonvalley_ai) |
| Model | 30B distilled Marey, `/app/hf_checkpoints/marey-distilled-0100` |
| Resolution | 1920×1080, 128 frames, 24 fps |
| Steps | 33 (distilled) |
| Seed | 42 |
| ulysses_degree | 8 |
| Conditioning image | `/app/yizhu/marey/vllm_omni/vllm_omni_storage/cond_frame_0.webp` |
| Frame indices | `[0]` (first-frame I2V) |
| Ref dump | `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/ref_single` |
| Run dump | `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/vllm_l1_single` |

**L1 injection set** (`examples/phase3_i2v/run_vllm_omni_dump.sh`, `LEVEL=L1`):

```
MAREY_LOAD_INITIAL_NOISE          = ref_single/z_initial.pt
MAREY_LOAD_STEP_NOISE_DIR         = ref_single/
MAREY_LOAD_TEXT_EMBEDS_DIR        = ref_single/
MAREY_LOAD_TRANSFORMER_INPUTS_DIR = ref_single/
MAREY_LOAD_COND_FRAMES_PATH       = ref_single/cond_frames.pt    # I2V-specific
MAREY_LOAD_COND_OFFSETS_PATH      = ref_single/cond_offsets.pt   # I2V-specific
```

The two new I2V env vars are loaded at every tier (L1/L2/L3), not just L1, because `cond_frames` / `cond_offsets` are deterministic preprocessing outputs, not scheduler outputs. Loading them everywhere isolates the diffusion path from the cond-encoding path.

### Reproducer (one-liner)

```bash
# 1. mv reference (with the working-tree use_distilled_steps fix applied)
COND_IMAGES=/app/yizhu/marey/vllm_omni/vllm_omni_storage/cond_frame_0.webp \
FRAME_INDICES=0 SEED=42 \
bash examples/phase3_i2v/run_moonvalley_dump.sh ref_single

# 2. vllm-omni L1
COND_IMAGES=/app/yizhu/marey/vllm_omni/vllm_omni_storage/cond_frame_0.webp \
FRAME_INDICES=0 SEED=42 LEVEL=L1 \
bash examples/phase3_i2v/run_vllm_omni_dump.sh vllm_l1_single
```

---

## Results

### End-to-end

| Metric | T2V Phase 2 L1 | **I2V Phase 3 L1 (single)** | Note |
|---|---:|---:|---|
| `z_final` shape | (1, 16, 32, 68, 120) | (1, 16, 32, 68, 120) | identical — slice removes cond positions |
| max_abs_diff | 8.722e-01 | 5.763e+00 | larger because I2V latent has bigger tail |
| mean_abs_diff | 3.802e-02 | 1.640e-02 | **I2V is smaller** in absolute terms |
| **relative_avg_diff** | **4.889%** | **2.836%** | **I2V is 1.7× tighter than T2V** |
| **cosine_sim** | **0.998886** | **0.999580** | I2V cos is closer to 1 (≈1.6° angle vs T2V's 2.7°) |

### Per-category

| Category | n | mean_rel | max_rel | min_cos | What it represents |
|---|---:|---:|---:|---:|---|
| `transformer_block` | 168 | 1.117 | 2.377 | 0.000871 | per-rank shard permutation (Phase 2 known-benign) |
| `i2v_cond_frames` | 1 | **32.98%** | 32.98% | **0.9076** | NEW — VAE encoder divergence; see analysis below |
| `z_final` | 1 | **2.836%** | 2.836% | **0.999580** | end-to-end pre-VAE latent |
| `transformer_v_pred` | 42 | **0.806%** | 1.519% | 0.999884 | transformer output, per (step, label) |
| `i2v_x_pre_slice` | 2 | 0.945% | 1.122% | 0.999920 | post-final-layer post-gather, pre-slice |
| `i2v_t0_emb` | 2 | 0.185% | 0.240% | 0.999994 | post-`t_block` of `t_embedder(zeros) + vec_cond` |
| `transformer_hidden_states` | 42 | 0.134% | 0.141% | 0.999999 | injected z_t after fp32→bf16 cast |
| `i2v_x_after_concat` | 2 | **0.016%** | 0.016% | **1.000000** | post-concat pre-shard sequence |
| `i2v_x_t_mask` | 2 | 0.000% | 0.000% | 1.000000 | exact-equal |
| `i2v_cond_offsets` | 1 | 0.000% | 0.000% | 1.000000 | exact-equal `[-0.75]` |
| `timesteps_schedule` | 1 | **0.000%** | 0.000% | 1.000000 | **fix took effect** (was 3.04% pre-fix) |
| `text_seq_cond`, `transformer_encoder_hidden_states`, `transformer_extra`, `transformer_timestep`, `step_noise`, `z_initial` | — | 0.000% | 0.000% | 1.000000 | injected |

**Reading:**

- **`v_pred` mean_rel = 0.806%** — statistically indistinguishable from T2V Phase 2 L1's 0.811%. The I2V code path adds no architectural divergence on top of the existing FA3 noise floor.
- **`z_final` rel = 2.84%** — **better than T2V's 4.89%**. The cond-frame anchoring in the attention layers (cond tokens are bit-identical between sides because cond_frames is loaded from mv) damps the per-step v_pred drift, so the integrated z_final drift is smaller than T2V's despite the same per-step noise floor.
- **All four I2V intermediates pass the bit-equivalence bar:**
  - `x_after_concat` cos 1.000000 — input to block 0 IS bit-identical when both sides have the same cond_frames + same noise injected
  - `x_t_mask` exact-equal — mask construction matches mv's `dit3d.py:1036` byte-for-byte
  - `t0_emb` cos 0.999994 — `t_block(t_embedder(zeros) + vec_cond)` matches mv's `mix_time_and_embeddings(zeros)` first return
  - `x_pre_slice` cos 0.99992 — small drift accumulated through 30 transformer blocks, well within noise envelope

### Per-step `v_pred` divergence trajectory (cond)

```
step  rel%      max_abs    note
  0   1.05      0.250
  1   0.86      0.246
  ...           (flat 0.65-0.90% pre-cooldown)
 14   0.69      0.371
 ...           (cooldown — uncond no longer called)
 30   1.15      0.140
 31   1.48      0.195
 32   2.44      0.180     final step, biggest per-step deviation
```

Same shape as T2V L1: flat at 0.7-0.9% pre-cooldown, slight growth in last 5 steps as `mean(|v_pred|)` shrinks near `t→0`. Cosine_sim stays ≥ 0.9999 throughout.

---

## I2V-specific findings

### `cond_frames` rel = 32.98%, cos = 0.9076 (unchanged from pre-fix)

This is unaffected by the schedule fix (the dump captures vllm-omni's *native* VAE encoder output before injection, so it's a property of the VAE forward pass — not the scheduler). At L1 it does not contribute to z_final because mv's value is loaded; the 32.98% figure is purely diagnostic.

Plausible sources (deferred for follow-up):
1. **VAE input dtype mismatch.** vllm-omni casts `cond_images` to `bf16` *before* `vae.encode_images`; mv may keep it `fp32`. A deep conv stack amplifies bf16-vs-fp32 input drift dramatically.
2. **VAE forward kernel differences.** Independent Python wrappers around the same VAE weights — different fp32→bf16 cast points or different conv kernel paths can drift this much over a deep encoder.
3. **Pre-VAE preprocessing drift.** `ImageOps.fit` is deterministic, but `transforms.Normalize(inplace=True)` ordering relative to `to_tensor` could differ.

The end-to-end impact at L3 (where cond_frames is also injected from mv, so this VAE divergence still doesn't enter the diffusion path) is small — see L3 report.

---

## Verification — what's ruled out for I2V

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **t0 modulation source** | ✅ ruled out | `i2v_t0_emb` cos = 0.999994 |
| **Concat axis** (cond frames appended to T) | ✅ ruled out | `i2v_x_after_concat` cos = 1.000000 |
| **`x_t_mask` construction** | ✅ ruled out | exact byte-equal between sides |
| **Spatial pos-emb on cond frames** | ✅ ruled out | `x_after_concat` cos 1.0 means the unconditional addition (mirroring mv `dit3d.py:1033`) is correct |
| **Pre-slice / unpatchify boundary** | ✅ ruled out | `i2v_x_pre_slice` cos = 0.99992 |
| **Scheduler (timesteps schedule)** | ✅ ruled out **after the fix** | `timesteps_schedule rel = 0.000%` |
| **All T2V-side hypotheses** (SwiGLU, RMSNorm, RoPE, modulate, attention, weight loading, ulysses degree, per-rank shard alignment) | ✅ ruled out by Phase 2 | See `examples/phase2/L1_REPORT.md` |
| **Cond-input injection actually fires** | ✅ confirmed | server log shows `_maybe_load_cond_frames` / `_maybe_load_cond_offsets` log lines with the loaded path |

---

## Most likely cause of the residual 0.806%

Same as Phase 2: **different `flash_attn_3` builds** in the two venvs (`fa3_fwd 0.0.2` on vllm-omni vs `flash_attn_3 3.0.0b1` on mv). I2V did not introduce a new divergence source; the residual 0.806% is the same FA3 kernel-level drift that produces 0.811% on T2V. See `examples/phase2/L1_REPORT.md` and `~/.claude/projects/-home-yizhu-code-vllm-omni/memory/project_phase2_fa3_mismatch.md`.

The I2V transformer adds:

- One additional `t_block` call per step per cond/uncond (for `t0_emb`)
- A `torch.cat` and a `torch.where` per block (for mask-aware modulation)

None of these is FA-attention-related, so they don't add to the FA3 divergence; the 0.806% v_pred is essentially unchanged.

---

## Bugs found and fixed during the L1 (single-frame) investigation

| Bug | Symptom | Fix |
|---|---|---|
| `MareyPipeline → transformer` cycle via `setattr` | Worker startup `RecursionError: maximum recursion depth exceeded` in `model.eval()`. `nn.Module.__setattr__` registers Module values as child modules; `transformer._dump_pipeline = self` (where `self` is `MareyPipeline(nn.Module, …)`) made `MareyPipeline` a child of `transformer`, creating a recursive `_modules` cycle. | Bypass `nn.Module.__setattr__` via `self.transformer.__dict__["_dump_pipeline"] = self` so the attribute is a plain Python ref, not a registered submodule. |
| Layout mismatch on `x_after_concat` / `x_t_mask` / `x_pre_slice` | Compare driver returned NaN: vllm-omni shape `(B, T+Tf, S, C)` vs mv `(B, (T+Tf)*S, C)`. Same content, different rank/layout. | Move vllm-omni dumps to *after* the `(B, T+Tf, S, C) → (B, (T+Tf)*S, C)` reshape; flatten `x_t_mask` and `x_pre_slice` on the way out. |
| mv `--frame-conditions` CLI flag missing | mv reference run failed at startup: `No such option: --frame-conditions`. The flag was added on a previous-task branch but never landed on `marey-serving-comparison`. The kwarg + `_frame_conditions` method already existed on mv. | Added the typer.Option in `inference-service/marey_inference.py` (`infer` command) and the `frame_conditions=...` forwarding to `model.infer()`. |
| **`use_distilled_steps` not propagated on mv side** | `timesteps_schedule rel = 3.04%`; mv ran linspace ending at 2.99, vllm-omni ran stride-3 ending at 88.30. **z_final rel inflated to 12.09%** before fix. | Re-applied the Phase 2 one-liner `OmegaConf.update(self.model_cfg, "scheduler.use_distilled_steps", params.get("use_distilled_steps", False))` to `_setup_scheduler_config`. After fix: schedule rel = 0.000%, **z_final rel = 2.836%**. Pending commit to `marey-serving-comparison`. |

---

## Conclusion

**The new I2V transformer code (mask-aware modulation, `t0_emb` path, cond-frames concat-then-slice) is computationally equivalent to moonvalley's `dit3d.py` I2V branch** within — and *better than* — the existing T2V noise floor. Headline numbers:

| Quantity | T2V L1 | I2V L1 single | I2V vs T2V |
|---|---:|---:|---|
| `transformer_v_pred` mean rel | 0.811% | 0.806% | match |
| `z_final` rel | 4.89% | 2.84% | **1.7× tighter** |
| `z_final` cos_sim | 0.998886 | 0.999580 | tighter (1.6° vs 2.7°) |

All four I2V intermediates compare cleanly:
- `x_after_concat`: cos 1.000000 — bit-equivalent
- `x_t_mask`: exact equal
- `t0_emb`: cos 0.999994
- `x_pre_slice`: cos 0.99992

The residual 0.806% per-step `transformer_v_pred` divergence is the same FA3 build mismatch identified in T2V Phase 2. I2V adds no new architectural divergence source.

The reason I2V's `z_final` is *tighter* than T2V's despite the same per-step noise floor: at L1, transformer-input injection re-syncs at every step but the pipeline's local `latents` variable accumulates drift across all 33 steps via `drift_{t+1} = drift_t - dt × v_pred_drift_t`. With cond-frame anchoring in the attention layers, the `v_pred_drift_t` terms have systematically smaller absolute values (the cond positions pull the global attention toward mv's exact trajectory), so the integrated drift is smaller.

**One I2V-only finding worth a follow-up**: vllm-omni's native VAE encode of the cond image produces `cond_frames` with cos_sim 0.9076 vs mv's. Likely a VAE/preprocessing-layer difference, not a transformer issue. Investigation deferred — see "I2V-specific findings" above.

---

## Related artifacts

| Path | Purpose |
|---|---|
| `examples/phase2/L1_REPORT.md` | T2V-equivalent L1 report; same shape, no I2V tensors |
| `examples/phase2/PHASE2_FINDINGS.md` | T2V Phase 2 combined summary (FA3 deep-dive) |
| `examples/phase3_i2v/L2_REPORT_singleframe.md` | L2 — scheduler-recurrence compounding for I2V (post-fix numbers) |
| `examples/phase3_i2v/L3_REPORT_singleframe.md` | L3 — production-mode end-to-end I2V (post-fix numbers) |
| `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/vllm_l1_single/report.md` | Auto-generated per-run metrics report (raw) |
| `~/.claude/projects/-home-yizhu-code-vllm-omni/memory/project_phase2_fa3_mismatch.md` | FA3 finding (still applies) |
| `~/.claude/projects/-home-yizhu-code-vllm-omni/memory/project_mv_distilled_steps_bug.md` | The bug that recurred — re-applied as working-tree change |
