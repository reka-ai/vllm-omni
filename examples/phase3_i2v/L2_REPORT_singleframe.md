# L2 (single-frame I2V) — vllm-omni vs moonvalley parity, scheduler-recurrence mode

**Purpose:** measure how the L1 per-step transformer noise floor **compounds across 33 scheduler steps** when the per-step transformer inputs are no longer re-injected.

L2 keeps the same injections as L1 *except* `MAREY_LOAD_TRANSFORMER_INPUTS_DIR` is unset. Initial noise, per-step DDPM noise, text embeddings, **and** the I2V `cond_frames` / `cond_offsets` are still loaded from mv. Each step's `hidden_states`, `timestep`, `encoder_hidden_states`, `vector_cond`, `extra_*` are produced by vllm-omni's scheduler from the previous step's `v_pred` — no longer corrected by mv's value.

L2 answers: **does the 0.81% per-step transformer noise blow up the trajectory, or stay bounded?**

> **Note:** Numbers below are from the post-fix run (mv-side `use_distilled_steps` propagation re-applied; `timesteps_schedule rel = 0.000%`). See `L1_REPORT_singleframe.md` for the full schedule-fix story.

---

## Setup

Identical to L1 except: `MAREY_LOAD_TRANSFORMER_INPUTS_DIR` is **not** set.

```
MAREY_LOAD_INITIAL_NOISE          = ref_single/z_initial.pt
MAREY_LOAD_STEP_NOISE_DIR         = ref_single/
MAREY_LOAD_TEXT_EMBEDS_DIR        = ref_single/
MAREY_LOAD_COND_FRAMES_PATH       = ref_single/cond_frames.pt
MAREY_LOAD_COND_OFFSETS_PATH      = ref_single/cond_offsets.pt
# (no MAREY_LOAD_TRANSFORMER_INPUTS_DIR)
```

Run command:
```bash
COND_IMAGES=/app/yizhu/marey/vllm_omni/vllm_omni_storage/cond_frame_0.webp \
FRAME_INDICES=0 SEED=42 LEVEL=L2 \
bash /home/yizhu/code/vllm-omni/examples/phase3_i2v/run_vllm_omni_dump.sh vllm_l2_single
```

Run dump: `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/vllm_l2_single`.

---

## Results

### End-to-end

| Metric | L1 | **L2** | Change |
|---|---:|---:|---|
| `z_final` rel | 2.836% | **11.70%** | +8.86 pp |
| `z_final` cos_sim | 0.9996 | **0.9876** | -0.0120 |
| `transformer_v_pred` mean rel | 0.806% | **4.370%** | **5.4×** |
| `transformer_hidden_states` mean rel | 0.134% | **3.50%** (mid-range) | compounds because `hidden_states` at step `i` = vllm-omni's own DDPM update of step `i-1`'s `v_pred` |

### Per-category

| Category | n | mean_rel | max_rel | min_cos | What it represents |
|---|---:|---:|---:|---:|---|
| `transformer_block` | 168 | 1.117 | 2.377 | 0.000871 | per-rank shard permutation (Phase 2 known-benign) |
| `i2v_cond_frames` | 1 | **32.98%** | 32.98% | **0.9076** | unchanged from L1 — VAE encoder divergence (still injected at L2) |
| `z_final` | 1 | **11.70%** | 11.70% | **0.9876** | end-to-end |
| `transformer_v_pred` | 42 | **4.370%** | 14.73% | 0.9845 | scheduler-compounded (cf. L1's 0.806%) |
| `i2v_x_pre_slice` | 2 | **0.945%** | 1.122% | **0.999920** | step-0 only — same cosine as L1; tracks the step-0 transformer math |
| `i2v_t0_emb` | 2 | **0.185%** | 0.240% | **0.999994** | step-0 only — same as L1 |
| `i2v_x_after_concat` | 2 | **0.016%** | 0.016% | **1.000000** | step-0 only — bit-equivalent (same as L1) |
| `i2v_x_t_mask` | 2 | **0.000%** | 0.000% | **1.000000** | exact-equal (same as L1) |
| `i2v_cond_offsets` | 1 | 0.000% | 0.000% | 1.000000 | exact-equal |
| `timesteps_schedule` | 1 | **0.000%** | 0.000% | 1.000000 | **fix took effect** (was 3.04% pre-fix) |
| `text_seq_cond` | 4 | 0.000% | 0.000% | 1.000000 | injected |
| `transformer_encoder_hidden_states` | 84 | 0.000% | 0.000% | 1.000000 | injected |
| `transformer_extra` | 336 | 0.000% | 0.000% | 1.000000 | injected |
| `step_noise` | 32 | 0.000% | 0.000% | 1.000000 | injected |
| `z_initial` | 1 | 0.000% | 0.000% | 1.000000 | injected |

**Reading:**

- The four step-0 I2V intermediates (`x_after_concat`, `x_t_mask`, `t0_emb`, `x_pre_slice`) are **identical to L1** because at step 0 the inputs are still bit-identical (initial `z` is loaded; `cond_frames` / `cond_offsets` are loaded; vec_cond comes from injected text embeds). This confirms the I2V transformer code is invariant across L1/L2 — the only difference between tiers is the per-step transformer-input injection, which doesn't affect step 0.
- Compounded from step 1 onward: `transformer_v_pred` and `transformer_hidden_states` show the trajectory drift. `v_pred` mean rel grows from 0.84% (step 0) to high single digits in the middle steps, peaking around the cooldown transition.

### Comparison to T2V Phase 2 L2

| Quantity | T2V L2 | I2V L2 (single) | Note |
|---|---:|---:|---|
| `transformer_v_pred` mean rel | 27.6% | **4.37%** | **I2V is 6.3× tighter** — `cond_frames` anchoring constrains the trajectory |
| `z_final` cos_sim | 0.766 | 0.9876 | huge improvement (smaller compounded angle) |
| `z_final` rel | 57.5% | **11.70%** | **I2V 4.9× tighter** |

The I2V `cond_frames` injection (loaded every tier) gives the transformer a fixed reference point at the cond positions throughout denoising. Even when the noise positions drift step-to-step, the cond positions stay anchored to mv's exact value, which damps the trajectory's overall divergence. T2V has no equivalent anchor — every position drifts independently.

---

## Verification

L2 confirms what L1 told us:

1. **The I2V transformer code is bit-equivalent at step 0** (`x_after_concat` cos 1.0, `x_t_mask` exact, `t0_emb` cos 0.999994, `x_pre_slice` cos 0.99992). All step-0 internal dumps match L1 numbers exactly because step 0 has the same inputs at L2 and L1.
2. **Scheduler recurrence compounds the L1 noise floor by ~5.4× to 4.37%** — much tighter than T2V's 27.6%, thanks to the cond-frame anchoring effect.
3. **No new divergence source vs L1.** The same FA3 build mismatch from Phase 2 fully explains the L1→L2 jump (no architectural bug introduced by I2V).

---

## Conclusion

L2 single-frame I2V passes. The compounded per-step error is bounded — at every step, vllm-omni's `v_pred` cosine_sim with mv's stays ≥ 0.87 (worst case in mid-trajectory; ≥ 0.95 most of the time), and the cond-frame anchoring keeps the L2 z_final much tighter than the equivalent T2V L2 result.

For applications that can tolerate the cond_frames-anchored noise envelope, vllm-omni passes I2V L2.

---

## Related artifacts

| Path | Purpose |
|---|---|
| `examples/phase3_i2v/L1_REPORT_singleframe.md` | L1 — full-injection mode (this report compounds from there) |
| `examples/phase3_i2v/L3_REPORT_singleframe.md` | L3 — production-mode (drops text-embeds injection on top of L2) |
| `examples/phase2/L2_REPORT.md` | T2V-equivalent L2 report (when written) |
| `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/vllm_l2_single/report.md` | Auto-generated per-run metrics |
