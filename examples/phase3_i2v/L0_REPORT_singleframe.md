# L0 (single-frame I2V) — vllm-omni vs moonvalley parity, transformer-isolation mode

**Purpose:** strictest possible isolation of the **I2V transformer forward math**. Identical to L1 except the pipeline's local `z` variable is also overwritten with mv's `step<i>_cond_hidden_states.pt` at the start of each denoising iteration. With both the transformer call AND the inline DDPM update operating on mv's exact state, only the LAST iteration's drift can propagate to `z_final`. See `examples/phase2/L0_REPORT.md` for the full L0 mechanism description.

L0 answers (for I2V): **what is the *true* per-step transformer noise floor for the new I2V code path when scheduler accumulation is removed?**

---

## Setup

Identical to single-frame L1 plus one new env var:

```
MAREY_LOAD_INITIAL_NOISE          = ref_single/z_initial.pt
MAREY_LOAD_STEP_NOISE_DIR         = ref_single/
MAREY_LOAD_TEXT_EMBEDS_DIR        = ref_single/
MAREY_LOAD_TRANSFORMER_INPUTS_DIR = ref_single/
MAREY_LOAD_COND_FRAMES_PATH       = ref_single/cond_frames.pt
MAREY_LOAD_COND_OFFSETS_PATH      = ref_single/cond_offsets.pt
MAREY_LOAD_PIPELINE_LATENTS_DIR   = ref_single/   # the L0-defining one
```

Reproduce:
```bash
COND_IMAGES=/app/yizhu/marey/vllm_omni/vllm_omni_storage/cond_frame_0.webp \
FRAME_INDICES=0 SEED=42 LEVEL=L0 \
bash examples/phase3_i2v/run_vllm_omni_dump.sh vllm_l0_single
```

Run dump: `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/vllm_l0_single`.

---

## Results

### End-to-end

| Metric | L0 (single) | L1 (single) | L0 / L1 |
|---|---:|---:|---|
| `z_final` rel | **0.713%** | 2.836% | **4.0× tighter** |
| `z_final` cos_sim | **0.999978** | 0.999580 | 4× smaller angle (0.38° vs 1.65°) |
| `transformer_v_pred` mean rel | 0.806% | 0.806% | unchanged |
| `transformer_hidden_states` mean rel | 0.134% | 0.134% | unchanged |
| All four step-0 I2V intermediates | unchanged | unchanged | identical to L1 |

**Headline finding:** at L0, **`z_final` rel = 0.713%** which is *below* the per-step `v_pred` rel (0.806%) — confirming that for I2V single-frame, the scheduler accumulation in L1 was the *only* meaningful driver of z_final divergence. The transformer code itself drifts at ~0.8% per step, and a single-step's contribution to z_final is now what dominates, slightly damped by the cond-frame anchoring across the (T+Tf)*S sequence at the last step.

### Per-category

| Category | n | mean_rel | max_rel | min_cos | What it represents |
|---|---:|---:|---:|---:|---|
| `transformer_block` | 168 | 1.117 | 2.377 | 0.000871 | per-rank shard permutation (benign) |
| `i2v_cond_frames` | 1 | 32.98% | 32.98% | 0.9076 | VAE divergence (still injected, doesn't enter z_final) |
| `transformer_v_pred` | 42 | **0.806%** | 1.519% | 0.999884 | transformer output (unchanged from L1) |
| `i2v_x_pre_slice` | 2 | 0.945% | 1.122% | 0.999920 | step 0 — same as L1 |
| `z_final` | 1 | **0.713%** | 0.713% | **0.999978** | end-to-end |
| `i2v_t0_emb` | 2 | 0.185% | 0.240% | 0.999994 | step 0 — same as L1 |
| `transformer_hidden_states` | 42 | 0.134% | 0.141% | 0.999999 | injected (unchanged from L1) |
| `i2v_x_after_concat` | 2 | 0.016% | 0.016% | 1.000000 | step 0 — bit-equivalent (same as L1) |
| `i2v_x_t_mask` | 2 | 0.000% | 0.000% | 1.000000 | exact-equal (same as L1) |
| `i2v_cond_offsets` | 1 | 0.000% | 0.000% | 1.000000 | exact-equal |
| `timesteps_schedule` | 1 | 0.000% | 0.000% | 1.000000 | distilled-steps fix in place |
| `text_seq_cond`, `transformer_encoder_hidden_states`, `transformer_extra`, `transformer_timestep`, `step_noise`, `z_initial` | — | 0.000% | 0.000% | 1.000000 | injected |

**Reading:**

- All four step-0 I2V intermediates are **byte-identical to L1** (cos 1.0 / 0.999994 / 0.99992; mask exact). The transformer's I2V branch is invariant across L0/L1/L2/L3 because step-0 inputs are always the same (loaded `z_initial`, loaded `cond_frames`, etc.).
- `transformer_v_pred` and `transformer_hidden_states` are also byte-identical to L1 — the L0 hook only affects `z` between iterations, not within a single transformer call.
- `z_final` collapses from 2.84% → 0.71% — exactly as predicted from "isolate single-step transformer drift". The 0.71% is essentially a slightly-damped version of the last step's v_pred drift (1.52% at step 32 ÷ scheduler amplification factor).

### Why I2V L0 is *tighter* than T2V L0

T2V L0: z_final rel 0.842%
I2V single L0: z_final rel **0.713%**

At L0, both runs reduce to "last-step transformer drift × scheduler amplification factor". For I2V the cond-frame anchoring (cond positions are bit-identical to mv via `MAREY_LOAD_COND_FRAMES_PATH`) damps the last-step v_pred drift through cross-attention, so the propagated z_final drift is smaller. T2V has no such anchor, so its full last-step drift propagates.

---

## Conclusion

**vllm-omni's I2V single-frame transformer is computationally equivalent to moonvalley_ai's `dit3d.py` I2V branch at the level of single-step forward pass.** End-to-end at L0, vllm-omni's `z_final` is at cos 0.999978 (≈ 0.38° angle from mv's reference) — *tighter* than T2V's L0 result (cos 0.999973). The cond-frame anchoring damps last-step drift in addition to the multi-step damping seen at L1/L2/L3.

The residual 0.806% per-step `v_pred` rel is the same FA3 build mismatch from T2V Phase 2; the I2V code path adds zero detectable divergence on top of it.

---

## Related artifacts

| Path | Purpose |
|---|---|
| `examples/phase2/L0_REPORT.md` | T2V L0 — same mechanism, no I2V tensors (z_final rel 0.842%) |
| `examples/phase3_i2v/L0_REPORT_multiframe.md` | I2V multi-keyframe L0 (z_final rel 0.850%) |
| `examples/phase3_i2v/L1_REPORT_singleframe.md` | L1 — adds scheduler-recurrence amplification on top of L0 |
| `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/vllm_l0_single/report.md` | Auto-generated per-run metrics |
