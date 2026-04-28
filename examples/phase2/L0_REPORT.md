# L0 — vllm-omni vs moonvalley parity, transformer-isolation mode

**Purpose:** strictest possible isolation of the **transformer forward math**. L1 already substitutes the transformer's `hidden_states` kwarg via a forward pre-hook, but the pipeline's local `latents` variable that flows into the inline DDPM update (`x0 = z - sigma * v_pred`) keeps drifting across all 33 steps via `drift_{t+1} = drift_t - dt × v_pred_drift_t` — so L1 `z_final` reflects *integrated* per-step transformer drift through the scheduler, not just one step's worth.

L0 fixes that: at the **start of each iteration**, before the transformer call, the pipeline's local `z` is overwritten with mv's `step<i>_cond_hidden_states.pt`. Both the transformer call AND the inline DDPM update then operate on mv's exact state, so only the LAST iteration's drift can propagate to `z_final`.

L0 answers: **what is the *true* per-step transformer noise floor when scheduler accumulation is removed?**

---

## Setup

Identical to L1 plus one new env var:

```
MAREY_LOAD_INITIAL_NOISE          = ref_30b/z_initial.pt
MAREY_LOAD_STEP_NOISE_DIR         = ref_30b/
MAREY_LOAD_TEXT_EMBEDS_DIR        = ref_30b/
MAREY_LOAD_TRANSFORMER_INPUTS_DIR = ref_30b/
MAREY_LOAD_PIPELINE_LATENTS_DIR   = ref_30b/   # the L0-defining one
```

Reproduce:
```bash
LEVEL=L0 bash examples/phase2/run_vllm_omni_dump.sh vllm_l0_t2v
```

Hook implementation: `vllm_omni/diffusion/debug/dump.py:_maybe_load_pipeline_latents` is called from `pipeline_marey.py` at the top of each denoising loop iteration. It loads `step{i}_cond_hidden_states.pt` for the current step `i` and overwrites `z` if the file exists and the shape matches. No-op when the env var is unset (any other tier).

Run dump: `/mnt/localdisk/vllm_omni_storage/phase2/vllm_l0_t2v/`.

---

## Results

### End-to-end

| Metric | L0 | L1 | L0 / L1 |
|---|---:|---:|---|
| `z_final` rel | **0.842%** | 4.889% | **5.8× tighter** |
| `z_final` cos_sim | **0.999973** | 0.998886 | 24× smaller angle (0.42° vs 2.7°) |
| `transformer_v_pred` mean rel | 0.811% | 0.811% | unchanged (transformer math unaffected) |
| `transformer_hidden_states` mean rel | 0.134% | 0.134% | unchanged (still injected, fp32→bf16 cast noise only) |

**Headline finding:** at L0, **`z_final` rel ≈ per-step `v_pred` rel** (0.84% vs 0.81%). This proves that L1's residual 4.89% z_final divergence was almost entirely cross-step accumulation through the scheduler's local `z` variable, *not* anything inside the transformer itself.

### Per-category

| Category | n | mean_rel | max_rel | min_cos | What it represents |
|---|---:|---:|---:|---:|---|
| `z_final` | 1 | **0.842%** | 0.842% | **0.999973** | end-to-end pre-VAE latent |
| `transformer_v_pred` | 42 | **0.811%** | 1.771% | 0.999844 | transformer output, per (step, label) |
| `transformer_hidden_states` | 42 | 0.134% | 0.143% | 0.999999 | injected z_t after fp32→bf16 cast (unchanged from L1) |
| `text_seq_cond` | 4 | 0.000% | 0.000% | 1.000000 | injected |
| `transformer_encoder_hidden_states` | 84 | 0.000% | 0.000% | 1.000000 | injected |
| `transformer_extra` | 336 | 0.000% | 0.000% | 1.000000 | injected |
| `transformer_timestep` | 42 | 0.000% | 0.000% | 1.000000 | injected |
| `step_noise` | 32 | 0.000% | 0.000% | 1.000000 | injected |
| `timesteps_schedule` | 1 | 0.000% | 0.000% | 1.000000 | distilled-steps fix in place |
| `z_initial` | 1 | 0.000% | 0.000% | 1.000000 | injected |

**Reading:**

- `transformer_v_pred` and `transformer_hidden_states` numbers are **byte-identical to L1** because the transformer math itself doesn't change between L0 and L1 (the pre-hook substitutes the transformer kwarg in both cases). Only the SCHEDULER path differs.
- `z_final` rel collapses from 4.89% → 0.84% — a 5.8× tightening that exactly matches the per-step `v_pred` rel. This is the pure last-step transformer drift, with zero cross-step amplification.

---

## Interpretation: the L1→L0 gap quantifies scheduler-recurrence amplification

For the flow-matching scheduler with the documented stride-3 distilled schedule:

```
drift_{t+1} = drift_t - dt_t × δ_t
            = -sum_{i=0..t} dt_i × δ_i
```

where δ_i = v_pred_i_omni - v_pred_i_mv at step i. The integrated sum over 33 steps amplifies a per-step ~0.005 abs drift into a final ~0.038 abs drift on z_final (the T2V L1 number). L0 truncates this sum to just the LAST step (`drift_final = -dt_32 × δ_32`), which is ~0.007 abs (the L0 number, slightly larger than v_pred_32 because of the cooldown step's larger sigma).

**Implications:**

1. **The entire L1 z_final divergence is scheduler-driven, not transformer-driven.** vllm-omni's transformer is computationally equivalent to mv's at the per-step level (cos 0.9998+ on every individual call, identical at L0 and L1).
2. **For pure transformer-correctness verification, L0 is the correct tier.** L1 conflates transformer math with scheduler recurrence in a way that overstates "transformer divergence".
3. **For end-to-end production parity assessment, L1 is still the right tier.** Production runs don't have per-step injection, so the scheduler-recurrence amplification IS the real-world divergence that downstream consumers see.

---

## Conclusion

**vllm-omni's Marey 30B T2V transformer is computationally equivalent to moonvalley_ai's reference implementation at the level of single-step forward pass**, with `z_final` cos 0.999973 (≈ 0.42° angle) when both are run on identical inputs throughout the entire denoising loop.

The residual ~0.81% per-step `v_pred` rel is the same FA3 build mismatch identified in Phase 2 (`fa3_fwd 0.0.2` on vllm-omni vs `flash_attn_3 3.0.0b1` on mv). It bounds the achievable parity given the current package set — L0 demonstrates that within that bound, the architecture is correctly implemented.

---

## Related artifacts

| Path | Purpose |
|---|---|
| `examples/phase2/L1_REPORT.md` | L1 — adds scheduler-recurrence amplification on top of L0 |
| `examples/phase2/L2_REPORT.md` | L2 — drops per-step transformer-input injection |
| `examples/phase2/L3_REPORT.md` | L3 — drops text-embed injection on top of L2 |
| `examples/phase2/PHASE2_FINDINGS.md` | Combined Phase 2 summary (FA3 deep-dive) |
| `examples/phase3_i2v/L0_REPORT_singleframe.md` | I2V-equivalent L0 result (z_final rel 0.713%) |
| `examples/phase3_i2v/L0_REPORT_multiframe.md` | I2V multi-keyframe L0 result (z_final rel 0.850%) |
| `/mnt/localdisk/vllm_omni_storage/phase2/vllm_l0_t2v/report.md` | Auto-generated per-run metrics |
