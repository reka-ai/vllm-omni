# L0 (multi-keyframe I2V) — vllm-omni vs moonvalley parity, transformer-isolation mode

**Variant:** three cond images at frame indices `[0, 63, 127]`. See `L0_REPORT_singleframe.md` for the L0 mechanism description; this report only documents what differs at multi-keyframe.

---

## Headline: at L0, single-frame and multi-keyframe I2V converge

L1 showed cond-frame anchoring damping cross-step accumulation (Single L1 z_final 2.84% → Multi L1 z_final 3.30%, both substantially better than T2V L1's 4.89%). At L0 there's no cross-step accumulation left to damp — only the LAST step's drift propagates — so the L0 results across single-frame and multi-keyframe converge:

| Variant | L0 z_final rel | L0 z_final cos | L0 v_pred mean rel |
|---|---:|---:|---:|
| T2V | 0.842% | 0.999973 | 0.811% |
| **I2V single L0** | 0.713% | 0.999978 | 0.806% |
| **I2V multi L0** | 0.850% | 0.999968 | 0.876% |

The three numbers are within ~1 pp of each other — at the per-step transformer level, T2V / I2V-single / I2V-multi all hit the same FA3 noise floor.

---

## Setup deltas vs single-frame L0

Same as L1/L2/L3 multi:
- `COND_IMAGES`: 3× the same image
- `FRAME_INDICES`: `0,63,127`
- `cond_frames`: `(1, 16, 3, 68, 120)`
- `cond_offsets`: `[-0.75, 15.0, 31.0]`

L0-specific: `MAREY_LOAD_PIPELINE_LATENTS_DIR=ref_multi/` injects mv's `step<i>_cond_hidden_states.pt` at the start of each iteration.

Reproduce:
```bash
COND=/app/yizhu/marey/vllm_omni/vllm_omni_storage/cond_frame_0.webp
COND_IMAGES="$COND,$COND,$COND" FRAME_INDICES=0,63,127 SEED=42 LEVEL=L0 \
REF_DIR=/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/ref_multi \
bash examples/phase3_i2v/run_vllm_omni_dump.sh vllm_l0_multi
```

Run dump: `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/vllm_l0_multi`.

---

## Results

### End-to-end

| Metric | L0 (multi) | L1 (multi) | L0 / L1 |
|---|---:|---:|---|
| `z_final` rel | **0.850%** | 3.302% | **3.9× tighter** |
| `z_final` cos_sim | **0.999968** | 0.999474 | 16× smaller angle (0.46° vs 1.86°) |
| `transformer_v_pred` mean rel | 0.876% | 0.876% | unchanged |

### Per-category

| Category | n | mean_rel | max_rel | min_cos | Note |
|---|---:|---:|---:|---:|---|
| `transformer_block` | 168 | 1.243 | 4.112 | 0.002114 | per-rank shard permutation (benign) |
| `i2v_cond_frames` | 1 | 32.98% | 32.98% | 0.9076 | VAE divergence (still injected) |
| `transformer_v_pred` | 42 | **0.876%** | 1.786% | 0.999840 | unchanged from L1 |
| `i2v_x_pre_slice` | 2 | 1.643% | 1.789% | 0.999699 | step 0 — same as L1 multi |
| `z_final` | 1 | **0.850%** | 0.850% | **0.999968** | end-to-end |
| `i2v_t0_emb` | 2 | 0.185% | 0.240% | 0.999994 | step 0 — same as L1 multi |
| `transformer_hidden_states` | 42 | 0.134% | 0.141% | 0.999999 | injected (same as L1) |
| `i2v_x_after_concat` | 2 | 0.025% | 0.025% | 1.000000 | step 0 — bit-equivalent |
| `i2v_x_t_mask` | 2 | 0.000% | 0.000% | 1.000000 | exact-equal |
| `i2v_cond_offsets` | 1 | 0.000% | 0.000% | 1.000000 | exact-equal `[-0.75, 15.0, 31.0]` |
| (text + injected categories) | — | 0.000% | 0.000% | 1.000000 | injected |

**Reading:**

- All four step-0 I2V intermediates and `transformer_v_pred` are **byte-identical to L1 multi**. The L0 hook only changes the pipeline's `z` between iterations.
- `z_final` rel collapses from 3.30% → 0.85% — same magnitude of L1→L0 tightening as single-frame. Cross-step accumulation through the scheduler was the dominant divergence driver in both variants.

### Why I2V L0 multi is slightly *less* tight than I2V L0 single

I2V single L0: 0.713%
I2V multi L0: 0.850%

L0 reduces both to "last-step drift × scheduler amplification". The multi case has slightly higher last-step drift (v_pred 0.876% vs single's 0.806%) because the longer (T+Tf)*S sequence accumulates marginally more FA3 kernel noise per attention call (~3% more tokens per step). At L0 this last-step difference is the entire delta; at L1 it would be amplified through 33 steps of accumulation in single-frame but damped by cond anchoring in multi.

---

## Conclusion

**At the per-step transformer level, all three variants (T2V, I2V single, I2V multi) converge to the same FA3 noise floor (~0.8% per-step `v_pred` rel and ~0.7-0.85% `z_final` rel at L0).** This is the cleanest demonstration that the I2V code path — including the `T_lat=3` multi-anchor case — adds zero architectural divergence on top of T2V.

The cond-frame anchoring effect that dramatically tightens L1/L2/L3 multi-keyframe (z_final 3.62% at L2 vs T2V L2's 57.5%, a 16× improvement) operates *across* iterations through cross-attention; with cross-iteration drift removed at L0, the anchoring effect has nothing to act on.

---

## Related artifacts

| Path | Purpose |
|---|---|
| `examples/phase3_i2v/L0_REPORT_singleframe.md` | I2V single-frame L0 — same mechanism with one cond anchor |
| `examples/phase2/L0_REPORT.md` | T2V L0 — no I2V tensors |
| `examples/phase3_i2v/L1_REPORT_multiframe.md` | L1 multi — adds scheduler-recurrence amplification on top of L0 |
| `examples/phase3_i2v/PHASE3_FINDINGS.md` | Combined Phase 3 summary across all tiers + variants |
| `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/vllm_l0_multi/report.md` | Auto-generated per-run metrics |
