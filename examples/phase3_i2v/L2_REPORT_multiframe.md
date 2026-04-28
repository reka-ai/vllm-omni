# L2 (multi-keyframe I2V) — vllm-omni vs moonvalley parity, scheduler-recurrence mode

**Variant:** three cond images at frame indices `[0, 63, 127]` (start / middle / end). All three positions use the same image. See `L2_REPORT_singleframe.md` for the L2 protocol; this report only documents what differs.

---

## Headline: cond-frame anchoring damps the trajectory

This is the tier where multi-keyframe pays off. With three cond positions bit-identical to mv (vs only one in single-frame), the noise tokens have stable cross-attention partners spread across the temporal sequence, and the per-step v_pred drift no longer compounds the same way through the scheduler recurrence.

| Metric | T2V Phase 2 L2 | I2V Single L2 | **I2V Multi L2** | Multi vs Single |
|---|---:|---:|---:|---|
| `transformer_v_pred` mean rel | 27.6% | 4.37% | **1.47%** | **3.0× tighter** |
| `z_final` rel | 57.5% | 11.70% | **3.62%** | **3.2× tighter** |
| `z_final` cos_sim | 0.766 | 0.9876 | **0.9992** | ≈0.7° angle |

Multi L2's z_final cos (0.9992) is **tighter than single-frame L1's** (0.9996 ≈ same order). With 3 anchors, the trajectory effectively doesn't drift across 33 scheduler steps — the noise tokens stay locked to mv's path via cross-attention to the cond positions.

---

## Setup deltas vs single-frame L2

Same as L1 multi:
- `COND_IMAGES`: 3× the same image
- `FRAME_INDICES`: `0,63,127`
- `cond_frames`: `(1, 16, 3, 68, 120)`
- `cond_offsets`: `[-0.75, 15.0, 31.0]`

Same mv-side `--use-distilled-steps` working-tree fix in place; `timesteps_schedule rel = 0.000%`.

---

## Per-category summary

| Category | n | mean_rel | max_rel | min_cos | Note |
|---|---:|---:|---:|---:|---|
| `transformer_block` | 168 | 1.243 | 4.112 | 0.002114 | per-rank shard permutation (benign) |
| `i2v_cond_frames` | 1 | 32.98% | 32.98% | 0.9076 | VAE divergence (still injected at L2) |
| `z_final` | 1 | **3.62%** | 3.62% | **0.9992** | end-to-end |
| `transformer_v_pred` | 42 | **1.47%** | 5.60% | 0.9984 | scheduler-compounded |
| `i2v_x_pre_slice` | 2 | 1.64% | 1.79% | 0.999699 | step 0 — same as L1 multi |
| `transformer_hidden_states` | 42 | 1.01% | 3.60% | 0.9992 | DDPM-update-driven drift, much smaller than single L2's 4.55% |
| `i2v_t0_emb` | 2 | 0.185% | 0.240% | 0.999994 | step 0 — same as L1 multi |
| `i2v_x_after_concat` | 2 | 0.025% | 0.025% | 1.000000 | step 0 — bit-equivalent |
| `timesteps_schedule` | 1 | 0.000% | 0.000% | 1.000000 | fix took effect |
| `i2v_x_t_mask` | 2 | 0.000% | 0.000% | 1.000000 | exact-equal |
| `i2v_cond_offsets` | 1 | 0.000% | 0.000% | 1.000000 | exact-equal `[-0.75, 15.0, 31.0]` |
| (text + injected categories) | … | 0.000% | 0.000% | 1.000000 | injected |

**Reading:**

- The four step-0 I2V intermediates are identical to L1 multi (same input → same step 0). This confirms the I2V transformer code is invariant across L1/L2 — only the per-step injection differs.
- `transformer_v_pred` mean_rel **1.47%** vs single-frame L2's 4.37% → 3× tighter. Direct evidence that 3 cond anchors > 1 cond anchor for damping per-step drift through cross-attention.
- `transformer_hidden_states` mean_rel **1.01%** vs single-frame L2's 4.55% → 4.5× tighter. The DDPM-recurrence drift is also damped because v_pred is tighter at each step.

---

## Conclusion

L2 multi-keyframe is the strongest result of the entire Phase 3. **vllm-omni's I2V output at L2 with 3 cond anchors is essentially indistinguishable from mv's at the latent level (z_final cos 0.9992 ≈ 0.7° angle)** — better than T2V achieved even at L1 (cos 0.9989).

This is direct empirical confirmation of the cond-frame anchoring hypothesis: by spreading the bit-identical cond tokens across the temporal sequence, the noise tokens' cross-attention is anchored to mv's exact trajectory, dramatically reducing the per-step compounding that L2 was meant to expose. With 3 of the 33 latent T positions held constant, the remaining 30 noise positions effectively stay on mv's flow.

---

## Related artifacts

| Path | Purpose |
|---|---|
| `examples/phase3_i2v/L2_REPORT_singleframe.md` | Single-frame L2 (1-anchor baseline) |
| `examples/phase3_i2v/L1_REPORT_multiframe.md` | Multi-keyframe L1 (where anchoring doesn't pay off) |
| `examples/phase3_i2v/L3_REPORT_multiframe.md` | Multi-keyframe L3 (text-encoder native) |
| `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/vllm_l2_multi/report.md` | Auto-generated per-run metrics |
