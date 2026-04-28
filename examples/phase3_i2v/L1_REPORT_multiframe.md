# L1 (multi-keyframe I2V) — vllm-omni vs moonvalley parity, full-injection mode

**Variant:** three cond images at frame indices `[0, 63, 127]` (start / middle / end of the 128-frame sequence). All three positions use the same image (`cond_frame_0.webp`) for self-consistency checks. See `L1_REPORT_singleframe.md` for the L1 protocol; this report only documents what differs at multi-keyframe.

---

## Setup deltas vs single-frame

| | Single | **Multi** |
|---|---|---|
| `COND_IMAGES` | `cond_frame_0.webp` | `cond_frame_0.webp,cond_frame_0.webp,cond_frame_0.webp` |
| `FRAME_INDICES` | `0` | `0,63,127` |
| `cond_frames` shape | `(1, 16, 1, 68, 120)` (T_lat=1) | `(1, 16, 3, 68, 120)` (T_lat=3) |
| `cond_offsets` | `[-0.75]` | `[-0.75, 15.0, 31.0]` |
| Tokens per step | T*S = 32×2040 = 65280 | (T+Tf)*S = 33×2040 = **67320** (≈3% more) |

Same mv-side `--use-distilled-steps` working-tree fix in place; `timesteps_schedule rel = 0.000%`.

---

## Results

### End-to-end

| Metric | Single L1 | **Multi L1** | Delta |
|---|---:|---:|---|
| `z_final` rel | 2.836% | **3.302%** | +0.47 pp |
| `z_final` cos_sim | 0.999580 | **0.999474** | -0.0001 |
| `transformer_v_pred` mean rel | 0.806% | **0.876%** | +0.07 pp |

The L1 deltas are small. At L1, transformer-input injection re-syncs every step, so cond_frames anchoring has nothing to damp; the only difference is +3% more tokens per attention call → marginally more accumulated FA3 noise per step. This was predicted before the run.

### I2V intermediates (step 0)

| Tensor | Single L1 | Multi L1 | Note |
|---|---:|---:|---|
| `i2v_x_after_concat` | cos 1.000000 | cos 1.000000 | bit-equivalent |
| `i2v_x_t_mask` | exact-equal | exact-equal | shape now `(1, 67320, 1)` (was `(1, 65280, 1)`) |
| `i2v_t0_emb` | cos 0.999994 | cos 0.999994 | unchanged (depends only on timestep + vec_cond) |
| `i2v_x_pre_slice` | cos 0.999920 | cos 0.999699 | small drop from longer sequence (more block-loop FA3 accumulation) |
| `i2v_cond_offsets` | `[-0.75]` exact | `[-0.75, 15.0, 31.0]` exact | three latent-time offsets all match |
| `i2v_cond_frames` | cos 0.9076 | cos 0.9076 | same VAE divergence finding (still injected, doesn't enter z_final) |

### Self-consistency check (multi-only)

Because the same input image was passed at all three positions, the three `cond_frames[:, :, i]` slices on each side should be bit-identical to each other. Verified before the run:

```
mv ref multi: cond_frames[:, :, 0] vs [:, :, 1] equal: True
              cond_frames[:, :, 0] vs [:, :, 2] equal: True
```

VAE encode is deterministic on identical input. ✓

### Per-category summary

| Category | n | mean_rel | max_rel | min_cos |
|---|---:|---:|---:|---:|
| `transformer_block` | 168 | 1.243 | 4.112 | 0.002114 | (per-rank shard permutation; benign) |
| `i2v_cond_frames` | 1 | 32.98% | 32.98% | 0.9076 | (VAE divergence — see L1 single report) |
| `z_final` | 1 | 3.30% | 3.30% | 0.999474 | end-to-end |
| `timesteps_schedule` | 1 | 0.000% | 0.000% | 1.000000 | fix took effect |
| `i2v_x_pre_slice` | 2 | 1.64% | 1.79% | 0.999699 | post-final-layer post-gather |
| `transformer_v_pred` | 42 | 0.876% | 1.786% | 0.999840 | per-step |
| `i2v_t0_emb` | 2 | 0.185% | 0.240% | 0.999994 | step 0 |
| `transformer_hidden_states` | 42 | 0.134% | 0.141% | 0.999999 | injected |
| `i2v_x_after_concat` | 2 | 0.025% | 0.025% | 1.000000 | step 0, bit-equivalent |
| `i2v_x_t_mask` | 2 | 0.000% | 0.000% | 1.000000 | exact-equal |
| `i2v_cond_offsets` | 1 | 0.000% | 0.000% | 1.000000 | exact-equal |
| (text + injected categories) | … | 0.000% | 0.000% | 1.000000 | injected |

---

## Conclusion

L1 multi-keyframe passes. The I2V transformer code path handles `T_lat=3` cond_frames identically to `T_lat=1` (cos 1.0 / 0.999994 / exact-equal across the four step-0 intermediates). The slight v_pred and z_final increase (~0.07 pp / 0.47 pp) is consistent with the ~3% more tokens per step adding marginal FA3 kernel noise.

The cond-anchoring damping effect predicted in the plan **does not show up at L1**, exactly as predicted in advance: with per-step transformer-input injection, the cond_frames are just additional bit-identical tokens with nowhere for the anchoring effect to act. L2/L3 are where multi-keyframe pays off — see `L2_REPORT_multiframe.md` and `L3_REPORT_multiframe.md`.

---

## Related artifacts

| Path | Purpose |
|---|---|
| `examples/phase3_i2v/L1_REPORT_singleframe.md` | Single-frame L1 (full setup + analysis) |
| `examples/phase3_i2v/L2_REPORT_multiframe.md` | Multi-keyframe L2 (where anchoring pays off) |
| `examples/phase3_i2v/L3_REPORT_multiframe.md` | Multi-keyframe L3 |
| `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/vllm_l1_multi/report.md` | Auto-generated per-run metrics |
