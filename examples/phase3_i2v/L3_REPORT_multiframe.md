# L3 (multi-keyframe I2V) — vllm-omni vs moonvalley parity, production mode

**Variant:** three cond images at frame indices `[0, 63, 127]` (start / middle / end). All three positions use the same image. See `L3_REPORT_singleframe.md` for the L3 protocol; this report only documents what differs.

---

## Headline: production-mode I2V output stays at L2-multi tightness

L3 drops the text-embeds injection on top of L2 (vllm-omni runs its own UL2/CLIP/ByT5). For the multi-keyframe variant, this adds a tiny amount on top of L2's already-very-tight numbers.

| Metric | T2V Phase 2 L3 | I2V Single L3 | **I2V Multi L3** | Multi vs Single |
|---|---:|---:|---:|---|
| `transformer_v_pred` mean rel | 27.8% | 4.65% | **1.68%** | **2.8× tighter** |
| `z_final` rel | 57.8% | 11.74% | **3.76%** | **3.1× tighter** |
| `z_final` cos_sim | 0.765 | 0.9871 | **0.9992** | ~0.7° angle |
| L2→L3 v_pred delta | +0.2 pp | +0.28 pp | +0.21 pp | small text-encoder add |

L3 multi z_final is essentially equal to L2 multi (3.76% vs 3.62%). The text-encoder native drift adds +0.21 pp on v_pred and +0.14 pp on z_final — same finding as T2V Phase 2 ("text encoder adds <1% over L2"), unchanged for I2V.

---

## Setup deltas vs single-frame L3

Same as L1/L2 multi:
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
| `i2v_cond_frames` | 1 | 32.98% | 32.98% | 0.9076 | VAE divergence (still injected) |
| `z_final` | 1 | **3.76%** | 3.76% | **0.9992** | end-to-end |
| `transformer_v_pred` | 42 | **1.68%** | 5.66% | 0.9983 | text-encoder drift adds ~0.21 pp on top of L2 |
| `i2v_x_pre_slice` | 2 | 1.75% | 1.81% | 0.999690 | step 0 — slightly different from L1/L2 due to native text-encoder mixing |
| `transformer_hidden_states` | 42 | 1.06% | 3.74% | 0.9992 | DDPM-update-driven drift, much smaller than single L3's 4.56% |
| `text_seq_cond` | 4 | **0.871%** | 1.847% | 0.999635 | newly native (UL2/ByT5 forward drift) |
| `transformer_encoder_hidden_states` | 84 | 0.841% | 1.847% | 0.999635 | mirrors `text_seq_cond` per-step |
| `i2v_t0_emb` | 2 | 0.193% | 0.246% | 0.999994 | step 0 — barely affected by text-encoder native mode |
| `i2v_x_after_concat` | 2 | 0.025% | 0.025% | 1.000000 | step 0 — bit-equivalent |
| `timesteps_schedule` | 1 | 0.000% | 0.000% | 1.000000 | fix took effect |
| `i2v_x_t_mask` | 2 | 0.000% | 0.000% | 1.000000 | exact-equal |
| `i2v_cond_offsets` | 1 | 0.000% | 0.000% | 1.000000 | exact-equal `[-0.75, 15.0, 31.0]` |
| (other injected categories) | … | 0.000% | 0.000% | 1.000000 | injected |

---

## Conclusion

L3 multi-keyframe passes. End-to-end production-mode I2V output sits at **`z_final` cos 0.9992 / rel 3.76%** vs mv's reference — **15× tighter than T2V Phase 2 L3** (rel 57.8%, cos 0.765).

The text-encoder native drift (0.87% on `text_seq_cond`) is the same finding as T2V Phase 2 — invariant of T2V vs I2V, invariant of single vs multi. It's a transformers/torch version mismatch (cross-session memory `project_phase2_text_encoder_drift.md`), not an I2V-specific issue.

The multi-keyframe L1/L2/L3 numbers cluster tightly:

| Tier | z_final rel | z_final cos |
|---|---:|---:|
| L1 multi | 3.30% | 0.999474 |
| L2 multi | 3.62% | 0.999222 |
| L3 multi | 3.76% | 0.999176 |

→ For multi-keyframe I2V, L1/L2/L3 are all within ~0.5 pp of each other on z_final. The cond-frame anchoring effect dominates the per-step compounding so thoroughly that the L2→L3 transition (text encoder native vs injected) barely moves the needle.

---

## Related artifacts

| Path | Purpose |
|---|---|
| `examples/phase3_i2v/L3_REPORT_singleframe.md` | Single-frame L3 (1-anchor baseline) |
| `examples/phase3_i2v/L2_REPORT_multiframe.md` | Multi-keyframe L2 (the same anchoring story) |
| `examples/phase3_i2v/PHASE3_FINDINGS.md` | Combined Phase 3 executive summary (when written) |
| `/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/vllm_l3_multi/report.md` | Auto-generated per-run metrics |
| `~/.claude/projects/-home-yizhu-code-vllm-omni/memory/project_phase2_text_encoder_drift.md` | Text-encoder version drift (unchanged for I2V) |
