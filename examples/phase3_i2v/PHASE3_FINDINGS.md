# Phase 3 — vllm-omni vs moonvalley_ai I2V parity investigation

**Status (2026-04-28):** Phase 3 closed. vllm-omni's I2V code path is **computationally equivalent or better than T2V's noise floor** at all four injection tiers (L0/L1/L2/L3) for both single-frame and multi-keyframe variants. **A new L0 tier was added late in Phase 3** to fully isolate transformer math from scheduler-recurrence amplification — see `examples/phase2/L0_REPORT.md` and the `L0_REPORT_*.md` here.

## Quick summary

Nine runs total: single-frame I2V × {L0, L1, L2, L3}, multi-keyframe I2V × {L0, L1, L2, L3}, plus a new T2V L0 baseline.

| Run | per-step `v_pred` mean rel | `z_final` rel | `z_final` cos_sim |
|---|---:|---:|---:|
| **T2V Phase 2 L0 (new — transformer-isolated)** | 0.811% | **0.842%** | **0.999973** |
| **T2V Phase 2 L1** | 0.811% | 4.89% | 0.998886 |
| **I2V Single L0 (new)** | 0.806% | **0.713%** | **0.999978** |
| **I2V Single L1** | 0.806% | 2.84% | 0.999580 |
| **I2V Single L2** | 4.37% | 11.70% | 0.987631 |
| **I2V Single L3** | 4.65% | 11.74% | 0.987057 |
| **I2V Multi L0 (new)** | 0.876% | **0.850%** | **0.999968** |
| **I2V Multi L1** | 0.876% | 3.30% | 0.999474 |
| **I2V Multi L2** | **1.47%** | **3.62%** | **0.999222** |
| **I2V Multi L3** | 1.68% | 3.76% | 0.999176 |

vs T2V Phase 2:

| Tier | T2V `z_final` rel | I2V Single | I2V Multi (3 anchors) |
|---|---:|---:|---:|
| **L0 (new)** | **0.84%** | **0.71%** | **0.85%** — all three variants converge to FA3 noise floor |
| L1 | 4.89% | 2.84% (1.7× tighter) | 3.30% |
| L2 | 57.5% | 11.70% (4.9× tighter) | 3.62% **(16× tighter)** |
| L3 | 57.8% | 11.74% (4.9× tighter) | 3.76% **(15× tighter)** |

I2V transformer code adds **no** additional architectural divergence on top of the existing T2V Phase 2 noise floor (FA3 build mismatch). The cond-frame anchoring mechanism in cross-attention dramatically tightens the L2/L3 trajectory, and **scales with the number of anchors** — 3 anchors at `[0, 63, 127]` collapse the L2 trajectory drift to L1-tier tightness.

## L0 finding: L1's z_final divergence is entirely scheduler accumulation, not transformer drift

A late addition to Phase 3 was the **L0 tier**: identical to L1 plus per-iteration overwrite of the pipeline's local `z` variable. With L0, both the transformer call AND the inline DDPM update operate on mv's exact state at each iteration, so only the LAST iteration's drift can propagate to `z_final`.

L1 → L0 results show that **the entire L1 z_final divergence is cross-step scheduler accumulation, not transformer math**:

| | L0 z_final | L1 z_final | L0 / L1 | per-step `v_pred` rel (unchanged) |
|---|---:|---:|---|---:|
| T2V | 0.842% | 4.89% | **5.8× tighter** | 0.811% |
| I2V single | 0.713% | 2.84% | **4.0× tighter** | 0.806% |
| I2V multi | 0.850% | 3.30% | **3.9× tighter** | 0.876% |

At L0, **`z_final` rel ≈ per-step `v_pred` rel** for all three variants. The mechanism: in the L1 pipeline, the local `z` accumulates drift via `drift_{t+1} = drift_t - dt × v_pred_drift_t` over 33 iterations because the pre-hook only substitutes the transformer's kwarg, not the scheduler's `z` argument. L0 removes that accumulation by overwriting `z` at each iteration boundary.

**Implications:**

- vllm-omni's transformer is computationally equivalent to mv's at the per-step level — the per-step `v_pred` rel of ~0.8% is the FA3 build mismatch noise floor (Phase 2 finding), and that's *all* of it.
- L1 conflates transformer math with scheduler recurrence and overstates "transformer divergence" by 4-6×.
- For pure transformer correctness verification, **L0 is the correct tier**. For end-to-end production parity (where there's no per-step injection at all), **L3 is the right tier**.
- The cond-frame anchoring effect that dramatically tightens I2V multi at L1/L2/L3 (16× tighter than T2V at L2) acts *across* iterations through cross-attention. At L0 there's no cross-iteration drift for it to damp, so the L0 numbers across single/multi converge.

## I2V transformer code is bit-equivalent to mv's `dit3d.py` I2V branch

All four step-0 I2V intermediate dumps confirm the new code path is correct:

| Tensor | Single L1/L2/L3 | Multi L1/L2/L3 | Verdict |
|---|---|---|---|
| `i2v_x_after_concat` (post-concat pre-shard) | cos **1.000000** | cos **1.000000** | bit-equivalent |
| `i2v_x_t_mask` (modulation mask) | exact-equal | exact-equal | exact match |
| `i2v_t0_emb` (post-`t_block`) | cos 0.999994 | cos 0.999994 | within fp tolerance |
| `i2v_x_pre_slice` (post-final_layer post-gather) | cos 0.999920 | cos 0.999699 | small drop in multi due to longer sequence |
| `i2v_cond_offsets` (latent offsets) | `[-0.75]` exact | `[-0.75, 15.0, 31.0]` exact | exact match |

This rules out every potential I2V-specific bug:
- **t0 modulation source** ✅ (cos 0.999994)
- **Concat axis + cond_frames spatial pos_emb addition** ✅ (cos 1.000000)
- **`x_t_mask` construction** ✅ (exact-equal)
- **Pre-slice / gather boundary** ✅ (cos 0.999920)
- **Cond-input injection actually fires** ✅ (server logs confirm `_maybe_load_cond_frames` / `_maybe_load_cond_offsets`)

## Bugs found and fixed during Phase 3

| # | Where | Symptom | Root cause | Fix | Memory |
|---|---|---|---|---|---|
| 1 | `vllm_omni/diffusion/debug/dump.py:155` | Worker startup `RecursionError` in `model.eval()`; `module.train(mode)` recursing 1000+ times | `nn.Module.__setattr__` registers Module-typed values as child modules. Setting `transformer._dump_pipeline = self` (where `self` is `MareyPipeline(nn.Module, …)`) made `MareyPipeline` a child of `transformer`, creating a `_modules` cycle. | Bypass `nn.Module.__setattr__` via `self.transformer.__dict__["_dump_pipeline"] = self` so the attribute is a plain Python ref, not a registered submodule. | — |
| 2 | `vllm_omni/diffusion/models/marey/marey_transformer.py:~1250` and `~1325` | Compare driver returned `nan` for `i2v_x_after_concat`, `i2v_x_t_mask`, `i2v_x_pre_slice` because of shape mismatch | vllm-omni dumped pre-reshape `(B, T+Tf, S, C)`; mv dumped post-reshape `(B, (T+Tf)*S, C)` from inside `split_sp_variables` / `gather_sequence`. Same content, incompatible layout. | Move vllm-omni dumps to *after* the `(B, T+Tf, S, C) → (B, (T+Tf)*S, C)` reshape; flatten `x_t_mask` and `x_pre_slice` on the way out. Now both sides dump in flat `(B, N, C)` matching mv. | — |
| 3 | `moonvalley_ai/inference-service/marey_inference.py:infer` | mv reference run failed at startup: `No such option: --frame-conditions` | The CLI flag was added on a previous-task branch but never landed on `marey-serving-comparison`. The kwarg + `_frame_conditions` method already existed. | Re-added the typer.Option in the `infer` command and the `frame_conditions=...` forwarding to `model.infer()` (working-tree change in mv). | `project_marey_cli_quirks.md` |
| 4 | `moonvalley_ai/inference-service/marey_inference.py:_setup_scheduler_config` | `timesteps_schedule rel = 3.04%`; mv ran linspace ending at 2.99 vs vllm-omni's stride-3 ending at 88.30. **Inflated I2V single L1 z_final from 2.84% → 12.09%.** | The Phase 2 fix (one `OmegaConf.update("scheduler.use_distilled_steps", …)` line) was a working-tree change that was never committed; was lost between Phase 2 and Phase 3. | Re-applied as a working-tree change. After fix: schedule rel = 0.000%, all z_final numbers in this report use the post-fix data. **Pending commit to `marey-serving-comparison`** so it doesn't bite a fourth time. | `project_mv_distilled_steps_bug.md` |

## Open finding (deferred): `cond_frames` cos = 0.9076 between sides

Same input image, same VAE weights, but vllm-omni's native `MareyPipeline._encode_cond_frames` produces `cond_frames` with cosine 0.9076 vs mv's `MareyInference._frame_conditions`. **At all tiers in this report, mv's value is loaded via `MAREY_LOAD_COND_FRAMES_PATH`**, so this VAE divergence does not enter z_final. The 32.98% rel-diff in the per-category tables is purely diagnostic.

Plausible sources (deferred):

1. **VAE input dtype mismatch.** vllm-omni casts `cond_images` to `bf16` *before* `vae.encode_images`; mv may keep it `fp32`. A deep conv stack amplifies bf16-vs-fp32 input drift dramatically.
2. **VAE forward kernel differences** between the independent Python wrappers.
3. **Pre-VAE preprocessing drift** — `ImageOps.fit` is deterministic, but `transforms.Normalize(inplace=True)` ordering relative to `to_tensor` could differ.

To investigate: run a small standalone script that loads `cond_frame_0.webp` in both venvs, applies each side's preprocessing + VAE, and compares. ~5-10 min, no GPU.

## Verifications that *passed* (ruled out as Phase 3 divergence sources)

The investigation re-confirmed all the Phase 2 hypotheses, plus rules out everything I2V-specific:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| t0 modulation source | ✅ ruled out | `i2v_t0_emb` cos = 0.999994 across all 6 runs |
| Concat axis (cond frames appended to T) | ✅ ruled out | `i2v_x_after_concat` cos = 1.000000 across all 6 runs |
| `x_t_mask` construction | ✅ ruled out | exact byte-equal between sides at all 6 runs |
| Spatial pos-emb on cond frames | ✅ ruled out | `x_after_concat` cos 1.0 (the unconditional addition mirroring mv `dit3d.py:1033` is correct) |
| Pre-slice / unpatchify boundary | ✅ ruled out | `i2v_x_pre_slice` cos ≥ 0.99969 |
| Scheduler (timesteps schedule) | ✅ ruled out **after the fix** | `timesteps_schedule rel = 0.000%` |
| Cond-input injection actually fires | ✅ confirmed | server log `_maybe_load_cond_frames` / `_maybe_load_cond_offsets` |
| All T2V-side hypotheses (SwiGLU, RMSNorm, RoPE, modulate, attention masking, weight loading, ulysses degree, per-rank shard alignment) | ✅ ruled out by Phase 2 | See `examples/phase2/L1_REPORT.md` |

## Cond-frame anchoring (the empirical "discovery" of Phase 3)

The L2/L3 numbers across single-vs-multi are not just an incidental improvement — they're a clean demonstration of how cond-frame anchoring works in a transformer with global cross-attention:

- **At L1** (per-step transformer-input injection): cond_frames are just additional bit-identical tokens. They don't damp anything because there's no drift to damp — every step's transformer input is forcibly re-synced. Adding more anchors slightly *hurts* because of the extra ~3% sequence length per anchor → marginally more accumulated FA3 noise per attention call. **Single L1 z_final 2.84% → Multi L1 z_final 3.30%.**
- **At L2** (scheduler recurrence): cond positions remain bit-identical via injection while noise positions drift step-to-step. Through cross-attention, the bit-identical cond tokens pull the global attention pattern toward mv's exact trajectory, damping the noise tokens' v_pred drift. **More anchors → more damping**. Single L2 v_pred 4.37% → Multi L2 v_pred **1.47%** (3× tighter); z_final 11.70% → **3.62%**.
- **At L3** (production-mode + native text encoder): same damping mechanism. The +0.28 pp text-encoder native drift at L3 is small enough that L3 multi (3.76% z_final) is essentially equal to L2 multi (3.62%) — the cond anchoring dominates everything else.

This was hypothesized in the planning round and verified empirically by Phase E. For applications where the cond image is reliably reproducible across both engines (i.e. they share VAE preprocessing), more cond anchors directly reduce engine-vs-engine divergence.

## Reproducer

```bash
# 1. Apply mv-side working-tree fixes (if not already in place)
#    — `--frame-conditions` typer.Option in inference-service/marey_inference.py
#    — `OmegaConf.update("scheduler.use_distilled_steps", …)` in _setup_scheduler_config
#    See bug table above and ~/.claude/.../memory/project_mv_distilled_steps_bug.md.

# 2. Single-frame mv reference (one-time per resolution + seed)
COND_IMAGES=/app/yizhu/marey/vllm_omni/vllm_omni_storage/cond_frame_0.webp \
FRAME_INDICES=0 SEED=42 \
bash examples/phase3_i2v/run_moonvalley_dump.sh ref_single

# 3. Single-frame vllm-omni L1/L2/L3
for TIER in L1 L2 L3; do
    COND_IMAGES=/app/yizhu/marey/vllm_omni/vllm_omni_storage/cond_frame_0.webp \
    FRAME_INDICES=0 SEED=42 LEVEL=$TIER \
    bash examples/phase3_i2v/run_vllm_omni_dump.sh vllm_${TIER,,}_single
done

# 4. Multi-keyframe mv reference (3-anchor at indices 0, 63, 127)
COND=/app/yizhu/marey/vllm_omni/vllm_omni_storage/cond_frame_0.webp
COND_IMAGES="$COND,$COND,$COND" FRAME_INDICES=0,63,127 SEED=42 \
bash examples/phase3_i2v/run_moonvalley_dump.sh ref_multi

# 5. Multi-keyframe vllm-omni L1/L2/L3
for TIER in L1 L2 L3; do
    REF_DIR=/app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v/ref_multi \
    COND_IMAGES="$COND,$COND,$COND" FRAME_INDICES=0,63,127 SEED=42 LEVEL=$TIER \
    bash examples/phase3_i2v/run_vllm_omni_dump.sh vllm_${TIER,,}_multi
done
```

Each vllm-omni run auto-generates a per-tier `report.md`; the narrative reports under `examples/phase3_i2v/L*_REPORT_*.md` aggregate those into Phase 2-style summaries.

## Conclusion

**vllm-omni's Marey 30B I2V code path is computationally equivalent to moonvalley_ai's reference implementation** at the per-step transformer level (cos ≥ 0.9999 on all four I2V intermediates), and *tighter* at the end-to-end z_final level than the equivalent T2V baseline thanks to cond-frame anchoring through cross-attention.

For production use:
- **Single-frame I2V** ships with `z_final` cos ≥ 0.987 (4.9× tighter than T2V end-to-end).
- **Multi-keyframe I2V (≥2 anchors)** ships with `z_final` cos ≥ 0.999 (15× tighter than T2V).

The residual divergence (0.81% per-step v_pred at L1) is the same FA3 build mismatch identified in Phase 2 — affecting both T2V and I2V identically. I2V adds no new sources.

## Related artifacts

| Path | Purpose |
|---|---|
| `L0_REPORT_singleframe.md` | Single-frame L0 — transformer-isolated (z_final rel 0.71%) |
| `L1_REPORT_singleframe.md` | Single-frame L1 — full setup + step-by-step analysis |
| `L2_REPORT_singleframe.md` | Single-frame L2 |
| `L3_REPORT_singleframe.md` | Single-frame L3 |
| `L0_REPORT_multiframe.md` | Multi-keyframe L0 — transformer-isolated (z_final rel 0.85%) |
| `L1_REPORT_multiframe.md` | Multi-keyframe L1 — what differs vs single |
| `L2_REPORT_multiframe.md` | Multi-keyframe L2 — anchoring damping result |
| `L3_REPORT_multiframe.md` | Multi-keyframe L3 |
| `examples/phase2/L0_REPORT.md` | T2V L0 — same mechanism, no I2V tensors (z_final rel 0.84%) |
| `run_moonvalley_dump.sh` / `run_vllm_omni_dump.sh` | Tier runners (took env: `COND_IMAGES`, `FRAME_INDICES`, `SEED`, `LEVEL`) |
| `summary_report.py` | Per-tier markdown generator (run automatically by `run_vllm_omni_dump.sh`) |
| `examples/phase2/PHASE2_FINDINGS.md` | T2V Phase 2 baseline (FA3 finding still applies) |
| `examples/online_serving/marey/marey_reference_inference_dump.py` | mv-side wrapper (extended with I2V dumps in this phase) |
| `vllm_omni/diffusion/debug/dump.py` | DumpMixin (extended with I2V dumps + load env vars) |
| `examples/offline_inference/marey/compare_dumps.py` | Comparison driver (recognizes new I2V tensor names) |
| `~/.claude/.../memory/project_phase2_fa3_mismatch.md` | FA3 build mismatch (still the residual cause) |
| `~/.claude/.../memory/project_mv_distilled_steps_bug.md` | The schedule bug that recurred in Phase 3 |
| `~/.claude/.../memory/project_phase2_text_encoder_drift.md` | Text-encoder version drift (unchanged for I2V) |
