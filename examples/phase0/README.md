# Phase 0 — branch-parity runs

Goal: prove that `marey-serving-comparison` produces the same 30B inference
output as `main` on each codebase.

Artifacts land under `${PHASE0_ROOT}` (default
`/mnt/localdisk/vllm_omni_storage/phase0/20260424`):

```
${PHASE0_ROOT}/
  vllm_main/  final_z.pt, output.mp4, server.log, client.log
  vllm_cmp/   final_z.pt, output.mp4, server.log, client.log
  mv_main/    latents.pt, output.mp4, run.log
  mv_cmp/     latents.pt, output.mp4, run.log
```

## Run order

```bash
# --- vllm-omni side ---
cd /home/yizhu/code/vllm-omni
git checkout main
bash examples/phase0/run_vllm_omni.sh vllm_main

git checkout marey-serving-comparison
bash examples/phase0/run_vllm_omni.sh vllm_cmp

# --- moonvalley_ai side ---
cd /home/yizhu/code/moonvalley_ai_master
git stash            # preserve WIP
git checkout main
bash /home/yizhu/code/vllm-omni/examples/phase0/run_moonvalley.sh mv_main

git checkout marey-serving-comparison
git stash pop        # restore WIP (inert for text-to-video)
bash /home/yizhu/code/vllm-omni/examples/phase0/run_moonvalley.sh mv_cmp

# --- compare ---
cd /home/yizhu/code/vllm-omni
bash examples/phase0/compare.sh
```

## Phase 0 gate

Both comparisons must report `PASS` (default tolerance:
`max_abs_diff < 1e-6 * max_abs(a)`).

- vllm-omni is expected to pass trivially: the only inference-path change on
  `marey-serving-comparison` is factoring `randn_tensor` / `torch.randn_like`
  into `_sample_initial_noise` / `_sample_step_noise` methods whose default
  bodies are byte-identical to the inline calls on `main`.
- moonvalley parity validates that `b96a1b1b1`'s rebased content
  (`use_rf_v2` scheduler-key drop, `sku_embedder.*` state-dict drop,
  `ul2-metaclip → ul2-clip` type rewrite, relative-vae-cp-path resolution)
  is numerically neutral on 30B. It should be; if not, bisect the hunks.

## Cleanup after Phase 0 closes

The `[TEMP Phase 0]` commit adding the `VLLM_OMNI_SAVE_FINAL_LATENT` hook on
both vllm-omni branches can be dropped:

```bash
cd /home/yizhu/code/vllm-omni
git checkout main
git reset --hard HEAD~1              # drop [TEMP Phase 0] commit on main
git checkout marey-serving-comparison
git reset --hard HEAD~1              # drop [TEMP Phase 0] commit on comparison
```
