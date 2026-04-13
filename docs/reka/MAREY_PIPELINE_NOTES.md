# Marey Pipeline Notes

## Summary

Marey is currently a `config.yaml`-native pipeline, not a standard HF/diffusers
checkpoint layout.

That means:

- The Marey pipeline reads its real runtime config from `config.yaml` in the
  checkpoint directory.
- A plain Marey checkpoint directory may not contain `config.json` or
  `model_index.json`.
- Older startup paths could fail early because they assumed every model had an
  HF/diffusers config file.

## Current Recommended Setup

1. Keep `config.yaml` as-is in the Marey checkpoint directory.
2. Run the server with `--model-class-name MareyPipeline`.
3. Use the newer `resolve_model_config_path()` behavior that falls back cleanly
   when no `config.json` or `model_index.json` exists.
4. Treat Marey as a special local pipeline during diffusion config resolution so
   it does not require `transformer/config.json`.

Example:

```bash
vllm-omni serve /path/to/marey-checkpoint \
  --omni \
  --model-class-name MareyPipeline
```

## What Was Built Here

This branch keeps Marey on the existing `config.yaml` contract and adds a small,
isolated escape hatch for it:

- `OmniDiffusion` now short-circuits HF config probing when the model is a local
  `MareyPipeline` checkpoint with `config.yaml`.
- `AsyncOmniDiffusion` does the same.
- If `model_index.json` sets `_class_name` to `MareyPipeline`, startup still
  skips `transformer/config.json` and uses the local `config.yaml` flow.
- The test coverage is isolated to a dedicated Marey-only test file so it can be
  deleted cleanly when Marey is migrated or removed.

## Why Not Convert To `config.json` Right Now

Rewriting Marey to be more upstream-compatible would mean more than generating a
single `config.json`. A real diffusers-style migration would require:

- `model_index.json` at the repo root
- component configs in standard locations
- weights arranged in the expected subfolders
- Marey loading code updated to consume those component configs instead of the
  current root `config.yaml`

That is a larger packaging and loader redesign. The changes in this branch are
meant to stabilize the current Marey checkpoint format without pretending it is
already a standard diffusers repo.
