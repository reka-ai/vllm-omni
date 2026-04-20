#!/bin/bash
# Run the moonvalley_ai marey_inference.py reference for the 7B (Flux-7B,
# ul2-metaclip) Marey checkpoint. Per-model differences from the 30B reference:
#
#   - single checkpoint dir (model-folder == checkpoint-folder)
#   - 480p / 64 frames (training-distribution max)
#   - 50 steps (not distilled), guidance-scale 7.0
#   - shift-value 0.0 (7B trained with use_timestep_transform=false)
#   - omits --use-distilled-steps and --add-quality-guidance
#
# *** OPEN ISSUE - use_timestep_transform ***
# moonvalley_ai's marey_inference.py:959 declares
#   typer.Option(True, "--use-timestep-transform", ...)
# as a Typer single-flag bool with no --no- counterpart. That means we cannot
# pass `--no-use-timestep-transform` from the CLI as-is. Two workarounds:
#   (a) Apply this 1-line local patch to marey_inference.py:959:
#         "--use-timestep-transform" -> "--use-timestep-transform/--no-use-timestep-transform"
#       then this script's --no-use-timestep-transform flag (below) takes effect.
#   (b) Pre-patch the script via sed in this wrapper (commented-out below).
# Without one of these, the run will use use_timestep_transform=True and apply
# shift=0.0 to a no-op transform — i.e. the same behavior as transform=False —
# but verify by inspecting the printed scheduler config (marey_inference.py:204-220).

set -euo pipefail

: "${MOONVALLEY_AI_PATH:?MOONVALLEY_AI_PATH must be set to the moonvalley_ai checkout}"

MODEL_FOLDER="${MODEL_FOLDER:-/home/claudio/marey_checkpoints/epoch5-global_step70000}"
CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-${MODEL_FOLDER}}"
WATERMARKER_PATH="${WATERMARKER_PATH:-/app/wlam/models/checkpoints/marey/videoseal/y_256b_img.jit}"
SEED="${SEED:-42}"
OUTPUT="${OUTPUT:-./marey_7b_ref_seed${SEED}_$(date +%Y%m%d_%H%M%S).mp4}"
PROMPT="${PROMPT:-Detailed Description: A majestic, aged eagle with mottled golden-brown feathers soars gracefully through a vast, ancient indoor chamber. Its expansive wings barely flap, catching the air as it glides effortlessly between towering stone pillars adorned with glinting metallic accents. Beams of morning light pierce the gloom, filtering through a cracked skylight high above and illuminating swirling dust motes in their path. The camera pans smoothly, following the eagle's silent flight as it navigates the cavernous space, its sharp eyes scanning the stone floor below, creating a scene of serene power and timeless solitude. Background: The far reaches of the chamber fade into deep shadow, with the silhouettes of distant pillars barely visible. High above, a cracked skylight serves as the primary light source, its fractured glass creating distinct rays of light. Middleground: The aged eagle glides on a steady path, its mottled golden-brown wings spread wide. It passes through the dramatic beams of light, which highlight the intricate details of its feathers and the dust particles dancing in the air. Foreground: The camera looks up from a low angle, tracking the eagle's movement across the expansive stone floor, which is patterned with the bright shafts of light and deep shadows cast by the pillars.}"

echo "Running moonvalley_ai 7B reference inference"
echo "  MOONVALLEY_AI_PATH:  ${MOONVALLEY_AI_PATH}"
echo "  MODEL_FOLDER:        ${MODEL_FOLDER}"
echo "  CHECKPOINT_FOLDER:   ${CHECKPOINT_FOLDER}"
echo "  SEED:                ${SEED}"
echo "  OUTPUT:              ${OUTPUT}"

# Workaround (b): in-place patch marey_inference.py to expose --no-use-timestep-transform
# Uncomment this block to auto-apply (or apply manually once and remove this).
# MAREY_INFER="${MOONVALLEY_AI_PATH}/inference-service/marey_inference.py"
# if ! grep -q -- '--use-timestep-transform/--no-use-timestep-transform' "${MAREY_INFER}"; then
#     sed -i 's|"--use-timestep-transform",|"--use-timestep-transform/--no-use-timestep-transform",|' "${MAREY_INFER}"
#     echo "Patched ${MAREY_INFER} to expose --no-use-timestep-transform"
# fi

# If any dump/load env var is set, route through the dump wrapper so we
# capture (or load) the same tensors as DumpMixin on the vllm-omni side.
INFERENCE_ENTRY="${MOONVALLEY_AI_PATH}/inference-service/marey_inference.py"
if [ -n "${MAREY_DUMP_DIR:-}" ] || [ -n "${MAREY_LOAD_INITIAL_NOISE:-}" ] || [ -n "${MAREY_LOAD_STEP_NOISE_DIR:-}" ]; then
    INFERENCE_ENTRY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/marey_reference_inference_dump.py"
    echo "Dump/load env var detected; using wrapper: ${INFERENCE_ENTRY}"
fi

# maybe_update_model_cfg uses a RELATIVE path "open_sora/configs/model/..."
# so CWD must be the moonvalley_ai repo root for the config lookup to work.
cd "${MOONVALLEY_AI_PATH}"

# Use moonvalley_ai/inference-service's project venv directly. Override via
# MOONVALLEY_TORCHRUN to use a different python (e.g. a unified env that has
# all moonvalley deps + dump-wrapper deps).
MOONVALLEY_TORCHRUN="${MOONVALLEY_TORCHRUN:-${MOONVALLEY_AI_PATH}/inference-service/.venv/bin/torchrun}"
if [ ! -x "${MOONVALLEY_TORCHRUN}" ]; then
    echo "ERROR: ${MOONVALLEY_TORCHRUN} not found or not executable"
    exit 1
fi
echo "Using torchrun: ${MOONVALLEY_TORCHRUN}"
LAUNCH_CMD=("${MOONVALLEY_TORCHRUN}" "--nproc_per_node=8")

# flash_attn_3 prebuilt wheels were compiled against GLIBCXX_3.4.32+, but
# Ubuntu 22.04's /lib/x86_64-linux-gnu/libstdc++.so.6 tops out at 3.4.30.
# Preload a newer libstdc++ from the system's miniconda install. Override
# LIBSTDCXX_PRELOAD to point elsewhere if your box doesn't have miniconda.
LIBSTDCXX_PRELOAD="${LIBSTDCXX_PRELOAD:-/home/yizhu/miniconda3/lib/libstdc++.so.6}"
if [ -e "${LIBSTDCXX_PRELOAD}" ]; then
    echo "Preloading libstdc++ from: ${LIBSTDCXX_PRELOAD}"
    export LD_PRELOAD="${LIBSTDCXX_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
else
    echo "WARN: ${LIBSTDCXX_PRELOAD} not found; flash_attn_3 import may fail with GLIBCXX_3.4.32"
fi

PYTHONPATH="${MOONVALLEY_AI_PATH}:${MOONVALLEY_AI_PATH}/open_sora" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${LAUNCH_CMD[@]}" "${INFERENCE_ENTRY}" infer \
  --num-seq-parallel-splits 8 \
  --offload-diffusion \
  --offload-vae \
  --offload-text-encoder \
  --model-folder "${MODEL_FOLDER}" \
  --checkpoint-folder "${CHECKPOINT_FOLDER}" \
  --watermarker-path "${WATERMARKER_PATH}" \
  --height 480 \
  --width 768 \
  --num-frames 64 \
  --fps 24 \
  --steps 50 \
  --guidance-scale 7.0 \
  --disable-caching \
  --use-negative-prompts \
  --negative-prompt "<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, bright, vignette, artifacts, still, noise, texture, scanlines, videogame, 360 camera, VR, transition, flare, saturation, distorted, warped, wide angle, contrast, saturated, vibrant, glowing, cross dissolve, texture, videogame, saturation, cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, blown out, horrible, blurry, worst quality, bad, transition, dissolve, cross-dissolve, melt, fade in, fade out, wobbly, weird, low quality, plastic, stock footage, video camera, boring, static" \
  --no-use-timestep-transform \
  --shift-value 0.0 \
  --use-guidance-schedule \
  --clip-value 10.0 \
  --seed "${SEED}" \
  --warmup-steps 4 \
  --cooldown-steps 18 \
  --output "${OUTPUT}" \
  "${PROMPT}"
