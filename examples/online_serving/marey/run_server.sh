#!/bin/bash
# Marey MMDiT online serving startup script.
#
# The model directory must contain a config.yaml with text_encoder, vae, model,
# and scheduler sections (see examples/offline_inference/marey/text_to_video.py).
#
# The model directory has neither model_index.json nor config.json, so stage
# config auto-detection cannot resolve a model_type and falls through to the
# default single-stage diffusion factory. --model-class-name MareyPipeline
# names the pipeline class that the factory should instantiate (looked up in
# DiffusionModelRegistry at vllm_omni/diffusion/registry.py).
#
# Required env vars:
#   MODEL                - Path to the Marey checkpoint directory (with config.yaml).
#   MOONVALLEY_AI_PATH   - Path to the moonvalley_ai checkout (containing open_sora/).
#                          Consumed by vllm_omni.diffusion.models.marey.pipeline_marey
#                          to locate the opensora VAE source tree.
#
# Optional env vars:
#   PORT                       - Server port (default: 8098).
#   FLOW_SHIFT                 - Flow shift value (default: 3.0).
#   ULYSSES_DEGREE             - Sequence parallel degree (default: 8).
#   GPU_MEMORY_UTILIZATION     - GPU memory utilization (default: 0.98).
#   HF_HOME                    - HuggingFace cache directory.
#   VLLM_OMNI_STORAGE_PATH     - vllm-omni storage directory.
#   VLLM_OMNI_PROJECT          - Path to the vllm-omni checkout for `uv run --project`
#                                (default: repo root inferred from this script's location).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${MODEL:?MODEL must be set to the Marey checkpoint directory}"
: "${MOONVALLEY_AI_PATH:?MOONVALLEY_AI_PATH must be set to the moonvalley_ai checkout}"

PORT="${PORT:-8098}"
FLOW_SHIFT="${FLOW_SHIFT:-3.0}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"
VLLM_OMNI_PROJECT="${VLLM_OMNI_PROJECT:-${REPO_ROOT}}"

echo "Starting Marey server..."
echo "Model:              $MODEL"
echo "MoonvalleyAI root:  $MOONVALLEY_AI_PATH"
echo "Port:               $PORT"
echo "Flow shift:         $FLOW_SHIFT"
echo "Ulysses degree:     $ULYSSES_DEGREE"

env_args=(
    MOONVALLEY_AI_PATH="${MOONVALLEY_AI_PATH}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
[[ -n "${HF_HOME:-}" ]]                       && env_args+=("HF_HOME=${HF_HOME}")
[[ -n "${VLLM_OMNI_STORAGE_PATH:-}" ]]        && env_args+=("VLLM_OMNI_STORAGE_PATH=${VLLM_OMNI_STORAGE_PATH}")

# Debug / reproducibility toggles — forwarded to the worker process and
# consumed by DumpMixin (see vllm_omni/diffusion/debug/dump.py). Inert unless
# MAREY_PIPELINE_CLASS=DumpMareyPipeline is also set.
[[ -n "${MAREY_DUMP_DIR:-}" ]]                && env_args+=("MAREY_DUMP_DIR=${MAREY_DUMP_DIR}")
[[ -n "${MAREY_LOAD_INITIAL_NOISE:-}" ]]      && env_args+=("MAREY_LOAD_INITIAL_NOISE=${MAREY_LOAD_INITIAL_NOISE}")
[[ -n "${MAREY_LOAD_STEP_NOISE_DIR:-}" ]]     && env_args+=("MAREY_LOAD_STEP_NOISE_DIR=${MAREY_LOAD_STEP_NOISE_DIR}")

# Pipeline class selector: defaults to MareyPipeline. Set to DumpMareyPipeline
# (registered in vllm_omni/diffusion/registry.py) to enable instrumentation.
MAREY_PIPELINE_CLASS="${MAREY_PIPELINE_CLASS:-MareyPipeline}"
echo "Pipeline class:     $MAREY_PIPELINE_CLASS"

VLLM_OMNI_PYTHON="${VLLM_OMNI_PYTHON:-${VLLM_OMNI_PROJECT}/.venv/bin/python}"

env "${env_args[@]}" \
"${VLLM_OMNI_PYTHON}" -m vllm_omni.entrypoints.cli.main serve "$MODEL" --omni \
    --port "$PORT" \
    --model-class-name "$MAREY_PIPELINE_CLASS" \
    --flow-shift "$FLOW_SHIFT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --ulysses-degree "$ULYSSES_DEGREE"
