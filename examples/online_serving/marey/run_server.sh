#!/bin/bash
# Marey MMDiT online serving startup script.
#
# The model directory must contain a config.yaml with text_encoder, vae, model,
# and scheduler sections (see examples/offline_inference/marey/text_to_video.py).
#
# Since the model directory does not have a model_index.json, we must explicitly
# pass --model-class-name MareyPipeline so that vllm-omni recognises it as a
# diffusion model and loads the correct pipeline.
#
# Required env vars:
#   MODEL                - Path to the Marey checkpoint directory (with config.yaml).
#   MOONVALLEY_AI_ROOT   - Path to the moonvalley_ai checkout (containing open_sora/).
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
: "${MOONVALLEY_AI_ROOT:?MOONVALLEY_AI_ROOT must be set to the moonvalley_ai checkout}"

PORT="${PORT:-8098}"
FLOW_SHIFT="${FLOW_SHIFT:-3.0}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"
VLLM_OMNI_PROJECT="${VLLM_OMNI_PROJECT:-${REPO_ROOT}}"

echo "Model:              $MODEL"
echo "MoonvalleyAI root:  $MOONVALLEY_AI_ROOT"
echo "Port:               $PORT"
echo "Flow shift:         $FLOW_SHIFT"
echo "Ulysses degree:     $ULYSSES_DEGREE"

env_args=(
    MOONVALLEY_AI_ROOT="${MOONVALLEY_AI_ROOT}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
[[ -n "${HF_HOME:-}" ]]                && env_args+=("HF_HOME=${HF_HOME}")
[[ -n "${VLLM_OMNI_STORAGE_PATH:-}" ]] && env_args+=("VLLM_OMNI_STORAGE_PATH=${VLLM_OMNI_STORAGE_PATH}")

env "${env_args[@]}" \
uv run --project "${VLLM_OMNI_PROJECT}" vllm-omni serve "$MODEL" --omni \
    --port "$PORT" \
    --model-class-name MareyPipeline \
    --flow-shift "$FLOW_SHIFT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --ulysses-degree "$ULYSSES_DEGREE"
