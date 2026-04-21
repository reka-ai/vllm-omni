#!/bin/bash
# Marey MMDiT online serving — TP=4 variant of run_server.sh.
#
# Uses marey_tp4.yaml (TP=4 on the DiT instead of SP=4) and does NOT pass
# --ulysses-degree so the stage config's ulysses_degree=1 is honored.
# Same required/optional env vars as run_server.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${MODEL:?MODEL must be set to the Marey checkpoint directory}"
: "${MOONVALLEY_AI_PATH:?MOONVALLEY_AI_PATH must be set to the moonvalley_ai checkout}"

PORT="${PORT:-8098}"
FLOW_SHIFT="${FLOW_SHIFT:-3.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"
VLLM_OMNI_PROJECT="${VLLM_OMNI_PROJECT:-${REPO_ROOT}}"

echo "Starting Marey server (TP=4 variant)..."
echo "Model:              $MODEL"
echo "MoonvalleyAI root:  $MOONVALLEY_AI_PATH"
echo "Port:               $PORT"
echo "Flow shift:         $FLOW_SHIFT"
echo "VLLM Omni project:  $VLLM_OMNI_PROJECT"

env_args=(
    MOONVALLEY_AI_PATH="${MOONVALLEY_AI_PATH}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    VLLM_OMNI_PROJECT="${VLLM_OMNI_PROJECT}"
)
[[ -n "${HF_HOME:-}" ]]                && env_args+=("HF_HOME=${HF_HOME}")
[[ -n "${VLLM_OMNI_STORAGE_PATH:-}" ]] && env_args+=("VLLM_OMNI_STORAGE_PATH=${VLLM_OMNI_STORAGE_PATH}")
[[ -n "${MAREY_DUMP_DIR:-}" ]]         && env_args+=("MAREY_DUMP_DIR=${MAREY_DUMP_DIR}")
[[ -n "${MAREY_DUMP_FLOAT32:-}" ]]     && env_args+=("MAREY_DUMP_FLOAT32=${MAREY_DUMP_FLOAT32}")

set -x
env "${env_args[@]}" \
uv run --no-sync --frozen --project /home/aormazabal/wlam/wlam-inference/vllm-omni/ vllm-omni serve "$MODEL" --omni \
    --port "$PORT" \
    --stage-configs-path "${VLLM_OMNI_PROJECT}/vllm_omni/model_executor/stage_configs/marey_tp4.yaml" \
    --flow-shift "$FLOW_SHIFT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
