#!/bin/bash
# Marey MMDiT online serving — per-stage launcher for multi-node or manually
# split single-node deployments.
#
# Stage 0 runs the API server + text encoders AND hosts the orchestrator
# (OmniMasterServer) that routes outputs between stages over ZMQ/TCP. Stages
# 1 (DiT) and 2 (VAE) run headless and register with that orchestrator.
#
# Usage:
#   # on node A (orchestrator + stage 0):
#   MASTER_ADDRESS=<node-A-ip> ./run_server_stage.sh --stage 0
#
#   # on node B (DiT, headless):
#   MASTER_ADDRESS=<node-A-ip> ./run_server_stage.sh --stage 1
#
#   # on node C (VAE, headless):
#   MASTER_ADDRESS=<node-A-ip> ./run_server_stage.sh --stage 2
#
#   # single-node dev: all three in the same terminal (starts in background):
#   MASTER_ADDRESS=127.0.0.1 ./run_server_stage.sh --stage all
#
# Required env vars (same as run_server.sh):
#   MODEL                - Path to the Marey checkpoint directory.
#   MOONVALLEY_AI_PATH   - Path to the moonvalley_ai checkout (for the opensora
#                          VAE source tree on whichever node runs stage 2).
#                          Not strictly needed on stages 0 or 1 but harmless.
#
# Optional env vars:
#   STAGE                - Stage selector (same as --stage flag, default: all).
#   MASTER_ADDRESS       - Orchestrator host IP (default: 127.0.0.1).
#                          Stages on other nodes must use the orchestrator's
#                          real, routable IP — not localhost.
#   MASTER_PORT          - Orchestrator port for the OmniMasterServer
#                          (default: 8091). ZMQ binds on MASTER_ADDRESS:this.
#   PORT                 - API server port, only meaningful on stage 0
#                          (default: 8098).
#   FLOW_SHIFT           - Flow shift (default: 3.0). Forwarded to every stage;
#                          stage-0 needs it but it's cheap to forward.
#   STAGE_INIT_TIMEOUT   - Per-stage handshake timeout in seconds (default: 900).
#                          Covers cold checkpoint load + vLLM warmup. Bump if
#                          the DiT weights live on slow storage.
#   INIT_TIMEOUT         - Overall orchestrator-startup timeout in seconds
#                          (default: 1800). Must be >= the longest single-stage
#                          init time + registration slack. Only stage 0
#                          consults this, but we forward to all for consistency.
#   GPU_MEMORY_UTILIZATION, HF_HOME, VLLM_OMNI_STORAGE_PATH,
#   MAREY_DUMP_DIR, MAREY_DUMP_FLOAT32 — forwarded as in run_server.sh.
#   STAGE_CONFIGS_PATH   - Which YAML to use (default: marey.yaml under
#                          vllm_omni/model_executor/stage_configs/).
#
# Extra CLI args (forwarded to the `vllm-omni serve` invocation of the selected
# stage) go after `--`:
#   ./run_server_stage.sh --stage 1 -- --gpu-memory-utilization 0.9

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

STAGE="${STAGE:-all}"
MASTER_ADDRESS="${MASTER_ADDRESS:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-33567}"
PORT="${PORT:-8098}"
FLOW_SHIFT="${FLOW_SHIFT:-3.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"
STAGE_INIT_TIMEOUT="${STAGE_INIT_TIMEOUT:-900}"
INIT_TIMEOUT="${INIT_TIMEOUT:-1800}"
VLLM_OMNI_PROJECT="${VLLM_OMNI_PROJECT:-${REPO_ROOT}}"
STAGE_CONFIGS_PATH="${STAGE_CONFIGS_PATH:-${VLLM_OMNI_PROJECT}/vllm_omni/model_executor/stage_configs/marey.yaml}"

EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [-- EXTRA_VLLM_ARGS...]

Options:
  --stage {0|1|2|all}        Stage to launch (default: $STAGE)
  --master-address ADDR      Orchestrator address (default: $MASTER_ADDRESS)
  --master-port PORT         Orchestrator master port (default: $MASTER_PORT)
  --port PORT                API port for stage 0 (default: $PORT)
  --stage-configs-path PATH  Stage config YAML (default: $STAGE_CONFIGS_PATH)
  --help                     Show this help message

Notes:
  - Stage 0 must start before stages 1/2. It will appear to hang until the
    headless stages register — that's normal.
  - --stage all launches all three in one session (useful for local dev).
  - Extra args after '--' are forwarded only to the selected stage.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)            STAGE="$2"; shift 2 ;;
        --master-address)   MASTER_ADDRESS="$2"; shift 2 ;;
        --master-port)      MASTER_PORT="$2"; shift 2 ;;
        --port)             PORT="$2"; shift 2 ;;
        --stage-configs-path) STAGE_CONFIGS_PATH="$2"; shift 2 ;;
        --help|-h)          usage; exit 0 ;;
        --)                 shift; EXTRA_ARGS=("$@"); break ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "$STAGE" != "0" && "$STAGE" != "1" && "$STAGE" != "2" && "$STAGE" != "all" ]]; then
    echo "Invalid --stage value: $STAGE" >&2
    usage
    exit 1
fi

: "${MODEL:?MODEL must be set to the Marey checkpoint directory}"
: "${MOONVALLEY_AI_PATH:?MOONVALLEY_AI_PATH must be set to the moonvalley_ai checkout}"

echo "Starting Marey server — stage=$STAGE"
echo "Model:              $MODEL"
echo "MoonvalleyAI root:  $MOONVALLEY_AI_PATH"
echo "Master:             $MASTER_ADDRESS:$MASTER_PORT"
echo "API port (stg 0):   $PORT"
echo "Flow shift:         $FLOW_SHIFT"
echo "Stage init timeout: $STAGE_INIT_TIMEOUT s"
echo "Init timeout:       $INIT_TIMEOUT s"
echo "Stage configs:      $STAGE_CONFIGS_PATH"
echo "VLLM Omni project:  $VLLM_OMNI_PROJECT"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "Extra args:         ${EXTRA_ARGS[*]}"

env_args=(
    MOONVALLEY_AI_PATH="${MOONVALLEY_AI_PATH}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    VLLM_OMNI_PROJECT="${VLLM_OMNI_PROJECT}"
)
[[ -n "${HF_HOME:-}" ]]                && env_args+=("HF_HOME=${HF_HOME}")
[[ -n "${VLLM_OMNI_STORAGE_PATH:-}" ]] && env_args+=("VLLM_OMNI_STORAGE_PATH=${VLLM_OMNI_STORAGE_PATH}")
[[ -n "${MAREY_DUMP_DIR:-}" ]]         && env_args+=("MAREY_DUMP_DIR=${MAREY_DUMP_DIR}")
[[ -n "${MAREY_DUMP_FLOAT32:-}" ]]     && env_args+=("MAREY_DUMP_FLOAT32=${MAREY_DUMP_FLOAT32}")

# Shared vllm-omni serve invocation parts
run_vllm_serve() {
    local stage_id="$1"
    shift
    env "${env_args[@]}" \
    uv run --no-sync --frozen --project "${VLLM_OMNI_PROJECT}" vllm-omni serve "$MODEL" --omni \
        --stage-configs-path "$STAGE_CONFIGS_PATH" \
        --flow-shift "$FLOW_SHIFT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --stage-init-timeout "$STAGE_INIT_TIMEOUT" \
        --init-timeout "$INIT_TIMEOUT" \
        --stage-id "$stage_id" \
        -oma "$MASTER_ADDRESS" \
        -omp "$MASTER_PORT" \
        "$@" \
        "${EXTRA_ARGS[@]}"
}

# Stage 0 — master orchestrator + API server + text encoders.
# Does NOT pass --headless so the API server comes up.
run_stage_0() {
    echo "[stage 0] Launching text encoders + orchestrator + API (port $PORT)..."
    set -x
    run_vllm_serve 0 --port "$PORT"
}

# Stage 1 — DiT, headless. Parallelism (ulysses_degree / tensor_parallel_size)
# comes from the stage config YAML's stage_args[1].engine_args.parallel_config.
run_stage_1() {
    echo "[stage 1] Launching DiT (headless)..."
    set -x
    run_vllm_serve 1 --headless
}

# Stage 2 — VAE, headless.
run_stage_2() {
    echo "[stage 2] Launching VAE (headless)..."
    set -x
    run_vllm_serve 2 --headless
}

case "$STAGE" in
    0) run_stage_0 ;;
    1) run_stage_1 ;;
    2) run_stage_2 ;;
    all)
        echo "Launching all 3 stages in this session (background jobs)..."
        run_stage_0 &
        STAGE_0_PID=$!
        cleanup() {
            for pid in "${STAGE_0_PID:-}" "${STAGE_1_PID:-}"; do
                [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
            done
            for pid in "${STAGE_0_PID:-}" "${STAGE_1_PID:-}"; do
                [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
            done
        }
        trap cleanup EXIT INT TERM

        sleep 3  # let stage 0 bind the master sockets
        run_stage_1 &
        STAGE_1_PID=$!

        sleep 1
        run_stage_2
        ;;
esac
