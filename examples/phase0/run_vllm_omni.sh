#!/usr/bin/env bash
# Phase 0: one vllm-omni 30B inference run producing final_z.pt + output.mp4.
#
# Usage:
#   bash examples/phase0/run_vllm_omni.sh <tag>
#
# <tag> is any label for the run dir, e.g. "vllm_main" or "vllm_cmp".
# Caller is responsible for having checked out the branch under test.

set -euo pipefail

TAG="${1:?Usage: $0 <tag>  (e.g. vllm_main | vllm_cmp)}"

PHASE0_ROOT="${PHASE0_ROOT:-/mnt/localdisk/vllm_omni_storage/phase0/20260424}"
OUT_DIR="${PHASE0_ROOT}/${TAG}"
mkdir -p "${OUT_DIR}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[phase0] vllm-omni run: tag=${TAG}  out=${OUT_DIR}"
echo "[phase0] branch: $(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
echo "[phase0] commit: $(git -C "${REPO_ROOT}" rev-parse HEAD)"

# Env passed to run_server.sh (forwarded to worker processes).
export VLLM_OMNI_SAVE_FINAL_LATENT="${OUT_DIR}/final_z.pt"
export HF_HOME="${HF_HOME:-/mnt/localdisk/vllm_omni_hf_cache}"
export VLLM_OMNI_STORAGE_PATH="${VLLM_OMNI_STORAGE_PATH:-/mnt/localdisk/vllm_omni_storage}"
export MODEL="${MODEL:-/app/hf_checkpoints/marey-distilled-0100/}"
export MOONVALLEY_AI_PATH="${MOONVALLEY_AI_PATH:-/home/yizhu/code/moonvalley_ai_master}"

SERVER_LOG="${OUT_DIR}/server.log"

# Symlink to the canonical repo-local path so `tail marey_30b_server.log`
# from the repo root still works (consistent with established convention).
ln -sf "${SERVER_LOG}" "${REPO_ROOT}/marey_30b_server.log"

echo "[phase0] starting server, log: ${SERVER_LOG}"
echo "[phase0]   (also tail-able at ${REPO_ROOT}/marey_30b_server.log)"
bash "${REPO_ROOT}/examples/online_serving/marey/run_server.sh" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
# TERM to the bash wrapper doesn't propagate to the `uv run` / `vllm-omni
# serve` / worker subprocess tree, so do a pattern-based recursive kill on
# exit. Targets the specific vllm-omni serve invocation and its spawn_main
# worker children.
cleanup_server() {
    pkill -TERM -f "vllm-omni serve" 2>/dev/null || true
    pkill -TERM -f "spawn_main" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "vllm-omni serve" 2>/dev/null || true
    pkill -KILL -f "spawn_main" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup_server EXIT

# Wait for the server to become ready. 30B checkpoint load (~50s) + dummy
# 1080p warmup (several minutes) runs before Uvicorn binds, so we allow up
# to 15 minutes. Also fail fast on common error signatures.
READY_TIMEOUT_S=900
echo "[phase0] waiting for server ready (up to ${READY_TIMEOUT_S}s)..."
for i in $(seq 1 ${READY_TIMEOUT_S}); do
    # vllm-omni 0.17.0 prints "Application startup complete" when the HTTP
    # server is ready and bound. The classic "Uvicorn running on ..." line is
    # not emitted in this version, so we key off the earlier marker.
    if grep -q "Application startup complete" "${SERVER_LOG}" 2>/dev/null; then
        echo "[phase0] server ready after ${i}s"
        break
    fi
    # Fail fast if the server process died or logged an unrecoverable error.
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "[phase0] server process exited early. Last 60 lines:"
        tail -n 60 "${SERVER_LOG}" || true
        exit 2
    fi
    # Fatal-signature fail-fast. Deliberately narrow: HF `Thread-auto_conversion`
    # happily raises OSError/Traceback during model load without killing the
    # main process, so plain "Traceback" is unreliable. These signatures are
    # always fatal in practice.
    if grep -qE "(CUDA out of memory|torch\.OutOfMemoryError|Address already in use|MareyVAEInitializationError|RuntimeError: CUDA error|NCCL error)" "${SERVER_LOG}" 2>/dev/null; then
        echo "[phase0] server log shows a fatal error. Last 60 lines:"
        tail -n 60 "${SERVER_LOG}" || true
        exit 2
    fi
    # Heartbeat every 60s so the caller knows we're still alive.
    if (( i % 60 == 0 )); then
        echo "[phase0]   still waiting (${i}s). Last log line:"
        tail -n 1 "${SERVER_LOG}" 2>/dev/null | sed 's/^/[phase0]     /'
    fi
    sleep 1
done
if ! grep -q "Application startup complete" "${SERVER_LOG}" 2>/dev/null; then
    echo "[phase0] server did not become ready within ${READY_TIMEOUT_S}s"
    tail -n 60 "${SERVER_LOG}" || true
    exit 2
fi

echo "[phase0] running client..."
SEED=42 OUTPUT_PATH="${OUT_DIR}/output.mp4" \
    bash "${REPO_ROOT}/examples/online_serving/marey/run_curl_text_to_video.sh" \
    >"${OUT_DIR}/client.log" 2>&1

echo "[phase0] done. Artifacts in ${OUT_DIR}:"
ls -lh "${OUT_DIR}"
