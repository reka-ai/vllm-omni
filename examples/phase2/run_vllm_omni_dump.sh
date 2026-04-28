#!/usr/bin/env bash
# Phase 2: one vllm-omni 30B inference run producing a DumpMareyPipeline
# dump dir (per-request subdir) consumable by compare_dumps.py against
# /mnt/localdisk/vllm_omni_storage/phase1/ref_30b/.
#
# Usage:
#   bash examples/phase2/run_vllm_omni_dump.sh <tag>
#
# <tag> is the subdir under $PHASE2_ROOT, e.g. "vllm_base" for the no-injection
# baseline, "vllm_runA" for Level 1 (maximally cheating), etc.
#
# Injection level is controlled by env vars (set by the caller; all optional):
#   MAREY_LOAD_INITIAL_NOISE            path to z_initial.pt
#   MAREY_LOAD_STEP_NOISE_DIR           dir containing step_noise_<i>.pt
#   MAREY_LOAD_TEXT_EMBEDS_DIR          dir containing encode_{cond,uncond}_*.pt
#   MAREY_LOAD_TRANSFORMER_INPUTS_DIR   dir containing step<i>_<label>_*.pt
#
# Presets — set one of these for convenience (or set the vars above directly):
#   LEVEL=base   (no injection; Step 2.3 Gate A/B)
#   LEVEL=L1     (initial + step_noise + text_embeds + transformer inputs)
#   LEVEL=L2     (initial + step_noise + text_embeds)
#   LEVEL=L3     (initial + step_noise)
#
# Reference dump (source for injections): $REF_DIR, default phase1/ref_30b/.

set -euo pipefail

TAG="${1:?Usage: $0 <tag>  (e.g. vllm_base | vllm_runA | vllm_runB | vllm_runC)}"

PHASE2_ROOT="${PHASE2_ROOT:-/mnt/localdisk/vllm_omni_storage/phase2}"
OUT_DIR="${PHASE2_ROOT}/${TAG}"
REF_DIR="${REF_DIR:-/mnt/localdisk/vllm_omni_storage/phase1/ref_30b}"
mkdir -p "${OUT_DIR}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---- Preset resolution ------------------------------------------------------
# LEVEL presets expand to the four MAREY_LOAD_* vars. Direct env assignments
# win (preset only sets a var if it's currently unset).
LEVEL="${LEVEL:-}"
case "${LEVEL}" in
    base|"")
        :
        ;;
    L0)
        # Strictest tier: L1 + per-iteration overwrite of pipeline `z` so
        # only the LAST step's transformer drift propagates to z_final.
        export MAREY_LOAD_INITIAL_NOISE="${MAREY_LOAD_INITIAL_NOISE:-${REF_DIR}/z_initial.pt}"
        export MAREY_LOAD_STEP_NOISE_DIR="${MAREY_LOAD_STEP_NOISE_DIR:-${REF_DIR}}"
        export MAREY_LOAD_TEXT_EMBEDS_DIR="${MAREY_LOAD_TEXT_EMBEDS_DIR:-${REF_DIR}}"
        export MAREY_LOAD_TRANSFORMER_INPUTS_DIR="${MAREY_LOAD_TRANSFORMER_INPUTS_DIR:-${REF_DIR}}"
        export MAREY_LOAD_PIPELINE_LATENTS_DIR="${MAREY_LOAD_PIPELINE_LATENTS_DIR:-${REF_DIR}}"
        ;;
    L1)
        export MAREY_LOAD_INITIAL_NOISE="${MAREY_LOAD_INITIAL_NOISE:-${REF_DIR}/z_initial.pt}"
        export MAREY_LOAD_STEP_NOISE_DIR="${MAREY_LOAD_STEP_NOISE_DIR:-${REF_DIR}}"
        export MAREY_LOAD_TEXT_EMBEDS_DIR="${MAREY_LOAD_TEXT_EMBEDS_DIR:-${REF_DIR}}"
        export MAREY_LOAD_TRANSFORMER_INPUTS_DIR="${MAREY_LOAD_TRANSFORMER_INPUTS_DIR:-${REF_DIR}}"
        ;;
    L2)
        export MAREY_LOAD_INITIAL_NOISE="${MAREY_LOAD_INITIAL_NOISE:-${REF_DIR}/z_initial.pt}"
        export MAREY_LOAD_STEP_NOISE_DIR="${MAREY_LOAD_STEP_NOISE_DIR:-${REF_DIR}}"
        export MAREY_LOAD_TEXT_EMBEDS_DIR="${MAREY_LOAD_TEXT_EMBEDS_DIR:-${REF_DIR}}"
        ;;
    L3)
        export MAREY_LOAD_INITIAL_NOISE="${MAREY_LOAD_INITIAL_NOISE:-${REF_DIR}/z_initial.pt}"
        export MAREY_LOAD_STEP_NOISE_DIR="${MAREY_LOAD_STEP_NOISE_DIR:-${REF_DIR}}"
        ;;
    *)
        echo "[phase2] ERROR: unknown LEVEL=${LEVEL}. Use base | L0 | L1 | L2 | L3."
        exit 2
        ;;
esac

echo "[phase2] vllm-omni run: tag=${TAG}  out=${OUT_DIR}"
echo "[phase2] branch: $(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
echo "[phase2] commit: $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "[phase2] LEVEL=${LEVEL:-<unset>}  ref_dir=${REF_DIR}"
echo "[phase2] MAREY_LOAD_INITIAL_NOISE=${MAREY_LOAD_INITIAL_NOISE:-<unset>}"
echo "[phase2] MAREY_LOAD_STEP_NOISE_DIR=${MAREY_LOAD_STEP_NOISE_DIR:-<unset>}"
echo "[phase2] MAREY_LOAD_TEXT_EMBEDS_DIR=${MAREY_LOAD_TEXT_EMBEDS_DIR:-<unset>}"
echo "[phase2] MAREY_LOAD_TRANSFORMER_INPUTS_DIR=${MAREY_LOAD_TRANSFORMER_INPUTS_DIR:-<unset>}"

# Pipeline selection: DumpMareyPipeline (DumpMixin + MareyPipeline).
export MAREY_PIPELINE_CLASS="DumpMareyPipeline"
export MAREY_DUMP_DIR="${OUT_DIR}"

export HF_HOME="${HF_HOME:-/mnt/localdisk/vllm_omni_hf_cache}"
export VLLM_OMNI_STORAGE_PATH="${VLLM_OMNI_STORAGE_PATH:-/mnt/localdisk/vllm_omni_storage}"
export MODEL="${MODEL:-/app/hf_checkpoints/marey-distilled-0100/}"
export MOONVALLEY_AI_PATH="${MOONVALLEY_AI_PATH:-/home/yizhu/code/moonvalley_ai_master}"

SERVER_LOG="${OUT_DIR}/server.log"
ln -sf "${SERVER_LOG}" "${REPO_ROOT}/marey_30b_server.log"

echo "[phase2] starting server, log: ${SERVER_LOG}"
echo "[phase2]   (also tail-able at ${REPO_ROOT}/marey_30b_server.log)"
bash "${REPO_ROOT}/examples/online_serving/marey/run_server.sh" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup_server() {
    pkill -TERM -f "vllm_omni.entrypoints.cli.main serve" 2>/dev/null || true
    pkill -TERM -f "spawn_main" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "vllm_omni.entrypoints.cli.main serve" 2>/dev/null || true
    pkill -KILL -f "spawn_main" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup_server EXIT

READY_TIMEOUT_S=900
echo "[phase2] waiting for server ready (up to ${READY_TIMEOUT_S}s)..."
for i in $(seq 1 ${READY_TIMEOUT_S}); do
    if grep -q "Application startup complete" "${SERVER_LOG}" 2>/dev/null; then
        echo "[phase2] server ready after ${i}s"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "[phase2] server process exited early. Last 60 lines:"
        tail -n 60 "${SERVER_LOG}" || true
        exit 2
    fi
    if grep -qE "(CUDA out of memory|torch\.OutOfMemoryError|Address already in use|MareyVAEInitializationError|RuntimeError: CUDA error|NCCL error)" "${SERVER_LOG}" 2>/dev/null; then
        echo "[phase2] server log shows a fatal error. Last 60 lines:"
        tail -n 60 "${SERVER_LOG}" || true
        exit 2
    fi
    if (( i % 60 == 0 )); then
        echo "[phase2]   still waiting (${i}s). Last log line:"
        tail -n 1 "${SERVER_LOG}" 2>/dev/null | sed 's/^/[phase2]     /'
    fi
    sleep 1
done
if ! grep -q "Application startup complete" "${SERVER_LOG}" 2>/dev/null; then
    echo "[phase2] server did not become ready within ${READY_TIMEOUT_S}s"
    tail -n 60 "${SERVER_LOG}" || true
    exit 2
fi

echo "[phase2] running client..."
SEED=42 OUTPUT_PATH="${OUT_DIR}/output.mp4" \
    bash "${REPO_ROOT}/examples/online_serving/marey/run_curl_text_to_video.sh" \
    >"${OUT_DIR}/client.log" 2>&1

# DumpMixin writes each request's tensors into ${OUT_DIR}/<auto_req_id>/.
# Vllm-omni's warmup/profile pass creates its own req_* subdir BEFORE the
# real request's video_gen_* subdir, so we pick the one with the most .pt
# files (the real 33-step inference dump) and symlink that to ${TAG}.
REQ_SUBDIR=""
REQ_MAX_PT=0
while IFS= read -r d; do
    [[ -z "${d}" || "$(basename "${d}")" == "${TAG}" ]] && continue
    n_pt=$(find "${d}" -maxdepth 1 -name '*.pt' | wc -l)
    if (( n_pt > REQ_MAX_PT )); then
        REQ_MAX_PT=$n_pt
        REQ_SUBDIR="${d}"
    fi
done < <(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -type d)
if [[ -n "${REQ_SUBDIR}" && "$(basename "${REQ_SUBDIR}")" != "${TAG}" ]]; then
    ln -sfn "$(basename "${REQ_SUBDIR}")" "${OUT_DIR}/${TAG}"
    echo "[phase2] symlink: ${OUT_DIR}/${TAG} -> $(basename "${REQ_SUBDIR}") (${REQ_MAX_PT} .pt files)"
fi

echo "[phase2] done. Artifacts in ${OUT_DIR}:"
ls -lh "${OUT_DIR}" | head -25
if [[ -n "${REQ_SUBDIR}" ]]; then
    N_PT="$(find "${REQ_SUBDIR}" -maxdepth 1 -name '*.pt' | wc -l)"
    echo "[phase2] dump tensors in ${REQ_SUBDIR}: ${N_PT} .pt files"

    # Generate the summary report alongside the dump so every run leaves a
    # durable markdown record — diff between runs to track L1→L2→L3 progress.
    REPORT="${OUT_DIR}/report.md"
    echo "[phase2] generating summary report: ${REPORT}"
    python "${REPO_ROOT}/examples/phase2/summary_report.py" \
        --run "${REQ_SUBDIR}" \
        --ref "${REF_DIR}" \
        --tag "${TAG}" \
        --out "${REPORT}" || echo "[phase2] WARN: summary_report.py failed (continuing)"
fi
