#!/usr/bin/env bash
# Phase 3 (I2V): one vllm-omni 30B *image-to-video* run producing a
# DumpMareyPipeline dump dir consumable by compare_dumps.py against
# ${PHASE3_ROOT}/<ref-tag>/.
#
# Mirrors examples/phase2/run_vllm_omni_dump.sh but adds:
#   - I2V-specific env vars MAREY_LOAD_COND_FRAMES_PATH /
#     MAREY_LOAD_COND_OFFSETS_PATH (loaded at every tier; cond_frames /
#     cond_offsets are deterministic pre-loop inputs of preprocessing, not
#     scheduler outputs, so loading them everywhere isolates the diffusion
#     path from the cond-encoding path)
#   - I2V curl request: multipart input_reference (one or more) + frame_indices
#
# Usage:
#   bash examples/phase3_i2v/run_vllm_omni_dump.sh <tag>
#
# Required env vars:
#   COND_IMAGES   — comma-separated list of conditioning image paths
#   FRAME_INDICES — comma-separated list of target frame indices (int)
#
# Tier preset (set LEVEL=L1|L2|L3|base, or set MAREY_LOAD_* directly):
#   L3   initial_z + step_noise + cond_frames + cond_offsets
#   L2   + text_embeds
#   L1   + per-step transformer inputs
#   base (no injection; runs vllm-omni natively)
#
# Optional env vars:
#   PHASE3_ROOT — default /app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v
#   REF_DIR     — default ${PHASE3_ROOT}/ref_single

set -euo pipefail

TAG="${1:?Usage: $0 <tag>  (e.g. vllm_l1_single | vllm_l2_single | vllm_l3_single | vllm_l1_multi)}"

: "${COND_IMAGES:?COND_IMAGES must be a comma-separated list of image paths}"
: "${FRAME_INDICES:?FRAME_INDICES must be a comma-separated list of int indices}"

STORAGE="${STORAGE:-/app/yizhu/marey/vllm_omni/vllm_omni_storage}"
PHASE3_ROOT="${PHASE3_ROOT:-${STORAGE}/phase3_i2v}"
OUT_DIR="${PHASE3_ROOT}/${TAG}"
REF_DIR="${REF_DIR:-${PHASE3_ROOT}/ref_single}"
mkdir -p "${OUT_DIR}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---- Tier preset resolution -----------------------------------------------
# The new I2V load vars (cond_frames + cond_offsets) are loaded at L3, L2,
# AND L1 — they're deterministic preprocessing outputs, not scheduler outputs.
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
        export MAREY_LOAD_COND_FRAMES_PATH="${MAREY_LOAD_COND_FRAMES_PATH:-${REF_DIR}/cond_frames.pt}"
        export MAREY_LOAD_COND_OFFSETS_PATH="${MAREY_LOAD_COND_OFFSETS_PATH:-${REF_DIR}/cond_offsets.pt}"
        export MAREY_LOAD_PIPELINE_LATENTS_DIR="${MAREY_LOAD_PIPELINE_LATENTS_DIR:-${REF_DIR}}"
        ;;
    L1)
        export MAREY_LOAD_INITIAL_NOISE="${MAREY_LOAD_INITIAL_NOISE:-${REF_DIR}/z_initial.pt}"
        export MAREY_LOAD_STEP_NOISE_DIR="${MAREY_LOAD_STEP_NOISE_DIR:-${REF_DIR}}"
        export MAREY_LOAD_TEXT_EMBEDS_DIR="${MAREY_LOAD_TEXT_EMBEDS_DIR:-${REF_DIR}}"
        export MAREY_LOAD_TRANSFORMER_INPUTS_DIR="${MAREY_LOAD_TRANSFORMER_INPUTS_DIR:-${REF_DIR}}"
        export MAREY_LOAD_COND_FRAMES_PATH="${MAREY_LOAD_COND_FRAMES_PATH:-${REF_DIR}/cond_frames.pt}"
        export MAREY_LOAD_COND_OFFSETS_PATH="${MAREY_LOAD_COND_OFFSETS_PATH:-${REF_DIR}/cond_offsets.pt}"
        ;;
    L2)
        export MAREY_LOAD_INITIAL_NOISE="${MAREY_LOAD_INITIAL_NOISE:-${REF_DIR}/z_initial.pt}"
        export MAREY_LOAD_STEP_NOISE_DIR="${MAREY_LOAD_STEP_NOISE_DIR:-${REF_DIR}}"
        export MAREY_LOAD_TEXT_EMBEDS_DIR="${MAREY_LOAD_TEXT_EMBEDS_DIR:-${REF_DIR}}"
        export MAREY_LOAD_COND_FRAMES_PATH="${MAREY_LOAD_COND_FRAMES_PATH:-${REF_DIR}/cond_frames.pt}"
        export MAREY_LOAD_COND_OFFSETS_PATH="${MAREY_LOAD_COND_OFFSETS_PATH:-${REF_DIR}/cond_offsets.pt}"
        ;;
    L3)
        export MAREY_LOAD_INITIAL_NOISE="${MAREY_LOAD_INITIAL_NOISE:-${REF_DIR}/z_initial.pt}"
        export MAREY_LOAD_STEP_NOISE_DIR="${MAREY_LOAD_STEP_NOISE_DIR:-${REF_DIR}}"
        export MAREY_LOAD_COND_FRAMES_PATH="${MAREY_LOAD_COND_FRAMES_PATH:-${REF_DIR}/cond_frames.pt}"
        export MAREY_LOAD_COND_OFFSETS_PATH="${MAREY_LOAD_COND_OFFSETS_PATH:-${REF_DIR}/cond_offsets.pt}"
        ;;
    *)
        echo "[phase3] ERROR: unknown LEVEL=${LEVEL}. Use base | L0 | L1 | L2 | L3."
        exit 2
        ;;
esac

echo "[phase3] vllm-omni I2V run: tag=${TAG}  out=${OUT_DIR}"
echo "[phase3]   COND_IMAGES:   ${COND_IMAGES}"
echo "[phase3]   FRAME_INDICES: ${FRAME_INDICES}"
echo "[phase3]   LEVEL=${LEVEL:-<unset>}  ref_dir=${REF_DIR}"
echo "[phase3]   branch:        $(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
echo "[phase3]   commit:        $(git -C "${REPO_ROOT}" rev-parse HEAD)"
for v in MAREY_LOAD_INITIAL_NOISE MAREY_LOAD_STEP_NOISE_DIR \
         MAREY_LOAD_TEXT_EMBEDS_DIR MAREY_LOAD_TRANSFORMER_INPUTS_DIR \
         MAREY_LOAD_COND_FRAMES_PATH MAREY_LOAD_COND_OFFSETS_PATH; do
    echo "[phase3]   ${v}=${!v:-<unset>}"
done

# Pipeline selection: DumpMareyPipeline (DumpMixin + MareyPipeline).
export MAREY_PIPELINE_CLASS="DumpMareyPipeline"
export MAREY_DUMP_DIR="${OUT_DIR}"
export MAREY_DUMP_BLOCKS_AT_STEPS="${MAREY_DUMP_BLOCKS_AT_STEPS:-0}"

export HF_HOME="${HF_HOME:-/mnt/localdisk/vllm_omni_hf_cache}"
export VLLM_OMNI_STORAGE_PATH="${VLLM_OMNI_STORAGE_PATH:-/mnt/localdisk/vllm_omni_storage}"
export MODEL="${MODEL:-/app/hf_checkpoints/marey-distilled-0100/}"
export MOONVALLEY_AI_PATH="${MOONVALLEY_AI_PATH:-/home/yizhu/code/moonvalley_ai_master}"

SERVER_LOG="${OUT_DIR}/server.log"
ln -sf "${SERVER_LOG}" "${REPO_ROOT}/marey_30b_server.log"

echo "[phase3] starting server, log: ${SERVER_LOG}"
echo "[phase3]   (also tail-able at ${REPO_ROOT}/marey_30b_server.log)"
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
echo "[phase3] waiting for server ready (up to ${READY_TIMEOUT_S}s)..."
for i in $(seq 1 ${READY_TIMEOUT_S}); do
    if grep -q "Application startup complete" "${SERVER_LOG}" 2>/dev/null; then
        echo "[phase3] server ready after ${i}s"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "[phase3] server process exited early. Last 60 lines:"
        tail -n 60 "${SERVER_LOG}" || true
        exit 2
    fi
    if grep -qE "(CUDA out of memory|torch\.OutOfMemoryError|Address already in use|MareyVAEInitializationError|RuntimeError: CUDA error|NCCL error)" "${SERVER_LOG}" 2>/dev/null; then
        echo "[phase3] server log shows a fatal error. Last 60 lines:"
        tail -n 60 "${SERVER_LOG}" || true
        exit 2
    fi
    if (( i % 60 == 0 )); then
        echo "[phase3]   still waiting (${i}s). Last log line:"
        tail -n 1 "${SERVER_LOG}" 2>/dev/null | sed 's/^/[phase3]     /'
    fi
    sleep 1
done
if ! grep -q "Application startup complete" "${SERVER_LOG}" 2>/dev/null; then
    echo "[phase3] server did not become ready within ${READY_TIMEOUT_S}s"
    tail -n 60 "${SERVER_LOG}" || true
    exit 2
fi

# ---- I2V curl request ------------------------------------------------------
SEED="${SEED:-42}"
PROMPT="${PROMPT:-$(cat "${STORAGE}/i2v_prompt.txt")}"
NEG_PROMPT="${NEG_PROMPT:-$(cat "${STORAGE}/i2v_negative_prompt.txt")}"
BASE_URL="${BASE_URL:-http://localhost:8098}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"

# Build frame_conditions JSON dict (OpenAI chat-completions nested image_url
# schema) from COND_IMAGES + FRAME_INDICES. Each entry's URL is a
# ``file://`` URI pointing at a local image (server reads from filesystem).
# Switch to ``data:image/...;base64,...`` for non-local servers, or
# ``https://...`` for remote-hosted images.
FRAME_CONDITIONS_JSON="$(python3 -c "
import json, os
from urllib.request import pathname2url
imgs = '${COND_IMAGES}'.split(',')
idxs = [int(x) for x in '${FRAME_INDICES}'.split(',')]
assert len(imgs) == len(idxs), f'mismatch: {len(imgs)} images vs {len(idxs)} indices'
out = {}
for idx, path in zip(idxs, imgs):
    assert os.path.exists(path), f'cond image not found: {path}'
    out[str(idx)] = {
        'image_url': {'url': 'file://' + pathname2url(os.path.abspath(path)), 'detail': 'auto'}
    }
print(json.dumps(out))
")"

CURL_ARGS=(
  -sS -X POST "${BASE_URL}/v1/videos"
  -H "Accept: application/json"
  -F "prompt=${PROMPT}"
  -F "negative_prompt=${NEG_PROMPT}"
  -F "size=1920x1080"
  -F "num_frames=128"
  -F "num_inference_steps=33"
  -F "guidance_scale=3.5"
  -F "seed=${SEED}"
  -F "frame_conditions=${FRAME_CONDITIONS_JSON}"
)

echo "[phase3] running I2V client..."
create_response="$(curl "${CURL_ARGS[@]}")"
echo "${create_response}" | tee "${OUT_DIR}/create_response.json" >/dev/null
video_id="$(echo "${create_response}" | jq -r '.id')"
if [[ -z "${video_id}" || "${video_id}" == "null" ]]; then
    echo "[phase3] failed to create video job:"
    echo "${create_response}" | jq .
    exit 1
fi
echo "[phase3] video_id=${video_id}"

while true; do
    status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
    status="$(echo "${status_response}" | jq -r '.status')"
    case "${status}" in
        queued|in_progress)
            echo "[phase3] status: ${status}"
            sleep "${POLL_INTERVAL}"
            ;;
        completed)
            echo "${status_response}" | jq . >"${OUT_DIR}/status_response.json"
            break
            ;;
        failed)
            echo "[phase3] video generation failed:"
            echo "${status_response}" | jq .
            exit 1
            ;;
        *)
            echo "[phase3] unexpected status: ${status_response}"
            exit 1
            ;;
    esac
done
curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUT_DIR}/output.mp4"

# DumpMixin writes per-request tensors into ${OUT_DIR}/<auto_req_id>/. Pick
# the subdir with the most .pt files (the real inference dump, not warmup).
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
    echo "[phase3] symlink: ${OUT_DIR}/${TAG} -> $(basename "${REQ_SUBDIR}") (${REQ_MAX_PT} .pt files)"
fi

echo "[phase3] done. Artifacts in ${OUT_DIR}:"
ls -lh "${OUT_DIR}" | head -25
if [[ -n "${REQ_SUBDIR}" ]]; then
    N_PT="$(find "${REQ_SUBDIR}" -maxdepth 1 -name '*.pt' | wc -l)"
    echo "[phase3] dump tensors in ${REQ_SUBDIR}: ${N_PT} .pt files"

    REPORT="${OUT_DIR}/report.md"
    echo "[phase3] generating summary report: ${REPORT}"
    python "${REPO_ROOT}/examples/phase3_i2v/summary_report.py" \
        --run "${REQ_SUBDIR}" \
        --ref "${REF_DIR}" \
        --tag "${TAG}" \
        --out "${REPORT}" || echo "[phase3] WARN: summary_report.py failed (continuing)"
fi
