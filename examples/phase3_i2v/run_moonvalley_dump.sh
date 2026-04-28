#!/usr/bin/env bash
# Phase 3 (I2V): moonvalley_ai 30B *image-to-video* inference on
# marey-serving-comparison branch with full dump instrumentation. Produces a
# reference dump dir consumable by the vllm-omni-side L1/L2/L3 runs and by
# examples/offline_inference/marey/compare_dumps.py.
#
# Mirrors examples/phase1/run_moonvalley_dump.sh but adds:
#   - --frame-conditions JSON dict (CLI flag added on marey-serving-comparison)
#   - I2V-specific dumps written by marey_reference_inference_dump.py
#     (cond_frames, cond_offsets, t0_emb, x_after_concat, x_t_mask, x_pre_slice)
#
# Usage:
#   bash examples/phase3_i2v/run_moonvalley_dump.sh <tag>
#
# <tag> is the subdir name under $PHASE3_ROOT, e.g. "ref_single" or
# "ref_multi". The mv wrapper writes its dump to ${PHASE3_ROOT}/${tag}.
#
# I2V-specific env vars (required):
#   COND_IMAGES   — comma-separated list of conditioning image paths
#   FRAME_INDICES — comma-separated list of target frame indices (int)
#
# Optional env vars:
#   PHASE3_ROOT   — default /app/yizhu/marey/vllm_omni/vllm_omni_storage/phase3_i2v
#   SEED          — default 42
#   PROMPT        — default reads from <STORAGE>/i2v_prompt.txt
#   NEG_PROMPT    — default reads from <STORAGE>/i2v_negative_prompt.txt
#   STORAGE       — default /app/yizhu/marey/vllm_omni/vllm_omni_storage

set -euo pipefail

TAG="${1:?Usage: $0 <tag>  (e.g. ref_single | ref_multi)}"

: "${COND_IMAGES:?COND_IMAGES must be a comma-separated list of image paths}"
: "${FRAME_INDICES:?FRAME_INDICES must be a comma-separated list of int indices}"

STORAGE="${STORAGE:-/app/yizhu/marey/vllm_omni/vllm_omni_storage}"
PHASE3_ROOT="${PHASE3_ROOT:-${STORAGE}/phase3_i2v}"
OUT_DIR="${PHASE3_ROOT}/${TAG}"
mkdir -p "${OUT_DIR}"

MV_REPO="${MV_REPO:-/home/yizhu/code/moonvalley_ai_master}"
MV_VENV="${MV_VENV:-${MV_REPO}/inference-service/.venv}"
MV_TORCHRUN="${MV_VENV}/bin/torchrun"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="${REPO_ROOT}/examples/online_serving/marey/marey_reference_inference_dump.py"

if [[ ! -x "${MV_TORCHRUN}" ]]; then
    echo "[phase3] ERROR: ${MV_TORCHRUN} not found or not executable"
    exit 1
fi
if [[ ! -f "${WRAPPER}" ]]; then
    echo "[phase3] ERROR: ${WRAPPER} not found"
    exit 1
fi

SEED="${SEED:-42}"
PROMPT="${PROMPT:-$(cat "${STORAGE}/i2v_prompt.txt")}"
NEG_PROMPT="${NEG_PROMPT:-$(cat "${STORAGE}/i2v_negative_prompt.txt")}"

# Build mv's --frame-conditions JSON from COND_IMAGES + FRAME_INDICES.
#   {"0": {"image_path": "/path/a.png"}, "127": {"image_path": "/path/b.png"}}
FRAME_CONDITIONS_JSON="$(python -c "
import json, os, sys
imgs = '${COND_IMAGES}'.split(',')
idxs = [int(x) for x in '${FRAME_INDICES}'.split(',')]
assert len(imgs) == len(idxs), f'mismatch: {len(imgs)} images vs {len(idxs)} indices'
for p in imgs:
    assert os.path.exists(p), f'cond image not found: {p}'
print(json.dumps({str(idx): {'image_path': path} for idx, path in zip(idxs, imgs)}))
")"

echo "[phase3] mv I2V dump run: tag=${TAG}  out=${OUT_DIR}"
echo "[phase3]   COND_IMAGES:   ${COND_IMAGES}"
echo "[phase3]   FRAME_INDICES: ${FRAME_INDICES}"
echo "[phase3]   --frame-conditions: ${FRAME_CONDITIONS_JSON}"
echo "[phase3]   SEED:          ${SEED}"
echo "[phase3]   branch:        $(git -C "${MV_REPO}" rev-parse --abbrev-ref HEAD)"
echo "[phase3]   commit:        $(git -C "${MV_REPO}" rev-parse HEAD)"

# vae.cp_path is relative; CWD must hold open_sora/configs/model/*.yaml. The
# moonvalley repo root works for both.
MODEL_FOLDER="${MODEL_FOLDER:-/app/hf_checkpoints/marey-distilled-0100}"
VAE_SYMLINK="${MV_REPO}/vae.ckpt"
if [[ ! -e "${VAE_SYMLINK}" ]]; then
    ln -sf "${MODEL_FOLDER}/vae.ckpt" "${VAE_SYMLINK}"
    cleanup_vae_symlink() { rm -f "${VAE_SYMLINK}"; }
    trap cleanup_vae_symlink EXIT
fi
cd "${MV_REPO}"

# flash_attn_3 wheels need GLIBCXX_3.4.32+; preload miniconda's libstdc++.
LIBSTDCXX_PRELOAD="${LIBSTDCXX_PRELOAD:-/home/yizhu/miniconda3/lib/libstdc++.so.6}"
if [[ -e "${LIBSTDCXX_PRELOAD}" ]]; then
    echo "[phase3] preloading libstdc++ from: ${LIBSTDCXX_PRELOAD}"
    export LD_PRELOAD="${LIBSTDCXX_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
else
    echo "[phase3] WARN: ${LIBSTDCXX_PRELOAD} not found; flash_attn_3 may fail"
fi

# Dump routing.
export MAREY_DUMP_DIR="${PHASE3_ROOT}"
export MAREY_DUMP_REQUEST_ID="${TAG}"
export MAREY_DUMP_BLOCKS_AT_STEPS="${MAREY_DUMP_BLOCKS_AT_STEPS:-0}"
export MOONVALLEY_AI_PATH="${MV_REPO}"

# DROPS --use-timestep-transform (project memory: switch-pair on cmp branch,
# default True is what we want; passing it would invert).
PYTHONPATH="${MV_REPO}/inference-service:${MV_REPO}:${MV_REPO}/open_sora" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${MV_TORCHRUN}" --nproc_per_node=8 \
    "${WRAPPER}" infer \
    --num-seq-parallel-splits 8 \
    --offload-diffusion --offload-vae --offload-text-encoder \
    --model-folder "${MODEL_FOLDER}" \
    --checkpoint-folder "${MODEL_FOLDER}" \
    --watermarker-path "/app/wlam/models/checkpoints/marey/videoseal/y_256b_img.jit" \
    --frame-conditions "${FRAME_CONDITIONS_JSON}" \
    --height 1080 --width 1920 --num-frames 128 --fps 24 \
    --steps 33 --guidance-scale 3.5 --disable-caching \
    --use-negative-prompts \
    --negative-prompt "${NEG_PROMPT}" \
    --use-distilled-steps --shift-value 3.0 \
    --use-guidance-schedule --add-quality-guidance --clip-value 10.0 \
    --seed "${SEED}" --warmup-steps 4 --cooldown-steps 18 \
    --save-latents \
    --output "${OUT_DIR}/output.mp4" \
    "${PROMPT}" 2>&1 | tee "${OUT_DIR}/run.log"

echo "[phase3] done. Artifacts in ${OUT_DIR}:"
ls -lh "${OUT_DIR}" | head -25
N_PT="$(find "${OUT_DIR}" -maxdepth 1 -name '*.pt' | wc -l)"
echo "[phase3] dump tensors: ${N_PT} .pt files"
