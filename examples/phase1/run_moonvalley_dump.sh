#!/usr/bin/env bash
# Phase 1: moonvalley 30B inference on marey-serving-comparison branch with
# full dump instrumentation. Produces a reference dump dir consumable by
# Phase 2 (vllm-omni injection runs) and by examples/offline_inference/marey/
# compare_dumps.py.
#
# Wraps marey_reference_inference_dump.py instead of marey_inference.py so:
#   - torch.randn_like is monkey-patched (initial noise + per-step noise)
#   - text_encoder.encode / text_encoder.null are wrapped (cond/uncond dump
#     and id() capture for B3 identity-based labeling)
#   - model() calls are buffered and labeled (cond/uncond via identity,
#     order fallback with warning)
#
# Usage:
#   bash examples/phase1/run_moonvalley_dump.sh <tag>
#
# <tag> is the subdir name under $PHASE1_ROOT, e.g. "ref_30b" for the
# canonical reference or "ref_30b_repro" for the reproducibility check.

set -euo pipefail

TAG="${1:?Usage: $0 <tag>  (e.g. ref_30b | ref_30b_repro)}"

PHASE1_ROOT="${PHASE1_ROOT:-/mnt/localdisk/vllm_omni_storage/phase1}"
OUT_DIR="${PHASE1_ROOT}/${TAG}"
mkdir -p "${OUT_DIR}"

MV_REPO="${MV_REPO:-/home/yizhu/code/moonvalley_ai_master}"
MV_VENV="${MV_VENV:-${MV_REPO}/inference-service/.venv}"
MV_TORCHRUN="${MV_VENV}/bin/torchrun"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="${REPO_ROOT}/examples/online_serving/marey/marey_reference_inference_dump.py"

if [[ ! -x "${MV_TORCHRUN}" ]]; then
    echo "[phase1] ERROR: ${MV_TORCHRUN} not found or not executable"
    exit 1
fi
if [[ ! -f "${WRAPPER}" ]]; then
    echo "[phase1] ERROR: ${WRAPPER} not found"
    exit 1
fi

echo "[phase1] moonvalley dump run: tag=${TAG}  out=${OUT_DIR}"
echo "[phase1] venv:    ${MV_VENV}"
echo "[phase1] wrapper: ${WRAPPER}"
echo "[phase1] branch:  $(git -C "${MV_REPO}" rev-parse --abbrev-ref HEAD)"
echo "[phase1] commit:  $(git -C "${MV_REPO}" rev-parse HEAD)"

# moonvalley resolves `vae.cp_path: vae.ckpt` (relative) against CWD. On the
# comparison branch this is absolutized in marey_inference.py, but CWD must
# still hold `open_sora/configs/model/*.yaml` (used by maybe_update_model_cfg).
# The moonvalley repo root satisfies both.
MODEL_FOLDER="/app/hf_checkpoints/marey-distilled-0100"
VAE_SYMLINK="${MV_REPO}/vae.ckpt"
if [[ ! -e "${VAE_SYMLINK}" ]]; then
    ln -sf "${MODEL_FOLDER}/vae.ckpt" "${VAE_SYMLINK}"
    cleanup_vae_symlink() { rm -f "${VAE_SYMLINK}"; }
    trap cleanup_vae_symlink EXIT
fi
cd "${MV_REPO}"

# flash_attn_3 wheels need GLIBCXX_3.4.32+; Ubuntu 22.04's system libstdc++
# tops out at 3.4.30. Preload the miniconda copy.
LIBSTDCXX_PRELOAD="${LIBSTDCXX_PRELOAD:-/home/yizhu/miniconda3/lib/libstdc++.so.6}"
if [[ -e "${LIBSTDCXX_PRELOAD}" ]]; then
    echo "[phase1] preloading libstdc++ from: ${LIBSTDCXX_PRELOAD}"
    export LD_PRELOAD="${LIBSTDCXX_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
else
    echo "[phase1] WARN: ${LIBSTDCXX_PRELOAD} not found; flash_attn_3 may fail"
fi

PROMPT="Detailed Description: A majestic, aged eagle with mottled golden-brown feathers soars gracefully through a vast, ancient indoor chamber. Its expansive wings barely flap, catching the air as it glides effortlessly between towering stone pillars adorned with glinting metallic accents. Beams of morning light pierce the gloom, filtering through a cracked skylight high above and illuminating swirling dust motes in their path. The camera pans smoothly, following the eagle's silent flight as it navigates the cavernous space, its sharp eyes scanning the stone floor below, creating a scene of serene power and timeless solitude. Background: The far reaches of the chamber fade into deep shadow, with the silhouettes of distant pillars barely visible. High above, a cracked skylight serves as the primary light source, its fractured glass creating distinct rays of light. Middleground: The aged eagle glides on a steady path, its mottled golden-brown wings spread wide. It passes through the dramatic beams of light, which highlight the intricate details of its feathers and the dust particles dancing in the air. Foreground: The camera looks up from a low angle, tracking the eagle's movement across the expansive stone floor, which is patterned with the bright shafts of light and deep shadows cast by the pillars."

NEG_PROMPT="<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, bright, vignette, artifacts, still, noise, texture, scanlines, videogame, 360 camera, VR, transition, flare, saturation, distorted, warped, wide angle, contrast, saturated, vibrant, glowing, cross dissolve, texture, videogame, saturation, cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, blown out, horrible, blurry, worst quality, bad, transition, dissolve, cross-dissolve, melt, fade in, fade out, wobbly, weird, low quality, plastic, stock footage, video camera, boring, static"

# Dump routing:
#   MAREY_DUMP_DIR       — root of the dump tree the wrapper writes into
#   MAREY_DUMP_REQUEST_ID — subdir name; use the tag so multiple runs coexist
export MAREY_DUMP_DIR="${PHASE1_ROOT}"
export MAREY_DUMP_REQUEST_ID="${TAG}"
export MOONVALLEY_AI_PATH="${MV_REPO}"

# Note: no --use-timestep-transform (see Phase 0 lessons — on comparison branch
# it's a switch pair that explicitly sets True, but also the default is True).
# --offload-* flags are explicit on the comparison branch (they don't need
# the Phase 0 shim's monkey-patch here).
PYTHONPATH="${MV_REPO}/inference-service:${MV_REPO}:${MV_REPO}/open_sora" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${MV_TORCHRUN}" --nproc_per_node=8 \
    "${WRAPPER}" infer \
    --num-seq-parallel-splits 8 \
    --offload-diffusion --offload-vae --offload-text-encoder \
    --model-folder "${MODEL_FOLDER}" \
    --checkpoint-folder "${MODEL_FOLDER}" \
    --watermarker-path "/app/wlam/models/checkpoints/marey/videoseal/y_256b_img.jit" \
    --height 1080 --width 1920 --num-frames 128 --fps 24 \
    --steps 33 --guidance-scale 3.5 --disable-caching \
    --use-negative-prompts \
    --negative-prompt "${NEG_PROMPT}" \
    --use-distilled-steps --shift-value 3.0 \
    --use-guidance-schedule --add-quality-guidance --clip-value 10.0 \
    --seed 42 --warmup-steps 4 --cooldown-steps 18 \
    --save-latents \
    --output "${OUT_DIR}/output.mp4" \
    "${PROMPT}" 2>&1 | tee "${OUT_DIR}/run.log"

echo "[phase1] done. Artifacts in ${OUT_DIR}:"
ls -lh "${OUT_DIR}" | head -25
