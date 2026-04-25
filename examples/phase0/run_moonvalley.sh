#!/usr/bin/env bash
# Phase 0: one moonvalley_ai 30B inference run producing latents.pt + output.mp4.
#
# Usage:
#   bash examples/phase0/run_moonvalley.sh <tag>
#
# <tag> is any label for the run dir, e.g. "mv_main" or "mv_cmp".
# Caller is responsible for having checked out the branch under test on
# /home/yizhu/code/moonvalley_ai_master and stash-popped any WIP back
# (text-to-video ignores WIP additions anyway).

set -euo pipefail

TAG="${1:?Usage: $0 <tag>  (e.g. mv_main | mv_cmp)}"

PHASE0_ROOT="${PHASE0_ROOT:-/mnt/localdisk/vllm_omni_storage/phase0/20260424}"
OUT_DIR="${PHASE0_ROOT}/${TAG}"
mkdir -p "${OUT_DIR}"

MV_REPO="${MV_REPO:-/home/yizhu/code/moonvalley_ai_master}"
MV_VENV="${MV_VENV:-${MV_REPO}/inference-service/.venv}"
MV_TORCHRUN="${MV_VENV}/bin/torchrun"

if [[ ! -x "${MV_TORCHRUN}" ]]; then
    echo "[phase0] ERROR: ${MV_TORCHRUN} not found or not executable"
    exit 1
fi

echo "[phase0] moonvalley run: tag=${TAG}  out=${OUT_DIR}"
echo "[phase0] venv: ${MV_VENV}"
echo "[phase0] branch: $(git -C "${MV_REPO}" rev-parse --abbrev-ref HEAD)"
echo "[phase0] commit: $(git -C "${MV_REPO}" rev-parse HEAD)"

# Same 30B canonical CLI as /home/yizhu/code/vllm-omni/README.md line 102-133,
# with --model-folder / --checkpoint-folder pointed at the unified ckpt path.
PROMPT="Detailed Description: A majestic, aged eagle with mottled golden-brown feathers soars gracefully through a vast, ancient indoor chamber. Its expansive wings barely flap, catching the air as it glides effortlessly between towering stone pillars adorned with glinting metallic accents. Beams of morning light pierce the gloom, filtering through a cracked skylight high above and illuminating swirling dust motes in their path. The camera pans smoothly, following the eagle's silent flight as it navigates the cavernous space, its sharp eyes scanning the stone floor below, creating a scene of serene power and timeless solitude. Background: The far reaches of the chamber fade into deep shadow, with the silhouettes of distant pillars barely visible. High above, a cracked skylight serves as the primary light source, its fractured glass creating distinct rays of light. Middleground: The aged eagle glides on a steady path, its mottled golden-brown wings spread wide. It passes through the dramatic beams of light, which highlight the intricate details of its feathers and the dust particles dancing in the air. Foreground: The camera looks up from a low angle, tracking the eagle's movement across the expansive stone floor, which is patterned with the bright shafts of light and deep shadows cast by the pillars."

NEG_PROMPT="<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, bright, vignette, artifacts, still, noise, texture, scanlines, videogame, 360 camera, VR, transition, flare, saturation, distorted, warped, wide angle, contrast, saturated, vibrant, glowing, cross dissolve, texture, videogame, saturation, cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, blown out, horrible, blurry, worst quality, bad, transition, dissolve, cross-dissolve, melt, fade in, fade out, wobbly, weird, low quality, plastic, stock footage, video camera, boring, static"

# Moonvalley resolves two distinct relative paths against CWD:
#   - `open_sora/configs/model/{type}.yaml` (maybe_update_model_cfg, ckpt_utils.py)
#   - `vae.cp_path: vae.ckpt` (config.yaml; main does NOT absolutize — that's a
#     fix on marey-serving-comparison)
# Both must resolve from a single CWD. The only CWD that works for (1) is the
# moonvalley repo root. For (2), symlink vae.ckpt into the repo root pointing
# at the real file. Harmless for cmp (it absolutizes anyway so the symlink is
# never consulted). Cleaned up on exit.
MODEL_FOLDER="/app/hf_checkpoints/marey-distilled-0100"
VAE_SYMLINK="${MV_REPO}/vae.ckpt"
if [[ ! -e "${VAE_SYMLINK}" ]]; then
    ln -sf "${MODEL_FOLDER}/vae.ckpt" "${VAE_SYMLINK}"
    cleanup_vae_symlink() { rm -f "${VAE_SYMLINK}"; }
    trap cleanup_vae_symlink EXIT
fi
cd "${MV_REPO}"

# Use the Phase 0 shim as the torchrun entrypoint so MareyInference.__init__
# gets offload_* kwargs forced on (main doesn't expose them via CLI; 30B OOMs
# without offload on 8x80GB). The shim then delegates to marey_inference.app().
SHIM="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mv_inference_shim.py"

# flash_attn_3 prebuilt wheels need GLIBCXX_3.4.32+, which Ubuntu 22.04's
# system libstdc++ doesn't have. Preload a newer one (same workaround as
# examples/online_serving/marey/marey_30b_reference_inference.sh).
LIBSTDCXX_PRELOAD="${LIBSTDCXX_PRELOAD:-/home/yizhu/miniconda3/lib/libstdc++.so.6}"
if [[ -e "${LIBSTDCXX_PRELOAD}" ]]; then
    echo "[phase0] preloading libstdc++ from: ${LIBSTDCXX_PRELOAD}"
    export LD_PRELOAD="${LIBSTDCXX_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
else
    echo "[phase0] WARN: ${LIBSTDCXX_PRELOAD} not found; flash_attn_3 import may fail"
fi

PYTHONPATH="${MV_REPO}/inference-service:${MV_REPO}:${MV_REPO}/open_sora" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${MV_TORCHRUN}" --nproc_per_node=8 \
    "${SHIM}" infer \
    --num-seq-parallel-splits 8 \
    --model-folder "/app/hf_checkpoints/marey-distilled-0100" \
    --checkpoint-folder "/app/hf_checkpoints/marey-distilled-0100" \
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

echo "[phase0] done. Artifacts in ${OUT_DIR}:"
ls -lh "${OUT_DIR}"
