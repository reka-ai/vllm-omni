#!/bin/bash
# Marey MMDiT online serving startup script.
#
# The model directory must contain a config.yaml with text_encoder, vae, model,
# and scheduler sections (see examples/offline_inference/marey/text_to_video.py).
#
# Since the model directory does not have a model_index.json, we must explicitly
# pass --model-class-name MareyPipeline so that vllm-omni recognises it as a
# diffusion model and loads the correct pipeline.

MODEL="${MODEL:-/home/aormazabal/wlam/wlam-inference/vllm-omni/ckpts/marey_distilled-0001}"
PORT="${PORT:-8098}"
FLOW_SHIFT="${FLOW_SHIFT:-3.0}"

echo "Starting Marey server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Flow shift: $FLOW_SHIFT"
# MAREY_DUMP_DIR="/home/aormazabal/wlam/wlam-inference/scratch/pipeline_dump/" \
# MAREY_LOAD_INITIAL_NOISE="/home/aormazabal/wlam/wlam-inference/moonvalley_ai/scratch/dumped/z_initial_noise.pt" \
# MAREY_LOAD_STEP_NOISE_DIR="/home/aormazabal/wlam/wlam-inference/moonvalley_ai/scratch/dumped/" \
HF_HOME="/home/aormazabal/wlam/wlam-inference/scratch/hf_home" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_OMNI_STORAGE_PATH="/home/aormazabal/wlam/wlam-inference/vllm-omni/storage" \
uv run --project /home/aormazabal/wlam/wlam-inference/vllm-omni vllm-omni serve "$MODEL" --omni \
    --port "$PORT" \
    --model-class-name MareyPipeline \
    --flow-shift "$FLOW_SHIFT" \
    --gpu-memory-utilization 0.98 \
    --ulysses-degree 8
