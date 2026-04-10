#!/usr/bin/env bash
set -euo pipefail

PROMPTS_FILE="/home/david/repos/vllm-omni/prompts.txt"
PROJECT_DIR="/home/david/repos/vllm-omni"
SCRIPT_PATH="/home/david/repos/vllm-omni/examples/offline_inference/marey/text_to_video.py"
OUTPUT_DIR="/home/david/repos/vllm-omni"

# Put the GPU ids you want to use here.
# Example for 8 GPUs:
GPUS=(0 1 2 3 4 5 6 7)

mapfile -t prompts < "$PROMPTS_FILE"

num_prompts="${#prompts[@]}"
num_gpus="${#GPUS[@]}"

if [[ "$num_prompts" -eq 0 ]]; then
  echo "No prompts found in $PROMPTS_FILE"
  exit 1
fi

echo "Found $num_prompts prompts"
echo "Using GPUs: ${GPUS[*]}"

pids=()

for i in "${!prompts[@]}"; do
  prompt="${prompts[$i]}"

  # Skip empty lines
  if [[ -z "${prompt// }" ]]; then
    continue
  fi

  gpu="${GPUS[$((i % num_gpus))]}"
  line_number=$((i + 1))
  timestamp="$(date +%Y%m%d_%H%M%S)"
  output_file="${OUTPUT_DIR}/vllm_omni_output_${timestamp}_${line_number}.mp4"

  echo "Launching line ${line_number} on GPU ${gpu}"

  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --project "${PROJECT_DIR}" python "${SCRIPT_PATH}" \
      --output "${output_file}" \
      --height 1080 \
      --width 1920 \
      --num-frames 128 \
      --steps 33 \
      --guidance-scale 3.5 \
      --prompt "${prompt}"
  ) &

  pids+=("$!")
done

# Wait for all jobs
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "All jobs finished."
