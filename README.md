# vllm-omni

## Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install vllm==0.17.0 --torch-backend=auto
# Install Open-Sora in editable mode (https://github.com/reka-ai/moonvalley_ai/tree/main/open_sora)
# I ported this over from the original repo to avoid the dependency issues.
uv pip install -e ../moonvalley_ai/open_sora --no-deps
```

## Text to Video Usage with vllm-omni

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --project /home/david/repos/vllm-omni \
python /home/david/repos/vllm-omni/examples/offline_inference/marey/text_to_video.py \
  --prompt "a woman demonstrates a beauty product, holding a golden cream jar in one hand and gesturing delicately with the other illustration style" \
  --output /home/david/repos/vllm-omni/vllm_omni_output_$(date +%Y%m%d_%H%M%S).mp4 \
  --height 1080 \
  --width 1920 \
  --num-frames 1
```

## Marey Inference Usage

```bash
PYTHONPATH=/home/david/repos/moonvalley_ai \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --project /home/david/repos/moonvalley_ai/inference-service \
torchrun --nproc_per_node=8 /home/david/repos/moonvalley_ai/inference-service/marey_inference.py infer \
  --num-seq-parallel-splits 8 \
  --offload-diffusion \
  --offload-vae \
  --offload-text-encoder \
  --model-folder "/app/wlam/models/checkpoints/marey/distilled-0001" \
  --checkpoint-folder "/app/wlam/models/checkpoints/marey/distilled-0001/epoch0-global_step7000_distilled" \
  --watermarker-path "/app/wlam/models/checkpoints/marey/videoseal/y_256b_img.jit" \
  --height 1080 \
  --width 1920 \
  --num-frames 32 \
  --steps 100 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output /home/david/repos/vllm-omni/marey_inference_output_$(date +%Y%m%d_%H%M%S).mp4 \
  "a woman demonstrates a beauty product, holding a golden cream jar in one hand and gesturing delicately with the other illustration style"
```

```bash
PYTHONPATH=/home/david/repos/moonvalley_ai \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --project /home/david/repos/moonvalley_ai/inference-service \
torchrun --nproc_per_node=8 /home/david/repos/moonvalley_ai/inference-service/marey_inference.py infer \
  --num-seq-parallel-splits 8 \
  --offload-diffusion \
  --offload-vae \
  --offload-text-encoder \
  --model-folder "/app/wlam/models/checkpoints/marey/distilled-0001" \
  --checkpoint-folder "/app/wlam/models/checkpoints/marey/distilled-0001/epoch0-global_step7000_distilled" \
  --watermarker-path "/app/wlam/models/checkpoints/marey/videoseal/y_256b_img.jit" \
  --output simple_marey_inference_output_$(date +%Y%m%d_%H%M%S).mp4 \
  "a woman demonstrates a beauty product, holding a golden cream jar in one hand and gesturing delicately with the other illustration style"
```