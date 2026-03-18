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

## Usage

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/offline_inference/marey/text_to_video.py \
  --prompt "cat playing guitar" \
  --output output_$(date +%Y%m%d_%H%M%S).mp4 \
  --height 1080 \
  --width 1920 \
  --num-frames 17
```

Faster command:
```bash
source .venv/bin/activate && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=8 examples/offline_inference/marey/text_to_video.py \
  --tp 8 \
  --steps 10 \
  --num-frames 17 \
  --height 360 --width 640 \
  --no-cfg \
  --prompt "cat playing guitar"
```