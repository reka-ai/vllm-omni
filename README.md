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
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/offline_inference/marey/text_to_video.py --prompt "cat playing guitar"
```
