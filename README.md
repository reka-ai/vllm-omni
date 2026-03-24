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
  --num-frames 32 \
  --prompt "Detailed Description: A majestic, aged eagle with mottled golden-brown feathers soars gracefully through a vast, ancient indoor chamber. Its expansive wings barely flap, catching the air as it glides effortlessly between towering stone pillars adorned with glinting metallic accents. Beams of morning light pierce the gloom, filtering through a cracked skylight high above and illuminating swirling dust motes in their path. The camera pans smoothly, following the eagle's silent flight as it navigates the cavernous space, its sharp eyes scanning the stone floor below, creating a scene of serene power and timeless solitude. Background: The far reaches of the chamber fade into deep shadow, with the silhouettes of distant pillars barely visible. High above, a cracked skylight serves as the primary light source, its fractured glass creating distinct rays of light. Middleground: The aged eagle glides on a steady path, its mottled golden-brown wings spread wide. It passes through the dramatic beams of light, which highlight the intricate details of its feathers and the dust particles dancing in the air. Foreground: The camera looks up from a low angle, tracking the eagle's movement across the expansive stone floor, which is patterned with the bright shafts of light and deep shadows cast by the pillars."
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --project /home/david/repos/vllm-omni \
python /home/david/repos/vllm-omni/examples/offline_inference/marey/text_to_video.py \
  --prompt "a woman demonstrates a beauty product, holding a golden cream jar in one hand and gesturing delicately with the other illustration style" \
  --output /home/david/repos/vllm-omni/vllm_omni_output_$(date +%Y%m%d_%H%M%S).mp4 \
  --height 1080 \
  --width 1920 \
  --num-frames 128 \
  --steps 33 \
  --guidance-scale 3.5 \
  --prompt "Detailed Description: A majestic, aged eagle with mottled golden-brown feathers soars gracefully through a vast, ancient indoor chamber. Its expansive wings barely flap, catching the air as it glides effortlessly between towering stone pillars adorned with glinting metallic accents. Beams of morning light pierce the gloom, filtering through a cracked skylight high above and illuminating swirling dust motes in their path. The camera pans smoothly, following the eagle's silent flight as it navigates the cavernous space, its sharp eyes scanning the stone floor below, creating a scene of serene power and timeless solitude. Background: The far reaches of the chamber fade into deep shadow, with the silhouettes of distant pillars barely visible. High above, a cracked skylight serves as the primary light source, its fractured glass creating distinct rays of light. Middleground: The aged eagle glides on a steady path, its mottled golden-brown wings spread wide. It passes through the dramatic beams of light, which highlight the intricate details of its feathers and the dust particles dancing in the air. Foreground: The camera looks up from a low angle, tracking the eagle's movement across the expansive stone floor, which is patterned with the bright shafts of light and deep shadows cast by the pillars."
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
  "Detailed Description: A majestic, aged eagle with mottled golden-brown feathers soars gracefully through a vast, ancient indoor chamber. Its expansive wings barely flap, catching the air as it glides effortlessly between towering stone pillars adorned with glinting metallic accents. Beams of morning light pierce the gloom, filtering through a cracked skylight high above and illuminating swirling dust motes in their path. The camera pans smoothly, following the eagle's silent flight as it navigates the cavernous space, its sharp eyes scanning the stone floor below, creating a scene of serene power and timeless solitude. Background: The far reaches of the chamber fade into deep shadow, with the silhouettes of distant pillars barely visible. High above, a cracked skylight serves as the primary light source, its fractured glass creating distinct rays of light. Middleground: The aged eagle glides on a steady path, its mottled golden-brown wings spread wide. It passes through the dramatic beams of light, which highlight the intricate details of its feathers and the dust particles dancing in the air. Foreground: The camera looks up from a low angle, tracking the eagle's movement across the expansive stone floor, which is patterned with the bright shafts of light and deep shadows cast by the pillars."
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

Matching test command in moonvalley_ai/inference-service/marey_inference.py

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
  --checkpoint-folder "/app/wlam/models/checkpoints/marey/distilled-0001/epoch0-global_step5000_distilled" \
  --watermarker-path "/app/wlam/models/checkpoints/marey/videoseal/y_256b_img.jit" \
  --height 1080 \
  --width 1920 \
  --num-frames 128 \
  --fps 24 \
  --steps 33 \
  --guidance-scale 3.5 \
  --use-negative-prompts \
  --negative-prompt "<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, bright, vignette, artifacts, still, noise, texture, scanlines, videogame, 360 camera, VR, transition, flare, saturation, distorted, warped, wide angle, contrast, saturated, vibrant, glowing, cross dissolve, texture, videogame, saturation, cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, blown out, horrible, blurry, worst quality, bad, transition, dissolve, cross-dissolve, melt, fade in, fade out, wobbly, weird, low quality, plastic, stock footage, video camera, boring, static" \
  --use-timestep-transform \
  --use-distilled-steps \
  --shift-value 3.0 \
  --use-guidance-schedule \
  --add-quality-guidance \
  --clip-value 10.0 \
  --seed 42 \
  --warmup-steps 4 \
  --cooldown-steps 18 \
  --output /home/david/repos/vllm-omni/marey_inference_test_output_$(date +%Y%m%d_%H%M%S).mp4 \
  "Detailed Description: A majestic, aged eagle with mottled golden-brown feathers soars gracefully through a vast, ancient indoor chamber. Its expansive wings barely flap, catching the air as it glides effortlessly between towering stone pillars adorned with glinting metallic accents. Beams of morning light pierce the gloom, filtering through a cracked skylight high above and illuminating swirling dust motes in their path. The camera pans smoothly, following the eagle's silent flight as it navigates the cavernous space, its sharp eyes scanning the stone floor below, creating a scene of serene power and timeless solitude. Background: The far reaches of the chamber fade into deep shadow, with the silhouettes of distant pillars barely visible. High above, a cracked skylight serves as the primary light source, its fractured glass creating distinct rays of light. Middleground: The aged eagle glides on a steady path, its mottled golden-brown wings spread wide. It passes through the dramatic beams of light, which highlight the intricate details of its feathers and the dust particles dancing in the air. Foreground: The camera looks up from a low angle, tracking the eagle's movement across the expansive stone floor, which is patterned with the bright shafts of light and deep shadows cast by the pillars."
  ```