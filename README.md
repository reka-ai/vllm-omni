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

You can launch the server through `vllm-omni serve`, you can find an example in [examples/online_serving/marey/run_server.sh](examples/online_serving/marey/run_server.sh)

```bash
# Launch the server on a GPU node
MODEL=/app/hf_checkpoints/marey-distilled-0100/ MOONVALLEY_AI_PATH=${PATH_TO_MOONVALLEY_AI} bash examples/online_serving/marey/run_server.sh

# On the same node
SEED=0 examples/online_serving/marey/run_curl_text_to_video.sh 
```

## Marey Inference Usage

Matching test command in moonvalley_ai/inference-service/marey_inference.py
*NOTE*: Due to a bug/quirk in the way the cli params are implemented note that many flags actually disable the corresponding variable, eg --add-quality-guidance actually sets this variable to false. After talking to Igor and Adithya it seems this CLI isn’t the preferred way to launch commands on the moonvalley side so this did not interfere with their workloads. This command sets the recommended setup for marey inference that the vllm-omni implementation was designed against.

```bash
PYTHONPATH=${PATH_TO_MOONVALLEY_AI} \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --project ${PATH_TO_MOONVALLEY_AI}/inference-service \
torchrun --nproc_per_node=8 ${PATH_TO_MOONVALLEY_AI}/inference-service/marey_inference.py infer \
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
  --disable-caching \
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
  --output /home/aormazabal/wlam/wlam-inference//vllm-omni/marey_inference_test_output_$(date +%Y%m%d_%H%M%S).mp4 \
  "Detailed Description: A majestic, aged eagle with mottled golden-brown feathers soars gracefully through a vast, ancient indoor chamber. Its expansive wings barely flap, catching the air as it glides effortlessly between towering stone pillars adorned with glinting metallic accents. Beams of morning light pierce the gloom, filtering through a cracked skylight high above and illuminating swirling dust motes in their path. The camera pans smoothly, following the eagle's silent flight as it navigates the cavernous space, its sharp eyes scanning the stone floor below, creating a scene of serene power and timeless solitude. Background: The far reaches of the chamber fade into deep shadow, with the silhouettes of distant pillars barely visible. High above, a cracked skylight serves as the primary light source, its fractured glass creating distinct rays of light. Middleground: The aged eagle glides on a steady path, its mottled golden-brown wings spread wide. It passes through the dramatic beams of light, which highlight the intricate details of its feathers and the dust particles dancing in the air. Foreground: The camera looks up from a low angle, tracking the eagle's movement across the expansive stone floor, which is patterned with the bright shafts of light and deep shadows cast by the pillars."
  ```