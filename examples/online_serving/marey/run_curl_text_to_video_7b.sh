#!/bin/bash
# Marey 7B text-to-video curl example using the async video job API.
#
# The 7B was trained at up to 480p (bucket_config max).
#
# Resolution constraint: H_patches * W_patches must be divisible by
# ULYSSES_DEGREE (8). For 480p the safe choice is 768x480 (patches 24x15,
# S=360, 360/8=45). 848x480 fails (patches 27x15, S=405, not div by 8).
# 1280x720 also works (patches 40x23, S=920/8=115) but is above training range.
#
# The checkpoint is not distilled → needs ~50 steps (vs 33 for distilled 30B)
# and a higher guidance scale (~7.0 vs 3.5).

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8098}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
SEED="${SEED:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-marey_7b_output_noquality_seed${SEED}.mp4}"
create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    -F "prompt=Detailed Description: A majestic, aged eagle with mottled golden-brown feathers soars gracefully through a vast, ancient indoor chamber. Its expansive wings barely flap, catching the air as it glides effortlessly between towering stone pillars adorned with glinting metallic accents. Beams of morning light pierce the gloom, filtering through a cracked skylight high above and illuminating swirling dust motes in their path. The camera pans smoothly, following the eagle's silent flight as it navigates the cavernous space, its sharp eyes scanning the stone floor below, creating a scene of serene power and timeless solitude. Background: The far reaches of the chamber fade into deep shadow, with the silhouettes of distant pillars barely visible. High above, a cracked skylight serves as the primary light source, its fractured glass creating distinct rays of light. Middleground: The aged eagle glides on a steady path, its mottled golden-brown wings spread wide. It passes through the dramatic beams of light, which highlight the intricate details of its feathers and the dust particles dancing in the air. Foreground: The camera looks up from a low angle, tracking the eagle's movement across the expansive stone floor, which is patterned with the bright shafts of light and deep shadows cast by the pillars." \
    -F "size=768x480" \
    -F "num_frames=64" \
    -F "num_inference_steps=50" \
    -F "guidance_scale=7.0" \
    -F "seed=${SEED}"
)

video_id="$(echo "${create_response}" | jq -r '.id')"
if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
  echo "Failed to create video job:"
  echo "${create_response}" | jq .
  exit 1
fi

echo "Created video job ${video_id}"
echo "${create_response}" | jq .

while true; do
  status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
  status="$(echo "${status_response}" | jq -r '.status')"

  case "${status}" in
    queued|in_progress)
      echo "Video job ${video_id} status: ${status}"
      sleep "${POLL_INTERVAL}"
      ;;
    completed)
      echo "${status_response}" | jq .
      break
      ;;
    failed)
      echo "Video generation failed:"
      echo "${status_response}" | jq .
      exit 1
      ;;
    *)
      echo "Unexpected status response:"
      echo "${status_response}" | jq .
      exit 1
      ;;
  esac
done

curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUTPUT_PATH}"
echo "Saved video to ${OUTPUT_PATH}"
