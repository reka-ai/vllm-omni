#!/usr/bin/env bash
# Build the vllm-omni serving image. Expects vllm-reka and moonvalley_ai as
# siblings of this repo; invoke from anywhere (dev laptop, CI) — the script
# discovers the build context via its own location.
#
# Usage:
#     docker/build-serving.sh -t rekaai/vllm-omni:<tag> [extra docker build args]
#
# Per-repo git state (HEAD SHA, HEAD commit summary, last 10 commits, dirty
# flag) is attached to the image as OCI labels under `org.reka.*`. Inspect
# with:
#     docker inspect --format '{{json .Config.Labels}}' <image> | jq
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_OMNI_DIR="$(dirname "$SCRIPT_DIR")"
CONTEXT_DIR="$(dirname "$VLLM_OMNI_DIR")"

for sibling in vllm-reka moonvalley_ai; do
    if [ ! -d "$CONTEXT_DIR/$sibling" ]; then
        echo "error: required sibling repo not found at $CONTEXT_DIR/$sibling" >&2
        echo "       vllm-omni, vllm-reka, and moonvalley_ai must live side-by-side." >&2
        exit 1
    fi
done

# Per-repo label helpers.
repo_sha()  { git -C "$1" rev-parse HEAD; }

VLLM_OMNI_SHA=$(repo_sha "$VLLM_OMNI_DIR")
VLLM_REKA_SHA=$(repo_sha "$CONTEXT_DIR/vllm-reka")
MOONVALLEY_SHA=$(repo_sha "$CONTEXT_DIR/moonvalley_ai")

LABELS=(
    --label "org.reka.vllm-omni.sha=${VLLM_OMNI_SHA}"
    --label "org.reka.vllm-reka.sha=${VLLM_REKA_SHA}"
    --label "org.reka.moonvalley_ai.sha=${MOONVALLEY_SHA}"
)

cd "$CONTEXT_DIR"
exec docker build -f vllm-omni/docker/Dockerfile.serving \
    "${LABELS[@]}" "$@" .
