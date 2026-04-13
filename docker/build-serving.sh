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
repo_head() { git -C "$1" log -1 --format='%h %ai %an: %s'; }
repo_log()  { git -C "$1" log -n 10 --format='%h %ai %s' HEAD; }
repo_dirty() {
    if git -C "$1" diff-index --quiet HEAD -- 2>/dev/null; then
        echo "false"
    else
        echo "true"
    fi
}

VLLM_OMNI_SHA=$(repo_sha "$VLLM_OMNI_DIR")
VLLM_REKA_SHA=$(repo_sha "$CONTEXT_DIR/vllm-reka")
MOONVALLEY_SHA=$(repo_sha "$CONTEXT_DIR/moonvalley_ai")
BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

LABELS=(
    # OCI standard labels (reference vllm-omni as the primary repo).
    --label "org.opencontainers.image.revision=${VLLM_OMNI_SHA}"
    --label "org.opencontainers.image.created=${BUILD_TIME}"

    # Build provenance.
    --label "org.reka.build-time=${BUILD_TIME}"
    --label "org.reka.build-host=$(hostname)"
    --label "org.reka.build-user=${USER:-unknown}"

    # vllm-omni
    --label "org.reka.vllm-omni.commit-sha=${VLLM_OMNI_SHA}"
    --label "org.reka.vllm-omni.head=$(repo_head "$VLLM_OMNI_DIR")"
    --label "org.reka.vllm-omni.log=$(repo_log "$VLLM_OMNI_DIR")"
    --label "org.reka.vllm-omni.dirty=$(repo_dirty "$VLLM_OMNI_DIR")"

    # vllm-reka
    --label "org.reka.vllm-reka.commit-sha=${VLLM_REKA_SHA}"
    --label "org.reka.vllm-reka.head=$(repo_head "$CONTEXT_DIR/vllm-reka")"
    --label "org.reka.vllm-reka.log=$(repo_log "$CONTEXT_DIR/vllm-reka")"
    --label "org.reka.vllm-reka.dirty=$(repo_dirty "$CONTEXT_DIR/vllm-reka")"

    # moonvalley_ai
    --label "org.reka.moonvalley_ai.commit-sha=${MOONVALLEY_SHA}"
    --label "org.reka.moonvalley_ai.head=$(repo_head "$CONTEXT_DIR/moonvalley_ai")"
    --label "org.reka.moonvalley_ai.log=$(repo_log "$CONTEXT_DIR/moonvalley_ai")"
    --label "org.reka.moonvalley_ai.dirty=$(repo_dirty "$CONTEXT_DIR/moonvalley_ai")"
)

cd "$CONTEXT_DIR"
exec docker build -f vllm-omni/docker/Dockerfile.serving "${LABELS[@]}" "$@" .
