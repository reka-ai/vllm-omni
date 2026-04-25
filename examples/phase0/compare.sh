#!/usr/bin/env bash
# Phase 0: compare the two within-codebase pairs of final latents.
#
# Usage:
#   bash examples/phase0/compare.sh

set -euo pipefail

PHASE0_ROOT="${PHASE0_ROOT:-/mnt/localdisk/vllm_omni_storage/phase0/20260424}"
REL_TOL="${REL_TOL:-1e-6}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPARE="${REPO_ROOT}/examples/offline_inference/marey/compare_final_latents.py"

echo "================================================================"
echo "vllm-omni: main  vs  marey-serving-comparison"
echo "================================================================"
python "${COMPARE}" \
    "${PHASE0_ROOT}/vllm_main/final_z.pt" \
    "${PHASE0_ROOT}/vllm_cmp/final_z.pt" \
    --rel-tol "${REL_TOL}"

echo ""
echo "================================================================"
echo "moonvalley_ai: main  vs  marey-serving-comparison"
echo "================================================================"
python "${COMPARE}" \
    "${PHASE0_ROOT}/mv_main/latents.pt" \
    "${PHASE0_ROOT}/mv_cmp/latents.pt" \
    --rel-tol "${REL_TOL}"
