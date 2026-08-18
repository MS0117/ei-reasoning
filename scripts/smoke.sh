#!/usr/bin/env bash
# End-to-end smoke test: 1 EI iteration with Qwen3-0.6B on tiny data.
# Usage: bash scripts/smoke.sh [GPU_ID] [CONFIG]
#   default GPU 0, config configs/smoke.yaml
#   the new-method path: bash scripts/smoke.sh 0 configs/methods/smoke_lora.yaml
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
GPU="${1:-0}"
CONFIG="${2:-configs/smoke.yaml}"
RUN_NAME="smoke_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -m expert_iter.loop \
  --config "$CONFIG" --override "run.name=$RUN_NAME"

echo
echo "=== smoke metrics ==="
cat "runs/$RUN_NAME/metrics.jsonl"
