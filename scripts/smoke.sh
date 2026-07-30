#!/usr/bin/env bash
# End-to-end smoke test: 1 EI iteration with Qwen3-0.6B on 50 questions.
# Usage: bash scripts/smoke.sh [GPU_ID]   (default GPU 0)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
GPU="${1:-0}"

CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -m expert_iter.loop \
  --config configs/smoke.yaml

echo
echo "=== smoke metrics ==="
cat runs/smoke/metrics.jsonl
