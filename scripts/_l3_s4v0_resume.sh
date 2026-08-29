#!/usr/bin/env bash
# One-off: finish the S4-v0 arm — its SFT checkpoint is valid on disk, only the
# DPO phase is missing (it OOM'd; run_dpo now runs with gradient checkpointing).
# --phase dpo skips straight to DPO; NOT the loop, whose upstream .done markers
# were invalidated by the parallel STAGED_UL config-schema change.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
R=runs/L3_S4v0_20260827_122953
S=runs/L2_freeze_20260825_040504
SP=$S/iter_0/cliff_split.json
GPU="${1:-0,1}"
export NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU"
# recover the ~2 GB the caching allocator strands as fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/3] DPO phase ($(date)) ==="
.venv/bin/accelerate launch --config_file configs/accelerate/zero2.yaml \
  --num_processes 2 --num_machines 1 \
  -m expert_iter.train --config "$R/config.yaml" --run-dir "$R" --iter 0 \
  --model-path "$R/iter_0/ckpt" --phase dpo

echo "=== [2/3] holdout eval ($(date)) ==="
.venv/bin/python -m expert_iter.eval --config "$R/config.yaml" --run-dir "$R" \
  --iter 0 --model-path "$R/iter_0/ckpt"

echo "=== [3/3] B re-roll + attractor compare ($(date)) ==="
.venv/bin/python scripts/cliff_reroll.py --run-dir "$S" \
  --model-path "$R/iter_0/ckpt" --qids-file "$SP:B" --passes 1 --out "$R/iter_0/reroll_B"
.venv/bin/python scripts/attractor_mass.py \
  --verdicts "$S/iter_0/reroll/pass_0/verdicts.jsonl" "$S/iter_0/reroll/pass_1/verdicts.jsonl" \
  --compare "$R/iter_0/reroll_B/pass_0/verdicts.jsonl" \
  --qids-file "$SP:B" --out "$R/iter_0/attractor_B_compare.json"

echo "=== S4v0 done ($(date)): $R ==="
