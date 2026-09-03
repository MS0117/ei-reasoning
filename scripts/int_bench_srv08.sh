#!/usr/bin/env bash
# srv08 = the HARD-SET measurement node. The hard sweep is redone here from
# scratch so base and every arm share one GPU architecture: the earlier attempts
# were split across srv07/A6000 and died on orphaned EngineCores, and a paired
# per-question test across two architectures is meaningless (measured on the same
# base model + hmmt25: aggregate -0.62pp, but 4pp mean per-question swing).
# The STANDARD 5 sets are NOT redone — base/S0/S1/S3 already sit on srv04/A100.
#
#   bash scripts/int_bench_srv08.sh              # OpenR1 arms, then the InT arms
#   ONLY=openr1 bash scripts/int_bench_srv08.sh  # stop after the OpenR1 sweep
#   ONLY=int    bash scripts/int_bench_srv08.sh  # wait for srv04's ckpts, InT only
#
# eval_bench.sh writes a fresh timestamped dir per model and skips finished
# benchmarks via .done markers, so re-running resumes instead of repeating.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU="${GPU:-0,1}"
CFG="${CFG:-configs/bench_eval_hard_n16.yaml}"
FREEZE="${FREEZE:-runs/L2_freeze_int_20260902_052108}"
ONLY="${ONLY:-all}"

# most informative first: base sets the reference, S3 is the arm under test
OPENR1=(
  "Qwen/Qwen3-4B-Instruct-2507|base"
  "runs/L3_S3_20260826_011420/iter_0/ckpt|OpenR1-S3"
  "runs/L3_S0_20260826_081534/iter_0/ckpt|OpenR1-S0"
  "runs/L3_S1_20260826_141540/iter_0/ckpt|OpenR1-S1"
)

run_one() {  # run_one <model> <label>
  echo ">>> [srv08] $2 on $(basename "$CFG") starting $(date +%m-%d_%H:%M)"
  bash scripts/eval_bench.sh "$1" -c "$CFG" -g "$GPU"
  echo ">>> [srv08] $2 done $(date +%m-%d_%H:%M)"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "openr1" ]; then
  for e in "${OPENR1[@]}"; do run_one "${e%%|*}" "${e##*|}"; done
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "int" ]; then
  for ARM in S3 S0; do
    echo ">>> [srv08] waiting for the InT-$ARM checkpoint from srv04 ..."
    until [ -n "$(ls -d runs/L3_${ARM}_int_*/iter_0/ckpt 2>/dev/null | head -1 || true)" ]; do sleep 180; done
    run_one "$(ls -d runs/L3_${ARM}_int_*/iter_0/ckpt | head -1)" "InT-$ARM"
  done
fi
echo ">>> [srv08] ALL DONE $(date +%m-%d_%H:%M)"
