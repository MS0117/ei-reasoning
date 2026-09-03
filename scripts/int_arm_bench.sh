#!/usr/bin/env bash
# One InT L3 arm, end to end: train -> external benchmark eval (the endpoint of
# this run: "do benchmark-aligned InT cliffs move the benchmarks vs base?").
# Runs on either node; the freeze dir under /shared is the only shared state.
#
#   ARM=S3 MAKE_SPLIT=1 REROLL_BASE=1 bash scripts/int_arm_bench.sh   # srv04
#   ARM=S0 GPU=0,1 JOBID=13979        bash scripts/int_arm_bench.sh   # srv08
#
# MAKE_SPLIT=1 creates cliff_split.json (exactly ONE node may do this; every
# other arm waits for that file so all arms share one A/B split).
# REROLL_BASE=1 appends the resumable base-floor re-roll on B after the eval.
#
# Already-measured references for the same 5 sets / same sampling:
#   base      runs/bench/Qwen_Qwen3-4B-Instruct-2507_bench_eval_20260901_110721
#   OpenR1-S3 runs/bench/runs_L3_S3_20260826_011420_iter_0_ckpt_bench_eval_20260901_234449
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARM="${ARM:-S3}"
FREEZE="${FREEZE:-runs/L2_freeze_int_20260902_052108}"
GPU="${GPU:-0,1}"
BENCH_CFG="${BENCH_CFG:-configs/bench_eval.yaml}"
B_FRAC="${B_FRAC:-0.25}"          # 536 cliffs -> B~134 (power 1.00 @3pp), A~402
MAKE_SPLIT="${MAKE_SPLIT:-0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"   # 1 = train here, run every eval on the benchmark node
REROLL_BASE="${REROLL_BASE:-0}"
N="${N:-32}"
IT="$FREEZE/iter_0"
SPLIT="$IT/cliff_split.json"
TAG="[${ARM}]"

wait_for() {  # wait_for <path> <what>
  local p="$1"
  [ -e "$p" ] && return 0
  echo ">>> $TAG waiting for $2 ($p) ..."
  until [ -e "$p" ]; do sleep 120; done
}

if [ "$MAKE_SPLIT" = "1" ]; then
  wait_for "$IT/dataset/stats.json" "the freeze's build_dataset"
  echo ">>> $TAG freeze complete at $(date +%H:%M)"
  [ -f "$SPLIT" ] || .venv/bin/python scripts/cliff_split.py --run-dir "$FREEZE" --b-frac "$B_FRAC"
else
  wait_for "$SPLIT" "the A/B split written by the split-owner node"
fi
.venv/bin/python -c "
import json; d=json.load(open('$SPLIT'))
print(f'>>> $TAG split: A={len(d[\"A\"])} trained, B={len(d[\"B\"])} held out')"

# ---- train (readout deferred; the benchmark is the endpoint) ----------------
EXISTING="$(ls -d runs/L3_${ARM}_int_*/iter_0/ckpt 2>/dev/null | head -1 || true)"
if [ -z "$EXISTING" ]; then
  SKIP_READOUT=1 ARM_TAG=int bash scripts/l3_arm.sh "$ARM" "$FREEZE" "$GPU"
fi
CKPT="$(ls -d runs/L3_${ARM}_int_*/iter_0/ckpt 2>/dev/null | head -1 || true)"
[ -n "$CKPT" ] || { echo ">>> $TAG no checkpoint — aborting" >&2; exit 1; }
echo ">>> $TAG trained: $CKPT"

if [ "$TRAIN_ONLY" = "1" ]; then
  echo ">>> $TAG TRAIN_ONLY — eval deferred to the benchmark node"
  exit 0
fi

# ---- the endpoint ----------------------------------------------------------
bash scripts/eval_bench.sh "$CKPT" -c "$BENCH_CFG" -g "$GPU"
echo ">>> $TAG benchmark eval done at $(date +%H:%M)"

# ---- optional resumable tail: the base floor for the attractor readout ------
if [ "$REROLL_BASE" = "1" ]; then
  echo ">>> $TAG base floor re-roll on B (resumable; safe to lose the allocation here)"
  NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/cliff_reroll.py \
    --run-dir "$FREEZE" --qids-file "$SPLIT:B" --n "$N" --passes 1
fi
echo ">>> $TAG ALL DONE at $(date +%H:%M)"
