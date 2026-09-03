#!/usr/bin/env bash
# The group-advantage (gadv) L3 arm on the InT freeze, chained end to end so it
# runs unattended:
#
#   [wait for the gadv smoke] -> TRAIN -> base floor on B -> arm re-roll on B
#   -> attractor compare
#
# Why this script instead of plain `l3_arm.sh P`:
#
#  1. The InT freeze has NO base floor. l3_arm.sh's readout (its step after the
#     train) hard-requires <freeze>/iter_0/reroll/pass_0/verdicts.jsonl, but
#     int_train_srv04.sh called int_arm_bench.sh with REROLL_BASE=0, so that
#     re-roll was never drawn. Running l3_arm.sh unmodified would train for
#     hours, re-roll B for hours, then die in attractor_mass.py on a missing
#     file. Here the floor is drawn explicitly as step 2.
#     (The OpenR1 freeze L2_freeze_20260825_040504 does have it: 353 qids x 32
#     x 2 passes -- which is why the same command works there.)
#
#  2. Ordering: TRAIN first. Both cliff_reroll passes are resumable through
#     their .done markers, so losing the allocation mid-re-roll costs only the
#     current shard; a killed training costs the whole run. Same reasoning as
#     scripts/int_s3_only.sh.
#
# The floor is the BASE policy (cliff_reroll's --model-path default) at the same
# n=32 as the arm re-roll -- a paired per-question comparison needs one n.
#
#   [SMOKE_PID=] [SMOKE_RUN=] [FREEZE=...] [GPU=0,1] bash scripts/gadv_arm_int.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FREEZE="${FREEZE:-runs/L2_freeze_int_20260902_052108}"
PRESET_YAML="${PRESET_YAML:-configs/methods/arms/gadv.yaml}"
GPU="${GPU:-0,1}"
N="${N:-32}"
SMOKE_PID="${SMOKE_PID:-}"
SMOKE_RUN="${SMOKE_RUN:-}"
SPLIT="$FREEZE/iter_0/cliff_split.json"

[ -f "$SPLIT" ]       || { echo ">>> [gadv] missing $SPLIT" >&2; exit 1; }
[ -f "$PRESET_YAML" ] || { echo ">>> [gadv] missing $PRESET_YAML" >&2; exit 1; }

# ---- 0) let the smoke finish and confirm it passed --------------------------
if [ -n "$SMOKE_PID" ]; then
  echo ">>> [gadv] waiting for the gadv smoke (pid $SMOKE_PID) ..."
  while kill -0 "$SMOKE_PID" 2>/dev/null; do sleep 60; done
  echo ">>> [gadv] smoke process exited at $(date +%H:%M)"
fi
if [ -n "$SMOKE_RUN" ]; then
  if [ -s "runs/$SMOKE_RUN/metrics.jsonl" ]; then
    echo ">>> [gadv] smoke PASSED: runs/$SMOKE_RUN/metrics.jsonl"
  else
    echo ">>> [gadv] smoke produced no metrics.jsonl -- the gadv path is broken." >&2
    echo ">>> [gadv] refusing to spend GPU hours on the arm. Aborting." >&2
    exit 1
  fi
fi

# vLLM leaves VLLM::EngineCore children behind on some exits; they hold the full
# gpu_memory_utilization share and the next boot then fails on free memory.
leftover="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
            | grep -i 'EngineCore' | cut -d, -f1 | tr -d ' ' || true)"
if [ -n "$leftover" ]; then
  echo ">>> [gadv] killing leftover EngineCore PIDs: $leftover"
  # shellcheck disable=SC2086
  kill -9 $leftover 2>/dev/null || true
  sleep 5
fi
echo ">>> [gadv] GPU before start:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# ---- 1) train (the only unrecoverable step) --------------------------------
if ! ls -d runs/L3_P_gadv_*/iter_0/ckpt >/dev/null 2>&1; then
  echo ">>> [gadv] TRAIN starting $(date +%m-%d_%H:%M)"
  SKIP_READOUT=1 PRESET="$PRESET_YAML" bash scripts/l3_arm.sh P "$FREEZE" "$GPU"
fi
CKPT="$(ls -dt runs/L3_P_gadv_*/iter_0/ckpt 2>/dev/null | head -1 || true)"
[ -n "$CKPT" ] || { echo ">>> [gadv] no checkpoint -- aborting" >&2; exit 1; }
ARM_DIR="$(dirname "$(dirname "$CKPT")")"
echo ">>> [gadv] trained: $CKPT  ($(date +%m-%d_%H:%M))"
.venv/bin/python -c "
import json; s=json.load(open('$ARM_DIR/iter_0/dataset/stats.json'))
print('[gadv] train set:', json.dumps({k: v for k, v in s.items() if isinstance(v, (int, float))}))" || true

# ---- 2) base floor on B (BASE policy; resumable) ---------------------------
echo ">>> [gadv] base floor on B starting $(date +%m-%d_%H:%M)"
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/cliff_reroll.py \
  --run-dir "$FREEZE" --qids-file "$SPLIT:B" --n "$N" --passes 1
echo ">>> [gadv] base floor done $(date +%m-%d_%H:%M)"

# ---- 3) the arm's own re-roll on the same B (resumable) --------------------
echo ">>> [gadv] arm re-roll on B starting $(date +%m-%d_%H:%M)"
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/cliff_reroll.py \
  --run-dir "$FREEZE" --model-path "$CKPT" --qids-file "$SPLIT:B" \
  --n "$N" --passes 1 --out "$ARM_DIR/iter_0/reroll_B"
echo ">>> [gadv] arm re-roll done $(date +%m-%d_%H:%M)"

# ---- 4) the readout --------------------------------------------------------
.venv/bin/python scripts/attractor_mass.py \
  --verdicts "$FREEZE/iter_0/reroll/pass_0/verdicts.jsonl" \
  --compare "$ARM_DIR/iter_0/reroll_B/pass_0/verdicts.jsonl" \
  --qids-file "$SPLIT:B" --out "$ARM_DIR/iter_0/attractor_B_compare.json"

echo
echo ">>> [gadv] ALL DONE $(date +%m-%d_%H:%M)"
echo ">>> arm:         $ARM_DIR"
echo ">>> dataset:     $ARM_DIR/iter_0/dataset/stats.json"
echo ">>> B transfer:  $ARM_DIR/iter_0/attractor_B_compare.json"
