#!/usr/bin/env bash
# srv08, one 2-GPU queue, unattended: the budget gadv arm + the two hard_n16
# benchmark reads that make it comparable to the full-budget arm.
#
#   0) wait for scripts/gadv_arm_int.sh (the full-budget arm's B readout) to let
#      go of the GPUs -- the allocation only has 2 and they are in use
#   1) TRAIN  the budget arm   (configs/methods/arms/budget_gadv.yaml)
#   2) BENCH  the FULL arm     runs/L3_P_gadv_20260903_204616/iter_0/ckpt
#   3) BENCH  the budget arm   runs/L3_P_budget_gadv_<TS>/iter_0/ckpt
#
# Train first because it is the only unrecoverable step (train.py saves once, at
# the end -- save_strategy="no"), while eval_bench.sh resumes per benchmark off
# its .done markers. Same reasoning as gadv_arm_int.sh.
#
# Both benches use configs/bench_eval_hard_n16.yaml on THIS node: the hard sweep
# is a paired per-question comparison and its rows may not be mixed across GPU
# architectures (srv04/A100 holds the 4-set bench_eval sweep, srv08/Blackwell
# the hard one).
#
#   [FREEZE=...] [GPU=0,1] [FULL_CKPT=...] [READOUT=1] bash scripts/gadv_budget_sweep.sh
#
# READOUT=1 appends the budget arm's own B re-roll + attractor compare after the
# benches (the base floor at n=32 already exists under $FREEZE/iter_0/reroll).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FREEZE="${FREEZE:-runs/L2_freeze_int_20260902_052108}"
PRESET_YAML="${PRESET_YAML:-configs/methods/arms/budget_gadv.yaml}"
FULL_CKPT="${FULL_CKPT:-runs/L3_P_gadv_20260903_204616/iter_0/ckpt}"
BENCH_CFG="${BENCH_CFG:-configs/bench_eval_hard_n16.yaml}"
GPU="${GPU:-0,1}"
N="${N:-32}"
READOUT="${READOUT:-0}"
SPLIT="$FREEZE/iter_0/cliff_split.json"
TAG="[budget]"

[ -f "$SPLIT" ]       || { echo ">>> $TAG missing $SPLIT" >&2; exit 1; }
[ -f "$PRESET_YAML" ] || { echo ">>> $TAG missing $PRESET_YAML" >&2; exit 1; }
[ -f "$BENCH_CFG" ]   || { echo ">>> $TAG missing $BENCH_CFG" >&2; exit 1; }
[ -d "$FULL_CKPT" ]   || { echo ">>> $TAG missing $FULL_CKPT" >&2; exit 1; }

# vLLM leaves VLLM::EngineCore children behind on some exits; each holds its full
# gpu_memory_utilization share and the next boot then fails on free memory.
reap_enginecores() {
  local leftover
  leftover="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
              | grep -i 'EngineCore' | cut -d, -f1 | tr -d ' ' || true)"
  # only ours: another user's job shares this node
  local mine=""
  for p in $leftover; do
    [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -un)" ] && mine="$mine $p"
  done
  if [ -n "$mine" ]; then
    echo ">>> $TAG killing leftover EngineCore PIDs:$mine"
    # shellcheck disable=SC2086
    kill -9 $mine 2>/dev/null || true
    sleep 10
  fi
}

# ---- 0) wait for the full-budget arm's chain to release the 2 GPUs ----------
if pgrep -u "$(id -u)" -f "scripts/gadv_arm_int.sh" >/dev/null 2>&1; then
  echo ">>> $TAG waiting for scripts/gadv_arm_int.sh to finish (it holds $GPU) ..."
  while pgrep -u "$(id -u)" -f "scripts/gadv_arm_int.sh" >/dev/null 2>&1; do sleep 120; done
  echo ">>> $TAG gadv_arm_int.sh exited at $(date +%m-%d_%H:%M)"
fi
# belt and braces: the loop/engine/accelerate processes of any of our runs
while pgrep -u "$(id -u)" -f "expert_iter\.(loop|engine|train)|cliff_reroll\.py|accelerate launch" >/dev/null 2>&1; do
  echo ">>> $TAG our GPU processes are still up, waiting ... $(date +%H:%M)"
  sleep 120
done
reap_enginecores
echo ">>> $TAG GPU before start:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# ---- 1) train the budget arm (the only unrecoverable step) ------------------
ARM_GLOB="runs/L3_P_budget_gadv_*"
if [ -z "$(ls -d $ARM_GLOB/iter_0/ckpt 2>/dev/null | head -1 || true)" ]; then
  echo ">>> $TAG TRAIN starting $(date +%m-%d_%H:%M)"
  SKIP_READOUT=1 PRESET="$PRESET_YAML" bash scripts/l3_arm.sh P "$FREEZE" "$GPU"
  reap_enginecores
fi
BUDGET_CKPT="$(ls -dt $ARM_GLOB/iter_0/ckpt 2>/dev/null | head -1 || true)"
[ -n "$BUDGET_CKPT" ] || { echo ">>> $TAG no budget checkpoint -- aborting" >&2; exit 1; }
ARM_DIR="$(dirname "$(dirname "$BUDGET_CKPT")")"
echo ">>> $TAG trained: $BUDGET_CKPT  ($(date +%m-%d_%H:%M))"
.venv/bin/python -c "
import json; s=json.load(open('$ARM_DIR/iter_0/dataset/stats.json'))
g=s.get('gadv', {})
print('[budget] rows:', s.get('sft_by_source'), 'mean_len:', s.get('mean_len'))
print('[budget] questions:', g.get('questions'), 'trunc_capped:', g.get('n_truncated_capped_rows'),
      'wrong_capped:', g.get('n_wrong_capped'))" || true

# ---- 2) hard_n16 on the FULL-budget arm ------------------------------------
echo ">>> $TAG BENCH full-budget arm $(date +%m-%d_%H:%M)"
bash scripts/eval_bench.sh "$FULL_CKPT" -c "$BENCH_CFG" -g "$GPU"
reap_enginecores
echo ">>> $TAG bench(full) done $(date +%m-%d_%H:%M)"

# ---- 3) hard_n16 on the budget arm -----------------------------------------
echo ">>> $TAG BENCH budget arm $(date +%m-%d_%H:%M)"
bash scripts/eval_bench.sh "$BUDGET_CKPT" -c "$BENCH_CFG" -g "$GPU"
reap_enginecores
echo ">>> $TAG bench(budget) done $(date +%m-%d_%H:%M)"

# ---- 4) optional: the budget arm's own B transfer readout -------------------
if [ "$READOUT" = "1" ]; then
  echo ">>> $TAG B re-roll under the budget arm $(date +%m-%d_%H:%M)"
  NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/cliff_reroll.py \
    --run-dir "$FREEZE" --model-path "$BUDGET_CKPT" --qids-file "$SPLIT:B" \
    --n "$N" --passes 1 --out "$ARM_DIR/iter_0/reroll_B"
  .venv/bin/python scripts/attractor_mass.py \
    --verdicts "$FREEZE/iter_0/reroll/pass_0/verdicts.jsonl" \
    --compare "$ARM_DIR/iter_0/reroll_B/pass_0/verdicts.jsonl" \
    --qids-file "$SPLIT:B" --out "$ARM_DIR/iter_0/attractor_B_compare.json"
fi

echo
echo ">>> $TAG ALL DONE $(date +%m-%d_%H:%M)"
echo ">>> budget arm:   $ARM_DIR"
echo ">>> dataset:      $ARM_DIR/iter_0/dataset/stats.json"
echo ">>> benches (newest first):"
ls -dt runs/bench/*bench_eval_hard_n16* 2>/dev/null | head -4
