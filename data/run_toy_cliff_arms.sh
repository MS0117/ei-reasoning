#!/usr/bin/env bash
# Run the toy-cliff arms back to back on ONE shared rollout, in the foreground,
# so arm k+1 starts the moment arm k exits. Each arm is a normal
# data/run_toy_cliff.sh launch WITHOUT -b (that flag backgrounds a single run,
# which is the opposite of what a queue wants).
#
# Usage:
#   bash data/run_toy_cliff_arms.sh -R runs/toy_cliff_2/_subset250 [options]
#
#   -R REUSE_DIR   run dir that owns the rollout (required; passed as
#                  --reuse-rollout, so every arm sees the SAME cliff set and the
#                  same failed trajectories — that is what makes the comparison
#                  paired)
#   -o OUT_ROOT    where run dirs go (default runs/toy_cliff_2)
#   -a "A B C"     arm list = data/configs/<name>.yaml
#                  (default: CONTROL LSPO BRIDGE STAGED STAGED_DPO)
#   -k             keep going after a failed arm (default: stop)
#   -n             dry run — print the commands and exit
#
# The whole queue is long (~20-30 GPU-h for five arms at ~250 cliffs), so start
# it detached inside the GPU tmux session and follow the summary log:
#   nohup bash data/run_toy_cliff_arms.sh -R runs/toy_cliff_2/_subset250 \
#       > logs/toy_cliff_arms_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Between arms it waits for the GPUs to actually drain: a killed or crashed run
# can leave VLLM::EngineCore holding ~78 GB, and the next arm then dies with
# "Free memory ... less than desired". Strays owned by this user are killed
# after the grace period.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REUSE=""
OUT_ROOT="runs/toy_cliff_2"
ARMS="CONTROL LSPO BRIDGE STAGED STAGED_DPO"
KEEP_GOING=false
DRY=false

while getopts ":R:o:a:knh" opt; do
  case "$opt" in
    R) REUSE="$OPTARG" ;;
    o) OUT_ROOT="$OPTARG" ;;
    a) ARMS="$OPTARG" ;;
    k) KEEP_GOING=true ;;
    n) DRY=true ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "unknown option -$OPTARG (try -h)" >&2; exit 2 ;;
    :) echo "option -$OPTARG needs a value" >&2; exit 2 ;;
  esac
done

[ -n "$REUSE" ] || { echo "-R REUSE_DIR is required (try -h)" >&2; exit 2; }
[ -d "$REUSE" ] || { echo "reuse dir not found: $REUSE" >&2; exit 1; }
[ -f "$REUSE/config.yaml" ] || { echo "no frozen config at $REUSE/config.yaml" >&2; exit 1; }
for a in $ARMS; do
  [ -f "data/configs/$a.yaml" ] || { echo "no config: data/configs/$a.yaml" >&2; exit 1; }
done

N_CLIFF=$(wc -l < "$REUSE/iter_0/partition/unsolved.jsonl" 2>/dev/null || echo "?")
echo "=== toy-cliff arm queue ==="
echo "  reuse   : $REUSE  ($N_CLIFF cliffs)"
echo "  out     : $OUT_ROOT"
echo "  arms    : $ARMS"
echo "  on fail : $($KEEP_GOING && echo 'keep going' || echo 'stop')"
echo

drain_gpus() {
  # A clean exit releases the engines by itself; this only catches leftovers.
  for _ in $(seq 1 12); do
    pgrep -u "$USER" -f "VLLM::EngineCore" >/dev/null 2>&1 || return 0
    sleep 10
  done
  echo ">>> VLLM::EngineCore still alive after 120s — killing strays"
  pkill -u "$USER" -f "VLLM::EngineCore" || true
  sleep 10
}

FAILED=""
DONE_ARMS=""
for ARM in $ARMS; do
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="logs/toy_cliff_${ARM}_${TS}.log"
  CMD=(bash data/run_toy_cliff.sh -c "data/configs/$ARM.yaml" -o "$OUT_ROOT"
       -- --reuse-rollout "$REUSE")

  echo "----------------------------------------------------------------"
  echo ">>> [$(date +%H:%M:%S)] START $ARM"
  echo ">>> ${CMD[*]}"
  echo ">>> log: $LOG"
  if $DRY; then echo ">>> (dry run)"; continue; fi

  mkdir -p logs
  SECONDS=0
  "${CMD[@]}" > "$LOG" 2>&1
  RC=$?
  H=$((SECONDS / 3600)); M=$(((SECONDS % 3600) / 60))

  if [ $RC -ne 0 ]; then
    echo ">>> [$(date +%H:%M:%S)] FAILED $ARM (rc=$RC) after ${H}h${M}m — see $LOG"
    tail -20 "$LOG" | sed 's/^/    | /'
    FAILED="$FAILED $ARM"
    drain_gpus
    $KEEP_GOING || { echo ">>> stopping (use -k to keep going)"; break; }
    continue
  fi

  RUN_DIR=$(grep -m1 -oP 'run_dir=\K\S+' "$LOG" || true)
  CONV=$(.venv/bin/python -c "
import json,sys
try:
    m=json.load(open('$RUN_DIR/metrics.json'))
    print(f\"conversion {m['cliff/conversion_rate']}  cliff {m['funnel/n_cliff']}  kept {m['funnel/n_kept']}\")
except Exception as e: print('metrics unavailable')" 2>/dev/null)
  echo ">>> [$(date +%H:%M:%S)] DONE  $ARM in ${H}h${M}m — $CONV"
  echo ">>>       $RUN_DIR"
  DONE_ARMS="$DONE_ARMS $ARM"
  drain_gpus
done

echo "================================================================"
echo "queue finished at $(date +%H:%M:%S)"
echo "  completed:${DONE_ARMS:- none}"
echo "  failed   :${FAILED:- none}"
echo
echo "analysis:  .venv/bin/python data/rank_toy_runs.py --runs-dir $OUT_ROOT"
[ -z "$FAILED" ] || exit 1
