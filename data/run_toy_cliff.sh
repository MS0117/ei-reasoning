#!/usr/bin/env bash
# Toy cliff experiment: one improvement cycle (rollout -> partition -> anchor
# -> improve -> filters) on the 137-question cliff set, then a survival
# funnel + alpha-scaling survival + C(y) statistics to stdout/metrics.json/wandb.
# MODEL is optional (default: the config's model.base = Qwen/Qwen3-4B-Instruct-2507).
#
# Usage:
#   bash data/run_toy_cliff.sh [MODEL] [-c CONFIG] [-g GPUS] [-o OUT_ROOT] [-r RUN_DIR] [-b] [-- EXTRA_ARGS]
#
#   bash data/run_toy_cliff.sh -g 0,1
#   bash data/run_toy_cliff.sh -g 0 -b
#   bash data/run_toy_cliff.sh -g 0 -- --override anchor.policy=none          # vanilla ablation
#   bash data/run_toy_cliff.sh -g 0 -- --override improve.lora_sft.project_back.enabled=false
#   bash data/run_toy_cliff.sh -r runs/toy_cliff/<dir> -g 0 -- --force filters
#
# Every launch gets a FRESH timestamped dir <OUT_ROOT>/<slug>_<ts>/ so
# ablations never overwrite each other. To RESUME a crashed/partial run
# (stages skip via their .done markers), point -r at its dir.
#
# -o OUT_ROOT (default runs/toy_cliff) keeps runs over DIFFERENT question sets
# apart. rank_toy_runs.py globs its --runs-dir and prints one table, so mixing
# two cliff sets there would put two different denominators in one conversion
# column. The ~300-cliff arms live in runs/toy_cliff_2:
#   bash data/run_toy_cliff.sh -c data/configs/CONTROL.yaml -o runs/toy_cliff_2 -b
#
# Prerequisite (one-time): backfill gold solutions —
#   .venv/bin/python scripts/backfill_gold_solutions.py \
#       --input data/cliff_sets/openr1_qwen3-4b-2507_n2000.jsonl
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL=""
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  MODEL="$1"
  shift
fi

CONFIG="data/configs/toy_cliff.yaml"
GPUS=""
BACKGROUND=false
RESUME_DIR=""
OUT_ROOT="runs/toy_cliff"

while getopts ":c:g:o:r:bh" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    o) OUT_ROOT="$OPTARG" ;;
    r) RESUME_DIR="$OPTARG" ;;
    b) BACKGROUND=true ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "unknown option -$OPTARG (try -h)" >&2; exit 2 ;;
    :) echo "option -$OPTARG needs a value" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
[ -x .venv/bin/python ] || { echo ".venv missing — run: bash scripts/setup.sh --skip-lean" >&2; exit 1; }

if [ -n "$GPUS" ]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# Same PCIe-without-NVLink NCCL guard as scripts/run.sh (only matters for
# engine.tensor_parallel > 1).
if [ -z "${NCCL_P2P_DISABLE:-}" ] && command -v nvidia-smi >/dev/null; then
  if ! nvidia-smi topo -m 2>/dev/null | grep -qE "NV[0-9]"; then
    export NCCL_P2P_DISABLE=1
    echo ">>> no NVLink detected: NCCL_P2P_DISABLE=1"
  fi
fi

SLUG="$(echo "${MODEL:-default}_$(basename "$CONFIG" .yaml)" | tr '/ ' '__')"
SLUG="${SLUG%/}"
if [ -n "$RESUME_DIR" ]; then
  [ -d "$RESUME_DIR" ] || { echo "resume dir not found: $RESUME_DIR" >&2; exit 1; }
  RUN_DIR="$RESUME_DIR"
else
  RUN_DIR="${OUT_ROOT%/}/${SLUG}_$(date +%Y%m%d_%H%M%S)"
fi
echo ">>> model=${MODEL:-<config model.base>}  config=$CONFIG  run_dir=$RUN_DIR  GPUs=${CUDA_VISIBLE_DEVICES:-<all visible>}"

CMD=(.venv/bin/python data/toy_cliff.py --config "$CONFIG" --run-dir "$RUN_DIR")
if [ -n "$MODEL" ]; then
  CMD+=(--model-path "$MODEL")
fi
CMD+=("$@")

if $BACKGROUND; then
  mkdir -p logs
  LOG="logs/toy_cliff_$(basename "$RUN_DIR").log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo ">>> started in background: PID $!"
  echo ">>> follow with: tail -f $LOG"
else
  exec "${CMD[@]}"
fi
