#!/usr/bin/env bash
# Standalone benchmark eval for ANY model: HF hub id, full checkpoint dir, or
# PEFT LoRA adapter dir (content-keyed merge under <adapter>/merged/).
# MODEL is optional (default: the config's model.base).
#
# Usage:
#   bash scripts/eval_bench.sh [MODEL] [-c CONFIG] [-g GPUS] [-r RUN_DIR] [-b] [-- EXTRA_ARGS]
#
#   bash scripts/eval_bench.sh -g 0,1                           # config's model.base
#   bash scripts/eval_bench.sh Qwen/Qwen3-4B-Instruct-2507
#   bash scripts/eval_bench.sh runs/ei_qwen3_4b/iter_2/ckpt -g 0,1
#   bash scripts/eval_bench.sh /path/to/lora_ckpt -c configs/bench_eval.yaml -b
#   bash scripts/eval_bench.sh Qwen/Qwen3-4B -- --override "eval.benchmarks=[{name: aime24, n: 32}]"
#
# Results land in runs/bench/<model-slug>_<ts>/iter_0/benchmark_eval/metrics.json —
# a fresh timestamped dir per launch, so identical re-runs never overwrite.
# <model-slug> names the model actually graded (the MODEL argument, else the
# config's model.base), so the dir and its wandb run are never mislabelled.
# To RESUME a partial run (finished benchmarks skip via .done markers):
#   bash scripts/eval_bench.sh MODEL -r runs/bench/<model-slug>_<ts> [-g ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Leading non-flag argument is MODEL; omitted -> the config's model.base.
MODEL=""
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  MODEL="${1%/}"        # trailing slash would double up in the run-dir slug
  shift
fi

CONFIG="configs/bench_eval.yaml"
GPUS=""
BACKGROUND=false
RESUME_DIR=""

while getopts ":c:g:r:bh" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
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

# The model that will actually be graded: the MODEL argument when given, else
# the config's model.base (after --override). benchmark_eval.py applies exactly
# this fallback, so resolving it here keeps the run-dir slug — and the wandb run
# name, which is the dir basename — honest about which model produced the
# numbers, instead of naming the dir "default" whenever MODEL was omitted.
CONFIG_MODEL="$(.venv/bin/python - "$CONFIG" "$@" <<'PY'
import sys

from expert_iter.config import Config

cfg_path, argv, overrides, i = sys.argv[1], sys.argv[2:], [], 0
while i < len(argv):
    if argv[i] == "--override" and i + 1 < len(argv):
        overrides.append(argv[i + 1]); i += 2
    elif argv[i].startswith("--override="):
        overrides.append(argv[i].split("=", 1)[1]); i += 1
    else:
        i += 1
print(Config.load(cfg_path, overrides=overrides).model.base)
PY
)"
if [ -n "$MODEL" ] && [ "$MODEL" != "$CONFIG_MODEL" ]; then
  echo ">>> NOTE: MODEL argument wins over the config's model.base ($CONFIG_MODEL)"
fi
MODEL="${MODEL:-$CONFIG_MODEL}"

# Fresh timestamped run dir per launch so identical re-runs never overwrite
# results; the frozen config snapshot stays truthful. To RESUME a partial run
# (per-benchmark .done markers skip finished benchmarks), pass -r <run_dir>.
SLUG="$(echo "${MODEL%/}_$(basename "$CONFIG" .yaml)" | tr '/ ' '__')"
if [ -n "$RESUME_DIR" ]; then
  [ -d "$RESUME_DIR" ] || { echo "resume dir not found: $RESUME_DIR" >&2; exit 1; }
  RUN_DIR="$RESUME_DIR"
else
  RUN_DIR="runs/bench/${SLUG}_$(date +%Y%m%d_%H%M%S)"
fi
echo ">>> model=$MODEL  config=$CONFIG  run_dir=$RUN_DIR  GPUs=${CUDA_VISIBLE_DEVICES:-<all visible>}"

# --model-path is always explicit now that MODEL is resolved, so the model named
# in the run dir is the model the stage loads and records in metrics.json.
CMD=(.venv/bin/python -m expert_iter.benchmark_eval
     --config "$CONFIG" --run-dir "$RUN_DIR" --iter 0 --model-path "$MODEL")
CMD+=("$@")

if $BACKGROUND; then
  mkdir -p logs
  LOG="logs/bench_$(basename "$RUN_DIR").log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo ">>> started in background: PID $!"
  echo ">>> follow with: tail -f $LOG"
else
  exec "${CMD[@]}"
fi
