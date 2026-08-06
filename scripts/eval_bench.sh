#!/usr/bin/env bash
# Standalone benchmark eval for ANY model: HF hub id, full checkpoint dir, or
# PEFT LoRA adapter dir (auto-merged into <adapter>/merged/ on first use).
#
# Usage:
#   bash scripts/eval_bench.sh MODEL [-c CONFIG] [-g GPUS] [-b] [-- EXTRA_ARGS]
#
#   bash scripts/eval_bench.sh Qwen/Qwen3-4B-Instruct-2507
#   bash scripts/eval_bench.sh runs/ei_qwen3_4b/iter_2/ckpt -g 0,1
#   bash scripts/eval_bench.sh /path/to/lora_ckpt -c configs/bench_eval.yaml -b
#   bash scripts/eval_bench.sh Qwen/Qwen3-4B -- --override "eval.benchmarks=[{name: aime24, n: 32}]"
#
# Results land in runs/bench/<model-slug>/iter_0/benchmark_eval/metrics.json.
# Re-running skips finished benchmarks (per-benchmark .done markers); a changed
# config or model re-generates only what changed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:?usage: eval_bench.sh MODEL [-c CONFIG] [-g GPUS] [-b] [-- EXTRA_ARGS]}"
shift

CONFIG="configs/bench_eval.yaml"
GPUS=""
BACKGROUND=false

while getopts ":c:g:bh" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
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

# One run dir per (model, config) so results for different models never mix
# and the frozen config snapshot stays truthful.
SLUG="$(echo "${MODEL%/}_$(basename "$CONFIG" .yaml)" | tr '/ ' '__')"
RUN_DIR="runs/bench/$SLUG"
echo ">>> model=$MODEL  config=$CONFIG  run_dir=$RUN_DIR  GPUs=${CUDA_VISIBLE_DEVICES:-<all visible>}"

CMD=(.venv/bin/python -m expert_iter.benchmark_eval
     --config "$CONFIG" --run-dir "$RUN_DIR" --iter 0 --model-path "$MODEL" "$@")

if $BACKGROUND; then
  mkdir -p logs
  LOG="logs/bench_${SLUG}_$(date +%Y%m%d_%H%M%S).log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo ">>> started in background: PID $!"
  echo ">>> follow with: tail -f $LOG"
else
  exec "${CMD[@]}"
fi
