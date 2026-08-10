#!/usr/bin/env bash
# Pass-rate distribution experiment: N sampled questions x K rollouts, graded,
# reported as a correct-count histogram with cliff/frontier/solved classes.
# MODEL is optional (default: the config's model.base); accepts an HF hub id,
# full checkpoint dir, or PEFT LoRA adapter dir (auto-merged).
#
# Usage:
#   bash data/run_passrate.sh [MODEL] [-c CONFIG] [-g GPUS] [-b] [-- EXTRA_ARGS]
#
#   bash data/run_passrate.sh -g 0,1
#   bash data/run_passrate.sh runs/ei_qwen3_4b/iter_2/ckpt -g 0 -b
#   bash data/run_passrate.sh -- --override data.adapter_args.n_questions=1000
#   bash data/run_passrate.sh -- --dry-run                    # no GPU: check data+prompts
#
# Results land in runs/passrate/<slug>/metrics.json (+ question_stats.jsonl).
# Re-running resumes: frozen questions and .done-marked samples are skipped;
# changed rollout/model params regenerate into a fresh pool dir.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL=""
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  MODEL="$1"
  shift
fi

CONFIG="data/configs/passrate.yaml"
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
SLUG="$(echo "${MODEL:-default}_$(basename "$CONFIG" .yaml)" | tr '/ ' '__')"
SLUG="${SLUG%/}"
RUN_DIR="runs/passrate/$SLUG"
echo ">>> model=${MODEL:-<config model.base>}  config=$CONFIG  run_dir=$RUN_DIR  GPUs=${CUDA_VISIBLE_DEVICES:-<all visible>}"

CMD=(.venv/bin/python data/passrate.py --config "$CONFIG" --run-dir "$RUN_DIR")
if [ -n "$MODEL" ]; then
  CMD+=(--model-path "$MODEL")
fi
CMD+=("$@")

if $BACKGROUND; then
  mkdir -p logs
  LOG="logs/passrate_${SLUG}_$(date +%Y%m%d_%H%M%S).log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo ">>> started in background: PID $!"
  echo ">>> follow with: tail -f $LOG"
else
  exec "${CMD[@]}"
fi
