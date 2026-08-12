#!/usr/bin/env bash
# Pass-rate distribution experiment: N sampled questions x K rollouts, graded,
# reported as a correct-count histogram with cliff/frontier/solved classes.
# MODEL is optional (default: the config's model.base); accepts an HF hub id,
# full checkpoint dir, or PEFT LoRA adapter dir (auto-merged).
#
# Usage:
#   bash data/run_passrate.sh [MODEL] [-c CONFIG] [-g GPUS] [-r RUN_DIR] [-b] [-- EXTRA_ARGS]
#
#   bash data/run_passrate.sh -g 0,1
#   bash data/run_passrate.sh runs/ei_qwen3_4b/iter_2/ckpt -g 0 -b
#   bash data/run_passrate.sh -- --override data.adapter_args.n_questions=1000
#   bash data/run_passrate.sh -- --dry-run                    # no GPU: check data+prompts
#
# Every launch gets a FRESH timestamped dir runs/passrate/<slug>_<ts>/ so
# identical re-runs never overwrite results. To RESUME a crashed/partial run
# (frozen questions + .done-marked samples are skipped), point -r at its dir:
#   bash data/run_passrate.sh -r runs/passrate/default_openr1_passrate_20260811_150000 -g 0,1
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

# One run dir per (model, dataset, config) so results never mix and the
# frozen config snapshot stays truthful. The dataset tag comes from
# adapter_args.preset, else the hf_name basename (comments never match the
# anchored greps); config-file renames still separate anything else.
DTAG="$(grep -m1 -oE '^[[:space:]]*preset:[[:space:]]*[^[:space:]#]+' "$CONFIG" | awk '{print $2}')"
if [ -z "$DTAG" ]; then
  DTAG="$(grep -m1 -oE '^[[:space:]]*hf_name:[[:space:]]*[^[:space:]#]+' "$CONFIG" | awk '{print $2}')"
  DTAG="${DTAG##*/}"
fi
SLUG="$(echo "${MODEL:-default}_${DTAG:+${DTAG}_}$(basename "$CONFIG" .yaml)" | tr '/ ' '__')"
SLUG="${SLUG%/}"
if [ -n "$RESUME_DIR" ]; then
  [ -d "$RESUME_DIR" ] || { echo "resume dir not found: $RESUME_DIR" >&2; exit 1; }
  RUN_DIR="$RESUME_DIR"
else
  RUN_DIR="runs/passrate/${SLUG}_$(date +%Y%m%d_%H%M%S)"
fi
echo ">>> model=${MODEL:-<config model.base>}  config=$CONFIG  run_dir=$RUN_DIR  GPUs=${CUDA_VISIBLE_DEVICES:-<all visible>}"

CMD=(.venv/bin/python data/passrate.py --config "$CONFIG" --run-dir "$RUN_DIR")
if [ -n "$MODEL" ]; then
  CMD+=(--model-path "$MODEL")
fi
CMD+=("$@")

if $BACKGROUND; then
  mkdir -p logs
  LOG="logs/passrate_$(basename "$RUN_DIR").log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo ">>> started in background: PID $!"
  echo ">>> follow with: tail -f $LOG"
else
  exec "${CMD[@]}"
fi
