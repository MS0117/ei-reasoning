#!/usr/bin/env bash
# L5 gold-SFT baseline (offline distillation) in one command.
#
#   bash scripts/l5_gold_sft.sh [-c CONFIG] [-g GPUS] [-r RUN_NAME] [-b] [-- --override a.b=c ...]
#
#   -c  config yaml            (default configs/methods/l5_gold_sft.yaml)
#   -g  CUDA_VISIBLE_DEVICES   (default: all visible)
#   -r  resume/reuse this run name instead of a fresh timestamped one
#   -b  run the loop in the background (same as run.sh -b)
#
# Two steps, because this arm has no rollout: build the dataset from y* on CPU,
# then run the loop, whose stage list (configs/methods/l5_gold_sft.yaml) is
# [train, eval, benchmark_eval]. Both calls take the SAME config, the same
# run.name and the same extra overrides, so the config hash matches and the loop
# finds the dataset already .done.
#
# A fresh timestamped run dir per launch, like run.sh — nothing is overwritten.
#
# Extra overrides go after a `--` separator, the same convention as run.sh and
# data/run_toy_cliff.sh (getopts stops there; without it `--override` is read as
# an unknown flag). The 6-epoch variant — see the config header for why 2 is the
# default:
#   bash scripts/l5_gold_sft.sh -g 0,1 -b -- --override train.sft.epochs=6
#
# A smaller/faster variant is a config, not a flag — copy the yaml and pass -c:
#   bash scripts/l5_gold_sft.sh -c /path/to/l5_gold_sft_smoke.yaml -g 0,1 -b
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/methods/l5_gold_sft.yaml"
GPUS=""
RUN_NAME=""
BACKGROUND=false

while getopts ":c:g:r:bh" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    r) RUN_NAME="$OPTARG" ;;
    b) BACKGROUND=true ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "unknown option -$OPTARG (try -h)" >&2; exit 2 ;;
    :) echo "option -$OPTARG needs a value" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA_OV=("$@")

[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
[ -x .venv/bin/python ] || { echo ".venv missing — run: bash scripts/setup.sh --skip-lean" >&2; exit 1; }

if [ -z "$RUN_NAME" ]; then
  BASE_NAME="$(grep -m1 -oE '^[[:space:]]*name:[[:space:]]*[^[:space:]#]+' "$CONFIG" | awk '{print $2}')"
  RUN_NAME="${BASE_NAME:-l5_gold_sft}_$(date +%Y%m%d_%H%M%S)"
fi
echo ">>> config=$CONFIG  run.name=$RUN_NAME  GPUs=${GPUS:-<all visible>}  extra: ${EXTRA_OV[*]:-<none>}"

# 1) dataset from y*, CPU, seconds. Idempotent: a rerun with a matching config
#    hash is a no-op, so this is also the resume path.
.venv/bin/python scripts/build_gold_sft.py -c "$CONFIG" \
  --override "run.name=$RUN_NAME" ${EXTRA_OV[@]+"${EXTRA_OV[@]}"}

# 2) the loop, which the config restricts to [train, eval, benchmark_eval].
# `if` (not `a && b`) throughout: a false test at top level trips `set -e`.
RUN_FLAGS=(-c "$CONFIG" -r "$RUN_NAME")
if [ -n "$GPUS" ]; then RUN_FLAGS+=(-g "$GPUS"); fi
if $BACKGROUND; then RUN_FLAGS+=(-b); fi
bash scripts/run.sh "${RUN_FLAGS[@]}" ${EXTRA_OV[@]+-- "${EXTRA_OV[@]}"}

echo ">>> dataset: runs/$RUN_NAME/iter_0/dataset/stats.json"
if $BACKGROUND; then echo ">>> follow:  tail -f logs/loop_$RUN_NAME.log"; fi
