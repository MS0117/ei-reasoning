#!/usr/bin/env bash
# End-to-end smoke test: 1 EI iteration with Qwen3-0.6B on tiny data.
#
#   bash scripts/smoke.sh [-c CONFIG] [-g GPU_ID] [-- --override a.b=c ...]
#
#   -c  config yaml   (default configs/smoke.yaml)
#   -g  GPU id(s)     (default 0)
#
# The new-method path (lora_sft + project-back + C(y) selection):
#   bash scripts/smoke.sh -c configs/methods/smoke_lora.yaml -g 0
#
# The positional form documented in CLAUDE.md still works:
#   bash scripts/smoke.sh [GPU_ID] [CONFIG]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU="0"
CONFIG="configs/smoke.yaml"

# Backwards-compatible positional form: `smoke.sh 0 configs/x.yaml`. Only taken
# when the first argument is not a flag, so the getopts form below is unaffected.
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  GPU="$1"; shift
  if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then CONFIG="$1"; shift; fi
fi

while getopts ":c:g:h" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    g) GPU="$OPTARG" ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "unknown option -$OPTARG (try -h)" >&2; exit 2 ;;
    :) echo "option -$OPTARG needs a value" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
RUN_NAME="smoke_$(date +%Y%m%d_%H%M%S)"
echo ">>> config=$CONFIG  GPU=$GPU  run.name=$RUN_NAME"

CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -m expert_iter.loop \
  --config "$CONFIG" --override "run.name=$RUN_NAME" "$@"

echo
echo "=== smoke metrics ==="
cat "runs/$RUN_NAME/metrics.jsonl"
