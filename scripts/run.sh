#!/usr/bin/env bash
# Convenience launcher for the expert-iteration loop.
#
# Usage:
#   bash scripts/run.sh                                   # ei_default.yaml, all visible GPUs
#   bash scripts/run.sh -g 0,1,2                          # pin GPUs
#   bash scripts/run.sh -c configs/smoke.yaml             # different config
#   bash scripts/run.sh -b                                # background (nohup) + log file
#   bash scripts/run.sh -g 2 -- --override rollout.n=4 --iterations 1
#
# Everything after `--` is passed straight to `expert_iter.loop`
# (e.g. --override a.b=c, --iterations N, --start-iter K, --force STAGE).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/ei_default.yaml"
GPUS=""
BACKGROUND=false

while getopts ":g:c:bh" opt; do
  case "$opt" in
    g) GPUS="$OPTARG" ;;
    c) CONFIG="$OPTARG" ;;
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

# On PCIe-only machines (no NVLink) P2P may be silently blocked by ACS/IOMMU:
# cudaDeviceCanAccessPeer still says yes, so NCCL picks P2P and its first
# collective hangs until the 600s watchdog (seen on srv04). Host shared memory
# is always safe on PCIe and costs little, so disable P2P there; NVLink
# machines keep P2P. Export NCCL_P2P_DISABLE yourself to override either way.
if [ -z "${NCCL_P2P_DISABLE:-}" ] && command -v nvidia-smi >/dev/null; then
  if ! nvidia-smi topo -m 2>/dev/null | grep -qE "NV[0-9]"; then
    export NCCL_P2P_DISABLE=1
    echo ">>> no NVLink detected: NCCL_P2P_DISABLE=1"
  fi
fi
echo ">>> config=$CONFIG  GPUs=${CUDA_VISIBLE_DEVICES:-<all visible>}  extra args: $*"

CMD=(.venv/bin/python -m expert_iter.loop --config "$CONFIG" "$@")

if $BACKGROUND; then
  mkdir -p logs
  LOG="logs/loop_$(date +%Y%m%d_%H%M%S).log"
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo ">>> started in background: PID $!"
  echo ">>> follow with: tail -f $LOG"
else
  exec "${CMD[@]}"
fi
