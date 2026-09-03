#!/usr/bin/env bash
# S3-only tail of the InT L2->L3 pipeline, sized to the srun allocation.
# (2026-09-02: srv04 expires ~12:40, and B x32 re-rolls are the binding cost, so
# B is chosen from the time actually left instead of the pre-registered b_frac.)
#
#   [FREEZE=runs/L2_freeze_int_...] [JOBID=11817] bash scripts/int_s3_only.sh
#
# Waits for the freeze's build_dataset, then, in cut-off-cost order:
#   cliff_split (adaptive b_frac)  ->  S3 TRAIN (not resumable, so first)
#   -> base floor re-roll on B     ->  S3 B re-roll  ->  attractor compare
# Both re-rolls are resumable, so losing the allocation mid-re-roll costs only
# the current shard; a killed training would cost the whole run.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FREEZE="${FREEZE:-runs/L2_freeze_int_20260902_052108}"
JOBID="${JOBID:-11817}"
GPU="${GPU:-0,1}"
N="${N:-32}"                 # samples per question (keep 32: avg@32 comparability)
SEC_PER_SAMPLE="${SEC_PER_SAMPLE:-3.5}"
TRAIN_H="${TRAIN_H:-3.2}"    # S3 train budget
BUFFER_H="${BUFFER_H:-0.6}"
B_MIN="${B_MIN:-60}"; B_MAX="${B_MAX:-268}"
IT="$FREEZE/iter_0"

hours_left() {   # from slurm; falls back to a fixed guess if scontrol is gone
  local end; end="$(scontrol show job "$JOBID" 2>/dev/null | grep -o 'EndTime=[^ ]*' | cut -d= -f2)"
  [ -n "$end" ] || { echo "${FALLBACK_H:-8}"; return; }
  .venv/bin/python -c "
import datetime,sys
end=datetime.datetime.fromisoformat('$end')
print(round(max(0,(end-datetime.datetime.now()).total_seconds())/3600, 2))"
}

echo ">>> [s3] freeze=$FREEZE job=$JOBID  waiting for build_dataset ..."
until [ -f "$IT/dataset/stats.json" ]; do
  if ! pgrep -f "expert_iter.loop .*$(basename "$FREEZE")" >/dev/null && \
     ! pgrep -f "expert_iter\.(improve|filters|build_dataset)" >/dev/null; then
    echo ">>> [s3] freeze process is gone and dataset/stats.json never appeared — aborting" >&2
    exit 1
  fi
  sleep 120
done
echo ">>> [s3] freeze complete at $(date +%H:%M)"

# ---- adaptive B ------------------------------------------------------------
LEFT="$(hours_left)"
N_CLIFF="$(wc -l < "$IT/partition/unsolved.jsonl")"
read -r B_TARGET B_FRAC <<EOF
$(.venv/bin/python -c "
left=float('$LEFT'); n=int('$N_CLIFF')
budget=max(0.0, left-float('$TRAIN_H')-float('$BUFFER_H'))
b=int(budget*3600/(2*int('$N')*float('$SEC_PER_SAMPLE')))
b=max(int('$B_MIN'), min(int('$B_MAX'), b, n//2))
print(b, round(b/n, 4))")
EOF
echo ">>> [s3] ${LEFT}h left, $N_CLIFF cliffs -> B=$B_TARGET (b_frac=$B_FRAC), A=$((N_CLIFF-B_TARGET))"

.venv/bin/python scripts/cliff_split.py --run-dir "$FREEZE" --b-frac "$B_FRAC"
SPLIT="$IT/cliff_split.json"

# ---- 1) S3 training (first: a killed train is the only unrecoverable loss) --
if ! ls -d runs/L3_S3_int_*/iter_0/ckpt >/dev/null 2>&1; then
  SKIP_READOUT=1 ARM_TAG=int bash scripts/l3_arm.sh S3 "$FREEZE" "$GPU"
fi
CKPT="$(ls -d runs/L3_S3_int_*/iter_0/ckpt 2>/dev/null | head -1 || true)"
[ -n "$CKPT" ] || { echo ">>> [s3] no S3 checkpoint — aborting" >&2; exit 1; }
ARM_DIR="$(dirname "$(dirname "$CKPT")")"
LEFT2="$(hours_left)"
NB="$(.venv/bin/python -c "
import json; print(len(json.load(open('$SPLIT'))['B']))")"
NEED="$(.venv/bin/python -c "
print(round(2*int('$NB')*int('$N')*float('$SEC_PER_SAMPLE')/3600, 2))")"
echo ">>> [s3] trained: $CKPT"
echo ">>> [s3] ${LEFT2}h left, B=$NB needs ~${NEED}h for both re-rolls"
.venv/bin/python -c "
import sys
sys.exit(0 if float('$LEFT2') >= float('$NEED') else 1)" || \
  echo ">>> [s3] WARNING: budget tight — re-rolls are resumable, resume with the same command"

# ---- 2) base floor on B, then the arm's B re-roll (both resumable) ---------
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/cliff_reroll.py \
  --run-dir "$FREEZE" --qids-file "$SPLIT:B" --n "$N" --passes 1
echo ">>> [s3] base floor done ($(hours_left)h left)"

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/cliff_reroll.py \
  --run-dir "$FREEZE" --model-path "$CKPT" --qids-file "$SPLIT:B" \
  --n "$N" --passes 1 --out "$ARM_DIR/iter_0/reroll_B"
echo ">>> [s3] arm re-roll done ($(hours_left)h left)"

# ---- 3) the readout --------------------------------------------------------
.venv/bin/python scripts/attractor_mass.py \
  --verdicts "$IT/reroll/pass_0/verdicts.jsonl" \
  --compare "$ARM_DIR/iter_0/reroll_B/pass_0/verdicts.jsonl" \
  --qids-file "$SPLIT:B" --out "$ARM_DIR/iter_0/attractor_B_compare.json"

echo
echo ">>> [s3] DONE. B transfer: $ARM_DIR/iter_0/attractor_B_compare.json"
