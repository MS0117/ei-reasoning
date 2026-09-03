#!/usr/bin/env bash
# srv04 (A100) — the node that carries the STANDARD-5-set comparison, because
# every existing reference on those sets (base, S0, S1, OpenR1-S3, all measured
# 2026-09-01) was produced on this exact hardware. Measuring InT-S3 here makes
# it directly comparable without re-running any reference.
#   S3 train -> S3 standard-set eval -> S0 train (last: least time-critical,
#   and srv08 can pick it up if this allocation ends first).
# The hard sets and all attractor re-rolls belong to srv08 (one architecture per
# comparison); see scripts/int_bench_srv08.sh.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FREEZE="${FREEZE:-runs/L2_freeze_int_20260902_052108}"
export B_FRAC="${B_FRAC:-0.25}"    # 536 cliffs -> B=134 held out, A=402 trained
export GPU="${GPU:-0,1}"

# S3: train + eval on configs/bench_eval.yaml (the A100 reference family)
MAKE_SPLIT=1 ARM=S3 BENCH_CFG=configs/bench_eval.yaml bash scripts/int_arm_bench.sh

# S0 control: train only — its eval belongs next to S3's on ONE architecture,
# so it is measured elsewhere. This is the expendable tail: a half-trained S0
# is worth nothing, so only start it if a full run still fits.
LEFT="$(scontrol show job "${JOBID:-11817}" 2>/dev/null | grep -o 'EndTime=[^ ]*' | cut -d= -f2 |
  xargs -I{} .venv/bin/python -c "
import datetime,sys
print(round(max(0,(datetime.datetime.fromisoformat('{}')-datetime.datetime.now()).total_seconds())/3600,2))" 2>/dev/null || echo 0)"
NEED="${S0_NEED_H:-3.2}"
if .venv/bin/python -c "import sys; sys.exit(0 if float('${LEFT:-0}') >= float('$NEED') else 1)"; then
  echo ">>> [srv04] ${LEFT}h left — training the S0 control"
  TRAIN_ONLY=1 ARM=S0 bash scripts/int_arm_bench.sh
else
  echo ">>> [srv04] only ${LEFT}h left (<${NEED}h) — SKIPPING S0; train it in the next allocation:"
  echo "    TRAIN_ONLY=1 ARM=S0 FREEZE=$FREEZE bash scripts/int_arm_bench.sh"
fi
echo ">>> [srv04] done"
