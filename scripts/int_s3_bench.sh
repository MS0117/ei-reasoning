#!/usr/bin/env bash
# InT L2 -> S3 -> BENCHMARK. The question this run answers is "does training on
# benchmark-aligned (InT) cliffs move the external benchmarks vs the base?", so
# there is no A/B cliff split and no attractor re-roll: the held-out-cliff
# transfer mechanism is already established on OpenR1 (docs/L3_results_20260826.md).
#   - every cliff rescue goes into training  (b_frac 0 -> exclude list empty)
#   - the readout is scripts/eval_bench.sh on the same 5 sets, same sampling,
#     as the ALREADY-MEASURED base and OpenR1-S3 runs -> a 3-way comparison
#     for free:
#       base      runs/bench/Qwen_Qwen3-4B-Instruct-2507_bench_eval_20260901_110721
#       OpenR1-S3 runs/bench/runs_L3_S3_20260826_011420_iter_0_ckpt_bench_eval_20260901_234449
#
#   [FREEZE=runs/L2_freeze_int_...] [BENCH_CFG=configs/bench_eval.yaml] bash scripts/int_s3_bench.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FREEZE="${FREEZE:-runs/L2_freeze_int_20260902_052108}"
JOBID="${JOBID:-11817}"
GPU="${GPU:-0,1}"
BENCH_CFG="${BENCH_CFG:-configs/bench_eval.yaml}"
IT="$FREEZE/iter_0"

hours_left() {
  local end; end="$(scontrol show job "$JOBID" 2>/dev/null | grep -o 'EndTime=[^ ]*' | cut -d= -f2)"
  [ -n "$end" ] || { echo "unknown"; return; }
  .venv/bin/python -c "
import datetime
end=datetime.datetime.fromisoformat('$end')
print(round(max(0,(end-datetime.datetime.now()).total_seconds())/3600, 2))"
}

echo ">>> [s3b] freeze=$FREEZE  waiting for build_dataset ..."
until [ -f "$IT/dataset/stats.json" ]; do
  if ! pgrep -f "expert_iter.loop .*$(basename "$FREEZE")" >/dev/null && \
     ! pgrep -f "expert_iter\.(improve|filters|build_dataset)" >/dev/null; then
    echo ">>> [s3b] freeze process is gone and dataset/stats.json never appeared — aborting" >&2
    exit 1
  fi
  sleep 120
done
echo ">>> [s3b] freeze complete at $(date +%H:%M), $(hours_left)h left in the allocation"

# ---- split with an EMPTY holdout: A = every cliff, exclude list = [] ---------
# (l3_arm.sh requires the file; build_dataset's `if excluded:` no-ops on [])
.venv/bin/python scripts/cliff_split.py --run-dir "$FREEZE" --b-frac 0.0
.venv/bin/python -c "
import json; d=json.load(open('$IT/cliff_split.json'))
assert d['exclude']==[], d['exclude'][:3]
print(f\"[s3b] A={len(d['A'])} cliffs train, B={len(d['B'])} held out\")"

# ---- S3: rho=0.3, per-question norm, guard on, negatives off ----------------
if ! ls -d runs/L3_S3_int_*/iter_0/ckpt >/dev/null 2>&1; then
  SKIP_READOUT=1 ARM_TAG=int bash scripts/l3_arm.sh S3 "$FREEZE" "$GPU"
fi
CKPT="$(ls -d runs/L3_S3_int_*/iter_0/ckpt 2>/dev/null | head -1 || true)"
[ -n "$CKPT" ] || { echo ">>> [s3b] no S3 checkpoint — aborting" >&2; exit 1; }
echo ">>> [s3b] trained: $CKPT  ($(hours_left)h left)"
.venv/bin/python -c "
import json; s=json.load(open('$(dirname "$CKPT")/dataset/stats.json'))
print('[s3b] train set:', json.dumps({k:v for k,v in s.items() if isinstance(v,(int,float))}))" || true

# ---- the actual endpoint ----------------------------------------------------
bash scripts/eval_bench.sh "$CKPT" -c "$BENCH_CFG" -g "$GPU"

echo
echo ">>> [s3b] DONE ($(hours_left)h left). Compare against the measured references:"
echo "    InT-S3    runs/bench/$(ls -t runs/bench | head -1)/iter_0/benchmark_eval/metrics.json"
echo "    base      runs/bench/Qwen_Qwen3-4B-Instruct-2507_bench_eval_20260901_110721/iter_0/benchmark_eval/metrics.json"
echo "    OpenR1-S3 runs/bench/runs_L3_S3_20260826_011420_iter_0_ckpt_bench_eval_20260901_234449/iter_0/benchmark_eval/metrics.json"
