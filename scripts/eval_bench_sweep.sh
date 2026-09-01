#!/usr/bin/env bash
# Run scripts/eval_bench.sh over SEVERAL models, one after another, then print
# one comparison table. Sequential by design: each model needs the whole GPU.
#
# Usage:
#   bash scripts/eval_bench_sweep.sh [-c CONFIG] [-g GPUS] [-b] [-F] MODEL...
#
#   MODEL   an HF hub id, a checkpoint dir, a LoRA adapter dir, or the literal
#           word `base` (= grade the config's own model.base — the floor every
#           other row is compared against).
#   -b      background the WHOLE sweep (nohup + one sweep log); the individual
#           models still run one at a time.
#   -F      re-run models that already have a finished run under runs/bench/.
#
# Example — the L3 objective arms against their base floor (5 x ~4.5 h):
#   bash scripts/eval_bench_sweep.sh -b base \
#       runs/L3_S0_20260826_081534/iter_0/ckpt \
#       runs/L3_S1_20260826_141540/iter_0/ckpt \
#       runs/L3_S3_20260826_011420/iter_0/ckpt \
#       runs/L3_S4v0_20260827_122953/iter_0/ckpt
#
# RESUMABLE, which matters because a full sweep outlives most srun allocations:
# a model whose newest runs/bench/<slug>_* already holds a finished
# metrics.json is SKIPPED, and a partial one is RESUMED into (`eval_bench.sh
# -r`, so finished benchmarks skip on their .done markers). Re-running the same
# command after a timeout therefore picks up where it stopped.
#
# A failing model does not stop the sweep; the summary marks it FAILED and the
# exit code is non-zero so a wrapper can notice.
#
# EVERY model must be graded at the same `n` — avg@n is unbiased at any n but
# pass@k is not comparable across n. That is why the base floor belongs in the
# same sweep rather than being reused from an older run at a different n.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/bench_eval.yaml"
GPUS=""
BACKGROUND=false
FORCE=false

while getopts ":c:g:bFh" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    b) BACKGROUND=true ;;
    F) FORCE=true ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "unknown option -$OPTARG (try -h)" >&2; exit 2 ;;
    :) echo "option -$OPTARG needs a value" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

[ $# -ge 1 ] || { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
[ -x .venv/bin/python ] || { echo ".venv missing — run: bash scripts/setup.sh --skip-lean" >&2; exit 1; }

MODELS=("$@")

# Re-exec into the background as ONE process so the models stay sequential.
if $BACKGROUND; then
  mkdir -p logs
  SWEEP_LOG="logs/bench_sweep_$(date +%Y%m%d_%H%M%S).log"
  ARGS=(-c "$CONFIG")
  if [ -n "$GPUS" ]; then ARGS+=(-g "$GPUS"); fi
  if $FORCE; then ARGS+=(-F); fi
  nohup bash "$0" "${ARGS[@]}" "${MODELS[@]}" > "$SWEEP_LOG" 2>&1 &
  echo ">>> sweep started in background: PID $!  (${#MODELS[@]} models, sequential)"
  echo ">>> follow with: tail -f $SWEEP_LOG"
  exit 0
fi

CONFIG_BASE="$(.venv/bin/python -c "
from expert_iter.config import Config
print(Config.load('$CONFIG').model.base)")"
CONFIG_SLUG="$(basename "$CONFIG" .yaml)"

# exit 0 iff two configs grade the same way (same benchmark list incl. n, same
# verifier). Kept as a string so the loop can call it without a nested heredoc.
GRADING_KEY_PY='
import dataclasses, json, sys
from expert_iter.config import Config
def key(p):
    c = Config.load(p)
    return (json.dumps([dataclasses.asdict(b) for b in c.eval.benchmarks],
                       sort_keys=True), c.eval.benchmark_verifier)
sys.exit(0 if key(sys.argv[1]) == key(sys.argv[2]) else 1)
'

echo "=== bench sweep: ${#MODELS[@]} model(s), config=$CONFIG, GPUs=${GPUS:-<all visible>}"
echo "=== base of config: $CONFIG_BASE"
echo

RUN_DIRS=()
STATUS=()
FAILED=0

for model in "${MODELS[@]}"; do
  # `base` -> no MODEL argument, so eval_bench.sh falls back to model.base.
  if [ "$model" = "base" ]; then
    resolved="$CONFIG_BASE"
    model_args=()
  else
    resolved="${model%/}"
    model_args=("$resolved")
  fi
  # Must match eval_bench.sh's slug exactly, or resume/skip would never hit.
  slug="$(echo "${resolved%/}_${CONFIG_SLUG}" | tr '/ ' '__')"

  existing=""
  for d in $(ls -d "runs/bench/${slug}_"* 2>/dev/null | sort); do existing="$d"; done

  # An older run of the SAME model may have been graded under different
  # benchmarks/n (the n=64 -> n=32 change of 2026-08-31 is exactly this case).
  # Reusing it would silently mix sample counts across rows, so compare the
  # frozen snapshot's grading settings and start fresh when they differ.
  comparable=false
  if [ -n "$existing" ] && [ -f "$existing/config.yaml" ]; then
    if .venv/bin/python -c "$GRADING_KEY_PY" "$CONFIG" "$existing/config.yaml" >/dev/null 2>&1; then
      comparable=true
    fi
  fi

  resume_args=()
  if [ -n "$existing" ] && ! $FORCE; then
    if ! $comparable; then
      echo "=== IGNORING $existing — graded with different benchmarks/n;"
      echo "===   running fresh so every row of the table shares one n."
    elif [ -e "$existing/iter_0/benchmark_eval/metrics.json.done" ]; then
      echo "=== SKIP  $resolved  (already finished: $existing)"
      RUN_DIRS+=("$existing"); STATUS+=("skipped"); echo
      continue
    else
      echo "=== RESUME $resolved  into $existing"
      resume_args=(-r "$existing")
    fi
  fi

  echo "=== [$(date '+%F %H:%M:%S')] START $resolved"
  gpu_args=()
  if [ -n "$GPUS" ]; then gpu_args=(-g "$GPUS"); fi
  # No -b: the point of the sweep is that these do NOT overlap.
  if bash scripts/eval_bench.sh "${model_args[@]}" -c "$CONFIG" \
        "${gpu_args[@]}" "${resume_args[@]}"; then
    ok=true
  else
    ok=false
  fi

  # eval_bench.sh made a fresh timestamped dir unless we resumed; find it now.
  newest=""
  for d in $(ls -d "runs/bench/${slug}_"* 2>/dev/null | sort); do newest="$d"; done
  RUN_DIRS+=("${newest:-<none>}")
  if $ok; then
    echo "=== [$(date '+%F %H:%M:%S')] DONE  $resolved -> ${newest:-<none>}"
    STATUS+=("ok")
  else
    echo "=== [$(date '+%F %H:%M:%S')] FAILED $resolved (continuing)" >&2
    STATUS+=("FAILED")
    FAILED=$((FAILED + 1))
  fi
  echo
done

echo "=============================================================="
echo "=== sweep summary"
echo "=============================================================="
.venv/bin/python - "${RUN_DIRS[@]}" <<'PY'
import json
import sys
from pathlib import Path

rows, sets = [], []
for d in sys.argv[1:]:
    m = Path(d) / "iter_0" / "benchmark_eval" / "metrics.json"
    if not m.exists():
        rows.append((d, None, {}))
        continue
    j = json.loads(m.read_text())
    per = {}
    for k, v in j.items():
        if "/" not in k:
            continue
        name, metric = k.split("/", 1)
        # avg@n IS pass@1 (the unbiased per-sample mean); n varies by config.
        if metric.startswith("avg@"):
            per[name] = (v, int(metric.split("@")[1]))
            if name not in sets:
                sets.append(name)
    rows.append((d, j.get("model_path", "?"), per))

if not sets:
    print("no finished benchmark metrics yet.")
    sys.exit(0)

ns = {n for _, _, per in rows for v in per.values() for n in [v[1]]}
w = max(28, *(len(str(mp)) for _, mp, _ in rows if mp))
print(f"{'model':<{w}} " + " ".join(f"{s:>13}" for s in sets) + "   (avg@n)")
print("-" * (w + 14 * len(sets) + 10))
for d, mp, per in rows:
    if mp is None:
        print(f"{d:<{w}} " + "  (no metrics.json — failed or not run)")
        continue
    cells = []
    for s in sets:
        cells.append(f"{per[s][0]:13.4f}" if s in per else f"{'-':>13}")
    print(f"{str(mp):<{w}} " + " ".join(cells))

if len(ns) > 1:
    print(f"\n!! sample counts differ across rows: n={sorted(ns)} — avg@n is still")
    print("   comparable, but pass@k across these rows is NOT. Re-run at one n.")
else:
    print(f"\nn = {ns.pop()} samples per question on every row.")
print("\nPaired arm-vs-arm tests need PER-QUESTION grading, not these means:")
print("  <run>/iter_0/benchmark_eval/<set>/samples.jsonl  (qid, sample_idx, correct)")
print("  scripts/collect_run_artifacts.sh packs a text-free copy of those.")
PY

echo
if [ "$FAILED" -gt 0 ]; then
  echo ">>> $FAILED model(s) FAILED — see the per-model output above." >&2
  exit 1
fi
echo ">>> sweep complete."
