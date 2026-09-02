#!/usr/bin/env bash
# Pack everything an L5 run needs for ANALYSIS into one small tarball.
#
#   bash scripts/collect_run_artifacts.sh runs/l5_* runs/bench/*_bench_eval_*
#   -> l5_artifacts_<timestamp>.tar.gz   (a few MB per run)
#
# ONE ARM PER CALL also works, and is the way to send arms as they finish from a
# shared machine holding unrelated runs:
#
#   bash scripts/collect_run_artifacts.sh runs/l5_rft_20260903_120000
#   -> l5_artifacts_l5_rft_20260903_120000_<timestamp>.tar.gz
#
# That call finds the arm's benchmark runs by itself: eval_bench.sh names its run
# dir after the model path with `/` -> `_`, so runs/<arm>/iter_K/ckpt always
# lands in runs/bench/runs_<arm>_iter_K_ckpt_bench_eval*. --no-bench opts out;
# naming bench dirs positionally still works either way.
#
# Tarballs from different calls extract into one tree without colliding (every
# path is run-scoped), so the receiving side untars them all in place — oldest
# first, since a re-collected run's newer files should win.
#
# Missing arguments are SKIPPED with a warning, never fatal. The cliff re-roll
# (scripts/cliff_reroll.py -> <run>/headline, runs/floor_holdout) is OPTIONAL:
# the same command works whether or not it was run, and MANIFEST.txt records
# which of the two it found. A plain FILE argument is packed verbatim.
#
# Why not send the run dir: a checkpoint is 7.9 GB and there are three per arm,
# but every number in the paper comes from JSON/JSONL that totals ~5 MB. The
# checkpoints stay on the machine that trained them; the headline cliff re-roll
# is run THERE too and its verdicts travel in here if it happened.
#
# Included, per run:
#   MANIFEST.txt                    what is inside + the headline numbers,
#                                   written here (tar root), read first
#   config.yaml, config.hash.json   the frozen settings (provenance)
#   metrics.jsonl                   per-iteration curve (the loop's own summary)
#   questions/*.jsonl               the exact train/holdout split
#   iter_*/*/stats.json|report.json partition / dataset / filtered / improve counts
#   iter_*/{eval,benchmark_eval}/metrics.json
#   iter_*/logs/*.log               loss + guard curves, stage timings
#   logs/{loop,bench}_<run>.log     the DRIVER log, which lives at the repo root,
#                                   not in the run dir — `run.sh` writes
#                                   logs/loop_<name>.log and `eval_bench.sh -b`
#                                   writes logs/bench_<name>.log
#   **/verdicts.jsonl               per-sample grading — attractor_mass.py and
#                                   the cliff re-roll passes both land here
#   **/summary.json                 cliff_reroll per-qid correct counts
#   benchmark_eval/*/samples_slim.jsonl   PER-QUESTION benchmark grading, derived
#                                   here from samples.jsonl by dropping
#                                   response_text (57 MB -> 1.2 MB per model).
#                                   metrics.json only has aggregates; the paired
#                                   arm-vs-arm bootstrap needs these rows.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
WITH_BENCH=true
ARGS=()
for a in "$@"; do
  case "$a" in
    --no-bench) WITH_BENCH=false ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) ARGS+=("$a") ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}
[ $# -ge 1 ] || { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

# eval_bench.sh slugs its run dir from the model path with `/` -> `_`, so an
# arm's benchmark runs are exactly runs/bench/<arm-with-underscores>_iter_*.
# Appending them to the argument list collects them like any other run dir, so
# naming one arm is enough — no need to paste timestamped bench dir names.
if $WITH_BENCH; then
  BENCH=()
  for run in "$@"; do
    run="${run%/}"
    case "$run" in runs/bench/*) continue ;; esac   # already a bench dir
    [ -d "$run" ] || continue
    for b in "runs/bench/${run//\//_}"_iter_*; do
      if [ -d "$b" ]; then BENCH+=("$b"); fi        # unmatched glob stays literal
    done
  done
  if [ ${#BENCH[@]} -gt 0 ]; then
    printf '>>> auto-collecting %d benchmark_eval run(s) for these arms:\n' "${#BENCH[@]}"
    printf '      %s\n' "${BENCH[@]}"
    set -- "$@" "${BENCH[@]}"
  else
    # Worth saying out loud: the usual cause is eval_bench.sh not having run yet.
    echo ">>> NOTE: no runs/bench/ dir matches these arms (--no-bench to silence)" >&2
  fi
fi

# Name the tarball after the arm when the call collects exactly one, so a stream
# of them arriving over weeks is self-identifying.
if [ -z "${OUT:-}" ]; then
  if [ ${#ARGS[@]} -eq 1 ] && [ -d "${ARGS[0]%/}" ]; then
    OUT="l5_artifacts_$(basename "${ARGS[0]%/}")_$(date +%Y%m%d_%H%M%S).tar.gz"
  else
    OUT="l5_artifacts_$(date +%Y%m%d_%H%M%S).tar.gz"
  fi
fi
TMPD=$(mktemp -d)
LIST="$TMPD/files.txt"
: > "$LIST"
trap 'rm -rf "$TMPD"' EXIT

ROOTS=()
SKIPPED=()
for run in "$@"; do
  run="${run%/}"
  if [ -f "$run" ]; then                       # loose file (a report, a log)
    echo "$run" >> "$LIST"; continue
  fi
  if [ ! -d "$run" ]; then
    # Not fatal: the optional re-roll outputs, or a glob that matched nothing.
    echo "[skip] not found: $run" >&2; SKIPPED+=("$run"); continue
  fi
  # A bench dir can arrive twice — named by a `runs/bench/*` glob AND found by
  # the auto-discovery above. Collect it once.
  case " ${ROOTS[*]-} " in *" $run "*) continue ;; esac
  ROOTS+=("$run")
  # benchmark samples: ship per-question grading without the generated text.
  # Written next to the source (derived, idempotent, ~1 MB) so the tar keeps one
  # root; delete samples_slim.jsonl any time, it is rebuilt from samples.jsonl.
  while IFS= read -r sam; do
    "$REPO_ROOT/.venv/bin/python" - "$sam" <<'PYSLIM'
import json, os, sys
from pathlib import Path
src = Path(sys.argv[1]); dst = src.with_name("samples_slim.jsonl")
if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
    sys.exit(0)
tmp = src.with_name("samples_slim.jsonl.tmp")
with open(src) as fin, open(tmp, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            # a background eval (-b) is still writing: keep the whole rows and
            # drop the torn tail. A later run rebuilds (src mtime moves).
            break
        r.pop("response_text", None)
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
os.replace(tmp, dst)          # never leave a half-built slim looking complete
PYSLIM
  done < <(find "$run" -path "*/benchmark_eval/*" -name "samples.jsonl")
  find "$run" \
    \( -name "config.yaml" -o -name "config.hash.json" -o -name "metrics.jsonl" \
       -o -name "stats.json" -o -name "report.json" -o -name "metrics.json" \
       -o -name "verdicts.jsonl" -o -name "summary.json" -o -name "*.log" \
       -o -name "samples_slim.jsonl" \
       -o -path "$run/questions/*.jsonl" \) \
    -not -path "*/wandb/*" -print >> "$LIST"
  # The driver log lives at the repo root, outside the run dir.
  base=$(basename "$run")
  for drv in "logs/loop_${base}.log" "logs/bench_${base}.log"; do
    [ -f "$drv" ] && echo "$drv" >> "$LIST"
  done
done

[ ${#ROOTS[@]} -gt 0 ] || { echo "nothing to collect: no argument resolved to a run dir" >&2; exit 1; }

# MANIFEST.txt: what the receiving side is holding, and the headline numbers, so
# a missing arm or an unfinished benchmark is obvious before any analysis runs.
"$REPO_ROOT/.venv/bin/python" - "$TMPD/MANIFEST.txt" "${ROOTS[@]}" <<'PYMAN'
import json, sys
from pathlib import Path

out = open(sys.argv[1], "w")
def emit(s=""):
    print(s); out.write(s + "\n")

for root in map(Path, sys.argv[2:]):
    emit(f"## {root}")
    iters = sorted(p for p in root.glob("iter_*") if p.is_dir())
    if (root / "metrics.jsonl").exists():
        emit(f"   loop run, {len(iters)} iteration(s): "
             + ", ".join(p.name for p in iters))
    for it in iters:
        bits = [s for s, p in (("ckpt", it / "ckpt"),
                               ("eval", it / "eval" / "metrics.json"),
                               ("bench", it / "benchmark_eval" / "metrics.json"))
                if p.exists()]
        if bits and (root / "metrics.jsonl").exists():
            emit(f"     {it.name}: {'+'.join(bits)}")
    for mp in sorted(root.glob("**/benchmark_eval/metrics.json")):
        m = json.loads(mp.read_text())
        benches = sorted({k.split("/")[0] for k in m if "/" in k})
        emit(f"   bench {mp.relative_to(root)}  model={m.get('model_path')}")
        if not benches:
            # e.g. {"skipped": "no benchmarks configured"} from the in-loop stage
            emit(f"     (no benchmark scores: {m.get('skipped', 'empty metrics')})")
            continue
        cells = []
        for b in benches:
            # n differs between the in-loop bench (avg@8) and the standalone
            # sweep (avg@32), so report whichever avg@ this file actually has.
            key = next((k for k in m if k.startswith(f"{b}/avg@")), None)
            cells.append(f"{key}={m[key]:.4f}" if key else f"{b}/avg@?=--")
        emit("     " + "  ".join(cells))
    for sp in sorted(root.glob("**/summary.json")):
        s = json.loads(sp.read_text())
        if "correct_counts" not in s:
            continue
        qids = s.get("qids", [])
        emit(f"   re-roll {sp.relative_to(root)}  model={s.get('model_path')}  "
             f"n={s.get('n')} passes={s.get('passes')} qids={len(qids)}")
        for name, counts in s["correct_counts"].items():
            pos = sum(1 for v in counts.values() if v > 0)
            emit(f"     {name}: {pos}/{len(counts)} re-roll to >0 correct "
                 f"({pos / max(len(counts), 1):.3f})")
    if not (root / "metrics.jsonl").exists() and not list(root.glob("**/benchmark_eval")) \
            and not list(root.glob("**/summary.json")):
        emit("   (no recognised artifacts — packed as-is)")
    emit()
out.close()
PYMAN

if [ ${#SKIPPED[@]} -gt 0 ]; then
  printf 'NOT COLLECTED (absent): %s\n' "${SKIPPED[*]}" >> "$TMPD/MANIFEST.txt"
fi

n=$(wc -l < "$LIST")
tar -czf "$OUT" -C "$TMPD" MANIFEST.txt -C "$REPO_ROOT" -T "$LIST"
echo ">>> $OUT  ($(stat -c %s "$OUT" | awk '{printf "%.1f MB", $1/1048576}'), $n files from ${#ROOTS[@]} run(s))"
if [ ${#SKIPPED[@]} -gt 0 ]; then echo ">>> skipped (absent): ${SKIPPED[*]}"; fi
echo ">>> 받는 쪽에서:  tar -xzf $OUT   → MANIFEST.txt + runs/... 구조 그대로 복원"
