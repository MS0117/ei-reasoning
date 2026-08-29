#!/usr/bin/env bash
# Pack everything an L5 run needs for ANALYSIS into one small tarball.
#
#   bash scripts/collect_run_artifacts.sh runs/l5_rft_20260901_120000 [MORE RUNS...]
#   -> l5_artifacts_<timestamp>.tar.gz   (a few MB per run)
#
# Why not send the run dir: a checkpoint is 7.9 GB and there are three per arm,
# but every number in the paper comes from JSON/JSONL that totals ~5 MB. The
# checkpoints stay on the machine that trained them; the headline cliff re-roll
# (scripts/cliff_reroll.py) is run THERE too and its verdicts travel in here.
#
# Included, per run:
#   config.yaml, config.hash.json   the frozen settings (provenance)
#   metrics.jsonl                   per-iteration curve (the loop's own summary)
#   questions/*.jsonl               the exact train/holdout split
#   iter_*/*/stats.json|report.json partition / dataset / filtered / improve counts
#   iter_*/{eval,benchmark_eval}/metrics.json
#   iter_*/logs/*.log               loss + guard curves, stage timings
#   **/verdicts.jsonl               per-sample grading — attractor_mass.py reads these
#   headline/, floor_holdout/       cliff_reroll output, if present
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ $# -ge 1 ] || { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

OUT="l5_artifacts_$(date +%Y%m%d_%H%M%S).tar.gz"
LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT

for run in "$@"; do
  run="${run%/}"
  [ -d "$run" ] || { echo "not a run dir: $run" >&2; exit 1; }
  find "$run" \
    \( -name "config.yaml" -o -name "config.hash.json" -o -name "metrics.jsonl" \
       -o -name "stats.json" -o -name "report.json" -o -name "metrics.json" \
       -o -name "verdicts.jsonl" -o -name "summary.json" -o -name "*.log" \
       -o -path "$run/questions/*.jsonl" \) \
    -not -path "*/wandb/*" -print >> "$LIST"
done

n=$(wc -l < "$LIST")
tar -czf "$OUT" -T "$LIST"
echo ">>> $OUT  ($(stat -c %s "$OUT" | awk '{printf "%.1f MB", $1/1048576}'), $n files from $# run(s))"
echo ">>> 받는 쪽에서:  tar -xzf $OUT   → runs/... 구조 그대로 복원"
