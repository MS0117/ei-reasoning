#!/usr/bin/env bash
# L2: freeze one improve pass for the L3 decision experiment.
# (docs/objective_decision_20260823.md §4; run inside the a100 tmux session)
#
#   bash scripts/l2_freeze.sh [-c CONFIG] [-g GPU_IDS] [-d DATASET] [-- --override a.b=c ...]
#
#   -c  base config yaml  (default configs/methods/staged_bridge_sft.yaml)
#   -g  GPU ids           (default 0,1)
#   -d  question pool     (or the DATASET env var)
#
# The positional GPU form still works: bash scripts/l2_freeze.sh 0,1
#
# Runs rollout(2000x8) -> partition -> anchor -> improve(STAGED 2+2, stage-2
# PURE DPO = the operator fixed from run default_STAGED_20260821_193219) ->
# filters(+C(y) scores for the guard) -> build_dataset, NO training. Then the
# base re-roll floor (cliff_reroll) and the A/B split + rho_legacy are run on
# the frozen dir. Every L3 arm forks this dir via scripts/fork_run.py.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
GPU="0,1"
CONFIG="configs/methods/staged_bridge_sft.yaml"

# Question pool: the cliff-enriched 400/2000 set (data/make_enriched_set.py —
# 400 clean 0/16 cliffs + 150 frontier for the S2' control + 1450 solved).
# B ~ 200 after the A/B split -> attractor mass resolves ~3pp at power 0.9.
DATASET="${DATASET:-data/cliff_sets/openr1_hybrid_c400_n2000.jsonl}"

# Backwards-compatible positional GPU form: `l2_freeze.sh 0,1`.
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then GPU="$1"; shift; fi

while getopts ":c:g:d:h" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    g) GPU="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "unknown option -$OPTARG (try -h)" >&2; exit 2 ;;
    :) echo "option -$OPTARG needs a value" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
RUN_NAME="L2_freeze_$(date +%Y%m%d_%H%M%S)"
echo ">>> config=$CONFIG  dataset=$DATASET  GPUs=$GPU  run.name=$RUN_NAME"

CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -m expert_iter.loop \
  --config "$CONFIG" \
  --override "run.name=$RUN_NAME" \
  --override "loop.stages=[rollout,partition,anchor,improve,filters,build_dataset]" \
  --override "loop.iterations=1" \
  --override "data.adapter=local_jsonl" \
  --override "data.adapter_args.path=$DATASET" \
  --override "improve.lora_sft.staged.stage2_objective=dpo" \
  --override "improve.lora_sft.staged.dpo.sft_weight=0.0" \
  --override "filter.selection.always_score=true" "$@"

echo
echo ">>> L2 frozen: runs/$RUN_NAME"
echo ">>> next (same GPUs, ~overnight together):"
echo "    .venv/bin/python scripts/cliff_reroll.py --run-dir runs/$RUN_NAME"
echo ">>> then (CPU):"
echo "    .venv/bin/python scripts/cliff_split.py --run-dir runs/$RUN_NAME --reroll-summary runs/$RUN_NAME/iter_0/reroll/summary.json"
echo "    .venv/bin/python scripts/rho_legacy.py --run-dir runs/$RUN_NAME"
echo "    .venv/bin/python scripts/attractor_mass.py --verdicts runs/$RUN_NAME/iter_0/reroll/pass_0/verdicts.jsonl --out runs/$RUN_NAME/iter_0/attractor_base.json"
echo "    .venv/bin/python scripts/power_table.py --attractor-json runs/$RUN_NAME/iter_0/attractor_base.json"
echo ">>> arms: bash scripts/l3_arm.sh <S0|S1|S1p|S3|S3tok|S4v0> runs/$RUN_NAME"
