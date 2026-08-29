#!/usr/bin/env bash
# L3 decision-experiment arm: fork the frozen L2 run and train+eval one arm.
# (docs/objective_decision_20260823.md §3 arm table + §8 amendments)
#
#   [ARM_TAG=<variant>] bash scripts/l3_arm.sh <ARM> <FROZEN_RUN_DIR> [GPU_IDS=0,1] [--override k=v ...]
#   ARM ∈ S0 | S1 | S1p | S3 | S3tok | S4v0 | S4v1   (S4v1 mu via S4_MU env)
#
# All arms exclude the B cliffs from training (data.exclude_train_qids ->
# <frozen>/iter_0/cliff_split.json); S0 additionally excludes the A cliffs
# (no cliff data at all). S1p needs the measured legacy share: run
# scripts/rho_legacy.py first and export S1P_RHO=<value> (default 0.06).
# Canary benchmarks are OFF for L3 arms (eval.benchmarks=[]) — rerun finalists
# with benchmarks via scripts/eval_bench.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ARM="${1:?usage: l3_arm.sh <S0|S1|S1p|S3|S3tok|S4v0> <FROZEN_RUN_DIR> [GPUS]}"
SRC="${2:?frozen L2 run dir required}"
GPU="${3:-0,1}"
shift $(( $# > 3 ? 3 : $# ))    # anything after the GPU arg = extra --override pairs
EXTRA_OV=("$@")
SPLIT="$SRC/iter_0/cliff_split.json"
[ -f "$SPLIT" ] || { echo "missing $SPLIT — run scripts/cliff_split.py first"; exit 1; }
S1P_RHO="${S1P_RHO:-0.06}"

TS="$(date +%Y%m%d_%H%M%S)"
# ARM_TAG labels a variant of the same arm (e.g. ARM_TAG=dropEOS). It must move
# the run DIR and run.name TOGETHER: loop.py derives its run_dir from
# cfg.run.name, so overriding run.name alone sends the loop to an empty dir and
# it re-runs rollout from scratch. Never pass --override run.name=... here.
NAME="L3_${ARM}${ARM_TAG:+_$ARM_TAG}_${TS}"
DST="runs/$NAME"
COMMON=(
  --override "run.name=$NAME"
  # the frozen L2 config stops at build_dataset with 4 iterations — arms train once:
  --override "loop.stages=[rollout,partition,anchor,improve,filters,build_dataset,train,eval]"
  --override "loop.iterations=1"
  --override "eval.benchmarks=[]"
  --override "data.exclude_train_qids=$SPLIT"
  --override "filter.selection.always_score=true"
)
CLIFF_ON=(--override "train.sft.cliff.enabled=true")

case "$ARM" in
  S0)  # no cliff data at all: exclude A ∪ B
    ALLQ="$SRC/iter_0/cliff_exclude_all.json"
    .venv/bin/python -c "
import json,sys; d=json.load(open('$SPLIT'))
json.dump({'exclude': sorted(set(d['A'])|set(d['B']))}, open('$ALLQ','w'))"
    ARM_OV=(--override "data.exclude_train_qids=$ALLQ") ;;
  S1)  # today's loss, B excluded
    ARM_OV=() ;;
  S1p) # legacy share, stratified supply (rho from scripts/rho_legacy.py)
    ARM_OV=("${CLIFF_ON[@]}" --override "train.sft.cliff.rho=$S1P_RHO") ;;
  S3)  # the dose arm
    ARM_OV=("${CLIFF_ON[@]}" --override "train.sft.cliff.rho=0.3") ;;
  S3tok) # per-question normalization ablation
    ARM_OV=("${CLIFF_ON[@]}" --override "train.sft.cliff.rho=0.3"
            --override "train.sft.cliff.per_question_norm=false") ;;
  S4v1) # + attractor negative, IN-SFT bounded unlikelihood (mu*L_N inside the rho bracket)
    ARM_OV=("${CLIFF_ON[@]}" --override "train.sft.cliff.rho=0.3"
            --override "train.sft.cliff.negative.mode=v1"
            --override "train.sft.cliff.negative.mu=${S4_MU:-0.1}") ;;
  S4v0) # + attractor negative, zero-code path (SFT then DPO w/ modal-wrong rejected)
    ARM_OV=("${CLIFF_ON[@]}" --override "train.sft.cliff.rho=0.3"
            --override "train.sft.cliff.negative.mode=v0"
            --override "train.objective=sft+dpo"
            --override "train.dpo.rejected_selection=modal_wrong") ;;
  *) echo "unknown arm: $ARM"; exit 1 ;;
esac

.venv/bin/python scripts/fork_run.py --src "$SRC" --dst "$DST" "${COMMON[@]}" "${ARM_OV[@]}" ${EXTRA_OV[@]+"${EXTRA_OV[@]}"}

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -m expert_iter.loop \
  --config "$DST/config.yaml"

# B transfer readout: re-roll B under the arm's checkpoint, compare vs base floor
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/cliff_reroll.py \
  --run-dir "$SRC" --model-path "$DST/iter_0/ckpt" \
  --qids-file "$SPLIT:B" --passes 1 --out "$DST/iter_0/reroll_B"
.venv/bin/python scripts/attractor_mass.py \
  --verdicts "$SRC/iter_0/reroll/pass_0/verdicts.jsonl" "$SRC/iter_0/reroll/pass_1/verdicts.jsonl" \
  --compare "$DST/iter_0/reroll_B/pass_0/verdicts.jsonl" \
  --qids-file "$SPLIT:B" --out "$DST/iter_0/attractor_B_compare.json"

echo
echo ">>> arm $ARM${ARM_TAG:+ ($ARM_TAG)} done: $DST"
echo ">>> holdout: $DST/metrics.jsonl | dataset: $DST/iter_0/dataset/stats.json"
echo ">>> B transfer: $DST/iter_0/attractor_B_compare.json (printed above)"
