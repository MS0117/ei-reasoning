#!/usr/bin/env bash
# One-shot InT L2->L3 pipeline (2026-09-02 time-crunch design; run inside the
# a100b tmux/srun session so the cgroup exposes the right GPUs):
#
#   L2 freeze on data/mixes/int_l2_2k_qwen3-4b-2507.jsonl with eval_holdout=0
#     (all 2000 q in train -> ~400 cliffs -> A/B ~ 200/200, no eval stage)
#   -> cliff_split WITHOUT the reroll (conversion strata only; base_pass empty)
#   -> base floor reroll on B ONLY, x32, 1 pass  (vs the old 353x32x2)
#   -> gate readout (attractor anatomy on B + improve conversion), advisory
#   -> arms S3, S0, S1 via l3_arm.sh (ARM_TAG=int; keepEOS defaults)
#
#   bash scripts/int_l2l3_auto.sh
#
# Resume after a crash: RUN_NAME=L2_freeze_int_<ts> bash scripts/int_l2l3_auto.sh
# (freeze stages skip via .done markers; finished arms are skipped by their
# attractor_B_compare.json; a HALF-finished arm is NOT resumed — delete its
# runs/L3_<ARM>_int_* dir and rerun.)
# GATE_STOP=1 stops after the gate readout instead of launching arms.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0,1}"
DATASET="data/mixes/int_l2_2k_qwen3-4b-2507.jsonl"
RUN_NAME="${RUN_NAME:-L2_freeze_int_$(date +%Y%m%d_%H%M%S)}"
FREEZE="runs/$RUN_NAME"
echo ">>> [auto] freeze=$FREEZE dataset=$DATASET GPUs=$GPU"

# ---- 1) L2 freeze (mirrors scripts/l2_freeze.sh + data.eval_holdout=0; inlined
#         so RUN_NAME is ours and a rerun resumes via the .done markers) --------
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -m expert_iter.loop \
  --config configs/methods/staged_bridge_sft.yaml \
  --override "run.name=$RUN_NAME" \
  --override "loop.stages=[rollout,partition,anchor,improve,filters,build_dataset]" \
  --override "loop.iterations=1" \
  --override "data.adapter=local_jsonl" \
  --override "data.adapter_args.path=$DATASET" \
  --override "data.eval_holdout=0" \
  --override "improve.lora_sft.staged.stage2_objective=dpo" \
  --override "improve.lora_sft.staged.dpo.sft_weight=0.0" \
  --override "filter.selection.always_score=true"

# ---- 2) A/B split, no reroll input (base_pass stratum empty by design) -------
.venv/bin/python scripts/cliff_split.py --run-dir "$FREEZE"
SPLIT="$FREEZE/iter_0/cliff_split.json"

# ---- 3) base floor: B only, x32, single pass ---------------------------------
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/cliff_reroll.py \
  --run-dir "$FREEZE" --qids-file "$SPLIT:B" --passes 1

# ---- 4) gates (advisory; printed for the human, logged to gates.json) --------
.venv/bin/python scripts/attractor_mass.py \
  --verdicts "$FREEZE/iter_0/reroll/pass_0/verdicts.jsonl" \
  --out "$FREEZE/iter_0/attractor_base.json"

A_CONV="$(.venv/bin/python - "$FREEZE" <<'EOF'
import json, sys
from pathlib import Path
from statistics import mean
it = Path(sys.argv[1]) / "iter_0"
unsolved = {json.loads(l)["qid"] for l in open(it / "partition" / "unsolved.jsonl")}
cands    = {json.loads(l)["qid"] for l in open(it / "improve" / "improved.jsonl")}
kept     = {json.loads(l)["qid"] for l in open(it / "filtered" / "kept.jsonl")}
split = json.load(open(it / "cliff_split.json"))
A, B = set(split["A"]), set(split["B"])
att = json.load(open(it / "attractor_base.json"))["base"]
agg, per = att["aggregate"], att["per_qid"]
gates = {
    # gate 2 — conversion (OpenR1 ref: 164/353 = 0.465)
    "n_cliffs": len(unsolved), "n_with_candidates": len(cands),
    "n_converted": len(kept), "conversion": round(len(kept) / max(1, len(unsolved)), 3),
    "n_converted_A": len(kept & A), "n_A": len(A), "n_B": len(B),
    # gate 1 — attractor anatomy on the B floor (OpenR1 B refs: mean_p_top1 0.71,
    # avg@32 0.020, base_pass share 0.29)
    "B_floor": agg,
    "B_mean_n_wrong_kinds": round(mean(r["n_wrong_kinds"] for r in per.values()), 2) if per else None,
}
json.dump(gates, open(it / "gates.json", "w"), indent=2)
print("[gates] " + json.dumps(gates), file=sys.stderr)
print(gates["n_converted_A"])
EOF
)"
echo ">>> [auto] gates written: $FREEZE/iter_0/gates.json (A-converted=$A_CONV)"

# hard floor only: with <20 converted A-cliffs the cliff term starves and the
# S3/S1 arms are meaningless — stop and leave the judgment call to the human.
if [ "$A_CONV" -lt 20 ]; then
  echo ">>> [auto] ABORT: only $A_CONV converted A-cliffs (<20). Not launching arms." >&2
  exit 1
fi
if [ "${GATE_STOP:-0}" = "1" ]; then
  echo ">>> [auto] GATE_STOP=1 — stopping after gates; launch arms with:"
  echo "    ARM_TAG=int bash scripts/l3_arm.sh S3 $FREEZE $GPU"
  exit 0
fi

# ---- 5) arms: S3 first (decisive), then the S0/S1 controls -------------------
for ARM in S3 S0 S1; do
  done_marker="$(ls -d runs/L3_${ARM}_int_*/iter_0/attractor_B_compare.json 2>/dev/null | head -1)"
  if [ -n "$done_marker" ]; then
    echo ">>> [auto] arm $ARM already complete ($done_marker) — skipping"
    continue
  fi
  ARM_TAG=int bash scripts/l3_arm.sh "$ARM" "$FREEZE" "$GPU"
done

echo ">>> [auto] all done. Readouts:"
echo "    gates:   $FREEZE/iter_0/gates.json"
for ARM in S3 S0 S1; do
  ls -d runs/L3_${ARM}_int_*/iter_0/attractor_B_compare.json 2>/dev/null | head -1
done
