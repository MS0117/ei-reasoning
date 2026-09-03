#!/usr/bin/env bash
# InT L3 readout sweep: the B-transfer / attractor readout the InT run still
# owes, plus the S0 control arm that was never trained.
#
# Run INSIDE the a100 tmux/srun session and pass NO -g / CUDA_VISIBLE_DEVICES,
# so every step inherits SLURM's allocation. The whole attractor family (base
# floor + every arm re-roll) MUST stay on ONE node: the readout is a paired
# arm-minus-floor delta, and A100-vs-Blackwell sampling numerics would land
# inside that delta. The existing OpenR1 attractor readouts are srv04/A100, and
# so is the 4-set bench_eval family -- keep both here. bench_eval_hard_n16 is
# the srv08/Blackwell family; never run that config on this node.
#
#   nohup bash scripts/int_l3_readout_sweep.sh > logs/int_l3_readout.log 2>&1 &
#
# Steps run in sequence (one GPU pair). Each skips when its output already
# exists, so a kill resumes by re-running the same command:
#   1  base B floor re-roll  134q x32 x1   ~3h   <freeze>/iter_0/reroll/pass_0/
#   2  S1(InT) re-roll + compare           ~3h   <S1>/iter_0/attractor_B_compare.json
#   3  S3(InT) re-roll + compare           ~3h   <S3>/iter_0/attractor_B_compare.json
#   4  S0(InT) arm: train + re-roll + compare  ~6h  runs/L3_S0_int_<ts>/
#   5  S0(InT) 4-set benchmark (A100 family)   ~3.5h runs/bench/<slug>/
#
# Step 1 is a hard prerequisite: the sweep aborts if it fails. Steps 2-5 are
# independent of one another, so a failure there is recorded and the sweep goes
# on. SKIP_S0=1 stops after step 3. FREEZE/S1_RUN/S3_RUN override the defaults.
set -uo pipefail          # deliberately NOT -e: steps 2-5 must outlive each other
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=.venv/bin/python
FREEZE="${FREEZE:-runs/L2_freeze_int_20260902_052108}"
S1_RUN="${S1_RUN:-runs/L3_S1_int_20260903_143350}"
S3_RUN="${S3_RUN:-runs/L3_S3_int_20260903_031559}"
SPLIT="$FREEZE/iter_0/cliff_split.json"
FLOOR="$FREEZE/iter_0/reroll/pass_0/verdicts.jsonl"
mkdir -p logs
FAILED=()

say() { echo; echo "=== [$(date '+%m-%d %H:%M:%S')] $* ==="; }

for p in "$SPLIT" "$S1_RUN/iter_0/ckpt/model.safetensors" "$S3_RUN/iter_0/ckpt/model.safetensors"; do
  [ -e "$p" ] || { echo "!!! missing prerequisite: $p" >&2; exit 1; }
done
say "sweep start  freeze=$FREEZE  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
$PY -c "
import json; d=json.load(open('$SPLIT')); print(f'    split: A={len(d[\"A\"])} B={len(d[\"B\"])}')"

# ---- 1) base B floor -- every compare below is paired against it -------------
if [ -f "$FLOOR" ]; then
  say "1/5 base B floor already present, skipping ($(wc -l < "$FLOOR") verdicts)"
else
  say "1/5 base B floor re-roll (base policy, 134 q x32, 1 pass)"
  $PY scripts/cliff_reroll.py --run-dir "$FREEZE" --qids-file "$SPLIT:B" --passes 1 \
    || { echo "!!! base floor FAILED -- every compare depends on it, aborting" >&2; exit 1; }
fi
[ -f "$FLOOR" ] || { echo "!!! floor verdicts missing after re-roll, aborting" >&2; exit 1; }

# ---- arm re-roll + paired compare vs that floor -----------------------------
readout() {                       # readout <label> <arm run dir>
  local lab="$1" run="$2"
  local out="$run/iter_0/reroll_B/pass_0/verdicts.jsonl"
  local cmp="$run/iter_0/attractor_B_compare.json"
  if [ -f "$cmp" ]; then say "$lab readout already present, skipping"; return 0; fi
  if [ -f "$out" ]; then
    say "$lab B re-roll already present, going straight to compare"
  else
    say "$lab B re-roll under its own ckpt"
    $PY scripts/cliff_reroll.py --run-dir "$FREEZE" --model-path "$run/iter_0/ckpt" \
        --qids-file "$SPLIT:B" --passes 1 --out "$run/iter_0/reroll_B" \
      || { FAILED+=("$lab re-roll"); return 1; }
  fi
  say "$lab attractor compare vs base floor"
  $PY scripts/attractor_mass.py --verdicts "$FLOOR" --compare "$out" \
      --qids-file "$SPLIT:B" --out "$cmp" \
    || { FAILED+=("$lab compare"); return 1; }
}

readout "2/5 S1(InT)" "$S1_RUN"
readout "3/5 S3(InT)" "$S3_RUN"

# ---- 4) S0(InT): the no-cliff-data control, never trained -------------------
S0_RUN=""
if [ "${SKIP_S0:-0}" = "1" ]; then
  say "SKIP_S0=1 -- stopping after the S1/S3 readouts"
else
  S0_RUN="$(ls -d runs/L3_S0_int_*/ 2>/dev/null | sort | tail -1)"; S0_RUN="${S0_RUN%/}"
  if [ -n "$S0_RUN" ] && [ -f "$S0_RUN/iter_0/attractor_B_compare.json" ]; then
    say "4/5 S0(InT) already complete ($S0_RUN), skipping"
  elif [ -n "$S0_RUN" ] && [ -f "$S0_RUN/iter_0/ckpt/model.safetensors" ]; then
    say "4/5 S0(InT) ckpt found ($S0_RUN) -- reusing it, readout only"
    readout "4/5 S0(InT)" "$S0_RUN"
  else
    say "4/5 S0(InT) arm: train + re-roll + compare (l3_arm.sh runs all three)"
    ARM_TAG=int bash scripts/l3_arm.sh S0 "$FREEZE" "${CUDA_VISIBLE_DEVICES:-0,1}" \
      || FAILED+=("4/5 S0(InT) arm")
    S0_RUN="$(ls -d runs/L3_S0_int_*/ 2>/dev/null | sort | tail -1)"; S0_RUN="${S0_RUN%/}"
  fi

  # ---- 5) S0(InT) 4-set benchmark -- the srv04/A100 comparison family -------
  if [ -z "$S0_RUN" ] || [ ! -f "$S0_RUN/iter_0/ckpt/model.safetensors" ]; then
    say "5/5 S0(InT) benchmark skipped -- no checkpoint to grade"
    FAILED+=("5/5 S0(InT) bench (no ckpt)")
  else
    HAVE="$($PY - "$S0_RUN/iter_0/ckpt" <<'PYEOF'
import glob, json, os, sys
want = sys.argv[1]
for m in sorted(glob.glob("runs/bench/*/iter_0/benchmark_eval/metrics.json")):
    try:
        d = json.load(open(m))
    except Exception:
        continue
    if d.get("model_path") == want and any(k.endswith("/avg@32") for k in d):
        print(os.path.dirname(m)); break
PYEOF
)"
    if [ -n "$HAVE" ]; then
      say "5/5 S0(InT) benchmark already present ($HAVE), skipping"
    else
      say "5/5 S0(InT) 4-set benchmark (configs/bench_eval.yaml, foreground)"
      bash scripts/eval_bench.sh "$S0_RUN/iter_0/ckpt" -c configs/bench_eval.yaml \
        || FAILED+=("5/5 S0(InT) bench")
    fi
  fi
fi

# ---- summary ----------------------------------------------------------------
say "sweep finished"
$PY - "$FREEZE" "$S1_RUN" "$S3_RUN" "${S0_RUN:-}" <<'PYEOF'
import json, os, sys
freeze, *arms = sys.argv[1:]
floor = os.path.join(freeze, "iter_0/reroll/pass_0/verdicts.jsonl")
print(f"  base B floor: {'present' if os.path.exists(floor) else 'MISSING'}  {floor}")
for run in arms:
    if not run:
        continue
    cmp = os.path.join(run, "iter_0/attractor_B_compare.json")
    lab = os.path.basename(run)
    if not os.path.exists(cmp):
        print(f"  {lab:<28} readout MISSING")
        continue
    d = json.load(open(cmp))
    b, c = d["base"]["aggregate"], d["compare"]["aggregate"]
    print(f"  {lab:<28} pass_rate {b['mean_pass_rate']:.4f} -> {c['mean_pass_rate']:.4f}"
          f"   p_top1 {b['mean_p_top1']:.4f} -> {c['mean_p_top1']:.4f}  (n={c['n_questions']})")
    for m, st in (d["compare"].get("paired") or {}).items():
        if isinstance(st, dict) and "mean_delta" in st:
            extra = "  ".join(f"{k}={v:.3g}" for k, v in st.items()
                              if k != "mean_delta" and isinstance(v, (int, float)))
            print(f"      d{m:<22} {st['mean_delta']:+.4f}   {extra}")
PYEOF
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo; echo "!!! steps that FAILED (re-run the same command to retry them):"
  printf '    %s\n' "${FAILED[@]}"
  exit 1
fi
echo; echo ">>> all steps done"
