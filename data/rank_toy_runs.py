"""Rank finished toy-cliff runs by conversion, with the confounders spelled out.

Headline conversion alone is misleading across arms: how many cliffs a run
converts at STAGE 1 swings by ±(6-10) questions on identical configs (vLLM
batching nondeterminism, measured 13-30 over the 2+2 arms), and a lucky stage 1
leaves a harder residual for stage 2. So this prints, per run:

  conv        overall cliff conversion (>=1 verifier-correct sample)
  conv|B+     conversion restricted to questions that GOT bridges (yield > 0) —
              removes the variation in how many cliffs produced usable fit
              targets at all
  conv|B0     conversion of the questions whose bridges ALL failed (yield = 0);
              these have no fit pairs, so nothing downstream can help them
  s2 lift     staged only: stage-2+ conversions / questions still unsolved after
              the stage-1 rollout — the schedule's own contribution, with the
              stage-1 draw divided out
  Sst / Ssmp  total gradient steps and max samples per question (the budget that
              explains most of the between-arm variance)

Usage:  .venv/bin/python data/rank_toy_runs.py [--runs-dir runs/toy_cliff]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from expert_iter.config import Config


def _bridge_yield(run_dir: Path) -> dict[str, float]:
    """qid -> fraction of its stage-1 bridge samples the verifier accepted.
    Empty for operators that generate no bridges (lora_sft, self_resample)."""
    path = run_dir / "iter_0" / "improve" / "bridge" / "bridges.jsonl"
    if not path.exists():
        return {}
    gen: dict[str, int] = defaultdict(int)
    ok: dict[str, int] = defaultdict(int)
    for line in path.open():
        row = json.loads(line)
        gen[row["qid"]] += 1
        ok[row["qid"]] += bool(row["correct"])
    return {q: ok[q] / gen[q] for q in gen}


def _converted(run_dir: Path) -> tuple[set[str], set[str]]:
    """(converted at stage 1, converted later). Non-staged operators report
    everything as stage 1. Operators that do NOT prefill `correct` (self_resample
    leaves the verdict to the filters stage) fall back to the kept candidates,
    which cleared the correctness gate."""
    first: set[str] = set()
    later: set[str] = set()
    for line in (run_dir / "iter_0" / "improve" / "improved.jsonl").open():
        cand = json.loads(line)
        if not cand.get("correct"):
            continue
        stage = (cand.get("op_meta") or {}).get("stage", "stage1")
        (first if stage == "stage1" else later).add(cand["qid"])
    if not first and not later:
        kept = run_dir / "iter_0" / "filtered" / "kept.jsonl"
        if kept.exists():
            first = {json.loads(l)["qid"] for l in kept.open()}
    return first, later - first


def _budget(cfg: Config) -> tuple[str, int, str, int, str]:
    """(step split, total steps, rollout split, max samples/question, knobs)."""
    ls = cfg.improve.lora_sft
    st = ls.staged
    if cfg.improve.operator == "staged_bridge_sft":
        steps = f"{ls.fit.steps}+{st.stage2_steps}" + (
            f"x{st.num_stages}" if st.num_stages > 1 else "")
        knobs = []
        if st.unsolved_targets != "reuse_bridge":
            knobs.append(st.unsolved_targets)
        if st.train_scope != "unsolved_only":
            knobs.append(f"{st.train_scope}/{st.solved_targets}")
        if st.stage2_chunk_size:
            knobs.append(f"chunk{st.stage2_chunk_size}")
        if st.stage2_objective != "sft":
            knobs.append(f"{st.stage2_objective}(w={st.dpo.sft_weight:g})")
        if st.final_rollout_scope != "unsolved":
            knobs.append("final=all")
        if st.emit != "all":
            knobs.append(st.emit)
        return (steps, ls.fit.steps + st.stage2_steps * st.num_stages,
                f"{st.rollout_n}+{st.final_rollout_n}",
                st.rollout_n + st.final_rollout_n, ", ".join(knobs) or "—")
    knobs = []
    if ls.chunk_size:
        knobs.append(f"chunk{ls.chunk_size}")
    if ls.adapter_scope != "pooled":
        knobs.append(ls.adapter_scope)
    if ls.refit_budget:
        knobs.append(f"refit{ls.refit_budget}")
    if cfg.improve.operator == "bridge_sft" and ls.bridge.max_keep != 4:
        knobs.append(f"max_keep{ls.bridge.max_keep}")
    if cfg.improve.operator == "bridge_sft" and ls.bridge.keep_selection != "shortest":
        knobs.append(ls.bridge.keep_selection)
    return (str(ls.fit.steps), ls.fit.steps, str(cfg.improve.n), cfg.improve.n,
            ", ".join(knobs) or "—")


def _quality(run_dir: Path, cfg: Config, cliffs: set[str], first: set[str]) -> dict:
    """Per-question correctness RATE and correct-response length over the
    candidates the operator emitted (all rollout rounds). The denominator is the
    nominal sampling budget per question (not the emitted count, which drops
    truncated samples): improve.n for single-fit operators; for staged,
    rollout_n (+ final_rollout_n if the question was still unsolved after
    stage 1). Operators that do not prefill `correct` (self_resample) report
    nothing here."""
    ls = cfg.improve.lora_sft
    st = ls.staged
    staged = cfg.improve.operator == "staged_bridge_sft"
    n_ok: dict[str, int] = defaultdict(int)
    lens_ok: list[int] = []
    any_graded = False
    for line in (run_dir / "iter_0" / "improve" / "improved.jsonl").open():
        cand = json.loads(line)
        if cand.get("correct") is None:
            continue
        any_graded = True
        if cand["correct"]:
            n_ok[cand["qid"]] += 1
            lens_ok.append(len(cand["continuation_token_ids"]))
    if not any_graded:
        return {}
    phats = []
    for q in sorted(cliffs):
        if staged:
            budget = st.rollout_n + (0 if q in first else st.final_rollout_n * st.num_stages)
        else:
            budget = cfg.improve.n
        phats.append(min(1.0, n_ok[q] / budget))
    return {
        "n_correct": sum(n_ok.values()),
        "mean_phat": sum(phats) / len(phats),
        "phat_ge_25": sum(p >= 0.25 for p in phats) / len(phats),
        "phat_ge_50": sum(p >= 0.5 for p in phats) / len(phats),
        "len_ok": sum(lens_ok) / len(lens_ok) if lens_ok else None,
    }


def collect(runs_dir: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(runs_dir.glob("*/")):
        metrics = run_dir / "metrics.json"
        cfg_path = run_dir / "config.yaml"
        if not (metrics.exists() and cfg_path.exists()):
            continue
        cfg = Config.load(cfg_path)
        m = json.loads(metrics.read_text())
        # Full run dirs carry the per-candidate artifacts; the committed
        # snapshot under docs/results/toy_cliff/ has only metrics + config, so
        # every derived column degrades to what metrics.json already records.
        unsolved_path = run_dir / "iter_0" / "partition" / "unsolved.jsonl"
        thin = not unsolved_path.exists()
        if thin:
            n_cliff = int(m.get("funnel/n_cliff") or m.get("cliff/count") or 0)
            by_stage = m.get("improve/n_resolved_by_stage") or {}
            s1 = int(m.get("improve/n_resolved_stage1") or 0)
            n_conv = int(m.get("funnel/n_kept") or 0)
            steps, tot_steps, roll, tot_samp, knobs = _budget(cfg)
            later_n = sum(v for k, v in by_stage.items() if k != "stage1")
            rows.append({
                "run": run_dir.name.replace("default_", ""), "op": cfg.improve.operator,
                "anchor": cfg.anchor.policy, "steps": steps, "tot_steps": tot_steps,
                "roll": roll, "tot_samp": tot_samp, "knobs": knobs,
                "n_cliff": n_cliff, "n_conv": n_conv,
                "conv": m.get("cliff/conversion_rate", 0.0),
                "conv_bplus": None, "conv_bzero": None, "n_bplus": 0, "n_bzero": 0,
                "s1": s1,
                "s2_lift": later_n / (n_cliff - s1) if later_n and n_cliff > s1 else None,
                "c_kept": m.get("selection/c_mean_kept"),
                "n_correct": None, "mean_phat": None, "phat25": None,
                "phat50": None, "corr_len": None,
            })
            continue
        cliffs = {json.loads(l)["qid"] for l in unsolved_path.open()}
        first, later = _converted(run_dir)
        conv = first | later
        yields = _bridge_yield(run_dir)
        with_b = {q for q, y in yields.items() if y > 0}
        no_b = {q for q, y in yields.items() if y == 0}
        steps, tot_steps, roll, tot_samp, knobs = _budget(cfg)
        unsolved_after_s1 = cliffs - first
        rows.append({
            "run": run_dir.name.replace("default_", ""),
            "op": cfg.improve.operator,
            "anchor": cfg.anchor.policy,
            "steps": steps, "tot_steps": tot_steps,
            "roll": roll, "tot_samp": tot_samp, "knobs": knobs,
            "n_cliff": len(cliffs), "n_conv": len(conv),
            "conv": m.get("cliff/conversion_rate", len(conv) / max(len(cliffs), 1)),
            "conv_bplus": len(conv & with_b) / len(with_b) if with_b else None,
            "conv_bzero": len(conv & no_b) / len(no_b) if no_b else None,
            "n_bplus": len(with_b), "n_bzero": len(no_b),
            "s1": len(first),
            "s2_lift": len(later) / len(unsolved_after_s1) if later and unsolved_after_s1 else None,
            "c_kept": m.get("selection/c_mean_kept"),
            **_quality(run_dir, cfg, cliffs, first),
        })
    return sorted(rows, key=lambda r: r["conv"], reverse=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs/toy_cliff")
    args = ap.parse_args(argv)
    rows = collect(Path(args.runs_dir))
    if not rows:
        raise SystemExit(f"no finished runs under {args.runs_dir}")

    def cell(v, n=None):
        if v is None:
            return f"{'—':>10}"
        return f"{v:.3f}({n:3d})" if n is not None else f"{v:.3f}     "

    print(f"{'conv':>6} {'#':>3} {'conv|B+ (n)':>11} {'conv|B0 (n)':>11} {'s2lift':>7} "
          f"{'s1':>3}  {'operator':17s} {'steps':7s} {'Sst':>3} {'roll':>6} {'Ssmp':>4}  "
          f"{'knobs':24s} run")
    for r in rows:
        anchor = "" if r["anchor"] == "none" else f" [anchor={r['anchor']}]"
        lift = f"{r['s2_lift']:.3f}" if r["s2_lift"] is not None else "—"
        print(f"{r['conv']:6.3f} {r['n_conv']:3d} {cell(r['conv_bplus'], r['n_bplus']):>11} "
              f"{cell(r['conv_bzero'], r['n_bzero']):>11} {lift:>7} "
              f"{r['s1']:3d}  {r['op']:17s} {r['steps']:7s} {r['tot_steps']:3d} "
              f"{r['roll']:>6} {r['tot_samp']:4d}  {r['knobs']:24s} {r['run']}{anchor}")
    print("\nconv|B+ / conv|B0 = conversion among questions that DID / did not get "
          "accepted bridges (n in parens; the split itself varies by run)")
    print("s2lift = stage-2+ conversions / questions still unsolved after the stage-1 "
          "rollout (staged only) — the schedule's contribution with the stage-1 draw "
          "divided out; s1 = converted at stage 1")
    print("Sst / Ssmp = total gradient steps / max samples per question. Arms inside "
          "one budget group differ by less than this set can resolve (~107 questions, "
          "single seed): treat <=6-question gaps as ties.")

    # ---- response quality beyond conversion --------------------------------
    print(f"\n{'conv':>6}  {'#correct':>8} {'mean p̂':>7} {'p̂>=.25':>7} {'p̂>=.5':>6} "
          f"{'corr len':>8} {'C(y) kept':>9}  {'operator':17s} {'steps':7s} {'roll':>6}  "
          f"{'knobs':24s} run")
    for r in rows:
        if "mean_phat" not in r:
            print(f"{r['conv']:6.3f}  {'— (correct not prefilled)':>49}  {r['op']:17s} "
                  f"{r['steps']:7s} {r['roll']:>6}  {r['knobs']:24s} {r['run']}")
            continue
        ck = f"{r['c_kept']:.3f}" if r["c_kept"] is not None else "—"
        print(f"{r['conv']:6.3f}  {r['n_correct']:8d} {r['mean_phat']:7.3f} "
              f"{r['phat_ge_25']:7.3f} {r['phat_ge_50']:6.3f} {r['len_ok'] or 0:8.0f} {ck:>9}  "
              f"{r['op']:17s} {r['steps']:7s} {r['roll']:>6}  {r['knobs']:24s} {r['run']}")
    print("\n#correct = verifier-correct candidates emitted over ALL rollout rounds; "
          "p̂ = per-question correct / nominal sample budget (improve.n, or staged "
          "rollout_n + final_rollout_n for questions unsolved after stage 1), averaged "
          "over all cliffs; corr len = mean tokens of correct candidates; C(y) kept = "
          "mean C over the candidates filters kept (lower = more student-like).")


if __name__ == "__main__":
    main()
