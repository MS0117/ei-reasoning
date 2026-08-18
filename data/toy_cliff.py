"""Toy cliff experiment: how effective are anchor / improve / filters on a
fixed all-cliff question set?

Runs ONE improvement cycle — rollout → partition → anchor → improve → filters
(no train/eval/loop) — by calling the REAL stage main() functions in-process,
then aggregates into run_dir/metrics.json + stdout + wandb:

  * a survival funnel (questions → cliff → anchored → candidates → correct
    (Y⁺) → kept → questions improved),
  * project-back alpha survival: per alpha, mean P̂(alpha) and how many
    questions clear tau_p (from improve/alpha_curves.jsonl),
  * C(y) statistics: mean C over Y⁺, mean C over the kept y† set, mean
    per-question min-C (from filtered/candidate_scores.jsonl),
  * anchor diagnostics (length/fraction, degenerate reasons, threshold rate).

Stages resume via their config-hash .done markers; --force STAGE|all mirrors
the loop driver. Ablations are config overrides on a FRESH run dir, e.g.
    --override anchor.policy=none
    --override improve.lora_sft.project_back.enabled=false

For a PAIRED method comparison, hand the second run the first run's rollout:
    --reuse-rollout runs/toy_cliff/<first run>
Both runs then see the identical cliff partition and the identical failed
trajectories (a fresh K=8 draw re-measures the cliff and lets low-pass-rate
questions escape — measured 137 -> 107), and the rollout cost is paid once.

Launcher: data/run_toy_cliff.sh. Hyperparameters (model, alpha grid,
m_rollouts, improve.n, anchor policy, ...) live in data/configs/toy_cliff.yaml.
Prerequisite: the question file must carry meta.gold_solution
(scripts/backfill_gold_solutions.py).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from collections import Counter
from pathlib import Path

from expert_iter import anchor, filters, improve, partition, rollout
from expert_iter.config import Config, freeze_run_config
from expert_iter.loop import STAGE_OUTPUT
from expert_iter.records import AnchorRecord, ImprovedCandidate
from expert_iter.utils import (
    done_marker,
    iter_dir,
    mark_done,
    read_json,
    read_jsonl,
    write_json,
)
from expert_iter.wandb_utils import log_metrics, stage_wandb

STAGES = [
    ("rollout", rollout),
    ("partition", partition),
    ("anchor", anchor),
    ("improve", improve),
    ("filters", filters),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One improvement cycle on a cliff set + metrics")
    p.add_argument("--config", default="data/configs/toy_cliff.yaml")
    p.add_argument("--override", action="append", default=[], metavar="a.b.c=value")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model-path", default=None, help="default: cfg.model.base")
    p.add_argument("--force", default=None, metavar="STAGE|all",
                   help="re-run this stage even if its .done marker matches")
    p.add_argument("--reuse-rollout", default=None, metavar="RUN_DIR",
                   help="copy questions + rollout + partition artifacts from a previous "
                        "toy run so method variants compare on the IDENTICAL cliff set "
                        "and the IDENTICAL failed trajectories (also skips the rollout cost)")
    args = p.parse_args(argv)
    if args.force is not None and args.force != "all" and args.force not in STAGE_OUTPUT:
        p.error(f"--force must be one of {list(STAGE_OUTPUT)} or 'all'")
    return args


# Config fields that DETERMINE the rollout/partition artifacts. Everything else
# (anchor policy, improvement operator, filters) runs downstream of them, so a
# variant differing only there may safely reuse another run's rollout.
ROLLOUT_INPUT_FIELDS = (
    "run.seed", "model.base", "model.dtype", "model.system_prompt",
    "model.chat_template_kwargs", "data.adapter", "data.adapter_args",
    "data.question_suffix", "data.eval_holdout", "rollout", "partition",
)


def _dotted(d: dict, path: str):
    for key in path.split("."):
        d = d[key]
    return d


def reuse_rollout(from_dir: str | Path, run_dir: Path, cfg: Config) -> None:
    """Import a previous run's questions/rollout/partition artifacts.

    Rollouts are re-sampled per run otherwise, and a cliff set defined by ONE
    K-sample measurement is noisy: questions with a small nonzero pass rate
    drop out of the cliff partition on a fresh draw (measured: 137 -> 107).
    Reusing the artifacts pins the cliff set AND the failed trajectories the
    anchor stage cuts from, so method variants are compared paired.

    The rollout-determining config fields must match exactly; anchor/improve/
    filter differences are what this is for."""
    import shutil

    src = Path(from_dir)
    src_config = src / "config.yaml"
    if not src_config.exists():
        raise SystemExit(f"[toy_cliff] --reuse-rollout: no frozen config at {src_config}")
    src_cfg = Config.load(src_config).to_dict()
    cur_cfg = cfg.to_dict()
    mismatched = [p for p in ROLLOUT_INPUT_FIELDS
                  if _dotted(src_cfg, p) != _dotted(cur_cfg, p)]
    if mismatched:
        raise SystemExit(
            f"[toy_cliff] --reuse-rollout refused: {src} differs in rollout-determining "
            f"config field(s) {mismatched} — its rollouts do not describe this run"
        )

    src_it, dst_it = src / "iter_0", iter_dir(run_dir, 0)
    copies = [
        (src / "questions" / "train.jsonl", run_dir / "questions" / "train.jsonl"),
        (src / "questions" / "holdout.jsonl", run_dir / "questions" / "holdout.jsonl"),
        (src_it / "rollout" / "rollouts.jsonl", dst_it / "rollout" / "rollouts.jsonl"),
        (src_it / "partition" / "verdicts.jsonl", dst_it / "partition" / "verdicts.jsonl"),
        (src_it / "partition" / "solved.jsonl", dst_it / "partition" / "solved.jsonl"),
        (src_it / "partition" / "unsolved.jsonl", dst_it / "partition" / "unsolved.jsonl"),
        (src_it / "partition" / "stats.json", dst_it / "partition" / "stats.json"),
    ]
    missing = [str(s) for s, _ in copies if not s.exists()]
    if missing:
        raise SystemExit(f"[toy_cliff] --reuse-rollout: source run is incomplete, missing {missing}")
    for s, d in copies:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    # questions/ markers are content-agnostic (ensure_questions only checks
    # existence); the stage-gating markers must carry THIS run's config hash.
    for s, d in [(src / "questions" / "train.jsonl", run_dir / "questions" / "train.jsonl"),
                 (src / "questions" / "holdout.jsonl", run_dir / "questions" / "holdout.jsonl")]:
        if done_marker(s).exists():
            shutil.copy2(done_marker(s), done_marker(d))
    for path in (dst_it / "rollout" / "rollouts.jsonl", dst_it / "partition" / "solved.jsonl"):
        mark_done(path, count=sum(1 for _ in read_jsonl(path)), config_hash=cfg.hash())
    n_cliff = sum(1 for _ in read_jsonl(dst_it / "partition" / "unsolved.jsonl"))
    print(f"[toy_cliff] reused rollout+partition from {src} "
          f"({n_cliff} cliff questions, rollout/partition stages will skip)")


def check_gold_solutions(cfg: Config) -> None:
    """Fail fast if y* is missing — both privileged_divergence and lora_sft
    need it (hf_math is covered by config validation; local files checked here)."""
    if cfg.data.adapter != "local_jsonl":
        return
    path = Path(cfg.data.adapter_args["path"])
    if not path.exists():
        raise SystemExit(
            f"[toy_cliff] question file not found: {path}\n"
            "  run: .venv/bin/python scripts/backfill_gold_solutions.py "
            "--input data/cliff_sets/openr1_qwen3-4b-2507_n2000.jsonl"
        )
    n = n_gold = 0
    for row in read_jsonl(path):
        n += 1
        if (row.get("meta") or {}).get("gold_solution"):
            n_gold += 1
    if n_gold == 0:
        raise SystemExit(
            f"[toy_cliff] {path} carries no meta.gold_solution — backfill first:\n"
            "  .venv/bin/python scripts/backfill_gold_solutions.py "
            "--input data/cliff_sets/openr1_qwen3-4b-2507_n2000.jsonl"
        )
    print(f"[toy_cliff] {n_gold}/{n} questions carry gold solutions")


def run_stages(args: argparse.Namespace, run_dir: Path, model_path: str) -> None:
    """Sequentially invoke the real stage entry points (their .done markers
    give stage-level resume; the frozen config carries the overrides)."""
    it_dir = iter_dir(run_dir, 0)
    common = [
        "--config", str(run_dir / "config.yaml"),
        "--run-dir", str(run_dir),
        "--iter", "0",
        "--model-path", model_path,
    ]
    for name, module in STAGES:
        if args.force in (name, "all"):
            marker = done_marker(it_dir / STAGE_OUTPUT[name])
            if marker.exists():
                marker.unlink()
        t0 = time.time()
        print(f"[toy_cliff] ===== stage {name} =====", flush=True)
        module.main(list(common))
        print(f"[toy_cliff] stage {name} finished in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# metrics aggregation — pure functions over the iter_0/ artifacts
# ---------------------------------------------------------------------------

def anchor_stats(anchors: list[AnchorRecord]) -> dict:
    nonempty = [a for a in anchors if a.anchor_len > 0]
    out: dict = {
        "anchor/n_total": len(anchors),
        "anchor/n_nonempty": len(nonempty),
        "anchor/empty_reasons": dict(Counter(
            a.meta.get("reason", "policy_zero") for a in anchors if a.anchor_len == 0
        )),
    }
    if nonempty:
        out["anchor/mean_len"] = round(
            sum(a.anchor_len for a in nonempty) / len(nonempty), 1)
        out["anchor/mean_frac"] = round(
            sum(a.anchor_len / a.base_response_len for a in nonempty if a.base_response_len)
            / len(nonempty), 4)
        out["anchor/lens"] = [a.anchor_len for a in nonempty]  # -> wandb histogram
        crossed = [a.meta["threshold_crossed"] for a in nonempty
                   if "threshold_crossed" in a.meta]
        if crossed:
            out["anchor/threshold_crossed_rate"] = round(
                sum(bool(x) for x in crossed) / len(crossed), 4)
        j_anchor = [a.meta["j_anchor"] for a in nonempty if "j_anchor" in a.meta]
        if j_anchor:
            out["anchor/mean_j_anchor"] = round(sum(j_anchor) / len(j_anchor), 2)
    return out


def alpha_survival(curves: list[dict], tau_p: float) -> dict:
    """Per-alpha survival from the improve stage's P̂ curves: mean P̂(alpha) and
    the number of questions with P̂(alpha) >= tau_p ('alpha 필터링 생존 수')."""
    if not curves:
        return {}
    out: dict = {
        "alpha/n_questions_swept": len(curves),
        "alpha/tau_p": tau_p,
        "alpha/n_refit": sum(bool(r.get("refit")) for r in curves),
    }
    keys = sorted({k for row in curves for k in row["p_hat_by_alpha"]}, key=float)
    for k in keys:
        vals = [row["p_hat_by_alpha"][k] for row in curves if k in row["p_hat_by_alpha"]]
        out[f"alpha/p_hat_mean/{k}"] = round(sum(vals) / len(vals), 4)
        out[f"alpha/n_survive/{k}"] = sum(v >= tau_p for v in vals)
    stars = [row["alpha_star"] for row in curves]
    out["alpha/star_mean"] = round(sum(stars) / len(stars), 4)
    out["alpha/star_values"] = stars  # -> wandb histogram
    return out


def selection_stats(scores: list[dict]) -> dict:
    """C(y) statistics over filtered/candidate_scores.jsonl. All scored rows
    already passed the correctness gate, so 'all scored' == Y⁺ survivors and
    the kept flags identify the selected y† set (top-k by min C)."""
    scored = [s for s in scores if s.get("c") is not None]
    out: dict = {"selection/n_scored": len(scores)}
    if not scored:
        return out
    kept = [s for s in scored if s.get("kept")]
    c_all = sorted(s["c"] for s in scored)
    by_qid: dict[str, list[float]] = {}
    for s in scored:
        by_qid.setdefault(s["qid"], []).append(s["c"])
    out.update({
        "selection/c_mean_all": round(sum(c_all) / len(c_all), 4),        # mean C over Y⁺
        "selection/c_median_all": _pct(c_all, 0.5),
        "selection/c_values": c_all,                                      # -> histogram
        "selection/c_min_per_question_mean": round(
            sum(min(v) for v in by_qid.values()) / len(by_qid), 4),       # mean_q C(y†_q)
    })
    if kept:
        out["selection/c_mean_kept"] = round(
            sum(s["c"] for s in kept) / len(kept), 4)                     # mean over kept y†
        out["selection/n_kept_scored"] = len(kept)
    for metric in ("s_mean", "s_tail", "d_tail"):
        vals = sorted(s[metric] for s in scored if s.get(metric) is not None)
        if vals:
            out[f"selection/{metric}_p50"] = _pct(vals, 0.5)
            out[f"selection/{metric}_p95"] = _pct(vals, 0.95)
    return out


def _pct(sorted_vals: list[float], q: float) -> float:
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))
    return round(sorted_vals[idx], 4)


def collect_metrics(run_dir: Path, cfg: Config) -> dict:
    it_dir = iter_dir(run_dir, 0)
    n_questions = sum(1 for _ in read_jsonl(run_dir / "questions" / "train.jsonl"))
    part = read_json(it_dir / "partition" / "stats.json")
    anchors = list(AnchorRecord.load_jsonl(it_dir / "anchors" / "anchors.jsonl"))
    improved = list(ImprovedCandidate.load_jsonl(it_dir / "improve" / "improved.jsonl"))
    report = read_json(it_dir / "filtered" / "report.json")

    # Operators that verify inline (lora_sft/bridge_sft) prefill `correct`;
    # self_resample leaves it None until the filters stage grades it, so fall
    # back to the conversion histogram (per-question correct counts) — this
    # keeps the funnel comparable across arms, including the no-LoRA control.
    n_correct = sum(bool(c.correct) for c in improved)
    if n_correct == 0 and improved and all(c.correct is None for c in improved):
        n_correct = sum(report.get("cliff/conversion_histogram", []))

    metrics: dict = {
        "funnel/n_questions": n_questions,
        "funnel/n_cliff": part["n_unsolved_questions"],
        "funnel/n_candidates": len(improved),
        "funnel/n_correct_candidates": n_correct,
        "funnel/n_kept": report["n_kept"],
        "funnel/n_questions_improved": report["n_questions_improved"],
        "partition/solve_rate": part.get("solve_rate"),
        "partition/sample_accuracy": part.get("sample_accuracy"),
        "filter/improve_yield": report["improve_yield"],
        "filter/rejects": report.get("rejects", {}),
        "cliff/conversion_rate": report.get("cliff/conversion_rate"),
        "cliff/conversion_histogram": report.get("cliff/conversion_histogram", []),
    }
    a_stats = anchor_stats(anchors)
    metrics.update(a_stats)
    metrics["funnel/n_anchored"] = a_stats["anchor/n_nonempty"]

    improve_stats_path = it_dir / "improve" / "stats.json"
    if improve_stats_path.exists():  # LoRA operators write it; others may not
        for k, v in read_json(improve_stats_path).items():
            if k != "operator":
                metrics[f"improve/{k}"] = v
        metrics["funnel/n_with_gold"] = metrics.get("improve/n_with_gold")
        if "improve/n_questions_bridged" in metrics:      # bridge_sft
            metrics["funnel/n_bridged"] = metrics["improve/n_questions_bridged"]

    # post-SFT RL reward curves (improve.rl): one curve per refined chunk
    curves = sorted((it_dir / "improve" / "adapters").glob("*_rl/*/reward_curve.jsonl"))
    if curves:
        rows = list(read_jsonl(curves[0]))
        rewards = [r["reward"] for r in rows]
        if rewards:
            # The per-step curve itself is a live wandb chart from the RL
            # subprocess (lora_rl.report_to); these are the end-of-run summary
            # numbers, so the toy metrics stay self-contained.
            metrics["rl/reward_curve"] = rewards          # -> wandb histogram
            metrics["rl/reward_first"] = rewards[0]
            metrics["rl/reward_last"] = rewards[-1]
            metrics["rl/reward_mean"] = sum(rewards) / len(rewards)
            metrics["rl/n_logged_steps"] = len(rewards)
            # Fraction of groups with zero advantage (all-right or all-wrong):
            # the gradient is exactly 0 there, so a flat reward curve with a
            # high value here means "no signal", not "method failed".
            zs = [r["frac_reward_zero_std"] for r in rows if "frac_reward_zero_std" in r]
            if zs:
                metrics["rl/frac_reward_zero_std_mean"] = sum(zs) / len(zs)
            cl = [r["completions/clipped_ratio"] for r in rows
                  if "completions/clipped_ratio" in r]
            if cl:
                metrics["rl/clipped_ratio_mean"] = sum(cl) / len(cl)

    curves_path = it_dir / "improve" / "alpha_curves.jsonl"
    if curves_path.exists():
        metrics.update(alpha_survival(
            list(read_jsonl(curves_path)), cfg.improve.lora_sft.project_back.tau_p))

    scores_path = it_dir / "filtered" / "candidate_scores.jsonl"
    if scores_path.exists():
        metrics.update(selection_stats(list(read_jsonl(scores_path))))
    return metrics


def print_funnel(m: dict) -> None:
    rows = [
        ("questions", "funnel/n_questions", None),
        ("cliff (routed to improve)", "funnel/n_cliff", "funnel/n_questions"),
        ("with gold solution", "funnel/n_with_gold", "funnel/n_cliff"),
        ("anchored (non-empty)", "funnel/n_anchored", "funnel/n_cliff"),
        ("bridged (B+ nonempty)", "funnel/n_bridged", "funnel/n_with_gold"),
        ("improve candidates", "funnel/n_candidates", None),
        ("correct candidates (Y+)", "funnel/n_correct_candidates", "funnel/n_candidates"),
        ("kept after filters", "funnel/n_kept", "funnel/n_candidates"),
        ("questions improved (kept)", "funnel/n_questions_improved", "funnel/n_cliff"),
    ]
    print("\n  stage                       |  count | share")
    print("  " + "-" * 50)
    for label, key, denom_key in rows:
        val = m.get(key)
        if val is None:
            continue
        denom = m.get(denom_key) if denom_key else None
        share = f"{100 * val / denom:5.1f}%" if denom else "      "
        print(f"  {label:<27} | {val:>6} | {share}")
    if m.get("cliff/conversion_rate") is not None:
        print(f"\n  cliff conversion (raw rescue power): {100 * m['cliff/conversion_rate']:.1f}%")

    alpha_keys = sorted(
        (k.rsplit("/", 1)[1] for k in m if k.startswith("alpha/p_hat_mean/")), key=float)
    if alpha_keys:
        print(f"\n  alpha | mean P_hat | n_survive (P>=tau={m['alpha/tau_p']})")
        print("  " + "-" * 44)
        for a in alpha_keys:
            print(f"  {a:>5} | {m[f'alpha/p_hat_mean/{a}']:>10.4f} | "
                  f"{m[f'alpha/n_survive/{a}']:>6}")
        print(f"  alpha* mean {m['alpha/star_mean']} over "
              f"{m['alpha/n_questions_swept']} questions")

    if "selection/c_mean_all" in m:
        parts = [f"mean C over Y+ = {m['selection/c_mean_all']}"]
        if "selection/c_mean_kept" in m:
            parts.append(f"mean C over kept y† = {m['selection/c_mean_kept']}")
        parts.append(f"mean_q min-C (y†_q) = {m['selection/c_min_per_question_mean']}")
        print("\n  C(y): " + " | ".join(parts))

    if "rl/reward_last" in m:
        print(f"\n  RL: reward {m.get('rl/reward_first')} -> {m['rl/reward_last']} "
              f"over {len(m.get('rl/reward_curve', []))} logged steps")
        if "rl/frac_reward_zero_std_mean" in m:
            print(f"      zero-advantage groups (no gradient): "
                  f"{m['rl/frac_reward_zero_std_mean']:.1%} of steps"
                  + (f" | truncated completions {m['rl/clipped_ratio_mean']:.1%}"
                     if "rl/clipped_ratio_mean" in m else ""))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = Config.load(args.config, overrides=args.override)
    run_dir = Path(args.run_dir)
    freeze_run_config(cfg, run_dir)
    model_path = args.model_path or cfg.model.base

    check_gold_solutions(cfg)
    if args.reuse_rollout:
        reuse_rollout(args.reuse_rollout, run_dir, cfg)
    run_stages(args, run_dir, model_path)

    metrics = collect_metrics(run_dir, cfg)
    write_json(run_dir / "metrics.json", metrics)
    print_funnel(metrics)

    wb = stage_wandb(
        cfg, name=run_dir.name, id_file=run_dir / "wandb_id.txt", job_type="toy_cliff",
        config={
            "model_path": model_path,
            "anchor": dataclasses.asdict(cfg.anchor),
            "improve": dataclasses.asdict(cfg.improve),
            "filter": dataclasses.asdict(cfg.filter),
            "rollout": dataclasses.asdict(cfg.rollout),
        },
    )
    # scalars -> charts, numeric lists (anchor/lens, alpha/star_values,
    # selection/c_values, cliff/conversion_histogram) -> wandb.Histogram,
    # dicts (rejects, empty_reasons) -> run summary.
    log_metrics(wb, metrics)
    if wb is not None:
        wb.finish()

    print(f"\n[toy_cliff] wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
