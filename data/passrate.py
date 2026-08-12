"""Pass-rate distribution toy experiment.

Sample N questions from an HF math dataset (hf_math adapter), generate K
rollouts each with the policy model, grade them, and report the per-question
correct-count distribution: how many questions land at c = 0/K, 1/K, ..., K/K.

Each question x is classified by its correct count c(x):
    c == 0                     -> cliff     (strict cliff — the hard problems)
    frontier_min <= c < solved -> frontier  (default 1 <= c <= 2)
    c >= solved_min            -> solved    (default c >= 3)

Outputs under --run-dir:
  questions.jsonl       frozen sampled questions (resampled only if the
                        adapter args change)
  samples.jsonl         graded generations, same schema as benchmark_eval
  question_stats.jsonl  {qid, n, c, pass_rate, class, n_truncated, final_answer}
  metrics.json          histogram + class counts + pass@k/avg@n and rates

Launcher: data/run_passrate.sh. Hyperparameters live in the config YAML
(data/configs/passrate.yaml); class thresholds are CLI flags here.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter
from pathlib import Path

from expert_iter.benchmark_eval import summarize_benchmark
from expert_iter.config import Config, freeze_run_config
from expert_iter.data import load_questions, resolve_adapter_args
from expert_iter.engine import GenRequest, run_pool
from expert_iter.lora import resolve_model_path
from expert_iter.records import QuestionRecord
from expert_iter.registry import VERIFIERS, build
from expert_iter.templates import render_question_prompt
from expert_iter.utils import (
    is_done,
    mark_done,
    read_jsonl,
    stable_hash,
    stable_seed,
    write_json,
    write_jsonl,
)
from expert_iter.wandb_utils import log_metrics, stage_wandb
from expert_iter import verifier as _verifier_registration  # noqa: F401 — @register side effect


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pass-rate distribution over K rollouts")
    p.add_argument("--config", default="data/configs/passrate.yaml")
    p.add_argument("--override", action="append", default=[], metavar="a.b.c=value")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model-path", default=None, help="default: cfg.model.base")
    p.add_argument("--frontier-min", type=int, default=1,
                   help="min correct count for the frontier class (below: cliff)")
    p.add_argument("--solved-min", type=int, default=3,
                   help="min correct count for the solved class")
    p.add_argument("--dry-run", action="store_true",
                   help="load+sample questions and render one prompt, no GPU work")
    args = p.parse_args(argv)
    if not 0 < args.frontier_min <= args.solved_min:
        p.error("need 0 < --frontier-min <= --solved-min (c=0 is always cliff)")
    return args


def classify(c: int, frontier_min: int, solved_min: int) -> str:
    if c >= solved_min:
        return "solved"
    if c >= frontier_min:
        return "frontier"
    return "cliff"


def frozen_questions(run_dir: Path, cfg: Config) -> list[QuestionRecord]:
    """Materialize the sampled questions once; keyed by the adapter spec only,
    so changing rollout/model params never resamples the question set. The
    spec is hashed in EXPANDED form so editing a preset in code re-freezes."""
    path = run_dir / "questions.jsonl"
    key = stable_hash("passrate_questions", cfg.data.adapter,
                      resolve_adapter_args(cfg.data.adapter, cfg.data.adapter_args))
    if not is_done(path, config_hash=key):
        records = load_questions(cfg.data.adapter, cfg.data.adapter_args)
        if not records:
            raise RuntimeError(f"adapter {cfg.data.adapter!r} produced no questions")
        n = QuestionRecord.dump_jsonl(path, records)
        mark_done(path, count=n, config_hash=key)
    return list(QuestionRecord.load_jsonl(path))


def generate_and_grade(cfg: Config, questions: list[QuestionRecord], tokenizer,
                       grader, model_path: str, work_dir: Path) -> list[dict]:
    prompts = {
        q.qid: render_question_prompt(
            tokenizer, q.question,
            system_prompt=cfg.model.system_prompt,
            question_suffix=cfg.data.question_suffix,
            chat_template_kwargs=cfg.model.chat_template_kwargs,
        ).token_ids
        for q in questions
    }
    needed = max(len(ids) for ids in prompts.values()) + cfg.rollout.max_tokens
    engine_cfg = dataclasses.replace(
        cfg.engine, max_model_len=max(cfg.engine.max_model_len, needed)
    )
    results = run_pool(
        [
            GenRequest(rid=q.qid, prompt_token_ids=prompts[q.qid], n=cfg.rollout.n,
                       seed=stable_seed(cfg.run.seed, "passrate", q.qid))
            for q in questions
        ],
        mode="generate", model_path=model_path,
        sampling={"temperature": cfg.rollout.temperature, "top_p": cfg.rollout.top_p,
                  "max_tokens": cfg.rollout.max_tokens},
        engine_cfg=engine_cfg, work_dir=work_dir, dtype=cfg.model.dtype,
    )
    rows = []
    for q, r in zip(questions, results):
        for i, s in enumerate(r.samples):
            v = grader.verify(q, s["text"])
            raw_correct = bool(v.correct)
            clean_correct = raw_correct and s.get("finish_reason") != "length"
            rows.append({
                "qid": q.qid, "sample_idx": i,
                "correct": clean_correct,
                "raw_correct": raw_correct,
                "formatted": bool(v.meta.get("formatted", v.extracted_answer is not None)),
                "extracted": v.extracted_answer,
                "finish_reason": s.get("finish_reason", ""),
                "n_tokens": len(s.get("token_ids") or []),
                "response_text": s["text"],
            })
    return rows


def summarize(rows: list[dict], k: int, grader, gold_by_qid: dict[str, str],
              frontier_min: int, solved_min: int) -> tuple[dict, list[dict]]:
    """(metrics, question_stats) — pure function of the graded samples."""
    by_qid: dict[str, list[dict]] = {}
    for row in rows:
        by_qid.setdefault(row["qid"], []).append(row)

    question_stats = []
    for qid, samples in by_qid.items():
        clean = [
            bool(s.get("raw_correct", s["correct"]))
            and s.get("finish_reason") != "length"
            for s in samples
        ]
        c = sum(clean)
        c_raw = sum(bool(s.get("raw_correct", s["correct"])) for s in samples)
        n = len(samples)
        question_stats.append({
            "qid": qid, "n": n, "c": c, "c_raw": c_raw,
            "pass_rate": round(c / n, 4),
            "raw_pass_rate": round(c_raw / n, 4),
            "class": classify(c, frontier_min, solved_min),
            "n_truncated": sum(s["finish_reason"] == "length" for s in samples),
            "final_answer": gold_by_qid.get(qid, ""),
        })
    question_stats.sort(key=lambda r: (r["c"], r["qid"]))

    counts = [r["c"] for r in question_stats]
    raw_counts = [r["c_raw"] for r in question_stats]
    classes = Counter(r["class"] for r in question_stats)
    n_q = len(question_stats)
    metrics = {
        "n_questions": n_q,
        "k": k,
        "hist": {str(c): n for c, n in sorted(Counter(counts).items())},
        "n_cliff": classes["cliff"],
        "n_frontier": classes["frontier"],
        "n_solved": classes["solved"],
        "frac_cliff": round(classes["cliff"] / n_q, 4),
        "frac_frontier": round(classes["frontier"] / n_q, 4),
        "frac_solved": round(classes["solved"] / n_q, 4),
        "thresholds": {"frontier_min": frontier_min, "solved_min": solved_min},
        "solve_rate": round(sum(c > 0 for c in counts) / n_q, 4),
        "raw_solve_rate": round(sum(c > 0 for c in raw_counts) / n_q, 4),
        "all_correct_rate": round(sum(c == k for c in counts) / n_q, 4),
        "raw_all_correct_rate": round(sum(c == k for c in raw_counts) / n_q, 4),
        "mean_pass_rate": round(sum(r["pass_rate"] for r in question_stats) / n_q, 4),
        "raw_mean_pass_rate": round(
            sum(r["raw_pass_rate"] for r in question_stats) / n_q, 4
        ),
    }
    metrics.update(summarize_benchmark("passrate", k, rows, gold_by_qid, grader))
    return metrics, question_stats


def print_table(metrics: dict, question_stats: list[dict], k: int) -> None:
    n_q = metrics["n_questions"]
    hist = metrics["hist"]
    thr = metrics["thresholds"]
    print(f"\n  c/{k} | #questions |     % | class")
    print("  " + "-" * 48)
    for c in range(k + 1):
        n = hist.get(str(c), 0)
        cls = classify(c, thr["frontier_min"], thr["solved_min"])
        bar = "#" * round(40 * n / n_q)
        print(f"  {c:>3}  | {n:>10} | {100 * n / n_q:>5.1f} | {cls:<8} {bar}")
    print(
        f"\n  cliff {metrics['n_cliff']} ({100 * metrics['frac_cliff']:.1f}%) | "
        f"frontier {metrics['n_frontier']} ({100 * metrics['frac_frontier']:.1f}%) | "
        f"solved {metrics['n_solved']} ({100 * metrics['frac_solved']:.1f}%)\n"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = Config.load(args.config, overrides=args.override)
    run_dir = Path(args.run_dir)
    freeze_run_config(cfg, run_dir)

    questions = frozen_questions(run_dir, cfg)
    print(f"[passrate] {len(questions)} questions, K={cfg.rollout.n}")

    raw_model_path = args.model_path or cfg.model.base

    from transformers import AutoTokenizer

    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(raw_model_path)
        lens = []
        for q in questions:
            lens.append(len(render_question_prompt(
                tokenizer, q.question,
                system_prompt=cfg.model.system_prompt,
                question_suffix=cfg.data.question_suffix,
                chat_template_kwargs=cfg.model.chat_template_kwargs,
            ).token_ids))
        example = render_question_prompt(
            tokenizer, questions[0].question,
            system_prompt=cfg.model.system_prompt,
            question_suffix=cfg.data.question_suffix,
            chat_template_kwargs=cfg.model.chat_template_kwargs,
        )
        print(f"[passrate] dry-run: prompt tokens min/mean/max = "
              f"{min(lens)}/{sum(lens) / len(lens):.0f}/{max(lens)}")
        print(f"[passrate] dry-run: example prompt for {questions[0].qid} "
              f"(gold={questions[0].final_answer!r}):\n{example.text}")
        return

    model_path = resolve_model_path(raw_model_path, cfg.model.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    grader = build(VERIFIERS, cfg.partition.verifier)

    samples_path = run_dir / "samples.jsonl"
    # Resume unit keyed by exactly what changes the samples — not the full
    # config hash, so e.g. threshold flags never trigger regeneration. The
    # adapter spec IS part of the key: it determines the question set, and
    # without it a dataset switch in the same run dir would silently reuse the
    # previous dataset's generations against the new gold set.
    key = stable_hash(
        "passrate_samples", model_path, cfg.data.adapter,
        resolve_adapter_args(cfg.data.adapter, cfg.data.adapter_args),
        dataclasses.asdict(cfg.rollout), cfg.run.seed,
        cfg.model.system_prompt, cfg.data.question_suffix, cfg.model.chat_template_kwargs,
    )
    if not is_done(samples_path, config_hash=key):
        # Pool work dir embeds the key: run_pool's worker-level resume is not
        # config-aware, so stale shards must never alias a new spec.
        rows = generate_and_grade(cfg, questions, tokenizer, grader, model_path,
                                  work_dir=run_dir / f"pool_{key[:8]}")
        write_jsonl(samples_path, rows)
        mark_done(samples_path, count=len(rows), config_hash=key)
    rows = list(read_jsonl(samples_path))

    gold_by_qid = {q.qid: q.final_answer for q in questions}
    metrics, question_stats = summarize(rows, cfg.rollout.n, grader, gold_by_qid,
                                        args.frontier_min, args.solved_min)
    metrics = {"model_path": model_path,
               # resolved form: with a preset, the raw args carry no hf_name
               "hf_name": resolve_adapter_args(cfg.data.adapter, cfg.data.adapter_args)
                          .get("hf_name", cfg.data.adapter),
               **metrics}
    write_jsonl(run_dir / "question_stats.jsonl", question_stats)
    write_json(run_dir / "metrics.json", metrics)

    wb = stage_wandb(
        cfg, name=run_dir.name, id_file=run_dir / "wandb_id.txt", job_type="passrate",
        config={"model_path": model_path, "rollout": dataclasses.asdict(cfg.rollout),
                "adapter_args": cfg.data.adapter_args,
                "thresholds": metrics["thresholds"]},
    )
    log_metrics(wb, {k_: v for k_, v in metrics.items() if not isinstance(v, dict)})
    # Raw per-question correct counts -> wandb.Histogram (numeric-list path).
    log_metrics(wb, {"passrate/correct_counts": [r["c"] for r in question_stats]})
    if wb is not None:
        wb.finish()

    print_table(metrics, question_stats, cfg.rollout.n)
    print(f"[passrate] metrics: {metrics}")
    print(f"[passrate] wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
