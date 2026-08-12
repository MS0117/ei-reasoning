"""Stage: benchmark_eval — external competition benchmarks (AIME/HMMT/MATH-500).

Complements eval.py: that stage tracks in-distribution progress on the run's
own holdout split; this one measures absolute capability on public benchmarks
with literature-comparable semantics ported from OPSD's evaluate_math.py —
strict last-\\boxed extraction, pass@n / avg@n / majority-vote@n / format rate.

Works on any checkpoint: EI ckpts, vanilla HF hub ids, or PEFT LoRA adapter
dirs (auto-merged, see lora.py). Standalone launcher: scripts/eval_bench.sh.

Outputs under iter_k/benchmark_eval/:
  <bench>/samples.jsonl   graded generations (resume unit, keyed per benchmark)
  metrics.json            {"<bench>/avg@n": ..., ...} — merged into the run's
                          metrics.jsonl by loop.py.
A benchmark whose questions fail to LOAD (e.g. an unpublished dataset id) is
reported as <bench>/error instead of killing the loop; generation/grading
failures still raise.
"""

from __future__ import annotations

import dataclasses
import sys
from math import comb
from pathlib import Path

from .config import BenchmarkCfg, load_stage_config, stage_argparser
from .data import load_benchmark_questions
from .engine import GenRequest, run_pool
from .lora import resolve_model_path
from .records import QuestionRecord
from .registry import VERIFIERS, build
from .templates import render_question_prompt
from .utils import (
    is_done,
    iter_dir,
    mark_done,
    read_jsonl,
    stable_hash,
    stable_seed,
    write_json,
    write_jsonl,
)
from .wandb_utils import log_metrics, stage_wandb
from . import verifier as _verifier_registration  # noqa: F401 — @register side effect


def main(argv: list[str] | None = None) -> None:
    args = stage_argparser("EI benchmark_eval stage", model_required=False).parse_args(argv)
    cfg = load_stage_config(args)
    # Standalone (scripts/eval_bench.sh with no MODEL): grade the config's own
    # model.base. Under the loop driver --model-path is always the current policy.
    args.model_path = args.model_path or cfg.model.base
    out_dir = iter_dir(args.run_dir, args.iteration) / "benchmark_eval"
    out_path = out_dir / "metrics.json"
    if is_done(out_path, config_hash=cfg.hash()):
        print(f"[benchmark_eval] {out_path} already done, skipping")
        return

    metrics: dict = {"iter": args.iteration, "model_path": args.model_path}
    if not cfg.eval.benchmarks:
        metrics["skipped"] = "no benchmarks configured"
    elif args.iteration % cfg.eval.benchmark_every != 0:
        metrics["skipped"] = f"cadence (benchmark_every={cfg.eval.benchmark_every})"
    if "skipped" in metrics:
        write_json(out_path, metrics)
        mark_done(out_path, count=0, config_hash=cfg.hash())
        print(f"[benchmark_eval] {metrics['skipped']}")
        return

    model_path = resolve_model_path(args.model_path, cfg.model.dtype)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    grader = build(VERIFIERS, cfg.eval.benchmark_verifier)
    # Standalone runs (scripts/eval_bench.sh) get their own wandb run; under
    # the loop driver this is a no-op and metrics flow through loop.py instead.
    wb = stage_wandb(
        cfg, name=Path(args.run_dir).name, id_file=Path(args.run_dir) / "wandb_id.txt",
        job_type="benchmark_eval",
        config={"model_path": args.model_path, "iter": args.iteration,
                "benchmarks": [dataclasses.asdict(b) for b in cfg.eval.benchmarks]},
    )

    failed_benchmarks = []
    for bench in cfg.eval.benchmarks:
        try:
            questions = _frozen_questions(args.run_dir, bench)
        except Exception as e:
            print(f"[benchmark_eval] {bench.name}: LOAD FAILED — {type(e).__name__}: {e}")
            metrics[f"{bench.name}/error"] = f"{type(e).__name__}: {e}"
            failed_benchmarks.append(bench.name)
            log_metrics(wb, {f"{bench.name}/error": metrics[f"{bench.name}/error"]})
            continue

        samples_path = out_dir / bench.name / "samples.jsonl"
        # Resume unit: regenerating is the expensive part, so samples are keyed
        # by exactly what changes them — not by the full config hash.
        bench_key = stable_hash(
            "bench_samples", dataclasses.asdict(bench), model_path, cfg.run.seed,
            cfg.model.system_prompt, cfg.data.question_suffix, cfg.model.chat_template_kwargs,
        )
        if not is_done(samples_path, config_hash=bench_key):
            # Keep pool artifacts grouped by the benchmark sample key for easy
            # inspection; run_pool additionally validates every worker cache.
            rows = _generate_and_grade(bench, questions, tokenizer, grader, cfg, model_path,
                                       work_dir=out_dir / bench.name / f"pool_{bench_key[:8]}")
            write_jsonl(samples_path, rows)
            mark_done(samples_path, count=len(rows), config_hash=bench_key)
        rows = list(read_jsonl(samples_path))
        metrics.update(summarize_benchmark(bench.name, bench.n, rows,
                                           {q.qid: q.final_answer for q in questions}, grader))
        # Log each benchmark as it lands so a many-hour run shows partial
        # results in the wandb UI instead of nothing until the very end.
        log_metrics(wb, {k: v for k, v in metrics.items() if k.startswith(f"{bench.name}/")})
        print(f"[benchmark_eval] {bench.name}: " + ", ".join(
            f"{k.split('/', 1)[1]}={v}" for k, v in metrics.items() if k.startswith(f"{bench.name}/")
        ))

    if failed_benchmarks:
        metrics["incomplete_benchmarks"] = failed_benchmarks
    write_json(out_path, metrics)
    if not failed_benchmarks:
        mark_done(out_path, count=len(cfg.eval.benchmarks), config_hash=cfg.hash())
    if wb is not None:
        wb.finish()
    print(f"[benchmark_eval] {metrics}")


def _frozen_questions(run_dir: str, bench: BenchmarkCfg) -> list[QuestionRecord]:
    """Materialize benchmark questions once per run (offline-reproducible;
    HF hub outages can't change or fail a later iteration's eval)."""
    path = Path(run_dir) / "benchmarks" / f"{bench.name}.jsonl"
    key = stable_hash("bench_questions", dataclasses.asdict(bench))
    if not is_done(path, config_hash=key):
        records = load_benchmark_questions(bench)
        if not records:
            raise RuntimeError(f"benchmark {bench.name!r} produced no questions")
        n = QuestionRecord.dump_jsonl(path, records)
        mark_done(path, count=n, config_hash=key)
    return list(QuestionRecord.load_jsonl(path))


def _generate_and_grade(bench, questions, tokenizer, grader, cfg, model_path, work_dir) -> list[dict]:
    prompts = {
        q.qid: render_question_prompt(
            tokenizer, q.question,
            system_prompt=cfg.model.system_prompt,
            question_suffix=cfg.data.question_suffix,
            chat_template_kwargs=cfg.model.chat_template_kwargs,
        ).token_ids
        for q in questions
    }
    # Competition problems need far more room than the training-loop default;
    # widen the worker context to fit prompt + bench.max_tokens.
    needed = max(len(ids) for ids in prompts.values()) + bench.max_tokens
    engine_cfg = dataclasses.replace(
        cfg.engine, max_model_len=max(cfg.engine.max_model_len, needed)
    )
    results = run_pool(
        [
            GenRequest(rid=q.qid, prompt_token_ids=prompts[q.qid], n=bench.n,
                       # No iteration in the seed: the same checkpoint re-graded
                       # at any point reproduces the same samples.
                       seed=stable_seed(cfg.run.seed, "bench", bench.name, q.qid))
            for q in questions
        ],
        mode="generate", model_path=model_path,
        sampling={"temperature": bench.temperature, "top_p": bench.top_p,
                  "top_k": bench.top_k, "min_p": bench.min_p, "max_tokens": bench.max_tokens},
        engine_cfg=engine_cfg, work_dir=work_dir, dtype=cfg.model.dtype,
    )
    rows = []
    for q, r in zip(questions, results):
        for i, s in enumerate(r.samples):
            v = grader.verify(q, s["text"])
            rows.append({
                "qid": q.qid, "sample_idx": i,
                "correct": bool(v.correct),
                "formatted": bool(v.meta.get("formatted", v.extracted_answer is not None)),
                "extracted": v.extracted_answer,
                "finish_reason": s.get("finish_reason", ""),
                "n_tokens": len(s.get("token_ids") or []),
                "response_text": s["text"],
            })
    return rows


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def summarize_benchmark(name: str, n: int, rows: list[dict],
                        gold_by_qid: dict[str, str], grader) -> dict:
    """pass@k / avg@n / maj@n / format & truncation rates from graded samples.
    Pure function of the samples file — metrics can always be recomputed.
    pass@k is reported at every power of 2 up to the per-question sample count
    (unbiased estimator; at k=n it reduces to "any sample correct")."""
    by_qid: dict[str, list[dict]] = {}
    for row in rows:
        by_qid.setdefault(row["qid"], []).append(row)

    n_q = len(by_qid)
    total = sum(len(v) for v in by_qid.values())
    if n_q == 0 or total == 0:
        return {f"{name}/error": "no graded samples"}

    n_correct = n_maj = n_formatted = n_truncated = 0
    counts: list[tuple[int, int]] = []  # (samples, correct) per question
    can_maj = hasattr(grader, "grade_extracted")
    for qid, samples in by_qid.items():
        oks = [s["correct"] for s in samples]
        counts.append((len(oks), sum(oks)))
        n_correct += sum(oks)
        n_formatted += sum(s["formatted"] for s in samples)
        n_truncated += sum(s["finish_reason"] == "length" for s in samples)
        if can_maj:
            answers = [s["extracted"] for s in samples if s["formatted"] and s["extracted"]]
            if answers:
                top = _majority_answer(answers, grader)
                n_maj += grader.grade_extracted(top, gold_by_qid[qid])

    min_n = min(c[0] for c in counts)
    ks = [k for k in (1 << i for i in range(min_n.bit_length())) if k <= min_n]
    if min_n not in ks:
        ks.append(min_n)
    out = {f"{name}/n_questions": n_q}
    for k in ks:
        out[f"{name}/pass@{k}"] = round(sum(_pass_at_k(ni, ci, k) for ni, ci in counts) / n_q, 4)
    out.update({
        f"{name}/avg@{n}": round(n_correct / total, 4),
        f"{name}/format_rate": round(n_formatted / total, 4),
        f"{name}/truncated_rate": round(n_truncated / total, 4),
    })
    if can_maj:
        out[f"{name}/maj@{n}"] = round(n_maj / n_q, 4)
    return out


def _majority_answer(answers: list[str], grader) -> str:
    """Vote by verifier equivalence, preserving first-answer tie breaking."""
    clusters: list[list] = []
    for answer in answers:
        for cluster in clusters:
            if grader.grade_extracted(answer, cluster[0]):
                cluster[1] += 1
                break
        else:
            clusters.append([answer, 1])
    return max(clusters, key=lambda cluster: cluster[1])[0]


if __name__ == "__main__":
    main(sys.argv[1:])
