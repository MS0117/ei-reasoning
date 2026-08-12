"""Stage: eval — grade the just-trained checkpoint on the held-out split.

Greedy pass@1 plus sampled pass@k. Metrics land in iter_k/eval/metrics.json
and are appended to runs/<name>/metrics.jsonl by loop.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import load_stage_config, stage_argparser
from .data import ensure_questions
from .engine import GenRequest, run_pool
from .registry import VERIFIERS, build
from .templates import render_question_prompt
from .utils import is_done, iter_dir, mark_done, stable_seed, write_json
from .wandb_utils import log_metrics, stage_wandb
from . import verifier  # noqa: F401 — imported for its @register side effect on VERIFIERS


def main(argv: list[str] | None = None) -> None:
    args = stage_argparser("EI eval stage").parse_args(argv)
    cfg = load_stage_config(args)
    out_dir = iter_dir(args.run_dir, args.iteration) / "eval"
    out_path = out_dir / "metrics.json"
    if is_done(out_path, config_hash=cfg.hash()):
        print(f"[eval] {out_path} already done, skipping")
        return

    _, holdout = ensure_questions(cfg, args.run_dir)
    if not holdout:
        write_json(out_path, {"iter": args.iteration, "skipped": "empty holdout"})
        mark_done(out_path, count=0, config_hash=cfg.hash())
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts = {
        q.qid: render_question_prompt(
            tokenizer, q.question,
            system_prompt=cfg.model.system_prompt,
            question_suffix=cfg.data.question_suffix,
            chat_template_kwargs=cfg.model.chat_template_kwargs,
        ).token_ids
        for q in holdout
    }
    verifier = build(VERIFIERS, cfg.partition.verifier)
    metrics: dict = {"iter": args.iteration, "model_path": args.model_path, "n_holdout": len(holdout)}

    if cfg.eval.greedy_pass1:
        results = run_pool(
            [GenRequest(rid=q.qid, prompt_token_ids=prompts[q.qid], n=1, seed=0) for q in holdout],
            mode="generate", model_path=args.model_path,
            sampling={"temperature": 0.0, "top_p": 1.0, "max_tokens": cfg.eval.max_tokens},
            engine_cfg=cfg.engine, work_dir=out_dir / "pool_greedy",
            dtype=cfg.model.dtype,
        )
        raw_correct = clean_correct = truncated = 0
        for q, result in zip(holdout, results):
            sample = result.samples[0]
            ok = verifier.verify(q, sample["text"]).correct
            is_clean = sample.get("finish_reason") != "length"
            raw_correct += ok
            clean_correct += ok and is_clean
            truncated += not is_clean
        metrics["pass@1_greedy"] = round(raw_correct / len(holdout), 4)
        metrics["pass@1_greedy_clean"] = round(clean_correct / len(holdout), 4)
        metrics["truncated_rate_greedy"] = round(truncated / len(holdout), 4)

    k = cfg.eval.passk.k
    if k > 1:
        results = run_pool(
            [
                GenRequest(rid=q.qid, prompt_token_ids=prompts[q.qid], n=k,
                           seed=stable_seed(cfg.run.seed, "eval", args.iteration, q.qid))
                for q in holdout
            ],
            mode="generate", model_path=args.model_path,
            sampling={"temperature": cfg.eval.passk.temperature,
                      "top_p": cfg.eval.passk.top_p, "max_tokens": cfg.eval.max_tokens},
            engine_cfg=cfg.engine, work_dir=out_dir / "pool_passk",
            dtype=cfg.model.dtype,
        )
        n_pass = n_avg = n_pass_clean = n_avg_clean = n_truncated = 0.0
        for q, r in zip(holdout, results):
            oks = [verifier.verify(q, s["text"]).correct for s in r.samples]
            clean_oks = [
                ok and s.get("finish_reason") != "length"
                for ok, s in zip(oks, r.samples)
            ]
            n_pass += any(oks)
            n_avg += sum(oks) / len(oks)
            n_pass_clean += any(clean_oks)
            n_avg_clean += sum(clean_oks) / len(clean_oks)
            n_truncated += sum(s.get("finish_reason") == "length" for s in r.samples)
        metrics[f"pass@{k}"] = round(n_pass / len(holdout), 4)
        metrics[f"avg@{k}"] = round(n_avg / len(holdout), 4)
        metrics[f"pass@{k}_clean"] = round(n_pass_clean / len(holdout), 4)
        metrics[f"avg@{k}_clean"] = round(n_avg_clean / len(holdout), 4)
        metrics[f"truncated_rate@{k}"] = round(n_truncated / (len(holdout) * k), 4)

    write_json(out_path, metrics)
    mark_done(out_path, count=len(holdout), config_hash=cfg.hash())
    # No-op under the loop driver (loop.py logs the merged iteration row);
    # standalone `python -m expert_iter.eval` gets its own wandb run.
    wb = stage_wandb(cfg, name=f"{Path(args.run_dir).name}-eval",
                     id_file=Path(args.run_dir) / "wandb_eval_id.txt", job_type="eval",
                     config={"model_path": args.model_path, "iter": args.iteration})
    log_metrics(wb, metrics, step=args.iteration)
    if wb is not None:
        wb.finish()
    print(f"[eval] {metrics}")


if __name__ == "__main__":
    main(sys.argv[1:])
