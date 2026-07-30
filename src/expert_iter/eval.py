"""Stage: eval — grade the just-trained checkpoint on the held-out split.

Greedy pass@1 plus sampled pass@k. Metrics land in iter_k/eval/metrics.json
and are appended to runs/<name>/metrics.jsonl by loop.py.
"""

from __future__ import annotations

import sys

from .config import load_stage_config, stage_argparser
from .data import ensure_questions
from .engine import GenRequest, run_pool
from .registry import VERIFIERS, build
from .templates import render_question_prompt
from .utils import is_done, iter_dir, mark_done, stable_seed, write_json


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
        )
        correct = sum(
            verifier.verify(q, r.samples[0]["text"]).correct
            for q, r in zip(holdout, results)
        )
        metrics["pass@1_greedy"] = round(correct / len(holdout), 4)

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
        )
        n_pass = n_avg = 0.0
        for q, r in zip(holdout, results):
            oks = [verifier.verify(q, s["text"]).correct for s in r.samples]
            n_pass += any(oks)
            n_avg += sum(oks) / len(oks)
        metrics[f"pass@{k}"] = round(n_pass / len(holdout), 4)
        metrics[f"avg@{k}"] = round(n_avg / len(holdout), 4)

    write_json(out_path, metrics)
    mark_done(out_path, count=len(holdout), config_hash=cfg.hash())
    print(f"[eval] {metrics}")


if __name__ == "__main__":
    main(sys.argv[1:])
