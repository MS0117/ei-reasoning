"""Stage: rollout — the current policy samples n responses per train question.

Output: iter_k/rollout/rollouts.jsonl (one RolloutSample per sample).
"""

from __future__ import annotations

import sys

from .config import load_stage_config, stage_argparser
from .data import ensure_questions
from .engine import GenRequest, run_pool
from .records import RolloutSample
from .templates import render_question_prompt
from .utils import is_done, iter_dir, mark_done, stable_seed


def main(argv: list[str] | None = None) -> None:
    args = stage_argparser("EI rollout stage").parse_args(argv)
    cfg = load_stage_config(args)
    out_dir = iter_dir(args.run_dir, args.iteration) / "rollout"
    out_path = out_dir / "rollouts.jsonl"
    if is_done(out_path, config_hash=cfg.hash()):
        print(f"[rollout] {out_path} already done, skipping")
        return

    train_questions, _ = ensure_questions(cfg, args.run_dir)
    print(f"[rollout] {len(train_questions)} questions, n={cfg.rollout.n}, model={args.model_path}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts = {
        q.qid: render_question_prompt(
            tokenizer, q.question,
            system_prompt=cfg.model.system_prompt,
            question_suffix=cfg.data.question_suffix,
            chat_template_kwargs=cfg.model.chat_template_kwargs,
        )
        for q in train_questions
    }

    requests = [
        GenRequest(
            rid=q.qid,
            prompt_token_ids=prompts[q.qid].token_ids,
            n=cfg.rollout.n,
            seed=stable_seed(cfg.run.seed, "rollout", args.iteration, q.qid),
        )
        for q in train_questions
    ]
    results = run_pool(
        requests,
        mode="generate",
        model_path=args.model_path,
        sampling={
            "temperature": cfg.rollout.temperature,
            "top_p": cfg.rollout.top_p,
            "max_tokens": cfg.rollout.max_tokens,
        },
        engine_cfg=cfg.engine,
        work_dir=out_dir / "pool",
        dtype=cfg.model.dtype,
    )

    def rows():
        for q, res in zip(train_questions, results):
            rp = prompts[q.qid]
            for si, s in enumerate(res.samples):
                yield RolloutSample(
                    qid=q.qid,
                    sample_idx=si,
                    prompt_text=rp.text,
                    prompt_token_ids=rp.token_ids,
                    response_text=s["text"],
                    response_token_ids=s["token_ids"],
                    finish_reason=s["finish_reason"],
                    gen={
                        "temperature": cfg.rollout.temperature,
                        "top_p": cfg.rollout.top_p,
                        "max_tokens": cfg.rollout.max_tokens,
                        "seed": stable_seed(cfg.run.seed, "rollout", args.iteration, q.qid),
                    },
                    model_path=args.model_path,
                    iter=args.iteration,
                ).to_dict()

    from .utils import write_jsonl

    n = write_jsonl(out_path, rows())
    mark_done(out_path, count=n, config_hash=cfg.hash())
    print(f"[rollout] wrote {n} samples -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
