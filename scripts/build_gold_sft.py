"""Build the OFFLINE gold-SFT dataset (L5 baseline arm) — CPU only, no rollout.

Gold SFT trains on the reference solution y* for every question, so it needs
neither sampling nor cliff detection. This writes iter_0/dataset/train_sft.jsonl
straight from questions/train.jsonl and stamps the .done marker, after which

    expert_iter.loop --config <run>/config.yaml

runs only [train, eval, benchmark_eval] (configs/methods/l5_gold_sft.yaml pins
that stage list). scripts/l5_gold_sft.sh chains the two steps.

y* becomes token ids through templates.gold_pair_ids — the one sanctioned
boundary where privileged text is tokenized, and the same one the lora_sft
operator's fit uses. Prompts are rendered by templates.render_question_prompt,
so this arm's prompt is byte-identical to every other arm's.

By default a gold solution is kept only if the run's verifier accepts it. That
is not a purity gate but a FORMAT one: on the 8k mix only 42.8% of cliff golds
carry \\boxed at all, and rows whose answer cannot be extracted teach the model
not to emit one, which collapses accuracy under the math_strict benchmark
verifier. --no-verify turns the screen off for a robustness arm.

Usage (seconds):
  .venv/bin/python scripts/build_gold_sft.py -c configs/methods/l5_gold_sft.yaml \
      --override run.name=l5_gold_sft_20260828
"""

from __future__ import annotations

import argparse
from pathlib import Path

from expert_iter import verifier  # noqa: F401 — @register side effect on VERIFIERS
from expert_iter.config import Config, freeze_run_config
from expert_iter.data import ensure_questions
from expert_iter.records import DPOExample, SFTExample
from expert_iter.registry import VERIFIERS, build
from expert_iter.templates import gold_pair_ids, render_question_prompt
from expert_iter.utils import is_done, mark_done, stable_hash, write_json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the offline gold-SFT dataset")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--override", action="append", default=[], metavar="a.b=c")
    ap.add_argument("--run-dir", default=None,
                    help="default: <run.output_root>/<run.name>, as the loop resolves it")
    ap.add_argument("--no-verify", action="store_true",
                    help="keep gold the verifier rejects (robustness arm; see module docstring)")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config, overrides=args.override)
    cfg.validate()
    run_dir = Path(args.run_dir or Path(cfg.run.output_root) / cfg.run.name)
    freeze_run_config(cfg, run_dir)

    out_dir = run_dir / "iter_0" / "dataset"
    train_sft_path = out_dir / "train_sft.jsonl"
    if is_done(train_sft_path, config_hash=cfg.hash()):
        print(f"[build_gold_sft] {train_sft_path} already done, skipping")
        _print_next(run_dir, cfg)
        return

    train_questions, holdout = ensure_questions(cfg, run_dir)
    print(f"[build_gold_sft] {len(train_questions)} train / {len(holdout)} holdout questions")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base)
    eos = tokenizer.eos_token_id
    verifier = None if args.no_verify else build(VERIFIERS, cfg.partition.verifier)

    rows: list[SFTExample] = []
    n_no_gold = n_rejected = n_tokens = 0
    for q in train_questions:
        y = str(q.meta.get("gold_solution") or "").strip()
        if not y:
            n_no_gold += 1
            continue
        if verifier is not None and not verifier.verify(q, y).correct:
            n_rejected += 1
            continue
        prompt_ids = render_question_prompt(
            tokenizer, q.question,
            system_prompt=cfg.model.system_prompt,
            question_suffix=cfg.data.question_suffix,
            chat_template_kwargs=cfg.model.chat_template_kwargs,
        ).token_ids
        input_ids, prompt_len = gold_pair_ids(tokenizer, prompt_ids, y, eos)
        completion_len = len(input_ids) - prompt_len
        n_tokens += completion_len
        row = SFTExample(
            uid=stable_hash("gold_sft", q.qid, 0),
            qid=q.qid,
            # "solved" (not "improved"): there is no anchor and no cliff bracket
            # here, so these are plain full-response rows in the `solution`
            # region. SFTExample.validate() enforces anchor_len == 0 for it.
            source="solved",
            input_ids=input_ids,
            prompt_len=prompt_len,
            anchor_len=0,
            completion_len=completion_len,
            text=y,
            iter_created=0,
        )
        row.validate()
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "no gold rows built — check that the question set carries "
            "meta.gold_solution (data/mixes/*.jsonl do; hf_math needs "
            "adapter_args.include_solution=true)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    SFTExample.dump_jsonl(out_dir / "examples_sft.jsonl", rows)
    n = SFTExample.dump_jsonl(train_sft_path, rows)
    # train reads train_dpo.jsonl only under a dpo objective, but an empty file
    # keeps the dataset dir shaped like build_dataset's output.
    DPOExample.dump_jsonl(out_dir / "train_dpo.jsonl", [])

    n_seen = len(train_questions)
    n_with_gold = n_seen - n_no_gold
    stats = {
        "source": "build_gold_sft",
        "verifier": None if verifier is None else cfg.partition.verifier,
        "n_questions": n_seen,
        "n_without_gold": n_no_gold,
        "n_with_gold": n_with_gold,
        "n_verifier_rejected": n_rejected,
        "n_rows": n,
        # the arm's yield, quoted in configs/methods/l5_gold_sft.yaml
        "gold_accept_rate": round(n / n_with_gold, 4) if n_with_gold else 0.0,
        "completion_tokens": n_tokens,
        "mean_completion_tokens": round(n_tokens / n, 1),
    }
    write_json(out_dir / "stats.json", stats)
    mark_done(train_sft_path, count=n, config_hash=cfg.hash(), extra=stats)
    print(f"[build_gold_sft] {stats}")
    _print_next(run_dir, cfg)


def _print_next(run_dir: Path, cfg: Config) -> None:
    print(f"[build_gold_sft] next: bash scripts/run.sh -c <this config> -r {cfg.run.name} -b")
    print(f"[build_gold_sft]   or: .venv/bin/python -m expert_iter.loop --config {run_dir}/config.yaml")


if __name__ == "__main__":
    main()
