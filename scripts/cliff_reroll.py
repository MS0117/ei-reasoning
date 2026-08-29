"""Base-policy re-rolls of the cliff questions — the mean-reversion noise floor.

Cliffs are selected as 0/8 under the base policy, so re-sampling alone flips
~24% of them to >=1 correct at n=32 (measured: self_resample 0.243). Every
transfer claim on the A/B cliff sets must therefore be read AGAINST this floor,
which is why it is drawn (twice, independently) before any training arm.

Usage (GPU; run inside the a100 tmux session):
  .venv/bin/python scripts/cliff_reroll.py --run-dir runs/<frozen L2 run> \
      [--n 32] [--passes 2] [--model-path <default: cfg.model.base>] \
      [--qids-file holdout|cliff_split.json] [--out <run>/iter_0/reroll]

Writes, per pass i: <out>/pass_{i}/rollouts.jsonl + verdicts.jsonl (+ .done),
and <out>/summary.json with per-qid correct counts per pass.

CPU-safe dry check: --list only prints the qid set and exits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from expert_iter.config import Config
from expert_iter.data import ensure_questions
from expert_iter.records import RolloutSample, UnsolvedQuestion, VerdictRecord
from expert_iter.registry import VERIFIERS, build
from expert_iter.templates import render_question_prompt
from expert_iter.utils import is_done, mark_done, stable_seed, write_json, write_jsonl


def _load_qids(args, run_dir: Path, holdout_qids: list[str]) -> list[str]:
    if args.qids_file == "holdout":
        # The L5 headline set: the run's external cliff holdout, never trained
        # on by any arm. Named rather than pathed so every arm quotes the same
        # target and the floor is drawn once for all of them.
        if not holdout_qids:
            raise SystemExit("[reroll] --qids-file holdout: this run has no holdout questions")
        return holdout_qids
    if args.qids_file:
        path, _, key = args.qids_file.partition(":")
        d = json.loads(Path(path).read_text())
        if isinstance(d, dict):
            if key:                       # e.g. cliff_split.json:B for post-train B re-rolls
                qids = d[key]
            elif "A" in d or "B" in d:
                qids = sorted(set(d.get("A", [])) | set(d.get("B", [])))
            else:
                qids = d.get("exclude")
        else:
            qids = d
        if not qids:
            raise SystemExit(f"{args.qids_file}: no qids found")
        return list(qids)
    unsolved = run_dir / f"iter_{args.iteration}" / "partition" / "unsolved.jsonl"
    return [u.qid for u in UnsolvedQuestion.load_jsonl(unsolved)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--model-path", default=None, help="default: cfg.model.base (the floor is the BASE policy)")
    ap.add_argument("--qids-file", default=None,
                    help="`holdout` (the run's external cliff holdout — the L5 headline set), "
                         "cliff_split.json[:A|B|exclude], or a qid list; default: all unsolved qids")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true", help="print the qid set and exit (CPU)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    cfg = Config.load(run_dir / "config.yaml")
    train_questions, holdout_questions = ensure_questions(cfg, run_dir)
    qids = _load_qids(args, run_dir, [q.qid for q in holdout_questions])
    print(f"[reroll] {len(qids)} cliff questions, n={args.n} x {args.passes} passes")
    if args.list:
        print("\n".join(qids))
        return

    model_path = args.model_path or cfg.model.base
    out = Path(args.out) if args.out else run_dir / f"iter_{args.iteration}" / "reroll"

    from transformers import AutoTokenizer
    from expert_iter.engine import GenRequest, run_pool

    # Both splits: the training cliffs (A/B work) live in train.jsonl, while the
    # L5 headline holdout lives in holdout.jsonl.
    pool = {q.qid: q for q in (*train_questions, *holdout_questions)}
    questions = {qid: pool[qid] for qid in qids if qid in pool}
    missing = [q for q in qids if q not in questions]
    if missing:
        raise SystemExit(
            f"[reroll] {len(missing)} qids in neither questions/train.jsonl nor "
            f"questions/holdout.jsonl, e.g. {missing[:3]}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = {
        qid: render_question_prompt(
            tokenizer, questions[qid].question,
            system_prompt=cfg.model.system_prompt,
            question_suffix=cfg.data.question_suffix,
            chat_template_kwargs=cfg.model.chat_template_kwargs,
        )
        for qid in qids
    }
    verifier = build(VERIFIERS, cfg.partition.verifier)
    summary: dict[str, dict] = {}

    for pi in range(args.passes):
        pdir = out / f"pass_{pi}"
        vpath = pdir / "verdicts.jsonl"
        if is_done(vpath):
            print(f"[reroll] pass {pi} already done, loading")
            counts = Counter()
            for v in VerdictRecord.load_jsonl(vpath):
                counts[v.qid] += bool(v.correct)
            summary[f"pass_{pi}"] = {q: counts.get(q, 0) for q in qids}
            continue
        requests = [
            GenRequest(
                rid=qid, prompt_token_ids=prompts[qid].token_ids, n=args.n,
                # seed namespace differs from the rollout stage AND per pass, so
                # the two passes are independent draws
                seed=stable_seed(cfg.run.seed, "reroll", pi, qid),
            )
            for qid in qids
        ]
        results = run_pool(
            requests, mode="generate", model_path=model_path,
            sampling={"temperature": cfg.rollout.temperature,
                      "top_p": cfg.rollout.top_p,
                      "max_tokens": cfg.rollout.max_tokens},
            engine_cfg=cfg.engine, work_dir=pdir / "pool", dtype=cfg.model.dtype,
        )
        roll_rows, pairs = [], []
        for qid, res in zip(qids, results):
            for si, s in enumerate(res.samples):
                roll_rows.append(RolloutSample(
                    qid=qid, sample_idx=si, prompt_text=prompts[qid].text,
                    prompt_token_ids=prompts[qid].token_ids,
                    response_text=s["text"], response_token_ids=s["token_ids"],
                    finish_reason=s["finish_reason"],
                    gen={"temperature": cfg.rollout.temperature, "pass": pi},
                    model_path=model_path, iter=args.iteration,
                ))
                pairs.append((questions[qid], s["text"]))
        verdicts = verifier.verify_batch(pairs)
        vrows = [
            VerdictRecord(qid=r.qid, sample_idx=r.sample_idx, correct=v.correct,
                          extracted_answer=v.extracted_answer,
                          verifier=cfg.partition.verifier, meta=v.meta)
            for r, v in zip(roll_rows, verdicts)
        ]
        write_jsonl(pdir / "rollouts.jsonl", (r.to_dict() for r in roll_rows))
        n = write_jsonl(vpath, (v.to_dict() for v in vrows))
        mark_done(vpath, count=n, config_hash=cfg.hash())
        counts = Counter()
        for v in vrows:
            counts[v.qid] += bool(v.correct)
        summary[f"pass_{pi}"] = {q: counts.get(q, 0) for q in qids}
        n_pos = sum(c > 0 for c in summary[f"pass_{pi}"].values())
        print(f"[reroll] pass {pi}: {n_pos}/{len(qids)} cliffs re-roll to >0 correct "
              f"(mean-reversion floor {n_pos / len(qids):.3f})")

    write_json(out / "summary.json", {
        "n": args.n, "passes": args.passes, "model_path": model_path,
        "qids": qids, "correct_counts": summary,
    })
    print(f"[reroll] summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
