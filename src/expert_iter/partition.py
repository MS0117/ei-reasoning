"""Stage: partition — grade every rollout sample, split questions into
solved (keep correct trajectories) and unsolved (send to improvement).

Outputs under iter_k/partition/:
  verdicts.jsonl  one VerdictRecord per rollout sample
  solved.jsonl    ≤ partition.solved_keep_max correct trajectories per question
  unsolved.jsonl  one UnsolvedQuestion per question with zero correct samples
  stats.json      solve-rate — the headline EI metric per iteration
"""

from __future__ import annotations

import random
import sys
import time
from collections import defaultdict

from .config import load_stage_config, stage_argparser
from .data import ensure_questions
from .records import (
    QuestionRecord,
    RolloutSample,
    SolvedTrajectory,
    UnsolvedQuestion,
    VerdictRecord,
)
from .registry import VERIFIERS, build
from .utils import is_done, iter_dir, mark_done, write_json
from . import verifier  # noqa: F401 — imported for its @register side effect on VERIFIERS


def main(argv: list[str] | None = None) -> None:
    args = stage_argparser("EI partition stage").parse_args(argv)
    cfg = load_stage_config(args)
    it_dir = iter_dir(args.run_dir, args.iteration)
    out_dir = it_dir / "partition"
    solved_path = out_dir / "solved.jsonl"
    if is_done(solved_path, config_hash=cfg.hash()):
        print(f"[partition] {out_dir} already done, skipping")
        return

    train_questions, _ = ensure_questions(cfg, args.run_dir)
    questions = {q.qid: q for q in train_questions}
    rollouts = list(RolloutSample.load_jsonl(it_dir / "rollout" / "rollouts.jsonl"))
    print(f"[partition] grading {len(rollouts)} samples with verifier={cfg.partition.verifier}")

    verifier = build(VERIFIERS, cfg.partition.verifier)
    t0 = time.time()
    verdicts = verifier.verify_batch([(questions[s.qid], s.response_text) for s in rollouts])
    print(f"[partition] verification took {time.time() - t0:.1f}s")

    verdict_rows = [
        VerdictRecord(
            qid=s.qid, sample_idx=s.sample_idx, correct=v.correct,
            extracted_answer=v.extracted_answer, verifier=cfg.partition.verifier, meta=v.meta,
        )
        for s, v in zip(rollouts, verdicts)
    ]

    (
        solved_rows, unsolved_rows, n_solved_q, n_q,
        n_clean_correct, n_truncated_correct,
    ) = _partition_rows(questions, rollouts, verdicts, cfg, args.iteration)

    VerdictRecord.dump_jsonl(out_dir / "verdicts.jsonl", verdict_rows)
    n_solved = SolvedTrajectory.dump_jsonl(solved_path, solved_rows)
    UnsolvedQuestion.dump_jsonl(out_dir / "unsolved.jsonl", unsolved_rows)

    stats = {
        "iter": args.iteration,
        "n_questions": n_q,
        "n_solved_questions": n_solved_q,
        "n_unsolved_questions": len(unsolved_rows),
        "solve_rate": round(n_solved_q / n_q, 4) if n_q else 0.0,
        "sample_accuracy": round(sum(v.correct for v in verdicts) / len(verdicts), 4) if verdicts else 0.0,
        "clean_sample_accuracy": round(n_clean_correct / len(verdicts), 4) if verdicts else 0.0,
        "n_truncated_correct": n_truncated_correct,
        "n_solved_trajectories_kept": n_solved,
    }
    write_json(out_dir / "stats.json", stats)
    mark_done(solved_path, count=n_solved, config_hash=cfg.hash(), extra=stats)
    print(f"[partition] {stats}")


def _select(samples: list[RolloutSample], keep: int, how: str, rng: random.Random):
    if how == "shortest":
        ranked = sorted(samples, key=lambda s: len(s.response_token_ids))
    elif how == "random":
        ranked = samples[:]
        rng.shuffle(ranked)
    else:  # first
        ranked = sorted(samples, key=lambda s: s.sample_idx)
    return ranked[:keep]


def _partition_rows(questions, rollouts, verdicts, cfg, iteration: int):
    """Split every question exactly once; truncated-correct samples are unsolved."""
    by_qid: dict[str, list[tuple[RolloutSample, bool]]] = defaultdict(list)
    for sample, verdict in zip(rollouts, verdicts, strict=True):
        by_qid[sample.qid].append((sample, verdict.correct))

    missing = sorted(set(questions) - set(by_qid))
    unexpected = sorted(set(by_qid) - set(questions))
    if missing or unexpected:
        raise RuntimeError(
            f"rollout/question mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    rng = random.Random(cfg.run.seed)
    solved_rows: list[SolvedTrajectory] = []
    unsolved_rows: list[UnsolvedQuestion] = []
    n_solved_q = 0
    n_clean_correct = 0
    n_truncated_correct = 0
    for qid, pairs in by_qid.items():
        q = questions[qid]
        clean_correct = [s for s, ok in pairs if ok and s.finish_reason == "stop"]
        n_clean_correct += len(clean_correct)
        n_truncated_correct += sum(
            1 for s, ok in pairs if ok and s.finish_reason != "stop"
        )
        if clean_correct:
            n_solved_q += 1
            solved_rows.extend(
                SolvedTrajectory(
                    qid=qid, question=q.question, final_answer=q.final_answer,
                    sample_idx=s.sample_idx, prompt_token_ids=s.prompt_token_ids,
                    response_token_ids=s.response_token_ids, response_text=s.response_text,
                    iter=iteration,
                )
                for s in _select(
                    clean_correct, cfg.partition.solved_keep_max,
                    cfg.partition.solved_selection, rng,
                )
            )
        else:
            rejected = [
                s for s, ok in pairs if not (ok and s.finish_reason == "stop")
            ]
            unsolved_rows.append(UnsolvedQuestion(
                qid=qid, question=q.question, final_answer=q.final_answer,
                failed_sample_idxs=[s.sample_idx for s in rejected],
                iter=iteration,
            ))

    if n_solved_q + len(unsolved_rows) != len(by_qid):
        raise AssertionError("partition did not classify every question")
    return (
        solved_rows, unsolved_rows, n_solved_q, len(by_qid),
        n_clean_correct, n_truncated_correct,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
