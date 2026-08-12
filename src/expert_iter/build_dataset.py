"""Stage: build_dataset — assemble the train-ready mixed dataset.

Per iteration this writes, under iter_k/dataset/:
  examples_sft.jsonl  THIS iteration's contribution (solved + improved)
  examples_dpo.jsonl  THIS iteration's anchor-conditioned preference pairs
  train_sft.jsonl     what the trainer actually reads: this iter's examples,
                      plus (if data.accumulate) every prior iteration's
                      examples_*.jsonl, deduped by input_ids hash
  train_dpo.jsonl     same, for DPO
  stats.json

The solved response / improved continuation gets EOS appended (idempotent) so
the model learns to terminate.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from .config import load_stage_config, stage_argparser
from .records import DPOExample, ImprovedCandidate, RolloutSample, SFTExample
from .templates import ensure_eos, training_input_ids
from .utils import is_done, iter_dir, mark_done, stable_hash, write_json


def main(argv: list[str] | None = None) -> None:
    args = stage_argparser("EI build_dataset stage").parse_args(argv)
    cfg = load_stage_config(args)
    it_dir = iter_dir(args.run_dir, args.iteration)
    out_dir = it_dir / "dataset"
    train_sft_path = out_dir / "train_sft.jsonl"
    if is_done(train_sft_path, config_hash=cfg.hash()):
        print(f"[build_dataset] {out_dir} already done, skipping")
        return

    from transformers import AutoTokenizer

    eos = AutoTokenizer.from_pretrained(args.model_path).eos_token_id

    from .records import SolvedTrajectory

    sft: list[SFTExample] = []
    for s in SolvedTrajectory.load_jsonl(it_dir / "partition" / "solved.jsonl"):
        completion = ensure_eos(s.response_token_ids, eos)
        sft.append(SFTExample(
            uid=stable_hash("solved", s.qid, s.sample_idx, args.iteration),
            qid=s.qid, source="solved",
            input_ids=training_input_ids(s.prompt_token_ids, [], completion),
            prompt_len=len(s.prompt_token_ids), anchor_len=0,
            completion_len=len(completion),
            text=s.response_text, iter_created=args.iteration,
        ))

    kept = list(ImprovedCandidate.load_jsonl(it_dir / "filtered" / "kept.jsonl"))
    for c in kept:
        completion = ensure_eos(c.continuation_token_ids, eos)
        sft.append(SFTExample(
            uid=stable_hash("improved", c.qid, c.base_sample_idx, c.attempt_idx, args.iteration),
            qid=c.qid, source="improved",
            input_ids=training_input_ids(c.prompt_token_ids, c.anchor_token_ids, completion),
            prompt_len=len(c.prompt_token_ids), anchor_len=len(c.anchor_token_ids),
            completion_len=len(completion),
            text=c.continuation_text, iter_created=args.iteration,
        ))
    for ex in sft:
        ex.validate()

    dpo = _build_dpo_pairs(kept, it_dir, eos, args.iteration)

    SFTExample.dump_jsonl(out_dir / "examples_sft.jsonl", sft)
    DPOExample.dump_jsonl(out_dir / "examples_dpo.jsonl", dpo)

    # Merge with prior iterations if accumulating.
    sft_all = _merge(args.run_dir, args.iteration, "examples_sft.jsonl", sft, SFTExample,
                     accumulate=cfg.data.accumulate)
    dpo_all = _merge(args.run_dir, args.iteration, "examples_dpo.jsonl", dpo, DPOExample,
                     accumulate=cfg.data.accumulate)
    n_sft = SFTExample.dump_jsonl(train_sft_path, sft_all)
    n_dpo = DPOExample.dump_jsonl(out_dir / "train_dpo.jsonl", dpo_all)

    stats = {
        "iter": args.iteration,
        "sft_this_iter": len(sft),
        "sft_by_source": dict(Counter(e.source for e in sft_all)),
        "sft_total": n_sft,
        "dpo_total": n_dpo,
        "mean_len": round(sum(len(e.input_ids) for e in sft_all) / n_sft, 1) if n_sft else 0,
    }
    write_json(out_dir / "stats.json", stats)
    mark_done(train_sft_path, count=n_sft, config_hash=cfg.hash(), extra=stats)
    mark_done(out_dir / "train_dpo.jsonl", count=n_dpo, config_hash=cfg.hash())
    print(f"[build_dataset] {stats}")


def _build_dpo_pairs(kept: list[ImprovedCandidate], it_dir: Path, eos: int, iteration: int):
    """chosen = improved continuation; rejected = the base failed rollout's own
    suffix after the SAME anchor — a textbook preference pair sharing its prompt."""
    need = {(c.qid, c.base_sample_idx) for c in kept}
    base: dict[tuple[str, int], RolloutSample] = {}
    if need:
        for s in RolloutSample.load_jsonl(it_dir / "rollout" / "rollouts.jsonl"):
            if (s.qid, s.sample_idx) in need:
                base[(s.qid, s.sample_idx)] = s
    pairs: list[DPOExample] = []
    for c in kept:
        b = base.get((c.qid, c.base_sample_idx))
        if b is None:
            continue
        rejected = b.response_token_ids[len(c.anchor_token_ids):]
        if not rejected:
            continue  # anchor covered the whole failed response
        pairs.append(DPOExample(
            uid=stable_hash("dpo", c.qid, c.base_sample_idx, c.attempt_idx, iteration),
            qid=c.qid,
            prompt_token_ids=c.prompt_token_ids + c.anchor_token_ids,
            chosen_token_ids=ensure_eos(c.continuation_token_ids, eos),
            rejected_token_ids=ensure_eos(rejected, eos) if b.finish_reason == "stop" else rejected,
            chosen_text=c.continuation_text,
            rejected_text=b.response_text,
            iter_created=iteration,
        ))
    return pairs


def _merge(run_dir, iteration: int, filename: str, current: list, record_cls, *, accumulate: bool):
    if not accumulate:
        return current
    merged: dict[str, object] = {}
    for it in range(iteration):
        prior = iter_dir(run_dir, it) / "dataset" / filename
        if prior.exists():
            for r in record_cls.load_jsonl(prior):
                merged[_content_key(r)] = r
    for r in current:
        merged[_content_key(r)] = r
    return list(merged.values())


def _content_key(r) -> str:
    input_ids = getattr(r, "input_ids", None)
    if input_ids is not None:
        return stable_hash("sft", r.qid, tuple(input_ids))
    return stable_hash(
        "dpo", r.qid,
        tuple(r.prompt_token_ids),
        tuple(r.chosen_token_ids),
        tuple(r.rejected_token_ids),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
