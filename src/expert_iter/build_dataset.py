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

Cliff-objective additions (train.sft.cliff — docs/objective_decision_20260823.md §3):
improved rows carry n_q (kept rescues per question, restamped on the merged set
every iteration) and ref_mean_nll (the C(y) pass's s_mean, displacement-guard
reference); negative.mode=v1 adds source="negative" rows = the base policy's
modal-wrong failures with any trailing EOS STRIPPED — unlikelihood on the
stop token suppresses termination and blows up generation length; train.dpo.rejected_selection=modal_wrong switches the DPO pairs'
rejected to the modal-wrong rollout (the v0 arm).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from .config import load_stage_config, stage_argparser
from .records import DPOExample, ImprovedCandidate, RolloutSample, SFTExample, VerdictRecord
from .records import load_any
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

    # Solved-only arms (RFT/ReST-EM) drop anchor/improve/filters from
    # loop.stages, so there is no kept.jsonl to join — 0 improved rows,
    # not an error.
    kept_path = it_dir / "filtered" / "kept.jsonl"
    kept = list(ImprovedCandidate.load_jsonl(kept_path)) if kept_path.exists() else []
    cliff = cfg.train.sft.cliff
    # Displacement-guard reference: the C(y) pass's s_mean, keyed like
    # filters._cand_key ("{qid}:{base_sample_idx}:{attempt_idx}").
    refs: dict[str, float | None] = {}
    if cliff.enabled and cliff.guard.enabled:
        scores_path = it_dir / "filtered" / "candidate_scores.jsonl"
        if not scores_path.exists():
            raise RuntimeError(
                f"train.sft.cliff.guard needs {scores_path} — run filters with "
                "filter.selection.method=c_score or always_score=true"
            )
        for row in load_any(scores_path):
            refs[row["key"]] = row.get("s_mean")
    n_ref_joined = n_ref_missing = 0
    for c in kept:
        completion = ensure_eos(c.continuation_token_ids, eos)
        ref = refs.get(f"{c.qid}:{c.base_sample_idx}:{c.attempt_idx}")
        if refs:
            n_ref_joined += ref is not None
            n_ref_missing += ref is None
        sft.append(SFTExample(
            uid=stable_hash("improved", c.qid, c.base_sample_idx, c.attempt_idx, args.iteration),
            qid=c.qid, source="improved",
            input_ids=training_input_ids(c.prompt_token_ids, c.anchor_token_ids, completion),
            prompt_len=len(c.prompt_token_ids), anchor_len=len(c.anchor_token_ids),
            completion_len=len(completion),
            text=c.continuation_text, iter_created=args.iteration,
            ref_mean_nll=ref,
        ))

    # Modal-wrong base failures (the attractor): needed by v1 negatives and by
    # the v0 DPO rejected_selection. Computed only when some consumer wants it.
    neg_mode = cliff.negative.mode
    want_modal = neg_mode == "v1" or cfg.train.dpo.rejected_selection == "modal_wrong"
    modal_by_qid: dict[str, list[RolloutSample]] = {}
    if want_modal:
        cliff_qids = {c.qid for c in kept}
        modal_by_qid = _modal_wrong_failures(it_dir, cliff_qids)

    n_negative = 0
    if neg_mode == "v1":
        for qid in sorted({c.qid for c in kept}):
            for b in modal_by_qid.get(qid, [])[: cliff.negative.max_per_question]:
                # EOS handling (docs/objective_loss_spec_20260825.md §1):
                # vLLM already puts the stop token in response_token_ids for
                # finish_reason=="stop", so NOT calling ensure_eos does not keep
                # EOS out of L_N — it must be dropped explicitly. Default keeps
                # it (termination is part of the attractor's commitment);
                # drop_terminal_eos=true is the paired ablation leg, motivated by
                # the 2026-08-27 S4-v1 measurement (+57% mean generation length,
                # p90 at the max_tokens cap, 4x truncation on held-out cliffs).
                resp = list(b.response_token_ids)
                if cliff.negative.drop_terminal_eos:
                    while resp and resp[-1] == eos:
                        resp.pop()
                if not resp:
                    continue
                sft.append(SFTExample(
                    uid=stable_hash("negative", qid, b.sample_idx, args.iteration),
                    qid=qid, source="negative",
                    input_ids=training_input_ids(b.prompt_token_ids, [], resp),
                    prompt_len=len(b.prompt_token_ids), anchor_len=0,
                    completion_len=len(resp),
                    text=b.response_text, iter_created=args.iteration,
                ))
                n_negative += 1
    for ex in sft:
        ex.validate()

    dpo, dpo_stats = _build_dpo_pairs(
        kept, it_dir, eos, args.iteration,
        modal_by_qid=modal_by_qid if cfg.train.dpo.rejected_selection == "modal_wrong" else None,
    )

    SFTExample.dump_jsonl(out_dir / "examples_sft.jsonl", sft)
    DPOExample.dump_jsonl(out_dir / "examples_dpo.jsonl", dpo)

    # Merge with prior iterations if accumulating.
    sft_all = _merge(args.run_dir, args.iteration, "examples_sft.jsonl", sft, SFTExample,
                     accumulate=cfg.data.accumulate)
    dpo_all = _merge(args.run_dir, args.iteration, "examples_dpo.jsonl", dpo, DPOExample,
                     accumulate=cfg.data.accumulate)
    excluded = _load_excluded_qids(cfg.data.exclude_train_qids)
    if excluded:
        n_sft_before, n_dpo_before = len(sft_all), len(dpo_all)
        sft_all = [e for e in sft_all if e.qid not in excluded]
        dpo_all = [e for e in dpo_all if e.qid not in excluded]
        print(f"[build_dataset] excluded {n_sft_before - len(sft_all)} sft / "
              f"{n_dpo_before - len(dpo_all)} dpo examples over {len(excluded)} held-out qids")
    # Per-question masses for the cliff term's per-question normalizer.
    # Restamped on the MERGED set every iteration so accumulated rows (including
    # rows written by pre-cliff code, which deserialize with n_q=0) always carry
    # this iteration's counts. Content-keyed dedup is unaffected (metadata only).
    _stamp_n_q(sft)
    _stamp_n_q(sft_all)
    n_sft = SFTExample.dump_jsonl(train_sft_path, sft_all)
    n_dpo = DPOExample.dump_jsonl(out_dir / "train_dpo.jsonl", dpo_all)

    n_q_hist = Counter(
        e.n_q for e in sft_all if e.source == "improved"
    )
    stats = {
        "iter": args.iteration,
        "sft_this_iter": len(sft),
        "sft_by_source": dict(Counter(e.source for e in sft_all)),
        "sft_total": n_sft,
        "dpo_total": n_dpo,
        "mean_len": round(sum(len(e.input_ids) for e in sft_all) / n_sft, 1) if n_sft else 0,
        "n_negative_this_iter": n_negative,
        "n_questions_with_negatives": len({e.qid for e in sft_all if e.source == "negative"}),
        "n_q_hist": {str(k): v for k, v in sorted(n_q_hist.items())},
        "n_ref_joined": n_ref_joined,
        "n_ref_missing": n_ref_missing,
        "n_excluded_qids": len(excluded),
        **dpo_stats,
    }
    write_json(out_dir / "stats.json", stats)
    mark_done(train_sft_path, count=n_sft, config_hash=cfg.hash(), extra=stats)
    mark_done(out_dir / "train_dpo.jsonl", count=n_dpo, config_hash=cfg.hash())
    print(f"[build_dataset] {stats}")


def _build_dpo_pairs(
    kept: list[ImprovedCandidate], it_dir: Path, eos: int, iteration: int,
    *, modal_by_qid: dict[str, list[RolloutSample]] | None = None,
):
    """chosen = improved continuation; rejected = a base failed rollout.

    rejected_selection=base_pick (modal_by_qid None): the anchor stage's
    base_sample_idx rollout, sliced after the SAME anchor — the legacy pair.
    rejected_selection=modal_wrong: the rollout carrying the question's modal
    wrong answer (the attractor). When that is a different sample than
    base_sample_idx the anchor is not its prefix, so the FULL response becomes
    rejected under the same prompt+anchor prompt (a sequence-level negative;
    moot under anchor.policy=none where anchors are empty). Questions without a
    valid modal-wrong failure fall back to base_pick and are counted.
    """
    need = {(c.qid, c.base_sample_idx) for c in kept}
    base: dict[tuple[str, int], RolloutSample] = {}
    if need:
        for s in RolloutSample.load_jsonl(it_dir / "rollout" / "rollouts.jsonl"):
            if (s.qid, s.sample_idx) in need:
                base[(s.qid, s.sample_idx)] = s
    pairs: list[DPOExample] = []
    n_modal = n_fallback = 0
    for c in kept:
        b = None
        if modal_by_qid is not None:
            modal = modal_by_qid.get(c.qid)
            if modal:
                b = modal[0]
                n_modal += 1
            else:
                n_fallback += 1
        if b is None:
            b = base.get((c.qid, c.base_sample_idx))
        if b is None:
            continue
        if b.sample_idx == c.base_sample_idx:
            rejected = b.response_token_ids[len(c.anchor_token_ids):]
        else:
            rejected = b.response_token_ids
        if not rejected:
            continue  # anchor covered the whole failed response
        pairs.append(DPOExample(
            uid=stable_hash("dpo", c.qid, b.sample_idx, c.attempt_idx, iteration),
            qid=c.qid,
            prompt_token_ids=c.prompt_token_ids + c.anchor_token_ids,
            chosen_token_ids=ensure_eos(c.continuation_token_ids, eos),
            rejected_token_ids=ensure_eos(rejected, eos) if b.finish_reason == "stop" else rejected,
            chosen_text=c.continuation_text,
            rejected_text=b.response_text,
            iter_created=iteration,
        ))
    stats = {"dpo_rejected_modal_wrong": n_modal, "dpo_rejected_fallback": n_fallback} \
        if modal_by_qid is not None else {}
    return pairs, stats


def _modal_wrong_failures(it_dir: Path, qids: set[str]) -> dict[str, list[RolloutSample]]:
    """The attractor sample set: per question, the clean (finish_reason=="stop")
    incorrect base rollouts whose extracted answer equals the MODAL wrong answer
    (exact-string grouping over partition/verdicts.jsonl; None excluded; ties
    broken deterministically by (-count, answer)). Sorted by sample_idx."""
    verdicts: dict[tuple[str, int], VerdictRecord] = {}
    for v in VerdictRecord.load_jsonl(it_dir / "partition" / "verdicts.jsonl"):
        if v.qid in qids:
            verdicts[(v.qid, v.sample_idx)] = v
    by_qid: dict[str, dict[str, list[RolloutSample]]] = {}
    for s in RolloutSample.load_jsonl(it_dir / "rollout" / "rollouts.jsonl"):
        if s.qid not in qids or s.finish_reason != "stop":
            continue
        v = verdicts.get((s.qid, s.sample_idx))
        if v is None or v.correct or v.extracted_answer is None:
            continue
        by_qid.setdefault(s.qid, {}).setdefault(v.extracted_answer, []).append(s)
    out: dict[str, list[RolloutSample]] = {}
    for qid, groups in by_qid.items():
        modal_answer = min(groups, key=lambda a: (-len(groups[a]), a))
        out[qid] = sorted(groups[modal_answer], key=lambda s: s.sample_idx)
    return out


def _load_excluded_qids(path: str) -> set[str]:
    """qids held out of training (the B cliff set). Empty path = no exclusion;
    a SET but MISSING file is a hard error (silently training on B would
    invalidate the transfer experiment)."""
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise RuntimeError(
            f"data.exclude_train_qids={path} does not exist — generate it with "
            "scripts/cliff_split.py (or clear the field)"
        )
    import json as _json
    d = _json.loads(p.read_text())
    if isinstance(d, dict):
        d = d.get("exclude", d.get("B"))
        if d is None:
            raise RuntimeError(f"{path}: dict must carry an 'exclude' or 'B' key")
    if not isinstance(d, list) or not all(isinstance(q, str) for q in d):
        raise RuntimeError(f"{path}: expected a JSON list of qid strings")
    return set(d)


def _stamp_n_q(rows: list[SFTExample]) -> None:
    """n_q = per-qid row count within each cliff slice (improved / negative)."""
    counts = Counter((e.qid, e.source) for e in rows if e.source in ("improved", "negative"))
    for e in rows:
        if e.source in ("improved", "negative"):
            e.n_q = counts[(e.qid, e.source)]


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
