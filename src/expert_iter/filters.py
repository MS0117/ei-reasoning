"""Stage: filters — ⚗ extension point (III): learnability gates over improved
candidates.

A candidate must be (a) correct where the base policy failed and (b) trainable:
nothing in it may depend on information absent at inference time. Cheap gates
run per-candidate in the config-ordered chain; the optional logprob gate runs
as one batched vLLM scoring pass over the survivors; a per-question quota is
applied last.

Outputs under iter_k/filtered/: kept.jsonl (ImprovedCandidate), report.json.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .config import Config, load_stage_config, stage_argparser
from .records import ImprovedCandidate, QuestionRecord, UnsolvedQuestion
from .registry import GATES, VERIFIERS, build, register
from .utils import is_done, iter_dir, mark_done, stable_hash, write_json


@dataclass
class FilterContext:
    cfg: Config
    verifier: object
    tokenizer: object
    questions: dict[str, UnsolvedQuestion]
    seen_hashes: set = field(default_factory=set)


class Gate(ABC):
    name: str

    @abstractmethod
    def check(self, cand: ImprovedCandidate, ctx: FilterContext) -> tuple[bool, str]:
        """Returns (passes, reason). May mutate cand (correctness fills .correct)."""


@register(GATES, "correctness")
class CorrectnessGate(Gate):
    def check(self, cand, ctx):
        if cand.correct is None:
            q = ctx.questions[cand.qid]
            full_text = ctx.tokenizer.decode(cand.anchor_token_ids) + cand.continuation_text
            verdict = ctx.verifier.verify(
                QuestionRecord(qid=q.qid, question=q.question, final_answer=q.final_answer),
                full_text,
            )
            cand.correct = verdict.correct
        return (bool(cand.correct), "incorrect")


@register(GATES, "no_external_context")
class NoExternalContextGate(Gate):
    """Structural learnability: operators must declare any information beyond
    question+anchor in external_context; here it is rejected. Replace/remove
    this gate only together with a training method designed to absorb the
    train/inference mismatch (methodology step 4b)."""

    def check(self, cand, ctx):
        return (cand.external_context is None, "external_context")


@register(GATES, "length")
class LengthGate(Gate):
    def check(self, cand, ctx):
        total = (len(cand.prompt_token_ids) + len(cand.anchor_token_ids)
                 + len(cand.continuation_token_ids))
        return (total <= ctx.cfg.filter.max_total_tokens, "too_long")


@register(GATES, "dedup")
class DedupGate(Gate):
    """Drop near-clones: best-of-n at temperature often yields identical
    continuations; key on (qid, continuation ids)."""

    def check(self, cand, ctx):
        key = stable_hash(cand.qid, tuple(cand.continuation_token_ids))
        if key in ctx.seen_hashes:
            return (False, "duplicate")
        ctx.seen_hashes.add(key)
        return (True, "")


def main(argv: list[str] | None = None) -> None:
    args = stage_argparser("EI filters stage").parse_args(argv)
    cfg = load_stage_config(args)
    it_dir = iter_dir(args.run_dir, args.iteration)
    out_path = it_dir / "filtered" / "kept.jsonl"
    if is_done(out_path, config_hash=cfg.hash()):
        print(f"[filters] {out_path} already done, skipping")
        return

    candidates = list(ImprovedCandidate.load_jsonl(it_dir / "improve" / "improved.jsonl"))
    unsolved = {u.qid: u for u in UnsolvedQuestion.load_jsonl(it_dir / "partition" / "unsolved.jsonl")}

    from transformers import AutoTokenizer

    ctx = FilterContext(
        cfg=cfg,
        verifier=build(VERIFIERS, cfg.partition.verifier),
        tokenizer=AutoTokenizer.from_pretrained(args.model_path),
        questions=unsolved,
    )
    gates: list[Gate] = [build(GATES, name) for name in cfg.filter.gates]

    rejects: Counter = Counter()
    survivors: list[ImprovedCandidate] = []
    for cand in candidates:
        for gate in gates:
            ok, reason = gate.check(cand, ctx)
            if not ok:
                rejects[f"{gate.name}:{reason}"] += 1
                break
        else:
            survivors.append(cand)

    if cfg.filter.logprob_gate.enabled and survivors:
        survivors, n_rej = _logprob_gate(survivors, cfg, args.model_path, it_dir)
        rejects["logprob:below_threshold"] = n_rej

    # Per-question quota, preferring the shortest total sequence (cheapest to
    # learn, least room for degenerate rambling).
    by_qid: dict[str, list[ImprovedCandidate]] = defaultdict(list)
    for c in survivors:
        by_qid[c.qid].append(c)
    kept: list[ImprovedCandidate] = []
    for qid, cands in by_qid.items():
        ranked = sorted(cands, key=lambda c: len(c.anchor_token_ids) + len(c.continuation_token_ids))
        quota = ranked[:cfg.filter.max_per_question]
        rejects["quota:over_max_per_question"] += len(ranked) - len(quota)
        kept.extend(quota)

    n = ImprovedCandidate.dump_jsonl(out_path, kept)
    report = {
        "iter": args.iteration,
        "n_candidates": len(candidates),
        "n_kept": n,
        "n_questions_improved": len({c.qid for c in kept}),
        "n_questions_unsolved": len(unsolved),
        "improve_yield": round(len({c.qid for c in kept}) / len(unsolved), 4) if unsolved else 0.0,
        "rejects": dict(rejects),
    }
    write_json(it_dir / "filtered" / "report.json", report)
    mark_done(out_path, count=n, config_hash=cfg.hash(), extra=report)
    print(f"[filters] {report}")


def _logprob_gate(survivors, cfg: Config, model_path: str, it_dir):
    """Score each surviving trajectory under the current policy; drop those the
    policy finds too improbable to be learnable (or whose probability is
    dominated by information it could not have produced)."""
    from .engine import GenRequest, run_pool

    scope = cfg.filter.logprob_gate.scope
    reqs = []
    for c in survivors:
        seq = c.prompt_token_ids + c.anchor_token_ids + c.continuation_token_ids
        score_from = (len(c.prompt_token_ids) + len(c.anchor_token_ids)
                      if scope == "continuation" else len(c.prompt_token_ids))
        reqs.append(GenRequest(
            rid=f"{c.qid}:{c.base_sample_idx}:{c.attempt_idx}",
            prompt_token_ids=seq, score_from=score_from,
        ))
    results = run_pool(
        reqs, mode="score", model_path=model_path, sampling={},
        engine_cfg=cfg.engine, work_dir=it_dir / "filtered" / "pool",
    )
    thr = cfg.filter.logprob_gate.min_mean_logprob
    kept, n_rej = [], 0
    for c, r in zip(survivors, results):
        if r.mean_logprob is not None and r.mean_logprob >= thr:
            kept.append(c)
        else:
            n_rej += 1
    return kept, n_rej


if __name__ == "__main__":
    main(sys.argv[1:])
