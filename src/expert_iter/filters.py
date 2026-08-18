"""Stage: filters — ⚗ extension point (III): learnability gates over improved
candidates.

A candidate must be (a) correct where the base policy failed and (b) trainable:
nothing in it may depend on information absent at inference time. Pass order:

  1. cheap per-candidate gates in the config-ordered chain (filter.gates);
  2. optional batched scoring passes over the survivors — logprob gate,
     leakage LLM judge;
  3. per-question selection LAST (so kept slots hold fully-vetted candidates):
     either the legacy shortest-first quota or C(y) ranking
     (filter.selection.method=c_score) — C(y) = S_mean + lambda*S_tail +
     gamma*D_tail under the student, keep the top max_per_question by min C.

Outputs under iter_k/filtered/: kept.jsonl (ImprovedCandidate), report.json,
and candidate_scores.jsonl when C(y) selection ran.
"""

from __future__ import annotations

import math
import random
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .config import Config, load_stage_config, stage_argparser
from .records import ImprovedCandidate, QuestionRecord, UnsolvedQuestion
from .registry import GATES, VERIFIERS, build, register
from .utils import (
    is_done,
    iter_dir,
    mark_done,
    read_json,
    stable_hash,
    stable_seed,
    write_json,
    write_jsonl,
)
from . import verifier  # noqa: F401 — imported for its @register side effect on VERIFIERS


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
            full_text = ctx.tokenizer.decode(
                cand.anchor_token_ids + cand.continuation_token_ids
            )
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


@register(GATES, "leakage_rules")
class LeakageRulesGate(Gate):
    """Rule half of the leakage filter (methodology 6c): reject continuations
    that textually presuppose an external reference solution/hint. Patterns
    come from filter.leakage.patterns (regex, case-insensitive). Activated by
    listing "leakage_rules" in filter.gates; NOT in the default list. The
    anchor needs no scanning — it is a slice of the pre-privileged rollout."""

    _compiled: list[re.Pattern] | None = None

    def check(self, cand, ctx):
        if self._compiled is None:
            self._compiled = [
                re.compile(p, re.IGNORECASE) for p in ctx.cfg.filter.leakage.patterns
            ]
        for rx in self._compiled:
            if rx.search(cand.continuation_text):
                return (False, "pattern")
        return (True, "")


def cvar(values: list[float], tail_fraction: float) -> float:
    """CVaR_{f}: mean of the worst (largest) ceil(f*T) values (methodology §CVaR)."""
    if not values:
        raise ValueError("cvar of an empty sequence")
    k = max(1, math.ceil(tail_fraction * len(values)))
    return sum(sorted(values, reverse=True)[:k]) / k


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

    leakage_report: dict = {}
    if cfg.filter.leakage.judge_enabled and survivors:
        survivors, n_flagged, leakage_report = _leakage_judge(
            survivors, cfg, args.model_path, it_dir,
        )
        rejects["leakage_judge:flagged"] = n_flagged

    # C(y) SCORING is deliberately separate from SELECTION: c_score ranks by it,
    # but any method can measure it (filter.selection.always_score) so arms with
    # different selection rules stay comparable on the same yardstick. Survivors
    # here have cleared the gate chain — with the default gates that means
    # verifier-correct — so these statistics describe exactly Y+.
    sel_cfg = cfg.filter.selection
    scores: dict[str, dict] = {}
    selection_report: dict = {"method": sel_cfg.method, "n_survivors": len(survivors)}
    if survivors and (sel_cfg.method == "c_score" or sel_cfg.always_score):
        scores, score_report = _score_candidates(survivors, cfg, args.model_path, it_dir)
        selection_report.update(score_report)

    # Per-question selection runs LAST so the kept slots hold fully-vetted candidates.
    if sel_cfg.method == "c_score" and survivors:
        kept, n_over = _rank_by_c_score(survivors, cfg, scores)
        rejects["selection:over_max_per_question"] = n_over
    elif sel_cfg.method == "random" and survivors:
        kept, n_over = _random_selection(survivors, cfg)
        rejects["selection:over_max_per_question"] = n_over
    else:
        # Quota preferring the shortest total sequence (cheapest to learn,
        # least room for degenerate rambling).
        by_qid: dict[str, list[ImprovedCandidate]] = defaultdict(list)
        for c in survivors:
            by_qid[c.qid].append(c)
        kept = []
        for qid, cands in by_qid.items():
            ranked = sorted(cands, key=lambda c: len(c.anchor_token_ids) + len(c.continuation_token_ids))
            quota = ranked[:cfg.filter.max_per_question]
            rejects["quota:over_max_per_question"] += len(ranked) - len(quota)
            kept.extend(quota)
    if scores:
        _write_candidate_scores(scores, kept, it_dir)

    n = ImprovedCandidate.dump_jsonl(out_path, kept)
    part_stats_path = it_dir / "partition" / "stats.json"
    n_total_questions = (read_json(part_stats_path).get("n_questions")
                         if part_stats_path.exists() else None)
    report = {
        "iter": args.iteration,
        "n_candidates": len(candidates),
        "n_kept": n,
        "n_questions_improved": len({c.qid for c in kept}),
        "n_questions_unsolved": len(unsolved),
        "improve_yield": round(len({c.qid for c in kept}) / len(unsolved), 4) if unsolved else 0.0,
        "rejects": dict(rejects),
        **({"selection": selection_report} if selection_report else {}),
        **({"leakage": leakage_report} if leakage_report else {}),
        **cliff_stats(candidates, unsolved, ctx, n_total_questions),
        # trainability/* metrics (D_mean, D_p95, D_max, per_token_max_deficit —
        # sequence/token-level log-ratio of a trajectory under the current policy
        # vs the policy that generated it) are reserved here but need
        # generation-time logprobs (rollout.capture_logprobs / improve) plus a
        # per-token scoring path in engine "score" mode; add them alongside that
        # machinery, not before.
    }
    write_json(it_dir / "filtered" / "report.json", report)
    mark_done(out_path, count=n, config_hash=cfg.hash(), extra=report)
    print(f"[filters] {report}")


def cliff_stats(candidates: list[ImprovedCandidate], unsolved: dict[str, UnsolvedQuestion],
                ctx: FilterContext, n_total_questions: int | None) -> dict:
    """Cliff = a question whose first rollouts ALL failed — exactly the unsolved
    partition, the population the improvement operator exists to rescue.
    cliff/count should fall across iterations; a flat curve means the loop is
    spinning. Conversion counts correctness ONLY, deliberately ignoring the
    learnability gates: it measures the operator's raw rescue power, while
    improve_yield measures what survives into training data."""
    correct_by_qid: dict[str, int] = {qid: 0 for qid in unsolved}
    gate = CorrectnessGate()
    for cand in candidates:
        if cand.correct is None:  # gate chain short-circuited before correctness
            gate.check(cand, ctx)
        if cand.correct and cand.qid in correct_by_qid:
            correct_by_qid[cand.qid] += 1

    n_cliff = len(unsolved)
    converted = sum(1 for v in correct_by_qid.values() if v > 0)
    stats = {
        "cliff/count": n_cliff,
        "cliff/conversion_rate": round(converted / n_cliff, 4) if n_cliff else 0.0,
        # Raw per-cliff correct-sample counts (out of improve.n * rounds);
        # wandb renders numeric lists as histograms via log_metrics.
        "cliff/conversion_histogram": sorted(correct_by_qid.values()),
    }
    if n_total_questions:
        stats["cliff/ratio"] = round(n_cliff / n_total_questions, 4)
    return stats


def _cand_key(c: ImprovedCandidate) -> str:
    return f"{c.qid}:{c.base_sample_idx}:{c.attempt_idx}"


def _random_selection(survivors, cfg: Config):
    """Uniform random pick of max_per_question survivors per question (the
    survivors already cleared the gate chain — with the default gates that
    means they are verifier-correct). Deterministic per (run.seed, qid), and
    independent of candidate order. Returns (kept, n_over_quota)."""
    by_qid: dict[str, list[ImprovedCandidate]] = defaultdict(list)
    for c in survivors:
        by_qid[c.qid].append(c)
    kept: list[ImprovedCandidate] = []
    n_over = 0
    for qid, cands in by_qid.items():
        ranked = sorted(cands, key=lambda c: (c.base_sample_idx, c.attempt_idx))
        rng = random.Random(stable_seed(cfg.run.seed, "selection", qid))
        picks = rng.sample(ranked, min(cfg.filter.max_per_question, len(ranked)))
        n_over += len(ranked) - len(picks)
        kept.extend(sorted(picks, key=lambda c: (c.base_sample_idx, c.attempt_idx)))
    return kept, n_over


def _score_candidates(survivors, cfg: Config, model_path: str, it_dir, *, pool_dir=None):
    """C(y) scoring (methodology 6b): score every survivor under the student
    pi_theta (S_mean, S_tail) and — when gamma_dtail > 0 — under its generating
    policy q_P via op_meta.lora_path (D_tail), giving
    C = S_mean + lambda*S_tail + gamma*D_tail. Pure MEASUREMENT: selection is a
    separate step, so these numbers are comparable across arms that select
    differently. Returns ({cand_key: {s_mean, s_tail, d_tail, c}}, report)."""
    from .engine import GenRequest, run_pool

    sel = cfg.filter.selection
    tail = sel.tail_fraction
    reqs: list[GenRequest] = []
    n_missing_lora_ref = 0
    for c in survivors:
        seq = c.prompt_token_ids + c.anchor_token_ids + c.continuation_token_ids
        score_from = (len(c.prompt_token_ids) + len(c.anchor_token_ids)
                      if sel.scope == "continuation" else len(c.prompt_token_ids))
        key = _cand_key(c)
        reqs.append(GenRequest(rid=f"{key}:s", prompt_token_ids=seq,
                               score_from=score_from))
        if sel.gamma_dtail > 0:
            lora_path = (c.op_meta or {}).get("lora_path")
            if lora_path:
                reqs.append(GenRequest(rid=f"{key}:d", prompt_token_ids=seq,
                                       score_from=score_from, lora_path=lora_path))
            else:
                n_missing_lora_ref += 1
    results = run_pool(
        reqs, mode="score", model_path=model_path,
        sampling={"return_token_logprobs": True},
        engine_cfg=cfg.engine,
        work_dir=pool_dir if pool_dir is not None else it_dir / "filtered" / "pool_selection",
        dtype=cfg.model.dtype,
    )
    by_rid = {r.rid: r for r in results}

    scored: dict[str, dict] = {}
    for c in survivors:
        key = _cand_key(c)
        lp_s = by_rid[f"{key}:s"].token_logprobs or []
        nll = [-v for v in lp_s if v is not None]
        d_tail = None
        r_d = by_rid.get(f"{key}:d")
        if r_d is not None:
            lp_q = r_d.token_logprobs or []
            deltas = [q - s for q, s in zip(lp_q, lp_s)
                      if q is not None and s is not None]
            if deltas:
                d_tail = cvar(deltas, tail)
        if nll:
            s_mean = sum(nll) / len(nll)
            s_tail = cvar(nll, tail)
            c_score = s_mean + sel.lambda_tail * s_tail
            if d_tail is not None:
                c_score += sel.gamma_dtail * d_tail
        else:  # nothing scorable — deprioritize without crashing
            s_mean = s_tail = None
            c_score = math.inf
        scored[key] = {
            "qid": c.qid, "key": key,
            "s_mean": _r4(s_mean), "s_tail": _r4(s_tail), "d_tail": _r4(d_tail),
            "c": _r4(c_score if c_score != math.inf else None),
            "_c_raw": c_score,          # unrounded, inf for unscorable; not persisted
        }

    report = {
        "lambda_tail": sel.lambda_tail,
        "gamma_dtail": sel.gamma_dtail,
        "tail_fraction": tail,
        "scope": sel.scope,
        "n_scored": len(scored),
        "n_missing_lora_ref": n_missing_lora_ref,
    }
    for metric in ("s_mean", "s_tail", "d_tail", "c"):
        vals = sorted(row[metric] for row in scored.values() if row[metric] is not None)
        if vals:
            report[metric] = {
                "mean": round(sum(vals) / len(vals), 4),
                "p50": _pct(vals, 0.50), "p90": _pct(vals, 0.90),
                "p95": _pct(vals, 0.95), "p99": _pct(vals, 0.99),
                "max": vals[-1],
            }
    return scored, report


def _rank_by_c_score(survivors, cfg: Config, scores: dict[str, dict]):
    """Keep the max_per_question candidates with the SMALLEST C(y) per question
    (ties: shorter first, then attempt order). Returns (kept, n_over_quota)."""
    by_qid: dict[str, list[ImprovedCandidate]] = defaultdict(list)
    for c in survivors:
        by_qid[c.qid].append(c)
    kept: list[ImprovedCandidate] = []
    n_over = 0
    for qid, cands in by_qid.items():
        ranked = sorted(cands, key=lambda c: (
            scores.get(_cand_key(c), {}).get("_c_raw", math.inf),
            len(c.anchor_token_ids) + len(c.continuation_token_ids),
            c.attempt_idx,
        ))
        quota = ranked[:cfg.filter.max_per_question]
        n_over += len(ranked) - len(quota)
        kept.extend(quota)
    return kept, n_over


def _write_candidate_scores(scores: dict[str, dict], kept, it_dir) -> None:
    """Persist per-candidate C(y) rows with the ACTUAL kept flags (whatever
    selection method produced them), so every arm's scores stay comparable."""
    kept_keys = {_cand_key(c) for c in kept}
    rows = []
    for key in sorted(scores):
        row = {k: v for k, v in scores[key].items() if not k.startswith("_")}
        row["kept"] = key in kept_keys
        rows.append(row)
    write_jsonl(it_dir / "filtered" / "candidate_scores.jsonl", rows)


def _r4(v):
    return round(v, 4) if isinstance(v, float) and math.isfinite(v) else (v if v is None else None)


def _pct(sorted_vals: list[float], q: float) -> float:
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[idx]


_JUDGE_PROMPT = (
    "You are auditing a math solution for information leakage. Does the text "
    "below presuppose, mention, or rely on an EXTERNAL reference solution, "
    "provided answer, or hint (e.g. phrases like \"the given solution\", "
    "\"as provided\", \"the reference\", \"according to the hint\")? "
    "Answer with exactly one word: YES or NO.\n\n---\n{body}\n---"
)


def _leakage_judge(survivors, cfg: Config, model_path: str, it_dir, *, pool_dir=None):
    """LLM half of the leakage filter: one batched generation pass asking a
    judge (filter.leakage.judge_model, default = current policy) whether each
    continuation presupposes external material. YES => rejected; anything
    unparsable => conservatively kept and counted. Also reused by bridge_sft
    at bridge-generation time (pool_dir points into the improve stage)."""
    from transformers import AutoTokenizer

    from .engine import GenRequest, run_pool
    from .templates import render_question_prompt

    lk = cfg.filter.leakage
    pool_dir = pool_dir if pool_dir is not None else it_dir / "filtered" / "pool_leakage"
    judge_model = lk.judge_model or model_path
    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    reqs = []
    for c in survivors:
        prompt = render_question_prompt(
            tokenizer, _JUDGE_PROMPT.format(body=c.continuation_text),
            system_prompt=None, question_suffix="",
            chat_template_kwargs=cfg.model.chat_template_kwargs if judge_model == model_path else None,
        )
        reqs.append(GenRequest(rid=_cand_key(c), prompt_token_ids=prompt.token_ids))
    results = run_pool(
        reqs, mode="generate", model_path=judge_model,
        sampling={
            "temperature": lk.judge_temperature,
            "top_p": 1.0,
            "max_tokens": lk.judge_max_tokens,
        },
        engine_cfg=cfg.engine, work_dir=pool_dir,
        dtype=cfg.model.dtype,
    )
    kept, n_flagged, n_unparsed = [], 0, 0
    for c, r in zip(survivors, results):
        text = (r.samples[0]["text"] if r.samples else "").strip().upper()
        verdict = next((w for w in ("YES", "NO") if text.startswith(w)), None)
        if verdict == "YES":
            n_flagged += 1
        else:
            if verdict is None:
                n_unparsed += 1
            kept.append(c)
    report = {"judge_model": judge_model, "n_judged": len(survivors),
              "n_flagged": n_flagged, "n_unparsed_kept": n_unparsed}
    return kept, n_flagged, report


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
        dtype=cfg.model.dtype,
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
