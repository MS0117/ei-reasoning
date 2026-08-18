"""Stage: anchor — ⚗ extension point (I): choose the anchor prefix of a failed
rollout that we commit to keeping.

An AnchorPolicy proposes, per unsolved question, an anchor that is ALWAYS an
id-slice of one failed rollout's response_token_ids — never re-tokenized text.
Two hooks:

  select_len(question, failed, params) -> int
      legacy per-sample hook: number of leading response tokens to keep
      (0 => resample from scratch). Enough for cheap policies.

  propose_batch(questions, ctx) -> list[AnchorRecord]
      batch hook with full context (tokenizer, engine access via ctx paths,
      gold solutions). The default implementation delegates to select_len on
      ctx.selected — override it for policies that need scoring passes.

Every routed question gets EXACTLY ONE AnchorRecord; degenerate cases emit
anchor_len == 0 with meta.reason so downstream consumers stay uniform
(anchor_len 0 == x-only improvement).

Failed-rollout selection (methodology G4) is a stage-level concern shared by
all policies: anchor.base_selection ∈ first_failed | longest | random |
min_mean_nll (the most policy-typical failure — one batched score pass).

Registering a new policy:

    @register(ANCHOR_POLICIES, "logprob_dip")
    class LogprobDipAnchor(AnchorPolicy):
        def select_len(self, question, failed, params): ...
"""

from __future__ import annotations

import math
import random
import sys
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, load_stage_config, stage_argparser
from .data import load_gold_solutions
from .records import AnchorRecord, RolloutSample, UnsolvedQuestion
from .registry import ANCHOR_POLICIES, build, register
from .steps import select_boundary, step_token_bounds
from .templates import render_privileged_prompt
from .utils import is_done, iter_dir, mark_done


@dataclass
class AnchorContext:
    """Everything a batch-level policy may need. Engine passes should write
    their pool dirs under `work_dir`."""

    cfg: Config
    tokenizer: Any
    model_path: str
    work_dir: Path
    iteration: int
    gold_solutions: dict[str, str]                  # qid -> y* text (may be empty)
    selected: dict[str, RolloutSample]              # G4 base-selection result per qid
    failed_by_qid: dict[str, list[RolloutSample]] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)


class AnchorPolicy(ABC):
    name: str

    def select_len(self, question: UnsolvedQuestion, failed: RolloutSample, params: dict) -> int:
        """Anchor length in response tokens (0 => resample). Override this OR
        propose_batch."""
        raise NotImplementedError(f"{type(self).__name__} implements neither hook")

    def propose_batch(
        self, questions: list[UnsolvedQuestion], ctx: AnchorContext
    ) -> list[AnchorRecord]:
        rows: list[AnchorRecord] = []
        for u in questions:
            failed = ctx.selected.get(u.qid)
            if failed is None:
                continue
            alen = self.select_len(u, failed, ctx.cfg.anchor.params)
            rows.append(_make_record(u.qid, failed, self.name, alen, ctx, meta={}))
        return rows


@register(ANCHOR_POLICIES, "fixed_fraction")
class FixedFractionAnchor(AnchorPolicy):
    """Keep the first `fraction` of the failed response, clamped to
    [min_tokens, max_tokens]. The simplest possible baseline."""

    def select_len(self, question: UnsolvedQuestion, failed: RolloutSample, params: dict) -> int:
        n = len(failed.response_token_ids)
        frac = float(params.get("fraction", 0.3))
        lo = int(params.get("min_tokens", 32))
        hi = int(params.get("max_tokens", 2048))
        return max(0, min(n, max(lo, min(hi, round(frac * n)))))


@register(ANCHOR_POLICIES, "none")
class NoAnchor(AnchorPolicy):
    """Empty anchor: the improvement operator resamples the whole response.
    Useful as the rejection-sampling / STaR ablation baseline."""

    def select_len(self, question, failed, params) -> int:
        return 0


def shuffled_reference_map(gold_solutions: dict[str, str]) -> dict[str, str]:
    """qid -> ANOTHER question's gold solution, paired by adjacent length.

    The control context for `matched_minus_shuffled`. Sorting by reference
    length and pairing neighbours cyclically keeps the two contexts comparable
    in size (a much longer or shorter reference would itself shift logprobs),
    is deterministic, and never pairs a question with itself."""
    order = sorted(gold_solutions, key=lambda q: (len(gold_solutions[q]), q))
    if len(order) < 2:
        return {}
    return {q: gold_solutions[order[(i + 1) % len(order)]]
            for i, q in enumerate(order)}


@register(ANCHOR_POLICIES, "privileged_divergence")
class PrivilegedDivergenceAnchor(AnchorPolicy):
    """Cut where the privileged context stops endorsing the failed rollout.

    Two teacher-forcing passes over y- with the SAME policy: privileged
    context = prompt(x, y*) (render_privileged_prompt) and base context =
    prompt(x) (the rollout's stored prompt ids). Per-position signal g_t:

      realized_logratio (default): g_t = lp_base(y-_t) - lp_priv(y-_t) — rises
          where knowing the reference solution LOWERS the probability of the
          failed rollout's own next token, i.e. at the first
          reference-detectable wrong turn. (Sign convention: verify on a real
          example during the GPU smoke before relying on it.)
      topk_kl: KL(q_priv || pi_base) approximated over the union of both
          top-K token sets (missing entries floored at that side's K-th
          logprob, then renormalized). Needs engine.max_logprobs >= top_k.
      matched_minus_shuffled: same subtraction, but the CONTROL pass also sees
          a reference — another question's, length-matched (see
          shuffled_reference_map). realized_logratio's control is prompt(x),
          so its signal conflates "the reference disagrees with this step" with
          "a reference was shown at all"; the latter fires hardest on a
          response's stylistic opening and dominated real runs (a correct
          preamble scored 1.03 vs a 0.26 max elsewhere, cutting at 4% of the
          response). With a reference on BOTH sides the format term cancels and
          only problem-specific disagreement survives. Same cost: two passes.

    Aggregated to step means over steps fully inside the first
    ceil(search_frac * L) tokens; boundary selection per steps.select_boundary
    (mu + c_sigma*sigma threshold, discrete-derivative fallback,
    j_a clamped to [min_steps, ceil(max_frac * J)]).

    params: signal=realized_logratio|topk_kl, top_k=100, c_sigma=2.0,
            search_frac=0.333, min_steps=2, max_frac=0.8.
    """

    def propose_batch(self, questions, ctx: AnchorContext) -> list[AnchorRecord]:
        from .engine import GenRequest, run_pool

        cfg = ctx.cfg
        p = cfg.anchor.params
        signal = p.get("signal", "realized_logratio")
        top_k = int(p.get("top_k", 100))
        c_sigma = float(p.get("c_sigma", 2.0))
        search_frac = float(p.get("search_frac", 0.333))
        min_steps = int(p.get("min_steps", 2))
        max_frac = float(p.get("max_frac", 0.8))

        if questions and not ctx.gold_solutions:
            raise RuntimeError(
                "privileged_divergence found no gold solutions in the frozen "
                "questions file; set data.adapter_args.include_solution=true or "
                "backfill with scripts/backfill_gold_solutions.py"
            )

        rows: list[AnchorRecord] = []
        tasks: list[dict] = []
        shuffled = (shuffled_reference_map(ctx.gold_solutions)
                    if signal == "matched_minus_shuffled" else {})
        for u in questions:
            failed = ctx.selected.get(u.qid)
            if failed is None:
                continue
            gold = ctx.gold_solutions.get(u.qid)
            if not gold:
                rows.append(_make_record(u.qid, failed, self.name, 0, ctx,
                                         meta={"reason": "no_gold"}))
                continue
            ids = failed.response_token_ids
            bounds = step_token_bounds(ids, ctx.tokenizer)
            cap = max(1, math.ceil(search_frac * len(ids)))
            searched = [b for b in bounds if b <= cap]
            if len(bounds) < 3 or len(searched) < 3:
                rows.append(_make_record(
                    u.qid, failed, self.name, 0, ctx,
                    meta={"reason": "too_few_steps", "n_steps": len(bounds),
                          "n_searched_steps": len(searched), "search_end_tok": cap},
                ))
                continue
            priv = render_privileged_prompt(
                ctx.tokenizer, u.question, gold,
                system_prompt=cfg.model.system_prompt,
                question_suffix=cfg.data.question_suffix,
                chat_template_kwargs=cfg.model.chat_template_kwargs,
            )
            # Control context. base = prompt(x): its difference from the
            # privileged pass mixes "the reference disagrees with this
            # reasoning" with "a reference was shown AT ALL", and the second
            # term dominates the opening of a response — a purely stylistic
            # preamble scored a g_bar of 1.03 against a 0.26 rest-of-response
            # max, so the cut landed at 4% of the response (2026-08-16).
            # matched_minus_shuffled shows a DIFFERENT question's reference
            # instead, so both contexts expect reference-style output and the
            # format term cancels, leaving only problem-specific disagreement.
            if signal == "matched_minus_shuffled" and u.qid in shuffled:
                other = shuffled[u.qid]
                ctrl = render_privileged_prompt(
                    ctx.tokenizer, u.question, other,
                    system_prompt=cfg.model.system_prompt,
                    question_suffix=cfg.data.question_suffix,
                    chat_template_kwargs=cfg.model.chat_template_kwargs,
                ).token_ids
            else:
                ctrl = list(failed.prompt_token_ids)
            tasks.append(dict(u=u, failed=failed, bounds=bounds, searched=searched,
                              cap=cap, priv_ids=priv.token_ids, ctrl_ids=ctrl))
        if not tasks:
            return rows

        sampling: dict = {"return_token_logprobs": True}
        if signal == "topk_kl":
            sampling["prompt_logprobs_k"] = top_k
        reqs = []
        for t in tasks:
            prefix = t["failed"].response_token_ids[:t["cap"]]
            reqs.append(GenRequest(
                rid=f"{t['u'].qid}:priv",
                prompt_token_ids=list(t["priv_ids"]) + prefix,
                score_from=len(t["priv_ids"]),
            ))
            reqs.append(GenRequest(
                rid=f"{t['u'].qid}:base",
                prompt_token_ids=list(t["ctrl_ids"]) + prefix,
                score_from=len(t["ctrl_ids"]),
            ))
        results = run_pool(
            reqs, mode="score", model_path=ctx.model_path, sampling=sampling,
            engine_cfg=cfg.engine, work_dir=ctx.work_dir / "pool_divergence",
            dtype=cfg.model.dtype,
        )
        by_rid = {r.rid: r for r in results}

        for t in tasks:
            u, failed = t["u"], t["failed"]
            r_priv, r_base = by_rid[f"{u.qid}:priv"], by_rid[f"{u.qid}:base"]
            g = _position_signal(signal, r_priv, r_base, n_positions=t["cap"])
            g_bar = _step_means(g, t["searched"])
            max_step = math.ceil(max_frac * len(t["bounds"]))
            j_a, sel_meta = select_boundary(
                g_bar, c_sigma=c_sigma, min_steps=min_steps, max_step=max_step,
            )
            meta = {
                "signal": signal,
                "search_end_tok": t["cap"],
                "n_steps": len(t["bounds"]),
                "step_bounds_searched": t["searched"],
                "g_bar": [round(v, 4) for v in g_bar],
                **sel_meta,
            }
            if j_a is None:
                rows.append(_make_record(u.qid, failed, self.name, 0, ctx, meta=meta))
                continue
            # j_a may exceed the searched window after the min_steps clamp-up;
            # step ends beyond it fall back to the full bounds list.
            all_bounds = t["bounds"]
            alen = all_bounds[min(j_a, len(all_bounds)) - 1]
            rows.append(_make_record(u.qid, failed, self.name, alen, ctx, meta=meta))
        return rows


def _position_signal(signal: str, r_priv, r_base, *, n_positions: int) -> list[float | None]:
    """Per-position divergence over the y- prefix; None where either pass has
    no entry. Arrays align by construction: index i <-> y- token i."""
    # Both realized_logratio and matched_minus_shuffled use the SAME
    # subtraction (control minus matched) — only the control PROMPT differs, and
    # that is decided when the requests are built. Anything else falls through
    # to the top-K branch, which needs prompt_logprobs_k to have been requested.
    if signal in ("realized_logratio", "matched_minus_shuffled"):
        lp_p = r_priv.token_logprobs or []
        lp_b = r_base.token_logprobs or []
        out: list[float | None] = []
        for i in range(n_positions):
            a = lp_b[i] if i < len(lp_b) else None
            b = lp_p[i] if i < len(lp_p) else None
            out.append(a - b if a is not None and b is not None else None)
        return out
    # topk_kl: KL(q_priv || pi_base) over the union of both top-K sets.
    tk_p = r_priv.topk_logprobs or []
    tk_b = r_base.topk_logprobs or []
    out = []
    for i in range(n_positions):
        dp = tk_p[i] if i < len(tk_p) else None
        db = tk_b[i] if i < len(tk_b) else None
        out.append(_union_topk_kl(dp, db) if dp and db else None)
    return out


def _union_topk_kl(dp: dict[str, float], db: dict[str, float]) -> float:
    """Approximate KL(p || q) restricted to the union of the two top-K sets.
    A token missing from one side is floored at that side's minimum (K-th)
    logprob — an upper bound on its true logprob — then both are renormalized
    over the union."""
    union = set(dp) | set(db)
    floor_p = min(dp.values())
    floor_q = min(db.values())
    lp = [dp.get(t, floor_p) for t in union]
    lq = [db.get(t, floor_q) for t in union]

    def _norm(logits: list[float]) -> list[float]:
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        z = sum(exps)
        return [e / z for e in exps]

    p = _norm(lp)
    q = _norm(lq)
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0)


def _step_means(g: list[float | None], searched_bounds: list[int]) -> list[float]:
    """Mean signal per step u_j = positions [b_{j-1}, b_j); steps with no valid
    positions contribute 0.0."""
    out: list[float] = []
    prev = 0
    for b in searched_bounds:
        vals = [v for v in g[prev:b] if v is not None]
        out.append(sum(vals) / len(vals) if vals else 0.0)
        prev = b
    return out


def _make_record(
    qid: str, failed: RolloutSample, policy: str, alen: int,
    ctx: AnchorContext, *, meta: dict,
) -> AnchorRecord:
    anchor_ids = failed.response_token_ids[:alen]
    return AnchorRecord(
        qid=qid,
        base_sample_idx=failed.sample_idx,
        policy=policy,
        anchor_token_ids=anchor_ids,
        anchor_text=ctx.tokenizer.decode(anchor_ids),
        anchor_len=len(anchor_ids),
        base_response_len=len(failed.response_token_ids),
        meta={"base_selection": ctx.cfg.anchor.base_selection, **meta},
        iter=ctx.iteration,
    )


def main(argv: list[str] | None = None) -> None:
    args = stage_argparser("EI anchor stage").parse_args(argv)
    cfg = load_stage_config(args)
    it_dir = iter_dir(args.run_dir, args.iteration)
    out_path = it_dir / "anchors" / "anchors.jsonl"
    if is_done(out_path, config_hash=cfg.hash()):
        print(f"[anchor] {out_path} already done, skipping")
        return

    unsolved = list(UnsolvedQuestion.load_jsonl(it_dir / "partition" / "unsolved.jsonl"))
    rollouts: dict[str, dict[int, RolloutSample]] = defaultdict(dict)
    unsolved_qids = {u.qid for u in unsolved}
    for s in RolloutSample.load_jsonl(it_dir / "rollout" / "rollouts.jsonl"):
        if s.qid in unsolved_qids:
            rollouts[s.qid][s.sample_idx] = s

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    policy: AnchorPolicy = build(ANCHOR_POLICIES, cfg.anchor.policy)
    rng = random.Random(cfg.run.seed * 7919 + args.iteration)

    failed_by_qid = {
        u.qid: [rollouts[u.qid][i] for i in u.failed_sample_idxs if i in rollouts[u.qid]]
        for u in unsolved
    }
    selected = _select_bases(failed_by_qid, cfg, args.model_path,
                             it_dir / "anchors", rng)

    ctx = AnchorContext(
        cfg=cfg,
        tokenizer=tokenizer,
        model_path=args.model_path,
        work_dir=it_dir / "anchors",
        iteration=args.iteration,
        gold_solutions=load_gold_solutions(args.run_dir),
        selected=selected,
        failed_by_qid=failed_by_qid,
        rng=rng,
    )
    rows = policy.propose_batch(unsolved, ctx)

    n = AnchorRecord.dump_jsonl(out_path, rows)
    mark_done(out_path, count=n, config_hash=cfg.hash())
    lens = [r.anchor_len for r in rows if r.anchor_len > 0]
    n_zero = sum(1 for r in rows if r.anchor_len == 0)
    print(
        f"[anchor] wrote {n} anchors (policy={cfg.anchor.policy}, "
        + (f"mean_len={sum(lens) / len(lens):.0f}, " if lens else "")
        + f"empty={n_zero})"
    )


def _select_bases(
    failed_by_qid: dict[str, list[RolloutSample]],
    cfg: Config,
    model_path: str,
    work_dir: Path,
    rng: random.Random,
) -> dict[str, RolloutSample]:
    """G4: pick the failed rollout to anchor on, per question."""
    how = cfg.anchor.base_selection
    if how == "min_mean_nll":
        return _select_min_mean_nll(failed_by_qid, cfg, model_path, work_dir)
    out: dict[str, RolloutSample] = {}
    for qid, failed in failed_by_qid.items():
        pick = _pick_base(failed, how, rng)
        if pick is not None:
            out[qid] = pick
    return out


def _select_min_mean_nll(
    failed_by_qid: dict[str, list[RolloutSample]],
    cfg: Config,
    model_path: str,
    work_dir: Path,
) -> dict[str, RolloutSample]:
    """The failure the policy finds most typical: max mean logprob == min mean
    NLL under pi_theta, one batched score pass; ties break to the lowest
    sample_idx, zero-length responses are skipped (fallback first_failed)."""
    from .engine import GenRequest, run_pool

    items: list[tuple[str, RolloutSample]] = [
        (qid, s)
        for qid, failed in sorted(failed_by_qid.items())
        for s in failed
        if s.response_token_ids
    ]
    if not items:
        return {}
    reqs = [
        GenRequest(
            rid=f"{qid}:{s.sample_idx}",
            prompt_token_ids=list(s.prompt_token_ids) + list(s.response_token_ids),
            score_from=len(s.prompt_token_ids),
        )
        for qid, s in items
    ]
    results = run_pool(
        reqs, mode="score", model_path=model_path, sampling={},
        engine_cfg=cfg.engine, work_dir=work_dir / "pool_base_nll",
        dtype=cfg.model.dtype,
    )
    best: dict[str, tuple[float, int, RolloutSample]] = {}
    for (qid, s), r in zip(items, results):
        if r.mean_logprob is None:
            continue
        cand = (r.mean_logprob, -s.sample_idx, s)
        if qid not in best or cand[:2] > best[qid][:2]:
            best[qid] = cand
    out = {qid: t[2] for qid, t in best.items()}
    for qid, failed in failed_by_qid.items():  # fallback for all-unscorable questions
        if qid not in out and failed:
            out[qid] = _pick_base(failed, "first_failed", random.Random(0))
    return out


def _pick_base(failed: list[RolloutSample], how: str, rng: random.Random) -> RolloutSample | None:
    if not failed:
        return None
    if how == "longest":
        return max(failed, key=lambda s: len(s.response_token_ids))
    if how == "random":
        return rng.choice(failed)
    return min(failed, key=lambda s: s.sample_idx)  # first_failed


if __name__ == "__main__":
    main(sys.argv[1:])
