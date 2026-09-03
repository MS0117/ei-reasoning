"""Group-advantage objective (train.objective: gadv) — data side.

Groups and per-row advantages are computed here from the iteration's OWN
rollouts (CPU, no torch); the clipped surrogate and the theta0 pre-pass live in
train.py. Spec: docs/objective_gadv_spec_20260903.md.

Per question q with n base rollouts, k = #clean-correct (verifier correct AND
finish_reason == "stop"; truncated-correct counts as wrong, like partition):
  k == n            excluded, unless train.gadv.solved_floor > 0 (a few correct
                    rows at that constant advantage, no negatives)
  1 <= k <= n-1     frontier: correct rows A = 1 - k/n, wrong rows negative
  k == 0, R >= 1    rescue group: the n failures + the R kept rescues,
                    rescue A = (1 - R/(n+R)) * rescue_dose, failures negative
  k == 0, R == 0    excluded
The negative total of a question is neg_scale x its positive total and is
split across the wrong rows in proportion to f_j ** gamma, f_j = share of the
wrong rows carrying the same extracted answer (None / truncated = singleton).
gamma = 0 reproduces Dr.GRPO's -p on every wrong row. Truncated failures are
capped per question by train.gadv.wrong_truncated_max_per_question BEFORE the
group is planned (n and the negative total are unchanged; 0 drops them).
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .records import ImprovedCandidate, RolloutSample, SFTExample, VerdictRecord
from .templates import ensure_eos, training_input_ids
from .utils import stable_hash, stable_seed

Member = tuple[int, bool, "str | None"]     # (sample_idx, clean_correct, answer_key)


@dataclass
class GroupPlan:
    kind: str                       # frontier | rescue | floor | excluded
    k: int                          # clean-correct count among the base rollouts
    p: float                        # k/n (frontier), R/(n+R) (rescue), 1.0 (floor)
    n: int = 0                      # base rollouts in the question
    n_rescues: int = 0
    pos: dict[int, float] = field(default_factory=dict)          # sample_idx -> A > 0
    neg: dict[int, float] = field(default_factory=dict)          # sample_idx -> A < 0
    rescue_adv: float = 0.0                                      # A of every rescue row
    answer_freq: dict[int, float] = field(default_factory=dict)  # sample_idx -> f_j
    reason: str = ""                                             # why excluded

    @property
    def group_size(self) -> int:
        return self.n + (self.n_rescues if self.kind == "rescue" else 0)

    @property
    def pos_total(self) -> float:
        return sum(self.pos.values()) + (self.n_rescues * self.rescue_adv if self.kind == "rescue" else 0.0)

    @property
    def neg_total(self) -> float:
        return sum(self.neg.values())


def _pick(rng: random.Random, items: list, cap: int) -> list:
    """Deterministic capped selection: everything when it fits, else a seeded
    random subset. Output keeps sample_idx order either way."""
    if len(items) <= cap:
        return list(items)
    chosen = rng.sample(items, cap)
    return sorted(chosen, key=lambda x: x[0] if isinstance(x, tuple) else x)


def group_advantages(members: list[Member], n_rescues: int, cfg, rng: random.Random,
                     n: int | None = None) -> GroupPlan:
    """Pure advantage computation for one question (see module docstring).
    `cfg` is train.gadv; `rng` decides which rows survive the caps."""
    n = n or len(members)
    correct = [i for i, ok, _ in members if ok]
    wrong = [(i, a) for i, ok, a in members if not ok]
    k = len(correct)
    if k >= n:
        if cfg.solved_floor <= 0:
            return GroupPlan("excluded", k, 1.0, n=n, n_rescues=n_rescues, reason="k==n")
        pick = _pick(rng, correct, cfg.solved_floor_max_per_question)
        return GroupPlan("floor", k, 1.0, n=n, n_rescues=n_rescues,
                         pos={i: float(cfg.solved_floor) for i in pick})
    if k == 0 and n_rescues == 0:
        return GroupPlan("excluded", 0, 0.0, n=n, n_rescues=0, reason="k==0,no_rescue")
    if k >= 1:
        kind, p = "frontier", k / n
        pick = _pick(rng, correct, cfg.correct_max_per_question)
        pos = {i: 1.0 - p for i in pick}
        rescue_adv = 0.0
        M = sum(pos.values())
    else:
        R = n_rescues
        kind, p = "rescue", R / (n + R)
        pos = {}
        rescue_adv = (1.0 - p) * cfg.rescue_dose
        M = R * rescue_adv
    sel = _pick(rng, wrong, cfg.wrong_max_per_question)
    n_neg = len(sel)
    counts = Counter(a for _, a in sel if a is not None)
    freq = {i: (counts[a] / n_neg if a is not None else 1.0 / n_neg) for i, a in sel}
    neg: dict[int, float] = {}
    if n_neg and M > 0:
        w = {i: f ** cfg.gamma for i, f in freq.items()}
        z = sum(w.values())
        neg = {i: -cfg.neg_scale * M * w[i] / z for i in w}
    return GroupPlan(kind, k, p, n=n, n_rescues=n_rescues, pos=pos, neg=neg,
                     rescue_adv=rescue_adv, answer_freq=freq)


def build_gadv_examples(cfg, it_dir: Path, iteration: int, eos: int,
                        kept: list[ImprovedCandidate],
                        refs: dict[str, float | None]) -> tuple[list[SFTExample], dict]:
    """Two streaming passes over rollout/rollouts.jsonl (hundreds of MB):
    pass 1 joins partition/verdicts.jsonl and keeps only (idx, clean, answer)
    per question to plan the groups; pass 2 retains token ids for the selected
    rows only. Rescue rows are built exactly like build_dataset's improved rows
    (same uid, same guard-ref join) with the group's rescue advantage."""
    gcfg = cfg.train.gadv
    n = cfg.rollout.n
    rollouts_path = it_dir / "rollout" / "rollouts.jsonl"
    verdicts = {
        (v.qid, v.sample_idx): v
        for v in VerdictRecord.load_jsonl(it_dir / "partition" / "verdicts.jsonl")
    }
    rescues_by_qid: dict[str, list[ImprovedCandidate]] = {}
    for c in kept:
        rescues_by_qid.setdefault(c.qid, []).append(c)

    # ---- pass 1: membership only ----
    members: dict[str, list[Member]] = {}
    truncated: dict[str, list[int]] = {}
    n_truncated_as_wrong = 0
    for s in RolloutSample.load_jsonl(rollouts_path):
        v = verdicts.get((s.qid, s.sample_idx))
        if v is None:
            raise RuntimeError(f"[gadv] rollout {s.qid}:{s.sample_idx} has no verdict")
        clean = bool(v.correct) and s.finish_reason == "stop"
        key = v.extracted_answer if s.finish_reason == "stop" else None
        if s.finish_reason != "stop":
            n_truncated_as_wrong += 1
            truncated.setdefault(s.qid, []).append(s.sample_idx)
        members.setdefault(s.qid, []).append((s.sample_idx, clean, key))

    # truncated-row cap: the question keeps n (so p = k/n is unchanged) and its
    # negative total; the mass just spreads over fewer rows. Own seed stream so
    # the default (cap >= n) is byte-identical to the uncapped builder.
    n_truncated_capped = 0
    for qid, idxs in truncated.items():
        if len(idxs) > gcfg.wrong_truncated_max_per_question:
            rng = random.Random(stable_seed(cfg.run.seed, "gadv-trunc", qid))
            keep = set(rng.sample(sorted(idxs), gcfg.wrong_truncated_max_per_question))
            drop = set(idxs) - keep
            members[qid] = [t for t in members[qid] if t[0] not in drop]
            n_truncated_capped += len(drop)

    plans: dict[str, GroupPlan] = {}
    for qid, m in members.items():
        m.sort(key=lambda t: t[0])
        rng = random.Random(stable_seed(cfg.run.seed, "gadv", qid))
        plans[qid] = group_advantages(m, len(rescues_by_qid.get(qid, [])), gcfg, rng, n=n)

    want: dict[tuple[str, int], tuple[str, float]] = {}
    for qid, plan in plans.items():
        for i, a in plan.pos.items():
            want[(qid, i)] = ("solved", a)
        for i, a in plan.neg.items():
            want[(qid, i)] = ("wrong", a)

    # ---- pass 2: token ids for the selected rows ----
    rows: list[SFTExample] = []
    n_empty_wrong = 0
    for s in RolloutSample.load_jsonl(rollouts_path):
        hit = want.get((s.qid, s.sample_idx))
        if hit is None:
            continue
        source, adv = hit
        plan = plans[s.qid]
        if source == "solved":
            completion = ensure_eos(s.response_token_ids, eos)
        else:
            # never APPEND an EOS to a wrong/truncated row; the stop token vLLM
            # already emitted on finish_reason=="stop" stays unless the ablation
            # knob strips it.
            completion = list(s.response_token_ids)
            if gcfg.wrong_drop_terminal_eos:
                while completion and completion[-1] == eos:
                    completion.pop()
            if not completion:
                n_empty_wrong += 1
                continue
        rows.append(SFTExample(
            uid=stable_hash("gadv", source, s.qid, s.sample_idx, iteration),
            qid=s.qid, source=source,
            input_ids=training_input_ids(s.prompt_token_ids, [], completion),
            prompt_len=len(s.prompt_token_ids), anchor_len=0,
            completion_len=len(completion),
            text=s.response_text, iter_created=iteration,
            advantage=float(adv), group_kind=plan.kind, group_size=plan.group_size,
        ))

    # ---- rescue rows (source="improved", guard ref joined) ----
    n_ref_joined = n_ref_missing = 0
    n_rescue_rows = n_rescue_dropped = 0
    for qid in sorted(rescues_by_qid):
        plan = plans.get(qid)
        if plan is None or plan.kind != "rescue":
            # a rescued question that is not 0/n clean-correct (only possible
            # with partition.cliff_max_correct > 0) trains as a frontier group
            n_rescue_dropped += len(rescues_by_qid[qid])
            continue
        for c in rescues_by_qid[qid]:
            completion = ensure_eos(c.continuation_token_ids, eos)
            ref = refs.get(f"{c.qid}:{c.base_sample_idx}:{c.attempt_idx}")
            if refs:
                n_ref_joined += ref is not None
                n_ref_missing += ref is None
            rows.append(SFTExample(
                uid=stable_hash("improved", c.qid, c.base_sample_idx, c.attempt_idx, iteration),
                qid=c.qid, source="improved",
                input_ids=training_input_ids(c.prompt_token_ids, c.anchor_token_ids, completion),
                prompt_len=len(c.prompt_token_ids), anchor_len=len(c.anchor_token_ids),
                completion_len=len(completion),
                text=c.continuation_text, iter_created=iteration,
                ref_mean_nll=ref,
                advantage=float(plan.rescue_adv), group_kind="rescue", group_size=plan.group_size,
            ))
            n_rescue_rows += 1

    # ---- stats ----
    def _adv_stats(src: str) -> dict:
        xs = [e.advantage for e in rows if e.source == src]
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "mean": round(sum(xs) / len(xs), 6),
                "min": round(min(xs), 6), "max": round(max(xs), 6)}

    active = [p for p in plans.values() if p.kind in ("frontier", "rescue")]
    # |Σpos + Σneg/neg_scale| per question: exactly 0 by construction (up to fp)
    resid = max((abs(p.pos_total + p.neg_total / gcfg.neg_scale) for p in active
                 if p.neg and gcfg.neg_scale > 0), default=0.0)
    q_kinds = Counter(p.kind for p in plans.values())
    questions = {k: q_kinds[k] for k in ("frontier", "rescue", "floor") if q_kinds[k]}
    questions.update({
        f"excluded_{r}": c for r, c in sorted(
            Counter(p.reason for p in plans.values() if p.kind == "excluded").items())
    })
    n_none_bucket = sum(
        1 for qid, p in plans.items() for (i, _ok, key) in members[qid]
        if i in p.neg and key is None
    )
    stats = {
        "gamma": gcfg.gamma, "rescue_dose": gcfg.rescue_dose, "neg_scale": gcfg.neg_scale,
        "solved_floor": gcfg.solved_floor,
        "questions": questions,
        "k_hist": {str(k): c for k, c in sorted(Counter(p.k for p in plans.values()).items())},
        "rows": dict(sorted(Counter(e.source for e in rows).items())),
        "adv": {src: _adv_stats(src) for src in ("solved", "wrong", "improved")},
        "zero_sum_max_abs_residual": resid,
        "n_truncated_as_wrong": n_truncated_as_wrong,
        "n_truncated_capped_rows": n_truncated_capped,
        "n_none_bucket_rows": n_none_bucket,
        "n_correct_capped": sum(1 for p in plans.values()
                                if p.kind == "frontier" and len(p.pos) < p.k),
        "n_wrong_capped": sum(1 for p in plans.values()
                              if p.kind in ("frontier", "rescue") and len(p.neg) < p.n - p.k),
        "n_empty_wrong_dropped": n_empty_wrong,
        "n_rescue_rows": n_rescue_rows,
        "n_rescue_rows_dropped_k_ge_1": n_rescue_dropped,
        "n_ref_joined": n_ref_joined,
        "n_ref_missing": n_ref_missing,
    }
    return rows, stats
