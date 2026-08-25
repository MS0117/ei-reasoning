"""Diagnostic: scaffold-attributed per-token credit on rescued trajectories.

For every verifier-correct candidate a bridge / staged operator produced (a
trajectory sampled from base + transient LoRA adapter), run two teacher-forcing
passes over the SAME continuation tokens:

    lp_lora_t = log pi_{theta+phi}(y_t | x, y_<t)      (op_meta.lora_path)
    lp_base_t = log pi_theta      (y_t | x, y_<t)      (adapter off)

With --scaffold privileged the second pass is instead the BASE model shown the
gold solution in the prompt (render_privileged_prompt: x + y*), adapter off:

    lp_priv_t = log pi_theta(y_t | x, y*, y_<t)

so g_t = lp_priv_t - lp_base_t measures where KNOWING the reference solution
raises the trajectory's own next token (a teacher that knows the answer).

and derive per token

    g_t  = lp_lora_t - lp_base_t          scaffold contribution ("the gap")
    s_t  = -lp_base_t                     base surprisal (= uniform-SFT loss)
    a_t  = clip(g_t^+ / s_t, 0, 1)        share of base surprisal attributable
                                          to the scaffold
    H_t  = top-K entropy of the BASE next-token distribution (truncated to K)

The report answers the three go/no-go questions for "learn the gap, not the
trajectory" token weighting:

  1. CONCENTRATION  is sum(g^+) carried by a few tokens (top-10% share, Gini)?
                    diffuse  -> the adapter rewrote everything (memorised y*)
  2. (A) vs (B)     among high-surprisal tokens (s_t > s_hi), how many are
                    scaffold-driven (a_t > a_min, class A) vs noise shared by
                    both policies (class B)? A large B mass is the gain from
                    DOWN-weighting.
  3. OVERLAP        Jaccard of top-q% by g^+ vs top-q% by base entropy / by
                    surprisal. High overlap = indistinguishable from the
                    forking-token (entropy) literature.
plus a FORKING view: how often a top-g token is followed by another top-g token,
and the same statistics at step granularity (steps.py boundaries).

Outputs under <run>/iter_<k>/diag_credit/ (or --out):
  tokens.jsonl       one row per scored token (key, pos, step, g, s, a, H)
  candidates.jsonl   one row per candidate with all summary statistics
  summary.json       aggregate over candidates (+ the knobs used)
  report.txt         the human-readable digest printed at the end

CPU driver; the two scoring passes run in the vLLM pool (GPU). Pass GPUs via
CUDA_VISIBLE_DEVICES (same convention as data/run_toy_cliff.sh). --dry-run
builds everything up to the pool call and prints the request plan.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/diag_scaffold_credit.py \
      --run-dir runs/toy_cliff/default_STAGED_20260821_154423 \
      --max-per-qid 2 --max-cands 120 --topk 20
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from expert_iter.config import Config
from expert_iter.records import ImprovedCandidate
from expert_iter.utils import iter_dir, stable_hash, stable_seed, write_json, write_jsonl


# --------------------------------------------------------------------------- #
# pure functions (CPU, unit-testable)                                          #
# --------------------------------------------------------------------------- #

def topk_entropy(topk: dict[str, float] | None) -> float | None:
    """Entropy (nats) of the top-K next-token distribution renormalised over
    the K entries. Truncated -> a LOWER bound on the true entropy; fine for
    ranking tokens against each other."""
    if not topk:
        return None
    vals = list(topk.values())
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    z = sum(exps)
    h = 0.0
    for e in exps:
        p = e / z
        if p > 0:
            h -= p * math.log(p)
    return h


def per_token_signals(
    lp_base: list[float | None],
    lp_lora: list[float | None],
    topk_base: list[dict[str, float] | None] | None,
    n_positions: int,
) -> list[dict | None]:
    """Align the two passes position-by-position. None where either pass has
    no entry (vLLM returns nothing for position 0 of the sequence — never an
    issue here since score_from >= len(prompt) > 0)."""
    out: list[dict | None] = []
    for i in range(n_positions):
        b = lp_base[i] if i < len(lp_base) else None
        q = lp_lora[i] if i < len(lp_lora) else None
        if b is None or q is None:
            out.append(None)
            continue
        g = q - b
        s = -b
        g_pos = max(g, 0.0)
        a = min(g_pos / s, 1.0) if s > 1e-6 else 0.0
        H = topk_entropy(topk_base[i]) if topk_base and i < len(topk_base) else None
        out.append({"g": g, "s": s, "a": a, "H": H})
    return out


def gini(values: list[float]) -> float:
    """Gini coefficient of a non-negative list (0 = uniform, 1 = one token)."""
    xs = sorted(v for v in values if v >= 0)
    n = len(xs)
    tot = sum(xs)
    if n == 0 or tot <= 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return (2.0 * cum) / (n * tot) - (n + 1.0) / n


def top_share(values: list[float], frac: float) -> float:
    """Share of sum(values) held by the top `frac` fraction of entries."""
    tot = sum(values)
    if tot <= 0 or not values:
        return 0.0
    k = max(1, int(math.ceil(frac * len(values))))
    return sum(sorted(values, reverse=True)[:k]) / tot


def top_index_set(values: list[float], frac: float) -> set[int]:
    k = max(1, int(math.ceil(frac * len(values))))
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return set(order[:k])


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def step_index_of(bounds: list[int], n: int) -> list[int]:
    """bounds = exclusive END indices b_1<...<b_J==n -> step id per position."""
    idx = [0] * n
    prev = 0
    for j, b in enumerate(bounds):
        for i in range(prev, min(b, n)):
            idx[i] = j
        prev = b
    for i in range(prev, n):  # defensive: bounds shorter than n
        idx[i] = len(bounds)
    return idx


def candidate_stats(
    sig: list[dict | None],
    bounds: list[int],
    *,
    q_frac: float,
    s_hi: float,
    a_min: float,
    g_fork: float,
) -> dict:
    """All per-candidate statistics from the aligned token signals."""
    valid = [(i, d) for i, d in enumerate(sig) if d is not None]
    n = len(sig)
    if not valid:
        return {"n_tokens": n, "n_scored": 0}
    idxs = [i for i, _ in valid]
    g = [d["g"] for _, d in valid]
    s = [d["s"] for _, d in valid]
    a = [d["a"] for _, d in valid]
    g_pos = [max(v, 0.0) for v in g]
    H = [d["H"] for _, d in valid]
    have_H = all(h is not None for h in H)

    sum_g = sum(g)
    sum_g_pos = sum(g_pos)
    sum_s = sum(s)

    # ---- 1. concentration of the gap ------------------------------------
    conc = {
        "top10_share_gpos": top_share(g_pos, 0.10),
        "top20_share_gpos": top_share(g_pos, 0.20),
        "gini_gpos": gini(g_pos),
        "frac_tokens_g_gt_0": sum(1 for v in g if v > 0) / len(g),
        "frac_tokens_g_gt_fork": sum(1 for v in g if v > g_fork) / len(g),
        "frac_tokens_g_lt_neg_fork": sum(1 for v in g if v < -g_fork) / len(g),
    }

    # ---- 2. (A) scaffold-driven vs (B) shared-noise among high-surprisal --
    hi = [(gi, si, ai) for gi, si, ai in zip(g, s, a) if si > s_hi]
    n_hi = len(hi)
    n_A = sum(1 for _, _, ai in hi if ai >= a_min)
    n_B = n_hi - n_A
    s_hi_mass = sum(si for _, si, _ in hi)
    s_A_mass = sum(si for _, si, ai in hi if ai >= a_min)
    classes = {
        "n_high_surprisal": n_hi,
        "frac_tokens_high_surprisal": n_hi / len(s),
        "n_A": n_A,
        "n_B": n_B,
        "frac_A_among_high": (n_A / n_hi) if n_hi else None,
        # how much of the uniform-SFT loss mass sits on noise tokens (B)
        "loss_mass_high_surprisal": (s_hi_mass / sum_s) if sum_s > 0 else None,
        "loss_mass_B": ((s_hi_mass - s_A_mass) / sum_s) if sum_s > 0 else None,
        "loss_mass_A": (s_A_mass / sum_s) if sum_s > 0 else None,
        # overall attributable share of base surprisal
        "attributed_share": (sum_g_pos / sum_s) if sum_s > 0 else None,
    }

    # ---- 3. overlap of selectors ------------------------------------------
    top_g = top_index_set(g_pos, q_frac)
    top_s = top_index_set(s, q_frac)
    overlap = {"jaccard_g_vs_surprisal": jaccard(top_g, top_s)}
    if have_H:
        top_H = top_index_set(H, q_frac)
        overlap["jaccard_g_vs_entropy"] = jaccard(top_g, top_H)
        overlap["jaccard_entropy_vs_surprisal"] = jaccard(top_H, top_s)
        # rank correlation (Spearman) between g^+ and H
        overlap["spearman_g_H"] = _spearman(g_pos, H)
    else:
        overlap["jaccard_g_vs_entropy"] = None
        overlap["jaccard_entropy_vs_surprisal"] = None
        overlap["spearman_g_H"] = None

    # ---- forking: does credit land on one token or on a span? -------------
    pos_of = {i: k for k, i in enumerate(idxs)}
    fork_hits = [k for k, v in enumerate(g) if v > g_fork]
    nxt = sum(1 for k in fork_hits if k + 1 < len(g) and g[k + 1] > g_fork)
    # mean g over the 8 tokens after a fork hit (excluding the hit itself)
    after = []
    for k in fork_hits:
        w = g[k + 1:k + 9]
        if w:
            after.append(sum(w) / len(w))
    forking = {
        "n_fork_hits": len(fork_hits),
        "p_next_also_fork": (nxt / len(fork_hits)) if fork_hits else None,
        "mean_g_8_after_fork": (sum(after) / len(after)) if after else None,
        "mean_g_all": sum_g / len(g),
    }

    # ---- step granularity -------------------------------------------------
    step_id = step_index_of(bounds, n)
    per_step: dict[int, list[float]] = defaultdict(list)
    per_step_s: dict[int, list[float]] = defaultdict(list)
    for i, d in valid:
        per_step[step_id[i]].append(d["g"])
        per_step_s[step_id[i]].append(d["s"])
    step_mean_g = [sum(v) / len(v) for _, v in sorted(per_step.items())]
    step_sum_gpos = [sum(max(x, 0.0) for x in v) for _, v in sorted(per_step.items())]
    steps = {
        "n_steps": len(step_mean_g),
        "top10_share_step_gpos": top_share(step_sum_gpos, 0.10),
        "top20_share_step_gpos": top_share(step_sum_gpos, 0.20),
        "gini_step_gpos": gini(step_sum_gpos),
        "frac_steps_mean_g_gt_fork": (sum(1 for v in step_mean_g if v > g_fork) / len(step_mean_g))
        if step_mean_g else None,
        "argmax_step_frac": (max(range(len(step_mean_g)), key=lambda j: step_mean_g[j]) / max(1, len(step_mean_g) - 1))
        if len(step_mean_g) > 1 else None,
    }

    # ---- the LSPO importance-sampling view --------------------------------
    # sequence log-ratio log pi_base(y)/pi_lora(y) = -sum g ; per-token ratio
    # exp(-g) clipped at [0.8, 1.28] (DAPO clip) -> which tokens LSPO keeps
    eps_lo, eps_hi = math.log(0.8), math.log(1.28)
    is_view = {
        "seq_log_ratio_base_over_lora": -sum_g,
        "frac_tokens_ratio_below_clip": sum(1 for v in g if -v < eps_lo) / len(g),
        "frac_tokens_ratio_above_clip": sum(1 for v in g if -v > eps_hi) / len(g),
        # share of the gap mass that sits on tokens LSPO's clip would flatten
        "gpos_mass_below_clip": (sum(v for v in g_pos if -v < eps_lo) / sum_g_pos) if sum_g_pos > 0 else None,
    }

    return {
        "n_tokens": n,
        "n_scored": len(valid),
        "sum_g": sum_g,
        "sum_g_pos": sum_g_pos,
        "sum_s": sum_s,
        "mean_s": sum_s / len(s),
        "mean_H": (sum(H) / len(H)) if have_H else None,
        "concentration": conc,
        "classes": classes,
        "overlap": overlap,
        "forking": forking,
        "steps": steps,
        "is_view": is_view,
    }


def _spearman(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _agg(rows: list[dict], path: tuple[str, ...]) -> dict | None:
    vals = []
    for r in rows:
        v = r
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                break
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    if not vals:
        return None
    vals.sort()
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "p10": vals[int(0.10 * (len(vals) - 1))],
        "p90": vals[int(0.90 * (len(vals) - 1))],
    }


AGG_KEYS = [
    ("concentration", "top10_share_gpos"),
    ("concentration", "top20_share_gpos"),
    ("concentration", "gini_gpos"),
    ("concentration", "frac_tokens_g_gt_fork"),
    ("concentration", "frac_tokens_g_lt_neg_fork"),
    ("classes", "frac_tokens_high_surprisal"),
    ("classes", "frac_A_among_high"),
    ("classes", "loss_mass_high_surprisal"),
    ("classes", "loss_mass_A"),
    ("classes", "loss_mass_B"),
    ("classes", "attributed_share"),
    ("overlap", "jaccard_g_vs_entropy"),
    ("overlap", "jaccard_g_vs_surprisal"),
    ("overlap", "jaccard_entropy_vs_surprisal"),
    ("overlap", "spearman_g_H"),
    ("forking", "p_next_also_fork"),
    ("forking", "mean_g_8_after_fork"),
    ("forking", "mean_g_all"),
    ("steps", "n_steps"),
    ("steps", "top10_share_step_gpos"),
    ("steps", "gini_step_gpos"),
    ("steps", "frac_steps_mean_g_gt_fork"),
    ("steps", "argmax_step_frac"),
    ("is_view", "seq_log_ratio_base_over_lora"),
    ("is_view", "frac_tokens_ratio_below_clip"),
    ("is_view", "gpos_mass_below_clip"),
    ("mean_s",),
    ("mean_H",),
    ("n_tokens",),
]


def aggregate(cands: list[dict]) -> dict:
    out = {"n_candidates": len(cands)}
    for path in AGG_KEYS:
        out["/".join(path)] = _agg(cands, path)
    # by group (stage / chunk) when present
    by_group: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_group[c.get("group", "all")].append(c)
    if len(by_group) > 1:
        out["by_group"] = {
            grp: {
                "n": len(rows),
                "top10_share_gpos": _agg(rows, ("concentration", "top10_share_gpos")),
                "loss_mass_B": _agg(rows, ("classes", "loss_mass_B")),
                "jaccard_g_vs_entropy": _agg(rows, ("overlap", "jaccard_g_vs_entropy")),
            }
            for grp, rows in sorted(by_group.items())
        }
    return out


def render_report(summary: dict, knobs: dict) -> str:
    def fmt(key, scale=1.0, unit=""):
        a = summary.get(key)
        if not a:
            return f"  {key:42s}  n/a"
        return (f"  {key:42s}  mean {a['mean'] * scale:7.3f}{unit}  "
                f"median {a['median'] * scale:7.3f}{unit}  "
                f"[p10 {a['p10'] * scale:6.3f}, p90 {a['p90'] * scale:6.3f}]")

    L = []
    L.append("=== scaffold-attributed credit diagnostic ===")
    L.append(f"candidates scored: {summary['n_candidates']}   knobs: {json.dumps(knobs)}")
    L.append("")
    L.append("[1] CONCENTRATION of the gap g^+ (uniform = 0.10 / 0.20 shares, Gini 0)")
    L.append(fmt("concentration/top10_share_gpos"))
    L.append(fmt("concentration/top20_share_gpos"))
    L.append(fmt("concentration/gini_gpos"))
    L.append(fmt("concentration/frac_tokens_g_gt_fork"))
    L.append(fmt("concentration/frac_tokens_g_lt_neg_fork"))
    L.append("    -> read: top10 share >> 0.10 and Gini > ~0.6 means credit is localised;")
    L.append("       near-uniform means the adapter rewrote the whole trajectory (memorised y*).")
    L.append("")
    L.append(f"[2] (A) scaffold-driven vs (B) shared-noise among HIGH-surprisal tokens (s > {knobs['s_hi']} nats)")
    L.append(fmt("classes/frac_tokens_high_surprisal"))
    L.append(fmt("classes/loss_mass_high_surprisal"))
    L.append(fmt("classes/frac_A_among_high"))
    L.append(fmt("classes/loss_mass_A"))
    L.append(fmt("classes/loss_mass_B"))
    L.append(fmt("classes/attributed_share"))
    L.append("    -> read: loss_mass_B is the share of uniform-SFT loss spent on tokens the")
    L.append("       scaffold did NOT cause (both policies unsure). Large = down-weighting gain.")
    L.append("")
    L.append(f"[3] OVERLAP of top-{int(knobs['q_frac'] * 100)}% selectors (Jaccard; random ~ {knobs['q_frac'] / (2 - knobs['q_frac']):.2f})")
    L.append(fmt("overlap/jaccard_g_vs_entropy"))
    L.append(fmt("overlap/jaccard_g_vs_surprisal"))
    L.append(fmt("overlap/jaccard_entropy_vs_surprisal"))
    L.append(fmt("overlap/spearman_g_H"))
    L.append("    -> read: g-vs-entropy near random = a different signal from forking-token")
    L.append("       (entropy) selection; near 1 = the same thing under a new name.")
    L.append("")
    L.append(f"[4] FORKING (fork hit = g > {knobs['g_fork']})")
    L.append(fmt("forking/p_next_also_fork"))
    L.append(fmt("forking/mean_g_8_after_fork"))
    L.append(fmt("forking/mean_g_all"))
    L.append("    -> read: p_next low + mean_g_after ~ mean_g_all means credit lands on single")
    L.append("       tokens and the span they open gets none -> needs step-level / decayed weights.")
    L.append("")
    L.append("[5] STEP granularity")
    L.append(fmt("steps/n_steps"))
    L.append(fmt("steps/top10_share_step_gpos"))
    L.append(fmt("steps/gini_step_gpos"))
    L.append(fmt("steps/frac_steps_mean_g_gt_fork"))
    L.append(fmt("steps/argmax_step_frac"))
    L.append("")
    L.append("[6] LSPO importance-sampling view (per-token ratio exp(-g), DAPO clip [0.8, 1.28])")
    L.append(fmt("is_view/seq_log_ratio_base_over_lora"))
    L.append(fmt("is_view/frac_tokens_ratio_below_clip"))
    L.append(fmt("is_view/gpos_mass_below_clip"))
    L.append("    -> read: gpos_mass_below_clip = share of the gap that LSPO's clipped ratio flattens.")
    L.append("")
    L.append("[misc]")
    L.append(fmt("n_tokens"))
    L.append(fmt("mean_s"))
    L.append(fmt("mean_H"))
    if "by_group" in summary:
        L.append("")
        L.append("[by group]")
        for grp, d in summary["by_group"].items():
            t10 = d["top10_share_gpos"]["mean"] if d["top10_share_gpos"] else float("nan")
            lb = d["loss_mass_B"]["mean"] if d["loss_mass_B"] else float("nan")
            je = d["jaccard_g_vs_entropy"]["mean"] if d["jaccard_g_vs_entropy"] else float("nan")
            L.append(f"  {grp:16s} n={d['n']:4d}  top10_share={t10:.3f}  loss_mass_B={lb:.3f}  J(g,H)={je:.3f}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #

def select_candidates(cands, *, max_per_qid: int, max_cands: int, seed: int,
                      include_incorrect: bool, balanced: bool = False,
                      require_lora: bool = True,
                      allowed_qids: set[str] | None = None) -> list[ImprovedCandidate]:
    """Up to max_per_qid candidates per qid (seeded). With `balanced`, the quota
    applies SEPARATELY to correct and incorrect candidates of each qid, and
    the max_cands cap is applied per class, so the two classes stay matched
    by question for the correct-vs-incorrect comparison. `require_lora` drops
    candidates without op_meta.lora_path (the adapter scaffold needs it; the
    privileged scaffold does not); `allowed_qids` restricts to questions that
    have what the scaffold needs (a gold solution)."""
    by_qid: dict[str, list[ImprovedCandidate]] = defaultdict(list)
    for c in cands:
        if require_lora and not (c.op_meta or {}).get("lora_path"):
            continue
        if allowed_qids is not None and c.qid not in allowed_qids:
            continue
        if c.correct is not True and not (include_incorrect or balanced):
            continue
        if not c.continuation_token_ids:
            continue
        by_qid[c.qid].append(c)

    def _pick(rows, qid, salt=None):
        rows = sorted(rows, key=lambda c: (c.base_sample_idx, c.attempt_idx))
        # no salt = the original (unbalanced) draw, so earlier runs stay reproducible
        parts = (seed, "diag_credit", qid) if salt is None else (seed, "diag_credit", qid, salt)
        rng = random.Random(stable_seed(*parts))
        return rng.sample(rows, min(max_per_qid, len(rows)))

    picked: list[ImprovedCandidate] = []
    if balanced:
        pos: list[ImprovedCandidate] = []
        neg: list[ImprovedCandidate] = []
        for qid in sorted(by_qid):
            rows = by_qid[qid]
            p = [c for c in rows if c.correct is True]
            n = [c for c in rows if c.correct is not True]
            if not p or not n:  # keep only questions that have BOTH classes
                continue
            pos.extend(_pick(p, qid, "pos"))
            neg.extend(_pick(n, qid, "neg"))
        # cap by QUESTION (both classes of a kept qid survive together) so the
        # by-question pairing the compare script relies on is preserved
        if len(pos) > max_cands or len(neg) > max_cands:
            qids = sorted({c.qid for c in pos})
            rng = random.Random(stable_seed(seed, "diag_credit", "cap_qids"))
            rng.shuffle(qids)
            keep: set[str] = set()
            n_pos = n_neg = 0
            for q in qids:
                p_q = sum(1 for c in pos if c.qid == q)
                n_q = sum(1 for c in neg if c.qid == q)
                if n_pos + p_q > max_cands or n_neg + n_q > max_cands:
                    break
                keep.add(q)
                n_pos += p_q
                n_neg += n_q
            pos = [c for c in pos if c.qid in keep]
            neg = [c for c in neg if c.qid in keep]
        picked = pos + neg
    else:
        for qid in sorted(by_qid):
            picked.extend(_pick(by_qid[qid], qid))
        if len(picked) > max_cands:
            rng = random.Random(stable_seed(seed, "diag_credit", "cap"))
            picked = rng.sample(picked, max_cands)
    picked.sort(key=lambda c: (c.qid, c.base_sample_idx, c.attempt_idx))
    return picked


def cand_key(c: ImprovedCandidate) -> str:
    return f"{c.qid}:{c.base_sample_idx}:{c.attempt_idx}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="toy / EI run dir holding config.yaml and iter_k/improve/improved.jsonl")
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument("--model-path", default=None, help="policy the candidates were sampled from (default: config model.base for iter 0, else iter_{k-1}/ckpt)")
    ap.add_argument("--out", default=None, help="output dir (default <run>/iter_k/diag_credit)")
    ap.add_argument("--max-per-qid", type=int, default=2)
    ap.add_argument("--max-cands", type=int, default=120)
    ap.add_argument("--include-incorrect", action="store_true", help="also score verifier-failed candidates (control)")
    ap.add_argument("--scaffold", choices=["lora", "privileged"], default="lora",
                    help="what the second pass conditions on: lora = base + op_meta.lora_path (adapter "
                         "contrast); privileged = base with the gold solution y* in the prompt "
                         "(render_privileged_prompt), adapter OFF. g_t is then lp_priv - lp_base.")
    ap.add_argument("--balanced", action="store_true",
                    help="per qid, up to --max-per-qid correct AND incorrect candidates (only qids with both); "
                         "--max-cands caps each class separately")
    ap.add_argument("--topk", type=int, default=20, help="top-K for base entropy (0 = skip entropy; >engine.max_logprobs is overridden)")
    ap.add_argument("--q-frac", type=float, default=0.20, help="top fraction for selector overlap")
    ap.add_argument("--s-hi", type=float, default=1.0, help="high-surprisal threshold in nats (1.0 <=> p_base < 0.37)")
    ap.add_argument("--a-min", type=float, default=0.5, help="attributed share a_t >= a_min -> class A")
    ap.add_argument("--g-fork", type=float, default=1.0, help="g_t > g_fork counts as a fork hit (nats)")
    ap.add_argument("--max-tokens", type=int, default=None, help="truncate continuations longer than this (saves prefill)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="build the request plan, skip the GPU pool")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = Config.load(run_dir / "config.yaml")
    cfg.engine.enable_lora = True
    if args.topk > cfg.engine.max_logprobs:
        cfg.engine.max_logprobs = args.topk
    it_dir = iter_dir(run_dir, args.iter)
    out = Path(args.out) if args.out else it_dir / ("diag_credit" if args.scaffold == "lora" else "diag_credit_priv")
    out.mkdir(parents=True, exist_ok=True)

    if args.model_path:
        model_path = args.model_path
    elif args.iter == 0:
        model_path = cfg.model.base
    else:
        model_path = str(iter_dir(run_dir, args.iter - 1) / "ckpt")

    all_cands = list(ImprovedCandidate.load_jsonl(it_dir / "improve" / "improved.jsonl"))

    # privileged scaffold: prompt(x, y*) rendered per qid, same tokenizer path as the
    # (deprecated) privileged_divergence anchor — y* never touches the continuation ids
    priv_ids: dict[str, list[int]] = {}
    if args.scaffold == "privileged":
        from transformers import AutoTokenizer
        from expert_iter.data import load_gold_solutions
        from expert_iter.records import QuestionRecord
        from expert_iter.templates import render_privileged_prompt

        gold = load_gold_solutions(run_dir)
        questions = {q.qid: q.question for q in QuestionRecord.load_jsonl(run_dir / "questions" / "train.jsonl")}
        tok_priv = AutoTokenizer.from_pretrained(cfg.model.base)
        for qid, sol in gold.items():
            if qid in questions:
                priv_ids[qid] = render_privileged_prompt(
                    tok_priv, questions[qid], sol,
                    system_prompt=cfg.model.system_prompt,
                    question_suffix=cfg.data.question_suffix,
                    chat_template_kwargs=cfg.model.chat_template_kwargs,
                ).token_ids
        print(f"[diag] privileged scaffold: {len(priv_ids)} qids with gold y*")
        if not priv_ids:
            raise SystemExit("no gold solutions in questions/train.jsonl — backfill first")

    cands = select_candidates(all_cands, max_per_qid=args.max_per_qid, max_cands=args.max_cands,
                              seed=args.seed, include_incorrect=args.include_incorrect,
                              balanced=args.balanced,
                              require_lora=(args.scaffold == "lora"),
                              allowed_qids=set(priv_ids) if args.scaffold == "privileged" else None)
    n_correct_total = sum(1 for c in all_cands if c.correct is True)
    print(f"[diag] {len(all_cands)} candidates on disk ({n_correct_total} correct); "
          f"scoring {len(cands)} over {len({c.qid for c in cands})} qids")
    if not cands:
        raise SystemExit("no usable candidates (lora scaffold needs op_meta.lora_path; privileged needs "
                         "gold y* for the candidate's qid) — is this a bridge/staged run with gold backfilled?")

    from expert_iter.engine import GenRequest, run_pool

    reqs: list[GenRequest] = []
    cut: dict[str, int] = {}
    for c in cands:
        cont = c.continuation_token_ids
        if args.max_tokens and len(cont) > args.max_tokens:
            cont = cont[:args.max_tokens]
        cut[cand_key(c)] = len(cont)
        seq = c.prompt_token_ids + c.anchor_token_ids + cont
        sf = len(c.prompt_token_ids) + len(c.anchor_token_ids)
        key = cand_key(c)
        reqs.append(GenRequest(rid=f"{key}:b", prompt_token_ids=seq, score_from=sf))
        if args.scaffold == "lora":
            reqs.append(GenRequest(rid=f"{key}:l", prompt_token_ids=seq, score_from=sf,
                                   lora_path=c.op_meta["lora_path"]))
        else:
            # same anchor + continuation ids behind the privileged prompt; only the
            # prompt differs, so index i of both passes is continuation token i
            pseq = priv_ids[c.qid] + c.anchor_token_ids + cont
            reqs.append(GenRequest(rid=f"{key}:l", prompt_token_ids=pseq,
                                   score_from=len(priv_ids[c.qid]) + len(c.anchor_token_ids)))
    n_tok = sum(cut.values())
    adapters = sorted({r.lora_path for r in reqs if r.lora_path})
    print(f"[diag] scaffold={args.scaffold}: {len(reqs)} score requests, {n_tok} continuation tokens "
          f"x2 passes, {len(adapters)} adapters, longest seq {max(len(r.prompt_token_ids) for r in reqs)}")
    for a in adapters:
        ok = Path(a, "adapter_config.json").exists()
        print(f"        adapter {'ok ' if ok else 'MISSING'} {a}")
    if args.dry_run:
        print("[diag] dry run: stopping before the pool")
        return

    sampling = {"return_token_logprobs": True}
    if args.topk > 0:
        sampling["prompt_logprobs_k"] = args.topk
    spec = stable_hash(model_path, sorted(r.rid for r in reqs), args.topk, args.max_tokens, run_dir.name,
                       args.scaffold)
    results = run_pool(
        reqs, mode="score", model_path=model_path, sampling=sampling,
        engine_cfg=cfg.engine, work_dir=out / f"pool_{spec[:8]}", dtype=cfg.model.dtype,
    )
    by_rid = {r.rid: r for r in results}

    from transformers import AutoTokenizer
    from expert_iter.steps import step_token_bounds

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base)

    knobs = {"scaffold": args.scaffold, "q_frac": args.q_frac, "s_hi": args.s_hi, "a_min": args.a_min,
             "g_fork": args.g_fork, "topk": args.topk, "max_tokens": args.max_tokens}
    token_rows: list[dict] = []
    cand_rows: list[dict] = []
    for c in cands:
        key = cand_key(c)
        rb, rl = by_rid[f"{key}:b"], by_rid[f"{key}:l"]
        n_pos = cut[key]
        sig = per_token_signals(rb.token_logprobs or [], rl.token_logprobs or [],
                                rb.topk_logprobs if args.topk > 0 else None, n_pos)
        cont = c.continuation_token_ids[:n_pos]
        bounds = step_token_bounds(cont, tokenizer)
        st = candidate_stats(sig, bounds, q_frac=args.q_frac, s_hi=args.s_hi,
                             a_min=args.a_min, g_fork=args.g_fork)
        om = c.op_meta or {}
        group = om.get("stage") or om.get("chunk") or "all"
        cand_rows.append({
            "key": key, "qid": c.qid, "correct": c.correct, "group": group,
            "scaffold": args.scaffold,
            "lora_path": om.get("lora_path"), "alpha": om.get("alpha"),
            "truncated": n_pos < len(c.continuation_token_ids),
            **st,
        })
        step_id = step_index_of(bounds, n_pos)
        pieces = tokenizer.batch_decode([[t] for t in cont])
        for i, d in enumerate(sig):
            if d is None:
                continue
            token_rows.append({"key": key, "pos": i, "step": step_id[i], "tok": pieces[i],
                               "g": round(d["g"], 4), "s": round(d["s"], 4),
                               "a": round(d["a"], 4),
                               "H": round(d["H"], 4) if d["H"] is not None else None})

    summary = aggregate(cand_rows)
    summary["knobs"] = knobs
    summary["model_path"] = model_path
    summary["run_dir"] = str(run_dir)
    summary["n_candidates_on_disk"] = len(all_cands)
    summary["n_correct_on_disk"] = n_correct_total
    # the report aggregates pool BOTH classes — record the mix so a reader of
    # report.txt/summary.json knows what the numbers are over (split with
    # scripts/diag_credit_compare.py)
    summary["n_correct_scored"] = sum(1 for c in cands if c.correct is True)
    summary["n_incorrect_scored"] = sum(1 for c in cands if c.correct is not True)
    summary["selection"] = {"balanced": args.balanced, "include_incorrect": args.include_incorrect,
                            "max_per_qid": args.max_per_qid, "max_cands": args.max_cands, "seed": args.seed}
    write_jsonl(out / "tokens.jsonl", token_rows)
    write_jsonl(out / "candidates.jsonl", cand_rows)
    write_json(out / "summary.json", summary)
    report = render_report(summary, knobs)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n[diag] wrote {out}/{{tokens,candidates}}.jsonl, summary.json, report.txt")


if __name__ == "__main__":
    main()
