#!/usr/bin/env python3
"""How far does the privileged gold solution y* leak — as text — into what the
student would be trained on?

The bridge operator generates z* from a prompt that SHOWS y*, fits a transient
LoRA on those z*, and then samples candidates from bare x.  This measures four
things at each point of that pipeline.  Definitions and the 2026-09-01 results:
docs/toy_cliff_leakage_metrics.md.

  refN / refB   did the trajectory SAY a reference solution exists (narrow /
                broad phrase regex).  Word matching is useless here: 70.6% of
                bridges contain "solution" and 56.7% contain "reference", almost
                all as ordinary math prose ("no solution", "the only solution is
                c = 1"), so both regexes match PHRASES only.  Both are lower
                bounds — a human reader catches more.
  gold_recall   fraction of the gold solution's 8-grams that appear verbatim in
                the trajectory.  0 = unrelated, 1 = contains gold whole.  The
                denominator is gold, so a longer trajectory does not inflate it.
  near_copy     gold_recall >= 0.30  OR  a shared 30-gram (~100+ consecutive
                chars).  The two clauses catch different copying: the first a
                scattered/reordered rewrite, the second a lifted paragraph.
                8-grams (~25 chars) collide by chance in math and cannot be used
                for a plagiarism call; 30-grams essentially do not.

Read every number against the CONTROL row (the base policy's own correct
samples, which never saw y*): that is the chance-collision floor, measured at
3.7% near-copy on this set.

Also reported, and NOT leakage metrics (see the doc's warning): ans@10/25 = where
the answer string first appears, show_that = "we need to show/verify that ..."
framing.  The gold solutions THEMSELVES score 31% on ans@10, so an early answer
is a property of terse math writing, not evidence of contamination.  show_that
does discriminate (gold 6% / bridge 30% / candidates 19-25%).

Usage (CPU, ~2 min per run dir):
  .venv/bin/python scripts/diag_leakage.py runs/toy_cliff_2/default_BRIDGE_*
  .venv/bin/python scripts/diag_leakage.py runs/toy_cliff_2/default_*_2026* \
      --questions data/cliff_sets/openr1_default_cliff450_k16_with_gold.jsonl \
      -o runs/toy_cliff_2/_analysis/leakage.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

# --- reference-mention regexes ---------------------------------------------
# NARROW: what the bridge prompt calls y*.  On bridges it is almost entirely
# "reference solution" (7,008 of 7,190 matches on the 2026-08-30 BRIDGE run).
NARROW = re.compile(
    r"reference (solution|answer)|given solution|provided solution|"
    r"the solution (states|says|given)|official solution|"
    r"according to the (solution|reference)|as (given|shown) in the solution",
    re.I)
# BROAD: NARROW plus hedged phrasings found by reading misses, e.g.
# "This is known from the problem's context and its solution."  On bridges the
# two agree (52.5% vs 53.0%) because bridges say it outright; on CANDIDATES they
# differ 5x (0-0.7% vs 3.1-6.9%), so report refB for post-LoRA text.
BROAD = re.compile(
    r"reference (solution|answer)|given solution|provided solution|official solution|"
    r"(the|its|this) solution('s)? (states|says|given|claims|shows|uses|suggests|indicates|mentions)|"
    r"according to (the )?(solution|reference|answer key)|"
    r"as (given|shown|stated) in the (solution|reference)|"
    r"from (the )?(problem'?s )?(context and its|known) solution|"
    r"the (intended|model|book) (solution|answer)", re.I)
SHOW = re.compile(
    r"\b(we (need|want|have|must) to (show|prove|verify|confirm)|"
    r"let'?s (verify|check|confirm)|to (verify|confirm|check) (that|this))\b", re.I)

NEAR_COPY_RECALL = 0.30
NEAR_COPY_NGRAM = 30


def toks(s: str) -> list[str]:
    """Word / number / symbol, one token each — tokenizer-independent so bridge
    text and gold text are comparable without re-tokenizing either."""
    return re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", s.lower())


def grams(t: list[str], n: int) -> set[tuple]:
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).replace("\\dfrac", "\\frac").replace("$", "")


def answer_first_pos(text: str, answer: str) -> float | None:
    """Where the answer string first appears, as a fraction of the whitespace-
    stripped trajectory.  None when it never appears verbatim."""
    a, t = _norm(answer), _norm(text)
    i = t.find(a) if a else -1
    return None if i < 0 else i / max(len(t), 1)


class GoldIndex:
    """8- and 30-gram sets per question, computed once."""

    def __init__(self, questions: dict[str, dict]):
        self.q = questions
        self._c: dict[tuple[str, int], set] = {}

    def g(self, qid: str, n: int) -> set[tuple]:
        k = (qid, n)
        if k not in self._c:
            self._c[k] = grams(toks(self.q[qid]["meta"].get("gold_solution", "")), n)
        return self._c[k]


def score_rows(rows: list[tuple[str, str, int]], gi: GoldIndex) -> dict:
    """rows: (qid, text, n_tokens)."""
    n = len(rows)
    if not n:
        return {"n": 0}
    pos = [p for p in (answer_first_pos(t, gi.q[q]["final_answer"]) for q, t, _ in rows)
           if p is not None]
    recalls, copies = [], 0
    for q, t, _ in rows:
        T = toks(t)
        g8 = gi.g(q, 8)
        r = len(grams(T, 8) & g8) / len(g8) if g8 else 0.0
        recalls.append(r)
        if r >= NEAR_COPY_RECALL or (grams(T, NEAR_COPY_NGRAM) & gi.g(q, NEAR_COPY_NGRAM)):
            copies += 1
    return {
        "n": n,
        "questions": len({q for q, _, _ in rows}),
        "len_mean": round(st.mean(x[2] for x in rows)),
        "ans_at_10": round(sum(p < 0.10 for p in pos) / len(pos), 4) if pos else None,
        "ans_at_25": round(sum(p < 0.25 for p in pos) / len(pos), 4) if pos else None,
        "show_that": round(sum(bool(SHOW.search(t)) for _, t, _ in rows) / n, 4),
        "gold_recall_median": round(st.median(recalls), 4),
        "near_copy": round(copies / n, 4),
        "refN": round(sum(bool(NARROW.search(t)) for _, t, _ in rows) / n, 4),
        "refB": round(sum(bool(BROAD.search(t)) for _, t, _ in rows) / n, 4),
    }


def load_bridges(run: Path) -> list[dict]:
    """bridges.jsonl carries metadata only — the TEXT lives in the generation
    pool and must be joined on (qid, sample_idx)."""
    meta_p = run / "iter_0/improve/bridge/bridges.jsonl"
    if not meta_p.exists():
        return []
    text: dict[tuple[str, int], str] = {}
    for f in run.glob("iter_0/improve/pool/bridge*/out_*.jsonl"):
        for line in f.open():
            d = json.loads(line)
            qid = d["rid"].split(":")[0]
            for i, s in enumerate(d["samples"]):
                text.setdefault((qid, i), s["text"])
    out = []
    for line in meta_p.open():
        x = json.loads(line)
        x["text"] = text.get((x["qid"], x["sample_idx"]), "")
        out.append(x)
    return out


def load_candidates(run: Path) -> list[tuple[str, str, int]]:
    """Verifier-correct improve candidates.  self_resample does NOT prefill
    `correct`, so for that operator the correct set is recovered from the
    filters stage's candidate_scores keys."""
    imp = run / "iter_0/improve/improved.jsonl"
    scores = run / "iter_0/filtered/candidate_scores.jsonl"
    rows, needs_join = [], None
    for line in imp.open():
        d = json.loads(line)
        if d.get("correct") is None:
            if needs_join is None:
                needs_join = ({json.loads(l)["key"] for l in scores.open()}
                              if scores.exists() else set())
            ok = f"{d['qid']}:{d['base_sample_idx']}:{d['attempt_idx']}" in needs_join
        else:
            ok = bool(d["correct"])
        if ok:
            rows.append((d["qid"], d["continuation_text"], len(d["continuation_token_ids"])))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--questions",
                    default="data/cliff_sets/openr1_default_cliff450_k16_with_gold.jsonl")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    Qs = {json.loads(l)["qid"]: json.loads(l) for l in open(args.questions)}
    gi = GoldIndex(Qs)
    cols = ("n", "questions", "len_mean", "ans_at_10", "ans_at_25", "show_that",
            "gold_recall_median", "near_copy", "refN", "refB")
    hdr = f"  {'row':<34}" + "".join(f"{c:>10}" for c in cols[:3]) + \
          "".join(f"{c:>11}" for c in cols[3:])

    report: dict[str, dict] = {}
    for r in args.runs:
        run = Path(r)
        if not (run / "iter_0").is_dir():
            continue
        print("=" * 132)
        print(f"### {run.name}")
        print(hdr)
        rows: dict[str, dict] = {}

        bs = load_bridges(run)
        if bs:
            B = lambda f: [(x["qid"], x["text"], x["n_tokens"]) for x in bs if f(x)]  # noqa: E731
            for lab, sel in (
                ("bridge: all generated", lambda x: True),
                ("bridge: verifier-correct", lambda x: x["correct"]),
                ("bridge: correct, no ref mention", lambda x: x["correct"] and not NARROW.search(x["text"])),
                ("bridge: correct, ref mention", lambda x: x["correct"] and NARROW.search(x["text"])),
                ("bridge: KEPT = LoRA fit target", lambda x: x["kept"]),
            ):
                rows[lab] = score_rows(B(sel), gi)

        rows["candidates: correct (bare x)"] = score_rows(load_candidates(run), gi)

        for lab, s in rows.items():
            if not s.get("n"):
                continue
            line = f"  {lab:<34}{s['n']:>10}{s['questions']:>10}{s['len_mean']:>10}"
            for c in cols[3:]:
                v = s[c]
                line += ("{:>11}".format("—") if v is None else
                         (f"{v:>11.3f}" if c == "gold_recall_median" else f"{v:>10.1%} "))
            print(line)
        report[run.name] = rows
        print()

    # reference row: the gold solutions themselves (ans_at_10 baseline)
    qs = sorted(Qs)
    gold_rows = [(q, Qs[q]["meta"].get("gold_solution", ""),
                  len(toks(Qs[q]["meta"].get("gold_solution", "")))) for q in qs
                 if Qs[q]["meta"].get("gold_solution")]
    g = score_rows(gold_rows, gi)
    print("=" * 132)
    print("### baseline: the gold solutions themselves (gold_recall/near_copy are 1.0 by construction)")
    print(hdr)
    print(f"  {'gold solution text':<34}{g['n']:>10}{g['questions']:>10}{g['len_mean']:>10}"
          f"{g['ans_at_10']:>10.1%} {g['ans_at_25']:>10.1%} {g['show_that']:>10.1%} "
          f"{g['gold_recall_median']:>11.3f}{g['near_copy']:>10.1%} {g['refN']:>10.1%} {g['refB']:>10.1%} ")
    report["_gold_baseline"] = g

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
