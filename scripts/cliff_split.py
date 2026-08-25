"""Stratified, deterministic A/B split of the cliff questions.

A = cliffs whose rescued trajectories may enter training.
B = held-out cliffs: improve runs on them, but NO example ever reaches the
trainer (wire the output file into data.exclude_train_qids). Transfer is read
on B, against the base re-roll floor (scripts/cliff_reroll.py).

Strata (assignment priority, so the same qid lands in exactly one):
  base_pass     re-rolls to >0 correct under the base at n=32 (mean-reversion
                risk group; needs --reroll-summary)
  converted     has >=1 kept rescued success (filtered/kept.jsonl)
  unconverted   improve produced candidates but none survived
  never_bridged improve produced no candidates at all

Each stratum is split B-first by a stable hash of (seed, qid), so the split is
reproducible and independent of file order.

Usage (CPU):
  .venv/bin/python scripts/cliff_split.py --run-dir runs/<frozen L2 run> \
      [--b-frac 0.5] [--seed 17] [--reroll-summary <run>/iter_0/reroll/summary.json] \
      [--out <run>/iter_0/cliff_split.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expert_iter.records import ImprovedCandidate, UnsolvedQuestion
from expert_iter.utils import stable_hash, write_json


def assign_strata(cliff_qids, converted, has_cands, base_pass):
    strata: dict[str, list[str]] = {"base_pass": [], "converted": [], "unconverted": [], "never_bridged": []}
    for q in cliff_qids:
        if q in base_pass:
            strata["base_pass"].append(q)
        elif q in converted:
            strata["converted"].append(q)
        elif q in has_cands:
            strata["unconverted"].append(q)
        else:
            strata["never_bridged"].append(q)
    return strata


def split_stratum(qids: list[str], b_frac: float, seed: int) -> tuple[list[str], list[str]]:
    order = sorted(qids, key=lambda q: stable_hash("cliff_split", seed, q))
    n_b = round(len(order) * b_frac)
    return order[n_b:], order[:n_b]          # (A, B)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--b-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--reroll-summary", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    it_dir = Path(args.run_dir) / f"iter_{args.iteration}"
    cliff_qids = sorted(u.qid for u in UnsolvedQuestion.load_jsonl(it_dir / "partition" / "unsolved.jsonl"))
    converted = {c.qid for c in ImprovedCandidate.load_jsonl(it_dir / "filtered" / "kept.jsonl")}
    has_cands = {c.qid for c in ImprovedCandidate.load_jsonl(it_dir / "improve" / "improved.jsonl")}
    base_pass: set[str] = set()
    if args.reroll_summary:
        summ = json.loads(Path(args.reroll_summary).read_text())
        for counts in summ["correct_counts"].values():
            base_pass |= {q for q, c in counts.items() if c > 0}
        base_pass &= set(cliff_qids)

    strata = assign_strata(cliff_qids, converted, has_cands, base_pass)
    A: list[str] = []
    B: list[str] = []
    stats = {}
    for name, qids in strata.items():
        a, b = split_stratum(qids, args.b_frac, args.seed)
        A += a
        B += b
        stats[name] = {"n": len(qids), "n_A": len(a), "n_B": len(b)}

    out = Path(args.out) if args.out else it_dir / "cliff_split.json"
    write_json(out, {
        "seed": args.seed, "b_frac": args.b_frac,
        "reroll_summary": args.reroll_summary,
        "strata": stats,
        "A": sorted(A), "B": sorted(B),
        # data.exclude_train_qids reads this key: B examples never reach train
        "exclude": sorted(B),
    })
    print(f"[cliff_split] {len(cliff_qids)} cliffs -> A={len(A)} B={len(B)}  strata={stats}")
    print(f"[cliff_split] wrote {out} — wire it into --override data.exclude_train_qids={out}")


if __name__ == "__main__":
    main()
