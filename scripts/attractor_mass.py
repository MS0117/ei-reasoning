"""Attractor mass — the continuous cliff endpoint.

A cliff is typically a CONFIDENT WRONG attractor: ~70% of the base policy's
samples commit to one wrong answer (measured). pass@k stays 0 long after
training starts helping, but the attractor's share P(modal wrong) moves — so
this is the powered per-question endpoint for the A/B transfer experiment
(docs/objective_decision_20260823.md §3 measurement panel).

Per question, over its n graded samples (partition/verdicts.jsonl schema):
  pass_rate      fraction correct
  p_top1         share of ALL samples on the modal wrong extracted answer
  p_top2         share on the two most common wrong answers (catches mass
                 sliding from the 1st to the 2nd wrong answer under a negative
                 term — the circularity guard the panel demanded)
  n_wrong_kinds  distinct wrong answers (None/unparsed excluded)

Usage (CPU):
  # one run
  .venv/bin/python scripts/attractor_mass.py --verdicts <...>/verdicts.jsonl \
      [--qids-file cliff_split.json:B] [--out attractor.json]
  # paired before/after (e.g. base re-roll floor vs post-train re-roll)
  .venv/bin/python scripts/attractor_mass.py --verdicts before.jsonl \
      --compare after.jsonl [--qids-file split.json:B]

--qids-file accepts path[:KEY] where KEY is A|B|exclude (default: all qids in file).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from statistics import mean

from expert_iter.records import VerdictRecord
from expert_iter.utils import write_json


def load_table(paths: list[str], qids: set[str] | None) -> dict[str, dict]:
    by_qid: dict[str, list[VerdictRecord]] = defaultdict(list)
    for p in paths:
        for v in VerdictRecord.load_jsonl(p):
            if qids is None or v.qid in qids:
                by_qid[v.qid].append(v)
    table = {}
    for qid, vs in by_qid.items():
        n = len(vs)
        wrong = Counter(v.extracted_answer for v in vs if not v.correct and v.extracted_answer is not None)
        top = wrong.most_common(2)
        table[qid] = {
            "n": n,
            "pass_rate": sum(v.correct for v in vs) / n,
            "modal_wrong": top[0][0] if top else None,
            "p_top1": (top[0][1] / n) if top else 0.0,
            "p_top2": (sum(c for _, c in top) / n) if top else 0.0,
            "n_wrong_kinds": len(wrong),
        }
    return table


def aggregate(table: dict[str, dict]) -> dict:
    if not table:
        return {}
    rows = table.values()
    return {
        "n_questions": len(table),
        "mean_pass_rate": round(mean(r["pass_rate"] for r in rows), 4),
        "frac_pass_gt0": round(mean(r["pass_rate"] > 0 for r in rows), 4),
        "mean_p_top1": round(mean(r["p_top1"] for r in rows), 4),
        "mean_p_top2": round(mean(r["p_top2"] for r in rows), 4),
        "frac_attractor_ge_half": round(mean(r["p_top1"] >= 0.5 for r in rows), 4),
    }


def sign_test(deltas: list[float]) -> tuple[int, int, float]:
    """two-sided exact sign test on nonzero deltas; returns (n_neg, n_nonzero, p)."""
    nz = [d for d in deltas if d != 0]
    k = sum(d < 0 for d in nz)
    n = len(nz)
    if n == 0:
        return 0, 0, float("nan")
    lo = min(k, n - k)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)
    return k, n, p


def _load_qids(spec: str | None) -> set[str] | None:
    if not spec:
        return None
    path, _, key = spec.partition(":")
    d = json.loads(Path(path).read_text())
    if isinstance(d, list):
        return set(d)
    if key:
        return set(d[key])
    out: set[str] = set()
    for k in ("A", "B"):
        out |= set(d.get(k, []))
    return out or set(d.get("exclude", []))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verdicts", nargs="+", required=True, help="verdicts.jsonl path(s); several are merged")
    ap.add_argument("--compare", nargs="+", default=None, help="second verdicts set -> paired per-qid deltas")
    ap.add_argument("--qids-file", default=None, help="path[:A|B|exclude] to restrict qids")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    qids = _load_qids(args.qids_file)
    base = load_table(args.verdicts, qids)
    print(f"[attractor] base: {json.dumps(aggregate(base))}")

    result: dict = {"base": {"per_qid": base, "aggregate": aggregate(base)}}
    if args.compare:
        other = load_table(args.compare, qids)
        common = sorted(set(base) & set(other))
        deltas = {m: [other[q][m] - base[q][m] for q in common]
                  for m in ("p_top1", "p_top2", "pass_rate")}
        print(f"[attractor] compare: {json.dumps(aggregate(other))}")
        comp_stats = {}
        for m, ds in deltas.items():
            k, n, p = sign_test(ds)
            comp_stats[m] = {"n_paired": len(common), "mean_delta": round(mean(ds), 4) if ds else None,
                             "n_down": k, "n_nonzero": n, "sign_p": round(p, 6) if n else None}
            print(f"[attractor] Δ{m}: mean {comp_stats[m]['mean_delta']:+.4f}  "
                  f"down {k}/{n}  sign-test p={comp_stats[m]['sign_p']}")
        result["compare"] = {"per_qid": other, "aggregate": aggregate(other), "paired": comp_stats}

    if args.out:
        write_json(args.out, result)
        print(f"[attractor] wrote {args.out}")


if __name__ == "__main__":
    main()
