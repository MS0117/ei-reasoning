"""Build the cliff-enriched question set for the L-ladder experiments.

Composes a 2000-question pool from the TWO OpenR1 halves:

  --default-table  the curated `default` config's passrate table (k=8; the
                   project's home distribution — every preset/toy/committed
                   cliff set lives here). Supplies ALL its cliffs (=the
                   committed 137, continuity), its frontier, and the whole
                   solved slice.
  --extended-table the non-curated `extended` config's full-sweep table
                   (k=16, mostly cn_k12, flagged easier/lower-quality by the
                   dataset card). Used ONLY to top up the cliff (and, if
                   needed, frontier) strata to quota — never for solved rows,
                   so the L_S data stays in the curated distribution.

Composition (defaults): 400 cliffs (137 default + 263 clean extended 0/16)
+ 150 frontier (106 default + 44 extended) + 1450 solved (default only).
Cross-config text duplicates share a qid (content hash) and are deduped with
default taking priority. meta.pool records each row's origin half.

Why enrich: cliffs are ~7% of a natural default sample -> B ~ 68 after the
A/B split resolves only ~5pp attractor-mass effects; 400 cliffs -> B ~ 200
resolves ~3pp at power 0.9 (scripts/power_table.py) at zero extra rollout
cost; only improve/reroll scale with cliff count. Deliberate curation — state
it in the paper (cliff-enriched, 20%).

Usage (CPU, seconds):
  .venv/bin/python data/make_enriched_set.py \
      --default-table runs/passrate/default_openr1_passrate_20260811_153259/openr1-math-220k_default_passrate.jsonl \
      --extended-table runs/passrate/friend_openr1_extended/openr1-math-220k_extended_passrate.jsonl \
      [--n-total 2000 --n-cliff 400 --n-frontier 150] [--seed 17] \
      [--out data/cliff_sets/openr1_hybrid_c400_n2000.jsonl]
Then point the run config at it:
  data.adapter=local_jsonl data.adapter_args.path=<out>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from expert_iter.records import QuestionRecord
from expert_iter.utils import stable_hash, write_jsonl
from expert_iter.verifier import MathVerifier


def pick(qids: list[str], n: int, seed: int, stratum: str) -> list[str]:
    return sorted(qids, key=lambda q: stable_hash("enriched_set", seed, stratum, q))[:n]


def _load_table(path: str) -> dict[str, dict]:
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[r["qid"]] = r
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--default-table", required=True,
                    help="curated `default` config passrate table (home distribution)")
    ap.add_argument("--extended-table", required=True,
                    help="non-curated `extended` config full-sweep table (cliff top-up only)")
    ap.add_argument("--n-total", type=int, default=2000)
    ap.add_argument("--n-cliff", type=int, default=400)
    ap.add_argument("--n-frontier", type=int, default=150)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="data/cliff_sets/openr1_hybrid_c400_n2000.jsonl")
    args = ap.parse_args(argv)
    if args.n_cliff + args.n_frontier >= args.n_total:
        ap.error("n_cliff + n_frontier must be < n_total")

    dflt = _load_table(args.default_table)
    ext = _load_table(args.extended_table)

    def bucket(rows, cls, *, clean=False):
        out = [q for q, r in rows.items() if r["class"] == cls
               and (not clean or int(r.get("n_truncated", 0)) == 0)]
        return out

    taken: set[str] = set()

    def take(qids: list[str], n: int, stratum: str, pool: str) -> list[tuple[str, str]]:
        picked = []
        for q in pick([q for q in qids if q not in taken], n, args.seed, f"{stratum}:{pool}"):
            taken.add(q)
            picked.append((q, pool))
        return picked

    # cliffs: ALL default cliffs (= the committed 137, continuity with every
    # prior toy/diagnostic), topped up with clean extended 0/16 cliffs
    cliffs = take(bucket(dflt, "cliff"), args.n_cliff, "cliff", "default")
    cliffs += take(bucket(ext, "cliff", clean=True), args.n_cliff - len(cliffs), "cliff", "extended")
    # frontier: default first (home distribution), extended fill
    frontier = take(bucket(dflt, "frontier"), args.n_frontier, "frontier", "default")
    frontier += take(bucket(ext, "frontier"), args.n_frontier - len(frontier), "frontier", "extended")
    # solved: default ONLY — the L_S data must stay in the curated distribution
    n_solved = args.n_total - len(cliffs) - len(frontier)
    solved = take(bucket(dflt, "solved"), n_solved, "solved", "default")

    for name, want, got in (("cliff", args.n_cliff, cliffs), ("frontier", args.n_frontier, frontier),
                            ("solved", n_solved, solved)):
        if len(got) < want:
            raise SystemExit(f"[enriched] stratum {name}: wanted {want}, tables gave {len(got)}")

    verifier = MathVerifier()
    out_rows: list[QuestionRecord] = []
    n_unparsable = 0
    for qid, pool in cliffs + frontier + solved:
        r = (dflt if pool == "default" else ext)[qid]
        if not verifier.gold_parsable(r["final_answer"]):
            n_unparsable += 1
            continue
        out_rows.append(QuestionRecord(
            qid=qid, question=r["question"], final_answer=r["final_answer"], domain="math",
            meta={
                "hf_name": "open-r1/OpenR1-Math-220k", "pool": pool,
                "uuid": r.get("uuid"), "row_idx": r.get("row_idx"),
                "source": r.get("source"), "problem_type": r.get("problem_type"),
                "gold_solution": r.get("solution") or "",
                "passrate_c": r.get("c"), "passrate_k": r.get("k"),
                "passrate_class": r["class"],
            },
        ))

    out = Path(args.out)
    write_jsonl(out, (q.to_dict() for q in out_rows))
    stats = Counter((q.meta["passrate_class"], q.meta["pool"]) for q in out_rows)
    n_gold = sum(1 for q in out_rows if q.meta["gold_solution"])
    print(f"[enriched] wrote {len(out_rows)} questions -> {out}")
    print(f"[enriched] composition (class, pool): {dict(stats)}")
    print(f"[enriched] gold_solution present {n_gold}/{len(out_rows)} | unparsable dropped {n_unparsable}")
    print(f"[enriched] wire up: --override data.adapter=local_jsonl "
          f"--override data.adapter_args.path={out}")


if __name__ == "__main__":
    main()
