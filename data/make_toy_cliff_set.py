"""Build the 450-candidate toy-cliff question set (successor to the committed 137).

WHY A NEW SET.  The committed set (`openr1_qwen3-4b-2507_n2000_with_gold.jsonl`,
137 questions -> 107 cliffs) is underpowered for its own effect sizes: the arm
spread on conversion is ~9pp, and McNemar at N=107 with the measured 25.4%
discordance reaches power 0.80 only at ~14pp.  Measured power for the headline
BRIDGE-vs-gold effect (11.2pp): N=107 -> 0.48, N=200 -> 0.78, N=300 -> 0.92.

WHY 0/16 AND NOT 0/8.  The committed 137 were selected c=0 at k=8; re-drawing 8
samples kept only 107 (78.1%).  Selecting at k=16 instead:

  * survives a fresh k=8 partition far better — of the 137, the 87 that are also
    0/16 kept 79 (90.8%) while the 50 that are not kept 28 (56.0%);
  * carries the discriminative signal.  Of the toy 107, the 79 that are 0/16
    separate the arms (conversion .342-.456) while the other 28 sit at a ceiling
    where every good arm ties (.786-.821).  The easy ones are dilution.

SOURCE.  runs/passrate/friend_openr1_default — the FULL default-config sweep
(59,344 questions, k=16, Qwen3-4B-Instruct-2507), 2,627 cliffs.  Default config
only: `extended` is non-curated (mostly cn_k12, flagged easier/lower-quality by
the dataset card) and its row_idx space differs, so it is never mixed in here.

COMPOSITION.  74 continuity questions (the toy-107 members that are also 0/16
AND carry a real gold solution) + fresh 0/16 cliffs to 450 candidates.

SIZING.  A 20-candidate probe of this set kept 15/20 = 75% through a fresh k=8
partition (continuity 4/5, fresh-only 11/15) — well under the 90.8% the legacy
double-selected questions suggested, exactly as expected for candidates screened
only at k=16.  450 -> ~338 cliffs at 75%, [238, 400] over the n=20 CI.  The
run's own rollout decides and the realized N is what gets reported.

Oversize deliberately: rollout cost scales with CANDIDATES (cheap) while the
expensive improve stage scales with SURVIVING CLIFFS, and questions cannot be
added later — a changed data path makes --reuse-rollout reject the shared
rollout, forcing every arm to re-run.

Unlike the committed set, meta records `config` and `split`: without them a
positional re-join by row_idx silently defaults to `default` (data.py:378 — and
row_idx 46513 is a different problem in `default` vs `extended`).

Seeded and prefix-stable: a larger --n-total is a strict superset of a smaller
one (the fresh pool is sorted by qid, then shuffled with --seed, then sliced).

Usage (CPU, seconds):
  .venv/bin/python data/make_toy_cliff_set.py
  .venv/bin/python data/make_toy_cliff_set.py --n-total 450 --seed 17
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from expert_iter.utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from expert_iter.verifier import MathVerifier  # noqa: E402

TABLE = "runs/passrate/friend_openr1_default/openr1-math-220k_default_passrate.jsonl"
LEGACY = "data/cliff_sets/openr1_qwen3-4b-2507_n2000_with_gold.jsonl"
TOY107 = "docs/results/toy_cliff/shared_rollout_cliff_qids.jsonl"
OUT = "data/cliff_sets/openr1_default_cliff450_k16_with_gold.jsonl"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
HF_NAME = "open-r1/OpenR1-Math-220k"

# A gold solution this short is a stub ("Reference answer: 3"), not a worked
# solution — the bridge operator conditions on y*, so a stub gives it nothing.
MIN_GOLD_CHARS = 200


def _is_stub(sol: str) -> bool:
    s = (sol or "").strip()
    return len(s) < MIN_GOLD_CHARS or s.lower().startswith("reference answer")


def _record(row: dict, *, continuity: bool) -> dict:
    return {
        "qid": row["qid"],
        "question": row["question"],
        "final_answer": row["final_answer"],
        "domain": "math",
        "meta": {
            "hf_name": HF_NAME,
            "config": "default",          # never omit: a row_idx re-join without
            "split": "train",             # these silently assumes `default`
            "row_idx": int(row["row_idx"]),
            "uuid": row.get("uuid"),
            "source": row.get("source"),
            "problem_type": row.get("problem_type"),
            "gold_solution": row["solution"],
            "passrate_c": int(row["c"]),
            "passrate_k": int(row["k"]),
            "passrate_run": "friend_openr1_default",
            "passrate_model": MODEL,
            "n_truncated": int(row.get("n_truncated", 0)),
            "continuity_107": continuity,  # also a member of the legacy toy 107
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--n-total", type=int, default=450)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = list(read_jsonl(args.table))
    cliffs = [r for r in rows if int(r["c"]) == 0]
    toy107 = {json.loads(l)["qid"] for l in open(TOY107)}
    legacy = {json.loads(l)["qid"] for l in open(LEGACY)}

    verifier = MathVerifier()
    n_stub = n_unparsable = 0
    usable: list[dict] = []
    for r in cliffs:
        if _is_stub(r["solution"]):
            n_stub += 1
            continue
        if not verifier.gold_parsable(r["final_answer"]):
            n_unparsable += 1
            continue
        usable.append(r)

    keep = [r for r in usable if r["qid"] in toy107]           # continuity
    pool = [r for r in usable if r["qid"] not in legacy]       # fresh only
    pool.sort(key=lambda r: r["qid"])                          # deterministic order
    random.Random(args.seed).shuffle(pool)
    need = max(0, args.n_total - len(keep))
    fresh = pool[:need]

    out = [_record(r, continuity=True) for r in keep] + \
          [_record(r, continuity=False) for r in fresh]
    out.sort(key=lambda r: r["qid"])
    assert len({r["qid"] for r in out}) == len(out), "duplicate qid"

    write_jsonl(Path(args.out), out)

    manifest = {
        "out": args.out,
        "built_by": "data/make_toy_cliff_set.py",
        "source_table": args.table,
        "source_sweep": {"hf_name": HF_NAME, "config": "default", "split": "train",
                         "model": MODEL, "k": 16, "rows": len(rows)},
        "criterion": "c == 0 at k=16 (base policy gets 0/16 correct)",
        "seed": args.seed,
        "counts": {
            "cliffs_in_table": len(cliffs),
            "dropped_gold_stub": n_stub,
            "dropped_gold_unparsable": n_unparsable,
            "usable": len(usable),
            "continuity_from_toy107": len(keep),
            "fresh_sampled": len(fresh),
            "total_candidates": len(out),
        },
        "legacy": {
            "set": LEGACY, "questions": len(legacy), "toy_cliffs": len(toy107),
            "toy_cliffs_that_are_0_of_16": len(toy107 & {r["qid"] for r in usable}),
        },
        "expected_cliffs_after_own_rollout": {
            "retention_estimate": 0.75,
            "note": "MEASURED on a 20-candidate probe of this very set "
                    "(runs/toy_cliff_2_smoke, 2026-08-30): 15/20 stayed 0/8 — "
                    "continuity 4/5, fresh-only 11/15. The legacy figure 0.908 "
                    "(79/87) came from questions selected 0/8 AND 0/16 and is "
                    "optimistic for fresh-only-0/16 candidates, which is why "
                    "this set is oversized. n=20, so the 95% CI is [.53, .89]; "
                    "the realized N is what gets reported.",
            "point": round(len(out) * 0.75),
            "ci_range": [round(len(out) * 0.53), round(len(out) * 0.89)],
        },
    }
    write_json(Path(args.out).with_name(Path(args.out).stem.replace(
        "_with_gold", "") + "_manifest.json"), manifest)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
