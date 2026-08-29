"""Stratified pass-rate mix: build an EI training set from joined pass-rate tables.

Cliff questions are ~3.6% of a math corpus, so a proportional sample spends
most of the rollout budget on questions the policy already solves 16/16 and
leaves almost nothing for the improvement operator to act on. This draws a
fixed quota per correct-count bucket instead, over-sampling the cliff ~12x.

    .venv/bin/python data/sample_mix.py \
        --input runs/passrate/friend_openr1_default/openr1-math-220k_default_passrate.jsonl \
        --input runs/passrate/friend_openr1_extended/openr1-math-220k_extended_passrate.jsonl \
        [--out-dir data/mixes] [--name openr1_mix8k] [--seed 17]
        [--mix "0-0=3400,1-2=2200,3-4=1100,5-8=700,9-15=400,16-16=200"]
        [--cliff-holdout 540] [--max-truncated 8]

Inputs are the tables written by data/join_passrate.py, given in PREFERENCE
order: a qid present in several is kept from the earliest one, and buckets are
filled from earlier inputs before later ones (openr1 `default` is the curated
half, so pass it first). The cliff bucket is exempt from that preference when
its quota plus the holdout consumes the whole pool — then both come from the
same hash-ordered split, which is what makes the holdout exchangeable with the
trained cliffs and the rescue-rate comparison meaningful.

Outputs (QuestionRecord JSONL, ready for the local_jsonl adapter):
  <name>.jsonl                 the mix; meta.gold_solution carries y*
  <name>_cliff_holdout.jsonl   reserved cliffs, DISJOINT from the mix — point
                               data.holdout_path at it with data.eval_holdout: 0
  <name>_manifest.json         realized per-bucket counts, sources, seed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from expert_iter.records import QuestionRecord  # noqa: E402
from expert_iter.utils import read_jsonl, stable_hash, write_json  # noqa: E402

DEFAULT_MIX = "0-0=3400,1-2=2200,3-4=1100,5-8=700,9-15=400,16-16=200"


def parse_mix(spec: str) -> list[tuple[int, int, int]]:
    """'0-0=3400,1-2=2200' -> [(lo, hi, quota), ...], validated non-overlapping."""
    out: list[tuple[int, int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rng, _, n = part.partition("=")
        lo, _, hi = rng.partition("-")
        try:
            lo_i, hi_i, n_i = int(lo), int(hi or lo), int(n)
        except ValueError:
            raise SystemExit(f"bad --mix entry {part!r}; expected LO-HI=COUNT")
        if lo_i > hi_i or n_i < 0:
            raise SystemExit(f"bad --mix entry {part!r}: need lo <= hi and count >= 0")
        out.append((lo_i, hi_i, n_i))
    for i, (lo, hi, _) in enumerate(out):
        for lo2, hi2, _ in out[i + 1:]:
            if lo <= hi2 and lo2 <= hi:
                raise SystemExit(f"--mix buckets {lo}-{hi} and {lo2}-{hi2} overlap")
    return out


def to_record(row: dict) -> QuestionRecord:
    """Joined pass-rate row -> QuestionRecord. The reference solution rides in
    meta.gold_solution (privileged: operators and scoring only, never a prompt),
    and the row_idx/config/split keys stay so the source row can be re-joined."""
    meta = {
        "hf_name": row.get("hf_name") or "open-r1/OpenR1-Math-220k",
        "config": row.get("sweep_config") or row.get("config"),
        "split": "train",
        "row_idx": row.get("row_idx"),
        "row_source": row.get("source") or "",
        "problem_type": row.get("problem_type"),
        "uuid": row.get("uuid"),
        "passrate_c": row["c"],
        "passrate_k": row["k"],
        "passrate_class": row["class"],
        "passrate_n_truncated": row["n_truncated"],
        "passrate_source": row["_source"],
    }
    if (row.get("solution") or "").strip():
        meta["gold_solution"] = row["solution"].strip()
    return QuestionRecord(
        qid=row["qid"], question=row["question"],
        final_answer=str(row["final_answer"]), domain="math", meta=meta,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", metavar="PATH",
                    help="joined pass-rate table (repeatable, in preference order)")
    ap.add_argument("--subset-of", metavar="MIX.jsonl",
                    help="downsample an ALREADY BUILT mix (QuestionRecord JSONL) "
                         "instead of drawing from pass-rate tables. Buckets come "
                         "from meta.passrate_c and are taken in the same hash "
                         "order, so the result is a strict subset of that mix — "
                         "a toy run then exercises the exact questions the full "
                         "run will use, and its reserved holdout stays disjoint "
                         "from both. Ignores --cliff-holdout: reuse the parent "
                         "mix's holdout file, which is already disjoint.")
    ap.add_argument("--out-dir", default="data/mixes")
    ap.add_argument("--name", default="openr1_mix8k")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--mix", default=DEFAULT_MIX, help=f"default: {DEFAULT_MIX}")
    ap.add_argument("--cliff-holdout", type=int, default=540,
                    help="cliff questions reserved OUT of the mix (0 disables)")
    ap.add_argument("--max-truncated", type=int, default=8,
                    help="cliff quality gate: drop cliffs with more truncated samples "
                         "than this (they failed for length, not for difficulty)")
    args = ap.parse_args(argv)

    buckets = parse_mix(args.mix)
    if bool(args.input) == bool(args.subset_of):
        raise SystemExit("pass exactly one of --input (build) or --subset-of (downsample)")

    if args.subset_of:
        return _subset(args, buckets)

    # Merge in preference order; first file wins a shared qid.
    merged: dict[str, dict] = {}
    for path in args.input:
        tag = Path(path).stem
        n_new = 0
        for row in read_jsonl(path):
            if row["qid"] in merged:
                continue
            row["_source"] = tag
            row["_rank"] = args.input.index(path)
            merged[row["qid"]] = row
            n_new += 1
        print(f"[mix] {path}: +{n_new} new qids (total {len(merged)})")
    rows = list(merged.values())

    def eligible(row: dict, lo: int, hi: int) -> bool:
        if not (lo <= row["c"] <= hi):
            return False
        if lo == 0 == hi:      # cliff quality gate
            return (row["n_truncated"] <= args.max_truncated
                    and (row.get("raw_pass_rate") or 0) == 0)
        return True

    picked: list[dict] = []
    holdout: list[dict] = []
    manifest_buckets = []
    for lo, hi, quota in buckets:
        pool = [r for r in rows if eligible(r, lo, hi)]
        reserve = args.cliff_holdout if lo == 0 == hi else 0
        # Preference order only matters when the pool exceeds what we take. If
        # quota+reserve consumes the pool, every row is used either way — and
        # ranking by preference would then push the whole reserve into the
        # least-preferred source, making the holdout systematically unlike the
        # trained cliffs. Hash order keeps the two exchangeable.
        exhausts = quota + reserve >= len(pool)
        if exhausts:
            ordered = sorted(pool, key=lambda r: stable_hash(args.seed, r["qid"]))
        else:
            ordered = sorted(pool, key=lambda r: (r["_rank"],
                                                  stable_hash(args.seed, r["qid"])))
        take, rest = ordered[:quota], ordered[quota:]
        if len(take) < quota:
            print(f"[mix] WARNING bucket c={lo}-{hi}: wanted {quota}, pool has "
                  f"{len(pool)} — taking all")
        picked.extend(take)
        if reserve:
            holdout.extend(rest[:reserve])
            if len(rest) < reserve:
                print(f"[mix] WARNING cliff holdout: wanted {reserve}, "
                      f"only {len(rest)} left after the quota")
        manifest_buckets.append({
            "c_range": [lo, hi], "quota": quota, "pool": len(pool),
            "taken": len(take), "holdout": len(holdout) - (len(holdout) - min(reserve, len(rest))) if reserve else 0,
            "by_source": dict(Counter(r["_source"] for r in take)),
            "order": "hash (pool exhausted)" if exhausts else "preference then hash",
        })

    picked_ids = {r["qid"] for r in picked}
    overlap = picked_ids & {r["qid"] for r in holdout}
    if overlap:
        raise SystemExit(f"BUG: {len(overlap)} qids in both the mix and the holdout")

    out_dir = Path(args.out_dir)
    mix_path = out_dir / f"{args.name}.jsonl"
    hold_path = out_dir / f"{args.name}_cliff_holdout.jsonl"
    QuestionRecord.dump_jsonl(mix_path, [to_record(r) for r in picked])
    if holdout:
        QuestionRecord.dump_jsonl(hold_path, [to_record(r) for r in holdout])

    src = Counter(r["_source"] for r in picked)
    manifest = {
        "name": args.name, "seed": args.seed, "mix": args.mix,
        "inputs": list(args.input), "max_truncated": args.max_truncated,
        "n_mix": len(picked), "n_cliff_holdout": len(holdout),
        "by_source": dict(src), "buckets": manifest_buckets,
        "n_with_gold_solution": sum(1 for r in picked if (r.get("solution") or "").strip()),
    }
    write_json(out_dir / f"{args.name}_manifest.json", manifest)

    print(f"\n[mix] wrote {len(picked)} -> {mix_path}")
    if holdout:
        print(f"[mix] wrote {len(holdout)} cliff holdout -> {hold_path}")
    print(f"[mix] sources: {dict(src)}")
    print(f"[mix] gold_solution present: {manifest['n_with_gold_solution']}/{len(picked)}")
    print(f"\n  {'c range':<10}{'quota':>7}{'pool':>8}{'taken':>7}   by source")
    for b in manifest_buckets:
        lo, hi = b["c_range"]
        print(f"  {f'{lo}-{hi}':<10}{b['quota']:>7}{b['pool']:>8}{b['taken']:>7}   {b['by_source']}")


def _subset(args, buckets) -> None:
    """Strict subset of an existing mix, same bucket structure, hash order."""
    parent = list(QuestionRecord.load_jsonl(args.subset_of))
    if not parent:
        raise SystemExit(f"{args.subset_of} is empty")
    missing = [r.qid for r in parent if r.meta.get("passrate_c") is None]
    if missing:
        raise SystemExit(
            f"{len(missing)} records lack meta.passrate_c (e.g. {missing[0]}) — "
            "--subset-of needs a mix built by this script"
        )

    picked, manifest_buckets = [], []
    for lo, hi, quota in buckets:
        pool = [r for r in parent if lo <= r.meta["passrate_c"] <= hi]
        ordered = sorted(pool, key=lambda r: stable_hash(args.seed, r.qid))
        take = ordered[:quota]
        if len(take) < quota:
            print(f"[mix] WARNING bucket c={lo}-{hi}: wanted {quota}, "
                  f"parent has {len(pool)} — taking all")
        picked.extend(take)
        manifest_buckets.append({
            "c_range": [lo, hi], "quota": quota, "pool": len(pool), "taken": len(take),
            "by_source": dict(Counter(r.meta.get("passrate_source", "?") for r in take)),
            "order": "hash (subset)",
        })

    out_dir = Path(args.out_dir)
    mix_path = out_dir / f"{args.name}.jsonl"
    QuestionRecord.dump_jsonl(mix_path, picked)
    src = Counter(r.meta.get("passrate_source", "?") for r in picked)
    write_json(out_dir / f"{args.name}_manifest.json", {
        "name": args.name, "seed": args.seed, "mix": args.mix,
        "subset_of": args.subset_of, "n_mix": len(picked),
        "by_source": dict(src), "buckets": manifest_buckets,
        "n_with_gold_solution": sum(1 for r in picked if r.meta.get("gold_solution")),
    })
    print(f"\n[mix] wrote {len(picked)} -> {mix_path}  (subset of {args.subset_of})")
    print(f"[mix] sources: {dict(src)}")
    print(f"\n  {'c range':<10}{'quota':>7}{'parent':>8}{'taken':>7}   by source")
    for b in manifest_buckets:
        lo, hi = b["c_range"]
        print(f"  {f'{lo}-{hi}':<10}{b['quota']:>7}{b['pool']:>8}{b['taken']:>7}   {b['by_source']}")


if __name__ == "__main__":
    main()
