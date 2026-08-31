"""Restrict a finished rollout+partition to a pre-specified subset of its cliffs.

WHY.  Cliff count is decided by the run's own partition, not by us: 450
candidates screened at 0/16 yield whatever survives a fresh 0/8 draw.  Running
every arm on all of them is the default, but N buys power with a shallow slope
(BRIDGE-vs-gold at 11.4pp / 36.7% discordance: N=250 -> .84, N=338 -> .93) while
GPU cost is linear in cliffs.  This builds a reuse bundle whose `unsolved.jsonl`
holds a fixed random subset, so every arm improves the SAME N questions and the
result is a genuine N=<subset> experiment — the transient adapters are fit on
those questions only, not on the full set.

WHAT IT TOUCHES.  Only `unsolved.jsonl` (the improve stage's input) and the
`n_unsolved_questions` field of partition stats.  questions/, rollouts, verdicts
and solved are copied verbatim, and `config.yaml` is copied unchanged — so
data.adapter_args still names the full candidate file and every arm config works
with --reuse-rollout as-is, no edits.

filters.py computes cliff/conversion_rate as converted / len(unsolved), so the
denominator follows the subset automatically.

THE SUBSET RULE MUST BE OUTCOME-INDEPENDENT.  Run this BEFORE any arm's improve
stage: at that point only the rollout exists, so a seeded sample cannot be
tuned to a result.  The manifest records the rule, the seed and the qids.

Usage (CPU, seconds):
  .venv/bin/python data/make_cliff_subset_bundle.py \
      --src runs/toy_cliff_2/default_CONTROL_<ts> --n 250

  bash data/run_toy_cliff.sh -c data/configs/CONTROL.yaml -o runs/toy_cliff_2 -b \
      -- --reuse-rollout runs/toy_cliff_2/_subset250
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from expert_iter.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="finished run that owns the rollout")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None, help="default: <src parent>/_subset<n>")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out) if args.out else src.parent / f"_subset{args.n}"
    src_it, out_it = src / "iter_0", out / "iter_0"

    unsolved = list(read_jsonl(src_it / "partition" / "unsolved.jsonl"))
    if len(unsolved) < args.n:
        raise SystemExit(
            f"[subset] source has only {len(unsolved)} cliffs, cannot take {args.n}"
        )

    # Deterministic and outcome-independent: sort by qid, then seeded sample.
    by_qid = sorted(unsolved, key=lambda u: u["qid"])
    chosen = random.Random(args.seed).sample(by_qid, args.n)
    chosen.sort(key=lambda u: u["qid"])
    chosen_qids = [u["qid"] for u in chosen]

    verbatim = [
        ("config.yaml", src / "config.yaml", out / "config.yaml"),
        ("questions/train.jsonl", src / "questions" / "train.jsonl",
         out / "questions" / "train.jsonl"),
        ("questions/holdout.jsonl", src / "questions" / "holdout.jsonl",
         out / "questions" / "holdout.jsonl"),
        ("iter_0/rollout/rollouts.jsonl", src_it / "rollout" / "rollouts.jsonl",
         out_it / "rollout" / "rollouts.jsonl"),
        ("iter_0/partition/verdicts.jsonl", src_it / "partition" / "verdicts.jsonl",
         out_it / "partition" / "verdicts.jsonl"),
        ("iter_0/partition/solved.jsonl", src_it / "partition" / "solved.jsonl",
         out_it / "partition" / "solved.jsonl"),
    ]
    missing = [n for n, s, _ in verbatim if not s.exists()]
    if missing:
        raise SystemExit(f"[subset] source is incomplete, missing {missing}")

    for _, s, d in verbatim:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        marker = s.with_name(s.name + ".done")
        if marker.exists():
            shutil.copy2(marker, d.with_name(d.name + ".done"))

    write_jsonl(out_it / "partition" / "unsolved.jsonl", chosen)

    stats = json.loads((src_it / "partition" / "stats.json").read_text())
    stats["n_unsolved_questions"] = len(chosen)
    stats["subset_of"] = str(src)
    stats["subset_full_cliff_count"] = len(unsolved)
    write_json(out_it / "partition" / "stats.json", stats)

    manifest = {
        "out": str(out),
        "built_by": "data/make_cliff_subset_bundle.py",
        "source_run": str(src),
        "rule": "sort cliffs by qid, then random.Random(seed).sample(n) — fixed "
                "before any arm's improve stage ran, so it cannot depend on results",
        "seed": args.seed,
        "cliffs_in_source": len(unsolved),
        "cliffs_in_subset": len(chosen),
        "qids": chosen_qids,
    }
    write_json(out / "subset_manifest.json", manifest)

    print(f"[subset] {len(unsolved)} cliffs -> {len(chosen)}  ({out})")
    print(f"[subset] verbatim: questions/, rollouts, verdicts, solved, config.yaml")
    print(f"[subset] rewritten: partition/unsolved.jsonl, partition/stats.json")
    print(f"[subset] run arms with:  -- --reuse-rollout {out}")


if __name__ == "__main__":
    main()
