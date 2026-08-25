"""Effective cliff share of the CURRENT (legacy) loss — the S1' arm's rho.

The legacy loss normalizes by the global loss-token mass, so the cliff
(improved) examples' effective gradient share is their share of that mass —
NOT their example-count share (rescues are ~3x longer than solved responses).
S1' is defined as "same share as today, but stratified per-step supply", so its
rho must be this measured number, computed on the FROZEN L2 dataset.

Usage (CPU, after L2's build_dataset):
  .venv/bin/python scripts/rho_legacy.py --run-dir runs/<frozen L2 run>
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from expert_iter.config import Config
from expert_iter.records import SFTExample


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    cfg = Config.load(run_dir / "config.yaml")
    w = cfg.train.sft.region_weights
    rows = list(SFTExample.load_jsonl(
        run_dir / f"iter_{args.iteration}" / "dataset" / "train_sft.jsonl"))
    rows = [r for r in rows if len(r.input_ids) <= cfg.train.max_seq_len]

    mass = Counter()
    count = Counter()
    for r in rows:
        completion_w = w["solution"] if r.source == "solved" else w["continuation"]
        m = w["prompt"] * r.prompt_len + w["anchor"] * r.anchor_len + completion_w * r.completion_len
        mass[r.source] += m
        count[r.source] += 1

    total = sum(mass.values())
    print(f"[rho_legacy] rows: {dict(count)}  (after max_seq_len filter)")
    for src in mass:
        print(f"[rho_legacy] {src:9s}: count share {count[src] / len(rows):.4f}  "
              f"loss-mass share {mass[src] / total:.4f}")
    rho = mass.get("improved", 0.0) / total if total else 0.0
    print(f"\n[rho_legacy] S1' arm: --override train.sft.cliff.rho={rho:.4f}")
    if not 0 < rho < 1:
        print("[rho_legacy] WARNING: rho outside (0,1) — no improved rows or empty dataset?")


if __name__ == "__main__":
    main()
