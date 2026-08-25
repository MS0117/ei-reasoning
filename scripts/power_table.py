"""Power table for the A/B cliff transfer experiment.

Answers, BEFORE the experiment: "with N_B held-out cliffs and n samples per
question, what effect size Δ can the paired sign test on each endpoint detect
with what probability?" — so that a flat L3 readout is a conclusion ("no effect
larger than X") instead of an ambiguous null (decision doc §7 risk).

Simulation: per sim, draw N_B questions (with replacement) from the empirical
per-question distribution (an attractor_mass.py JSON, or built-in defaults
matching the measured cliff anatomy: p_top1 mean ~0.65, pass ~0); "before"
counts ~ Binom(n, p_i), "after" ~ Binom(n, p_i ∓ Δ); endpoint = two-sided
exact sign test on per-question deltas at alpha.

Usage (CPU, seconds):
  .venv/bin/python scripts/power_table.py [--attractor-json attractor.json[:B]]
      [--n-b 35 68 100 200] [--delta-pp 1 2 3 5] [--n 32] [--sims 1000]
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path


def sign_p(deltas: list[float]) -> float:
    nz = [d for d in deltas if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    k = min(sum(d < 0 for d in nz), sum(d > 0 for d in nz))
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def power(pop: list[float], n_b: int, n: int, delta: float, direction: int,
          sims: int, alpha: float, rng: random.Random) -> float:
    hits = 0
    for _ in range(sims):
        ps = [pop[rng.randrange(len(pop))] for _ in range(n_b)]
        deltas = []
        for p in ps:
            before = sum(rng.random() < p for _ in range(n)) / n
            q = min(1.0, max(0.0, p + direction * delta))
            after = sum(rng.random() < q for _ in range(n)) / n
            deltas.append(after - before)
        if sign_p(deltas) < alpha:
            hits += 1
    return hits / sims


def _table(name, pop, direction, args, rng):
    print(f"\n## endpoint: {name} (per-question share, n={args.n} samples, "
          f"paired sign test, alpha={args.alpha}, {args.sims} sims)")
    header = "| N_B | " + " | ".join(f"Δ={d}pp" for d in args.delta_pp) + " |"
    print(header)
    print("|" + "---|" * (len(args.delta_pp) + 1))
    rows = {}
    for n_b in args.n_b:
        cells = []
        for d in args.delta_pp:
            pw = power(pop, n_b, args.n, d / 100.0, direction, args.sims, args.alpha, rng)
            cells.append(pw)
        rows[n_b] = cells
        print(f"| {n_b} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attractor-json", default=None,
                    help="attractor_mass.py output (path[:A|B]); default = built-in cliff-shaped population")
    ap.add_argument("--n-b", type=int, nargs="+", default=[35, 68, 100, 200])
    ap.add_argument("--delta-pp", type=float, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    if args.attractor_json:
        path, _, key = args.attractor_json.partition(":")
        d = json.loads(Path(path).read_text())["base"]["per_qid"]
        rows = list(d.values())
        pop_top1 = [r["p_top1"] for r in rows]
        pop_pass = [r["pass_rate"] for r in rows]
        print(f"[power] empirical population: {len(rows)} questions from {path}")
    else:
        # built-in population matching the measured cliff anatomy: attractor
        # share ~0.65 mean with spread; pass ~ the re-roll floor tail
        pop_top1 = [min(0.95, max(0.15, rng.gauss(0.65, 0.18))) for _ in range(500)]
        pop_pass = [0.0] * 380 + [rng.uniform(0.03, 0.15) for _ in range(120)]
        print("[power] built-in population (measured-shaped); rerun with "
              "--attractor-json after L2 re-rolls for the final table")

    out = {
        "p_top1_down": _table("attractor mass P(top-1 wrong), DECREASE", pop_top1, -1, args, rng),
        "pass_rate_up": _table("pass rate avg@n, INCREASE", pop_pass, +1, args, rng),
    }
    if args.out:
        from expert_iter.utils import write_json
        write_json(args.out, {"args": vars(args), "power": out})
        print(f"\n[power] wrote {args.out}")


if __name__ == "__main__":
    main()
