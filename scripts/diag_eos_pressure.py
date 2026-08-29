"""Usage: .venv/bin/python scripts/diag_eos_pressure.py <diag_dir> [<diag_dir> ...]
           [--delta 0.02] [--json OUT.json]

How much unlikelihood gradient would L_N put on the TERMINAL (EOS) token?

Context: v1 negatives are whole failed trajectories, and the bounded-unlikelihood
gradient on the observed token's own logit is exactly p = exp(-ce) — so a token
the policy is confident about takes the LARGEST push down, while the delta clamp
(p > 1-delta) zeroes it entirely. Whether the terminal EOS of a wrong-answer
trajectory sits above or below that clamp decides whether "don't stop after
committing a wrong answer" is a live training signal or a no-op.
See docs/objective_loss_spec_20260825.md 4.3.

Reads the diag dumps (candidates.jsonl + tokens.jsonl, written by
scripts/diag_scaffold_credit.py), where `s` is the BASE surprisal of each token.
The terminal token of every sequence is taken as-is (no id hardcoding) and
p = exp(-s) is reported per correctness class, next to the trajectory's own mean
token probability so the EOS push can be compared with a typical token's.

CPU only. Caveat: these trajectories are adapter-sampled and base-scored, while
v1's negatives are the base's OWN rollouts — self-generated samples should sit
somewhat higher. Treat as a lower bound on p(EOS), i.e. an upper bound on how
much gradient survives the clamp.
"""
import argparse, collections, json, math, statistics as st
from pathlib import Path


def load(d: Path):
    """{key: {correct, eos_tok, eos_p, mean_p, n}} for one diag dir."""
    correct = {}
    for line in open(d / "candidates.jsonl"):
        r = json.loads(line)
        correct[r["key"]] = r.get("correct")
    last: dict[str, tuple[int, str, float]] = {}
    acc: dict[str, list[float]] = collections.defaultdict(list)
    for line in open(d / "tokens.jsonl"):
        r = json.loads(line)
        k, pos, s = r["key"], r["pos"], r["s"]
        acc[k].append(math.exp(-s))
        if k not in last or pos > last[k][0]:
            last[k] = (pos, r["tok"], s)
    out = {}
    for k, (_, tok, s) in last.items():
        if correct.get(k) is None:
            continue
        out[k] = {
            "correct": bool(correct[k]), "eos_tok": tok,
            "eos_p": math.exp(-s), "mean_p": st.mean(acc[k]), "n": len(acc[k]),
        }
    return out


def summarize(rows: list[dict], delta: float) -> dict:
    p = sorted(r["eos_p"] for r in rows)
    n = len(p)
    live = [x for x in p if x <= 1 - delta]
    return {
        "n": n,
        "median_eos_p": st.median(p),
        "p10_eos_p": p[n // 10],
        "min_eos_p": p[0],
        "n_clamped": n - len(live),
        "frac_clamped": (n - len(live)) / n,
        "frac_live": len(live) / n,
        # mean gradient magnitude on the EOS logit: p below the clamp, 0 above
        "mean_grad": sum(live) / n,
        "median_mean_token_p": st.median([r["mean_p"] for r in rows]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--delta", type=float, default=0.02,
                    help="train.sft.cliff.negative.delta (clamp: p <= 1-delta)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    report = {"delta": args.delta, "dirs": {}}
    for d in args.dirs:
        if not (d / "tokens.jsonl").exists():
            print(f"skip {d}: no tokens.jsonl")
            continue
        rows = list(load(d).values())
        toks = collections.Counter(r["eos_tok"] for r in rows)
        by = {"incorrect": [r for r in rows if not r["correct"]],
              "correct": [r for r in rows if r["correct"]]}
        entry = {"terminal_tokens": dict(toks.most_common(3)),
                 "classes": {k: summarize(v, args.delta) for k, v in by.items() if v}}
        report["dirs"][str(d)] = entry

        print(f"\n== {d}  (n={len(rows)}, terminal token(s): "
              f"{', '.join(f'{t!r}x{c}' for t, c in toks.most_common(3))})")
        print(f"  {'class':<10} {'n':>4} {'med p(EOS)':>11} {'p10':>7} {'min':>7} "
              f"{'clamped':>9} {'mean|grad|':>11} {'med tok p':>10}")
        for cls in ("incorrect", "correct"):
            s = entry["classes"].get(cls)
            if not s:
                continue
            print(f"  {cls:<10} {s['n']:>4} {s['median_eos_p']:>11.4f} "
                  f"{s['p10_eos_p']:>7.4f} {s['min_eos_p']:>7.4f} "
                  f"{s['frac_clamped']:>8.1%} {s['mean_grad']:>11.4f} "
                  f"{s['median_mean_token_p']:>10.4f}")

    # verdict across all dirs/classes: is the EOS gradient live anywhere?
    live = [s["frac_live"] for e in report["dirs"].values()
            for s in e["classes"].values()]
    if live:
        report["max_frac_live"] = max(live)
        print(f"\nmax fraction below the clamp (= gradient flows): {max(live):.1%}")
        print("→ " + ("LIVE: terminal-EOS suppression is a real, uncontrolled "
                      "intervention in v1 — make it an explicit arm"
                      if max(live) > 0.2 else
                      "mostly clamped: terminal-EOS suppression is ~a no-op"))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
