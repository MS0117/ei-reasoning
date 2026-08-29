"""Regenerate the L3 arm comparison table from the run dirs on disk.

Reads, per arm run dir: metrics.jsonl (holdout envelope), iter_0/dataset/stats.json
(what was actually trained), iter_0/attractor_B_compare.json (the paired B
transfer readout written by scripts/attractor_mass.py), and the arm's frozen
config.yaml (rho / negative mode / cliff on-off). Prints a markdown table.

Usage (CPU, instant):
  .venv/bin/python scripts/l3_summary.py runs/L3_*            # all arms
  .venv/bin/python scripts/l3_summary.py runs/L3_* --md docs/L3_results_table.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run: Path) -> dict | None:
    cmp_path = run / "iter_0" / "attractor_B_compare.json"
    met_path = run / "metrics.jsonl"
    if met_path.exists():
        met = json.loads(met_path.read_text().strip().split("\n")[-1])
    else:
        # metrics.jsonl is written by the LOOP driver; an arm whose eval stage was
        # re-run standalone (e.g. a phase-scoped resume) only has the stage's own
        # output. Same holdout, same numbers — just not merged with improve stats.
        stage = run / "iter_0" / "eval" / "metrics.json"
        if not stage.exists():
            return None
        met = json.loads(stage.read_text())
    ds = json.loads((run / "iter_0" / "dataset" / "stats.json").read_text())
    from expert_iter.config import Config
    cfg = Config.load(run / "config.yaml")
    cl = cfg.train.sft.cliff
    row = {
        "arm": run.name.split("_")[1],
        "run": run.name,
        "cliff_on": cl.enabled,
        "rho": cl.rho if cl.enabled else None,
        "neg": cl.negative.mode if cl.enabled else "off",
        "mu": cl.negative.mu if (cl.enabled and cl.negative.mode != "off") else None,
        "n_improved": ds["sft_by_source"].get("improved", 0),
        "n_negative": ds["sft_by_source"].get("negative", 0),
        "n_solved": ds["sft_by_source"].get("solved", 0),
        "greedy": met.get("pass@1_greedy"),
        "avg8": met.get("avg@8"),
        "trunc8": met.get("truncated_rate@8"),
    }
    if cmp_path.exists():
        c = json.loads(cmp_path.read_text())
        pr = c["compare"]["paired"]
        base, comp = c["base"]["aggregate"], c["compare"]["aggregate"]
        row.update({
            "n_B": pr["p_top1"]["n_paired"],
            "attr_base": base["mean_p_top1"], "attr_arm": comp["mean_p_top1"],
            "d_attr": pr["p_top1"]["mean_delta"], "p_attr": pr["p_top1"]["sign_p"],
            "d_top2": pr["p_top2"]["mean_delta"], "p_top2": pr["p_top2"]["sign_p"],
            "pass_base": base["mean_pass_rate"], "pass_arm": comp["mean_pass_rate"],
            "d_pass": pr["pass_rate"]["mean_delta"], "p_pass": pr["pass_rate"]["sign_p"],
            # coverage = share of B cliffs with >=1 correct in the arm's 32 samples.
            # NOT compared against c["base"] here: the base floor merges two 32-sample
            # passes (64 draws), which inflates its >=1 rate. The matched-32 baseline
            # is each floor pass on its own (0.290 / 0.312 — also the noise scale).
            "cov": comp["frac_pass_gt0"],
        })
    return row


def fmt(v, pp=False, digits=3):
    if v is None:
        return "—"
    if pp:
        return f"{v * 100:+.1f}pp"
    if isinstance(v, float):
        return f"{v:.{digits}g}" if v < 0.01 else f"{v:.{digits}f}"
    return str(v)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md", default=None, help="also write the table to this markdown file")
    args = ap.parse_args(argv)

    rows = [r for r in (load(Path(p)) for p in sorted(args.runs)) if r]
    if not rows:
        raise SystemExit("no arm run dirs with metrics.jsonl found")

    out = []
    out.append("| arm | cliff term | trained rows (solved/rescue/neg) | "
               "B: Δattractor (p) | B: Δp_top2 (p) | B: Δavg@32 (p) | B: ≥1정답 비율 | "
               "holdout greedy / avg@8 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        term = "off (legacy loss)" if not r["cliff_on"] else \
            f"rho={r['rho']}" + (f", {r['neg']} mu={r['mu']}" if r["neg"] != "off" else "")
        data = f"{r['n_solved']} / {r['n_improved']} / {r['n_negative']}"
        if "d_attr" in r:
            a = f"{fmt(r['d_attr'], pp=True)} (p={fmt(r['p_attr'])})"
            t2 = f"{fmt(r['d_top2'], pp=True)} (p={fmt(r['p_top2'])})"
            pa = f"{fmt(r['d_pass'], pp=True)} (p={fmt(r['p_pass'])})"
        else:
            a = t2 = pa = "pending"
        cov = fmt(r["cov"]) if "cov" in r else "pending"
        out.append(f"| **{r['arm']}** | {term} | {data} | {a} | {t2} | {pa} | {cov} | "
                   f"{fmt(r['greedy'])} / {fmt(r['avg8'])} |")
    text = "\n".join(out)
    print(text)
    print()
    for r in rows:
        if "attr_base" in r:
            print(f"[{r['arm']}] N_B={r['n_B']}  attractor {r['attr_base']:.3f} -> {r['attr_arm']:.3f}  "
                  f"| avg@32 {r['pass_base']:.3f} -> {r['pass_arm']:.3f}  | {r['run']}")
    if args.md:
        Path(args.md).write_text(text + "\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
