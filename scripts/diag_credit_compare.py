"""Compare diag_scaffold_credit outputs: correct vs incorrect, by adapter group,
and by token kind. Reads one or more <out_dir>s written by
scripts/diag_scaffold_credit.py (candidates.jsonl + tokens.jsonl).

  .venv/bin/python scripts/diag_credit_compare.py \
      runs/toy_cliff/<run>/iter_0/diag_credit_bal [more dirs...]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
from pathlib import Path

MOVE = set("but wait however alternatively so thus hence therefore let check verify recheck suppose assume "
           "since because need must consider note actually hmm instead try recall observe claim now then case "
           "cases if unless contradiction indeed first second finally conclusion answer".split())
OPEN = set("we reason step solution these this the given".split())
FUNC = set("the a an of in and to that is are for at from with on as by it we".split())
MATH = re.compile(r"[0-9=+\-*/^<>]|\\(frac|sqrt|cdot|le|ge)")
FMT = re.compile(r"^[\s\$\*\-#:\.,\(\)\{\}\\]+$")


def kind(r: dict) -> str:
    t = r["tok"].strip().lower().strip("*:.,")
    if not t or FMT.match(r["tok"]):
        return "format"
    if t in MOVE:
        return "reasoning-move"
    if r["pos"] < 60 and t in OPEN:
        return "opening-template"
    if t in FUNC:
        return "function"
    if MATH.search(r["tok"]):
        return "math"
    return "content-word"


def load(d: Path):
    c = [json.loads(l) for l in open(d / "candidates.jsonl")]
    t = [json.loads(l) for l in open(d / "tokens.jsonl")]
    return c, t


def _m(cands, path):
    vals = []
    for x in cands:
        v = x
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(v)
    return st.mean(vals) if vals else float("nan")


def tokstats(toks):
    n = len(toks)
    if n == 0:
        return None
    hi = [r for r in toks if r["H"] is not None and r["H"] > 1.0]
    return dict(
        n=n,
        mean_g=st.mean(r["g"] for r in toks),
        per1k_gH=1000 * sum(1 for r in hi if r["g"] > 0.5) / n,
        per1k_gLo=1000 * sum(1 for r in toks if r["g"] > 0.5 and r["H"] is not None and r["H"] <= 1.0) / n,
        g_on_hi=st.mean(r["g"] for r in hi) if hi else float("nan"),
        a_on_hi=st.mean(r["a"] for r in hi) if hi else float("nan"),
        frac_hi=len(hi) / n,
        neg_mass=sum(-r["g"] for r in toks if r["g"] < 0) / n,
    )


def row(label, cands, toks):
    keys = {x["key"] for x in cands}
    tt = [r for r in toks if r["key"] in keys]
    s = tokstats(tt)
    if not cands or not s:
        print(f"{label:24s} (empty)")
        return
    print(f"{label:24s} n={len(cands):3d} tok={s['n'] / len(cands):5.0f} | g={s['mean_g']:.4f} neg={s['neg_mass']:.4f} "
          f"attr={_m(cands, ('classes', 'attributed_share')):.3f} lossB={_m(cands, ('classes', 'loss_mass_B')):.3f} "
          f"top10={_m(cands, ('concentration', 'top10_share_gpos')):.3f} J(g,H)={_m(cands, ('overlap', 'jaccard_g_vs_entropy')):.3f} | "
          f"g>.5&H>1/1k={s['per1k_gH']:5.2f} g>.5&H<=1/1k={s['per1k_gLo']:5.2f} g|H>1={s['g_on_hi']:.3f} a|H>1={s['a_on_hi']:.3f}")


def kind_table(label, toks):
    big = [r for r in toks if r["g"] > 1]
    if not big:
        print(f"  {label}: no g>1 tokens")
        return
    tot = sum(r["g"] for r in big)
    n_all = len(toks)
    print(f"  {label}: g>1 tokens {len(big)} ({1000 * len(big) / n_all:.2f}/1k)")
    print(f"    {'kind':18s} {'n':>6} {'/1k':>6} {'g-mass':>7} {'meanH':>6}")
    for k, cnt in sorted(collections.Counter(kind(r) for r in big).items(), key=lambda kv: -kv[1]):
        rs = [r for r in big if kind(r) == k]
        print(f"    {k:18s} {cnt:6d} {1000 * cnt / n_all:6.2f} {sum(r['g'] for r in rs) / tot:7.2f} "
              f"{st.mean((r['H'] or 0) for r in rs):6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--no-kinds", action="store_true")
    args = ap.parse_args()
    for d in args.dirs:
        d = Path(d)
        c, t = load(d)
        print(f"\n===== {d} =====")
        pos = [x for x in c if x["correct"]]
        neg = [x for x in c if not x["correct"]]
        row("ALL", c, t)
        row("correct", pos, t)
        row("incorrect", neg, t)
        groups = sorted({x["group"] for x in c})
        if len(groups) > 1:
            for g in groups:
                row(f"correct/{g}", [x for x in pos if x["group"] == g], t)
                row(f"incorrect/{g}", [x for x in neg if x["group"] == g], t)
        if not args.no_kinds:
            kp = {x["key"] for x in pos}
            kn = {x["key"] for x in neg}
            kind_table("correct", [r for r in t if r["key"] in kp])
            kind_table("incorrect", [r for r in t if r["key"] in kn])
        # paired-by-question difference (correct minus incorrect mean g) when both present
        byq = collections.defaultdict(lambda: {"pos": [], "neg": []})
        tok_by_key = collections.defaultdict(list)
        for r in t:
            tok_by_key[r["key"]].append(r["g"])
        for x in c:
            byq[x["qid"]]["pos" if x["correct"] else "neg"].append(st.mean(tok_by_key[x["key"]]))
        diffs = [st.mean(v["pos"]) - st.mean(v["neg"]) for v in byq.values() if v["pos"] and v["neg"]]
        if diffs:
            wins = sum(1 for d_ in diffs if d_ > 0)
            print(f"  paired by question (mean g, correct - incorrect): n={len(diffs)} mean diff={st.mean(diffs):+.4f} "
                  f"correct>incorrect in {wins}/{len(diffs)} questions")


if __name__ == "__main__":
    main()
