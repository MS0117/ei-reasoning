"""Usage: .venv/bin/python scripts/diag_priv_windows.py <run>/iter_0/diag_credit_priv [n_show]

Print the teacher-flagged (most negative g') 64-token window for a few
incorrect and correct candidates, with context, to judge qualitatively whether
it is a wrong turn in the math or a style mismatch with the gold solution."""
import json, sys, collections
from pathlib import Path

d = Path(sys.argv[1]); n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 3
cands = {json.loads(l)["key"]: json.loads(l) for l in open(d / "candidates.jsonl")}
per = collections.defaultdict(list)
for line in open(d / "tokens.jsonl"):
    r = json.loads(line); per[r["key"]].append((r["pos"], r["tok"], r["g"], r["s"]))

W = 64
rows_out = []
for key, rows in per.items():
    rows.sort(); n = len(rows)
    if n < W: continue
    cs = [0.0]
    for r in rows: cs.append(cs[-1] + r[2])
    wins = [(cs[i + W] - cs[i]) / W for i in range(n - W + 1)]
    i_min = min(range(len(wins)), key=wins.__getitem__)
    neg_mass = sum(-r[2] for r in rows if r[2] < 0)
    win_neg = sum(-r[2] for r in rows[i_min:i_min + W] if r[2] < 0)
    last_third_neg = sum(-r[2] for r in rows[2 * n // 3:] if r[2] < 0)
    rows_out.append(dict(key=key, correct=cands[key]["correct"], n=n, i_min=i_min, loc=(i_min + W / 2) / n,
                         worst=wins[i_min], win_share=win_neg / max(neg_mass, 1e-9),
                         last3_share=last_third_neg / max(neg_mass, 1e-9), rows=rows))

# aggregate: how concentrated is the negative mass?
for cls in (False, True):
    sub = [r for r in rows_out if r["correct"] == cls]
    import statistics as st
    print(f"class correct={cls}: n={len(sub)}  worst-window share of neg mass: mean {st.mean(r['win_share'] for r in sub):.3f} "
          f"median {st.median(r['win_share'] for r in sub):.3f} | last-third share mean {st.mean(r['last3_share'] for r in sub):.3f} | loc mean {st.mean(r['loc'] for r in sub):.2f}")

def show(r, ctx=120):
    rows = r["rows"]; a = max(0, r["i_min"] - ctx); b = min(len(rows), r["i_min"] + W + 40)
    print("=" * 100)
    print(f"key={r['key']} correct={r['correct']} n={r['n']} window@{r['i_min']} (loc {r['loc']:.2f}) mean g'={r['worst']:.2f} "
          f"window holds {r['win_share']:.0%} of neg mass; last third holds {r['last3_share']:.0%}")
    out = []
    for pos, tok, g, s in rows[a:b]:
        if pos == r["i_min"]: out.append("\n>>>>[WINDOW START]>>>> ")
        if pos == r["i_min"] + W: out.append(" <<<<[WINDOW END]<<<<\n")
        if g < -1.0: out.append(f"{tok}⟨{g:.1f}⟩")
        else: out.append(tok)
    print("".join(out).replace("\n\n", "\n"))

inc = sorted([r for r in rows_out if not r["correct"]], key=lambda r: r["worst"])[:n_show]
cor = sorted([r for r in rows_out if r["correct"]], key=lambda r: r["worst"])[:n_show]
for r in inc: show(r)
print("\n\n########## CORRECT trajectories (their worst windows) ##########")
for r in cor: show(r)
