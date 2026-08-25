"""Usage: .venv/bin/python scripts/diag_priv_anatomy.py <run>/iter_0/diag_credit_priv

Where does the privileged teacher's disagreement mass sit, by class?
- neg mass per token by position decile (correct vs incorrect)
- share of neg mass on answer-like tokens (digits/boxed region) vs prose
- share on 'give-up / guess' cue tokens
"""
import json, sys, re, collections, statistics as st
from pathlib import Path

d = Path(sys.argv[1])
cands = {json.loads(l)["key"]: json.loads(l) for l in open(d / "candidates.jsonl")}
per = collections.defaultdict(list)
for line in open(d / "tokens.jsonl"):
    r = json.loads(line); per[r["key"]].append((r["pos"], r["tok"], r["g"], r["s"]))

GUESS = re.compile(r"(sources?|online|typically|standard|known|likely|perhaps|probably|guess|final choice|go with|must be|I think|maybe|should be|safe)", re.I)
NUM = re.compile(r"[0-9]|\\(frac|dfrac|sqrt|pi)|boxed")

def anatomy(rows):
    rows.sort(); n = len(rows)
    dec = [0.0] * 10; cnt = [0] * 10
    kinds = collections.Counter(); tot = 0.0
    after_first_box = 0.0; box_pos = None
    for pos, tok, g, s in rows:
        if "boxed" in tok and box_pos is None: box_pos = pos
    for pos, tok, g, s in rows:
        dd = min(9, pos * 10 // n); cnt[dd] += 1
        if g < 0:
            m = -g; dec[dd] += m; tot += m
            if NUM.search(tok): kinds["number/answer"] += m
            elif GUESS.search(tok): kinds["guess-cue"] += m
            elif tok.strip() == "" or re.match(r"^[\s\$\*\-#:\.,\(\)\{\}\\]+$", tok): kinds["format"] += m
            else: kinds["prose"] += m
            if box_pos is not None and pos >= box_pos - 20: after_first_box += m
    return dict(n=n, tot=tot, dec=[x / max(c, 1) for x, c in zip(dec, cnt)],
                kinds={k: v / max(tot, 1e-9) for k, v in kinds.items()},
                from_first_box=after_first_box / max(tot, 1e-9),
                first_box_loc=(box_pos / n) if box_pos is not None else None,
                per_tok=tot / n)

by_cls = {True: [], False: []}
for key, rows in per.items():
    by_cls[cands[key]["correct"]].append(anatomy(rows))

for cls in (False, True):
    A = by_cls[cls]
    print(f"\n=== correct={cls}  n={len(A)}  neg mass/token {st.mean(a['per_tok'] for a in A):.4f}")
    print("  by decile (neg nats/token): " + " ".join(f"{st.mean(a['dec'][i] for a in A):.3f}" for i in range(10)))
    for k in ["number/answer", "guess-cue", "prose", "format"]:
        print(f"  share on {k:14s}: {st.mean(a['kinds'].get(k, 0.0) for a in A):.3f}")
    fb = [a for a in A if a["first_box_loc"] is not None]
    print(f"  has \\boxed: {len(fb)}/{len(A)}; first-box location mean {st.mean(a['first_box_loc'] for a in fb):.2f}; "
          f"share of neg mass from 20 tokens before first \\boxed onward: {st.mean(a['from_first_box'] for a in fb):.3f}")

# paired: excess of incorrect over correct, by decile, per question
byq = collections.defaultdict(lambda: {True: [], False: []})
for key, rows in per.items():
    byq[cands[key]["qid"]][cands[key]["correct"]].append(anatomy(rows))
ex = [0.0] * 10; npairs = 0
for q, g in byq.items():
    if g[True] and g[False]:
        npairs += 1
        for i in range(10):
            ex[i] += st.mean(a["dec"][i] for a in g[False]) - st.mean(a["dec"][i] for a in g[True])
print(f"\npaired excess (incorrect - correct) neg nats/token by decile over {npairs} questions: "
      + " ".join(f"{x / npairs:+.3f}" for x in ex))
