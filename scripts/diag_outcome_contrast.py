"""Usage: .venv/bin/python scripts/diag_outcome_contrast.py <diag_dir> [...]
(diag_dir = output of scripts/diag_scaffold_credit.py, e.g. <run>/iter_0/diag_credit_bal or diag_credit_priv)

Falsification test for the 'scaffold + outcome contrast' weighting proposal.

For each diag dir (tokens.jsonl + candidates.jsonl) compute per-candidate
  G_mean  = mean_t g_t^+
  G_top20 = mean of the top-20% g_t^+
  G_negmass = mean_t g_t^-   (|negative| mass, for the privileged runs)
  worst64 = min over 64-token windows of mean g_t
then, per question with BOTH correct and incorrect candidates,
  B_x = mean over incorrect of G
  C_r = [G_r - B_x]^+ for each correct r
and report: paired sign test (correct mean vs B_x), fraction of correct cands
with C_r > 0, the effective weight distribution w = 1 + lam*q under the
proposal, and how much of the weighted loss mass lands on which token kinds.
"""
from __future__ import annotations
import json, sys, math, re, collections, statistics as st
from pathlib import Path

MOVE = set("but wait however alternatively so thus hence therefore let check verify recheck suppose assume "
           "since because need must consider note actually hmm instead try recall observe claim now then case "
           "cases if unless contradiction indeed first second finally conclusion answer".split())
OPEN = set("we reason step solution these this the given".split())
FUNC = set("the a an of in and to that is are for at from with on as by it we".split())
MATH = re.compile(r"[0-9=+\-*/^<>]|\\(frac|sqrt|cdot|le|ge)")
FMT = re.compile(r"^[\s\$\*\-#:\.,\(\)\{\}\\]+$")

def kind(tok, pos):
    t = tok.strip().lower().strip("*:.,")
    if not t or FMT.match(tok): return "format"
    if t in MOVE: return "reasoning-move"
    if pos < 60 and t in OPEN: return "opening-template"
    if t in FUNC: return "function"
    if MATH.search(tok): return "math"
    return "content-word"

def binom_p(k, n):
    # two-sided sign test p-value
    from math import comb
    if n == 0: return float("nan")
    lo = min(k, n - k)
    p = sum(comb(n, i) for i in range(0, lo + 1)) / 2 ** n
    return min(1.0, 2 * p)

def analyze(d: Path, lam=3.0):
    cands = {json.loads(l)["key"]: json.loads(l) for l in open(d / "candidates.jsonl")}
    per = collections.defaultdict(list)  # key -> list of (pos, tok, g, s, H)
    for line in open(d / "tokens.jsonl"):
        r = json.loads(line)
        per[r["key"]].append((r["pos"], r["tok"], r["g"], r["s"], r.get("H", 0.0)))
    stats = {}
    for key, rows in per.items():
        rows.sort()
        g = [x[2] for x in rows]; s = [x[3] for x in rows]; H = [x[4] for x in rows]
        gp = [max(v, 0.0) for v in g]; gn = [max(-v, 0.0) for v in g]
        n = len(g)
        k = max(1, int(0.2 * n))
        top = sorted(gp, reverse=True)[:k]
        # worst 64-window
        w = 64; worst = 0.0; worst_loc = 0.5
        if n >= w:
            cs = [0.0]
            for v in g: cs.append(cs[-1] + v)
            wins = [(cs[i + w] - cs[i]) / w for i in range(n - w + 1)]
            i_min = min(range(len(wins)), key=wins.__getitem__)
            worst = wins[i_min]; worst_loc = (i_min + w / 2) / n
        else:
            worst = sum(g) / max(n, 1)
        smh = [max(si - hi, 0.0) for si, hi in zip(s, H)]   # confident-surprise
        stats[key] = dict(
            G_mean=sum(gp) / n, G_top20=sum(top) / len(top), G_neg=sum(gn) / n,
            worst64=worst, worst_loc=worst_loc,
            mean_s=sum(s) / n, mean_H=sum(H) / n, SmH_pos=sum(smh) / n,
            frac_s_gt1=sum(1 for v in s if v > 1) / n,
            n=n, sum_s=sum(s), rows=[r[:4] for r in rows], gp=gp,
        )
    # group by question
    byq = collections.defaultdict(lambda: {"c": [], "i": []})
    for key, c in cands.items():
        if key not in stats: continue
        byq[c["qid"]]["c" if c["correct"] else "i"].append(key)
    out = {}
    for metric, better_is_high in [("G_mean", True), ("G_top20", True), ("G_neg", False), ("worst64", True),
                                   ("worst_loc", False), ("mean_s", False), ("mean_H", False),
                                   ("SmH_pos", False), ("frac_s_gt1", False), ("n", False)]:
        wins = ties = n_pairs = 0
        n_pos = n_corr = 0
        deltas = []
        for q, grp in byq.items():
            if not grp["c"] or not grp["i"]: continue
            B = st.mean(stats[k][metric] for k in grp["i"])
            cm = st.mean(stats[k][metric] for k in grp["c"])
            n_pairs += 1
            if cm == B: ties += 1
            elif (cm > B) == better_is_high: wins += 1
            deltas.append(cm - B)
            for k in grp["c"]:
                n_corr += 1
                if (stats[k][metric] - B > 0) == better_is_high and stats[k][metric] != B:
                    n_pos += 1
        out[metric] = dict(paired_wins=wins, n_pairs=n_pairs, p=binom_p(wins, n_pairs - ties),
                           frac_correct_with_C_gt0=n_pos / max(n_corr, 1),
                           mean_delta=st.mean(deltas) if deltas else float("nan"),
                           mean_correct=st.mean(stats[k][metric] for q, g in byq.items() for k in g["c"]) if any(g["c"] for g in byq.values()) else float("nan"),
                           mean_incorrect=st.mean(stats[k][metric] for q, g in byq.items() for k in g["i"]) if any(g["i"] for g in byq.values()) else float("nan"))
    # effective weights under the proposal, G=G_top20, f = seq-normalised g+
    wmass_kind = collections.Counter(); umass_kind = collections.Counter()
    extra_mass_total = 0.0; base_mass_total = 0.0
    w_eff = []  # per-candidate: effective multiplier on its own loss (sum w*s / sum s)
    for q, grp in byq.items():
        if not grp["c"] or not grp["i"]: continue
        B = st.mean(stats[k]["G_top20"] for k in grp["i"])
        for k in grp["c"]:
            S = stats[k]; C = max(S["G_top20"] - B, 0.0)
            sg = sum(S["gp"]) + 1e-8
            num = 0.0
            for (pos, tok, g, s), gp in zip(S["rows"], S["gp"]):
                qv = C * gp / sg
                wt = 1 + lam * qv
                num += wt * s
                kd = kind(tok, pos)
                umass_kind[kd] += s
                wmass_kind[kd] += (wt - 1) * s
            w_eff.append(num / max(S["sum_s"], 1e-8))
            extra_mass_total += num - S["sum_s"]; base_mass_total += S["sum_s"]
    tot_u = sum(umass_kind.values()) or 1; tot_w = sum(wmass_kind.values()) or 1
    out["proposal_weights"] = dict(
        lam=lam, n_correct=len(w_eff),
        mean_eff_multiplier=st.mean(w_eff) if w_eff else float("nan"),
        p90_eff_multiplier=sorted(w_eff)[int(0.9 * len(w_eff))] if w_eff else float("nan"),
        extra_loss_mass_over_uniform=extra_mass_total / max(base_mass_total, 1e-8),
        uniform_loss_mass_by_kind={k: round(v / tot_u, 3) for k, v in umass_kind.most_common()},
        extra_mass_by_kind={k: round(v / tot_w, 3) for k, v in wmass_kind.most_common()},
    )
    return out

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("=" * 80); print(p)
        res = analyze(Path(p))
        for k, v in res.items():
            print(f"  {k}:")
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
