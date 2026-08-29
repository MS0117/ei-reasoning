#!/usr/bin/env python3
r"""Does the attractor survive under the staged operator's own adapters — and is
it still the SAME wrong answer the base committed to?

CPU-only, reads existing dumps.  Motivation: an in-operator unlikelihood term
(a stage-2 objective that penalizes the adapter's confident wrong answer instead
of running DPO against it) needs a mode to aim at, and the failures it would aim
at are the ones staged_bridge_sft already uses as DPO negatives — the preceding
stage's rollout (`neg_stage=last_tag`).  So we ask, per stage:

  * is there still a modal wrong answer, and how concentrated is it
  * is that mode the base policy's attractor, or a different one
  * does each stage install a fresh mode, or does the first fit relocate it once
    and later stages inherit it (stage-to-stage chain, printed after the table)

Grouping is the exact-string rule build_dataset._modal_wrong_failures uses on
base rollouts (ties broken by (-count, answer)); answers come from the last
\boxed of each incorrect candidate.  Two controls, because mode identity is a
small-n statistic: a split-half null (base's own 8 failures split 4/4 — how
often does a mode agree with itself under resampling alone) and an n-matched
comparison (stage failures subsampled to the base's failure count per question).

  .venv/bin/python scripts/diag_stage_attractor.py runs/L2_freeze_* [-o out.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from expert_iter.verifier import last_boxed  # noqa: E402

N_NULL_REPEATS = 20


def _modal(xs: list[str]) -> tuple[str | None, float, int]:
    if not xs:
        return None, 0.0, 0
    c = Counter(xs)
    ans, n = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return ans, n / len(xs), len(c)


def _base_failures(it_dir: Path) -> dict[str, list[str]]:
    """qid -> extracted answers of the base policy's incorrect rollouts."""
    out: dict[str, list[str]] = defaultdict(list)
    with (it_dir / "partition" / "verdicts.jsonl").open() as f:
        for line in f:
            v = json.loads(line)
            if not v["correct"] and v.get("extracted_answer"):
                out[v["qid"]].append(v["extracted_answer"])
    return out


def _stage_failures(it_dir: Path) -> tuple[dict[str, dict[str, list[str]]], dict[str, Counter]]:
    """stage -> qid -> answers of that stage's incorrect candidates."""
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    book: dict[str, Counter] = defaultdict(Counter)
    with (it_dir / "improve" / "improved.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            st = r["op_meta"].get("stage") or "?"
            book[st]["candidates"] += 1
            if r["correct"]:
                book[st]["correct"] += 1
                continue
            book[st]["failures"] += 1
            ans = last_boxed(r["continuation_text"] or "")
            if ans is None:
                book[st]["no_box"] += 1
            else:
                out[st][r["qid"]].append(ans)
    return out, book


def report(run_dir: Path, iteration: int = 0) -> dict:
    it_dir = run_dir / f"iter_{iteration}"
    base, (stages, book) = _base_failures(it_dir), _stage_failures(it_dir)
    rng = random.Random(0)
    res: dict = {"run": run_dir.name, "iter": iteration, "stages": {}}
    print(f"\n=== {run_dir.name} (iter_{iteration}) ===")

    for st in sorted(stages):
        per_q = {q: _modal(a) for q, a in stages[st].items() if a}
        common = [q for q in per_q if base.get(q)]
        if not per_q:
            continue
        bm = {q: _modal(base[q]) for q in common}
        share = sum(v[1] for v in per_q.values()) / len(per_q)
        ge50 = sum(v[1] >= 0.5 for v in per_q.values()) / len(per_q)
        distinct = sum(v[2] for v in per_q.values()) / len(per_q)
        same = sum(per_q[q][0] == bm[q][0] for q in common) / len(common)
        # mass the stage still puts on the BASE attractor (vs bm share at base)
        mass = sum(sum(x == bm[q][0] for x in stages[st][q]) / len(stages[st][q])
                   for q in common) / len(common)
        b_share = sum(bm[q][1] for q in common) / len(common)

        # n-matched mode agreement + null floor, both on the same question set
        matched = null = m_share = 0.0
        n_m = n_n = 0
        for q in common:
            xs, ys = base[q], list(stages[st][q])
            k = min(len(xs), len(ys))
            if k >= 2:
                sub = rng.sample(ys, k)
                matched += _modal(xs)[0] == _modal(sub)[0]
                m_share += _modal(sub)[1]
                n_m += 1
        for _ in range(N_NULL_REPEATS):
            for q in common:
                xs = list(base[q])
                if len(xs) < 4:
                    continue
                rng.shuffle(xs)
                h = len(xs) // 2
                null += _modal(xs[:h])[0] == _modal(xs[h:])[0]
                n_n += 1
        row = {
            "questions": len(per_q), "failures": book[st]["failures"],
            "no_box_rate": round(book[st]["no_box"] / max(book[st]["failures"], 1), 4),
            "failures_per_q": round(sum(len(a) for a in stages[st].values()) / len(per_q), 2),
            "modal_share": round(share, 4), "frac_ge_0.5": round(ge50, 4),
            "distinct_per_q": round(distinct, 2),
            "base_modal_share_same_qs": round(b_share, 4),
            "mode_eq_base": round(same, 4),
            "mode_eq_base_n_matched": round(matched / max(n_m, 1), 4),
            "modal_share_n_matched": round(m_share / max(n_m, 1), 4),
            "null_split_half_agreement": round(null / max(n_n, 1), 4),
            "mass_on_base_mode": round(mass, 4),
        }
        res["stages"][st] = row
        print(f"{st:8s} {row['questions']:4d}q  share {row['modal_share']:.3f} "
              f"(n-matched {row['modal_share_n_matched']:.3f}, base {row['base_modal_share_same_qs']:.3f})"
              f"  >=0.5 {ge50:.1%}  distinct/q {distinct:.2f}  fails/q {row['failures_per_q']:.1f}"
              f"\n         mode==base {same:.1%} (n-matched {row['mode_eq_base_n_matched']:.1%}, "
              f"null {row['null_split_half_agreement']:.1%})  mass on base mode {mass:.3f}")

    # Chain: does the mode move once (at the first fit) or churn every stage?
    order = sorted(stages)
    for prev, cur in zip(order, order[1:]):
        common = [q for q in stages[cur] if stages[prev].get(q)]
        if not common:
            continue
        mp = {q: _modal(stages[prev][q])[0] for q in common}
        eq = sum(_modal(stages[cur][q])[0] == mp[q] for q in common) / len(common)
        mass = sum(sum(x == mp[q] for x in stages[cur][q]) / len(stages[cur][q])
                   for q in common) / len(common)
        res["chain"] = res.get("chain", {})
        res["chain"][f"{prev}->{cur}"] = {
            "questions": len(common), "mode_eq_prev": round(eq, 4),
            "mass_on_prev_mode": round(mass, 4),
        }
        print(f"  chain {prev}->{cur}: {len(common)}q  mode unchanged {eq:.1%}"
              f"  mass on {prev} mode {mass:.3f}")
    return res


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    rows = [report(Path(d), a.iter) for d in a.run_dirs]
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
