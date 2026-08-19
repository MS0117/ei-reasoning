"""Join a passrate run's per-question stats back onto the source HF dataset.

Produces one JSONL row per rolled-out question: the original dataset columns
you ask for (default: OpenR1's problem/answer/solution/source/problem_type/
uuid — NOT generations/messages, which are multi-GB and consumed by nothing)
plus the pass-rate columns {c, k, pass_rate, class, n_truncated} and the
join keys {qid, row_idx}. This is the curation table a stratified sampler
draws from ("x% cliff, y% low pass-rate, ...").

    .venv/bin/python data/join_passrate.py \
        --run-dir runs/passrate/<slug>/            # needs question_stats.jsonl
        [--questions <path>]                       #   + questions.jsonl (default: run dir)
        [--output PATH]     # default: <run_dir>/<dataset>_<config>_passrate.jsonl
        [--hf-cols problem,answer,solution,source,problem_type,uuid]
        [--no-hf-join]                             # skip the HF join: stats + stored fields only

Only the two small run artifacts are needed from the machine that did the
rollouts (question_stats.jsonl + questions.jsonl); the HF join runs locally
against the cached dataset. Same misjoin guard as backfill_gold_solutions.py:
every joined row's question text must match the stored question
(whitespace-normalized) or the run aborts BEFORE writing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from expert_iter.utils import read_jsonl, stable_hash, write_jsonl  # noqa: E402

DEFAULT_HF_COLS = "problem,answer,solution,source,problem_type,uuid"


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="passrate run dir holding question_stats.jsonl")
    ap.add_argument("--questions", default=None,
                    help="questions.jsonl path (default: <run-dir>/questions.jsonl)")
    ap.add_argument("--output", default=None,
                    help="default: <run-dir>/passrate_joined.jsonl")
    ap.add_argument("--hf-cols", default=DEFAULT_HF_COLS,
                    help=f"comma-separated original columns to carry (default: {DEFAULT_HF_COLS})")
    ap.add_argument("--no-hf-join", action="store_true",
                    help="skip the HF dataset join; emit stats + stored question fields only")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    stats_path = run_dir / "question_stats.jsonl"
    questions_path = Path(args.questions) if args.questions else run_dir / "questions.jsonl"
    for p in (stats_path, questions_path):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    stats = {row["qid"]: row for row in read_jsonl(stats_path)}
    questions = {row["qid"]: row for row in read_jsonl(questions_path)}
    if not stats:
        raise SystemExit(f"{stats_path} is empty")
    missing = [qid for qid in stats if qid not in questions]
    if missing:
        raise SystemExit(
            f"{len(missing)} stats qids absent from {questions_path} "
            f"(first: {missing[0]}) — stats and questions are from different runs?"
        )

    # Group by HF source so multi-dataset runs (none yet, but meta carries the
    # key) load each dataset once. Mirrors backfill_gold_solutions.py.
    by_source: dict[tuple[str, str, str], list[str]] = {}
    for qid in stats:
        meta = questions[qid].get("meta") or {}
        if not args.no_hf_join and "hf_name" not in meta:
            raise SystemExit(
                f"{qid}: meta.hf_name required for the HF join "
                "(re-run with --no-hf-join to skip it)"
            )
        key = (str(meta.get("hf_name") or ""), str(meta.get("config") or "default"),
               str(meta.get("split") or "train"))
        by_source.setdefault(key, []).append(qid)

    # Default output name identifies the slice: <dataset>_<config>_passrate.jsonl
    # (e.g. openr1-math-220k_extended_passrate.jsonl), so default/extended sweeps
    # of the same dataset never overwrite each other's joined table.
    if args.output:
        out_path = Path(args.output)
    elif len(by_source) == 1:
        (hf_name, config, _split), = by_source
        ds_tag = (hf_name.rsplit("/", 1)[-1] or "dataset").lower()
        out_path = run_dir / f"{ds_tag}_{config}_passrate.jsonl"
    else:
        out_path = run_dir / "passrate_joined.jsonl"

    hf_cols = [c.strip() for c in args.hf_cols.split(",") if c.strip()]
    hf_rows: dict[str, dict] = {}       # qid -> {col: value}
    if not args.no_hf_join:
        import datasets as hf_datasets

        mismatches: list[str] = []
        for (hf_name, config, split), qids in by_source.items():
            print(f"[join] loading {hf_name} [{config}/{split}] for {len(qids)} rows")
            ds = hf_datasets.load_dataset(hf_name, config, split=split)
            bad = [c for c in hf_cols if c not in ds.column_names]
            if bad:
                raise SystemExit(
                    f"{hf_name} has no column(s) {bad}; columns: {ds.column_names}"
                )
            # qid-hash fallback for runs from before the adapter recorded
            # row_idx: qid == "hfm-" + stable_hash(hf_name, question), so one
            # dataset scan rebuilds the mapping. Adapter-era runs skip the scan.
            no_idx = [qid for qid in qids if "row_idx" not in questions[qid]["meta"]]
            qid_to_idx: dict[str, int] = {}
            if no_idx:
                print(f"[join] {len(no_idx)} rows lack meta.row_idx — scanning by qid hash")
                wanted = set(no_idx)
                q_col = "problem" if "problem" in ds.column_names else "question"
                for row_idx, question in enumerate(ds[q_col]):
                    qid = "hfm-" + stable_hash(hf_name, str(question or "").strip())
                    if qid in wanted:
                        qid_to_idx.setdefault(qid, row_idx)
            for qid in qids:
                q = questions[qid]
                idx = q["meta"].get("row_idx", qid_to_idx.get(qid))
                if idx is None:
                    mismatches.append(f"{qid}: not found by row_idx or qid hash")
                    continue
                idx = int(idx)
                row = ds[idx]
                if _norm(row.get("problem") or row.get("question") or "") != _norm(q["question"]):
                    mismatches.append(f"{qid}: row_idx {idx} question mismatch")
                    continue
                hf_rows[qid] = {"row_idx": idx, **{c: row.get(c) for c in hf_cols}}
        if mismatches:
            for m in mismatches[:10]:
                print(f"[join] MISMATCH {m}", file=sys.stderr)
            raise SystemExit(
                f"{len(mismatches)} question mismatches — dataset revision drifted? "
                "Nothing written."
            )

    out_rows = []
    for qid, stat in sorted(stats.items()):
        q = questions[qid]
        row = {
            "qid": qid,
            "row_idx": (q.get("meta") or {}).get("row_idx"),
            # stored (filtered/extracted) fields — always present, HF join or not
            "question": q["question"],
            "final_answer": q["final_answer"],
            **hf_rows.get(qid, {}),   # includes resolved row_idx when joined
            # pass-rate columns (raw_* absent in older runs -> None)
            "c": stat["c"],
            "k": stat["n"],
            "pass_rate": stat["pass_rate"],
            "raw_pass_rate": stat.get("raw_pass_rate"),
            "class": stat["class"],
            "n_truncated": stat["n_truncated"],
        }
        out_rows.append(row)

    write_jsonl(out_path, out_rows)
    classes = Counter(r["class"] for r in out_rows)
    print(f"[join] wrote {len(out_rows)} rows -> {out_path}")
    print(f"[join] classes: {dict(sorted(classes.items()))}")


if __name__ == "__main__":
    main()
