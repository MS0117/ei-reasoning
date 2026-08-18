"""Backfill meta.gold_solution into a QuestionRecord JSONL (e.g. a committed
cliff set) by re-joining the source HF dataset on meta.row_idx.

    .venv/bin/python scripts/backfill_gold_solutions.py \
        --input data/cliff_sets/openr1_qwen3-4b-2507_n2000.jsonl
        [--output <input stem>_with_gold.jsonl]
        [--solution-col solution]

Every input row must carry meta.hf_name and meta.row_idx. The joined row's
question text must match the stored question (whitespace-normalized) — any
mismatch aborts the whole run BEFORE writing, because a silent misjoin would
train on the wrong solution. CPU + network only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from expert_iter.utils import read_jsonl, write_jsonl  # noqa: E402


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None,
                    help="default: <input stem>_with_gold.jsonl next to the input")
    ap.add_argument("--solution-col", default="solution")
    args = ap.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(
        in_path.stem + "_with_gold.jsonl"
    )
    rows = list(read_jsonl(in_path))
    if not rows:
        raise SystemExit(f"{in_path} is empty")

    by_source: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        meta = row.get("meta") or {}
        if "hf_name" not in meta or "row_idx" not in meta:
            raise SystemExit(f"{row.get('qid')}: meta.hf_name/meta.row_idx required for the join")
        key = (meta["hf_name"], str(meta.get("config") or "default"), str(meta.get("split") or "train"))
        by_source.setdefault(key, []).append(row)

    import datasets as hf_datasets

    mismatches: list[str] = []
    n_empty = 0
    for (hf_name, config, split), group in by_source.items():
        print(f"[backfill] loading {hf_name} [{config}/{split}] for {len(group)} rows")
        ds = hf_datasets.load_dataset(hf_name, config, split=split)
        if args.solution_col not in ds.column_names:
            raise SystemExit(f"{hf_name} has no column {args.solution_col!r}: {ds.column_names}")
        for row in group:
            idx = int(row["meta"]["row_idx"])
            src = ds[idx]
            src_q = str(src.get("problem") or src.get("question") or "")
            if _norm(src_q) != _norm(row["question"]):
                mismatches.append(
                    f"{row['qid']}: row_idx {idx} question mismatch "
                    f"(stored {_norm(row['question'])[:60]!r} vs hf {_norm(src_q)[:60]!r})"
                )
                continue
            solution = str(src.get(args.solution_col) or "").strip()
            if not solution:
                n_empty += 1
                continue
            row["meta"]["gold_solution"] = solution

    if mismatches:
        for m in mismatches:
            print(f"[backfill] MISMATCH {m}")
        raise SystemExit(
            f"{len(mismatches)} question-text mismatches — aborting without writing "
            f"(a misjoin would attach the wrong solution)"
        )

    n_with = sum(1 for r in rows if (r.get("meta") or {}).get("gold_solution"))
    write_jsonl(out_path, rows)
    print(
        f"[backfill] wrote {len(rows)} rows -> {out_path} "
        f"({n_with} with gold_solution, {n_empty} empty at source)"
    )


if __name__ == "__main__":
    main()
