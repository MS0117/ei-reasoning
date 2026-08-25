#!/usr/bin/env bash
# Integrity check for a pass-rate run received from another machine, BEFORE
# joining it. Confirms the shipped stats/questions are a matched pair and that
# the remote question set is the one this checkout would have produced —
# a dataset-revision drift there would silently shift every row_idx.
#
#   bash scripts/verify_friend_passrate.sh runs/passrate/friend_openr1_extended [LOCAL_REF_DIR]
#
# LOCAL_REF_DIR defaults to this repo's frozen dry-run copy for the same config.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DIR="${1:?usage: verify_friend_passrate.sh RUN_DIR [LOCAL_REF_DIR]}"
REF="${2:-}"

.venv/bin/python - "$DIR" "$REF" <<'PY'
import json, sys
from pathlib import Path

d = Path(sys.argv[1]); ref = sys.argv[2]
qs, st = d / "questions.jsonl", d / "question_stats.jsonl"
for p in (qs, st):
    if not p.exists():
        raise SystemExit(f"MISSING {p}")

def scan(path):
    """(rows, n_torn) — a truncated transfer leaves a half-written final line,
    so parse leniently and REPORT the damage instead of dying on it."""
    rows, torn = [], 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                torn += 1
    return rows, torn

q_rows, q_torn = scan(qs)
stats, s_torn = scan(st)
q_ids = {r["qid"] for r in q_rows}
s_ids = {r["qid"] for r in stats}
k_seen = {r["n"] for r in stats}

print(f"questions      {len(q_ids):>7}" + (f"   MALFORMED LINES: {q_torn}" if q_torn else ""))
print(f"question_stats {len(s_ids):>7}   K={sorted(k_seen)}"
      + (f"   MALFORMED LINES: {s_torn}" if s_torn else ""))
if q_torn or s_torn:
    print("  -> truncated/corrupt transfer. question_stats is the file that MUST be")
    print("     re-sent; a damaged questions.jsonl is recoverable from a local copy")
    print("     of the same config (see the local-reference check below).")

orphans = s_ids - q_ids
print(f"stats qids missing from questions: {len(orphans)}"
      + (f"  e.g. {sorted(orphans)[:3]}" if orphans else "  OK"))
ungraded = q_ids - s_ids
if ungraded:
    print(f"WARNING questions with no stats row: {len(ungraded)} (partial run?)")

if ref:
    # Compare against the STATS qids, not the shipped questions.jsonl: stats is
    # the authoritative set (one row per rolled-out question) and survives a
    # truncated questions transfer, which would otherwise read as drift.
    r_ids = {r["qid"] for r in scan(Path(ref) / "questions.jsonl")[0]}
    same = r_ids >= s_ids
    print(f"local reference {len(r_ids)} qids -> "
          f"{'COVERS every stats qid' if same else 'MISSING stats qids'}")
    if not same:
        print(f"  stats qids absent from the local reference: {len(s_ids - r_ids)}")
        print("  -> dataset revision drifted; do NOT trust row_idx, re-derive locally")
    elif r_ids != s_ids:
        print(f"  (reference is a superset: {len(r_ids - s_ids)} extra, fine for joining)")

from collections import Counter
print("classes:", dict(sorted(Counter(r["class"] for r in stats).items())))
if s_torn:
    raise SystemExit("FAILED: question_stats.jsonl is damaged — ask for a re-send")
if orphans and not (ref and not (s_ids - r_ids)):
    raise SystemExit("FAILED: stats reference questions that were not shipped")
if q_torn:
    if ref and s_ids == r_ids:
        print("OK (shipped questions.jsonl is truncated but UNNEEDED: the local "
              "reference covers every stats qid — join with --questions <ref>)")
    else:
        raise SystemExit("FAILED: questions.jsonl truncated and no matching local reference")
else:
    print("OK")
PY
