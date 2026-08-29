#!/usr/bin/env python3
"""Re-stamp a finished run's .done markers after a config SCHEMA change.

Why this exists: the .done markers that make stages resumable are keyed by
Config.hash(), which covers the whole config. Adding a FIELD to the dataclasses
(not changing any value) therefore changes the hash of every previously frozen
run, so `expert_iter.loop --config <run>/config.yaml` would re-run every stage
from scratch even though the frozen config is semantically unchanged.
scripts/fork_run.py already re-stamps when forking; this is the in-place case.

Safety: a marker is only rewritten when its recorded hash equals the run's OLD
frozen hash (config.hash.json) — i.e. it was consistent before the schema
change. Markers carrying any other hash are reported and left alone, so a
genuine value mismatch is never masked. Dry-run by default.

  .venv/bin/python scripts/restamp_config_hash.py runs/L2_freeze_*        # report
  .venv/bin/python scripts/restamp_config_hash.py runs/L2_freeze_* --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from expert_iter.config import Config  # noqa: E402


def restamp(run: Path, apply: bool) -> tuple[int, int, int]:
    """(rewritten, already-current, skipped-foreign) marker counts."""
    cfg_path, hash_path = run / "config.yaml", run / "config.hash.json"
    if not cfg_path.exists() or not hash_path.exists():
        print(f"{run}: skip (no frozen config/hash)")
        return 0, 0, 0
    old = json.loads(hash_path.read_text())["config_hash"]
    try:
        new = Config.load(cfg_path).hash()
    except Exception as e:                                  # noqa: BLE001
        # An old run frozen before a validation rule was added no longer loads.
        # It cannot be resumed under today's code anyway, so leave it untouched.
        print(f"{run}: skip (frozen config no longer validates — {type(e).__name__}: "
              f"{str(e).splitlines()[0][:90]})")
        return 0, 0, 0
    if old == new:
        print(f"{run}: already current ({new})")
        return 0, 0, 0

    hit = cur = foreign = 0
    for marker in sorted(run.rglob("*.done")):
        try:
            data = json.loads(marker.read_text())
        except (json.JSONDecodeError, OSError):
            foreign += 1
            continue
        h = data.get("config_hash")
        if h == new:
            cur += 1
        elif h == old:
            hit += 1
            if apply:
                data["config_hash"] = new
                marker.write_text(json.dumps(data, indent=2))
        else:
            foreign += 1
            print(f"    ! {marker.relative_to(run)} carries {h} (neither old nor new) — left alone")
    if apply:
        hash_path.write_text(json.dumps({"config_hash": new}, indent=2))
    verb = "re-stamped" if apply else "would re-stamp"
    print(f"{run}: {old} -> {new}  |  {verb} {hit}, already-current {cur}, foreign {foreign}")
    return hit, cur, foreign


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    a = ap.parse_args(argv)
    tot = [0, 0, 0]
    for d in a.runs:
        for i, v in enumerate(restamp(Path(d), a.apply)):
            tot[i] += v
    print(f"\ntotal: {'re-stamped' if a.apply else 'would re-stamp'} {tot[0]}, "
          f"already-current {tot[1]}, foreign {tot[2]}")
    if not a.apply and tot[0]:
        print("re-run with --apply to write")


if __name__ == "__main__":
    main(sys.argv[1:])
