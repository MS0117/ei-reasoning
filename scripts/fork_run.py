"""Fork a frozen run dir into a new arm run dir (the L3 workflow).

Why: the config hash covers the ENTIRE config, so a train.*/data.* override
invalidates every stage's .done in the source run (docs/objective_decision §7
risk). This tool clones the frozen data stages by hardlink into a fresh run
dir, writes the arm's config there, and re-stamps the data stages' .done
markers with the ARM's hash — so `expert_iter.loop --config <arm>/config.yaml`
skips rollout..filters, re-runs build_dataset (cheap, CPU — it must see the
arm's exclusion/negative/rejected_selection settings), then trains.

Usage (CPU, seconds):
  .venv/bin/python scripts/fork_run.py --src runs/<frozen L2 run> \
      --dst runs/L3_S3_<ts> --override train.sft.cliff.enabled=true [...]
  # or apply a whole arm preset (sparse YAML deep-merged onto the snapshot,
  # BEFORE any --override): --overlay configs/methods/arms/gadv.yaml
Then:
  .venv/bin/python -m expert_iter.loop --config runs/L3_S3_<ts>/config.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from expert_iter.config import Config
from expert_iter.utils import done_marker, mark_done, read_json, write_json

# stage output -> its .done-gated artifact (see loop.STAGE_OUTPUT); dataset/ is
# deliberately NOT transplanted so build_dataset re-runs under the arm config.
FROZEN_STAGES = {
    "rollout": "rollout/rollouts.jsonl",
    "partition": "partition/solved.jsonl",
    "anchor": "anchors/anchors.jsonl",
    "improve": "improve/improved.jsonl",
    "filters": "filtered/kept.jsonl",
}
COPY_DIRS = ["rollout", "partition", "anchors", "improve", "filtered", "reroll", "cliff_split.json"]


def _link_tree(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        return
    shutil.copytree(src, dst, copy_function=os.link, dirs_exist_ok=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--override", action="append", default=[], metavar="a.b.c=value")
    ap.add_argument("--overlay", action="append", default=[], metavar="sparse.yaml",
                    help="arm preset YAML deep-merged onto the frozen snapshot "
                         "(applied in order, before --override)")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"[fork_run] {dst} already exists — arms get fresh dirs")
    for ovl in args.overlay:
        if not Path(ovl).is_file():
            raise SystemExit(f"[fork_run] overlay not found: {ovl}")
    cfg = Config.load(src / "config.yaml", overrides=args.override, overlays=args.overlay)

    dst.mkdir(parents=True)
    cfg.save(dst / "config.yaml")
    write_json(dst / "config.hash.json", {"config_hash": cfg.hash()})
    _link_tree(src / "questions", dst / "questions")

    src_it, dst_it = src / f"iter_{args.iteration}", dst / f"iter_{args.iteration}"
    for name in COPY_DIRS:
        p = src_it / name
        if p.exists():
            _link_tree(p, dst_it / name)

    for stage, rel in FROZEN_STAGES.items():
        artifact = dst_it / rel
        old_marker = done_marker(src_it / rel)
        if not artifact.exists() or not old_marker.exists():
            raise SystemExit(f"[fork_run] source stage '{stage}' is not frozen: missing {rel}(.done)")
        payload = read_json(old_marker)
        mark_done(artifact, count=payload.get("count", 0), config_hash=cfg.hash(),
                  extra={k: v for k, v in payload.items() if k not in ("count", "config_hash")})

    print(f"[fork_run] {src} -> {dst}  (hash {cfg.hash()}; {len(args.overlay)} overlays, "
          f"{len(args.override)} overrides)")
    print(f"[fork_run] next: .venv/bin/python -m expert_iter.loop --config {dst}/config.yaml")


if __name__ == "__main__":
    main()
