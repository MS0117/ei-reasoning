"""Shared plumbing: atomic JSONL io, .done markers, hashing, seeding, GPU parsing."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator


# ---------------------------------------------------------------------------
# JSONL io
# ---------------------------------------------------------------------------

def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Atomically write rows to path (tmp file + rename). Returns row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, row: dict) -> None:
    """Append a single row (used for metrics logs; not atomic, single-writer only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stage completion markers
# ---------------------------------------------------------------------------
# A stage output is trusted iff its `.done` sidecar exists AND was produced under
# the same config hash. Partial outputs from a killed run are therefore ignored.

def done_marker(output: str | Path) -> Path:
    output = Path(output)
    return output.parent / (output.name + ".done")


def mark_done(output: str | Path, *, count: int, config_hash: str, extra: dict | None = None) -> None:
    payload = {"count": count, "config_hash": config_hash}
    if extra:
        payload.update(extra)
    write_json(done_marker(output), payload)


def is_done(output: str | Path, *, config_hash: str | None = None) -> bool:
    output = Path(output)
    marker = done_marker(output)
    # A marker without its primary artifact can be left behind by a manual
    # cleanup or by a buggy/partial producer. Never trust that phantom marker.
    if not output.exists() or not marker.exists():
        return False
    if config_hash is None:
        return True
    try:
        return read_json(marker).get("config_hash") == config_hash
    except (json.JSONDecodeError, OSError):
        return False


# ---------------------------------------------------------------------------
# Hashing / seeding
# ---------------------------------------------------------------------------

def stable_hash(*parts: Any, digits: int = 16) -> str:
    """Deterministic hex hash of the given parts (order-sensitive)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:digits]


def stable_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from parts — used for per-request vLLM seeds so
    results do not depend on pool size or request sharding."""
    return int(stable_hash(*parts, digits=8), 16) & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# GPU / run-dir helpers
# ---------------------------------------------------------------------------

def visible_gpus(configured: list[int] | None = None) -> list[int]:
    """Resolve the GPU id list: explicit config > CUDA_VISIBLE_DEVICES > device_count.

    Returned ids are *raw* ids usable as CUDA_VISIBLE_DEVICES values for worker
    subprocesses (i.e. if CUDA_VISIBLE_DEVICES="1,2" we return [1, 2]).
    """
    if configured:
        return list(configured)
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip():
        return [int(x) for x in env.split(",") if x.strip()]
    try:
        import torch

        return list(range(torch.cuda.device_count()))
    except ImportError:
        return []


def iter_dir(run_dir: str | Path, iteration: int) -> Path:
    return Path(run_dir) / f"iter_{iteration}"


def atomic_symlink(target: str | Path, link: str | Path) -> None:
    """Point `link` at `target` (relative), replacing atomically if it exists."""
    link = Path(link)
    tmp = link.parent / (link.name + ".tmp")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(os.path.relpath(Path(target), link.parent))
    os.replace(tmp, link)
