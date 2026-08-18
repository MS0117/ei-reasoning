"""Lazy wandb helpers — the one place a process decides if/how it talks to wandb.

Ownership: the loop driver holds the per-iteration metrics run and exports
EI_LOOP_DRIVER=1 to its stage subprocesses, which then skip their own init so
each metric is logged exactly once. train is the deliberate exception — its
per-step loss curves flow through HF Trainer's report_to="wandb" (see
train._report_to) and appear as separate runs grouped under run.name.
Standalone stage invocations (scripts/eval_bench.sh, python -m expert_iter.eval)
have no driver and get their own run via stage_wandb().

Auth: a requested online mode silently downgrades to offline when no API key
is configured (wandb login / WANDB_API_KEY) — a background job must never hang
on an interactive login prompt. Offline runs land in <dir>/wandb/ and are
uploaded later with `wandb sync <run-dir>`; note wandb ignores id/resume in
offline mode, so each offline rerun is a fresh run.
"""

from __future__ import annotations

import os
from pathlib import Path

DRIVER_ENV = "EI_LOOP_DRIVER"


def resolve_wandb_mode(mode: str) -> str:
    """A requested online mode with no API key downgrades to offline — a
    background job must never hang on an interactive login prompt."""
    if mode != "online":
        return mode
    import wandb

    if wandb.api.api_key is None:
        print("[wandb] no API key (run `wandb login`) — falling back to offline; "
              "upload later with `wandb sync`", flush=True)
        return "offline"
    return mode


def init_wandb(cfg, *, name: str, id_file: Path, job_type: str,
               config: dict | None = None):
    """Start (or resume) a wandb run; None when cfg.run.wandb.mode is disabled.
    The run id persists in id_file so crash-recovery reruns continue the same
    wandb run instead of littering the project (online mode only)."""
    if cfg.run.wandb.mode == "disabled":
        return None
    import wandb

    mode = resolve_wandb_mode(cfg.run.wandb.mode)
    # Trainer runs in a later subprocess. Propagate the resolved mode so it
    # cannot retry online auth after the driver already chose offline.
    os.environ["WANDB_MODE"] = mode
    id_file.parent.mkdir(parents=True, exist_ok=True)
    run_id = id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id()
    id_file.write_text(run_id)
    return wandb.init(
        project=cfg.run.wandb.project,
        entity=cfg.run.wandb.entity,
        mode=mode,
        group=cfg.run.name,
        name=name,
        id=run_id,
        resume="allow",
        job_type=job_type,
        dir=str(id_file.parent),
        config=config or {},
    )


def stage_wandb(cfg, *, name: str, id_file: Path, job_type: str,
                config: dict | None = None):
    """init_wandb for a stage process: no-op under the loop driver, which logs
    the merged per-iteration metrics itself."""
    if os.environ.get(DRIVER_ENV):
        return None
    return init_wandb(cfg, name=name, id_file=id_file, job_type=job_type, config=config)


def log_metrics(run, metrics: dict, step: int | None = None) -> None:
    """Numeric values become charts, numeric lists become wandb histograms
    (e.g. cliff/conversion_histogram), everything else goes to the run summary
    (strings like model_path or a benchmark's error message can't be plotted)."""
    if run is None:
        return
    import wandb  # already imported by whoever created `run`

    payload: dict = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            run.summary[k] = v
        elif isinstance(v, (int, float)):
            payload[k] = v
        elif (isinstance(v, (list, tuple)) and v
              and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
            payload[k] = wandb.Histogram(v)
        else:
            run.summary[k] = v
    if payload:
        run.log(payload, step=step)
