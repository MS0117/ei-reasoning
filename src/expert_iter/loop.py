"""Driver: run EI iterations as sequenced stage subprocesses.

    python -m expert_iter.loop --config configs/ei_default.yaml \
        [--override a.b=c ...] [--iterations N] [--force STAGE|all]

Model resolution per iteration k:
  data-producing stages (rollout/partition/anchor/improve/filters/build_dataset)
      -> current policy = iter_{k-1}/ckpt (k=0: model.base)
  train stage
      -> initialization = model.base if train.init_from=="base" else current policy
  eval + benchmark_eval
      -> the just-trained iter_k/ckpt (falls back to current policy when the
         stage list has no train, e.g. eval-only runs)

Stages themselves skip when their .done marker matches the config hash, so
resuming after a crash is just re-running the same command. `--force` clears
that behavior for one stage name (or `all`).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import Config
from .wandb_utils import DRIVER_ENV, init_wandb, log_metrics
from .utils import (
    append_jsonl,
    atomic_symlink,
    done_marker,
    iter_dir,
    read_json,
    visible_gpus,
    write_json,
)

# stage -> primary output (relative to iter dir) whose .done marker gates a skip
STAGE_OUTPUT = {
    "rollout": "rollout/rollouts.jsonl",
    "partition": "partition/solved.jsonl",
    "anchor": "anchors/anchors.jsonl",
    "improve": "improve/improved.jsonl",
    "filters": "filtered/kept.jsonl",
    "build_dataset": "dataset/train_sft.jsonl",
    "train": "ckpt/config.json",
    "eval": "eval/metrics.json",
    "benchmark_eval": "benchmark_eval/metrics.json",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Expert-iteration loop driver")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[], metavar="a.b=c")
    ap.add_argument("--iterations", type=int, default=None, help="override loop.iterations")
    ap.add_argument("--start-iter", type=int, default=0)
    ap.add_argument("--force", default=None, help="re-run this stage (or 'all') even if .done matches")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config, overrides=args.override)
    run_dir = Path(cfg.run.output_root) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")
    n_iters = args.iterations if args.iterations is not None else cfg.loop.iterations
    n_gpus = len(visible_gpus(cfg.engine.gpus))
    print(f"[loop] run={cfg.run.name} iterations={n_iters} gpus={n_gpus} "
          f"backend={cfg.train.backend} objective={cfg.train.objective}")
    wb = init_wandb(cfg, name=cfg.run.name, id_file=run_dir / "wandb_id.txt",
                    job_type="loop", config=cfg.to_dict())

    for k in range(args.start_iter, n_iters):
        it_dir = iter_dir(run_dir, k)
        (it_dir / "logs").mkdir(parents=True, exist_ok=True)
        policy_path = _policy_path(cfg, run_dir, k)
        train_init = cfg.model.base if cfg.train.init_from == "base" else policy_path
        print(f"\n[loop] ===== iteration {k} | policy={policy_path} =====")

        for stage in cfg.loop.stages:
            out = it_dir / STAGE_OUTPUT[stage]
            if args.force in (stage, "all"):
                marker = done_marker(out)
                if marker.exists():
                    marker.unlink()
            if stage == "train":
                model_path = train_init
            elif stage in ("eval", "benchmark_eval") and (it_dir / "ckpt" / "config.json").exists():
                # Eval stages grade the JUST-trained checkpoint (they run after
                # train); fall back to the current policy in train-less configs.
                model_path = str(it_dir / "ckpt")
            else:
                model_path = policy_path
            _run_stage(stage, cfg, args, run_dir, k, model_path, it_dir, n_gpus)

        atomic_symlink(it_dir, run_dir / "latest")
        atomic_symlink(it_dir / "ckpt", run_dir / "latest_ckpt")
        row = _collect_metrics(run_dir, it_dir, k)
        log_metrics(wb, row, step=k)
        _update_state(run_dir, cfg, k)

    if wb is not None:
        wb.finish()
    print(f"\n[loop] done — metrics: {run_dir / 'metrics.jsonl'}")


def _policy_path(cfg: Config, run_dir: Path, k: int) -> str:
    if k == 0:
        return cfg.model.base
    prev_ckpt = iter_dir(run_dir, k - 1) / "ckpt"
    if not (prev_ckpt / "config.json").exists():
        raise RuntimeError(f"iteration {k} needs {prev_ckpt}, but it has no checkpoint")
    return str(prev_ckpt)


def _run_stage(stage: str, cfg: Config, args, run_dir: Path, k: int,
               model_path: str, it_dir: Path, n_gpus: int) -> None:
    common = [
        "--config", str(run_dir / "config.yaml"),
        "--run-dir", str(run_dir),
        "--iter", str(k),
        "--model-path", model_path,
    ]
    if stage == "train":
        accel_cfg = Path("configs/accelerate") / f"{_accel_file(cfg.train.backend)}"
        accelerate = shutil.which("accelerate") or str(Path(sys.executable).parent / "accelerate")
        cmd = [accelerate, "launch",
               "--config_file", str(accel_cfg),
               "--num_processes", str(n_gpus),
               "--num_machines", "1",
               "-m", "expert_iter.train", *common]
    else:
        cmd = [sys.executable, "-m", f"expert_iter.{stage}", *common]

    log_path = it_dir / "logs" / f"{stage}.log"
    print(f"[loop] iter{k}:{stage} -> {log_path}")
    t0 = time.time()
    with log_path.open("a") as log_f:
        # DRIVER_ENV tells stages the driver owns wandb metric logging, so they
        # skip their own init (train's HF-Trainer reporting is unaffected).
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env={**os.environ, DRIVER_ENV: "1"})
        for line in proc.stdout:  # tee: keep the console alive AND the log complete
            log_f.write(line)
            if line.startswith("[") or "error" in line.lower():
                print(f"  {line.rstrip()}", flush=True)
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"stage iter{k}:{stage} failed (exit {proc.returncode}); see {log_path}"
        )
    print(f"[loop] iter{k}:{stage} finished in {time.time() - t0:.0f}s")


def _accel_file(backend: str) -> str:
    return {"single": "single_gpu.yaml", "zero2": "zero2.yaml",
            "zero3": "zero3.yaml", "fsdp2": "fsdp2.yaml"}[backend]


def _collect_metrics(run_dir: Path, it_dir: Path, k: int) -> dict:
    row = {"iter": k}
    for rel, keys in [
        ("partition/stats.json", ["solve_rate", "sample_accuracy", "n_unsolved_questions"]),
        ("filtered/report.json", ["n_kept", "improve_yield", "cliff/count", "cliff/ratio",
                                  "cliff/conversion_rate", "cliff/conversion_histogram"]),
        ("dataset/stats.json", ["sft_total", "dpo_total"]),
        ("eval/metrics.json", None),
        ("benchmark_eval/metrics.json", None),
    ]:
        p = it_dir / rel
        if p.exists():
            data = read_json(p)
            for key, val in data.items():
                if keys is None or key in keys:
                    if key != "iter":
                        row[key] = val
    append_jsonl(run_dir / "metrics.jsonl", row)
    print(f"[loop] iter{k} metrics: {row}")
    return row


def _update_state(run_dir: Path, cfg: Config, k: int) -> None:
    write_json(run_dir / "state.json", {
        "last_completed_iter": k,
        "config_hash": cfg.hash(),
        "latest_ckpt": str(iter_dir(run_dir, k) / "ckpt"),
    })


if __name__ == "__main__":
    main(sys.argv[1:])
