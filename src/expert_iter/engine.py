"""vLLM data-parallel pool.

Client side (`run_pool`): shard requests round-robin into per-worker JSONL
files, spawn one `python -m expert_iter.engine --worker` subprocess per worker
with its own CUDA_VISIBLE_DEVICES slice, wait, merge outputs in request order.

Why subprocesses + files instead of an in-process pool: vLLM owns the CUDA
context and does not tolerate fork; separate processes guarantee GPUs are
fully released between stages (so `accelerate launch` never fights vLLM for
memory), crashes surface as exit codes with a log tail, and per-shard `.done`
markers give worker-level resume for free.

Modes:
  generate — sample continuations; per-request seed = stable_seed(...) so
             results are independent of request ORDER within a fixed pool
             topology (verified: same 1-GPU rerun is bitwise identical).
             They are NOT independent of pool size/sharding or GPU model:
             batch composition changes kernel numerics, and at temperature
             sampling the sequences diverge (verified: 1-GPU vs 2-GPU pool
             produced 0/160 identical responses). Pin engine.gpus for runs
             you may want to reproduce exactly.
  score    — teacher-forced per-token logprobs over a suffix of the sequence
             (prompt_logprobs trick); used by the trainability logprob gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .config import EngineCfg
from .utils import is_done, mark_done, read_jsonl, stable_hash, visible_gpus, write_jsonl


@dataclass
class GenRequest:
    """One unit of work. `rid` must be unique and stable across reruns."""

    rid: str
    prompt_token_ids: list[int]
    n: int = 1
    seed: int = 0
    # score mode only: score tokens from this index onward
    score_from: int = 0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class GenResult:
    rid: str
    # generate mode: one entry per sample
    samples: list[dict] = field(default_factory=list)   # {text, token_ids, finish_reason}
    # score mode:
    sum_logprob: float | None = None
    mean_logprob: float | None = None
    n_scored: int | None = None
    meta: dict = field(default_factory=dict)


def run_pool(
    requests: list[GenRequest],
    *,
    mode: str,                       # "generate" | "score"
    model_path: str,
    sampling: dict,                  # generate: {temperature, top_p, max_tokens, stop?}
    engine_cfg: EngineCfg,
    work_dir: str | Path,
    gpus: list[int] | None = None,   # override engine_cfg.gpus (e.g. teacher split)
    dtype: str = "bfloat16",
) -> list[GenResult]:
    assert mode in ("generate", "score"), mode
    if not requests:
        return []
    extra_tokens = sampling.get("max_tokens", 0) if mode == "generate" else 0
    required_context = max(len(r.prompt_token_ids) for r in requests) + extra_tokens
    if required_context > engine_cfg.max_model_len:
        engine_cfg = replace(engine_cfg, max_model_len=required_context)
    rids = [r.rid for r in requests]
    if len(set(rids)) != len(rids):
        dupes = sorted(rid for rid in set(rids) if rids.count(rid) > 1)
        raise ValueError(f"GenRequest.rid must be unique; duplicates: {dupes[:5]}")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = visible_gpus(gpus if gpus is not None else engine_cfg.gpus)
    if not gpu_ids:
        raise RuntimeError("no GPUs available (engine.gpus / CUDA_VISIBLE_DEVICES empty)")
    tp = max(1, engine_cfg.tensor_parallel)
    if tp > len(gpu_ids) or len(gpu_ids) % tp:
        raise ValueError(
            f"tensor_parallel={tp} must divide the {len(gpu_ids)} visible GPU(s): {gpu_ids}"
        )
    n_workers = len(gpu_ids) // tp
    worker_gpus = [gpu_ids[i * tp:(i + 1) * tp] for i in range(n_workers)]

    # Deterministic sharding (round-robin over rid-sorted requests).
    ordered = sorted(requests, key=lambda r: r.rid)
    shards: list[list[GenRequest]] = [ordered[i::n_workers] for i in range(n_workers)]

    procs: list[tuple[subprocess.Popen, Path, Path]] = []
    for wi, (shard, wgpus) in enumerate(zip(shards, worker_gpus)):
        in_path = work_dir / f"shard_{wi}.jsonl"
        out_path = work_dir / f"out_{wi}.jsonl"
        log_path = work_dir / f"worker_{wi}.log"
        write_jsonl(in_path, (r.to_dict() for r in shard))
        worker_key = stable_hash(
            "pool_worker_v2", mode, _model_fingerprint(model_path), sampling,
            asdict(engine_cfg), dtype, tuple(wgpus),
            tuple(r.to_dict() for r in shard),
        )
        if not shard:
            write_jsonl(out_path, [])
            mark_done(out_path, count=0, config_hash=worker_key)
            continue
        if is_done(out_path, config_hash=worker_key):
            continue  # worker-level resume
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in wgpus)
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        cmd = [
            sys.executable, "-m", "expert_iter.engine", "--worker",
            "--model", model_path, "--mode", mode,
            "--input", str(in_path), "--output", str(out_path),
            "--sampling-json", json.dumps(sampling),
            "--cache-key", worker_key,
            "--engine-json", json.dumps({
                "tensor_parallel": tp,
                "gpu_memory_utilization": engine_cfg.gpu_memory_utilization,
                "dtype": dtype,
                "max_model_len": engine_cfg.max_model_len,
                "enable_prefix_caching": engine_cfg.enable_prefix_caching,
                "enforce_eager": engine_cfg.enforce_eager,
                "score_batch_size": engine_cfg.score_batch_size,
            }),
        ]
        log_f = log_path.open("w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        log_f.close()  # the child owns its duplicated descriptor
        procs.append((proc, out_path, log_path))

    _wait_all(procs, total=len(ordered), work_dir=work_dir, n_workers=n_workers)

    results: dict[str, GenResult] = {}
    for wi in range(n_workers):
        for row in read_jsonl(work_dir / f"out_{wi}.jsonl"):
            results[row["rid"]] = GenResult(**row)
    missing = [r.rid for r in requests if r.rid not in results]
    if missing:
        raise RuntimeError(f"pool returned no result for {len(missing)} requests, e.g. {missing[:5]}")
    return [results[r.rid] for r in requests]


def _model_fingerprint(model_path: str) -> str:
    """Fast cache identity for a hub reference or local checkpoint.

    Local model weights are often many GB, so use a root-file manifest rather
    than rereading them before every stage. Size + nanosecond mtime catches
    ordinary checkpoint replacement; immutable/revision-pinned hub ids remain
    reproducible by their literal reference.
    """
    path = Path(model_path)
    if not path.exists():
        return stable_hash("hub_model", model_path)
    if path.is_file():
        st = path.stat()
        return stable_hash("model_file", str(path.resolve()), st.st_size, st.st_mtime_ns)
    manifest = []
    for p in sorted(path.iterdir(), key=lambda x: x.name):
        if p.is_file():
            st = p.stat()
            manifest.append((p.name, st.st_size, st.st_mtime_ns))
    return stable_hash("model_dir", str(path.resolve()), tuple(manifest))


def _wait_all(procs, *, total: int, work_dir: Path, n_workers: int) -> None:
    last_report = 0.0
    while True:
        alive = [p for p, _, _ in procs if p.poll() is None]
        now = time.time()
        if now - last_report > 30:
            done_count = sum(
                sum(1 for _ in read_jsonl(work_dir / f"out_{wi}.jsonl"))
                if (work_dir / f"out_{wi}.jsonl").exists() else 0
                for wi in range(n_workers)
            )
            print(f"[pool] {done_count}/{total} results, {len(alive)} workers alive", flush=True)
            last_report = now
        if not alive:
            break
        time.sleep(2)
    for p, out_path, log_path in procs:
        if p.returncode != 0:
            tail = ""
            if log_path.exists():
                tail = "".join(log_path.read_text(errors="replace").splitlines(keepends=True)[-40:])
            raise RuntimeError(
                f"vLLM worker for {out_path.name} exited {p.returncode}; log tail:\n{tail}"
            )


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["generate", "score"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sampling-json", required=True)
    ap.add_argument("--engine-json", required=True)
    ap.add_argument("--cache-key", required=True)
    args = ap.parse_args(argv)

    sampling = json.loads(args.sampling_json)
    ecfg = json.loads(args.engine_json)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model,
        dtype=ecfg["dtype"],
        tensor_parallel_size=ecfg["tensor_parallel"],
        gpu_memory_utilization=ecfg["gpu_memory_utilization"],
        max_model_len=ecfg["max_model_len"],
        enable_prefix_caching=ecfg["enable_prefix_caching"],
        enforce_eager=ecfg["enforce_eager"],
    )

    reqs = [GenRequest(**row) for row in read_jsonl(args.input)]
    rows: list[dict] = []

    if args.mode == "generate":
        prompts = [TokensPrompt(prompt_token_ids=r.prompt_token_ids) for r in reqs]
        params = [
            SamplingParams(
                n=r.n,
                temperature=sampling.get("temperature", 1.0),
                top_p=sampling.get("top_p", 1.0),
                top_k=sampling.get("top_k", -1),
                min_p=sampling.get("min_p", 0.0),
                max_tokens=sampling.get("max_tokens", 1024),
                stop=sampling.get("stop"),
                seed=r.seed,
            )
            for r in reqs
        ]
        outs = llm.generate(prompts, params)
        for r, out in zip(reqs, outs):
            rows.append(GenResult(
                rid=r.rid,
                samples=[
                    {
                        "text": o.text,
                        "token_ids": list(o.token_ids),
                        "finish_reason": o.finish_reason or "stop",
                    }
                    for o in out.outputs
                ],
                meta=r.meta,
            ).__dict__)
    else:  # score
        bs = max(1, ecfg.get("score_batch_size", 256))
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        for start in range(0, len(reqs), bs):
            chunk = reqs[start:start + bs]
            outs = llm.generate(
                [TokensPrompt(prompt_token_ids=r.prompt_token_ids) for r in chunk],
                params,
            )
            for r, out in zip(chunk, outs):
                plp = out.prompt_logprobs or []
                lps: list[float] = []
                for pos in range(max(1, r.score_from), len(plp)):  # pos 0 is always None
                    d = plp[pos]
                    if d:
                        lps.append(next(iter(d.values())).logprob)
                rows.append(GenResult(
                    rid=r.rid,
                    sum_logprob=float(sum(lps)) if lps else None,
                    mean_logprob=float(sum(lps) / len(lps)) if lps else None,
                    n_scored=len(lps),
                    meta=r.meta,
                ).__dict__)

    n = write_jsonl(args.output, rows)
    mark_done(args.output, count=n, config_hash=args.cache_key)
    print(f"[worker] wrote {n} results to {args.output}", flush=True)


if __name__ == "__main__":
    _worker(sys.argv[1:])
