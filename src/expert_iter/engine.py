"""vLLM data-parallel pool.

Client side (`run_pool`): split rid-sorted requests into MANY single-
lora-path shards (~8 per worker), spawn one `python -m expert_iter.engine
--worker` subprocess per GPU slice, and let workers CLAIM shards one at a
time (atomic mkdir) — dynamic assignment, so one straggler shard never
leaves the other GPUs idle. Outputs are merged in request order.

Why subprocesses + files instead of an in-process pool: vLLM owns the CUDA
context and does not tolerate fork; separate processes guarantee GPUs are
fully released between stages (so `accelerate launch` never fights vLLM for
memory), crashes surface as exit codes with a log tail, and per-shard `.done`
markers give worker-level resume for free.

Modes:
  generate — sample continuations; per-request seed = stable_seed(...) so
             results are independent of request ORDER within a fixed pool
             topology (verified: same 1-GPU rerun is bitwise identical; with
             >1 worker the dynamic shard claiming makes even a same-topology
             rerun batch-compose differently, so only 1-worker pools are
             bitwise rerunnable).
             They are NOT independent of pool size/sharding or GPU model:
             batch composition changes kernel numerics, and at temperature
             sampling the sequences diverge (verified: 1-GPU vs 2-GPU pool
             produced 0/160 identical responses). Pin engine.gpus for runs
             you may want to reproduce exactly.
  score    — teacher-forced per-token logprobs over a suffix of the sequence
             (prompt_logprobs trick); used by the trainability logprob gate,
             anchor divergence signals, and C(y) candidate selection. Optional
             sampling keys: return_token_logprobs=true adds the per-position
             realized-token logprob array to each result; prompt_logprobs_k=K
             adds per-position top-K {token_id: logprob} maps (needs
             engine.max_logprobs >= K).

Per-request LoRA: GenRequest.lora_path routes that request through a PEFT
adapter dir (engine.enable_lora must be true). Requests with different
adapters (or none) can share one pool launch.
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
from .utils import (append_jsonl, is_done, mark_done, read_json, read_jsonl,
                    stable_hash, visible_gpus, write_json, write_jsonl)


@dataclass
class GenRequest:
    """One unit of work. `rid` must be unique and stable across reruns."""

    rid: str
    prompt_token_ids: list[int]
    n: int = 1
    seed: int = 0
    # score mode only: score tokens from this index onward
    score_from: int = 0
    # PEFT adapter dir for THIS request (generate and score); None = base model.
    # Adapter dirs must be content-addressed: the pool cache key hashes this
    # path string, not the weights behind it.
    lora_path: str | None = None
    # generate mode: per-request override of sampling["max_tokens"]
    max_tokens: int | None = None
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
    # score mode extras. Alignment contract: index i corresponds to sequence
    # position max(1, score_from) + i — the same positions the aggregates
    # cover. Positions vLLM returned no entry for are None-filled.
    token_logprobs: list[float | None] | None = None    # sampling["return_token_logprobs"]
    topk_logprobs: list[dict[str, float] | None] | None = None  # sampling["prompt_logprobs_k"]>0;
    # per-position {str(token_id): logprob}, realized token first
    meta: dict = field(default_factory=dict)


# Shards per worker. Higher = finer-grained dynamic balancing (shorter idle
# tail) but more llm.generate calls and more .done bookkeeping; 8 empties the
# measured 12% tail without making shards smaller than a generate chunk.
SHARDS_PER_WORKER = 8


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
    if mode == "generate":
        default_max = sampling.get("max_tokens", 0)
        required_context = max(
            len(r.prompt_token_ids) + (r.max_tokens if r.max_tokens is not None else default_max)
            for r in requests
        )
    else:
        # +1: score mode sends the whole sequence as the prompt and asks for
        # one token, and vLLM rejects a prompt whose length EQUALS
        # max_model_len on a generate runner.
        required_context = max(len(r.prompt_token_ids) for r in requests) + 1
        # Leave headroom for the full-vocab logprobs spike (see EngineCfg).
        if engine_cfg.score_gpu_memory_utilization < engine_cfg.gpu_memory_utilization:
            engine_cfg = replace(
                engine_cfg,
                gpu_memory_utilization=engine_cfg.score_gpu_memory_utilization,
            )
    if required_context > engine_cfg.max_model_len:
        engine_cfg = replace(engine_cfg, max_model_len=required_context)
    if any(r.lora_path for r in requests) and not engine_cfg.enable_lora:
        raise ValueError("requests carry lora_path but engine.enable_lora is false")
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

    # Deterministic sharding: rid-sorted, split by lora_path, then cut into many
    # small shards (~SHARDS_PER_WORKER each) that workers CLAIM dynamically.
    #
    # Why not the old round-robin one-shard-per-worker: generation length varies
    # by an order of magnitude (a cliff question runs to the token cap while an
    # easy one stops in hundreds), so a statically assigned shard finishes at a
    # time nobody can predict. MEASURED on a 500-question rollout: worker 0 sat
    # idle for the last 16 of 134 minutes (12% of one GPU) waiting for worker 1.
    # Splitting finer does NOT help under static assignment — every worker still
    # owns the same total work — so the fix has to be dynamic claiming.
    #
    # Shards never straddle a lora_path, which preserves the worker's
    # one-adapter-per-generate-call property (see the grouping loop in _worker);
    # that is what lets engine.max_loras stay small no matter how many transient
    # adapters a run produces.
    ordered = sorted(requests, key=lambda r: r.rid)
    by_path: dict[str | None, list[GenRequest]] = {}
    for r in ordered:
        by_path.setdefault(r.lora_path, []).append(r)
    target_shards = max(n_workers, n_workers * SHARDS_PER_WORKER)
    shard_size = max(1, -(-len(ordered) // target_shards))   # ceil
    shards: list[list[GenRequest]] = [
        group[i:i + shard_size]
        for path in sorted(by_path, key=lambda p: (p is not None, p or ""))
        for group in [by_path[path]]
        for i in range(0, len(group), shard_size)
    ]

    # Outputs from a DIFFERENT sharding scheme must never be mistaken for this
    # one's: shard i now covers different requests, and the worker appends in
    # place, so a stale out_i.jsonl would be silently extended (and, in a
    # hardlink fork, would corrupt the source run). The .done keys already
    # differ, but a partial output carries no key at all — so gate the whole
    # directory on a scheme stamp instead.
    scheme_path = work_dir / "pool_scheme.json"
    scheme = {"scheme": "shard_claim_v3"}
    if scheme_path.exists():
        stale = read_json(scheme_path) != scheme
    else:
        stale = any(work_dir.glob("out_*.jsonl"))     # pre-stamp layout
    if stale:
        print(f"[pool] {work_dir}: clearing outputs from a previous sharding scheme",
              flush=True)
        for old_path in (*work_dir.glob("out_*.jsonl"), *work_dir.glob("out_*.jsonl.done"),
                         *work_dir.glob("shard_*.jsonl")):
            old_path.unlink()
        for claim in work_dir.glob("claim_*"):
            claim.rmdir()
    write_json(scheme_path, scheme)

    manifest: list[dict] = []
    pending = 0
    for si, shard in enumerate(shards):
        in_path = work_dir / f"shard_{si}.jsonl"
        out_path = work_dir / f"out_{si}.jsonl"
        claim_path = work_dir / f"claim_{si}"
        write_jsonl(in_path, (r.to_dict() for r in shard))
        # The GPU slice is deliberately NOT in the key: which worker runs a
        # shard is now a scheduling detail, and results do not depend on it.
        shard_key = stable_hash(
            "pool_shard_v3", mode, _model_fingerprint(model_path), sampling,
            asdict(engine_cfg), dtype, tuple(r.to_dict() for r in shard),
        )
        if is_done(out_path, config_hash=shard_key):
            manifest.append({"index": si, "input": str(in_path), "output": str(out_path),
                             "claim": str(claim_path), "key": shard_key, "done": True})
            continue
        # A claim left by a killed run would block this shard forever.
        if claim_path.exists():
            claim_path.rmdir()
        pending += 1
        manifest.append({"index": si, "input": str(in_path), "output": str(out_path),
                         "claim": str(claim_path), "key": shard_key, "done": False})

    procs: list[tuple[subprocess.Popen, Path, Path]] = []
    if pending:
        manifest_path = work_dir / "shards.json"
        write_json(manifest_path, {"shards": manifest})
    for wi, wgpus in enumerate(worker_gpus):
        if not pending:
            break
        log_path = work_dir / f"worker_{wi}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in wgpus)
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        cmd = [
            sys.executable, "-m", "expert_iter.engine", "--worker",
            "--model", model_path, "--mode", mode,
            "--manifest", str(manifest_path),
            "--sampling-json", json.dumps(sampling),
            "--engine-json", json.dumps({
                "tensor_parallel": tp,
                "gpu_memory_utilization": engine_cfg.gpu_memory_utilization,
                "dtype": dtype,
                "max_model_len": engine_cfg.max_model_len,
                "enable_prefix_caching": engine_cfg.enable_prefix_caching,
                "enforce_eager": engine_cfg.enforce_eager,
                "score_batch_size": engine_cfg.score_batch_size,
                "enable_lora": engine_cfg.enable_lora,
                "max_loras": engine_cfg.max_loras,
                "max_lora_rank": engine_cfg.max_lora_rank,
                "max_logprobs": engine_cfg.max_logprobs,
                "score_max_num_batched_tokens": engine_cfg.score_max_num_batched_tokens,
                "generate_chunk_size": engine_cfg.generate_chunk_size,
            }),
        ]
        log_f = log_path.open("w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        log_f.close()  # the child owns its duplicated descriptor
        procs.append((proc, log_path, log_path))

    _wait_all(procs, total=len(ordered), work_dir=work_dir)

    results: dict[str, GenResult] = {}
    for entry in manifest:
        out_path = Path(entry["output"])
        if not out_path.exists():
            continue
        for row in read_jsonl(out_path):
            results[row["rid"]] = GenResult(**row)
    missing = [r.rid for r in requests if r.rid not in results]
    if missing:
        raise RuntimeError(f"pool returned no result for {len(missing)} requests, e.g. {missing[:5]}")
    return [results[r.rid] for r in requests]


def _resume_done_rids(out_path: str | Path) -> set[str]:
    """rids already emitted into a PARTIAL worker output (no .done marker).

    A kill mid-append can leave a torn final line, so parse leniently and
    rewrite the file from the intact rows — appending after a torn line would
    corrupt the shard permanently. A completed (.done) output is left alone;
    run_pool skips those shards before ever starting a worker.
    """
    out_path = Path(out_path)
    if not out_path.exists():
        return set()
    rows, torn = [], False
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                torn = True          # only ever the last line of a killed write
    if torn:
        write_jsonl(out_path, rows)
    return {r["rid"] for r in rows if "rid" in r}


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


def _wait_all(procs, *, total: int, work_dir: Path) -> None:
    last_report = 0.0
    while True:
        alive = [p for p, _, _ in procs if p.poll() is None]
        now = time.time()
        if now - last_report > 30:
            done_count = sum(
                sum(1 for _ in read_jsonl(out)) for out in work_dir.glob("out_*.jsonl")
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
    ap.add_argument("--manifest", required=True,
                    help="shards.json written by run_pool: the shards to claim")
    ap.add_argument("--sampling-json", required=True)
    ap.add_argument("--engine-json", required=True)
    args = ap.parse_args(argv)

    sampling = json.loads(args.sampling_json)
    ecfg = json.loads(args.engine_json)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm_kwargs = dict(
        model=args.model,
        dtype=ecfg["dtype"],
        tensor_parallel_size=ecfg["tensor_parallel"],
        gpu_memory_utilization=ecfg["gpu_memory_utilization"],
        max_model_len=ecfg["max_model_len"],
        enable_prefix_caching=ecfg["enable_prefix_caching"],
        enforce_eager=ecfg["enforce_eager"],
    )
    # Added only when non-default so a plain run constructs the engine with the
    # exact argument set that existed before LoRA support.
    if ecfg.get("enable_lora"):
        llm_kwargs.update(
            enable_lora=True,
            max_loras=int(ecfg.get("max_loras", 8)),
            max_lora_rank=int(ecfg.get("max_lora_rank", 16)),
        )
    if int(ecfg.get("max_logprobs", 20)) != 20:
        llm_kwargs["max_logprobs"] = int(ecfg["max_logprobs"])
    # Score mode only: bound the per-chunk full-vocab fp32 log_softmax spike.
    if args.mode == "score" and ecfg.get("score_max_num_batched_tokens"):
        llm_kwargs["max_num_batched_tokens"] = int(ecfg["score_max_num_batched_tokens"])
    llm = LLM(**llm_kwargs)

    shards = read_json(args.manifest)["shards"]

    # Claim shards one at a time instead of owning a fixed slice: whoever
    # finishes first takes the next unclaimed shard, so a long-running shard
    # cannot leave the other GPUs idle. mkdir is the lock — atomic on POSIX and
    # on NFS — and the engine is built once, outside this loop, so dynamic
    # assignment costs no extra model loads.
    for shard in shards:
        if shard["done"]:
            continue
        try:
            os.mkdir(shard["claim"])
        except FileExistsError:
            continue                      # another worker has it
        _run_shard(shard, args, ecfg, sampling, llm, SamplingParams, TokensPrompt)
    print("[worker] no unclaimed shards left", flush=True)


# vLLM caches adapters by lora_int_id, so the id must identify the ADAPTER for
# the whole life of the engine — not just within one shard. A worker now
# processes many shards, each with its own adapter, so numbering restarted per
# shard would hand two different adapters the same id and silently serve the
# cached one. Ids are process-local (engine-local), which is all vLLM needs.
_LORA_INT_IDS: dict[str, int] = {}


def _lora_int_id(path: str) -> int:
    return _LORA_INT_IDS.setdefault(path, len(_LORA_INT_IDS) + 1)


def _run_shard(shard, args, ecfg, sampling, llm, SamplingParams, TokensPrompt) -> None:
    out_path = shard["output"]
    reqs = [GenRequest(**row) for row in read_jsonl(shard["input"])]

    # Incremental checkpointing. Results are appended per chunk and
    # already-emitted rids are skipped on rerun, so a kill costs at most one
    # chunk rather than the shard (and _wait_all's progress readout moves
    # while a shard is still running).
    done_rids = _resume_done_rids(out_path)
    if done_rids:
        print(f"[worker] shard {shard['index']}: resuming, {len(done_rids)} of "
              f"{len(reqs)} results already on disk", flush=True)
        reqs = [r for r in reqs if r.rid not in done_rids]
    n_written = len(done_rids)

    def emit(new_rows: list[dict]) -> None:
        nonlocal n_written
        for row in new_rows:
            append_jsonl(out_path, row)
        n_written += len(new_rows)

    # Group by adapter: one llm.generate call per adapter (or None = base
    # model). Per-request seeds keep results independent of request order, so
    # grouping does not affect reproducibility beyond the already-documented
    # batch-composition numerics caveat.
    lora_paths = sorted({r.lora_path for r in reqs if r.lora_path})
    if lora_paths and not ecfg.get("enable_lora"):
        raise RuntimeError("requests carry lora_path but the engine was built without enable_lora")

    def lora_kw(path: str | None) -> dict:
        if path is None:
            return {}
        from vllm.lora.request import LoRARequest

        return {"lora_request": LoRARequest(
            lora_name=path, lora_int_id=_lora_int_id(path), lora_path=path,
        )}

    groups: dict[str | None, list[GenRequest]] = {}
    for r in reqs:
        groups.setdefault(r.lora_path, []).append(r)

    if args.mode == "generate":
        # Chunked so a kill costs at most one chunk, not the shard. vLLM cannot
        # run more than max_num_seqs concurrently anyway, so a chunk of
        # generate_chunk_size requests keeps the batch just as full.
        gcs = max(1, int(ecfg.get("generate_chunk_size", 256)))
        for path in [None, *lora_paths]:
            group = groups.get(path)
            if not group:
                continue
            for start in range(0, len(group), gcs):
                chunk = group[start:start + gcs]
                prompts = [TokensPrompt(prompt_token_ids=r.prompt_token_ids) for r in chunk]
                params = [
                    SamplingParams(
                        n=r.n,
                        temperature=sampling.get("temperature", 1.0),
                        top_p=sampling.get("top_p", 1.0),
                        top_k=sampling.get("top_k", -1),
                        min_p=sampling.get("min_p", 0.0),
                        max_tokens=(r.max_tokens if r.max_tokens is not None
                                    else sampling.get("max_tokens", 1024)),
                        stop=sampling.get("stop"),
                        seed=r.seed,
                    )
                    for r in chunk
                ]
                outs = llm.generate(prompts, params, **lora_kw(path))
                emit([GenResult(
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
                ).__dict__ for r, out in zip(chunk, outs)])
    else:  # score
        bs = max(1, ecfg.get("score_batch_size", 256))
        want_tokens = bool(sampling.get("return_token_logprobs"))
        top_k = int(sampling.get("prompt_logprobs_k", 0) or 0)
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=top_k)
        for path in [None, *lora_paths]:
            group = groups.get(path)
            if not group:
                continue
            for start in range(0, len(group), bs):
                chunk = group[start:start + bs]
                chunk_rows: list[dict] = []
                outs = llm.generate(
                    [TokensPrompt(prompt_token_ids=r.prompt_token_ids) for r in chunk],
                    params,
                    **lora_kw(path),
                )
                for r, out in zip(chunk, outs):
                    plp = out.prompt_logprobs or []
                    # Aligned per-position views from max(1, score_from) on;
                    # vLLM's per-position dict lists the realized token FIRST.
                    lps: list[float | None] = []
                    topk: list[dict[str, float] | None] = []
                    for pos in range(max(1, r.score_from), len(plp)):  # pos 0 is always None
                        d = plp[pos]
                        if d:
                            lps.append(next(iter(d.values())).logprob)
                            if top_k > 0:
                                topk.append({str(t): lp.logprob for t, lp in d.items()})
                        else:  # None-fill keeps index i <-> position score_from+i
                            lps.append(None)
                            if top_k > 0:
                                topk.append(None)
                    scored = [v for v in lps if v is not None]
                    chunk_rows.append(GenResult(
                        rid=r.rid,
                        sum_logprob=float(sum(scored)) if scored else None,
                        mean_logprob=float(sum(scored) / len(scored)) if scored else None,
                        n_scored=len(scored),
                        token_logprobs=lps if want_tokens else None,
                        topk_logprobs=topk if top_k > 0 else None,
                        meta=r.meta,
                    ).__dict__)
                emit(chunk_rows)

    mark_done(out_path, count=n_written, config_hash=shard["key"])
    print(f"[worker] shard {shard['index']}: wrote {n_written} results to {out_path}",
          flush=True)


if __name__ == "__main__":
    _worker(sys.argv[1:])
