import json
import os
from pathlib import Path

import pytest

from expert_iter import engine
from expert_iter.config import EngineCfg
from expert_iter.engine import GenRequest, GenResult
from expert_iter.utils import (is_done, mark_done, read_json, read_jsonl,
                               write_jsonl)


def test_is_done_requires_primary_artifact(tmp_path):
    output = tmp_path / "missing.jsonl"
    mark_done(output, count=1, config_hash="x")
    assert not is_done(output, config_hash="x")


def test_run_pool_invalidates_stale_worker_output(tmp_path, monkeypatch):
    work = tmp_path / "pool"
    work.mkdir()
    stale_out = work / "out_0.jsonl"
    write_jsonl(stale_out, [{"rid": "q", "samples": [{"text": "STALE"}]}])
    mark_done(stale_out, count=1, config_hash="old-key")

    calls = []

    class FakeProc:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            calls.append(cmd)
            _serve_shards(cmd, text="FRESH")

        def poll(self):
            return self.returncode

    monkeypatch.setattr(engine.subprocess, "Popen", FakeProc)
    request = GenRequest(rid="q", prompt_token_ids=[1, 2], seed=3)
    cfg = EngineCfg(gpus=[0], max_model_len=4)

    result = engine.run_pool(
        [request], mode="generate", model_path="org/model",
        sampling={"temperature": 0.7, "max_tokens": 8},
        engine_cfg=cfg, work_dir=work, dtype="float16",
    )
    assert result[0].samples[0]["text"] == "FRESH"
    assert len(calls) == 1
    engine_json = json.loads(calls[0][calls[0].index("--engine-json") + 1])
    assert engine_json["max_model_len"] == 10
    assert engine_json["dtype"] == "float16"

    engine.run_pool(
        [request], mode="generate", model_path="org/model",
        sampling={"temperature": 0.7, "max_tokens": 8},
        engine_cfg=cfg, work_dir=work, dtype="float16",
    )
    assert len(calls) == 1

    engine.run_pool(
        [request], mode="generate", model_path="org/model",
        sampling={"temperature": 0.9, "max_tokens": 8},
        engine_cfg=cfg, work_dir=work, dtype="float16",
    )
    assert len(calls) == 2


def _serve_shards(cmd, *, text="ok"):
    """Stand in for a vLLM worker: claim every unclaimed shard in the manifest
    and answer its requests. Mirrors the real worker's contract (claim dir,
    per-shard output + .done stamped with that shard's key)."""
    manifest = read_json(cmd[cmd.index("--manifest") + 1])["shards"]
    served = []
    for shard in manifest:
        if shard["done"]:
            continue
        try:
            os.mkdir(shard["claim"])
        except FileExistsError:
            continue
        rows = [
            {"rid": row["rid"],
             "samples": [{"text": text, "token_ids": [7], "finish_reason": "stop"}]}
            for row in read_jsonl(shard["input"])
        ]
        write_jsonl(shard["output"], rows)
        mark_done(shard["output"], count=len(rows), config_hash=shard["key"])
        served.append(shard["index"])
    return served


class _FakeProcFactory:
    """Reusable fake worker: records commands, serves every unclaimed shard."""

    def __init__(self):
        self.calls = []
        self.served = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        self.served.append(_serve_shards(cmd))
        return self

    returncode = 0

    def poll(self):
        return self.returncode


def test_new_request_fields_invalidate_worker_cache(tmp_path, monkeypatch):
    fake = _FakeProcFactory()
    monkeypatch.setattr(engine.subprocess, "Popen", fake)
    cfg = EngineCfg(gpus=[0], max_model_len=4, enable_lora=True)

    def run(req, sampling=None):
        return engine.run_pool(
            [req], mode="generate", model_path="org/model",
            sampling=sampling or {"temperature": 0.7, "max_tokens": 8},
            engine_cfg=cfg, work_dir=tmp_path, dtype="float16",
        )

    base = GenRequest(rid="q", prompt_token_ids=[1, 2], seed=3)
    run(base)
    assert len(fake.calls) == 1
    run(base)                                                   # cache hit
    assert len(fake.calls) == 1
    run(GenRequest(rid="q", prompt_token_ids=[1, 2], seed=3, lora_path="/a"))
    assert len(fake.calls) == 2                                 # lora_path in the key
    run(GenRequest(rid="q", prompt_token_ids=[1, 2], seed=3, max_tokens=4))
    assert len(fake.calls) == 3                                 # per-request max_tokens too
    run(base, sampling={"temperature": 0.7, "max_tokens": 8,
                        "return_token_logprobs": True})
    assert len(fake.calls) == 4                                 # sampling extras too
    engine_json = json.loads(fake.calls[1][fake.calls[1].index("--engine-json") + 1])
    assert engine_json["enable_lora"] is True
    assert engine_json["max_loras"] == 8 and engine_json["max_lora_rank"] == 16


def test_lora_path_requires_enable_lora(tmp_path):
    with pytest.raises(ValueError, match="enable_lora"):
        engine.run_pool(
            [GenRequest(rid="q", prompt_token_ids=[1], lora_path="/x")],
            mode="generate", model_path="org/model",
            sampling={"max_tokens": 4},
            engine_cfg=EngineCfg(gpus=[0]), work_dir=tmp_path,
        )


def test_per_request_max_tokens_grows_context(tmp_path, monkeypatch):
    fake = _FakeProcFactory()
    monkeypatch.setattr(engine.subprocess, "Popen", fake)
    cfg = EngineCfg(gpus=[0], max_model_len=4)
    engine.run_pool(
        [GenRequest(rid="q", prompt_token_ids=[1, 2], max_tokens=20)],
        mode="generate", model_path="org/model",
        sampling={"temperature": 0.7, "max_tokens": 8},
        engine_cfg=cfg, work_dir=tmp_path, dtype="float16",
    )
    engine_json = json.loads(fake.calls[0][fake.calls[0].index("--engine-json") + 1])
    assert engine_json["max_model_len"] == 22   # per-request override, not sampling's 8


def test_old_schema_result_rows_deserialize():
    row = {"rid": "x", "samples": [], "sum_logprob": -1.0, "mean_logprob": -0.5,
           "n_scored": 2, "meta": {}}
    r = GenResult(**row)
    assert r.token_logprobs is None and r.topk_logprobs is None


# ---------------------------------------------------------------------------
# Worker-level incremental checkpointing (_resume_done_rids)
#
# A shard is one worker's whole slice of the pool — hours to days on a
# full-dataset sweep. These pin the contract that makes a kill cost one chunk
# instead of the entire shard.
# ---------------------------------------------------------------------------

def test_resume_rids_absent_output(tmp_path):
    assert engine._resume_done_rids(tmp_path / "never_started.jsonl") == set()


def test_resume_rids_reads_complete_rows(tmp_path):
    out = tmp_path / "out_0.jsonl"
    write_jsonl(out, [{"rid": "a", "samples": []}, {"rid": "b", "samples": []}])
    assert engine._resume_done_rids(out) == {"a", "b"}


def test_resume_rids_drops_and_repairs_torn_final_line(tmp_path):
    # A kill mid-append leaves a half-written last line. Appending after it
    # would corrupt the shard permanently, so the file is rewritten intact.
    out = tmp_path / "out_0.jsonl"
    out.write_text('{"rid": "a", "samples": []}\n{"rid": "b", "samples": []}\n{"rid": "c", "sam')
    assert engine._resume_done_rids(out) == {"a", "b"}
    assert [r["rid"] for r in read_jsonl(out)] == ["a", "b"]


def test_generate_chunk_size_reaches_the_worker(tmp_path, monkeypatch):
    fake = _FakeProcFactory()
    monkeypatch.setattr(engine.subprocess, "Popen", fake)
    engine.run_pool(
        [GenRequest(rid="q", prompt_token_ids=[1, 2])],
        mode="generate", model_path="org/model", sampling={"max_tokens": 4},
        engine_cfg=EngineCfg(gpus=[0], generate_chunk_size=32), work_dir=tmp_path,
    )
    engine_json = json.loads(fake.calls[0][fake.calls[0].index("--engine-json") + 1])
    assert engine_json["generate_chunk_size"] == 32


# ---------------------------------------------------------------------------
# Dynamic shard claiming (replaces the old one-shard-per-worker round robin).
#
# Generation length varies by an order of magnitude, so a statically assigned
# shard strands a GPU at the tail (measured: 12% of one GPU on a 500-question
# rollout). These pin the properties that make dynamic claiming safe.
# ---------------------------------------------------------------------------

def _shards(work_dir):
    return read_json(work_dir / "shards.json")["shards"]


def test_many_shards_and_results_stay_in_request_order(tmp_path, monkeypatch):
    fake = _FakeProcFactory()
    monkeypatch.setattr(engine.subprocess, "Popen", fake)
    monkeypatch.setattr(engine, "SHARDS_PER_WORKER", 4)
    reqs = [GenRequest(rid=f"q{i:02d}", prompt_token_ids=[i]) for i in range(20)]
    out = engine.run_pool(
        list(reversed(reqs)), mode="generate", model_path="org/model",
        sampling={"max_tokens": 4}, engine_cfg=EngineCfg(gpus=[0, 1]),
        work_dir=tmp_path,
    )
    assert len(_shards(tmp_path)) > 2          # finer than one shard per worker
    assert len(out) == 20                       # merged back in REQUEST order
    assert all(r.samples[0]["text"] == "ok" for r in out)


def test_a_shard_never_mixes_lora_paths(tmp_path, monkeypatch):
    """The worker issues one generate call per adapter; keeping each shard on a
    single path is what lets engine.max_loras stay small."""
    fake = _FakeProcFactory()
    monkeypatch.setattr(engine.subprocess, "Popen", fake)
    monkeypatch.setattr(engine, "SHARDS_PER_WORKER", 3)
    reqs = [
        GenRequest(rid=f"q{i:02d}", prompt_token_ids=[i],
                   lora_path=None if i < 4 else f"/ad{i % 3}")
        for i in range(16)
    ]
    engine.run_pool(
        reqs, mode="generate", model_path="org/model", sampling={"max_tokens": 4},
        engine_cfg=EngineCfg(gpus=[0, 1], enable_lora=True), work_dir=tmp_path,
    )
    for shard in _shards(tmp_path):
        paths = {row.get("lora_path") for row in read_jsonl(shard["input"])}
        assert len(paths) == 1, f"shard {shard['index']} mixes {paths}"


def test_finished_shards_are_skipped_and_stale_claims_cleared(tmp_path, monkeypatch):
    fake = _FakeProcFactory()
    monkeypatch.setattr(engine.subprocess, "Popen", fake)
    monkeypatch.setattr(engine, "SHARDS_PER_WORKER", 4)
    reqs = [GenRequest(rid=f"q{i:02d}", prompt_token_ids=[i]) for i in range(12)]
    kw = dict(mode="generate", model_path="org/model", sampling={"max_tokens": 4},
              engine_cfg=EngineCfg(gpus=[0]), work_dir=tmp_path)
    engine.run_pool(reqs, **kw)
    n_shards = len(_shards(tmp_path))
    assert fake.served[0] == list(range(n_shards))

    # A killed run leaves a claim behind on an unfinished shard: without the
    # sweep that shard would never be claimed again.
    victim = _shards(tmp_path)[1]
    Path(victim["output"]).unlink()
    Path(victim["output"] + ".done").unlink()
    engine.run_pool(reqs, **kw)
    assert fake.served[-1] == [victim["index"]]     # only the unfinished one reran


def test_lora_int_ids_are_stable_across_shards():
    """Two adapters must never share an int id inside one engine: vLLM caches
    by that id, so a per-shard restart would silently serve the wrong adapter."""
    engine._LORA_INT_IDS.clear()
    a, b = engine._lora_int_id("/ad_a"), engine._lora_int_id("/ad_b")
    assert a != b
    assert engine._lora_int_id("/ad_a") == a       # stable on revisit


def test_outputs_from_a_previous_sharding_scheme_are_cleared(tmp_path, monkeypatch):
    """Shard i means something different under the old round-robin layout, and
    the worker APPENDS to its output — so a stale out_i.jsonl left in the work
    dir (or hardlinked in from a forked run) must be dropped, not extended."""
    fake = _FakeProcFactory()
    monkeypatch.setattr(engine.subprocess, "Popen", fake)
    stale = tmp_path / "out_0.jsonl"
    write_jsonl(stale, [{"rid": "from-old-scheme", "samples": []}])
    mark_done(stale, count=1, config_hash="v2-key")

    reqs = [GenRequest(rid=f"q{i}", prompt_token_ids=[i]) for i in range(4)]
    out = engine.run_pool(
        reqs, mode="generate", model_path="org/model", sampling={"max_tokens": 4},
        engine_cfg=EngineCfg(gpus=[0]), work_dir=tmp_path,
    )
    assert [r.samples[0]["text"] for r in out] == ["ok"] * 4
    rids = {row["rid"] for f in tmp_path.glob("out_*.jsonl") for row in read_jsonl(f)}
    assert "from-old-scheme" not in rids
    assert read_json(tmp_path / "pool_scheme.json") == {"scheme": "shard_claim_v3"}


def test_resume_after_a_kill_mid_shard(tmp_path, monkeypatch):
    """A kill leaves: finished shards (.done), one shard with a stale claim and
    a PARTIAL output, and untouched shards. The rerun must keep the finished
    work, re-claim the interrupted shard, skip the rows it already emitted, and
    finish the rest."""
    monkeypatch.setattr(engine, "SHARDS_PER_WORKER", 4)
    reqs = [GenRequest(rid=f"q{i:02d}", prompt_token_ids=[i]) for i in range(16)]
    kw = dict(mode="generate", model_path="org/model", sampling={"max_tokens": 4},
              engine_cfg=EngineCfg(gpus=[0]), work_dir=tmp_path)

    # --- first attempt: a worker that dies after two shards, mid-way through
    #     the second (partial output, claim still held).
    class _DyingProc:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            manifest = read_json(cmd[cmd.index("--manifest") + 1])["shards"]
            for n_done, shard in enumerate(manifest):
                os.mkdir(shard["claim"])
                rows = [{"rid": r["rid"], "samples": [{"text": "ok", "token_ids": [7],
                                                       "finish_reason": "stop"}]}
                        for r in read_jsonl(shard["input"])]
                if n_done < 2:
                    write_jsonl(shard["output"], rows)
                    mark_done(shard["output"], count=len(rows), config_hash=shard["key"])
                else:
                    write_jsonl(shard["output"], rows[:1])   # torn off mid-shard
                    return                                    # ...and killed

        def poll(self):
            return self.returncode

    monkeypatch.setattr(engine.subprocess, "Popen", _DyingProc)
    with pytest.raises(RuntimeError, match="no result"):
        engine.run_pool(reqs, **kw)

    shards = read_json(tmp_path / "shards.json")["shards"]
    assert Path(shards[2]["claim"]).exists()                  # stale claim left behind
    assert not Path(shards[2]["output"] + ".done").exists()
    partial = {r["rid"] for r in read_jsonl(shards[2]["output"])}
    assert len(partial) == 1

    # --- resume
    fake = _FakeProcFactory()
    monkeypatch.setattr(engine.subprocess, "Popen", fake)
    out = engine.run_pool(reqs, **kw)

    assert [r.rid for r in (reqs)] == [r.rid for r in reqs]   # order preserved
    assert len(out) == 16 and all(r.samples for r in out)
    served = fake.served[0]
    assert 0 not in served and 1 not in served                # finished shards untouched
    assert 2 in served                                        # interrupted shard re-claimed
    rows = list(read_jsonl(shards[2]["output"]))
    assert len({r["rid"] for r in rows}) == len(rows)         # no duplicated rows
