import json

import pytest

from expert_iter import engine
from expert_iter.config import EngineCfg
from expert_iter.engine import GenRequest, GenResult
from expert_iter.utils import is_done, mark_done, read_jsonl, write_jsonl


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
            output = cmd[cmd.index("--output") + 1]
            input_path = cmd[cmd.index("--input") + 1]
            cache_key = cmd[cmd.index("--cache-key") + 1]
            rows = [
                {
                    "rid": row["rid"],
                    "samples": [{
                        "text": "FRESH",
                        "token_ids": [7],
                        "finish_reason": "stop",
                    }],
                }
                for row in read_jsonl(input_path)
            ]
            write_jsonl(output, rows)
            mark_done(output, count=len(rows), config_hash=cache_key)

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


class _FakeProcFactory:
    """FakeProc as in the test above, but reusable: records commands, answers
    every request from its input shard."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        output = cmd[cmd.index("--output") + 1]
        cache_key = cmd[cmd.index("--cache-key") + 1]
        rows = [
            {"rid": row["rid"],
             "samples": [{"text": "ok", "token_ids": [7], "finish_reason": "stop"}]}
            for row in read_jsonl(cmd[cmd.index("--input") + 1])
        ]
        write_jsonl(output, rows)
        mark_done(output, count=len(rows), config_hash=cache_key)
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
