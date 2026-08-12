import json

from expert_iter import engine
from expert_iter.config import EngineCfg
from expert_iter.engine import GenRequest
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
