"""staged_bridge_sft operator: stage-1 bridge fit -> rollout -> stage-2 fits
(reuse/regen bridges, self-wash) -> final rollout — monkeypatched pool and fit
(no GPU, no downloads)."""

import json
import subprocess
import sys

import pytest

from expert_iter import lora_sft as ls_mod
from expert_iter.bridge_sft import BridgeSftOperator
from expert_iter.staged_bridge_sft import StagedBridgeSftOperator
from expert_iter.config import Config
from expert_iter.engine import GenResult
from expert_iter.records import AnchorRecord, UnsolvedQuestion
from expert_iter.utils import stable_seed


class FakeTok:
    eos_token_id = 0
    PROMPT_IDS = [9, 9, 9]
    vocab = {
        101: "The answer is \\boxed{4}",
        102: "wrong: \\boxed{5}",
        104: "recovered: \\boxed{7}",
        105: "Longer correct solution text \\boxed{4}",
    }

    def apply_chat_template(self, messages, **kw):
        return "BRIDGE:" + messages[-1]["content"][:8]

    def __call__(self, text, **kw):
        return {"input_ids": list(self.PROMPT_IDS)}

    def decode(self, ids):
        return " ".join(self.vocab.get(i, "") for i in ids)


def _sample(tok_id, ids=None):
    return {"text": FakeTok.vocab[tok_id],
            "token_ids": list(ids) if ids is not None else [tok_id],
            "finish_reason": "stop"}


def _fixtures():
    questions = [
        UnsolvedQuestion(qid="q1", question="2+2?", final_answer="4"),
        UnsolvedQuestion(qid="q2", question="3+4?", final_answer="7"),
    ]
    anchors = [
        AnchorRecord(qid=q.qid, base_sample_idx=0, policy="none",
                     anchor_token_ids=[], anchor_text="", anchor_len=0,
                     base_response_len=10)
        for q in questions
    ]
    prompts = {"q1": [1, 2], "q2": [1, 2]}
    gold = {"q1": "It is 4.", "q2": "It is 7."}
    return questions, anchors, prompts, gold


def _cfg(*extra):
    return Config.load(None, overrides=[
        "improve.operator=staged_bridge_sft", "engine.enable_lora=true",
        "improve.lora_sft.bridge.n=2",
        "improve.lora_sft.staged.rollout_n=2",
        "improve.lora_sft.staged.final_rollout_n=3",
        *extra,
    ])


def _stage_of(work_dir):
    """Launch site from the run_pool work_dir: 'bridge' / 'stage1' /
    'stage1_bridge' / 'stage2' / 'wash:stageK' ..."""
    parts = work_dir.parts
    name = work_dir.name
    if name in ("bridge", "bridge_retry"):
        for p in parts:
            if p.endswith("_bridge"):
                return p                      # regen: stage{k}_bridge
        return "bridge"                       # stage-1 bridges
    if name.endswith("_roll"):
        return name[:-len("_roll")]           # stage{k}
    if name == "pool_wash":
        return f"wash:{parts[-2]}"
    return name


def _setup(monkeypatch, responder):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: FakeTok())
    fits = {"order": [], "pairs": {}, "params": {}, "init": {}}

    def fake_fit(policy, pairs, adapters_dir, name, params, cfg, *,
                 init_adapter=None, wandb=None):
        fits["order"].append(name)
        fits["pairs"][name] = list(pairs)
        fits["params"][name] = dict(params)
        fits["init"][name] = init_adapter
        adapter_dir = adapters_dir / name / "FAKEKEY"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"lora_alpha": params["lora_alpha"], "r": params["r"]}))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
        return adapter_dir, 0.0

    monkeypatch.setattr(ls_mod, "_fit_adapter", fake_fit)
    launches = []

    def fake_run_pool(requests, **kw):
        stage = _stage_of(kw["work_dir"])
        launches.append({"stage": stage, "requests": list(requests), "kw": kw})
        out = []
        for r in requests:
            qid, kind = r.rid.split(":", 1)
            out.append(GenResult(rid=r.rid, samples=responder(qid, kind, r, stage)))
        return out

    monkeypatch.setattr(ls_mod, "run_pool", fake_run_pool)
    return fits, launches


def _propose(cfg, tmp_path, fixtures, op=None):
    questions, anchors, prompts, gold = fixtures
    return (op or StagedBridgeSftOperator()).propose(
        questions, anchors, prompts, cfg,
        model_paths={"policy": "org/policy"},
        work_dir=tmp_path / "pool", iteration=0, gold_solutions=gold)


def _default_responder(qid, kind, r, stage):
    """Bridges succeed for both; q1 solves at the stage-1 rollout, q2 only at
    the final rollout."""
    if kind in ("bridge", "bridge_retry"):
        return [_sample(101)] if qid == "q1" else [_sample(104)]
    assert kind == "a1"
    if stage == "stage1":
        return [_sample(101), _sample(102)] if qid == "q1" else [_sample(102)]
    return [_sample(102)] if qid == "q1" else [_sample(104)]


def test_default_flow(monkeypatch, tmp_path):
    cfg = _cfg()
    fits, launches = _setup(monkeypatch, _default_responder)
    cands = _propose(cfg, tmp_path, _fixtures())

    # two fits: stage1 at fit.steps, stage2 at stage2_steps chained from stage1
    assert fits["order"] == ["stage1", "stage2"]
    assert fits["params"]["stage1"]["steps"] == cfg.improve.lora_sft.fit.steps
    assert fits["params"]["stage2"]["steps"] == cfg.improve.lora_sft.staged.stage2_steps
    assert fits["init"]["stage1"] is None
    assert str(fits["init"]["stage2"]).endswith("stage1/FAKEKEY")
    # default unsolved_only + reuse_bridge: stage2 pairs = q2's stage-1 bridge
    assert {p["qid"] for p in fits["pairs"]["stage2"]} == {"q2"}
    assert fits["pairs"]["stage2"][0]["input_ids"] == [1, 2, 104, 0]
    # stage-1 rollout covers both qids at rollout_n; final covers only q2 at final_rollout_n
    roll1 = next(l for l in launches if l["stage"] == "stage1")
    assert {r.rid for r in roll1["requests"]} == {"q1:a1", "q2:a1"}
    assert all(r.n == 2 for r in roll1["requests"])
    final = next(l for l in launches if l["stage"] == "stage2")
    assert [r.rid for r in final["requests"]] == ["q2:a1"]
    assert final["requests"][0].n == 3
    # pool: correct prefilled, per-round stage stamped, staged provenance
    assert all(c.correct is not None for c in cands)
    q1 = next(c for c in cands if c.qid == "q1" and c.correct)
    assert q1.op_meta["stage"] == "stage1" and q1.op_meta["staged"] is True
    assert q1.external_context is None and q1.operator == "staged_bridge_sft"
    q2 = next(c for c in cands if c.qid == "q2" and c.correct)
    assert q2.op_meta["stage"] == "stage2"
    # attempt_idx unique across rounds
    keys = [(c.qid, c.base_sample_idx, c.attempt_idx) for c in cands]
    assert len(keys) == len(set(keys))
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_resolved_stage1"] == 1
    assert stats["n_resolved_by_stage"] == {"stage1": 1, "stage2": 1}
    assert stats["n_resolved_final"] == 2 and stats["n_stages_run"] == 1
    assert stats["lora_yield"] == 1.0 and stats["pool_size"] == len(cands)


def test_chain_adapter_off(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.chain_adapter=false")
    fits, _ = _setup(monkeypatch, _default_responder)
    _propose(cfg, tmp_path, _fixtures())
    assert fits["init"]["stage2"] is None


def test_regen_bridge_through_adapter(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.unsolved_targets=regen_bridge")

    def responder(qid, kind, r, stage):
        if stage == "bridge":                       # stage-1 bridges: base policy
            assert r.lora_path is None
            assert r.seed == stable_seed(cfg.run.seed, kind, 0, qid)
            return [_sample(101)] if qid == "q1" else [_sample(102)]
        if stage == "stage1_bridge":                # regen: THROUGH the adapter
            assert qid == "q2"
            assert str(r.lora_path).endswith("stage1/FAKEKEY")
            assert r.seed == stable_seed(cfg.run.seed, kind, 0, qid, "stage1")
            return [_sample(104)]
        if kind == "a1" and stage == "stage1":
            return [_sample(101)] if qid == "q1" else [_sample(102)]
        return [_sample(104)]

    fits, launches = _setup(monkeypatch, responder)
    _propose(cfg, tmp_path, _fixtures())
    # q2 had no stage-1 bridge; the regenerated one becomes the stage-2 pair
    assert {p["qid"] for p in fits["pairs"]["stage2"]} == {"q2"}
    assert fits["pairs"]["stage2"][0]["input_ids"] == [1, 2, 104, 0]
    assert any(l["stage"] == "stage1_bridge" for l in launches)
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stage1_bridge"]["n_questions_bridged"] == 1
    # seam reset after the regen call
    op = StagedBridgeSftOperator()
    assert getattr(op, "_bridge_lora_path", None) is None


def test_full_pool_self_wash_min_c(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.train_scope=full_pool")

    def responder(qid, kind, r, stage):
        if kind in ("bridge", "bridge_retry"):
            return [_sample(101)] if qid == "q1" else [_sample(104)]
        if stage == "stage1":
            if qid == "q1":     # two correct rollouts: short and long
                return [_sample(101), _sample(105, ids=[105, 105, 105])]
            return [_sample(102)]
        return [_sample(104)]

    fits, _ = _setup(monkeypatch, responder)

    import expert_iter.filters as filters_mod
    from expert_iter.filters import _cand_key
    wash_calls = {}

    def fake_score(survivors, cfg_, model_path, it_dir, *, pool_dir=None):
        wash_calls["pool_dir"] = pool_dir
        # give the LONG rollout the smallest C so the pick is score-driven,
        # not length-driven
        return ({_cand_key(c): {"_c_raw": 0.1 if len(c.continuation_token_ids) > 1
                                else 0.9}
                 for c in survivors}, {})

    monkeypatch.setattr(filters_mod, "_score_candidates", fake_score)
    _propose(cfg, tmp_path, _fixtures())

    assert wash_calls["pool_dir"] is not None
    assert wash_calls["pool_dir"].name == "pool_wash"
    # stage2 pairs: q2's bridge (unsolved) + q1's min-C rollout (solved)
    by_qid = {p["qid"]: p for p in fits["pairs"]["stage2"]}
    assert set(by_qid) == {"q1", "q2"}
    assert by_qid["q1"]["input_ids"] == [1, 2, 105, 105, 105, 0]
    assert by_qid["q1"]["prompt_len"] == 2


@pytest.mark.parametrize("mode,expected_cont", [
    ("shortest", [101, 0]),
    ("longest", [105, 105, 105, 0]),
    ("bridge", [101, 0]),   # q1's stage-1 bridge is [101]
])
def test_solved_targets_without_scoring(monkeypatch, tmp_path, mode, expected_cont):
    cfg = _cfg("improve.lora_sft.staged.train_scope=full_pool",
               f"improve.lora_sft.staged.solved_targets={mode}")

    def responder(qid, kind, r, stage):
        if kind in ("bridge", "bridge_retry"):
            return [_sample(101)] if qid == "q1" else [_sample(104)]
        if stage == "stage1":
            if qid == "q1":
                return [_sample(101), _sample(105, ids=[105, 105, 105])]
            return [_sample(102)]
        return [_sample(104)]

    fits, _ = _setup(monkeypatch, responder)

    import expert_iter.filters as filters_mod

    def boom(*a, **k):
        raise AssertionError("_score_candidates must not run for " + mode)

    monkeypatch.setattr(filters_mod, "_score_candidates", boom)
    _propose(cfg, tmp_path, _fixtures())
    q1 = next(p for p in fits["pairs"]["stage2"] if p["qid"] == "q1")
    assert q1["input_ids"] == [1, 2] + expected_cont


def test_final_rollout_scope_all(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.final_rollout_scope=all")
    _, launches = _setup(monkeypatch, _default_responder)
    _propose(cfg, tmp_path, _fixtures())
    final = next(l for l in launches if l["stage"] == "stage2")
    assert {r.rid.split(":")[0] for r in final["requests"]} == {"q1", "q2"}


def test_num_stages_two(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.num_stages=2")

    def responder(qid, kind, r, stage):
        if kind in ("bridge", "bridge_retry"):
            return [_sample(101)] if qid == "q1" else [_sample(104)]
        if stage in ("stage1", "stage2"):   # q2 stays wrong until the final roll
            return [_sample(101)] if qid == "q1" else [_sample(102)]
        return [_sample(104)]

    fits, launches = _setup(monkeypatch, responder)
    _propose(cfg, tmp_path, _fixtures())

    assert fits["order"] == ["stage1", "stage2", "stage3"]
    mid = next(l for l in launches if l["stage"] == "stage2")
    assert [r.rid for r in mid["requests"]] == ["q2:a1"]     # unsolved only
    assert mid["requests"][0].n == 2                          # rollout_n
    final = next(l for l in launches if l["stage"] == "stage3")
    assert final["requests"][0].n == 3                        # final_rollout_n
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_stages_run"] == 2
    assert stats["n_resolved_by_stage"] == {"stage1": 1, "stage2": 0, "stage3": 1}


def test_emit_final_only(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.emit=final_only")
    _, _ = _setup(monkeypatch, _default_responder)
    cands = _propose(cfg, tmp_path, _fixtures())
    assert cands and all(c.op_meta["stage"] == "stage2" for c in cands)
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_emitted"] == len(cands) < stats["pool_size"]


@pytest.mark.parametrize("bad", [
    "improve.lora_sft.fit.adaptive.enabled=true",
    "improve.lora_sft.project_back.enabled=true",
    "improve.lora_sft.refit_budget=1",
    "improve.lora_sft.adapter_scope=per_problem",
    "improve.lora_sft.chunk_size=2",
    "improve.rl.enabled=true",
])
def test_v1_unsupported_combinations_rejected(bad):
    with pytest.raises(ValueError, match="staged_bridge_sft"):
        _cfg(bad)


@pytest.mark.parametrize("bad,match", [
    ("improve.lora_sft.staged.unsolved_targets=nope", "unsolved_targets"),
    ("improve.lora_sft.staged.solved_targets=nope", "solved_targets"),
    ("improve.lora_sft.staged.train_scope=nope", "train_scope"),
    ("improve.lora_sft.staged.final_rollout_scope=nope", "final_rollout_scope"),
    ("improve.lora_sft.staged.emit=nope", "emit"),
    ("improve.lora_sft.staged.num_stages=0", "num_stages"),
    ("improve.lora_sft.staged.stage2_steps=0", "stage2_steps"),
])
def test_staged_enum_and_range_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        _cfg(bad)


def test_base_bridge_sft_seeds_unchanged(monkeypatch, tmp_path):
    """The lora_path/seed-salt seam must leave plain bridge_sft requests
    byte-identical (seed + no lora_path)."""
    cfg = Config.load(None, overrides=[
        "improve.operator=bridge_sft", "engine.enable_lora=true",
        "improve.lora_sft.bridge.n=2",
    ])

    def responder(qid, kind, r, stage):
        if kind in ("bridge", "bridge_retry"):
            assert r.lora_path is None
            assert r.seed == stable_seed(cfg.run.seed, kind, 0, qid)
            return [_sample(101)] if qid == "q1" else [_sample(104)]
        return [_sample(102)]

    _setup(monkeypatch, responder)
    _propose(cfg, tmp_path, _fixtures(), op=BridgeSftOperator())


def test_import_has_no_cycle_back_into_improve():
    proc = subprocess.run(
        [sys.executable, "-c",
         "import expert_iter.staged_bridge_sft, sys; "
         "assert 'expert_iter.improve' not in sys.modules, 'import cycle back into improve'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
