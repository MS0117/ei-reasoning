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

    def decode(self, ids, **kw):
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
    # final_only requires final_rollout_scope=all (the unsolved combination is
    # rejected at config load — see test below)
    cfg = _cfg("improve.lora_sft.staged.emit=final_only",
               "improve.lora_sft.staged.final_rollout_scope=all")
    _, _ = _setup(monkeypatch, _default_responder)
    cands = _propose(cfg, tmp_path, _fixtures())
    assert cands and all(c.op_meta["stage"] == "stage2" for c in cands)
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_emitted"] == len(cands) < stats["pool_size"]


def test_emit_final_only_with_unsolved_scope_rejected():
    with pytest.raises(ValueError, match="final_rollout_scope"):
        _cfg("improve.lora_sft.staged.emit=final_only")


def test_regen_bridge_falls_back_to_stage1_bridges(monkeypatch, tmp_path):
    """A question whose regen pass produces no accepted bridge keeps its
    stage-1 bridge pairs instead of dropping out of the stage-2 fit."""
    cfg = _cfg("improve.lora_sft.staged.unsolved_targets=regen_bridge")

    def responder(qid, kind, r, stage):
        if stage == "bridge":                       # stage-1: both get bridges
            return [_sample(101)] if qid == "q1" else [_sample(104)]
        if stage == "stage1_bridge":                # regen FAILS for q2
            assert qid == "q2"
            return [_sample(102)]
        if kind == "a1" and stage == "stage1":
            return [_sample(101)] if qid == "q1" else [_sample(102)]
        return [_sample(104)]

    fits, _ = _setup(monkeypatch, responder)
    _propose(cfg, tmp_path, _fixtures())
    # q2 still fits on its stage-1 bridge pair
    assert [p["input_ids"] for p in fits["pairs"]["stage2"]] == [[1, 2, 104, 0]]
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stage1_regen_fallback"] == 1


def test_add_bridge_merges_old_and_new(monkeypatch, tmp_path):
    """add_bridge: union of stage-1 and regenerated bridges, deduped by
    input_ids, re-ranked shortest-first, capped at stage_max_keep."""
    cfg = _cfg("improve.lora_sft.staged.unsolved_targets=add_bridge",
               "improve.lora_sft.staged.stage_max_keep=2",
               "improve.lora_sft.staged.stage_bridge_n=3")

    def responder(qid, kind, r, stage):
        if stage == "bridge":
            assert r.n == 2                          # bridge.n untouched
            return [_sample(101)] if qid == "q1" else [_sample(104)]
        if stage == "stage1_bridge":
            assert qid == "q2"
            assert r.n == 3                          # stage_bridge_n seam
            # one exact duplicate of the stage-1 bridge + one longer new one
            return [_sample(104), _sample(104, ids=[104, 104])]
        if kind == "a1" and stage == "stage1":
            return [_sample(101)] if qid == "q1" else [_sample(102)]
        return [_sample(104)]

    fits, _ = _setup(monkeypatch, responder)
    op = StagedBridgeSftOperator()
    _propose(cfg, tmp_path, _fixtures(), op=op)
    q2_pairs = [p["input_ids"] for p in fits["pairs"]["stage2"]]
    assert q2_pairs == [[1, 2, 104, 0], [1, 2, 104, 104, 0]]   # dedup + shortest rank
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stage1_pairs_old"] == 1 and stats["stage1_pairs_new"] == 1
    # seams released after the regen call
    assert op._bridge_n is None and op._bridge_max_keep is None


def test_add_bridge_stage_max_keep_caps_merge(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.unsolved_targets=add_bridge",
               "improve.lora_sft.staged.stage_max_keep=1")

    def responder(qid, kind, r, stage):
        if stage == "bridge":
            return [_sample(101)] if qid == "q1" else [_sample(104)]
        if stage == "stage1_bridge":
            return [_sample(104, ids=[104, 104])]
        if kind == "a1" and stage == "stage1":
            return [_sample(101)] if qid == "q1" else [_sample(102)]
        return [_sample(104)]

    fits, _ = _setup(monkeypatch, responder)
    _propose(cfg, tmp_path, _fixtures())
    # shortest wins the single slot: the stage-1 bridge
    assert [p["input_ids"] for p in fits["pairs"]["stage2"]] == [[1, 2, 104, 0]]


def test_keep_selection_longest(monkeypatch, tmp_path):
    """bridge.keep_selection=longest flips the max_keep quota ordering (applies
    to the base bridge_sft operator too)."""
    cfg = Config.load(None, overrides=[
        "improve.operator=bridge_sft", "engine.enable_lora=true",
        "improve.lora_sft.bridge.n=2", "improve.lora_sft.bridge.max_keep=1",
        "improve.lora_sft.bridge.keep_selection=longest",
    ])

    def responder(qid, kind, r, stage):
        if kind in ("bridge", "bridge_retry"):
            if qid == "q1":
                return [_sample(101), _sample(105, ids=[105, 105, 105])]
            return [_sample(104)]
        return [_sample(102)]

    fits, _ = _setup(monkeypatch, responder)
    _propose(cfg, tmp_path, _fixtures(), op=BridgeSftOperator())
    q1 = [p for p in fits["pairs"]["pooled_c0"] if p["qid"] == "q1"]
    assert [p["input_ids"] for p in q1] == [[1, 2, 105, 105, 105, 0]]   # LONGEST kept


def _both_unsolved_responder(qid, kind, r, stage):
    """Bridges succeed for both questions; neither converts at stage 1, so both
    reach the stage-2 fit."""
    if kind in ("bridge", "bridge_retry"):
        return [_sample(101)] if qid == "q1" else [_sample(104)]
    if stage == "stage1":
        return [_sample(102)]
    return [_sample(101)] if qid == "q1" else [_sample(104)]


def test_stage2_chunking_one_adapter_per_question(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.stage2_chunk_size=1")
    fits, launches = _setup(monkeypatch, _both_unsolved_responder)
    _propose(cfg, tmp_path, _fixtures())

    # one stage-2 fit per question, each on its own pairs, both chained from stage1
    assert fits["order"] == ["stage1", "stage2_c0", "stage2_c1"]
    assert {p["qid"] for p in fits["pairs"]["stage2_c0"]} == {"q1"}
    assert {p["qid"] for p in fits["pairs"]["stage2_c1"]} == {"q2"}
    for name in ("stage2_c0", "stage2_c1"):
        assert str(fits["init"][name]).endswith("stage1/FAKEKEY")
    # the final rollout serves each question from ITS OWN shard adapter
    final = next(l for l in launches if l["stage"] == "stage2")
    lora_of = {r.rid.split(":")[0]: str(r.lora_path) for r in final["requests"]}
    assert lora_of["q1"].endswith("stage2_c0/FAKEKEY")
    assert lora_of["q2"].endswith("stage2_c1/FAKEKEY")
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stage2_chunks"] == [
        {"name": "stage2_c0", "n_questions": 1, "n_pairs": 1},
        {"name": "stage2_c1", "n_questions": 1, "n_pairs": 1},
    ]
    assert stats["n_fits"] == 3


def test_stage2_chunk_size_larger_than_set_stays_single_adapter(monkeypatch, tmp_path):
    """One chunk keeps the un-sharded fit name, so cache keys/logs are unchanged."""
    cfg = _cfg("improve.lora_sft.staged.stage2_chunk_size=25")
    fits, _ = _setup(monkeypatch, _both_unsolved_responder)
    _propose(cfg, tmp_path, _fixtures())
    assert fits["order"] == ["stage1", "stage2"]
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stage2_chunks"] == [{"name": "stage2", "n_questions": 2, "n_pairs": 2}]


def test_chunking_leaves_unfitted_question_on_its_previous_adapter(monkeypatch, tmp_path):
    """train_scope=unsolved_only + final_rollout_scope=all: a question solved at
    stage 1 is absent from the stage-2 fit, so it rolls out from stage-1."""
    cfg = _cfg("improve.lora_sft.staged.stage2_chunk_size=1",
               "improve.lora_sft.staged.final_rollout_scope=all")
    _, launches = _setup(monkeypatch, _default_responder)   # q1 solves at stage 1
    _propose(cfg, tmp_path, _fixtures())
    final = next(l for l in launches if l["stage"] == "stage2")
    lora_of = {r.rid.split(":")[0]: str(r.lora_path) for r in final["requests"]}
    assert lora_of["q1"].endswith("stage1/FAKEKEY")       # never re-fitted
    assert lora_of["q2"].endswith("stage2/FAKEKEY")       # sole stage-2 chunk


def test_chunking_chains_each_shard_from_its_own_previous_adapter(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.stage2_chunk_size=1",
               "improve.lora_sft.staged.num_stages=2")

    def responder(qid, kind, r, stage):
        if kind in ("bridge", "bridge_retry"):
            return [_sample(101)] if qid == "q1" else [_sample(104)]
        if stage in ("stage1", "stage2"):   # unsolved through the intermediate roll
            return [_sample(102)]
        return [_sample(101)] if qid == "q1" else [_sample(104)]

    fits, _ = _setup(monkeypatch, responder)
    _propose(cfg, tmp_path, _fixtures())
    assert fits["order"] == ["stage1", "stage2_c0", "stage2_c1", "stage3_c0", "stage3_c1"]
    # shard membership is stable, so stage3_cN chains from stage2_cN (not a neighbour)
    assert str(fits["init"]["stage3_c0"]).endswith("stage2_c0/FAKEKEY")
    assert str(fits["init"]["stage3_c1"]).endswith("stage2_c1/FAKEKEY")
    assert {p["qid"] for p in fits["pairs"]["stage3_c0"]} == {"q1"}


def test_chunking_without_chaining(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.stage2_chunk_size=1",
               "improve.lora_sft.staged.chain_adapter=false")
    fits, _ = _setup(monkeypatch, _both_unsolved_responder)
    _propose(cfg, tmp_path, _fixtures())
    assert fits["init"]["stage2_c0"] is None and fits["init"]["stage2_c1"] is None


def _dpo_responder(qid, kind, r, stage):
    """Bridges succeed for both; at stage 1 both questions FAIL with two
    distinct wrong rollouts (the on-policy negatives); both convert at the end."""
    if kind in ("bridge", "bridge_retry"):
        return [_sample(101)] if qid == "q1" else [_sample(104)]
    if stage == "stage1":
        return [_sample(102), _sample(102, ids=[102, 102, 102])]
    return [_sample(101)] if qid == "q1" else [_sample(104)]


def test_stage2_dpo_pairs_use_own_stage1_failures(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=dpo",
               "improve.lora_sft.staged.dpo.beta=0.2",
               "improve.lora_sft.staged.dpo.lr=5e-5",
               "improve.lora_sft.staged.dpo.negative_selection=shortest")
    fits, _ = _setup(monkeypatch, _dpo_responder)
    _propose(cfg, tmp_path, _fixtures())

    assert fits["order"] == ["stage1", "stage2"]
    # stage 1 is untouched SFT
    assert "input_ids" in fits["pairs"]["stage1"][0]
    assert fits["params"]["stage1"].get("objective", "sft") == "sft"
    # stage 2 carries DPO pairs: chosen = the bridge, rejected = the SHORTEST
    # stage-1 failure, same prompt_len; params carry the DPO knobs
    pairs = {p["qid"]: p for p in fits["pairs"]["stage2"]}
    assert set(pairs) == {"q1", "q2"}
    assert pairs["q1"]["chosen_ids"] == [1, 2, 101, 0]
    assert pairs["q1"]["rejected_ids"] == [1, 2, 102, 0]      # shortest failure
    assert pairs["q2"]["chosen_ids"] == [1, 2, 104, 0]
    assert pairs["q1"]["prompt_len"] == 2
    assert all("input_ids" not in p for p in fits["pairs"]["stage2"])
    p2 = fits["params"]["stage2"]
    assert p2["objective"] == "dpo" and p2["beta"] == 0.2 and p2["lr"] == 5e-5
    assert p2["reference"] == "init" and p2["sft_weight"] == 0.0
    assert str(fits["init"]["stage2"]).endswith("stage1/FAKEKEY")   # chained
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stage2_dpo_pairs"] == 2
    assert stats["stage2_dpo_no_negative"] == 0


def test_stage2_dpo_longest_and_random_negatives(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=dpo",
               "improve.lora_sft.staged.dpo.negative_selection=longest")
    fits, _ = _setup(monkeypatch, _dpo_responder)
    _propose(cfg, tmp_path, _fixtures())
    pairs = {p["qid"]: p for p in fits["pairs"]["stage2"]}
    assert pairs["q1"]["rejected_ids"] == [1, 2, 102, 102, 102, 0]

    cfg = _cfg("improve.lora_sft.staged.stage2_objective=dpo",
               "improve.lora_sft.staged.dpo.negative_selection=random")
    picks = []
    for _ in range(2):   # seeded -> identical across repeats
        fits, _ = _setup(monkeypatch, _dpo_responder)
        _propose(cfg, tmp_path / str(len(picks)), _fixtures())
        picks.append({p["qid"]: p["rejected_ids"] for p in fits["pairs"]["stage2"]})
    assert picks[0] == picks[1]


def test_stage2_dpo_drops_questions_without_negatives(monkeypatch, tmp_path):
    """A question whose stage-1 samples were all truncated has no graded
    failure in the pool -> no DPO pair, counted in stats."""
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=dpo")

    def responder(qid, kind, r, stage):
        if kind in ("bridge", "bridge_retry"):
            return [_sample(101)] if qid == "q1" else [_sample(104)]
        if stage == "stage1":
            if qid == "q2":   # truncated -> _collect emits nothing for q2
                return [{"text": "", "token_ids": [102], "finish_reason": "length"}]
            return [_sample(102)]
        return [_sample(101)] if qid == "q1" else [_sample(104)]

    fits, _ = _setup(monkeypatch, responder)
    _propose(cfg, tmp_path, _fixtures())
    assert {p["qid"] for p in fits["pairs"]["stage2"]} == {"q1"}
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stage2_dpo_no_negative"] == 1


def test_stage2_dpo_max_pairs_per_question(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=dpo",
               "improve.lora_sft.staged.dpo.max_pairs_per_question=1",
               "improve.lora_sft.bridge.max_keep=2")

    def responder(qid, kind, r, stage):
        if kind in ("bridge", "bridge_retry"):   # two bridges each -> two SFT pairs
            return ([_sample(101), _sample(105, ids=[105, 105, 105])] if qid == "q1"
                    else [_sample(104), _sample(104, ids=[104, 104])])
        if stage == "stage1":
            return [_sample(102)]
        return [_sample(101)] if qid == "q1" else [_sample(104)]

    fits, _ = _setup(monkeypatch, responder)
    _propose(cfg, tmp_path, _fixtures())
    assert len(fits["pairs"]["stage1"]) == 4                 # SFT: all bridges
    assert sorted(p["qid"] for p in fits["pairs"]["stage2"]) == ["q1", "q2"]   # DPO: capped


def test_stage2_dpo_composes_with_chunking(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=dpo",
               "improve.lora_sft.staged.stage2_chunk_size=1")
    fits, _ = _setup(monkeypatch, _dpo_responder)
    _propose(cfg, tmp_path, _fixtures())
    assert fits["order"] == ["stage1", "stage2_c0", "stage2_c1"]
    for name in ("stage2_c0", "stage2_c1"):
        assert fits["params"][name]["objective"] == "dpo"
        assert "chosen_ids" in fits["pairs"][name][0]


@pytest.mark.parametrize("bad,match", [
    ("improve.lora_sft.staged.stage2_objective=nope", "stage2_objective"),
    ("improve.lora_sft.staged.dpo.beta=0", "beta"),
    ("improve.lora_sft.staged.dpo.sft_weight=-1", "sft_weight"),
    ("improve.lora_sft.staged.dpo.negative_selection=nope", "negative_selection"),
    ("improve.lora_sft.staged.dpo.reference=nope", "reference"),
    ("improve.lora_sft.staged.dpo.max_pairs_per_question=0", "max_pairs_per_question"),
])
def test_dpo_config_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        _cfg(bad)


def test_stage1_chunking_shards_fit_and_rollout(monkeypatch, tmp_path):
    """stage1_chunk_size=1: one stage-1 adapter per question; the stage-1
    rollout serves each question from its own shard; stage-2 shards inherit
    the membership and chain from their OWN stage-1 shard."""
    cfg = _cfg("improve.lora_sft.staged.stage1_chunk_size=1",
               "improve.lora_sft.staged.stage2_chunk_size=1")
    fits, launches = _setup(monkeypatch, _both_unsolved_responder)
    _propose(cfg, tmp_path, _fixtures())

    assert fits["order"] == ["stage1_c0", "stage1_c1", "stage2_c0", "stage2_c1"]
    assert {p["qid"] for p in fits["pairs"]["stage1_c0"]} == {"q1"}
    assert {p["qid"] for p in fits["pairs"]["stage1_c1"]} == {"q2"}
    assert fits["init"]["stage1_c0"] is None
    assert str(fits["init"]["stage2_c0"]).endswith("stage1_c0/FAKEKEY")
    assert str(fits["init"]["stage2_c1"]).endswith("stage1_c1/FAKEKEY")
    roll1 = next(l for l in launches if l["stage"] == "stage1")
    lora_of = {r.rid.split(":")[0]: str(r.lora_path) for r in roll1["requests"]}
    assert lora_of["q1"].endswith("stage1_c0/FAKEKEY")
    assert lora_of["q2"].endswith("stage1_c1/FAKEKEY")
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert [c["name"] for c in stats["stage1_chunks"]] == ["stage1_c0", "stage1_c1"]
    assert stats["n_fits"] == 4


def test_stage1_chunking_bridge_skipped_question_gets_a_shard(monkeypatch, tmp_path):
    """A question with no accepted bridge has no fit pairs but is still sampled
    (bridge.sample_skipped): it is attached to a shard adapter round-robin."""
    cfg = _cfg("improve.lora_sft.staged.stage1_chunk_size=1",
               "improve.lora_sft.staged.stage2_chunk_size=1")

    def responder(qid, kind, r, stage):
        if kind in ("bridge", "bridge_retry"):
            return [_sample(101)] if qid == "q1" else [_sample(102)]   # q2 never bridges
        if stage == "stage1":
            return [_sample(102)]
        return [_sample(101)] if qid == "q1" else [_sample(104)]

    fits, launches = _setup(monkeypatch, responder)
    _propose(cfg, tmp_path, _fixtures())
    assert fits["order"][:1] == ["stage1"]          # one shard -> un-sharded name
    roll1 = next(l for l in launches if l["stage"] == "stage1")
    assert {r.rid.split(":")[0] for r in roll1["requests"]} == {"q1", "q2"}
    assert all(str(r.lora_path).endswith("stage1/FAKEKEY") for r in roll1["requests"])


def test_stage1_chunking_without_chaining_allows_pooled_stage2(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.stage1_chunk_size=1",
               "improve.lora_sft.staged.chain_adapter=false")
    fits, _ = _setup(monkeypatch, _both_unsolved_responder)
    _propose(cfg, tmp_path, _fixtures())
    assert fits["order"] == ["stage1_c0", "stage1_c1", "stage2"]
    assert fits["init"]["stage2"] is None


@pytest.mark.parametrize("bad,match", [
    ("improve.lora_sft.staged.stage1_chunk_size=-1", "stage1_chunk_size"),
    # stage-1 sharded + chained pooled stage-2: no well-defined warm start
    ("improve.lora_sft.staged.stage1_chunk_size=1", "chain_adapter"),
])
def test_stage1_chunking_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        _cfg(bad)


def test_stage2_chunk_cannot_exceed_stage1_chunk():
    with pytest.raises(ValueError, match="cannot exceed"):
        _cfg("improve.lora_sft.staged.stage1_chunk_size=1",
             "improve.lora_sft.staged.stage2_chunk_size=2")


def test_hierarchical_chunking_splits_each_stage1_shard(monkeypatch, tmp_path):
    """stage1_chunk_size=2 (both questions in ONE stage-1 shard), stage2_chunk_size=1:
    stage 2 sub-splits that shard into two adapters, each warm-started from the
    parent stage-1 adapter."""
    cfg = _cfg("improve.lora_sft.staged.stage1_chunk_size=2",
               "improve.lora_sft.staged.stage2_chunk_size=1")
    fits, launches = _setup(monkeypatch, _both_unsolved_responder)
    _propose(cfg, tmp_path, _fixtures())
    assert fits["order"] == ["stage1", "stage2_c0_0", "stage2_c0_1"]   # 1 parent -> 2 children
    assert {p["qid"] for p in fits["pairs"]["stage2_c0_0"]} == {"q1"}
    assert {p["qid"] for p in fits["pairs"]["stage2_c0_1"]} == {"q2"}
    for name in ("stage2_c0_0", "stage2_c0_1"):
        assert str(fits["init"][name]).endswith("stage1/FAKEKEY")
    final = next(l for l in launches if l["stage"] == "stage2")
    lora_of = {r.rid.split(":")[0]: str(r.lora_path) for r in final["requests"]}
    assert lora_of["q1"].endswith("stage2_c0_0/FAKEKEY")
    assert lora_of["q2"].endswith("stage2_c0_1/FAKEKEY")


def test_assign_chunks_helper():
    from expert_iter.staged_bridge_sft import _assign_chunks

    key: dict = {}
    # first split == a slice of the sorted fit set; keys record the shard path
    assert _assign_chunks(["c", "a", "d", "b", "e"], 2, key) == [
        ("c0", ["a", "b"]), ("c1", ["c", "d"]), ("c2", ["e"])]
    assert key == {"a": (0,), "b": (0,), "c": (1,), "d": (1,), "e": (2,)}
    # same size later: solved questions drop out, survivors keep their shard
    assert _assign_chunks(["b", "c", "e"], 2, key) == [("c0", ["b"]), ("c1", ["c"]), ("c2", ["e"])]
    # a newcomer joins the smallest parent group
    _assign_chunks(["b", "c", "e", "f"], 2, key)
    assert key["f"] in ((0,), (1,), (2,))
    # smaller size: each parent shard is sub-split, children get (parent, j) keys
    key = {"a": (0,), "b": (0,), "c": (0,), "d": (1,)}
    assert _assign_chunks(["a", "b", "c", "d"], 2, key) == [
        ("c0_0", ["a", "b"]), ("c0_1", ["c"]), ("c1", ["d"])]
    assert key["a"] == (0, 0) and key["c"] == (0, 1) and key["d"] == (1,)
    # size <= 0 -> one pooled shard; keys reset to () (nothing to chain from)
    assert _assign_chunks(["b", "a"], 0, key) == [("c0", ["a", "b"])]
    assert key["a"] == () and key["b"] == ()


@pytest.mark.parametrize("bad,match", [
    ("improve.lora_sft.staged.stage2_chunk_size=-1", "stage2_chunk_size"),
    ("improve.lora_sft.chunk_size=25", "stage2_chunk_size"),
    ("improve.lora_sft.bridge.keep_selection=nope", "keep_selection"),
    ("improve.lora_sft.staged.stage_bridge_n=0", "stage_bridge_n"),
    ("improve.lora_sft.staged.stage_max_keep=0", "stage_max_keep"),
])
def test_new_knob_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        _cfg(bad)


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


def test_stage2_ul_pairs_and_params(monkeypatch, tmp_path):
    """objective=ul reuses the dpo pair path (chosen = bridge, rejected = own
    stage-1 failure) and forwards the StagedUlCfg knobs to lora_fit."""
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=ul",
               "improve.lora_sft.staged.ul.mu=0.3",
               "improve.lora_sft.staged.ul.delta=0.05",
               "improve.lora_sft.staged.ul.guard=false",
               "improve.lora_sft.staged.ul.lr=7e-5",
               "improve.lora_sft.staged.ul.negative_selection=shortest")
    fits, _ = _setup(monkeypatch, _dpo_responder)
    _propose(cfg, tmp_path, _fixtures())

    assert fits["order"] == ["stage1", "stage2"]
    assert fits["params"]["stage1"].get("objective", "sft") == "sft"
    pairs = {p["qid"]: p for p in fits["pairs"]["stage2"]}
    assert set(pairs) == {"q1", "q2"}
    assert pairs["q1"]["chosen_ids"] == [1, 2, 101, 0]
    assert pairs["q1"]["rejected_ids"] == [1, 2, 102, 0]      # shortest failure
    assert all("input_ids" not in p for p in fits["pairs"]["stage2"])
    p2 = fits["params"]["stage2"]
    assert p2["objective"] == "ul" and p2["mu"] == 0.3 and p2["delta"] == 0.05
    assert p2["guard"] is False and p2["lr"] == 7e-5 and p2["reference"] == "init"
    assert "beta" not in p2 and "sft_weight" not in p2
    assert str(fits["init"]["stage2"]).endswith("stage1/FAKEKEY")   # chained
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stage2_dpo_pairs"] == 2


def _modal_responder(qid, kind, r, stage):
    """q1's stage-1 failures box 5, 5, 8 -> the modal wrong answer is 5; the
    boxed-8 sample (103) is longest AND first by attempt order, so both the
    longest rule and attempt-order would pick it — only `modal` picks a 5."""
    if kind in ("bridge", "bridge_retry"):
        return [_sample(101)] if qid == "q1" else [_sample(104)]
    if stage == "stage1":
        if qid == "q1":
            return [_sample(103, ids=[103, 103, 103, 103]),
                    _sample(102), _sample(102, ids=[102, 102])]
        return [_sample(106)]      # q2 fails without any boxed answer
    return [_sample(101)] if qid == "q1" else [_sample(104)]


def test_stage2_ul_modal_negative_selection(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=ul")   # default: modal
    FakeTok.vocab[103] = "also wrong: \\boxed{8}"
    FakeTok.vocab[106] = "gave up, no box"
    try:
        fits, _ = _setup(monkeypatch, _modal_responder)
        _propose(cfg, tmp_path, _fixtures())
    finally:
        FakeTok.vocab.pop(103), FakeTok.vocab.pop(106)
    pairs = {p["qid"]: p for p in fits["pairs"]["stage2"]}
    # q1: the modal wrong answer is \boxed{5} (2 of 3), first such sample = 102
    assert pairs["q1"]["rejected_ids"] == [1, 2, 102, 0]
    # q2: no boxed failure -> random fallback still yields a pair, counted
    assert pairs["q2"]["rejected_ids"] == [1, 2, 106, 0]
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stage2_neg_modal_fallback"] == 1


def test_stage2_dpo_modal_negative_selection(monkeypatch, tmp_path):
    """`modal` is also a valid dpo selection (a DPO+modal arm needs no code)."""
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=dpo",
               "improve.lora_sft.staged.dpo.negative_selection=modal")
    FakeTok.vocab[103] = "also wrong: \\boxed{8}"
    FakeTok.vocab[106] = "gave up, no box"
    try:
        fits, _ = _setup(monkeypatch, _modal_responder)
        _propose(cfg, tmp_path, _fixtures())
    finally:
        FakeTok.vocab.pop(103), FakeTok.vocab.pop(106)
    pairs = {p["qid"]: p for p in fits["pairs"]["stage2"]}
    assert pairs["q1"]["rejected_ids"] == [1, 2, 102, 0]
    assert fits["params"]["stage2"]["objective"] == "dpo"


# --- ul span localization --------------------------------------------------
# The unlikelihood's normalizer is global, so WHICH rejected tokens carry it is
# the whole methodology: uniform spreads mu over ~4.7k tokens of confident math
# prose (measured inert at 2 steps, and harmful at full budget — L3 S4-v1),
# boxed charges it to the answer the trajectory committed to.


def test_ul_span_boxed_stamps_last_boxed_token(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=ul",
               "improve.lora_sft.staged.ul.span=boxed",
               "improve.lora_sft.staged.ul.negative_selection=longest")
    fits, _ = _setup(monkeypatch, _dpo_responder)
    _propose(cfg, tmp_path, _fixtures())

    pairs = {p["qid"]: p for p in fits["pairs"]["stage2"]}
    # rejected = [1, 2, 102, 102, 102, 0]: prompt_len 2, three boxed tokens.
    # The span must start at the LAST one (index 4), not the first.
    assert pairs["q1"]["rejected_ids"] == [1, 2, 102, 102, 102, 0]
    assert pairs["q1"]["rejected_span_start"] == 4
    assert fits["params"]["stage2"]["span"] == "boxed"     # in the fit cache key
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stage2_span_no_boxed"] == 0
    # 2 penalized tokens per pair (the boxed token + EOS) instead of the 4
    # response tokens uniform would charge
    assert stats["stage2_span_tokens_mean"] == 2.0


def test_ul_span_pad_widens_the_span(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=ul",
               "improve.lora_sft.staged.ul.span=boxed",
               "improve.lora_sft.staged.ul.span_pad=2",
               "improve.lora_sft.staged.ul.negative_selection=longest")
    fits, _ = _setup(monkeypatch, _dpo_responder)
    _propose(cfg, tmp_path, _fixtures())
    pairs = {p["qid"]: p for p in fits["pairs"]["stage2"]}
    assert pairs["q1"]["rejected_span_start"] == 2          # 4 - 2, clamped at plen


def test_ul_span_boxed_drops_negatives_with_no_answer(monkeypatch, tmp_path):
    """A negative with no \\boxed has no attractor to aim at. It must be dropped,
    not silently widened back to the full response: one 4.7k-token fallback pair
    would dominate a normalizer the other pairs contribute ~10 tokens to."""
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=ul",
               "improve.lora_sft.staged.ul.span=boxed",
               "improve.lora_sft.staged.ul.negative_selection=modal")
    FakeTok.vocab[103] = "also wrong: \\boxed{8}"
    FakeTok.vocab[106] = "gave up, no box"
    try:
        fits, _ = _setup(monkeypatch, _modal_responder)
        _propose(cfg, tmp_path, _fixtures())
    finally:
        FakeTok.vocab.pop(103), FakeTok.vocab.pop(106)

    pairs = {p["qid"]: p for p in fits["pairs"]["stage2"]}
    assert set(pairs) == {"q1"}                             # q2's negative had no box
    assert "rejected_span_start" in pairs["q1"]
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stage2_span_no_boxed"] == 1


def test_ul_span_uniform_is_unchanged(monkeypatch, tmp_path):
    """Default stays byte-identical to the pre-span behaviour: no span field, so
    lora_fit falls back to prompt_len and charges every response token."""
    cfg = _cfg("improve.lora_sft.staged.stage2_objective=ul",
               "improve.lora_sft.staged.ul.negative_selection=longest")
    fits, _ = _setup(monkeypatch, _dpo_responder)
    _propose(cfg, tmp_path, _fixtures())
    assert all("rejected_span_start" not in p for p in fits["pairs"]["stage2"])
    assert fits["params"]["stage2"]["span"] == "uniform"
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert "stage2_span_no_boxed" not in stats


class _CharTok:
    """decode(ids) = the ids as characters — lets the suffix search be checked
    against an exact, hand-countable string."""

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids)


def _ids(text):
    return [ord(c) for c in text]


@pytest.mark.parametrize("text", [
    "2+2 is \\boxed{4}",
    "a \\boxed{1} then \\boxed{2}",          # the LAST one wins
    "\\boxed{9}",                            # at position 0
    "\\boxed{1} and a tail long enough to force the doubling probe past 8",
    "no answer at all",
    "",
])
def test_boxed_span_start_finds_the_last_boxed(text):
    """Checked against an independent oracle: with a 1-char-per-token decoder the
    token index IS the character index, so str.rfind is the ground truth."""
    from expert_iter.templates import boxed_span_start
    want = text.rfind("\\boxed")
    got = boxed_span_start(_CharTok(), _ids(text))
    assert got == (None if want < 0 else want)


def test_boxed_span_start_pad_is_clamped_at_zero():
    from expert_iter.templates import boxed_span_start
    ids = _ids("ab \\boxed{4}")
    assert boxed_span_start(_CharTok(), ids) == 3
    assert boxed_span_start(_CharTok(), ids, pad=2) == 1
    assert boxed_span_start(_CharTok(), ids, pad=99) == 0
