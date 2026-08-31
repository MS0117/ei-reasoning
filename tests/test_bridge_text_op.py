"""bridge_text operator (L5 arm: STaR-style rationalization on cliffs) — the
accepted bridges ARE the candidates; no fit, no resample. Monkeypatched pool,
no GPU, no downloads. Fixtures mirror tests/test_bridge_sft_op.py."""

import json

import pytest

from expert_iter import lora_sft as ls_mod
from expert_iter.bridge_sft import BridgeTextOperator
from expert_iter.config import Config
from expert_iter.engine import GenResult
from expert_iter.records import AnchorRecord, UnsolvedQuestion
from expert_iter.utils import read_jsonl


class FakeTok:
    eos_token_id = 0
    PROMPT_IDS = [9, 9, 9]
    vocab = {
        101: "The answer is \\boxed{4}",
        102: "wrong: \\boxed{5}",
        103: "By the reference solution, \\boxed{4}",   # correct but leaky
        104: "recovered: \\boxed{7}",
    }

    def apply_chat_template(self, messages, **kw):
        return "BRIDGE:" + messages[-1]["content"][:8]

    def __call__(self, text, **kw):
        return {"input_ids": list(self.PROMPT_IDS)}

    def decode(self, ids):
        return " ".join(self.vocab.get(i, "") for i in ids)


def _sample(*ids):
    """A generated sample whose ids repeat a vocab entry — the verifier grades
    the LAST \\boxed, so repeats stay correct/wrong as their id says."""
    return {"text": " ".join(FakeTok.vocab[i] for i in ids),
            "token_ids": list(ids), "finish_reason": "stop"}


def _fixtures(anchored=False):
    questions = [
        UnsolvedQuestion(qid="q1", question="2+2?", final_answer="4"),
        UnsolvedQuestion(qid="q2", question="3+4?", final_answer="7"),
    ]
    anchor_ids = {"q1": [7, 8] if anchored else [], "q2": []}
    anchors = [
        AnchorRecord(qid=q.qid, base_sample_idx=3, policy="none",
                     anchor_token_ids=anchor_ids[q.qid], anchor_text="",
                     anchor_len=len(anchor_ids[q.qid]), base_response_len=10)
        for q in questions
    ]
    prompts = {"q1": [1, 2], "q2": [1, 2]}
    gold = {"q1": "It is 4.", "q2": "It is 7."}
    return questions, anchors, prompts, gold


def _cfg(*extra):
    return Config.load(None, overrides=[
        "improve.operator=bridge_text", "improve.n=2",
        "improve.lora_sft.bridge.n=2", *extra,
    ])


def _setup(monkeypatch, responder):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: FakeTok())

    def never_fit(*a, **k):
        raise AssertionError("bridge_text must never fit an adapter")

    monkeypatch.setattr(ls_mod, "_fit_adapter", never_fit)
    calls = {"n": 0, "requests": []}

    def fake_run_pool(requests, **kw):
        calls["n"] += 1
        calls["requests"].append(list(requests))
        out = []
        for r in requests:
            qid, kind = r.rid.split(":", 1)
            out.append(GenResult(rid=r.rid, samples=responder(qid, kind, r)))
        return out

    monkeypatch.setattr(ls_mod, "run_pool", fake_run_pool)
    return calls


def _propose(cfg, tmp_path, fixtures):
    questions, anchors, prompts, gold = fixtures
    return BridgeTextOperator().propose(
        questions, anchors, prompts, cfg,
        model_paths={"policy": "org/policy"},
        work_dir=tmp_path / "pool", iteration=0, gold_solutions=gold)


def test_candidates_are_the_kept_bridges_verbatim(monkeypatch, tmp_path):
    cfg = _cfg()
    fixtures = _fixtures()

    def responder(qid, kind, r):
        assert kind == "bridge"                 # the ONLY pass that runs
        return [_sample(101), _sample(102)] if qid == "q1" else [_sample(104), _sample(102)]

    calls = _setup(monkeypatch, responder)
    cands = _propose(cfg, tmp_path, fixtures)

    assert calls["n"] == 1                      # one bridge launch, no resample
    assert not (tmp_path / "adapters").exists()
    assert sorted(c.qid for c in cands) == ["q1", "q2"]
    c = next(c for c in cands if c.qid == "q1")
    assert c.continuation_token_ids == [101]    # ids straight from generation
    assert c.continuation_text == FakeTok.vocab[101]
    assert c.prompt_token_ids == [1, 2]         # PLAIN prompt, not the privileged one
    assert c.anchor_token_ids == []
    assert c.correct is True
    assert c.external_context == "It is 4."    # learnability contract: y* recorded
    assert c.operator == "bridge_text"
    assert c.op_meta["channel"] == "training_text"
    assert c.op_meta["via"] == "bridge_trajectories"
    assert c.op_meta["pass"] == "bridge"
    assert c.base_sample_idx == 3 and c.attempt_idx == 0
    # bridge bookkeeping is the shared one
    rows = list(read_jsonl(tmp_path / "bridge" / "bridges.jsonl"))
    assert sum(r["kept"] for r in rows) == 2
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["operator"] == "bridge_text"
    assert stats["n_bridge_generated"] == 4 and stats["n_bridge_correct"] == 2
    assert stats["n_questions_bridged"] == 2 and stats["n_candidates"] == 2
    assert stats["n_bridge_leak_phrasing_reported"] == 0


def test_max_keep_shortest_is_the_pre_filter_cap(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.bridge.max_keep=2", "improve.lora_sft.bridge.n=3")
    fixtures = _fixtures()

    def responder(qid, kind, r):
        if qid == "q1":
            return [_sample(101, 101, 101), _sample(101), _sample(101, 101)]
        return [_sample(102)] * 3 if kind == "bridge" else [_sample(102)]

    _setup(monkeypatch, responder)
    cands = [c for c in _propose(cfg, tmp_path, fixtures) if c.qid == "q1"]
    assert sorted(len(c.continuation_token_ids) for c in cands) == [1, 2]   # two shortest
    assert {c.attempt_idx for c in cands} == {1, 2}   # original sample_idx kept


def test_retry_then_skip_emits_nothing_for_the_empty_cliff(monkeypatch, tmp_path):
    cfg = _cfg()
    fixtures = _fixtures()

    def responder(qid, kind, r):
        if kind == "bridge":
            return [_sample(101)] if qid == "q1" else [_sample(102)]
        assert kind == "bridge_retry" and qid == "q2"
        return [_sample(102)]                   # still wrong -> skip list

    calls = _setup(monkeypatch, responder)
    cands = _propose(cfg, tmp_path, fixtures)
    assert calls["n"] == 2                      # bridge + retry, nothing else
    assert [c.qid for c in cands] == ["q1"]
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_bridge_retried"] == 1
    assert stats["bridge_skipped_qids"] == ["q2"]
    assert stats["bridge_yield"] == 0.5 and stats["n_candidates"] == 1


def test_leaky_bridge_is_reported_and_still_emitted(monkeypatch, tmp_path):
    """The arm's definition: correctness screened, leakage counted but never
    filtered (the STaR original does not screen)."""
    cfg = _cfg()
    fixtures = _fixtures()

    def responder(qid, kind, r):
        return [_sample(103)] if qid == "q1" else [_sample(104)]

    _setup(monkeypatch, responder)
    cands = _propose(cfg, tmp_path, fixtures)
    leaky = next(c for c in cands if c.qid == "q1")
    assert "reference solution" in leaky.continuation_text
    assert leaky.correct is True
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_bridge_leak_phrasing_reported"] == 1
    assert stats["n_bridge_leak_rule"] == 0     # the DROPPING screen stayed off
    assert stats["n_candidates"] == 2


def test_anchored_bridge_carries_the_anchor(monkeypatch, tmp_path):
    cfg = _cfg()
    fixtures = _fixtures(anchored=True)

    def responder(qid, kind, r):
        if qid == "q1":
            # generation prompt = privileged ids + anchor; candidate keeps the anchor
            assert r.prompt_token_ids == FakeTok.PROMPT_IDS + [7, 8]
            return [_sample(101)]
        return [_sample(104)]

    _setup(monkeypatch, responder)
    c = next(c for c in _propose(cfg, tmp_path, fixtures) if c.qid == "q1")
    assert c.prompt_token_ids == [1, 2] and c.anchor_token_ids == [7, 8]
    assert c.continuation_token_ids == [101]


def test_no_gold_is_a_hard_error_and_no_eligible_is_empty(monkeypatch, tmp_path):
    cfg = _cfg()
    questions, anchors, prompts, _ = _fixtures()
    _setup(monkeypatch, lambda *a: [])
    with pytest.raises(RuntimeError, match="gold_solution"):
        BridgeTextOperator().propose(questions, anchors, prompts, cfg,
                                     model_paths={"policy": "org/policy"},
                                     work_dir=tmp_path / "pool", iteration=0,
                                     gold_solutions={})
    assert BridgeTextOperator().propose([], [], {}, cfg,
                                        model_paths={"policy": "org/policy"},
                                        work_dir=tmp_path / "pool2", iteration=0,
                                        gold_solutions={}) == []


def test_validation_rejects_lora_only_phases():
    for bad in ("improve.lora_sft.project_back.enabled=true",
                "improve.lora_sft.fit.adaptive.enabled=true",
                "improve.lora_sft.refit_budget=1",
                "improve.rl.enabled=true"):
        with pytest.raises(ValueError, match="bridge_text"):
            Config.load(None, overrides=["improve.operator=bridge_text", bad])
    # and no adapter is needed: enable_lora may stay off
    assert _cfg().engine.enable_lora is False


def test_preset_pins_the_arm_definition():
    cfg = Config.load("configs/methods/l5_bridge_inloop.yaml")
    cfg.validate()
    assert cfg.improve.operator == "bridge_text"
    assert cfg.engine.enable_lora is False
    assert cfg.train.sft.cliff.enabled is False          # today's loss
    assert "no_external_context" not in cfg.filter.gates  # y* rides in external_context
    assert "leakage_rules" not in cfg.filter.gates        # STaR original: no screen
    assert cfg.improve.lora_sft.bridge.leakage_rules is False
    assert cfg.filter.max_per_question == 2 and cfg.partition.solved_keep_max == 4
    # the bridge block matches the main arm so iteration-0 requests are identical
    main = Config.load("configs/methods/l5_staged_dpo_s3.yaml")
    assert cfg.improve.lora_sft.bridge == main.improve.lora_sft.bridge
    assert cfg.improve.lora_sft.fit.max_pair_tokens == main.improve.lora_sft.fit.max_pair_tokens
    for k in ("temperature", "top_p", "max_tokens"):
        assert getattr(cfg.improve, k) == getattr(main.improve, k)
