"""gold_text operator (L5 gold-SFT baseline): y* becomes the training text,
and it is the one operator that deliberately trips the learnability gate.
CPU-only — the tokenizer is faked, nothing is downloaded."""

import pytest

from expert_iter import improve as mod
from expert_iter.config import Config
from expert_iter.filters import NoExternalContextGate
from expert_iter.improve import GoldTextOperator
from expert_iter.records import AnchorRecord, UnsolvedQuestion

EOS = 99


class _FakeTokenizer:
    """One id per character — enough to assert splicing, not text handling."""

    eos_token_id = EOS

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


@pytest.fixture
def patched_tokenizer(monkeypatch):
    class _Auto:
        @staticmethod
        def from_pretrained(*a, **k):
            return _FakeTokenizer()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _Auto)
    return _Auto


def _propose(gold, *, anchor_ids=(), qids=("q1", "q2")):
    questions = [UnsolvedQuestion(qid=q, question=f"question {q}", final_answer="1",
                                 failed_sample_idxs=[0], iter=0) for q in qids]
    anchors = [AnchorRecord(qid=q, base_sample_idx=3, policy="none",
                            anchor_token_ids=list(anchor_ids), anchor_text="",
                            anchor_len=len(anchor_ids), base_response_len=10)
               for q in qids]
    return GoldTextOperator().propose(
        questions, anchors, {q: [1, 2] for q in qids}, Config.load(None),
        model_paths={"policy": "fake"}, work_dir="/tmp", iteration=0,
        gold_solutions=gold,
    )


def test_emits_one_candidate_per_question_with_gold(patched_tokenizer):
    out = _propose({"q1": "AB", "q2": "C"})
    assert [c.qid for c in out] == ["q1", "q2"]
    assert all(c.attempt_idx == 0 for c in out)
    assert all(c.operator == "gold_text" for c in out)


def test_questions_without_gold_are_skipped(patched_tokenizer):
    out = _propose({"q1": "AB", "q2": "   "})
    assert [c.qid for c in out] == ["q1"]


def test_continuation_is_gold_ids_plus_eos(patched_tokenizer):
    (cand,) = _propose({"q1": "AB"}, qids=("q1",))
    assert cand.continuation_token_ids == [ord("A"), ord("B"), EOS]
    assert cand.prompt_token_ids == [1, 2]
    assert cand.continuation_text == "AB"


def test_anchor_is_dropped_even_when_one_exists(patched_tokenizer):
    """y* is a standalone solution, never a continuation of a failed prefix."""
    (cand,) = _propose({"q1": "AB"}, anchor_ids=(7, 8), qids=("q1",))
    assert cand.anchor_token_ids == []


def test_external_context_carries_gold_and_trips_the_gate(patched_tokenizer):
    """The learnability contract: this operator conditions on y*, declares it,
    and no_external_context therefore rejects it — which is why the gold-SFT
    arm (configs/methods/l5_gold_sft.yaml) drops that gate on purpose."""
    (cand,) = _propose({"q1": "AB"}, qids=("q1",))
    assert cand.external_context == "AB"
    ok, reason = NoExternalContextGate().check(cand, None)
    assert not ok and reason == "external_context"


def test_operator_is_registered_and_config_accepts_it():
    from expert_iter.registry import OPERATORS

    assert "gold_text" in OPERATORS
    cfg = Config.load(None, ["improve.operator=gold_text"])
    cfg.validate()
    assert cfg.improve.operator == "gold_text"
