"""cliff_stats: cliff (= unsolved) counting and improvement-operator conversion."""

from types import SimpleNamespace

from expert_iter.filters import FilterContext, cliff_stats
from expert_iter.records import ImprovedCandidate, UnsolvedQuestion


def _cand(qid, correct, attempt=0):
    return ImprovedCandidate(
        qid=qid, base_sample_idx=0, attempt_idx=attempt,
        prompt_token_ids=[1], anchor_token_ids=[2], continuation_token_ids=[3],
        continuation_text="", correct=correct,
    )


def _unsolved(*qids):
    return {q: UnsolvedQuestion(qid=q, question="?", final_answer="42") for q in qids}


def test_cliff_counts_rate_and_histogram():
    unsolved = _unsolved("q1", "q2", "q3")
    cands = [_cand("q1", True), _cand("q1", True, 1), _cand("q1", False, 2),
             _cand("q2", False), _cand("q3", True)]
    stats = cliff_stats(cands, unsolved, ctx=None, n_total_questions=10)
    assert stats["cliff/count"] == 3
    assert stats["cliff/ratio"] == 0.3
    assert stats["cliff/conversion_rate"] == round(2 / 3, 4)
    assert stats["cliff/conversion_histogram"] == [0, 1, 2]


def test_cliff_empty_unsolved():
    stats = cliff_stats([], {}, ctx=None, n_total_questions=None)
    assert stats["cliff/count"] == 0
    assert stats["cliff/conversion_rate"] == 0.0
    assert stats["cliff/conversion_histogram"] == []
    assert "cliff/ratio" not in stats


def test_cliff_verifies_unfilled_candidates():
    """Candidates the gate chain never reached (correct=None) are graded here so
    conversion is independent of gate order."""
    unsolved = _unsolved("q1")
    ctx = FilterContext(
        cfg=None,
        verifier=SimpleNamespace(verify=lambda q, text: SimpleNamespace(correct="42" in text)),
        tokenizer=SimpleNamespace(decode=lambda ids: ""),
        questions=unsolved,
    )
    good, bad = _cand("q1", None), _cand("q1", None, 1)
    good.continuation_text = "the answer is 42"
    bad.continuation_text = "no idea"
    stats = cliff_stats([good, bad], unsolved, ctx=ctx, n_total_questions=1)
    assert stats["cliff/conversion_rate"] == 1.0
    assert stats["cliff/conversion_histogram"] == [1]
    assert good.correct is True and bad.correct is False
