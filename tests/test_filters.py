"""cliff_stats + the C(y) candidate-selection pass."""

from types import SimpleNamespace

import pytest

from expert_iter import engine as engine_mod
from expert_iter.config import Config
from expert_iter.engine import GenResult
from expert_iter.filters import (
    FilterContext,
    _rank_by_c_score,
    _score_candidates,
    _write_candidate_scores,
    cliff_stats,
)
from expert_iter.records import ImprovedCandidate, UnsolvedQuestion
from expert_iter.utils import read_jsonl


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
    decoded = []

    def decode(ids):
        decoded.append(ids)
        return "the answer is 42" if 42 in ids else "no idea"

    ctx = FilterContext(
        cfg=None,
        verifier=SimpleNamespace(verify=lambda q, text: SimpleNamespace(correct="42" in text)),
        tokenizer=SimpleNamespace(decode=decode),
        questions=unsolved,
    )
    good, bad = _cand("q1", None), _cand("q1", None, 1)
    good.continuation_token_ids = [42]
    bad.continuation_token_ids = [0]
    stats = cliff_stats([good, bad], unsolved, ctx=ctx, n_total_questions=1)
    assert stats["cliff/conversion_rate"] == 1.0
    assert stats["cliff/conversion_histogram"] == [1]
    assert good.correct is True and bad.correct is False
    assert decoded == [[2, 42], [2, 0]]


# ---------------------------------------------------------------------------
# C(y) selection (filter.selection.method=c_score)
# ---------------------------------------------------------------------------

def _score_cand(i, op_meta=None):
    return ImprovedCandidate(
        qid="q", base_sample_idx=0, attempt_idx=i, prompt_token_ids=[1],
        anchor_token_ids=[], continuation_token_ids=[10 + i],
        continuation_text="t", correct=True, op_meta=op_meta or {},
    )


def test_c_score_ranks_by_mean_plus_tail(monkeypatch, tmp_path):
    """The CVaR term punishes a single bottleneck token: cand2 has a LOWER mean
    NLL than cand1 but a worse tail, so cand1 wins the second slot."""
    cfg = Config.load(None, overrides=["filter.selection.method=c_score"])
    cands = [_score_cand(0), _score_cand(1), _score_cand(2)]
    lps = {
        "q:0:0:s": [-0.1] * 10,           # C = 0.1 + 0.1  = 0.2
        "q:0:1:s": [-1.0] * 10,           # C = 1.0 + 1.0  = 2.0
        "q:0:2:s": [-0.1] * 9 + [-3.0],   # C = 0.39 + 3.0 = 3.39
    }
    monkeypatch.setattr(
        engine_mod, "run_pool",
        lambda requests, **kw: [GenResult(rid=r.rid, token_logprobs=lps[r.rid])
                                for r in requests],
    )
    scores, report = _score_candidates(cands, cfg, "org/policy", tmp_path)
    kept, n_over = _rank_by_c_score(cands, cfg, scores)
    _write_candidate_scores(scores, kept, tmp_path)

    assert sorted(c.attempt_idx for c in kept) == [0, 1]
    assert n_over == 1
    assert report["n_scored"] == 3 and report["s_tail"]["max"] == 3.0
    assert report["c"]["mean"] == pytest.approx((0.2 + 2.0 + 3.39) / 3, abs=1e-3)
    persisted = {s["key"]: s for s in read_jsonl(tmp_path / "filtered" / "candidate_scores.jsonl")}
    assert persisted["q:0:2"]["kept"] is False
    assert persisted["q:0:0"]["c"] == pytest.approx(0.2)
    assert persisted["q:0:2"]["s_mean"] == pytest.approx(0.39)
    assert all(not k.startswith("_") for row in persisted.values() for k in row)


def test_random_selection_is_deterministic_and_capped():
    from expert_iter.filters import _random_selection

    cfg = Config.load(None, overrides=[
        "filter.selection.method=random", "filter.max_per_question=1",
    ])
    cands = [_score_cand(i) for i in range(5)]
    kept1, n_over1 = _random_selection(cands, cfg)
    kept2, n_over2 = _random_selection(list(reversed(cands)), cfg)  # order-independent
    assert len(kept1) == 1 and n_over1 == 4
    assert [c.attempt_idx for c in kept1] == [c.attempt_idx for c in kept2]
    # different seed -> (very likely) different draw is allowed; same seed must match
    kept3, _ = _random_selection(cands, cfg)
    assert [c.attempt_idx for c in kept3] == [c.attempt_idx for c in kept1]


def test_random_selection_keeps_all_under_cap():
    from expert_iter.filters import _random_selection

    cfg = Config.load(None, overrides=[
        "filter.selection.method=random", "filter.max_per_question=8",
    ])
    cands = [_score_cand(i) for i in range(3)]
    kept, n_over = _random_selection(cands, cfg)
    assert len(kept) == 3 and n_over == 0


def test_c_score_d_tail_pass_and_missing_ref(monkeypatch, tmp_path):
    cfg = Config.load(None, overrides=[
        "filter.selection.method=c_score", "filter.selection.gamma_dtail=1.0",
        "improve.operator=lora_sft", "engine.enable_lora=true",
    ])
    cands = [_score_cand(0, {"lora_path": "/fake/adapter"}), _score_cand(1)]
    seen = []

    def fake(requests, **kw):
        out = []
        for r in requests:
            seen.append((r.rid, r.lora_path))
            lp = [-0.05] * 10 if r.rid.endswith(":d") else [-0.1] * 10
            out.append(GenResult(rid=r.rid, token_logprobs=lp))
        return out

    monkeypatch.setattr(engine_mod, "run_pool", fake)
    scores, report = _score_candidates(cands, cfg, "org/policy", tmp_path)
    kept, n_over = _rank_by_c_score(cands, cfg, scores)
    _write_candidate_scores(scores, kept, tmp_path)

    assert ("q:0:0:d", "/fake/adapter") in seen          # q_P pass for the ref'd cand
    assert all(rid != "q:0:1:d" for rid, _ in seen)      # none for the missing one
    assert report["n_missing_lora_ref"] == 1
    assert "d_tail" in report
    persisted = {s["key"]: s for s in read_jsonl(tmp_path / "filtered" / "candidate_scores.jsonl")}
    assert persisted["q:0:0"]["d_tail"] == pytest.approx(0.05)   # lp_q - lp_theta
    assert persisted["q:0:1"]["d_tail"] is None
    assert len(kept) == 2 and n_over == 0


def test_always_score_measures_c_without_changing_selection(monkeypatch, tmp_path):
    """always_score: random/shortest arms still report comparable C(y) stats,
    and the persisted kept flags follow the ACTUAL selection."""
    from expert_iter.filters import _random_selection

    cfg = Config.load(None, overrides=[
        "filter.selection.method=random", "filter.selection.always_score=true",
        "filter.max_per_question=1",
    ])
    cands = [_score_cand(0), _score_cand(1), _score_cand(2)]
    lps = {"q:0:0:s": [-0.1] * 10, "q:0:1:s": [-1.0] * 10, "q:0:2:s": [-0.5] * 10}
    monkeypatch.setattr(
        engine_mod, "run_pool",
        lambda requests, **kw: [GenResult(rid=r.rid, token_logprobs=lps[r.rid])
                                for r in requests],
    )
    scores, report = _score_candidates(cands, cfg, "org/policy", tmp_path)
    kept, _ = _random_selection(cands, cfg)          # selection ignores C
    _write_candidate_scores(scores, kept, tmp_path)

    assert report["n_scored"] == 3
    assert report["c"]["mean"] == pytest.approx((0.2 + 2.0 + 1.0) / 3, abs=1e-4)
    persisted = {s["key"]: s for s in read_jsonl(tmp_path / "filtered" / "candidate_scores.jsonl")}
    assert sum(r["kept"] for r in persisted.values()) == 1        # random kept exactly one
    assert persisted[f"q:0:{kept[0].attempt_idx}"]["kept"] is True
