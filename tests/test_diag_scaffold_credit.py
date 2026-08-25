"""CPU tests for scripts/diag_scaffold_credit.py (pure statistics only; the
GPU pool call is not exercised)."""
import importlib.util
import math
import random
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "diag_scaffold_credit",
    Path(__file__).resolve().parents[1] / "scripts" / "diag_scaffold_credit.py",
)
D = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(D)


def _synthetic(n=400, seed=0):
    """31 fork tokens (base unsure, lora sure), ~37 noise tokens (both unsure),
    the rest shared-confident. Returns aligned (lp_base, lp_lora, topk_base)."""
    rng = random.Random(seed)
    lp_b, lp_l, topk = [], [], []
    for i in range(n):
        if i % 13 == 0:
            b, l = -3.0 + rng.gauss(0, 0.3), -0.2
        elif i % 10 == 0:
            b = -2.5 + rng.gauss(0, 0.3)
            l = b + 0.1 + rng.gauss(0, 0.3)
        else:
            b = -0.05 + rng.gauss(0, 0.02)
            l = b + rng.gauss(0, 0.02)
        lp_b.append(b)
        lp_l.append(l)
        p = math.exp(b)
        rest = (1 - p) / 19
        topk.append({str(j): math.log(max(rest if j else p, 1e-9)) for j in range(20)})
    return lp_b, lp_l, topk


def test_per_token_signals_alignment_and_missing():
    lp_b, lp_l, topk = _synthetic()
    lp_b[5] = None
    sig = D.per_token_signals(lp_b, lp_l[:-3], topk, len(lp_b))  # lora pass 3 short
    assert len(sig) == len(lp_b)
    assert sig[5] is None and sig[-1] is None and sig[-3] is None
    d = sig[0]  # a fork token
    assert d["g"] == pytest.approx(lp_l[0] - lp_b[0])
    assert d["s"] == pytest.approx(-lp_b[0])
    assert 0.0 <= d["a"] <= 1.0 and d["a"] > 0.8
    assert d["H"] is not None and d["H"] > 0


def test_candidate_stats_separates_forks_from_noise():
    lp_b, lp_l, topk = _synthetic()
    sig = D.per_token_signals(lp_b, lp_l, topk, len(lp_b))
    st = D.candidate_stats(sig, list(range(25, 401, 25)), q_frac=0.2, s_hi=1.0,
                           a_min=0.5, g_fork=1.0)
    assert st["n_scored"] == 400
    # gap mass sits on the forks
    assert st["concentration"]["top10_share_gpos"] > 0.8
    assert st["concentration"]["gini_gpos"] > 0.8
    # both classes present and the loss mass splits between them
    assert st["classes"]["n_A"] >= 28 and st["classes"]["n_B"] >= 30
    assert 0.3 < st["classes"]["loss_mass_B"] < 0.7
    # forks are isolated single tokens
    assert st["forking"]["p_next_also_fork"] < 0.1
    assert st["forking"]["mean_g_8_after_fork"] < st["forking"]["mean_g_all"]
    # sequence IS ratio is exactly -sum g
    assert st["is_view"]["seq_log_ratio_base_over_lora"] == pytest.approx(-st["sum_g"])
    # overlap bounded
    assert 0.0 <= st["overlap"]["jaccard_g_vs_entropy"] <= 1.0


def test_candidate_stats_all_missing_and_helpers():
    st = D.candidate_stats([None] * 5, [5], q_frac=0.2, s_hi=1.0, a_min=0.5, g_fork=1.0)
    assert st["n_scored"] == 0
    assert D.gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert D.gini([0, 0, 0, 4]) == pytest.approx(0.75)
    assert D.top_share([1, 1, 8], 1 / 3) == pytest.approx(0.8)
    assert D.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert D.step_index_of([2, 5], 5) == [0, 0, 1, 1, 1]
    assert D.topk_entropy(None) is None
    assert D.topk_entropy({"a": 0.0}) == pytest.approx(0.0)
    assert D.topk_entropy({"a": math.log(0.5), "b": math.log(0.5)}) == pytest.approx(math.log(2))


def _cand(qid, idx, correct, lora=True):
    from expert_iter.records import ImprovedCandidate
    return ImprovedCandidate(
        qid=qid, base_sample_idx=0, attempt_idx=idx, prompt_token_ids=[1, 2],
        anchor_token_ids=[], continuation_token_ids=[3, 4, 5], continuation_text="x",
        correct=correct, operator="t", op_meta={"lora_path": "a/b"} if lora else {},
    )


def test_select_candidates_balanced_keeps_questions_paired():
    cands = []
    for q in range(10):
        for i in range(3):
            cands.append(_cand(f"q{q}", i, correct=(i == 0) or (q < 3)))  # q0-2: all correct
        for i in range(3, 6):
            cands.append(_cand(f"q{q}", i, correct=False))
    cands.append(_cand("only_neg", 0, correct=False))
    cands.append(_cand("only_pos", 0, correct=True))
    cands.append(_cand("no_lora", 0, correct=True, lora=False))
    picked = D.select_candidates(cands, max_per_qid=2, max_cands=1000, seed=0,
                                 include_incorrect=False, balanced=True)
    pos = [c for c in picked if c.correct]
    neg = [c for c in picked if not c.correct]
    assert {c.qid for c in pos} == {c.qid for c in neg} == {f"q{q}" for q in range(10)}
    assert all(sum(1 for c in pos if c.qid == q) <= 2 for q in {c.qid for c in pos})
    # the per-class cap removes whole questions, never one class of a question
    capped = D.select_candidates(cands, max_per_qid=2, max_cands=9, seed=0,
                                 include_incorrect=False, balanced=True)
    cp = {c.qid for c in capped if c.correct}
    cn = {c.qid for c in capped if not c.correct}
    assert cp == cn and 0 < len(cp) < 10
    assert sum(1 for c in capped if c.correct) <= 9 and sum(1 for c in capped if not c.correct) <= 9
    # unbalanced path: lora required by default, privileged path may waive it
    plain = D.select_candidates(cands, max_per_qid=5, max_cands=1000, seed=0, include_incorrect=False)
    assert all(c.correct for c in plain) and not any(c.qid == "no_lora" for c in plain)
    waived = D.select_candidates(cands, max_per_qid=5, max_cands=1000, seed=0, include_incorrect=False,
                                 require_lora=False, allowed_qids={"no_lora", "q1"})
    assert {c.qid for c in waived} == {"no_lora", "q1"}


def test_aggregate_and_report_render():
    lp_b, lp_l, topk = _synthetic()
    sig = D.per_token_signals(lp_b, lp_l, topk, len(lp_b))
    st = D.candidate_stats(sig, [400], q_frac=0.2, s_hi=1.0, a_min=0.5, g_fork=1.0)
    rows = [{"key": f"k{j}", "group": "stage1" if j % 2 else "stage2", **st} for j in range(4)]
    summ = D.aggregate(rows)
    assert summ["n_candidates"] == 4
    assert summ["concentration/top10_share_gpos"]["n"] == 4
    assert set(summ["by_group"]) == {"stage1", "stage2"}
    knobs = {"q_frac": 0.2, "s_hi": 1.0, "a_min": 0.5, "g_fork": 1.0, "topk": 20, "max_tokens": None}
    rep = D.render_report(summ, knobs)
    assert "[1] CONCENTRATION" in rep and "[by group]" in rep
