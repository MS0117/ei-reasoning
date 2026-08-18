"""privileged_divergence anchor + min_mean_nll base selection, with a
monkeypatched engine pool (no GPU)."""

import random

import pytest

from expert_iter import engine as engine_mod
from expert_iter.anchor import (
    AnchorContext,
    PrivilegedDivergenceAnchor,
    _select_min_mean_nll,
    _step_means,
    _union_topk_kl,
)
from expert_iter.config import Config
from expert_iter.engine import GenResult
from expert_iter.records import RolloutSample, UnsolvedQuestion


class FakeTok:
    """30-token responses split into 3-token steps by a '\\n\\n' piece every
    third token. Also serves the chat-template surface the privileged prompt
    rendering needs."""

    eos_token_id = 0

    def piece(self, i):
        return "x\n\n" if i % 3 == 2 else f"t{i} "

    def batch_decode(self, batches):
        return [self.piece(b[0]) for b in batches]

    def decode(self, ids):
        return "".join(self.piece(i) for i in ids)

    def apply_chat_template(self, messages, **kw):
        return "PROMPT:" + messages[-1]["content"][:16]

    def __call__(self, text, **kw):
        return {"input_ids": [9] * 4}  # privileged context = 4 ids


def _rollout(qid, n_tokens=30, sample_idx=0):
    return RolloutSample(
        qid=qid, sample_idx=sample_idx, prompt_text="p", prompt_token_ids=[1, 2, 3],
        response_text="r", response_token_ids=list(range(n_tokens)),
        finish_reason="stop",
    )


def _ctx(cfg, questions, gold, selected, tmp_path):
    return AnchorContext(
        cfg=cfg, tokenizer=FakeTok(), model_path="org/policy",
        work_dir=tmp_path, iteration=0, gold_solutions=gold,
        selected=selected, failed_by_qid={}, rng=random.Random(0),
    )


def test_divergence_anchor_cuts_before_the_spike(monkeypatch, tmp_path):
    cfg = Config.load(None, overrides=["anchor.policy=privileged_divergence"])
    q = UnsolvedQuestion(qid="q1", question="?", final_answer="1")
    selected = {"q1": _rollout("q1")}

    def fake_run_pool(requests, **kw):
        # cap = ceil(30/3) = 10 scored positions per request
        out = []
        for r in requests:
            if r.rid.endswith(":base"):
                lps = [-1.0] * 10
            else:  # privileged pass stops endorsing tokens 6..9 (step 3)
                lps = [-1.0] * 6 + [-5.0] * 4
            out.append(GenResult(rid=r.rid, token_logprobs=lps))
        return out

    monkeypatch.setattr(engine_mod, "run_pool", fake_run_pool)
    rows = PrivilegedDivergenceAnchor().propose_batch(
        [q], _ctx(cfg, [q], {"q1": "gold text"}, selected, tmp_path),
    )
    assert len(rows) == 1
    r = rows[0]
    # g_bar = [0, 0, 4] over searched steps [3, 6, 9]; no threshold crossing at
    # 3 steps -> discrete-argmax j*=3 -> j_a=2 -> anchor ends at bound 6
    assert r.anchor_len == 6
    assert r.anchor_token_ids == list(range(6))       # exact id-slice
    assert r.meta["j_star"] == 3 and r.meta["j_anchor"] == 2
    assert r.meta["threshold_crossed"] is False
    assert r.meta["g_bar"] == [0.0, 0.0, 4.0]
    assert r.meta["search_end_tok"] == 10


def test_no_gold_degenerates_to_empty_anchor(monkeypatch, tmp_path):
    cfg = Config.load(None, overrides=["anchor.policy=privileged_divergence"])
    qs = [UnsolvedQuestion(qid="q1", question="?", final_answer="1"),
          UnsolvedQuestion(qid="q2", question="?", final_answer="2")]
    selected = {"q1": _rollout("q1"), "q2": _rollout("q2")}

    def fake_run_pool(requests, **kw):
        return [GenResult(rid=r.rid, token_logprobs=[-1.0] * 10) for r in requests]

    monkeypatch.setattr(engine_mod, "run_pool", fake_run_pool)
    rows = PrivilegedDivergenceAnchor().propose_batch(
        qs, _ctx(cfg, qs, {"q1": "gold"}, selected, tmp_path),
    )
    by_qid = {r.qid: r for r in rows}
    assert by_qid["q2"].anchor_len == 0
    assert by_qid["q2"].meta["reason"] == "no_gold"


def test_zero_gold_overall_raises(tmp_path):
    cfg = Config.load(None, overrides=["anchor.policy=privileged_divergence"])
    q = UnsolvedQuestion(qid="q1", question="?", final_answer="1")
    with pytest.raises(RuntimeError, match="gold"):
        PrivilegedDivergenceAnchor().propose_batch(
            [q], _ctx(cfg, [q], {}, {"q1": _rollout("q1")}, tmp_path),
        )


def test_too_few_searched_steps_degenerates(monkeypatch, tmp_path):
    cfg = Config.load(None, overrides=["anchor.policy=privileged_divergence"])
    q = UnsolvedQuestion(qid="q1", question="?", final_answer="1")
    selected = {"q1": _rollout("q1", n_tokens=6)}  # cap=2 -> <3 searched steps
    monkeypatch.setattr(engine_mod, "run_pool",
                        lambda requests, **kw: pytest.fail("no pool call expected"))
    rows = PrivilegedDivergenceAnchor().propose_batch(
        [q], _ctx(cfg, [q], {"q1": "gold"}, selected, tmp_path),
    )
    assert rows[0].anchor_len == 0
    assert rows[0].meta["reason"] == "too_few_steps"


# ---- helpers ----------------------------------------------------------------

def test_step_means_skips_none():
    g = [1.0, None, 3.0, 5.0]
    assert _step_means(g, [2, 4]) == [1.0, 4.0]


def test_step_means_empty_step_is_zero():
    assert _step_means([None, None, 2.0], [2, 3]) == [0.0, 2.0]


def test_union_topk_kl_identical_is_zero():
    d = {"1": -0.5, "2": -1.5}
    assert _union_topk_kl(d, d) == pytest.approx(0.0, abs=1e-9)


def test_union_topk_kl_positive_when_shifted():
    p = {"1": -0.1, "2": -3.0}
    q = {"1": -3.0, "2": -0.1}
    assert _union_topk_kl(p, q) > 0.5


def test_min_mean_nll_picks_most_typical_failure(monkeypatch, tmp_path):
    cfg = Config.load(None)
    failed = {
        "q1": [_rollout("q1", sample_idx=0), _rollout("q1", sample_idx=1)],
    }

    def fake_run_pool(requests, **kw):
        # sample 1 has HIGHER mean logprob = LOWER mean NLL -> most typical
        return [GenResult(rid=r.rid,
                          mean_logprob=-1.0 if r.rid.endswith(":1") else -2.0)
                for r in requests]

    monkeypatch.setattr(engine_mod, "run_pool", fake_run_pool)
    picked = _select_min_mean_nll(failed, cfg, "org/policy", tmp_path)
    assert picked["q1"].sample_idx == 1


def test_min_mean_nll_tie_breaks_to_lowest_idx(monkeypatch, tmp_path):
    cfg = Config.load(None)
    failed = {"q1": [_rollout("q1", sample_idx=0), _rollout("q1", sample_idx=1)]}
    monkeypatch.setattr(
        engine_mod, "run_pool",
        lambda requests, **kw: [GenResult(rid=r.rid, mean_logprob=-1.0) for r in requests],
    )
    picked = _select_min_mean_nll(failed, cfg, "org/policy", tmp_path)
    assert picked["q1"].sample_idx == 0


# ---------------------------------------------------------------------------
# matched_minus_shuffled: cancel the "a reference was shown at all" confound
# ---------------------------------------------------------------------------

def test_shuffled_reference_map_pairs_by_length_never_self():
    from expert_iter.anchor import shuffled_reference_map

    gold = {"q1": "a" * 10, "q2": "b" * 12, "q3": "c" * 90, "q4": "d" * 91}
    m = shuffled_reference_map(gold)
    assert set(m) == set(gold)
    assert all(m[q] != gold[q] for q in gold)          # never its own reference
    # neighbours in the length ordering -> comparable context size
    assert m["q1"] == gold["q2"] and m["q3"] == gold["q4"]
    assert shuffled_reference_map({"only": "x"}) == {}  # degenerate, no partner


def test_shuffled_reference_map_is_deterministic():
    from expert_iter.anchor import shuffled_reference_map

    gold = {f"q{i}": "x" * (i % 7) for i in range(20)}
    assert shuffled_reference_map(gold) == shuffled_reference_map(dict(reversed(gold.items())))


def test_matched_minus_shuffled_scores_two_privileged_contexts(monkeypatch, tmp_path):
    """The control pass must be prompt(x, y*_other) — NOT the bare prompt(x),
    which is what makes realized_logratio spike on a response's opening."""
    cfg = Config.load(None, overrides=[
        "anchor.policy=privileged_divergence",
        "anchor.params.signal=matched_minus_shuffled",
    ])
    qs = [UnsolvedQuestion(qid=f"q{i}", question="?", final_answer="1") for i in (1, 2)]
    selected = {q.qid: _rollout(q.qid) for q in qs}
    gold = {"q1": "short gold", "q2": "a much longer gold reference text"}
    seen = {}

    def fake_run_pool(requests, **kw):
        seen.update({r.rid: (list(r.prompt_token_ids[:r.score_from]), r.score_from)
                     for r in requests})
        return [GenResult(rid=r.rid, token_logprobs=[-1.0] * 10) for r in requests]

    monkeypatch.setattr(engine_mod, "run_pool", fake_run_pool)
    PrivilegedDivergenceAnchor().propose_batch(
        qs, _ctx(cfg, qs, gold, selected, tmp_path),
    )
    for qid in ("q1", "q2"):
        base_ids, _ = seen[f"{qid}:base"]
        priv_ids, _ = seen[f"{qid}:priv"]
        # the control is a PRIVILEGED render, not the bare rollout prompt
        assert base_ids != selected[qid].prompt_token_ids, "control fell back to prompt(x)"
        assert len(base_ids) == len(priv_ids), "both contexts are privileged renders"


def test_realized_logratio_still_uses_the_bare_prompt(monkeypatch, tmp_path):
    """The original signal stays selectable — it is the ablation baseline."""
    cfg = Config.load(None, overrides=["anchor.policy=privileged_divergence"])
    q = UnsolvedQuestion(qid="q1", question="?", final_answer="1")
    selected = {"q1": _rollout("q1")}
    seen = {}

    def fake_run_pool(requests, **kw):
        seen.update({r.rid: list(r.prompt_token_ids[:r.score_from]) for r in requests})
        return [GenResult(rid=r.rid, token_logprobs=[-1.0] * 10) for r in requests]

    monkeypatch.setattr(engine_mod, "run_pool", fake_run_pool)
    PrivilegedDivergenceAnchor().propose_batch(
        [q], _ctx(cfg, [q], {"q1": "gold"}, selected, tmp_path),
    )
    assert seen["q1:base"] == selected["q1"].prompt_token_ids


def test_position_signal_routes_matched_minus_shuffled_to_logprobs():
    """Regression: a signal name missing from this branch silently falls through
    to the top-K path, which has no data unless prompt_logprobs_k was requested —
    producing an all-None signal and therefore an all-zero g_bar (observed on a
    real 99-question run before this guard existed)."""
    from expert_iter.anchor import _position_signal

    priv = GenResult(rid="q:priv", token_logprobs=[-2.0, -2.0, -2.0])
    base = GenResult(rid="q:base", token_logprobs=[-1.0, -3.0, -1.0])
    for signal in ("realized_logratio", "matched_minus_shuffled"):
        g = _position_signal(signal, priv, base, n_positions=3)
        assert g == [1.0, -1.0, 1.0], f"{signal} did not use token_logprobs"
    # top-K without prompt_logprobs_k is the degenerate case the guard protects
    assert _position_signal("topk_kl", priv, base, n_positions=3) == [None] * 3
