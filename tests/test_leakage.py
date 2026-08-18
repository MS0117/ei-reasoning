"""Leakage filter: rule gate + LLM-judge pass (filters.py). No GPU."""

from expert_iter import engine as engine_mod
from expert_iter.config import Config
from expert_iter.engine import GenResult
from expert_iter.filters import FilterContext, LeakageRulesGate, _leakage_judge
from expert_iter.records import ImprovedCandidate


def _cand(text, i=0):
    return ImprovedCandidate(
        qid="q", base_sample_idx=0, attempt_idx=i, prompt_token_ids=[1],
        anchor_token_ids=[], continuation_token_ids=[i], continuation_text=text,
    )


def _ctx(cfg=None):
    return FilterContext(cfg=cfg or Config.load(None), verifier=None,
                         tokenizer=None, questions={})


def test_rules_gate_flags_english_reference():
    ok, reason = LeakageRulesGate().check(
        _cand("By the reference solution, we conclude x = 2."), _ctx(),
    )
    assert not ok and reason == "pattern"


def test_rules_gate_flags_korean_pattern():
    ok, _ = LeakageRulesGate().check(_cand("주어진 풀이에 따르면 답은 3이다."), _ctx())
    assert not ok


def test_rules_gate_is_case_insensitive():
    ok, _ = LeakageRulesGate().check(_cand("The GIVEN SOLUTION shows..."), _ctx())
    assert not ok


def test_rules_gate_passes_clean_text():
    ok, reason = LeakageRulesGate().check(
        _cand("Therefore the sum telescopes and x = 4."), _ctx(),
    )
    assert ok and reason == ""


class FakeTok:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kw):
        return "JUDGE:" + messages[-1]["content"][:8]

    def __call__(self, text, **kw):
        return {"input_ids": [1, 2]}


def test_judge_pass_parses_verdicts(monkeypatch, tmp_path):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: FakeTok())
    cands = [_cand("a", 0), _cand("b", 1), _cand("c", 2)]
    replies = {"q:0:0": "YES", "q:0:1": "No, it does not.", "q:0:2": "I think maybe"}

    def fake_run_pool(requests, **kw):
        return [GenResult(rid=r.rid, samples=[{
            "text": replies[r.rid], "token_ids": [1], "finish_reason": "stop",
        }]) for r in requests]

    monkeypatch.setattr(engine_mod, "run_pool", fake_run_pool)
    cfg = Config.load(None, overrides=["filter.leakage.judge_enabled=true"])
    kept, n_flagged, report = _leakage_judge(cands, cfg, "org/policy", tmp_path)

    assert [c.attempt_idx for c in kept] == [1, 2]      # YES rejected
    assert n_flagged == 1
    assert report["n_unparsed_kept"] == 1               # garbage kept conservatively
    assert report["judge_model"] == "org/policy"
