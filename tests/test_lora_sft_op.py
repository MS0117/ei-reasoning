"""lora_sft operator: chunking, alpha* selection, candidate emission, refit —
all with a monkeypatched pool and fit (no GPU, no downloads)."""

import json

import pytest

from expert_iter import lora_sft as mod
from expert_iter.config import Config
from expert_iter.engine import GenResult
from expert_iter.lora_sft import LoraSftOperator, select_alpha_star
from expert_iter.records import AnchorRecord, UnsolvedQuestion
from expert_iter.utils import read_jsonl


def _curves(improve_dir):
    return {r["qid"]: r for r in read_jsonl(improve_dir / "alpha_curves.jsonl")}


def test_lora_sft_import_has_no_cycle_back_into_improve():
    """`python -m expert_iter.improve` executes improve.py as __main__; if
    lora_sft imported expert_iter.improve, that would RE-EXECUTE the module
    body under its canonical name -> duplicate operator registrations (seen
    live on GPU). Pin: importing lora_sft alone must not pull in improve."""
    import subprocess
    import sys as _sys

    proc = subprocess.run(
        [_sys.executable, "-c",
         "import expert_iter.lora_sft, sys; "
         "assert 'expert_iter.improve' not in sys.modules, 'import cycle back into improve'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_select_alpha_star_min_crossing():
    assert select_alpha_star({0.05: 0.0, 0.1: 0.3, 1.0: 0.5}, 0.25) == 0.1


def test_select_alpha_star_boundary_inclusive():
    assert select_alpha_star({0.5: 0.25, 1.0: 1.0}, 0.25) == 0.5


def test_select_alpha_star_fallback_to_one():
    assert select_alpha_star({0.5: 0.1, 1.0: 0.2}, 0.25) == 1.0


def test_chunks():
    cfg = Config.load(None)
    ls = cfg.improve.lora_sft
    qids = ["a", "b", "c", "d", "e"]
    assert LoraSftOperator._chunks(qids, ls) == [qids]          # pooled, no chunk_size
    ls.chunk_size = 2
    assert LoraSftOperator._chunks(qids, ls) == [["a", "b"], ["c", "d"], ["e"]]
    ls.adapter_scope = "per_problem"
    assert LoraSftOperator._chunks(qids, ls) == [["a"], ["b"], ["c"], ["d"], ["e"]]


# ---------------------------------------------------------------------------
# end-to-end propose() with fakes
# ---------------------------------------------------------------------------

class FakeTok:
    eos_token_id = 0
    vocab = {
        101: "The answer is \\boxed{4}",
        102: "wrong: \\boxed{5}",
        103: "recovered: \\boxed{7}",
    }

    def __call__(self, text, **kw):
        return {"input_ids": [1, 2, 3]}

    def decode(self, ids):
        return " ".join(self.vocab.get(i, "") for i in ids)


def _fake_fit(policy, pairs, adapters_dir, name, params, cfg, **kw):
    adapter_dir = adapters_dir / name / "FAKEKEY"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": params["lora_alpha"], "r": params["r"]})
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    return adapter_dir, 0.0


def _sample(tok_id):
    return {"text": FakeTok.vocab[tok_id], "token_ids": [tok_id], "finish_reason": "stop"}


def _setup(monkeypatch, cfg, responder):
    """Patch tokenizer, fit subprocess, and pool. responder(qid, alpha, n, call_idx)
    -> list of sample dicts."""
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: FakeTok())
    monkeypatch.setattr(mod, "_fit_adapter", _fake_fit)
    calls = {"n": 0, "requests": []}

    def fake_run_pool(requests, **kw):
        calls["n"] += 1
        calls["requests"].append(list(requests))
        out = []
        for r in requests:
            qid, alpha = r.rid.rsplit(":a", 1)
            out.append(GenResult(
                rid=r.rid,
                samples=responder(qid, float(alpha), r.n, calls["n"]),
            ))
        return out

    monkeypatch.setattr(mod, "run_pool", fake_run_pool)
    return calls


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
        "improve.operator=lora_sft", "engine.enable_lora=true", "improve.n=2", *extra,
    ])


def test_vanilla_propose(monkeypatch, tmp_path):
    cfg = _cfg()
    questions, anchors, prompts, gold = _fixtures()

    def responder(qid, alpha, n, call_idx):
        assert alpha == 1.0
        return [_sample(101), _sample(102)] if qid == "q1" else [_sample(102), _sample(102)]

    calls = _setup(monkeypatch, cfg, responder)
    op = LoraSftOperator()
    cands = op.propose(questions, anchors, prompts, cfg,
                       model_paths={"policy": "org/policy"},
                       work_dir=tmp_path / "pool", iteration=0,
                       gold_solutions=gold)

    assert calls["n"] == 1  # ONE pool launch
    by_qid = {}
    for c in cands:
        by_qid.setdefault(c.qid, []).append(c)
    # all stop-finished samples at alpha* are emitted, correct prefilled
    assert [c.correct for c in by_qid["q1"]] == [True, False]
    assert all(c.correct is False for c in by_qid["q2"])
    c = by_qid["q1"][0]
    assert c.op_meta["alpha_star"] == 1.0
    assert c.op_meta["channel"] == "lora_weights"
    assert c.op_meta["privileged"] == "gold_solution"
    assert c.external_context is None
    assert c.prompt_token_ids == [1, 2] and c.anchor_token_ids == []
    # per-request max_tokens cap: min(improve.max_tokens, budget)
    req = calls["requests"][0][0]
    budget = cfg.filter.max_total_tokens - 2
    assert req.max_tokens == min(cfg.improve.max_tokens, budget)
    assert req.lora_path and "FAKEKEY" in req.lora_path
    # stats
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_eligible"] == 2 and stats["n_with_gold"] == 2
    assert stats["n_resolved_pool"] == 1 and stats["lora_yield"] == 0.5
    assert stats["n_fits"] == 1
    # per-question P_hat curves for alpha-survival analysis
    curves = _curves(tmp_path)
    assert curves["q1"]["p_hat_by_alpha"] == {"1": 0.5}
    assert curves["q2"]["p_hat_by_alpha"] == {"1": 0.0}
    assert curves["q1"]["alpha_star"] == 1.0 and curves["q1"]["refit"] is False


def test_project_back_selects_min_alpha(monkeypatch, tmp_path):
    cfg = _cfg(
        "improve.lora_sft.project_back.enabled=true",
        "improve.lora_sft.project_back.alphas=[0.5, 1.0]",
        "improve.lora_sft.project_back.m_rollouts=2",
        "improve.lora_sft.project_back.tau_p=0.25",
    )
    questions, anchors, prompts, gold = _fixtures()

    def responder(qid, alpha, n, call_idx):
        if qid == "q1":  # correct at BOTH alphas -> alpha* must be the smaller
            return [_sample(101), _sample(102)]
        return [_sample(102), _sample(102)]

    calls = _setup(monkeypatch, cfg, responder)
    cands = LoraSftOperator().propose(
        questions, anchors, prompts, cfg,
        model_paths={"policy": "org/policy"},
        work_dir=tmp_path / "pool", iteration=0, gold_solutions=gold,
    )
    # 2 qids x 2 alphas requests in one launch
    assert len(calls["requests"][0]) == 4
    q1 = [c for c in cands if c.qid == "q1"]
    assert q1 and all(c.op_meta["alpha_star"] == 0.5 for c in q1)
    assert q1[0].op_meta["p_hat_by_alpha"] == {"0.5": 0.5, "1": 0.5}
    assert "alpha/0.5" in q1[0].op_meta["lora_path"]
    # q2 never crosses tau -> fallback alpha*=1.0
    q2 = [c for c in cands if c.qid == "q2"]
    assert q2 and all(c.op_meta["alpha_star"] == 1.0 for c in q2)
    curves = _curves(tmp_path)
    assert curves["q1"] == {"qid": "q1", "p_hat_by_alpha": {"0.5": 0.5, "1": 0.5},
                            "alpha_star": 0.5, "refit": False}
    assert curves["q2"]["alpha_star"] == 1.0


def test_refit_replaces_unresolved(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.refit_budget=1")
    questions, anchors, prompts, gold = _fixtures()

    def responder(qid, alpha, n, call_idx):
        if call_idx == 1:  # main pass: q2 unresolved
            return [_sample(101), _sample(102)] if qid == "q1" else [_sample(102), _sample(102)]
        assert qid == "q2"  # refit pass only covers the unresolved cliff
        return [_sample(103), _sample(102)]

    calls = _setup(monkeypatch, cfg, responder)
    cands = LoraSftOperator().propose(
        questions, anchors, prompts, cfg,
        model_paths={"policy": "org/policy"},
        work_dir=tmp_path / "pool", iteration=0, gold_solutions=gold,
    )
    assert calls["n"] == 2
    q2 = [c for c in cands if c.qid == "q2"]
    assert all(c.op_meta["refit"] for c in q2)          # first-pass q2 rows replaced
    assert [c.correct for c in q2] == [True, False]
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_refits"] == 1 and stats["n_resolved_refit"] == 1
    assert stats["lora_yield"] == 1.0
    # attempt_idx unique within (qid, refit-pass)
    assert [c.attempt_idx for c in q2] == [0, 1]
    # refit curve rows replace the first-pass rows for refitted questions
    curves = _curves(tmp_path)
    assert curves["q2"]["refit"] is True and curves["q1"]["refit"] is False
    assert curves["q2"]["p_hat_by_alpha"] == {"1": 0.5}


def test_all_gold_missing_raises(monkeypatch, tmp_path):
    cfg = _cfg()
    questions, anchors, prompts, _ = _fixtures()
    _setup(monkeypatch, cfg, lambda *a: [])
    with pytest.raises(RuntimeError, match="gold_solution"):
        LoraSftOperator().propose(
            questions, anchors, prompts, cfg,
            model_paths={"policy": "org/policy"},
            work_dir=tmp_path / "pool", iteration=0, gold_solutions={},
        )


def test_partial_gold_counts(monkeypatch, tmp_path):
    cfg = _cfg()
    questions, anchors, prompts, gold = _fixtures()
    gold = {"q1": gold["q1"]}  # q2 has no solution -> skipped, counted

    def responder(qid, alpha, n, call_idx):
        assert qid == "q1"
        return [_sample(101), _sample(101)]

    _setup(monkeypatch, cfg, responder)
    cands = LoraSftOperator().propose(
        questions, anchors, prompts, cfg,
        model_paths={"policy": "org/policy"},
        work_dir=tmp_path / "pool", iteration=0, gold_solutions=gold,
    )
    assert {c.qid for c in cands} == {"q1"}
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_no_gold"] == 1 and stats["n_with_gold"] == 1
