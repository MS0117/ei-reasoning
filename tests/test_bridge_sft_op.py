"""bridge_sft operator: bridge generation/verify/retry/skip + pair building —
monkeypatched pool and fit (no GPU, no downloads)."""

import json

import pytest

from expert_iter import lora_sft as ls_mod
from expert_iter.bridge_sft import BridgeSftOperator
from expert_iter.config import Config
from expert_iter.engine import GenResult
from expert_iter.records import AnchorRecord, UnsolvedQuestion
from expert_iter.utils import read_jsonl


class FakeTok:
    """decode maps ids -> text via vocab; call/apply_chat_template serve the
    bridge-prompt rendering; PROMPT_IDS marks the rendered privileged prompt."""

    eos_token_id = 0
    PROMPT_IDS = [9, 9, 9]
    vocab = {
        101: "The answer is \\boxed{4}",
        102: "wrong: \\boxed{5}",
        103: "By the reference solution, \\boxed{4}",   # correct but leaky
        104: "recovered: \\boxed{7}",
        105: "Longer correct solution text \\boxed{4}",
    }

    def apply_chat_template(self, messages, **kw):
        return "BRIDGE:" + messages[-1]["content"][:8]

    def __call__(self, text, **kw):
        return {"input_ids": list(self.PROMPT_IDS)}

    def decode(self, ids):
        return " ".join(self.vocab.get(i, "") for i in ids)


def _fake_fit(policy, pairs, adapters_dir, name, params, cfg, **kw):
    adapter_dir = adapters_dir / name / "FAKEKEY"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": params["lora_alpha"], "r": params["r"]}))
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    return adapter_dir, 0.0


def _sample(tok_id):
    return {"text": FakeTok.vocab[tok_id], "token_ids": [tok_id], "finish_reason": "stop"}


def _fixtures(anchored=False):
    questions = [
        UnsolvedQuestion(qid="q1", question="2+2?", final_answer="4"),
        UnsolvedQuestion(qid="q2", question="3+4?", final_answer="7"),
    ]
    anchor_ids = {"q1": [7, 8] if anchored else [], "q2": []}
    anchors = [
        AnchorRecord(qid=q.qid, base_sample_idx=0, policy="none",
                     anchor_token_ids=anchor_ids[q.qid], anchor_text="",
                     anchor_len=len(anchor_ids[q.qid]), base_response_len=10)
        for q in questions
    ]
    prompts = {"q1": [1, 2], "q2": [1, 2]}
    gold = {"q1": "It is 4.", "q2": "It is 7."}
    return questions, anchors, prompts, gold


def _cfg(*extra):
    return Config.load(None, overrides=[
        "improve.operator=bridge_sft", "engine.enable_lora=true", "improve.n=2",
        "improve.lora_sft.bridge.n=2", *extra,
    ])


def _setup(monkeypatch, responder):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: FakeTok())
    monkeypatch.setattr(ls_mod, "_fit_adapter", _fake_fit)
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


def _propose(op, cfg, tmp_path, fixtures):
    questions, anchors, prompts, gold = fixtures
    return op.propose(questions, anchors, prompts, cfg,
                      model_paths={"policy": "org/policy"},
                      work_dir=tmp_path / "pool", iteration=0,
                      gold_solutions=gold)


def test_bridge_pairs_and_candidates(monkeypatch, tmp_path):
    cfg = _cfg()
    fixtures = _fixtures()

    def responder(qid, kind, r):
        if kind == "bridge":
            # q1 bridges: one correct + one wrong; q2: one correct
            return [_sample(101), _sample(102)] if qid == "q1" else [_sample(104), _sample(102)]
        assert kind == "a1"      # candidate sampling from q_phi
        return [_sample(101), _sample(102)] if qid == "q1" else [_sample(102), _sample(102)]

    calls = _setup(monkeypatch, responder)
    cands = _propose(BridgeSftOperator(), cfg, tmp_path, fixtures)

    assert calls["n"] == 2  # ONE bridge launch + ONE candidate launch (no retry)
    # fit pairs recorded in bridges.jsonl; kept flags on the correct ones
    rows = list(read_jsonl(tmp_path / "bridge" / "bridges.jsonl"))
    assert sum(r["kept"] for r in rows) == 2
    assert all(r["pass"] == "bridge" for r in rows)
    # stats: bridge counters
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_bridge_generated"] == 4 and stats["n_bridge_correct"] == 2
    assert stats["n_questions_bridged"] == 2 and stats["bridge_yield"] == 1.0
    assert stats["n_bridge_skipped"] == 0 and stats["n_bridge_retried"] == 0
    # candidates carry bridge provenance and operator name
    c = next(c for c in cands if c.qid == "q1" and c.correct)
    assert c.operator == "bridge_sft"
    assert c.op_meta["via"] == "bridge_trajectories"
    assert c.op_meta["channel"] == "lora_weights"
    assert c.external_context is None


def test_bridge_pair_ids_use_plain_prompt_and_anchor(monkeypatch, tmp_path):
    """Fit pair = PLAIN x prompt + anchor + z+ + eos, loss from len(x)+len(a)."""
    cfg = _cfg()
    fixtures = _fixtures(anchored=True)
    captured = {}

    def fake_fit(policy, pairs, adapters_dir, name, params, cfg_, **kw):
        captured[name] = pairs
        return _fake_fit(policy, pairs, adapters_dir, name, params, cfg_)

    def responder(qid, kind, r):
        if kind == "bridge":
            # anchored q1: bridge generation prompt must be privileged ids + anchor
            if qid == "q1":
                assert r.prompt_token_ids == FakeTok.PROMPT_IDS + [7, 8]
            return [_sample(101)]
        return [_sample(102)]

    _setup(monkeypatch, responder)
    import expert_iter.lora_sft as m
    monkeypatch.setattr(m, "_fit_adapter", fake_fit)
    _propose(BridgeSftOperator(), cfg, tmp_path, fixtures)

    pair = next(p for p in captured["pooled_c0"] if p["qid"] == "q1")
    assert pair["input_ids"] == [1, 2] + [7, 8] + [101, 0]   # plain prompt+anchor+z+eos
    assert pair["prompt_len"] == 4                            # len(x) + len(anchor)


def test_retry_then_skip(monkeypatch, tmp_path):
    cfg = _cfg()
    fixtures = _fixtures()

    def responder(qid, kind, r):
        if kind == "bridge":
            return [_sample(101)] if qid == "q1" else [_sample(102)]
        if kind == "bridge_retry":
            assert qid == "q2"          # only the empty cliff retries
            return [_sample(102)]       # still wrong -> skip list
        return [_sample(102)]

    calls = _setup(monkeypatch, responder)
    _propose(BridgeSftOperator(), cfg, tmp_path, fixtures)

    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_bridge_retried"] == 1
    assert stats["bridge_skipped_qids"] == ["q2"]
    assert stats["bridge_yield"] == 0.5
    # sample_skipped=true (default): q2 still sampled from the pooled adapter
    sampled_qids = {r.rid.split(":")[0]
                    for r in calls["requests"][-1]}  # last launch = candidates
    assert sampled_qids == {"q1", "q2"}


def test_sample_skipped_off_excludes_skip_list(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.bridge.sample_skipped=false")
    fixtures = _fixtures()

    def responder(qid, kind, r):
        if kind in ("bridge", "bridge_retry"):
            return [_sample(101)] if qid == "q1" else [_sample(102)]
        return [_sample(102)]

    calls = _setup(monkeypatch, responder)
    _propose(BridgeSftOperator(), cfg, tmp_path, fixtures)
    sampled_qids = {r.rid.split(":")[0] for r in calls["requests"][-1]}
    assert sampled_qids == {"q1"}


def test_leakage_rules_reject_when_enabled(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.bridge.leakage_rules=true")
    fixtures = _fixtures()

    def responder(qid, kind, r):
        if kind in ("bridge", "bridge_retry"):
            # correct but mentions "the reference solution" -> rules reject
            return [_sample(103)]
        return [_sample(102)]

    _setup(monkeypatch, responder)
    _propose(BridgeSftOperator(), cfg, tmp_path, fixtures)
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_bridge_leak_rule"] >= 1
    assert stats["n_questions_bridged"] == 0    # everything leaked -> all skipped
    assert set(stats["bridge_skipped_qids"]) == {"q1", "q2"}


def test_leakage_rules_off_by_default(monkeypatch, tmp_path):
    cfg = _cfg()  # default: verifier-only acceptance (confirmed)
    fixtures = _fixtures()

    def responder(qid, kind, r):
        if kind == "bridge":
            return [_sample(103)]       # leaky text accepted when rules off
        return [_sample(102)]

    _setup(monkeypatch, responder)
    _propose(BridgeSftOperator(), cfg, tmp_path, fixtures)
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_bridge_leak_rule"] == 0
    assert stats["n_questions_bridged"] == 1    # q1 accepted despite leaky wording


def test_judge_rejects_when_enabled(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.bridge.judge_enabled=true")
    fixtures = _fixtures()

    def responder(qid, kind, r):
        if kind in ("bridge", "bridge_retry"):
            return [_sample(101)] if qid == "q1" else [_sample(102)]
        return [_sample(102)]

    _setup(monkeypatch, responder)

    import expert_iter.filters as filters_mod

    def fake_judge(survivors, cfg_, model_path, it_dir, *, pool_dir=None):
        return [], len(survivors), {"n_flagged": len(survivors)}  # flag everything

    monkeypatch.setattr(filters_mod, "_leakage_judge", fake_judge)
    _propose(BridgeSftOperator(), cfg, tmp_path, fixtures)
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_bridge_leak_judge"] >= 1
    assert stats["n_questions_bridged"] == 0


def test_max_keep_caps_pairs_shortest_first(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.bridge.max_keep=1", "improve.lora_sft.bridge.n=2")
    fixtures = _fixtures()

    def responder(qid, kind, r):
        if kind in ("bridge", "bridge_retry"):
            if qid == "q1":
                long_ok = {"text": FakeTok.vocab[105], "token_ids": [105, 105, 105],
                           "finish_reason": "stop"}
                return [long_ok, _sample(101)]   # two correct; shortest kept
            return [_sample(104)]
        return [_sample(102)]

    captured = {}

    def fake_fit(policy, pairs, adapters_dir, name, params, cfg_, **kw):
        captured[name] = pairs
        return _fake_fit(policy, pairs, adapters_dir, name, params, cfg_)

    _setup(monkeypatch, responder)
    import expert_iter.lora_sft as m
    monkeypatch.setattr(m, "_fit_adapter", fake_fit)
    _propose(BridgeSftOperator(), cfg, tmp_path, fixtures)

    q1_pairs = [p for p in captured["pooled_c0"] if p["qid"] == "q1"]
    assert len(q1_pairs) == 1
    assert q1_pairs[0]["input_ids"] == [1, 2, 101, 0]   # the SHORT correct bridge
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["bridge_pairs_kept"] == 2              # 1 per question


def test_import_has_no_cycle_back_into_improve():
    import subprocess
    import sys as _sys

    proc = subprocess.run(
        [_sys.executable, "-c",
         "import expert_iter.bridge_sft, sys; "
         "assert 'expert_iter.improve' not in sys.modules, 'import cycle back into improve'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
