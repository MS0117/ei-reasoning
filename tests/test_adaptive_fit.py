"""Adaptive tau_E fit termination: round chaining, early stop, hard cap —
monkeypatched fit + probe pool (no GPU)."""

import json

from expert_iter import lora_sft as mod
from expert_iter.config import Config
from expert_iter.engine import GenResult
from expert_iter.lora_sft import LoraSftOperator
from expert_iter.records import AnchorRecord, UnsolvedQuestion


class FakeTok:
    eos_token_id = 0
    vocab = {101: "The answer is \\boxed{4}", 102: "wrong: \\boxed{5}"}

    def __call__(self, text, **kw):
        return {"input_ids": [1, 2, 3]}

    def decode(self, ids):
        return " ".join(self.vocab.get(i, "") for i in ids)


def _ctx(cfg, tmp_path, qids=("q1", "q2")):
    questions = {q: UnsolvedQuestion(qid=q, question="2+2?", final_answer="4")
                 for q in qids}
    anchors = {q: AnchorRecord(qid=q, base_sample_idx=0, policy="none",
                               anchor_token_ids=[], anchor_text="", anchor_len=0,
                               base_response_len=10) for q in qids}
    return {
        "chunk_qids": list(qids),
        "prompts": {q: [1, 2] for q in qids},
        "anchors_by_qid": anchors,
        "questions_by_qid": questions,
        "tokenizer": FakeTok(),
        "grader": __import__("expert_iter.registry", fromlist=["build"]).build(
            __import__("expert_iter.registry", fromlist=["VERIFIERS"]).VERIFIERS, "math"),
        "pool_base": tmp_path / "pool",
        "iteration": 0,
        "stats": {},
    }


def _setup(monkeypatch, round_solutions):
    """round_solutions[r-1] = set of qids whose probe samples are correct in
    round r. Fake fit records (name, steps, init_adapter)."""
    fits = []

    def fake_fit(policy, pairs, adapters_dir, name, params, cfg, *,
                 init_adapter=None, wandb=None):
        d = adapters_dir / name / "FAKEKEY"
        fits.append({"name": name, "steps": params["steps"],
                     "init": str(init_adapter) if init_adapter else None})
        return d, 1.0

    state = {"round": 0}

    def fake_run_pool(requests, **kw):
        state["round"] += 1
        solved = round_solutions[min(state["round"], len(round_solutions)) - 1]
        out = []
        for r in requests:
            qid = r.rid.split(":")[0]
            tok = 101 if qid in solved else 102
            out.append(GenResult(rid=r.rid, samples=[
                {"text": FakeTok.vocab[tok], "token_ids": [tok], "finish_reason": "stop"}
                for _ in range(r.n)
            ]))
        return out

    monkeypatch.setattr(mod, "_fit_adapter", fake_fit)
    monkeypatch.setattr(mod, "run_pool", fake_run_pool)
    return fits


def _cfg(*extra):
    return Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
        "improve.lora_sft.fit.adaptive.enabled=true",
        "improve.lora_sft.fit.adaptive.eval_every=2",
        "improve.lora_sft.fit.adaptive.max_steps=6",
        "improve.lora_sft.fit.adaptive.m_rollouts=2",
        "improve.lora_sft.fit.adaptive.tau_e=0.5", *extra,
    ])


def test_early_stop_when_criterion_clears(monkeypatch, tmp_path):
    cfg = _cfg()
    # round 1: q1 solved -> frac_solved = 0.5 >= tau_e -> stop after 1 round
    fits = _setup(monkeypatch, [{"q1"}])
    op = LoraSftOperator()
    ctx = _ctx(cfg, tmp_path)
    adapter, secs = op._fit("org/p", [{"qid": "q1", "input_ids": [1], "prompt_len": 0}],
                            tmp_path / "adapters", "pooled_c0",
                            {"r": 16, "lora_alpha": 32, "steps": 3}, cfg, probe_ctx=ctx)
    assert len(fits) == 1 and fits[0]["name"] == "pooled_c0_r1"
    assert fits[0]["steps"] == 2 and fits[0]["init"] is None
    meta = ctx["stats"]["adaptive"]["pooled_c0"]
    assert meta["rounds_used"] == 1 and meta["steps_used"] == 2
    assert meta["stopped_early"] is True
    assert ctx["stats"]["adaptive_rounds_total"] == 1


def test_hard_cap_with_round_chaining(monkeypatch, tmp_path):
    cfg = _cfg()
    fits = _setup(monkeypatch, [set(), set(), set()])   # never solves
    op = LoraSftOperator()
    ctx = _ctx(cfg, tmp_path)
    adapter, _ = op._fit("org/p", [{"qid": "q1", "input_ids": [1], "prompt_len": 0}],
                         tmp_path / "adapters", "pooled_c0",
                         {"r": 16, "lora_alpha": 32, "steps": 3}, cfg, probe_ctx=ctx)
    # 6 max_steps / 2 eval_every = 3 rounds, chained via init_adapter
    assert [f["name"] for f in fits] == ["pooled_c0_r1", "pooled_c0_r2", "pooled_c0_r3"]
    assert fits[0]["init"] is None
    assert fits[1]["init"].endswith("pooled_c0_r1/FAKEKEY")
    assert fits[2]["init"].endswith("pooled_c0_r2/FAKEKEY")
    meta = ctx["stats"]["adaptive"]["pooled_c0"]
    assert meta["rounds_used"] == 3 and meta["steps_used"] == 6
    assert meta["stopped_early"] is False
    assert str(adapter).endswith("pooled_c0_r3/FAKEKEY")


def test_mean_p_hat_criterion(monkeypatch, tmp_path):
    cfg = _cfg("improve.lora_sft.fit.adaptive.criterion=mean_p_hat",
               "improve.lora_sft.fit.adaptive.tau_e=0.4")
    # q1 fully solved (p=1.0), q2 not (0.0) -> mean 0.5 >= 0.4 -> stop round 1
    _setup(monkeypatch, [{"q1"}])
    op = LoraSftOperator()
    ctx = _ctx(cfg, tmp_path)
    op._fit("org/p", [{"qid": "q1", "input_ids": [1], "prompt_len": 0}],
            tmp_path / "adapters", "c", {"r": 16, "lora_alpha": 32, "steps": 3},
            cfg, probe_ctx=ctx)
    meta = ctx["stats"]["adaptive"]["c"]
    assert meta["stopped_early"] is True
    assert meta["rounds"][0]["criterion"] == 0.5


def test_adaptive_off_uses_plain_fit(monkeypatch, tmp_path):
    cfg = Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
    ])
    fits = _setup(monkeypatch, [set()])
    op = LoraSftOperator()
    ctx = _ctx(cfg, tmp_path)
    op._fit("org/p", [{"qid": "q1", "input_ids": [1], "prompt_len": 0}],
            tmp_path / "adapters", "c", {"r": 16, "lora_alpha": 32, "steps": 3},
            cfg, probe_ctx=ctx)
    assert len(fits) == 1 and fits[0]["name"] == "c"    # no _r1 suffix, no probe
    assert fits[0]["steps"] == 3 and "adaptive" not in ctx["stats"]


def test_propose_wires_probe_ctx(monkeypatch, tmp_path):
    """End-to-end: adaptive stats land in improve/stats.json via propose."""
    import transformers

    cfg = _cfg("improve.n=2")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: FakeTok())
    _setup(monkeypatch, [{"q1", "q2"}])   # solves immediately -> 1 round
    questions = [UnsolvedQuestion(qid=q, question="2+2?", final_answer="4")
                 for q in ("q1", "q2")]
    anchors = [AnchorRecord(qid=q.qid, base_sample_idx=0, policy="none",
                            anchor_token_ids=[], anchor_text="", anchor_len=0,
                            base_response_len=10) for q in questions]
    LoraSftOperator().propose(
        questions, anchors, {"q1": [1, 2], "q2": [1, 2]}, cfg,
        model_paths={"policy": "org/p"}, work_dir=tmp_path / "pool", iteration=0,
        gold_solutions={"q1": "4.", "q2": "4."},
    )
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["adaptive"]["pooled_c0"]["stopped_early"] is True
    assert stats["adaptive_rounds_total"] == 1


# ---------------------------------------------------------------------------
# _probe_correct_counts — the probe body shared with the RL prompt filter
# ---------------------------------------------------------------------------

def _probe(op, cfg, ctx, tmp_path, m, tag="probe"):
    return op._probe_correct_counts(
        ctx["chunk_qids"], tmp_path / "adapter", m, cfg=cfg, policy="org/p",
        prompts=ctx["prompts"], anchors_by_qid=ctx["anchors_by_qid"],
        questions_by_qid=ctx["questions_by_qid"], tokenizer=ctx["tokenizer"],
        grader=ctx["grader"], pool_dir=tmp_path / "pool" / tag,
        iteration=0, tag=tag,
    )


def test_probe_correct_counts(monkeypatch, tmp_path):
    _setup(monkeypatch, [{"q1"}])          # q1 solves, q2 does not
    cfg = _cfg()
    ctx = _ctx(cfg, tmp_path)
    assert _probe(LoraSftOperator(), cfg, ctx, tmp_path, 4) == {"q1": 4, "q2": 0}


def test_probe_excludes_truncated_but_keeps_m_as_denominator(monkeypatch, tmp_path):
    """A truncated trajectory has no answer, so it cannot be correct — but it
    still consumed one of the m draws, so it must count as a failure."""
    def fake_run_pool(requests, **kw):
        out = []
        for r in requests:
            # half correct-but-truncated, half correct-and-finished
            samples = [{"text": "", "token_ids": [101], "finish_reason": "length"}
                       for _ in range(r.n // 2)]
            samples += [{"text": "", "token_ids": [101], "finish_reason": "stop"}
                        for _ in range(r.n - r.n // 2)]
            out.append(GenResult(rid=r.rid, samples=samples))
        return out

    monkeypatch.setattr(mod, "run_pool", fake_run_pool)
    cfg = _cfg()
    ctx = _ctx(cfg, tmp_path)
    counts = _probe(LoraSftOperator(), cfg, ctx, tmp_path, 8)
    assert counts == {"q1": 4, "q2": 4}    # 4 of 8, not 4 of 4


def test_probe_uses_its_own_pool_dir_and_seed_tag(monkeypatch, tmp_path):
    """Request ids repeat across launches, so two probes MUST differ in
    pool_dir (shard files) and in tag (per-request seeds)."""
    seen = []

    def fake_run_pool(requests, **kw):
        seen.append({"work_dir": str(kw["work_dir"]),
                     "seeds": sorted(r.seed for r in requests)})
        return [GenResult(rid=r.rid, samples=[]) for r in requests]

    monkeypatch.setattr(mod, "run_pool", fake_run_pool)
    cfg = _cfg()
    ctx = _ctx(cfg, tmp_path)
    op = LoraSftOperator()
    _probe(op, cfg, ctx, tmp_path, 4, tag="adaptive_r1")
    _probe(op, cfg, ctx, tmp_path, 4, tag="rlfilter_pooled_c0")
    assert seen[0]["work_dir"] != seen[1]["work_dir"]
    assert seen[0]["seeds"] != seen[1]["seeds"]
