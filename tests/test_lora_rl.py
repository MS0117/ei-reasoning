"""improve.rl plumbing: dataset assembly + seam counting, rl_key stability,
PlateauCallback stop logic, operator _maybe_rl wiring — no GPU, no trl run."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from expert_iter import lora_rl as ls_rl
from expert_iter import lora_sft as ls_mod
from expert_iter.config import Config, resolve_batch_shape
from expert_iter.lora_rl import PlateauCallback, budget_kwargs, build_rl_rows, rl_key
from expert_iter.lora_sft import LoraSftOperator
from expert_iter.records import AnchorRecord, UnsolvedQuestion


class FakeTok:
    """__call__ returns PROMPT_IDS regardless of text -> an anchored row's
    retokenization can never reproduce prompt_ids + anchor_ids (seam mismatch);
    plain rows match when prompts[qid] == PROMPT_IDS."""

    eos_token_id = 0
    PROMPT_IDS = [9, 9, 9, 9]
    vocab = {7: "step one\n\n", 8: "step two\n\n", 101: "The answer is \\boxed{4}",
             102: "wrong: \\boxed{5}"}

    def apply_chat_template(self, messages, **kw):
        return "P:" + messages[-1]["content"][:6]

    def __call__(self, text, **kw):
        return {"input_ids": list(self.PROMPT_IDS)}

    def decode(self, ids):
        return "".join(self.vocab.get(i, "") for i in ids)


def _fixtures(anchored_qids=()):
    qids = ["q1", "q2"]
    questions = {q: UnsolvedQuestion(qid=q, question="2+2?", final_answer="4")
                 for q in qids}
    anchors = {q: AnchorRecord(qid=q, base_sample_idx=0, policy="p",
                               anchor_token_ids=[7, 8] if q in anchored_qids else [],
                               anchor_text="", anchor_len=0, base_response_len=10)
               for q in qids}
    prompts = {q: list(FakeTok.PROMPT_IDS) for q in qids}
    return qids, questions, anchors, prompts


def test_build_rl_rows_plain_no_mismatch():
    cfg = Config.load(None)
    qids, questions, anchors, prompts = _fixtures()
    rows, n_mismatch = build_rl_rows(
        qids, questions_by_qid=questions, prompts=prompts,
        anchors_by_qid=anchors, tokenizer=FakeTok(), cfg=cfg,
    )
    assert [r["qid"] for r in rows] == ["q1", "q2"]
    assert n_mismatch == 0
    assert all(r["final_answer"] == "4" for r in rows)
    assert "step one" not in rows[0]["prompt"]          # no anchor text


def test_build_rl_rows_counts_anchored_seam_mismatch():
    cfg = Config.load(None)
    qids, questions, anchors, prompts = _fixtures(anchored_qids=("q1",))
    rows, n_mismatch = build_rl_rows(
        qids, questions_by_qid=questions, prompts=prompts,
        anchors_by_qid=anchors, tokenizer=FakeTok(), cfg=cfg,
    )
    assert n_mismatch == 1                              # q1 retokenizes differently
    q1 = next(r for r in rows if r["qid"] == "q1")
    assert q1["prompt"].endswith("step one\n\nstep two\n\n")  # anchor text appended


def test_rl_key_stability_and_sensitivity(tmp_path):
    params = {"algo": "grpo", "steps": 10}
    rows = [{"qid": "q", "prompt": "p", "final_answer": "4"}]
    k1 = rl_key(params, "fp", tmp_path / "a", rows)
    assert k1 == rl_key(params, "fp", tmp_path / "a", rows)
    assert k1 != rl_key(params, "fp", tmp_path / "b", rows)         # input adapter
    assert k1 != rl_key({**params, "steps": 5}, "fp", tmp_path / "a", rows)
    assert k1 != rl_key(params, "fp2", tmp_path / "a", rows)        # model


def _log(cb, step, reward):
    control = SimpleNamespace(should_training_stop=False)
    cb.on_log(None, SimpleNamespace(global_step=step), control, logs={"reward": reward})
    return control.should_training_stop


def test_plateau_callback_stops_after_patience(tmp_path):
    cb = PlateauCallback(patience=2, min_delta=0.01, curve_path=tmp_path / "curve.jsonl")
    assert _log(cb, 1, 0.10) is False       # first -> best
    assert _log(cb, 2, 0.30) is False       # improvement
    assert _log(cb, 3, 0.30) is False       # stale 1
    assert _log(cb, 4, 0.305) is True       # stale 2 (< min_delta) -> stop
    assert cb.stopped_early is True
    curve = [json.loads(l) for l in open(tmp_path / "curve.jsonl")]
    assert [c["reward"] for c in curve] == [0.10, 0.30, 0.30, 0.305]


def test_plateau_callback_stops_at_full_reward(tmp_path):
    cb = PlateauCallback(patience=3, min_delta=0.01, curve_path=tmp_path / "c.jsonl")
    assert _log(cb, 1, 1.0) is True
    assert cb.stopped_early is True


def test_plateau_callback_ignores_rewardless_logs(tmp_path):
    cb = PlateauCallback(patience=1, min_delta=0.0, curve_path=tmp_path / "c.jsonl")
    control = SimpleNamespace(should_training_stop=False)
    cb.on_log(None, SimpleNamespace(global_step=1), control, logs={"loss": 0.5})
    assert control.should_training_stop is False and cb.rewards == []


# ---------------------------------------------------------------------------
# operator wiring (_maybe_rl) with a faked subprocess launcher
# ---------------------------------------------------------------------------

def test_maybe_rl_replaces_adapter_and_records_stats(monkeypatch, tmp_path):
    cfg = Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
        "improve.rl.enabled=true", "improve.rl.epochs=null", "improve.rl.steps=3",
    ])
    qids, questions, anchors, prompts = _fixtures()
    launched = {}

    def fake_launch(policy, adapter_dir, rows, adapters_dir, name, params, cfg_):
        launched.update(adapter=str(adapter_dir), n_rows=len(rows), params=params)
        out = adapters_dir / f"{name}_rl" / "RLKEY"
        out.mkdir(parents=True, exist_ok=True)
        return out, {"algo": params["algo"], "steps_done": 3, "reward_first": 0.1,
                     "reward_last": 0.4, "plateau_stopped": False, "seconds": 1.0}

    monkeypatch.setattr(ls_mod, "_launch_lora_rl", fake_launch)
    op = LoraSftOperator()
    stats: dict = {}
    out_dir, rl_info = op._maybe_rl(
        tmp_path / "fit_adapter", qids, "pooled_c0", cfg=cfg, policy="org/p",
        prompts=prompts, anchors_by_qid=anchors, questions_by_qid=questions,
        tokenizer=FakeTok(), adapters_dir=tmp_path / "adapters", iteration=0,
        stats=stats,
    )
    assert str(out_dir).endswith("pooled_c0_rl/RLKEY")   # phi_E replaces the fit adapter
    assert launched["n_rows"] == 2
    assert launched["params"]["algo"] == "grpo"
    assert launched["params"]["max_completion_length"] == cfg.improve.max_tokens
    assert launched["params"]["verifier"] == "math"
    assert stats["rl"]["pooled_c0"]["reward_last"] == 0.4
    assert rl_info["seam_mismatches"] == 0


def test_maybe_rl_disabled_is_noop(tmp_path):
    cfg = Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
    ])
    qids, questions, anchors, prompts = _fixtures()
    op = LoraSftOperator()
    stats: dict = {}
    out_dir, rl_info = op._maybe_rl(
        tmp_path / "fit_adapter", qids, "c", cfg=cfg, policy="org/p",
        prompts=prompts, anchors_by_qid=anchors, questions_by_qid=questions,
        tokenizer=FakeTok(), adapters_dir=tmp_path / "adapters", iteration=0,
        stats=stats,
    )
    assert out_dir == tmp_path / "fit_adapter" and rl_info is None and stats == {}


@pytest.mark.parametrize("gpus,num_processes,expect_accelerate,expect_cvd", [
    ([0], None, False, "0"),            # single GPU -> plain python subprocess
    ([0, 1], None, True, "0,1"),        # all GPUs -> accelerate DDP (GPU-verified)
    ([0, 1], 1, False, "0"),            # forced single-process
    ([0, 1, 2], 2, True, "0,1"),        # capped at num_processes
])
def test_launch_lora_rl_topology(monkeypatch, tmp_path, gpus, num_processes,
                                 expect_accelerate, expect_cvd):
    overrides = ["improve.operator=lora_sft", "engine.enable_lora=true",
                 "improve.rl.enabled=true", f"engine.gpus={gpus}"]
    if num_processes is not None:
        overrides.append(f"improve.rl.num_processes={num_processes}")
    cfg = Config.load(None, overrides=overrides)
    captured = {}

    class FakeProc:
        returncode = 0

    def fake_run(cmd, env=None, **kw):
        captured["cmd"] = cmd
        captured["env"] = env
        out = Path(cmd[cmd.index("--out") + 1])
        (out / "adapter_config.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text("{}")
        from expert_iter.utils import mark_done, write_json

        write_json(out / "rl_meta.json", {"algo": "grpo", "steps_done": 1})
        mark_done(out / "adapter_config.json", count=1,
                  config_hash=cmd[cmd.index("--cache-key") + 1])
        return FakeProc()

    monkeypatch.setattr(ls_mod.subprocess, "run", fake_run)
    rows = [{"qid": "q1", "prompt": "p", "final_answer": "4"}]
    ls_mod._launch_lora_rl("org/policy", tmp_path / "fit", rows,
                           tmp_path / "adapters", "pooled_c0",
                           {"algo": "grpo", "steps": 3}, cfg)

    is_accelerate = "launch" in captured["cmd"][:2]
    assert is_accelerate is expect_accelerate
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == expect_cvd
    if expect_accelerate:
        n = captured["cmd"][captured["cmd"].index("--num_processes") + 1]
        assert n == str(len(expect_cvd.split(",")))
        assert "-m" in captured["cmd"] and "expert_iter.lora_rl" in captured["cmd"]


def test_maybe_rl_seam_strict_raises(monkeypatch, tmp_path):
    cfg = Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
        "improve.rl.enabled=true", "improve.rl.seam_strict=true",
    ])
    qids, questions, anchors, prompts = _fixtures(anchored_qids=("q1",))
    monkeypatch.setattr(ls_mod, "_launch_lora_rl",
                        lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(RuntimeError, match="seam"):
        LoraSftOperator()._maybe_rl(
            tmp_path / "fit", qids, "c", cfg=cfg, policy="org/p",
            prompts=prompts, anchors_by_qid=anchors, questions_by_qid=questions,
            tokenizer=FakeTok(), adapters_dir=tmp_path / "adapters", iteration=0,
            stats={},
        )


# ---------------------------------------------------------------------------
# training budget: epochs over the question set vs a raw step count
# ---------------------------------------------------------------------------

def test_budget_defaults_to_one_epoch():
    """A fixed step count silently covers only the first slice of a large cliff
    set (10 steps on 2 GPUs = 20 of 107 questions), so epochs is the default."""
    rl = Config.load(None).improve.rl
    assert (rl.epochs, rl.steps) == (1.0, None)


@pytest.mark.parametrize("params,expected", [
    ({"epochs": 1.0, "steps": None}, {"max_steps": -1, "num_train_epochs": 1.0}),
    ({"epochs": 2.5, "steps": None}, {"max_steps": -1, "num_train_epochs": 2.5}),
    # max_steps > 0 wins over num_train_epochs in transformers' Trainer
    ({"epochs": None, "steps": 7}, {"max_steps": 7, "num_train_epochs": 3.0}),
])
def test_budget_kwargs(params, expected):
    assert budget_kwargs(params) == expected


@pytest.mark.parametrize("params", [
    {"epochs": 1.0, "steps": 5},      # both -> ambiguous
    {"epochs": None, "steps": None},  # neither -> no budget
])
def test_budget_kwargs_rejects_ambiguous(params):
    with pytest.raises(ValueError, match="exactly one"):
        budget_kwargs(params)


@pytest.mark.parametrize("overrides,message", [
    (["improve.rl.steps=5"], "exactly one"),                      # epochs still 1.0
    (["improve.rl.epochs=null"], "exactly one"),                  # neither set
    (["improve.rl.epochs=0"], "epochs must be > 0"),
    (["improve.rl.epochs=null", "improve.rl.steps=0"], "steps must be >= 1"),
])
def test_config_rejects_bad_budget(overrides, message):
    with pytest.raises(ValueError, match=message):
        Config.load(None, overrides=["improve.operator=lora_sft",
                                     "engine.enable_lora=true",
                                     "improve.rl.enabled=true", *overrides])


def test_maybe_rl_forwards_epoch_budget(monkeypatch, tmp_path):
    cfg = Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
        "improve.rl.enabled=true", "improve.rl.epochs=2.0",
    ])
    qids, questions, anchors, prompts = _fixtures()
    seen = {}

    def fake_launch(policy, adapter_dir, rows, adapters_dir, name, params, cfg_):
        seen.update(params)
        out = adapters_dir / "rl"
        out.mkdir(parents=True, exist_ok=True)
        return out, {"algo": params["algo"], "steps_planned": 4, "steps_done": 4}

    monkeypatch.setattr(ls_mod, "_launch_lora_rl", fake_launch)
    _, rl_info = LoraSftOperator()._maybe_rl(
        tmp_path / "fit", qids, "c", cfg=cfg, policy="org/p",
        prompts=prompts, anchors_by_qid=anchors, questions_by_qid=questions,
        tokenizer=FakeTok(), adapters_dir=tmp_path / "adapters", iteration=0,
        stats={},
    )
    assert (seen["epochs"], seen["steps"]) == (2.0, None)
    assert budget_kwargs(seen) == {"max_steps": -1, "num_train_epochs": 2.0}
    # what the epoch budget actually resolved to is reported back to stats
    assert rl_info["steps_planned"] == 4


def test_rl_key_separates_epoch_and_step_budgets(tmp_path):
    rows = [{"qid": "q1", "prompt": "p", "final_answer": "4"}]
    base = {"algo": "grpo", "lr": 1e-5}
    k_epoch = rl_key({**base, "epochs": 1.0, "steps": None}, "fp", tmp_path, rows)
    k_steps = rl_key({**base, "epochs": None, "steps": 10}, "fp", tmp_path, rows)
    assert k_epoch != k_steps          # budget change -> fresh adapter, no stale reuse


# ---------------------------------------------------------------------------
# batch shape (grad_accum) and plateau robustness
# ---------------------------------------------------------------------------

def test_grad_accum_default_and_forwarding(monkeypatch, tmp_path):
    """Questions per optimizer step = num_processes x grad_accum; the per-device
    batch stays pinned to ONE group (per_device_train_batch_size == group_size)."""
    assert Config.load(None).improve.rl.grad_accum == 1
    cfg = Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
        "improve.rl.enabled=true", "improve.rl.grad_accum=4",
    ])
    qids, questions, anchors, prompts = _fixtures()
    seen = {}

    def fake_launch(policy, adapter_dir, rows, adapters_dir, name, params, cfg_):
        seen.update(params)
        out = adapters_dir / "rl"
        out.mkdir(parents=True, exist_ok=True)
        return out, {"algo": params["algo"], "steps_done": 1}

    monkeypatch.setattr(ls_mod, "_launch_lora_rl", fake_launch)
    LoraSftOperator()._maybe_rl(
        tmp_path / "fit", qids, "c", cfg=cfg, policy="org/p", prompts=prompts,
        anchors_by_qid=anchors, questions_by_qid=questions, tokenizer=FakeTok(),
        adapters_dir=tmp_path / "adapters", iteration=0, stats={},
    )
    assert seen["grad_accum"] == 4
    assert seen["plateau_enabled"] is False       # off by default
    assert seen["plateau_window"] is None


def test_config_rejects_bad_grad_accum_and_window():
    for ov, msg in [(["improve.rl.grad_accum=0"], "grad_accum must be >= 1"),
                    (["improve.rl.plateau.window=0"], "window must be >= 1")]:
        with pytest.raises(ValueError, match=msg):
            Config.load(None, overrides=ov)


def _run_plateau(cb, rewards, max_steps=53, num_train_epochs=1.0):
    args = SimpleNamespace(max_steps=-1, num_train_epochs=num_train_epochs)
    for step, r in enumerate(rewards, start=1):
        control = SimpleNamespace(should_training_stop=False)
        cb.on_log(args, SimpleNamespace(global_step=step, max_steps=max_steps),
                  control, logs={"reward": r})
        if control.should_training_stop:
            return step
    return None


def test_plateau_window_resolves_to_one_epoch(tmp_path):
    cb = PlateauCallback(3, 0.01, tmp_path / "c.jsonl")
    _run_plateau(cb, [0.1], max_steps=53, num_train_epochs=1.0)
    assert cb.window == 53                       # 53 steps / 1.0 epoch
    cb2 = PlateauCallback(3, 0.01, tmp_path / "c2.jsonl")
    _run_plateau(cb2, [0.1], max_steps=100, num_train_epochs=2.0)
    assert cb2.window == 50


def test_plateau_disabled_records_curve_but_never_stops(tmp_path):
    cb = PlateauCallback(1, 0.01, tmp_path / "c.jsonl", window=1, enabled=False)
    assert _run_plateau(cb, [0.5, 0.1, 0.1, 0.1, 1.0]) is None
    assert cb.stopped_early is False
    assert len(cb.rewards) == 5                  # still logged for inspection
    assert len(open(tmp_path / "c.jsonl").readlines()) == 5


def test_plateau_window_survives_question_noise(tmp_path):
    """Regression guard for the reason plateau defaults to OFF: with window=1 the
    per-step reward is a different question each step, so patience fires almost
    immediately; averaging over an epoch makes it survive the same noise."""
    import random
    import statistics
    rates = [0.0] * 40 + [0.05] * 20 + [0.1] * 15 + [0.2] * 10 + [0.35] * 8 + [0.5] * 5 + [0.75] * 2

    def series(seed, n=53, G=8):
        rng = random.Random(seed)
        return [sum(rng.random() < rng.choice(rates) for _ in range(G)) / G for _ in range(n)]

    def median_stop(window):
        stops = []
        for seed in range(60):
            cb = PlateauCallback(3, 0.01, tmp_path / f"c{window}_{seed}.jsonl", window=window)
            stops.append(_run_plateau(cb, series(seed)) or 53)
        return statistics.median(stops)

    assert median_stop(1) <= 10                  # raw per-step signal: stops early
    assert median_stop(53) == 53                 # one-epoch window: runs the epoch


# ---------------------------------------------------------------------------
# micro_batch_size: activation-memory knob, constrained by trl's whole-group rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("micro,ga,ws,per_device,qps", [
    (None, 1, 1, 8, 1),    # default: one group per device slot
    (None, 1, 2, 8, 2),
    (None, 4, 2, 8, 8),    # grad_accum multiplies questions per step
    (1, 8, 1, 1, 1),       # same update as the default, 1/8 the activations
    (1, 4, 2, 1, 1),
    (16, 1, 2, 16, 4),     # two groups resident per device
])
def test_resolve_batch_shape(micro, ga, ws, per_device, qps):
    shape = resolve_batch_shape(group_size=8, micro_batch_size=micro,
                                grad_accum=ga, world_size=ws)
    assert shape["micro_batch"] == per_device
    assert shape["questions_per_step"] == qps
    assert shape["generation_batch_size"] == per_device * ws * ga


@pytest.mark.parametrize("micro,ga,ws", [(1, 3, 2), (2, 1, 1), (1, 1, 4)])
def test_resolve_batch_shape_rejects_partial_groups(micro, ga, ws):
    """trl generates whole groups: micro x world x grad_accum must be a multiple
    of group_size. The message must name the legal grad_accum values."""
    with pytest.raises(ValueError, match="Legal grad_accum here"):
        resolve_batch_shape(group_size=8, micro_batch_size=micro,
                            grad_accum=ga, world_size=ws)


def test_default_micro_batch_is_legal_for_every_topology():
    """Why micro_batch_size defaults to group_size."""
    for ws in range(1, 9):
        for ga in range(1, 9):
            resolve_batch_shape(group_size=8, micro_batch_size=None,
                                grad_accum=ga, world_size=ws)


def test_config_rejects_illegal_shape_when_processes_pinned():
    with pytest.raises(ValueError, match="Legal grad_accum here"):
        Config.load(None, overrides=["improve.rl.num_processes=2",
                                     "improve.rl.micro_batch_size=1",
                                     "improve.rl.grad_accum=3"])
    # the same shape with a legal grad_accum loads
    cfg = Config.load(None, overrides=["improve.rl.num_processes=2",
                                       "improve.rl.micro_batch_size=1",
                                       "improve.rl.grad_accum=4"])
    assert cfg.improve.rl.micro_batch_size == 1


def test_launch_rejects_illegal_shape_before_spawning(monkeypatch, tmp_path):
    """The shape check must run before any subprocess: an hour into a run is not
    when to discover the batch shape is invalid."""
    cfg = Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
        "improve.rl.enabled=true", "improve.rl.micro_batch_size=1",
        "improve.rl.grad_accum=1",
    ])
    monkeypatch.setattr(ls_mod, "visible_gpus", lambda _: [0, 1])
    monkeypatch.setattr(ls_mod.subprocess, "run",
                        lambda *a, **k: pytest.fail("must not spawn"))
    with pytest.raises(ValueError, match="Legal grad_accum here"):
        ls_mod._launch_lora_rl(
            "org/p", tmp_path / "adapter", [{"qid": "q", "prompt": "p", "final_answer": "4"}],
            tmp_path / "adapters", "c", {"algo": "grpo", "epochs": 1.0, "steps": None}, cfg,
        )


def test_rl_key_separates_batch_topology(tmp_path):
    """micro_batch/world_size change the update, so a rerun under a different
    topology must not silently reuse a cached adapter."""
    rows = [{"qid": "q", "prompt": "p", "final_answer": "4"}]
    base = {"algo": "grpo", "epochs": 1.0, "steps": None, "grad_accum": 1}
    k1 = rl_key({**base, "micro_batch": 8, "world_size": 1}, "fp", tmp_path, rows)
    k2 = rl_key({**base, "micro_batch": 8, "world_size": 2}, "fp", tmp_path, rows)
    k3 = rl_key({**base, "micro_batch": 1, "world_size": 1}, "fp", tmp_path, rows)
    assert len({k1, k2, k3}) == 3


# ---------------------------------------------------------------------------
# wandb reporting for the RL phase
# ---------------------------------------------------------------------------

def test_report_to_disabled_and_empty(monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    assert ls_rl.report_to(None) == []
    assert ls_rl.report_to({"mode": "disabled", "project": "p", "group": "g"}) == []
    assert "WANDB_PROJECT" not in os.environ


def test_report_to_sets_env(monkeypatch):
    for k in ("WANDB_PROJECT", "WANDB_RUN_GROUP", "WANDB_MODE", "WANDB_ENTITY"):
        monkeypatch.delenv(k, raising=False)
    out = ls_rl.report_to({"project": "ei_reasoning", "group": "bridge_toy_cliff",
                           "mode": "offline", "entity": "acme", "name": "n"})
    assert out == ["wandb"]
    assert os.environ["WANDB_PROJECT"] == "ei_reasoning"
    assert os.environ["WANDB_RUN_GROUP"] == "bridge_toy_cliff"
    assert os.environ["WANDB_MODE"] == "offline"
    assert os.environ["WANDB_ENTITY"] == "acme"


def test_maybe_rl_forwards_wandb_params(monkeypatch, tmp_path):
    cfg = Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
        "improve.rl.enabled=true", "run.name=bridge_toy_cliff",
        "run.wandb.mode=offline", "run.wandb.project=ei_reasoning",
    ])
    qids, questions, anchors, prompts = _fixtures()
    seen = {}

    def fake_launch(policy, adapter_dir, rows, adapters_dir, name, params, cfg_):
        seen.update(params)
        out = adapters_dir / "rl"
        out.mkdir(parents=True, exist_ok=True)
        return out, {"algo": params["algo"], "steps_done": 1}

    monkeypatch.setattr(ls_mod, "_launch_lora_rl", fake_launch)
    LoraSftOperator()._maybe_rl(
        tmp_path / "fit", qids, "pooled_c0", cfg=cfg, policy="org/p", prompts=prompts,
        anchors_by_qid=anchors, questions_by_qid=questions, tokenizer=FakeTok(),
        adapters_dir=tmp_path / "adapters", iteration=0, stats={},
    )
    wb = seen["wandb"]
    assert wb["project"] == "ei_reasoning" and wb["mode"] == "offline"
    assert wb["group"] == "bridge_toy_cliff"          # same group as the toy run
    assert wb["name"] == "bridge_toy_cliff/iter0/rl_pooled_c0"
    assert ls_rl.report_to(wb) == ["wandb"]


def test_maybe_rl_wandb_disabled_passes_through(monkeypatch, tmp_path):
    cfg = Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
        "improve.rl.enabled=true", "run.wandb.mode=disabled",
    ])
    qids, questions, anchors, prompts = _fixtures()
    seen = {}
    monkeypatch.setattr(ls_mod, "_launch_lora_rl",
                        lambda *a, **k: (seen.update(a[5]), tmp_path)[1:] and (tmp_path, {}))
    LoraSftOperator()._maybe_rl(
        tmp_path / "fit", qids, "c", cfg=cfg, policy="org/p", prompts=prompts,
        anchors_by_qid=anchors, questions_by_qid=questions, tokenizer=FakeTok(),
        adapters_dir=tmp_path / "adapters", iteration=0, stats={},
    )
    assert seen["wandb"]["mode"] == "disabled"
    assert ls_rl.report_to(seen["wandb"]) == []


def test_plateau_curve_records_zero_std_diagnostic(tmp_path):
    """frac_reward_zero_std explains a flat reward curve on a cliff set, so it
    must survive into reward_curve.jsonl alongside the reward itself."""
    cb = PlateauCallback(3, 0.01, tmp_path / "c.jsonl", window=1, enabled=False)
    args = SimpleNamespace(max_steps=-1, num_train_epochs=1.0)
    control = SimpleNamespace(should_training_stop=False)
    cb.on_log(args, SimpleNamespace(global_step=1, max_steps=10), control,
              logs={"reward": 0.25, "reward_std": 0.43, "frac_reward_zero_std": 0.75,
                    "completions/clipped_ratio": 0.1, "entropy": 1.8, "loss": 0.0})
    row = json.loads(open(tmp_path / "c.jsonl").read())
    assert row == {"step": 1, "reward": 0.25, "reward_std": 0.43,
                   "frac_reward_zero_std": 0.75, "completions/clipped_ratio": 0.1,
                   "entropy": 1.8}          # `loss` is not a curve field


# ---------------------------------------------------------------------------
# vLLM importance-sampling correction (the long-CoT vanishing-gradient fix)
# ---------------------------------------------------------------------------

def test_importance_sampling_defaults_are_token_level():
    """trl's sequence_mask default exponentiates the SUM of per-token logprob
    gaps and underflows to 0 past a few thousand tokens (api_notes 24)."""
    rl = Config.load(None).improve.rl
    assert rl.vllm_importance_sampling_correction is True
    assert rl.vllm_importance_sampling_mode == "token_truncate"
    assert rl.vllm_importance_sampling_cap == 2.0


@pytest.mark.parametrize("overrides,msg", [
    (["improve.rl.vllm_importance_sampling_mode=seq"], "vllm_importance_sampling_mode"),
    (["improve.rl.vllm_importance_sampling_cap=1.0"], "cap must be > 1"),
    (["improve.rl.vllm_importance_sampling_cap=0.5"], "cap must be > 1"),
])
def test_config_rejects_bad_importance_sampling(overrides, msg):
    with pytest.raises(ValueError, match=msg):
        Config.load(None, overrides=overrides)


def test_maybe_rl_forwards_importance_sampling(monkeypatch, tmp_path):
    cfg = Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
        "improve.rl.enabled=true", "improve.rl.vllm_importance_sampling_mode=token_mask",
        "improve.rl.vllm_importance_sampling_cap=2.5",
    ])
    qids, questions, anchors, prompts = _fixtures()
    seen = {}
    monkeypatch.setattr(ls_mod, "_launch_lora_rl",
                        lambda *a, **k: (seen.update(a[5]), (tmp_path, {"algo": "grpo"}))[1])
    LoraSftOperator()._maybe_rl(
        tmp_path / "fit", qids, "c", cfg=cfg, policy="org/p", prompts=prompts,
        anchors_by_qid=anchors, questions_by_qid=questions, tokenizer=FakeTok(),
        adapters_dir=tmp_path / "adapters", iteration=0, stats={},
    )
    assert seen["vllm_importance_sampling_mode"] == "token_mask"
    assert seen["vllm_importance_sampling_cap"] == 2.5
    assert seen["vllm_importance_sampling_correction"] is True


def test_rl_lr_default_is_1e6():
    """Both long-CoT math-RL studies at this completion length used 1e-6."""
    assert Config.load(None).improve.rl.lr == 1.0e-6


# ---------------------------------------------------------------------------
# prompt filter (DAPO-style zero-variance filtering)
# ---------------------------------------------------------------------------

def _filter_cfg(*extra):
    return Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
        "improve.rl.enabled=true", *extra,
    ])


def _run_maybe_rl(cfg, monkeypatch, tmp_path, counts, qids=None, questions=None,
                  anchors=None, prompts=None):
    """Drive _maybe_rl with a scripted probe (qid -> n_correct)."""
    if qids is None:
        qids, questions, anchors, prompts = _fixtures()
    captured = {}
    monkeypatch.setattr(
        LoraSftOperator, "_probe_correct_counts",
        lambda self, qs, *a, **k: {q: counts[q] for q in qs},
    )

    def fake_launch(policy, adapter_dir, rows, adapters_dir, name, params, cfg_):
        captured["rows"] = rows
        out = adapters_dir / "rl"
        out.mkdir(parents=True, exist_ok=True)
        return out, {"algo": params["algo"], "steps_done": 1}

    monkeypatch.setattr(ls_mod, "_launch_lora_rl", fake_launch)
    stats: dict = {}
    out_dir, rl_info = LoraSftOperator()._maybe_rl(
        tmp_path / "fit", qids, "pooled_c0", cfg=cfg, policy="org/p",
        prompts=prompts, anchors_by_qid=anchors, questions_by_qid=questions,
        tokenizer=FakeTok(), adapters_dir=tmp_path / "adapters", iteration=0,
        stats=stats, pool_base=tmp_path / "pool", grader=object(),
    )
    return out_dir, rl_info, stats, captured


def test_prompt_filter_drops_uniform_groups(monkeypatch, tmp_path):
    """q1 mixed (2 of 8) is kept; q2 all-wrong contributes zero advantage."""
    cfg = _filter_cfg("improve.rl.prompt_filter.enabled=true")
    _, rl_info, stats, captured = _run_maybe_rl(
        cfg, monkeypatch, tmp_path, {"q1": 2, "q2": 0})
    assert [r["qid"] for r in captured["rows"]] == ["q1"]
    assert stats["n_rl_prompts_probed"] == 2
    assert stats["n_rl_prompts_kept"] == 1
    assert stats["n_rl_prompts_all_wrong"] == 1
    assert stats["n_rl_prompts_all_right"] == 0
    assert stats["rl_prompt_filter_yield"] == 0.5
    assert rl_info["n_prompts"] == 1


def test_prompt_filter_drops_all_right_groups(monkeypatch, tmp_path):
    """A group the adapter already solves every time has advantage 0 too."""
    cfg = _filter_cfg("improve.rl.prompt_filter.enabled=true")
    _, _, stats, captured = _run_maybe_rl(
        cfg, monkeypatch, tmp_path, {"q1": 8, "q2": 3})
    assert [r["qid"] for r in captured["rows"]] == ["q2"]
    assert stats["n_rl_prompts_all_right"] == 1


def test_prompt_filter_pass_rate_band(monkeypatch, tmp_path):
    """(0.2, 0.8) is the literature's band heuristic: 1/8 and 7/8 fall outside."""
    cfg = _filter_cfg("improve.rl.prompt_filter.enabled=true",
                      "improve.rl.prompt_filter.min_pass_rate=0.2",
                      "improve.rl.prompt_filter.max_pass_rate=0.8")
    _, _, stats, captured = _run_maybe_rl(
        cfg, monkeypatch, tmp_path, {"q1": 1, "q2": 4})     # 0.125 out, 0.5 in
    assert [r["qid"] for r in captured["rows"]] == ["q2"]
    assert (stats["n_rl_prompts_all_wrong"], stats["n_rl_prompts_all_right"]) == (1, 0)


def test_prompt_filter_empty_skips_rl_entirely(monkeypatch, tmp_path):
    """Never hand trl a zero-row dataset — return the fit adapter untouched."""
    cfg = _filter_cfg("improve.rl.prompt_filter.enabled=true")
    monkeypatch.setattr(ls_mod, "_launch_lora_rl",
                        lambda *a, **k: pytest.fail("must not launch"))
    monkeypatch.setattr(LoraSftOperator, "_probe_correct_counts",
                        lambda self, qs, *a, **k: {q: 0 for q in qs})
    qids, questions, anchors, prompts = _fixtures()
    stats: dict = {}
    out_dir, rl_info = LoraSftOperator()._maybe_rl(
        tmp_path / "fit", qids, "c", cfg=cfg, policy="org/p", prompts=prompts,
        anchors_by_qid=anchors, questions_by_qid=questions, tokenizer=FakeTok(),
        adapters_dir=tmp_path / "adapters", iteration=0, stats=stats,
        pool_base=tmp_path / "pool", grader=object(),
    )
    assert out_dir == tmp_path / "fit" and rl_info is None
    assert stats["rl"]["c"] == {"skipped": "prompt_filter_empty"}


def test_prompt_filter_disabled_launches_no_probe(monkeypatch, tmp_path):
    """Default OFF must cost nothing — not even one probe pool."""
    cfg = _filter_cfg()                       # prompt_filter.enabled defaults false
    monkeypatch.setattr(LoraSftOperator, "_probe_correct_counts",
                        lambda self, *a, **k: pytest.fail("must not probe"))
    _, _, stats, captured = _run_maybe_rl(cfg, monkeypatch, tmp_path, {})
    assert [r["qid"] for r in captured["rows"]] == ["q1", "q2"]   # unchanged
    assert "n_rl_prompts_probed" not in stats


def test_prompt_filter_changes_rl_key(tmp_path):
    """A filtered prompt set must not reuse the unfiltered run's adapter."""
    all_rows = [{"qid": "q1", "prompt": "p1", "final_answer": "4"},
                {"qid": "q2", "prompt": "p2", "final_answer": "5"}]
    params = {"algo": "grpo", "epochs": 1.0, "steps": None}
    assert rl_key(params, "fp", tmp_path, all_rows) != \
        rl_key(params, "fp", tmp_path, all_rows[:1])


# ---------------------------------------------------------------------------
# RL reward must grade the same text the pipeline scores (anchor + completion)
# ---------------------------------------------------------------------------

def test_rl_rows_carry_the_anchor_when_anchored():
    cfg = Config.load(None)
    qids, questions, anchors, prompts = _fixtures(anchored_qids=("q1",))
    rows, _ = build_rl_rows(qids, questions_by_qid=questions, prompts=prompts,
                            anchors_by_qid=anchors, tokenizer=FakeTok(), cfg=cfg)
    by = {r["qid"]: r for r in rows}
    assert by["q1"]["anchor"] == "step one\n\nstep two\n\n"
    assert by["q2"]["anchor"] == ""        # schema stays consistent across rows


def test_rl_rows_drop_the_anchor_field_when_unanchored():
    """anchor.policy=none must leave rows — and therefore rl_key — byte-identical
    to the pre-anchor runs, so cached adapters stay valid."""
    cfg = Config.load(None)
    qids, questions, anchors, prompts = _fixtures()
    rows, _ = build_rl_rows(qids, questions_by_qid=questions, prompts=prompts,
                            anchors_by_qid=anchors, tokenizer=FakeTok(), cfg=cfg)
    assert all("anchor" not in r for r in rows)
    assert rl_key({"algo": "grpo"}, "fp", Path("/a"), rows) == \
        rl_key({"algo": "grpo"}, "fp", Path("/a"),
               [{"qid": r["qid"], "prompt": r["prompt"],
                 "final_answer": r["final_answer"]} for r in rows])


def test_reward_curve_written_by_rank_zero_only(tmp_path):
    """trl gathers rewards before logging, so both ranks see the same value —
    writing from every rank doubles every row (observed on a 2-GPU run)."""
    cb = PlateauCallback(3, 0.01, tmp_path / "c.jsonl", window=1, enabled=False)
    args = SimpleNamespace(max_steps=-1, num_train_epochs=1.0)
    for rank_zero in (True, False):
        state = SimpleNamespace(global_step=1, max_steps=10,
                                is_world_process_zero=rank_zero)
        cb.on_log(args, state, SimpleNamespace(should_training_stop=False),
                  logs={"reward": 0.25})
    assert len(open(tmp_path / "c.jsonl").readlines()) == 1   # not 2
    assert len(cb.rewards) == 2      # in-memory history is per-rank, unaffected
