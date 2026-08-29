import pytest

from expert_iter.config import Config, freeze_run_config


def test_defaults_load():
    cfg = Config.load(None)
    assert cfg.train.backend == "zero2"
    assert cfg.anchor.policy == "fixed_fraction"


def test_yaml_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("rollout:\n  n: 4\ntrain:\n  backend: single\n")
    cfg = Config.load(p, overrides=["rollout.temperature=0.5", "train.sft.lr=2e-5"])
    assert cfg.rollout.n == 4
    assert cfg.train.backend == "single"
    assert cfg.rollout.temperature == 0.5
    assert cfg.train.sft.lr == 2e-5


def test_unknown_key_errors(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("rollout:\n  n_samples: 4\n")  # typo for n
    with pytest.raises(KeyError):
        Config.load(p)


def test_packing_with_region_weights_rejected():
    with pytest.raises(ValueError, match="packing"):
        Config.load(None, overrides=["train.sft.packing=true"])


def test_max_len_budget_validation():
    with pytest.raises(ValueError, match="max_total_tokens"):
        Config.load(None, overrides=["filter.max_total_tokens=99999"])


def test_hash_stable_and_sensitive():
    a = Config.load(None)
    b = Config.load(None)
    assert a.hash() == b.hash()
    c = Config.load(None, overrides=["rollout.n=3"])
    assert a.hash() != c.hash()


@pytest.mark.parametrize("override", [
    "rollout.capture_logprobs=true",
    "improve.rounds=2",
    "improve.teacher.model=org/teacher",
])
def test_unimplemented_options_are_rejected(override):
    with pytest.raises(ValueError, match="not implemented"):
        Config.load(None, overrides=[override])


@pytest.mark.parametrize("override", [
    "run.name=''",
    "model.base=''",
    "engine.gpus=[-1]",
    "loop.iterations=0",
    "train.sft.lr=0",
])
def test_invalid_runtime_values_are_rejected(override):
    with pytest.raises(ValueError):
        Config.load(None, overrides=[override])


def test_new_method_defaults_load():
    cfg = Config.load(None)
    assert cfg.improve.lora_sft.fit.r == 16
    assert cfg.improve.lora_sft.fit.lora_alpha == 32
    assert cfg.improve.lora_sft.fit.steps == 3
    assert cfg.improve.lora_sft.adapter_scope == "pooled"
    assert cfg.improve.lora_sft.refit_budget == 0
    assert cfg.improve.lora_sft.project_back.enabled is False
    assert cfg.improve.lora_sft.project_back.granularity == "per_problem"
    assert cfg.improve.lora_sft.project_back.alphas[-1] == 1.0
    assert cfg.filter.selection.method == "shortest"
    assert cfg.filter.selection.lambda_tail == 1.0
    assert cfg.filter.selection.gamma_dtail == 0.0
    assert cfg.filter.leakage.judge_enabled is False
    assert cfg.partition.cliff_max_correct == 0
    assert cfg.engine.enable_lora is False and cfg.engine.max_logprobs == 20


def test_lora_sft_config_loads_when_consistent():
    cfg = Config.load(None, overrides=[
        "improve.operator=lora_sft", "engine.enable_lora=true",
    ])
    assert cfg.improve.operator == "lora_sft"


def test_unknown_operator_rejected_at_config_load():
    """A typo'd operator must not survive until the improve stage (an hour into
    a real run); RL in particular is a phase, not an operator."""
    with pytest.raises(ValueError, match="is not one of"):
        Config.load(None, overrides=["improve.operator=bogus"])
    with pytest.raises(ValueError, match="RL is not an operator"):
        Config.load(None, overrides=["improve.operator=rl"])
    with pytest.raises(ValueError, match="RL is not an operator"):
        Config.load(None, overrides=["improve.operator=grpo"])


def test_bridge_defaults_load():
    cfg = Config.load(None)
    br = cfg.improve.lora_sft.bridge
    assert br.n == 8 and br.max_keep == 4 and br.retry_temperature == 1.5
    assert br.leakage_rules is False and br.judge_enabled is False   # confirmed defaults
    assert br.sample_skipped is True


@pytest.mark.parametrize("overrides", [
    # lora_sft cross-requirements
    "improve.operator=lora_sft",                                        # needs enable_lora
    "improve.operator=bridge_sft",                                      # same gate
    # bridge bounds
    "improve.lora_sft.bridge.n=0",
    "improve.lora_sft.bridge.max_keep=0",
    "improve.lora_sft.bridge.retry_temperature=-1",
    "improve.lora_sft.bridge.max_tokens=0",
    # adaptive tau_E bounds
    "improve.lora_sft.fit.adaptive.tau_e=0",
    "improve.lora_sft.fit.adaptive.eval_every=0",
    "improve.lora_sft.fit.adaptive.max_steps=100",
    "improve.lora_sft.fit.adaptive.eval_every=11",  # > max_steps (10)
    "improve.lora_sft.fit.adaptive.criterion=bogus",
    "improve.lora_sft.fit.adaptive.m_rollouts=0",
    # rl bounds / gates
    "improve.rl.algo=ppo",                                              # dedicated rejection
    "improve.rl.algo=bogus",
    "improve.rl.kl_beta=0.01",                                          # > 1e-3
    "improve.rl.enabled=true",                                          # needs LoRA operator
    "improve.rl.group_size=1",
    "improve.rl.backend=pool",                                          # reserved fallback
    "improve.rl.vllm_gpu_memory_utilization=0",
    "improve.operator=lora_sft|engine.enable_lora=true|engine.max_lora_rank=8",
    # project-back grid rules
    "improve.lora_sft.project_back.alphas=[0.5]",                       # missing 1.0
    "improve.lora_sft.project_back.alphas=[1.0, 0.5]",                  # not increasing
    "improve.lora_sft.project_back.granularity=bogus",
    "improve.lora_sft.project_back.tau_p=0",
    # fit ranges
    "improve.lora_sft.fit.steps=0",
    "improve.lora_sft.fit.dropout=1.0",
    # selection
    "filter.selection.method=bogus",
    "filter.selection.method=c_score|filter.selection.gamma_dtail=0.5",  # self_resample has no q_P
    "filter.selection.tail_fraction=0",
    # partition threshold
    "partition.cliff_max_correct=8",                                    # >= rollout.n
    "partition.cliff_max_correct=-1",
    # anchor
    "anchor.base_selection=bogus",
    "anchor.policy=privileged_divergence|data.adapter=hf_math",         # include_solution off
    "anchor.policy=privileged_divergence|anchor.params.signal=topk_kl",  # max_logprobs < top_k
    "anchor.policy=privileged_divergence|anchor.params.signal=bogus",
    # leakage
    "filter.gates=[leakage_rules]|filter.leakage.patterns=[]",
    # engine
    "engine.max_logprobs=0",
])
def test_new_validations_reject(overrides):
    with pytest.raises(ValueError):
        Config.load(None, overrides=overrides.split("|"))


def test_unknown_key_under_new_sections_errors():
    with pytest.raises(KeyError):
        Config.load(None, overrides=["improve.lora_sft.bogus=1"])
    with pytest.raises(KeyError):
        Config.load(None, overrides=["filter.selection.threshold=3"])


def test_hash_sensitive_to_new_fields():
    a = Config.load(None)
    b = Config.load(None, overrides=["improve.lora_sft.fit.steps=4"])
    assert a.hash() != b.hash()


def test_frozen_run_config_rejects_mismatched_resume(tmp_path):
    cfg = Config.load(None)
    frozen = freeze_run_config(cfg, tmp_path)
    assert frozen.exists()
    assert freeze_run_config(Config.load(None), tmp_path) == frozen

    changed = Config.load(None, overrides=["rollout.n=3"])
    with pytest.raises(ValueError, match="config mismatch"):
        freeze_run_config(changed, tmp_path)


# ---------------------------------------------------------------------------
# cliff objective (train.sft.cliff) — docs/objective_decision_20260823.md §3
# ---------------------------------------------------------------------------

def test_cliff_defaults_load():
    cfg = Config.load(None)
    cl = cfg.train.sft.cliff
    assert cl.enabled is False
    assert cl.rho == 0.1 and cl.per_question_norm is True and cl.m_per_batch == 1
    assert cl.negative.mode == "off" and cl.negative.mu == 0.1
    assert cl.negative.max_per_question == 8 and cl.negative.delta == 0.02
    assert cl.guard.enabled is True
    assert cfg.train.dpo.rejected_selection == "base_pick"


_CLIFF_ON = "train.sft.cliff.enabled=true|filter.selection.always_score=true"


@pytest.mark.parametrize("overrides", [
    f"{_CLIFF_ON}|train.sft.cliff.rho=0",
    f"{_CLIFF_ON}|train.sft.cliff.rho=1.0",
    f"{_CLIFF_ON}|train.sft.cliff.m_per_batch=0",
    f"{_CLIFF_ON}|train.sft.cliff.m_per_batch=20",          # > global_batch_size // 2
    f"{_CLIFF_ON}|train.sft.cliff.negative.mode=bogus",
    f"{_CLIFF_ON}|train.sft.cliff.negative.mode=v1|train.sft.cliff.negative.mu=-1",
    f"{_CLIFF_ON}|train.sft.cliff.negative.mode=v1|train.sft.cliff.negative.delta=0",
    f"{_CLIFF_ON}|train.sft.cliff.negative.mode=v1|train.objective=sft+dpo",  # double negative
    f"{_CLIFF_ON}|train.sft.cliff.negative.mode=v0",        # v0 needs sft+dpo + modal_wrong
    f"{_CLIFF_ON}|train.sft.cliff.negative.mode=v0|train.objective=sft+dpo",  # still no modal_wrong
    "train.sft.cliff.negative.mode=v1",                     # negative without cliff.enabled
    "train.dpo.rejected_selection=bogus",
    # guard needs the C(y) scores file + matching scope
    "train.sft.cliff.enabled=true",                         # shortest + always_score=false
    f"{_CLIFF_ON}|filter.selection.scope=full",
])
def test_cliff_validations_reject(overrides):
    with pytest.raises(ValueError):
        Config.load(None, overrides=overrides.split("|"))


def test_cliff_valid_arms_load():
    # S3 / S3-tok / S4-v0 / S4-v1 override combos from the preset cookbook
    Config.load(None, overrides=_CLIFF_ON.split("|") + ["train.sft.cliff.rho=0.3"])
    Config.load(None, overrides=_CLIFF_ON.split("|") + ["train.sft.cliff.per_question_norm=false"])
    Config.load(None, overrides=_CLIFF_ON.split("|") + [
        "train.sft.cliff.negative.mode=v0", "train.objective=sft+dpo",
        "train.dpo.rejected_selection=modal_wrong"])
    Config.load(None, overrides=_CLIFF_ON.split("|") + ["train.sft.cliff.negative.mode=v1"])
    # guard off lifts the always_score requirement
    Config.load(None, overrides=[
        "train.sft.cliff.enabled=true", "train.sft.cliff.guard.enabled=false"])


def test_cliff_unknown_key_and_hash():
    with pytest.raises(KeyError):
        Config.load(None, overrides=["train.sft.cliff.bogus=1"])
    with pytest.raises(KeyError):
        Config.load(None, overrides=["train.sft.cliff.negative.bogus=1"])
    assert Config.load(None).hash() != Config.load(
        None, overrides=["train.sft.cliff.rho=0.2"]).hash()


_STAGED_ON = "improve.operator=staged_bridge_sft|engine.enable_lora=true"


@pytest.mark.parametrize("overrides", [
    f"{_STAGED_ON}|improve.lora_sft.staged.stage2_objective=bogus",
    f"{_STAGED_ON}|improve.lora_sft.staged.ul.mu=-0.1",
    f"{_STAGED_ON}|improve.lora_sft.staged.ul.delta=0",
    f"{_STAGED_ON}|improve.lora_sft.staged.ul.delta=1.5",
    f"{_STAGED_ON}|improve.lora_sft.staged.ul.lr=0",
    f"{_STAGED_ON}|improve.lora_sft.staged.ul.negative_selection=bogus",
    f"{_STAGED_ON}|improve.lora_sft.staged.ul.max_pairs_per_question=0",
    f"{_STAGED_ON}|improve.lora_sft.staged.ul.reference=bogus",
    f"{_STAGED_ON}|improve.lora_sft.staged.dpo.negative_selection=bogus",
])
def test_staged_ul_validations_reject(overrides):
    with pytest.raises(ValueError):
        Config.load(None, overrides=overrides.split("|"))


def test_staged_ul_arm_loads():
    cfg = Config.load(None, overrides=[*_STAGED_ON.split("|"),
                                       "improve.lora_sft.staged.stage2_objective=ul"])
    ul = cfg.improve.lora_sft.staged.ul
    assert (ul.mu, ul.delta, ul.guard, ul.negative_selection) == (0.1, 0.02, True, "modal")
    # modal is legal for the dpo arm too (a DPO+modal control needs no code)
    Config.load(None, overrides=[*_STAGED_ON.split("|"),
                                 "improve.lora_sft.staged.stage2_objective=dpo",
                                 "improve.lora_sft.staged.dpo.negative_selection=modal"])


# --- train.dpo memory knobs (added 2026-08-28 after the S4-v0 OOM) -----------

def test_dpo_memory_knobs_default_to_the_settings_that_fit():
    """Defaults must reproduce the configuration the S4-v0 arm actually ran:
    checkpointing on, reference log-probs precomputed, max_length inherited."""
    cfg = Config.load(None)
    assert cfg.train.dpo.gradient_checkpointing is True
    assert cfg.train.dpo.precompute_ref_log_probs is True
    assert cfg.train.dpo.max_length is None


@pytest.mark.parametrize("value", [0, -1, 999_999])
def test_dpo_max_length_must_fit_inside_max_seq_len(tmp_path, value):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("train:\n  max_seq_len: 4096\nfilter:\n  max_total_tokens: 4096\n")
    with pytest.raises(ValueError, match="train.dpo.max_length"):
        Config.load(cfg_path, overrides=[f"train.dpo.max_length={value}"])


def test_dpo_max_length_accepts_a_value_at_or_below_max_seq_len(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("train:\n  max_seq_len: 4096\nfilter:\n  max_total_tokens: 4096\n")
    for v in (1, 2048, 4096):
        assert Config.load(cfg_path, overrides=[f"train.dpo.max_length={v}"]).train.dpo.max_length == v
