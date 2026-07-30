import pytest

from expert_iter.config import Config


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
