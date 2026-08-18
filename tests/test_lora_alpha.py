"""make_alpha_variant: the adapter_config lora_alpha scaling trick (lora.py)."""

import json

import pytest

from expert_iter.lora import fmt_alpha, make_alpha_variant


def _fake_adapter(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps(
        {"lora_alpha": 32, "r": 16, "base_model_name_or_path": "org/base"}
    ))
    (d / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
    return d


def test_alpha_one_returns_base_dir(tmp_path):
    d = _fake_adapter(tmp_path)
    assert make_alpha_variant(d, 1.0) == d


def test_variant_scales_config_and_shares_weights(tmp_path):
    d = _fake_adapter(tmp_path)
    v = make_alpha_variant(d, 0.5)
    assert v == d / "alpha" / "0.5"
    cfg = json.loads((v / "adapter_config.json").read_text())
    assert cfg["lora_alpha"] == 16          # effective scale = lora_alpha / r
    assert cfg["r"] == 16
    weights = v / "adapter_model.safetensors"
    assert weights.exists()
    # hardlink (fallback copy also acceptable — content must match)
    assert weights.read_bytes() == (d / "adapter_model.safetensors").read_bytes()


def test_variant_is_idempotent(tmp_path):
    d = _fake_adapter(tmp_path)
    v1 = make_alpha_variant(d, 0.2)
    v2 = make_alpha_variant(d, 0.2)
    assert v1 == v2
    assert json.loads((v1 / "adapter_config.json").read_text())["lora_alpha"] == pytest.approx(6.4)


def test_invalid_alpha_rejected(tmp_path):
    d = _fake_adapter(tmp_path)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            make_alpha_variant(d, bad)


def test_not_an_adapter_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_alpha_variant(tmp_path, 0.5)


def test_fmt_alpha():
    assert fmt_alpha(0.05) == "0.05"
    assert fmt_alpha(1.0) == "1"
    assert fmt_alpha(0.2) == "0.2"
