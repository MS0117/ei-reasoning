from types import SimpleNamespace

import pytest

from expert_iter.config import Config
from expert_iter.train import run_dpo


def test_pure_dpo_rejects_empty_pairs_before_loading_trainer(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("")
    cfg = Config.load(None)
    args = SimpleNamespace(iteration=0)

    with pytest.raises(RuntimeError, match="empty DPO"):
        run_dpo(cfg, args, pairs, "unused", tmp_path / "ckpt")

    assert run_dpo(
        cfg, args, pairs, "unused", tmp_path / "ckpt", allow_empty=True,
    ) is False
