import os
import sys
from types import SimpleNamespace

from expert_iter.config import Config
from expert_iter.wandb_utils import init_wandb


def test_offline_fallback_propagates_to_trainer_subprocess(tmp_path, monkeypatch):
    seen = {}
    fake_wandb = SimpleNamespace(
        api=SimpleNamespace(api_key=None),
        util=SimpleNamespace(generate_id=lambda: "run-id"),
        init=lambda **kwargs: seen.update(kwargs) or SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    cfg = Config.load(None, overrides=["run.wandb.mode=online"])

    init_wandb(
        cfg, name="test", id_file=tmp_path / "wandb_id.txt",
        job_type="loop",
    )

    assert seen["mode"] == "offline"
    assert os.environ["WANDB_MODE"] == "offline"
