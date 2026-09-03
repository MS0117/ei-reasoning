"""train.gadv config surface, YAML overlays (Config.load(overlays=...) +
scripts/fork_run.py --overlay), the arm/L5/smoke presets, and the SFTExample
fields the objective adds. CPU, no model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from expert_iter.config import Config, deep_merge
from expert_iter.records import SFTExample
from expert_iter.utils import is_done, mark_done

ROOT = Path(__file__).resolve().parents[1]
GADV_ON = ["train.objective=gadv", "filter.selection.always_score=true"]


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_gadv_defaults_load_and_are_inert_under_sft():
    cfg = Config.load(None)
    g = cfg.train.gadv
    assert cfg.train.objective == "sft"
    assert g.gamma == 1.0 and g.rescue_dose == 1.0 and g.neg_scale == 1.0
    assert g.solved_floor == 0.0 and g.correct_max_per_question == 8 == g.wrong_max_per_question
    assert g.wrong_truncated_max_per_question == 8                    # >= n: byte-identical builder
    assert g.clip.enabled and g.clip.eps_lo == 0.2 == g.clip.eps_hi
    assert g.guard_weight == 1.0 and g.accumulate is False and g.wrong_drop_terminal_eos is False
    assert g.cache_dtype == "float32" and g.prepass_batch_size == 1
    Config.load(None, overrides=GADV_ON)                            # the objective itself loads
    Config.load(None, overrides=["train.objective=gadv", "train.gadv.guard_weight=0"])  # no scores needed


@pytest.mark.parametrize("overrides", [
    "train.objective=bogus",
    "train.objective=gadv",                                          # guard needs always_score/c_score
    "train.objective=gadv|filter.selection.always_score=true|train.sft.cliff.enabled=true",
    "train.objective=gadv|filter.selection.always_score=true|filter.selection.scope=full",
    "train.gadv.gamma=-1",
    "train.gadv.neg_scale=-0.1",
    "train.gadv.rescue_dose=-1",
    "train.gadv.solved_floor=-0.5",
    "train.gadv.guard_weight=-1",
    "train.gadv.correct_max_per_question=0",
    "train.gadv.wrong_max_per_question=0",
    "train.gadv.wrong_truncated_max_per_question=-1",                # 0 is legal (drop them)
    "train.gadv.solved_floor_max_per_question=0",
    "train.gadv.prepass_batch_size=0",
    "train.gadv.clip.eps_lo=0",
    "train.gadv.clip.eps_lo=1.0",
    "train.gadv.clip.eps_hi=0",
    "train.gadv.cache_dtype=bfloat16",
])
def test_gadv_validations_reject(overrides):
    with pytest.raises(ValueError):
        Config.load(None, overrides=overrides.split("|"))


def test_gadv_unknown_key_and_hash():
    with pytest.raises(KeyError):
        Config.load(None, overrides=["train.gadv.bogus=1"])
    with pytest.raises(KeyError):
        Config.load(None, overrides=["train.gadv.clip.bogus=1"])
    assert Config.load(None).hash() != Config.load(None, overrides=["train.gadv.gamma=0.5"]).hash()


def test_cliff_guard_messages_unchanged_after_refactor():
    with pytest.raises(ValueError, match="train.sft.cliff.guard needs the C\\(y\\) scores file"):
        Config.load(None, overrides=["train.sft.cliff.enabled=true"])
    with pytest.raises(ValueError, match="train.sft.cliff.guard requires filter.selection.scope=continuation"):
        Config.load(None, overrides=["train.sft.cliff.enabled=true", "filter.selection.always_score=true",
                                     "filter.selection.scope=full"])


# ---------------------------------------------------------------------------
# overlays
# ---------------------------------------------------------------------------

def test_deep_merge_semantics():
    base = {"a": {"x": 1, "y": [1, 2], "z": {"k": 1}}, "b": 2}
    out = deep_merge(base, {"a": {"y": [9], "z": {"j": 2}}, "c": None})
    assert out == {"a": {"x": 1, "y": [9], "z": {"k": 1, "j": 2}}, "b": 2, "c": None}
    assert out is base                                                # in place


def test_overlay_precedence_and_unknown_keys(tmp_path):
    ovl = tmp_path / "o.yaml"
    ovl.write_text(yaml.safe_dump({"train": {"objective": "gadv", "gadv": {"gamma": 3.0}},
                                   "filter": {"selection": {"always_score": True}}}))
    cfg = Config.load(None, overlays=[ovl])
    assert cfg.train.objective == "gadv" and cfg.train.gadv.gamma == 3.0
    # --override beats the overlay; the overlay beats the file
    cfg = Config.load(None, overlays=[ovl], overrides=["train.gadv.gamma=0"])
    assert cfg.train.gadv.gamma == 0.0
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"train": {"gadv": {"gamma": 7.0, "neg_scale": 0.5}}}))
    cfg = Config.load(base, overlays=[ovl])
    assert cfg.train.gadv.gamma == 3.0 and cfg.train.gadv.neg_scale == 0.5   # sibling key kept
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"train": {"gadv": {"bogus": 1}}}))
    with pytest.raises(KeyError):
        Config.load(None, overlays=[bad])
    notmap = tmp_path / "list.yaml"
    notmap.write_text("- 1\n")
    with pytest.raises(ValueError, match="mapping"):
        Config.load(None, overlays=[notmap])


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_frozen_run(root: Path) -> Path:
    src = root / "L2"
    cfg = Config.load(None)
    src.mkdir()
    cfg.save(src / "config.yaml")
    (src / "questions").mkdir()
    (src / "questions" / "train.jsonl").write_text("{}\n")
    it = src / "iter_0"
    for rel in ["rollout/rollouts.jsonl", "partition/solved.jsonl", "anchors/anchors.jsonl",
                "improve/improved.jsonl", "filtered/kept.jsonl"]:
        p = it / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}\n")
        mark_done(p, count=1, config_hash=cfg.hash())
    return src


COMMON = ["loop.stages=[rollout,partition,anchor,improve,filters,build_dataset,train]",
          "loop.iterations=1", "eval.benchmarks=[]", "filter.selection.always_score=true"]


def test_fork_run_overlay_s3_equals_override_arm(tmp_path):
    fr = _load_script("fork_run")
    src = _fake_frozen_run(tmp_path)
    fr.main(["--src", str(src), "--dst", str(tmp_path / "s3_ov"),
             *sum((["--override", o] for o in COMMON), []),
             "--override", "train.sft.cliff.enabled=true", "--override", "train.sft.cliff.rho=0.3"])
    fr.main(["--src", str(src), "--dst", str(tmp_path / "s3_p"),
             "--overlay", str(ROOT / "configs/methods/arms/s3.yaml"),
             *sum((["--override", o] for o in COMMON), [])])
    a = Config.load(tmp_path / "s3_ov" / "config.yaml")
    b = Config.load(tmp_path / "s3_p" / "config.yaml")
    assert a.hash() == b.hash() and b.train.sft.cliff.enabled and b.train.sft.cliff.rho == 0.3
    for rel in fr.FROZEN_STAGES.values():                             # restamped under the arm hash
        assert is_done(tmp_path / "s3_p" / "iter_0" / rel, config_hash=b.hash()), rel
    with pytest.raises(SystemExit, match="overlay not found"):
        fr.main(["--src", str(src), "--dst", str(tmp_path / "x"), "--overlay", "nope.yaml"])


def test_fork_run_overlay_gadv_arm(tmp_path):
    fr = _load_script("fork_run")
    src = _fake_frozen_run(tmp_path)
    fr.main(["--src", str(src), "--dst", str(tmp_path / "gadv"),
             "--overlay", str(ROOT / "configs/methods/arms/gadv.yaml"),
             *sum((["--override", o] for o in COMMON), [])])
    cfg = Config.load(tmp_path / "gadv" / "config.yaml")
    assert cfg.train.objective == "gadv" and not cfg.train.sft.cliff.enabled
    assert cfg.train.gadv.clip.enabled and cfg.train.gadv.guard_weight == 1.0
    assert cfg.hash() != Config.load(src / "config.yaml").hash()


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

def test_l5_gadv_preset_is_the_s3_preset_with_a_gadv_train_block():
    s3 = Config.load(ROOT / "configs/methods/l5_staged_dpo_s3.yaml")
    g = Config.load(ROOT / "configs/methods/l5_gadv.yaml")
    g.validate()
    assert g.train.objective == "gadv" and g.train.init_from == "last"
    assert not g.train.sft.cliff.enabled and g.train.gadv.accumulate is False
    assert g.filter.selection.always_score is True                    # guard refs
    # everything outside train.* and run.name is shared with the S3 arm
    ds, dg = s3.to_dict(), g.to_dict()
    for k in ds:
        if k in ("train", "run"):
            continue
        assert ds[k] == dg[k], k
    assert (s3.train.sft.lr, s3.train.sft.epochs, s3.train.sft.global_batch_size,
            s3.train.max_seq_len, s3.train.backend, s3.train.sft.region_weights) == (
        g.train.sft.lr, g.train.sft.epochs, g.train.sft.global_batch_size,
        g.train.max_seq_len, g.train.backend, g.train.sft.region_weights)


def test_budget_gadv_arm_is_the_gadv_arm_at_one_epoch_with_caps(tmp_path):
    fr = _load_script("fork_run")
    src = _fake_frozen_run(tmp_path)
    cfgs = {}
    for name in ("gadv", "budget_gadv"):
        fr.main(["--src", str(src), "--dst", str(tmp_path / name),
                 "--overlay", str(ROOT / f"configs/methods/arms/{name}.yaml"),
                 *sum((["--override", o] for o in COMMON), [])])
        cfgs[name] = Config.load(tmp_path / name / "config.yaml")
    full, budget = cfgs["gadv"], cfgs["budget_gadv"]
    budget.validate()
    assert budget.train.sft.epochs == 1.0 and full.train.sft.epochs == 2.0
    assert budget.train.gadv.wrong_max_per_question == 4
    assert budget.train.gadv.wrong_truncated_max_per_question == 1
    # only the budget knobs differ: same objective, clip, guard, gamma, lr, batch
    df, db = full.to_dict()["train"], budget.to_dict()["train"]
    df["sft"].pop("epochs"); db["sft"].pop("epochs")
    for k in ("wrong_max_per_question", "wrong_truncated_max_per_question"):
        df["gadv"].pop(k); db["gadv"].pop(k)
    assert df == db
    assert full.hash() != budget.hash()


def test_smoke_gadv_preset_loads():
    cfg = Config.load(ROOT / "configs/methods/smoke_gadv.yaml")
    cfg.validate()
    assert cfg.train.objective == "gadv" and not cfg.train.sft.cliff.enabled
    assert cfg.filter.selection.method == "c_score"                   # guard refs for the smoke


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def test_sft_example_gadv_fields_and_wrong_source():
    old = SFTExample(uid="u", qid="q", source="solved", input_ids=list(range(6)),
                     prompt_len=3, anchor_len=0, completion_len=3).to_dict()
    for k in ("advantage", "group_kind", "group_size"):
        del old[k]
    back = SFTExample.from_dict(old)
    assert back.advantage == 0.0 and back.group_kind == "" and back.group_size == 0
    wrong = SFTExample(uid="w", qid="q", source="wrong", input_ids=list(range(6)),
                       prompt_len=3, anchor_len=0, completion_len=3, advantage=-0.25,
                       group_kind="frontier", group_size=8)
    wrong.validate()
    bad = SFTExample(uid="w2", qid="q", source="wrong", input_ids=list(range(6)),
                     prompt_len=2, anchor_len=1, completion_len=3)
    with pytest.raises(ValueError, match="anchor_len"):
        bad.validate()
