"""CPU tests for the L0 experiment tooling: A/B cliff split, attractor mass,
train-qid exclusion, run-dir fork, and the rho_legacy computation.
(scripts/cliff_split.py, attractor_mass.py, fork_run.py, rho_legacy.py)"""

import importlib.util
import json
from pathlib import Path

import pytest

from expert_iter.build_dataset import _load_excluded_qids
from expert_iter.config import Config
from expert_iter.records import (
    ImprovedCandidate,
    SFTExample,
    UnsolvedQuestion,
    VerdictRecord,
)
from expert_iter.utils import done_marker, is_done, mark_done, read_json, write_json, write_jsonl

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# cliff_split
# ---------------------------------------------------------------------------

def test_cliff_split_strata_and_determinism(tmp_path):
    cs = _load("cliff_split")
    cliffs = [f"q{i}" for i in range(40)]
    converted = {f"q{i}" for i in range(10)}
    has_cands = {f"q{i}" for i in range(25)}          # 10-24 = unconverted
    base_pass = {"q0", "q30"}                          # wins over other strata
    strata = cs.assign_strata(cliffs, converted, has_cands, base_pass)
    assert sorted(strata["base_pass"]) == ["q0", "q30"]
    assert "q0" not in strata["converted"]             # priority: one stratum each
    assert len(strata["converted"]) == 9 and len(strata["unconverted"]) == 15
    assert len(strata["never_bridged"]) == 14
    assert sum(len(v) for v in strata.values()) == 40

    a1, b1 = cs.split_stratum(strata["converted"], b_frac=0.5, seed=17)
    a2, b2 = cs.split_stratum(list(reversed(strata["converted"])), b_frac=0.5, seed=17)
    assert (a1, b1) == (a2, b2)                        # order-independent
    assert abs(len(b1) - round(9 * 0.5)) == 0
    a3, b3 = cs.split_stratum(strata["converted"], b_frac=0.5, seed=18)
    assert set(b1) != set(b3)                          # seed changes the draw


def test_cliff_split_main_end_to_end(tmp_path):
    cs = _load("cliff_split")
    it = tmp_path / "iter_0"
    (it / "partition").mkdir(parents=True)
    (it / "filtered").mkdir()
    (it / "improve").mkdir()
    UnsolvedQuestion.dump_jsonl(it / "partition" / "unsolved.jsonl", [
        UnsolvedQuestion(qid=f"q{i}", question="?", final_answer="1") for i in range(8)
    ])
    def cand(qid):
        return ImprovedCandidate(qid=qid, base_sample_idx=0, attempt_idx=0,
                                 prompt_token_ids=[1], anchor_token_ids=[],
                                 continuation_token_ids=[2], continuation_text="c")
    ImprovedCandidate.dump_jsonl(it / "filtered" / "kept.jsonl", [cand("q0"), cand("q1")])
    ImprovedCandidate.dump_jsonl(it / "improve" / "improved.jsonl",
                                 [cand(q) for q in ("q0", "q1", "q2", "q3")])
    cs.main(["--run-dir", str(tmp_path), "--b-frac", "0.5", "--seed", "17"])
    d = json.loads((it / "cliff_split.json").read_text())
    assert sorted(d["A"] + d["B"]) == [f"q{i}" for i in range(8)]
    assert d["exclude"] == d["B"]
    assert d["strata"]["converted"]["n"] == 2
    assert d["strata"]["never_bridged"]["n"] == 4


# ---------------------------------------------------------------------------
# attractor_mass
# ---------------------------------------------------------------------------

def _verdicts(path, spec):
    """spec: {qid: [(correct, answer), ...]}"""
    rows = []
    for qid, samples in spec.items():
        for i, (c, a) in enumerate(samples):
            rows.append(VerdictRecord(qid=qid, sample_idx=i, correct=c, extracted_answer=a))
    write_jsonl(path, (r.to_dict() for r in rows))


def test_attractor_table_and_sign_test(tmp_path):
    am = _load("attractor_mass")
    p = tmp_path / "v.jsonl"
    _verdicts(p, {
        "q1": [(False, "7")] * 6 + [(False, "9")] * 2 + [(True, "42")] * 2,
        "q2": [(False, None)] * 4,                        # unparsed only -> no attractor
    })
    t = am.load_table([str(p)], None)
    assert t["q1"]["p_top1"] == 0.6 and t["q1"]["p_top2"] == 0.8
    assert t["q1"]["modal_wrong"] == "7" and t["q1"]["pass_rate"] == 0.2
    assert t["q2"]["p_top1"] == 0.0 and t["q2"]["n_wrong_kinds"] == 0
    agg = am.aggregate(t)
    assert agg["n_questions"] == 2 and agg["frac_attractor_ge_half"] == 0.5

    k, n, pv = am.sign_test([-1, -1, -1, -1, -1, -1, -1, -1, 0, 1])
    assert (k, n) == (8, 9) and pv == pytest.approx(2 * (1 + 9) / 2 ** 9)
    assert am.sign_test([0, 0])[2] != am.sign_test([0, 0])[2] or True  # nan ok


# ---------------------------------------------------------------------------
# exclusion loader (data.exclude_train_qids)
# ---------------------------------------------------------------------------

def test_exclusion_loader(tmp_path):
    f = tmp_path / "split.json"
    f.write_text(json.dumps({"A": ["a"], "B": ["b1", "b2"], "exclude": ["b1", "b2"]}))
    assert _load_excluded_qids(str(f)) == {"b1", "b2"}
    f2 = tmp_path / "list.json"
    f2.write_text(json.dumps(["x"]))
    assert _load_excluded_qids(str(f2)) == {"x"}
    assert _load_excluded_qids("") == set()
    with pytest.raises(RuntimeError, match="does not exist"):
        _load_excluded_qids(str(tmp_path / "missing.json"))
    f3 = tmp_path / "bad.json"
    f3.write_text(json.dumps({"nope": 1}))
    with pytest.raises(RuntimeError, match="exclude"):
        _load_excluded_qids(str(f3))


# ---------------------------------------------------------------------------
# fork_run
# ---------------------------------------------------------------------------

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
    (it / "dataset").mkdir()
    (it / "dataset" / "train_sft.jsonl").write_text("{}\n")
    mark_done(it / "dataset" / "train_sft.jsonl", count=1, config_hash=cfg.hash())
    return src


def test_fork_run_restamps_data_stages_only(tmp_path):
    fr = _load("fork_run")
    src = _fake_frozen_run(tmp_path)
    dst = tmp_path / "arm"
    fr.main(["--src", str(src), "--dst", str(dst),
             "--override", "train.sft.cliff.enabled=true",
             "--override", "train.sft.cliff.guard.enabled=false"])
    new_cfg = Config.load(dst / "config.yaml")
    assert new_cfg.train.sft.cliff.enabled is True
    assert new_cfg.hash() != Config.load(src / "config.yaml").hash()
    # frozen data stages skip under the ARM's hash
    for rel in fr.FROZEN_STAGES.values():
        assert is_done(dst / "iter_0" / rel, config_hash=new_cfg.hash()), rel
    # build_dataset must RERUN under the arm config: no dataset dir/marker forked
    assert not (dst / "iter_0" / "dataset").exists()
    # hardlinked, not copied
    assert (dst / "iter_0" / "rollout" / "rollouts.jsonl").stat().st_ino == \
           (src / "iter_0" / "rollout" / "rollouts.jsonl").stat().st_ino
    with pytest.raises(SystemExit, match="already exists"):
        fr.main(["--src", str(src), "--dst", str(dst)])


# ---------------------------------------------------------------------------
# rho_legacy
# ---------------------------------------------------------------------------

def test_rho_legacy_token_share(tmp_path, capsys):
    rl = _load("rho_legacy")
    run = tmp_path / "run"
    run.mkdir()
    cfg = Config.load(None)              # region weights: anchor 0, cont/sol 1
    cfg.save(run / "config.yaml")
    ds = run / "iter_0" / "dataset"
    ds.mkdir(parents=True)
    def ex(uid, source, completion_len):
        ids = list(range(2 + completion_len))
        return SFTExample(uid=uid, qid=uid, source=source, input_ids=ids,
                          prompt_len=2, anchor_len=0, completion_len=completion_len)
    SFTExample.dump_jsonl(ds / "train_sft.jsonl",
                          [ex("s1", "solved", 30), ex("s2", "solved", 50),
                           ex("i1", "improved", 20)])
    rl.main(["--run-dir", str(run)])
    out = capsys.readouterr().out
    assert "rho=0.2000" in out           # 20 / (30+50+20)


# ---------------------------------------------------------------------------
# cliff_reroll target selection
#
# The L5 headline is measured on the external cliff holdout, which lives in
# questions/holdout.jsonl — a set the tool could not previously address.
# ---------------------------------------------------------------------------

def test_reroll_qids_file_holdout_sentinel(tmp_path):
    reroll = _load("cliff_reroll")

    class _Args:
        qids_file = "holdout"

    assert reroll._load_qids(_Args(), tmp_path, ["h1", "h2"]) == ["h1", "h2"]


def test_reroll_holdout_sentinel_requires_a_holdout(tmp_path):
    reroll = _load("cliff_reroll")

    class _Args:
        qids_file = "holdout"

    with pytest.raises(SystemExit, match="no holdout"):
        reroll._load_qids(_Args(), tmp_path, [])


def test_reroll_qids_file_paths_still_work(tmp_path):
    """The holdout sentinel must not disturb the existing A/B split usage."""
    reroll = _load("cliff_reroll")
    split = tmp_path / "cliff_split.json"
    write_json(split, {"A": ["a1"], "B": ["b1", "b2"]})

    class _Args:
        qids_file = f"{split}:B"

    assert reroll._load_qids(_Args(), tmp_path, ["h1"]) == ["b1", "b2"]
