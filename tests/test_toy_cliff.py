"""data/toy_cliff.py: metrics aggregation + stage sequencing (no GPU)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

import toy_cliff  # noqa: E402

from expert_iter.config import Config  # noqa: E402
from expert_iter.records import AnchorRecord  # noqa: E402
from expert_iter.utils import (  # noqa: E402
    done_marker,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def _anchor(qid, alen, base_len=100, meta=None):
    return AnchorRecord(qid=qid, base_sample_idx=0, policy="p",
                        anchor_token_ids=list(range(alen)), anchor_text="",
                        anchor_len=alen, base_response_len=base_len, meta=meta or {})


def test_anchor_stats():
    anchors = [
        _anchor("q1", 30, meta={"threshold_crossed": True, "j_anchor": 3}),
        _anchor("q2", 50, meta={"threshold_crossed": False, "j_anchor": 2}),
        _anchor("q3", 0, meta={"reason": "no_gold"}),
        _anchor("q4", 0, meta={"reason": "too_few_steps"}),
    ]
    s = toy_cliff.anchor_stats(anchors)
    assert s["anchor/n_total"] == 4 and s["anchor/n_nonempty"] == 2
    assert s["anchor/empty_reasons"] == {"no_gold": 1, "too_few_steps": 1}
    assert s["anchor/mean_len"] == 40.0
    assert s["anchor/mean_frac"] == pytest.approx(0.4)
    assert s["anchor/threshold_crossed_rate"] == 0.5
    assert s["anchor/mean_j_anchor"] == 2.5
    assert s["anchor/lens"] == [30, 50]


def test_anchor_stats_empty():
    s = toy_cliff.anchor_stats([])
    assert s["anchor/n_total"] == 0 and "anchor/mean_len" not in s


def test_alpha_survival():
    curves = [
        {"qid": "q1", "p_hat_by_alpha": {"0.5": 0.5, "1": 0.5}, "alpha_star": 0.5,
         "refit": False},
        {"qid": "q2", "p_hat_by_alpha": {"0.5": 0.0, "1": 0.25}, "alpha_star": 1.0,
         "refit": True},
    ]
    s = toy_cliff.alpha_survival(curves, tau_p=0.25)
    assert s["alpha/n_questions_swept"] == 2 and s["alpha/n_refit"] == 1
    assert s["alpha/p_hat_mean/0.5"] == 0.25 and s["alpha/p_hat_mean/1"] == 0.375
    assert s["alpha/n_survive/0.5"] == 1          # only q1 clears tau at 0.5
    assert s["alpha/n_survive/1"] == 2            # boundary inclusive (>=)
    assert s["alpha/star_mean"] == 0.75
    assert s["alpha/star_values"] == [0.5, 1.0]


def test_alpha_survival_empty():
    assert toy_cliff.alpha_survival([], tau_p=0.25) == {}


def test_selection_stats():
    scores = [
        {"qid": "q1", "key": "a", "s_mean": 0.1, "s_tail": 0.2, "d_tail": None,
         "c": 0.3, "kept": True},
        {"qid": "q1", "key": "b", "s_mean": 0.5, "s_tail": 1.0, "d_tail": 0.05,
         "c": 1.5, "kept": False},
        {"qid": "q2", "key": "c", "s_mean": 0.2, "s_tail": 0.4, "d_tail": None,
         "c": 0.6, "kept": True},
        {"qid": "q3", "key": "d", "s_mean": None, "s_tail": None, "d_tail": None,
         "c": None, "kept": False},  # unscorable row is excluded from means
    ]
    s = toy_cliff.selection_stats(scores)
    assert s["selection/n_scored"] == 4
    assert s["selection/c_mean_all"] == pytest.approx(0.8)      # mean over Y+
    assert s["selection/c_mean_kept"] == pytest.approx(0.45)    # mean over y† set
    assert s["selection/c_min_per_question_mean"] == pytest.approx(0.45)  # (0.3+0.6)/2
    assert s["selection/n_kept_scored"] == 2
    assert s["selection/s_tail_p95"] == 1.0
    assert s["selection/d_tail_p50"] == 0.05
    assert s["selection/c_values"] == [0.3, 0.6, 1.5]


def test_selection_stats_no_scorable():
    s = toy_cliff.selection_stats([{"qid": "q", "c": None}])
    assert s == {"selection/n_scored": 1}


def test_collect_metrics_end_to_end(tmp_path):
    run_dir = tmp_path
    it_dir = run_dir / "iter_0"
    write_jsonl(run_dir / "questions" / "train.jsonl",
                [{"qid": f"q{i}", "question": "?", "final_answer": "1"} for i in range(5)])
    write_json(it_dir / "partition" / "stats.json",
               {"n_unsolved_questions": 4, "solve_rate": 0.2, "sample_accuracy": 0.05})
    AnchorRecord.dump_jsonl(it_dir / "anchors" / "anchors.jsonl", [
        _anchor("q1", 20), _anchor("q2", 0, meta={"reason": "no_gold"}),
    ])
    write_jsonl(it_dir / "improve" / "improved.jsonl", [
        {"qid": "q1", "base_sample_idx": 0, "attempt_idx": 0, "prompt_token_ids": [1],
         "anchor_token_ids": [], "continuation_token_ids": [2], "continuation_text": "",
         "correct": True},
        {"qid": "q1", "base_sample_idx": 0, "attempt_idx": 1, "prompt_token_ids": [1],
         "anchor_token_ids": [], "continuation_token_ids": [3], "continuation_text": "",
         "correct": False},
    ])
    write_json(it_dir / "improve" / "stats.json",
               {"operator": "lora_sft", "n_with_gold": 4, "lora_yield": 0.25})
    write_jsonl(it_dir / "improve" / "alpha_curves.jsonl", [
        {"qid": "q1", "p_hat_by_alpha": {"1": 0.5}, "alpha_star": 1.0, "refit": False},
    ])
    write_json(it_dir / "filtered" / "report.json", {
        "n_kept": 1, "n_questions_improved": 1, "improve_yield": 0.25,
        "rejects": {"correctness:incorrect": 1},
        "cliff/conversion_rate": 0.25, "cliff/conversion_histogram": [0, 0, 0, 1],
    })
    write_jsonl(it_dir / "filtered" / "candidate_scores.jsonl", [
        {"qid": "q1", "key": "k", "s_mean": 0.1, "s_tail": 0.2, "d_tail": None,
         "c": 0.3, "kept": True},
    ])

    cfg = Config.load(None)
    m = toy_cliff.collect_metrics(run_dir, cfg)

    assert m["funnel/n_questions"] == 5 and m["funnel/n_cliff"] == 4
    assert m["funnel/n_anchored"] == 1
    assert m["funnel/n_candidates"] == 2 and m["funnel/n_correct_candidates"] == 1
    assert m["funnel/n_kept"] == 1 and m["funnel/n_questions_improved"] == 1
    assert m["funnel/n_with_gold"] == 4 and m["improve/lora_yield"] == 0.25
    assert m["alpha/p_hat_mean/1"] == 0.5
    assert m["selection/c_mean_all"] == 0.3
    assert m["filter/rejects"] == {"correctness:incorrect": 1}
    toy_cliff.print_funnel(m)  # smoke: must not raise


def test_collect_metrics_counts_correct_for_uninlined_operators(tmp_path):
    """The no-LoRA control (self_resample) leaves candidate.correct None until
    the filters stage grades it — the funnel must still report the correct
    count, from the conversion histogram."""
    run_dir = tmp_path
    it_dir = run_dir / "iter_0"
    write_jsonl(run_dir / "questions" / "train.jsonl",
                [{"qid": "q1", "question": "?", "final_answer": "1"}])
    write_json(it_dir / "partition" / "stats.json",
               {"n_unsolved_questions": 1, "solve_rate": 0.0})
    AnchorRecord.dump_jsonl(it_dir / "anchors" / "anchors.jsonl", [_anchor("q1", 0)])
    write_jsonl(it_dir / "improve" / "improved.jsonl", [
        {"qid": "q1", "base_sample_idx": 0, "attempt_idx": i, "prompt_token_ids": [1],
         "anchor_token_ids": [], "continuation_token_ids": [i], "continuation_text": "",
         "correct": None}                      # self_resample: not graded yet
        for i in range(3)
    ])
    write_json(it_dir / "filtered" / "report.json", {
        "n_kept": 1, "n_questions_improved": 1, "improve_yield": 1.0, "rejects": {},
        "cliff/conversion_rate": 1.0, "cliff/conversion_histogram": [2],
    })
    m = toy_cliff.collect_metrics(run_dir, Config.load(None))
    assert m["funnel/n_candidates"] == 3
    assert m["funnel/n_correct_candidates"] == 2      # from the histogram


def test_run_stages_order_and_force(tmp_path, monkeypatch):
    calls = []

    def fake_stage(name):
        return SimpleNamespace(main=lambda argv: calls.append((name, list(argv))))

    monkeypatch.setattr(toy_cliff, "STAGES",
                        [(n, fake_stage(n)) for n, _ in toy_cliff.STAGES])
    it_dir = tmp_path / "iter_0"
    marker = done_marker(it_dir / "rollout" / "rollouts.jsonl")
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    other = done_marker(it_dir / "improve" / "improved.jsonl")
    other.parent.mkdir(parents=True)
    other.write_text("{}")

    args = SimpleNamespace(force="rollout", model_path=None)
    toy_cliff.run_stages(args, tmp_path, "org/policy")

    assert [n for n, _ in calls] == ["rollout", "partition", "anchor", "improve", "filters"]
    _, argv = calls[0]
    assert argv[argv.index("--iter") + 1] == "0"
    assert argv[argv.index("--model-path") + 1] == "org/policy"
    assert argv[argv.index("--run-dir") + 1] == str(tmp_path)
    assert not marker.exists()      # --force rollout unlinked its marker
    assert other.exists()           # other stages untouched


def _fake_source_run(tmp_path, cfg):
    """A minimal completed toy run dir: frozen config + questions + rollout +
    partition artifacts."""
    src = tmp_path / "src_run"
    cfg.save(src / "config.yaml")
    write_jsonl(src / "questions" / "train.jsonl",
                [{"qid": "q1", "question": "?", "final_answer": "1"}])
    write_jsonl(src / "questions" / "holdout.jsonl", [])
    from expert_iter.utils import mark_done
    mark_done(src / "questions" / "train.jsonl", count=1, config_hash="whatever")
    mark_done(src / "questions" / "holdout.jsonl", count=0, config_hash="whatever")
    it = src / "iter_0"
    write_jsonl(it / "rollout" / "rollouts.jsonl", [
        {"qid": "q1", "sample_idx": 0, "prompt_text": "p", "prompt_token_ids": [1],
         "response_text": "r", "response_token_ids": [2, 3], "finish_reason": "stop"},
    ])
    write_jsonl(it / "partition" / "verdicts.jsonl",
                [{"qid": "q1", "sample_idx": 0, "correct": False}])
    write_jsonl(it / "partition" / "solved.jsonl", [])
    write_jsonl(it / "partition" / "unsolved.jsonl",
                [{"qid": "q1", "question": "?", "final_answer": "1",
                  "failed_sample_idxs": [0]}])
    write_json(it / "partition" / "stats.json", {"n_questions": 1, "n_unsolved_questions": 1})
    return src


def test_reuse_rollout_copies_and_remarks_done(tmp_path):
    cfg = Config.load(None)
    src = _fake_source_run(tmp_path, cfg)
    dst = tmp_path / "dst_run"

    toy_cliff.reuse_rollout(src, dst, cfg)

    it = dst / "iter_0"
    assert (dst / "questions" / "train.jsonl").exists()
    assert list(read_jsonl(it / "rollout" / "rollouts.jsonl"))[0]["response_token_ids"] == [2, 3]
    assert list(read_jsonl(it / "partition" / "unsolved.jsonl"))[0]["qid"] == "q1"
    # the stage-gating markers must carry THIS run's config hash so rollout and
    # partition skip themselves
    for rel in ("rollout/rollouts.jsonl", "partition/solved.jsonl"):
        assert read_json(done_marker(it / rel))["config_hash"] == cfg.hash()
    # ensure_questions only checks marker existence
    assert done_marker(dst / "questions" / "train.jsonl").exists()


def test_reuse_rollout_allows_downstream_only_differences(tmp_path):
    """anchor/improve/filter differences are exactly what reuse is for."""
    cfg = Config.load(None)
    src = _fake_source_run(tmp_path, cfg)
    variant = Config.load(None, overrides=[
        "anchor.policy=none", "filter.selection.method=random",
        "improve.lora_sft.fit.steps=4",
    ])
    toy_cliff.reuse_rollout(src, tmp_path / "dst2", cfg=variant)
    assert (tmp_path / "dst2" / "iter_0" / "rollout" / "rollouts.jsonl").exists()


@pytest.mark.parametrize("override", [
    "rollout.n=4",                 # different sampling -> different rollouts
    "run.seed=99",                 # different per-request seeds
    "model.base=org/other",        # different policy
    "partition.cliff_max_correct=1",  # different cliff routing
])
def test_reuse_rollout_refuses_incompatible_config(tmp_path, override):
    cfg = Config.load(None)
    src = _fake_source_run(tmp_path, cfg)
    with pytest.raises(SystemExit, match="rollout-determining"):
        toy_cliff.reuse_rollout(src, tmp_path / "dst3", Config.load(None, overrides=[override]))


def test_reuse_rollout_refuses_incomplete_source(tmp_path):
    cfg = Config.load(None)
    src = _fake_source_run(tmp_path, cfg)
    (src / "iter_0" / "partition" / "unsolved.jsonl").unlink()
    with pytest.raises(SystemExit, match="incomplete"):
        toy_cliff.reuse_rollout(src, tmp_path / "dst4", cfg)


def test_check_gold_solutions(tmp_path, capsys):
    qfile = tmp_path / "q.jsonl"
    write_jsonl(qfile, [{"qid": "q", "question": "?", "final_answer": "1",
                         "meta": {}}])
    cfg = Config.load(None, overrides=[
        "data.adapter=local_jsonl", f"data.adapter_args.path={qfile}",
    ])
    with pytest.raises(SystemExit, match="gold_solution"):
        toy_cliff.check_gold_solutions(cfg)

    write_jsonl(qfile, [{"qid": "q", "question": "?", "final_answer": "1",
                         "meta": {"gold_solution": "because"}}])
    toy_cliff.check_gold_solutions(cfg)
    assert "1/1 questions carry gold solutions" in capsys.readouterr().out


def test_check_gold_skips_hf_adapters():
    cfg = Config.load(None)  # openthoughts_math adapter -> no local file check
    toy_cliff.check_gold_solutions(cfg)


def test_parse_args_rejects_unknown_force():
    with pytest.raises(SystemExit):
        toy_cliff.parse_args(["--run-dir", "x", "--force", "bogus"])


def test_toy_config_validates():
    cfg = Config.load("data/configs/toy_cliff.yaml")
    assert cfg.model.base == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg.anchor.policy == "privileged_divergence"
    assert cfg.improve.operator == "lora_sft"
    assert cfg.improve.lora_sft.project_back.enabled is False   # off for now (confirmed)
    assert cfg.filter.selection.method == "c_score"
    assert cfg.rollout.max_tokens == 16384
    assert cfg.improve.max_tokens == 16384
    assert cfg.improve.lora_sft.fit.max_pair_tokens == 16384
    assert cfg.data.eval_holdout == 0


def test_lspo_config_validates():
    cfg = Config.load("data/configs/LSPO.yaml")
    assert cfg.model.base == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg.anchor.policy == "none"                          # no anchor
    assert cfg.improve.operator == "lora_sft"
    assert cfg.improve.lora_sft.project_back.enabled is False   # no project-back
    assert cfg.filter.selection.method == "random"              # random pick among correct
    assert cfg.filter.max_per_question == 1                     # exactly one
    assert cfg.rollout.max_tokens == 16384
