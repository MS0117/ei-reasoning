"""scripts/build_gold_sft.py — the offline gold-SFT dataset builder (L5 arm a).

CPU-only: the tokenizer and the question adapter are faked, so nothing is
downloaded and no rollout is involved (which is the point of the arm)."""

import importlib.util
import json
from pathlib import Path

import pytest

from expert_iter.records import QuestionRecord, SFTExample
from expert_iter.utils import read_json

REPO = Path(__file__).resolve().parents[1]
EOS = 151645


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_gold_sft", REPO / "scripts" / "build_gold_sft.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeTokenizer:
    """One id per character — enough to assert region splicing."""

    eos_token_id = EOS

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


QUESTIONS = [
    # (qid, gold, final_answer) — q3 has no gold, q4's gold boxes a wrong answer
    ("q1", r"work \boxed{7}", "7"),
    ("q2", r"more work \boxed{9}", "9"),
    ("q3", "", "5"),
    ("q4", r"bad work \boxed{3}", "11"),
]


@pytest.fixture
def built(tmp_path, monkeypatch):
    """Run the builder against a fake tokenizer and a fake question set."""
    mod = _load_module()
    import transformers

    class _Auto:
        @staticmethod
        def from_pretrained(*a, **k):
            return _FakeTokenizer()

    monkeypatch.setattr(transformers, "AutoTokenizer", _Auto)

    records = [
        QuestionRecord(qid=q, question=f"question {q}", final_answer=ans,
                       meta={"gold_solution": gold})
        for q, gold, ans in QUESTIONS
    ]
    monkeypatch.setattr(mod, "ensure_questions", lambda cfg, run_dir: (records, []))
    # render_question_prompt would need a real chat template; the prompt region is
    # opaque to this builder beyond its length.
    monkeypatch.setattr(mod, "render_question_prompt",
                        lambda *a, **k: type("P", (), {"token_ids": [1, 2, 3]})())

    def _run(*extra):
        run_dir = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
        mod.main(["-c", "configs/methods/l5_gold_sft.yaml",
                  "--run-dir", str(run_dir),
                  "--override", "run.name=test_gold",
                  *extra])
        return run_dir

    return mod, _run


def _rows(run_dir):
    return list(SFTExample.load_jsonl(run_dir / "iter_0" / "dataset" / "train_sft.jsonl"))


def test_skips_questions_without_gold_and_verifier_rejects(built):
    _, run = built
    rows = _rows(run())
    assert [r.qid for r in rows] == ["q1", "q2"]   # q3 no gold, q4 wrong answer


def test_no_verify_keeps_the_rejected_gold(built):
    _, run = built
    rows = _rows(run("--no-verify"))
    assert [r.qid for r in rows] == ["q1", "q2", "q4"]


def test_rows_are_plain_solved_examples(built):
    _, run = built
    for r in _rows(run()):
        r.validate()                    # region lengths sum to len(input_ids)
        assert r.source == "solved"
        assert r.anchor_len == 0        # y* is never a continuation of a prefix
        assert r.iter_created == 0


def test_completion_is_gold_ids_plus_eos_after_an_untouched_prompt(built):
    _, run = built
    (row, _) = _rows(run())[:2]
    gold = QUESTIONS[0][1]
    assert row.prompt_len == 3
    assert row.input_ids[:3] == [1, 2, 3]
    assert row.input_ids[3:] == [ord(c) for c in gold] + [EOS]
    assert row.text == gold


def test_stats_report_the_arm_yield(built):
    _, run = built
    stats = read_json(run() / "iter_0" / "dataset" / "stats.json")
    assert stats["n_questions"] == 4
    assert stats["n_without_gold"] == 1
    assert stats["n_verifier_rejected"] == 1
    assert stats["n_rows"] == 2
    assert stats["gold_accept_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_rerun_is_a_no_op(built, capsys):
    mod, run = built
    run_dir = run()
    before = (run_dir / "iter_0" / "dataset" / "train_sft.jsonl").read_text()
    mod.main(["-c", "configs/methods/l5_gold_sft.yaml", "--run-dir", str(run_dir),
              "--override", "run.name=test_gold"])
    assert "already done" in capsys.readouterr().out
    assert (run_dir / "iter_0" / "dataset" / "train_sft.jsonl").read_text() == before


def test_empty_dpo_file_is_written(built):
    _, run = built
    dpo = run() / "iter_0" / "dataset" / "train_dpo.jsonl"
    assert dpo.exists() and dpo.read_text() == ""


def test_config_pins_a_rollout_free_stage_list():
    """The arm's whole point: no sampling stage runs."""
    from expert_iter.config import Config

    cfg = Config.load("configs/methods/l5_gold_sft.yaml")
    cfg.validate()
    assert cfg.loop.stages == ["train", "eval", "benchmark_eval"]
    assert cfg.loop.iterations == 1
    assert not cfg.train.sft.cliff.enabled


def test_gold_sft_and_the_other_l5_arms_share_the_trainer():
    """Only the data recipe may differ — lr/batch/regions/seq len must not.

    epochs is the ONE documented exception: gold_sft trains offline on a static
    ~4k-row set, so no single value matches the looped arms (deliverable-matched
    = 2, step-matched = ~12). It carries 6 deliberately — see the EPOCHS block
    in configs/methods/l5_gold_sft.yaml. Every other knob still has to match, and
    the looped arms must match each other on epochs too.
    """
    from expert_iter.config import Config

    arms = ["l5_staged_dpo_s3", "l5_lspo", "l5_rft", "l5_gold_sft", "l5_gold_inloop",
            "l5_bridge_inloop"]
    cfgs = {}
    shape = set()
    for name in arms:
        c = Config.load(f"configs/methods/{name}.yaml")
        c.validate()
        cfgs[name] = c
        shape.add((c.train.sft.lr, c.train.sft.global_batch_size,
                   c.train.max_seq_len, c.train.init_from, c.train.backend,
                   json.dumps(c.train.sft.region_weights, sort_keys=True),
                   c.partition.solved_keep_max, c.partition.solved_selection))
    assert len(shape) == 1, shape

    looped = {n: cfgs[n].train.sft.epochs for n in arms if n != "l5_gold_sft"}
    assert len(set(looped.values())) == 1, looped
    # Not an arbitrary number: 6 is the "total epochs consumed" reading (3
    # iterations x 2), the value that cannot be read as under-training the
    # distillation baseline.
    assert cfgs["l5_gold_sft"].train.sft.epochs == 6
