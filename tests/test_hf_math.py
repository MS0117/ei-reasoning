"""CPU-only tests for the hf_math adapter and the data/passrate.py summary
logic: column preference, config passthrough, seeded sampling, drop rules,
classification thresholds, histogram math. No HF downloads, no GPU."""

import importlib.util
from pathlib import Path

import pytest

from expert_iter.config import Config
from expert_iter.data import HFMathAdapter
from expert_iter.verifier import StrictMathVerifier

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_passrate():
    spec = importlib.util.spec_from_file_location("passrate", REPO_ROOT / "data" / "passrate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.column_names = sorted({k for r in rows for k in r})

    def __iter__(self):
        return iter(self.rows)


@pytest.fixture
def fake_load(monkeypatch):
    """Patch datasets.load_dataset; returns a dict capturing rows + call args."""
    state = {"rows": [], "calls": []}

    def load_dataset(*args, **kwargs):
        state["calls"].append((args, kwargs))
        return FakeDataset(state["rows"])

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return state


def _rows(n: int) -> list[dict]:
    return [{"problem": f"What is {i}+{i}?", "answer": str(2 * i), "solution": rf"Working. \boxed{{{3 * i}}}"}
            for i in range(1, n + 1)]


def test_answer_col_preferred_over_solution(fake_load):
    fake_load["rows"] = _rows(3)
    records = HFMathAdapter().load({"hf_name": "org/ds"})
    # `answer` (clean final answer) must win over `solution` (worked trace).
    assert [r.final_answer for r in records] == ["2", "4", "6"]


def test_config_passthrough(fake_load):
    fake_load["rows"] = _rows(1)
    HFMathAdapter().load({"hf_name": "org/ds", "config": "default", "split": "train"})
    HFMathAdapter().load({"hf_name": "org/ds", "split": "train"})
    assert fake_load["calls"][0] == (("org/ds", "default"), {"split": "train"})
    assert fake_load["calls"][1] == (("org/ds",), {"split": "train"})


def test_seeded_sampling_deterministic_and_order_independent(fake_load):
    fake_load["rows"] = _rows(20)
    args = {"hf_name": "org/ds", "n_questions": 5, "seed": 7}
    first = [r.qid for r in HFMathAdapter().load(args)]
    fake_load["rows"] = list(reversed(fake_load["rows"]))
    second = [r.qid for r in HFMathAdapter().load(args)]
    assert len(first) == 5
    assert sorted(first) == sorted(second)
    fake_load["rows"] = _rows(20)
    other_seed = [r.qid for r in HFMathAdapter().load({**args, "seed": 8})]
    assert sorted(first) != sorted(other_seed)


def test_n_questions_exceeding_available_keeps_all(fake_load):
    fake_load["rows"] = _rows(4)
    assert len(HFMathAdapter().load({"hf_name": "org/ds", "n_questions": 99})) == 4


def test_empty_gold_and_question_dropped(fake_load):
    fake_load["rows"] = _rows(2) + [
        {"problem": "no gold?", "answer": "", "solution": ""},
        {"problem": "", "answer": "5", "solution": ""},
    ]
    assert len(HFMathAdapter().load({"hf_name": "org/ds"})) == 2


def test_dedup_by_question(fake_load):
    fake_load["rows"] = _rows(2) + _rows(2)
    assert len(HFMathAdapter().load({"hf_name": "org/ds"})) == 2


def test_missing_columns_raise(fake_load):
    fake_load["rows"] = [{"text": "hi", "label": "1"}]
    with pytest.raises(KeyError):
        HFMathAdapter().load({"hf_name": "org/ds"})


# ---------------------------------------------------------------------------
# passrate.py: classification + summary
# ---------------------------------------------------------------------------

def test_passrate_config_parses():
    cfg = Config.load(REPO_ROOT / "data" / "configs" / "passrate.yaml")
    assert cfg.data.adapter == "hf_math"
    assert cfg.data.adapter_args["n_questions"] == 100
    assert cfg.rollout.n == 8


def test_classify_thresholds():
    passrate = _load_passrate()
    assert passrate.classify(0, 1, 3) == "cliff"
    assert passrate.classify(1, 1, 3) == "frontier"
    assert passrate.classify(2, 1, 3) == "frontier"
    assert passrate.classify(3, 1, 3) == "solved"
    assert passrate.classify(8, 1, 3) == "solved"
    assert passrate.classify(2, 2, 5) == "frontier"
    assert passrate.classify(1, 2, 5) == "cliff"


def test_summarize_histogram_and_classes():
    passrate = _load_passrate()
    k = 4
    # 3 questions: c = 0 (cliff, with one truncation), 2 (frontier), 4 (solved).
    per_q = {"q0": 0, "q1": 2, "q2": 4}
    rows = [
        {"qid": qid, "sample_idx": i, "correct": i < c, "formatted": True,
         "extracted": "1" if i < c else "2",
         "finish_reason": "length" if qid == "q0" and i == 0 else "stop",
         "n_tokens": 10, "response_text": "x"}
        for qid, c in per_q.items() for i in range(k)
    ]
    gold = {qid: "1" for qid in per_q}
    metrics, stats = passrate.summarize(rows, k, StrictMathVerifier(), gold,
                                        frontier_min=1, solved_min=3)
    assert metrics["hist"] == {"0": 1, "2": 1, "4": 1}
    assert (metrics["n_cliff"], metrics["n_frontier"], metrics["n_solved"]) == (1, 1, 1)
    assert metrics["solve_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert metrics["all_correct_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["mean_pass_rate"] == pytest.approx((0 + 0.5 + 1) / 3, abs=1e-3)
    assert metrics["passrate/avg@4"] == pytest.approx(6 / 12, abs=1e-3)
    by_qid = {s["qid"]: s for s in stats}
    assert by_qid["q0"]["class"] == "cliff" and by_qid["q0"]["n_truncated"] == 1
    assert by_qid["q1"]["class"] == "frontier"
    assert by_qid["q2"]["class"] == "solved"
    # stats sorted hardest-first
    assert [s["c"] for s in stats] == [0, 2, 4]
