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
# metadata row filters (where / require_any_true) — the OpenR1-220k cleanup
# ---------------------------------------------------------------------------

def test_where_filter_drops_disallowed(fake_load):
    rows = _rows(3)
    for row, qt in zip(rows, ["math-word-problem", "MCQ", "math-word-problem"]):
        row["question_type"] = qt
    fake_load["rows"] = rows
    records = HFMathAdapter().load(
        {"hf_name": "org/ds", "where": {"question_type": ["math-word-problem"]}}
    )
    assert [r.final_answer for r in records] == ["2", "6"]


def test_where_accepts_scalar_value(fake_load):
    rows = _rows(2)
    rows[0]["question_type"] = "MCQ"
    rows[1]["question_type"] = "math-word-problem"
    fake_load["rows"] = rows
    records = HFMathAdapter().load(
        {"hf_name": "org/ds", "where": {"question_type": "math-word-problem"}}
    )
    assert [r.final_answer for r in records] == ["4"]


def test_require_any_true_semantics(fake_load):
    rows = _rows(4)
    rows[0]["verified"] = [True, False]   # any true -> kept
    rows[1]["verified"] = [False, False]  # all false -> dropped
    rows[2]["verified"] = []              # empty list -> dropped
    rows[3]["verified"] = True            # scalar bool -> kept
    fake_load["rows"] = rows
    records = HFMathAdapter().load(
        {"hf_name": "org/ds", "require_any_true": ["verified"]}
    )
    assert [r.final_answer for r in records] == ["2", "8"]


def test_filter_on_missing_column_raises(fake_load):
    fake_load["rows"] = _rows(1)
    with pytest.raises(KeyError):
        HFMathAdapter().load({"hf_name": "org/ds", "where": {"nope": ["x"]}})
    with pytest.raises(KeyError):
        HFMathAdapter().load({"hf_name": "org/ds", "require_any_true": ["nope"]})


def test_require_boxed_gold_drops_prooflike(fake_load):
    fake_load["rows"] = [
        {"problem": "Compute 1+1.", "solution": r"Adding gives \boxed{2}."},
        {"problem": "Prove X.", "solution": "We proceed by induction. QED."},
    ]
    records = HFMathAdapter().load({"hf_name": "org/ds", "require_boxed_gold": True})
    assert [r.final_answer for r in records] == ["2"]


def test_preset_expansion_and_override(fake_load):
    rows = _rows(3)
    for row, qt in zip(rows, ["math-word-problem", "MCQ", "math-word-problem"]):
        row["question_type"] = qt
        row["correctness_math_verify"] = [True]
    fake_load["rows"] = rows
    records = HFMathAdapter().load({"preset": "openr1", "hf_name": "org/ds"})
    # Preset filters apply; the explicit hf_name overrides the preset's id and
    # the preset's config/split flow through to load_dataset.
    assert fake_load["calls"][-1] == (("org/ds", "default"), {"split": "train"})
    assert [r.final_answer for r in records] == ["2", "6"]


def test_openthoughts_preset_filters(fake_load):
    fake_load["rows"] = [
        {"problem": "Q1?", "solution": r"So \boxed{7}.", "correct": True},
        {"problem": "Q2?", "solution": "no box at all", "correct": True},
        {"problem": "Q3?", "solution": r"\boxed{9}", "correct": False},
        {"problem": r"Prove that $x \ge \frac{5}{6}$.", "solution": r"\boxed{\frac{5}{6}}",
         "correct": True},
    ]
    records = HFMathAdapter().load({"preset": "openthoughts", "hf_name": "org/ds"})
    # boxed gold required, R1-verified only, proof-worded question excluded
    assert [r.final_answer for r in records] == ["7"]


def test_exclude_question_regex(fake_load):
    rows = _rows(2)
    rows[1]["problem"] = "Show that the sum is even."
    fake_load["rows"] = rows
    records = HFMathAdapter().load(
        {"hf_name": "org/ds", "exclude_question_regex": r"(?i)^\s*(prove|show)\b"})
    assert [r.final_answer for r in records] == ["2"]


def test_unknown_preset_raises(fake_load):
    fake_load["rows"] = _rows(1)
    with pytest.raises(KeyError):
        HFMathAdapter().load({"preset": "nope"})


def test_resolve_adapter_args_expands_presets():
    from expert_iter.data import HF_MATH_PRESETS, resolve_adapter_args

    raw = {"preset": "openthoughts", "n_questions": 500}
    expanded = resolve_adapter_args("hf_math", raw)
    # Cache keys hash the expanded form, so editing a preset definition in
    # code must change the key even though the raw args are unchanged.
    assert expanded["hf_name"] == HF_MATH_PRESETS["openthoughts"]["hf_name"]
    assert expanded["require_any_true"] == ["correct"]
    assert expanded["n_questions"] == 500
    assert "preset" not in expanded
    # Explicit keys win; non-hf_math adapters pass through untouched.
    assert resolve_adapter_args("hf_math", {**raw, "hf_name": "x/y"})["hf_name"] == "x/y"
    assert resolve_adapter_args("local_jsonl", {"path": "p"}) == {"path": "p"}
    with pytest.raises(KeyError):
        resolve_adapter_args("hf_math", {"preset": "nope"})


def test_sampling_draws_from_filtered_pool(fake_load):
    rows = _rows(20)
    for i, row in enumerate(rows):
        row["question_type"] = "math-word-problem" if i % 2 == 0 else "MCQ"
    fake_load["rows"] = rows
    records = HFMathAdapter().load(
        {"hf_name": "org/ds", "n_questions": 5, "seed": 7,
         "where": {"question_type": ["math-word-problem"]}}
    )
    assert len(records) == 5
    # _rows answers are 2*(i+1); the filter keeps even i, whose answers are ≡2 mod 4.
    assert all(int(r.final_answer) % 4 == 2 for r in records)


# ---------------------------------------------------------------------------
# passrate.py: classification + summary
# ---------------------------------------------------------------------------

def test_passrate_config_parses():
    cfg = Config.load(REPO_ROOT / "data" / "configs" / "passrate.yaml")
    assert cfg.data.adapter == "hf_math"
    # N/K are experiment knobs the user retunes freely — assert shape, not values.
    assert cfg.data.adapter_args["n_questions"] > 0
    assert cfg.rollout.n > 0


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


def test_passrate_counts_truncated_correct_as_raw_only():
    passrate = _load_passrate()
    rows = [{
        "qid": "q", "sample_idx": 0, "correct": True,
        "raw_correct": True, "formatted": True, "extracted": "1",
        "finish_reason": "length", "n_tokens": 10, "response_text": "x",
    }]
    metrics, stats = passrate.summarize(
        rows, 1, StrictMathVerifier(), {"q": "1"},
        frontier_min=1, solved_min=1,
    )
    assert stats[0]["c"] == 0 and stats[0]["c_raw"] == 1
    assert metrics["solve_rate"] == 0
    assert metrics["raw_solve_rate"] == 1
