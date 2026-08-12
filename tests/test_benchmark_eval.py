"""CPU-only tests for the benchmark_eval stack: strict grading, boxed
extraction, preset resolution, qid namespace, metrics math, config parsing,
template kwargs, LoRA passthrough. No HF downloads, no GPU."""

import json

import pytest

from expert_iter.config import BenchmarkCfg, Config
from expert_iter.data import BENCHMARK_PRESETS, load_benchmark_questions
from expert_iter.benchmark_eval import summarize_benchmark
from expert_iter.lora import _adapter_cache_key, resolve_model_path
from expert_iter.records import QuestionRecord
from expert_iter.templates import render_question_prompt
from expert_iter.verifier import StrictMathVerifier, last_boxed


def q(answer: str) -> QuestionRecord:
    return QuestionRecord(qid="bench-t-0000", question="?", final_answer=answer)


# ---------------------------------------------------------------------------
# last_boxed extraction
# ---------------------------------------------------------------------------

def test_last_boxed_simple():
    assert last_boxed(r"the answer is \boxed{42}.") == "42"


def test_last_boxed_nested_braces():
    assert last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_last_boxed_takes_last():
    assert last_boxed(r"\boxed{1} then \boxed{2}") == "2"


def test_last_boxed_missing_or_unbalanced():
    assert last_boxed("no box here") is None
    assert last_boxed(r"\boxed{unclosed") is None


# ---------------------------------------------------------------------------
# math_strict verifier (OPSD grading semantics)
# ---------------------------------------------------------------------------

def test_strict_correct_boxed():
    v = StrictMathVerifier()
    verdict = v.verify(q("42"), r"Thus \boxed{42}.")
    assert verdict.correct and verdict.meta["formatted"]
    assert verdict.extracted_answer == "42"


def test_strict_wrong_boxed():
    assert not StrictMathVerifier().verify(q("42"), r"\boxed{41}").correct


def test_strict_no_box_is_unformatted_and_wrong():
    verdict = StrictMathVerifier().verify(q("42"), "The answer is 42.")
    assert not verdict.correct
    assert verdict.meta["formatted"] is False


def test_strict_fraction_equivalence():
    assert StrictMathVerifier().verify(q("1/2"), r"\boxed{\frac{1}{2}}").correct


def test_strict_string_fallback_on_unparsable():
    # Both sides fail sympy parsing identically -> normalized string equality.
    v = StrictMathVerifier()
    assert v.grade_extracted("AB = CD", "ab=cd")


def test_grade_extracted_used_for_majority():
    v = StrictMathVerifier()
    assert v.grade_extracted(r"\frac{1}{2}", "1/2")
    assert not v.grade_extracted("3", "1/2")


# ---------------------------------------------------------------------------
# Benchmark presets + adapter resolution
# ---------------------------------------------------------------------------

def test_presets_cover_requested_suite():
    for name in ["aime24", "aime25", "aime26", "hmmt25", "math500", "math500_hard"]:
        assert name in BENCHMARK_PRESETS


def test_unknown_benchmark_name_raises():
    with pytest.raises(KeyError, match="not a preset"):
        load_benchmark_questions(BenchmarkCfg(name="nope"))


def test_local_jsonl_benchmark_gets_bench_namespace(tmp_path):
    p = tmp_path / "bench.jsonl"
    rows = [{"qid": "x1", "question": "1+1?", "final_answer": "2"}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    recs = load_benchmark_questions(
        BenchmarkCfg(name="mini", adapter="local_jsonl", adapter_args={"path": str(p)})
    )
    assert len(recs) == 1
    assert recs[0].qid == "bench-mini-x1"  # namespace enforced for any adapter


# ---------------------------------------------------------------------------
# summarize_benchmark metrics
# ---------------------------------------------------------------------------

def _row(qid, correct, formatted=True, extracted=None, finish="stop"):
    return {"qid": qid, "correct": correct, "formatted": formatted,
            "extracted": extracted, "finish_reason": finish}


def test_summarize_pass_avg_format_truncation():
    rows = [
        # q1: 1/2 correct; q2: 0/2, one truncated+unformatted
        _row("q1", True, extracted="7"), _row("q1", False, extracted="8"),
        _row("q2", False, extracted="1"), _row("q2", False, formatted=False, finish="length"),
    ]
    m = summarize_benchmark("b", 2, rows, {"q1": "7", "q2": "9"}, StrictMathVerifier())
    assert m["b/n_questions"] == 2
    assert m["b/pass@2"] == 0.5
    assert m["b/avg@2"] == 0.25
    assert m["b/format_rate"] == 0.75
    assert m["b/truncated_rate"] == 0.25


def test_summarize_majority_vote():
    # q1: answers 7,7,3 -> majority 7 == gold -> maj credit even though avg < 1
    rows = [_row("q1", True, extracted="7"), _row("q1", True, extracted="7"),
            _row("q1", False, extracted="3")]
    m = summarize_benchmark("b", 3, rows, {"q1": "7"}, StrictMathVerifier())
    assert m["b/maj@3"] == 1.0


def test_majority_vote_groups_equivalent_answer_forms():
    rows = [
        _row("q1", True, extracted=r"\frac{1}{2}"),
        _row("q1", True, extracted="1/2"),
        _row("q1", False, extracted="3"),
        _row("q1", False, extracted="3"),
        _row("q1", False, extracted="4"),
    ]
    m = summarize_benchmark("b", 5, rows, {"q1": "1/2"}, StrictMathVerifier())
    assert m["b/maj@5"] == 1.0


def test_summarize_empty_rows():
    assert "b/error" in summarize_benchmark("b", 4, [], {}, StrictMathVerifier())


# ---------------------------------------------------------------------------
# Config: list-of-dataclass parsing + validation
# ---------------------------------------------------------------------------

def test_benchmarks_parse_from_yaml(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "eval:\n  benchmarks:\n"
        "    - {name: aime24, n: 4, max_tokens: 512}\n"
        "    - {name: mini, adapter: local_jsonl, adapter_args: {path: x.jsonl}}\n"
    )
    cfg = Config.load(cfg_path)
    assert [b.name for b in cfg.eval.benchmarks] == ["aime24", "mini"]
    assert isinstance(cfg.eval.benchmarks[0], BenchmarkCfg)
    assert cfg.eval.benchmarks[0].n == 4
    assert cfg.eval.benchmarks[1].adapter == "local_jsonl"


def test_benchmark_unknown_key_is_hard_error(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("eval:\n  benchmarks:\n    - {name: aime24, temprature: 1.0}\n")
    with pytest.raises(KeyError, match="temprature"):
        Config.load(cfg_path)


def test_benchmark_validation():
    with pytest.raises(ValueError, match="name is required"):
        Config.load(None, overrides=["eval.benchmarks=[{n: 4}]"])
    with pytest.raises(ValueError, match="duplicate"):
        Config.load(None, overrides=["eval.benchmarks=[{name: a}, {name: a}]"])


def test_default_config_includes_benchmark_eval_stage():
    cfg = Config.load(None)
    assert "benchmark_eval" in cfg.loop.stages
    assert cfg.eval.benchmarks == []  # opt-in via yaml


# ---------------------------------------------------------------------------
# chat_template_kwargs passthrough
# ---------------------------------------------------------------------------

class _StubTokenizer:
    def __init__(self):
        self.seen_kwargs = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        self.seen_kwargs = kwargs
        return "".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) % 100 for c in text]}


def test_chat_template_kwargs_forwarded():
    tok = _StubTokenizer()
    render_question_prompt(tok, "hi", chat_template_kwargs={"enable_thinking": False})
    assert tok.seen_kwargs == {"enable_thinking": False}


def test_chat_template_kwargs_default_empty():
    tok = _StubTokenizer()
    render_question_prompt(tok, "hi")
    assert tok.seen_kwargs == {}


# ---------------------------------------------------------------------------
# LoRA resolution
# ---------------------------------------------------------------------------

def test_resolve_model_path_passthrough_for_full_model(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert resolve_model_path(str(tmp_path)) == str(tmp_path)


def test_resolve_model_path_passthrough_for_hub_id():
    assert resolve_model_path("Qwen/Qwen3-4B-Instruct-2507") == "Qwen/Qwen3-4B-Instruct-2507"


def test_lora_cache_key_tracks_weight_content(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "org/base"}'
    )
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"first")
    first = _adapter_cache_key(tmp_path, "bfloat16")

    weights.write_bytes(b"other")
    second = _adapter_cache_key(tmp_path, "bfloat16")
    assert first != second
    assert second != _adapter_cache_key(tmp_path, "float16")
