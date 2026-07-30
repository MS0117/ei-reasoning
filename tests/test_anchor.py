from expert_iter.anchor import FixedFractionAnchor, NoAnchor
from expert_iter.records import RolloutSample, UnsolvedQuestion


def make_failed(n_tokens: int) -> RolloutSample:
    return RolloutSample(
        qid="q", sample_idx=0, prompt_text="p", prompt_token_ids=[1, 2, 3],
        response_text="r", response_token_ids=list(range(n_tokens)),
        finish_reason="stop",
    )


U = UnsolvedQuestion(qid="q", question="?", final_answer="1")


def test_fixed_fraction_basic():
    n = FixedFractionAnchor().select_len(U, make_failed(1000), {"fraction": 0.3, "min_tokens": 32, "max_tokens": 2048})
    assert n == 300


def test_fixed_fraction_clamps_min():
    n = FixedFractionAnchor().select_len(U, make_failed(50), {"fraction": 0.1, "min_tokens": 32, "max_tokens": 2048})
    assert n == 32


def test_fixed_fraction_clamps_max():
    n = FixedFractionAnchor().select_len(U, make_failed(100000), {"fraction": 0.9, "min_tokens": 32, "max_tokens": 2048})
    assert n == 2048


def test_fixed_fraction_never_exceeds_response():
    n = FixedFractionAnchor().select_len(U, make_failed(10), {"fraction": 0.5, "min_tokens": 32, "max_tokens": 2048})
    assert n == 10  # min_tokens clamp cannot exceed the actual response


def test_no_anchor():
    assert NoAnchor().select_len(U, make_failed(500), {}) == 0


def test_anchor_is_id_slice():
    failed = make_failed(100)
    n = FixedFractionAnchor().select_len(U, failed, {"fraction": 0.25, "min_tokens": 1, "max_tokens": 2048})
    anchor_ids = failed.response_token_ids[:n]
    assert anchor_ids == list(range(25))
