from expert_iter.records import (
    AnchorRecord,
    ImprovedCandidate,
    RolloutSample,
    SFTExample,
)


def test_roundtrip(tmp_path):
    rec = RolloutSample(
        qid="q1", sample_idx=0, prompt_text="p", prompt_token_ids=[1, 2],
        response_text="r", response_token_ids=[3, 4, 5], finish_reason="stop",
        gen={"temperature": 1.0}, model_path="m", iter=2,
    )
    path = tmp_path / "x.jsonl"
    RolloutSample.dump_jsonl(path, [rec])
    back = list(RolloutSample.load_jsonl(path))
    assert len(back) == 1
    assert back[0] == rec


def test_from_dict_ignores_unknown_fields():
    d = AnchorRecord(
        qid="q", base_sample_idx=0, policy="p", anchor_token_ids=[1],
        anchor_text="a", anchor_len=1, base_response_len=4,
    ).to_dict()
    d["future_field"] = 123  # forward compatibility with newer writers
    rec = AnchorRecord.from_dict(d)
    assert rec.qid == "q"


def test_sft_example_validate():
    ex = SFTExample(
        uid="u", qid="q", source="improved",
        input_ids=list(range(10)), prompt_len=4, anchor_len=3, completion_len=3,
    )
    ex.validate()  # ok

    bad = SFTExample(
        uid="u", qid="q", source="improved",
        input_ids=list(range(10)), prompt_len=4, anchor_len=3, completion_len=4,
    )
    try:
        bad.validate()
        assert False, "expected ValueError"
    except ValueError:
        pass

    solved_with_anchor = SFTExample(
        uid="u", qid="q", source="solved",
        input_ids=list(range(10)), prompt_len=4, anchor_len=3, completion_len=3,
    )
    try:
        solved_with_anchor.validate()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_external_context_default_none():
    c = ImprovedCandidate(
        qid="q", base_sample_idx=0, attempt_idx=0,
        prompt_token_ids=[1], anchor_token_ids=[2],
        continuation_token_ids=[3], continuation_text="t",
    )
    assert c.external_context is None
