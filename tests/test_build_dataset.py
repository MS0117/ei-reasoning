from expert_iter.build_dataset import _content_key
from expert_iter.records import DPOExample


def _pair(rejected):
    return DPOExample(
        uid="u", qid="q", prompt_token_ids=[1],
        chosen_token_ids=[2], rejected_token_ids=rejected,
    )


def test_dpo_content_key_includes_rejected_sequence():
    assert _content_key(_pair([3])) != _content_key(_pair([4]))
