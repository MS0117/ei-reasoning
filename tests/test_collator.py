import math

import torch

from expert_iter.train import WeightedCausalCollator

W = {"prompt": 0.0, "anchor": 0.0, "continuation": 1.0, "solution": 1.0}
PAD = 0


def improved(ids, p, a, c, uid="u1"):
    return {"uid": uid, "source": "improved", "input_ids": ids,
            "prompt_len": p, "anchor_len": a, "completion_len": c}


def solved(ids, p, c, uid="u2"):
    return {"uid": uid, "source": "solved", "input_ids": ids,
            "prompt_len": p, "anchor_len": 0, "completion_len": c}


def test_regions_and_padding():
    coll = WeightedCausalCollator(PAD, W)
    batch = coll([
        improved(list(range(1, 9)), 3, 2, 3),   # len 8: [p p p][a a][c c c]
        solved(list(range(1, 6)), 2, 3),        # len 5: [p p][s s s]
    ])
    assert batch["input_ids"].shape == (2, 8)
    # zero-weight regions and pad have label -100
    assert batch["labels"][0].tolist() == [-100] * 5 + [6, 7, 8]
    assert batch["labels"][1].tolist() == [-100, -100, 3, 4, 5, -100, -100, -100]
    assert batch["loss_weights"][0].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert batch["loss_weights"][1].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert batch["attention_mask"][1].tolist() == [1] * 5 + [0] * 3


def test_anchor_weight_nonzero():
    w = dict(W, anchor=0.5)
    coll = WeightedCausalCollator(PAD, w)
    batch = coll([improved(list(range(1, 9)), 3, 2, 3)])
    assert batch["loss_weights"][0].tolist() == [0, 0, 0, 0.5, 0.5, 1, 1, 1]
    # anchor tokens now carry labels
    assert batch["labels"][0].tolist() == [-100, -100, -100, 4, 5, 6, 7, 8]


def test_weighted_loss_hand_computed():
    """Loss from a hand-built logits tensor must equal sum(w*ce)/sum(w)."""
    coll = WeightedCausalCollator(PAD, W)
    batch = coll([improved([1, 2, 3, 4], 1, 1, 2)])  # regions: [p][a][c c]
    vocab = 5
    torch.manual_seed(0)
    logits = torch.randn(1, 4, vocab)

    shift_logits = logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    weights = batch["loss_weights"][:, 1:]
    ce = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, vocab), labels.reshape(-1).clamp(min=0), reduction="none"
    ).view(labels.shape)
    ce = ce * (labels != -100) * weights
    got = (ce.sum() / weights[labels != -100].sum()).item()

    # manual: only positions with weight 1 (the two completion tokens at
    # positions 2,3 -> shifted indices 1,2) contribute
    lp = torch.log_softmax(shift_logits[0], dim=-1)
    expected = (-(lp[1, 3]) - (lp[2, 4])).item() / 2
    assert math.isclose(got, expected, rel_tol=1e-5)


def test_num_items_matches_weight_sum():
    coll = WeightedCausalCollator(PAD, W)
    b = coll([improved(list(range(1, 9)), 3, 2, 3), solved(list(range(1, 6)), 2, 3)])
    num = (b["loss_weights"][:, 1:] * (b["labels"][:, 1:] != -100)).sum().item()
    assert num == 3 + 3  # completion tokens per example (labels shifted off pos 0)
