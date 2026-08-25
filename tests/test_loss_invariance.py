"""Grad-accum invariance: training on the same 8 examples must produce the same
weights whether they arrive as 1 batch of 8 or 8 accumulated micro-batches of 1.

This is the M4 acceptance check for the num_items normalization contract in
WeightedSFTTrainer (sum(w*ce)/global_weight_sum + no per-accum-step division).
Runs on CPU with a tiny random model — no GPU, no downloads.
"""

import copy

import pytest
import torch

torch.manual_seed(0)

from expert_iter.train import WeightedCausalCollator, make_weighted_trainer_cls

W = {"prompt": 0.0, "anchor": 0.5, "continuation": 1.0, "solution": 1.0}


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    torch.manual_seed(1234)
    return LlamaForCausalLM(cfg)


def synth_examples(n=8):
    torch.manual_seed(99)
    rows = []
    for i in range(n):
        p, a, c = 3 + i % 3, (0 if i % 2 else 2 + i % 4), 4 + i % 5
        ids = torch.randint(1, 127, (p + a + c,)).tolist()
        rows.append({
            "uid": f"u{i}", "source": "improved" if a else "solved",
            "input_ids": ids, "prompt_len": p, "anchor_len": a, "completion_len": c,
        })
    return rows


def run_once(model, rows, micro_bs, accum, tmp_path, tag):
    from datasets import Dataset
    from transformers import TrainingArguments

    args = TrainingArguments(
        output_dir=str(tmp_path / tag),
        per_device_train_batch_size=micro_bs,
        gradient_accumulation_steps=accum,
        num_train_epochs=1,
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        max_grad_norm=1e9,          # disable clipping (clip depends on step count only via total grad — same anyway)
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=7,
        use_cpu=True,
        average_tokens_across_devices=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer_cls = make_weighted_trainer_cls()
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        data_collator=WeightedCausalCollator(pad_token_id=0, region_weights=W),
    )
    trainer.train()
    return model


@pytest.mark.slow
def test_accum_topology_invariance(tmp_path):
    base = tiny_model()
    rows = synth_examples(8)

    m_a = run_once(copy.deepcopy(base), rows, micro_bs=8, accum=1, tmp_path=tmp_path, tag="a")
    m_b = run_once(copy.deepcopy(base), rows, micro_bs=1, accum=8, tmp_path=tmp_path, tag="b")

    sd_a, sd_b = m_a.state_dict(), m_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.allclose(sd_a[k], sd_b[k], atol=1e-5, rtol=1e-4), (
            f"parameter {k} diverged between accum topologies: "
            f"max diff {(sd_a[k] - sd_b[k]).abs().max().item():.3e}"
        )


# ---------------------------------------------------------------------------
# cliff objective: same invariance with the two-normalizer loss + stratified
# sampler + v1 negatives + guard (train.sft.cliff.enabled)
# ---------------------------------------------------------------------------

def cliff_rows():
    torch.manual_seed(101)
    rows = []
    for i in range(12):                                    # solved fill
        p, c = 3 + i % 3, 4 + i % 5
        ids = torch.randint(1, 127, (p + c,)).tolist()
        rows.append({"uid": f"s{i}", "qid": f"sq{i}", "source": "solved",
                     "input_ids": ids, "prompt_len": p, "anchor_len": 0,
                     "completion_len": c, "n_q": 0, "ref_mean_nll": None})
    for i in range(2):                                     # rescued successes
        ids = torch.randint(1, 127, (9,)).tolist()
        rows.append({"uid": f"i{i}", "qid": f"cq{i}", "source": "improved",
                     "input_ids": ids, "prompt_len": 3, "anchor_len": 2,
                     "completion_len": 4, "n_q": 1,
                     "ref_mean_nll": 0.5 if i == 0 else None})   # one guarded, one skipped
    for i in range(2):                                     # modal-wrong negatives
        ids = torch.randint(1, 127, (8,)).tolist()
        rows.append({"uid": f"n{i}", "qid": f"cq{i}", "source": "negative",
                     "input_ids": ids, "prompt_len": 3, "anchor_len": 0,
                     "completion_len": 5, "n_q": 1, "ref_mean_nll": None})
    return rows


def run_once_cliff(model, rows, micro_bs, accum, tmp_path, tag):
    from datasets import Dataset
    from transformers import TrainingArguments

    from expert_iter.config import Config
    from expert_iter.train import StratifiedWindowSampler

    G = micro_bs * accum                                   # world 1
    cfg = Config.load(None, overrides=[
        "train.sft.cliff.enabled=true", "filter.selection.always_score=true",
        "train.sft.cliff.rho=0.3", "train.sft.cliff.negative.mode=v1",
        "train.sft.cliff.negative.mu=0.5", f"train.sft.global_batch_size={G}",
    ])
    solved_idx = [i for i, r in enumerate(rows) if r["source"] == "solved"]
    cliff_pairs = [(i, r["qid"]) for i, r in enumerate(rows) if r["source"] == "improved"]
    neg_by_qid = {}
    for i, r in enumerate(rows):
        if r["source"] == "negative":
            neg_by_qid.setdefault(r["qid"], []).append(i)
    sampler = StratifiedWindowSampler(
        solved_idx=solved_idx, cliff_idx=[i for i, _ in cliff_pairs],
        neg_idx_by_qid=neg_by_qid, cliff_qids=[q for _, q in cliff_pairs],
        window=G, m_c=1, m_n=1, seed=7)
    args = TrainingArguments(
        output_dir=str(tmp_path / tag),
        per_device_train_batch_size=micro_bs,
        gradient_accumulation_steps=accum,
        num_train_epochs=1, learning_rate=1e-3, lr_scheduler_type="constant",
        weight_decay=0.0, max_grad_norm=1e9, logging_steps=1,
        save_strategy="no", report_to=[], seed=7, use_cpu=True,
        average_tokens_across_devices=True, remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer_cls = make_weighted_trainer_cls()
    trainer = trainer_cls(
        model=model, args=args, train_dataset=Dataset.from_list(rows),
        data_collator=WeightedCausalCollator(pad_token_id=0, region_weights=W,
                                             cliff_mode=True),
        cliff_cfg=cfg.train.sft.cliff, train_sampler=sampler,
    )
    trainer.train()
    return model


@pytest.mark.slow
def test_cliff_accum_topology_invariance(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    base = tiny_model()
    rows = cliff_rows()                                    # 12 solved -> 2 windows of G=8

    m_a = run_once_cliff(copy.deepcopy(base), rows, micro_bs=8, accum=1, tmp_path=tmp_path, tag="ca")
    m_b = run_once_cliff(copy.deepcopy(base), rows, micro_bs=1, accum=8, tmp_path=tmp_path, tag="cb")

    sd_a, sd_b = m_a.state_dict(), m_b.state_dict()
    for k in sd_a:
        assert torch.allclose(sd_a[k], sd_b[k], atol=1e-5, rtol=1e-4), (
            f"parameter {k} diverged between accum topologies (cliff objective): "
            f"max diff {(sd_a[k] - sd_b[k]).abs().max().item():.3e}"
        )
