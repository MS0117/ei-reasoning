"""Cliff objective (train.sft.cliff): build_dataset helpers, collator channels,
stratified sampler, and the two-normalizer trainer mechanics.

CPU-only. The composed loss L = (1-rho)*L_S + rho*(L_C + mu*L_N + L_G) is
checked against an independent reference computed from the same tiny model's
logits; the DDP property is checked as rank-shard additivity of the gathered
denominator vector (the round-robin sharding itself is asserted against
accelerate's BatchSamplerShard).
"""

from __future__ import annotations

import copy

import pytest
import torch

from expert_iter.build_dataset import _build_dpo_pairs, _modal_wrong_failures, _stamp_n_q
from expert_iter.config import Config
from expert_iter.records import (
    ImprovedCandidate,
    RolloutSample,
    SFTExample,
    VerdictRecord,
)
from expert_iter.train import StratifiedWindowSampler, WeightedCausalCollator

W = {"prompt": 0.0, "anchor": 0.0, "continuation": 1.0, "solution": 1.0}


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """GPUs on this box are cgroup-isolated (see memory: srv04 srun isolation);
    CUDA init fails outside the allocation. All tests here are CPU-only."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


# ---------------------------------------------------------------------------
# build_dataset helpers
# ---------------------------------------------------------------------------

def _rollout(qid, idx, finish="stop", resp=None):
    resp = resp if resp is not None else [10 + idx, 11, 12]
    return RolloutSample(
        qid=qid, sample_idx=idx, prompt_text="p", prompt_token_ids=[1, 2],
        response_text="r", response_token_ids=resp, finish_reason=finish,
    )


def _verdict(qid, idx, correct, ans):
    return VerdictRecord(qid=qid, sample_idx=idx, correct=correct, extracted_answer=ans)


def _write_iter_dir(tmp_path, rollouts, verdicts):
    (tmp_path / "rollout").mkdir(parents=True, exist_ok=True)
    (tmp_path / "partition").mkdir(parents=True, exist_ok=True)
    RolloutSample.dump_jsonl(tmp_path / "rollout" / "rollouts.jsonl", rollouts)
    VerdictRecord.dump_jsonl(tmp_path / "partition" / "verdicts.jsonl", verdicts)


def test_modal_wrong_failures_grouping(tmp_path):
    rollouts = [
        _rollout("q1", 0), _rollout("q1", 1), _rollout("q1", 2),
        _rollout("q1", 3, finish="length"),      # truncated: excluded
        _rollout("q1", 4),                        # correct: excluded
        _rollout("q1", 5),                        # extracted None: excluded
        _rollout("q2", 0), _rollout("q2", 1),
        _rollout("q3", 0),                        # qid not requested
    ]
    verdicts = [
        _verdict("q1", 0, False, "7"), _verdict("q1", 1, False, "7"),
        _verdict("q1", 2, False, "9"),
        _verdict("q1", 3, False, "7"),
        _verdict("q1", 4, True, "42"),
        _verdict("q1", 5, False, None),
        # q2: tie 1-1 -> deterministic tie-break by (-count, answer): "3" < "5"
        _verdict("q2", 0, False, "5"), _verdict("q2", 1, False, "3"),
        _verdict("q3", 0, False, "1"),
    ]
    _write_iter_dir(tmp_path, rollouts, verdicts)
    out = _modal_wrong_failures(tmp_path, {"q1", "q2"})
    assert [s.sample_idx for s in out["q1"]] == [0, 1]     # modal answer "7", clean only
    assert [s.sample_idx for s in out["q2"]] == [1]        # tie -> answer "3"
    assert "q3" not in out


def test_stamp_n_q_counts_per_slice():
    def ex(uid, qid, source):
        return SFTExample(uid=uid, qid=qid, source=source,
                          input_ids=[1, 2], prompt_len=1, anchor_len=0, completion_len=1)
    rows = [ex("s", "qa", "solved"), ex("i1", "qa", "improved"), ex("i2", "qa", "improved"),
            ex("i3", "qb", "improved"), ex("n1", "qa", "negative")]
    _stamp_n_q(rows)
    assert rows[0].n_q == 0                                # solved untouched
    assert rows[1].n_q == 2 and rows[2].n_q == 2           # qa improved
    assert rows[3].n_q == 1                                # qb improved
    assert rows[4].n_q == 1                                # qa negative (n^-_q)


def _cand(qid, base_idx, attempt):
    return ImprovedCandidate(
        qid=qid, base_sample_idx=base_idx, attempt_idx=attempt,
        prompt_token_ids=[1, 2], anchor_token_ids=[],
        continuation_token_ids=[5, 6], continuation_text="c",
    )


def test_dpo_pairs_modal_wrong_and_fallback(tmp_path):
    rollouts = [_rollout("q1", 0, resp=[30, 31]), _rollout("q1", 1, resp=[40, 41, 42]),
                _rollout("q2", 0, resp=[50])]
    verdicts = [_verdict("q1", 0, False, "7"), _verdict("q1", 1, False, "7"),
                _verdict("q2", 0, False, None)]           # q2 has no valid modal
    _write_iter_dir(tmp_path, rollouts, verdicts)
    modal = _modal_wrong_failures(tmp_path, {"q1", "q2"})
    kept = [_cand("q1", 1, 0), _cand("q2", 0, 0)]
    pairs, stats = _build_dpo_pairs(kept, tmp_path, eos=99, iteration=0, modal_by_qid=modal)
    assert stats == {"dpo_rejected_modal_wrong": 1, "dpo_rejected_fallback": 1}
    by_qid = {p.qid: p for p in pairs}
    # q1: modal pick = sample 0 (differs from base_sample_idx 1) -> FULL response
    assert by_qid["q1"].rejected_token_ids == [30, 31, 99]
    # q2: fallback to base_pick (sample 0)
    assert by_qid["q2"].rejected_token_ids == [50, 99]
    # legacy path unchanged: no modal dict -> stats empty, base_pick rejected
    pairs2, stats2 = _build_dpo_pairs(kept, tmp_path, eos=99, iteration=0)
    assert stats2 == {} and {p.qid: p for p in pairs2}["q1"].rejected_token_ids == [40, 41, 42, 99]


# ---------------------------------------------------------------------------
# collator channels
# ---------------------------------------------------------------------------

def _row(uid, source, n_q=0, ref=None, p=2, a=0, c=3):
    ids = list(range(1, 1 + p + a + c))
    return {"uid": uid, "source": source, "input_ids": ids,
            "prompt_len": p, "anchor_len": a, "completion_len": c,
            "n_q": n_q, "ref_mean_nll": ref}


def test_collator_legacy_output_unchanged():
    legacy = WeightedCausalCollator(pad_token_id=0, region_weights=W)
    out = legacy([_row("s", "solved"), _row("i", "improved", c=4)])
    assert set(out) == {"input_ids", "attention_mask", "labels", "loss_weights"}


def test_collator_cliff_channels():
    col = WeightedCausalCollator(pad_token_id=0, region_weights=W, cliff_mode=True)
    out = col([
        _row("s", "solved", c=3),
        _row("i", "improved", n_q=2, ref=1.5, c=4),
        _row("n", "negative", n_q=3, ref=None, c=2),
        _row("old", "improved", n_q=0, ref=None, c=2),    # pre-cliff accumulated row
    ])
    assert out["slice_ids"].tolist() == [0, 1, 2, 1]
    assert out["n_q"].tolist() == [0.0, 2.0, 3.0, 1.0]     # solved 0; old row clamped to 1
    assert out["ref_mean_nll"].tolist() == [-1.0, 1.5, -1.0, -1.0]
    # completion mask covers exactly the completion region, zero on padding
    m = out["completion_mask"]
    assert m.shape == out["labels"].shape
    assert m[1].tolist() == [0, 0, 1, 1, 1, 1]             # p=2 then c=4 (max_len 6)
    assert m[2].tolist() == [0, 0, 1, 1, 0, 0]             # c=2 then padding


# ---------------------------------------------------------------------------
# stratified sampler
# ---------------------------------------------------------------------------

def _mk_sampler(n_solved=30, n_cliff=3, negs=None, window=8, m_c=1, m_n=0, seed=7):
    solved = list(range(n_solved))
    cliff = [100 + i for i in range(n_cliff)]
    qids = [f"q{i}" for i in range(n_cliff)]
    return StratifiedWindowSampler(
        solved_idx=solved, cliff_idx=cliff, neg_idx_by_qid=negs or {},
        cliff_qids=qids, window=window, m_c=m_c, m_n=m_n, seed=seed)


def test_sampler_window_composition_and_truncation():
    negs = {"q0": [200], "q1": [201, 202]}
    s = _mk_sampler(n_solved=30, n_cliff=3, negs=negs, m_n=1)
    stream = list(iter(s))
    assert len(stream) == len(s) == (30 // 6) * 8          # fill=6 -> 5 windows of 8
    for w in range(5):
        win = stream[w * 8:(w + 1) * 8]
        assert sum(i >= 100 and i < 200 for i in win) == 1  # exactly m_c cliff
        assert sum(i >= 200 for i in win) == 1              # exactly m_n negative
    # no even_batches padding possible; every solved index used at most once
    solved_used = [i for i in stream if i < 100]
    assert len(solved_used) == len(set(solved_used))


def test_sampler_epoch_reshuffle_and_determinism():
    a, b = _mk_sampler(), _mk_sampler()
    assert list(iter(a)) == list(iter(b))                   # rank-independence
    a.set_epoch(1)
    assert list(iter(a)) != list(iter(b))                   # reshuffled
    a.set_epoch(0)
    assert list(iter(a)) == list(iter(b))                   # deterministic in epoch


def test_sampler_zero_cliff_degrades_and_too_small_raises():
    s = StratifiedWindowSampler(
        solved_idx=list(range(16)), cliff_idx=[], neg_idx_by_qid={}, cliff_qids=[],
        window=8, m_c=1, m_n=0, seed=0)
    assert s.m_c == 0 and len(list(iter(s))) == 16          # solved-only windows
    # too few solved AND no cliff rows to fall back on -> actionable error
    with pytest.raises(ValueError, match="too small"):
        StratifiedWindowSampler(
            solved_idx=list(range(3)), cliff_idx=[], neg_idx_by_qid={}, cliff_qids=[],
            window=8, m_c=1, m_n=0, seed=0)


def test_sampler_survives_batch_sampler_shard_round_robin():
    """The G-block <-> optimizer-window identity under accelerate's sharding."""
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler

    s = _mk_sampler(n_solved=30, n_cliff=3, m_c=1)
    stream = list(iter(s))
    per_rank = []
    for r in (0, 1):
        bs = BatchSamplerShard(
            BatchSampler(s, batch_size=1, drop_last=False),
            num_processes=2, process_index=r, split_batches=False)
        per_rank.append([b[0] for b in bs])
    G, accum = 8, 4                                        # micro=1, world=2
    for w in range(len(stream) // G):
        window_union = set(per_rank[0][w * accum:(w + 1) * accum]
                           + per_rank[1][w * accum:(w + 1) * accum])
        assert window_union == set(stream[w * G:(w + 1) * G])


# ---------------------------------------------------------------------------
# trainer mechanics (tiny model, CPU)
# ---------------------------------------------------------------------------

def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(1234)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32))


def _mk_trainer(tmp_path, cliff_overrides=None, rows=None):
    from transformers import TrainingArguments
    from datasets import Dataset
    from expert_iter.train import make_weighted_trainer_cls

    overrides = cliff_overrides or []
    cfg = Config.load(None, overrides=overrides)
    cl = cfg.train.sft.cliff
    targs = TrainingArguments(
        output_dir=str(tmp_path / "t"), use_cpu=True, report_to=[],
        per_device_train_batch_size=2, average_tokens_across_devices=True,
        remove_unused_columns=False, dataloader_num_workers=0,
    )
    trainer = make_weighted_trainer_cls()(
        model=_tiny_model(), args=targs,
        train_dataset=Dataset.from_list(rows or [_row("s", "solved")]),
        data_collator=WeightedCausalCollator(0, W, cliff_mode=cl.enabled),
        cliff_cfg=cl,
    )
    return trainer, cl


CLIFF_ON = ["train.sft.cliff.enabled=true", "filter.selection.always_score=true",
            "train.sft.cliff.negative.mode=v1", "train.objective=sft",
            "train.sft.cliff.rho=0.3", "train.sft.cliff.negative.mu=0.5"]


def _window_batches(cliff_mode):
    col = WeightedCausalCollator(0, W, cliff_mode=cliff_mode)
    rows = [
        [_row("s1", "solved", c=3), _row("s2", "solved", c=4)],
        [_row("i1", "improved", n_q=2, ref=0.001, c=4),      # guard hinge active
         _row("n1", "negative", n_q=1, c=3)],
    ]
    return [col(b) for b in rows]


def _reference_loss(model, batches, cl):
    """Independent recomputation of the composed loss from raw logits."""
    import torch.nn.functional as F
    num = {"S": 0.0, "C": 0.0, "N": 0.0, "G": 0.0}
    den = {"S": 0.0, "C": 0.0, "N": 0.0, "G": 0.0}
    for b in batches:
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        logits = out.logits[:, :-1, :].float()
        lab = b["labels"][:, 1:]
        wt = b["loss_weights"][:, 1:]
        valid = (lab != -100).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             lab.reshape(-1).clamp(min=0), reduction="none"
                             ).view(lab.shape)
        row_wce = (ce * valid * wt).sum(1)
        row_W = (wt * valid).sum(1)
        comp = b["completion_mask"][:, 1:].float()
        mean_ce = (ce * comp * valid).sum(1) / (comp * valid).sum(1).clamp(min=1)
        for i in range(lab.size(0)):
            sl = int(b["slice_ids"][i])
            nq = max(float(b["n_q"][i]), 1.0)
            if sl == 0:
                num["S"] += float(row_wce[i]); den["S"] += float(row_W[i])
            elif sl == 1:
                num["C"] += float(row_wce[i]) / (nq * float(row_W[i]))
                den["C"] += 1.0 / nq
                ref = float(b["ref_mean_nll"][i])
                if ref >= 0:
                    num["G"] += max(0.0, float(mean_ce[i]) - ref) / nq
                    den["G"] += 1.0 / nq
            else:
                p = torch.exp(-ce[i]).clamp(max=1 - cl.negative.delta)
                u = (-torch.log1p(-p) * valid[i] * wt[i]).sum()
                num["N"] += float(u) / (nq * float(row_W[i]))
                den["N"] += 1.0 / nq
    t = {k: (num[k] / den[k] if den[k] > 0 else 0.0) for k in num}
    return (1 - cl.rho) * t["S"] + cl.rho * (t["C"] + cl.negative.mu * t["N"] + t["G"])


def test_composed_loss_matches_reference(tmp_path):
    trainer, cl = _mk_trainer(tmp_path, CLIFF_ON)
    batches = _window_batches(cliff_mode=True)
    scalar = trainer._get_num_items_in_batch([dict(b) for b in batches], torch.device("cpu"))
    assert trainer._window_denoms is not None and scalar > 0
    total = sum(
        float(trainer.compute_loss(trainer.model, dict(b), num_items_in_batch=scalar).detach())
        for b in batches
    )
    expected = _reference_loss(trainer.model, batches, cl)
    assert total == pytest.approx(expected, rel=1e-5)


def test_s3_tok_normalization(tmp_path):
    trainer, cl = _mk_trainer(tmp_path, CLIFF_ON + ["train.sft.cliff.per_question_norm=false"])
    batches = _window_batches(cliff_mode=True)
    trainer._get_num_items_in_batch([dict(b) for b in batches], torch.device("cpu"))
    # token-normalized cliff denominator = the cliff rows' weight-sum, not 1/n_q units
    b = batches[1]
    valid = b["labels"][:, 1:].ne(-100)
    row_W = (b["loss_weights"][:, 1:] * valid).sum(1)
    assert float(trainer._window_denoms["C"]) == pytest.approx(float(row_W[0]))


def test_zero_cliff_window_no_nan(tmp_path):
    trainer, cl = _mk_trainer(tmp_path, CLIFF_ON)
    col = WeightedCausalCollator(0, W, cliff_mode=True)
    batches = [col([_row("s1", "solved"), _row("s2", "solved", c=4)])]
    scalar = trainer._get_num_items_in_batch([dict(b) for b in batches], torch.device("cpu"))
    loss = trainer.compute_loss(trainer.model, dict(batches[0]), num_items_in_batch=scalar)
    assert torch.isfinite(loss)


def test_legacy_reproduction(tmp_path):
    """cliff.enabled=false must reproduce the current loss exactly."""
    trainer, _ = _mk_trainer(tmp_path)          # defaults: cliff off
    assert trainer._cliff is None and trainer._train_sampler is None
    col = WeightedCausalCollator(0, W)          # legacy collator
    batch = col([_row("s1", "solved"), _row("i1", "improved", c=4)])
    scalar = trainer._get_num_items_in_batch([dict(batch)], torch.device("cpu"))
    valid = batch["labels"][:, 1:].ne(-100)
    assert float(scalar) == pytest.approx(
        float((batch["loss_weights"][:, 1:] * valid).sum()))
    loss = trainer.compute_loss(trainer.model, dict(batch), num_items_in_batch=scalar)
    # reference: sum(w*ce)/weight-sum
    import torch.nn.functional as F
    out = trainer.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :].float()
    lab = batch["labels"][:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                         lab.reshape(-1).clamp(min=0), reduction="none").view(lab.shape)
    ref = float(((ce * valid * batch["loss_weights"][:, 1:]).sum() / scalar).detach())
    assert float(loss) == pytest.approx(ref, rel=1e-6)


def test_rank_shard_denominator_additivity(tmp_path):
    """gather() sums per-rank local vectors; the round-robin shards' local
    denominators must add up to the full window's — the DDP invariance core."""
    trainer, _ = _mk_trainer(tmp_path, CLIFF_ON)
    col = WeightedCausalCollator(0, W, cliff_mode=True)
    micros = [col([r]) for r in [
        _row("s1", "solved"), _row("i1", "improved", n_q=2, ref=0.5, c=4),
        _row("s2", "solved", c=4), _row("n1", "negative", n_q=1),
    ]]
    def denoms(batch_list):
        trainer._window_denoms = None
        trainer._get_num_items_in_batch([dict(b) for b in batch_list], torch.device("cpu"))
        return {k: float(v) for k, v in trainer._window_denoms.items()}
    full = denoms(micros)
    r0, r1 = denoms(micros[0::2]), denoms(micros[1::2])
    for k in full:
        assert full[k] == pytest.approx(r0[k] + r1[k])


def test_sampler_scarce_solved_cliff_fill():
    """0 (or too few) solved rows: fill slots cycle the cliff order instead of
    erroring — the smoke-scale / cliff-only dataset case."""
    s = StratifiedWindowSampler(
        solved_idx=[], cliff_idx=[100, 101, 102, 103], neg_idx_by_qid={},
        cliff_qids=["a", "b", "c", "d"], window=8, m_c=1, m_n=0, seed=0)
    stream = list(iter(s))
    assert len(stream) == len(s) == 8                     # 1 window, all cliff
    assert set(stream) == {100, 101, 102, 103}
    # a few solved rows: each used exactly once, remainder cliff-filled
    s2 = StratifiedWindowSampler(
        solved_idx=[0, 1, 2], cliff_idx=[100, 101], neg_idx_by_qid={},
        cliff_qids=["a", "b"], window=8, m_c=1, m_n=0, seed=0)
    stream2 = list(iter(s2))
    assert len(stream2) == len(s2) == 8
    assert sorted(i for i in stream2 if i < 100) == [0, 1, 2]
    # nothing at all -> still an actionable error
    with pytest.raises(ValueError, match="too small"):
        StratifiedWindowSampler(
            solved_idx=[], cliff_idx=[], neg_idx_by_qid={}, cliff_qids=[],
            window=8, m_c=1, m_n=0, seed=0)


def test_negative_rows_eos_ablation_flag(tmp_path):
    """negative.drop_terminal_eos is the paired ablation leg (spec §1): vLLM
    already includes the stop token in response_token_ids, so 'not calling
    ensure_eos' does NOT keep EOS out of the unlikelihood. Default false keeps
    it (every arm so far); true drops it (measured motivation: keeping it gave
    +57% mean generation length and 4x truncation on held-out cliffs)."""
    import expert_iter.build_dataset as bd

    EOS = 99
    it = tmp_path / "iter_0"
    (it / "rollout").mkdir(parents=True)
    (it / "partition").mkdir()
    RolloutSample.dump_jsonl(it / "rollout" / "rollouts.jsonl", [
        RolloutSample(qid="q1", sample_idx=0, prompt_text="p", prompt_token_ids=[1, 2],
                      response_text="r", response_token_ids=[7, 8, EOS], finish_reason="stop"),
        RolloutSample(qid="q1", sample_idx=1, prompt_text="p", prompt_token_ids=[1, 2],
                      response_text="r", response_token_ids=[7, 9, EOS, EOS], finish_reason="stop"),
    ])
    VerdictRecord.dump_jsonl(it / "partition" / "verdicts.jsonl", [
        VerdictRecord(qid="q1", sample_idx=0, correct=False, extracted_answer="7"),
        VerdictRecord(qid="q1", sample_idx=1, correct=False, extracted_answer="7"),
    ])
    modal = bd._modal_wrong_failures(it, {"q1"})
    assert len(modal["q1"]) == 2
    assert Config.load(None).train.sft.cliff.negative.drop_terminal_eos is False
    for b in modal["q1"]:
        raw = list(b.response_token_ids)
        assert raw[-1] == EOS, "vLLM stop samples carry the stop token"
        dropped = list(raw)
        while dropped and dropped[-1] == EOS:
            dropped.pop()
        assert EOS not in dropped and dropped, "drop leg removes all trailing EOS, keeps content"


# ---------------------------------------------------------------------------
# m_per_batch: auto
#
# An epoch is sized by the SOLVED pool, so a fixed m_C decides how many rescue
# trajectories are ever seen only by accident. MEASURED: L3 (118 improved rows,
# 173 windows) over-sampled at m_C=1, while a 6k L5 mix would show ~24% of its
# rescue rows per epoch and a 300-question smoke 4%.
# ---------------------------------------------------------------------------

def test_auto_m_c_covers_every_improved_row():
    from expert_iter.train import StratifiedWindowSampler as S

    for n_solved, n_cliff in ((5366, 118), (526, 61), (1237, 110), (194, 153), (5724, 780)):
        m = S._auto_m_c(n_solved, n_cliff, window=32, m_n=0)
        n_win = n_solved // (32 - m)
        assert n_win * m >= n_cliff, (n_solved, n_cliff, m)
        if m > 1:                                    # and it is the SMALLEST such m
            prev_win = n_solved // (32 - (m - 1))
            assert prev_win * (m - 1) < n_cliff


def test_auto_m_c_matches_the_validated_l3_setting():
    """L3's ratio already over-sampled at m_C=1, so auto must not disturb it."""
    from expert_iter.train import StratifiedWindowSampler as S

    assert S._auto_m_c(5366, 118, window=32, m_n=0) == 1


def test_auto_m_c_keeps_a_fill_slot_and_reports_short_coverage():
    """A window of pure cliff rows would starve L_S, so m_C is clamped — and
    then coverage is honestly below 1."""
    from expert_iter.train import StratifiedWindowSampler as S

    m = S._auto_m_c(n_solved=40, n_cliff=100_000, window=32, m_n=0)
    assert m == 31                                   # window - m_n - 1
    sampler = S(solved_idx=list(range(40)), cliff_idx=list(range(100_000)),
                neg_idx_by_qid={}, cliff_qids=[f"q{i}" for i in range(100_000)],
                window=32, m_c="auto", m_n=0, seed=1)
    assert sampler.m_c == 31
    assert sampler.cliff_coverage() < 1.0


def test_auto_m_c_resolves_inside_the_sampler_and_draws_every_row():
    from expert_iter.train import StratifiedWindowSampler as S

    n_solved, n_cliff = 526, 61
    sampler = S(solved_idx=list(range(n_solved)),
                cliff_idx=list(range(10_000, 10_000 + n_cliff)),
                neg_idx_by_qid={}, cliff_qids=[f"q{i}" for i in range(n_cliff)],
                window=32, m_c="auto", m_n=0, seed=7)
    assert sampler.m_c == 4 and sampler.cliff_coverage() == 1.0
    drawn = [i for i in sampler if i >= 10_000]
    assert set(drawn) == set(range(10_000, 10_000 + n_cliff))   # every row, no gaps
