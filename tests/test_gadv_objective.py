"""Group-advantage objective (train.objective: gadv): the pure advantage
function, the two-pass dataset builder, the collator channels, the clipped
surrogate trainer mechanics and the theta0 pre-pass.

CPU-only. The loss is checked against an independent recomputation from the
same tiny model's logits (at rho == 1 it must also equal the clip-free form),
the clip is checked as a literal zero gradient outside the band, and the DDP
property as rank-shard additivity of the gathered denominator vector.
"""

from __future__ import annotations

import copy
import random

import pytest
import torch

from expert_iter.config import Config
from expert_iter.gadv import build_gadv_examples, group_advantages
from expert_iter.records import ImprovedCandidate, RolloutSample, SFTExample, VerdictRecord
from expert_iter.train import (
    GadvCollator,
    WeightedCausalCollator,
    make_gadv_prepass_callback,
    make_gadv_trainer_cls,
)

W = {"prompt": 0.0, "anchor": 0.0, "continuation": 1.0, "solution": 1.0}
GADV_ON = ["train.objective=gadv", "filter.selection.always_score=true"]


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _gcfg(**ov):
    cfg = Config.load(None, overrides=GADV_ON + [f"train.gadv.{k}={v}" for k, v in ov.items()])
    return cfg.train.gadv


# ---------------------------------------------------------------------------
# pure advantage function
# ---------------------------------------------------------------------------

def test_frontier_zero_sum_signs_and_attractor_allocation():
    g = _gcfg()
    # k=2 of 8; wrong answers: "7" x3 (the attractor), "9", None (truncated), "3"
    m = [(0, True, "42"), (1, True, "42"), (2, False, "7"), (3, False, "7"),
         (4, False, "9"), (5, False, None), (6, False, "7"), (7, False, "3")]
    plan = group_advantages(m, 0, g, random.Random(0), n=8)
    assert plan.kind == "frontier" and plan.k == 2 and plan.p == pytest.approx(0.25)
    assert set(plan.pos) == {0, 1} and all(v == pytest.approx(0.75) for v in plan.pos.values())
    assert set(plan.neg) == {2, 3, 4, 5, 6, 7} and all(v < 0 for v in plan.neg.values())
    assert plan.pos_total + plan.neg_total == pytest.approx(0.0, abs=1e-12)
    # attractor rows carry more negative mass than the scattered answers
    assert plan.neg[2] == plan.neg[3] == plan.neg[6] < plan.neg[4] < 0
    # a None (truncated) row is its own bucket: same weight as a unique answer
    assert plan.neg[5] == pytest.approx(plan.neg[7]) and plan.neg[7] == pytest.approx(plan.neg[4])
    assert plan.answer_freq[2] == pytest.approx(0.5) and plan.answer_freq[5] == pytest.approx(1 / 6)
    assert plan.group_size == 8


def test_gamma0_reproduces_dr_grpo_minus_p():
    g = _gcfg(gamma=0)
    m = [(i, i < 3, "x" if i < 6 else str(i)) for i in range(8)]       # k=3
    plan = group_advantages(m, 0, g, random.Random(0), n=8)
    for v in plan.neg.values():
        assert v == pytest.approx(-3 / 8)
    for v in plan.pos.values():
        assert v == pytest.approx(5 / 8)
    # rescue group: 8 failures + 2 rescues, p = 2/10
    m0 = [(i, False, "x" if i < 5 else str(i)) for i in range(8)]
    plan = group_advantages(m0, 2, g, random.Random(0), n=8)
    assert plan.kind == "rescue" and plan.p == pytest.approx(0.2) and plan.group_size == 10
    assert plan.rescue_adv == pytest.approx(0.8) and not plan.pos
    for v in plan.neg.values():
        assert v == pytest.approx(-0.2)
    assert plan.pos_total + plan.neg_total == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("gamma", [0.0, 1.0, 2.5])
def test_same_answer_and_all_none_are_uniform(gamma):
    g = _gcfg(gamma=gamma)
    same = [(0, True, "1")] + [(i, False, "7") for i in range(1, 8)]
    plan = group_advantages(same, 0, g, random.Random(0), n=8)
    assert len(set(round(v, 12) for v in plan.neg.values())) == 1
    assert plan.neg_total == pytest.approx(-plan.pos_total)
    nones = [(0, True, "1")] + [(i, False, None) for i in range(1, 8)]
    plan = group_advantages(nones, 0, g, random.Random(0), n=8)
    assert len(set(round(v, 12) for v in plan.neg.values())) == 1
    assert all(f == pytest.approx(1 / 7) for f in plan.answer_freq.values())


def test_rescue_dose_and_neg_scale():
    g = _gcfg(rescue_dose=0.5, neg_scale=0.3)
    m0 = [(i, False, "x") for i in range(8)]
    plan = group_advantages(m0, 2, g, random.Random(0), n=8)
    assert plan.rescue_adv == pytest.approx(0.8 * 0.5)
    assert plan.neg_total == pytest.approx(-0.3 * 2 * 0.4)


def test_exclusion_floor_and_caps_are_deterministic():
    g = _gcfg()
    solved = [(i, True, "1") for i in range(8)]
    assert group_advantages(solved, 0, g, random.Random(0), n=8).kind == "excluded"
    assert group_advantages(solved, 0, g, random.Random(0), n=8).reason == "k==n"
    gf = _gcfg(solved_floor=0.5, solved_floor_max_per_question=2)
    plan = group_advantages(solved, 0, gf, random.Random(0), n=8)
    assert plan.kind == "floor" and len(plan.pos) == 2 and not plan.neg
    assert all(v == 0.5 for v in plan.pos.values())
    dead = [(i, False, "x") for i in range(8)]
    plan = group_advantages(dead, 0, g, random.Random(0), n=8)
    assert plan.kind == "excluded" and plan.reason == "k==0,no_rescue"
    # caps: 2 of 4 correct rows, 3 of 4 wrong rows; zero-sum on what is trained
    gc = _gcfg(correct_max_per_question=2, wrong_max_per_question=3)
    m = [(i, i < 4, "a" if i >= 4 else "ok") for i in range(8)]
    p1 = group_advantages(m, 0, gc, random.Random(5), n=8)
    p2 = group_advantages(m, 0, gc, random.Random(5), n=8)
    assert len(p1.pos) == 2 and len(p1.neg) == 3
    assert p1.pos == p2.pos and p1.neg == p2.neg                     # seeded => reproducible
    assert p1.pos_total == pytest.approx(2 * 0.5)
    assert p1.neg_total == pytest.approx(-1.0)
    assert group_advantages(m, 0, gc, random.Random(6), n=8).pos != p1.pos


# ---------------------------------------------------------------------------
# dataset builder on a tmp iter dir (rollout.n = 4)
# ---------------------------------------------------------------------------

def _rollout(qid, idx, finish="stop", resp=None):
    resp = resp if resp is not None else [10 + idx, 11, 12]
    return RolloutSample(qid=qid, sample_idx=idx, prompt_text="p", prompt_token_ids=[1, 2],
                         response_text="r", response_token_ids=resp, finish_reason=finish)


def _verdict(qid, idx, correct, ans):
    return VerdictRecord(qid=qid, sample_idx=idx, correct=correct, extracted_answer=ans)


def _cand(qid, base_idx, attempt):
    return ImprovedCandidate(qid=qid, base_sample_idx=base_idx, attempt_idx=attempt,
                             prompt_token_ids=[1, 2], anchor_token_ids=[],
                             continuation_token_ids=[5, 6], continuation_text="c")


def _write_iter_dir(root):
    rollouts, verdicts = [], []

    def q(qid, spec):   # spec: list of (correct, answer, finish)
        for i, (ok, ans, fin) in enumerate(spec):
            rollouts.append(_rollout(qid, i, finish=fin))
            verdicts.append(_verdict(qid, i, ok, ans))

    q("qA", [(True, "1", "stop")] * 4)                                          # 4/4 -> excluded
    q("qB", [(True, "1", "stop"), (True, "1", "stop"),
             (False, "7", "stop"), (False, "7", "stop")])                       # frontier k=2
    q("qC", [(False, "5", "stop"), (False, "5", "stop"),
             (False, "6", "stop"), (False, "6", "length")])                     # cliff + rescue
    q("qD", [(False, "9", "stop")] * 4)                                          # cliff, no rescue
    q("qE", [(True, "1", "length"), (False, "2", "stop"),
             (False, "2", "stop"), (False, "2", "stop")])                       # truncated-correct => k=0
    (root / "rollout").mkdir(parents=True)
    (root / "partition").mkdir(parents=True)
    RolloutSample.dump_jsonl(root / "rollout" / "rollouts.jsonl", rollouts)
    VerdictRecord.dump_jsonl(root / "partition" / "verdicts.jsonl", verdicts)


def _cfg4(*extra):
    return Config.load(None, overrides=GADV_ON + ["rollout.n=4", *extra])


def test_build_gadv_examples_groups_rows_and_stats(tmp_path):
    _write_iter_dir(tmp_path)
    cfg = _cfg4()
    kept = [_cand("qC", 0, 0)]
    rows, stats = build_gadv_examples(cfg, tmp_path, 0, eos=99, kept=kept, refs={"qC:0:0": 0.42})
    for r in rows:
        r.validate()
    by = {}
    for r in rows:
        by.setdefault((r.qid, r.source), []).append(r)
    assert not any(qid in ("qA", "qD", "qE") for qid, _ in by)
    # frontier qB: both correct rows at 1-p, EOS appended; both wrong rows uniform
    pos = by[("qB", "solved")]
    assert len(pos) == 2 and all(r.advantage == pytest.approx(0.5) for r in pos)
    assert all(r.input_ids[-1] == 99 and r.completion_len == 4 for r in pos)
    neg = by[("qB", "wrong")]
    assert len(neg) == 2 and all(r.advantage == pytest.approx(-0.5) for r in neg)
    assert all(r.input_ids[-1] != 99 and r.completion_len == 3 for r in neg)   # no EOS appended
    assert all(r.group_kind == "frontier" and r.group_size == 4 for r in pos + neg)
    # rescue group qC: rescue A = 1 - 1/5, guard ref joined, failures ~ answer frequency
    resc = by[("qC", "improved")]
    assert len(resc) == 1 and resc[0].advantage == pytest.approx(0.8)
    assert resc[0].ref_mean_nll == 0.42 and resc[0].group_size == 5 and resc[0].group_kind == "rescue"
    assert resc[0].input_ids == [1, 2, 5, 6, 99]
    fails = {r.uid: r for r in by[("qC", "wrong")]}
    assert len(fails) == 4
    advs = sorted(r.advantage for r in fails.values())
    # answers: "5" x2 (f=.5), "6" (f=.25), truncated None (f=.25) -> total -0.8
    assert sum(advs) == pytest.approx(-0.8)
    assert advs[0] == advs[1] == pytest.approx(-0.8 * 0.5 / 1.5)
    assert advs[2] == advs[3] == pytest.approx(-0.8 * 0.25 / 1.5)
    # stats
    assert stats["questions"] == {"frontier": 1, "rescue": 1, "excluded_k==0,no_rescue": 2,
                                  "excluded_k==n": 1}
    assert stats["rows"] == {"improved": 1, "solved": 2, "wrong": 6}
    assert stats["zero_sum_max_abs_residual"] < 1e-9
    assert stats["n_truncated_as_wrong"] == 2 and stats["n_none_bucket_rows"] == 1
    assert stats["n_ref_joined"] == 1 and stats["n_ref_missing"] == 0
    assert stats["k_hist"] == {"0": 3, "2": 1, "4": 1}


def test_build_gadv_examples_floor_and_drop_eos(tmp_path):
    _write_iter_dir(tmp_path)
    cfg = _cfg4("train.gadv.solved_floor=0.25", "train.gadv.wrong_drop_terminal_eos=true")
    rows, stats = build_gadv_examples(cfg, tmp_path, 0, eos=12, kept=[], refs={})
    floor = [r for r in rows if r.qid == "qA"]
    assert len(floor) == 1 and floor[0].source == "solved" and floor[0].advantage == 0.25
    assert floor[0].group_kind == "floor"
    # every wrong row's response ended with 12 (= eos here) -> stripped
    assert all(r.input_ids[-1] != 12 and r.completion_len == 2 for r in rows if r.source == "wrong")
    assert stats["questions"]["floor"] == 1 and "rescue" not in stats["questions"]


def test_build_gadv_examples_truncated_cap(tmp_path):
    """wrong_truncated_max_per_question caps the 16k answer-less failures
    BEFORE the group is planned: n (and so p) and the negative total are
    unchanged, the mass moves onto the surviving rows, and the default is
    byte-identical to the uncapped builder."""
    _write_iter_dir(tmp_path)
    kept = [_cand("qC", 0, 0)]
    base, base_stats = build_gadv_examples(_cfg4(), tmp_path, 0, eos=99, kept=kept, refs={})
    same, same_stats = build_gadv_examples(_cfg4("train.gadv.wrong_truncated_max_per_question=4"),
                                           tmp_path, 0, eos=99, kept=kept, refs={})
    assert [(r.uid, r.advantage) for r in same] == [(r.uid, r.advantage) for r in base]
    assert base_stats["n_truncated_capped_rows"] == 0 == same_stats["n_truncated_capped_rows"]

    rows, stats = build_gadv_examples(_cfg4("train.gadv.wrong_truncated_max_per_question=0"),
                                      tmp_path, 0, eos=99, kept=kept, refs={})
    # qC lost its truncated failure (idx 3), qE its truncated-correct row (idx 0,
    # which counts as a wrong row): both are gone, nothing else changed
    assert stats["n_truncated_capped_rows"] == 2 and stats["n_truncated_as_wrong"] == 2
    assert stats["rows"] == {"improved": 1, "solved": 2, "wrong": 5}
    qc = [r for r in rows if r.qid == "qC" and r.source == "wrong"]
    assert len(qc) == 3 and all(r.group_size == 5 for r in qc)          # n stays 4 (+1 rescue)
    advs = sorted(r.advantage for r in qc)
    # answers now "5" x2 (f=2/3), "6" (f=1/3): total still -0.8, no None bucket
    assert sum(advs) == pytest.approx(-0.8)
    assert advs[0] == advs[1] == pytest.approx(-0.8 * (2 / 3) / (5 / 3))
    assert advs[2] == pytest.approx(-0.8 * (1 / 3) / (5 / 3))
    assert stats["n_none_bucket_rows"] == 0
    assert stats["zero_sum_max_abs_residual"] < 1e-9
    # qB (no truncated rows) is untouched
    qb = {r.uid: r.advantage for r in rows if r.qid == "qB"}
    assert qb == {r.uid: r.advantage for r in base if r.qid == "qB"}


# ---------------------------------------------------------------------------
# collator
# ---------------------------------------------------------------------------

def _row(uid, source, adv, row_idx, n_q=0, ref=None, p=2, a=0, c=3, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(1, 60, (p + a + c,)).tolist()
    return {"uid": uid, "source": source, "input_ids": ids, "prompt_len": p, "anchor_len": a,
            "completion_len": c, "n_q": n_q, "ref_mean_nll": ref, "advantage": adv,
            "row_idx": row_idx}


def test_gadv_collator_channels():
    rows = [_row("s", "solved", 0.5, 0, c=3), _row("i", "improved", 0.8, 1, n_q=2, ref=0.3, a=1, c=4),
            _row("w", "wrong", -0.4, 2, c=2)]
    out = GadvCollator(0, W)(rows)
    legacy = WeightedCausalCollator(0, W, cliff_mode=False)(rows)
    for k in ("input_ids", "attention_mask", "labels", "loss_weights"):
        assert torch.equal(out[k], legacy[k])
    assert out["advantage"].tolist() == pytest.approx([0.5, 0.8, -0.4])
    assert out["row_idx"].tolist() == [0, 1, 2]
    assert out["slice_ids"].tolist() == [0, 1, 2]
    assert out["n_q"].tolist() == [0.0, 2.0, 0.0]
    assert out["ref_mean_nll"].tolist() == pytest.approx([-1.0, 0.3, -1.0])
    assert out["completion_mask"].shape == out["input_ids"].shape
    assert out["completion_mask"][0].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert out["completion_mask"][1].tolist() == [0, 0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="unknown source"):
        GadvCollator(0, W)([_row("n", "negative", -1.0, 0)])


# ---------------------------------------------------------------------------
# trainer mechanics (tiny model, CPU)
# ---------------------------------------------------------------------------

def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(1234)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32))


def _rows():
    return [_row("s1", "solved", 0.75, 0, c=3, seed=1), _row("s2", "solved", 0.5, 1, c=4, seed=2),
            _row("i1", "improved", 0.8, 2, n_q=2, ref=0.001, a=1, c=4, seed=3),   # hinge active
            _row("i2", "improved", 0.8, 3, n_q=2, ref=None, c=2, seed=4),         # ref missing
            _row("w1", "wrong", -0.25, 4, c=3, seed=5), _row("w2", "wrong", -0.6, 5, c=5, seed=6)]


def _mk_trainer(tmp_path, rows, overrides=(), model=None):
    from datasets import Dataset
    from transformers import TrainingArguments

    cfg = Config.load(None, overrides=GADV_ON + list(overrides))
    targs = TrainingArguments(
        output_dir=str(tmp_path / "t"), use_cpu=True, report_to=[],
        per_device_train_batch_size=2, average_tokens_across_devices=True,
        remove_unused_columns=False, dataloader_num_workers=0,
    )
    collator = GadvCollator(0, W)
    trainer = make_gadv_trainer_cls()(
        model=model or _tiny_model(), args=targs, train_dataset=Dataset.from_list(rows),
        data_collator=collator, gadv_cfg=cfg.train.gadv,
    )
    return trainer, collator, cfg.train.gadv


def _prepass(trainer, collator, g):
    cb = make_gadv_prepass_callback(trainer, collator, batch_size=g.prepass_batch_size,
                                    cache_dtype=g.cache_dtype)
    cb.on_train_begin(trainer.args, trainer.state, trainer.control)
    return cb


def _batches(collator, rows):
    return [collator(rows[0:2]), collator(rows[2:4]), collator(rows[4:6])]


@torch.no_grad()
def _reference(model, batches, rows, g, old=None):
    """Independent recomputation from raw logits. old=None -> clip-free form
    (-A * log pi); with old -> the clipped surrogate, whose VALUE at rho == 1
    is -A per token (only its gradient equals the clip-free form)."""
    import torch.nn.functional as F
    num = n_tok = num_g = d_g = 0.0
    for b in batches:
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        logits = out.logits[:, :-1, :].float()
        lab = b["labels"][:, 1:]
        wt = b["loss_weights"][:, 1:].float()
        comp = b["completion_mask"][:, 1:].bool()
        valid = (lab != -100) & comp
        logp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab.reshape(-1).clamp(min=0),
                                reduction="none").view(lab.shape)
        for i in range(lab.size(0)):
            r = rows[int(b["row_idx"][i])]
            A = float(r["advantage"])
            m = valid[i].float() * wt[i]
            if old is None:
                num += float((-A * logp[i] * m).sum())
            else:
                o = torch.zeros_like(logp[i]); o[comp[i]] = old[int(b["row_idx"][i])].float()
                ratio = torch.exp(logp[i] - o)
                surr = torch.minimum(ratio * A, ratio.clamp(1 - g.clip.eps_lo, 1 + g.clip.eps_hi) * A)
                num += float((-surr * m).sum())
            n_tok += float(m.sum())
            if r["source"] == "improved" and r["ref_mean_nll"] is not None:
                mean_ce = float(-(logp[i] * valid[i]).sum() / valid[i].sum())
                num_g += max(0.0, mean_ce - r["ref_mean_nll"]) / r["n_q"]
                d_g += 1.0 / r["n_q"]
    return num / n_tok + g.guard_weight * (num_g / d_g if d_g > 0 else 0.0)


def test_gadv_loss_matches_reference_at_ratio_one(tmp_path):
    rows = _rows()
    trainer, collator, g = _mk_trainer(tmp_path, rows)
    _prepass(trainer, collator, g)
    batches = _batches(collator, rows)
    scalar = trainer._get_num_items_in_batch([dict(b) for b in batches], torch.device("cpu"))
    assert trainer._window_denoms is not None and float(scalar) > 0
    total = sum(float(trainer.compute_loss(trainer.model, dict(b), num_items_in_batch=scalar).detach())
                for b in batches)
    assert total == pytest.approx(_reference(trainer.model, batches, rows, g, old=trainer._old_logp), rel=1e-5)
    # at rho == 1 the surrogate's value is -sum(A*m)/N_tok + guard (its gradient
    # is the clip-free one — checked in the next test)
    n_tok = sum(r["completion_len"] for r in rows)
    main = -sum(r["advantage"] * r["completion_len"] for r in rows) / n_tok
    guard = _reference(trainer.model, batches, rows, g, old=None) - _reference(
        trainer.model, batches, rows, Config.load(None, overrides=GADV_ON + ["train.gadv.guard_weight=0"]).train.gadv, old=None)
    assert total == pytest.approx(main + guard, rel=1e-5)
    # clip disabled: the trainer is the plain -A*log pi form
    tr_free, col_free, g_free = _mk_trainer(tmp_path, rows, overrides=["train.gadv.clip.enabled=false"],
                                            model=copy.deepcopy(trainer.model))
    scalar_f = tr_free._get_num_items_in_batch([dict(b) for b in batches], torch.device("cpu"))
    total_f = sum(float(tr_free.compute_loss(tr_free.model, dict(b), num_items_in_batch=scalar_f).detach())
                  for b in batches)
    assert total_f == pytest.approx(_reference(tr_free.model, batches, rows, g_free, old=None), rel=1e-5)
    # monitoring: no clipping happened, ratio is 1, masses match the window vector
    c = trainer._comp_sums
    assert c["_clip_hit_pos"] == 0 and c["_clip_hit_neg"] == 0
    assert c["_ratio_sum"] / c["_ratio_tok"] == pytest.approx(1.0, abs=1e-5)
    D = trainer._window_denoms
    assert float(D["n_pos"]) == 4 and float(D["n_neg"]) == 2
    assert float(D["pos_mass"]) == pytest.approx(0.75 * 3 + 0.5 * 4 + 0.8 * 4 + 0.8 * 2)
    assert float(D["neg_mass"]) == pytest.approx(0.25 * 3 + 0.6 * 5)


def test_gadv_gradient_at_ratio_one_equals_clip_free_gradient(tmp_path):
    rows = _rows()
    base = _tiny_model()
    tr_clip, col, g = _mk_trainer(tmp_path, rows, model=copy.deepcopy(base))
    tr_free, _, _ = _mk_trainer(tmp_path, rows, overrides=["train.gadv.clip.enabled=false"],
                                model=copy.deepcopy(base))
    _prepass(tr_clip, col, g)
    batches = _batches(col, rows)
    grads = []
    for tr in (tr_clip, tr_free):
        scalar = tr._get_num_items_in_batch([dict(b) for b in batches], torch.device("cpu"))
        loss = sum(tr.compute_loss(tr.model, dict(b), num_items_in_batch=scalar) for b in batches)
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in tr.model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and grads[0]
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], atol=1e-6, rtol=1e-4), n


def test_gadv_clip_zeroes_gradient_outside_band(tmp_path):
    rows = _rows()
    trainer, collator, g = _mk_trainer(tmp_path, rows)
    _prepass(trainer, collator, g)
    # push a positive row above 1+eps_hi (old logp lower => ratio e^1) and a
    # negative row below 1-eps_lo (old logp higher => ratio e^-1)
    trainer._old_logp[0] = trainer._old_logp[0] - 1.0
    trainer._old_logp[4] = trainer._old_logp[4] + 1.0
    for i in (0, 4):
        b = collator([rows[i]])
        trainer.model.zero_grad()
        scalar = trainer._get_num_items_in_batch([dict(b)], torch.device("cpu"))
        loss = trainer.compute_loss(trainer.model, dict(b), num_items_in_batch=scalar)
        loss.backward()
        assert all(p.grad is None or torch.all(p.grad == 0) for p in trainer.model.parameters())
    c = trainer._comp_sums
    assert c["_clip_hit_pos"] == pytest.approx(c["_clip_mass_pos"]) and c["_clip_mass_pos"] > 0
    assert c["_clip_hit_neg"] == pytest.approx(c["_clip_mass_neg"]) and c["_clip_mass_neg"] > 0
    # an unperturbed row still trains
    trainer.model.zero_grad()
    b = collator([rows[1]])
    scalar = trainer._get_num_items_in_batch([dict(b)], torch.device("cpu"))
    trainer.compute_loss(trainer.model, dict(b), num_items_in_batch=scalar).backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in trainer.model.parameters())


def test_gadv_rank_shard_denominator_additivity(tmp_path):
    rows = _rows()
    trainer, collator, g = _mk_trainer(tmp_path, rows)
    micros = [collator([r]) for r in rows]
    trainer._get_num_items_in_batch([dict(b) for b in micros], torch.device("cpu"))
    full = {k: float(v) for k, v in trainer._window_denoms.items()}
    trainer._get_num_items_in_batch([dict(b) for b in micros[0::2]], torch.device("cpu"))
    r0 = {k: float(v) for k, v in trainer._window_denoms.items()}
    trainer._get_num_items_in_batch([dict(b) for b in micros[1::2]], torch.device("cpu"))
    r1 = {k: float(v) for k, v in trainer._window_denoms.items()}
    for k in full:
        assert full[k] == pytest.approx(r0[k] + r1[k]), k
    assert full["N"] == pytest.approx(sum(r["completion_len"] for r in rows))
    assert full["G"] == pytest.approx(1 / 2)          # one guarded improved row with n_q=2


def test_gadv_prepass_row_alignment_and_skip(tmp_path):
    rows = _rows()
    trainer, collator, g = _mk_trainer(tmp_path, rows, overrides=["train.gadv.prepass_batch_size=2"])
    _prepass(trainer, collator, g)
    assert len(trainer._old_logp) == len(rows)
    for i, r in enumerate(rows):
        assert trainer._old_logp[i].numel() == r["completion_len"]
        assert trainer._old_logp[i].dtype == torch.float32
        assert torch.all(trainer._old_logp[i] <= 0)
    assert list(trainer.train_dataset["row_idx"]) == list(range(len(rows)))
    # a second call is a no-op; clip disabled -> no pre-pass at all
    cache = trainer._old_logp
    _prepass(trainer, collator, g)
    assert trainer._old_logp is cache
    tr2, col2, g2 = _mk_trainer(tmp_path, rows, overrides=["train.gadv.clip.enabled=false"])
    _prepass(tr2, col2, g2)
    assert tr2._old_logp is None
    # batch-size-2 pre-pass equals batch-size-1 (padding must not change logp)
    tr1, col1, g1 = _mk_trainer(tmp_path, rows, model=copy.deepcopy(trainer.model))
    _prepass(tr1, col1, g1)
    for i in range(len(rows)):
        assert torch.allclose(tr1._old_logp[i], trainer._old_logp[i], atol=1e-5)
