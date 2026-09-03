"""Stage: train — ⚗ extension point (IV): the training objective.

Runs under `accelerate launch` (loop.py passes --num_processes and the backend
config file). Reads iter_k/dataset/train_{sft,dpo}.jsonl, writes iter_k/ckpt/
as a full HF-format checkpoint that vLLM loads directly next iteration.

Objectives:
  sft      WeightedSFT — per-token region-weighted cross entropy.
  dpo      Anchor-conditioned DPO (prompt = question + anchor).
  sft+dpo  SFT first, then DPO initialized from the SFT result.
  gadv     Group-advantage SFT (docs/objective_gadv_spec_20260903.md): every
           row carries a signed advantage A from its question's group
           (build_dataset/gadv.py); per-token loss = PPO-clipped surrogate
           -min(rho*A, clip(rho)*A) with rho = pi_theta/pi_theta0, theta0 =
           the weights the trainer starts from (per-token log-probs cached by
           a no-grad pre-pass in on_train_begin), token-mean over the global
           window. Separate GadvTrainer/GadvCollator — the sft paths are untouched.

Why WeightedSFTTrainer subclasses transformers.Trainer, not trl.SFTTrainer:
our examples are pre-tokenized with region annotations, so SFTTrainer's
dataset preparation/packing pipeline is unused surface area — and on the
pinned bleeding-edge trl it is the most version-volatile part. The collator +
compute_loss below are the whole custom surface. DPO does use trl.DPOTrainer.

Loss normalization contract: compute_loss returns SUM(w * ce) / num_items,
where num_items is the sum of loss weights over the FULL global batch (all
grad-accum micro-steps; transformers averages across ranks when
average_tokens_across_devices is on). get_batch_samples is overridden to
compute that weight-sum, so `loss` is invariant to micro-batch/accum topology.
With the default 0/1 region weights this equals standard token-count
normalization; fractional region weights stay correctly normalized too.

Cliff objective (train.sft.cliff.enabled — docs/objective_decision_20260823.md §3):
L = (1-rho)·L_S + rho·(L_C + mu·L_N + L_G) with SEPARATE per-slice normalizers
(solved token-sum / cliff per-question 1/n_q units / negative units / guard
units), gathered once per window in _get_num_items_in_batch and stashed on the
trainer; a StratifiedWindowSampler puts exactly m_c cliff (+ m_n negative) rows
in every global batch window so rho is a realized per-step share. Disabled ⇒
every code path above is byte-identical to the legacy loss.
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
from pathlib import Path

import torch

from .config import Config, load_stage_config, stage_argparser
from .utils import is_done, iter_dir, mark_done, read_jsonl


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class WeightedCausalCollator:
    """Right-pads a batch of pre-tokenized region-annotated examples.

    Emits `loss_weights` aligned with labels: weight 0 also masks the label to
    -100 (pad / prompt / zero-weight regions never contribute to CE).
    Region -> weight mapping comes from config, per example `source`:
      solved:   [prompt][solution]           -> prompt, solution weights
      improved: [prompt][anchor][completion] -> prompt, anchor, continuation
    """

    SLICE = {"solved": 0, "improved": 1, "negative": 2}

    def __init__(self, pad_token_id: int, region_weights: dict[str, float],
                 cliff_mode: bool = False):
        self.pad_token_id = pad_token_id
        self.w = region_weights
        # cliff_mode adds DATA CHANNELS only (slice ids, per-question mass, guard
        # ref, completion mask); all cliff math lives in the trainer. loss_weights
        # stay pure region weights, so the w==0 => label -100 coupling below is
        # untouched and the legacy output is byte-identical when cliff_mode=False.
        self.cliff_mode = cliff_mode

    def example_weights(self, ex: dict) -> list[float]:
        # "negative" rows share the continuation weight; their tokens are trained
        # with unlikelihood, selected by slice id — never by weight.
        completion_w = self.w["solution"] if ex["source"] == "solved" else self.w["continuation"]
        return (
            [self.w["prompt"]] * ex["prompt_len"]
            + [self.w["anchor"]] * ex["anchor_len"]
            + [completion_w] * ex["completion_len"]
        )

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(ex["input_ids"]) for ex in examples)
        input_ids, attention_mask, labels, weights = [], [], [], []
        slice_ids, n_q, refs, comp_mask = [], [], [], []
        for ex in examples:
            ids = ex["input_ids"]
            w = self.example_weights(ex)
            assert len(w) == len(ids), f"region lens != input_ids for uid={ex.get('uid')}"
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append([t if wi > 0 else -100 for t, wi in zip(ids, w)] + [-100] * pad)
            weights.append(w + [0.0] * pad)
            if self.cliff_mode:
                sl = self.SLICE.get(ex["source"], 0)
                slice_ids.append(sl)
                # clamp >= 1 defensively: rows accumulated from pre-cliff files
                # deserialize with n_q=0 (build_dataset restamps, this is belt+braces)
                n_q.append(float(max(ex.get("n_q") or 0, 1)) if sl else 0.0)
                ref = ex.get("ref_mean_nll")
                refs.append(float(ref) if ref is not None else -1.0)  # -1 = missing (NLL >= 0)
                start = ex["prompt_len"] + ex["anchor_len"]
                comp_mask.append(
                    [0] * start + [1] * ex["completion_len"] + [0] * pad
                )
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(weights, dtype=torch.float),
        }
        if self.cliff_mode:
            out["slice_ids"] = torch.tensor(slice_ids, dtype=torch.long)
            out["n_q"] = torch.tensor(n_q, dtype=torch.float)
            out["ref_mean_nll"] = torch.tensor(refs, dtype=torch.float)
            out["completion_mask"] = torch.tensor(comp_mask, dtype=torch.uint8)
        return out


class GadvCollator(WeightedCausalCollator):
    """Collator for train.objective=gadv: the legacy four channels (identical to
    WeightedCausalCollator with cliff_mode=False) plus the group-advantage data
    channels. `advantage` is emitted for inspection and for the UN-cast
    _get_num_items_in_batch hook only — compute_loss re-reads the fp32
    advantage by `row_idx`, because accelerate/DeepSpeed cast every float
    input to the model dtype (bf16) before compute_loss."""

    GADV_SLICE = {"solved": 0, "improved": 1, "wrong": 2}

    def __init__(self, pad_token_id: int, region_weights: dict[str, float]):
        super().__init__(pad_token_id, region_weights, cliff_mode=False)

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        out = super().__call__(examples)
        max_len = out["input_ids"].shape[1]
        adv, row_idx, slice_ids, n_q, refs, comp = [], [], [], [], [], []
        for ex in examples:
            sl = self.GADV_SLICE.get(ex["source"])
            if sl is None:
                raise ValueError(f"gadv: unknown source {ex['source']!r} (uid={ex.get('uid')})")
            adv.append(float(ex["advantage"]))
            row_idx.append(int(ex["row_idx"]))
            slice_ids.append(sl)
            n_q.append(float(max(ex.get("n_q") or 0, 1)) if sl == 1 else 0.0)
            ref = ex.get("ref_mean_nll")
            refs.append(float(ref) if ref is not None else -1.0)   # -1 = missing (NLL >= 0)
            start = ex["prompt_len"] + ex["anchor_len"]
            comp.append([0] * start + [1] * ex["completion_len"]
                        + [0] * (max_len - len(ex["input_ids"])))
        out["advantage"] = torch.tensor(adv, dtype=torch.float)
        out["row_idx"] = torch.tensor(row_idx, dtype=torch.long)
        out["slice_ids"] = torch.tensor(slice_ids, dtype=torch.long)
        out["n_q"] = torch.tensor(n_q, dtype=torch.float)
        out["ref_mean_nll"] = torch.tensor(refs, dtype=torch.float)
        out["completion_mask"] = torch.tensor(comp, dtype=torch.uint8)
        return out


# ---------------------------------------------------------------------------
# Stratified sampler (cliff objective)
# ---------------------------------------------------------------------------

class StratifiedWindowSampler(torch.utils.data.Sampler):
    """Global index stream where EVERY consecutive window of `window` indices
    holds exactly m_c cliff rows (+ m_n negative rows), rest solved.

    Load-bearing for the cliff term: rho is a per-STEP share only if every
    optimizer window sees the cliff slice (docs/objective_decision_20260823.md §3).
    accelerate's BatchSamplerShard deals the stream round-robin across ranks, so
    each consecutive `window`-block IS one optimizer window regardless of world
    size. The stream length is truncated to a multiple of `window` so
    even_batches padding and partial trailing windows can never occur, and every
    rank runs the same number of full windows (gather collectives stay in
    lockstep with no filler micro-batches).

    Deliberately NOT a RandomSampler subclass: accelerate swaps exact
    RandomSampler instances for its seedable variant; a custom Sampler passes
    through untouched. Deterministic from (seed, epoch), identical on all ranks.
    """

    def __init__(self, solved_idx: list[int], cliff_idx: list[int],
                 neg_idx_by_qid: dict[str, list[int]], cliff_qids: list[str],
                 window: int, m_c: int, m_n: int, seed: int):
        if m_c and not cliff_idx:
            print("[train] WARNING: cliff term enabled but no improved rows — "
                  "windows degrade to solved-only, loss = (1-rho)*L_S")
            m_c = 0
        elif m_c == "auto":
            m_c = self._auto_m_c(len(solved_idx), len(cliff_idx), window, m_n)
        self.solved_idx = list(solved_idx)
        self.cliff_idx = list(cliff_idx)
        self.neg_idx_by_qid = {q: list(v) for q, v in neg_idx_by_qid.items()}
        self.all_neg_idx = [i for v in neg_idx_by_qid.values() for i in v]
        self.cliff_qids = list(cliff_qids)          # parallel to cliff_idx
        if m_n and not self.all_neg_idx:
            print("[train] WARNING: negative.m_per_batch > 0 but no negative rows — "
                  "filling those slots with solved rows")
            m_n = 0
        self.window, self.m_c, self.m_n, self.seed = window, m_c, m_n, seed
        self.epoch = 0
        fill = window - m_c - m_n
        # Scarce-solved fallback (smoke/cliff-only datasets): when the solved
        # pool cannot fill even one window, fill slots cycle the CLIFF order
        # instead (solved rows, if any, are still each used once). D_S = 0
        # windows are fine — compute_loss zero-guards the L_S term.
        self.cliff_fill = fill > 0 and len(self.solved_idx) < fill
        if self.cliff_fill and self.cliff_idx:
            print(f"[train] WARNING: only {len(self.solved_idx)} solved rows for a "
                  f"{fill}-slot fill — filling windows from the cliff cycle instead")
            per_win = window - m_n
            self.n_win = max(1, -(-(len(self.solved_idx) + len(self.cliff_idx)) // per_win))
            self.n_solved_dropped = 0
        else:
            self.n_win = len(self.solved_idx) // fill if fill > 0 else 0
            self.n_solved_dropped = len(self.solved_idx) - self.n_win * fill
        if self.n_win == 0:
            raise ValueError(
                f"dataset too small for stratification: {len(self.solved_idx)} solved rows "
                f"cannot fill one {fill}-slot window (global_batch_size={window}, "
                f"m_c={m_c}, m_n={m_n}) and there are no cliff rows to fill from"
            )

    @staticmethod
    def _auto_m_c(n_solved: int, n_cliff: int, window: int, m_n: int) -> int:
        """Smallest cliff-rows-per-window that shows EVERY improved row at least
        once per epoch.

        An epoch is sized by the SOLVED pool (n_win = n_solved // fill), so the
        number of cliff DRAWS is n_win * m_C — independent of how many improved
        rows exist. Whether that over- or under-samples them is therefore an
        accident of the solved:cliff ratio, and the L5 mixes sit on the opposite
        side of it from L3 (which had 118 improved rows against 173 windows).
        Solving for m_C restores the intent: every rescue trajectory the
        operator paid to produce enters training, and per_question_norm — inert
        at m_C=1, where its 1/n_q cancels against the window denominator —
        starts equalising questions again.

        fill shrinks as m_C grows, so n_win moves too; scan rather than divide.
        At least one fill slot is kept: a window of pure cliff rows would leave
        L_S with no data.
        """
        if n_cliff <= 0:
            return 0
        hi = max(1, window - m_n - 1)
        for m in range(1, hi + 1):
            fill = window - m - m_n
            n_win = n_solved // fill if fill > 0 else 0
            if n_win * m >= n_cliff:
                return m
        return hi

    def cliff_coverage(self) -> float:
        """Fraction of improved rows the epoch actually draws (cycled order, so
        draws beyond the pool are a second pass, not a re-roll)."""
        if not self.cliff_idx:
            return 0.0
        return min(1.0, self.n_win * self.m_c / len(self.cliff_idx))

    def set_epoch(self, epoch: int) -> None:   # reached via DataLoaderShard.set_epoch
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_win * self.window

    def __iter__(self):
        rng = random.Random((self.seed * 1000003) ^ (self.epoch + 1))
        solved = self.solved_idx[:]
        rng.shuffle(solved)
        # one shuffled cliff order, cycled for the whole epoch (intentional
        # oversampling when n_cliff < n_win * m_c)
        order = list(range(len(self.cliff_idx)))
        rng.shuffle(order)
        neg_cycles = {q: rng.sample(v, len(v)) for q, v in self.neg_idx_by_qid.items()}
        neg_cursor = {q: 0 for q in neg_cycles}
        all_neg = self.all_neg_idx[:]
        rng.shuffle(all_neg)
        gcur = 0
        ci = si = 0
        fill = self.window - self.m_c - self.m_n
        for _ in range(self.n_win):
            win: list[int] = []
            win_qids: list[str] = []
            for _ in range(self.m_c):
                k = order[ci % len(order)]
                ci += 1
                win.append(self.cliff_idx[k])
                win_qids.append(self.cliff_qids[k])
            for j in range(self.m_n):
                # best-effort same-qid negative for the window's cliff rows
                q = win_qids[j % len(win_qids)] if win_qids else None
                pool = neg_cycles.get(q)
                if pool:
                    win.append(pool[neg_cursor[q] % len(pool)])
                    neg_cursor[q] += 1
                else:
                    win.append(all_neg[gcur % len(all_neg)])
                    gcur += 1
            n_fill = fill
            if self.cliff_fill:
                take = min(n_fill, len(solved) - si)   # remaining solved, each once
                win.extend(solved[si:si + take])
                si += take
                for _ in range(n_fill - take):         # rest from the cliff cycle
                    k = order[ci % len(order)]
                    ci += 1
                    win.append(self.cliff_idx[k])
            else:
                win.extend(solved[si:si + n_fill])
                si += n_fill
            # in-window shuffle: cliff rows should not always land on rank 0
            rng.shuffle(win)
            yield from win


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def make_weighted_trainer_cls():
    """Deferred import so light-weight tooling can import this module without
    a full transformers install."""
    from transformers import Trainer

    class WeightedSFTTrainer(Trainer):
        def __init__(self, *args, cliff_cfg=None, train_sampler=None, **kwargs):
            # cliff_cfg = cfg.train.sft.cliff (None/disabled -> legacy paths verbatim)
            self._cliff = cliff_cfg if (cliff_cfg is not None and cliff_cfg.enabled) else None
            self._train_sampler = train_sampler
            self._window_denoms = None
            self._comp_sums = {}
            super().__init__(*args, **kwargs)
            # Pin the num_items code path instead of trusting signature
            # inspection of a wrapped model: with this True and num_items
            # provided, transformers 5.7 training_step skips its per-accum-step
            # loss division (and disables DeepSpeed gas-scaling), which is
            # required for our sum/global-count normalization.
            self.model_accepts_loss_kwargs = True

        def _get_train_sampler(self, train_dataset=None):
            if self._train_sampler is not None:
                return self._train_sampler
            return super()._get_train_sampler(train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("loss_weights")
            labels = inputs.pop("labels")
            cl = self._cliff
            if cl is not None:
                slice_ids = inputs.pop("slice_ids", None)
                n_q = inputs.pop("n_q", None)
                refs = inputs.pop("ref_mean_nll", None)
                comp = inputs.pop("completion_mask", None)
                if slice_ids is None:   # non-cliff collator output (defensive)
                    cl = None
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            shift_weights = weights[:, 1:]
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                shift_labels.reshape(-1).clamp(min=0),
                reduction="none",
            ).view(shift_labels.shape)
            valid = shift_labels != -100
            if cl is None:
                ce = ce * valid * shift_weights
                if num_items_in_batch is not None:
                    denom = num_items_in_batch
                    if torch.is_tensor(denom):
                        denom = denom.to(ce.device)
                    loss = ce.sum() / denom
                    # DP compensation, mirroring base Trainer.compute_loss: with a
                    # globally-gathered denominator each rank holds sum_local/num_global,
                    # and the DP all-reduce AVERAGES gradients across ranks — without
                    # this factor both the gradient and the logged loss shrink by
                    # world_size (verified empirically: 2-GPU zero2 logged exactly
                    # half the single-GPU loss and grad_norm on identical data).
                    if self.args.average_tokens_across_devices and self.args.world_size > 1:
                        loss = loss * self.args.world_size
                else:
                    loss = ce.sum() / shift_weights[shift_labels != -100].sum().clamp(min=1.0)
                return (loss, outputs) if return_outputs else loss

            # ---- cliff objective: L = (1-rho)·L_S + rho·(L_C + mu·L_N + L_G) ----
            # Numerators are LOCAL micro-batch sums; denominators are the window-
            # GLOBAL per-slice sums stashed by _get_num_items_in_batch. Summing the
            # per-micro losses over the window therefore reproduces each term
            # exactly, and the DP all-reduce average is undone by the same
            # world_size factor as the legacy path.
            wce = ce * valid * shift_weights                       # [B, T]
            row_wce = wce.sum(dim=1)                               # [B]
            row_W = (shift_weights * valid).sum(dim=1)             # [B]
            solved = slice_ids == 0
            cliffm = slice_ids == 1
            negm = slice_ids == 2

            num_S = row_wce[solved].sum()
            inv_nq = torch.where(slice_ids > 0, 1.0 / n_q.clamp(min=1.0),
                                 torch.zeros_like(n_q))
            if cl.per_question_norm:
                num_C = (row_wce * inv_nq / row_W.clamp(min=1.0))[cliffm].sum()
            else:
                num_C = row_wce[cliffm].sum()                      # S3-tok: token norm

            num_N = ce.new_zeros(())
            if cl.negative.mode == "v1" and bool(negm.any()):
                # bounded unlikelihood, fp32: u = -log1p(-p), p = exp(-ce) <= 1-delta
                p = torch.exp(-ce[negm]).clamp(max=1.0 - cl.negative.delta)
                u = -torch.log1p(-p) * valid[negm] * shift_weights[negm]
                num_N = (u.sum(dim=1) * inv_nq[negm] / row_W[negm].clamp(min=1.0)).sum()

            num_G = ce.new_zeros(())
            if cl.guard.enabled:
                shift_comp = comp[:, 1:].to(ce.dtype)
                comp_tokens = (shift_comp * valid).sum(dim=1)
                mean_ce = (ce * shift_comp * valid).sum(dim=1) / comp_tokens.clamp(min=1.0)
                guard_rows = cliffm & (refs >= 0)
                if bool(guard_rows.any()):
                    hinge = torch.relu(mean_ce - refs)
                    num_G = (hinge * inv_nq)[guard_rows].sum()

            D = self._window_denoms
            if D is None:  # prediction_step / unexpected path: local fallback
                D = {"S": row_W[solved].sum(), "C": inv_nq[cliffm].sum() if cl.per_question_norm else row_W[cliffm].sum(),
                     "N": inv_nq[negm].sum(), "G": inv_nq[cliffm & (refs >= 0)].sum()}

            def _term(num, key):
                d = D[key]
                d = d.to(num.device) if torch.is_tensor(d) else torch.as_tensor(d, device=num.device)
                return num / d if float(d) > 0 else num * 0.0

            L_S = _term(num_S, "S")
            L_C = _term(num_C, "C")
            L_N = _term(num_N, "N")
            L_G = _term(num_G, "G")
            loss = (1.0 - cl.rho) * L_S + cl.rho * (L_C + cl.negative.mu * L_N + L_G)
            if self.args.average_tokens_across_devices and self.args.world_size > 1:
                loss = loss * self.args.world_size

            with torch.no_grad():  # rank-local monitoring, flushed by log()
                c = self._comp_sums
                for k, v in (("loss/solved", L_S), ("loss/cliff", L_C),
                             ("loss/negative", L_N), ("loss/guard", L_G)):
                    c[k] = c.get(k, 0.0) + float(v)
                c["cliff/rows"] = c.get("cliff/rows", 0.0) + float(cliffm.sum())
                c["negative/rows"] = c.get("negative/rows", 0.0) + float(negm.sum())
                c["guard/skipped_ref"] = c.get("guard/skipped_ref", 0.0) + float((cliffm & (refs < 0)).sum())
                c["_micro"] = c.get("_micro", 0.0) + 1.0
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, start_time=None):
            c, self._comp_sums = self._comp_sums, {}
            micro = c.pop("_micro", 0.0)
            if micro and self._cliff is not None and "loss" in logs:
                accum = max(self.args.gradient_accumulation_steps, 1)
                steps = max(micro / accum, 1.0)
                for k, v in c.items():
                    # per-window component means (loss/*: sum of micro contributions
                    # per window; rows/skipped: per-window counts on this rank)
                    logs[k] = round(v / steps, 6)
            return super().log(logs, start_time)

        def _get_num_items_in_batch(self, batch_samples, device):
            """num_items = global sum of (label-shifted) loss weights across the
            full grad-accum window (and all ranks, via the same gather the base
            class uses), so the weighted loss normalizes identically under any
            micro-batch/accum/DP topology. Signature verified against the
            installed transformers 5.7 (see docs/api_notes.md).

            Cliff mode: this hook runs once per optimizer window BEFORE any of
            the window's compute_loss calls (verified against the installed
            trainer), so the per-slice global denominators [D_S, D_C, D_N, D_G]
            are gathered here in ONE collective, stashed on self, and the
            RETURN VALUE stays the legacy scalar — the base class only checks
            it for None, and a scalar keeps any future consumer sane."""
            if not batch_samples or "loss_weights" not in batch_samples[0]:
                return super()._get_num_items_in_batch(batch_samples, device)
            cl = self._cliff
            if cl is None or "slice_ids" not in batch_samples[0]:
                num_items = sum(
                    (b["loss_weights"][:, 1:] * b["labels"][:, 1:].ne(-100)).sum()
                    for b in batch_samples
                )
                if self.args.average_tokens_across_devices and self.args.world_size > 1:
                    num_items = self.accelerator.gather(num_items.to(device)).sum()
                return num_items

            # follow the data's device: under accelerate/DeepSpeed the prefetched
            # micro-batches are already on CUDA (CPU only in unit tests)
            vec = torch.zeros(5, device=batch_samples[0]["loss_weights"].device)
            for b in batch_samples:
                valid = b["labels"][:, 1:].ne(-100)
                row_W = (b["loss_weights"][:, 1:] * valid).sum(dim=1)
                sl = b["slice_ids"]
                inv_nq = torch.where(sl > 0, 1.0 / b["n_q"].clamp(min=1.0),
                                     torch.zeros_like(b["n_q"]))
                solved, cliffm, negm = sl == 0, sl == 1, sl == 2
                vec[0] += row_W[solved].sum()
                vec[1] += (inv_nq[cliffm].sum() if cl.per_question_norm
                           else row_W[cliffm].sum())
                vec[2] += inv_nq[negm].sum()
                vec[3] += inv_nq[cliffm & (b["ref_mean_nll"] >= 0)].sum()
                vec[4] += row_W.sum()          # legacy scalar (all slices)
            if self.args.average_tokens_across_devices and self.args.world_size > 1:
                vec = self.accelerator.gather(vec.to(device)).view(-1, 5).sum(dim=0)
            self._window_denoms = {"S": vec[0], "C": vec[1], "N": vec[2], "G": vec[3]}
            return vec[4]

    return WeightedSFTTrainer


# ---------------------------------------------------------------------------
# Group-advantage objective (train.objective: gadv)
# ---------------------------------------------------------------------------

_PREPASS_CHUNK = 2048   # positions per fp32 log-softmax chunk in the theta0 pre-pass


def make_gadv_trainer_cls():
    """Deferred import, like make_weighted_trainer_cls."""
    from transformers import Trainer

    class GadvTrainer(Trainer):
        """Advantage-weighted, theta0-clipped token loss:

            L = sum_rows sum_t m_t * -min(rho_t*A, clip(rho_t, 1-eps_lo, 1+eps_hi)*A) / N_tok
                + guard_weight * sum_{rescue rows with ref} relu(mean_ce - ref)/n_q / D_G

        N_tok and D_G are gathered once per optimizer window (fixed-shape
        vector, same collective pattern as the cliff trainer) so the loss is
        invariant to micro-batch/accum/DP topology; the world_size factor undoes
        the DP gradient average exactly as the legacy path does. At rho == 1
        the gradient is -A * grad(log pi): SFT for A > 0, NSR-form push-down for
        A < 0. Advantages and the theta0 cache are looked up by `row_idx`
        (fp32 on this side; batch floats arrive bf16 under DeepSpeed)."""

        def __init__(self, *args, gadv_cfg, **kwargs):
            self._gadv = gadv_cfg
            self._window_denoms = None
            self._comp_sums: dict[str, float] = {}
            self._old_logp: list[torch.Tensor] | None = None
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = True   # see WeightedSFTTrainer
            self._adv = torch.tensor(list(self.train_dataset["advantage"]), dtype=torch.float32)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch.nn.functional as F

            weights = inputs.pop("loss_weights")
            labels = inputs.pop("labels")
            row_idx = inputs.pop("row_idx")
            inputs.pop("advantage", None)            # bf16-cast copy: never used for math
            comp = inputs.pop("completion_mask")
            sl = inputs.pop("slice_ids")
            n_q = inputs.pop("n_q")
            refs = inputs.pop("ref_mean_nll")
            g = self._gadv
            A = self._adv[row_idx.cpu()].to(weights.device)            # fp32 [B]

            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            shift_w = weights[:, 1:].float()
            shift_comp = comp[:, 1:].bool()
            logp = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                shift_labels.reshape(-1).clamp(min=0),
                reduction="none",
            ).view(shift_labels.shape)
            valid = (shift_labels != -100) & shift_comp
            m = valid.to(logp.dtype) * shift_w                          # per-token weight [B, T']
            Ab = A[:, None]

            ratio = None
            if g.clip.enabled:
                if self._old_logp is None:
                    raise RuntimeError("gadv: theta0 log-prob cache missing — the pre-pass "
                                       "callback did not run before the first step")
                lengths = shift_comp.sum(dim=1).tolist()
                chunks = []
                for j, i in enumerate(row_idx.tolist()):
                    c = self._old_logp[i]
                    if c.numel() != lengths[j]:
                        raise RuntimeError(
                            f"gadv: theta0 cache for row {i} holds {c.numel()} tokens, "
                            f"batch has {lengths[j]} completion tokens (row_idx misaligned?)"
                        )
                    chunks.append(c)
                old = torch.zeros_like(logp)
                if chunks:
                    old[shift_comp] = torch.cat(chunks).to(device=old.device, dtype=old.dtype)
                ratio = torch.exp(logp - old)                          # old is a constant
                clipped = ratio.clamp(1.0 - g.clip.eps_lo, 1.0 + g.clip.eps_hi)
                per_tok = -torch.minimum(ratio * Ab, clipped * Ab)
            else:
                per_tok = -Ab * logp

            weighted = per_tok * m
            num = weighted.sum()

            D = self._window_denoms
            if D is None:   # prediction_step / unexpected path: local fallback
                D = {"N": m.sum(),
                     "G": torch.where(sl == 1, 1.0 / n_q.clamp(min=1.0), torch.zeros_like(n_q))[
                         (sl == 1) & (refs >= 0)].sum()}

            def _term(x, key):
                d = D[key]
                d = d.to(x.device) if torch.is_tensor(d) else torch.as_tensor(d, device=x.device)
                return x / d if float(d) > 0 else x * 0.0

            loss = _term(num, "N")
            L_G = num.new_zeros(())
            if g.guard_weight > 0:
                comp_tok = valid.sum(dim=1)
                mean_ce = -(logp * valid).sum(dim=1) / comp_tok.clamp(min=1.0)
                guard_rows = (sl == 1) & (refs >= 0)
                if bool(guard_rows.any()):
                    inv_nq = torch.where(sl == 1, 1.0 / n_q.clamp(min=1.0), torch.zeros_like(n_q))
                    hinge = torch.relu(mean_ce - refs)
                    L_G = _term((hinge * inv_nq)[guard_rows].sum(), "G")
                loss = loss + g.guard_weight * L_G
            if self.args.average_tokens_across_devices and self.args.world_size > 1:
                loss = loss * self.args.world_size

            with torch.no_grad():   # rank-local monitoring, flushed by log()
                c = self._comp_sums
                pos_rows, neg_rows = A > 0, A < 0
                dN = D["N"]
                dN = float(dN) if float(dN) > 0 else 1.0
                c["loss/pos"] = c.get("loss/pos", 0.0) + float(weighted[pos_rows].sum()) / dN
                c["loss/neg"] = c.get("loss/neg", 0.0) + float(weighted[neg_rows].sum()) / dN
                c["loss/guard"] = c.get("loss/guard", 0.0) + float(L_G)
                c["rows/pos"] = c.get("rows/pos", 0.0) + float(pos_rows.sum())
                c["rows/neg"] = c.get("rows/neg", 0.0) + float(neg_rows.sum())
                c["rows/rescue"] = c.get("rows/rescue", 0.0) + float((sl == 1).sum())
                c["guard/skipped_ref"] = c.get("guard/skipped_ref", 0.0) + float(((sl == 1) & (refs < 0)).sum())
                if ratio is not None:
                    mp, mn = m * pos_rows[:, None], m * neg_rows[:, None]
                    c["_clip_mass_pos"] = c.get("_clip_mass_pos", 0.0) + float(mp.sum())
                    c["_clip_mass_neg"] = c.get("_clip_mass_neg", 0.0) + float(mn.sum())
                    c["_clip_hit_pos"] = c.get("_clip_hit_pos", 0.0) + float((mp * (ratio > 1.0 + g.clip.eps_hi)).sum())
                    c["_clip_hit_neg"] = c.get("_clip_hit_neg", 0.0) + float((mn * (ratio < 1.0 - g.clip.eps_lo)).sum())
                    vm = m > 0
                    c["_ratio_sum"] = c.get("_ratio_sum", 0.0) + float(ratio[vm].sum())
                    c["_ratio_tok"] = c.get("_ratio_tok", 0.0) + float(vm.sum())
                    c["_ratio_max"] = max(c.get("_ratio_max", 0.0), float(ratio[vm].max()) if bool(vm.any()) else 0.0)
                c["_micro"] = c.get("_micro", 0.0) + 1.0
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, start_time=None):
            c, self._comp_sums = self._comp_sums, {}
            micro = c.pop("_micro", 0.0)
            if micro and "loss" in logs:
                accum = max(self.args.gradient_accumulation_steps, 1)
                steps = max(micro / accum, 1.0)
                for k in ("loss/pos", "loss/neg", "loss/guard", "rows/pos", "rows/neg",
                          "rows/rescue", "guard/skipped_ref"):
                    if k in c:
                        logs[k] = round(c[k] / steps, 6)
                for sign in ("pos", "neg"):
                    mass = c.get(f"_clip_mass_{sign}", 0.0)
                    if mass > 0:
                        logs[f"clip/frac_{sign}"] = round(c.get(f"_clip_hit_{sign}", 0.0) / mass, 6)
                if c.get("_ratio_tok", 0.0) > 0:
                    logs["ratio/mean"] = round(c["_ratio_sum"] / c["_ratio_tok"], 6)
                    logs["ratio/max"] = round(c.get("_ratio_max", 0.0), 6)
                D = self._window_denoms
                if D is not None:   # last window's GLOBAL (gathered) masses
                    logs["gadv/n_tok"] = round(float(D["N"]), 1)
                    logs["gadv/pos_mass"] = round(float(D["pos_mass"]), 3)
                    logs["gadv/neg_mass"] = round(float(D["neg_mass"]), 3)
            return super().log(logs, start_time)

        def _get_num_items_in_batch(self, batch_samples, device):
            """Once per optimizer window, BEFORE its compute_loss calls (see
            WeightedSFTTrainer): gather [N_tok, D_G, pos_mass, neg_mass, n_pos,
            n_neg] across ranks in one collective; return N_tok as the legacy
            scalar. Runs on the un-cast batches, so `advantage` is fp32 here."""
            if not batch_samples or "row_idx" not in batch_samples[0]:
                return super()._get_num_items_in_batch(batch_samples, device)
            vec = torch.zeros(6, device=batch_samples[0]["loss_weights"].device)
            for b in batch_samples:
                valid = b["labels"][:, 1:].ne(-100) & b["completion_mask"][:, 1:].bool()
                row_tok = (b["loss_weights"][:, 1:].float() * valid).sum(dim=1)
                A = self._adv[b["row_idx"].cpu()].to(row_tok.device)
                sl = b["slice_ids"]
                inv_nq = torch.where(sl == 1, 1.0 / b["n_q"].clamp(min=1.0), torch.zeros_like(b["n_q"]))
                vec[0] += row_tok.sum()
                vec[1] += inv_nq[(sl == 1) & (b["ref_mean_nll"] >= 0)].sum()
                vec[2] += (A.clamp(min=0) * row_tok).sum()
                vec[3] += ((-A).clamp(min=0) * row_tok).sum()
                vec[4] += (A > 0).sum()
                vec[5] += (A < 0).sum()
            if self.args.average_tokens_across_devices and self.args.world_size > 1:
                vec = self.accelerator.gather(vec.to(device)).view(-1, 6).sum(dim=0)
            self._window_denoms = {"N": vec[0], "G": vec[1], "pos_mass": vec[2],
                                   "neg_mass": vec[3], "n_pos": vec[4], "n_neg": vec[5]}
            return vec[0]

    return GadvTrainer


def make_gadv_prepass_callback(trainer, collator, batch_size: int = 1,
                               cache_dtype: str = "float32"):
    """TrainerCallback that fills trainer._old_logp (per-row completion-token
    log-probs under theta0) at on_train_begin — after accelerate/DeepSpeed have
    prepared the model and before the first optimizer step (verl's "recompute
    old_log_probs"). Each rank scores a strided share of the rows through the
    prepared model (trainer.model_wrapped, replicated params under ZeRO-2), then
    the shards are exchanged with all_gather_object so any rank can train on any
    row in any epoch. Skipped when the clip is disabled (no ratio needed)."""
    import time

    from transformers import TrainerCallback

    dtype = torch.float32 if cache_dtype == "float32" else torch.float16

    class OldLogpPrepass(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            import torch.distributed as dist
            import torch.nn.functional as F

            tr = trainer
            if tr._old_logp is not None or not tr._gadv.clip.enabled:
                return control
            t0 = time.time()
            fwd = getattr(tr, "model_wrapped", None) or tr.model
            was_training = fwd.training
            fwd.eval()
            dev = args.device
            ds = tr.train_dataset
            n_rows = len(ds)
            rank, world = args.process_index, args.world_size
            lens = [p + a + c for p, a, c in zip(ds["prompt_len"], ds["anchor_len"], ds["completion_len"])]
            mine = sorted(range(rank, n_rows, world), key=lambda i: -lens[i])
            local: dict[int, torch.Tensor] = {}
            with torch.no_grad():
                for s in range(0, len(mine), batch_size):
                    chunk = mine[s:s + batch_size]
                    b = collator([ds[i] for i in chunk])
                    out = fwd(input_ids=b["input_ids"].to(dev),
                              attention_mask=b["attention_mask"].to(dev), use_cache=False)
                    logits = out.logits[:, :-1, :]
                    tgt = b["labels"][:, 1:].clamp(min=0).to(dev)
                    logp = torch.empty(tgt.shape, dtype=torch.float32, device=dev)
                    for st in range(0, tgt.shape[1], _PREPASS_CHUNK):
                        lg = logits[:, st:st + _PREPASS_CHUNK, :].float()
                        tg = tgt[:, st:st + _PREPASS_CHUNK]
                        logp[:, st:st + _PREPASS_CHUNK] = -F.cross_entropy(
                            lg.reshape(-1, lg.size(-1)), tg.reshape(-1), reduction="none"
                        ).view(tg.shape)
                    shift_comp = b["completion_mask"][:, 1:].bool().to(dev)
                    for j, i in enumerate(chunk):
                        local[i] = logp[j][shift_comp[j]].to(dtype).cpu()
                    del out, logits, logp
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                parts: list = [None] * dist.get_world_size()
                dist.all_gather_object(parts, local)
                merged = {k: v for p in parts for k, v in p.items()}
            else:
                merged = local
            if len(merged) != n_rows:
                raise RuntimeError(f"gadv pre-pass covered {len(merged)}/{n_rows} rows")
            tr._old_logp = [merged[i] for i in range(n_rows)]
            fwd.train(was_training)
            if tr.is_world_process_zero():
                n_tok = sum(t.numel() for t in tr._old_logp)
                mb = sum(t.numel() * t.element_size() for t in tr._old_logp) / 1e6
                print(f"[train] gadv theta0 pre-pass: {n_rows} rows, {n_tok} completion tokens, "
                      f"{mb:.0f} MB {cache_dtype} cache, {time.time() - t0:.0f}s")
            return control

    return OldLogpPrepass()


# ---------------------------------------------------------------------------
# Entrypoints per objective
# ---------------------------------------------------------------------------

def _load_policy(init_path: str, sft):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        init_path,
        torch_dtype=torch.bfloat16 if sft.bf16 else None,
        attn_implementation="flash_attention_2",
    )
    if sft.gradient_checkpointing:
        model.config.use_cache = False
    return model


def _sft_training_args(cfg: Config, args, sft, out_dir: Path, grad_accum: int, run_tag: str = "sft"):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(out_dir / "trainer_state"),
        per_device_train_batch_size=sft.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=sft.epochs,
        learning_rate=sft.lr,
        lr_scheduler_type=sft.scheduler,
        warmup_ratio=sft.warmup_ratio,
        weight_decay=sft.weight_decay,
        max_grad_norm=sft.max_grad_norm,
        bf16=sft.bf16,
        gradient_checkpointing=sft.gradient_checkpointing,
        logging_steps=sft.logging_steps,
        save_strategy="no",              # we save exactly once, at the end
        report_to=_report_to(cfg),
        run_name=f"{cfg.run.name}/iter{args.iteration}/{run_tag}",
        seed=cfg.run.seed,
        average_tokens_across_devices=True,
        remove_unused_columns=False,     # keep source/region columns for the collator
        dataloader_num_workers=2,
    )


def run_gadv(cfg: Config, args, dataset_path: Path, init_path: str, out_dir: Path) -> None:
    from collections import Counter

    from datasets import Dataset

    sft = cfg.train.sft
    g = cfg.train.gadv
    rows = [r for r in read_jsonl(dataset_path)]
    n_before = len(rows)
    rows = [r for r in rows if len(r["input_ids"]) <= cfg.train.max_seq_len]
    if len(rows) < n_before:
        print(f"[train] dropped {n_before - len(rows)} examples > max_seq_len={cfg.train.max_seq_len}")
    n_missing = sum("advantage" not in r for r in rows)
    if n_missing:
        raise RuntimeError(
            f"gadv: {n_missing} rows carry no `advantage` — was {dataset_path} built with "
            "train.objective=gadv? (legacy train_sft.jsonl cannot be trained with gadv)"
        )
    bad = Counter(r["source"] for r in rows if r["source"] not in GadvCollator.GADV_SLICE)
    if bad:
        raise RuntimeError(f"gadv: unsupported sources in the dataset: {dict(bad)}")
    n_zero = sum(float(r["advantage"]) == 0.0 for r in rows)
    if n_zero:
        print(f"[train] WARNING: dropping {n_zero} rows with advantage == 0")
        rows = [r for r in rows if float(r["advantage"]) != 0.0]
    if not rows:
        raise RuntimeError("empty gadv training set — no group produced any row this iteration")

    keep_cols = ["uid", "source", "input_ids", "prompt_len", "anchor_len", "completion_len",
                 "n_q", "ref_mean_nll", "advantage"]
    ds = Dataset.from_list([
        {**{k: r.get(k) for k in keep_cols}, "n_q": r.get("n_q", 0), "row_idx": i}
        for i, r in enumerate(rows)
    ])
    if list(ds["row_idx"]) != list(range(len(ds))):
        raise RuntimeError("gadv: row_idx column is not the dataset position")

    by_src = Counter(r["source"] for r in rows)
    pos_mass = sum(float(r["advantage"]) * r["completion_len"] for r in rows if float(r["advantage"]) > 0)
    neg_mass = -sum(float(r["advantage"]) * r["completion_len"] for r in rows if float(r["advantage"]) < 0)
    print(
        f"[train] gadv objective: gamma={g.gamma} rescue_dose={g.rescue_dose} "
        f"neg_scale={g.neg_scale} clip={'on' if g.clip.enabled else 'off'}"
        f"({g.clip.eps_lo}/{g.clip.eps_hi}) guard_weight={g.guard_weight} | rows {dict(by_src)} | "
        f"token-weighted mass pos={pos_mass:.0f} neg={neg_mass:.0f} "
        f"({neg_mass / pos_mass:.2f}x)" if pos_mass else "[train] gadv objective: no positive rows"
    )

    tokenizer = _load_tokenizer(init_path)
    model = _load_policy(init_path, sft)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = _grad_accum(sft.global_batch_size, sft.micro_batch_size, world)
    targs = _sft_training_args(cfg, args, sft, out_dir, grad_accum, run_tag="gadv")
    collator = GadvCollator(
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        region_weights=sft.region_weights,
    )
    trainer = make_gadv_trainer_cls()(
        model=model, args=targs, train_dataset=ds, data_collator=collator, gadv_cfg=g,
    )
    trainer.add_callback(make_gadv_prepass_callback(
        trainer, collator, batch_size=g.prepass_batch_size, cache_dtype=g.cache_dtype))
    trainer.train()
    _save_and_check(trainer, tokenizer, out_dir)
    trainer.accelerator.free_memory()
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def run_sft(cfg: Config, args, dataset_path: Path, init_path: str, out_dir: Path) -> None:
    from datasets import Dataset

    sft = cfg.train.sft
    cliff = sft.cliff
    rows = [r for r in read_jsonl(dataset_path)]
    n_before = len(rows)
    rows = [r for r in rows if len(r["input_ids"]) <= cfg.train.max_seq_len]
    if len(rows) < n_before:
        print(f"[train] dropped {n_before - len(rows)} examples > max_seq_len={cfg.train.max_seq_len}")
    # gadv's negative-advantage rows must never be CE-trained (possible via
    # data.accumulate after a gadv iteration): drop them loudly.
    n_wrong = sum(r.get("source") == "wrong" for r in rows)
    if n_wrong:
        print(f"[train] WARNING: dropping {n_wrong} gadv 'wrong' rows (objective={cfg.train.objective})")
        rows = [r for r in rows if r.get("source") != "wrong"]
    if not cliff.enabled:
        # Legacy path must never CE-train unlikelihood rows (possible via
        # data.accumulate after a v1 iteration): drop them loudly.
        n_neg = sum(r.get("source") == "negative" for r in rows)
        if n_neg:
            print(f"[train] WARNING: dropping {n_neg} negative rows (cliff term disabled)")
            rows = [r for r in rows if r.get("source") != "negative"]
    if not rows:
        raise RuntimeError("empty training set — nothing to learn this iteration")
    keep_cols = ["uid", "source", "input_ids", "prompt_len", "anchor_len", "completion_len"]
    if cliff.enabled:
        keep_cols += ["n_q", "ref_mean_nll"]
        rows = [{**r, "n_q": r.get("n_q", 0), "ref_mean_nll": r.get("ref_mean_nll")}
                for r in rows]
    ds = Dataset.from_list([{k: r[k] for k in keep_cols} for r in rows])

    sampler = None
    if cliff.enabled:
        solved_idx = [i for i, r in enumerate(rows) if r["source"] == "solved"]
        cliff_pairs = [(i, r["qid"]) for i, r in enumerate(rows) if r["source"] == "improved"]
        neg_idx_by_qid: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            if r["source"] == "negative":
                neg_idx_by_qid.setdefault(r["qid"], []).append(i)
        m_n = cliff.negative.m_per_batch if cliff.negative.mode == "v1" else 0
        sampler = StratifiedWindowSampler(
            solved_idx=solved_idx,
            cliff_idx=[i for i, _ in cliff_pairs],
            neg_idx_by_qid=neg_idx_by_qid,
            cliff_qids=[q for _, q in cliff_pairs],
            window=sft.global_batch_size,
            m_c=cliff.m_per_batch, m_n=m_n, seed=cfg.run.seed,
        )
        n_neg_rows = sum(len(v) for v in neg_idx_by_qid.values())
        print(
            f"[train] cliff objective on: rho={cliff.rho} mu={cliff.negative.mu} "
            f"m_c={sampler.m_c}(cliff coverage {sampler.cliff_coverage():.0%}) "
            f"m_n={sampler.m_n} per_question_norm={cliff.per_question_norm} "
            f"guard={cliff.guard.enabled} | rows solved={len(solved_idx)} "
            f"cliff={len(cliff_pairs)} negative={n_neg_rows} | "
            f"{sampler.n_win} windows/epoch, {sampler.n_solved_dropped} solved rows dropped per epoch"
        )

    tokenizer = _load_tokenizer(init_path)
    model = _load_policy(init_path, sft)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = _grad_accum(sft.global_batch_size, sft.micro_batch_size, world)

    targs = _sft_training_args(cfg, args, sft, out_dir, grad_accum)
    trainer_cls = make_weighted_trainer_cls()
    trainer = trainer_cls(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=WeightedCausalCollator(
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            region_weights=sft.region_weights,
            cliff_mode=cliff.enabled,
        ),
        cliff_cfg=cliff,
        train_sampler=sampler,
    )
    trainer.train()
    _save_and_check(trainer, tokenizer, out_dir)
    # objective=sft+dpo continues in THIS process: run_dpo then loads a policy
    # AND a reference model on top of whatever is still resident. The SFT
    # DeepSpeed engine (params + grads + optimizer shards) outlives function
    # scope because Trainer <-> CallbackHandler is a reference cycle, so a
    # refcount drop is not enough — break it explicitly and hand the caching
    # allocator's blocks back before returning.
    trainer.accelerator.free_memory()
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_dpo(
    cfg: Config, args, dataset_path: Path, init_path: str, out_dir: Path,
    *, allow_empty: bool = False,
) -> bool:
    """Anchor-conditioned DPO. v1 feeds decoded text (prompt/chosen/rejected) to
    trl's DPOTrainer; special tokens round-trip through the Qwen tokenizer.
    If exact id-level control turns out to matter for DPO too, switch to trl's
    pre-tokenized columns."""
    rows = list(read_jsonl(dataset_path))
    if not rows:
        if allow_empty:
            print("[train] no DPO pairs this iteration — keeping the SFT checkpoint")
            return False
        raise RuntimeError("empty DPO training set for train.objective=dpo")

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dpo = cfg.train.dpo
    tokenizer = _load_tokenizer(init_path)
    ds = Dataset.from_list([
        {
            "prompt": tokenizer.decode(r["prompt_token_ids"]),
            "chosen": tokenizer.decode(r["chosen_token_ids"]),
            "rejected": tokenizer.decode(r["rejected_token_ids"]),
        }
        for r in rows
    ])
    model = AutoModelForCausalLM.from_pretrained(
        init_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    # With precompute on we supply the reference ourselves, loaded AND PLACED,
    # because trl's precompute pass runs inside DPOTrainer.__init__ — before the
    # DeepSpeed engine exists. Passing ref_model=None there makes trl fall back
    # to self.model, which under DeepSpeed is still on the CPU at that point and
    # under ZeRO-3 is partitioned with no gather hooks, i.e. silently wrong.
    # After __init__ the reference is dead weight (the loss reads cached logps),
    # so we drop it before training starts. With precompute off we hand trl None
    # and it builds/prepares the reference itself, as before.
    ref_model = None
    if dpo.precompute_ref_log_probs:
        ref_model = AutoModelForCausalLM.from_pretrained(
            init_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).eval()
        ref_model.to(torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))))
    grad_ckpt = dpo.gradient_checkpointing
    if grad_ckpt:
        model.config.use_cache = False
    world = int(os.environ.get("WORLD_SIZE", "1"))
    dargs = DPOConfig(
        output_dir=str(out_dir / "trainer_state_dpo"),
        per_device_train_batch_size=dpo.micro_batch_size,
        gradient_accumulation_steps=_grad_accum(dpo.global_batch_size, dpo.micro_batch_size, world),
        num_train_epochs=dpo.epochs,
        learning_rate=dpo.lr,
        beta=dpo.beta,
        loss_type=dpo.loss_type,
        max_grad_norm=dpo.max_grad_norm,
        max_length=dpo.max_length or cfg.train.max_seq_len,
        bf16=True,
        gradient_checkpointing=grad_ckpt,
        # Precompute the reference log-probs ONCE up front, then train with the
        # policy forward only. This is exact — the reference is the frozen SFT
        # checkpoint, so its logps cannot change — and it is what makes DPO fit:
        # measured 2026-08-27 on the S4-v0 arm, a training step costs ~4.4 MB of
        # logits/softmax buffers per token of the longer side of a pair (152k
        # vocab, chosen+rejected, .contiguous() copies, fp32 log_softmax), and
        # running policy AND reference together doubled that into an OOM at
        # 76 GiB on an 80 GB A100 while only 33.7 GiB was resident.
        precompute_ref_log_probs=dpo.precompute_ref_log_probs,
        precompute_ref_batch_size=1,
        # use_reentrant=False is REQUIRED, not cosmetic: with the reentrant
        # autograd checkpoint the recompute is skipped whenever the block's
        # inputs do not require grad, so checkpointing silently no-ops and the
        # full activation stack is kept (2026-08-27: enabling checkpointing
        # moved the S4-v0 DPO peak by 0.06 of 74.6 GiB).
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=dpo.logging_steps,
        save_strategy="no",
        report_to=_report_to(cfg),
        run_name=f"{cfg.run.name}/iter{args.iteration}/dpo",
        seed=cfg.run.seed,
    )
    trainer = DPOTrainer(model=model, ref_model=ref_model, args=dargs,
                         train_dataset=ds, processing_class=tokenizer)
    if dpo.precompute_ref_log_probs:
        trainer.ref_model = None      # logps are cached; release the ~8 GB copy
        del ref_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    trainer.add_callback(_DpoMemoryProbe())
    trainer.train()
    _save_and_check(trainer, tokenizer, out_dir)
    return True


def _trainer_callback_base():
    from transformers import TrainerCallback
    return TrainerCallback


class _DpoMemoryProbe(_trainer_callback_base()):
    """Reports what actually occupies the GPU during DPO.

    DPO is the memory-critical phase (policy + reference, both forwards over
    chosen AND rejected, logits at a 152k vocab) and it OOMs in ways that are
    easy to misdiagnose — checkpointing that silently no-ops looks exactly like
    checkpointing that is on but insufficient. Print the two facts that tell
    them apart: whether checkpointing is live on the prepared model, and the
    allocated bytes before any activation is built."""

    def _log(self, tag, model):
        if int(os.environ.get("RANK", "0")) != 0:
            return
        inner = getattr(model, "module", model)
        print(f"[train] dpo/{tag}: grad_ckpt={getattr(inner, 'is_gradient_checkpointing', '?')} "
              f"alloc={torch.cuda.memory_allocated() / 2**30:.1f} GiB "
              f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kw):
        self._log("train_begin", model)

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step <= 2:
            self._log(f"step{state.global_step}", model)
        torch.cuda.reset_peak_memory_stats()

# ---------------------------------------------------------------------------

def _grad_accum(global_bs: int, micro_bs: int, world: int) -> int:
    denom = micro_bs * world
    if global_bs % denom:
        raise ValueError(
            f"global_batch_size={global_bs} not divisible by micro*world={denom}; "
            "adjust the config for this GPU count."
        )
    return global_bs // denom

def _report_to(cfg: Config) -> list[str]:
    if cfg.run.wandb.mode == "disabled":
        return []
    os.environ.setdefault("WANDB_PROJECT", cfg.run.wandb.project)
    os.environ.setdefault("WANDB_RUN_GROUP", cfg.run.name)
    os.environ.setdefault("WANDB_MODE", cfg.run.wandb.mode)
    if cfg.run.wandb.entity:
        os.environ.setdefault("WANDB_ENTITY", cfg.run.wandb.entity)
    return ["wandb"]


def _save_and_check(trainer, tokenizer, out_dir: Path) -> None:
    """One final full-weights save, valid under all backends: the accelerate/DS
    configs set ZeRO-3 gather-on-save and FSDP FULL_STATE_DICT, so save_model
    always lands a complete HF checkpoint that vLLM can load."""
    # Tokenizer FIRST: it is kilobytes, the weights are ~9 GB of NFS write. If
    # the job dies mid-save, a dir with weights but no tokenizer is the
    # dangerous state (see _load_tokenizer) — a dir with a tokenizer but no
    # weights fails loudly on the next load.
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(out_dir))
    trainer.save_model(str(out_dir))
    if trainer.is_world_process_zero():
        n_shards, total = _check_checkpoint(out_dir)
        print(f"[train] saved checkpoint: {n_shards} shard(s), {total / 1e9:.2f} GB -> {out_dir}")
    # Barrier: only rank 0 writes the weights, and on NFS an 8-9 GB safetensors
    # takes ~90 s. Without this the objective=sft+dpo path races — the other
    # ranks return from save_model and enter run_dpo, which loads this very
    # directory, before the file exists ("no file named model.safetensors",
    # observed 2026-08-27 on a 2-GPU zero2 S4-v0 arm).
    trainer.accelerator.wait_for_everyone()


def _load_tokenizer(path: str):
    """Load a tokenizer that is actually the model's tokenizer.

    transformers 5.x does NOT raise when a local dir holds model files but no
    tokenizer files: it builds an EMPTY tokenizer from config.json's
    `tokenizer_class` (vocab_size 1, `decode()` returns ""). On 2026-08-27 that
    silent fallback made a DPO phase train on blank prompt/chosen/rejected text
    for a full epoch and then overwrite the SFT checkpoint with the result —
    no error until an unrelated stage tripped over the missing chat template.
    Never trust the fallback; fail here instead."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.vocab_size < 1000 or tok.chat_template is None:
        raise RuntimeError(
            f"degenerate tokenizer from {path}: vocab_size={tok.vocab_size}, "
            f"chat_template={'set' if tok.chat_template else 'MISSING'} — the dir is "
            "missing its tokenizer files and transformers fell back to an empty "
            "tokenizer. Re-save the checkpoint; do not train against this."
        )
    return tok


def _check_checkpoint(out_dir: Path) -> tuple[int, int]:
    if not (out_dir / "config.json").exists():
        raise RuntimeError(f"checkpoint missing config.json: {out_dir}")
    shards = list(out_dir.glob("*.safetensors")) + list(out_dir.glob("*.bin"))
    total = sum(p.stat().st_size for p in shards)
    if not shards or total <= 1_000_000:
        raise RuntimeError(
            f"checkpoint at {out_dir} looks empty ({total} bytes) — "
            "sharded save without gather? Check backend save flags."
        )
    # A checkpoint without its tokenizer is silently loadable and silently
    # wrong (see _load_tokenizer), so treat it as an incomplete save.
    _load_tokenizer(str(out_dir))
    return len(shards), total


def main(argv: list[str] | None = None) -> None:
    parser = stage_argparser("EI train stage (run under `accelerate launch`)")
    # Resume helper for objective=sft+dpo, whose two phases are ~3 h apart: if
    # DPO dies, --phase dpo re-runs it against the SFT checkpoint already on
    # disk. A CLI flag, NOT a config field, on purpose — Config.hash() covers
    # every field, so a new one would invalidate the .done marker of every
    # stage in every existing run dir (and --override cannot help either:
    # freeze_run_config rejects any hash that differs from the run snapshot).
    parser.add_argument("--phase", choices=["all", "sft", "dpo"], default="all",
                        help="run only one phase of objective=sft+dpo (default: all)")
    args = parser.parse_args(argv)
    cfg = load_stage_config(args)
    it_dir = iter_dir(args.run_dir, args.iteration)
    out_dir = it_dir / "ckpt"
    done_key = out_dir / "config.json"
    if is_done(done_key, config_hash=cfg.hash()):
        print(f"[train] {out_dir} already done, skipping")
        return

    # --model-path here is the INITIALIZATION checkpoint (loop.py resolves
    # base-vs-last); inference stages resolve the current policy separately.
    objective = cfg.train.objective
    if args.phase != "all" and objective != "sft+dpo":
        raise SystemExit(f"--phase {args.phase} needs train.objective=sft+dpo, got {objective!r}")
    if objective == "gadv":
        run_gadv(cfg, args, it_dir / "dataset" / "train_sft.jsonl", args.model_path, out_dir)
    if objective in ("sft", "sft+dpo") and args.phase in ("all", "sft"):
        run_sft(cfg, args, it_dir / "dataset" / "train_sft.jsonl", args.model_path, out_dir)
    if objective in ("dpo", "sft+dpo") and args.phase in ("all", "dpo"):
        dpo_init = str(out_dir) if objective == "sft+dpo" else args.model_path
        if args.phase == "dpo":
            # resuming: the SFT phase must have left a complete checkpoint
            _check_checkpoint(out_dir)
        run_dpo(
            cfg, args, it_dir / "dataset" / "train_dpo.jsonl", dpo_init, out_dir,
            allow_empty=objective == "sft+dpo",
        )

    # --phase sft leaves the objective half-done, so it must NOT be marked.
    if args.phase != "sft" and int(os.environ.get("RANK", "0")) == 0:
        _check_checkpoint(out_dir)
        mark_done(done_key, count=1, config_hash=cfg.hash())


if __name__ == "__main__":
    main(sys.argv[1:])
