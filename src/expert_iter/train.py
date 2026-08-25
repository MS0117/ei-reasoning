"""Stage: train — ⚗ extension point (IV): the training objective.

Runs under `accelerate launch` (loop.py passes --num_processes and the backend
config file). Reads iter_k/dataset/train_{sft,dpo}.jsonl, writes iter_k/ckpt/
as a full HF-format checkpoint that vLLM loads directly next iteration.

Objectives:
  sft      WeightedSFT — per-token region-weighted cross entropy.
  dpo      Anchor-conditioned DPO (prompt = question + anchor).
  sft+dpo  SFT first, then DPO initialized from the SFT result.

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
# Entrypoints per objective
# ---------------------------------------------------------------------------

def run_sft(cfg: Config, args, dataset_path: Path, init_path: str, out_dir: Path) -> None:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    sft = cfg.train.sft
    cliff = sft.cliff
    rows = [r for r in read_jsonl(dataset_path)]
    n_before = len(rows)
    rows = [r for r in rows if len(r["input_ids"]) <= cfg.train.max_seq_len]
    if len(rows) < n_before:
        print(f"[train] dropped {n_before - len(rows)} examples > max_seq_len={cfg.train.max_seq_len}")
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
            f"m_c={sampler.m_c} m_n={sampler.m_n} per_question_norm={cliff.per_question_norm} "
            f"guard={cliff.guard.enabled} | rows solved={len(solved_idx)} "
            f"cliff={len(cliff_pairs)} negative={n_neg_rows} | "
            f"{sampler.n_win} windows/epoch, {sampler.n_solved_dropped} solved rows dropped per epoch"
        )

    tokenizer = AutoTokenizer.from_pretrained(init_path)
    model = AutoModelForCausalLM.from_pretrained(
        init_path,
        torch_dtype=torch.bfloat16 if sft.bf16 else None,
        attn_implementation="flash_attention_2",
    )
    if sft.gradient_checkpointing:
        model.config.use_cache = False

    world = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = _grad_accum(sft.global_batch_size, sft.micro_batch_size, world)

    targs = TrainingArguments(
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
        run_name=f"{cfg.run.name}/iter{args.iteration}/sft",
        seed=cfg.run.seed,
        average_tokens_across_devices=True,
        remove_unused_columns=False,     # keep source/region columns for the collator
        dataloader_num_workers=2,
    )
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
    tokenizer = AutoTokenizer.from_pretrained(init_path)
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
        max_length=cfg.train.max_seq_len,
        bf16=True,
        logging_steps=dpo.logging_steps,
        save_strategy="no",
        report_to=_report_to(cfg),
        run_name=f"{cfg.run.name}/iter{args.iteration}/dpo",
        seed=cfg.run.seed,
    )
    trainer = DPOTrainer(model=model, args=dargs, train_dataset=ds, processing_class=tokenizer)
    trainer.train()
    _save_and_check(trainer, tokenizer, out_dir)
    return True


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
    trainer.save_model(str(out_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(out_dir))
        n_shards, total = _check_checkpoint(out_dir)
        print(f"[train] saved checkpoint: {n_shards} shard(s), {total / 1e9:.2f} GB -> {out_dir}")


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
    return len(shards), total


def main(argv: list[str] | None = None) -> None:
    args = stage_argparser("EI train stage (run under `accelerate launch`)").parse_args(argv)
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
    if objective in ("sft", "sft+dpo"):
        run_sft(cfg, args, it_dir / "dataset" / "train_sft.jsonl", args.model_path, out_dir)
    if objective in ("dpo", "sft+dpo"):
        dpo_init = str(out_dir) if objective == "sft+dpo" else args.model_path
        run_dpo(
            cfg, args, it_dir / "dataset" / "train_dpo.jsonl", dpo_init, out_dir,
            allow_empty=objective == "sft+dpo",
        )

    if int(os.environ.get("RANK", "0")) == 0:
        _check_checkpoint(out_dir)
        mark_done(done_key, count=1, config_hash=cfg.hash())


if __name__ == "__main__":
    main(sys.argv[1:])
