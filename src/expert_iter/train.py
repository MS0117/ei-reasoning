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
"""

from __future__ import annotations

import json
import os
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

    def __init__(self, pad_token_id: int, region_weights: dict[str, float]):
        self.pad_token_id = pad_token_id
        self.w = region_weights

    def example_weights(self, ex: dict) -> list[float]:
        completion_w = self.w["solution"] if ex["source"] == "solved" else self.w["continuation"]
        return (
            [self.w["prompt"]] * ex["prompt_len"]
            + [self.w["anchor"]] * ex["anchor_len"]
            + [completion_w] * ex["completion_len"]
        )

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(ex["input_ids"]) for ex in examples)
        input_ids, attention_mask, labels, weights = [], [], [], []
        for ex in examples:
            ids = ex["input_ids"]
            w = self.example_weights(ex)
            assert len(w) == len(ids), f"region lens != input_ids for uid={ex.get('uid')}"
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append([t if wi > 0 else -100 for t, wi in zip(ids, w)] + [-100] * pad)
            weights.append(w + [0.0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(weights, dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def make_weighted_trainer_cls():
    """Deferred import so light-weight tooling can import this module without
    a full transformers install."""
    from transformers import Trainer

    class WeightedSFTTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Pin the num_items code path instead of trusting signature
            # inspection of a wrapped model: with this True and num_items
            # provided, transformers 5.7 training_step skips its per-accum-step
            # loss division (and disables DeepSpeed gas-scaling), which is
            # required for our sum/global-count normalization.
            self.model_accepts_loss_kwargs = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("loss_weights")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            shift_weights = weights[:, 1:]
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                shift_labels.reshape(-1).clamp(min=0),
                reduction="none",
            ).view(shift_labels.shape)
            ce = ce * (shift_labels != -100) * shift_weights
            if num_items_in_batch is not None:
                denom = num_items_in_batch
                if torch.is_tensor(denom):
                    denom = denom.to(ce.device)
                loss = ce.sum() / denom
            else:
                loss = ce.sum() / shift_weights[shift_labels != -100].sum().clamp(min=1.0)
            return (loss, outputs) if return_outputs else loss

        def _get_num_items_in_batch(self, batch_samples, device):
            """num_items = global sum of (label-shifted) loss weights across the
            full grad-accum window (and all ranks, via the same gather the base
            class uses), so the weighted loss normalizes identically under any
            micro-batch/accum/DP topology. Signature verified against the
            installed transformers 5.7 (see docs/api_notes.md)."""
            if not batch_samples or "loss_weights" not in batch_samples[0]:
                return super()._get_num_items_in_batch(batch_samples, device)
            num_items = sum(
                (b["loss_weights"][:, 1:] * b["labels"][:, 1:].ne(-100)).sum()
                for b in batch_samples
            )
            if self.args.average_tokens_across_devices and self.args.world_size > 1:
                num_items = self.accelerator.gather(num_items.to(device)).sum()
            return num_items

    return WeightedSFTTrainer


# ---------------------------------------------------------------------------
# Entrypoints per objective
# ---------------------------------------------------------------------------

def run_sft(cfg: Config, args, dataset_path: Path, init_path: str, out_dir: Path) -> None:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    sft = cfg.train.sft
    rows = [r for r in read_jsonl(dataset_path)]
    n_before = len(rows)
    rows = [r for r in rows if len(r["input_ids"]) <= cfg.train.max_seq_len]
    if len(rows) < n_before:
        print(f"[train] dropped {n_before - len(rows)} examples > max_seq_len={cfg.train.max_seq_len}")
    if not rows:
        raise RuntimeError("empty training set — nothing to learn this iteration")
    keep_cols = ["uid", "source", "input_ids", "prompt_len", "anchor_len", "completion_len"]
    ds = Dataset.from_list([{k: r[k] for k in keep_cols} for r in rows])

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
        ),
    )
    trainer.train()
    _save_and_check(trainer, tokenizer, out_dir)


def run_dpo(cfg: Config, args, dataset_path: Path, init_path: str, out_dir: Path) -> None:
    """Anchor-conditioned DPO. v1 feeds decoded text (prompt/chosen/rejected) to
    trl's DPOTrainer; special tokens round-trip through the Qwen tokenizer.
    If exact id-level control turns out to matter for DPO too, switch to trl's
    pre-tokenized columns."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dpo = cfg.train.dpo
    tokenizer = AutoTokenizer.from_pretrained(init_path)
    rows = list(read_jsonl(dataset_path))
    if not rows:
        print("[train] no DPO pairs this iteration — skipping DPO phase")
        return
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
        assert (out_dir / "config.json").exists(), "checkpoint missing config.json"
        shards = list(out_dir.glob("*.safetensors")) + list(out_dir.glob("*.bin"))
        total = sum(p.stat().st_size for p in shards)
        assert shards and total > 1_000_000, (
            f"checkpoint at {out_dir} looks empty ({total} bytes) — "
            "sharded save without gather? Check backend save flags."
        )
        print(f"[train] saved checkpoint: {len(shards)} shard(s), {total / 1e9:.2f} GB -> {out_dir}")


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
        run_dpo(cfg, args, it_dir / "dataset" / "train_dpo.jsonl", dpo_init, out_dir)

    if int(os.environ.get("RANK", "0")) == 0:
        mark_done(done_key, count=1, config_hash=cfg.hash())


if __name__ == "__main__":
    main(sys.argv[1:])
