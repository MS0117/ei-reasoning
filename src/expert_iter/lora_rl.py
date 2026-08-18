"""Post-SFT RL on LoRA params — GPU subprocess for improve.rl (GRPO/REINFORCE).

    python -m expert_iter.lora_rl --model <policy> --adapter <fit adapter dir> \
        --dataset <rl_rows.jsonl> --out <dir> --params-json '{...}' --cache-key <rl_key>

Wraps trl 1.3 GRPOTrainer (algo=grpo) / RLOOTrainer (algo=reinforce) with a
colocated vLLM engine (sleep mode) generating on-policy rollouts, training
ONLY the LoRA parameters warm-started from the fit adapter
(PeftModel.from_pretrained(..., is_trainable=True)). Reward = answer matching
via the math verifier — no reward model. Prompts contain NO y*.

CAVEAT (probed before trusting, see check_env.gpu_trl_grpo_probe): trl 1.3
officially supports vllm <= 0.18; this stack pins vllm 0.20. The colocate path
exercises merge_adapter -> load_weights -> llm.sleep/wake_up. This module runs
as its own subprocess so a failure surfaces as an exit code + log tail without
poisoning the pipeline.

Dataset rows (assembled by the OPERATOR, which owns tokenizer + ids):
    {"qid", "prompt": <rendered x text + decoded anchor text>, "final_answer",
     "anchor": <decoded anchor text, present only on anchored runs>}
The reward grades anchor+completion, matching what lora_sft._collect scores when
it decides whether a candidate rescued the question.
trl is text-based, so anchored prompts are re-rendered text; the operator
counts retokenization seam mismatches (anchors end at "\\n\\n" step boundaries,
so the seam is normally stable) and the candidate-sampling tail keeps using
id-spliced prompts regardless.

Outputs in --out: adapter files (peft save_pretrained), reward_curve.jsonl,
rl_meta.json, and a .done marker keyed by --cache-key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from transformers import TrainerCallback

from .records import QuestionRecord
from .templates import render_question_prompt
from .utils import (
    append_jsonl,
    is_done,
    mark_done,
    read_jsonl,
    seed_everything,
    stable_hash,
    write_json,
)


def build_rl_rows(qids, *, questions_by_qid, prompts, anchors_by_qid, tokenizer,
                  cfg) -> tuple[list[dict], int]:
    """RL dataset rows for the true target states (x [+ anchor]) — NO y*.
    Returns (rows, n_seam_mismatches): a mismatch means retokenizing the text
    prompt does not reproduce the id-spliced prompt (anchored rows only in
    practice; anchors end at step boundaries so this should be rare)."""
    rows: list[dict] = []
    n_mismatch = 0
    for qid in sorted(qids):
        q = questions_by_qid[qid]
        anchor_ids = list(anchors_by_qid[qid].anchor_token_ids)
        rendered = render_question_prompt(
            tokenizer, q.question,
            system_prompt=cfg.model.system_prompt,
            question_suffix=cfg.data.question_suffix,
            chat_template_kwargs=cfg.model.chat_template_kwargs,
        )
        anchor_text = tokenizer.decode(anchor_ids) if anchor_ids else ""
        prompt_text = rendered.text + anchor_text
        retok = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if list(retok) != list(prompts[qid]) + anchor_ids:
            n_mismatch += 1
        rows.append({"qid": qid, "prompt": prompt_text,
                     "final_answer": q.final_answer, "anchor": anchor_text})
    # The final candidate is graded as anchor+continuation (lora_sft._collect),
    # so the reward must grade the same text or RL optimizes something the
    # pipeline never scores. Under anchor.policy=none every anchor is empty and
    # the field is dropped entirely — the rows, and therefore rl_key, stay
    # byte-identical to the pre-anchor runs.
    if not any(r["anchor"] for r in rows):
        for r in rows:
            del r["anchor"]
    return rows, n_mismatch


def budget_kwargs(params: dict) -> dict:
    """Resolve improve.rl's epochs/steps budget into trainer config kwargs.

    epochs -> max_steps=-1 so transformers derives the step count from the
    dataloader length. With per_device_train_batch_size == num_generations == G
    and steps_per_generation 1, trl's RepeatSampler hands exactly one question
    per process per optimizer step, so one epoch is floor(n_questions /
    world_size) steps — every question seen once, instead of a fixed count that
    only ever reaches the first slice of a large cliff set.
    """
    epochs, steps = params.get("epochs"), params.get("steps")
    if (epochs is None) == (steps is None):
        raise ValueError(f"exactly one of epochs / steps must be set (got {epochs=}, {steps=})")
    if epochs is not None:
        return {"max_steps": -1, "num_train_epochs": float(epochs)}
    # num_train_epochs is ignored once max_steps > 0, but transformers still
    # validates it, so keep its default rather than passing a meaningless value.
    return {"max_steps": int(steps), "num_train_epochs": 3.0}


def report_to(wandb_params: dict | None) -> list[str]:
    """Mirror of train._report_to: per-step RL curves flow through HF Trainer's
    wandb integration as their OWN run, grouped under run.name — the driver's
    metrics run does not exist yet while improve is executing, so joining it is
    not an option. The operator resolves the mode (online may already have been
    downgraded to offline) and passes it here."""
    wp = wandb_params or {}
    if not wp or wp.get("mode") == "disabled":
        return []
    os.environ.setdefault("WANDB_PROJECT", wp["project"])
    os.environ.setdefault("WANDB_RUN_GROUP", wp["group"])
    os.environ.setdefault("WANDB_MODE", wp["mode"])
    if wp.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", wp["entity"])
    return ["wandb"]


def rl_key(params: dict, model_fingerprint: str, adapter_dir, rows: list[dict]) -> str:
    return stable_hash(
        "lora_rl_v1",
        json.dumps(params, sort_keys=True),
        model_fingerprint,
        str(adapter_dir),
        tuple((r["qid"], r["prompt"], r["final_answer"]) for r in rows),
    )


class PlateauCallback(TrainerCallback):
    """Stops training when the reward plateaus (patience comparisons without an
    improvement >= min_delta) or saturates at 1.0; appends every logged reward to
    reward_curve.jsonl regardless.

    The comparison runs on a rolling mean of the last `window` logs, NOT on the
    raw per-step reward: trl logs the mean reward of the questions in the current
    step, and each step draws different questions, so the raw series measures
    question difficulty. window=None resolves on the first log to one epoch of
    steps (epochs budget) or 1 (raw steps budget) — see PlateauCfg.
    """

    def __init__(self, patience: int, min_delta: float, curve_path: Path,
                 window: int | None = None, enabled: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.curve_path = Path(curve_path)
        self.window = window
        # enabled=False still records reward_curve.jsonl, it just never stops.
        self.enabled = enabled
        self.best: float | None = None
        self.stale = 0
        self.stopped_early = False
        self.rewards: list[float] = []

    def _resolve_window(self, args, state) -> int:
        """One epoch of optimizer steps under an epochs budget, else 1. Resolved
        lazily on the first log, where state.max_steps is populated."""
        if self.window is not None:
            return self.window
        epochs = float(getattr(args, "num_train_epochs", 0) or 0)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if getattr(args, "max_steps", -1) > 0 or epochs <= 0 or max_steps <= 0:
            return 1
        return max(1, round(max_steps / epochs))

    def on_log(self, args, state, control, logs=None, **kwargs):
        reward = (logs or {}).get("reward")
        if reward is None:
            return control
        self.window = self._resolve_window(args, state)
        self.rewards.append(float(reward))
        # trl gathers rewards across processes before logging, so every rank sees
        # the SAME value — without this guard a 2-rank run writes each step
        # twice and any row count (rl/n_logged_steps) doubles.
        if getattr(state, "is_world_process_zero", True):
            append_jsonl(self.curve_path, {
                "step": state.global_step, "reward": float(reward),
                # groups whose rewards are all-equal contribute ZERO gradient —
                # on a cliff set this is the number that explains a flat curve
                **{k: logs[k] for k in ("reward_std", "frac_reward_zero_std",
                                        "completions/clipped_ratio", "entropy")
                   if k in (logs or {})},
            })
        if not self.enabled:
            return control
        if reward >= 1.0:  # every group solved — nothing left to optimize
            self.stopped_early = True
            control.should_training_stop = True
            return control
        if len(self.rewards) < self.window:
            return control            # not enough history to compare windows yet
        smoothed = sum(self.rewards[-self.window:]) / self.window
        if self.best is None or smoothed > self.best + self.min_delta:
            self.best = smoothed
            self.stale = 0
        else:
            self.stale += 1
            if self.stale >= self.patience:
                self.stopped_early = True
                control.should_training_stop = True
        return control


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="post-SFT RL on LoRA params (see module docstring)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--cache-key", required=True)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    if is_done(out_dir / "adapter_config.json", config_hash=args.cache_key):
        print(f"[lora_rl] {out_dir} already done, skipping")
        return

    params = json.loads(args.params_json)
    rows = list(read_jsonl(args.dataset))
    if not rows:
        raise SystemExit(f"[lora_rl] {args.dataset} is empty — nothing to train on")
    seed_everything(int(params["seed"]))

    import dataclasses

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if params["algo"] == "grpo":
        from trl import GRPOConfig as ConfigCls, GRPOTrainer as TrainerCls
    else:  # reinforce -> RLOO (REINFORCE with leave-one-out baseline)
        from trl import RLOOConfig as ConfigCls, RLOOTrainer as TrainerCls

    from .registry import VERIFIERS, build
    from . import verifier as _verifier_registration  # noqa: F401

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    policy = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    grader = build(VERIFIERS, params["verifier"])

    def reward_answer_match(prompts, completions, qid=None, final_answer=None,
                            anchor=None, **kw):
        """Grades anchor+completion — the same text lora_sft._collect scores when
        it decides whether a candidate rescued the question. `anchor` arrives as
        a dataset column (absent, hence None, on unanchored runs)."""
        anchors = anchor if anchor is not None else [""] * len(completions)
        return [
            1.0 if grader.verify(
                QuestionRecord(qid=q, question="", final_answer=fa), (a or "") + c,
            ).correct else 0.0
            for q, fa, a, c in zip(qid, final_answer, anchors, completions)
        ]

    cfg_kwargs = dict(
        output_dir=str(out_dir / "trainer"),
        **budget_kwargs(params),       # epochs over the question set, or raw steps
        learning_rate=float(params["lr"]),
        # Sequences per forward/backward. The operator resolved it (and checked
        # micro_batch x world_size x grad_accum % group_size == 0) in
        # config.resolve_batch_shape; questions per optimizer step is that
        # product / group_size.
        per_device_train_batch_size=int(params.get("micro_batch") or params["group_size"]),
        num_generations=int(params["group_size"]),
        gradient_accumulation_steps=int(params["grad_accum"]),
        gradient_checkpointing=bool(params.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=float(params["vllm_gpu_memory_utilization"]),
        vllm_max_model_length=int(params["vllm_max_model_length"]),
        # Long-CoT correction: token-level ratios stay near exp(+-0.1) and do
        # not compound over the sequence the way trl's sequence_* default does
        # (see the RlCfg comment / api_notes finding 24). RLOOConfig declares
        # none of these three — the `allowed` filter below drops them.
        vllm_importance_sampling_correction=bool(
            params.get("vllm_importance_sampling_correction", True)),
        vllm_importance_sampling_mode=params.get(
            "vllm_importance_sampling_mode", "token_truncate"),
        vllm_importance_sampling_cap=float(
            params.get("vllm_importance_sampling_cap", 2.0)),
        temperature=float(params["temperature"]),
        top_p=float(params["top_p"]),
        max_completion_length=int(params["max_completion_length"]),
        epsilon=float(params["epsilon"]),
        beta=float(params["kl_beta"]),
        num_iterations=1,
        loss_type="grpo",
        mask_truncated_completions=True,
        scale_rewards="group",
        bf16=True,
        save_strategy="no",
        report_to=report_to(params.get("wandb")),
        run_name=params.get("wandb", {}).get("name") if params.get("wandb") else None,
        logging_steps=1,
        seed=int(params["seed"]),
    )
    # RLOOConfig lacks the GRPO-only loss knobs — keep only fields the chosen
    # config class actually declares.
    allowed = {f.name for f in dataclasses.fields(ConfigCls)}
    trainer_args = ConfigCls(**{k: v for k, v in cfg_kwargs.items() if k in allowed})

    # Always attached — it owns reward_curve.jsonl; plateau_enabled only gates
    # whether a plateau actually stops training.
    plateau = PlateauCallback(
        int(params["plateau_patience"]), float(params["plateau_min_delta"]),
        out_dir / "reward_curve.jsonl", window=params["plateau_window"],
        enabled=bool(params["plateau_enabled"]),
    )
    trainer = TrainerCls(
        model=policy,
        reward_funcs=reward_answer_match,
        args=trainer_args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        callbacks=[plateau],
    )
    trainer.train()

    # Under multi-GPU (accelerate launch) every rank trains the same LoRA;
    # only rank 0 writes the adapter, meta and .done marker.
    if not trainer.is_world_process_zero():
        print(f"[lora_rl] rank {trainer.args.process_index}: training done, "
              "save handled by rank 0")
        return
    trainer.model.save_pretrained(str(out_dir))
    write_json(out_dir / "rl_meta.json", {
        "algo": params["algo"],
        "epochs": params.get("epochs"),
        # what the epoch budget resolved to (or the configured step cap)
        "steps_planned": int(trainer.state.max_steps),
        "steps_done": int(trainer.state.global_step),
        "reward_first": plateau.rewards[0] if plateau.rewards else None,
        "reward_last": plateau.rewards[-1] if plateau.rewards else None,
        "plateau_stopped": plateau.stopped_early,
        "n_questions": len(rows),
        "world_size": int(trainer.args.world_size),
        "seconds": round(time.time() - t0, 1),
        "seed": int(params["seed"]),
        "params": params,
    })
    mark_done(out_dir / "adapter_config.json", count=len(rows), config_hash=args.cache_key)
    print(f"[lora_rl] saved adapter -> {out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1:])
