"""Single nested config: YAML <-> dataclasses, dot-path CLI overrides, frozen snapshot.

Usage from any stage:
    cfg = Config.load("configs/ei_default.yaml", overrides=["rollout.n=4"])
Unknown YAML keys are hard errors (typos must not silently no-op).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
import types
from typing import Union, get_args, get_origin, get_type_hints

import yaml

from .utils import write_json

LOOP_STAGES = (
    "rollout", "partition", "anchor", "improve", "filters",
    "build_dataset", "train", "eval", "benchmark_eval",
)

# Registered improvement operators (registry.OPERATORS is populated by import
# side effects, which config.py cannot trigger without a cycle — so the names
# are mirrored here to fail a typo at CONFIG LOAD instead of an hour into the
# run). The operator decides what the transient LoRA is fitted on; the
# optional post-SFT RL phase is NOT an operator (improve.rl.enabled).
IMPROVE_OPERATORS = ("self_resample", "lora_sft", "bridge_sft", "staged_bridge_sft",
                     "gold_text", "teacher")

# anchor.params is a free-form dict, so a key belonging to a DIFFERENT policy
# would otherwise be read by nobody and leave the active policy silently on its
# hardcoded defaults (observed: a privileged_divergence run carrying
# fixed_fraction's fraction/min_tokens/max_tokens). Mirrored here for the same
# reason as IMPROVE_OPERATORS — registry.ANCHOR_POLICIES cannot be imported
# without a cycle.
ANCHOR_POLICY_PARAMS = {
    "none": set(),
    "fixed_fraction": {"fraction", "min_tokens", "max_tokens"},
    "privileged_divergence": {"signal", "top_k", "c_sigma", "search_frac",
                              "min_steps", "max_frac"},
}


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass
class WandbCfg:
    project: str = "ei_reasoning"
    entity: str | None = None
    mode: str = "online"          # online | offline | disabled


@dataclass
class RunCfg:
    name: str = "ei_dev"
    output_root: str = "runs"
    seed: int = 17
    wandb: WandbCfg = field(default_factory=WandbCfg)


@dataclass
class ModelCfg:
    base: str = "Qwen/Qwen3-4B-Instruct-2507"
    dtype: str = "bfloat16"
    system_prompt: str | None = None
    # Forwarded to apply_chat_template by ALL prompt-rendering stages. Hybrid
    # Qwen3 (0.6B/4B/8B) takes {enable_thinking: true|false}; Qwen3-*-2507
    # Instruct and R1-Distill templates take no switches — leave empty.
    chat_template_kwargs: dict = field(default_factory=dict)


@dataclass
class EngineCfg:
    gpus: list[int] | None = None      # null -> CUDA_VISIBLE_DEVICES -> device_count
    tensor_parallel: int = 1           # >1 groups GPUs per vLLM worker
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 12288         # must cover the longest scored sequence
    score_batch_size: int = 256
    # Score-mode utilization cap: prompt_logprobs materializes a full-vocab
    # fp32 log_softmax per prefill chunk (~8192 tokens x 151k vocab x 4B ~= 5
    # GiB spike ON TOP of the utilization budget). At 0.90 this OOMed on 16k
    # sequences (A100 80GB, 2026-08-14); score pools therefore run at
    # min(gpu_memory_utilization, this).
    score_gpu_memory_utilization: float = 0.80
    # vLLM's prefill chunk in SCORE mode. The full-vocab fp32 log_softmax is
    # materialized per chunk, so the spike is chunk x vocab x 4 bytes — 4.64 GiB
    # at vLLM's 8192 default on a 152k vocab, which OOMed even at 0.80
    # utilization once the KV cache was sized ("Tried to allocate 4.58 GiB",
    # A100 80GB, 2026-08-16). Halving the chunk halves the spike and leaves the
    # KV cache untouched, which is cheaper than lowering the utilization.
    score_max_num_batched_tokens: int = 4096
    # Requests per llm.generate call in a pool worker. Bounds what a kill
    # costs: results are appended after each chunk, so a crashed sweep resumes
    # at the last chunk instead of redoing the worker's whole shard (hours to
    # days on a full-dataset run). vLLM caps concurrency at max_num_seqs
    # regardless, so chunking at this size does not shrink the running batch.
    generate_chunk_size: int = 256
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    # ---- LoRA serving (needed by the lora_sft improvement operator) ----
    # All default-off: a non-LoRA run constructs a byte-identical vLLM engine.
    enable_lora: bool = False
    max_loras: int = 8                 # adapters co-resident per vLLM batch
    max_lora_rank: int = 16            # must cover improve.lora_sft.fit.r
    # vLLM cap on per-position logprob entries; raise to >= anchor top_k for
    # the privileged_divergence topk_kl signal (vLLM default is 20).
    max_logprobs: int = 20


@dataclass
class DataCfg:
    adapter: str = "openthoughts_math"
    adapter_args: dict = field(default_factory=dict)
    # Appended to every question when rendering the user turn; keeps grading
    # reliable by pinning the final-answer format.
    question_suffix: str = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    eval_holdout: int = 200            # split off by stable qid hash, fixed across iterations
    # Use an EXTERNAL QuestionRecord JSONL as the eval holdout instead of
    # splitting one off the train set. For a holdout that must be a specific
    # curated slice rather than a random sample — e.g. reserved cliff questions,
    # where "rescue rate on cliffs never trained on" is the endpoint and a
    # proportional random holdout would carry too few of them. Mutually
    # exclusive with eval_holdout > 0; qids overlapping the train set are a
    # hard error, since the whole point is that these were never trained on.
    holdout_path: str | None = None
    accumulate: bool = True            # STaR-style union of all iterations' filtered data
    # Held-out-cliff transfer (A/B split, docs/objective_decision_20260823.md §4):
    # path to a JSON file listing qids whose examples must NEVER reach the
    # trainer (train_sft/train_dpo). Rollout/improve still run on them, so they
    # stay measurable ("improved-but-never-trained" = the B set). Accepts a
    # plain list, or a dict with an "exclude" (preferred) or "B" key —
    # scripts/cliff_split.py writes a compatible file. "" = no exclusion.
    exclude_train_qids: str = ""


@dataclass
class RolloutCfg:
    n: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 8192
    capture_logprobs: bool = False


@dataclass
class PartitionCfg:
    verifier: str = "math"
    solved_keep_max: int = 4           # cap correct trajectories kept per question
    solved_selection: str = "shortest"  # shortest | first | random
    # Route a question to improvement (unsolved.jsonl) iff its clean-correct
    # count <= this. 0 == the classic cliff rule (all rollouts failed). With
    # >=1 a question can be BOTH solved (keeps native trajectories) and
    # improvement-eligible.
    cliff_max_correct: int = 0


@dataclass
class AnchorCfg:
    policy: str = "fixed_fraction"     # ⚗ extension point I
    params: dict = field(default_factory=lambda: {"fraction": 0.3, "min_tokens": 32, "max_tokens": 2048})
    base_selection: str = "first_failed"  # first_failed | longest | random


@dataclass
class TeacherCfg:
    model: str | None = None
    gpus: list[int] | None = None
    quantization: str | None = None


@dataclass
class AdaptiveStopCfg:
    """tau_E fit termination (applies to lora_sft AND bridge_sft fits): fit
    eval_every gradient steps -> probe m_rollouts samples per question from the
    round adapter -> stop when the criterion clears tau_e or the hard step cap
    is reached. Disabled (default) = today's fixed fit.steps behavior. Each
    round costs one lora_fit subprocess + one engine boot for the probe."""

    enabled: bool = False
    tau_e: float = 0.5
    max_steps: int = 10                # hard cap on TOTAL gradient steps (spec ~10)
    eval_every: int = 2                # gradient steps per round
    m_rollouts: int = 4                # probe samples per question per round
    criterion: str = "frac_solved"     # frac_solved (confirmed) | mean_p_hat


@dataclass
class LoraFitCfg:
    """Transient LoRA SFT on (x -> y*) pairs (lora_fit.py subprocess)."""

    r: int = 16
    lora_alpha: int = 32
    lr: float = 1.0e-4
    steps: int = 3                     # full-batch gradient steps (spec: 2-4)
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    micro_batch_size: int = 1          # grad-accum reaches the full batch each step
    max_grad_norm: float = 1.0         # 0 disables clipping
    max_pair_tokens: int = 8192        # drop (x -> y*) pairs longer than this
    bf16: bool = True
    # Data-parallel ranks for the fit: null -> all engine GPUs (accelerate DDP,
    # capped at the pair count); 1 forces the single-GPU launcher. The update is
    # mathematically identical either way (DDP averaging is compensated), but
    # not bitwise — so the topology is part of the fit cache key.
    num_processes: int | None = None
    # DDP only: all-reduce every K micro-batches instead of once per step. 0 =
    # once per step, which makes the gap between collectives the whole shard —
    # 264 s at 313 pairs on 2 GPUs, but 1264 s at 1500 pairs, past NCCL's 600 s
    # watchdog, so a healthy large fit would abort. The update is identical
    # either way (averaging running sums == averaging the totals).
    sync_every: int = 8
    adaptive: AdaptiveStopCfg = field(default_factory=AdaptiveStopCfg)


@dataclass
class ProjectBackCfg:
    """Project-back / LoRA scaling (methodology filter 6a; executed in improve):
    sample q_alpha = pi_{theta + alpha*phi} over the alpha grid, pick per problem
    alpha* = min{alpha : P_hat(correct) >= tau_p} (else 1.0); candidates are the
    correct samples at alpha* only."""

    enabled: bool = False
    alphas: list[float] = field(default_factory=lambda: [
        0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    ])
    tau_p: float = 0.25
    m_rollouts: int = 16               # samples per (problem, alpha) for P_hat
    granularity: str = "per_problem"   # per_problem (C2-1 default) | per_batch


@dataclass
class BridgeCfg:
    """bridge_sft operator: LoRA-fit targets are self-generated bridge
    trajectories z+ (base policy prompted WITH y* — privileged, generation-time
    only) instead of gold y*. Default acceptance check is verifier correctness
    only (confirmed); the G5 leakage screens are opt-in."""

    n: int = 8                          # N_B bridge samples per cliff
    temperature: float | None = None    # null -> improve.temperature
    retry_temperature: float = 1.5      # ONE hotter retry pass for B+-empty cliffs
    max_keep: int = 4                   # bridge pairs kept per question
    keep_selection: str = "shortest"    # shortest | longest | random — which accepted
                                        # bridges fill the max_keep quota (also the
                                        # re-rank rule for staged add_bridge merges)
    leakage_rules: bool = False         # G5 regex screen over z+ text (patterns from
                                        # filter.leakage.patterns); default OFF
    judge_enabled: bool = False         # G5 LLM-judge screen over z+ text; default OFF
    max_tokens: int | None = None       # null -> improve.max_tokens
    sample_skipped: bool = True         # B+-empty cliffs still sampled from the
                                        # pooled chunk adapter (confirmed)


@dataclass
class StagedDpoCfg:
    """staged.stage2_objective=dpo: the stage-2 fit optimizes a preference pair
    per bridge instead of NLL. chosen = the verifier-accepted bridge; rejected =
    a failure the CURRENT adapter itself produced in the preceding rollout, so
    the contrast lands exactly on what this policy still gets wrong."""

    beta: float = 0.1                  # DPO temperature
    lr: float | None = None            # null -> fit.lr (DPO usually wants lower)
    sft_weight: float = 0.0            # + w * NLL(chosen) in the SAME loss (RPO-style).
                                       # Pure DPO only widens the chosen-rejected GAP and
                                       # can push both down; this keeps "make chosen
                                       # likely" in the objective. Raise if reward_margin
                                       # climbs while conversion falls.
    negative_selection: str = "random"  # random | longest | shortest | modal (the
                                       # failure boxing the stage's modal wrong answer)
    max_pairs_per_question: int | None = None   # null -> one pair per kept bridge, which
                                       # keeps the chosen side identical to the SFT arm
    reference: str = "init"            # init (the fit's starting weights = the adapter
                                       # that produced the negatives) | base (adapter
                                       # disabled). Identical when chain_adapter: false.


@dataclass
class StagedUlCfg:
    """staged.stage2_objective=ul: NLL(bridge) + mu * bounded unlikelihood on a
    failure the CURRENT adapter produced in the preceding rollout + displacement
    guard on the bridge — the outer cliff objective's three terms (train.py
    L_C + mu*L_N + L_G) without rho/stratification, which are meaningless in a
    full-batch fit with no solved slice. Aimed at conversion: the transient
    adapter absorbs uniform unlikelihood's diffusion/length pathologies that
    disqualified it as a policy objective (L3 S4-v1), and the verifier filters
    them out of everything the operator emits."""

    mu: float = 0.1                    # weight on the unlikelihood term
    delta: float = 0.02                # p clamp (u bounded by -log delta), same
                                       # default as train.sft.cliff.negative.delta
    guard: bool = True                 # + relu(mean chosen NLL - reference mean
                                       # chosen NLL) per pair: the negative must
                                       # not displace the bridge it protects
    lr: float | None = None            # null -> fit.lr
    negative_selection: str = "modal"  # modal | random | longest | shortest —
                                       # modal aims at the adapter's own attractor
                                       # (measured: random hits it only ~62-69%)
    max_pairs_per_question: int | None = None   # null -> one pair per kept bridge
    reference: str = "init"            # guard baseline: init | base (as in dpo)


@dataclass
class StagedCfg:
    """staged_bridge_sft operator: stage-1 bridge LoRA fit (steps = fit.steps)
    -> rollout (b) off the adapter (correct samples = converted cliffs, pooled)
    -> num_stages stage-2 fits (unsolved: bridge pairs, reused or regenerated
    through the adapter; solved: self-generated rollouts, when train_scope is
    full_pool) -> final rollout. Emitted candidates are the pooled rollout
    samples; per-question selection stays in filter.selection."""

    rollout_n: int = 8                 # (b) samples per cliff off the stage-1 adapter
    num_stages: int = 1                # stage-2 fit passes
    chain_adapter: bool = True         # warm-start each stage-2 fit from the previous adapter
    unsolved_targets: str = "reuse_bridge"   # reuse_bridge | regen_bridge | add_bridge
                                             # regen/add generate bridges THROUGH the
                                             # current adapter; regen falls back to the
                                             # stage-1 bridges when generation fails,
                                             # add merges old+new (dedup, re-ranked by
                                             # bridge.keep_selection, capped at
                                             # stage_max_keep)
    stage_bridge_n: int | None = None        # stage-2+ bridge samples per question
                                             # (null -> bridge.n)
    stage_max_keep: int | None = None        # per-question pair cap for stage-2 fits,
                                             # applied after the add_bridge merge
                                             # (null -> bridge.max_keep)
    solved_targets: str = "self_wash_min_c"  # self_wash_min_c | bridge | random |
                                             # longest | shortest (read only when
                                             # train_scope: full_pool)
    train_scope: str = "unsolved_only"       # unsolved_only | full_pool
    stage2_objective: str = "sft"      # sft | dpo | ul — what the stage-2 fit optimizes.
                                       # dpo/ul turn the preceding rollout's FAILURES into
                                       # on-policy negatives (see StagedDpoCfg/StagedUlCfg).
    dpo: StagedDpoCfg = field(default_factory=StagedDpoCfg)
    ul: StagedUlCfg = field(default_factory=StagedUlCfg)
    stage2_steps: int = 2              # full-batch gradient steps per stage-2 fit
    stage1_chunk_size: int = 0         # 0 = ONE stage-1 adapter (pooled); N = one
                                       # adapter per N questions of the stage-1 fit
                                       # set. Sharding exists to keep PAIRS PER
                                       # ADAPTER in the validated regime (~250-320
                                       # pairs) as the cliff set grows — at 107
                                       # cliffs shards of 25 (~85 pairs) HURT, so
                                       # size chunks by pairs, not by question count.
    stage2_chunk_size: int = 0         # 0 = ONE stage-2 adapter; N = one adapter per
                                       # N questions of the stage-2 fit set. When
                                       # stage 1 is sharded too, stage-2 shards are
                                       # SUB-shards of their stage-1 shard (equal
                                       # sizes = one-to-one, a shard just shrinks as
                                       # questions get solved; smaller = each stage-1
                                       # shard is split further, e.g. 100 -> 5x20),
                                       # each warm-starting from the parent adapter.
                                       # Must be <= stage1_chunk_size.
    final_rollout_n: int = 16
    final_rollout_scope: str = "unsolved"    # unsolved | all
    emit: str = "all"                  # all | final_only (which pool rounds reach filters)


@dataclass
class LoraSftCfg:
    fit: LoraFitCfg = field(default_factory=LoraFitCfg)
    adapter_scope: str = "pooled"      # pooled | per_problem
    chunk_size: int = 0                # 0 = one pooled chunk; >0 = n problems per adapter
    refit_budget: int = 0              # per-problem refits for cliffs the pooled
                                       # adapter failed to resolve (0 = off)
    project_back: ProjectBackCfg = field(default_factory=ProjectBackCfg)
    bridge: BridgeCfg = field(default_factory=BridgeCfg)
    staged: StagedCfg = field(default_factory=StagedCfg)


@dataclass
class PlateauCfg:
    """Reward-plateau early stop for the RL phase. DEFAULT OFF: trl logs the
    mean reward of the CURRENT step's questions, and every step draws different
    questions, so the raw per-step series tracks question difficulty rather than
    learning progress — with window=1 the default patience fires at a median of
    step 5 of 53 (simulated over the cliff difficulty distribution, see
    tests/test_lora_rl.py::test_plateau_window_survives_question_noise).
    Averaging over `window` logs is what makes the comparison meaningful."""

    enabled: bool = False
    patience: int = 3                  # windows without reward improvement
    min_delta: float = 0.01
    # Logs averaged before comparing. null -> one epoch of steps under an epochs
    # budget (i.e. "did this epoch beat the last?"), 1 under a raw steps budget.
    window: int | None = None


@dataclass
class RlPromptFilterCfg:
    """Zero-variance prompt filtering before RL (DAPO's "dynamic sampling",
    done ONCE offline instead of by resampling). A group whose rewards are all
    equal has advantage 0 and contributes NO gradient; on the first real cliff
    run 74.2% of groups were all-wrong, so most of the budget bought nothing.
    Every published long-CoT recipe drops these prompts — DAPO resamples until
    0 < correct < G (its single largest ablation win), Skywork-OR1 filters
    pass-rate 0 or 1 offline, ScaleRL calls it zero-variance filtering — and trl
    implements none of it, so it lives here.

    DEFAULT OFF, which keeps a run byte-comparable with the pre-filter arms;
    flipping it on is the paired comparison. Cost is one probe pool
    (m_rollouts samples per question) off the freshly fitted adapter.

    NOTE a cliff set is DEFINED by pass rate 0, i.e. exactly what these recipes
    exclude. What makes the filter meaningful here is that it probes the
    TRANSIENT LoRA, not the base policy: only questions the fit already lifted
    off the floor are trainable by RL."""

    enabled: bool = False
    m_rollouts: int = 8                # probe samples per question (denominator)
    # Keep a question iff min_pass_rate < n_correct/m_rollouts < max_pass_rate.
    # The exclusive defaults drop all-wrong and all-right groups = DAPO's rule;
    # (0.2, 0.8) reproduces the literature's pass-rate-band heuristic.
    min_pass_rate: float = 0.0
    max_pass_rate: float = 1.0


@dataclass
class RlCfg:
    """Optional post-SFT RL on the LoRA params only (improve.rl): runs after
    the fit (and adaptive stop), before candidate sampling. Prompts show NO
    y* — the true target state (x [+ anchor]); reward = answer matching via
    the partition verifier. phi_E replaces the fit adapter for the tail
    (project-back and candidate sampling operate on it)."""

    enabled: bool = False
    algo: str = "grpo"                 # grpo (trl GRPOTrainer) | reinforce (trl RLOOTrainer)
    group_size: int = 8                # G = num_generations
    # Training budget: set EXACTLY ONE of epochs / steps.
    #   epochs — passes over the chunk's question set; trl gets
    #            num_train_epochs (max_steps=-1) and transformers derives the
    #            step count from the dataloader, so the budget scales with the
    #            cliff set instead of silently covering a fixed slice of it.
    #   steps  — raw optimizer-step budget (an escape hatch / hard cap).
    # One optimizer step consumes num_processes questions x group_size
    # rollouts, so an epoch over N questions is floor(N / num_processes) steps
    # (trl's RepeatSampler drops the N mod num_processes remainder; it reshuffles
    # per epoch, so the dropped questions rotate).
    epochs: float | None = 1.0
    steps: int | None = None
    # Questions per optimizer step = num_processes x grad_accum (per_device_train_
    # batch_size is pinned to group_size = ONE group per device slot, so the GPU
    # count alone would otherwise decide the batch). Raising this makes each
    # update average over more questions; trl ties steps_per_generation to it, so
    # the GENERATION batch grows the same way (grad_accum=4, G=8 -> 32 rollouts
    # generated per step) — it costs memory, not extra generation.
    grad_accum: int = 1
    # Sequences per forward/backward (trl's per_device_train_batch_size), i.e.
    # the activation-memory knob — NOT the number of questions per update.
    # null -> group_size, the only value that keeps EVERY (world_size,
    # grad_accum) pair legal (see resolve_batch_shape). Smaller values cut
    # activation memory at long max_completion_length but constrain grad_accum.
    micro_batch_size: int | None = None
    # On by default (as in lora_fit): without it a 4B policy storing full
    # activations for a multi-thousand-token completion dwarfs the colocate vLLM
    # share. trl calls enable_input_require_grads() for PEFT models when this is
    # set, so the LoRA path is safe.
    gradient_checkpointing: bool = True
    # 1e-6 is what both long-CoT math-RL studies at this completion length used
    # (Tina; the LoRA-alpha study, which reported that RAISING it degraded
    # results) and matches DAPO/Skywork full-parameter values. TRL/verl's
    # "LoRA wants 10x the full-FT lr" guidance points at 1e-5 instead — this is
    # a genuine split in the literature, so treat lr as the thing to sweep.
    lr: float = 1.0e-6
    epsilon: float = 0.2               # PPO clip range
    kl_beta: float = 0.0               # 0 (no ref model) or <= 1e-3
    temperature: float | None = None   # null -> improve.temperature
    max_completion_length: int | None = None  # null -> improve.max_tokens
    # ---- vLLM <-> training logprob mismatch correction (GRPO only) ----------
    # The rollouts come out of vLLM but the gradient is computed by a
    # transformers forward, and the two disagree on logprobs for the SAME tokens
    # (different kernels, attention impl, bf16 rounding). trl corrects for it
    # with exp(log pi_train - log pi_vllm).
    #
    # trl's DEFAULT MODE IS UNUSABLE FOR LONG CoT. "sequence_*" exponentiates
    # the SUM over the completion, so the small per-token gaps (measured
    # 0.0002-0.10) compound: ratio 0.983 at 73 tokens, 1.1e-33 at 3755, and 0
    # past ~4k, which multiplies the whole sequence's loss by zero. A 62-step
    # 16k-completion run produced grad_norm == 0 on every step over ~4k tokens
    # (docs/api_notes.md finding 24). Token-level ratios stay near exp(+-0.1),
    # inside the cap, with no accumulation.
    #
    # cap 2.0, not trl's 3.0: corroborated by verl's rollout_is_threshold
    # default, TRL's own recipe configs, and the "Diagnosing TIM" tau_tok.
    vllm_importance_sampling_correction: bool = True
    vllm_importance_sampling_mode: str = "token_truncate"
    vllm_importance_sampling_cap: float = 2.0
    prompt_filter: RlPromptFilterCfg = field(default_factory=RlPromptFilterCfg)
    plateau: PlateauCfg = field(default_factory=PlateauCfg)
    vllm_gpu_memory_utilization: float = 0.3   # colocate vLLM share (per rank)
    # Data-parallel ranks: null -> all engine GPUs (accelerate launch, one
    # colocate vLLM per rank — GPU-verified); 1 forces the single-GPU path.
    num_processes: int | None = None
    backend: str = "trl"               # trl (colocate vLLM) | pool (reserved fallback)
    seam_strict: bool = False          # anchored prompt retokenization seam:
                                       # true = hard-fail on any mismatch, false = count+warn
    seed: int | None = None            # null -> stable_seed(run.seed, "lora_rl", iter, chunk)


def resolve_batch_shape(*, group_size: int, micro_batch_size: int | None,
                        grad_accum: int, world_size: int) -> dict:
    """The RL batch shape trl will actually run, or a ValueError naming the fix.

    trl generates WHOLE groups per optimizer step, so its only hard constraint is

        generation_batch_size = micro_batch x world_size x grad_accum
        must be divisible by group_size

    (rewards are gathered across processes before advantage normalization, so a
    group may straddle ranks). micro_batch is transformers' per_device_train_
    batch_size = sequences per forward/backward = the activation-memory knob;
    questions per optimizer step is generation_batch_size / group_size. At
    micro_batch == group_size the constraint is satisfied for every world_size
    and grad_accum, which is why that is the default.
    """
    micro_batch = micro_batch_size or group_size
    generation_batch_size = micro_batch * world_size * grad_accum
    if generation_batch_size % group_size:
        legal = [g for g in range(1, group_size + 1)
                 if (micro_batch * world_size * g) % group_size == 0]
        raise ValueError(
            f"improve.rl batch shape rejected by trl: micro_batch_size({micro_batch}) x "
            f"world_size({world_size}) x grad_accum({grad_accum}) = {generation_batch_size} "
            f"is not divisible by group_size({group_size}) — trl generates whole groups. "
            f"Legal grad_accum here: {legal or 'none'}; or set "
            f"improve.rl.micro_batch_size=null (= group_size), which makes every "
            f"grad_accum legal at any GPU count."
        )
    return {
        "micro_batch": micro_batch,
        "generation_batch_size": generation_batch_size,
        "questions_per_step": generation_batch_size // group_size,
    }


@dataclass
class ImproveCfg:
    operator: str = "self_resample"    # ⚗ extension point II
    n: int = 8                         # continuations sampled per anchor
    temperature: float = 1.2
    top_p: float = 0.98
    max_tokens: int = 8192
    rounds: int = 1                    # ⚗ multi-round retry budget for future operators
    teacher: TeacherCfg = field(default_factory=TeacherCfg)
    lora_sft: LoraSftCfg = field(default_factory=LoraSftCfg)
    rl: RlCfg = field(default_factory=RlCfg)


@dataclass
class LogprobGateCfg:
    enabled: bool = False
    min_mean_logprob: float = -1.5
    scope: str = "continuation"        # continuation | full


@dataclass
class SelectionCfg:
    """Per-question candidate selection (the CVaR component, methodology 6b).

    Among verified-correct candidates keep the top max_per_question by
    C(y) = S_mean + lambda_tail * S_tail + gamma_dtail * D_tail, computed under
    the STUDENT policy (S_*) and the candidate's generating policy (D_tail).
    method=shortest keeps the legacy shortest-first quota byte-for-byte."""

    method: str = "shortest"           # shortest | c_score | random
    # Measure C(y) even when the method does not rank by it, so arms that select
    # differently (random / shortest) still report comparable C statistics.
    # Costs one extra scoring pass over the survivors.
    always_score: bool = False
    lambda_tail: float = 1.0           # weight on S_tail (0 -> S_mean-only)
    gamma_dtail: float = 0.0           # weight on D_tail (0 -> no q_P scoring pass)
    tail_fraction: float = 0.1         # CVaR tail: mean of the worst ceil(f*T) tokens
    scope: str = "continuation"        # continuation | full (= anchor+continuation)


@dataclass
class LeakageCfg:
    """Leakage filter (methodology 6c), default OFF. The rule gate activates by
    listing "leakage_rules" in filter.gates; the LLM judge is a batched
    generation pass controlled by judge_enabled."""

    judge_enabled: bool = False
    judge_model: str | None = None     # null -> current policy
    judge_max_tokens: int = 16
    judge_temperature: float = 0.0
    # \b-anchored: unanchored "as given" fired on benign substrings ("was given
    # 5 candies"), inflating the measured floor ~2x on the 107-cliff toy runs.
    patterns: list[str] = field(default_factory=lambda: [
        r"주어진 풀이", r"참고 풀이", r"\breference solution\b", r"\bgiven solution\b",
        r"\bprovided solution\b", r"\bmodel solution\b", r"\bofficial solution\b",
        r"\bas given\b", r"\bthe hint\b", r"\baccording to the (solution|reference)\b",
    ])


@dataclass
class FilterCfg:
    gates: list[str] = field(default_factory=lambda: ["correctness", "no_external_context", "length", "dedup"])
    max_total_tokens: int = 10240      # prompt+anchor+continuation cap (must fit train.max_seq_len)
    max_per_question: int = 2          # quota of improved trajectories kept per qid
    logprob_gate: LogprobGateCfg = field(default_factory=LogprobGateCfg)
    selection: SelectionCfg = field(default_factory=SelectionCfg)
    leakage: LeakageCfg = field(default_factory=LeakageCfg)


@dataclass
class NegativeTermCfg:
    """L_N — the attractor negative on the base policy's own modal-wrong failures.

    mode "v0" is the zero-code arm: train.objective sft+dpo with the DPO pairs'
    rejected switched to the modal-wrong sample (train.dpo.rejected_selection);
    no training-code branch reads v0 — it is a validated documentation marker.
    mode "v1" adds bounded-unlikelihood rows (source="negative") to the SFT set:
    per-token loss -log(1 - p) with p clamped <= 1-delta, weighted mu inside the
    rho bracket. Negatives never get CE and never get an appended EOS.
    """

    mode: str = "off"                  # off | v0 | v1
    mu: float = 0.1                    # weight on L_N (v1 only)
    m_per_batch: int = 1               # negative rows per global batch window (v1)
    max_per_question: int = 8          # cap on modal-wrong failures kept per cliff
    delta: float = 0.02                # unlikelihood clamp: p <= 1 - delta
    # Drop the trailing EOS from negative completions before the unlikelihood.
    # DEFAULT false = keep it, which is the documented decision of
    # docs/objective_loss_spec_20260825.md §1 (the attractor is the whole
    # "confidently write and stop" behaviour, so termination is part of the
    # commitment) AND the behaviour of every arm run so far.
    # true is the paired ablation leg: with EOS kept, S4-v1 measured +57% mean
    # generation length, p90 at the max_tokens cap and 4x truncation on held-out
    # cliffs (2026-08-27) — the risk that section flagged, now materialised.
    drop_terminal_eos: bool = False


@dataclass
class GuardCfg:
    """L_G — displacement guard: hinge on each rescued success's mean completion
    CE rising above its stored reference (the C(y) pass's s_mean). Requires the
    scores file, i.e. filter.selection.method=c_score or always_score, and
    selection.scope=continuation (the live region must match the scored one)."""

    enabled: bool = True


@dataclass
class CliffTermCfg:
    """Separately-normalized cliff term (docs/objective_decision_20260823.md §3):

        L = (1-rho)·L_S + rho·(L_C + mu·L_N + L_G)

    enabled=false keeps the train stage byte-identical to the legacy single-
    normalizer loss. When enabled, a stratified sampler places exactly
    m_per_batch improved rows (and negative.m_per_batch negative rows under v1)
    in EVERY global batch window, and L_C normalizes per QUESTION (1/(n_q·T_j))
    so one converted cliff = one unit of loss regardless of rescue count or
    length; per_question_norm=false is the S3-tok ablation (token-normalized
    within the cliff slice, still under its own normalizer)."""

    enabled: bool = False
    rho: float = 0.1                   # per-step share of the cliff bracket, (0,1)
    per_question_norm: bool = True     # false -> S3-tok
    # Cliff rows per global batch window (m_C), or "auto".
    #
    # The sampler sizes an epoch by the SOLVED pool (n_win = n_solved // fill),
    # so a fixed m_C decides coverage only by accident. MEASURED: L3 had 118
    # improved rows against 173 windows -> every row seen 1.47x, but the L5
    # mixes over-sample cliffs ~12x, and at m_C=1 a 6k run would show only ~24%
    # of its rescue trajectories per epoch (a 300-question smoke: 4%).
    # "auto" picks the smallest m_C whose n_win * m_C covers every improved row
    # once — self-adjusting per iteration and per arm, and it resolves to 1 at
    # L3-like ratios, so the validated setting is unchanged there.
    m_per_batch: int | str = 1
    negative: NegativeTermCfg = field(default_factory=NegativeTermCfg)
    guard: GuardCfg = field(default_factory=GuardCfg)


@dataclass
class SftCfg:
    lr: float = 1.0e-5
    epochs: float = 2.0
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    global_batch_size: int = 32
    micro_batch_size: int = 1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    packing: bool = False              # errors if true with non-uniform region_weights
    logging_steps: int = 5
    # ⚗ extension point IV: per-region loss weights.
    # "solution" = the response region of natively-solved examples.
    region_weights: dict = field(default_factory=lambda: {
        "prompt": 0.0, "anchor": 0.0, "continuation": 1.0, "solution": 1.0,
    })
    # ⚗ extension point IV (cliff objective): L = (1-rho)·L_S + rho·(L_C + mu·L_N + L_G)
    cliff: CliffTermCfg = field(default_factory=CliffTermCfg)


@dataclass
class DpoCfg:
    lr: float = 5.0e-7
    beta: float = 0.1
    epochs: float = 1.0
    global_batch_size: int = 16
    micro_batch_size: int = 1
    loss_type: str = "sigmoid"
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    # Which base failure becomes `rejected` in build_dataset's DPO pairs:
    # base_pick = the anchor stage's base_sample_idx pick (legacy);
    # modal_wrong = the rollout carrying the question's modal wrong answer
    # (the attractor) — the S4-v0 negative arm.
    rejected_selection: str = "base_pick"   # base_pick | modal_wrong
    # --- memory. DPO is the memory-critical objective: it runs a policy AND a
    # reference forward over BOTH sides of every pair, and at a 152k vocab the
    # logits/softmax buffers cost ~4.4 MB per token of the longer side. Measured
    # 2026-08-27 on the S4-v0 arm (Qwen3-4B, 2x80 GB, zero2): the defaults below
    # peak at ~70 GiB, turning either of them off OOMs on ordinary 5-6k pairs.
    gradient_checkpointing: bool = True
    # Compute the reference log-probs once up front, then train with the policy
    # forward only. EXACT (the reference is the frozen SFT checkpoint, so its
    # logps cannot change), and roughly halves the per-step transient.
    precompute_ref_log_probs: bool = True
    max_length: int | None = None      # null -> train.max_seq_len. Lower it only
                                       # to fit; trl truncates whole pairs.


@dataclass
class TrainCfg:
    objective: str = "sft"             # sft | dpo | sft+dpo
    init_from: str = "base"            # base (STaR-style) | last
    backend: str = "zero2"             # single | zero2 | zero3 | fsdp2
    max_seq_len: int = 10240
    sft: SftCfg = field(default_factory=SftCfg)
    dpo: DpoCfg = field(default_factory=DpoCfg)


@dataclass
class LoopCfg:
    iterations: int = 4
    stages: list[str] = field(default_factory=lambda: [
        "rollout", "partition", "anchor", "improve", "filters",
        "build_dataset", "train", "eval", "benchmark_eval",
    ])


@dataclass
class PasskCfg:
    k: int = 8
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class BenchmarkCfg:
    """One external benchmark for the benchmark_eval stage.

    `name` alone selects a preset from data.BENCHMARK_PRESETS (aime24, aime25,
    aime26, hmmt25, math500, math500_hard); adapter/adapter_args override or
    extend it. Sampling knobs follow Qwen3 best practices per model mode —
    non-thinking: temp 0.7 / top_p 0.8 / top_k 20; thinking: 0.6 / 0.95 / 20.
    n=1 with temperature 0.0 gives greedy pass@1.
    """

    name: str = ""
    adapter: str = ""                  # "" -> preset lookup by name
    adapter_args: dict = field(default_factory=dict)
    n: int = 8                         # samples per problem -> pass@n / avg@n / maj@n
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20                    # -1 disables
    min_p: float = 0.0
    max_tokens: int = 16384


@dataclass
class EvalCfg:
    greedy_pass1: bool = True
    passk: PasskCfg = field(default_factory=PasskCfg)
    max_tokens: int = 8192
    # ---- benchmark_eval stage ----
    benchmarks: list[BenchmarkCfg] = field(default_factory=list)
    benchmark_every: int = 1           # run benchmarks every N iterations (others skip)
    benchmark_verifier: str = "math_strict"


@dataclass
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    engine: EngineCfg = field(default_factory=EngineCfg)
    data: DataCfg = field(default_factory=DataCfg)
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    partition: PartitionCfg = field(default_factory=PartitionCfg)
    anchor: AnchorCfg = field(default_factory=AnchorCfg)
    improve: ImproveCfg = field(default_factory=ImproveCfg)
    filter: FilterCfg = field(default_factory=FilterCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    loop: LoopCfg = field(default_factory=LoopCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None, overrides: list[str] | None = None) -> "Config":
        raw: dict = {}
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for ov in overrides or []:
            if "=" not in ov:
                raise ValueError(f"override must be key.path=value, got {ov!r}")
            key, value = ov.split("=", 1)
            _set_dotted(raw, key.strip(), yaml.safe_load(value))
        cfg = _from_dict(cls, raw, path="")
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.run.name, self.run.output_root)
        ):
            raise ValueError("run.name/output_root must be non-empty")
        if self.run.wandb.mode not in ("online", "offline", "disabled"):
            raise ValueError(f"run.wandb.mode: {self.run.wandb.mode!r}")
        if not isinstance(self.model.base, str) or not self.model.base.strip():
            raise ValueError("model.base must be non-empty")
        if self.model.dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError(f"model.dtype: {self.model.dtype!r}")
        e = self.engine
        if e.tensor_parallel < 1:
            raise ValueError("engine.tensor_parallel must be >= 1")
        if e.gpus is not None:
            if (not e.gpus or any(not isinstance(g, int) or g < 0 for g in e.gpus)
                    or len(set(e.gpus)) != len(e.gpus)):
                raise ValueError("engine.gpus must be null or a non-empty unique list of non-negative ints")
            if len(e.gpus) % e.tensor_parallel:
                raise ValueError("engine.tensor_parallel must divide len(engine.gpus)")
        if not 0 < e.gpu_memory_utilization <= 1:
            raise ValueError("engine.gpu_memory_utilization must be in (0, 1]")
        if e.score_max_num_batched_tokens < 1:
            raise ValueError("engine.score_max_num_batched_tokens must be >= 1")
        if e.max_model_len < 1 or e.score_batch_size < 1:
            raise ValueError("engine max_model_len/score_batch_size must be >= 1")
        if e.max_loras < 1 or e.max_lora_rank < 1 or e.max_logprobs < 1:
            raise ValueError("engine max_loras/max_lora_rank/max_logprobs must be >= 1")
        if self.rollout.n < 1 or self.rollout.max_tokens < 1:
            raise ValueError("rollout.n/max_tokens must be >= 1")
        if self.rollout.temperature < 0 or not 0 < self.rollout.top_p <= 1:
            raise ValueError("rollout temperature must be >= 0 and top_p in (0, 1]")
        if self.rollout.capture_logprobs:
            raise ValueError("rollout.capture_logprobs=true is not implemented")
        if self.partition.solved_keep_max < 1:
            raise ValueError("partition.solved_keep_max must be >= 1")
        if self.partition.solved_selection not in ("shortest", "first", "random"):
            raise ValueError(f"partition.solved_selection: {self.partition.solved_selection!r}")
        if not 0 <= self.partition.cliff_max_correct < self.rollout.n:
            raise ValueError(
                f"partition.cliff_max_correct ({self.partition.cliff_max_correct}) must be "
                f"in [0, rollout.n) = [0, {self.rollout.n})"
            )
        if self.anchor.base_selection not in ("first_failed", "longest", "random", "min_mean_nll"):
            raise ValueError(f"anchor.base_selection: {self.anchor.base_selection!r}")
        # Only EXPLICIT params are checked: AnchorCfg's default dict is
        # fixed_fraction's, so every config that never writes a params block
        # carries it regardless of policy — and each policy reads its own keys
        # with its own defaults, so that case is legible-but-harmless rather
        # than wrong. Truthiness on `known` skips policy "none" (reads nothing)
        # and any unregistered policy (the registry rejects it later).
        known = ANCHOR_POLICY_PARAMS.get(self.anchor.policy)
        if known and self.anchor.params != AnchorCfg().params:
            stray = sorted(set(self.anchor.params) - known)
            if stray:
                owners = {k: pol for pol, keys in ANCHOR_POLICY_PARAMS.items()
                          for k in keys if k in stray}
                raise ValueError(
                    f"anchor.params {stray} are not read by "
                    f"anchor.policy={self.anchor.policy!r} (it takes "
                    f"{sorted(known) or 'no params'}). "
                    + (f"They belong to {owners}. " if owners else "")
                    + "Leaving them would silently run the active policy on its "
                    "hardcoded defaults — swap the params block with the policy."
                )
        if self.anchor.policy == "privileged_divergence":
            signal = self.anchor.params.get("signal", "realized_logratio")
            if signal not in ("realized_logratio", "topk_kl", "matched_minus_shuffled"):
                raise ValueError(f"anchor.params.signal: {signal!r}")
            top_k = int(self.anchor.params.get("top_k", 100))
            if signal == "topk_kl" and self.engine.max_logprobs < top_k:
                raise ValueError(
                    f"anchor.params.signal=topk_kl needs engine.max_logprobs >= "
                    f"anchor.params.top_k ({top_k}), got {self.engine.max_logprobs}"
                )
            if self.data.adapter == "hf_math" and not self.data.adapter_args.get("include_solution"):
                raise ValueError(
                    "anchor.policy=privileged_divergence needs gold solutions: set "
                    "data.adapter_args.include_solution=true (hf_math), or use a "
                    "local_jsonl file with meta.gold_solution "
                    "(scripts/backfill_gold_solutions.py)"
                )
        if self.improve.n < 1 or self.improve.max_tokens < 1:
            raise ValueError("improve.n/max_tokens must be >= 1")
        if self.improve.temperature < 0 or not 0 < self.improve.top_p <= 1:
            raise ValueError("improve temperature must be >= 0 and top_p in (0, 1]")
        if self.improve.rounds != 1:
            raise ValueError("improve.rounds other than 1 is not implemented")
        if self.improve.operator not in IMPROVE_OPERATORS:
            hint = ""
            if self.improve.operator in ("rl", "grpo", "reinforce", "ppo"):
                hint = (" — RL is not an operator: it is an optional phase that "
                        "refines whatever the operator fitted, enabled with "
                        "improve.rl.enabled=true (+ improve.rl.algo=grpo|reinforce)")
            raise ValueError(
                f"improve.operator: {self.improve.operator!r} is not one of "
                f"{list(IMPROVE_OPERATORS)}{hint}"
            )
        teacher_values = dataclasses.asdict(self.improve.teacher)
        if self.improve.operator == "teacher" or any(v is not None for v in teacher_values.values()):
            raise ValueError("the teacher improvement operator/options are not implemented")
        ls = self.improve.lora_sft
        fit = ls.fit
        if fit.r < 1 or fit.lora_alpha <= 0 or fit.lr <= 0:
            raise ValueError("improve.lora_sft.fit r/lora_alpha/lr must be positive")
        if not 1 <= fit.steps <= 32:
            raise ValueError("improve.lora_sft.fit.steps must be in [1, 32]")
        if not 0 <= fit.dropout < 1:
            raise ValueError("improve.lora_sft.fit.dropout must be in [0, 1)")
        if not fit.target_modules:
            raise ValueError("improve.lora_sft.fit.target_modules must be non-empty")
        if fit.micro_batch_size < 1 or fit.max_grad_norm < 0 or fit.max_pair_tokens < 1:
            raise ValueError(
                "improve.lora_sft.fit micro_batch_size/max_grad_norm/max_pair_tokens invalid"
            )
        if fit.num_processes is not None and fit.num_processes < 1:
            raise ValueError(
                "improve.lora_sft.fit.num_processes must be >= 1 or null (all GPUs)"
            )
        if ls.adapter_scope not in ("pooled", "per_problem"):
            raise ValueError(f"improve.lora_sft.adapter_scope: {ls.adapter_scope!r}")
        if ls.chunk_size < 0 or ls.refit_budget < 0:
            raise ValueError("improve.lora_sft chunk_size/refit_budget must be >= 0")
        pb = ls.project_back
        if (not pb.alphas or sorted(pb.alphas) != list(pb.alphas)
                or len(set(pb.alphas)) != len(pb.alphas)
                or any(not 0 < a <= 1 for a in pb.alphas)):
            raise ValueError(
                "improve.lora_sft.project_back.alphas must be strictly increasing in (0, 1]"
            )
        if 1.0 not in pb.alphas:
            raise ValueError(
                "improve.lora_sft.project_back.alphas must contain 1.0 (the alpha*=1 fallback)"
            )
        if not 0 < pb.tau_p <= 1 or pb.m_rollouts < 1:
            raise ValueError("improve.lora_sft.project_back tau_p/m_rollouts invalid")
        if pb.granularity not in ("per_problem", "per_batch"):
            raise ValueError(f"improve.lora_sft.project_back.granularity: {pb.granularity!r}")
        ad = fit.adaptive
        if not 0 < ad.tau_e <= 1:
            raise ValueError("improve.lora_sft.fit.adaptive.tau_e must be in (0, 1]")
        if not 1 <= ad.eval_every <= ad.max_steps <= 64:
            raise ValueError(
                "improve.lora_sft.fit.adaptive needs 1 <= eval_every <= max_steps <= 64"
            )
        if fit.sync_every < 0:
            raise ValueError("improve.lora_sft.fit.sync_every must be >= 0 (0 = once per step)")
        if ad.m_rollouts < 1:
            raise ValueError("improve.lora_sft.fit.adaptive.m_rollouts must be >= 1")
        if ad.criterion not in ("frac_solved", "mean_p_hat"):
            raise ValueError(f"improve.lora_sft.fit.adaptive.criterion: {ad.criterion!r}")
        br = ls.bridge
        if br.n < 1 or br.max_keep < 1:
            raise ValueError("improve.lora_sft.bridge n/max_keep must be >= 1")
        if br.retry_temperature < 0 or (br.temperature is not None and br.temperature < 0):
            raise ValueError("improve.lora_sft.bridge temperatures must be >= 0")
        if br.max_tokens is not None and br.max_tokens < 1:
            raise ValueError("improve.lora_sft.bridge.max_tokens must be >= 1 or null")
        if br.keep_selection not in ("shortest", "longest", "random"):
            raise ValueError(f"improve.lora_sft.bridge.keep_selection: {br.keep_selection!r}")
        st = ls.staged
        if st.rollout_n < 1 or st.final_rollout_n < 1 or st.num_stages < 1:
            raise ValueError(
                "improve.lora_sft.staged rollout_n/final_rollout_n/num_stages must be >= 1"
            )
        if not 1 <= st.stage2_steps <= 32:
            raise ValueError("improve.lora_sft.staged.stage2_steps must be in [1, 32]")
        if st.stage2_chunk_size < 0 or st.stage1_chunk_size < 0:
            raise ValueError(
                "improve.lora_sft.staged.stage1_chunk_size/stage2_chunk_size must be >= 0 "
                "(0 = pooled)"
            )
        if st.stage1_chunk_size and st.stage2_chunk_size and \
                st.stage2_chunk_size > st.stage1_chunk_size:
            raise ValueError(
                "improve.lora_sft.staged: stage-2 shards are SUB-shards of their stage-1 "
                f"shard, so stage2_chunk_size ({st.stage2_chunk_size}) cannot exceed "
                f"stage1_chunk_size ({st.stage1_chunk_size}); equal = one-to-one shards, "
                "smaller = each stage-1 shard is split further"
            )
        if st.stage1_chunk_size and not st.stage2_chunk_size and st.chain_adapter:
            raise ValueError(
                "improve.lora_sft.staged: a pooled stage-2 fit cannot warm-start from "
                "several stage-1 shard adapters — set stage2_chunk_size to match "
                "stage1_chunk_size, or chain_adapter: false"
            )
        if st.stage2_objective not in ("sft", "dpo", "ul"):
            raise ValueError(
                f"improve.lora_sft.staged.stage2_objective: {st.stage2_objective!r} "
                "(sft | dpo | ul)"
            )
        dpo = st.dpo
        if dpo.beta <= 0:
            raise ValueError("improve.lora_sft.staged.dpo.beta must be > 0")
        if dpo.lr is not None and dpo.lr <= 0:
            raise ValueError("improve.lora_sft.staged.dpo.lr must be > 0 or null")
        if dpo.sft_weight < 0:
            raise ValueError("improve.lora_sft.staged.dpo.sft_weight must be >= 0")
        if dpo.negative_selection not in ("random", "longest", "shortest", "modal"):
            raise ValueError(
                f"improve.lora_sft.staged.dpo.negative_selection: {dpo.negative_selection!r}"
            )
        if dpo.max_pairs_per_question is not None and dpo.max_pairs_per_question < 1:
            raise ValueError(
                "improve.lora_sft.staged.dpo.max_pairs_per_question must be >= 1 or null"
            )
        if dpo.reference not in ("init", "base"):
            raise ValueError(f"improve.lora_sft.staged.dpo.reference: {dpo.reference!r}")
        ul = st.ul
        if ul.mu < 0:
            raise ValueError("improve.lora_sft.staged.ul.mu must be >= 0")
        if not (0.0 < ul.delta < 1.0):
            raise ValueError("improve.lora_sft.staged.ul.delta must be in (0, 1)")
        if ul.lr is not None and ul.lr <= 0:
            raise ValueError("improve.lora_sft.staged.ul.lr must be > 0 or null")
        if ul.negative_selection not in ("random", "longest", "shortest", "modal"):
            raise ValueError(
                f"improve.lora_sft.staged.ul.negative_selection: {ul.negative_selection!r}"
            )
        if ul.max_pairs_per_question is not None and ul.max_pairs_per_question < 1:
            raise ValueError(
                "improve.lora_sft.staged.ul.max_pairs_per_question must be >= 1 or null"
            )
        if ul.reference not in ("init", "base"):
            raise ValueError(f"improve.lora_sft.staged.ul.reference: {ul.reference!r}")
        if st.unsolved_targets not in ("reuse_bridge", "regen_bridge", "add_bridge"):
            raise ValueError(f"improve.lora_sft.staged.unsolved_targets: {st.unsolved_targets!r}")
        if st.stage_bridge_n is not None and st.stage_bridge_n < 1:
            raise ValueError("improve.lora_sft.staged.stage_bridge_n must be >= 1 or null")
        if st.stage_max_keep is not None and st.stage_max_keep < 1:
            raise ValueError("improve.lora_sft.staged.stage_max_keep must be >= 1 or null")
        if st.solved_targets not in ("self_wash_min_c", "bridge", "random", "longest", "shortest"):
            raise ValueError(f"improve.lora_sft.staged.solved_targets: {st.solved_targets!r}")
        if st.train_scope not in ("unsolved_only", "full_pool"):
            raise ValueError(f"improve.lora_sft.staged.train_scope: {st.train_scope!r}")
        if st.final_rollout_scope not in ("unsolved", "all"):
            raise ValueError(
                f"improve.lora_sft.staged.final_rollout_scope: {st.final_rollout_scope!r}"
            )
        if st.emit not in ("all", "final_only"):
            raise ValueError(f"improve.lora_sft.staged.emit: {st.emit!r}")
        if self.improve.operator == "staged_bridge_sft":
            if st.emit == "final_only" and st.final_rollout_scope == "unsolved":
                raise ValueError(
                    "improve.lora_sft.staged: emit=final_only with "
                    "final_rollout_scope=unsolved silently drops early-solved "
                    "questions from the training data (they are absent from the "
                    "final round) — use final_rollout_scope: all"
                )
            # v1 scope: the intermediate rollout IS the fit probe, alpha stays 1.0,
            # and propose() is overridden wholesale — these phases would silently
            # not run, so reject them loudly at load.
            unsupported = [
                ("improve.lora_sft.fit.adaptive.enabled", ad.enabled),
                ("improve.lora_sft.project_back.enabled", pb.enabled),
                ("improve.lora_sft.refit_budget > 0", ls.refit_budget > 0),
                ("improve.lora_sft.adapter_scope != 'pooled'", ls.adapter_scope != "pooled"),
                # stage-2 sharding has its own knob (staged.stage2_chunk_size);
                # the global one has no defined stage to apply to here.
                ("improve.lora_sft.chunk_size != 0 (use "
                 "improve.lora_sft.staged.stage2_chunk_size instead)", ls.chunk_size != 0),
                ("improve.rl.enabled", self.improve.rl.enabled),
            ]
            bad = [name for name, hit in unsupported if hit]
            if bad:
                raise ValueError(
                    "not implemented for improve.operator=staged_bridge_sft yet: "
                    + ", ".join(bad)
                )
        if self.improve.operator in ("lora_sft", "bridge_sft", "staged_bridge_sft"):
            if not self.engine.enable_lora:
                raise ValueError(
                    f"improve.operator={self.improve.operator} needs engine.enable_lora=true"
                )
            if self.engine.max_lora_rank < fit.r:
                raise ValueError(
                    f"engine.max_lora_rank ({self.engine.max_lora_rank}) < "
                    f"improve.lora_sft.fit.r ({fit.r})"
                )
        rl = self.improve.rl
        if rl.algo == "ppo":
            raise ValueError(
                "improve.rl.algo=ppo is not supported: grpo's epsilon IS the PPO clip "
                "(true value-model PPO is deferred) — use improve.rl.algo=grpo"
            )
        if rl.algo not in ("grpo", "reinforce"):
            raise ValueError(f"improve.rl.algo: {rl.algo!r} (grpo | reinforce)")
        if not 0 <= rl.kl_beta <= 1e-3:
            raise ValueError("improve.rl.kl_beta must be 0 or <= 1e-3 (methodology: no/tiny KL)")
        if rl.group_size < 2 or rl.epsilon <= 0 or rl.lr <= 0:
            raise ValueError("improve.rl group_size>=2, epsilon>0, lr>0 required")
        if (rl.epochs is None) == (rl.steps is None):
            raise ValueError(
                "improve.rl: set exactly one of epochs / steps "
                f"(got epochs={rl.epochs!r}, steps={rl.steps!r}) — epochs scales the "
                "budget with the cliff set, steps pins it to a fixed count"
            )
        if rl.epochs is not None and rl.epochs <= 0:
            raise ValueError("improve.rl.epochs must be > 0 or null (then set steps)")
        if rl.steps is not None and rl.steps < 1:
            raise ValueError("improve.rl.steps must be >= 1 or null (then set epochs)")
        if not 0 < rl.vllm_gpu_memory_utilization < 1:
            raise ValueError("improve.rl.vllm_gpu_memory_utilization must be in (0, 1)")
        if rl.num_processes is not None and rl.num_processes < 1:
            raise ValueError("improve.rl.num_processes must be >= 1 or null (all GPUs)")
        if rl.grad_accum < 1:
            raise ValueError("improve.rl.grad_accum must be >= 1")
        if rl.micro_batch_size is not None and rl.micro_batch_size < 1:
            raise ValueError("improve.rl.micro_batch_size must be >= 1 or null (= group_size)")
        # World size is only known here when it is pinned; otherwise the same
        # check runs in _launch_lora_rl, before any GPU work starts.
        if rl.num_processes is not None:
            resolve_batch_shape(group_size=rl.group_size,
                                micro_batch_size=rl.micro_batch_size,
                                grad_accum=rl.grad_accum,
                                world_size=rl.num_processes)
        if rl.plateau.patience < 1 or rl.plateau.min_delta < 0:
            raise ValueError("improve.rl.plateau patience>=1 and min_delta>=0 required")
        if rl.plateau.window is not None and rl.plateau.window < 1:
            raise ValueError("improve.rl.plateau.window must be >= 1 or null (one epoch)")
        is_modes = ("token_truncate", "token_mask", "sequence_truncate", "sequence_mask")
        if rl.vllm_importance_sampling_mode not in is_modes:
            raise ValueError(
                f"improve.rl.vllm_importance_sampling_mode: "
                f"{rl.vllm_importance_sampling_mode!r} (one of {list(is_modes)}). "
                "Prefer a token_* mode for long completions — sequence_* "
                "exponentiates the SUM of per-token logprob gaps and underflows "
                "to zero past a few thousand tokens (api_notes finding 24)"
            )
        if rl.vllm_importance_sampling_cap <= 1:
            raise ValueError(
                "improve.rl.vllm_importance_sampling_cap must be > 1 "
                "(verl/TRL recipes and the Diagnosing-TIM paper all use 2.0)"
            )
        pf = rl.prompt_filter
        if pf.m_rollouts < 2:
            raise ValueError(
                "improve.rl.prompt_filter.m_rollouts must be >= 2 — a single "
                "sample cannot distinguish a mixed group from a uniform one"
            )
        if not 0 <= pf.min_pass_rate < pf.max_pass_rate <= 1:
            raise ValueError(
                "improve.rl.prompt_filter needs 0 <= min_pass_rate < max_pass_rate <= 1 "
                f"(got {pf.min_pass_rate}, {pf.max_pass_rate})"
            )
        if rl.max_completion_length is not None and rl.max_completion_length < 1:
            raise ValueError("improve.rl.max_completion_length must be >= 1 or null")
        # The colocate engine is capped at engine.max_model_len, so the completion
        # budget plus its prompt has to fit inside it.
        rl_completion = rl.max_completion_length or self.improve.max_tokens
        if rl.enabled and rl_completion >= self.engine.max_model_len:
            raise ValueError(
                f"improve.rl completion budget ({rl_completion}) >= engine.max_model_len "
                f"({self.engine.max_model_len}) — the RL vLLM engine is capped at "
                "max_model_len and still has to fit the prompt; lower "
                "improve.rl.max_completion_length or raise engine.max_model_len"
            )
        if rl.temperature is not None and rl.temperature < 0:
            raise ValueError("improve.rl.temperature must be >= 0 or null")
        if rl.backend != "trl":
            raise ValueError(
                "improve.rl.backend=pool is reserved for the engine-pool fallback "
                "(lands only if the trl<->vllm probe fails) — use backend=trl"
            )
        if rl.enabled and self.improve.operator not in ("lora_sft", "bridge_sft"):
            raise ValueError(
                "improve.rl.enabled needs a LoRA operator "
                "(improve.operator=lora_sft|bridge_sft)"
            )
        if self.filter.max_total_tokens < 1 or self.filter.max_per_question < 1:
            raise ValueError("filter token/per-question limits must be >= 1")
        if not self.filter.gates or len(set(self.filter.gates)) != len(self.filter.gates):
            raise ValueError("filter.gates must be a non-empty list without duplicates")
        if self.filter.logprob_gate.scope not in ("continuation", "full"):
            raise ValueError(f"filter.logprob_gate.scope: {self.filter.logprob_gate.scope!r}")
        sel = self.filter.selection
        if sel.method not in ("shortest", "c_score", "random"):
            raise ValueError(f"filter.selection.method: {sel.method!r}")
        if sel.lambda_tail < 0 or sel.gamma_dtail < 0:
            raise ValueError("filter.selection lambda_tail/gamma_dtail must be >= 0")
        if not 0 < sel.tail_fraction <= 1:
            raise ValueError("filter.selection.tail_fraction must be in (0, 1]")
        if sel.scope not in ("continuation", "full"):
            raise ValueError(f"filter.selection.scope: {sel.scope!r}")
        if sel.method == "c_score" and sel.gamma_dtail > 0:
            if self.improve.operator not in ("lora_sft", "bridge_sft") or not self.engine.enable_lora:
                raise ValueError(
                    "filter.selection.gamma_dtail > 0 needs a LoRA operator "
                    "(improve.operator=lora_sft|bridge_sft) and engine.enable_lora=true "
                    "(D_tail scores under the generating LoRA policy; for self_resample "
                    "q == pi_theta makes D_tail identically 0)"
                )
        lk = self.filter.leakage
        if lk.judge_max_tokens < 1 or lk.judge_temperature < 0:
            raise ValueError("filter.leakage judge_max_tokens/judge_temperature invalid")
        if "leakage_rules" in self.filter.gates and not lk.patterns:
            raise ValueError("filter.gates includes leakage_rules but filter.leakage.patterns is empty")
        if self.loop.iterations < 1:
            raise ValueError("loop.iterations must be >= 1")
        if not self.loop.stages or len(set(self.loop.stages)) != len(self.loop.stages):
            raise ValueError("loop.stages must be a non-empty list without duplicates")
        unknown_stages = set(self.loop.stages) - set(LOOP_STAGES)
        if unknown_stages:
            raise ValueError(f"loop.stages contains unknown names: {sorted(unknown_stages)}")
        order = [LOOP_STAGES.index(stage) for stage in self.loop.stages]
        if order != sorted(order):
            raise ValueError(f"loop.stages must follow pipeline order: {LOOP_STAGES}")
        if self.data.eval_holdout < 0:
            raise ValueError("data.eval_holdout must be >= 0")
        if self.data.holdout_path and self.data.eval_holdout > 0:
            raise ValueError(
                "data.holdout_path and data.eval_holdout are mutually exclusive: "
                f"set eval_holdout=0 to use {self.data.holdout_path!r} as the holdout"
            )
        t = self.train
        if t.objective not in ("sft", "dpo", "sft+dpo"):
            raise ValueError(f"train.objective: {t.objective!r}")
        if t.init_from not in ("base", "last"):
            raise ValueError(f"train.init_from: {t.init_from!r}")
        if t.backend not in ("single", "zero2", "zero3", "fsdp2"):
            raise ValueError(f"train.backend: {t.backend!r}")
        if t.max_seq_len < 2:
            raise ValueError("train.max_seq_len must be >= 2")
        if t.sft.lr <= 0 or t.sft.epochs <= 0 or t.dpo.lr <= 0 or t.dpo.epochs <= 0:
            raise ValueError("train learning rates and epochs must be positive")
        required_regions = {"prompt", "anchor", "continuation", "solution"}
        if set(t.sft.region_weights) != required_regions:
            raise ValueError(f"train.sft.region_weights needs exactly {sorted(required_regions)}")
        if any(not isinstance(v, (int, float)) or v < 0 for v in t.sft.region_weights.values()):
            raise ValueError("train.sft.region_weights values must be non-negative numbers")
        if t.sft.packing:
            raise ValueError(
                "train.sft.packing=true is not implemented by WeightedCausalCollator"
            )
        cl = t.sft.cliff
        if cl.negative.mode not in ("off", "v0", "v1"):
            raise ValueError(f"train.sft.cliff.negative.mode: {cl.negative.mode!r} (off | v0 | v1)")
        if t.dpo.rejected_selection not in ("base_pick", "modal_wrong"):
            raise ValueError(
                f"train.dpo.rejected_selection: {t.dpo.rejected_selection!r} (base_pick | modal_wrong)"
            )
        if t.dpo.max_length is not None and not 0 < t.dpo.max_length <= t.max_seq_len:
            raise ValueError(
                f"train.dpo.max_length={t.dpo.max_length} must be in (0, train.max_seq_len="
                f"{t.max_seq_len}] or null (= train.max_seq_len)"
            )
        if cl.negative.mode != "off" and not cl.enabled:
            raise ValueError(
                "train.sft.cliff.negative.mode != off requires train.sft.cliff.enabled "
                "(L_N lives inside the rho bracket)"
            )
        if cl.enabled:
            if not 0 < cl.rho < 1:
                raise ValueError("train.sft.cliff.rho must be in (0, 1)")
            if cl.m_per_batch != "auto" and (
                    not isinstance(cl.m_per_batch, int) or cl.m_per_batch < 1):
                raise ValueError('train.sft.cliff.m_per_batch must be >= 1 or "auto"')
            neg = cl.negative
            if neg.mu < 0:
                raise ValueError("train.sft.cliff.negative.mu must be >= 0")
            if neg.m_per_batch < 1 or neg.max_per_question < 1:
                raise ValueError("train.sft.cliff.negative m_per_batch/max_per_question must be >= 1")
            if not 0 < neg.delta < 1:
                raise ValueError("train.sft.cliff.negative.delta must be in (0, 1)")
            # "auto" is resolved (and clamped to the same budget) inside the
            # sampler, where the row counts are known.
            reserved = ((cl.m_per_batch if cl.m_per_batch != "auto" else 1)
                        + (neg.m_per_batch if neg.mode == "v1" else 0))
            if reserved > t.sft.global_batch_size // 2:
                raise ValueError(
                    f"cliff m_per_batch (+negative) reserves {reserved} of the "
                    f"{t.sft.global_batch_size}-example global batch — must be <= half"
                )
            if neg.mode == "v0" and (t.objective != "sft+dpo" or t.dpo.rejected_selection != "modal_wrong"):
                raise ValueError(
                    "negative.mode=v0 is the zero-code arm: it requires train.objective=sft+dpo "
                    "AND train.dpo.rejected_selection=modal_wrong"
                )
            if neg.mode == "v1" and t.objective != "sft":
                raise ValueError(
                    "negative.mode=v1 requires train.objective=sft (sft+dpo would apply the "
                    "modal-wrong negatives twice: in-SFT unlikelihood + the DPO phase)"
                )
            sel = self.filter.selection
            if cl.guard.enabled and not (sel.method == "c_score" or sel.always_score):
                raise ValueError(
                    "train.sft.cliff.guard needs the C(y) scores file: set "
                    "filter.selection.method=c_score or filter.selection.always_score=true"
                )
            if cl.guard.enabled and sel.scope != "continuation":
                raise ValueError(
                    "train.sft.cliff.guard requires filter.selection.scope=continuation "
                    "(the guard's live region must match the scored region)"
                )
        for name, global_bs, micro_bs in (
            ("sft", t.sft.global_batch_size, t.sft.micro_batch_size),
            ("dpo", t.dpo.global_batch_size, t.dpo.micro_batch_size),
        ):
            if global_bs < 1 or micro_bs < 1 or global_bs % micro_bs:
                raise ValueError(f"train.{name} batch sizes must be positive and global divisible by micro")
        if self.filter.max_total_tokens > t.max_seq_len:
            raise ValueError(
                f"filter.max_total_tokens ({self.filter.max_total_tokens}) exceeds "
                f"train.max_seq_len ({t.max_seq_len}): filtered examples would be truncated."
            )
        if self.engine.max_model_len < t.max_seq_len:
            raise ValueError(
                f"engine.max_model_len ({self.engine.max_model_len}) < train.max_seq_len "
                f"({t.max_seq_len}): logprob scoring of full trajectories would fail."
            )
        if self.eval.passk.k < 1 or self.eval.max_tokens < 1:
            raise ValueError("eval.passk.k/max_tokens must be >= 1")
        if self.eval.passk.temperature < 0 or not 0 < self.eval.passk.top_p <= 1:
            raise ValueError("eval.passk temperature must be >= 0 and top_p in (0, 1]")
        if self.eval.benchmark_every < 1:
            raise ValueError(f"eval.benchmark_every must be >= 1, got {self.eval.benchmark_every}")
        seen_bench = set()
        for i, b in enumerate(self.eval.benchmarks):
            if not b.name:
                raise ValueError(f"eval.benchmarks[{i}]: name is required")
            if b.name in seen_bench:
                raise ValueError(f"eval.benchmarks: duplicate name {b.name!r}")
            seen_bench.add(b.name)
            if b.n < 1:
                raise ValueError(f"eval.benchmarks[{i}] ({b.name}): n must be >= 1")
            if b.temperature < 0 or not 0 < b.top_p <= 1:
                raise ValueError(f"eval.benchmarks[{i}] sampling values are invalid")
            if not 0 <= b.min_p <= 1 or b.top_k < -1 or b.max_tokens < 1:
                raise ValueError(f"eval.benchmarks[{i}] sampling values are invalid")


# ---------------------------------------------------------------------------
# dict -> dataclass with unknown-key errors
# ---------------------------------------------------------------------------

def _from_dict(cls, raw: dict, path: str):
    if not isinstance(raw, dict):
        raise TypeError(f"config section {path or '<root>'} must be a mapping, got {type(raw).__name__}")
    hints = get_type_hints(cls)
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise KeyError(f"unknown config key(s) {sorted(unknown)} under {path or '<root>'}")
    kwargs = {}
    for name, f in known.items():
        if name not in raw:
            continue
        value = raw[name]
        target = hints[f.name]
        # list[Dataclass] fields (e.g. eval.benchmarks): map each element.
        if get_origin(target) is list:
            (elem,) = get_args(target)
            if dataclasses.is_dataclass(elem) and isinstance(value, list):
                kwargs[name] = [
                    _from_dict(elem, v, f"{path}.{name}[{i}]" if path else f"{name}[{i}]")
                    for i, v in enumerate(value)
                ]
                continue
        # Unwrap Optional[X] / unions: pick the sole dataclass member if any.
        union_args = [a for a in get_args(target) if dataclasses.is_dataclass(a)]
        if union_args:
            target = union_args[0]
        if dataclasses.is_dataclass(target) and isinstance(value, dict):
            kwargs[name] = _from_dict(target, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = _coerce_scalar(target, value, f"{path}.{name}" if path else name)
    return cls(**kwargs)


def _coerce_scalar(target, value, path: str):
    """YAML 1.1 parses '2e-5' (no decimal point) as a STRING; coerce scalars to
    the field's declared type so numeric overrides always behave. `X | None`
    fields coerce to X (a `--override a.b=5e-5` on an optional float must not
    arrive as the string '5e-5')."""
    if get_origin(target) in (Union, types.UnionType):
        members = [a for a in get_args(target) if a is not type(None)]
        if value is None or len(members) != 1:
            return value
        target = members[0]
    try:
        if target is float and isinstance(value, (int, str)):
            return float(value)
        if target is int and isinstance(value, str):
            return int(value)
    except ValueError as e:
        raise ValueError(f"config value {path}={value!r} is not a valid {target.__name__}") from e
    return value


def _set_dotted(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            raise ValueError(f"override path {dotted!r} collides with non-mapping value")
    cur[keys[-1]] = value


# ---------------------------------------------------------------------------
# Shared stage CLI
# ---------------------------------------------------------------------------

def stage_argparser(description: str, model_required: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="path to the YAML config")
    p.add_argument("--override", action="append", default=[], metavar="a.b=c",
                   help="dot-path config override (repeatable)")
    p.add_argument("--run-dir", required=True, help="runs/<name> directory")
    p.add_argument("--iter", type=int, required=True, dest="iteration")
    # model_required=False for stages that also run standalone on an arbitrary
    # model (benchmark_eval): they fall back to cfg.model.base. Under the loop
    # driver the current policy is always passed explicitly.
    p.add_argument("--model-path", required=model_required, default=None,
                   help="HF hub id or local checkpoint dir of the CURRENT policy"
                        + ("" if model_required else " (default: cfg.model.base)"))
    return p


def freeze_run_config(cfg: Config, run_dir: str | Path) -> Path:
    """Create the immutable run snapshot, or reject a mismatched resume."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "config.yaml"
    if frozen.exists():
        frozen_cfg = Config.load(frozen)
        if frozen_cfg.hash() != cfg.hash():
            raise ValueError(
                f"run config mismatch for {run_dir}: frozen={frozen_cfg.hash()} "
                f"requested={cfg.hash()}; use the frozen config or a new run.name"
            )
    else:
        cfg.save(frozen)
        write_json(run_dir / "config.hash.json", {"config_hash": cfg.hash()})
    return frozen


def load_stage_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config, overrides=args.override)
    run_dir = Path(args.run_dir)
    freeze_run_config(cfg, run_dir)
    return cfg
