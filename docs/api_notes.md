# M0 API notes — verified against the installed environment (2026-07-30)

Raw output: `docs/api_notes_raw.txt` (regenerate with `.venv/bin/python scripts/check_env.py`;
add `--skip-gpu` to keep it CPU-only).

Environment: python 3.11.15 · torch 2.11.0+cu130 · transformers 5.7.0 · trl 1.3.0 ·
accelerate 1.13.0 · vllm 0.20.0 · deepspeed 0.18.9 · flash_attn 2.8.3

## Findings that shaped the code

1. **`apply_chat_template(tokenize=True)` returns a `BatchEncoding` (dict), not a
   flat id list** in transformers 5.x. `list(...)` over it yields the KEYS
   (`'input_ids'`, ...), which crashed vLLM with
   `TypeError: '>' not supported between 'str' and 'int'` during the live test.
   → `templates.py` renders text and encodes it with
   `tokenizer(text, add_special_tokens=False)` — the only text→ids path in the repo.

2. **`TrainingArguments.group_by_length` was removed** in transformers 5.x (112
   fields remain; no length-grouped sampler knob). → feature dropped from the
   config; padding-waste mitigation is a future packed-collator optimization.

3. **Loss normalization contract** (`transformers/trainer.py`):
   - `get_batch_samples` → `_get_num_items_in_batch(batch_samples, device)`;
     overriding the latter is the minimal hook (WeightedSFTTrainer sums
     label-shifted `loss_weights` there, with the same
     `average_tokens_across_devices` gather the base class uses).
   - `training_step` skips its per-accum-step `loss /= accum` division only when
     `model_accepts_loss_kwargs and num_items_in_batch is not None` (it also
     disables DeepSpeed gas-scaling then). WeightedSFTTrainer pins
     `self.model_accepts_loss_kwargs = True` instead of trusting model-signature
     inspection. Verified by `tests/test_loss_invariance.py` (1×8 accum ≡ 8×1).

4. **trl 1.3**: `SFTTrainer(model, args, data_collator, train_dataset,
   processing_class, ...)`, `SFTConfig.completion_only_loss/packing/padding_free`
   exist; `DPOTrainer(model, ref_model, args, ...)` with `DPOConfig.beta`,
   `loss_type`, `max_length`. We use plain `transformers.Trainer` for weighted
   SFT (pre-tokenized data; smaller API surface) and `DPOTrainer` for DPO.

5. **vLLM 0.20**: `SamplingParams`/`LLM.__init__` are not introspectable via
   `inspect.signature` (msgspec/`**kwargs`) — "MISSING" rows in the raw output
   are an introspection artifact, not absent params. `vllm.inputs.TokensPrompt`
   exists. On the RTX PRO 6000 Blackwell the engine loads and warms up
   (flashinfer autotune ran); the harmless startup warning
   `Failed to get device capability: SM 12.x requires CUDA >= 12.9` appears twice.
   Full generate/score round-trip still needs a GPU smoke run
   (`bash scripts/smoke.sh <gpu>` — coordinate GPU usage first).

6. **Qwen3-4B-Instruct-2507 chat template**: renders
   `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`; **no `<think>`
   block** (non-thinking variant), EOS = `<|im_end|>` (151645).

## GPU-verified findings (2026-07-31, RTX A6000 48GB + RTX A4000 16GB, SM 8.6)

7. **DP loss compensation must be replicated when overriding `compute_loss`.**
   Base `Trainer.compute_loss` ends with `loss *= num_processes` when
   `average_tokens_across_devices and model_accepts_loss_kwargs and
   num_items_in_batch is not None`; `training_step` does NOT apply it. An
   override that normalizes by the globally-gathered `num_items` without this
   factor shrinks BOTH the logged loss AND the gradient by exactly world_size
   (DP all-reduce averages gradients). Observed: 2-GPU zero2 logged half the
   single-GPU loss and grad_norm on identical data; after adding the factor,
   steps match (zero2 ≡ single steps 1–3; fsdp2 ≡ single all steps).
   Residual zero2-vs-single late-step difference is DeepSpeed fp32 master
   weights vs bf16-weight update rounding — benign (DS is more precise).

8. **accelerate 1.13 + `deepspeed_config_file` rejects a top-level
   `mixed_precision` key** in the accelerate yaml
   (`ValueError: ...following accelerate config variables will be ignored`).
   bf16 comes from the DS json `bf16.enabled: "auto"` (filled from
   `TrainingArguments.bf16`). → removed from `configs/accelerate/zero{2,3}.yaml`.

9. **vLLM generation is bitwise-reproducible only for a fixed pool topology.**
   Same 1-GPU rerun: 160/160 identical responses. 1-GPU vs 2-GPU pool (or a
   different GPU model): 0/160 identical — batch composition changes kernel
   numerics and temperature-1 sampling diverges. Per-request seeds make results
   independent of *request order*, not of *pool size/sharding*. Plan repro
   accordingly (fix `engine.gpus` for a run you may want to reproduce).

10. **Registry registration is an import side effect** — a stage subprocess that
   consumes a registry must import the module that defines the components.
   `partition`/`filters`/`eval` crashed on GPU with
   `unknown component 'math'; registered: []` (CPU tests import
   `expert_iter.verifier` directly and masked it). → explicit
   `from . import verifier  # noqa: F401` in all three stages.

11. **Killing a pool worker does not kill its vLLM `EngineCore` child** — the
   orphan keeps holding GPU memory and a relaunch fails with
   `Free memory ... less than desired GPU memory utilization`. After killing a
   run manually, check `nvidia-smi` for `VLLM::EngineCore` leftovers.

## Benchmark-eval findings (2026-08-06)

12. **math-verify (installed pin) supports the OPSD grading knobs**:
    `parse(pred, fallback_mode="no_fallback")` (returns `[]` on prose instead
    of guessing) and `verify(gold, target, timeout_seconds=5)` — both verified
    by signature inspection and a live equivalence check
    (`$1/2$` ≡ `$\frac{1}{2}$` → True). `math_strict` in `verifier.py` relies
    on both.

13. **transformers 5.7 `from_pretrained` takes `dtype=` (with `torch_dtype`
    kept as a deprecated alias)** — both appear as `kwargs.pop` in the source.
    `lora.py` uses `dtype=`.

14. **All six benchmark presets load from the hub** (verified 2026-08-06):
    `HuggingFaceH4/aime_2024` (30), `yentinglin/aime_2025` (30),
    `MathArena/aime_2026` (30 — exists!), `MathArena/hmmt_feb_2025` (30),
    `HuggingFaceH4/MATH-500` (500 / 134 at level 5). All expose a
    `problem`-like question column and an `answer`-like gold column that
    `_first_present` resolves.

## LoRA / logprob-K findings (2026-08-13, A100 80GB via slurm step, `gpu_lora_probes`)

15. **vLLM 0.20 per-request LoRA works in BOTH generate and score modes.**
    `LLM(model, enable_lora=True, max_loras=4, max_lora_rank=16)` constructs;
    `llm.generate(prompts, params, lora_request=LoRARequest(name, int_id, path))`
    applies the adapter at generation AND at prefill scoring
    (`SamplingParams(max_tokens=1, prompt_logprobs=K)`): a 2-step probe fit on
    "The answer is 4." moved the target's sum logprob from −9.54 (base) to
    −2.76 (α=1), and greedy generation under the adapter emits the trained
    target. `engine.py` routes this via `GenRequest.lora_path` (one
    `llm.generate` call per adapter group).

16. **The `lora_alpha`-scaled adapter-config copy IS q_α.** A sibling dir with
    `lora_alpha *= α` in `adapter_config.json` and HARDLINKED
    `adapter_model.safetensors` loads fine (vLLM computes scaling =
    lora_alpha/r at load) and interpolates monotonically: α=0.5 scored −4.15,
    between base −9.54 and full −2.76. NOT linear in logprob space (delta
    ratio 0.795, softmax nonlinearity) — project-back only needs monotone
    P̂(α), which holds. `lora.make_alpha_variant` implements the copy.

17. **`max_logprobs=100` + `prompt_logprobs=100` returns 100 entries per
    position with the realized token FIRST** — the `next(iter(d.values()))`
    aggregate and the topk_kl union approximation stay correct at K=100.

18. **GPU memory sequencing holds**: vLLM at 0.85 utilization constructs
    immediately after a `python -m expert_iter.lora_fit` subprocess exits
    (the improve-stage pattern: fit subprocess → pool launch).

19. **peft 0.20 `LoraConfig`/`get_peft_model`/2-step AdamW/`save_pretrained`
    round-trips** on Qwen3 with the standard 7-module target list;
    `lora_fit.py`'s plain-torch loop needs no Trainer.

20. **slurm on srv04: steps attached with `srun --jobid --overlap` get only 1
    of the job's 2 GPUs** (device cgroup exposes one device despite sacct
    granting `gres/gpu=2`; `--whole`/`--gpu-bind=none`/CVD override don't
    help). Run multi-GPU work in the job's PRIMARY interactive step (the
    srun --pty bash shell) until gres.conf/cgroup.conf is fixed; also pass
    `--ntasks=1` to attached steps — the default here duplicated the command
    (two tasks → two vLLM engines on one GPU → OOM).

## trl RL findings (2026-08-13, A100 80GB, `gpu_trl_grpo_probe`)

21. **trl 1.3 GRPO + colocate vLLM WORKS on vllm 0.20 — single AND multi-GPU.**
    Despite trl's import-time warning ("supports vLLM 0.12.0–0.18.0"), a real
    `GRPOTrainer(model=PeftModel(..., is_trainable=True), reward_funcs=<fn>,
    args=GRPOConfig(use_vllm=True, vllm_mode="colocate",
    vllm_enable_sleep_mode=True, vllm_gpu_memory_utilization=0.3,
    num_generations=4))` completed `train()` on Qwen3-0.6B, exercising
    merge_adapter → `llm.load_weights` → `llm.sleep(2)`/`wake_up`. Peak CUDA
    24.9 GiB single-process; 25.0 GiB PER RANK under
    `accelerate launch --num_processes 2` (world_size=2, 2 steps in 5.5 s,
    both ranks trained, both saved). `improve.rl` therefore ships
    `backend=trl`, single-GPU by default and DDP when
    `improve.rl.num_processes > 1` (`_launch_lora_rl` picks the launcher).
    NOTE multi-GPU trl is DDP-ONLY: a plain `python script.py` with 2 visible
    GPUs makes transformers' Trainer wrap the model in DataParallel, which trl
    does not support — probes and production both pin one GPU per process.

22. **Never build a second vLLM engine in a process that already ran a
    colocate trl engine.** After `train()` + sleep/wake, an in-process
    `LLM(...)` dies with `CUDA driver error: invalid argument`; under
    `accelerate launch` a leaked `TORCHELASTIC_*`/`MASTER_*` env var instead
    makes the child engine attempt a torch.distributed rendezvous and hang for
    its full 600 s timeout. Both are teardown/env artifacts, not serving bugs:
    production always serves adapters from a FRESH `engine.py` worker
    subprocess (and `lora_rl` exits after saving). The probe now does the same
    and scrubs every distributed marker from child envs.

## trl RL findings on a REAL long-CoT run (2026-08-15, A100 80GB x2, Qwen3-4B)

23. **`GRPOConfig.vllm_max_model_length` MUST be set — the colocate engine
    otherwise boots at the MODEL's native context.** Left at its `None`
    default, the RL engine tried to serve Qwen3-4B-Instruct-2507's full
    262144-token context and demanded a 36.0 GiB KV cache against the 31.07 GiB
    its `vllm_gpu_memory_utilization=0.5` share allowed:

        ValueError: To serve at least one request with the model's max seq len
        (262144), 36.0 GiB KV cache is needed, which is larger than the
        available KV cache memory (31.07 GiB).

    This is unrelated to the completion budget — no `max_completion_length`
    lowers it. `_maybe_rl` now always passes `vllm_max_model_length =
    engine.max_model_len`, so the RL engine is capped exactly like every other
    pool in the run, and config validation rejects a resolved
    `improve.rl.max_completion_length >= engine.max_model_len` at LOAD time.

24. **trl 1.3's default vLLM importance-sampling correction silently zeroes the
    gradient for long completions.** `GRPOConfig` defaults are
    `vllm_importance_sampling_correction=True`,
    `vllm_importance_sampling_mode="sequence_mask"`, `cap=3.0`. The correction
    compensates for vLLM and the training forward disagreeing on logprobs for
    the same tokens, but `sequence_*` modes exponentiate the SUM over the
    completion:

        ratio = exp( Σ_t (log π_train(t) − log π_vllm(t)) )

    Per-token gaps are small (measured 0.0002–0.10 mean) but they accumulate,
    and `per_token_loss *= importance_sampling_ratio` then multiplies the whole
    sequence by ~0. Measured on a 107-question cliff set, 16384-token budget:

        completion len   mean logp gap   ratio mean   grad_norm
             73            0.00024         0.983       7.3e-04
          1,913            0.0276          3.7e-04     ~0
          3,755            0.0999          1.1e-33      0
          9,162            —               —            0      (loss 0)
         12,340            —               —            0      (loss 1e-43)

    Both tails are lost: an overshoot is zeroed by the `*_mask` cap, an
    undershoot underflows. Only sub-3k completions kept any gradient, so a
    62-step run learned nothing despite 16 steps having a mixed-reward group.
    Use `vllm_importance_sampling_mode="token_truncate"` for long CoT —
    per-token ratios stay near exp(±0.1), inside the cap, with no accumulation.
    trl's defaults assume the few-hundred-token completions of its examples.

    The literature agrees on the mechanism and on the cap. BF16 rollout/training
    mismatch grows *exponentially* with response length (Defeating the
    Training-Inference Mismatch via FP16, arXiv 2510.26788 — FP16 shrinks it
    ~24x), and sequence-level MIS plateaued at 95% vs 99% training accuracy
    there, attributed to "the high variance of its sequence-level importance
    ratio". **Cap 2.0, not trl's 3.0**, is corroborated three independent ways:
    verl's `algorithm.rollout_correction.rollout_is_threshold` default, TRL's
    own recipe configs (`vllm_importance_sampling_clip_max=2.0`), and the
    Diagnosing-TIM paper's tau_tok=2. The tradeoff is real in both directions:
    GSPO (arXiv 2507.18071) argues token-level ratios are ill-posed for
    distribution correction, so token-level = biased but low variance,
    sequence-level = unbiased but variance explodes with length. The best
    configuration reported by Diagnosing TIM combines them (token truncation
    tau=2 + sequence-level *rejection* at tau_seq=0.001). Root-cause fixes that
    sidestep the choice: fp32 LM head (ScaleRL `cast_lm_head_to_fp32`) or fp16
    end-to-end.

    `improve.rl` therefore ships `vllm_importance_sampling_mode=token_truncate`
    and `vllm_importance_sampling_cap=2.0` as defaults, both hyperparameters.

25. **`reward` in trl's logs is the MEAN over the step's completions, so with
    one group per step it is a multiple of 1/G**, not 0/1 — on the cliff set it
    took values {0: 46, 0.125: 4, 0.25: 10, 0.375: 2} over 62 steps (mean
    0.0605, matching the 5.35% per-candidate rate measured offline).
    `frac_reward_zero_std` (fraction of groups whose rewards are all equal →
    advantage 0 → NO gradient) averaged 74.2%, against 75.6% predicted from the
    per-question correct-counts. Log it: on a cliff set most groups are
    all-wrong by construction, and a flat reward curve means "no signal", not
    "the method failed". `loss ≈ 0` is NOT a diagnostic here — with
    `num_iterations=1` the ratio is 1 and the group-centered advantages make the
    scalar loss vanish by construction; `grad_norm` is the number to watch.

    Every published long-CoT recipe removes these prompts, and trl implements
    none of it: DAPO resamples until `0 < correct < G` (generation batch 3x the
    train batch, up to 10 batches — its single largest ablation win, AIME24
    42 -> 50), Skywork-OR1 drops pass-rate-0-or-1 prompts offline, ScaleRL calls
    it zero-variance filtering and also drops prompts once pass rate >= 0.9.
    RL-ZVP (arXiv 2509.21880) measures zero-variance prompts at 30-99% of every
    batch, so 74.2% here is ordinary. `improve.rl.prompt_filter` implements the
    offline form (probe the FIT adapter m times, keep
    `min_pass_rate < p < max_pass_rate`), default OFF so a run stays comparable
    with the pre-filter arms. NOTE the awkward fit with this project: a cliff
    set is DEFINED by pass rate 0, i.e. exactly what those recipes exclude —
    what makes the filter meaningful is that it probes the transient LoRA, not
    the base policy.

26. **Killing an EI GPU job must also kill `VLLM::EngineCore`.** vLLM renames
    its engine subprocesses, so `pkill -f expert_iter` leaves them holding the
    full `gpu_memory_utilization` share; the next launch then dies with
    "Free memory on device cuda:0 (6.36/79.25 GiB) ... less than desired GPU
    memory utilization". Worse, on this cluster `nvidia-smi` run OUTSIDE the
    srun step reports the wrong devices (it showed 0 MiB while the allocated
    pair was full). Check and clean from INSIDE the step:
    `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` then
    `pkill -f "VLLM::EngineCore"`.
