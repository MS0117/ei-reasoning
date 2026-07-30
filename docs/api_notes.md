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
