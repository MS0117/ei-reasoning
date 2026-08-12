# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Expert Iteration (EI) for math (later Lean) reasoning with pluggable improvement operators. Per iteration: rollout → partition (verifier grades) → anchor → improve → filters → build_dataset → train → eval → benchmark_eval, then repeat with the new checkpoint. `eval` tracks the run's own holdout split; `benchmark_eval` grades the just-trained checkpoint on external competition benchmarks (AIME/HMMT/MATH-500) with literature-comparable `math_strict` grading.

## Commands

Always use the project venv (`.venv/bin/python`), never system python. Python is hard-pinned to 3.11 (flash-attn wheel is cp311-only).

```bash
# environment setup (uv-based, no sudo)
bash scripts/setup.sh --skip-lean          # math-only; omit flag for Lean/kimina/Mathlib

# tests (CPU-only; pytest from the dev dependency group)
.venv/bin/python -m pytest                 # testpaths = tests/
.venv/bin/python -m pytest -m "not slow"   # skip tiny-model training-loop tests
.venv/bin/python -m pytest tests/test_anchor.py::test_name   # single test

# full EI loop
.venv/bin/python -m expert_iter.loop --config configs/ei_default.yaml

# any stage standalone (stages compose via JSONL under runs/<name>/iter_k/)
.venv/bin/python -m expert_iter.rollout --config configs/ei_default.yaml \
    --run-dir runs/ei_qwen3_4b --iter 0 --model-path Qwen/Qwen3-4B-Instruct-2507

# end-to-end smoke test: 1 iteration, Qwen3-0.6B, 50 questions, 1 GPU
bash scripts/smoke.sh [GPU_ID]

# standalone benchmark eval (AIME24/25/26, HMMT25, MATH-500-hard) of any model:
# HF hub id, EI checkpoint, or LoRA adapter dir (auto-merged)
bash scripts/eval_bench.sh Qwen/Qwen3-4B-Instruct-2507 [-c configs/bench_eval.yaml] [-g 0,1] [-b]

# run.sh / eval_bench.sh / data/run_passrate.sh all create a FRESH timestamped
# run dir per launch (never overwrite); resume a crashed run with -r <name|dir>


# verify the installed API surface (regenerates data behind docs/api_notes.md)
.venv/bin/python scripts/check_env.py --skip-gpu
```

**GPUs are shared on this machine.** Validate CPU-only (unit tests, config loads, dry runs) and let the user launch GPU jobs (smoke runs, training) after coordinating GPU usage.

## Architecture

**Driver + stage subprocesses.** `expert_iter.loop` runs each stage as its own subprocess (`python -m expert_iter.<stage>`; `train` runs under `accelerate launch`). Stages communicate only through JSONL files under `runs/<name>/iter_k/` (see `STAGE_OUTPUT` in `loop.py`). Each stage writes a `.done` marker tied to the config hash and skips itself if the marker matches — so crash recovery is just re-running the same command; `--force STAGE|all` overrides the skip.

**Model resolution per iteration k:** inference stages use the current policy = `iter_{k-1}/ckpt` (k=0: `model.base`); train initializes from `model.base` when `train.init_from: base` (STaR-style default) or from the current policy when `last`.

**Registry pattern — the four research extension points.** `registry.py` holds name→class registries; a new component is one class + one `@register` decorator, selected by name in config:
- (I) anchor policies (`anchor.py`) — how much of a failed rollout's prefix to keep
- (II) improvement operators (`improve.py`) — how to continue from question+anchor; operators MUST record any conditioning beyond question+anchor in `external_context` so the learnability gate can reject it
- (III) filter gates (`filters.py`) — learnability gates over improved candidates
- (IV) training objective (`train.py`) — region-weighted SFT and/or anchor-conditioned DPO

Dataset adapters (`data.py`) and verifiers (`verifier.py`) use the same registry mechanism. The lean verifier is imported lazily so math-only machines never touch kimina.

**Token-id splicing invariant (load-bearing everywhere).** Token ids are the source of truth; text fields are for human inspection only. Anchors are always id-slices of the original rollout's `response_token_ids`, and prompts/training inputs are built by concatenating id lists — never by re-tokenizing decoded text, because BPE merges across a splice point silently shift region boundaries. `templates.py` is the ONE place text becomes token ids.

**vLLM pool (`engine.py`).** Data-parallel generation/scoring via one subprocess per worker with its own `CUDA_VISIBLE_DEVICES` slice, communicating through sharded JSONL files. Subprocesses (not in-process) because vLLM doesn't tolerate fork and GPUs must be fully released before `accelerate launch`. Per-request seeds are stable, so results are independent of pool size/sharding. GPU count is never hardcoded: `engine.gpus` → `CUDA_VISIBLE_DEVICES` → all visible.

**Config (`config.py`).** Single nested YAML ↔ dataclasses. Unknown YAML keys are hard errors (typos must not silently no-op). Any field overridable via `--override a.b.c=value`. The driver snapshots config into the run dir; stages load that snapshot.

**Training (`train.py`).** `WeightedSFTTrainer` subclasses `transformers.Trainer` (not `trl.SFTTrainer` — data is pre-tokenized with region annotations, and SFTTrainer's prep pipeline is the most version-volatile surface on the pinned bleeding-edge trl). Per-token region weights (`prompt`/`anchor`/`continuation`/`solution`) are the key methodology knob; loss normalization is invariant to micro-batch/accum topology (verified by `tests/test_loss_invariance.py`). DPO uses `trl.DPOTrainer`. Backend selected by `train.backend: single|zero2|zero3|fsdp2` mapping to `configs/accelerate/*.yaml`.

## Pinned bleeding-edge dependencies

transformers 5.7 / trl 1.3 / vllm 0.20 / torch 2.11 are newer than public training recipes. Do not trust documentation memory for these APIs — consult `docs/api_notes.md` (empirically verified findings that shaped the code, e.g. `apply_chat_template(tokenize=True)` returns a dict in transformers 5.x, `group_by_length` was removed, the Trainer loss-normalization contract) and re-verify with `scripts/check_env.py` when in doubt. vllm and flash-attn install from pinned wheel URLs in `[tool.uv.sources]` (CUDA 12.9 builds); their versions live only there.

## Layout notes

Two importable roots (see `pyproject.toml` hatch config): the core library `src/expert_iter/`, and `inference_controlled_reflection/` at the repo root — currently an empty placeholder that keeps the editable build working until the controlled-reflection inference code is copied in.
