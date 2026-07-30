# expert-iter

Expert Iteration with pluggable improvement operators, for math (and later Lean) reasoning.

The loop, per iteration `k`:

1. **rollout** — the policy samples `n` responses per question (vLLM, data-parallel workers).
2. **partition** — a verifier grades every sample; questions split into *solved*
   (correct trajectories kept for training) and *unsolved* (sent to improvement).
3. **anchor** *(extension point I)* — for each unsolved question, pick a failed rollout and
   fix an *anchor prefix* (leading token span we commit to keeping).
4. **improve** *(extension point II)* — an improvement operator continues from
   `question + anchor` and tries to reach the correct answer.
5. **filters** *(extension point III)* — learnability gates: correctness, no residual
   dependence on external feedback, dedup, length, optional policy-logprob gate.
6. **train** *(extension point IV)* — mixed objective over natively-solved trajectories and
   anchor-prefixed improved trajectories (region-weighted SFT and/or anchor-conditioned DPO).
7. Retrain (from base or last checkpoint), then iterate.

## Setup

```bash
bash scripts/setup.sh --skip-lean    # math-only (no Lean/kimina/Mathlib)
bash scripts/setup.sh                # full setup including Lean proof verification
```

## Running

```bash
# full loop
.venv/bin/python -m expert_iter.loop --config configs/ei_default.yaml

# any stage standalone (they compose via JSONL under runs/<name>/iter_k/)
.venv/bin/python -m expert_iter.rollout --config configs/ei_default.yaml \
    --run-dir runs/ei_qwen3_4b --iter 0 --model-path Qwen/Qwen3-4B-Instruct-2507

# smoke test (small model, 1 GPU, 1 iteration)
bash scripts/smoke.sh
```

Config is a single YAML (see `configs/ei_default.yaml`); any field can be overridden on the
CLI with `--override a.b.c=value`. Distributed training backend is selected with
`train.backend: single|zero2|zero3|fsdp2` (accelerate configs under `configs/accelerate/`);
GPU count is never hardcoded — it comes from `engine.gpus` / `CUDA_VISIBLE_DEVICES`.
