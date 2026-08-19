# expert-iter

Expert Iteration with pluggable improvement operators, for math (and later Lean) reasoning.

## Install

```bash
git clone https://github.com/MS0117/ei-reasoning.git && cd ei-reasoning
bash scripts/setup.sh --skip-lean
```

`--skip-lean` gives the math-only environment; drop it for the full setup including Lean
proof verification (kimina/Mathlib). Everything runs out of the project venv
(`.venv/bin/python`) — the launcher scripts call it directly, so no `activate` needed.

## Pass-rate sweeps

Measure, for every question in a dataset, how many of `K` rollouts the policy gets right —
the table the cliff sets and the stratified sampler are drawn from.

```bash
bash data/run_passrate.sh -c data/configs/passrate_openr1_default.yaml -b

bash data/run_passrate.sh -c data/configs/passrate_openr1_extended.yaml -b
```

`-b` runs in the background (nohup) and prints the log path; add `-g 0,1` to pin GPUs
(the default is every visible device). Each launch creates a fresh timestamped
`runs/passrate/<slug>_<ts>/`; resume a partial run with `-r <that dir>`. Results are
`metrics.json` (histogram + cliff/frontier/solved counts), `question_stats.jsonl`
(per-question `c`/`pass_rate`/`class`) and `samples.jsonl` (every graded generation).
Join the stats back onto the source dataset with `data/join_passrate.py`.

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

**Deep dive:** [`src/expert_iter/README.md`](src/expert_iter/README.md) explains the
architecture, the run-directory data flow, every module's role, and how to add new
anchor policies / operators / gates. API quirks of the pinned bleeding-edge deps are
in [`docs/api_notes.md`](docs/api_notes.md).

## Running

```bash
# full loop (convenience launcher: -g GPUs, -c config, -b background/nohup, -h help)
bash scripts/run.sh -g 0,1,2
bash scripts/run.sh -g 2 -b -- --iterations 1        # background, log under logs/
# equivalent direct call:
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
