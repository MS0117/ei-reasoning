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

**Deep dive:** [`src/expert_iter/README.md`](src/expert_iter/README.md) explains the
architecture, the run-directory data flow, every module's role, and how to add new
operators / gates. API quirks of the pinned bleeding-edge deps are
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

# one-cycle toy-cliff comparison of improvement operators (no train/eval)
bash data/run_toy_cliff.sh -c data/configs/STAGED.yaml -b \
    -- --reuse-rollout runs/toy_cliff/default_LSPO_20260813_155520
```

## L5 main experiment

Five arms on the same 6k question mix, the same held-out 300-cliff set, and the same
trainer.

Run all five **with the same code revision** — the vLLM pool's batch composition changes
kernel numerics, so a different revision draws a different sample and the arm-to-arm
comparison stops being one. No overrides: the arms are aligned by their YAMLs.

### Training (one at a time; each creates a fresh timestamped run dir)

```bash
# 1) Ours — staged bridge operator + cliff objective S3
bash scripts/run.sh -c configs/methods/l5_staged_dpo_s3.yaml -b

# 2) LSPO — transient LoRA fitted on gold y* (operator control)
bash scripts/run.sh -c configs/methods/l5_lspo.yaml -b

# 3) Gold-in-loop — gold y* verbatim as the cliff row (data-source control)
bash scripts/run.sh -c configs/methods/l5_gold_inloop.yaml -b

# 4) RFT / ReST-EM — solved trajectories only (standard baseline)
bash scripts/run.sh -c configs/methods/l5_rft.yaml -b

# 5) Gold SFT — offline distillation on y*, no rollout (dedicated launcher)
bash scripts/l5_gold_sft.sh -b
```

Arms 1-4 run 3 iterations (final checkpoint `iter_2`); arm 5 runs one (`iter_0`).
Resume a crashed run with `-r <run name>` — stages, shards and rows all skip what is done.

### Evaluation (after a run reaches `[loop] done`)

`<arm>` below is the run directory, e.g. `runs/l5_staged_dpo_s3_20260901_093000`.

```bash
# competition benchmarks — aime24/25/26 + hmmt25 at n=64, math500_hard at n=8
bash scripts/eval_bench.sh <arm>/iter_2/ckpt -b          # arms 1-4
bash scripts/eval_bench.sh <arm>/iter_0/ckpt -b          # arm 5

# held-out cliff transfer, per arm (the paper's headline measurement)
.venv/bin/python scripts/cliff_reroll.py --run-dir <arm> \
    --qids-file holdout --n 32 --passes 1 \
    --model-path <arm>/iter_2/ckpt --out <arm>/headline

# base floor — ONCE for the whole experiment, shared by every arm
.venv/bin/python scripts/cliff_reroll.py --run-dir <any arm> \
    --qids-file holdout --n 32 --passes 2 --out runs/floor_holdout
```

`--qids-file holdout` resolves to the run's frozen `questions/holdout.jsonl` (the 300
reserved cliffs), so no path is typed by hand and every arm measures the same questions.
Omitting `--model-path` silently grades `model.base` instead of the checkpoint — that is
what the floor wants, and what an arm must never do, so check `summary.json:model_path`
afterwards.

Cliffs are selected as "0 correct out of 8", so re-sampling alone rescues a share of them;
the floor is that share (L3 measured 29% coverage / avg@32 0.020 with no training at all),
and its two passes double as the null check for the measurement procedure.

Metrics come from `scripts/attractor_mass.py` over the resulting `verdicts.jsonl`:
`mean_p_top1` (attractor mass — the primary endpoint), `mean_pass_rate` (= avg@32),
`frac_pass_gt0` (>=1-correct coverage), plus per-question paired sign tests.

### Handing results back

Checkpoints are 7.9 GB each; everything the analysis needs is ~5 MB per run.

```bash
bash scripts/collect_run_artifacts.sh runs/l5_* runs/floor_holdout runs/bench/*
# -> l5_artifacts_<timestamp>.tar.gz
```


Toy-cliff experiments (which dataset, which YAML, which code to touch, results so far and the
analysis tooling) are documented in [`docs/toy_cliff_playbook.md`](docs/toy_cliff_playbook.md);
committed per-run summaries live under `docs/results/toy_cliff/`.

Config is a single YAML (see `configs/ei_default.yaml`); any field can be overridden on the
CLI with `--override a.b.c=value`. Distributed training backend is selected with
`train.backend: single|zero2|zero3|fsdp2` (accelerate configs under `configs/accelerate/`);
GPU count is never hardcoded — it comes from `engine.gpus` / `CUDA_VISIBLE_DEVICES`.
