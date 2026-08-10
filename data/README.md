# data/ — data-curation code

Code (never artifacts) for characterizing and curating training data for the
EI loop. All generated outputs go under `runs/` (gitignored).

## Pass-rate distribution experiment

Sample N questions from an HF math dataset, generate K rollouts each with the
policy model, grade them, and report how many questions land at each correct
count c = 0/K, 1/K, ..., K/K.

```bash
bash data/run_passrate.sh -g 0,1                  # defaults: N=100, K=8, model.base
bash data/run_passrate.sh Qwen/Qwen3-4B -g 0 -b   # any HF id / EI ckpt / LoRA dir
bash data/run_passrate.sh -- --dry-run            # no GPU: verify data + prompts
```

| hyperparameter | where |
|---|---|
| dataset | `data.adapter_args.hf_name` (+ `config`) in [configs/passrate.yaml](configs/passrate.yaml) |
| N questions | `data.adapter_args.n_questions` (seeded, order-independent sampling) |
| K rollouts | `rollout.n` |
| sampling | `rollout.{temperature, top_p, max_tokens}` (defaults match the EI rollout stage) |
| model | `model.base` or the launcher's `MODEL` argument |
| grading | `partition.verifier` (`math` = lenient, same as EI partition; `math_strict` = requires `\boxed{}`) |
| class thresholds | `passrate.py --frontier-min 1 --solved-min 3` |

Any config key is also overridable ad hoc, e.g.
`bash data/run_passrate.sh -- --override rollout.n=16 --override data.adapter_args.n_questions=1000`.

## Difficulty classes

Each problem x is classified by its correct count c(x) over the K rollouts:

| class | condition (defaults) | meaning |
|---|---|---|
| **cliff** | c = 0 | strict cliff — the model never solves it. **We call hard problems "cliff problems"** (consistent with `cliff_stats` in `src/expert_iter/filters.py`). |
| **frontier** | 1 ≤ c ≤ 2 | rarely solved — the learnability frontier |
| **solved** | c ≥ 3 | reliably within the model's reach |

Thresholds are configurable via `--frontier-min` / `--solved-min`; the resolved
values are recorded in `metrics.json`.

## Outputs (`runs/passrate/<slug>/`)

- `questions.jsonl` — frozen sampled questions (resampled only if adapter args change)
- `samples.jsonl` — graded generations: `{qid, sample_idx, correct, formatted, extracted, finish_reason, n_tokens, response_text}` (same schema as `benchmark_eval`)
- `question_stats.jsonl` — per question: `{qid, n, c, pass_rate, class, n_truncated, final_answer}` — filter by `class` for curation
- `metrics.json` — `hist` (count of questions per c), class counts/fractions, `solve_rate`, `mean_pass_rate`, `passrate/pass@{1,2,4,...,K}`, `passrate/avg@K`, format/truncation rates

Re-running resumes: frozen questions and `.done`-marked samples are skipped;
changing rollout/model params regenerates into a fresh `pool_<hash>` dir.
Truncated generations (`finish_reason == "length"`) are graded incorrect but
tracked via `truncated_rate` / per-question `n_truncated`.
