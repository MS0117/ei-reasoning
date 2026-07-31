# `expert_iter` — how the codebase works

Expert Iteration (EI) with a pluggable **improvement operator**. Per iteration the
policy keeps what it already solves and gets a constructed-but-learnable trajectory
for what it cannot solve, then retrains on both.

## The loop at a glance

```
                    ┌──────────────────────────────────────────────────────┐
                    │              iteration k (policy π_k)                │
                    │                                                      │
 questions ──► rollout ──► partition ──┬─► solved.jsonl ────────────┐      │
                (π_k, n samples)       │   (correct trajectories)   │      │
                                       │                            ▼      │
                                       └─► unsolved ─► anchor ─► improve   │
                                                        (I)       (II)    │
                                                                    │      │
                                                              filters (III)│
                                                                    │      │
                                                                    ▼      │
                                                         build_dataset     │
                                                       (mixed SFT/DPO set) │
                                                                    │      │
                                                              train (IV)   │
                                                                    │      │
                    └───────────────── eval ◄── ckpt (π_{k+1}) ◄────┘      │
                                                                           │
                                        π_{k+1} becomes the rollout policy ┘
```

Roman numerals mark the four **research extension points** — each is a registry of
pluggable classes selected by config, with a simple baseline implemented.

## How an iteration actually executes

`loop.py` runs each stage as a **separate subprocess** (`python -m expert_iter.<stage>`;
`train` runs under `accelerate launch`). Stages communicate only through JSONL files
under `runs/<name>/iter_k/`:

```
runs/<name>/
  config.yaml              frozen config snapshot (stages load this)
  state.json  metrics.jsonl
  questions/{train,holdout}.jsonl      split materialized once per run
  iter_0/
    rollout/rollouts.jsonl             one line per sampled response
    partition/{verdicts,solved,unsolved}.jsonl + stats.json (solve-rate)
    anchors/anchors.jsonl              chosen prefixes of failed rollouts
    improve/improved.jsonl             ALL operator attempts (kept + failed)
    filtered/kept.jsonl + report.json  per-gate reject counts
    dataset/train_{sft,dpo}.jsonl      what the trainer reads (incl. accumulation)
    ckpt/                              full HF checkpoint → π_{k+1}
    eval/metrics.json                  pass@1 / pass@k on holdout
    logs/<stage>.log
  latest -> iter_k     latest_ckpt -> iter_k/ckpt
```

Every stage output has a `.done` sidecar tied to the config hash; a stage **skips
itself** if the marker matches. Crash recovery = re-run the same command.
`--force STAGE|all` overrides. Subprocess isolation also guarantees vLLM fully
releases GPUs before training starts (and vice versa).

**Model resolution:** inference stages always use the current policy
(`iter_{k-1}/ckpt`; `model.base` at k=0). Training initializes from `model.base`
(`train.init_from: base`, STaR-style default, usually with `data.accumulate: true`)
or from the current policy (`last`).

## Load-bearing invariants

1. **Token ids are the source of truth; text is for humans.** Anchors are id-slices
   of the failing rollout's `response_token_ids`. Continuation prompts and training
   inputs are built by **concatenating id lists** — never by re-tokenizing decoded
   text, because BPE can merge across a splice point and silently shift region
   boundaries. `templates.py` is the single place where text becomes ids.
2. **Learnability is structural.** Any operator that conditions on information
   beyond `question + anchor` (a hint, critique, teacher text) must record it in
   `ImprovedCandidate.external_context`; the `no_external_context` gate rejects such
   candidates unless a mismatch-absorbing objective is deliberately configured.
3. **Nothing hardcodes GPU topology.** GPU list: `engine.gpus` →
   `CUDA_VISIBLE_DEVICES` → all visible. Grad accumulation is derived at runtime
   from `global_batch / (micro_batch × world_size)`.
4. **Unknown config keys are hard errors** — a typo can never silently no-op.

## File-by-file

### Plumbing
| file | role |
|---|---|
| `config.py` | Nested dataclasses ↔ one YAML; dot-path `--override a.b=c`; validation (e.g. packing × region-weights conflict); config hash for `.done` markers; shared stage CLI (`--config/--run-dir/--iter/--model-path`). |
| `records.py` | Typed records for every stage boundary (`RolloutSample`, `SolvedTrajectory`, `AnchorRecord`, `ImprovedCandidate`, `SFTExample`, `DPOExample`, …) with JSONL (de)serialization and invariant checks (`SFTExample.validate`). |
| `registry.py` | Tiny name→class registries (`ADAPTERS`, `VERIFIERS`, `ANCHOR_POLICIES`, `OPERATORS`, `GATES`). Adding a component = one class + one `@register` decorator. |
| `utils.py` | Atomic JSONL writes (tmp+rename), `.done` markers, stable hashing/seeding, GPU-list resolution, run-dir helpers, atomic symlinks. |
| `templates.py` | The ONE text→ids location: chat-templates the question (`add_generation_prompt=True`), builds continuation prompts / training inputs by id concatenation, `ensure_eos`. |
| `engine.py` | vLLM **data-parallel pool**: shards requests round-robin into per-worker JSONL, spawns one worker subprocess per GPU group (own `CUDA_VISIBLE_DEVICES`), merges results. Two modes: `generate` (sampling; per-request stable seeds ⇒ reproducible for a FIXED pool topology — pool size/GPU model changes alter numerics and thus samples) and `score` (teacher-forced per-token logprobs via `prompt_logprobs`, used by the trainability gate). |

### Data & grading
| file | role |
|---|---|
| `data.py` | `DatasetAdapter` registry → canonical `QuestionRecord{qid, question, final_answer, domain}`. `openthoughts_math` (filters OpenThoughts-114k to verifiable math; never reads its R1 traces), `local_jsonl`. Deterministic qid-hash holdout split, frozen per run. |
| `verifier.py` | `Verifier` registry: `math` (math-verify; gold wrapped in `\boxed{}`, pred = last boxed expression) and `lean` (kimina HTTP client, lazily imported — math-only machines never need the Lean stack). |

### The loop stages
| file | role |
|---|---|
| `rollout.py` | π_k samples `rollout.n` responses per train question → `rollouts.jsonl`. |
| `partition.py` | Grades every sample; questions with ≥1 correct **cleanly-finished** sample → `solved.jsonl` (≤ `solved_keep_max` per question, shortest-first); zero-correct questions → `unsolved.jsonl`; writes `stats.json` (solve-rate = the headline metric). |
| `anchor.py` | ⚗ **(I)** `AnchorPolicy.select_len(question, failed_rollout, params) -> int`. Baselines: `fixed_fraction` (keep first ρ of the failed response, clamped), `none` (pure resample ⇒ STaR/rejection-sampling ablation). |
| `improve.py` | ⚗ **(II)** `ImprovementOperator.propose(...) -> list[ImprovedCandidate]`. Baseline `self_resample`: best-of-n continuation of `question+anchor` by the policy itself at higher temperature (trivially learnability-safe). `teacher` is a stub whose docstring specifies the external-context contract. |
| `filters.py` | ⚗ **(III)** ordered gate chain: `correctness` (verifier on anchor+continuation), `no_external_context`, `length`, `dedup` (continuation-id hash), optional batched `logprob_gate` (mean policy logprob threshold via engine `score` mode), then a `max_per_question` quota (shortest-first). Writes per-gate reject counts. |
| `build_dataset.py` | Assembles train-ready examples. Solved → `[prompt][solution]`; improved → `[prompt][anchor][continuation]` (+EOS). Stores **region lengths, not baked weights**, so weight sweeps need no data rebuild. Builds DPO pairs sharing `prompt+anchor` (chosen = improved continuation, rejected = the same failed rollout's suffix). Merges prior iterations when `data.accumulate`. |
| `train.py` | ⚗ **(IV)** `WeightedSFTTrainer` (subclasses `transformers.Trainer`; per-token region-weighted CE; normalization invariant to micro-batch/accum/DP topology — see `tests/test_loss_invariance.py`) and anchor-conditioned `trl.DPOTrainer`. `objective: sft | dpo | sft+dpo`. Saves one full HF checkpoint (gather flags per backend) with a post-save sanity assertion. |
| `eval.py` | Greedy pass@1 + sampled pass@k / avg@k on the frozen holdout. |
| `loop.py` | The driver described above: stage sequencing, model resolution, resume, symlinks, `metrics.jsonl` aggregation. |

### Outside `src/`
| path | role |
|---|---|
| `configs/ei_default.yaml` | Full config surface with commented defaults (Qwen3-4B). `configs/smoke.yaml`: 1-iteration 0.6B pipeline test. |
| `configs/accelerate/*.yaml` + `configs/deepspeed/*.json` | Backends for `train.backend`: `single`, `zero2` (default; PCIe-friendly), `zero3`, `fsdp2`. `num_processes` is intentionally absent — the driver passes it. |
| `scripts/setup.sh` | uv-based env setup (`--skip-lean` = math-only). `scripts/run.sh`: loop launcher (`-g` GPUs, `-b` nohup). `scripts/smoke.sh`: end-to-end smoke. `scripts/check_env.py`: prints the *installed* API surface (see below). |
| `docs/api_notes.md` | Empirical findings about the pinned bleeding-edge deps that shaped the code (transformers 5.x `apply_chat_template(tokenize=True)` returns a dict; `group_by_length` removed; the Trainer loss-normalization contract; vLLM introspection caveats). Read this before touching trainer/template code. |
| `tests/` | CPU-only: verifier verdicts, anchor slicing, collator region weights (hand-computed loss), config validation, record round-trips, and the grad-accum invariance test. |

## Extending it (the intended workflow)

```python
# 1. a new anchor policy — anchor.py (or a new module imported by it)
@register(ANCHOR_POLICIES, "logprob_dip")
class LogprobDipAnchor(AnchorPolicy):
    def select_len(self, question, failed, params):
        ...  # e.g. cut where per-token logprob first craters

# 2. select it in config — nothing else changes
# anchor: {policy: logprob_dip, params: {...}}
```

Same pattern for operators (`improve.py`), gates (`filters.py`), adapters
(`data.py`), verifiers (`verifier.py`). For a new training objective, extend
`train.py` (the collator already delivers per-token region weights; `objective`
dispatch lives in `main`).
