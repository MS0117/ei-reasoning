"""Improvement operator (II): transient LoRA-SFT ("vanilla" baseline + project-back).

Methodology: fit a transient LoRA phi on (x -> y*) pairs of the cliff batch
(2-4 gradient steps; y* flows ONLY through the adapter weights, never any
prompt), then sample continuations of x [+ anchor] from q_phi and keep what
verifies. Optional project-back sweeps alpha-scaled adapters
q_alpha = pi_{theta + alpha*phi} and keeps samples from the smallest alpha
whose success rate clears tau_p (the most student-like policy that solves it).

Per adapter chunk (adapter_scope pooled | per_problem, chunk_size for pooled
batching):
  1. write pre-tokenized pairs, run `python -m expert_iter.lora_fit` as a GPU
     subprocess (exits => GPU freed before any vLLM pool starts);
  2. ONE run_pool generate launch covering every (question, alpha) with
     per-request lora_path — adapters mix freely inside one pool;
  3. verify inline (P_hat per (question, alpha) needs it; the filters stage's
     correctness gate sees `correct` prefilled and does not re-verify);
  4. emit ALL stop-finished samples at alpha* (correct and incorrect — the
     gate chain and quota do the trimming, matching self_resample semantics);
  5. cliffs with zero correct samples at ANY alpha optionally get a
     per-problem refit (improve.lora_sft.refit_budget), whose candidates
     REPLACE the (all-incorrect) first-pass candidates for those questions.

Learnability bookkeeping: external_context stays None — the gold solution is
weights-channel conditioning, not shown text; op_meta records the provenance
({"privileged": "gold_solution", "channel": "lora_weights"}) and the C(y)
selection / leakage gates own the residual risk.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .config import Config, resolve_batch_shape
from .engine import GenRequest, _model_fingerprint, run_pool
from .lora import fmt_alpha, make_alpha_variant
from .records import ImprovedCandidate, QuestionRecord
from .registry import OPERATORS, VERIFIERS, build, register
from .templates import continuation_prompt_ids, gold_pair_ids
from .wandb_utils import resolve_wandb_mode
from .utils import (
    is_done,
    read_json,
    stable_hash,
    stable_seed,
    visible_gpus,
    write_json,
    write_jsonl,
)
from . import verifier  # noqa: F401 — @register side effect on VERIFIERS


def wandb_params(cfg: Config, *, run_name: str) -> dict:
    """Settings for a training subprocess's OWN wandb run, grouped under
    run.name — the convention train.py and lora_rl.py already follow (the
    driver's metrics run does not exist yet while improve is executing, so
    joining it is not an option). resolve_wandb_mode downgrades a keyless
    "online" to offline so a background job never blocks on auth."""
    return {
        "project": cfg.run.wandb.project,
        "entity": cfg.run.wandb.entity,
        "mode": resolve_wandb_mode(cfg.run.wandb.mode),
        "group": cfg.run.name,
        "name": run_name,
    }


def select_alpha_star(p_hat_by_alpha: dict[float, float], tau_p: float) -> float:
    """alpha* = min{alpha : P_hat(alpha) >= tau_p}, else 1.0 (methodology C2)."""
    for alpha in sorted(p_hat_by_alpha):
        if p_hat_by_alpha[alpha] >= tau_p:
            return alpha
    return 1.0


@register(OPERATORS, "lora_sft")
class LoraSftOperator:
    """Duck-typed to improve.ImprovementOperator, deliberately NOT subclassed:
    improve.py imports this module for its registration side effect, and when
    improve runs as `python -m expert_iter.improve` an import back into
    `expert_iter.improve` would re-execute its module body under the canonical
    name — duplicate operator registrations. Keeping this module free of
    improve imports breaks the cycle."""

    required_models = ("policy",)
    # Provenance stamped into every candidate's op_meta. Subclasses (e.g.
    # bridge_sft) extend this to describe HOW y* reached the weights.
    PROVENANCE = {"privileged": "gold_solution", "channel": "lora_weights"}

    def propose(self, questions, anchors, prompts, cfg: Config, *,
                model_paths, work_dir, iteration, gold_solutions=None):
        gold_solutions = gold_solutions or {}
        ls = cfg.improve.lora_sft
        pb = ls.project_back
        pool_base = Path(work_dir)
        improve_dir = pool_base.parent
        adapters_dir = improve_dir / "adapters"
        policy = model_paths["policy"]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(policy)
        grader = build(VERIFIERS, cfg.partition.verifier)

        anchors_by_qid = {a.qid: a for a in anchors}
        questions_by_qid = {q.qid: q for q in questions}
        eligible = sorted(q.qid for q in questions if q.qid in anchors_by_qid)
        with_gold = [qid for qid in eligible if gold_solutions.get(qid)]
        stats: dict = {
            "operator": self.name,
            "n_eligible": len(eligible),
            "n_with_gold": len(with_gold),
            "n_no_gold": len(eligible) - len(with_gold),
        }
        if eligible and not with_gold:
            raise RuntimeError(
                f"{self.name}: no eligible question has meta.gold_solution — set "
                "data.adapter_args.include_solution=true or backfill the dataset "
                "(scripts/backfill_gold_solutions.py)"
            )
        if not with_gold:
            write_json(improve_dir / "stats.json", stats)
            return []

        # ---- fit-target building (subclass hook) + adapter fits ------------
        targets = self._build_targets(
            with_gold, tokenizer=tokenizer, prompts=prompts,
            anchors_by_qid=anchors_by_qid, questions_by_qid=questions_by_qid,
            grader=grader, cfg=cfg, policy=policy, pool_base=pool_base,
            improve_dir=improve_dir, iteration=iteration,
            gold_solutions=gold_solutions, stats=stats,
        )
        chunks = self._chunks(with_gold, ls)
        fit_params = {
            "r": ls.fit.r, "lora_alpha": ls.fit.lora_alpha, "lr": ls.fit.lr,
            "steps": ls.fit.steps, "dropout": ls.fit.dropout,
            "target_modules": list(ls.fit.target_modules),
            "micro_batch_size": ls.fit.micro_batch_size,
            "max_grad_norm": ls.fit.max_grad_norm, "bf16": ls.fit.bf16,
            "sync_every": ls.fit.sync_every,
        }
        sample_unpaired = self._sample_unpaired(cfg) and ls.adapter_scope == "pooled"
        n_pairs = n_dropped_long = 0
        chunk_of: dict[str, str] = {}
        adapter_of: dict[str, Path] = {}
        fit_meta_of: dict[str, dict] = {}
        fit_seconds: list[float] = []
        for ci, chunk_qids in enumerate(chunks):
            name = f"q_{chunk_qids[0]}" if ls.adapter_scope == "per_problem" else f"pooled_c{ci}"
            pairs = []
            for qid in chunk_qids:
                for pair in targets.get(qid, []):
                    if len(pair["input_ids"]) > ls.fit.max_pair_tokens:
                        n_dropped_long += 1
                        continue
                    pairs.append(pair)
            if not pairs:
                continue  # nothing to fit on -> chunk's questions get no adapter
            n_pairs += len(pairs)
            seed = stable_seed(cfg.run.seed, "lora_fit", iteration, name)
            adapter_dir, secs = self._fit(
                policy, pairs, adapters_dir, name, {**fit_params, "seed": seed}, cfg,
                probe_ctx={
                    "chunk_qids": sorted({p["qid"] for p in pairs}),
                    "prompts": prompts, "anchors_by_qid": anchors_by_qid,
                    "questions_by_qid": questions_by_qid, "tokenizer": tokenizer,
                    "grader": grader, "pool_base": pool_base,
                    "iteration": iteration, "stats": stats,
                },
            )
            fit_seconds.append(secs)
            paired = {p["qid"] for p in pairs}
            rl_qids = [q for q in chunk_qids if q in paired or sample_unpaired]
            adapter_dir, rl_info = self._maybe_rl(
                adapter_dir, rl_qids, name, cfg=cfg, policy=policy,
                prompts=prompts, anchors_by_qid=anchors_by_qid,
                questions_by_qid=questions_by_qid, tokenizer=tokenizer,
                adapters_dir=adapters_dir, iteration=iteration, stats=stats,
                pool_base=pool_base, grader=grader,
            )
            for qid in chunk_qids:
                if qid in paired or sample_unpaired:
                    chunk_of[qid] = name
                    adapter_of[qid] = adapter_dir
                    fit_meta_of[qid] = {**fit_params, "n_pairs": len(pairs),
                                        **({"rl": rl_info} if rl_info else {})}
        stats.update(n_pairs=n_pairs, n_dropped_long=n_dropped_long,
                     n_fits=len(fit_seconds), fit_seconds=fit_seconds)

        # ---- sampling pass over (question, alpha) ---------------------------
        alphas = list(pb.alphas) if pb.enabled else [1.0]
        n_per_req = pb.m_rollouts if pb.enabled else cfg.improve.n
        sampled_qids = sorted(adapter_of)
        results, n_budget_skipped = self._sample(
            sampled_qids, alphas, n_per_req, prompts, anchors_by_qid, adapter_of,
            cfg, policy, pool_base / "main", iteration, tag="",
        )
        stats["n_budget_skipped"] = n_budget_skipped
        candidates, correct_total, curve_rows = self._collect(
            results, questions_by_qid, anchors_by_qid, prompts, tokenizer, grader, cfg,
            n_per_req, pb, chunk_of, adapter_of, fit_meta_of, iteration, refit=False,
        )
        resolved = {qid for qid, n_ok in correct_total.items() if n_ok > 0}
        stats["n_resolved_pool"] = len(resolved)

        # ---- per-problem refit for unresolved cliffs ------------------------
        n_resolved_refit = 0
        refit_qids: list[str] = []
        if ls.refit_budget > 0:
            refit_qids = [q for q in sampled_qids if q not in resolved][:ls.refit_budget]
        refit_qids = [q for q in refit_qids if targets.get(q)]  # need fit pairs to refit
        if refit_qids:
            refit_adapter_of: dict[str, Path] = {}
            refit_meta_of: dict[str, dict] = {}
            for qid in refit_qids:
                pairs = [p for p in targets[qid]
                         if len(p["input_ids"]) <= ls.fit.max_pair_tokens]
                if not pairs:
                    continue
                seed = stable_seed(cfg.run.seed, "lora_fit", iteration, f"refit_{qid}")
                adapter_dir, secs = self._fit(
                    policy, pairs, adapters_dir, f"refit_{qid}",
                    {**fit_params, "seed": seed}, cfg,
                    probe_ctx={
                        "chunk_qids": [qid],
                        "prompts": prompts, "anchors_by_qid": anchors_by_qid,
                        "questions_by_qid": questions_by_qid, "tokenizer": tokenizer,
                        "grader": grader, "pool_base": pool_base,
                        "iteration": iteration, "stats": stats,
                    },
                )
                fit_seconds.append(secs)
                refit_adapter_of[qid] = adapter_dir
                refit_meta_of[qid] = {**fit_params, "n_pairs": len(pairs)}
            refit_qids = sorted(refit_adapter_of)
            refit_results, skipped = self._sample(
                refit_qids, alphas, n_per_req, prompts, anchors_by_qid, refit_adapter_of,
                cfg, policy, pool_base / "refit", iteration, tag="refit",
            )
            stats["n_budget_skipped"] += skipped
            refit_cands, refit_correct, refit_curves = self._collect(
                refit_results, questions_by_qid, anchors_by_qid, prompts, tokenizer,
                grader, cfg, n_per_req, pb, {q: f"refit_{q}" for q in refit_qids},
                refit_adapter_of, refit_meta_of, iteration, refit=True,
            )
            n_resolved_refit = sum(1 for q in refit_qids if refit_correct.get(q, 0) > 0)
            # refit candidates/curves REPLACE the (all-incorrect) first-pass ones
            refit_set = set(refit_qids)
            candidates = [c for c in candidates if c.qid not in refit_set] + refit_cands
            curve_rows = [r for r in curve_rows if r["qid"] not in refit_set] + refit_curves
            resolved |= {q for q in refit_qids if refit_correct.get(q, 0) > 0}
        stats.update(
            n_fits=len(fit_seconds),
            n_refits=len(refit_qids),
            n_resolved_refit=n_resolved_refit,
            lora_yield=round(len(resolved) / len(eligible), 4) if eligible else 0.0,
        )
        per_q_alpha = {c.qid: c.op_meta["alpha_star"] for c in candidates}
        stats["alpha_star_hist"] = dict(Counter(fmt_alpha(a) for a in per_q_alpha.values()))
        stats["alpha_star_mean"] = (
            round(sum(per_q_alpha.values()) / len(per_q_alpha), 4) if per_q_alpha else None
        )
        # Per-question P_hat(alpha) curves — kept separate from candidates so a
        # question that emitted ZERO candidates (e.g. all samples truncated at
        # alpha*) still contributes its curve to alpha-survival analysis.
        write_jsonl(improve_dir / "alpha_curves.jsonl",
                    sorted(curve_rows, key=lambda r: r["qid"]))
        write_json(improve_dir / "stats.json", stats)
        print(f"[improve:lora_sft] {stats}")
        return candidates

    # -- subclass hooks ------------------------------------------------------

    def _build_targets(self, qids: list[str], *, tokenizer, prompts, anchors_by_qid,
                       questions_by_qid, grader, cfg: Config, policy: str,
                       pool_base: Path, improve_dir: Path, iteration: int,
                       gold_solutions: dict[str, str], stats: dict,
                       ) -> dict[str, list[dict]]:
        """qid -> pre-tokenized fit pairs [{qid, input_ids, prompt_len}, ...].
        A qid absent from the result contributes no pairs (and, unless
        _sample_unpaired, no adapter). Base: one gold (x -> y*) pair per qid."""
        eos = tokenizer.eos_token_id
        out: dict[str, list[dict]] = {}
        for qid in qids:
            ids, plen = gold_pair_ids(tokenizer, prompts[qid], gold_solutions[qid], eos)
            out[qid] = [{"qid": qid, "input_ids": ids, "prompt_len": plen}]
        return out

    def _sample_unpaired(self, cfg: Config) -> bool:
        """Whether pooled-chunk questions WITHOUT fit pairs still get sampled
        from their chunk's adapter (bridge_sft's sample_skipped)."""
        return False

    def _fit(self, policy: str, pairs: list[dict], adapters_dir: Path, name: str,
             params: dict, cfg: Config, *, probe_ctx: dict | None = None,
             ) -> tuple[Path, float]:
        """Fit dispatcher: fixed-steps fit, or the adaptive tau_E round loop
        when improve.lora_sft.fit.adaptive.enabled (the optional post-SFT RL
        phase hooks in after either)."""
        iteration = (probe_ctx or {}).get("iteration", 0)
        wb = wandb_params(cfg, run_name=f"{cfg.run.name}/iter{iteration}/fit_{name}")
        if cfg.improve.lora_sft.fit.adaptive.enabled and probe_ctx is not None:
            return self._fit_adapter_adaptive(
                policy, pairs, adapters_dir, name, params, cfg, probe_ctx=probe_ctx,
            )
        return _fit_adapter(policy, pairs, adapters_dir, name, params, cfg, wandb=wb)

    def _fit_adapter_adaptive(self, policy: str, pairs: list[dict], adapters_dir: Path,
                              name: str, params: dict, cfg: Config, *, probe_ctx: dict,
                              ) -> tuple[Path, float]:
        """tau_E early-stopped fit (methodology 4b): rounds of eval_every
        gradient steps, each warm-started from the previous round's adapter
        (content-addressed chain via init_adapter in the fit key), probed with
        m_rollouts samples per question. Stops when the criterion (frac of
        questions with >=1 correct probe sample, or mean P_hat) clears tau_e,
        or at the max_steps hard cap. Each round costs one lora_fit subprocess
        + one engine boot for the probe pool."""
        ad = cfg.improve.lora_sft.fit.adaptive
        ctx = probe_ctx
        qids = list(ctx["chunk_qids"])
        prev: Path | None = None
        adapter_dir: Path | None = None
        steps_done = 0
        total_secs = 0.0
        value = 0.0
        rounds_meta: list[dict] = []
        n_rounds = -(-ad.max_steps // ad.eval_every)  # ceil
        r = 0
        for r in range(1, n_rounds + 1):
            step_budget = min(ad.eval_every, ad.max_steps - steps_done)
            round_params = {
                **params, "steps": step_budget,
                "seed": stable_seed(cfg.run.seed, "lora_fit", ctx["iteration"], name, r),
            }
            adapter_dir, secs = _fit_adapter(
                policy, pairs, adapters_dir, f"{name}_r{r}", round_params, cfg,
                init_adapter=prev,
                wandb=wandb_params(
                    cfg, run_name=f"{cfg.run.name}/iter{ctx['iteration']}/fit_{name}_r{r}"),
            )
            total_secs += secs
            steps_done += step_budget
            counts = self._probe_correct_counts(
                qids, adapter_dir, ad.m_rollouts, cfg=cfg, policy=policy,
                prompts=ctx["prompts"], anchors_by_qid=ctx["anchors_by_qid"],
                questions_by_qid=ctx["questions_by_qid"], tokenizer=ctx["tokenizer"],
                grader=ctx["grader"],
                pool_dir=ctx["pool_base"] / f"adaptive_{name}_r{r}",
                iteration=ctx["iteration"], tag=f"adaptive_r{r}",
            )
            p_hat = {q: n / ad.m_rollouts for q, n in counts.items()}
            if ad.criterion == "frac_solved":
                value = sum(v > 0 for v in p_hat.values()) / len(p_hat)
            else:  # mean_p_hat
                value = sum(p_hat.values()) / len(p_hat)
            rounds_meta.append({"round": r, "steps": steps_done,
                                "criterion": round(value, 4), "seconds": secs})
            print(f"[improve:{self.name}] adaptive {name} round {r}: "
                  f"{ad.criterion}={value:.3f} (tau_e={ad.tau_e}, steps {steps_done})")
            if value >= ad.tau_e or steps_done >= ad.max_steps:
                break
            prev = adapter_dir
        meta = {"rounds_used": r, "steps_used": steps_done,
                "stopped_early": value >= ad.tau_e, "rounds": rounds_meta}
        stats = ctx["stats"]
        stats.setdefault("adaptive", {})[name] = meta
        stats["adaptive_rounds_total"] = stats.get("adaptive_rounds_total", 0) + r
        return adapter_dir, round(total_secs, 1)

    def _probe_correct_counts(self, qids: list[str], adapter_dir: Path, m: int, *,
                              cfg: Config, policy: str, prompts, anchors_by_qid,
                              questions_by_qid, tokenizer, grader, pool_dir: Path,
                              iteration: int, tag: str) -> dict[str, int]:
        """n_correct out of m samples of `adapter_dir`, per question.

        Shared by the adaptive-fit criterion and the RL prompt filter. Two
        conventions carried over from `_collect`: only stop-finished samples can
        be correct (a truncated trajectory has no answer), but the denominator
        stays `m`, so truncation counts as failure. alpha 1.0 is a no-op in
        make_alpha_variant, so no adapter variant is materialized. Every caller
        must pass its OWN pool_dir and tag — `_sample` reuses request ids across
        launches and only the seed varies by tag."""
        results, _ = self._sample(
            qids, [1.0], m, prompts, anchors_by_qid,
            {q: adapter_dir for q in qids}, cfg, policy, pool_dir, iteration,
            tag=tag,
        )
        counts = {q: 0 for q in qids}
        for (qid, _alpha), res in results:
            q = questions_by_qid[qid]
            anchor_ids = list(anchors_by_qid[qid].anchor_token_ids)
            counts[qid] = sum(
                bool(grader.verify(
                    QuestionRecord(qid=q.qid, question=q.question,
                                   final_answer=q.final_answer),
                    tokenizer.decode(anchor_ids + list(s["token_ids"])),
                ).correct)
                for s in res.samples if s["finish_reason"] == "stop"
            )
        return counts

    def _filter_rl_prompts(self, qids: list[str], adapter_dir: Path, name: str, *,
                           cfg: Config, policy: str, prompts, anchors_by_qid,
                           questions_by_qid, tokenizer, grader, pool_base: Path,
                           iteration: int, stats: dict) -> list[str]:
        """DAPO-style zero-variance filtering (improve.rl.prompt_filter): keep
        only questions whose probed pass rate is strictly inside
        (min_pass_rate, max_pass_rate). A group with uniform rewards has
        advantage 0 and produces NO gradient, so those questions spend rollouts
        for nothing. Returns qids unchanged when the filter is off."""
        pf = cfg.improve.rl.prompt_filter
        if not pf.enabled or not qids:
            return qids
        t0 = time.time()
        counts = self._probe_correct_counts(
            qids, adapter_dir, pf.m_rollouts, cfg=cfg, policy=policy,
            prompts=prompts, anchors_by_qid=anchors_by_qid,
            questions_by_qid=questions_by_qid, tokenizer=tokenizer, grader=grader,
            pool_dir=pool_base / f"rlfilter_{name}", iteration=iteration,
            tag=f"rlfilter_{name}",
        )
        rates = {q: n / pf.m_rollouts for q, n in counts.items()}
        kept = [q for q in qids if pf.min_pass_rate < rates[q] < pf.max_pass_rate]
        n_low = sum(1 for r in rates.values() if r <= pf.min_pass_rate)
        n_high = sum(1 for r in rates.values() if r >= pf.max_pass_rate)
        stats.update(
            n_rl_prompts_probed=len(qids),
            n_rl_prompts_kept=len(kept),
            n_rl_prompts_all_wrong=n_low,
            n_rl_prompts_all_right=n_high,
            rl_prompt_filter_yield=round(len(kept) / len(qids), 4),
            rl_prompt_filter_seconds=round(time.time() - t0, 1),
            rl_prompt_filter_mean_pass_rate=round(sum(rates.values()) / len(rates), 4),
        )
        print(f"[improve:{self.name}] rl prompt filter {name}: kept {len(kept)}/{len(qids)} "
              f"(pass rate in ({pf.min_pass_rate}, {pf.max_pass_rate}) over "
              f"{pf.m_rollouts} probes; {n_low} too low, {n_high} too high)")
        return kept

    def _maybe_rl(self, adapter_dir: Path, qids: list[str], name: str, *,
                  cfg: Config, policy: str, prompts, anchors_by_qid,
                  questions_by_qid, tokenizer, adapters_dir: Path,
                  iteration: int, stats: dict,
                  pool_base: Path | None = None, grader=None,
                  ) -> tuple[Path, dict | None]:
        """Optional post-SFT RL phase (improve.rl): refine the freshly fitted
        chunk adapter with GRPO/REINFORCE on the true target states (x [+ a],
        NO y*), returning phi_E for the candidate-sampling tail. Main-pass
        adapters only — refits are targeted rescues and skip RL."""
        rl = cfg.improve.rl
        if not rl.enabled or not qids:
            return adapter_dir, None
        from . import lora_rl

        # Zero-variance prompt filtering, before any rows are built. Needs a
        # pool + a verifier, so it is skipped when the caller did not supply
        # them (the fit path does; unit tests calling _maybe_rl directly do not).
        if pool_base is not None and grader is not None:
            qids = self._filter_rl_prompts(
                qids, adapter_dir, name, cfg=cfg, policy=policy, prompts=prompts,
                anchors_by_qid=anchors_by_qid, questions_by_qid=questions_by_qid,
                tokenizer=tokenizer, grader=grader, pool_base=pool_base,
                iteration=iteration, stats=stats,
            )
            if not qids:
                print(f"[improve:{self.name}] rl SKIPPED for {name}: the prompt "
                      "filter left no question with a mixed-reward group")
                stats.setdefault("rl", {})[name] = {"skipped": "prompt_filter_empty"}
                return adapter_dir, None

        rows, n_mismatch = lora_rl.build_rl_rows(
            qids, questions_by_qid=questions_by_qid, prompts=prompts,
            anchors_by_qid=anchors_by_qid, tokenizer=tokenizer, cfg=cfg,
        )
        if n_mismatch:
            msg = (f"[improve:{self.name}] rl seam: {n_mismatch}/{len(rows)} anchored "
                   f"prompts retokenize differently from their id-spliced form")
            if rl.seam_strict:
                raise RuntimeError(msg + " (improve.rl.seam_strict=true)")
            print(msg + " — continuing (count+warn)")
        params = {
            "algo": rl.algo,
            "epochs": rl.epochs,
            "steps": rl.steps,
            "grad_accum": rl.grad_accum,
            "gradient_checkpointing": rl.gradient_checkpointing,
            "lr": rl.lr,
            "epsilon": rl.epsilon,
            "kl_beta": rl.kl_beta,
            "group_size": rl.group_size,
            "temperature": rl.temperature if rl.temperature is not None
                           else cfg.improve.temperature,
            "top_p": cfg.improve.top_p,
            "max_completion_length": rl.max_completion_length or cfg.improve.max_tokens,
            "plateau_enabled": rl.plateau.enabled,
            "plateau_patience": rl.plateau.patience,
            "plateau_min_delta": rl.plateau.min_delta,
            "plateau_window": rl.plateau.window,
            "vllm_importance_sampling_correction": rl.vllm_importance_sampling_correction,
            "vllm_importance_sampling_mode": rl.vllm_importance_sampling_mode,
            "vllm_importance_sampling_cap": rl.vllm_importance_sampling_cap,
            "vllm_gpu_memory_utilization": rl.vllm_gpu_memory_utilization,
            # Without this the colocate engine boots at the MODEL's native
            # context (262k for Qwen3-4B) and demands a KV cache far larger
            # than its utilization share; every other pool in the run is
            # capped the same way.
            "vllm_max_model_length": cfg.engine.max_model_len,
            "verifier": cfg.partition.verifier,
            "seed": rl.seed if rl.seed is not None
                    else stable_seed(cfg.run.seed, "lora_rl", iteration, name),
            # Per-step RL curves get their own wandb run grouped under run.name
            # (same convention as train.py); resolve_wandb_mode downgrades a
            # keyless "online" to offline so the subprocess cannot hang on auth.
            "wandb": wandb_params(
                cfg, run_name=f"{cfg.run.name}/iter{iteration}/rl_{name}"),
        }
        out_dir, meta = _launch_lora_rl(policy, adapter_dir, rows, adapters_dir,
                                        name, params, cfg)
        rl_info = {
            **{k: meta.get(k) for k in ("algo", "steps_planned", "steps_done",
                                        "reward_first", "reward_last",
                                        "plateau_stopped", "seconds")},
            "seam_mismatches": n_mismatch,
            "n_prompts": len(rows),
        }
        stats.setdefault("rl", {})[name] = rl_info
        return out_dir, rl_info

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _chunks(qids: list[str], ls) -> list[list[str]]:
        if ls.adapter_scope == "per_problem":
            return [[q] for q in qids]
        if ls.chunk_size and ls.chunk_size > 0:
            return [qids[i:i + ls.chunk_size] for i in range(0, len(qids), ls.chunk_size)]
        return [qids]

    def _sample(self, qids, alphas, n_per_req, prompts, anchors_by_qid, adapter_of,
                cfg: Config, policy: str, pool_dir: Path, iteration: int, tag: str):
        """One generate launch over qids x alphas; returns ([(qid, alpha, GenResult)],
        n_budget_skipped)."""
        requests, keys = [], []
        n_budget_skipped = 0
        for qid in qids:
            anchor = anchors_by_qid[qid]
            prompt_ids = continuation_prompt_ids(prompts[qid], anchor.anchor_token_ids)
            budget = cfg.filter.max_total_tokens - len(prompt_ids)
            if budget <= 0:
                n_budget_skipped += 1
                continue
            for alpha in alphas:
                variant = make_alpha_variant(adapter_of[qid], alpha)
                requests.append(GenRequest(
                    rid=f"{qid}:a{fmt_alpha(alpha)}",
                    prompt_token_ids=prompt_ids,
                    n=n_per_req,
                    seed=stable_seed(cfg.run.seed, "improve", iteration, qid,
                                     fmt_alpha(alpha), tag),
                    lora_path=str(variant),
                    max_tokens=min(cfg.improve.max_tokens, budget),
                ))
                keys.append((qid, alpha))
        if not requests:
            return [], n_budget_skipped
        results = run_pool(
            requests, mode="generate", model_path=policy,
            sampling={
                "temperature": cfg.improve.temperature,
                "top_p": cfg.improve.top_p,
                "max_tokens": cfg.improve.max_tokens,
            },
            engine_cfg=cfg.engine, work_dir=pool_dir, dtype=cfg.model.dtype,
        )
        return list(zip(keys, results)), n_budget_skipped

    def _collect(self, results, questions_by_qid, anchors_by_qid, prompts, tokenizer, grader,
                 cfg: Config, n_per_req: int, pb, chunk_of, adapter_of, fit_meta_of,
                 iteration: int, *, refit: bool):
        """Verify every sample, pick alpha* per question (or per chunk), emit
        ImprovedCandidates for the alpha* samples. Returns (candidates,
        correct_total per qid across ALL alphas, alpha-curve rows
        [{qid, p_hat_by_alpha, alpha_star, refit}])."""
        verdicts: dict[tuple[str, float], list[tuple[dict, bool]]] = {}
        for (qid, alpha), res in results:
            q = questions_by_qid[qid]
            anchor = anchors_by_qid[qid]
            graded = []
            for s in res.samples:
                if s["finish_reason"] != "stop":
                    continue  # truncated continuations are untrainable
                full_text = tokenizer.decode(
                    list(anchor.anchor_token_ids) + list(s["token_ids"])
                )
                verdict = grader.verify(
                    QuestionRecord(qid=q.qid, question=q.question,
                                   final_answer=q.final_answer),
                    full_text,
                )
                graded.append((s, verdict.correct))
            verdicts[(qid, alpha)] = graded

        p_hat: dict[str, dict[float, float]] = {}
        for (qid, alpha), graded in verdicts.items():
            n_ok = sum(1 for _, ok in graded if ok)
            p_hat.setdefault(qid, {})[alpha] = n_ok / n_per_req
        correct_total = {
            qid: sum(int(round(p * n_per_req)) for p in by_alpha.values())
            for qid, by_alpha in p_hat.items()
        }

        alpha_star: dict[str, float] = {}
        if pb.enabled and pb.granularity == "per_batch":
            by_chunk: dict[str, list[str]] = {}
            for qid in p_hat:
                by_chunk.setdefault(chunk_of[qid], []).append(qid)
            for chunk_qids in by_chunk.values():
                mean_p = {
                    alpha: sum(p_hat[q][alpha] for q in chunk_qids) / len(chunk_qids)
                    for alpha in p_hat[chunk_qids[0]]
                }
                star = select_alpha_star(mean_p, pb.tau_p)
                for q in chunk_qids:
                    alpha_star[q] = star
        else:
            for qid, by_alpha in p_hat.items():
                alpha_star[qid] = select_alpha_star(by_alpha, pb.tau_p) if pb.enabled else 1.0

        curve_rows = [
            {
                "qid": qid,
                "p_hat_by_alpha": {fmt_alpha(a): round(p, 4)
                                   for a, p in sorted(p_hat[qid].items())},
                "alpha_star": alpha_star[qid],
                "refit": refit,
            }
            for qid in sorted(alpha_star)
        ]

        out: list[ImprovedCandidate] = []
        for qid in sorted(alpha_star):
            star = alpha_star[qid]
            anchor = anchors_by_qid[qid]
            graded = verdicts.get((qid, star), [])
            meta_common = {
                "adapter_path": str(adapter_of[qid]),
                "lora_path": str(make_alpha_variant(adapter_of[qid], star)),
                "alpha": star,
                "alpha_star": star,
                "p_hat_by_alpha": {fmt_alpha(a): round(p, 4)
                                   for a, p in sorted(p_hat[qid].items())},
                "adapter_scope": cfg.improve.lora_sft.adapter_scope,
                "chunk": chunk_of[qid],
                "refit": refit,
                "temperature": cfg.improve.temperature,
                **type(self).PROVENANCE,
                "fit": fit_meta_of[qid],
            }
            for ai, (s, ok) in enumerate(graded):
                out.append(ImprovedCandidate(
                    qid=qid,
                    base_sample_idx=anchor.base_sample_idx,
                    attempt_idx=ai,
                    prompt_token_ids=list(prompts[qid]),
                    anchor_token_ids=list(anchor.anchor_token_ids),
                    continuation_token_ids=list(s["token_ids"]),
                    continuation_text=s["text"],
                    correct=ok,
                    operator=self.name,
                    op_meta=dict(meta_common),
                    external_context=None,
                    iter=iteration,
                ))
        return out, correct_total, curve_rows


def _launch_lora_rl(policy: str, adapter_dir: Path, rows: list[dict],
                    adapters_dir: Path, name: str, params: dict,
                    cfg: Config) -> tuple[Path, dict]:
    """Run `python -m expert_iter.lora_rl` as a GPU subprocess (exits before
    any candidate pool boots). Content-addressed output keyed over params +
    model fingerprint + input adapter path + dataset rows."""
    from . import lora_rl

    gpus = visible_gpus(cfg.engine.gpus)
    if not gpus:
        raise RuntimeError("improve.rl: no GPUs available")
    n_proc = cfg.improve.rl.num_processes or len(gpus)
    n_proc = max(1, min(n_proc, len(gpus)))
    # Reject an illegal (micro_batch, grad_accum, world_size) combination HERE,
    # before any GPU work — the message names the legal values. The resolved
    # shape is part of params (and therefore of the cache key): the topology
    # changes the update, so a rerun under a different one must not reuse it.
    shape = resolve_batch_shape(
        group_size=cfg.improve.rl.group_size,
        micro_batch_size=cfg.improve.rl.micro_batch_size,
        grad_accum=cfg.improve.rl.grad_accum, world_size=n_proc,
    )
    params = {**params, "micro_batch": shape["micro_batch"], "world_size": n_proc}

    key = lora_rl.rl_key(params, _model_fingerprint(policy), adapter_dir, rows)
    out_dir = adapters_dir / f"{name}_rl" / key
    if is_done(out_dir / "adapter_config.json", config_hash=key):
        return out_dir, read_json(out_dir / "rl_meta.json")
    dataset_path = adapters_dir / f"rl_rows_{name}.jsonl"
    write_jsonl(dataset_path, rows)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    log_path = adapters_dir / f"{name}.rl.log"
    module_args = [
        "--model", policy,
        "--adapter", str(adapter_dir),
        "--dataset", str(dataset_path),
        "--out", str(out_dir),
        "--params-json", json.dumps(params),
        "--cache-key", key,
    ]
    if n_proc > 1:
        # Data-parallel RL: one rank (and one colocate vLLM) per GPU. trl
        # supports multi-GPU ONLY through accelerate/DDP — never the Trainer's
        # implicit DataParallel. Both topologies are GPU-verified
        # (docs/api_notes.md); NCCL_P2P_DISABLE comes from the launcher on
        # PCIe-only nodes.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus[:n_proc])
        accelerate = shutil.which("accelerate") or str(Path(sys.executable).parent / "accelerate")
        cmd = [accelerate, "launch", "--num_processes", str(n_proc),
               "--num_machines", "1", "-m", "expert_iter.lora_rl", *module_args]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        cmd = [sys.executable, "-m", "expert_iter.lora_rl", *module_args]
    t0 = time.time()
    # One optimizer step consumes n_proc questions x group_size rollouts, so an
    # epoch is floor(len(rows) / n_proc) steps (transformers derives the exact
    # count from the dataloader; rl_meta records it as steps_planned).
    qps = shape["questions_per_step"]
    budget = (f"{params['epochs']} epoch(s) ~ {len(rows) // qps} steps"
              if params.get("epochs") is not None else f"{params['steps']} steps")
    print(f"[improve] rl ({params['algo']}, {n_proc} gpu) refining {name} "
          f"({len(rows)} questions, {budget}, {qps} question(s) x "
          f"{cfg.improve.rl.group_size} rollouts per step, micro_batch "
          f"{shape['micro_batch']}) -> {out_dir}")
    with log_path.open("w") as log_f:
        proc = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "".join(log_path.read_text(errors="replace").splitlines(keepends=True)[-40:])
        raise RuntimeError(f"lora_rl for {name} exited {proc.returncode}; log tail:\n{tail}")
    if not is_done(out_dir / "adapter_config.json", config_hash=key):
        raise RuntimeError(f"lora_rl for {name} finished but left no done marker")
    meta = read_json(out_dir / "rl_meta.json")
    meta.setdefault("seconds", round(time.time() - t0, 1))
    return out_dir, meta


def _fit_adapter(policy: str, pairs: list[dict], adapters_dir: Path, name: str,
                 params: dict, cfg: Config, *,
                 init_adapter: Path | None = None,
                 wandb: dict | None = None) -> tuple[Path, float]:
    """Write pairs, run lora_fit in a subprocess on one engine GPU, return the
    content-addressed adapter dir. Skips fitting when the .done marker matches.
    init_adapter warm-starts from a previous (content-addressed) adapter dir —
    its path is part of the fit key, keeping the round chain honest."""
    pairs_path = adapters_dir / f"pairs_{name}.jsonl"
    write_jsonl(pairs_path, pairs)
    gpus = visible_gpus(cfg.engine.gpus)
    if not gpus:
        raise RuntimeError("lora_sft: no GPUs available for the LoRA fit")
    # Data-parallel ranks, capped at the pair count (a rank with no pairs would
    # deadlock DDP's collectives).
    n_proc = cfg.improve.lora_sft.fit.num_processes or len(gpus)
    n_proc = max(1, min(n_proc, len(gpus), len(pairs)))
    fit_key = stable_hash(
        "lora_fit_v1",
        json.dumps(params, sort_keys=True),
        _model_fingerprint(policy),
        tuple((p["qid"], tuple(p["input_ids"]), p["prompt_len"]) for p in pairs),
        str(init_adapter) if init_adapter else "",
        n_proc,   # DDP is mathematically but not bitwise identical to 1 process
    )
    adapter_dir = adapters_dir / name / fit_key
    if is_done(adapter_dir / "adapter_config.json", config_hash=fit_key):
        return adapter_dir, 0.0
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    log_path = adapters_dir / f"{name}.fit.log"
    module_args = [
        "--model", policy,
        "--pairs", str(pairs_path),
        "--out", str(adapter_dir),
        "--params-json", json.dumps(params),
        "--cache-key", fit_key,
    ]
    # Deliberately NOT part of fit_key above: where the loss curve is reported
    # must never invalidate a cached adapter.
    if wandb:
        module_args += ["--wandb-json", json.dumps(wandb)]
    if init_adapter is not None:
        module_args += ["--init-adapter", str(init_adapter)]
    if n_proc > 1:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus[:n_proc])
        accelerate = shutil.which("accelerate") or str(Path(sys.executable).parent / "accelerate")
        cmd = [accelerate, "launch", "--num_processes", str(n_proc),
               "--num_machines", "1", "-m", "expert_iter.lora_fit", *module_args]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        cmd = [sys.executable, "-m", "expert_iter.lora_fit", *module_args]
    t0 = time.time()
    print(f"[improve:lora_sft] fitting {name} ({len(pairs)} pairs, {n_proc} gpu) "
          f"-> {adapter_dir}")
    with log_path.open("w") as log_f:
        proc = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "".join(log_path.read_text(errors="replace").splitlines(keepends=True)[-40:])
        raise RuntimeError(f"lora_fit for {name} exited {proc.returncode}; log tail:\n{tail}")
    if not is_done(adapter_dir / "adapter_config.json", config_hash=fit_key):
        raise RuntimeError(f"lora_fit for {name} finished but left no done marker")
    return adapter_dir, round(time.time() - t0, 1)
