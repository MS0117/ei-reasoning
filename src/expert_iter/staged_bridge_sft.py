"""Improvement operator (II): staged bridge SFT — multi-stage bridge LoRA.

Extends bridge_sft from one fit->sample pass to a staged loop:

  stage 1   bridges z+ from the privileged y*-showing prompt (inherited
            _build_targets, unchanged) -> LoRA fit (steps = fit.steps)
  (b)       rollout_n samples per cliff off the adapter; verifier-correct
            samples enter the candidate POOL (the cliff is "converted")
  stage 2   fit again for num_stages passes (steps = staged.stage2_steps),
            warm-started from the previous adapter (chain_adapter):
              unsolved: stage-1 bridge pairs reused, or bridges REGENERATED
                        through the current adapter (unsolved_targets)
              solved:   only when train_scope=full_pool — one self-generated
                        correct rollout per question, picked by solved_targets
                        (self-wash min C(y) / stage-1 bridge / random /
                        longest / shortest)
            then roll out again (intermediate: rollout_n over unsolved;
            final: final_rollout_n over final_rollout_scope)
  emit      the POOL (all rounds, or final round only) — correct is prefilled,
            filters' gates + filter.selection do per-question trimming exactly
            as for bridge_sft. Bridges never become training data.

v1 scope (rejected at config load): adaptive tau_E (the staged rollouts ARE
the probe), project-back, refit_budget, per_problem/chunked adapters, RL.

Module rule: like lora_sft/bridge_sft, NEVER import expert_iter.improve;
shared infrastructure is reached through the lora_sft module object so tests
can fake one `run_pool` / `_fit_adapter` seam.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

from .bridge_sft import BridgeSftOperator
from .config import Config
from .records import ImprovedCandidate
from .registry import OPERATORS, VERIFIERS, build, register
from .templates import bridge_pair_ids
from .utils import stable_seed, write_json, write_jsonl
from . import lora_sft as _ls

# stats keys copied from a regen _build_targets call into stats[f"stage{k}_bridge"]
_BRIDGE_STAT_KEYS = (
    "n_bridge_generated", "n_bridge_correct", "n_bridge_leak_rule",
    "n_bridge_leak_judge", "n_bridge_retried", "n_bridge_budget_skipped",
    "n_questions_bridged", "n_bridge_skipped", "bridge_pairs_kept",
    "bridge_yield",
)


@register(OPERATORS, "staged_bridge_sft")
class StagedBridgeSftOperator(BridgeSftOperator):
    PROVENANCE = {
        "privileged": "gold_solution",
        "channel": "lora_weights",
        "via": "bridge_trajectories",
        "staged": True,
    }

    def propose(self, questions, anchors, prompts, cfg: Config, *,
                model_paths, work_dir, iteration, gold_solutions=None):
        gold_solutions = gold_solutions or {}
        ls = cfg.improve.lora_sft
        st = ls.staged
        pb = ls.project_back  # disabled (validated) -> alpha* stays 1.0 in _collect
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

        fit_params = {
            "r": ls.fit.r, "lora_alpha": ls.fit.lora_alpha, "lr": ls.fit.lr,
            "steps": ls.fit.steps, "dropout": ls.fit.dropout,
            "target_modules": list(ls.fit.target_modules),
            "micro_batch_size": ls.fit.micro_batch_size,
            "max_grad_norm": ls.fit.max_grad_norm, "bf16": ls.fit.bf16,
            "sync_every": ls.fit.sync_every,
        }
        max_pair = ls.fit.max_pair_tokens
        eos = tokenizer.eos_token_id

        pool: list[ImprovedCandidate] = []
        curve_rows: list[dict] = []
        next_attempt: dict[str, int] = defaultdict(int)
        solved: set[str] = set()
        fit_seconds: list[float] = []
        stats["n_budget_skipped"] = 0
        # qid -> the adapter it currently samples from, and its shard bookkeeping.
        # A question the stage-2 fit does not cover keeps its previous adapter,
        # so no fallback branch is needed anywhere below.
        adapter_of_qid: dict[str, Path] = {}
        chunk_key_of: dict[str, tuple[int, ...]] = {}
        chunk_label_of: dict[str, str] = {}

        def _roll(name: str, qids: list[str], n: int, fit_meta: dict):
            """Sample n per qid from that question's CURRENT adapter (pooled at
            stage 1, its own shard once stage-2 chunking is on), grade, re-index
            attempt_idx to stay unique across rounds, stamp op_meta.stage, extend
            the pool. Returns the per-qid correct counts."""
            qids = sorted(qids)
            if not qids:
                return {}
            adapter_of = {q: adapter_of_qid[q] for q in qids}
            results, skipped = self._sample(
                qids, [1.0], n, prompts, anchors_by_qid,
                adapter_of, cfg, policy,
                pool_base / f"{name}_roll", iteration, tag=name,
            )
            stats["n_budget_skipped"] += skipped
            cands, correct_total, curves = self._collect(
                results, questions_by_qid, anchors_by_qid, prompts, tokenizer,
                grader, cfg, n, pb,
                {q: chunk_label_of.get(q, name) for q in qids},
                adapter_of,
                {q: {**fit_meta, "stage": name} for q in qids},
                iteration, refit=False,
            )
            for c in cands:
                c.attempt_idx = next_attempt[c.qid]
                next_attempt[c.qid] += 1
                c.op_meta["stage"] = name
            pool.extend(cands)
            for row in curves:
                curve_rows.append({**row, "stage": name})
            return correct_total

        # ---- stage 1: bridges -> fit (steps = fit.steps) --------------------
        targets1 = self._build_targets(
            with_gold, tokenizer=tokenizer, prompts=prompts,
            anchors_by_qid=anchors_by_qid, questions_by_qid=questions_by_qid,
            grader=grader, cfg=cfg, policy=policy, pool_base=pool_base,
            improve_dir=improve_dir, iteration=iteration,
            gold_solutions=gold_solutions, stats=stats,
        )
        pairs1, n_dropped_long = _flatten_pairs(targets1, with_gold, max_pair)
        stats.update(n_pairs=len(pairs1), n_dropped_long=n_dropped_long)
        if not pairs1:
            stats["stage1_skipped"] = "no_pairs"
            write_json(improve_dir / "stats.json", stats)
            print(f"[improve:{self.name}] {stats}")
            return []
        # ---- stage-1 fit: pooled, or one adapter per stage1_chunk_size questions
        pairs1_by_qid: dict[str, list[dict]] = defaultdict(list)
        for pair in pairs1:
            pairs1_by_qid[pair["qid"]].append(pair)
        chunks1 = _assign_chunks(sorted(pairs1_by_qid), st.stage1_chunk_size, chunk_key_of)
        self._warn_max_loras(len(chunks1), cfg, "stage-1")
        chunk1_rows = []
        shard_adapters: list[Path] = []
        for label, chunk_qids in chunks1:
            name = "stage1" if len(chunks1) == 1 else f"stage1_{label}"
            chunk_pairs = [p for q in chunk_qids for p in pairs1_by_qid[q]]
            seed = stable_seed(cfg.run.seed, "lora_fit", iteration, name)
            adapter, secs = _ls._fit_adapter(
                policy, chunk_pairs, adapters_dir, name,
                {**fit_params, "seed": seed}, cfg,
                wandb=_ls.wandb_params(
                    cfg, run_name=f"{cfg.run.name}/iter{iteration}/fit_{name}"),
            )
            fit_seconds.append(secs)
            shard_adapters.append(adapter)
            for q in chunk_qids:
                adapter_of_qid[q] = adapter
                chunk_label_of[q] = name
            chunk1_rows.append({"name": name, "n_questions": len(chunk_qids),
                                "n_pairs": len(chunk_pairs)})
        if st.stage1_chunk_size > 0:
            stats["stage1_chunks"] = chunk1_rows
        # questions without fit pairs (bridge-skipped; sampled when
        # bridge.sample_skipped) still need an adapter: round-robin over shards
        for i, q in enumerate(q for q in with_gold if q not in adapter_of_qid):
            s = i % len(shard_adapters)
            adapter_of_qid[q] = shard_adapters[s]
            chunk_key_of[q] = chunk_key_of[chunks1[s][1][0]]   # same shard path
            chunk_label_of[q] = chunk1_rows[s]["name"]

        # ---- rollout (b) off the stage-1 adapter ----------------------------
        sample_unpaired = self._sample_unpaired(cfg)
        roll_qids = [q for q in with_gold if q in targets1 or sample_unpaired]
        meta1 = {**fit_params, "n_pairs": len(pairs1)}
        correct = _roll("stage1", roll_qids, st.rollout_n, meta1)
        solved |= {q for q, n_ok in correct.items() if n_ok > 0}
        stats["n_resolved_stage1"] = stats["n_resolved_pool"] = len(solved)
        stats["n_resolved_by_stage"] = {"stage1": len(solved)}

        # ---- stage-2 passes -------------------------------------------------
        last_tag = "stage1"
        stages_run = 0
        for k in range(1, st.num_stages + 1):
            unsolved_k = [q for q in with_gold if q not in solved]
            # c-1: unsolved questions -> bridge pairs
            if st.unsolved_targets in ("regen_bridge", "add_bridge") and unsolved_k:
                regen_stats: dict = {}
                cap = st.stage_max_keep or ls.bridge.max_keep
                self._bridge_lora_path = str(adapter)
                # regenerate through each question's CURRENT adapter (pooled at
                # stage 1; its own shard once stage-2 chunking has run)
                self._bridge_lora_of = {q: str(adapter_of_qid[q]) for q in unsolved_k}
                self._bridge_seed_salt = f"stage{k}"
                self._bridge_n = st.stage_bridge_n
                self._bridge_max_keep = st.stage_max_keep
                try:
                    regen = self._build_targets(
                        unsolved_k, tokenizer=tokenizer, prompts=prompts,
                        anchors_by_qid=anchors_by_qid,
                        questions_by_qid=questions_by_qid, grader=grader,
                        cfg=cfg, policy=policy,
                        pool_base=pool_base / f"stage{k}_bridge",
                        improve_dir=improve_dir / f"stage{k}",
                        iteration=iteration, gold_solutions=gold_solutions,
                        stats=regen_stats,
                    )
                finally:
                    self._bridge_lora_path = None
                    self._bridge_lora_of = None
                    self._bridge_seed_salt = None
                    self._bridge_n = None
                    self._bridge_max_keep = None
                stats[f"stage{k}_bridge"] = {
                    key: regen_stats[key] for key in _BRIDGE_STAT_KEYS
                    if key in regen_stats
                }
                if st.unsolved_targets == "add_bridge":
                    n_old = n_new = 0
                    targets_k = {}
                    for q in unsolved_k:
                        merged = _merge_pairs(
                            targets1.get(q, []), regen.get(q, []), cap,
                            ls.bridge.keep_selection,
                            stable_seed(cfg.run.seed, "bridge_merge", iteration, q, k),
                        )
                        targets_k[q] = merged
                        old_ids = {tuple(p["input_ids"]) for p in targets1.get(q, [])}
                        n_old += sum(1 for p in merged if tuple(p["input_ids"]) in old_ids)
                        n_new += sum(1 for p in merged if tuple(p["input_ids"]) not in old_ids)
                    stats[f"stage{k}_pairs_old"] = n_old
                    stats[f"stage{k}_pairs_new"] = n_new
                else:  # regen_bridge: successful regens replace, failures fall
                    # back to the stage-1 bridges instead of losing the question
                    targets_k = {q: (regen.get(q) or targets1.get(q, []))
                                 for q in unsolved_k}
                    stats[f"stage{k}_regen_fallback"] = sum(
                        1 for q in unsolved_k
                        if not regen.get(q) and targets1.get(q))
                pairs_k, dropped = _flatten_pairs(targets_k, unsolved_k, max_pair)
            else:
                pairs_k, dropped = _flatten_pairs(targets1, unsolved_k, max_pair)
            stats["n_dropped_long"] += dropped
            # c-2: solved questions -> self-generated (or bridge) pairs
            if st.train_scope == "full_pool" and solved:
                pairs_k += self._solved_pairs(
                    k, sorted(solved), pool, targets1, cfg=cfg, policy=policy,
                    prompts=prompts, anchors_by_qid=anchors_by_qid, eos=eos,
                    improve_dir=improve_dir, iteration=iteration, stats=stats,
                )
            if not pairs_k:
                stats[f"stage{k + 1}_skipped"] = "no_pairs"
                break
            stages_run = k
            # ---- objective: NLL on the bridges, or DPO against the adapter's
            # own failures from the preceding rollout (tag = last_tag) ----------
            fit_extra: dict = {}
            if st.stage2_objective == "dpo":
                pairs_k = _dpo_pairs(
                    pairs_k, pool, neg_stage=last_tag, prompts=prompts,
                    anchors_by_qid=anchors_by_qid, eos=eos, max_pair_tokens=max_pair,
                    selection=st.dpo.negative_selection,
                    max_per_question=st.dpo.max_pairs_per_question,
                    seed_of=lambda q: stable_seed(cfg.run.seed, "dpo_neg", iteration, q, k),
                    stats=stats, prefix=f"stage{k + 1}",
                )
                fit_extra = {"objective": "dpo", "beta": st.dpo.beta,
                             "sft_weight": st.dpo.sft_weight,
                             "reference": st.dpo.reference,
                             "lr": st.dpo.lr if st.dpo.lr is not None else ls.fit.lr}
                if not pairs_k:
                    stats[f"stage{k + 1}_skipped"] = "no_dpo_pairs"
                    break
            # ---- fit: one adapter for the whole set, or one per shard --------
            pairs_by_qid: dict[str, list[dict]] = defaultdict(list)
            for pair in pairs_k:
                pairs_by_qid[pair["qid"]].append(pair)
            chunks = _assign_chunks(sorted(pairs_by_qid), st.stage2_chunk_size,
                                    chunk_key_of)
            self._warn_max_loras(len(chunks), cfg, f"stage-{k + 1}")
            chunk_rows = []
            for label, chunk_qids in chunks:
                # a single chunk keeps the un-sharded name, so its fit cache key
                # and logs stay identical to the pooled path
                name = f"stage{k + 1}" if len(chunks) == 1 else f"stage{k + 1}_{label}"
                chunk_pairs = [p for q in chunk_qids for p in pairs_by_qid[q]]
                seed = stable_seed(cfg.run.seed, "lora_fit", iteration, name)
                init = adapter_of_qid[chunk_qids[0]] if st.chain_adapter else None
                chunk_adapter, secs = _ls._fit_adapter(
                    policy, chunk_pairs, adapters_dir, name,
                    {**fit_params, "steps": st.stage2_steps, "seed": seed, **fit_extra}, cfg,
                    init_adapter=init,
                    wandb=_ls.wandb_params(
                        cfg, run_name=f"{cfg.run.name}/iter{iteration}/fit_{name}"),
                )
                fit_seconds.append(secs)
                for q in chunk_qids:
                    adapter_of_qid[q] = chunk_adapter
                    chunk_label_of[q] = name
                chunk_rows.append({"name": name, "n_questions": len(chunk_qids),
                                   "n_pairs": len(chunk_pairs)})
            if st.stage2_chunk_size > 0:
                stats[f"stage{k + 1}_chunks"] = chunk_rows
            last = k == st.num_stages
            n = st.final_rollout_n if last else st.rollout_n
            qids = (roll_qids if (last and st.final_rollout_scope == "all")
                    else [q for q in roll_qids if q not in solved])
            last_tag = f"stage{k + 1}"
            meta_k = {**fit_params, "steps": st.stage2_steps, **fit_extra,
                      "n_pairs": len(pairs_k)}
            correct = _roll(last_tag, qids, n, meta_k)
            newly = {q for q, n_ok in correct.items() if n_ok > 0} - solved
            solved |= newly
            stats["n_resolved_by_stage"][last_tag] = len(newly)

        # ---- emit -----------------------------------------------------------
        out = (pool if st.emit == "all"
               else [c for c in pool if c.op_meta.get("stage") == last_tag])
        stats.update(
            n_fits=len(fit_seconds),
            fit_seconds=fit_seconds,
            n_stages_run=stages_run,
            n_resolved_final=len(solved),
            pool_size=len(pool),
            n_emitted=len(out),
            lora_yield=round(len(solved) / len(eligible), 4) if eligible else 0.0,
            alpha_star_mean=1.0,
        )
        write_jsonl(improve_dir / "alpha_curves.jsonl",
                    sorted(curve_rows, key=lambda r: (r["qid"], r["stage"])))
        write_json(improve_dir / "stats.json", stats)
        print(f"[improve:{self.name}] {stats}")
        return out

    def _warn_max_loras(self, n_adapters: int, cfg: Config, where: str) -> None:
        if n_adapters > cfg.engine.max_loras:
            print(f"[improve:{self.name}] {n_adapters} {where} adapters exceed "
                  f"engine.max_loras={cfg.engine.max_loras}: vLLM will swap adapters "
                  "during the rollout (slower, still correct)")

    def _solved_pairs(self, k: int, solved_qids: list[str],
                      pool: list[ImprovedCandidate], targets1: dict,
                      *, cfg: Config, policy: str, prompts, anchors_by_qid,
                      eos, improve_dir: Path, iteration: int, stats: dict,
                      ) -> list[dict]:
        """One fit pair per solved question, from its correct pool rollouts
        ranked by staged.solved_targets (or the stage-1 bridges for 'bridge').
        Over-budget picks fall through to the next-ranked candidate."""
        st = cfg.improve.lora_sft.staged
        max_pair = cfg.improve.lora_sft.fit.max_pair_tokens
        if st.solved_targets == "bridge":
            pairs, dropped = _flatten_pairs(targets1, solved_qids, max_pair)
            stats["n_dropped_long"] += dropped
            return pairs
        # candidates: this operator's own pool entries, correct only
        solved_set = set(solved_qids)
        by_qid: dict[str, list[ImprovedCandidate]] = defaultdict(list)
        for c in pool:
            if c.correct and c.qid in solved_set:
                by_qid[c.qid].append(c)
        scores: dict[str, dict] = {}
        if st.solved_targets == "self_wash_min_c":
            from .filters import _cand_key, _score_candidates
            flat = [c for qid in solved_qids for c in by_qid.get(qid, [])]
            if flat:
                scores, _report = _score_candidates(
                    flat, cfg, policy, improve_dir,
                    pool_dir=improve_dir / f"stage{k}" / "pool_wash",
                )
        pairs: list[dict] = []
        n_dropped = 0
        for qid in solved_qids:
            cands = sorted(by_qid.get(qid, []), key=lambda c: c.attempt_idx)
            if not cands:
                continue
            if st.solved_targets == "self_wash_min_c":
                ranked = sorted(cands, key=lambda c: (
                    scores.get(_cand_key(c), {}).get("_c_raw", math.inf),
                    len(c.anchor_token_ids) + len(c.continuation_token_ids),
                    c.attempt_idx,
                ))
            elif st.solved_targets == "shortest":
                ranked = sorted(cands, key=lambda c: (
                    len(c.continuation_token_ids), c.attempt_idx))
            elif st.solved_targets == "longest":
                ranked = sorted(cands, key=lambda c: (
                    -len(c.continuation_token_ids), c.attempt_idx))
            else:  # random
                rng = random.Random(
                    stable_seed(cfg.run.seed, "staged_wash", iteration, qid, k))
                ranked = list(cands)
                rng.shuffle(ranked)
            picked = False
            anchor_ids = list(anchors_by_qid[qid].anchor_token_ids)
            for c in ranked:
                ids, plen = bridge_pair_ids(
                    prompts[qid], anchor_ids, c.continuation_token_ids, eos)
                if len(ids) > max_pair:
                    continue
                pairs.append({"qid": qid, "input_ids": ids, "prompt_len": plen})
                picked = True
                break
            if not picked:
                n_dropped += 1
        stats["n_wash_dropped_long"] = stats.get("n_wash_dropped_long", 0) + n_dropped
        stats.setdefault("n_wash_pairs", {})[f"stage{k}"] = len(pairs)
        return pairs


def _dpo_pairs(sft_pairs: list[dict], pool: list[ImprovedCandidate], *,
               neg_stage: str, prompts, anchors_by_qid, eos: int,
               max_pair_tokens: int, selection: str, max_per_question: int | None,
               seed_of, stats: dict, prefix: str) -> list[dict]:
    """Turn the stage-2 SFT pairs (chosen = bridges) into DPO pairs whose
    rejected side is a failure the CURRENT adapter produced in the rollout
    tagged `neg_stage` — an on-policy negative, free of extra generation.

    One pair per bridge (negatives cycle when fewer than bridges), so the chosen
    side is identical to the SFT arm and an SFT-vs-DPO comparison isolates the
    objective. Both sequences are built by templates.bridge_pair_ids, so they
    share the prompt (+anchor) ids and a single prompt_len. Questions with no
    negative are dropped (counted), as are pairs over max_pair_tokens."""
    negs: dict[str, list[ImprovedCandidate]] = defaultdict(list)
    for c in pool:
        if not c.correct and c.op_meta.get("stage") == neg_stage:
            negs[c.qid].append(c)
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for p in sft_pairs:
        by_qid[p["qid"]].append(p)

    out: list[dict] = []
    n_no_neg = n_long = 0
    for qid in sorted(by_qid):
        cands = sorted(negs.get(qid, []), key=lambda c: c.attempt_idx)
        if not cands:
            n_no_neg += 1
            continue
        if selection == "shortest":
            cands.sort(key=lambda c: (len(c.continuation_token_ids), c.attempt_idx))
        elif selection == "longest":
            cands.sort(key=lambda c: (-len(c.continuation_token_ids), c.attempt_idx))
        else:
            random.Random(seed_of(qid)).shuffle(cands)
        anchor_ids = list(anchors_by_qid[qid].anchor_token_ids)
        chosen_list = by_qid[qid][:max_per_question] if max_per_question else by_qid[qid]
        for i, chosen in enumerate(chosen_list):
            neg = cands[i % len(cands)]
            rej_ids, plen = bridge_pair_ids(prompts[qid], anchor_ids,
                                            neg.continuation_token_ids, eos)
            if plen != chosen["prompt_len"]:   # same prompt+anchor -> must match
                raise RuntimeError(f"dpo pair prompt_len mismatch for {qid}")
            if len(rej_ids) > max_pair_tokens or len(chosen["input_ids"]) > max_pair_tokens:
                n_long += 1
                continue
            out.append({"qid": qid, "prompt_len": plen,
                        "chosen_ids": list(chosen["input_ids"]),
                        "rejected_ids": rej_ids})
    stats[f"{prefix}_dpo_pairs"] = len(out)
    stats[f"{prefix}_dpo_no_negative"] = n_no_neg
    stats[f"{prefix}_dpo_dropped_long"] = n_long
    return out


def _assign_chunks(fit_qids: list[str], size: int,
                   key_of: dict[str, tuple[int, ...]]) -> list[tuple[str, list[str]]]:
    """Hierarchical, stable sharding of a fit set. Returns [(label, qids)].

    `key_of` maps a question to the shard path it has been assigned so far —
    () for "pooled / not yet sharded", (i,) after a first split, (i, j) after a
    sub-split. Rules:
      * size <= 0 -> ONE pooled shard (label "c0"); every question's key is
        reset to () (a pooled fit has no shard identity to chain from).
      * size > 0  -> questions are grouped by their CURRENT key (the parent
        shard) and each parent group is cut into sub-shards of `size`. A parent
        that yields a single sub-shard keeps its key (so with equal sizes a
        shard simply shrinks as questions get solved and chains from its own
        previous adapter); a parent that splits gets child keys (parent..., j).
      * newcomers (no key yet, e.g. solved questions joining via full_pool) join
        the smallest parent group before splitting.
    Labels are "c" + "_".join(path): c0, c1, ... / c0_0, c0_1, ... With
    num_stages: 1 and a pooled stage 1 this is exactly a slice of the sorted fit
    set — the same semantics as lora_sft's pooled chunking."""
    qids = sorted(fit_qids)
    if size <= 0:
        for q in qids:
            key_of[q] = ()
        return [("c0", qids)]
    parents: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for q in qids:
        parents[key_of.get(q, ())].append(q)
    if () in parents and len(parents) > 1:        # newcomers among assigned ones
        for q in parents.pop(()):
            k = min(parents, key=lambda k: (len(parents[k]), k))
            parents[k].append(q)
    out: list[tuple[str, list[str]]] = []
    for pkey in sorted(parents):
        members = sorted(parents[pkey])
        subs = [members[i:i + size] for i in range(0, len(members), size)]
        for j, sub in enumerate(subs):
            key = pkey if (len(subs) == 1 and pkey) else pkey + (j,)
            for q in sub:
                key_of[q] = key
            out.append(("c" + "_".join(map(str, key)), sub))
    return out


def _merge_pairs(old: list[dict], new: list[dict], cap: int, selection: str,
                 rng_seed: int) -> list[dict]:
    """add_bridge merge: union of a question's stage-1 and freshly generated
    pairs, deduped by input_ids (same qid -> same prompt, so this equals
    continuation dedup), re-ranked by bridge.keep_selection, capped at cap."""
    seen: set[tuple] = set()
    pool: list[dict] = []
    for pair in list(old) + list(new):
        key = tuple(pair["input_ids"])
        if key in seen:
            continue
        seen.add(key)
        pool.append(pair)
    if selection == "random":
        random.Random(rng_seed).shuffle(pool)
    else:
        pool.sort(key=lambda p: len(p["input_ids"]),
                  reverse=(selection == "longest"))
    return pool[:cap]


def _flatten_pairs(targets: dict[str, list[dict]], qids: list[str],
                   max_pair_tokens: int) -> tuple[list[dict], int]:
    """Flatten qid->pairs for the given qids, dropping over-budget pairs.
    Returns (pairs, n_dropped_long)."""
    pairs: list[dict] = []
    dropped = 0
    for qid in qids:
        for pair in targets.get(qid, []):
            if len(pair["input_ids"]) > max_pair_tokens:
                dropped += 1
                continue
            pairs.append(pair)
    return pairs, dropped
