"""Improvement operator (II): bridge SFT — LoRA fit on self-generated bridges.

Instead of fitting the transient LoRA on gold (x -> y*) pairs (lora_sft), the
base policy pi_theta first generates BRIDGE trajectories z+ from a privileged
prompt that shows y* (self-improvement setting; templates.render_bridge_prompt):

  anchored:  [x] [y* "never mention its existence"] [attempt a] "continue
             seamlessly, only the continuation, self-correct naturally"
  plain:     [x] [y*] "write your own complete solution guided by it"

N_B samples per cliff -> verifier acceptance (default) -> optional G5 leakage
screens (regex rules and/or LLM judge, both OFF by default — confirmed) -> B+.
B+ empty -> ONE retry pass at bridge.retry_temperature -> still empty -> the
cliff joins the skip list (with sample_skipped=true it is still sampled from
its pooled chunk adapter — it merely contributed no fit pairs).

Fit pairs are (PLAIN x prompt ids [+ anchor ids] -> z+ ids) built by pure id
concatenation (templates.bridge_pair_ids) with loss on z+ only — y* reached
the weights through generation, never the candidate's prompt or training text.
Everything downstream (chunked fits, q_phi sampling, project-back, C(y)) is
inherited from LoraSftOperator unchanged.

Module rule: like lora_sft, NEVER import expert_iter.improve (module-as-main
re-execution would duplicate registrations). Shared infrastructure is reached
through the lora_sft module object so tests can fake one `run_pool`.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from .config import Config
from .engine import GenRequest
from .records import ImprovedCandidate, QuestionRecord
from .registry import OPERATORS, VERIFIERS, build, register
from .templates import bridge_pair_ids, continuation_prompt_ids, render_bridge_prompt
from .utils import stable_seed, write_json, write_jsonl
from . import lora_sft as _ls


@register(OPERATORS, "bridge_sft")
class BridgeSftOperator(_ls.LoraSftOperator):
    PROVENANCE = {
        "privileged": "gold_solution",
        "channel": "lora_weights",
        "via": "bridge_trajectories",
    }

    def _sample_unpaired(self, cfg: Config) -> bool:
        return cfg.improve.lora_sft.bridge.sample_skipped

    def _build_targets(self, qids, *, tokenizer, prompts, anchors_by_qid,
                       questions_by_qid, grader, cfg: Config, policy, pool_base,
                       improve_dir, iteration, gold_solutions, stats):
        br = cfg.improve.lora_sft.bridge
        fit_budget = cfg.improve.lora_sft.fit.max_pair_tokens
        eos = tokenizer.eos_token_id
        rules = ([re.compile(p, re.IGNORECASE) for p in cfg.filter.leakage.patterns]
                 if br.leakage_rules else [])

        # ---- per-qid privileged prompts + generation budgets ----------------
        gen_prompt: dict[str, list[int]] = {}
        gen_max: dict[str, int] = {}
        n_budget_skipped = 0
        for qid in qids:
            anchor = anchors_by_qid[qid]
            bp = render_bridge_prompt(
                tokenizer, questions_by_qid[qid].question, gold_solutions[qid],
                anchored=len(anchor.anchor_token_ids) > 0,
                system_prompt=cfg.model.system_prompt,
                question_suffix=cfg.data.question_suffix,
                chat_template_kwargs=cfg.model.chat_template_kwargs,
            )
            # the FIT PAIR (plain prompt + anchor + z+), not the bridge prompt,
            # must fit the pair budget
            pair_room = fit_budget - len(prompts[qid]) - len(anchor.anchor_token_ids)
            max_tok = min(br.max_tokens or cfg.improve.max_tokens, pair_room)
            if max_tok <= 0:
                n_budget_skipped += 1
                continue
            gen_prompt[qid] = continuation_prompt_ids(bp.token_ids, anchor.anchor_token_ids)
            gen_max[qid] = max_tok

        # ---- generate -> verify -> (rules) -> (judge), with ONE retry -------
        accepted: dict[str, list[dict]] = {}
        log_rows: list[dict] = []
        counters = {"generated": 0, "correct": 0, "leak_rule": 0, "leak_judge": 0}
        pending = sorted(gen_prompt)
        passes = [
            ("bridge", br.temperature if br.temperature is not None else cfg.improve.temperature),
            ("bridge_retry", br.retry_temperature),
        ]
        n_retried = 0
        for pass_name, temperature in passes:
            if not pending:
                break
            if pass_name == "bridge_retry":
                n_retried = len(pending)
            # staged_bridge_sft seams: when set, bridges are regenerated THROUGH
            # the current adapter with a per-stage seed salt and (optionally) a
            # per-stage sample count. All default to None, keeping this
            # operator's requests (and seeds) byte-identical.
            bridge_lora = getattr(self, "_bridge_lora_path", None)
            # per-question override: with stage-2 sharding each question's
            # "current adapter" is its own shard, not one pooled adapter
            bridge_lora_of = getattr(self, "_bridge_lora_of", None) or {}
            seed_salt = getattr(self, "_bridge_seed_salt", None)
            bridge_n = getattr(self, "_bridge_n", None) or br.n
            results = _ls.run_pool(
                [GenRequest(
                    rid=f"{qid}:{pass_name}",
                    prompt_token_ids=gen_prompt[qid],
                    n=bridge_n,
                    seed=(stable_seed(cfg.run.seed, pass_name, iteration, qid)
                          if seed_salt is None
                          else stable_seed(cfg.run.seed, pass_name, iteration, qid, seed_salt)),
                    max_tokens=gen_max[qid],
                    lora_path=bridge_lora_of.get(qid, bridge_lora),
                ) for qid in pending],
                mode="generate", model_path=policy,
                sampling={
                    "temperature": temperature,
                    "top_p": cfg.improve.top_p,
                    "max_tokens": cfg.improve.max_tokens,
                },
                engine_cfg=cfg.engine, work_dir=pool_base / pass_name,
                dtype=cfg.model.dtype,
            )
            pass_accepted: dict[str, list[dict]] = {}
            pass_log: dict[tuple[str, int], dict] = {}
            for qid, res in zip(pending, results):
                q = questions_by_qid[qid]
                anchor_ids = list(anchors_by_qid[qid].anchor_token_ids)
                for si, s in enumerate(res.samples):
                    counters["generated"] += 1
                    row = {"qid": qid, "pass": pass_name, "sample_idx": si,
                           "n_tokens": len(s["token_ids"]),
                           "finish_reason": s["finish_reason"],
                           "correct": False, "leak_rule": False,
                           "leak_judge": False, "kept": False}
                    log_rows.append(row)
                    pass_log[(qid, si)] = row
                    if s["finish_reason"] != "stop":
                        continue
                    verdict = grader.verify(
                        QuestionRecord(qid=q.qid, question=q.question,
                                       final_answer=q.final_answer),
                        tokenizer.decode(anchor_ids + list(s["token_ids"])),
                    )
                    row["correct"] = bool(verdict.correct)
                    if not verdict.correct:
                        continue
                    counters["correct"] += 1
                    if rules and any(rx.search(s["text"]) for rx in rules):
                        row["leak_rule"] = True
                        counters["leak_rule"] += 1
                        continue
                    pass_accepted.setdefault(qid, []).append(
                        {"sample_idx": si, "sample": s, "row": row})
            if br.judge_enabled and pass_accepted:
                pass_accepted = self._judge_bridges(
                    pass_accepted, pass_log, counters, cfg, policy,
                    anchors_by_qid, improve_dir, pass_name,
                )
            accepted.update(pass_accepted)
            pending = [qid for qid in pending if qid not in accepted]
        skipped = pending  # empty B+ even after the retry

        # ---- fit pairs: PLAIN prompt [+ anchor] -> z+ (id concat) -----------
        targets: dict[str, list[dict]] = {}
        n_pairs_kept = 0
        max_keep = getattr(self, "_bridge_max_keep", None) or br.max_keep
        # (sample_idx, sample, pass) of every kept bridge — read by bridge_text,
        # which emits them as candidates instead of fitting on them
        kept: dict[str, list[tuple[int, dict, str]]] = {}
        for qid, items in accepted.items():
            anchor_ids = list(anchors_by_qid[qid].anchor_token_ids)
            if br.keep_selection == "random":
                items = sorted(items, key=lambda it: it["sample_idx"])
                rng = random.Random(stable_seed(cfg.run.seed, "bridge_keep", iteration, qid))
                rng.shuffle(items)
            else:
                items = sorted(items, key=lambda it: len(it["sample"]["token_ids"]),
                               reverse=(br.keep_selection == "longest"))
            pairs = []
            for it in items[:max_keep]:
                ids, plen = bridge_pair_ids(
                    prompts[qid], anchor_ids, it["sample"]["token_ids"], eos)
                pairs.append({"qid": qid, "input_ids": ids, "prompt_len": plen})
                it["row"]["kept"] = True
                kept.setdefault(qid, []).append(
                    (it["sample_idx"], it["sample"], it["row"]["pass"]))
            targets[qid] = pairs
            n_pairs_kept += len(pairs)
        self._kept_bridges = kept

        write_jsonl(improve_dir / "bridge" / "bridges.jsonl", log_rows)
        stats.update({
            "n_bridge_generated": counters["generated"],
            "n_bridge_correct": counters["correct"],
            "n_bridge_leak_rule": counters["leak_rule"],
            "n_bridge_leak_judge": counters["leak_judge"],
            "n_bridge_retried": n_retried,
            "n_bridge_budget_skipped": n_budget_skipped,
            "n_questions_bridged": len(targets),
            "n_bridge_skipped": len(skipped),
            "bridge_skipped_qids": sorted(skipped),
            "bridge_pairs_kept": n_pairs_kept,
            "bridge_yield": round(len(targets) / len(qids), 4) if qids else 0.0,
        })
        return targets

    @staticmethod
    def _judge_bridges(pass_accepted, pass_log, counters, cfg: Config, policy,
                       anchors_by_qid, improve_dir, pass_name):
        """Batched G5 LLM judge over this pass's accepted bridges, reusing the
        filters-stage judge (throwaway ImprovedCandidate wrappers carry the z+
        text; the anchor needs no screening — it predates the privileged
        prompt)."""
        from .filters import _leakage_judge  # late import: monkeypatch-friendly

        wrappers = []
        origin: dict[tuple[str, int], tuple[str, dict]] = {}
        for qid, items in pass_accepted.items():
            for it in items:
                wrappers.append(ImprovedCandidate(
                    qid=qid,
                    base_sample_idx=anchors_by_qid[qid].base_sample_idx,
                    attempt_idx=it["sample_idx"],
                    prompt_token_ids=[], anchor_token_ids=[],
                    continuation_token_ids=list(it["sample"]["token_ids"]),
                    continuation_text=it["sample"]["text"],
                ))
                origin[(qid, it["sample_idx"])] = (qid, it)
        kept_wrappers, n_flagged, _report = _leakage_judge(
            wrappers, cfg, policy, improve_dir.parent,
            pool_dir=improve_dir / "bridge" / f"pool_judge_{pass_name}",
        )
        counters["leak_judge"] += n_flagged
        kept_keys = {(w.qid, w.attempt_idx) for w in kept_wrappers}
        for key, row in pass_log.items():
            if row["correct"] and not row["leak_rule"] and key not in kept_keys:
                row["leak_judge"] = True
        out: dict[str, list[dict]] = {}
        for key in kept_keys:
            qid, it = origin[key]
            out.setdefault(qid, []).append(it)
        return out


@register(OPERATORS, "bridge_text")
class BridgeTextOperator(BridgeSftOperator):
    """L5 BASELINE: STaR-style rationalization on cliffs — the bridge
    trajectories z+ ARE the training text.

    Same bridge generation as bridge_sft / staged_bridge_sft (privileged prompt
    showing y*, verifier acceptance, one retry, keep_selection / max_keep), but
    the accepted bridges are emitted as candidates directly: no LoRA fit, no
    plain-prompt resample, no adapter served. z+ was written while y* was
    visible, so like gold_text this operator VIOLATES the learnability contract
    on purpose — external_context carries y* and the arm has to drop the
    no_external_context gate (configs/methods/l5_bridge_inloop.yaml does, and
    says why).

    What the resample step does for the LoRA operators and this one skips,
    MEASURED on runs/L2_freeze_20260825_040504: 35.5% of kept bridges mention a
    "reference solution" (regex over filter.leakage.patterns) against 0.2% of
    the plain-prompt candidates; verbatim y* copying is ~0 (8-gram) either way.
    Leakage screening stays OFF here — the STaR original does not screen — and
    the phrasing rate is REPORTED in stats.json, never filtered on.
    """

    PROVENANCE = {
        "privileged": "gold_solution",
        "channel": "training_text",
        "via": "bridge_trajectories",
    }

    def propose(self, questions, anchors, prompts, cfg: Config, *,
                model_paths, work_dir, iteration, gold_solutions=None):
        from transformers import AutoTokenizer

        gold_solutions = gold_solutions or {}
        pool_base = Path(work_dir)
        improve_dir = pool_base.parent
        policy = model_paths["policy"]
        tokenizer = AutoTokenizer.from_pretrained(policy)
        grader = build(VERIFIERS, cfg.partition.verifier)

        # Mirrors LoraSftOperator.propose up to the fit-target hook, then stops.
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

        self._kept_bridges = {}
        self._build_targets(
            with_gold, tokenizer=tokenizer, prompts=prompts,
            anchors_by_qid=anchors_by_qid, questions_by_qid=questions_by_qid,
            grader=grader, cfg=cfg, policy=policy, pool_base=pool_base,
            improve_dir=improve_dir, iteration=iteration,
            gold_solutions=gold_solutions, stats=stats,
        )

        # Report-only: the same regex the leakage_rules gate would apply. It
        # counts, it never drops — the arm's definition.
        rules = [re.compile(p, re.IGNORECASE) for p in cfg.filter.leakage.patterns]
        out: list[ImprovedCandidate] = []
        n_leak = 0
        for qid in sorted(self._kept_bridges):
            anchor = anchors_by_qid[qid]
            for sample_idx, sample, pass_name in self._kept_bridges[qid]:
                if rules and any(rx.search(sample["text"]) for rx in rules):
                    n_leak += 1
                out.append(ImprovedCandidate(
                    qid=qid,
                    base_sample_idx=anchor.base_sample_idx,
                    attempt_idx=sample_idx,
                    prompt_token_ids=list(prompts[qid]),
                    anchor_token_ids=list(anchor.anchor_token_ids),
                    # ids straight from generation — never re-tokenized
                    continuation_token_ids=list(sample["token_ids"]),
                    continuation_text=sample["text"],
                    correct=True,               # bridge acceptance is verifier-gated
                    operator=self.name,
                    op_meta={**self.PROVENANCE, "pass": pass_name},
                    external_context=gold_solutions[qid],
                    iter=iteration,
                ))
        stats["n_bridge_leak_phrasing_reported"] = n_leak
        stats["n_candidates"] = len(out)
        write_json(improve_dir / "stats.json", stats)
        return out
