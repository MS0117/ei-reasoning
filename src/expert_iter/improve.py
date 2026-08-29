"""Stage: improve — ⚗ extension point (II): the improvement operator.

An ImprovementOperator receives (question, anchor) and produces candidate
continuations of `question_prompt + anchor` that hopefully reach the correct
answer. The operator CONTRACT for learnability (methodology step 4):

  * The continuation must be generatable from question+anchor alone at
    inference time. If the operator conditioned on anything else (a hint, a
    critique, the gold answer, a teacher's text shown in-context), it MUST
    record that in ImprovedCandidate.external_context. The no_external_context
    gate then rejects such candidates unless a mismatch-absorbing training
    method is explicitly configured.

Default operator `self_resample`: best-of-n continuation by the policy itself
at a higher temperature — trivially learnability-safe (external_context=None).
`teacher` is a registered stub documenting the intended contract.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

from .config import Config, load_stage_config, stage_argparser
from .engine import GenRequest, run_pool
from .records import AnchorRecord, ImprovedCandidate, UnsolvedQuestion
from .registry import OPERATORS, build, register
from .templates import continuation_prompt_ids, gold_pair_ids, render_question_prompt
from .utils import is_done, iter_dir, mark_done, stable_seed


class ImprovementOperator(ABC):
    name: str
    required_models: tuple[str, ...] = ("policy",)

    @abstractmethod
    def propose(
        self,
        questions: list[UnsolvedQuestion],
        anchors: list[AnchorRecord],
        prompts: dict[str, list[int]],      # qid -> chat-templated question prompt ids
        cfg: Config,
        *,
        model_paths: dict[str, str],        # {"policy": path, "teacher": path?}
        work_dir,
        iteration: int,
        gold_solutions: dict[str, str] | None = None,  # qid -> y* (privileged; weights/
        # scoring channels ONLY — an operator that shows y* in a prompt must
        # record that in external_context)
    ) -> list[ImprovedCandidate]:
        ...


@register(OPERATORS, "self_resample")
class SelfResampleOperator(ImprovementOperator):
    """Continue from the anchor with the policy itself, improve.n samples at
    improve.temperature. No external information — external_context is None
    by construction, so every candidate passes the learnability gate."""

    def propose(self, questions, anchors, prompts, cfg, *, model_paths, work_dir,
                iteration, gold_solutions=None):
        anchors_by_qid = {a.qid: a for a in anchors}
        usable = [q for q in questions if q.qid in anchors_by_qid]
        requests = [
            GenRequest(
                rid=q.qid,
                prompt_token_ids=continuation_prompt_ids(
                    prompts[q.qid], anchors_by_qid[q.qid].anchor_token_ids
                ),
                n=cfg.improve.n,
                seed=stable_seed(cfg.run.seed, "improve", iteration, q.qid),
            )
            for q in usable
        ]
        results = run_pool(
            requests,
            mode="generate",
            model_path=model_paths["policy"],
            sampling={
                "temperature": cfg.improve.temperature,
                "top_p": cfg.improve.top_p,
                "max_tokens": cfg.improve.max_tokens,
            },
            engine_cfg=cfg.engine,
            work_dir=work_dir,
            dtype=cfg.model.dtype,
        )
        out: list[ImprovedCandidate] = []
        for q, res in zip(usable, results):
            a = anchors_by_qid[q.qid]
            for ai, s in enumerate(res.samples):
                if s["finish_reason"] != "stop":
                    continue  # truncated continuations are untrainable
                out.append(ImprovedCandidate(
                    qid=q.qid,
                    base_sample_idx=a.base_sample_idx,
                    attempt_idx=ai,
                    prompt_token_ids=prompts[q.qid],
                    anchor_token_ids=a.anchor_token_ids,
                    continuation_token_ids=s["token_ids"],
                    continuation_text=s["text"],
                    operator=self.name,
                    op_meta={"temperature": cfg.improve.temperature},
                    external_context=None,
                    iter=iteration,
                ))
        return out


@register(OPERATORS, "gold_text")
class GoldTextOperator(ImprovementOperator):
    """L5 BASELINE: off-policy gold SFT — the reference solution y* IS the
    training text.

    Every other operator here routes y* through adapter weights or a
    generation-time prompt and trains on the POLICY's own writing; this one
    trains on y* verbatim. That is the control for "does the rescue trajectory
    have to be on-policy, or is any correct solution enough?", and it is
    deliberately the one operator that VIOLATES the learnability contract:
    external_context carries y*, so the no_external_context gate rejects every
    candidate unless the arm drops that gate (configs/methods/l5_gold_sft.yaml
    does, and documents why).

    No generation, so the stage is CPU-only and instant. One candidate per
    cliff question — half the per-question dose of an operator that samples,
    which is inherent to gold (there is only one y*) and must be read off the
    dataset stats rather than papered over.
    """

    def propose(self, questions, anchors, prompts, cfg, *, model_paths, work_dir,
                iteration, gold_solutions=None):
        from transformers import AutoTokenizer

        gold = gold_solutions or {}
        anchors_by_qid = {a.qid: a for a in anchors}
        tokenizer = AutoTokenizer.from_pretrained(model_paths["policy"])
        eos = tokenizer.eos_token_id
        out: list[ImprovedCandidate] = []
        n_missing = 0
        for q in questions:
            a = anchors_by_qid.get(q.qid)
            if a is None:
                continue
            y = (gold.get(q.qid) or "").strip()
            if not y:
                n_missing += 1
                continue
            # gold_pair_ids is the sanctioned boundary where y* becomes ids: y*
            # is a training TARGET, not a splice of generated ids, so fresh
            # tokenization is legitimate exactly here.
            ids, prompt_len = gold_pair_ids(tokenizer, prompts[q.qid], y, eos)
            out.append(ImprovedCandidate(
                qid=q.qid,
                base_sample_idx=a.base_sample_idx,
                attempt_idx=0,
                prompt_token_ids=prompts[q.qid],
                # y* is a standalone solution, never a continuation of a failed
                # prefix, so it carries no anchor even when one was computed.
                anchor_token_ids=[],
                continuation_token_ids=ids[prompt_len:],
                continuation_text=y,
                operator=self.name,
                op_meta={"source": "meta.gold_solution"},
                external_context=y,
                iter=iteration,
            ))
        if n_missing:
            print(f"[improve] gold_text: {n_missing} question(s) without "
                  "meta.gold_solution — skipped")
        return out


@register(OPERATORS, "teacher")
class TeacherOperator(ImprovementOperator):
    """⚗ STUB — feedback/teacher-guided improvement.

    Intended contract when implemented:
      * `improve.teacher.model` is loaded on `improve.teacher.gpus` as a second
        pool; the policy pool gets the remaining GPUs.
      * Whatever the teacher contributes (a critique shown in-context, a hint
        prepended to the continuation prompt, a rewritten step) must be either
        (a) absent from the final continuation (the policy re-derives it), or
        (b) recorded verbatim in external_context so the learnability gate /
            a mismatch-absorbing objective can handle it downstream.
      * Candidates must still be pure continuations of question+anchor at the
        token-id level: teacher text never leaks into continuation_token_ids
        unless it is exactly what we want the policy to learn to emit.
    """

    required_models = ("policy", "teacher")

    def propose(self, *a, **k):
        raise NotImplementedError(
            "TeacherOperator is a stub. Implement propose() per the class "
            "docstring contract, then select it with improve.operator=teacher "
            "and set improve.teacher.model / improve.teacher.gpus."
        )


def main(argv: list[str] | None = None) -> None:
    args = stage_argparser("EI improve stage").parse_args(argv)
    cfg = load_stage_config(args)
    it_dir = iter_dir(args.run_dir, args.iteration)
    out_path = it_dir / "improve" / "improved.jsonl"
    if is_done(out_path, config_hash=cfg.hash()):
        print(f"[improve] {out_path} already done, skipping")
        return

    unsolved = list(UnsolvedQuestion.load_jsonl(it_dir / "partition" / "unsolved.jsonl"))
    anchors = list(AnchorRecord.load_jsonl(it_dir / "anchors" / "anchors.jsonl"))
    print(f"[improve] operator={cfg.improve.operator} on {len(anchors)} anchored questions")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts = {
        u.qid: render_question_prompt(
            tokenizer, u.question,
            system_prompt=cfg.model.system_prompt,
            question_suffix=cfg.data.question_suffix,
            chat_template_kwargs=cfg.model.chat_template_kwargs,
        ).token_ids
        for u in unsolved
    }

    from . import bridge_sft, lora_sft, staged_bridge_sft  # noqa: F401 — @register side effects (deferred: circular import)
    from .data import load_gold_solutions

    operator: ImprovementOperator = build(OPERATORS, cfg.improve.operator)
    model_paths = {"policy": args.model_path}
    if "teacher" in operator.required_models:
        if not cfg.improve.teacher.model:
            raise ValueError(f"operator {operator.name!r} needs improve.teacher.model")
        model_paths["teacher"] = cfg.improve.teacher.model

    candidates = operator.propose(
        unsolved, anchors, prompts, cfg,
        model_paths=model_paths,
        work_dir=it_dir / "improve" / "pool",
        iteration=args.iteration,
        gold_solutions=load_gold_solutions(args.run_dir),
    )
    n = ImprovedCandidate.dump_jsonl(out_path, candidates)
    mark_done(out_path, count=n, config_hash=cfg.hash())
    print(f"[improve] wrote {n} candidates -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
