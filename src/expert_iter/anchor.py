"""Stage: anchor — ⚗ extension point (I): choose the anchor prefix of a failed
rollout that we commit to keeping.

An AnchorPolicy sees the question and one failed rollout and returns the number
of leading RESPONSE tokens to keep. The anchor is materialized as an id-slice
of that rollout's response_token_ids — never re-tokenized text.

Registering a new policy:

    @register(ANCHOR_POLICIES, "logprob_dip")
    class LogprobDipAnchor(AnchorPolicy):
        def select_len(self, question, failed, params): ...
"""

from __future__ import annotations

import random
import sys
from abc import ABC, abstractmethod
from collections import defaultdict

from .config import load_stage_config, stage_argparser
from .records import AnchorRecord, RolloutSample, UnsolvedQuestion
from .registry import ANCHOR_POLICIES, build, register
from .utils import is_done, iter_dir, mark_done


class AnchorPolicy(ABC):
    name: str

    @abstractmethod
    def select_len(self, question: UnsolvedQuestion, failed: RolloutSample, params: dict) -> int:
        """Return anchor length in response tokens (0 => resample from scratch)."""


@register(ANCHOR_POLICIES, "fixed_fraction")
class FixedFractionAnchor(AnchorPolicy):
    """Keep the first `fraction` of the failed response, clamped to
    [min_tokens, max_tokens]. The simplest possible baseline."""

    def select_len(self, question: UnsolvedQuestion, failed: RolloutSample, params: dict) -> int:
        n = len(failed.response_token_ids)
        frac = float(params.get("fraction", 0.3))
        lo = int(params.get("min_tokens", 32))
        hi = int(params.get("max_tokens", 2048))
        return max(0, min(n, max(lo, min(hi, round(frac * n)))))


@register(ANCHOR_POLICIES, "none")
class NoAnchor(AnchorPolicy):
    """Empty anchor: the improvement operator resamples the whole response.
    Useful as the rejection-sampling / STaR ablation baseline."""

    def select_len(self, question, failed, params) -> int:
        return 0


def main(argv: list[str] | None = None) -> None:
    args = stage_argparser("EI anchor stage").parse_args(argv)
    cfg = load_stage_config(args)
    it_dir = iter_dir(args.run_dir, args.iteration)
    out_path = it_dir / "anchors" / "anchors.jsonl"
    if is_done(out_path, config_hash=cfg.hash()):
        print(f"[anchor] {out_path} already done, skipping")
        return

    unsolved = list(UnsolvedQuestion.load_jsonl(it_dir / "partition" / "unsolved.jsonl"))
    rollouts: dict[str, dict[int, RolloutSample]] = defaultdict(dict)
    unsolved_qids = {u.qid for u in unsolved}
    for s in RolloutSample.load_jsonl(it_dir / "rollout" / "rollouts.jsonl"):
        if s.qid in unsolved_qids:
            rollouts[s.qid][s.sample_idx] = s

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    policy: AnchorPolicy = build(ANCHOR_POLICIES, cfg.anchor.policy)
    rng = random.Random(cfg.run.seed * 7919 + args.iteration)

    rows: list[AnchorRecord] = []
    for u in unsolved:
        failed = _pick_base(
            [rollouts[u.qid][i] for i in u.failed_sample_idxs if i in rollouts[u.qid]],
            cfg.anchor.base_selection, rng,
        )
        if failed is None:
            continue
        alen = policy.select_len(u, failed, cfg.anchor.params)
        anchor_ids = failed.response_token_ids[:alen]
        rows.append(AnchorRecord(
            qid=u.qid,
            base_sample_idx=failed.sample_idx,
            policy=cfg.anchor.policy,
            anchor_token_ids=anchor_ids,
            anchor_text=tokenizer.decode(anchor_ids),
            anchor_len=len(anchor_ids),
            base_response_len=len(failed.response_token_ids),
            iter=args.iteration,
        ))

    n = AnchorRecord.dump_jsonl(out_path, rows)
    mark_done(out_path, count=n, config_hash=cfg.hash())
    lens = [r.anchor_len for r in rows]
    print(f"[anchor] wrote {n} anchors (policy={cfg.anchor.policy}, "
          f"mean_len={sum(lens) / len(lens):.0f})" if lens else f"[anchor] wrote 0 anchors")


def _pick_base(failed: list[RolloutSample], how: str, rng: random.Random) -> RolloutSample | None:
    if not failed:
        return None
    if how == "longest":
        return max(failed, key=lambda s: len(s.response_token_ids))
    if how == "random":
        return rng.choice(failed)
    return min(failed, key=lambda s: s.sample_idx)  # first_failed


if __name__ == "__main__":
    main(sys.argv[1:])
