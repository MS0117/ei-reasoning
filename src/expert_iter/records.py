"""Typed records for every stage boundary, serialized as JSONL.

Token ids are the source of truth wherever text is spliced or loss-weighted;
text fields exist for human inspection only. The anchor is ALWAYS an id-slice
of the original rollout's response_token_ids — never re-tokenized text —
because BPE merges across a splice point silently shift region boundaries.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Type, TypeVar

from .utils import read_jsonl, write_jsonl

SCHEMA_VERSION = 1

R = TypeVar("R", bound="Record")


@dataclass
class Record:
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls: Type[R], d: dict) -> R:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load_jsonl(cls: Type[R], path: str | Path) -> Iterator[R]:
        for row in read_jsonl(path):
            yield cls.from_dict(row)

    @staticmethod
    def dump_jsonl(path: str | Path, records: Iterator["Record"] | list["Record"]) -> int:
        return write_jsonl(path, (r.to_dict() for r in records))


# ---------------------------------------------------------------------------
# Canonical question (produced by data.py adapters)
# ---------------------------------------------------------------------------

@dataclass
class QuestionRecord(Record):
    qid: str
    question: str
    final_answer: str          # gold answer string, parseable by the verifier
    domain: str = "math"
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# rollout/rollouts.jsonl — one line per sample
# ---------------------------------------------------------------------------

@dataclass
class RolloutSample(Record):
    qid: str
    sample_idx: int
    prompt_text: str
    prompt_token_ids: list[int]
    response_text: str
    response_token_ids: list[int]
    finish_reason: str          # "stop" | "length"
    gen: dict = field(default_factory=dict)   # {temperature, top_p, max_tokens, seed}
    model_path: str = ""
    iter: int = 0


# ---------------------------------------------------------------------------
# partition/ outputs
# ---------------------------------------------------------------------------

@dataclass
class VerdictRecord(Record):
    qid: str
    sample_idx: int
    correct: bool
    extracted_answer: str | None = None
    verifier: str = "math"
    meta: dict = field(default_factory=dict)


@dataclass
class SolvedTrajectory(Record):
    qid: str
    question: str
    final_answer: str
    sample_idx: int
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_text: str
    source: str = "rollout"
    iter: int = 0


@dataclass
class UnsolvedQuestion(Record):
    qid: str
    question: str
    final_answer: str
    failed_sample_idxs: list[int] = field(default_factory=list)
    iter: int = 0


# ---------------------------------------------------------------------------
# anchors/anchors.jsonl — ⚗ extension point I output
# ---------------------------------------------------------------------------

@dataclass
class AnchorRecord(Record):
    qid: str
    base_sample_idx: int        # which failed rollout the anchor was cut from
    policy: str
    anchor_token_ids: list[int]  # slice of the rollout's response_token_ids
    anchor_text: str            # decode of the ids — debug only, never re-encoded
    anchor_len: int
    base_response_len: int
    meta: dict = field(default_factory=dict)
    iter: int = 0


# ---------------------------------------------------------------------------
# improve/improved.jsonl — ⚗ extension point II output (ALL attempts kept)
# ---------------------------------------------------------------------------

@dataclass
class ImprovedCandidate(Record):
    qid: str
    base_sample_idx: int
    attempt_idx: int
    prompt_token_ids: list[int]        # chat-templated question incl. generation prompt
    anchor_token_ids: list[int]
    continuation_token_ids: list[int]
    continuation_text: str
    correct: bool | None = None        # None until the correctness gate fills it
    operator: str = ""
    op_meta: dict = field(default_factory=dict)
    # Structural learnability field: anything shown to the model beyond
    # question+anchor (hint, critique, gold solution, ...) MUST be recorded
    # here by the operator. Non-None ⇒ rejected by the no_external_context
    # gate unless a mismatch-absorbing training method is configured.
    external_context: str | None = None
    iter: int = 0


# ---------------------------------------------------------------------------
# filtered/ + dataset/ — train-ready examples
# ---------------------------------------------------------------------------

@dataclass
class SFTExample(Record):
    """input_ids = prompt ids + anchor ids + completion ids (pre-concatenated).

    Region lengths — not baked weights — are stored so region->weight mapping
    stays a config knob (`train.sft.region_weights`): weight sweeps re-train
    without rebuilding data. Invariant: prompt_len+anchor_len+completion_len
    == len(input_ids); anchor_len == 0 for source in ("solved", "negative").

    Cliff-objective fields (train.sft.cliff — see docs/objective_decision_20260823.md §3):
      source "negative" = a base-policy modal-wrong failure rollout, trained with
      bounded unlikelihood (L_N v1), never with CE.
      n_q = per-question mass for the per-question normalizer: on "improved" rows
      the number of kept rescued successes for that qid, on "negative" rows the
      number of negative rows for that qid (n⁻_q). 0 = not a cliff-term row /
      written by pre-cliff code; build_dataset restamps it every iteration.
      ref_mean_nll = the C(y) scoring pass's s_mean (mean NLL over the
      continuation region under the pre-train policy) — the displacement
      guard's reference. None = unavailable (guard skips the row).

    Group-advantage fields (train.objective: gadv — docs/objective_gadv_spec_20260903.md):
      source "wrong" = a base rollout with a NEGATIVE advantage (frontier
      failure, truncated sample, or a rescued cliff's failure); gadv only.
      advantage = the signed per-row weight A (0 on rows written by other
      objectives); group_kind = "frontier" | "rescue" | "floor";
      group_size = the group's member count the advantage was computed under.
    """

    uid: str
    qid: str
    source: str                 # "solved" | "improved" | "negative" | "wrong"
    input_ids: list[int]
    prompt_len: int
    anchor_len: int
    completion_len: int
    text: str = ""
    iter_created: int = 0
    n_q: int = 0
    ref_mean_nll: float | None = None
    advantage: float = 0.0
    group_kind: str = ""
    group_size: int = 0

    def validate(self) -> None:
        total = self.prompt_len + self.anchor_len + self.completion_len
        if total != len(self.input_ids):
            raise ValueError(
                f"{self.uid}: region lengths {total} != len(input_ids) {len(self.input_ids)}"
            )
        if self.source in ("solved", "negative", "wrong") and self.anchor_len != 0:
            raise ValueError(f"{self.uid}: {self.source} example with anchor_len != 0")


@dataclass
class DPOExample(Record):
    """Anchor-conditioned preference pair: prompt = question prompt ids + anchor ids;
    chosen = improved continuation; rejected = failed continuation after the SAME anchor."""

    uid: str
    qid: str
    prompt_token_ids: list[int]
    chosen_token_ids: list[int]
    rejected_token_ids: list[int]
    chosen_text: str = ""
    rejected_text: str = ""
    iter_created: int = 0


def load_any(path: str | Path) -> Iterator[dict]:
    """Raw dict loader for stages that only need a few fields."""
    yield from read_jsonl(path)
