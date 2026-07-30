"""Pluggable correctness verifiers.

`math` (default) grades free-form responses against a gold answer string with
math-verify. `lean` is a stub for the kimina Lean server (optional install —
imported lazily so math-only machines never touch it).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .records import QuestionRecord
from .registry import VERIFIERS, register


@dataclass
class Verdict:
    correct: bool
    extracted_answer: str | None = None
    meta: dict = field(default_factory=dict)


class Verifier(ABC):
    name: str

    @abstractmethod
    def verify(self, question: QuestionRecord, response_text: str) -> Verdict:
        ...

    def verify_batch(self, items: list[tuple[QuestionRecord, str]]) -> list[Verdict]:
        """Default: sequential. Verifiers with batch endpoints override this."""
        return [self.verify(q, r) for q, r in items]


@register(VERIFIERS, "math")
class MathVerifier(Verifier):
    """math-verify based grading.

    Gold: parsed from the question's final_answer (already normalized by the
    dataset adapter). Pred: parsed from the response text — math_verify.parse
    prefers the last \\boxed{...}, falling back to the last expression.
    """

    def __init__(self) -> None:
        # Import here so merely constructing other verifiers never needs math_verify.
        from math_verify import parse, verify

        self._parse = parse
        self._verify = verify

    def verify(self, question: QuestionRecord, response_text: str) -> Verdict:
        try:
            gold = self._parse(_ensure_boxed(question.final_answer))
            if not gold:
                return Verdict(False, meta={"error": "gold_unparsable"})
            pred = self._parse(response_text)
            if not pred:
                return Verdict(False, meta={"error": "pred_unparsable"})
            ok = bool(self._verify(gold, pred))
            return Verdict(ok, extracted_answer=_repr_short(pred))
        except Exception as e:  # math-verify can raise on pathological latex
            return Verdict(False, meta={"error": f"{type(e).__name__}: {e}"})

    def gold_parsable(self, final_answer: str) -> bool:
        """Pre-verifiability check used by dataset adapters."""
        try:
            return bool(self._parse(_ensure_boxed(final_answer)))
        except Exception:
            return False


@register(VERIFIERS, "lean")
class LeanVerifier(Verifier):
    """Kimina lean-server client. Requires the optional Lean stack
    (scripts/setup.sh WITHOUT --skip-lean) and a running server."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 120.0) -> None:
        import httpx  # lazy: only needed when Lean verification is selected

        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def verify(self, question: QuestionRecord, response_text: str) -> Verdict:
        return self.verify_batch([(question, response_text)])[0]

    def verify_batch(self, items: list[tuple[QuestionRecord, str]]) -> list[Verdict]:
        codes = [
            {"custom_id": f"{i}", "code": _extract_lean_code(resp)}
            for i, (_, resp) in enumerate(items)
        ]
        r = self._client.post("/verify", json={"codes": codes})
        r.raise_for_status()
        by_id = {res["custom_id"]: res for res in r.json().get("results", [])}
        verdicts = []
        for i in range(len(items)):
            res = by_id.get(str(i), {})
            # Empty diagnostics == proof closed (kimina convention).
            errors = [m for m in res.get("messages", []) if m.get("severity") == "error"]
            verdicts.append(Verdict(correct=bool(res) and not errors, meta={"n_errors": len(errors)}))
        return verdicts


def _ensure_boxed(answer: str) -> str:
    r"""math_verify.parse extracts most reliably from \boxed{...}; wrap bare
    gold answers so short forms like `3/4` parse as math, not prose."""
    a = answer.strip()
    if "\\boxed" in a:
        return a
    return f"\\boxed{{{a}}}"


def _repr_short(parsed, limit: int = 200) -> str:
    try:
        s = repr(parsed[0]) if isinstance(parsed, list) and parsed else repr(parsed)
    except Exception:
        s = "<unrepr>"
    return s[:limit]


def _extract_lean_code(response_text: str) -> str:
    """Pull the last ```lean fenced block, else the raw text."""
    marker = "```lean"
    if marker in response_text:
        block = response_text.rsplit(marker, 1)[1]
        return block.split("```", 1)[0].strip()
    return response_text.strip()
