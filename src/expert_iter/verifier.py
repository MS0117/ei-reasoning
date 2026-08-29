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


# ---------------------------------------------------------------------------
# Parallel verify_batch for the math verifiers.
#
# math_verify is pure-Python sympy work, so threads are useless (GIL) and the
# per-sample cost is dominated by a few pathological expressions that grind
# for seconds. MEASURED on 300 real rollout samples (128-core srv04): 39.9s
# sequential -> 6.5s with 8 processes, and NO further gain at 32/64 processes
# (7.7x is the ceiling because the slow outliers serialize) — hence Pool(8),
# not Pool(cpu_count). Results were element-identical to sequential.
#
# The worker builds its verifier once (initializer), items travel as plain
# picklable dataclasses, and Pool.map preserves order. Any pool failure falls
# back to the sequential path so grading can never be lost to an mp quirk.
# ---------------------------------------------------------------------------

_POOL_VERIFIER: Verifier | None = None


def _pool_init(verifier_name: str) -> None:
    global _POOL_VERIFIER
    from .registry import build

    _POOL_VERIFIER = build(VERIFIERS, verifier_name)


def _pool_verify(item: tuple[QuestionRecord, str]) -> Verdict:
    q, response_text = item
    return _POOL_VERIFIER.verify(q, response_text)


class _ParallelVerifyBatch:
    """Mixin: verify_batch via a process pool, sequential below the threshold."""

    pool_processes = 8      # measured saturation point (see block comment above)
    pool_min_items = 64     # below this, pool spawn overhead beats the speedup

    def verify_batch(self, items: list[tuple[QuestionRecord, str]]) -> list[Verdict]:
        if len(items) < self.pool_min_items:
            return [self.verify(q, r) for q, r in items]
        import multiprocessing as mp

        try:
            with mp.Pool(self.pool_processes, initializer=_pool_init,
                         initargs=(self.name,)) as pool:
                return pool.map(_pool_verify, items, chunksize=4)
        except Exception as e:
            print(f"[verifier] parallel verify_batch failed ({type(e).__name__}: {e}); "
                  "falling back to sequential", flush=True)
            return [self.verify(q, r) for q, r in items]


@register(VERIFIERS, "math")
class MathVerifier(_ParallelVerifyBatch, Verifier):
    """math-verify based grading.

    Gold: parsed from the question's final_answer (already normalized by the
    dataset adapter). Pred: parsed from the response text — math_verify.parse
    prefers the last \\boxed{...}, falling back to the last expression.
    """

    def __init__(self, timeout_seconds: int = 5) -> None:
        # Import here so merely constructing other verifiers never needs math_verify.
        from math_verify import parse, verify

        self._parse = parse
        self._verify = verify
        self._timeout = timeout_seconds

    def verify(self, question: QuestionRecord, response_text: str) -> Verdict:
        try:
            gold = self._parse(_ensure_boxed(question.final_answer))
            if not gold:
                return Verdict(False, meta={"error": "gold_unparsable"})
            pred = self._parse(response_text)
            if not pred:
                return Verdict(False, meta={"error": "pred_unparsable"})
            # timeout: sympy can hang on pathological latex (same guard as strict).
            ok = bool(self._verify(gold, pred, timeout_seconds=self._timeout))
            return Verdict(ok, extracted_answer=_repr_short(pred))
        except Exception as e:  # math-verify can raise on pathological latex
            return Verdict(False, meta={"error": f"{type(e).__name__}: {e}"})

    def gold_parsable(self, final_answer: str) -> bool:
        """Pre-verifiability check used by dataset adapters."""
        try:
            return bool(self._parse(_ensure_boxed(final_answer)))
        except Exception:
            return False


@register(VERIFIERS, "math_strict")
class StrictMathVerifier(_ParallelVerifyBatch, Verifier):
    """Benchmark-grade grading, semantics ported from OPSD's evaluate_math.py
    so external-benchmark numbers are comparable to published evals:

    1. Extract the LAST \\boxed{...} from the response (balanced-brace scan).
       No box => wrong AND meta.formatted=False (the format-rate metric).
    2. math_verify with fallback_mode="no_fallback" (prose must not parse as
       math) and a verify timeout (sympy can hang on pathological latex).
    3. On parse/verify exception: normalized string equality as a last resort
       (rescues plain answers like "42" that sympy chokes on).

    extracted_answer is the RAW boxed string — majority voting groups by it.
    """

    def __init__(self, timeout_seconds: int = 5) -> None:
        from math_verify import parse, verify

        self._parse = parse
        self._verify = verify
        self._timeout = timeout_seconds

    def verify(self, question: QuestionRecord, response_text: str) -> Verdict:
        pred_str = last_boxed(response_text)
        if pred_str is None:
            return Verdict(False, meta={"formatted": False, "error": "no_boxed_answer"})
        ok = self.grade_extracted(pred_str, question.final_answer)
        return Verdict(ok, extracted_answer=pred_str, meta={"formatted": True})

    def grade_extracted(self, pred_str: str, gold_str: str) -> bool:
        """Grade an already-extracted answer string (also used for majority vote)."""
        try:
            pred = self._parse(_dollar_wrap(pred_str), fallback_mode="no_fallback")
            gold = self._parse(_dollar_wrap(gold_str), fallback_mode="no_fallback")
            return bool(self._verify(gold, pred, timeout_seconds=self._timeout))
        except Exception:
            return _normalize_str(pred_str) == _normalize_str(gold_str)


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


def last_boxed(text: str) -> str | None:
    r"""Content of the last \boxed{...} in text (balanced-brace scan), or None.
    The single source of truth for boxed extraction — data.py gold extraction
    and strict grading both use it."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None  # unbalanced braces


def _dollar_wrap(s: str) -> str:
    """math_verify parses latex most reliably inside $...$; leave already-
    delimited strings alone."""
    s = s.strip()
    return s if "$" in s else f"${s}$"


def _normalize_str(s: str) -> str:
    return s.replace("$", "").replace(" ", "").lower().strip()


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
