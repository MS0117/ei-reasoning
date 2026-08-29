from expert_iter.records import QuestionRecord
from expert_iter.verifier import MathVerifier, _ensure_boxed


def q(answer: str) -> QuestionRecord:
    return QuestionRecord(qid="q", question="?", final_answer=answer)


def test_boxed_answer_correct():
    v = MathVerifier()
    assert v.verify(q("42"), r"So the answer is \boxed{42}.").correct


def test_wrong_answer():
    v = MathVerifier()
    assert not v.verify(q("42"), r"The answer is \boxed{41}.").correct


def test_fraction_equivalence():
    v = MathVerifier()
    assert v.verify(q("1/2"), r"\boxed{\frac{1}{2}}").correct


def test_gold_parsable_check():
    v = MathVerifier()
    assert v.gold_parsable("3/4")
    assert v.gold_parsable(r"\boxed{x^2+1}")


def test_ensure_boxed_idempotent():
    assert _ensure_boxed(r"\boxed{3}") == r"\boxed{3}"
    assert _ensure_boxed("3") == r"\boxed{3}"


def test_garbage_response_not_correct():
    v = MathVerifier()
    assert not v.verify(q("42"), "I have no idea!!! ####").correct


def test_timeout_configurable_grading_unchanged():
    v = MathVerifier(timeout_seconds=1)
    assert v.verify(q("42"), r"\boxed{42}").correct
    assert not v.verify(q("42"), r"\boxed{41}").correct


# ---- parallel verify_batch (mixin on the math verifiers) --------------------

def test_parallel_verify_batch_matches_sequential(monkeypatch):
    """Force the pool path with a tiny threshold: results and order must be
    identical to the sequential loop (also the measured property on real
    rollouts — see the mixin's block comment)."""
    from expert_iter.records import QuestionRecord
    from expert_iter.verifier import MathVerifier, StrictMathVerifier, _ParallelVerifyBatch

    monkeypatch.setattr(_ParallelVerifyBatch, "pool_min_items", 2)
    monkeypatch.setattr(_ParallelVerifyBatch, "pool_processes", 2)
    qs = [QuestionRecord(qid=f"q{i}", question="?", final_answer=str(i)) for i in range(9)]
    items = [(q, rf"steps \boxed{{{i if i % 2 else i + 1}}}") for i, q in enumerate(qs)]
    for cls in (MathVerifier, StrictMathVerifier):
        v = cls()
        seq = [v.verify(q, r) for q, r in items]
        par = v.verify_batch(items)
        assert [x.correct for x in par] == [x.correct for x in seq], cls.__name__
        assert [x.extracted_answer for x in par] == [x.extracted_answer for x in seq]


def test_parallel_verify_batch_falls_back_on_pool_failure(monkeypatch):
    import multiprocessing as mp

    from expert_iter.records import QuestionRecord
    from expert_iter.verifier import MathVerifier, _ParallelVerifyBatch

    monkeypatch.setattr(_ParallelVerifyBatch, "pool_min_items", 1)

    def boom(*a, **k):
        raise RuntimeError("no pool for you")

    monkeypatch.setattr(mp, "Pool", boom)
    v = MathVerifier()
    q = QuestionRecord(qid="q", question="?", final_answer="4")
    out = v.verify_batch([(q, r"\boxed{4}"), (q, r"\boxed{5}")])
    assert [x.correct for x in out] == [True, False]
