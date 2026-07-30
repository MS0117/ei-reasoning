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
