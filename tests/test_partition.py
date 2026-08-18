from expert_iter.config import Config
from expert_iter.partition import _partition_rows
from expert_iter.records import QuestionRecord, RolloutSample, VerdictRecord


def _sample(finish_reason, sample_idx=0, qid="q"):
    return RolloutSample(
        qid=qid, sample_idx=sample_idx, prompt_text="p", prompt_token_ids=[1],
        response_text="\\boxed{1}", response_token_ids=[2],
        finish_reason=finish_reason,
    )


def test_truncated_correct_sample_is_classified_as_unsolved():
    cfg = Config.load(None)
    questions = {"q": QuestionRecord(qid="q", question="1?", final_answer="1")}
    verdicts = [VerdictRecord(qid="q", sample_idx=0, correct=True)]

    solved, unsolved, n_solved, n_questions, n_clean, n_truncated, n_overlap = _partition_rows(
        questions, [_sample("length")], verdicts, cfg, 0,
    )

    assert solved == []
    assert n_solved == 0 and n_questions == 1 and n_clean == 0
    assert n_truncated == 1 and n_overlap == 0
    assert unsolved[0].failed_sample_idxs == [0]


def test_clean_correct_sample_is_solved():
    cfg = Config.load(None)
    questions = {"q": QuestionRecord(qid="q", question="1?", final_answer="1")}
    verdicts = [VerdictRecord(qid="q", sample_idx=0, correct=True)]

    solved, unsolved, n_solved, _, n_clean, n_truncated, n_overlap = _partition_rows(
        questions, [_sample("stop")], verdicts, cfg, 2,
    )

    assert len(solved) == 1 and solved[0].iter == 2
    assert unsolved == []
    assert n_solved == 1 and n_clean == 1 and n_truncated == 0 and n_overlap == 0


def test_cliff_threshold_routes_partially_solved_to_both():
    """cliff_max_correct=1: a question with exactly 1 clean-correct sample keeps
    its solved trajectory AND is improvement-eligible (overlap)."""
    cfg = Config.load(None, overrides=["partition.cliff_max_correct=1"])
    questions = {"q": QuestionRecord(qid="q", question="1?", final_answer="1")}
    samples = [_sample("stop", 0), _sample("stop", 1)]
    verdicts = [
        VerdictRecord(qid="q", sample_idx=0, correct=True),
        VerdictRecord(qid="q", sample_idx=1, correct=False),
    ]

    solved, unsolved, n_solved, n_questions, _, _, n_overlap = _partition_rows(
        questions, samples, verdicts, cfg, 0,
    )

    assert n_solved == 1 and len(solved) == 1
    assert len(unsolved) == 1 and unsolved[0].failed_sample_idxs == [1]
    assert n_overlap == 1 and n_questions == 1


def test_cliff_threshold_excludes_well_solved_questions():
    """cliff_max_correct=1: a question with 2 clean-correct samples is solved
    only — no UnsolvedQuestion row."""
    cfg = Config.load(None, overrides=["partition.cliff_max_correct=1"])
    questions = {"q": QuestionRecord(qid="q", question="1?", final_answer="1")}
    samples = [_sample("stop", 0), _sample("stop", 1)]
    verdicts = [
        VerdictRecord(qid="q", sample_idx=0, correct=True),
        VerdictRecord(qid="q", sample_idx=1, correct=True),
    ]

    solved, unsolved, n_solved, _, _, _, n_overlap = _partition_rows(
        questions, samples, verdicts, cfg, 0,
    )

    assert n_solved == 1 and len(solved) == 2
    assert unsolved == [] and n_overlap == 0
