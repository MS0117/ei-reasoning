"""Step splitting + boundary-selection math (steps.py). Pure CPU, fake tokenizer."""

from expert_iter.steps import select_boundary, step_token_bounds


class FakeTok:
    """id -> fixed decoded piece; ids are indices into the piece list."""

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def batch_decode(self, batches):
        return [self.pieces[b[0]] for b in batches]

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def _bounds(pieces, **kw):
    return step_token_bounds(list(range(len(pieces))), FakeTok(pieces), **kw)


def test_delimiter_inside_one_piece():
    assert _bounds(["Step1", ".\n\n", "Step2", ".\n\n", "Step3", "."]) == [2, 4, 6]


def test_delimiter_split_across_pieces():
    assert _bounds(["A", ".\n", "\nB", ".\n", "\nC", "."]) == [3, 5, 6]


def test_blank_line_runs_coalesce():
    # the whitespace-only middle segment merges into its neighbor
    assert _bounds(["A.", "\n\n", "\n\n", "B."]) == [2, 4]


def test_trailing_whitespace_attaches_to_last_step():
    assert _bounds(["A.", "\n\n", "B.", "\n\n"]) == [2, 4]


def test_sentence_fallback_when_too_few_primary_steps():
    assert _bounds(["Hello", ". ", "world", ". ", "bye", "."]) == [2, 4, 6]


def test_no_delimiter_yields_single_step():
    assert _bounds(["onlyone"]) == [1]


def test_empty_input():
    assert _bounds([]) == []


def test_final_bound_is_len():
    for pieces in (["a", "b\n\nc", "d"], ["x."], ["p", ". ", "q"]):
        b = _bounds(pieces)
        assert b[-1] == len(pieces)
        assert b == sorted(set(b))


# ---- select_boundary --------------------------------------------------------

def test_threshold_crossing_spike():
    j_a, meta = select_boundary([0.0, 0.0, 0.0, 5.0, 0.0],
                                c_sigma=1.0, min_steps=2, max_step=5)
    # mu=1, sigma=2, thr=3 -> j*=4 -> j_a=3
    assert j_a == 3
    assert meta["threshold_crossed"] is True and meta["j_star"] == 4


def test_fallback_discrete_argmax_when_nothing_crosses():
    j_a, meta = select_boundary([1.0, 1.0, 4.0, 4.0, 4.0],
                                c_sigma=2.0, min_steps=2, max_step=5)
    # biggest jump at j=3 -> j_a=2
    assert j_a == 2
    assert meta["threshold_crossed"] is False and meta["j_star"] == 3


def test_clamps_up_to_min_steps():
    j_a, meta = select_boundary([5.0, 0.0, 0.0, 0.0, 0.0],
                                c_sigma=1.0, min_steps=2, max_step=5)
    assert meta["j_anchor_raw"] == 0 and j_a == 2


def test_clamps_down_to_max_step():
    j_a, _ = select_boundary([0.0, 0.0, 0.0, 5.0, 0.0],
                             c_sigma=1.0, min_steps=2, max_step=2)
    assert j_a == 2


def test_too_few_steps_returns_none():
    j_a, meta = select_boundary([1.0, 2.0], c_sigma=2.0, min_steps=2, max_step=5)
    assert j_a is None and meta["reason"] == "too_few_steps"


def test_empty_clamp_window_returns_none():
    j_a, meta = select_boundary([1.0, 2.0, 3.0], c_sigma=2.0, min_steps=2, max_step=1)
    assert j_a is None and meta["reason"] == "no_valid_window"
