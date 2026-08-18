"""cvar() semantics (filters.py) — the tail operator behind S_tail / D_tail."""

import pytest

from expert_iter.filters import cvar


def test_ceil_tail_semantics():
    # ceil(0.1 * 10) = 1 -> just the worst value
    assert cvar(list(map(float, range(1, 11))), 0.1) == 10.0


def test_tail_of_more_than_one():
    # ceil(0.5 * 3) = 2 -> mean of the two largest
    assert cvar([1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)


def test_single_value():
    assert cvar([5.0], 0.5) == 5.0


def test_full_tail_is_mean():
    assert cvar([2.0, 4.0, 6.0], 1.0) == pytest.approx(4.0)


def test_ties():
    assert cvar([2.0, 2.0, 2.0], 0.34) == 2.0


def test_empty_raises():
    with pytest.raises(ValueError):
        cvar([], 0.1)
