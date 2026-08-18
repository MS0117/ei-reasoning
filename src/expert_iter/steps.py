"""Step segmentation of a generated response + anchor-boundary selection math.

Step boundaries are TOKEN indices derived from per-token decoded pieces —
never from re-tokenizing decoded text — so an anchor built as
`token_ids[:bounds[j]]` stays an exact id-slice of the original rollout
(the pipeline's splicing invariant, see templates.py).

Convention: a delimiter "completes" inside some token; the boundary goes AFTER
that token. A token like ".\n\nNext" therefore carries the next step's first
characters inside the previous step — an accepted approximation (the anchor is
an exact id-slice either way; only the step ATTRIBUTION of a few characters is
approximate).
"""

from __future__ import annotations

import re
from statistics import pstdev

_SENT_END = (".", "!", "?")


def step_token_bounds(
    token_ids: list[int],
    tokenizer,
    *,
    primary: str = "\n\n",
    fallback_min_steps: int = 3,
) -> list[int]:
    """END indices (exclusive) b_1 < ... < b_J == len(token_ids) delimiting steps
    u_j = token_ids[b_{j-1}:b_j].

    Splits on `primary` ("\\n\\n" blank lines); if that yields fewer than
    `fallback_min_steps` steps, rescans on sentence boundaries. Whitespace-only
    segments are merged into the preceding step (runs of blank lines never
    create empty steps). Returns [] for empty input and [len] when no
    delimiter exists (J=1 — callers should degenerate to "no anchor").
    """
    if not token_ids:
        return []
    pieces = tokenizer.batch_decode([[tid] for tid in token_ids])
    bounds = _bounds_from_pieces(pieces, primary=primary)
    if len(bounds) < fallback_min_steps:
        sentence = _sentence_bounds(pieces)
        if len(sentence) > len(bounds):
            bounds = sentence
    return bounds


def _bounds_from_pieces(pieces: list[str], *, primary: str) -> list[int]:
    n = len(pieces)
    cand: list[int] = []
    for i, piece in enumerate(pieces):
        if primary in piece:
            cand.append(i + 1)
        elif i > 0 and pieces[i - 1].endswith(primary[0]) and piece.startswith(primary[-1]):
            # delimiter split across two tokens ("...\n" + "\n...") completes here
            cand.append(i + 1)
    return _finalize_bounds(cand, pieces)


def _sentence_bounds(pieces: list[str]) -> list[int]:
    """Fallback: boundary after a token that ends a sentence. Either the
    punctuation+whitespace pair completes inside one piece, or the piece ends
    with sentence punctuation and the next piece starts with whitespace."""
    n = len(pieces)
    cand: list[int] = []
    for i, piece in enumerate(pieces):
        if re.search(r"[.!?][\"')\]]*\s", piece):
            cand.append(i + 1)
        elif (piece.rstrip('"\')]').endswith(_SENT_END)
                and i + 1 < n and pieces[i + 1][:1] in (" ", "\n", "\t")):
            cand.append(i + 1)
    return _finalize_bounds(cand, pieces)


def _finalize_bounds(cand: list[int], pieces: list[str]) -> list[int]:
    """Dedup/sort candidates, drop boundaries that would close a whitespace-only
    segment (merging it into the previous step), and terminate with len."""
    n = len(pieces)
    bounds: list[int] = []
    prev = 0
    for b in sorted(set(cand)):
        if b <= prev or b >= n:
            continue
        if "".join(pieces[prev:b]).strip() == "":
            continue  # whitespace-only segment: merge into the neighbor
        bounds.append(b)
        prev = b
    if bounds and "".join(pieces[bounds[-1]:]).strip() == "":
        bounds.pop()  # trailing whitespace-only segment attaches to the last real step
    bounds.append(n)
    return bounds


def select_boundary(
    g_bar: list[float],
    *,
    c_sigma: float,
    min_steps: int,
    max_step: int,
) -> tuple[int | None, dict]:
    """Anchor step count j_a from per-step divergence means (methodology §anchor).

    j* = min{j : g_bar_j > mu + c_sigma * sigma} (1-based, population sigma over
    the searched steps); if nothing crosses, j* = argmax_{j>=2}(g_bar_j -
    g_bar_{j-1}). j_a = j* - 1 clamped into [min_steps, max_step]. Returns
    (None, meta) when the window is degenerate (fewer than 3 searched steps or
    an empty clamp window).
    """
    meta: dict = {
        "n_searched_steps": len(g_bar),
        "min_steps": min_steps,
        "max_step": max_step,
    }
    if len(g_bar) < 3 or max_step < min_steps:
        meta["reason"] = "too_few_steps" if len(g_bar) < 3 else "no_valid_window"
        return None, meta
    mu = sum(g_bar) / len(g_bar)
    sigma = pstdev(g_bar)
    threshold = mu + c_sigma * sigma
    j_star = next((j for j, g in enumerate(g_bar, start=1) if g > threshold), None)
    crossed = j_star is not None
    if j_star is None:
        deltas = [g_bar[j - 1] - g_bar[j - 2] for j in range(2, len(g_bar) + 1)]
        j_star = 2 + max(range(len(deltas)), key=deltas.__getitem__)
    j_a_raw = j_star - 1
    j_a = max(min_steps, min(max_step, j_a_raw))
    meta.update(
        mu_g=mu, sigma_g=sigma, threshold=threshold, threshold_crossed=crossed,
        j_star=j_star, j_anchor_raw=j_a_raw, j_anchor=j_a,
    )
    return j_a, meta
