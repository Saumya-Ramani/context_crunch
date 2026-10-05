"""Tests for the splitter.

The important invariant is that no piece ever exceeds ``max_chars``. That is
what keeps a state inside the Laya window, so it is checked against randomly
generated text rather than only the neat examples.
"""

from __future__ import annotations

import json
import random
import string

import pytest

from contextcrunch.core.splitter import CLIP_MARKER, Piece, clip_head_tail, split_text

TARGET = 200
MAX = 300


def test_short_text_is_a_single_piece() -> None:
    """Nothing to gain from splitting something that already fits."""
    pieces = split_text("a short note", TARGET, MAX)
    assert pieces == [Piece(anchor="part 1", text="a short note")]


def test_empty_text_yields_no_pieces() -> None:
    """Whitespace only is not content."""
    assert split_text("", TARGET, MAX) == []
    assert split_text("   \n\n  ", TARGET, MAX) == []


def test_paragraphs_merge_up_to_the_target() -> None:
    """Small paragraphs are merged so a piece is a useful amount of context."""
    text = "\n\n".join(f"paragraph number {n} with a few words" for n in range(6))
    pieces = split_text(text, TARGET, MAX)

    assert len(pieces) > 1
    assert all(len(piece.text) <= MAX for piece in pieces)
    assert [p.anchor for p in pieces] == [f"part {n}" for n in range(1, len(pieces) + 1)]


def test_anchors_are_numbered_from_one() -> None:
    """Anchors say which part of the original a decision was about."""
    text = "\n\n".join("x" * 90 for _ in range(8))
    pieces = split_text(text, TARGET, MAX)
    assert pieces[0].anchor == "part 1"
    assert pieces[-1].anchor == f"part {len(pieces)}"


def test_a_long_paragraph_is_split_on_line_boundaries() -> None:
    """A big block keeps its lines whole where they fit."""
    line = "x" * 90
    text = "\n".join(line for _ in range(20))
    pieces = split_text(text, TARGET, MAX)

    assert len(pieces) > 1
    assert all(len(piece.text) <= MAX for piece in pieces)
    for piece in pieces:
        assert all(len(row) == 90 for row in piece.text.splitlines())


def test_a_long_line_is_cut_at_max_chars() -> None:
    """There is nothing finer than a character to split one long line on."""
    pieces = split_text("y" * 1000, TARGET, MAX)
    assert all(len(piece.text) <= MAX for piece in pieces)
    assert "".join(piece.text for piece in pieces) == "y" * 1000


def test_no_content_is_lost_when_splitting() -> None:
    """Splitting must not discard text: the pieces have to add back up.

    Whitespace at the split boundaries is allowed to differ, because paragraphs
    are stripped and rejoined, but every visible character must survive.
    """
    text = "\n\n".join(f"content block {n} with words" for n in range(12))
    joined = "".join(piece.text for piece in split_text(text, TARGET, MAX))
    assert "".join(joined.split()) == "".join(text.split())


@pytest.mark.parametrize("seed", range(12))
def test_no_piece_ever_exceeds_max_chars(seed: int) -> None:
    """Property check: random text, random sizes, the bound always holds."""
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + " \n" * 20
    text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 4000)))
    target = rng.randint(50, 400)
    cap = rng.randint(target, 900)

    for piece in split_text(text, target, cap):
        assert len(piece.text) <= cap, f"seed={seed} cap={cap}"


def test_json_array_becomes_one_piece_per_element() -> None:
    """A ledger is judged invoice by invoice, which is the whole point."""
    data = [{"id": n, "amount": n * 10} for n in range(5)]
    pieces = split_text(json.dumps(data), TARGET, MAX)

    assert [piece.anchor for piece in pieces] == [f"item {n}" for n in range(1, 6)]
    for piece in pieces:
        assert json.loads(piece.text) in data


def test_long_json_element_is_cut() -> None:
    """One huge element must not become one huge piece."""
    data = [{"id": 1, "note": "z" * 2000}, {"id": 2, "note": "small"}]
    pieces = split_text(json.dumps(data), TARGET, MAX)

    assert all(len(piece.text) <= MAX for piece in pieces)


def test_broken_json_falls_back_to_text_splitting() -> None:
    """Text that starts like JSON but is not must still split cleanly."""
    text = "[not really json\n\nsecond paragraph here"
    pieces = split_text(text, TARGET, MAX)

    assert pieces
    assert all(piece.anchor.startswith("part") for piece in pieces)


def test_json_object_is_not_treated_as_an_array() -> None:
    """Only an array splits per element; an object is ordinary text."""
    pieces = split_text(json.dumps({"a": 1, "b": 2}), TARGET, MAX)
    assert all(piece.anchor.startswith("part") for piece in pieces)


def test_max_chars_must_be_positive() -> None:
    """A zero or negative bound is a configuration bug, not something to guess at."""
    with pytest.raises(ValueError):
        split_text("some text", TARGET, 0)


# --------------------------------------------------------------------------- #
# clip_head_tail
# --------------------------------------------------------------------------- #


def test_clip_keeps_the_start_and_the_end() -> None:
    """A tool output carries its identity in the head and the tail."""
    text = "HEAD" + "x" * 500 + "TAIL"
    clipped = clip_head_tail(text, 100, 50)

    assert clipped.startswith("HEAD")
    assert clipped.endswith("TAIL")
    assert CLIP_MARKER in clipped


def test_clip_returns_short_text_unchanged() -> None:
    """Never add noise to something that did not need shortening."""
    assert clip_head_tail("short", 100, 50) == "short"


def test_clip_at_the_boundary_is_a_no_op() -> None:
    """Exactly head+tail characters is already short enough."""
    assert clip_head_tail("x" * 150, 100, 50) == "x" * 150


def test_clip_with_zero_tail_keeps_only_the_head() -> None:
    """A caller that wants no tail still gets the head."""
    clipped = clip_head_tail("HEAD" + "x" * 500, 100, 0)
    assert clipped.startswith("HEAD")
    assert not clipped.endswith("x" * 10)


def test_clip_rejects_negative_sizes() -> None:
    """Negative sizes are a bug and would silently slice from the end."""
    with pytest.raises(ValueError):
        clip_head_tail("text", -1, 10)