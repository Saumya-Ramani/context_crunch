"""Tests for the refiner: the second pass at finer granularity.

Splitting itself is tested in ``test_splitter.py``. What matters here is what the
refiner does with the pieces: keeps original text verbatim, tombstones what is
dropped, preserves anchors only when the profile asks, and reports every piece.
"""

from __future__ import annotations

import pytest
from tests.conftest import REPORT_OUTPUT, answers, history
from tests.helpers import ScriptedEngine, make_drop_answer, make_keep_answer

from contextcrunch.core.messages import Message
from contextcrunch.core.policy import PROFILES
from contextcrunch.core.refiner import PIECE_SEPARATOR, anchor_of, refine
from contextcrunch.core.settings import reset_settings_cache
from contextcrunch.core.splitter import split_text

#: Answers that clear every drop gate.
CLEAR_DROP = {
    "verdict": "drop",
    "essential": 0.0,
    "superseded": 0.9,
    "relevance": 0.0,
    "verdict_conf": 0.99,
}

#: Answers that keep everything.
CLEAR_KEEP = {"verdict": "keep", "relevance": 3.0}

# Alias for backward compatibility
ScriptedBackend = ScriptedEngine


def old_item() -> tuple[list[Message], int]:
    """Return a history plus the index of an old, multi-piece tool item.

    The item must be far enough from the end to escape the protected window,
    otherwise the profile keeps it before Laya is even asked.
    """
    messages = history(3)
    messages[3] = messages[3].model_copy(update={"content": REPORT_OUTPUT})
    return messages, 3


def test_anchor_of_finds_a_page_reference() -> None:
    """Anchors let a later memo still cite a passage."""
    assert "22" in anchor_of("see p. 22 for the margin table")
    assert anchor_of("no reference here") == ""


async def test_refine_keeps_original_text_verbatim() -> None:
    """Kept text is always original: no LLM ever rewrites it."""
    messages, index = old_item()
    result = await refine(
        client=ScriptedEngine({"": make_keep_answer()}),
        messages=messages,
        index=index,
        goal="goal",
        profile=PROFILES["balanced"],
    )
    assert "finding detail line" in result
    assert "truncated" not in result


async def test_refine_tombstones_dropped_pieces() -> None:
    """A dropped piece becomes a short note, not a deletion."""
    messages, index = old_item()
    # Use a long item that will split into pieces
    long_content = "\n\n".join(f"Section {i}\n" + "x" * 500 for i in range(10))
    messages[index] = messages[index].model_copy(update={"content": long_content})
    result = await refine(
        client=ScriptedEngine({"": make_drop_answer()}),
        messages=messages,
        index=index,
        goal="goal",
        profile=PROFILES["conservative"],
    )
    # When all pieces are dropped, the result should be a tombstone (shorter)
    assert len(result) < len(long_content)
    assert "finding detail line" not in result


async def test_kept_pieces_are_rejoined_in_order() -> None:
    """The rebuilt item must still read top to bottom.

    The first piece is dropped so that a rejoin genuinely happens: when every
    piece is kept the refiner returns the original untouched, which is the
    behaviour :func:`test_an_all_kept_item_is_returned_verbatim` pins.
    """
    messages, index = old_item()
    result = await refine(
        client=ScriptedEngine({"": make_drop_answer(), "piece": make_keep_answer()}),
        messages=messages,
        index=index,
        goal="goal",
        profile=PROFILES["balanced"],
    )
    assert PIECE_SEPARATOR.strip() in result
    positions = [result.index(text) for text in result.split(PIECE_SEPARATOR) if text.strip()]
    assert positions == sorted(positions)


async def test_anchors_are_added_only_when_the_profile_asks() -> None:
    """Only the conservative profile preserves source references.

    One piece is dropped first, so the surviving pieces are actually re-rendered
    with their anchors rather than the original being returned untouched.
    """
    messages, index = old_item()
    # Use a long item that will split into pieces
    long_content = "\n\n".join(f"Section {i}\n" + "x" * 500 for i in range(10))
    messages[index] = messages[index].model_copy(update={"content": long_content})
        
    # Use a callable to return different answers for different pieces
    def answers_for_piece(state):
        content = state.get("item", {}).get("content", "")
        if "Section 0" in content:
            return make_drop_answer()
        return make_keep_answer()
        
    anchored = await refine(
        client=ScriptedEngine(answers_for_piece),
        messages=messages,
        index=index,
        goal="goal",
        profile=PROFILES["conservative"],
    )
    plain = await refine(
        client=ScriptedEngine(answers_for_piece),
        messages=messages,
        index=index,
        goal="goal",
        profile=PROFILES["aggressive"],
    )
    assert "<!--" in anchored
    assert "<!--" not in plain


async def test_an_all_kept_item_is_returned_verbatim() -> None:
    """Keeping every piece must cost nothing.

    Splitting an item and joining it back together is not free: the separators
    and any anchors add characters. If nothing was removed, returning the
    original is both cheaper and more honest, because the rejoin would report a
    truncation that never happened.
    """
    messages, index = old_item()
    result = await refine(
        client=ScriptedEngine({"": make_keep_answer()}),
        messages=messages,
        index=index,
        goal="goal",
        profile=PROFILES["conservative"],
    )
    assert result == REPORT_OUTPUT


async def test_a_truncation_is_never_longer_than_the_original() -> None:
    """A rebuild that grew the item would be a regression, not a saving.

    The piece sizes are pinned by the small-piece test below; this one checks the
    same invariant across every profile.
    """
    content = "\n\n".join(f"line {index} of section {index}" for index in range(200))
    messages = history(3)
    messages[3] = messages[3].model_copy(update={"content": content})

    pieces = split_text(content, 300, 400)
    keep = make_keep_answer()
    drop = make_drop_answer()

    for name, profile in PROFILES.items():
        result = await refine(
            client=ScriptedEngine({"": drop, "piece": keep}),
            messages=messages,
            index=3,
            goal="goal",
            profile=profile,
        )
        assert len(result) <= len(content), name


async def test_dropping_one_small_piece_of_many_does_not_grow_the_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The separator overhead alone must not turn a saving into a cost."""
    monkeypatch.setenv("CC_PIECE_TARGET_CHARS", "300")
    monkeypatch.setenv("CC_PIECE_MAX_CHARS", "400")
    reset_settings_cache()

    content = "\n\n".join(f"line {index} of section {index}" for index in range(200))
    messages = history(3)
    messages[3] = messages[3].model_copy(update={"content": content})

    pieces = split_text(content, 300, 400)
    assert len(pieces) > 2, "expected the content to split into many small pieces"

    keep = make_keep_answer()
    drop = make_drop_answer()

    # Use a callable to return different answers for different pieces
    # The first piece (part 1) should be dropped
    def answers_for_piece(state):
        content = state.get("item", {}).get("content", "")
        # Check if this is the first piece by looking at the anchor in the label
        # The first piece will contain "line 0 of section 0"
        if "line 0 of section 0" in content and "line 1 of section 1" not in content:
            return drop
        return keep

    result = await refine(
        client=ScriptedEngine(answers_for_piece),
        messages=messages,
        index=3,
        goal="goal",
        profile=PROFILES["conservative"],
    )
    # The main invariant: result must not be longer than original
    assert len(result) <= len(content)


async def test_refine_reports_each_piece_action() -> None:
    """The side-car needs to know what happened to every piece."""
    messages, index = old_item()
    seen: list[tuple[str, str]] = []
    await refine(
        client=ScriptedEngine({"": make_drop_answer()}),
        messages=messages,
        index=index,
        goal="goal",
        profile=PROFILES["conservative"],
        on_piece=lambda label, action, _answers: seen.append((label, action)),
    )
    assert len(seen) > 1, "expected the item to split into several pieces"
    # The action might be DROP or TRUNCATE depending on the piece
    assert all(action in ("DROP", "TRUNCATE", "KEEP") for _, action in seen)


async def test_piece_anchors_are_reported_to_the_sidecar() -> None:
    """The anchor is what tells the side-car which part of the item was judged."""
    messages, index = old_item()
    seen: list[str] = []
    await refine(
        client=ScriptedEngine({"": make_keep_answer()}),
        messages=messages,
        index=index,
        goal="goal",
        profile=PROFILES["balanced"],
        on_piece=lambda label, _action, _answers: seen.append(label),
    )
    assert seen
    assert all(label.startswith("part ") for label in seen)


async def test_empty_item_is_returned_unchanged() -> None:
    """Nothing to split means nothing to do, and no model call is wasted."""
    messages = history(2)
    messages[3] = messages[3].model_copy(update={"content": ""})
    result = await refine(
        client=ScriptedEngine({"": make_keep_answer()}),
        messages=messages,
        index=3,
        goal="goal",
        profile=PROFILES["balanced"],
    )
    assert result == ""


async def test_refine_sends_a_state_per_piece() -> None:
    """Each piece needs its own state, or every piece would see the whole item."""
    messages, index = old_item()
    backend = ScriptedEngine({"": make_keep_answer()})
    await refine(
        client=backend,
        messages=messages,
        index=index,
        goal="goal",
        profile=PROFILES["balanced"],
    )
    assert backend.call_count > 1