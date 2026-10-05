"""Tests for the tombstone helper."""

from __future__ import annotations

from contextcrunch.core.messages import Message
from contextcrunch.core.tombstone import (
    TOMBSTONE_PREFIX,
    is_tombstone,
    tombstone_message,
)


def test_keeps_role_and_tool_call_id() -> None:
    """The pairing between a tool call and its result must survive a drop."""
    original = Message(
        role="tool", name="read_file", tool_call_id="call_42", content="a" * 5000
    )
    stone = tombstone_message(original)

    assert stone.role == "tool"
    assert stone.name == "read_file"
    assert stone.tool_call_id == "call_42"


def test_keeps_the_tool_calls_that_point_at_this_message() -> None:
    """An assistant message must keep its tool_calls, or the history breaks."""
    original = Message(
        role="assistant",
        tool_call_id="call_1",
        content="ignored",
        tool_calls=[{"id": "call_1", "function": {"name": "read_file"}}],
    )
    assert tombstone_message(original).tool_calls == original.tool_calls


def test_content_names_the_tool_and_the_reason() -> None:
    """A tombstone must say what went and where it can be recovered."""
    original = Message(role="tool", name="read_file", tool_call_id="c1", content="x" * 100)
    stone = tombstone_message(original)

    assert stone.content == (
        f"[{TOMBSTONE_PREFIX}: read_file. judged no longer relevant. Recoverable via sidecar.]"
    )
    assert len(stone.content) < 100, "a tombstone must be far shorter than the original"


def test_reason_can_be_overridden() -> None:
    """A caller can say why something went, which is useful in the side-car."""
    original = Message(role="tool", name="grep", tool_call_id="c1", content="x")
    stone = tombstone_message(original, reason="superseded by a later run")
    assert "superseded by a later run" in stone.content


def test_falls_back_to_item_when_there_is_no_tool_name() -> None:
    """A message with no tool still gets a readable tombstone."""
    stone = tombstone_message(Message(role="tool", tool_call_id="c1", content="x"))
    assert f"{TOMBSTONE_PREFIX}: item." in stone.content


def test_does_not_mutate_the_original() -> None:
    """The caller's history must be untouched: fail-open depends on it."""
    original = Message(role="tool", name="read_file", tool_call_id="c1", content="original")
    tombstone_message(original)
    assert original.content == "original"


def test_is_tombstone_round_trips() -> None:
    """Detection must work, or a restore could not tell a tombstone from content."""
    original = Message(role="tool", name="read_file", tool_call_id="c1", content="x" * 100)
    assert is_tombstone(tombstone_message(original)) is True
    assert is_tombstone(original) is False


def test_plain_ascii_only() -> None:
    """Tombstones land in logs and consoles, so no smart quotes or dashes."""
    original = Message(role="tool", name="read_file", tool_call_id="c1", content="x")
    assert tombstone_message(original).content.isascii()