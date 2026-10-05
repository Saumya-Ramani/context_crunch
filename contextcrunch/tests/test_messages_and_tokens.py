"""Tests for the message model and the token helpers."""

from __future__ import annotations

from contextcrunch.core.messages import Message, as_messages, to_dicts
from contextcrunch.core.tokens import clip_to_chars, clip_to_tokens, count_tokens


def test_as_messages_skips_malformed_entries() -> None:
    """A bad entry must not stop a real history from loading."""
    messages = as_messages(
        [
            {"role": "user", "content": "hello"},
            {"no_role": True},
            "not-a-dict",
            {"role": "tool", "content": None},
        ]
    )
    assert [m.role for m in messages] == ["user", "tool"]
    assert messages[1].content == ""


def test_extra_keys_survive_the_round_trip() -> None:
    """ContextCrunch must not strip keys it does not understand."""
    raw = [{"role": "user", "content": "hi", "custom_flag": 7}]
    assert to_dicts(as_messages(raw))[0]["custom_flag"] == 7


def test_label_describes_the_source() -> None:
    """Labels appear in tombstones and the side-car."""
    assert Message(role="tool", content="x", name="read_file").label() == "tool:read_file"
    assert Message(role="assistant", content="x").label() == "assistant"


def test_count_tokens_grows_with_text() -> None:
    """The counter must actually count."""
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0
    assert count_tokens("hello world " * 10) > count_tokens("hello world")


def test_clip_helpers_respect_their_limits() -> None:
    """Clipping must never exceed the budget."""
    assert clip_to_chars("abcdef", 3) == "abc"
    assert clip_to_chars("abc", 10) == "abc"
    assert clip_to_tokens("hello world " * 20, 5).count(" ") < 20
    assert clip_to_tokens("abc", 0) == ""
