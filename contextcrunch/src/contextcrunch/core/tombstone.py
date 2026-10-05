"""Tombstones: what a dropped message turns into.

A tombstone keeps the original ``role``, ``tool_name`` and ``tool_call_id``. That
is the whole point. An assistant message carries a ``tool_calls`` entry pointing
at a ``tool_call_id``, and if the matching tool result disappears or changes role
the provider rejects the request. The tombstone stands in for the content while
leaving the shape of the conversation untouched.

The text says where the content went, because ContextCrunch never destroys
anything: the original is in the sidecar Box and can be put back.
"""

from __future__ import annotations

from contextcrunch.core.messages import Message

#: Prefix every tombstone starts with. Cheap to detect and cheap to keep.
TOMBSTONE_PREFIX = "Removed by ContextCrunch"

#: The text used when the item has no identifiable tool.
DEFAULT_LABEL = "item"


def tombstone_text(message: Message, reason: str) -> str:
    """Return the tombstone text for a message."""
    label = message.name or DEFAULT_LABEL
    return f"[{TOMBSTONE_PREFIX}: {label}. {reason}. Recoverable via sidecar.]"


def tombstone_message(message: Message, reason: str = "judged no longer relevant") -> Message:
    """Return a copy of ``message`` replaced by a short tombstone note.

    The ``role``, ``name`` and ``tool_call_id`` are all carried over unchanged, so
    the tombstone can stand in for the original without breaking the pairing
    between an assistant tool call and its result.
    """
    return message.model_copy(update={"content": tombstone_text(message, reason)})


def is_tombstone(message: Message) -> bool:
    """Return True when a message is a tombstone ContextCrunch produced."""
    return TOMBSTONE_PREFIX in message.content