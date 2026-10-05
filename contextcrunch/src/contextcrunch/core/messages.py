"""The message shape ContextCrunch reads and writes.

Only the keys a compaction decision needs are modelled. Anything else the caller
sent is preserved untouched, so ContextCrunch is safe to put in front of any
agent's message list.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]

#: Roles that are pinned by policy and never judged.
PINNED_ROLES: frozenset[str] = frozenset({"system", "user"})


class Message(BaseModel):
    """One entry of an agent message history."""

    model_config = ConfigDict(extra="allow")

    role: Role
    content: str = ""
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    def is_tool_output(self) -> bool:
        """Return True when this message is a tool result."""
        return self.role == "tool"

    def fingerprint(self) -> str:
        """Return a short description of the tool call behind this message.

        This is the recoverability signal the policy reasons about: a tool output
        is only dangerous to remove if the same call cannot be re-issued. An
        assistant message points at the function it invoked, a tool result falls
        back to its own name, and anything else has nothing to point at.
        """
        for call in self.tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            name = function.get("name") or call.get("name")
            if name:
                return str(name)
        if self.role == "tool":
            return self.name or self.tool_call_id or "tool"
        return ""

    def label(self) -> str:
        """Return a short label for logs."""
        if self.role == "tool" and self.name:
            return f"tool:{self.name}"
        return self.role


def as_messages(raw: list[dict[str, Any]]) -> list[Message]:
    """Build messages from plain dicts, skipping anything that does not fit."""
    messages: list[Message] = []
    for item in raw:
        if not isinstance(item, dict) or "role" not in item:
            continue
        content = item.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        messages.append(Message(**{**item, "content": content}))
    return messages


def to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    """Turn messages back into plain dicts, keeping every extra key."""
    return [message.model_dump(exclude_none=True) for message in messages]
