"""Build the state handed to Laya for one history item.

The state is deliberately domain-free. It exposes four relations and nothing else:

``goal``           what the agent is trying to do;
``next_step``      the most recent assistant text, a signal for future need;
``item``           the message on trial;
``later_findings`` assistant messages *after* the item, nearest first.

``later_findings`` is what makes the ``consumed`` question answerable at all: Laya
can see that a 14000-token fetch was already distilled into a short later message.
That is the most important input to the whole engine, so it is kept first and
shrunk first.

Sizing has one subtlety worth stating plainly. ``item["tokens"]`` reports the
**full** untruncated size of the item, while ``item["content"]`` is only what Laya
is actually shown. Laya must be able to judge "this item is enormous" without
reading all of it, so the size signal must not be distorted by the clipping that
keeps the request inside the window.
"""

from __future__ import annotations

import json
from typing import Any

from contextcrunch.core.messages import Message
from contextcrunch.core.questions import QUESTION_SET
from contextcrunch.core.settings import Settings
from contextcrunch.core.tokens import count_tokens

#: Hard floor for the item content. Shrinking stops here rather than emptying it.
MIN_ITEM_CHARS = 200

#: Fraction of the item content kept per round while the state is still too big.
ITEM_SHRINK_FACTOR = 0.8

#: Character cap on ``next_step``, which is a hint rather than evidence.
NEXT_STEP_CHARS = 300


def later_findings(messages: list[Message], idx: int, cap_chars: int) -> list[str]:
    """Return assistant messages after ``idx``, nearest first.

    The list stops *before* the cap would be exceeded, so the result never ends in a
    half-finding. Nearest first because the most recent conclusion is the one most
    likely to say whether this item's content was already extracted.

    Assistant messages before ``idx`` are skipped, as are tool outputs: only the
    agent's own distilled conclusions count as findings.
    """
    findings: list[str] = []
    used = 0
    for message in reversed(messages[idx + 1 :]):
        if message.role != "assistant" or not message.content:
            continue
        if used + len(message.content) > cap_chars:
            break
        findings.append(message.content)
        used += len(message.content)
    return findings


def build_state(
    messages: list[Message],
    idx: int,
    goal: str,
    settings: Settings,
    content: str | None = None,
) -> dict[str, Any]:
    """Build the Laya state for the message at ``idx``.

    ``content`` lets the refiner build a state for one piece of an item rather than
    the whole thing, which is why the same five questions work at any granularity.
    """
    message = messages[idx]
    full = message.content if content is None else content

    state: dict[str, Any] = {
        "goal": goal[: settings.goal_max_chars],
        "next_step": _next_step(messages),
        "item": {
            "index": idx,
            "role": message.role,
            "tool_call": message.fingerprint(),
            "content": full[: settings.item_max_chars],
            # The full size, not the clipped size: Laya must be able to tell that an
            # item is enormous without being sent all of it.
            "tokens": count_tokens(full),
            "age_in_turns": len(messages) - idx,
        },
        "later_findings": later_findings(messages, idx, settings.findings_cap_chars),
    }
    return fit_state(state, settings)


def fit_state(state: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Shrink ``state`` until it fits the Laya input window.

    The order is deliberate. Findings go first, because a clipped or invented
    finding is worse than no finding at all, and the goal is never cut, because
    without it every relevance judgement is meaningless. Only once the findings are
    exhausted does the item content shrink, and it stops at
    :data:`MIN_ITEM_CHARS` rather than being emptied.

    Mutates and returns ``state``, so a caller can pass a freshly built dict.
    """
    item = state.get("item")
    if not isinstance(item, dict):
        return state

    findings = state.get("later_findings")
    if not isinstance(findings, list):
        findings = []
        state["later_findings"] = findings

    questions_cost = count_tokens(json.dumps(dict(QUESTION_SET), ensure_ascii=False))
    while _estimate(state, questions_cost) > settings.state_max_tokens:
        if findings:
            # Drop the furthest finding first, so the nearest is kept longest.
            findings.pop()
            continue
        content = str(item.get("content", ""))
        if len(content) <= MIN_ITEM_CHARS:
            break
        item["content"] = content[: max(MIN_ITEM_CHARS, int(len(content) * ITEM_SHRINK_FACTOR))]
    return state


def _estimate(state: dict[str, Any], questions_cost: int) -> int:
    """Return the token cost of a state plus the question set sent with it."""
    return count_tokens(json.dumps(state, ensure_ascii=False)) + questions_cost


def reviewable_indices(messages: list[Message]) -> list[int]:
    """Return the indices worth judging.

    System and user messages are pinned by policy and never judged, so they are
    excluded here rather than filtered after paying for a model call.
    """
    return [i for i, message in enumerate(messages) if message.role not in ("system", "user")]


def goal_from_messages(messages: list[Message]) -> str:
    """Return the goal, taken from the first user message."""
    for message in messages:
        if message.role == "user":
            return message.content
    return ""


def _next_step(messages: list[Message]) -> str:
    """Return the last message's content, when it came from the assistant."""
    if not messages:
        return ""
    last = messages[-1]
    return last.content[:NEXT_STEP_CHARS] if last.role == "assistant" else ""