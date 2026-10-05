"""Scout: pick the policy profile and extract the goal.

Scout is the only agent that looks at the whole conversation. It answers two
questions and nothing else:

- what is the goal, in one piece of text;
- which policy profile fits this trace.

Scout proposes. The pipeline is free to overrule it: an explicit profile in the
request always wins, and the Guardian can still escalate afterwards.

Scout is a plain rule-based loop. It calls no LLM and makes no network request:
the archetype is inferred from the tool names already present in the history,
which are free to read and deterministic to score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contextcrunch.core.messages import Message
from contextcrunch.core.policy import PROFILES
from contextcrunch.core.settings import get_settings
from contextcrunch.core.state import goal_from_messages
from contextcrunch.core.tokens import clip_to_chars, count_tokens

#: Value that means "work it out from the history".
AUTO = "auto"

#: Tool-name fragments for each archetype, and the profile each one maps to.
#: Matched case-insensitively as substrings, so ``kb_search`` counts as support
#: and ``run_tests`` counts as coding without listing every variant.
ARCHETYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "coding": (
        "read_file",
        "write_file",
        "edit",
        "pytest",
        "run_tests",
        "grep",
        "bash",
        "shell",
    ),
    "support": ("ticket", "refund", "customer", "invoice", "billing", "kb_", "crm"),
    "research": (
        "filing",
        "citation",
        "cite",
        "document",
        "pdf",
        "transcript",
        "search",
    ),
}

#: The profile each archetype maps to once it wins the keyword count.
ARCHETYPE_PROFILE: dict[str, str] = {
    "coding": "aggressive",
    "support": "balanced",
    "research": "conservative",
}

#: Tool-name fragments that mark a tool as too dangerous to compact away.
#: These are the side-effecting verbs: an agent that calls one of them has changed
#: something outside its own context, and its output is the only record of that.
PINNED_KEYWORDS: tuple[str, ...] = (
    "send",
    "pay",
    "refund",
    "delete",
    "create",
    "update",
    "post",
    "submit",
    "charge",
)


@dataclass(frozen=True)
class SessionContext:
    """What Scout worked out about this session before any compaction happens."""

    profile: str
    goal: str
    pinned_tools: frozenset[str]


class ScoutAgent:
    """Choose a profile, a goal and the pinned tools for one session.

    Rule-based and deterministic: the same history always produces the same answer,
    which is what makes a compaction reproducible and testable.
    """

    def run(
        self,
        messages: list[Message],
        requested_profile: str = AUTO,
        default_profile: str = "",
    ) -> SessionContext:
        """Return the session context for ``messages``.

        An explicit, valid ``requested_profile`` wins outright. Otherwise the
        archetype is scored from the tool names, and the highest count maps to its
        profile. No signal at all, or a tie between archetypes, falls back to
        ``default_profile``: guessing wrong in the aggressive direction is the one
        mistake that can lose information.
        """
        return SessionContext(
            profile=self.pick_profile(messages, requested_profile, default_profile),
            goal=self.extract_goal(messages),
            pinned_tools=self.pinned_tools(messages),
        )

    def pick_profile(
        self,
        messages: list[Message],
        requested_profile: str = AUTO,
        default_profile: str = "",
    ) -> str:
        """Return the profile name to use for this session."""
        if requested_profile != AUTO and requested_profile in PROFILES:
            return requested_profile

        scores = self.score_archetypes(messages)
        best = max(scores.values(), default=0)
        if best == 0:
            return default_profile

        # A tie means no archetype actually dominates, so fall back rather than
        # letting dict order silently decide.
        winners = [name for name, score in scores.items() if score == best]
        if len(winners) != 1:
            return default_profile
        return ARCHETYPE_PROFILE[winners[0]]

    def score_archetypes(self, messages: list[Message]) -> dict[str, int]:
        """Return how many keyword hits each archetype got from the tool names."""
        scores = dict.fromkeys(ARCHETYPE_KEYWORDS, 0)
        for name in self.tool_names(messages):
            lowered = name.lower()
            for archetype, keywords in ARCHETYPE_KEYWORDS.items():
                if any(keyword in lowered for keyword in keywords):
                    scores[archetype] += 1
        return scores

    def extract_goal(self, messages: list[Message]) -> str:
        """Return every user message joined with ``" | "``.

        Deliberately untruncated: ``build_state`` applies ``settings.goal_max_chars``,
        so clipping here too would make the limit impossible to reason about.
        """
        return " | ".join(m.content for m in messages if m.role == "user" and m.content)

    def pinned_tools(self, messages: list[Message]) -> frozenset[str]:
        """Return the tool names that must never be compacted away.

        A name is pinned when it contains any side-effecting verb. Those calls have
        already changed something outside the conversation, so their output is the
        only surviving record of what was done.
        """
        return frozenset(
            name
            for name in self.tool_names(messages)
            if any(keyword in name.lower() for keyword in PINNED_KEYWORDS)
        )

    def tool_names(self, messages: list[Message]) -> set[str]:
        """Return the distinct tool names seen in this history."""
        names: set[str] = set()
        for message in messages:
            if message.role == "tool" and message.name:
                names.add(message.name)
            for call in message.tool_calls:
                if isinstance(call, dict):
                    function = call.get("function") or {}
                    if function.get("name"):
                        names.add(str(function["name"]))
        return names


#: Words that suggest the agent is editing files and running commands.
_CODE_HINTS = (
    "fix",
    "bug",
    "refactor",
    "test",
    "compile",
    "stack trace",
    "repository",
    "function",
    "code",
    "patch",
    "build",
)
#: Words that suggest a human is in the loop.
_SUPPORT_HINTS = (
    "customer",
    "ticket",
    "refund",
    "invoice",
    "complaint",
    "account",
    "cancel",
    "dispute",
)
#: Words that suggest evidence must stay citable.
_RESEARCH_HINTS = (
    "cite",
    "citation",
    "citation",
    "memo",
    "report",
    "evidence",
    "source",
    "quote",
    "filing",
    "contract",
)

_TOOL_WORDS = {
    "read_file",
    "write_file",
    "edit_file",
    "run_terminal",
    "run_in_terminal",
    "search",
    "grep",
    "pytest",
    "web_search",
    "fetch_url",
}


@dataclass(frozen=True)
class Plan:
    """What Scout decided: the goal, the profile and why."""

    goal: str
    profile: str
    reason: str


def extract_goal(messages: list[Message]) -> str:
    """Return the goal, taken from the first user message."""
    return goal_from_messages(messages)


def pick_profile(messages: list[Message], requested: str | None = None) -> Plan:
    """Choose a policy profile for this trace, the way the pipeline asks for it.

    Kept for the pipeline, which reasons about the goal's *wording* as well as the
    profile. :class:`ScoutAgent` is the rule-based entry point; both share the
    keyword tables above, so the two cannot drift apart.
    """
    if requested and requested in PROFILES:
        return Plan(goal=extract_goal(messages), profile=requested, reason="requested")

    goal = extract_goal(messages).lower()
    tools = _tool_names(messages)
    score = {
        "aggressive": _hits(goal, _CODE_HINTS) + (2 if tools & _TOOL_WORDS else 0),
        "balanced": _hits(goal, _SUPPORT_HINTS),
        "conservative": _hits(goal, _RESEARCH_HINTS),
    }
    best = max(score, key=lambda name: score[name])
    if score[best] == 0:
        best = "conservative"
    return Plan(
        goal=extract_goal(messages),
        profile=best,
        reason="guessed from " + ", ".join(f"{k}={v}" for k, v in score.items()),
    )


def goal_text(messages: list[Message]) -> str:
    """Return the goal clipped to the configured budget."""
    return clip_to_chars(extract_goal(messages), get_settings().goal_max_chars)


def total_tokens(messages: list[Message]) -> int:
    """Return the token count of a whole history."""
    return sum(count_tokens(message.content) for message in messages)


def should_compact(messages: list[Message]) -> bool:
    """Return True when the history is big enough to be worth compacting."""
    return total_tokens(messages) >= get_settings().trigger_tokens


def _hits(text: str, hints: tuple[str, ...]) -> int:
    """Count how many hint phrases appear in ``text``."""
    return sum(1 for hint in hints if hint in text)


def _tool_names(messages: list[Message]) -> set[str]:
    """Collect the tool names seen in this trace."""
    names: set[str] = set()
    for message in messages:
        if message.role == "tool" and message.name:
            names.add(message.name)
        for call in message.tool_calls:
            if isinstance(call, dict):
                function = call.get("function") or {}
                if function.get("name"):
                    names.add(str(function["name"]))
    return {name for name in names if name}


_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


def first_sentence(text: str) -> str:
    """Return the first sentence of ``text``, used for a short goal preview."""
    match = _SENTENCE.search(text.strip())
    return match.group(0).strip() if match else text.strip()
