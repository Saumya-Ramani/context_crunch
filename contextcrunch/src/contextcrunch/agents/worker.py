"""Worker: the Compactor. PLAN -> ACT -> VERIFY.

ContextCrunch sits in front of somebody else's LLM call, so the Worker is the
component that actually touches the message history. It runs in three phases, and
each one exists to stop a compaction from doing damage.

**PLAN** decides whether to act at all, and on what. It counts the tokens, and a
history below ``CC_TRIGGER_TOKENS`` is returned completely untouched: paying for
model calls on a short history cannot pay for itself. It then works out which
messages are even eligible. Only ``tool`` output is judged. System, user and
assistant turns are the conversation itself, and a human turn cannot be re-issued,
so it is never on trial. Three further things make an item untouchable: the
Guardian asked for it back, its tool is pinned because that call changed the
outside world, or it is one of the most recent turns the agent is probably still
working with. None of those costs a model call to establish.

**ACT** asks the engine about the eligible items and applies the policy. Items
small enough to fit in one state are all judged in a single parallel batch. An
item too big to send whole is never sent whole: it is *split first* into pieces
that are each judgeable, then rebuilt from whatever survives. That is the
difference between a 14000-token filing losing twelve of its fourteen invoices and
the filing being kept intact because it will not fit in the window.

**VERIFY** then checks the Worker's own output rather than trusting it. The same
number of messages, the same roles in the same order, every ``tool_call_id`` still
attached, no empty content. If any of that fails the original history is returned,
because a malformed history costs the caller far more than a long one.

The Worker never calls an LLM and never rewrites text. What it emits is either the
original string, a short tombstone, or a rejoin of original pieces, so kept
evidence is always verbatim. Nothing is destroyed: the original content of every
reviewed item goes to the :class:`~contextcrunch.storage.store.Store`, which is
what makes a compaction reversible.

One deliberate omission: there is no ``try``/``except`` around the work, and no
deadline. Fail-open belongs to the API layer, and an agent that quietly swallows a
failure would report a compaction that never actually happened.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from contextcrunch.agents.guardian import Overrides
from contextcrunch.agents.scout import SessionContext
from contextcrunch.core.messages import Message
from contextcrunch.core.policy import DROP, KEEP, TRUNCATE, Profile, decide, get_profile
from contextcrunch.core.settings import Settings, get_settings
from contextcrunch.core.splitter import Piece, clip_head_tail, split_text
from contextcrunch.core.state import build_state
from contextcrunch.core.tokens import count_tokens
from contextcrunch.core.tombstone import tombstone_message
from contextcrunch.laya.engine import LayaEngine
from contextcrunch.laya.types import LayaResult
from contextcrunch.storage.store import Store, build_store

#: Separator between the surviving pieces of one rebuilt item.
PIECE_SEPARATOR = "\n---\n"

#: Prefix added to a kept piece when the profile preserves source references.
ANCHOR_PREFIX = "[{anchor}] "

#: Text left in place of a piece the policy dropped.
PIECE_DROP_TEXT = "[removed: {anchor} judged irrelevant]"


@dataclass
class Stats:
    """What one compaction changed.

    All-zero stats mean "nothing was touched", which is also what a skipped or a
    failed compaction returns, so a caller can tell a no-op from a real run by
    looking at this one object.
    """

    tokens_before: int = 0
    tokens_after: int = 0
    kept: int = 0
    shortened: int = 0
    removed: int = 0
    actions: dict[int, str] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        """Return True when the history was actually altered."""
        return bool(self.shortened or self.removed)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view for the API response."""
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "reduction_pct": (
                round(100.0 * (self.tokens_before - self.tokens_after) / self.tokens_before, 2)
                if self.tokens_before
                else 0.0
            ),
            "kept": self.kept,
            "shortened": self.shortened,
            "removed": self.removed,
            "actions": {str(k): v for k, v in sorted(self.actions.items())},
        }


@dataclass
class _Plan:
    """Which items are worth a model call, and what happens to the rest."""

    #: Every index that will be judged.
    reviewable: list[int] = field(default_factory=list)
    #: Items sent whole, because they fit in a single state.
    whole: list[int] = field(default_factory=list)
    #: Items too big for one state, so they are split first.
    split_first: list[int] = field(default_factory=list)
    #: Action per index, including the KEEPs that needed no model call.
    actions: dict[int, str] = field(default_factory=dict)


class CompactorAgent:
    """Compact one session's history.

    Collaborators are injected rather than looked up: the engine, the store and
    the settings are exactly the seams a test needs to control, and building any
    of them here would stop a test exercising a policy path without a network
    call.
    """

    def __init__(self, engine: LayaEngine, store: Store, settings: Settings) -> None:
        self.engine = engine
        self.store = store
        self.settings = settings

    async def run(
        self,
        session_id: str,
        messages: list[Message],
        ctx: SessionContext,
        overrides: Overrides | None = None,
    ) -> tuple[list[Message], Stats, str]:
        """Compact ``messages`` and return the new history, the stats and the profile used.

        PLAN, ACT and VERIFY in that order. A verification failure returns the
        original history with zero-change stats rather than something that might
        be malformed.

        Raises whatever the engine raises. Fail-open is the caller's decision, so
        a model failure is never reported here as a successful compaction.
        """
        overrides = overrides or Overrides()

        # ---------------------------------------------------------- PLAN
        tokens_before = sum(count_tokens(m.content) for m in messages)
        if tokens_before < self.settings.trigger_tokens:
            # Not a failure: a short history is simply not worth compacting.
            return messages, Stats(tokens_before=tokens_before), ctx.profile

        profile_name = overrides.profile or ctx.profile
        profile = get_profile(profile_name)
        plan = self._plan(messages, ctx, overrides, profile)
        stats = Stats(tokens_before=tokens_before, actions=dict(plan.actions))

        if not plan.reviewable:
            return messages, stats, profile_name

        # ----------------------------------------------------------- ACT
        compacted = await self._act(
            session_id=session_id,
            messages=messages,
            ctx=ctx,
            profile=profile,
            plan=plan,
            stats=stats,
        )

        # -------------------------------------------------------- VERIFY
        if self._verify(messages, compacted):
            # Returning the originals rather than a history that might be
            # malformed: a long history is recoverable, a broken one is not.
            return messages, Stats(tokens_before=tokens_before), profile_name

        stats.tokens_after = sum(count_tokens(m.content) for m in compacted)
        return compacted, stats, profile_name

    # ------------------------------------------------------------------ #
    # PLAN
    # ------------------------------------------------------------------ #

    def _plan(
        self, messages: list[Message], ctx: SessionContext, overrides: Overrides, profile: Profile
    ) -> _Plan:
        """Work out what to review, recording the KEEPs that need no model call."""
        plan = _Plan()
        total = len(messages)

        for index, message in enumerate(messages):
            if message.role != "tool":
                # The conversation itself is never on trial.
                plan.actions[index] = KEEP
                continue
            if (
                index in overrides.restore
                or message.name in ctx.pinned_tools
                or (total - index) <= profile.K
            ):
                plan.actions[index] = KEEP
                continue

            plan.reviewable.append(index)
            if len(message.content) <= self.settings.item_max_chars:
                plan.whole.append(index)
            else:
                plan.split_first.append(index)

        return plan

    # ------------------------------------------------------------------ #
    # ACT
    # ------------------------------------------------------------------ #

    async def _act(
        self,
        *,
        session_id: str,
        messages: list[Message],
        ctx: SessionContext,
        profile: Profile,
        plan: _Plan,
        stats: Stats,
    ) -> list[Message]:
        """Judge every planned item in one parallel batch, then rebuild the history."""
        answers = await self._judge(messages, ctx, plan.whole, goal=ctx.goal)
        compacted: list[Message] = []
        total = len(messages)

        for index, message in enumerate(messages):
            answer = answers.get(index)
            rebuilt: str | None = None
            action = KEEP

            if answer is None:
                # Not judged as a single state: either pinned or recent, so it is
                # already correct, or too big to send whole, in which case the
                # split-first pass judges the pieces instead.
                if index in plan.split_first:
                    rebuilt, action = await self._rebuild(message, index, ctx, profile)
            else:
                action = self._judge_action(message, index, total, answer, profile, ctx)
                if action == TRUNCATE:
                    # The policy said "shorten this"; the rebuild decides by how
                    # much, and may find there was nothing worth removing.
                    rebuilt, rebuilt_action = await self._rebuild(message, index, ctx, profile)
                    if rebuilt is not None:
                        action = rebuilt_action

            stats.actions[index] = action
            if action == DROP:
                stats.removed += 1
                compacted.append(tombstone_message(message))
            elif rebuilt is not None:
                stats.shortened += 1
                compacted.append(message.model_copy(update={"content": rebuilt}))
            else:
                stats.kept += 1
                compacted.append(message)

            if index in plan.reviewable:
                # The ORIGINAL content is logged, never the rewritten text: being
                # able to undo a removal is the entire purpose of the store.
                self.store.log_decision(
                    session_id=session_id,
                    item_index=index,
                    fingerprint=message.fingerprint(),
                    action=action,
                    relevance=_relevance(answer) if answer is not None else 0.0,
                    original_content=message.content,
                )

        return compacted

    def _judge_action(
        self,
        message: Message,
        index: int,
        total: int,
        answer: LayaResult,
        profile: Profile,
        ctx: SessionContext,
    ) -> str:
        """Apply the policy to one judged item."""
        return decide(
            {
                "role": message.role,
                "tokens": count_tokens(message.content),
                "age_in_turns": total - index,
                "tool_name": message.name,
            },
            answer.answers,
            profile,
            pinned_tools=ctx.pinned_tools,
        )

    async def _judge(
        self, messages: list[Message], ctx: SessionContext, indexes: list[int], *, goal: str
    ) -> dict[int, LayaResult]:
        """Ask the engine about ``indexes`` in one parallel batch."""
        if not indexes:
            return {}
        states = [
            build_state(messages, index, goal, self.settings, content=messages[index].content)
            for index in indexes
        ]
        results = await self.engine.decide(states, [f"item:{index}" for index in indexes])
        return dict(zip(indexes, results, strict=True))

    async def _rebuild(
        self, message: Message, index: int, ctx: SessionContext, profile: Profile
    ) -> tuple[str | None, str]:
        """Split one item, judge the pieces, and join whatever survives.

        Returns the rebuilt text (``None`` when nothing was removed) and the
        action the item ended up with. An item whose every piece is dropped
        becomes a whole-item DROP: a filing reduced to fourteen tombstone lines
        carries no more information than one tombstone and costs more.
        """
        pieces = split_text(
            message.content, self.settings.piece_target_chars, self.settings.piece_max_chars
        )
        if not pieces:
            return None, KEEP

        if len(pieces) == 1:
            # Already a single piece, so judging it separately gains nothing.
            head, tail = self._clip_sizes(pieces[0].text)
            clipped = clip_head_tail(pieces[0].text, head, tail)
            if len(clipped) >= len(message.content):
                return None, KEEP
            return clipped, TRUNCATE

        states = [
            build_piece_state(piece.text, message, ctx.goal, self.settings) for piece in pieces
        ]
        answers = await self.engine.decide(
            states, [f"piece:{index}:{piece.anchor}" for piece in pieces]
        )

        rendered: list[str] = []
        dropped = 0
        for piece, answer in zip(pieces, answers, strict=True):
            # is_sub=True: the parent item's size and position say nothing about
            # the value of an individual passage.
            action = decide(
                {
                    "role": message.role,
                    "tokens": count_tokens(piece.text),
                    "age_in_turns": 0,
                    "tool_name": message.name,
                },
                answer.answers,
                profile,
                pinned_tools=ctx.pinned_tools,
                is_sub=True,
            )
            if action == DROP:
                dropped += 1
                rendered.append(PIECE_DROP_TEXT.format(anchor=piece.anchor))
                continue

            if action == TRUNCATE:
                head, tail = self._clip_sizes(piece.text)
                body = clip_head_tail(piece.text, head, tail)
            else:
                body = piece.text
            rendered.append(self._anchor(body, piece, profile))

        if dropped == len(pieces):
            return None, DROP

        rebuilt = PIECE_SEPARATOR.join(rendered)
        if len(rebuilt) >= len(message.content):
            return None, KEEP
        return rebuilt, TRUNCATE

    def _anchor(self, body: str, piece: Piece, profile: Profile) -> str:
        """Prefix a kept piece with its source reference when the profile wants one."""
        if not profile.anchors:
            return body
        return ANCHOR_PREFIX.format(anchor=piece.anchor) + body

    def _clip_sizes(self, text: str) -> tuple[int, int]:
        """Return the head and tail lengths to keep when clipping ``text``.

        The fractions come from ``piece_max_chars`` because a piece is sized by
        that setting, but the sum is capped at half the text: a clip that would
        keep more than the original is not a clip, and returning unchanged text is
        the only honest outcome when the content is already smaller than the
        budget.
        """
        budget = self.settings.piece_max_chars
        head = max(budget // 4, 100)
        tail = max(budget // 8, 50)
        room = max(len(text) // 2, 0)
        if head + tail >= room:
            return room // 2, room - room // 2
        return head, tail

    # ------------------------------------------------------------------ #
    # VERIFY
    # ------------------------------------------------------------------ #

    def _verify(self, before: list[Message], after: list[Message]) -> list[str]:
        """Return every structural problem found in the compacted history.

        An empty list means the history is safe to use. These are exactly the
        things a provider rejects a request over: a changed count, a changed role,
        a lost ``tool_call_id``, or content that is now empty.
        """
        if len(before) != len(after):
            return [f"message count changed: {len(before)} -> {len(after)}"]

        problems: list[str] = []
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            if old.role != new.role:
                problems.append(f"role changed at index {index}")
            if old.tool_call_id and new.tool_call_id != old.tool_call_id:
                problems.append(f"tool_call_id lost at index {index}")
            if not isinstance(new.content, str):
                problems.append(f"content is not text at index {index}")
            elif not new.content.strip():
                problems.append(f"empty content at index {index}")
        return problems


def build_piece_state(
    content: str, message: Message, goal: str, settings: Settings
) -> dict[str, Any]:
    """Build the state for one piece of an item.

    A piece is judged without the surrounding history on purpose. The parent
    item's age and size say nothing about the value of one passage inside it, so
    only the goal and the piece itself are supplied.
    """
    return {
        "goal": goal[: settings.goal_max_chars],
        "next_step": "",
        "item": {
            "index": 0,
            "role": message.role,
            "tool_call": message.fingerprint(),
            "content": content,
            "tokens": count_tokens(content),
            "age_in_turns": 0,
        },
        "later_findings": [],
    }


def _relevance(answer: LayaResult) -> float:
    """Return the relevance score Laya gave, or ``0`` when it gave none.

    The store's column is not nullable, so an unanswered question is recorded as
    zero rather than as a number nobody measured.
    """
    verdict = answer.answer("relevance")
    if verdict is None:
        return 0.0
    score = getattr(verdict, "score", None)
    return float(score) if isinstance(score, int | float) else 0.0


# ---------------------------------------------------------------------- #
# Compatibility shim
#
# The API layer still calls the module-level ``compact()``. It is kept so both
# entry points keep working while ``CompactorAgent`` is the primary interface.
# ---------------------------------------------------------------------- #


@dataclass
class Outcome:
    """The result of one compaction through the legacy entry point."""

    run_id: str
    messages: list[Message]
    profile: str
    goal: str
    actions: dict[int, str] = field(default_factory=dict)
    tokens_before: int = 0
    tokens_after: int = 0
    degraded: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view for the API response."""
        return {
            "run_id": self.run_id,
            "profile": self.profile,
            "goal": self.goal,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "reduction_pct": (
                round(100.0 * (self.tokens_before - self.tokens_after) / self.tokens_before, 2)
                if self.tokens_before
                else 0.0
            ),
            "actions": {str(k): v for k, v in sorted(self.actions.items())},
            "degraded": self.degraded,
            "notes": self.notes,
        }


async def compact(
    *,
    client: Any,
    messages: list[Message],
    goal: str,
    profile: Profile,
    store: Store | None = None,
    box: Any | None = None,
    profile_name: str = "",
    run_id: str | None = None,
    deadline_s: float | None = None,
) -> Outcome:
    """Compact a history, failing open to the original messages on any error.

    The older, function-shaped entry point. It keeps the fail-open wrapper the API
    depends on and delegates the judgement to :class:`CompactorAgent`.

    ``box`` is the per-run side-car the Guardian reads; ``store`` is the
    session-scoped one. Both are optional so a caller with neither still gets a
    compaction, which is what makes the fail-open path reachable in a test.
    """
    settings = get_settings()
    run = run_id or uuid.uuid4().hex[:12]
    tokens_before = sum(count_tokens(m.content) for m in messages)
    outcome = Outcome(
        run_id=run, messages=messages, profile=profile_name, goal=goal, tokens_before=tokens_before
    )

    try:
        agent = CompactorAgent(
            engine=LayaEngine(client=client, settings=settings),
            store=store if store is not None else build_store(),
            # The trigger is the pipeline's job, not the agent's, so it is lifted
            # out of the way here rather than being re-checked against it.
            settings=settings.model_copy(update={"trigger_tokens": 0}),
        )
        compacted, stats, used = await agent.run(
            session_id=run,
            messages=messages,
            ctx=SessionContext(
                profile=profile_name or settings.default_profile,
                goal=goal,
                pinned_tools=frozenset(),
            ),
        )
        outcome.messages = compacted
        outcome.actions = dict(stats.actions)
        outcome.tokens_after = stats.tokens_after
        outcome.notes.append(f"profile {used}")
        return outcome
    except Exception as exc:  # noqa: BLE001 - fail-open is a hard requirement
        outcome.messages = messages
        outcome.actions = {}
        outcome.degraded = True
        outcome.notes = [f"fail-open: {type(exc).__name__}: {exc}"]
        outcome.tokens_after = tokens_before
        return outcome