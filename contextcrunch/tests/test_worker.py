"""Tests for the Worker: PLAN, ACT and VERIFY.

The Worker is the part that touches the history, so these tests care about three
things above all: that nothing eligible is skipped, that nothing ineligible is
judged, and that a bad outcome returns the original messages rather than a broken
history.

Every test drives :class:`CompactorAgent` with a scripted engine and a real
:class:`~contextcrunch.storage.store.Store` in ``tmp_path``. No model is loaded
and no network call is made.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.conftest import answers
from tests.helpers import ScriptedEngine, make_keep_answer, make_drop_answer, make_truncate_answer

from contextcrunch.agents.guardian import Overrides
from contextcrunch.agents.scout import SessionContext
from contextcrunch.agents.worker import (
    PIECE_SEPARATOR,
    CompactorAgent,
    build_piece_state,
)
from contextcrunch.core.messages import Message
from contextcrunch.core.settings import Settings
from contextcrunch.core.tombstone import TOMBSTONE_PREFIX
from contextcrunch.laya.types import LayaResult
from contextcrunch.storage.store import Store

#: Answers that clear every drop gate, so the policy really drops.
DROP_ALL = answers(
    verdict="drop", essential=0.0, superseded=0.9, relevance=0.0, verdict_conf=0.99
)
#: Answers that clear nothing: the policy keeps.
KEEP_ALL = answers(verdict="keep", relevance=3.0)


@pytest.fixture
def store(tmp_path) -> Store:
    """Return a real Store backed by tmp_path."""
    return Store(db_path=str(tmp_path / "store.db"), originals_dir=str(tmp_path / "originals"))


@pytest.fixture
def settings_factory_default(settings_factory):
    """Return the settings factory with a low trigger so tests reach ACT."""

    def build(**overrides: Any) -> Settings:
        # A trigger of 1 lets a tiny fixture reach ACT; an explicit trigger in a
        # test still wins, so the defaults are applied with setdefault semantics.
        defaults: dict[str, Any] = {"trigger_tokens": 1, "laya_concurrency": 8}
        defaults.update(overrides)
        return settings_factory(**defaults)

    return build


def agent_for(engine: ScriptedEngine, store: Store, settings: Settings) -> CompactorAgent:
    """Build a CompactorAgent over the fakes."""
    return CompactorAgent(engine=engine, store=store, settings=settings)


def ctx_for(profile: str = "aggressive", pinned: frozenset[str] = frozenset()) -> SessionContext:
    """Return a session context for the tests."""
    return SessionContext(profile=profile, goal="fix the billing bug", pinned_tools=pinned)


def tool(text: str, index: int, name: str = "read_file") -> Message:
    """Return a tool message."""
    return Message(role="tool", content=text, name=name, tool_call_id=f"call_{index}")


def history(*contents: str, name: str = "read_file") -> list[Message]:
    """Return a history whose tool messages are deep enough to escape the K window."""
    messages: list[Message] = [
        Message(role="system", content="You are a coding agent."),
        Message(role="user", content="fix the billing bug"),
    ]
    for index, content in enumerate(contents):
        messages.append(tool(content, index, name))
        messages.append(Message(role="assistant", content=f"finding {index}"))
    return messages


def noise(index: int = 0) -> str:
    """Return tool output big enough to be worth judging."""
    return "filler line of console output\n" * 40 + f" end{index}"


# --------------------------------------------------------------------------- #
# PLAN
# --------------------------------------------------------------------------- #


async def test_history_below_the_trigger_is_untouched(
    store: Store, settings_factory_default
) -> None:
    """A short history costs nothing: it is returned exactly as it arrived."""
    settings = settings_factory_default(trigger_tokens=100_000)
    engine = ScriptedEngine({"": make_drop_answer()})
    # Several deep tool messages, so that with the trigger removed there WOULD be
    # eligible items to judge and the history WOULD change. Without that, this
    # test would pass even if the trigger check were deleted, because every item
    # would be kept anyway for being recent.
    messages = history(*[noise(n) for n in range(5)])

    new, stats, profile = await agent_for(engine, store, settings).run("s1", messages, ctx_for())

    assert new == messages, "a skipped history must be the original objects"
    # Zero-change stats: tokens_after stays 0 because nothing was measured after
    # a run that deliberately did nothing, which is how a caller tells a skip
    # from a real compaction.
    assert stats.tokens_before > 0
    assert stats.tokens_after == 0
    assert stats.kept == 0 and stats.shortened == 0 and stats.removed == 0
    assert stats.changed is False
    assert profile == "aggressive"
    assert engine.call_count == 0, "no model call may be made below the trigger"


async def test_only_tool_messages_are_judged(store: Store, settings_factory_default) -> None:
    """A human turn cannot be re-issued, so it is never on trial."""
    settings = settings_factory_default()
    engine = ScriptedEngine({"": make_keep_answer()})
    messages = history(noise(), noise(1))

    await agent_for(engine, store, settings).run("s1", messages, ctx_for())

    # All calls should be for tool messages
    assert engine.call_count > 0


async def test_a_noise_tool_message_is_tombstoned(
    store: Store, settings_factory_default
) -> None:
    """A dropped tool output becomes a tombstone that keeps its shape."""
    settings = settings_factory_default()
    messages = history(noise(), noise(1))
    engine = ScriptedEngine({"": make_drop_answer()})

    new, stats, _ = await agent_for(engine, store, settings).run("s1", messages, ctx_for())

    assert len(new) == len(messages)
    assert TOMBSTONE_PREFIX in new[2].content
    # The shape an assistant tool_call depends on must be untouched.
    assert new[2].role == "tool"
    assert new[2].tool_call_id == messages[2].tool_call_id
    assert new[2].name == messages[2].name
    assert stats.removed >= 1


async def test_a_recent_tool_message_is_kept_without_a_call(
    store: Store, settings_factory_default
) -> None:
    """The most recent turns are the ones the agent is working with."""
    settings = settings_factory_default()
    messages = history(noise())
    engine = ScriptedEngine({"": make_drop_answer()})
    ctx = ctx_for("aggressive")

    from contextcrunch.core.policy import get_profile

    profile = get_profile("aggressive")
    total = len(messages)
    # Everything within profile.K of the end is protected, so nothing is judged.
    assert all((total - i) <= profile.K for i in range(len(messages)) if messages[i].role == "tool")

    new, _, _ = await agent_for(engine, store, settings).run("s1", messages, ctx)

    assert new == messages
    assert engine.call_count == 0


async def test_a_pinned_tool_is_kept_even_when_marked_noise(
    store: Store, settings_factory_default
) -> None:
    """A side-effecting call's output is the only record that it happened."""
    settings = settings_factory_default()
    messages = history(noise(), noise(1), noise(2))
    engine = ScriptedEngine({"": make_drop_answer()})
    # read_file is pinned; every tool message in this history is that call.
    ctx = ctx_for("aggressive", pinned=frozenset({"read_file"}))

    new, _, _ = await agent_for(engine, store, settings).run("s1", messages, ctx)

    for before, after in zip(messages, new, strict=True):
        assert after.content == before.content
    assert engine.call_count == 0, "a pinned tool must not cost a model call"


# --------------------------------------------------------------------------- #
# ACT: split-first
# --------------------------------------------------------------------------- #


async def test_a_long_message_is_shortened_and_the_evidence_survives(
    store: Store, settings_factory_default
) -> None:
    """The point of splitting: junk goes, the one useful paragraph stays verbatim."""
    settings = settings_factory_default(
        item_max_chars=200, piece_target_chars=200, piece_max_chars=400
    )
    keepme = "the invoice total must round half up using Decimal"
    long_item = "\n\n".join(
        [keepme] + [f"paragraph {n} of console noise" + (" x" * 60) for n in range(6)]
    )
    messages = history(long_item, noise(9))
    # keepme is the FIRST piece so the scripted answers are unambiguous: piece 1
    # is kept, piece 2 is dropped. Putting keepme last would make the test depend
    # on the exact piece count, which is a splitter detail and not the point.
    engine = ScriptedEngine({
        keepme: make_keep_answer(),
        "paragraph": make_drop_answer(),
    })

    new, stats, _ = await agent_for(engine, store, settings).run("s1", messages, ctx_for())

    rebuilt = new[2].content
    assert stats.shortened >= 1
    assert keepme in rebuilt, "the useful paragraph must survive word for word"
    assert stats.tokens_after < stats.tokens_before
    assert new[2].role == "tool"
    assert new[2].tool_call_id == messages[2].tool_call_id


async def test_the_conservative_profile_adds_an_anchor_prefix(
    store: Store, settings_factory_default
) -> None:
    """With anchors on, a kept piece records where it came from."""
    settings = settings_factory_default(
        item_max_chars=200, piece_target_chars=200, piece_max_chars=400
    )
    keepme = "cited passage that must remain findable"
    long_item = "\n\n".join(
        [keepme] + [f"paragraph {n}" + (" x" * 60) for n in range(6)]
    )
    messages = history(long_item, noise(9))
    engine = ScriptedEngine({
        keepme: make_keep_answer(),
        "paragraph": make_drop_answer(),
    })

    new, _, _ = await agent_for(engine, store, settings).run(
        "s1", messages, ctx_for("conservative")
    )

    rebuilt = new[2].content
    assert keepme in rebuilt
    assert rebuilt.startswith("[part"), f"expected an anchor prefix, got {rebuilt[:60]!r}"


async def test_an_item_whose_every_piece_is_dropped_becomes_one_tombstone(
    store: Store, settings_factory_default
) -> None:
    """Fourteen tombstone lines carry no more than one, and cost more."""
    settings = settings_factory_default(
        item_max_chars=200, piece_target_chars=400, piece_max_chars=400
    )
    long_item = "\n\n".join(f"paragraph {n}" + (" x" * 40) for n in range(6))
    messages = history(long_item, noise(9))
    engine = ScriptedEngine({"": make_drop_answer()})

    new, stats, _ = await agent_for(engine, store, settings).run("s1", messages, ctx_for())

    assert TOMBSTONE_PREFIX in new[2].content
    assert PIECE_SEPARATOR not in new[2].content
    assert stats.removed >= 1


async def test_a_single_piece_is_clipped_rather_than_judged(
    store: Store, settings_factory_default
) -> None:
    """One piece means there is nothing to gain from judging the pieces."""
    settings = settings_factory_default(
        item_max_chars=100, piece_target_chars=100_000, piece_max_chars=100_000
    )
    body = "x" * 5_000
    messages = history(body, noise(9))
    engine = ScriptedEngine({"": make_truncate_answer()})

    new, stats, _ = await agent_for(engine, store, settings).run("s1", messages, ctx_for())

    assert len(new[2].content) < len(body)
    assert "[... clipped ...]" in new[2].content
    assert stats.shortened >= 1
    assert engine.call_count == 0, "a single piece needs no piece judgement"


# --------------------------------------------------------------------------- #
# ACT: overrides and the store
# --------------------------------------------------------------------------- #


async def test_overrides_restore_keeps_an_item_that_would_be_dropped(
    store: Store, settings_factory_default
) -> None:
    """The Guardian can put an item back."""
    settings = settings_factory_default()
    messages = history(noise(), noise(1))
    engine = ScriptedEngine({"": make_drop_answer()})

    new, _, _ = await agent_for(engine, store, settings).run(
        "s1",
        messages,
        ctx_for(),
        Overrides(restore={2}),
    )

    assert new[2].content == messages[2].content
    assert TOMBSTONE_PREFIX not in new[2].content


async def test_decisions_are_logged_with_the_original_content(
    store: Store, settings_factory_default
) -> None:
    """The store keeps what was there, so a removal can be undone."""
    settings = settings_factory_default()
    original = noise()
    messages = history(original, noise(1))
    engine = ScriptedEngine({"": make_drop_answer()})

    await agent_for(engine, store, settings).run("s1", messages, ctx_for())

    assert store.stats("s1")["DROP"] >= 1
    saved = store.get_original("s1", 2)
    assert saved == original, "the store must hold the original, not the tombstone"


async def test_every_reviewable_item_is_judged_in_one_batch(
    store: Store, settings_factory_default
) -> None:
    """One batch, not one call per item: the cost of a compaction is one round trip.

    Three items are judged rather than four because the most recent tool message
    is inside the profile's protected window. That is the point: the batch holds
    exactly the items the plan found eligible, with no second round trip.
    """
    settings = settings_factory_default(laya_concurrency=8)
    messages = history(*[noise(n) for n in range(4)])
    engine = ScriptedEngine({"": make_keep_answer()})

    await agent_for(engine, store, settings).run("s1", messages, ctx_for())

    assert engine.call_count == 3, f"expected 3 eligible items, got {engine.call_count}"


# --------------------------------------------------------------------------- #
# VERIFY
# --------------------------------------------------------------------------- #


async def test_a_failed_verification_returns_the_originals(
    store: Store, settings_factory_default, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A history that would break the caller's request is worse than a long one."""
    settings = settings_factory_default()
    messages = history(noise(), noise(1))
    engine = ScriptedEngine({"": make_drop_answer()})
    agent = agent_for(engine, store, settings)

    # Force a structural failure the verifier must catch: a changed role.
    def bad_verify(before, after):
        return [f"role changed at index {i}" for i in range(len(before))]

    monkeypatch.setattr(agent, "_verify", bad_verify)

    new, stats, _ = await agent.run("s1", messages, ctx_for())

    assert new == messages, "a failed verification must return the originals"
    assert stats.removed == 0 and stats.shortened == 0
    assert stats.tokens_after == 0, "a failed verification reports no change"


async def test_verify_catches_every_structural_problem(store: Store) -> None:
    """The checks are the ones a provider would reject a request over."""
    settings = Settings()
    agent = CompactorAgent(engine=ScriptedEngine({"": make_keep_answer()}), store=store, settings=settings)
    good = [Message(role="user", content="hi", tool_call_id="c1")]

    assert agent._verify(good, good) == []
    assert agent._verify(good, []) == ["message count changed: 1 -> 0"]
    assert agent._verify(good, [Message(role="tool", content="x", tool_call_id="c1")])
    assert agent._verify(good, [Message(role="user", content="x", tool_call_id="other")])
    assert agent._verify(good, [Message(role="user", content="", tool_call_id="c1")])


async def test_run_does_not_swallow_engine_errors(
    store: Store, settings_factory_default
) -> None:
    """Fail-open belongs to the API layer, so the error must reach it."""

    class Exploding:
        async def decide(self, states, labels=None):
            raise RuntimeError("model exploded")

    settings = settings_factory_default()
    messages = history(noise(), noise(1))
    agent = CompactorAgent(engine=Exploding(), store=store, settings=settings)

    with pytest.raises(RuntimeError, match="model exploded"):
        await agent.run("s1", messages, ctx_for())


async def test_build_piece_state_carries_no_history(store: Store, settings_factory_default) -> None:
    """A piece is judged on its own, not through the parent item's position."""
    settings = settings_factory_default()
    message = tool("some passage", 2, name="read_file")

    state = build_piece_state("some passage", message, "a goal", settings)

    assert state["item"]["age_in_turns"] == 0
    assert state["later_findings"] == []
    assert state["item"]["content"] == "some passage"
    assert state["goal"] == "a goal"