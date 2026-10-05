"""Tests for ScoutAgent: profile selection, goal and pinned tools.

Scout is rule-based, so every test here is exact rather than approximate.
"""

from __future__ import annotations

import pytest
from tests.conftest import history

from contextcrunch.agents.scout import (
    ARCHETYPE_PROFILE,
    AUTO,
    ScoutAgent,
    SessionContext,
    extract_goal,
    first_sentence,
    pick_profile,
    should_compact,
)
from contextcrunch.core.messages import Message
from contextcrunch.core.policy import PROFILES


def with_tools(*names: str) -> list[Message]:
    """Return a user message followed by tool messages with the given names."""
    messages = [Message(role="user", content="do the thing")]
    for index, name in enumerate(names):
        messages.append(Message(role="tool", content="out", tool_call_id=f"c{index}", name=name))
    return messages


# --------------------------------------------------------------------------- #
# Profile selection from tool names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("read_file", "aggressive"),
        ("write_file", "aggressive"),
        ("edit", "aggressive"),
        ("pytest", "aggressive"),
        ("run_tests", "aggressive"),
        ("grep", "aggressive"),
        ("bash", "aggressive"),
        ("shell", "aggressive"),
        ("ticket", "balanced"),
        ("refund", "balanced"),
        ("customer", "balanced"),
        ("invoice", "balanced"),
        ("billing", "balanced"),
        ("crm", "balanced"),
        ("filing", "conservative"),
        ("citation", "conservative"),
        ("cite", "conservative"),
        ("document", "conservative"),
        ("pdf", "conservative"),
        ("transcript", "conservative"),
        ("search", "conservative"),
    ],
)
def test_each_keyword_maps_to_its_profile(tool_name: str, expected: str) -> None:
    """Every keyword in the spec must reach the profile it is mapped to."""
    context = ScoutAgent().run(with_tools(tool_name), AUTO, "conservative")
    assert context.profile == expected


def test_the_highest_keyword_count_wins() -> None:
    """Two coding tools beat one research tool."""
    context = ScoutAgent().run(with_tools("read_file", "pytest", "pdf"), AUTO, "balanced")
    assert context.profile == "aggressive"


@pytest.mark.parametrize("name", ["READ_FILE", "read_file", "rEaD_fIlE"])
def test_matching_is_case_insensitive(name: str) -> None:
    """Tool names arrive in whatever case the agent used."""
    assert ScoutAgent().run(with_tools(name), AUTO, "balanced").profile == "aggressive"


@pytest.mark.parametrize("name", ["ReadFile", "readFile"])
def test_camel_case_tool_names_do_not_match(name: str) -> None:
    """Matching is substring-based, so a camelCase name misses a snake_case keyword.

    A real limitation of the specified rules rather than a bug in them, and it
    fails safe: an unmatched name contributes nothing, so the session falls back to
    the default profile instead of being guessed into the aggressive one.
    """
    scores = ScoutAgent().score_archetypes(with_tools(name))
    assert sum(scores.values()) == 0
    assert ScoutAgent().run(with_tools(name), AUTO, "balanced").profile == "balanced"


def test_kb_search_is_ambiguous_and_takes_the_default() -> None:
    """``kb_`` is support and ``search`` is research, so this name ties.

    The specified keyword lists overlap here. Scoring both and falling back to the
    default is the specified behaviour, and it is the safe outcome: a tie must never
    be resolved in the aggressive direction.
    """
    scout = ScoutAgent()
    scores = scout.score_archetypes(with_tools("kb_search"))
    assert scores["support"] == 1
    assert scores["research"] == 1

    assert scout.run(with_tools("kb_search"), AUTO, "balanced").profile == "balanced"


def test_one_tool_can_score_for_two_archetypes() -> None:
    """A name matching several tables scores in each, and can cause a tie."""
    scores = ScoutAgent().score_archetypes(with_tools("invoice", "pdf"))
    assert scores["support"] == 1
    assert scores["research"] == 1
    assert scores["coding"] == 0


def test_zero_hits_falls_back_to_the_default() -> None:
    """No signal must not become a guess."""
    assert ScoutAgent().run(with_tools("frobnicate"), AUTO, "balanced").profile == "balanced"


def test_no_tool_messages_falls_back_to_the_default() -> None:
    """A conversation with no tools has nothing to score."""
    context = ScoutAgent().run([Message(role="user", content="hi")], AUTO, "balanced")
    assert context.profile == "balanced"


def test_a_tie_falls_back_to_the_default() -> None:
    """When no archetype dominates, defer rather than let dict order decide."""
    messages = with_tools("invoice", "pdf")
    scores = ScoutAgent().score_archetypes(messages)
    assert scores["support"] == scores["research"], "the fixture must be a real tie"

    assert ScoutAgent().run(messages, AUTO, "conservative").profile == "conservative"


def test_a_tie_does_not_return_a_profile_name() -> None:
    """The tie result is the default, never an arbitrary winner."""
    messages = with_tools("invoice", "pdf")
    assert ScoutAgent().run(messages, AUTO, "aggressive").profile == "aggressive"


def test_an_empty_default_is_returned_as_is() -> None:
    """An empty default must not crash or be replaced by a guess."""
    assert ScoutAgent().run(with_tools("frobnicate"), AUTO, "").profile == ""


# --------------------------------------------------------------------------- #
# Explicit profiles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_an_explicit_valid_profile_wins(profile: str) -> None:
    """An explicit request must never be second-guessed by the keyword count."""
    context = ScoutAgent().run(with_tools("read_file", "pytest"), profile, "conservative")
    assert context.profile == profile


def test_auto_is_not_returned_as_a_profile_name() -> None:
    """AUTO means guess, and must not be mistaken for a profile."""
    assert ScoutAgent().run(with_tools("pdf"), AUTO, "balanced").profile == "conservative"


def test_an_invalid_requested_profile_is_ignored() -> None:
    """A bogus name must fall through to guessing, not be trusted."""
    assert ScoutAgent().run(with_tools("read_file"), "wat", "balanced").profile == "aggressive"


def test_an_invalid_requested_profile_falls_back_to_the_default() -> None:
    """With no signal either, the default is the answer."""
    assert ScoutAgent().run(with_tools("frobnicate"), "wat", "conservative").profile == (
        "conservative"
    )


# --------------------------------------------------------------------------- #
# Goal
# --------------------------------------------------------------------------- #


def test_goal_joins_every_user_message() -> None:
    """All user turns are part of the goal, not just the first."""
    messages = [
        Message(role="user", content="fix the bug"),
        Message(role="assistant", content="working"),
        Message(role="user", content="and run the tests"),
    ]
    assert ScoutAgent().run(messages).goal == "fix the bug | and run the tests"


def test_goal_is_not_truncated_here() -> None:
    """Clipping belongs to build_state, so the limit stays in one place."""
    long_goal = "g" * 9000
    assert ScoutAgent().run([Message(role="user", content=long_goal)]).goal == long_goal


def test_goal_ignores_non_user_messages() -> None:
    """Only the human's own words are the goal."""
    messages = [
        Message(role="system", content="be helpful"),
        Message(role="user", content="the real goal"),
        Message(role="tool", content="noise", tool_call_id="c1", name="read_file"),
    ]
    assert ScoutAgent().run(messages).goal == "the real goal"


def test_goal_of_a_history_with_no_user_message_is_empty() -> None:
    """No human turn means no goal, not a guess."""
    assert ScoutAgent().run([Message(role="system", content="be helpful")]).goal == ""


def test_empty_user_messages_are_skipped() -> None:
    """An empty turn would otherwise produce a stray separator."""
    messages = [Message(role="user", content="a"), Message(role="user", content="")]
    assert ScoutAgent().run(messages).goal == "a"


# --------------------------------------------------------------------------- #
# Pinned tools
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "send_email",
        "pay_invoice",
        "refund_order",
        "delete_file",
        "create_ticket",
        "update_crm",
        "post_message",
        "submit_form",
        "charge_card",
    ],
)
def test_side_effecting_tools_are_pinned(name: str) -> None:
    """A call that changed the world must keep its output."""
    assert name in ScoutAgent().run(with_tools(name)).pinned_tools


def test_read_only_tools_are_not_pinned() -> None:
    """Reading a file removes nothing, so its output is safe to compact."""
    context = ScoutAgent().run(with_tools("read_file", "grep", "pdf"))
    assert context.pinned_tools == frozenset()


def test_a_read_only_tool_with_a_scary_word_is_pinned() -> None:
    """Substring matching is the specified rule, so this pins too."""
    context = ScoutAgent().run(with_tools("get_delete_policy"))
    assert context.pinned_tools == frozenset({"get_delete_policy"})


def test_pinned_tools_collect_several_names() -> None:
    """Every pinned tool in the session is reported, not just the first."""
    context = ScoutAgent().run(with_tools("read_file", "send_email", "refund_order"))
    assert context.pinned_tools == frozenset({"send_email", "refund_order"})


def test_pinned_tools_are_empty_without_tools() -> None:
    """Nothing to pin in a conversation with no tool calls."""
    context = ScoutAgent().run([Message(role="user", content="hi")])
    assert context.pinned_tools == frozenset()


def test_a_pinned_tool_does_not_change_the_profile() -> None:
    """Pinning is independent of profile selection."""
    context = ScoutAgent().run(with_tools("send_email", "read_file"), AUTO, "balanced")
    assert context.profile == "aggressive"
    assert context.pinned_tools == frozenset({"send_email"})


# --------------------------------------------------------------------------- #
# Shape and determinism
# --------------------------------------------------------------------------- #


def test_run_returns_a_session_context() -> None:
    """All three fields come back together."""
    context = ScoutAgent().run(with_tools("read_file", "send_email"), AUTO, "balanced")
    assert isinstance(context, SessionContext)
    assert context.profile == "aggressive"
    assert context.goal == "do the thing"
    assert context.pinned_tools == frozenset({"send_email"})


def test_run_is_deterministic() -> None:
    """The same history must always give the same answer."""
    scout = ScoutAgent()
    messages = with_tools("read_file", "pdf", "invoice")
    assert scout.run(messages, AUTO, "balanced") == scout.run(messages, AUTO, "balanced")


def test_archetype_profile_mapping_matches_the_spec() -> None:
    """The mapping is the domain configuration and must not drift."""
    assert ARCHETYPE_PROFILE == {
        "coding": "aggressive",
        "support": "balanced",
        "research": "conservative",
    }


# --------------------------------------------------------------------------- #
# Legacy helpers the pipeline still uses
# --------------------------------------------------------------------------- #


def test_legacy_goal_comes_from_the_user_message() -> None:
    """The goal is the pinned human request."""
    assert "rounding bug" in extract_goal(history(1))


def test_legacy_requested_profile_always_wins() -> None:
    """An explicit profile must never be second-guessed."""
    plan = pick_profile(history(1), "aggressive")
    assert plan.profile == "aggressive"
    assert plan.reason == "requested"


def test_legacy_unknown_requested_profile_is_not_trusted() -> None:
    """A bogus profile name falls through to a guess, not to aggressive."""
    plan = pick_profile([Message(role="user", content="fix the bug in the test")], "wat")
    assert plan.profile != "wat"


def test_should_compact_respects_the_trigger() -> None:
    """Small histories are left alone."""
    assert should_compact(history(1)) is True
    assert should_compact([Message(role="user", content="hi")]) is False


def test_first_sentence() -> None:
    """Used for a short goal preview."""
    assert first_sentence("Do this. Then that.") == "Do this."