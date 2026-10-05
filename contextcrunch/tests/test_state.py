"""Tests for the state builder and the two-phase size fitting."""

from __future__ import annotations

import json

import pytest
from tests.conftest import history, tool_message

from contextcrunch.core.messages import Message
from contextcrunch.core.questions import QUESTION_SET
from contextcrunch.core.settings import Settings, get_settings
from contextcrunch.core.state import (
    MIN_ITEM_CHARS,
    build_state,
    fit_state,
    goal_from_messages,
    later_findings,
    reviewable_indices,
)
from contextcrunch.core.tokens import count_tokens


def _settings(**overrides: object) -> Settings:
    """Return the test settings with ``overrides`` applied.

    The question set costs about 300 tokens on its own and is sent on every call, so
    the totals here are measured against that rather than guessed.
    """
    base = {
        field: getattr(get_settings(), field)
        for field in Settings.model_fields
        if field != "laya_api_key"
    }
    return Settings(**{**base, "laya_api_key": "", **overrides})


#: Tight enough that the findings must go, loose enough that the item does not have
#: to shrink. Measured: the findings-first fixture costs 632 tokens with its findings
#: and 514 without them, so 570 sits between the two and isolates the ordering rule.
TIGHT = _settings(state_max_tokens=570, findings_cap_chars=400, item_max_chars=1200)


def state_cost(state: dict) -> int:
    """Return the token cost of a state plus the question set sent with it."""
    return count_tokens(json.dumps(state)) + count_tokens(json.dumps(dict(QUESTION_SET)))


# --------------------------------------------------------------------------- #
# build_state
# --------------------------------------------------------------------------- #


def test_state_carries_the_four_relations() -> None:
    """The state must expose goal, next step, item and later findings."""
    state = build_state(history(2), 3, "fix the bug", get_settings())
    assert set(state) == {"goal", "next_step", "item", "later_findings"}


def test_item_describes_the_message_on_trial() -> None:
    """The item block must describe the message being judged."""
    messages = history(2)
    item = build_state(messages, 3, "fix the bug", get_settings())["item"]

    assert item["index"] == 3
    assert item["role"] == "tool"
    assert item["age_in_turns"] == len(messages) - 3
    assert item["tokens"] > 0


def test_tool_call_uses_the_fingerprint() -> None:
    """The tool name is the recoverability signal the policy reasons about."""
    messages = [Message(role="user", content="go"), tool_message("data", 1)]
    state = build_state(messages, 1, "go", get_settings())
    assert state["item"]["tool_call"] == "read_file"


def test_fingerprint_of_an_assistant_tool_call() -> None:
    """An assistant message points at the function it invoked."""
    message = Message(
        role="assistant",
        content="",
        tool_calls=[{"id": "c1", "function": {"name": "run_tests"}}],
    )
    assert message.fingerprint() == "run_tests"


def test_next_step_is_the_last_message_when_the_assistant_spoke_last() -> None:
    """next_step is the future-need signal."""
    state = build_state(history(1), 3, "fix the bug", get_settings())
    assert state["next_step"] == "finding 0"


def test_next_step_is_empty_when_the_last_message_is_a_tool() -> None:
    """A tool result is not a stated next step."""
    messages = [Message(role="user", content="go"), tool_message("data", 1)]
    assert build_state(messages, 1, "go", get_settings())["next_step"] == ""


def test_next_step_is_capped() -> None:
    """A long final message must not eat the whole state budget."""
    messages = [Message(role="assistant", content="n" * 5000)]
    state = build_state(messages, 0, "goal", get_settings())
    assert len(state["next_step"]) == 300


def test_content_can_be_overridden_for_one_piece() -> None:
    """The refiner reuses this for a piece of an item."""
    messages = history(1)
    state = build_state(messages, 3, "goal", get_settings(), content="just this piece")
    assert state["item"]["content"] == "just this piece"


def test_item_content_is_truncated_but_tokens_report_the_full_size() -> None:
    """Laya must be able to see that an item is huge without being sent all of it."""
    settings = get_settings()
    messages = history(1, filler="y" * 50000)
    item = build_state(messages, 3, "goal", settings)["item"]

    assert len(item["content"]) <= settings.item_max_chars
    full_tokens = count_tokens(messages[3].content)
    assert item["tokens"] == full_tokens
    assert item["tokens"] > count_tokens(item["content"])


def test_goal_is_clipped() -> None:
    """The goal must fit, or nothing else can be judged against it."""
    settings = get_settings()
    state = build_state(history(1), 3, "g" * 5000, settings)
    assert len(state["goal"]) == settings.goal_max_chars


# --------------------------------------------------------------------------- #
# later_findings
# --------------------------------------------------------------------------- #


def test_findings_are_only_later_assistant_messages() -> None:
    """later_findings is what makes the 'consumed' question answerable."""
    findings = later_findings(history(3), 3, 4000)
    assert findings
    assert all(text.startswith("finding") for text in findings)


def test_findings_are_nearest_first() -> None:
    """The most recent conclusion is the one that says what was already extracted."""
    messages = [
        Message(role="user", content="go"),
        Message(role="tool", content="data", tool_call_id="c1"),
        Message(role="assistant", content="older finding"),
        Message(role="assistant", content="newest finding"),
    ]
    assert later_findings(messages, 1, 4000) == ["newest finding", "older finding"]


def test_findings_respect_the_cap_without_truncating() -> None:
    """The list stops before the cap, so no entry is ever a half-finding."""
    messages = [
        Message(role="user", content="go"),
        Message(role="tool", content="data", tool_call_id="c1"),
        Message(role="assistant", content="x" * 60),
        Message(role="assistant", content="y" * 60),
    ]
    findings = later_findings(messages, 1, 100)

    assert findings == ["y" * 60]
    assert sum(len(text) for text in findings) <= 100


def test_findings_skip_earlier_assistant_messages() -> None:
    """A conclusion formed before the item cannot say it was consumed."""
    messages = [
        Message(role="assistant", content="before the item"),
        Message(role="tool", content="data", tool_call_id="c1"),
    ]
    assert later_findings(messages, 1, 4000) == []


def test_findings_skip_tool_output() -> None:
    """Only the agent's own distilled conclusions count as findings."""
    messages = [
        Message(role="tool", content="data", tool_call_id="c1"),
        Message(role="tool", content="more data", tool_call_id="c2"),
    ]
    assert later_findings(messages, 0, 4000) == []


# --------------------------------------------------------------------------- #
# fit_state
# --------------------------------------------------------------------------- #


def test_a_state_within_budget_is_untouched() -> None:
    """Fitting must not clip anything that already fits."""
    settings = get_settings()
    state = build_state(history(1), 3, "goal", settings)
    before = json.dumps(state, sort_keys=True)
    assert json.dumps(fit_state(state, settings), sort_keys=True) == before


def test_fit_state_shrinks_findings_before_the_item() -> None:
    """A clipped finding is worse than no finding, so findings go first.

    The item is sized so that dropping all three findings brings the state under
    budget on its own. If the item still had to shrink, that would prove nothing
    about the ordering. Text is varied rather than one repeated character, because a
    run of identical characters compresses to far fewer tokens than its length
    suggests and would quietly weaken the fixture.
    """
    filler = "the quick brown fox jumps over the lazy dog " * 20
    state = {
        "goal": "assess the margin",
        "next_step": "",
        "item": {"content": filler, "tokens": count_tokens(filler)},
        "later_findings": ["a" * 300, "b" * 300, "c" * 300],
    }
    assert state_cost(state) > TIGHT.state_max_tokens, "the fixture must start over budget"

    fitted = fit_state(state, TIGHT)

    # Enough findings go that the item never has to shrink. The last one standing is
    # the nearest, because the furthest is always dropped first.
    assert len(fitted["later_findings"]) < 3
    assert fitted["item"]["content"] == filler, "the item must be untouched"
    assert state_cost(fitted) <= TIGHT.state_max_tokens


def test_fit_state_drops_the_furthest_finding_first() -> None:
    """The nearest finding is kept longest: it is the most informative."""
    state = {
        "goal": "assess the margin",
        "next_step": "",
        "item": {"content": "z" * 300, "tokens": 300},
        "later_findings": ["nearest", "filler " * 30, "far " * 30, "furthest " * 30],
    }
    fitted = fit_state(state, TIGHT)
    assert fitted["later_findings"][0] == "nearest"


def test_fit_state_shrinks_the_item_once_findings_run_out() -> None:
    """The item is the last thing to go, and only in steps."""
    state = {
        "goal": "assess the margin",
        "next_step": "",
        "item": {"content": "z" * 9000, "tokens": 9000},
        "later_findings": [],
    }
    fitted = fit_state(state, _settings(state_max_tokens=1200))

    assert fitted["later_findings"] == []
    assert len(fitted["item"]["content"]) < 9000
    assert len(fitted["item"]["content"]) >= MIN_ITEM_CHARS
    assert state_cost(fitted) <= 1200


def test_fit_state_never_cuts_the_goal() -> None:
    """Without a goal every relevance judgement is meaningless."""
    state = {
        "goal": "assess the margin " * 50,
        "next_step": "",
        "item": {"content": "z" * 9000, "tokens": 9000},
        "later_findings": [],
    }
    original_goal = state["goal"]
    fit_state(state, TIGHT)
    assert state["goal"] == original_goal


def test_fit_state_stops_at_the_item_floor() -> None:
    """Shrinking must not empty the item, however tight the budget gets."""
    state = {
        "goal": "g",
        "next_step": "",
        "item": {"content": "z" * 5000, "tokens": 5000},
        "later_findings": [],
    }
    fitted = fit_state(state, _settings(state_max_tokens=1))
    assert len(fitted["item"]["content"]) == MIN_ITEM_CHARS


def test_fit_state_leaves_tokens_reporting_the_full_item() -> None:
    """Fitting must not make Laya think a huge item is a small one."""
    state = {
        "goal": "g",
        "next_step": "",
        "item": {"content": "z" * 9000, "tokens": 9000},
        "later_findings": [],
    }
    assert fit_state(state, TIGHT)["item"]["tokens"] == 9000


def test_fit_state_tolerates_a_state_with_no_item() -> None:
    """Defensive: a malformed state must not raise inside the pipeline."""
    assert fit_state({"goal": "g"}, TIGHT) == {"goal": "g"}


def test_fit_state_replaces_a_non_list_findings_field() -> None:
    """A malformed findings field is repaired rather than crashing."""
    state = {"goal": "g", "next_step": "", "item": {"content": "z", "tokens": 1},
             "later_findings": None}
    assert fit_state(state, TIGHT)["later_findings"] == []


@pytest.mark.parametrize(
    ("size", "filler"),
    [(20000, "y" * 20000), (60000, "z" * 60000)],
    ids=["20k", "60k"],
)
def test_build_state_always_fits_its_budget(size: int, filler: str) -> None:
    """End to end: however large the history, the state fits the window."""
    settings = get_settings()
    state = build_state(history(1, filler=filler), 3, "goal" * 200, settings)
    assert state_cost(state) <= settings.state_max_tokens
    assert len(state["item"]["content"]) >= MIN_ITEM_CHARS


# --------------------------------------------------------------------------- #
# Helpers that did not change
# --------------------------------------------------------------------------- #


def test_reviewable_excludes_pinned_roles() -> None:
    """System and user messages are never judged."""
    indices = reviewable_indices(history(2))
    assert 0 not in indices
    assert 1 not in indices


def test_goal_is_taken_from_the_first_user_message() -> None:
    """The goal is pinned to the human request."""
    assert goal_from_messages(history(1)) == "Fix the rounding bug in the billing service."