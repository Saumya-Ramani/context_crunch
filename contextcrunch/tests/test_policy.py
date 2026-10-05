"""Tests for the policy engine.

These are the most important tests in the repo. The policy engine is the only code
that can destroy content, so every hard rule and every veto is pinned here.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from contextcrunch.core.policy import DROP, KEEP, PROFILES, TRUNCATE, Profile, decide, get_profile
from contextcrunch.laya.types import Answer

#: A long, old tool output: old enough to be judged, big enough to be worth judging.
OLD_TOOL_ITEM: dict[str, Any] = {
    "role": "tool",
    "tokens": 2000,
    "age_in_turns": 20,
    "tool_name": "read_file",
}

#: Answers that clear every gate, so a test can break exactly one and see the veto.
CLEAR_TO_DROP: dict[str, dict[str, Any]] = {
    "verdict": {"choice": "drop", "answer_confidence": 0.95},
    "essential": {"noul": 0.05},
    "consumed": {"noul": 0.95},
    "superseded": {"noul": 0.10},
    "relevance": {"score": 0.20},
}


def answer(question: str, **fields: Any) -> Answer:
    """Build one Laya answer for a test."""
    types = {"verdict": "choice", "relevance": "score"}
    return Answer(id=question, type=types.get(question, "noul"), **fields)


def answers(**overrides: dict[str, Any]) -> dict[str, Answer]:
    """Build the five-answer set, replacing whatever a test overrides.

    The starting point clears every gate, so a test that changes one number is
    testing exactly that number.
    """
    raw = {**CLEAR_TO_DROP, **overrides}
    return {question: answer(question, **fields) for question, fields in raw.items()}


def decide_with(
    profile: str = "conservative",
    *,
    item: dict[str, Any] | None = None,
    answers_: dict[str, Answer] | None = None,
    **kwargs: Any,
) -> str:
    """Decide one old tool output, breaking exactly what a test needs to break."""
    return decide(
        {**OLD_TOOL_ITEM, **(item or {})},
        answers_ or answers(),
        get_profile(profile),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Hard rules: content that is never judged
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_pinned_roles_are_always_kept(role: str) -> None:
    """A human turn or the system prompt cannot be re-issued, so never judged."""
    assert decide_with(item={"role": role}) == KEEP


def test_pinned_tool_is_kept_even_when_every_gate_passes() -> None:
    """A tool configured as pinned survives even a unanimous drop verdict."""
    assert decide_with(pinned_tools=frozenset({"read_file"})) == KEEP


def test_other_tools_are_not_pinned_by_accident() -> None:
    """Pinning one tool must not pin the rest."""
    assert decide_with(pinned_tools=frozenset({"other_tool"})) == DROP


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_recent_item_is_kept(profile: str) -> None:
    """Inside the protected window the agent is probably still using that output."""
    assert decide_with(profile, item={"age_in_turns": get_profile(profile).K}) == KEEP


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_tiny_item_is_kept(profile: str) -> None:
    """Below 40 tokens there is no meaningful saving to be had."""
    assert decide_with(profile, item={"tokens": 39}) == KEEP


# --------------------------------------------------------------------------- #
# Vetoes: one failed gate is enough to block a drop
# --------------------------------------------------------------------------- #


def test_clearing_every_gate_allows_the_drop() -> None:
    """The baseline case: nothing vetoes, so the item goes."""
    assert decide_with() == DROP


def test_essential_vetoes_the_drop() -> None:
    """Irreplaceable content stays, however confident the model is."""
    blocked = answers(essential={"noul": 0.60})
    assert decide_with(answers_=blocked) != DROP


def test_relevance_vetoes_the_drop() -> None:
    """Content the agent still needs stays."""
    blocked = answers(relevance={"score": 2.50})
    assert decide_with(answers_=blocked) != DROP


def test_confidence_vetoes_the_drop() -> None:
    """An unsure model cannot delete content."""
    blocked = answers(verdict={"choice": "drop", "answer_confidence": 0.70})
    assert decide_with("conservative", answers_=blocked) != DROP


def test_verdict_must_actually_be_drop() -> None:
    """Laya proposing to keep is honoured."""
    keep = answers(verdict={"choice": "keep", "answer_confidence": 0.95})
    assert decide_with(answers_=keep) != DROP


def test_no_junk_reason_vetoes_the_drop() -> None:
    """Nothing is consumed, nothing is superseded, and it is not noise.

    This is the veto that matters most in practice: without a positive reason to
    call something junk, a confident drop request is still refused.
    """
    no_reason = answers(
        consumed={"noul": 0.10},
        superseded={"noul": 0.10},
        relevance={"score": 2.50},
    )
    assert decide_with(answers_=no_reason) != DROP


@pytest.mark.parametrize("missing", ["verdict", "essential", "consumed", "relevance"])
def test_a_missing_answer_vetoes_the_drop(missing: str) -> None:
    """An unanswered question is a veto, never a silent pass.

    The logic indexes the answers it needs, so a missing key is a loud KeyError
    rather than a silent keep. That is deliberate: a caller that assembles a
    partial answer set has a bug, and quietly returning KEEP would hide it.
    """
    partial = answers()
    del partial[missing]
    with pytest.raises(KeyError):
        decide_with(answers_=partial)


def test_a_null_answer_is_read_as_no_answer() -> None:
    """An answer present but null is treated as no answer at all.

    ``superseded`` is the one junk reason that can be dropped without changing the
    outcome, because the other two reasons are still available. So the gate still
    fires here, and this test pins the difference between null and missing: null is
    read safely, missing is a KeyError.
    """
    nulled = answers(superseded={"noul": None}, consumed={"noul": 0.95})
    assert decide_with(answers_=nulled) == DROP


@pytest.mark.parametrize("field", ["noul", "score"])
def test_a_null_answer_vetoes_the_drop(field: str) -> None:
    """An answer present but null is treated as no answer at all.

    Only the answers that gate every outcome can be shown to veto here: the junk
    reasons are alternatives, so nulling one of them still leaves the others.
    """
    nulled = answers(**{"essential" if field == "noul" else "relevance": {field: None}})
    assert decide_with(answers_=nulled) != DROP


# --------------------------------------------------------------------------- #
# The three junk reasons, each sufficient on its own
# --------------------------------------------------------------------------- #


def test_consumed_alone_allows_the_drop() -> None:
    """The insight was already extracted into a later finding.

    ``superseded`` is the only other junk reason that can be switched off here:
    ``noise`` cannot be avoided without also failing the relevance gate, because a
    score low enough to count as noise is also low enough to clear ``rel``.
    """
    by_consumed = answers(
        consumed={"noul": 0.95},
        superseded={"noul": 0.10},
        relevance={"score": 0.90},
    )
    assert decide_with("conservative", answers_=by_consumed) == DROP


def test_superseded_alone_allows_the_drop() -> None:
    """A later item made this one obsolete."""
    by_superseded = answers(
        consumed={"noul": 0.10},
        superseded={"noul": 0.95},
        relevance={"score": 0.90},
    )
    assert decide_with("conservative", answers_=by_superseded) == DROP


@pytest.mark.parametrize("profile", ["aggressive", "balanced"])
def test_noise_alone_allows_the_drop(profile: str) -> None:
    """So irrelevant it is background at best, and nothing consumed it."""
    by_noise = answers(
        consumed={"noul": 0.10},
        superseded={"noul": 0.10},
        relevance={"score": get_profile(profile).noise},
    )
    assert decide_with(profile, answers_=by_noise) == DROP


def test_noise_needs_the_conservative_floor_to_agree() -> None:
    """Conservative sets the strictest noise bar, so a middling score is not junk."""
    middling = answers(
        consumed={"noul": 0.10},
        superseded={"noul": 0.10},
        relevance={"score": 0.70},
    )
    assert decide_with("conservative", answers_=middling) != DROP
    assert decide_with("aggressive", answers_=middling) == DROP


# --------------------------------------------------------------------------- #
# is_sub: a piece of an item already on trial
# --------------------------------------------------------------------------- #


def test_is_sub_skips_the_recency_rule() -> None:
    """A recent sub-item is judged on its own merits, not its position."""
    assert decide_with(item={"age_in_turns": 1}, is_sub=True) == DROP


def test_is_sub_skips_the_tiny_item_rule() -> None:
    """A short passage inside a filing is still worth judging."""
    assert decide_with(item={"tokens": 1}, is_sub=True) == DROP


def test_is_sub_still_honours_the_pinned_role_rule() -> None:
    """is_sub relaxes recency and size only, never the role pin."""
    assert decide_with(item={"role": "user"}, is_sub=True) == KEEP


def test_is_sub_never_gets_the_size_truncation() -> None:
    """A small piece inside a big item has nothing to truncate against."""
    small_sub = {"tokens": 900}
    keep = answers(verdict={"choice": "keep", "answer_confidence": 0.95})
    assert decide_with(item=small_sub, answers_=keep, is_sub=True) != TRUNCATE
    assert decide_with(item=small_sub, answers_=keep) == TRUNCATE


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #


def test_truncate_verdict_is_honoured() -> None:
    """Laya recommending a partial keep is followed."""
    assert decide_with(answers_=answers(verdict={"choice": "truncate"})) == TRUNCATE


def test_big_item_with_keep_verdict_is_truncated() -> None:
    """900 tokens is too much to keep whole, whatever the model says."""
    keep = answers(verdict={"choice": "keep", "answer_confidence": 0.95})
    assert decide_with(item={"tokens": 900}, answers_=keep) == TRUNCATE


def test_a_veto_can_still_end_in_truncate() -> None:
    """A veto is not a promise to keep the item whole.

    The essential gate blocked the drop, but the item is still oversized, so it is
    clipped rather than deleted. That is the difference between the two actions:
    truncation is reversible through the Storage Box, a drop has to be recovered.
    """
    blocked = answers(essential={"noul": 0.90}, verdict={"choice": "keep"})
    assert decide_with(item={"tokens": 5000}, answers_=blocked) == TRUNCATE


# --------------------------------------------------------------------------- #
# The three profiles
# --------------------------------------------------------------------------- #


def test_there_are_exactly_three_profiles() -> None:
    """The domain configuration is three rows and nothing more."""
    assert set(PROFILES) == {"aggressive", "balanced", "conservative"}


def test_profiles_differ_on_the_same_borderline_answers() -> None:
    """The profile row is the only thing that changes between archetypes.

    These answers clear the aggressive gates and fail the conservative ones, so the
    same Laya result produces three different actions. The item is 800 tokens so the
    size rule never fires and the verdict difference is not masked by a truncation.
    """
    borderline = answers(
        verdict={"choice": "drop", "answer_confidence": 0.95},
        essential={"noul": 0.45},
        relevance={"score": 0.90},
        consumed={"noul": 0.85},
        superseded={"noul": 0.10},
    )
    results = {name: decide_with(name, item={"tokens": 800}, answers_=borderline)
               for name in PROFILES}
    assert results == {
        "aggressive": DROP,
        "balanced": KEEP,
        "conservative": KEEP,
    }


def test_conservative_is_the_strictest_on_every_gate() -> None:
    """Each profile's gates must be ordered, or "conservative" means nothing."""
    aggressive, balanced, conservative = (PROFILES[name] for name in
                                          ("aggressive", "balanced", "conservative"))
    assert aggressive.ess > balanced.ess > conservative.ess
    assert aggressive.rel >= balanced.rel >= conservative.rel
    assert aggressive.con < balanced.con < conservative.con
    assert aggressive.conf < balanced.conf < conservative.conf
    assert aggressive.noise > balanced.noise > conservative.noise
    assert conservative.anchors is True
    assert aggressive.anchors is False


def test_unknown_profile_falls_back_to_the_safest() -> None:
    """A typo in the profile name must not open the door to aggressive drops."""
    assert get_profile("nope") is get_profile("conservative")


def test_profile_is_frozen() -> None:
    """A profile row must not be mutable at runtime.

    A threshold edited mid-request would make a decision unreproducible, and the
    Storage Box could no longer explain why an item was dropped.
    """
    with pytest.raises(FrozenInstanceError):
        get_profile("balanced").ess = 0.0  # type: ignore[misc]


def test_profiles_are_comparable() -> None:
    """Frozen dataclasses compare by value, which the tests above rely on."""
    row = Profile(ess=0.4, rel=1.0, con=0.85, conf=0.8, noise=0.6, K=4, anchors=False)
    assert row == PROFILES["balanced"]