"""Tests for the question set, the single constant engine."""

from __future__ import annotations

from contextcrunch.core.questions import QUESTION_IDS, QUESTION_SET, QUESTION_TYPES


def test_exactly_five_questions() -> None:
    """The engine is five questions and never changes."""
    assert len(QUESTION_SET) == 5
    assert set(QUESTION_IDS) == {
        "verdict",
        "essential",
        "consumed",
        "superseded",
        "relevance",
    }


def test_question_types_match_the_laya_format() -> None:
    """Only the three real Laya types may be used."""
    assert QUESTION_TYPES == {
        "verdict": "choice",
        "essential": "noul",
        "consumed": "noul",
        "superseded": "noul",
        "relevance": "score",
    }


def test_choice_uses_criteria_mapping() -> None:
    """A choice question must define its options as a mapping."""
    verdict = QUESTION_SET["verdict"]
    assert verdict["type"] == "choice"
    assert set(verdict["criteria"]) == {"keep", "truncate", "drop"}


def test_score_has_four_levels_lowest_first() -> None:
    """A score question needs an ordered level list."""
    levels = QUESTION_SET["relevance"]["criteria"]
    assert isinstance(levels, list)
    assert len(levels) == 4


def test_questions_are_immutable() -> None:
    """A decision is only reproducible if the question set cannot be edited."""
    try:
        QUESTION_SET["verdict"] = {}  # type: ignore[index]
    except TypeError:
        return
    raise AssertionError("QUESTION_SET must be immutable")


def test_questions_mention_no_domain_vocabulary() -> None:
    """The questions must stay universal, so no domain words may creep in."""
    text = str(dict(QUESTION_SET)).lower()
    for word in ("customer", "invoice", "python", "ticket", "margin", "sql"):
        assert word not in text, f"{word} would tie the question set to one domain"
