"""The five questions asked about every history item.

These are the only questions ContextCrunch ever asks. They probe structural
relations between an item, the goal and the findings that came after it, so the
same set works for a coding agent, a support agent and a research agent.

Wording is copied from the Laya question format: ``choice`` uses ``criteria``
(a mapping), ``noul`` takes no options, and ``score`` uses a list of level
descriptions, most important last.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

#: The five questions (v1 - current). Immutable: a decision is only reproducible if the set is fixed.
QUESTION_SET_V1: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "verdict": {
            "type": "choice",
            "instructions": "What should happen to this history item for the ongoing goal?",
            "criteria": {
                "keep": "still needed: the agent will need this exact item to finish the goal",
                "truncate": "partly needed: the useful part should be kept, the rest removed",
                "drop": "no longer needed: a short note is enough, the original is not",
            },
        },
        "essential": {
            "type": "noul",
            "instructions": (
                "Does this item contain information the agent could not get back if it "
                "was removed, such as unique findings, exact numbers, ids, file paths or "
                "quotes that are not repeated anywhere else?"
            ),
        },
        "consumed": {
            "type": "noul",
            "instructions": (
                "Has the useful information in this item already been extracted into one "
                "of the later findings shown in the state, so the original text is no "
                "longer needed?"
            ),
        },
        "superseded": {
            "type": "noul",
            "instructions": (
                "Has this item been made obsolete or duplicated by a later item, for "
                "example by a newer run, a fresher fetch or a corrected version?"
            ),
        },
        "relevance": {
            "type": "score",
            "instructions": "How relevant is this item to the current goal?",
            "criteria": [
                "noise or boilerplate with no future use",
                "background, unlikely to be needed again",
                "supporting, may be referenced later",
                "critical, the agent needs this to finish the goal",
            ],
        },
    }
)

#: The five questions (v2 - sharper wording). Same five ids and score levels.
QUESTION_SET_V2: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "verdict": {
            "type": "choice",
            "instructions": "What should happen to this history item for the ongoing goal?",
            "criteria": {
                "keep": "still needed: the agent will need this exact item to finish the goal",
                "truncate": "partly needed: the useful part should be kept, the rest removed",
                "drop": "no longer needed: a short note is enough, the original is not",
            },
        },
        "essential": {
            "type": "noul",
            "instructions": (
                "This item contains a specific fact, number, quote or identifier that is not "
                "repeated in the later findings and that the goal needs."
            ),
        },
        "consumed": {
            "type": "noul",
            "instructions": (
                "The later findings already state everything from this item that the goal needs."
            ),
        },
        "superseded": {
            "type": "noul",
            "instructions": (
                "A later item replaces this item, so this item is outdated."
            ),
        },
        "relevance": {
            "type": "score",
            "instructions": "How much does the goal still depend on this item?",
            "criteria": [
                "noise or boilerplate with no future use",
                "background, unlikely to be needed again",
                "supporting, may be referenced later",
                "critical, the agent needs this to finish the goal",
            ],
        },
    }
)

#: Available question variants.
QUESTION_VARIANTS: Mapping[str, Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        "v1": QUESTION_SET_V1,
        "v2": QUESTION_SET_V2,
    }
)

#: The active question set (selected by CC_QUESTION_VARIANT, default v1).
QUESTION_SET = QUESTION_SET_V1

#: Question id to declared type, used to type the answers Laya returns.
QUESTION_TYPES: Mapping[str, str] = MappingProxyType(
    {question_id: str(spec["type"]) for question_id, spec in QUESTION_SET.items()}
)

#: The five questions, in a plain dict for backends that need a mutable copy.
QUESTION_IDS: tuple[str, ...] = tuple(QUESTION_SET)


def get_question_set(variant: str = "v1") -> Mapping[str, Mapping[str, Any]]:
    """Return the question set for the given variant."""
    return QUESTION_VARIANTS.get(variant, QUESTION_SET_V1)


def get_question_types(variant: str = "v1") -> Mapping[str, str]:
    """Return the question types for the given variant."""
    question_set = get_question_set(variant)
    return MappingProxyType(
        {question_id: str(spec["type"]) for question_id, spec in question_set.items()}
    )
