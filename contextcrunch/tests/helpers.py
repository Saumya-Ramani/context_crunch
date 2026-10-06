"""Test helpers for ContextCrunch.

This module provides testing utilities that do NOT depend on any model or
markers. It lives ONLY under tests/ and must not be importable from src/.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from contextcrunch.core.questions import QUESTION_TYPES
from contextcrunch.laya.engine import LayaEngine
from contextcrunch.laya.types import LayaResult, parse_result


class ScriptedEngine(LayaEngine):
    """Test engine that returns pre-scripted answers.

    This engine takes an explicit callable or dict mapping a substring of
    state["item"]["content"] to an answers dict. It is for TESTS ONLY and
    must not be used in production code.
    """

    def __init__(
        self,
        answers: dict[str, dict[str, Any]] | Callable[[dict[str, Any]], dict[str, Any]],
        settings: Any = None,
        delay: float = 0.0,
    ) -> None:
        """Create a scripted engine.

        Args:
            answers: Either a dict mapping content substrings to answer dicts,
                     or a callable that takes a state and returns an answer dict.
            settings: Optional settings object (for concurrency limit).
            delay: Optional delay in seconds for each call (for timeout testing).
        """
        # Don't call parent __init__ - we don't need a real client
        self._answers_map = answers
        self._settings = settings
        self._call_count = 0
        self._delay = delay

    async def decide(
        self, states: Sequence[dict[str, Any]], labels: Sequence[str] | None = None
    ) -> list[LayaResult]:
        """Return scripted answers for all states."""
        if not states:
            return []

        results = []
        for state in states:
            self._call_count += 1
            if self._delay:
                await asyncio.sleep(self._delay)
            content = str(state.get("item", {}).get("content", ""))

            if callable(self._answers_map):
                answer = self._answers_map(state)
            else:
                # Find matching key in the map
                answer = None
                for key, value in self._answers_map.items():
                    if key in content:
                        answer = value
                        break
                if answer is None:
                    # Default to a safe "keep" answer
                    answer = _make_keep_payload()

            # If answer is already a LayaResult, use it directly; otherwise parse it
            if isinstance(answer, LayaResult):
                results.append(answer)
            else:
                results.append(parse_result(answer, dict(QUESTION_TYPES)))

        return results

    async def predict(self, state: dict[str, Any], label: str = "") -> LayaResult:
        """Single-state predict for backward compatibility."""
        results = await self.decide([state], [label])
        return results[0] if results else parse_result(_make_keep_payload(), dict(QUESTION_TYPES))

    async def aclose(self) -> None:
        """No resources to release."""
        pass

    @property
    def call_count(self) -> int:
        """Return the number of times decide was called."""
        return self._call_count


def _make_keep_payload() -> dict[str, Any]:
    """Return a payload that parses to a 'keep' LayaResult."""
    return {
        "model": "scripted",
        "answers": {
            "verdict": {
                "type": "choice",
                "choice": "keep",
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "probabilities": {"keep": 0.8, "truncate": 0.1, "drop": 0.1},
                "action": {"act_probability": 1.0},
            },
            "essential": {
                "type": "noul",
                "noul": 0.1,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "action": {"act_probability": 1.0},
            },
            "consumed": {
                "type": "noul",
                "noul": 0.1,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "action": {"act_probability": 1.0},
            },
            "superseded": {
                "type": "noul",
                "noul": 0.1,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "action": {"act_probability": 1.0},
            },
            "relevance": {
                "type": "score",
                "score": 2.5,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "probabilities": {"0": 0.0, "1": 0.0, "2": 0.3, "3": 0.7},
                "legend": {
                    "0": "noise",
                    "1": "background",
                    "2": "supporting",
                    "3": "critical",
                },
                "action": {"act_probability": 1.0},
            },
        },
    }


def _make_drop_payload() -> dict[str, Any]:
    """Return a payload that parses to a 'drop' LayaResult."""
    return {
        "model": "scripted",
        "answers": {
            "verdict": {
                "type": "choice",
                "choice": "drop",
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "probabilities": {"keep": 0.1, "truncate": 0.1, "drop": 0.8},
                "action": {"act_probability": 1.0},
            },
            "essential": {
                "type": "noul",
                "noul": 0.1,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "action": {"act_probability": 1.0},
            },
            "consumed": {
                "type": "noul",
                "noul": 0.9,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "action": {"act_probability": 1.0},
            },
            "superseded": {
                "type": "noul",
                "noul": 0.9,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "action": {"act_probability": 1.0},
            },
            "relevance": {
                "type": "score",
                "score": 0.2,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "probabilities": {"0": 0.8, "1": 0.15, "2": 0.05, "3": 0.0},
                "legend": {
                    "0": "noise",
                    "1": "background",
                    "2": "supporting",
                    "3": "critical",
                },
                "action": {"act_probability": 1.0},
            },
        },
    }


def _make_truncate_payload() -> dict[str, Any]:
    """Return a payload that parses to a 'truncate' LayaResult."""
    return {
        "model": "scripted",
        "answers": {
            "verdict": {
                "type": "choice",
                "choice": "truncate",
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "probabilities": {"keep": 0.1, "truncate": 0.8, "drop": 0.1},
                "action": {"act_probability": 1.0},
            },
            "essential": {
                "type": "noul",
                "noul": 0.3,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "action": {"act_probability": 1.0},
            },
            "consumed": {
                "type": "noul",
                "noul": 0.5,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "action": {"act_probability": 1.0},
            },
            "superseded": {
                "type": "noul",
                "noul": 0.3,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "action": {"act_probability": 1.0},
            },
            "relevance": {
                "type": "score",
                "score": 1.5,
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "probabilities": {"0": 0.1, "1": 0.3, "2": 0.4, "3": 0.2},
                "legend": {
                    "0": "noise",
                    "1": "background",
                    "2": "supporting",
                    "3": "critical",
                },
                "action": {"act_probability": 1.0},
            },
        },
    }


def make_keep_answer() -> dict[str, Any]:
    """Return a standard 'keep' answer payload for testing."""
    return _make_keep_payload()


def make_drop_answer() -> dict[str, Any]:
    """Return a standard 'drop' answer payload for testing."""
    return _make_drop_payload()


def make_truncate_answer() -> dict[str, Any]:
    """Return a standard 'truncate' answer payload for testing."""
    return _make_truncate_payload()