"""Shared test fixtures.

Tests never load the real model. They run against the fake backend, or against
a scripted backend that returns exactly the answers a test needs, so a policy
test asserts on code rather than on a model's mood.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from contextcrunch.core.messages import Message
from contextcrunch.core.questions import QUESTION_TYPES
from contextcrunch.core.settings import Settings, reset_settings_cache
from contextcrunch.laya.types import LayaResult, parse_result

#: Every tunable the tests need, so a test never depends on the developer's .env.
BASE_ENV: dict[str, str] = {
    "CC_LAYA_MODE": "fake",
    "CC_LAYA_MODEL": "convaiinnovations/laya",
    "CC_LAYA_API_URL": "https://api.example.test/v1/systemone",
    "CC_LAYA_API_KEY": "test-key",
    "CC_LAYA_MAX_LEN": "8192",
    "CC_LAYA_CONCURRENCY": "4",
    "CC_LAYA_RETRIES": "1",
    "CC_LAYA_BACKOFF_S": "0",
    "CC_LAYA_HTTP_BASE_URL": "http://127.0.0.1:8001",
    "CC_TIKTOKEN_ENCODING": "cl100k_base",
    "CC_ITEM_MAX_CHARS": "12000",
    "CC_FINDINGS_CAP_CHARS": "4000",
    "CC_GOAL_MAX_CHARS": "1000",
    "CC_STATE_MAX_TOKENS": "6000",
    "CC_PIECE_TARGET_CHARS": "2500",
    "CC_PIECE_MAX_CHARS": "3500",
    "CC_TRIGGER_TOKENS": "200",
    "CC_DEADLINE_S": "30",
    "CC_DEFAULT_PROFILE": "conservative",
    "CC_DB_PATH": "data/contextcrunch.db",
    "CC_ORIGINALS_DIR": "data/originals",
    # The real key must never leak into a test, even if .env is present.
    "Laya_Api_key": "",
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give every test a known configuration backed by a temporary data dir.

    The variables are also removed from the process environment first, so a real
    .env file in the working tree cannot change a test's result.
    """
    for key in BASE_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in BASE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CC_DB_PATH", str(tmp_path / "box.db"))
    monkeypatch.setenv("CC_ORIGINALS_DIR", str(tmp_path / "originals"))
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings_factory(monkeypatch: pytest.MonkeyPatch):
    """Return a factory that builds a real ``Settings`` with chosen overrides.

    Overrides go through the environment because that is where ``Settings`` reads
    its values from. The store paths are deliberately left alone: the autouse
    ``clean_env`` fixture has already pointed them at ``tmp_path``, and rewriting
    them here would make a test write its databases into the repo.
    """

    def build(**overrides: object) -> Settings:
        for name, value in overrides.items():
            monkeypatch.setenv(f"CC_{name.upper()}", str(value))
        reset_settings_cache()
        return Settings()

    return build


def answers(
    *,
    verdict: str = "keep",
    verdict_conf: float = 0.95,
    essential: float = 0.1,
    consumed: float = 0.2,
    superseded: float = 0.1,
    relevance: float = 2.5,
    relevance_probs: dict[str, float] | None = None,
) -> LayaResult:
    """Build a scripted Laya result with the five answers a policy test needs.

    The defaults describe a plainly relevant, unconsumed item that should be kept
    as-is, so a test that changes one number is testing exactly that number.
    """
    return parse_result(
        {
            "model": "scripted",
            "answers": {
                "verdict": {
                    "type": "choice",
                    "choice": verdict,
                    "confidence": verdict_conf,
                    "answer_confidence": verdict_conf,
                    "probabilities": {
                        "keep": 0.1 if verdict == "drop" else 0.8,
                        "truncate": 0.1,
                        "drop": 0.8 if verdict == "drop" else 0.1,
                    },
                    "action": {"act_probability": 1.0},
                },
                "essential": {
                    "type": "noul",
                    "noul": essential,
                    "answer_confidence": verdict_conf,
                    "action": {"act_probability": 1.0},
                },
                "consumed": {
                    "type": "noul",
                    "noul": consumed,
                    "answer_confidence": verdict_conf,
                    "action": {"act_probability": 1.0},
                },
                "superseded": {
                    "type": "noul",
                    "noul": superseded,
                    "answer_confidence": verdict_conf,
                    "action": {"act_probability": 1.0},
                },
                "relevance": {
                    "type": "score",
                    "score": relevance,
                    "answer_confidence": verdict_conf,
                    "probabilities": relevance_probs
                    or {"0": 0.7, "1": 0.2, "2": 0.1, "3": 0.0},
                    "action": {"act_probability": 1.0},
                },
            },
        },
        dict(QUESTION_TYPES),
    )


class ScriptedBackend:
    """A Laya backend that returns pre-canned answers, one call at a time."""

    def __init__(self, *results: LayaResult) -> None:
        self.queue = list(results)
        self.calls: list[dict[str, Any]] = []

    async def predict(self, state: dict[str, Any], label: str = "") -> LayaResult:
        """Return the next scripted answer, or a default one when exhausted."""
        self.calls.append({"state": state, "label": label})
        if self.queue:
            return self.queue.pop(0)
        return answers()

    async def aclose(self) -> None:
        """Nothing to release."""


@pytest.fixture
def scripted() -> type[ScriptedBackend]:
    """Return the scripted backend class."""
    return ScriptedBackend


#: A realistic tool output. Real code resists BPE compression, which is what a
#: real trace looks like, so the fixtures must not be a run of one character.
CODE_OUTPUT = "def round_price(value):\n    return int(value) + 1\n"

#: A long, structured report, for tests that need several distinct paragraphs.
#: Deliberately larger than CC_PIECE_TARGET_CHARS, because a smaller report merges
#: into a single piece and a test that expects several pieces would then pass
#: without exercising the split at all.
REPORT_OUTPUT = "\n\n".join(
    f"## Section {index}\n" + f"finding detail line {index} {index * 7}\n" * 40
    for index in range(4)
)


def tool_message(text: str, index: int = 1) -> Message:
    """Return a tool message for fixtures."""
    return Message(
        role="tool", content=text, tool_call_id=f"call_{index}", name="read_file"
    )


def history(count: int = 6, filler: str = CODE_OUTPUT * 20) -> list[Message]:
    """Return a small but realistic history: goal, plan, tool output, finding."""
    messages: list[Message] = [
        Message(role="system", content="You are a coding agent."),
        Message(role="user", content="Fix the rounding bug in the billing service."),
        Message(role="assistant", content="Plan: read the file, patch it, run the tests."),
    ]
    for index in range(count):
        messages.append(tool_message(filler, index))
        messages.append(Message(role="assistant", content=f"finding {index}"))
    return messages
