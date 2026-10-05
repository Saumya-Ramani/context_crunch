"""The only place ContextCrunch talks to a model.

Four backends sit behind one small interface:

- ``http`` (the default) calls the hosted inference API with a bearer token.
  This is the fast path, so it is the one the service uses;
- ``serve`` calls a self-hosted ``python -m laya.serve`` instance;
- ``inprocess`` runs the model in this process. It needs no network but it is
  slow, because it pays model load and inference cost on every call;
- ``fake`` returns deterministic answers so tests never load a model.

Every backend honours the token budget, the question set and the concurrency
limit from :mod:`contextcrunch.core.settings`.
"""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from contextcrunch.core.questions import QUESTION_SET, QUESTION_TYPES
from contextcrunch.core.settings import Settings, get_settings
from contextcrunch.laya.hosted import HostedBackend
from contextcrunch.laya.types import LayaResult, parse_result

if TYPE_CHECKING:
    from contextcrunch.laya.engine import LayaEngine


class LayaUnavailable(RuntimeError):
    """Raised when the configured Laya backend cannot be used."""


class LayaClient(ABC):
    """Judge a state against the question set and return typed answers."""

    @abstractmethod
    async def predict(self, state: dict[str, Any], label: str = "") -> LayaResult:
        """Return Laya's answers for one state."""

    @property
    def mode(self) -> str:
        """Return the backend name, for logs and metrics."""
        return type(self).__name__.removesuffix("Backend").lower()

    async def aclose(self) -> None:
        """Release any network resources. Subclasses that hold any override this."""
        return None

    async def __aenter__(self) -> LayaClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


class InprocessBackend(LayaClient):
    """Run the real Laya model in this process."""

    def __init__(self) -> None:
        settings = get_settings()
        self._agent: Any = None
        self._lock = asyncio.Lock()
        self._model_id = settings.laya_model
        self._max_len = settings.laya_max_len

    def _load(self) -> Any:
        """Load the model once. Laya has no ``max_len`` here; predict takes it."""
        if self._agent is None:
            try:
                import laya
            except ImportError as exc:
                raise LayaUnavailable("laya is not installed") from exc
            self._agent = laya.load(self._model_id)
        return self._agent

    async def predict(self, state: dict[str, Any], label: str = "") -> LayaResult:
        agent = self._load()
        async with self._lock:
            payload = await asyncio.to_thread(
                agent.predict, state, dict(QUESTION_SET), max_len=self._max_len
            )
        return parse_result(payload, dict(QUESTION_TYPES))


class ServeBackend(LayaClient):
    """Call a self-hosted Laya server such as ``python -m laya.serve``."""

    def __init__(self) -> None:
        import httpx

        settings = get_settings()
        self._base_url = settings.laya_http_base_url
        self._max_len = settings.laya_max_len
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=settings.deadline_s)

    async def predict(self, state: dict[str, Any], label: str = "") -> LayaResult:
        response = await self._client.post(
            "/v1/systemone",
            json={"state": state, "questions": dict(QUESTION_SET), "max_len": self._max_len},
        )
        response.raise_for_status()
        return parse_result(response.json(), dict(QUESTION_TYPES))

    async def aclose(self) -> None:
        await self._client.aclose()


class FakeBackend(LayaClient):
    """Deterministic stand-in for Laya, used by tests and offline runs.

    It reads the state and answers the way a well-behaved model would, so the
    policy engine, the refiner and the API can all be exercised without a model.

    Two modes. Content carrying ``[[KEEPME]]`` is always kept and content carrying
    ``[[NOISE]]`` is always droppable, which is what lets a synthetic trace with
    a known answer be scored end to end without a model. Anything unmarked falls
    back to a stable hash, so an unmarked item still gets a reproducible verdict
    rather than a random one.
    """

    #: Probability an unmarked item is treated as droppable.
    DROP_CHANCE = 0.5

    #: Marks content that must survive.
    KEEP_MARKER = "[[KEEPME]]"
    #: Marks content that may go.
    NOISE_MARKER = "[[NOISE]]"

    async def predict(self, state: dict[str, Any], label: str = "") -> LayaResult:
        item = state.get("item") if isinstance(state, dict) else None
        item = item if isinstance(item, dict) else {}
        digest = _digest(f"{label}|{item.get('content', '')}")
        content = str(item.get("content", ""))
        goal = str(state.get("goal", "")) if isinstance(state, dict) else ""
        overlap = _overlap(content, goal)
        big = len(content) > 800

        # A marker beats everything else: it is a planted, known answer, and
        # guessing at it would make the evaluation meaningless.
        if self.KEEP_MARKER in content:
            drop = False
        elif self.NOISE_MARKER in content:
            drop = True
        else:
            drop = int(digest[0], 16) / 255.0 < self.DROP_CHANCE

        return parse_result(
            {
                "model": "fake",
                "answers": {
                    "verdict": {
                        "type": "choice",
                        "choice": "drop" if drop else ("truncate" if big else "keep"),
                        "confidence": 0.9,
                        "answer_confidence": 0.9,
                        "probabilities": {"keep": 0.1, "truncate": 0.1, "drop": 0.8}
                        if drop
                        else {"keep": 0.7, "truncate": 0.2, "drop": 0.1},
                        "action": {"act_probability": 1.0},
                    },
                    "essential": {
                        "type": "noul",
                        "noul": 0.1 + (0.4 * overlap),
                        "confidence": 0.9,
                        "answer_confidence": 0.9,
                        "action": {"act_probability": 1.0},
                    },
                    "consumed": {
                        "type": "noul",
                        "noul": 0.9 if drop else 0.2,
                        "confidence": 0.9,
                        "answer_confidence": 0.9,
                        "action": {"act_probability": 1.0},
                    },
                    "superseded": {
                        "type": "noul",
                        "noul": 0.8 if drop else 0.1,
                        "confidence": 0.9,
                        "answer_confidence": 0.9,
                        "action": {"act_probability": 1.0},
                    },
                    "relevance": {
                        "type": "score",
                        "score": 0.3 if drop else 2.0,
                        "confidence": 0.9,
                        "answer_confidence": 0.9,
                        "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1, "3": 0.0}
                        if drop
                        else {"0": 0.0, "1": 0.1, "2": 0.5, "3": 0.4},
                        "legend": {
                            "0": "noise",
                            "1": "background",
                            "2": "supporting",
                            "3": "critical",
                        },
                        "action": {"act_probability": 1.0},
                    },
                },
                "usage": {"input_tokens": 0, "output_tokens": 0, "state_tokens": 0},
            },
            dict(QUESTION_TYPES),
        )


def _digest(text: str) -> str:
    """Return a short stable hash of ``text``."""
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def _overlap(content: str, goal: str) -> float:
    """Return how much of ``content``'s vocabulary also appears in ``goal``."""
    if not goal:
        return 0.0
    goal_words = {word for word in goal.lower().split() if len(word) > 3}
    if not goal_words:
        return 0.0
    words = {word for word in content.lower().split() if len(word) > 3}
    if not words:
        return 0.0
    return len(words & goal_words) / len(words)


def build_client() -> LayaClient:
    """Build the backend named by ``CC_LAYA_MODE``.

    ``http`` is the default because the hosted API is far faster than loading
    the model locally.
    """
    mode = get_settings().laya_mode
    if mode == "http":
        return HostedBackend()
    if mode == "serve":
        return ServeBackend()
    if mode == "inprocess":
        return InprocessBackend()
    return FakeBackend()


def build_engine(settings: Settings | None = None) -> LayaEngine:
    """Build the engine the Worker uses, from the configured backend.

    The one place that turns a ``Settings`` into something the Compactor can ask
    questions. ``LayaEngine`` is imported lazily because ``laya.engine`` depends on
    this module, so a real import here would be circular.
    """
    from contextcrunch.laya.engine import LayaEngine

    return LayaEngine(client=build_client(), settings=settings)
