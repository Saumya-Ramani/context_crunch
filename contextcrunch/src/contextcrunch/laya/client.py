"""The only place ContextCrunch talks to a model.

Three backends sit behind one small interface:

- ``http`` (the default) calls the hosted inference API with a bearer token.
  This is the fast path, so it is the one the service uses;
- ``serve`` calls a self-hosted ``python -m laya.serve`` instance;
- ``inprocess`` runs the model in this process. It needs no network but it is
  slow, because it pays model load and inference cost on every call;
- ``replay`` reads recorded real-Laya answers from a cache directory.

Every backend honours the token budget, the question set and the concurrency
limit from :mod:`contextcrunch.core.settings`.
"""

from __future__ import annotations

import asyncio
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


class ReplayBackend(LayaClient):
    """Read recorded real-Laya answers from a cache directory."""

    def __init__(self, replay_dir: str) -> None:
        import json
        from pathlib import Path

        self._replay_dir = Path(replay_dir)
        self._cache: dict[str, LayaResult] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load all JSONL cache files from the replay directory."""
        import json
        from contextcrunch.laya.types import parse_result
        from contextcrunch.core.questions import QUESTION_TYPES

        if not self._replay_dir.exists():
            raise LayaUnavailable(f"Replay directory does not exist: {self._replay_dir}")

        for cache_file in self._replay_dir.glob("*.jsonl"):
            with cache_file.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    if line_num == 0:
                        # Skip header line
                        continue
                    try:
                        record = json.loads(line)
                        key = record["key"]
                        answers = parse_result(record["raw"], dict(QUESTION_TYPES))
                        self._cache[key] = answers
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def predict(self, state: dict[str, Any], label: str = "") -> LayaResult:
        from contextcrunch.laya.cache import cache_key
        from contextcrunch.core.questions import QUESTION_SET

        key = cache_key(state, QUESTION_SET)
        if key not in self._cache:
            raise LayaUnavailable(f"Cache miss for key: {key[:16]}... (run collect_answers.py first)")
        return self._cache[key]

    @property
    def mode(self) -> str:
        return "replay"


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
    if mode == "replay":
        replay_dir = get_settings().replay_dir
        if not replay_dir:
            raise LayaUnavailable("CC_REPLAY_DIR must be set when laya_mode=replay")
        return ReplayBackend(replay_dir)
    raise LayaUnavailable(f"Unknown laya_mode: {mode}")


def build_engine(settings: Settings | None = None) -> LayaEngine:
    """Build the engine the Worker uses, from the configured backend.

    The one place that turns a ``Settings`` into something the Compactor can ask
    questions. ``LayaEngine`` is imported lazily because ``laya.engine`` depends on
    this module, so a real import here would be circular.
    """
    from contextcrunch.laya.engine import LayaEngine
    from contextcrunch.laya.cached_engine import CachedEngine
    from contextcrunch.laya.cache import LayaCache

    settings = settings or get_settings()
    mode = settings.laya_mode

    if mode == "replay":
        if not settings.replay_dir:
            raise LayaUnavailable("CC_REPLAY_DIR must be set when laya_mode=replay")
        cache = LayaCache(settings.replay_dir)
        return CachedEngine(cache, settings)

    client = build_client()
    return LayaEngine(client=client, settings=settings)

    return LayaEngine(client=build_client(), settings=settings)
