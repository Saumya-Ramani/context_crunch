"""Cached engine for replaying recorded real-Laya answers.

This engine reads from a JSONL cache directory and returns recorded answers
without calling any model. It raises CacheMiss if a key is not found.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from contextcrunch.core.settings import Settings
from contextcrunch.laya.cache import LayaCache, cache_key, CacheMiss
from contextcrunch.laya.client import LayaClient
from contextcrunch.laya.engine import LayaEngine
from contextcrunch.laya.types import LayaResult
from contextcrunch.core.questions import QUESTION_SET


class CachedEngine(LayaEngine):
    """Engine that replays recorded Laya answers from a cache."""

    def __init__(self, cache: LayaCache, settings: Settings | None = None) -> None:
        """Wrap a cache.

        Args:
            cache: The LayaCache to read from.
            settings: Optional settings (for concurrency limit).
        """
        from contextcrunch.core.settings import get_settings

        self.settings = settings or get_settings()
        self._cache = cache

    async def decide(
        self, states: Sequence[dict[str, Any]], labels: Sequence[str] | None = None
    ) -> list[LayaResult]:
        """Return recorded answers for all states, raising CacheMiss if any missing."""
        if not states:
            return []

        if labels is None or len(labels) == 0:
            names = [f"item:{index}" for index in range(len(states))]
        elif len(labels) != len(states):
            raise ValueError(
                f"got {len(labels)} labels for {len(states)} states; they must match"
            )
        else:
            names = list(labels)

        # A semaphore of at least one for consistency with LayaEngine
        limit = max(self.settings.laya_concurrency, 1)
        semaphore = asyncio.Semaphore(limit)

        async def one(state: dict[str, Any], label: str) -> LayaResult:
            async with semaphore:
                key = cache_key(state, QUESTION_SET)
                record = self._cache.get(key)
                if record is None:
                    raise CacheMiss(key)
                return LayaResult.model_validate(record["answers"])

        return list(
            await asyncio.gather(
                *(one(state, label) for state, label in zip(states, names, strict=True))
            )
        )

    async def aclose(self) -> None:
        """No resources to release for cached engine."""
        pass