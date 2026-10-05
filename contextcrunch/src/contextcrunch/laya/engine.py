"""The engine the Compactor asks questions of.

:class:`LayaEngine` is the *only* way the Worker reaches a model. It exists so
that two concerns stay out of the Worker itself:

- **batching.** A history has many items, and judging them one at a time would
  make a compaction as slow as the sum of every model call. The engine runs a
  whole batch concurrently and returns the answers in the order they were asked,
  so the Worker can zip the results back onto indexes without tracking which
  answer came from which call;
- **bounding.** Concurrency is capped by ``CC_LAYA_CONCURRENCY``. A 200-message
  history must not open 200 simultaneous requests, which would be refused by the
  hosted API and would waste the caller's rate limit.

The engine judges a *state*, never a message. Building the state is the caller's
job, because only the caller knows the surrounding history.

It deliberately holds no fail-open behaviour. The Worker is required to let
exceptions escape and the API layer decides what to do about them; an engine that
quietly returned a default answer would hide a real failure behind a plausible
looking verdict.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from contextcrunch.core.settings import Settings
from contextcrunch.laya.client import LayaClient, build_client
from contextcrunch.laya.types import LayaResult


class LayaEngine:
    """Ask Laya about many states at once, with a bounded number in flight."""

    def __init__(self, client: LayaClient | None = None, settings: Settings | None = None) -> None:
        """Wrap a backend.

        Both arguments are optional so a caller can inject a fake backend in a
        test, while production leaves them alone and gets the configured one.
        """
        from contextcrunch.core.settings import get_settings

        self.settings = settings or get_settings()
        self.client = client if client is not None else build_client()

    async def decide(
        self, states: Sequence[dict[str, Any]], labels: Sequence[str] | None = None
    ) -> list[LayaResult]:
        """Judge every state concurrently and return the answers in the same order.

        ``labels`` are for logs and for backends that key off the item; they do
        not affect ordering. The result list is always the same length as
        ``states`` and positionally aligned with it, which is the property the
        Worker relies on.
        """
        if not states:
            return []
        # An absent or empty label list means "label them yourself". A non-empty
        # list of the wrong length is a caller bug and is raised on rather than
        # quietly relabelled, because a mislabelled batch still returns the right
        # number of answers and would otherwise fail silently.
        if labels is None or len(labels) == 0:
            names = [f"item:{index}" for index in range(len(states))]
        elif len(labels) != len(states):
            raise ValueError(
                f"got {len(labels)} labels for {len(states)} states; they must match"
            )
        else:
            names = list(labels)

        # A semaphore of at least one: a configured concurrency of zero would
        # otherwise deadlock the batch instead of simply not limiting it.
        limit = max(self.settings.laya_concurrency, 1)
        semaphore = asyncio.Semaphore(limit)

        async def one(state: dict[str, Any], label: str) -> LayaResult:
            async with semaphore:
                return await self.client.predict(state, label)

        return list(
            await asyncio.gather(
                *(one(state, label) for state, label in zip(states, names, strict=True))
            )
        )

    async def aclose(self) -> None:
        """Release the backend's network resources."""
        await self.client.aclose()