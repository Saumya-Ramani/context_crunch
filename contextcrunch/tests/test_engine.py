"""Tests for the engine: the Worker's only route to a model.

The engine owns two guarantees that the Worker relies on and cannot see for
itself:

- every state in a batch is judged, concurrently, and the answers come back in
  the order they were asked;
- no more than ``CC_LAYA_CONCURRENCY`` calls are ever in flight.

Both are asserted here rather than through the Worker, because the Worker talks
to a batch interface and cannot observe either property from outside.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.conftest import answers

from contextcrunch.laya.engine import LayaEngine
from contextcrunch.laya.types import LayaResult


class CountingBackend:
    """A backend that records how many calls overlapped."""

    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay
        self.calls: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def predict(self, state: dict[str, Any], label: str = "") -> LayaResult:
        """Record the call, then wait long enough for overlap to be observable."""
        self.calls.append(label)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return answers()
        finally:
            self.in_flight -= 1

    async def aclose(self) -> None:
        """Nothing to release."""


@pytest.fixture
def settings_factory_concurrency(settings_factory):
    """Return the settings factory with a chosen concurrency limit."""

    def build(concurrency: int):
        return settings_factory(laya_concurrency=concurrency)

    return build


def states(count: int) -> list[dict[str, Any]]:
    """Return ``count`` distinguishable states."""
    return [{"item": {"index": index}} for index in range(count)]


async def test_every_state_is_answered(settings_factory_concurrency) -> None:
    """A short batch must not silently answer only part of it."""
    backend = CountingBackend(delay=0)
    engine = LayaEngine(client=backend, settings=settings_factory_concurrency(4))

    results = await engine.decide(states(6), [f"item:{i}" for i in range(6)])

    assert len(results) == 6
    assert len(backend.calls) == 6


async def test_an_empty_batch_asks_nothing(settings_factory_concurrency) -> None:
    """No states means no model calls, and not an error."""
    backend = CountingBackend(delay=0)
    engine = LayaEngine(client=backend, settings=settings_factory_concurrency(4))

    assert await engine.decide([], []) == []
    assert backend.calls == []


async def test_the_batch_is_judged_concurrently(settings_factory_concurrency) -> None:
    """Calls must overlap, or a compaction costs the sum of every call."""
    backend = CountingBackend(delay=0.02)
    engine = LayaEngine(client=backend, settings=settings_factory_concurrency(8))

    await engine.decide(states(6), [f"item:{i}" for i in range(6)])

    assert backend.max_in_flight == 6, (
        f"only {backend.max_in_flight} calls overlapped, so the batch was sequential"
    )


async def test_concurrency_is_capped_by_the_setting(settings_factory_concurrency) -> None:
    """A long history must not open one request per item at the hosted API."""
    backend = CountingBackend(delay=0.02)
    engine = LayaEngine(client=backend, settings=settings_factory_concurrency(2))

    await engine.decide(states(8), [f"item:{i}" for i in range(8)])

    assert len(backend.calls) == 8, "every state must still be answered"
    assert backend.max_in_flight <= 2, f"limit was {backend.max_in_flight}, expected at most 2"


async def test_a_zero_concurrency_does_not_deadlock(settings_factory_concurrency) -> None:
    """A misconfigured limit of zero must not hang the request forever."""
    backend = CountingBackend(delay=0)
    engine = LayaEngine(client=backend, settings=settings_factory_concurrency(0))

    results = await asyncio.wait_for(engine.decide(states(3), []), timeout=5)

    assert len(results) == 3


async def test_answers_come_back_in_the_order_they_were_asked(
    settings_factory_concurrency,
) -> None:
    """The Worker zips answers back onto indexes positionally, so order matters."""

    class Ordered:
        """Answers with a marker identifying which state it was given."""

        async def predict(self, state: dict[str, Any], label: str = "") -> LayaResult:
            await asyncio.sleep(0.01 * (3 - int(state["item"]["index"])))
            return answers(relevance=float(state["item"]["index"]))

        async def aclose(self) -> None:
            """Nothing to release."""

    engine = LayaEngine(client=Ordered(), settings=settings_factory_concurrency(4))

    results = await engine.decide(states(3), [])

    assert [r.answer("relevance").score for r in results] == [0.0, 1.0, 2.0]


async def test_a_backend_error_propagates(settings_factory_concurrency) -> None:
    """The engine must not swallow a failure: the API layer owns fail-open."""

    class Exploding:
        async def predict(self, state: dict[str, Any], label: str = "") -> LayaResult:
            raise RuntimeError("model exploded")

        async def aclose(self) -> None:
            """Nothing to release."""

    engine = LayaEngine(client=Exploding(), settings=settings_factory_concurrency(4))

    with pytest.raises(RuntimeError, match="model exploded"):
        await engine.decide(states(2), [])
