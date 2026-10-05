"""Tests for the pipeline, the Laya client and the API surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.conftest import ScriptedBackend, answers, history

from contextcrunch.core.pipeline import Cruncher, compact_history
from contextcrunch.core.settings import reset_settings_cache
from contextcrunch.laya.client import FakeBackend, build_client

DROP_ALL = answers(
    verdict="drop", essential=0.0, superseded=0.9, relevance=0.0, verdict_conf=0.99
)


async def test_pipeline_shrinks_a_history() -> None:
    """The end-to-end path must actually save tokens."""
    cruncher = Cruncher(client=ScriptedBackend(*([DROP_ALL] * 8)), box=None)
    report = await compact_history(
        [m.model_dump() for m in history(3)], cruncher=cruncher, profile="aggressive"
    )
    assert report.degraded is False
    assert report.tokens_after < report.tokens_before
    assert report.as_dict()["reduction_pct"] > 0


async def test_pipeline_keeps_a_small_history_untouched() -> None:
    """Below the trigger there is nothing to do, and nothing is paid for."""
    cruncher = Cruncher(client=ScriptedBackend(), box=None)
    report = await compact_history(
        [{"role": "user", "content": "hi"}], cruncher=cruncher, profile="aggressive"
    )
    assert report.skipped is True
    assert report.tokens_after == report.tokens_before


async def test_pipeline_fails_open_on_a_broken_client() -> None:
    """A model failure must still return a usable history."""

    class Exploding:
        async def predict(self, state, label=""):
            raise RuntimeError("boom")

        async def aclose(self):
            return None

    raw = [m.model_dump() for m in history(2)]
    report = await compact_history(raw, cruncher=Cruncher(client=Exploding(), box=None))
    assert report.degraded is True
    assert len(report.messages) == len(history(2))


async def test_pipeline_handles_an_empty_history() -> None:
    """An empty request must not crash."""
    report = await compact_history([], cruncher=Cruncher(client=FakeBackend(), box=None))
    assert report.messages == []


async def test_fake_backend_answers_all_five_questions() -> None:
    """The fake backend must honour the contract the policy engine relies on."""
    from contextcrunch.core.settings import get_settings
    from contextcrunch.core.state import build_state

    result = await FakeBackend().predict(build_state(history(1), 3, "goal", get_settings()))
    assert set(result.answers) == {
        "verdict",
        "essential",
        "consumed",
        "superseded",
        "relevance",
    }


async def test_fake_backend_is_deterministic() -> None:
    """The same state must give the same answers, so tests are reproducible."""
    from contextcrunch.core.settings import get_settings
    from contextcrunch.core.state import build_state

    state = build_state(history(1), 3, "goal", get_settings())
    first = await FakeBackend().predict(state, "label")
    second = await FakeBackend().predict(state, "label")
    assert first.answers["verdict"].choice == second.answers["verdict"].choice


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("fake", "FakeBackend"),
        ("inprocess", "InprocessBackend"),
        ("http", "HostedBackend"),
    ],
)
def test_build_client_follows_the_mode(mode: str, expected: str) -> None:
    """CC_LAYA_MODE selects the backend. http is the hosted inference API."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("CC_LAYA_MODE", mode)
    reset_settings_cache()
    try:
        assert type(build_client()).__name__ == expected
    finally:
        monkeypatch.undo()
        reset_settings_cache()


def test_health_endpoint() -> None:
    """The gateway needs a health check."""
    with TestClient(_app()) as client:
        payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["laya_mode"] == "fake"


def test_compact_endpoint_returns_a_history() -> None:
    """The main endpoint must always return a usable message list."""
    with TestClient(_app()) as client:
        response = client.post(
            "/v1/compact",
            json={
                "messages": [m.model_dump() for m in history(3)],
                "profile": "aggressive",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["messages"]
    assert body["profile"] == "aggressive"
    assert body["tokens_after"] <= body["tokens_before"]


def test_compact_endpoint_rejects_an_empty_history() -> None:
    """An empty history is a client error, not a silent success."""
    with TestClient(_app()) as client:
        response = client.post("/v1/compact", json={"messages": []})
    assert response.status_code == 422


def test_unknown_run_is_a_404() -> None:
    """Reading back an unknown run must be explicit."""
    with TestClient(_app()) as client:
        assert client.get("/v1/runs/nope").status_code == 404


def _app():
    """Return an app wired to the fake backend and no database."""
    from contextcrunch.api.app import create_app

    return create_app(Cruncher(client=FakeBackend(), box=None))
