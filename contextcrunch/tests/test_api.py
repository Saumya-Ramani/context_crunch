"""Tests for the HTTP service.

The service sits in front of somebody else's LLM call, so the property that
matters most is that it never breaks that call. Most of these tests are therefore
about failure: a dead engine, a timeout, a malformed request. Each one asserts
that the caller gets its own history back and can carry on.

A scripted engine is injected throughout. No model is loaded and no network call is
made, so the suite is fast and deterministic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.conftest import answers
from tests.helpers import ScriptedEngine, make_drop_answer, make_keep_answer

from contextcrunch.agents.worker import CompactorAgent
from contextcrunch.api.main import create_app
from contextcrunch.core.messages import as_messages
from contextcrunch.laya.engine import LayaEngine

#: Answers that clear every drop gate.
DROP_ALL = answers(
    verdict="drop", essential=0.0, superseded=0.9, relevance=0.0, verdict_conf=0.99
)


class Exploding:
    """An engine whose model is down."""

    async def decide(self, states, labels=None):
        raise RuntimeError("model is down")

    async def aclose(self) -> None:
        """Nothing to release."""


# Alias for backward compatibility with tests that still use FakeLaya
class FakeLaya:
    """Backward-compatible wrapper for tests using the old FakeLaya interface."""

    def __init__(self, *results, delay: float = 0.0) -> None:
        self._engine = ScriptedEngine({"": results[0] if results else make_keep_answer()}, delay=delay)
        self._results = list(results)
        self._delay = delay
        self.calls: list[list[str]] = []

    async def decide(self, states, labels=None):
        """Answer every state in one batch."""
        self.calls.append([str(label) for label in (labels or [])])
        if self._delay:
            await asyncio.sleep(self._delay)
        return [self._results.pop(0) if self._results else make_keep_answer() for _ in states]

    async def aclose(self) -> None:
        """Nothing to release."""


def noise(index: int = 0) -> str:
    """Return tool output big enough to be worth judging."""
    return "filler line of console output\n" * 40 + f" end{index}"


def trace(*contents: str, filler: int = 3) -> list[dict[str, Any]]:
    """Return a raw history, as a host app would POST it.

    ``filler`` adds trailing assistant turns. They are not filler for the
    profile's sake: the most recent tool message sits inside the protected
    window of recent turns, so a history with nothing after it has nothing the
    Compactor is allowed to judge.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "fix the billing rounding bug"},
    ]
    for index, content in enumerate(contents):
        messages.append(
            {
                "role": "tool",
                "name": "read_file",
                "tool_call_id": f"call_{index}",
                "content": content,
            }
        )
        messages.append({"role": "assistant", "content": f"finding {index}"})
    for extra in range(filler):
        messages.append({"role": "assistant", "content": f"follow-up step {extra}"})
    return messages


def build_client(settings, engine, tmp_path) -> TestClient:
    """Return a TestClient over an app wired to a temporary store.

    ``engine`` is injected as the ``LayaEngine`` itself rather than wrapped in
    one: the fakes here answer ``decide``, which is the interface the Compactor
    depends on, so there is nothing left for a real engine to add.
    """
    resolved = settings.model_copy(
        update={
            "store_db_path": str(tmp_path / "store.db"),
            "store_originals_dir": str(tmp_path / "originals"),
            # A low trigger so a small fixture actually reaches the model.
            "trigger_tokens": 1,
        }
    )
    engine.settings = resolved
    return TestClient(create_app(settings=resolved, engine=engine))


@pytest.fixture
def settings_factory_api(settings_factory):
    """Return the settings factory with a generous deadline by default."""

    def build(**overrides: Any):
        defaults: dict[str, Any] = {"deadline_s": 10, "laya_concurrency": 4}
        defaults.update(overrides)
        return settings_factory(**defaults)

    return build


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


def test_health_reports_the_backend_in_use(
    settings_factory_api, tmp_path
) -> None:
    """A host needs to know which model this instance will call."""
    settings = settings_factory_api()
    engine = ScriptedEngine({"": make_keep_answer()})
    with build_client(settings, engine, tmp_path) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["laya_mode"] == settings.laya_mode


def test_health_needs_no_session(settings_factory_api, tmp_path) -> None:
    """Health must answer even when nothing has been compacted yet."""
    engine = ScriptedEngine({"": make_keep_answer()})
    with build_client(settings_factory_api(), engine, tmp_path) as client:
        assert client.get("/health").status_code == 200


# --------------------------------------------------------------------------- #
# Compaction
# --------------------------------------------------------------------------- #


def test_a_noise_tool_message_is_tombstoned(settings_factory_api, tmp_path) -> None:
    """The end-to-end path removes junk and keeps the conversation's shape."""
    engine = ScriptedEngine({"": make_drop_answer()})
    with build_client(settings_factory_api(), engine, tmp_path) as client:
        response = client.post(
            "/v1/compact",
            json={"messages": trace(noise(), noise(1)), "session_id": "s1"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["fail_open"] is False
    assert body["removed"] >= 1
    assert body["tokens_after"] < body["tokens_before"]
    assert len(body["messages"]) == len(trace(noise(), noise(1)))

    sent = body["messages"]
    original = trace(noise(), noise(1))
    assert [m["role"] for m in sent] == [m["role"] for m in original]
    ids = [m.get("tool_call_id") for m in sent if m.get("tool_call_id")]
    assert ids == ["call_0", "call_1"], "tool_call_ids must survive compaction"


def test_a_compacted_history_is_accepted_by_the_message_model(
    settings_factory_api, tmp_path
) -> None:
    """Whatever we return must be something the host can send on.

    A tombstone that lost its ``tool_call_id`` would be rejected by the provider,
    so the response is parsed back through the real model rather than inspected
    field by field.
    """
    engine = ScriptedEngine({"": make_drop_answer()})
    with build_client(settings_factory_api(), engine, tmp_path) as client:
        body = client.post(
            "/v1/compact", json={"messages": trace(noise(), noise(1)), "session_id": "s1"}
        ).json()

    reparsed = as_messages(body["messages"])
    assert len(reparsed) == len(body["messages"])
    assert all(isinstance(m.content, str) for m in reparsed)


def test_an_explicit_profile_is_reported_back(settings_factory_api, tmp_path) -> None:
    """The host asked for a profile, so the host is told which one was used."""
    engine = ScriptedEngine({"": make_drop_answer()})
    with build_client(settings_factory_api(), engine, tmp_path) as client:
        body = client.post(
            "/v1/compact",
            json={"messages": trace(noise()), "session_id": "s1", "profile": "aggressive"},
        ).json()

    assert body["profile_used"] == "aggressive"


# --------------------------------------------------------------------------- #
# Fail-open
# --------------------------------------------------------------------------- #


def test_invalid_input_is_rejected(settings_factory_api, tmp_path) -> None:
    """A caller bug must be visible, not silently compacted.

    This is the one case that does return an error: an empty ``messages`` list is
    a mistake in the host, and hiding it would hide the bug.
    """
    engine = ScriptedEngine({"": make_keep_answer()})
    with build_client(settings_factory_api(), engine, tmp_path) as client:
        assert client.post("/v1/compact", json={"messages": []}).status_code == 422
        assert client.post("/v1/compact", json={}).status_code == 422
        assert client.post("/v1/compact", json={"messages": "not a list"}).status_code == 422
        # A field of the wrong type is also the caller's bug, not ours.
        assert client.post("/v1/compact", json={"messages": [1, 2]}).status_code == 422


def test_a_broken_engine_fails_open(settings_factory_api, tmp_path) -> None:
    """A model that is down must not take the host's LLM call with it."""
    messages = trace(noise(), noise(1))
    engine = Exploding()
    with build_client(settings_factory_api(), engine, tmp_path) as client:
        response = client.post(
            "/v1/compact", json={"messages": messages, "session_id": "s1"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["fail_open"] is True
    assert body["profile_used"] == "none"
    assert body["messages"] == messages, "the caller's own history must come back untouched"
    assert body["tokens_before"] == body["tokens_after"]
    assert body["reduction_pct"] == 0.0
    assert "RuntimeError" in body["notes"][0]


def test_a_timeout_fails_open(settings_factory_api, tmp_path) -> None:
    """A slow model must not stall the host past its own deadline."""
    messages = trace(noise(), noise(1))
    settings = settings_factory_api(deadline_s=1)
    engine = ScriptedEngine({"": make_drop_answer()}, delay=5)

    with build_client(settings, engine, tmp_path) as client:
        response = client.post(
            "/v1/compact", json={"messages": messages, "session_id": "s1"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["fail_open"] is True
    assert body["profile_used"] == "none"
    assert body["messages"] == messages
    assert body["tokens_before"] == body["tokens_after"]
    assert "timeout" in body["notes"][0]


def test_a_traceback_is_logged_but_never_returned(
    settings_factory_api, tmp_path, caplog
) -> None:
    """The operator gets the traceback; the caller gets only a reason.

    A host cannot act on a stack trace, and returning one would leak internals
    for no benefit.
    """
    messages = trace(noise())
    # `create_app` calls `logging.basicConfig`, which attaches a handler to the
    # root logger. `caplog` needs its own handler on that logger, so the level
    # is set here rather than relying on the app's configuration.
    with caplog.at_level(logging.WARNING, logger="contextcrunch.api"):
        engine = Exploding()
        with build_client(settings_factory_api(), engine, tmp_path) as client:
            body = client.post(
                "/v1/compact", json={"messages": messages, "session_id": "s1"}
            ).json()

    assert any("compaction failed" in record.message for record in caplog.records)
    assert any(record.exc_info for record in caplog.records), "the traceback must be logged"

    joined = " ".join(body["notes"]) + str(body)
    assert "Traceback" not in joined
    assert "contextcrunch/api/main.py" not in joined


def test_a_broken_store_still_returns_a_history(
    settings_factory_api, tmp_path
) -> None:
    """Even a failure inside the Guardian must not reach the host.

    The Guardian is meant to be unbreakable, but the guarantee that matters is
    the one at the edge: whatever happens inside, the caller gets its history.
    """
    messages = trace(noise(), noise(1))
    with build_client(settings_factory_api(), FakeLaya(*([DROP_ALL] * 16)), tmp_path) as client:
        # A store that raises the moment it is used, rather than one that is
        # merely absent.
        class Broken:
            def __getattr__(self, name):
                def fail(*args, **kwargs):
                    raise RuntimeError("store is gone")

                return fail

        client.app.state.guardian.store = Broken()  # type: ignore[assignment]
        client.app.state.worker.store = Broken()  # type: ignore[assignment]
        response = client.post("/v1/compact", json={"messages": messages, "session_id": "s1"})

    assert response.status_code == 200
    assert response.json()["messages"] == messages
    assert response.json()["fail_open"] is True


# --------------------------------------------------------------------------- #
# Session learning
# --------------------------------------------------------------------------- #


def test_a_re_requested_removal_is_logged_as_a_mistake(
    settings_factory_api, tmp_path
) -> None:
    """Removing something the host then asks for again is evidence we were wrong.

    This is the loop that makes the thresholds learnable: the first call drops
    an item, the host comes back needing it, and the Guardian records that.
    """
    settings = settings_factory_api()
    original = noise()
    first = trace(original, noise(1))

    with build_client(settings, FakeLaya(*([DROP_ALL] * 16)), tmp_path) as client:
        body = client.post(
            "/v1/compact", json={"messages": first, "session_id": "s2"}
        ).json()
        assert body["removed"] >= 1, "the first call should drop the noise"
        assert body["fail_open"] is False

        # The host asks again with the item it needs back, at a later index.
        second = trace(noise(2), original)
        body2 = client.post(
            "/v1/compact", json={"messages": second, "session_id": "s2"}
        ).json()

        stats = client.get("/v1/sessions/s2/stats").json()

    assert stats["mistakes"] >= 1, "a re-requested removal must be recorded"
    assert stats["decisions"], "the decisions must be visible too"
    assert body2["messages"], "the second call must still return a usable history"


def test_session_stats_for_an_unknown_session_is_empty_not_an_error(
    settings_factory_api, tmp_path
) -> None:
    """Asking about a session that never happened is a 200 with zeroes."""
    with build_client(settings_factory_api(), FakeLaya(), tmp_path) as client:
        body = client.get("/v1/sessions/nobody/stats").json()

    assert body["session_id"] == "nobody"
    assert body["decisions"] == {}
    assert body["mistakes"] == 0


def test_two_calls_in_one_session_do_not_duplicate_decisions(
    settings_factory_api, tmp_path
) -> None:
    """Re-judging an item replaces its verdict rather than accumulating.

    One tool message is judged, not two: the other sits inside the aggressive
    profile's protected window of recent turns. The point is that the row count
    does not grow between calls, which is what UPSERT on ``(session, index)``
    buys.
    """
    messages = trace(noise(), noise(1))
    with build_client(settings_factory_api(), FakeLaya(*([DROP_ALL] * 32)), tmp_path) as client:
        client.post("/v1/compact", json={"messages": messages, "session_id": "s3"})
        first = client.get("/v1/sessions/s3/stats").json()
        client.post("/v1/compact", json={"messages": messages, "session_id": "s3"})
        second = client.get("/v1/sessions/s3/stats").json()

    assert first["decisions"], "the first call must record something"
    assert second["decisions"] == first["decisions"], (
        f"a repeat call must replace decisions, not add to them: "
        f"{first['decisions']} then {second['decisions']}"
    )


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_nothing_is_built_at_import_time() -> None:
    """Importing the module must not need a model, a network or a database.

    If it did, ``uvicorn --factory`` would fail before it could serve anything,
    and every test that imports the module would need the environment.
    """
    import importlib

    module = importlib.import_module("contextcrunch.api.main")
    assert hasattr(module, "create_app")


def test_the_app_holds_one_of_each_collaborator(
    settings_factory_api, tmp_path
) -> None:
    """Built once at startup, not per request.

    A per-request Store would mean a new SQLite connection on every call, and a
    per-request engine a new HTTP pool.
    """
    from contextcrunch.agents.guardian import GuardianAgent
    from contextcrunch.agents.scout import ScoutAgent
    from contextcrunch.storage.store import Store

    settings = settings_factory_api()
    app = create_app(
        settings=settings,
        engine=LayaEngine(client=FakeLaya(), settings=settings),  # type: ignore[arg-type]
    )
    assert isinstance(app.state.store, Store)
    assert isinstance(app.state.scout, ScoutAgent)
    assert isinstance(app.state.guardian, GuardianAgent)
    assert isinstance(app.state.worker, CompactorAgent)


def test_the_store_writes_inside_the_given_directory(
    settings_factory_api, tmp_path
) -> None:
    """A test must never leave a database behind in the repository."""
    settings = settings_factory_api(
        store_db_path=str(tmp_path / "nested" / "store.db"),
        store_originals_dir=str(tmp_path / "nested" / "originals"),
    )
    app = create_app(settings=settings, engine=FakeLaya(*([DROP_ALL] * 16)))
    with TestClient(app) as client:
        client.post("/v1/compact", json={"messages": trace(noise()), "session_id": "s4"})

    assert (tmp_path / "nested" / "store.db").exists()
    assert (tmp_path / "nested" / "originals").is_dir()