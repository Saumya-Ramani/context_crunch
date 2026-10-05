"""Tests for the hosted Laya backend.

No test here touches the network or the real API key. The HTTP layer is replaced
with an ``httpx.MockTransport``, so these assert on the request we build and the
parsing we do, not on a live service.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from contextcrunch.core.questions import QUESTION_SET
from contextcrunch.core.settings import Settings, get_settings, reset_settings_cache
from contextcrunch.laya.hosted import HostedBackend, HostedLayaError, _unwrap
from contextcrunch.laya.types import parse_result

Handler = Callable[[httpx.Request], httpx.Response]

#: A response shaped like the real hosted API: no answer_confidence, no action.
HOSTED_ANSWERS: dict[str, dict[str, object]] = {
    "verdict": {
        "type": "choice",
        "choice": "drop",
        "confidence": 0.05,
        "probabilities": {"keep": 0.1, "truncate": 0.15, "drop": 0.75},
    },
    "essential": {"type": "noul", "noul": 0.12},
    "consumed": {"type": "noul", "noul": 0.91},
    "superseded": {"type": "noul", "noul": 0.83},
    "relevance": {
        "type": "score",
        "score": 0.4,
        "confidence": 0.3,
        "probabilities": {"0": 0.8, "1": 0.15, "2": 0.04, "3": 0.01},
    },
}

HOSTED_BODY: dict[str, object] = {
    "model": "laya-0.3.4/english",
    "answers": HOSTED_ANSWERS,
    "usage": {"input_tokens": 373, "output_tokens": 0},
}


def mock(handler: Handler) -> HostedBackend:
    """Return a backend whose HTTP client is a mock transport."""
    return HostedBackend(transport=httpx.MockTransport(handler))


async def test_sends_the_documented_payload_and_bearer_header() -> None:
    """The request must match the agreed shape exactly."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=HOSTED_BODY)

    backend = mock(handler)
    await backend.predict({"item": {"content": "x"}}, "unit")
    await backend.aclose()

    assert seen["url"] == "https://api.example.test/v1/systemone"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert set(body) == {"model", "state", "questions"}
    assert body["state"] == {"item": {"content": "x"}}
    assert body["questions"] == dict(QUESTION_SET)


def test_parses_hosted_answers_and_derives_answer_confidence() -> None:
    """The missing answer_confidence must be recovered, not left as None."""
    result = parse_result(HOSTED_BODY, {})
    verdict = result.answer("verdict")
    assert verdict is not None
    # 0.75 is the probability of the chosen option, matching the local model.
    assert verdict.answer_confidence == pytest.approx(0.75)
    assert verdict.act_ok() is True  # the hosted API sends no action block

    essential = result.answer("essential")
    assert essential is not None
    assert essential.answer_confidence == pytest.approx(1.0 - 0.12)


def test_score_answer_confidence_is_the_largest_level() -> None:
    """A score has no single chosen option, so the top level probability is used."""
    result = parse_result(HOSTED_BODY, {})
    relevance = result.answer("relevance")
    assert relevance is not None
    assert relevance.answer_confidence == pytest.approx(0.8)


async def test_predict_returns_answers_the_policy_engine_can_use() -> None:
    """A full predict call parses into the same shape as the local backend."""
    backend = mock(lambda request: httpx.Response(200, json=HOSTED_BODY))
    result = await backend.predict({"item": {"content": "x"}}, "unit")
    await backend.aclose()

    assert set(result.answers) == {"verdict", "essential", "consumed", "superseded", "relevance"}
    assert result.noul("consumed") == pytest.approx(0.91)
    assert result.answer("relevance").score == pytest.approx(0.4)
    assert result.model == "laya-0.3.4/english"


def test_missing_api_key_fails_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key means no request, and a clear error naming the variable."""
    for name in ("CC_LAYA_API_KEY", "Laya_Api_key", "LAYA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    reset_settings_cache()

    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    with pytest.raises(HostedLayaError, match="Laya_Api_key"):
        HostedBackend(transport=transport).build_headers()


def test_api_key_is_read_from_the_underscore_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key is accepted from the name used in .env, not only the CC_ prefix."""
    for name in ("CC_LAYA_API_KEY", "Laya_Api_key", "LAYA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("Laya_Api_key", "secret-token")
    reset_settings_cache()
    try:
        assert get_settings().laya_api_key == "secret-token"
        assert HostedBackend().build_headers()["Authorization"] == "Bearer secret-token"
    finally:
        reset_settings_cache()


async def test_client_error_is_not_retried() -> None:
    """A 401 must fail fast rather than burn the retry budget."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="unauthorized")

    backend = mock(handler)
    with pytest.raises(HostedLayaError, match="HTTP 401"):
        await backend.predict({"item": {}}, "unit")
    await backend.aclose()
    assert len(calls) == 1


async def test_server_error_is_retried_then_raises() -> None:
    """A 503 is transient, so it is retried the configured number of times."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="unavailable")

    backend = mock(handler)
    with pytest.raises(HostedLayaError, match="failed after 2 attempts"):
        await backend.predict({"item": {}}, "unit")
    await backend.aclose()
    assert len(calls) == 2  # 1 initial + 1 retry (CC_LAYA_RETRIES=1)


async def test_retry_succeeds_after_transient_failure() -> None:
    """A retry that works returns a normal result."""
    responses = [httpx.Response(500, text="boom"), httpx.Response(200, json=HOSTED_BODY)]
    backend = mock(lambda request: responses.pop(0))
    result = await backend.predict({"item": {}}, "unit")
    await backend.aclose()
    assert result.answer("verdict").choice == "drop"


async def test_response_without_answers_raises() -> None:
    """A 200 with no answers is an error, not an empty result."""
    backend = mock(lambda request: httpx.Response(200, json={"model": "x"}))
    with pytest.raises(HostedLayaError, match="no answers"):
        await backend.predict({"item": {}}, "unit")
    await backend.aclose()


async def test_network_failure_raises_a_hosted_error() -> None:
    """A connection error must surface as a hosted error, never raw httpx."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    backend = mock(handler)
    with pytest.raises(HostedLayaError, match="failed after"):
        await backend.predict({"item": {}}, "unit")
    await backend.aclose()


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"answers": {"a": {}}}, True),
        ({"result": {"answers": {"a": {}}}}, True),
        ({"data": {"answers": {"a": {}}}}, True),
        ({"results": [{"answers": {"a": {}}}]}, True),
        ({"results": [{"a": {}}]}, True),
        ({"nothing": 1}, False),
    ],
)
def test_unwrap_accepts_common_wrapper_shapes(data: dict, expected: bool) -> None:
    """Different platforms wrap the answer map differently."""
    assert ("answers" in _unwrap(data)) is expected


def test_unwrap_rejects_non_objects() -> None:
    """A list or string at the top level is a protocol error."""
    with pytest.raises(HostedLayaError):
        _unwrap(["not", "an", "object"])