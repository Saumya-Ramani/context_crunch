"""The hosted Laya backend.

Posts to the inference API with a bearer token:

    POST https://api.impossibl.com/v1/systemone
    Authorization: Bearer <key>
    {"model": ..., "state": ..., "questions": {...}}

The request shape is exactly the one agreed with the platform: ``state`` is the
packed context, ``questions`` is our five-question set, and nothing else is sent.
Every response is parsed through the same contract as the local model, so the
policy engine cannot tell the two apart.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from contextcrunch.core.questions import QUESTION_SET, QUESTION_TYPES
from contextcrunch.core.settings import get_settings
from contextcrunch.laya.types import LayaResult, parse_result

#: Status codes worth one more attempt: rate limits and transient server faults.
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class HostedLayaError(RuntimeError):
    """Raised when the hosted API cannot give us a usable answer."""


class HostedBackend:
    """Call the hosted Laya API over HTTPS with a bearer token."""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        settings = get_settings()
        self._url = settings.laya_api_url
        self._api_key = settings.laya_api_key
        self._model = settings.laya_model
        self._max_len = settings.laya_max_len
        self._retries = max(settings.laya_retries, 0)
        self._backoff = max(settings.laya_backoff_s, 0.0)
        self._timeout = float(settings.deadline_s)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    @property
    def mode(self) -> str:
        """Return the backend name, for logs and metrics."""
        return "hosted"

    def _get_client(self) -> httpx.AsyncClient:
        """Return a lazily created client, so import stays cheap and testable."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=min(self._timeout, 10.0)),
                transport=self._transport,
            )
        return self._client

    def build_payload(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build the request body. Exposed so tests can assert the exact shape."""
        return {
            "model": self._model,
            "state": state,
            "questions": dict(QUESTION_SET),
        }

    def build_headers(self) -> dict[str, str]:
        """Build the auth headers. Exposed so tests never touch the real key."""
        if not self._api_key:
            raise HostedLayaError(
                "no API key configured; set Laya_Api_key in .env or CC_LAYA_API_KEY"
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def predict(self, state: dict[str, Any], label: str = "") -> LayaResult:
        """Return Laya's answers for one state, retrying transient failures."""
        headers = self.build_headers()
        payload = self.build_payload(state)
        client = self._get_client()
        last_error: Exception | None = None

        for attempt in range(self._retries + 1):
            try:
                response = await client.post(self._url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    return self._parse(response.json(), label)
                detail = response.text[:300]
                last_error = HostedLayaError(
                    f"HTTP {response.status_code} from {self._url}: {detail}"
                )
                if response.status_code not in RETRY_STATUS:
                    raise last_error from None
            if attempt < self._retries:
                await asyncio.sleep(self._backoff * (2**attempt))

        raise HostedLayaError(
            f"hosted Laya failed after {self._retries + 1} attempts: {last_error}"
        )

    def _parse(self, data: Any, label: str) -> LayaResult:
        """Parse a hosted response, tolerating the small wrapper shapes used."""
        payload = _unwrap(data)
        if "answers" not in payload:
            raise HostedLayaError(f"hosted Laya returned no answers for {label!r}: {list(payload)}")
        return parse_result(payload, dict(QUESTION_TYPES))

    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _unwrap(data: Any) -> dict[str, Any]:
    """Return the payload with the answers in it.

    Platforms wrap the result differently, so accept the answer map itself, a
    top-level ``answers`` key, or a single-item list of results.
    """
    if not isinstance(data, dict):
        raise HostedLayaError(f"hosted Laya returned {type(data).__name__}, not an object")
    if "answers" in data:
        return data
    for key in ("result", "data", "output"):
        inner = data.get(key)
        if isinstance(inner, dict) and "answers" in inner:
            return inner
    if isinstance(data.get("results"), list) and data["results"]:
        first = data["results"][0]
        if isinstance(first, dict):
            return first if "answers" in first else {"answers": first}
    return data
