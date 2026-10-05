"""ContextCrunch as an HTTP service.

A host application calls ``POST /v1/compact`` with its full message history just
before it makes its own LLM call, then sends the returned messages to the LLM
instead of the originals. The host keeps its own copy of the history throughout,
so a removal here costs the host nothing: it can always send the full thing
again. That is the whole contract, and it is why this service can be aggressive.

Run it with:

    uvicorn contextcrunch.api.main:create_app --factory --port 8080

**Fail-open is mandatory.** The host's LLM call has to happen whether or not
compaction worked. So no endpoint here ever returns an error for a compaction
problem. A model timeout, a dead backend, a malformed history from a third
party: all of them return the caller's original messages with ``fail_open``
set, and ``tokens_before == tokens_after`` so the host can see nothing changed.
Only genuinely invalid *input* is rejected, with 422, because that is a bug in
the caller rather than a failure of this service, and silently accepting it would
hide the bug.

The three agents run in the order the design intends: Scout decides the profile
and the goal, the Guardian says what would make the session safer, and only then
does the Compactor run. Everything is built once inside :func:`create_app` and
held on ``app.state``, because building a client at import time would make the
module unimportable without a valid configuration, and building one per request
would open a new HTTP connection pool on every call.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from contextcrunch.agents.guardian import GuardianAgent
from contextcrunch.agents.scout import AUTO, ScoutAgent
from contextcrunch.agents.worker import CompactorAgent
from contextcrunch.core.messages import as_messages, to_dicts
from contextcrunch.core.settings import Settings, get_settings
from contextcrunch.core.tokens import count_tokens
from contextcrunch.laya.engine import LayaEngine
from contextcrunch.storage.store import Store

LOG = logging.getLogger("contextcrunch.api")

#: Reported as ``profile_used`` when compaction did not run at all.
NO_PROFILE = "none"


class CompactRequest(BaseModel):
    """One history to compact.

    ``session_id`` is how the service remembers a session across calls: it is
    what lets the Guardian recognise an item it removed earlier. It is optional,
    and a request without one still compacts, it just cannot learn.
    """

    messages: list[dict[str, Any]] = Field(min_length=1)
    session_id: str = ""
    profile: str | None = None
    goal: str | None = None


class CompactResponse(BaseModel):
    """The history to send to the LLM, plus what happened on the way.

    ``messages`` is always usable. On any failure it is the caller's original
    input and ``fail_open`` is true.
    """

    messages: list[dict[str, Any]]
    session_id: str
    profile_used: str
    goal: str
    tokens_before: int
    tokens_after: int
    reduction_pct: float
    fail_open: bool
    kept: int = 0
    shortened: int = 0
    removed: int = 0
    actions: dict[str, str] = Field(default_factory=dict)
    restored: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SessionStats(BaseModel):
    """What one session has cost so far."""

    session_id: str
    decisions: dict[str, int]
    mistakes: int


def _reduction_pct(before: int, after: int) -> float:
    """Return the percentage of tokens saved, guarding a zero baseline."""
    return round(100.0 * (before - after) / before, 2) if before else 0.0


def _fail_open(
    messages: list[dict[str, Any]],
    *,
    session_id: str,
    goal: str,
    reason: str,
) -> CompactResponse:
    """Return the caller's original messages, untouched.

    ``tokens_before`` and ``tokens_after`` are set to the same value on purpose:
    the host should see that its history came back whole, not merely that
    something went wrong.
    """
    total = sum(count_tokens(str(m.get("content", ""))) for m in messages)
    return CompactResponse(
        messages=messages,
        session_id=session_id,
        profile_used=NO_PROFILE,
        goal=goal,
        tokens_before=total,
        tokens_after=total,
        reduction_pct=0.0,
        fail_open=True,
        notes=[reason],
    )


def create_app(settings: Settings | None = None, engine: LayaEngine | None = None) -> FastAPI:
    """Build the application.

    Every collaborator is built here, once, and attached to ``app.state``. The
    ``settings`` and ``engine`` arguments exist so a test can inject a fake model
    and a known configuration; production passes neither and gets the configured
    ones. Nothing is constructed at import time, so importing this module never
    needs a valid environment.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    resolved = settings or get_settings()
    resolved_engine = engine or LayaEngine(settings=resolved)
    store = Store(db_path=resolved.store_db_path, originals_dir=resolved.store_originals_dir)
    scout = ScoutAgent()
    guardian = GuardianAgent(store)
    worker = CompactorAgent(engine=resolved_engine, store=store, settings=resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Close the engine's network resources when the service stops."""
        yield
        await resolved_engine.aclose()

    app = FastAPI(
        title="ContextCrunch",
        version="0.2.0",
        summary="Shrink an agent message history without rewriting it.",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.engine = resolved_engine
    app.state.store = store
    app.state.scout = scout
    app.state.guardian = guardian
    app.state.worker = worker

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Report that the service is up and which backend it will use."""
        return {"status": "ok", "laya_mode": resolved.laya_mode}

    @app.post("/v1/compact", response_model=CompactResponse)
    async def compact(payload: CompactRequest, request: Request) -> CompactResponse:
        """Compact one history, always returning something usable.

        The order matters: Scout decides the profile and the goal, the Guardian
        says what would make this session safer, and the Compactor runs under a
        deadline. A failure at any step returns the caller's original messages.
        """
        app_state = request.app.state
        goal = payload.goal or ""

        try:
            # Parse before anything else: Scout, the Guardian and the Worker all
            # take Message objects, and a raw dict reaching any of them would be
            # an AttributeError that fail-open would then report as a compaction
            # failure. Parsing first turns a malformed message into a clear error.
            messages = as_messages(payload.messages)

            ctx = app_state.scout.run(
                messages, payload.profile or AUTO, resolved.default_profile
            )
            goal = goal or ctx.goal

            overrides = app_state.guardian.observe(payload.session_id, messages)

            compacted, stats, profile_used = await asyncio.wait_for(
                app_state.worker.run(
                    session_id=payload.session_id,
                    messages=messages,
                    ctx=ctx,
                    overrides=overrides,
                ),
                timeout=resolved.deadline_s,
            )

            tokens_before = stats.tokens_before or sum(
                count_tokens(m.content) for m in messages
            )
            # A verification failure returns the originals with zero-change stats,
            # so `changed` is the honest signal that nothing was altered.
            if not stats.changed and stats.tokens_after == 0:
                return CompactResponse(
                    messages=payload.messages,
                    session_id=payload.session_id,
                    profile_used=profile_used,
                    goal=goal,
                    tokens_before=tokens_before,
                    tokens_after=tokens_before,
                    reduction_pct=0.0,
                    fail_open=True,
                    restored=sorted(overrides.restore),
                    notes=["compaction made no change"],
                )

            return CompactResponse(
                messages=to_dicts(compacted),
                session_id=payload.session_id,
                profile_used=profile_used,
                goal=goal,
                tokens_before=tokens_before,
                tokens_after=stats.tokens_after,
                reduction_pct=_reduction_pct(tokens_before, stats.tokens_after),
                fail_open=False,
                kept=stats.kept,
                shortened=stats.shortened,
                removed=stats.removed,
                actions={str(index): action for index, action in sorted(stats.actions.items())},
                restored=sorted(overrides.restore),
            )
        except Exception as exc:  # noqa: BLE001 - fail-open is the whole point
            # The traceback goes to the log, never to the caller: a host cannot act
            # on it, and returning it would leak internals for no benefit.
            LOG.warning("compaction failed, returning the original history", exc_info=True)
            reason = (
                f"timeout after {resolved.deadline_s}s"
                if isinstance(exc, TimeoutError)
                else f"{type(exc).__name__}: {exc}"
            )
            return _fail_open(
                payload.messages, session_id=payload.session_id, goal=goal, reason=reason
            )

    @app.get("/v1/sessions/{session_id}/stats", response_model=SessionStats)
    async def session_stats(session_id: str, request: Request) -> SessionStats:
        """Return how many decisions and mistakes one session has accumulated."""
        store_: Store = request.app.state.store
        return SessionStats(
            session_id=session_id,
            decisions=store_.stats(session_id),
            mistakes=store_.mistake_count(session_id),
        )

    return app