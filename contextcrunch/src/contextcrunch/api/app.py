"""The gateway endpoint.

ContextCrunch is meant to sit in front of an existing LLM gateway as a pre-call
hook, so the API surface is intentionally tiny: one endpoint that takes a message
history and returns a smaller one, plus a health check and a way to read back a
run from the Storage Box.

The endpoint never returns an error for a compaction problem. If anything goes
wrong it returns the original history and ``degraded: true``, because the
caller's LLM call must still happen.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from contextcrunch.core.messages import to_dicts
from contextcrunch.core.pipeline import Cruncher, compact_history
from contextcrunch.core.policy import PROFILES
from contextcrunch.core.settings import get_settings
from contextcrunch.storage.box import build_box


class CompactRequest(BaseModel):
    """A request to compact one agent message history."""

    messages: list[dict[str, Any]] = Field(min_length=1)
    profile: str | None = None
    goal: str | None = None


class CompactResponse(BaseModel):
    """The compacted history plus the numbers that explain what happened."""

    messages: list[dict[str, Any]]
    profile: str
    goal: str
    tokens_before: int
    tokens_after: int
    reduction_pct: float
    degraded: bool
    skipped: bool
    actions: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    guardian: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline once at startup and close it at shutdown."""
    app.state.cruncher = Cruncher.build()
    yield
    await app.state.cruncher.aclose()


def create_app(cruncher: Cruncher | None = None) -> FastAPI:
    """Build the FastAPI application.

    ``cruncher`` lets a test inject a fake pipeline. When it is omitted the real
    one is built from the environment at startup.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.cruncher = cruncher if cruncher is not None else Cruncher.build()
        yield
        if cruncher is None:
            await app.state.cruncher.aclose()

    app = FastAPI(
        title="ContextCrunch",
        version="0.1.0",
        summary="Shrink an agent message history without rewriting it.",
        lifespan=_lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Report that the service is up and which backend it will use."""
        settings = get_settings()
        return {
            "status": "ok",
            "laya_mode": settings.laya_mode,
            "default_profile": settings.default_profile,
            "profiles": sorted(PROFILES),
        }

    @app.post("/v1/compact", response_model=CompactResponse)
    async def compact(
        request: CompactRequest, http_request: Request
    ) -> CompactResponse:
        """Compact a message history.

        Always returns a usable history. On failure the original messages come
        back untouched with ``degraded`` set.
        """
        cruncher: Cruncher = http_request.app.state.cruncher
        report = await compact_history(
            request.messages,
            cruncher=cruncher,
            profile=request.profile,
            goal=request.goal,
        )
        return CompactResponse(
            messages=to_dicts(report.messages),
            profile=report.profile,
            goal=report.goal,
            tokens_before=report.tokens_before,
            tokens_after=report.tokens_after,
            reduction_pct=report.as_dict()["reduction_pct"],
            degraded=report.degraded,
            skipped=report.skipped,
            actions={str(index): action for index, action in report.actions.items()},
            notes=report.notes,
            guardian=report.guardian.as_dict(),
        )

    @app.get("/v1/runs/{run_id}")
    async def run(run_id: str) -> dict[str, Any]:
        """Return the decisions and savings recorded for one run."""
        summary = build_box().summary(run_id)
        if not summary["counts"]:
            raise HTTPException(status_code=404, detail="run not found")
        return summary

    return app


app = create_app()
