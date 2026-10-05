"""The compaction pipeline: Scout -> Worker -> Guardian.

This is the single entry point the API calls. It is deliberately boring: three
agents, in order, with a fail-open wrapper around all of them.

The wrapper is the important part. ContextCrunch sits in front of somebody
else's LLM call, so any bug, timeout or model failure must return the original
history rather than a broken one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contextcrunch.agents import guardian, scout, worker
from contextcrunch.core.messages import Message, as_messages
from contextcrunch.core.policy import Profile, get_profile
from contextcrunch.core.settings import get_settings
from contextcrunch.core.tokens import count_tokens
from contextcrunch.laya.client import LayaClient, build_client
from contextcrunch.storage.box import Box, build_box


@dataclass
class Cruncher:
    """A ready-to-use ContextCrunch instance."""

    client: LayaClient
    box: Box | None = None

    @classmethod
    def build(cls, *, with_box: bool = True) -> Cruncher:
        """Build an instance from the environment."""
        return cls(client=build_client(), box=build_box() if with_box else None)

    async def aclose(self) -> None:
        """Release the model client."""
        await self.client.aclose()


@dataclass
class Report:
    """The full result of one compaction, ready for the API."""

    messages: list[Message]
    profile: str
    goal: str
    tokens_before: int = 0
    tokens_after: int = 0
    degraded: bool = False
    skipped: bool = False
    actions: dict[int, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    guardian: guardian.Report = field(default_factory=guardian.Report)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view for the API response."""
        return {
            "profile": self.profile,
            "goal": self.goal,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "reduction_pct": (
                round(100.0 * (self.tokens_before - self.tokens_after) / self.tokens_before, 2)
                if self.tokens_before
                else 0.0
            ),
            "degraded": self.degraded,
            "skipped": self.skipped,
            "actions": {str(k): v for k, v in sorted(self.actions.items())},
            "notes": self.notes,
            "guardian": self.guardian.as_dict(),
        }


async def compact_history(
    raw_messages: list[dict[str, Any]],
    *,
    cruncher: Cruncher,
    profile: str | None = None,
    goal: str | None = None,
) -> Report:
    """Compact one message history.

    Never raises: any failure returns the original messages with ``degraded``
    set, so the caller's LLM call can proceed exactly as if we were not here.
    """
    settings = get_settings()
    try:
        messages = as_messages(raw_messages)
        if not messages:
            return Report(messages=[], profile=profile or settings.default_profile, goal=goal or "")

        plan = scout.pick_profile(messages, profile)
        target = goal or plan.goal
        report = Report(
            messages=messages,
            profile=plan.profile,
            goal=target,
            tokens_before=scout.total_tokens(messages),
        )

        if not scout.should_compact(messages):
            report.skipped = True
            report.tokens_after = report.tokens_before
            report.notes.append(
                f"history below CC_TRIGGER_TOKENS={settings.trigger_tokens}, left untouched"
            )
            return report

        profile_row: Profile = get_profile(plan.profile)
        outcome = await worker.compact(
            client=cruncher.client,
            messages=messages,
            goal=target,
            profile=profile_row,
            box=cruncher.box,
            profile_name=plan.profile,
            deadline_s=settings.deadline_s,
        )
        report.degraded = outcome.degraded
        report.actions = dict(outcome.actions)
        report.notes = list(outcome.notes)

        guarded, report.guardian = guardian.review(
            before=messages,
            after=outcome.messages,
            tokens_before=report.tokens_before,
            profile_name=plan.profile,
            box=cruncher.box,
            run_id=outcome.run_id,
        )
        report.messages = guarded
        report.tokens_after = sum(count_tokens(m.content) for m in guarded)
        if report.guardian.escalated_to:
            escalated = f"{report.guardian.escalated_from} -> {report.guardian.escalated_to}"
            report.notes.append(f"guardian escalated {escalated}")
        return report
    except Exception as exc:  # noqa: BLE001 - fail-open is a hard requirement
        return Report(
            messages=as_messages(raw_messages),
            profile=profile or get_settings().default_profile,
            goal=goal or "",
            degraded=True,
            notes=[f"fail-open: {type(exc).__name__}: {exc}"],
        )
