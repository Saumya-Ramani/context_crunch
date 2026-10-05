"""Guardian: detect mistakes, restore items, switch to the safer mode.

The Guardian is the last line of defence. It runs after the Worker and it is
allowed to be paranoid, because its mistakes cost tokens while the Worker's cost
information.

Two things live here.

:class:`GuardianAgent` is the session-scoped observer. It reads the Storage Box
and the current history and returns :class:`Overrides` telling the pipeline what
to change: items to restore, or a profile to switch to.

The module-level functions (``check_integrity``, ``review``, ``restore``) are the
per-compaction review that runs against a :class:`~contextcrunch.storage.box.Box`.

The Guardian may only ever make things **safer** by itself. It can restore items
and it can switch to the conservative profile; it can never loosen a threshold,
drop more, or keep less than it was asked to. That asymmetry is deliberate: the
Guardian is guessing, and a guess that removes information is worse than a guess
that wastes tokens.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from contextcrunch.core.messages import Message
from contextcrunch.core.tokens import count_tokens
from contextcrunch.storage.box import Box
from contextcrunch.storage.store import Store

#: Order from safest to most aggressive. Escalation only ever moves left.
SAFEST_FIRST: tuple[str, ...] = ("conservative", "balanced", "aggressive")

#: Markers that identify an error in a tool output.
STUCK_MARKERS: tuple[str, ...] = ("traceback", "error:")

#: How many characters of an error identify it as the same one.
STUCK_PREFIX_CHARS = 200

#: How many repeats of the same error mean the agent is going in circles.
STUCK_REPEAT_THRESHOLD = 3

#: How many of the most relevant removals to restore when stuck.
STUCK_RESTORE_COUNT = 3

#: Mistake count at which the Guardian forces the safest profile.
REGRET_DOWNSHIFT_AT = 2


@dataclass
class Overrides:
    """What the Guardian wants changed about the next compaction.

    ``profile`` of ``None`` means "carry on with whatever was chosen". ``restore``
    is a set of item indexes to put back from the side-car.
    """

    profile: str | None = None
    restore: set[int] = field(default_factory=set)

    def __bool__(self) -> bool:
        """Return True when the Guardian wants something changed."""
        return self.profile is not None or bool(self.restore)


class GuardianAgent:
    """Observe a session and say what would make it safer.

    Rule-based and never calls a model. Every rule it applies can only add
    information back or move to a stricter profile.
    """

    def __init__(self, store: Store) -> None:
        self.store = store

    def observe(self, session_id: str, messages: list[Message]) -> Overrides:
        """Return the overrides this session needs. Never raises.

        A Guardian that crashes must not be able to break the caller's request, so
        any failure is logged and reported as "no change". Returning empty overrides
        is the safe outcome: the compaction proceeds exactly as it would have
        without the Guardian.
        """
        try:
            overrides = Overrides()
            self._apply_regret(session_id, messages, overrides)
            self._apply_stuck(session_id, messages, overrides)
            return overrides
        except Exception as exc:  # noqa: BLE001 - required by the contract
            print(
                f"contextcrunch: guardian failed ({type(exc).__name__}: {exc}); "
                "continuing without overrides",
                file=sys.stderr,
            )
            return Overrides()

    def _apply_regret(
        self, session_id: str, messages: list[Message], overrides: Overrides
    ) -> None:
        """Rule 1: the same item was removed before, and is being needed again.

        Each hit is recorded as a mistake, and once there are enough of them the
        Guardian stops trusting the current profile and forces the safest one. Two
        repeats of the same mistake is evidence the threshold is wrong, not bad luck.
        """
        for index, message in enumerate(messages):
            if message.role != "tool":
                continue
            fingerprint = message.fingerprint()
            if not fingerprint:
                continue
            if self.store.was_removed(session_id, fingerprint, before_index=index):
                self.store.log_mistake(session_id, fingerprint, index)

        if self.store.mistake_count(session_id) >= REGRET_DOWNSHIFT_AT:
            overrides.profile = "conservative"

    def _apply_stuck(
        self, session_id: str, messages: list[Message], overrides: Overrides
    ) -> None:
        """Rule 2: the same error keeps coming back, so evidence is being lost.

        An agent stuck on one error has probably lost the context that told it what
        it had already tried. The most relevant recent removals are put back.
        """
        if not self.is_stuck(messages):
            return
        overrides.restore = set(self.store.top_removed(session_id, STUCK_RESTORE_COUNT))

    def is_stuck(self, messages: list[Message]) -> bool:
        """Return True when the same error appears at least three times.

        Matching on the first 200 characters identifies the same failure without
        being fooled by a changing line number or timing in the output.
        """
        seen: dict[str, int] = {}
        for message in messages:
            lowered = message.content.lower()
            if not any(marker in lowered for marker in STUCK_MARKERS):
                continue
            prefix = lowered[:STUCK_PREFIX_CHARS]
            seen[prefix] = seen.get(prefix, 0) + 1
            if seen[prefix] >= STUCK_REPEAT_THRESHOLD:
                return True
        return False


@dataclass
class Report:
    """What the per-compaction review found."""

    ok: bool = True
    restored: list[int] = field(default_factory=list)
    escalated_from: str | None = None
    escalated_to: str | None = None
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly view for the API response."""
        return {
            "ok": self.ok,
            "restored": self.restored,
            "escalated_from": self.escalated_from,
            "escalated_to": self.escalated_to,
            "reasons": self.reasons,
        }


def check_integrity(before: list[Message], after: list[Message]) -> list[str]:
    """Return the structural problems found in a compacted history."""
    problems: list[str] = []
    if len(before) != len(after):
        return [f"message count changed: {len(before)} -> {len(after)}"]
    for old, new in zip(before, after, strict=True):
        if old.role != new.role:
            problems.append(f"role changed at index {old.tool_call_id or old.label()}")
        if old.tool_call_id and new.tool_call_id != old.tool_call_id:
            problems.append(f"tool_call_id lost at index {old.tool_call_id}")
        if new.content and not isinstance(new.content, str):
            problems.append(f"content is not text at index {new.label()}")
    return problems


def check_sanity(tokens_before: int, tokens_after: int, dropped: int) -> list[str]:
    """Return the reasons to distrust this compaction."""
    reasons: list[str] = []
    if tokens_before > 0 and tokens_after >= tokens_before:
        reasons.append("compaction saved nothing")
    if dropped == 0 and tokens_before > 0:
        reasons.append("nothing was dropped but the history was above the trigger")
    return reasons


def review(
    *,
    before: list[Message],
    after: list[Message],
    tokens_before: int,
    profile_name: str,
    box: Box | None = None,
    run_id: str = "",
) -> tuple[list[Message], Report]:
    """Review a compaction and repair it when needed.

    Returns the history that should be sent on, plus a report explaining what
    happened. The repair is always "put the originals back", never "rewrite".
    """
    report = Report()
    problems = check_integrity(before, after)
    dropped = sum(1 for message in after if _is_tombstone(message))
    sanity = check_sanity(tokens_before, sum(count_tokens(m.content) for m in after), dropped)

    if problems:
        report.ok = False
        report.reasons = problems
        return list(before), report

    report.reasons = sanity
    if not sanity:
        return after, report

    restored, report.restored = restore(after, box=box, run_id=run_id)
    report.escalated_from = profile_name
    report.escalated_to = safer(profile_name)
    return restored, report


def restore(
    messages: list[Message], *, box: Box | None = None, run_id: str = ""
) -> tuple[list[Message], list[int]]:
    """Put the most relevant dropped items back, in place.

    Restoration is in-place so the message order and the tool-call pairing stay
    valid. Without a Box there is nothing to restore from, so the tombstones stay.
    """
    if box is None or not run_id:
        return list(messages), []

    candidates = box.dropped(run_id)
    if not candidates:
        return list(messages), []

    restored = list(messages)
    touched: list[int] = []
    for row in candidates:
        index = row.item_index
        if index < 0 or index >= len(restored) or not _is_tombstone(restored[index]):
            continue
        original = box.load_original(row.original_ref or "")
        if not original:
            continue
        restored[index] = restored[index].model_copy(update={"content": original})
        touched.append(index)
    return restored, touched


def safer(profile_name: str) -> str:
    """Return the next safer profile name, or the safest one when unknown."""
    if profile_name not in SAFEST_FIRST:
        return SAFEST_FIRST[0]
    position = SAFEST_FIRST.index(profile_name)
    return SAFEST_FIRST[max(position - 1, 0)]


def _is_tombstone(message: Message) -> bool:
    """Return True when this message is a tombstone ContextCrunch produced."""
    return "ContextCrunch" in message.content and message.content.startswith("[")
