"""Tests for the Guardian: integrity, sanity and recovery."""

from __future__ import annotations

import pytest
from tests.conftest import history

from contextcrunch.agents.guardian import (
    SAFEST_FIRST,
    GuardianAgent,
    Overrides,
    check_integrity,
    check_sanity,
    restore,
    review,
    safer,
)
from contextcrunch.core.messages import Message
from contextcrunch.storage.box import Box, DecisionRow
from contextcrunch.storage.store import Store


@pytest.fixture
def box(tmp_path) -> Box:
    """Return a Storage Box backed by a temporary directory."""
    return Box(db_path=tmp_path / "box.db", originals_dir=tmp_path / "originals")


def _tombstone(message: Message) -> Message:
    """Return a message that looks like a ContextCrunch tombstone."""
    return message.model_copy(
        update={"content": "[removed by ContextCrunch: tool - judged not needed]"}
    )


def test_integrity_passes_for_a_clean_compaction() -> None:
    """Unchanged roles and ids mean the history is safe."""
    before = history(2)
    after = [_tombstone(m) if m.role == "tool" else m for m in before]
    assert check_integrity(before, after) == []


def test_integrity_catches_a_changed_count() -> None:
    """A lost message is a hard failure."""
    before = history(2)
    assert check_integrity(before, before[:-1])


def test_integrity_catches_a_lost_tool_call_id() -> None:
    """A tombstone without its id would break the tool-call pairing."""
    before = history(1)
    after = [m.model_copy(update={"tool_call_id": None}) for m in before]
    problems = check_integrity(before, after)
    assert any("tool_call_id lost" in problem for problem in problems)


def test_review_returns_the_original_when_broken() -> None:
    """A malformed history is worse than a long one."""
    before = history(2)
    result, report = review(
        before=before, after=before[:-1], tokens_before=100, profile_name="aggressive"
    )
    assert report.ok is False
    assert result == before


def test_sanity_notices_when_nothing_was_saved() -> None:
    """A compaction that saved nothing is suspicious."""
    assert check_sanity(1000, 1000, 0)


def test_sanity_is_happy_with_a_real_saving() -> None:
    """A normal compaction must not raise a false alarm."""
    assert check_sanity(1000, 400, 2) == []


def test_restore_puts_originals_back_in_place(box: Box) -> None:
    """Recovery replaces the tombstone in place, keeping order and ids."""
    before = history(2)
    ref = box.save_original(before[3].content)
    box.record(
        DecisionRow(
            run_id="r1",
            item_index=3,
            role="tool",
            label="tool:read_file",
            item_tokens=100,
            action="DROP",
            profile="aggressive",
            relevance_score=0.1,
            original_ref=ref,
        )
    )
    after = list(before)
    after[3] = _tombstone(after[3])
    restored, touched = restore(after, box=box, run_id="r1")
    assert touched == [3]
    assert restored[3].content == before[3].content


def test_restore_without_a_box_is_a_no_op() -> None:
    """With nothing to restore from, the tombstones stay."""
    before = history(1)
    after = [_tombstone(m) if m.role == "tool" else m for m in before]
    restored, touched = restore(after, box=None, run_id="r1")
    assert touched == []
    assert restored == after


def test_safer_only_moves_towards_safety() -> None:
    """Escalation must never make the policy more aggressive."""
    assert safer("aggressive") == "balanced"
    assert safer("balanced") == "conservative"
    assert safer("conservative") == "conservative"
    assert safer("unknown") == "conservative"


# --------------------------------------------------------------------------- #
# GuardianAgent: session-scoped observation over a Store
# --------------------------------------------------------------------------- #


@pytest.fixture
def store(tmp_path) -> Store:
    """Return a Store backed by a temporary directory."""
    return Store(db_path=str(tmp_path / "store.db"), originals_dir=str(tmp_path / "originals"))


def tool_history(*names: str) -> list[Message]:
    """Return a user message followed by tool messages with the given names.

    The user message sits at index 0, so the first tool is at index 1. A prior
    removal has to be recorded below that to count, because ``was_removed`` only
    accepts strictly earlier indexes: an item is never evidence against its own
    removal.
    """
    messages = [Message(role="user", content="do the thing")]
    for index, name in enumerate(names):
        messages.append(Message(role="tool", content="out", tool_call_id=f"c{index}", name=name))
    return messages


def record_earlier_removals(store: Store, *fingerprints: str, session: str = "s1") -> None:
    """Record removals at indexes below anything ``tool_history`` produces."""
    for index, fingerprint in enumerate(fingerprints):
        store.log_decision(session, index, fingerprint, "DROP", 0.1, f"old {fingerprint}")


def test_a_clean_session_needs_no_overrides(store: Store) -> None:
    """Nothing removed, nothing repeated, nothing stuck: carry on."""
    store.log_decision("s1", 2, "read_file", "DROP", 0.1, "old")
    overrides = GuardianAgent(store).observe("s1", tool_history("read_file", "grep"))

    assert overrides.profile is None
    assert overrides.restore == set()
    assert not overrides


def test_empty_overrides_are_falsey() -> None:
    """An empty Overrides must read as 'nothing to do'."""
    assert not Overrides()
    assert Overrides(restore={3})


# --- Rule 1: regret --------------------------------------------------------- #


def test_a_previously_removed_item_is_logged_as_a_mistake(store: Store) -> None:
    """The same item removed before, and needed again, is a mistake.

    The prior removal is recorded at index 0 and the tool observed at index 1, so
    ``was_removed`` sees a genuinely earlier decision.
    """
    record_earlier_removals(store, "read_file")
    overrides = GuardianAgent(store).observe("s1", tool_history("read_file"))

    assert store.mistake_count("s1") == 1
    assert overrides.profile is None, "one mistake is not enough to escalate"


def test_two_mistakes_force_the_conservative_profile(store: Store) -> None:
    """Two repeats is evidence the threshold is wrong, not bad luck."""
    record_earlier_removals(store, "read_file", "grep")
    overrides = GuardianAgent(store).observe("s1", tool_history("read_file", "grep"))

    assert store.mistake_count("s1") == 2
    assert overrides.profile == "conservative"


def test_a_single_mistake_does_not_escalate(store: Store) -> None:
    """The threshold must be exact: one mistake leaves the profile alone."""
    record_earlier_removals(store, "read_file")
    GuardianAgent(store).observe("s1", tool_history("read_file"))
    assert store.mistake_count("s1") == 1


def test_a_kept_item_is_not_a_mistake(store: Store) -> None:
    """Keeping something is not a removal, so it is not evidence of regret."""
    record_earlier_removals(store, "read_file")
    store.log_decision("s1", 0, "read_file", "KEEP", 2.0, "kept")
    GuardianAgent(store).observe("s1", tool_history("read_file"))

    assert store.mistake_count("s1") == 0


def test_a_different_tool_is_not_a_mistake(store: Store) -> None:
    """The same tool producing different content is a different item."""
    record_earlier_removals(store, "read_file")
    GuardianAgent(store).observe("s1", tool_history("grep"))
    assert store.mistake_count("s1") == 0


def test_an_item_is_not_evidence_against_itself(store: Store) -> None:
    """A removal at the same index cannot be an earlier removal."""
    store.log_decision("s1", 1, "read_file", "DROP", 0.1, "a")
    GuardianAgent(store).observe("s1", tool_history("read_file"))
    assert store.mistake_count("s1") == 0


def test_non_tool_messages_are_skipped(store: Store) -> None:
    """Only tool output has a fingerprint to reason about."""
    store.log_decision("s1", 0, "assistant", "DROP", 0.1, "a")
    GuardianAgent(store).observe(
        "s1",
        [Message(role="assistant", content="hi", tool_call_id="c0")],
    )
    assert store.mistake_count("s1") == 0


def test_mistakes_are_not_double_counted_on_a_second_pass(store: Store) -> None:
    """Observing the same session twice must not inflate the count."""
    record_earlier_removals(store, "read_file")
    guardian = GuardianAgent(store)
    messages = tool_history("read_file")

    guardian.observe("s1", messages)
    guardian.observe("s1", messages)
    assert store.mistake_count("s1") == 1


def test_regret_does_not_escape_the_session(store: Store) -> None:
    """Another agent's removals say nothing about this one."""
    record_earlier_removals(store, "read_file", session="other")
    GuardianAgent(store).observe("s1", tool_history("read_file"))
    assert store.mistake_count("s1") == 0


# --- Rule 2: stuck ---------------------------------------------------------- #


def test_the_same_error_three_times_triggers_a_restore(store: Store) -> None:
    """An agent going in circles has probably lost the context that helped it."""
    for index in (1, 2, 3):
        store.log_decision("s1", index, f"tool{index}", "DROP", float(index), f"body {index}")

    stuck = [Message(role="tool", content="Traceback (most recent call last): boom", name="t",
                     tool_call_id=f"c{i}") for i in range(3)]
    overrides = GuardianAgent(store).observe("s1", stuck)

    assert overrides.restore == {1, 2, 3}


def test_two_repeats_are_not_enough_to_be_stuck(store: Store) -> None:
    """The threshold is three, and it must be exact."""
    two = [Message(role="tool", content="ERROR: nope", name="t", tool_call_id=f"c{i}")
           for i in range(2)]
    assert GuardianAgent(store).observe("s1", two).restore == set()


def test_three_different_errors_are_not_being_stuck(store: Store) -> None:
    """Repetition is the signal, not the error count."""
    messages = [
        Message(
            role="tool",
            content=f"Traceback: failure number {n}",
            name="t",
            tool_call_id=f"c{n}",
        )
        for n in range(3)
    ]
    assert GuardianAgent(store).observe("s1", messages).restore == set()


def test_a_restored_set_comes_from_the_most_relevant_removals(store: Store) -> None:
    """Only the three most relevant items are restored, not everything."""
    for index in range(6):
        store.log_decision("s1", index, f"tool{index}", "DROP", float(index), f"body {index}")

    stuck = [Message(role="tool", content="Traceback: boom", name="t", tool_call_id=f"c{i}")
             for i in range(3)]
    overrides = GuardianAgent(store).observe("s1", stuck)

    assert overrides.restore == {3, 4, 5}, "the highest-relevance removals come first"


def test_being_stuck_with_nothing_removed_restores_nothing(store: Store) -> None:
    """An empty side-car must not produce invented indexes."""
    stuck = [Message(role="tool", content="Traceback: boom", name="t", tool_call_id=f"c{i}")
             for i in range(3)]
    assert GuardianAgent(store).observe("s1", stuck).restore == set()


def test_output_without_an_error_marker_is_not_stuck(store: Store) -> None:
    """Only tracebacks and 'error:' count as being stuck."""
    repeated = [Message(role="tool", content="all good", name="t", tool_call_id=f"c{i}")
                for i in range(5)]
    assert GuardianAgent(store).observe("s1", repeated).restore == set()


def test_the_markers_are_matched_case_insensitively(store: Store) -> None:
    """Tool output is not consistent about capitalisation."""
    for marker in ("Traceback", "TRACEBACK", "Error:", "ERROR:"):
        messages = [Message(role="tool", content=f"{marker} boom", name="t", tool_call_id=f"c{i}")
                    for i in range(3)]
        assert GuardianAgent(store).is_stuck(messages), marker


def test_differences_beyond_the_prefix_are_ignored(store: Store) -> None:
    """A changing line number or timing must not hide a repeat.

    Only the first 200 characters identify the failure, so trailing noise cannot
    disguise the same error as a new one.
    """
    messages = [
        Message(role="tool", content="Traceback: boom " + "x" * 250 + f" line {n}",
                name="t", tool_call_id=f"c{n}")
        for n in range(3)
    ]
    assert GuardianAgent(store).is_stuck(messages)


# --- Contract: never raise --------------------------------------------------- #


def test_a_broken_store_returns_empty_overrides(capsys) -> None:
    """A Guardian that crashes must not break the caller's request."""
    broken = Store(db_path=":memory:", originals_dir=".")
    broken._conn.close()  # every later call raises

    overrides = GuardianAgent(broken).observe("s1", tool_history("read_file"))

    assert overrides.profile is None
    assert overrides.restore == set()
    assert not overrides
    assert "guardian failed" in capsys.readouterr().err


def test_a_store_raising_midway_still_returns_overrides() -> None:
    """The except must cover the whole rule set, not just one call."""
    class HalfBroken(Store):
        def mistake_count(self, session_id: str) -> int:
            raise RuntimeError("disk gone")

    store = HalfBroken(db_path=":memory:", originals_dir=".")
    assert not GuardianAgent(store).observe("s1", tool_history("read_file"))


# --- The Guardian may only make things safer -------------------------------- #


def test_the_guardian_only_ever_escalates(store: Store) -> None:
    """Overriding must never move towards a more aggressive policy."""
    for index in range(4):
        store.log_decision("s1", index, f"tool{index}", "DROP", 1.0, f"b{index}")

    stuck = [Message(role="tool", content="Traceback: boom", name="t", tool_call_id=f"c{i}")
             for i in range(3)]
    overrides = GuardianAgent(store).observe("s1", stuck)

    assert overrides.profile in (None, "conservative")
    assert overrides.restore
    assert SAFEST_FIRST.index("conservative") == 0, "conservative must be the safest row"
