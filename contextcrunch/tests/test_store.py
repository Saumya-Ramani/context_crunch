"""Tests for the Storage Box.

Every test uses ``tmp_path``, so nothing touches a real database or a real
directory of originals. The path-traversal tests matter most: session ids come
from an agent, so they are untrusted input.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from contextcrunch.storage.store import REMOVING_ACTIONS, Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    """Return a Store backed by a temporary directory."""
    return Store(db_path=str(tmp_path / "store.db"), originals_dir=str(tmp_path / "originals"))


def log(store: Store, index: int, *, action: str = "DROP", fp: str = "fp1",
        relevance: float | None = 1.0, content: str = "body") -> None:
    """Log one decision with sensible defaults."""
    store.log_decision("s1", index, fp, action, relevance, content)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_creates_folders_and_tables(tmp_path: Path) -> None:
    """A Store must be usable immediately, without a separate setup step."""
    db = tmp_path / "nested" / "store.db"
    originals = tmp_path / "nested" / "originals"
    store = Store(db_path=str(db), originals_dir=str(originals))

    assert db.is_file()
    assert originals.is_dir()
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in rows}
    assert {"decisions", "mistakes"} <= tables
    store.close()


def test_reopening_an_existing_store_keeps_its_rows(tmp_path: Path) -> None:
    """Reopening must not wipe the side-car: that would destroy recovery."""
    paths = (str(tmp_path / "store.db"), str(tmp_path / "originals"))
    first = Store(*paths)
    first.log_decision("s1", 1, "fp1", "DROP", 1.0, "body")
    first.close()

    second = Store(*paths)
    assert second.stats("s1") == {"DROP": 1}
    assert second.get_original("s1", 1) == "body"
    second.close()


# --------------------------------------------------------------------------- #
# log_decision
# --------------------------------------------------------------------------- #


def test_log_decision_records_the_action(store: Store) -> None:
    """The decision row is what the Guardian reads to judge a later removal."""
    store.log_decision("s1", 4, "fp1", "TRUNCATE", 2.5, "the original text")
    assert store.stats("s1") == {"TRUNCATE": 1}


def test_log_decision_upserts_rather_than_duplicating(store: Store) -> None:
    """Re-judging an item replaces the old verdict, so stats stay truthful."""
    log(store, 3, action="KEEP")
    log(store, 3, action="DROP", relevance=0.5)

    assert store.stats("s1") == {"DROP": 1}, "the second decision must replace the first"
    assert store.get_original("s1", 3) == "body", "the original must be refreshed too"


def test_upsert_updates_the_fingerprint_and_relevance(store: Store) -> None:
    """Every field of the verdict has to be replaced, not just the action."""
    log(store, 3, fp="old", relevance=0.1)
    log(store, 3, fp="new", relevance=9.0)

    assert store.was_removed("s1", "new", before_index=10) is True
    assert store.was_removed("s1", "old", before_index=10) is False
    assert store.top_removed("s1", 5) == [3]


def test_decisions_are_scoped_to_a_session(store: Store) -> None:
    """Two agents sharing a database must not see each other's decisions."""
    log(store, 1)
    store.log_decision("s2", 1, "fp1", "DROP", 1.0, "body")

    assert store.stats("s1") == {"DROP": 1}
    assert store.stats("s2") == {"DROP": 1}
    # Both sessions removed index 1, but neither can use the other's removal as
    # evidence: a mistake in one agent's run says nothing about another's.
    assert store.was_removed("s3", "fp1", before_index=5) is False
    assert store.mistake_count("s3") == 0


def test_a_null_fingerprint_is_allowed(store: Store) -> None:
    """Some items have no identifiable tool, and must still be recordable."""
    log(store, 1, fp=None)
    assert store.stats("s1") == {"DROP": 1}


# --------------------------------------------------------------------------- #
# was_removed
# --------------------------------------------------------------------------- #


def test_was_removed_true_for_an_earlier_matching_removal(store: Store) -> None:
    """The Guardian's core question: have we removed this before?"""
    log(store, 2, fp="fp1", action="DROP")
    assert store.was_removed("s1", "fp1", before_index=5) is True


def test_was_removed_true_for_truncate(store: Store) -> None:
    """Truncation removes the middle, so it counts as a removal too."""
    log(store, 2, fp="fp1", action="TRUNCATE")
    assert store.was_removed("s1", "fp1", before_index=5) is True


def test_was_removed_false_at_the_same_index(store: Store) -> None:
    """An item is never evidence against its own removal."""
    log(store, 5, fp="fp1", action="DROP")
    assert store.was_removed("s1", "fp1", before_index=5) is False


def test_was_removed_false_for_a_later_index(store: Store) -> None:
    """Only earlier removals count as evidence of a repeat."""
    log(store, 9, fp="fp1", action="DROP")
    assert store.was_removed("s1", "fp1", before_index=5) is False


def test_was_removed_false_for_a_keep(store: Store) -> None:
    """Keeping an item is not a removal, however old it is."""
    log(store, 2, fp="fp1", action="KEEP")
    assert store.was_removed("s1", "fp1", before_index=5) is False


def test_was_removed_false_for_a_different_fingerprint(store: Store) -> None:
    """The same tool producing different content is a different item."""
    log(store, 2, fp="fp_read_file", action="DROP")
    assert store.was_removed("s1", "fp_grep", before_index=5) is False


def test_was_removed_false_for_a_null_fingerprint(store: Store) -> None:
    """An unidentified item cannot be shown to be the same one."""
    log(store, 2, fp=None, action="DROP")
    assert store.was_removed("s1", "fp1", before_index=5) is False


def test_was_removed_false_in_an_unknown_session(store: Store) -> None:
    """No history means no evidence."""
    log(store, 2, fp="fp1", action="DROP")
    assert store.was_removed("other", "fp1", before_index=5) is False


def test_every_removing_action_counts(store: Store) -> None:
    """Any action that removes content is evidence; nothing else is."""
    log(store, 2, action="KEEP")
    assert store.was_removed("s1", "fp1", before_index=9) is False
    for action in REMOVING_ACTIONS:
        log(store, 2, action=action)
        assert store.was_removed("s1", "fp1", before_index=9) is True


# --------------------------------------------------------------------------- #
# Mistakes
# --------------------------------------------------------------------------- #


def test_mistake_count_starts_at_zero(store: Store) -> None:
    """A clean session has no mistakes."""
    assert store.mistake_count("s1") == 0


def test_log_mistake_counts_once(store: Store) -> None:
    """A recorded mistake must be visible to the Guardian."""
    store.log_mistake("s1", "fp1", 3)
    assert store.mistake_count("s1") == 1


def test_log_mistake_is_idempotent(store: Store) -> None:
    """A Guardian that runs repeatedly must not inflate the count."""
    store.log_mistake("s1", "fp1", 3)
    store.log_mistake("s1", "fp1", 3)
    store.log_mistake("s1", "fp1", 3)
    assert store.mistake_count("s1") == 1


def test_distinct_mistakes_are_counted_separately(store: Store) -> None:
    """Different items, or the same item in different sessions, are different."""
    store.log_mistake("s1", "fp1", 3)
    store.log_mistake("s1", "fp2", 3)
    store.log_mistake("s1", "fp1", 4)
    store.log_mistake("s2", "fp1", 3)

    assert store.mistake_count("s1") == 3
    assert store.mistake_count("s2") == 1


# --------------------------------------------------------------------------- #
# top_removed
# --------------------------------------------------------------------------- #


def test_top_removed_orders_by_relevance(store: Store) -> None:
    """The most relevant removals are the ones most worth restoring."""
    log(store, 1, relevance=0.2)
    log(store, 2, relevance=2.9)
    log(store, 3, relevance=1.5)

    assert store.top_removed("s1", 10) == [2, 3, 1]


def test_top_removed_excludes_kept_items(store: Store) -> None:
    """Only removed items are candidates for restoration."""
    log(store, 1, relevance=9.9, action="KEEP")
    log(store, 2, relevance=0.1, action="DROP")

    assert store.top_removed("s1", 10) == [2]


def test_top_removed_includes_truncations(store: Store) -> None:
    """A truncated item lost its middle, so it is restorable too."""
    log(store, 1, relevance=5.0, action="TRUNCATE")
    assert store.top_removed("s1", 10) == [1]


def test_top_removed_respects_the_limit(store: Store) -> None:
    """The Guardian restores a few items, not the whole history."""
    for index in range(5):
        log(store, index, relevance=float(index))
    assert store.top_removed("s1", 2) == [4, 3]


def test_top_removed_with_zero_or_negative_n(store: Store) -> None:
    """Asking for nothing must return nothing, not everything."""
    log(store, 1, relevance=1.0)
    assert store.top_removed("s1", 0) == []
    assert store.top_removed("s1", -5) == []


def test_top_removed_puts_unscored_items_last(store: Store) -> None:
    """An item with no relevance must not be lost in the ordering."""
    log(store, 1, relevance=None)
    log(store, 2, relevance=0.1)
    assert store.top_removed("s1", 10) == [2, 1]


def test_top_removed_of_an_unknown_session_is_empty(store: Store) -> None:
    """No session, no removals."""
    assert store.top_removed("nope", 10) == []


def test_ties_break_on_index_so_ordering_is_stable(store: Store) -> None:
    """Equal relevance must still give a deterministic order."""
    log(store, 3, relevance=1.0)
    log(store, 1, relevance=1.0)
    assert store.top_removed("s1", 10) == [1, 3]


# --------------------------------------------------------------------------- #
# Originals
# --------------------------------------------------------------------------- #


def test_get_original_round_trips_verbatim(store: Store) -> None:
    """Recovery must reproduce the item exactly, byte for byte."""
    content = "line one\nline two\ttabbed\nunicode: éè中文\n"
    store.log_decision("s1", 2, "fp1", "DROP", 1.0, content)
    assert store.get_original("s1", 2) == content


def test_get_original_of_an_unknown_item_is_none(store: Store) -> None:
    """None and an empty string must be distinguishable."""
    assert store.get_original("s1", 99) is None


def test_get_original_is_scoped_to_the_session(store: Store) -> None:
    """Two sessions must not read each other's originals."""
    store.log_decision("s1", 0, "fp1", "DROP", 1.0, "first session")
    store.log_decision("s2", 0, "fp1", "DROP", 1.0, "second session")

    assert store.get_original("s1", 0) == "first session"
    assert store.get_original("s2", 0) == "second session"


def test_relogging_replaces_the_original_file(store: Store) -> None:
    """A re-judged item must recover to its latest text, not the first."""
    log(store, 1, content="first version")
    log(store, 1, content="second version")
    assert store.get_original("s1", 1) == "second version"


def test_empty_content_round_trips(store: Store) -> None:
    """An empty original is still an original, and must not read as missing."""
    log(store, 1, content="")
    assert store.get_original("s1", 1) == ""


# --------------------------------------------------------------------------- #
# Path safety: the session id is untrusted input
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "session_id",
    ["../x", "a/b", "../../etc/passwd", "..", "a/../../b", "/absolute", "a\\b", ""],
)
def test_hostile_session_ids_stay_inside_originals_dir(
    store: Store, tmp_path: Path, session_id: str
) -> None:
    """A session id must never be able to write outside originals_dir.

    Session ids come from the calling agent, so a value like ``../x`` is hostile
    input, not a hypothetical.
    """
    store.log_decision(session_id, 1, "fp1", "DROP", 1.0, "body")

    originals = (tmp_path / "originals").resolve()
    written = [p for p in originals.rglob("*") if p.is_file()]
    assert len(written) == 1, "exactly one original should have been written"
    assert written[0].resolve().is_relative_to(originals)
    assert store.get_original(session_id, 1) == "body"


@pytest.mark.parametrize("session_id", ["../x", "a/b", ".."])
def test_hostile_session_ids_cannot_read_outside(store: Store, session_id: str) -> None:
    """A traversal id must not be able to read a file it did not write."""
    assert store.get_original(session_id, 1) is None


def test_original_ref_is_a_bare_filename(store: Store) -> None:
    """The reference stored in the database must not itself contain a path."""
    store.log_decision("a/b", 7, "fp1", "DROP", 1.0, "body")
    with sqlite3.connect(store.db_path) as conn:
        ref = conn.execute("SELECT original_ref FROM decisions").fetchone()[0]
    assert "/" not in ref
    assert ref.endswith(".txt")


def test_a_hostile_item_index_cannot_escape(store: Store) -> None:
    """The index reaches the path too, so it is coerced rather than trusted."""
    store.log_decision("s1", 5, "fp1", "DROP", 1.0, "body")
    assert store.get_original("s1", 5) == "body"
    assert store.get_original("s1", 6) is None


# --------------------------------------------------------------------------- #
# Threading
# --------------------------------------------------------------------------- #


def test_concurrent_writes_do_not_lose_rows(store: Store) -> None:
    """One connection is shared across threads, so the lock has to hold.

    ``check_same_thread=False`` allows sharing but does not serialise writes, so
    without the lock this either raises "database is locked" or drops rows.
    """
    errors: list[Exception] = []

    def write(index: int) -> None:
        try:
            for step in range(10):
                store.log_decision("s1", index * 10 + step, "fp", "DROP", 1.0, f"body {step}")
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.stats("s1") == {"DROP": 40}


def test_concurrent_mistake_logging_is_idempotent(store: Store) -> None:
    """Racing INSERT OR IGNORE must still record the mistake exactly once."""
    def write() -> None:
        store.log_mistake("s1", "fp1", 3)

    threads = [threading.Thread(target=write) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.mistake_count("s1") == 1