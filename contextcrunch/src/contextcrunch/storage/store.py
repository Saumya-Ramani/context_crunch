"""The Storage Box: the side-car that makes every compaction reversible.

Two things are recorded for every decision:

1. the **original** message, byte for byte, as a file, so a removed item can be
   put back;
2. the **decision**: the action taken, the item's index, and its relevance.

Plus a third table, ``mistakes``, which the Guardian fills in when an item turns
out to have been needed after all. That table is what makes the system learn: a
repeated mistake is evidence that a profile threshold is wrong.

Nothing is ever truly destroyed, so a compaction is always reversible.

Originals are stored as files under a directory named after a **hash** of the
session id, never the raw id. Session ids come from an agent, so they are not
trusted to be path-safe; a value like ``../../etc`` must not be able to write
outside ``originals_dir``.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

#: Actions that mean content was removed from the history, in whole or in part.
#: Truncation counts: the missing middle is still gone and may still be needed.
REMOVING_ACTIONS = ("DROP", "TRUNCATE")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    session_id    TEXT    NOT NULL,
    item_index    INTEGER NOT NULL,
    fingerprint   TEXT,
    action        TEXT    NOT NULL,
    relevance     REAL,
    original_ref  TEXT,
    created_at    TEXT    NOT NULL,
    PRIMARY KEY (session_id, item_index)
);
CREATE TABLE IF NOT EXISTS mistakes (
    session_id    TEXT    NOT NULL,
    fingerprint   TEXT    NOT NULL,
    item_index    INTEGER NOT NULL,
    created_at    TEXT    NOT NULL,
    PRIMARY KEY (session_id, fingerprint, item_index)
);
"""


class Store:
    """SQLite side-car plus a directory of original messages.

    ``check_same_thread=False`` lets one connection be shared across worker
    threads, and the lock is what keeps that safe. SQLite still only serialises one
    writer at a time, so the lock is held for the whole read-modify-write of every
    method, not just the execute call.
    """

    def __init__(self, db_path: str, originals_dir: str) -> None:
        """Create the folders and tables if they do not exist yet."""
        self.db_path = Path(db_path)
        self.originals_dir = Path(originals_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.originals_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # Decisions
    # ------------------------------------------------------------------ #

    def log_decision(
        self,
        session_id: str,
        item_index: int,
        fingerprint: str | None,
        action: str,
        relevance: float | None,
        original_content: str,
    ) -> None:
        """Record one decision and save the original it refers to.

        Upserts on ``(session_id, item_index)``. Re-judging the same item therefore
        replaces the earlier verdict rather than accumulating duplicates, so
        ``top_removed`` and ``stats`` always describe the latest decision.
        """
        ref = self._save_original(session_id, item_index, original_content)
        with self._lock:
            self._conn.execute(
                "INSERT INTO decisions"
                " (session_id, item_index, fingerprint, action, relevance,"
                "  original_ref, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (session_id, item_index) DO UPDATE SET"
                "  fingerprint = excluded.fingerprint,"
                "  action = excluded.action,"
                "  relevance = excluded.relevance,"
                "  original_ref = excluded.original_ref,"
                "  created_at = excluded.created_at",
                (
                    session_id,
                    item_index,
                    fingerprint,
                    action,
                    relevance,
                    ref,
                    _now(),
                ),
            )
            self._conn.commit()

    def was_removed(self, session_id: str, fingerprint: str, before_index: int) -> bool:
        """Return True when this exact item was removed **earlier** in the session.

        The Guardian asks "did we remove something like this before?", to decide
        whether a fresh removal is a repeat of a known mistake. Three conditions
        must all hold:

        - the fingerprint matches, so it is the same content, not merely the same
          tool;
        - the action removed content, so a KEEP is not a removal;
        - the earlier index is strictly less than ``before_index``, so an item is
          not treated as evidence against its own removal.

        A fingerprint logged as ``NULL`` never matches, because an unidentified item
        cannot be shown to be the same one.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT action FROM decisions"
                " WHERE session_id = ? AND fingerprint = ? AND item_index < ?",
                (session_id, fingerprint, before_index),
            ).fetchone()
        return row is not None and str(row["action"]) in REMOVING_ACTIONS

    def top_removed(self, session_id: str, n: int) -> list[int]:
        """Return up to ``n`` removed item indexes, most relevant first.

        Relevance is the model's own estimate of how much the item mattered, so the
        highest-scoring removals are the ones most worth restoring. Items with no
        recorded relevance sort last rather than being treated as zero and lost in
        the ordering.
        """
        placeholders = ", ".join("?" for _ in REMOVING_ACTIONS)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT item_index, relevance FROM decisions"
                f" WHERE session_id = ? AND action IN ({placeholders})"
                f" ORDER BY relevance IS NULL, relevance DESC, item_index ASC"
                f" LIMIT ?",
                (session_id, *REMOVING_ACTIONS, max(n, 0)),
            ).fetchall()
        return [int(row["item_index"]) for row in rows]

    def stats(self, session_id: str) -> dict[str, int]:
        """Return how many times each action was taken in a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT action, COUNT(*) AS n FROM decisions"
                " WHERE session_id = ? GROUP BY action",
                (session_id,),
            ).fetchall()
        return {str(row["action"]): int(row["n"]) for row in rows}

    # ------------------------------------------------------------------ #
    # Mistakes
    # ------------------------------------------------------------------ #

    def log_mistake(self, session_id: str, fingerprint: str, item_index: int) -> None:
        """Record that a removal turned out to be wrong.

        Idempotent: logging the same mistake twice is a no-op, so a Guardian that
        runs repeatedly does not inflate the count.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO mistakes"
                " (session_id, fingerprint, item_index, created_at) VALUES (?, ?, ?, ?)",
                (session_id, fingerprint, item_index, _now()),
            )
            self._conn.commit()

    def mistake_count(self, session_id: str) -> int:
        """Return how many distinct mistakes a session has accumulated."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM mistakes WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------ #
    # Originals
    # ------------------------------------------------------------------ #

    def get_original(self, session_id: str, item_index: int) -> str | None:
        """Return the stored original, or ``None`` when there is none."""
        path = self._original_path(session_id, item_index)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def _save_original(self, session_id: str, item_index: int, content: str) -> str:
        """Write an original to disk and return the relative reference."""
        path = self._original_path(session_id, item_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path.name

    def _original_path(self, session_id: str, item_index: int) -> Path:
        """Return the path an original lives at.

        The session directory is a hash of the session id, so a hostile or merely
        careless id such as ``../x`` or ``a/b`` cannot escape ``originals_dir``.
        The index is coerced to ``int`` for the same reason.
        """
        digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:16]
        return self.originals_dir / digest / f"{int(item_index)}.txt"

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()


def _now() -> str:
    """Return the current UTC time as an ISO string."""
    return datetime.now(UTC).isoformat()


def build_store() -> Store:
    """Build a Store from ``CC_STORE_DB_PATH`` and ``CC_STORE_ORIGINALS_DIR``.

    These are deliberately not the same paths :class:`~contextcrunch.storage.box.Box`
    uses. Both classes create a ``decisions`` table, with different columns, and
    SQLite's ``CREATE TABLE IF NOT EXISTS`` means the first one to open a file
    decides the schema for both. Separate files make that impossible.
    """
    from contextcrunch.core.settings import get_settings

    settings = get_settings()
    return Store(db_path=settings.store_db_path, originals_dir=settings.store_originals_dir)