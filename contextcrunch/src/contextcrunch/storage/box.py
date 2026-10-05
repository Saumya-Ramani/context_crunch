"""The Storage Box: the side-car that makes every decision reversible.

Two things are stored for every compaction:

1. the **original** message, byte for byte, so a dropped or truncated item can
   always be restored;
2. the **decision row**: the action, the profile and every number Laya returned.

Nothing is ever truly lost. The Guardian uses this to put an item back.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contextcrunch.core.settings import get_settings
from contextcrunch.laya.types import LayaResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,
    item_index        INTEGER NOT NULL,
    role              TEXT    NOT NULL,
    label             TEXT,
    item_tokens       INTEGER NOT NULL,
    action            TEXT    NOT NULL,
    profile           TEXT    NOT NULL,
    verdict_choice    TEXT,
    verdict_conf      REAL,
    essential_noul    REAL,
    consumed_noul     REAL,
    superseded_noul   REAL,
    relevance_score   REAL,
    original_ref      TEXT,
    created_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions (run_id);
CREATE TABLE IF NOT EXISTS originals (
    ref     TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class DecisionRow:
    """One recorded decision. Every number Laya returned is kept."""

    run_id: str
    item_index: int
    role: str
    label: str
    item_tokens: int
    action: str
    profile: str
    verdict_choice: str | None = None
    verdict_conf: float | None = None
    essential_noul: float | None = None
    consumed_noul: float | None = None
    superseded_noul: float | None = None
    relevance_score: float | None = None
    original_ref: str | None = None


@dataclass
class Box:
    """SQLite side-car plus an on-disk store of originals."""

    db_path: Path
    originals_dir: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        """Create the database and the originals directory."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived connection. Async callers must not block on this."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_original(self, content: str) -> str:
        """Store an original message and return its reference.

        The reference is content addressed, so saving the same text twice costs
        nothing and always maps to the same row.
        """
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
        path = self.originals_dir / f"{digest}.txt"
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO originals (ref, payload) VALUES (?, ?)",
                (digest, content),
            )
        return digest

    def load_original(self, ref: str) -> str:
        """Return a stored original by reference, or an empty string."""
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM originals WHERE ref = ?", (ref,)).fetchone()
        if row is not None:
            return str(row["payload"])
        path = self.originals_dir / f"{ref}.txt"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def record(self, row: DecisionRow, answers: LayaResult | None = None) -> None:
        """Store one decision row."""
        payload = asdict(row)
        payload["created_at"] = datetime.now(UTC).isoformat()
        columns = ", ".join(payload)
        placeholders = ", ".join("?" for _ in payload)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT INTO decisions ({columns}) VALUES ({placeholders})",
                tuple(payload.values()),
            )
        if answers is not None:
            self.save_trace(row.run_id, row.item_index, answers)

    def save_trace(self, run_id: str, item_index: int, answers: LayaResult) -> None:
        """Store the full Laya answer set for one item, for later diagnosis."""
        path = self.originals_dir / "traces"
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{run_id}-{item_index}.json"
        target.write_text(answers.model_dump_json(indent=2), encoding="utf-8")

    def dropped(self, run_id: str) -> list[DecisionRow]:
        """Return the dropped items of a run, most relevant first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE run_id = ? AND action = ?"
                " ORDER BY relevance_score DESC",
                (run_id, "DROP"),
            ).fetchall()
        return [_to_row(row) for row in rows]

    def by_run(self, run_id: str) -> list[DecisionRow]:
        """Return every decision of a run, in item order."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE run_id = ? ORDER BY item_index", (run_id,)
            ).fetchall()
        return [_to_row(row) for row in rows]

    def summary(self, run_id: str) -> dict[str, Any]:
        """Return counts and token savings for one run, for the API response."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT action, COUNT(*) AS n, SUM(item_tokens) AS tokens"
                " FROM decisions WHERE run_id = ? GROUP BY action",
                (run_id,),
            ).fetchall()
        counts = {str(row["action"]): int(row["n"]) for row in rows}
        tokens = {str(row["action"]): int(row["tokens"] or 0) for row in rows}
        before = sum(tokens.values())
        after = before - tokens.get("DROP", 0)
        return {
            "run_id": run_id,
            "counts": counts,
            "tokens_before": before,
            "tokens_after": after,
            "reduction_pct": round(100.0 * (before - after) / before, 2) if before else 0.0,
        }


def build_box() -> Box:
    """Build the Storage Box described by ``CC_DB_PATH`` and ``CC_ORIGINALS_DIR``."""
    settings = get_settings()
    return Box(db_path=Path(settings.db_path), originals_dir=Path(settings.originals_dir))


def _to_row(row: sqlite3.Row) -> DecisionRow:
    """Turn a database row into a :class:`DecisionRow`."""
    data = {key: row[key] for key in row.keys() if key in DecisionRow.__annotations__}
    data["item_tokens"] = int(data.get("item_tokens") or 0)
    return DecisionRow(**data)


def dump_row(row: DecisionRow) -> str:
    """Return a decision row as compact JSON, for logs."""
    return json.dumps(asdict(row), default=str)
