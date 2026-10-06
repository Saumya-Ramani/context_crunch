"""Cache for recorded real-Laya answers.

This module provides deterministic caching of Laya responses so they can be
replayed exactly without calling the model again. The cache is append-only
JSONL with a header record containing metadata.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from contextcrunch.core.questions import get_question_set
from contextcrunch.laya.types import LayaResult


@dataclass
class CacheHeader:
    """Metadata header for a cache file."""

    recorded_at: str
    checkpoint: str
    laya_version: str
    question_variant: str
    question_set_hash: str
    total_entries: int = 0


def cache_key(state: dict[str, Any], question_set: dict[str, Any]) -> str:
    """Return a deterministic SHA256 key for a state + question set pair.

    The key is computed from canonical JSON (sorted keys) of both inputs.
    """
    canonical = json.dumps(
        {"state": state, "questions": question_set},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def question_set_hash(question_set: dict[str, Any]) -> str:
    """Return a short hash of the question set for cache versioning."""
    canonical = json.dumps(question_set, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class CacheMiss(Exception):
    """Raised when a requested key is not in the cache."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"Cache miss for key: {key[:16]}...")


class LayaCache:
    """Append-only JSONL cache for Laya answers."""

    def __init__(self, replay_dir: str | Path) -> None:
        self._replay_dir = Path(replay_dir)
        self._replay_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._replay_dir / "laya_cache.jsonl"
        self._header: CacheHeader | None = None
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load existing cache entries from disk."""
        if not self._cache_file.exists():
            return

        with self._cache_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if line_num == 0 and "recorded_at" in record:
                        # Header record
                        self._header = CacheHeader(**record)
                    else:
                        # Data record
                        key = record["key"]
                        self._entries[key] = record
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached record for a key, or None if not found."""
        return self._entries.get(key)

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._entries

    def add(
        self,
        key: str,
        answers: LayaResult,
        raw: dict[str, Any],
        seconds: float,
        meta: dict[str, Any],
    ) -> None:
        """Add a new entry to the cache and flush to disk."""
        record = {
            "key": key,
            "answers": answers.model_dump(mode="json"),
            "raw": raw,
            "seconds": seconds,
            "meta": meta,
        }
        self._entries[key] = record

        # Append to file
        with self._cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_header(self, header: CacheHeader) -> None:
        """Write or update the header record."""
        self._header = header
        # Rewrite the entire file with new header
        self._rewrite_file()

    def _rewrite_file(self) -> None:
        """Rewrite the entire cache file with current header and entries."""
        with self._cache_file.open("w", encoding="utf-8") as f:
            if self._header:
                f.write(json.dumps(asdict(self._header), ensure_ascii=False) + "\n")
            for record in self._entries.values():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> list[str]:
        return list(self._entries.keys())


def create_cache(
    replay_dir: str | Path,
    checkpoint: str,
    laya_version: str,
    question_variant: str,
) -> LayaCache:
    """Create a new cache with a header record."""
    cache = LayaCache(replay_dir)
    question_set = get_question_set(question_variant)
    header = CacheHeader(
        recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        checkpoint=checkpoint,
        laya_version=laya_version,
        question_variant=question_variant,
        question_set_hash=question_set_hash(dict(question_set)),
    )
    cache.write_header(header)
    return cache