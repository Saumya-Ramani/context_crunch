"""All tunable numbers for ContextCrunch come from the environment.

Rules enforced here:
- every tunable is a ``Settings`` field with the ``CC_`` prefix and **no default**;
- profile thresholds live in :mod:`contextcrunch.core.policy` and nowhere else;
- if the environment is incomplete we warn on stderr and fall back to
  :data:`EMERGENCY_DEFAULTS` so the service keeps serving (fail-open).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

LayaMode = Literal["http", "serve", "inprocess", "fake"]

#: Last-resort values, used only when a ``CC_`` variable is missing.
#: They are deliberately kept in one place so they are easy to spot in review.
EMERGENCY_DEFAULTS: dict[str, str] = {
    "CC_LAYA_MODE": "fake",
    "CC_LAYA_MODEL": "convaiinnovations/laya",
    "CC_LAYA_API_URL": "https://api.impossibl.com/v1/systemone",
    "CC_LAYA_API_KEY": "",
    "CC_LAYA_MAX_LEN": "8192",
    "CC_LAYA_CONCURRENCY": "8",
    "CC_LAYA_RETRIES": "2",
    "CC_LAYA_BACKOFF_S": "0.5",
    "CC_TIKTOKEN_ENCODING": "cl100k_base",
    "CC_LAYA_HTTP_BASE_URL": "http://127.0.0.1:8001",
    "CC_ITEM_MAX_CHARS": "12000",
    "CC_FINDINGS_CAP_CHARS": "4000",
    "CC_GOAL_MAX_CHARS": "1000",
    "CC_STATE_MAX_TOKENS": "6000",
    "CC_PIECE_TARGET_CHARS": "2500",
    "CC_PIECE_MAX_CHARS": "3500",
    "CC_TRIGGER_TOKENS": "2000",
    "CC_DEADLINE_S": "30",
    "CC_DEFAULT_PROFILE": "conservative",
    "CC_DB_PATH": "data/contextcrunch.db",
    "CC_ORIGINALS_DIR": "data/originals",
    # The session-scoped Store keeps its own database on purpose. Box and Store
    # both define a `decisions` table with different columns, so sharing one file
    # would mean whichever opened it first defines the schema for both.
    "CC_STORE_DB_PATH": "data/contextcrunch_store.db",
    "CC_STORE_ORIGINALS_DIR": "data/store_originals",
}


class Settings(BaseSettings):
    """Runtime configuration read from ``.env`` using the ``CC_`` prefix."""

    model_config = SettingsConfigDict(
        env_prefix="CC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Laya engine: http | serve | inprocess | fake
    laya_mode: LayaMode
    laya_model: str
    #: Full URL of the hosted Laya inference endpoint, bearer authenticated.
    laya_api_url: str    #: Bearer token for the hosted endpoint. Read from the environment only,
    #: never from the emergency defaults, and hidden from reprs and logs.
    laya_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("CC_LAYA_API_KEY", "Laya_Api_key", "LAYA_API_KEY"),
    )
    laya_max_len: int
    laya_concurrency: int
    #: How many times a transient hosted-API failure is retried.
    laya_retries: int
    #: Base delay between hosted-API retries, in seconds.
    laya_backoff_s: float
    #: Base URL of a self-hosted ``python -m laya.serve`` instance.
    laya_http_base_url: str
    tiktoken_encoding: str

    # Size budgets for one Laya call
    item_max_chars: int
    findings_cap_chars: int
    goal_max_chars: int
    state_max_tokens: int

    # Splitting big items into pieces
    piece_target_chars: int
    piece_max_chars: int

    # Behaviour
    trigger_tokens: int
    deadline_s: int
    default_profile: str

    # Storage Box (Box, per compaction run)
    db_path: str
    originals_dir: str
    # Storage Box (Store, per session). Separate from the above by design.
    store_db_path: str
    store_originals_dir: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, warning and degrading if config is broken."""
    try:
        return Settings()
    except ValidationError as exc:
        missing = sorted({str(err["loc"][0]) for err in exc.errors() if err["type"] == "missing"})
        recoverable = {name: EMERGENCY_DEFAULTS[f"CC_{name.upper()}"] for name in missing}
        # The API key is a secret, never defaulted. Leaving it out keeps it empty,
        # which makes the hosted backend fail open instead of sending a wrong one.
        recoverable.pop("laya_api_key", None)
        print(
            f"contextcrunch: incomplete configuration ({', '.join(missing)}); "
            "falling back to emergency defaults",
            file=sys.stderr,
        )
        return Settings(**recoverable)


def reset_settings_cache() -> None:
    """Clear the settings cache. Used by tests that change the environment."""
    get_settings.cache_clear()
