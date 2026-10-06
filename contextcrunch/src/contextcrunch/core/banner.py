"""Engine banner for reports.

Every report (evaluate, demo, diagnose, tune) must print a banner at the top
identifying the engine mode, Laya checkpoint, version, and date. In replay mode
it also prints cache metadata.
"""

from __future__ import annotations

import datetime
from typing import Any

from contextcrunch.core.settings import Settings


def engine_banner(settings: Settings, engine: Any) -> str:
    """Return a banner string for the current engine configuration.

    Args:
        settings: The current settings.
        engine: The engine instance (LayaEngine, CachedEngine, or ScriptedEngine).

    Returns:
        A formatted banner string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ContextCrunch Engine Banner")
    lines.append("=" * 70)
    lines.append(f"Engine mode:     {settings.laya_mode}")
    lines.append(f"Laya model:      {settings.laya_model}")
    lines.append(f"Question variant: {getattr(settings, 'question_variant', 'v1')}")
    lines.append(f"UTC date:        {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    # Try to get Laya version
    try:
        import laya
        lines.append(f"Laya version:    {laya.__version__}")
    except (ImportError, AttributeError):
        lines.append("Laya version:    unknown (not installed)")

    # Replay mode extra info
    if settings.laya_mode == "replay" and settings.replay_dir:
        from pathlib import Path
        import json

        cache_file = Path(settings.replay_dir) / "laya_cache.jsonl"
        if cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    try:
                        header = json.loads(first_line)
                        lines.append(f"REPLAY of recorded real-Laya answers")
                        lines.append(f"  recorded:      {header.get('recorded_at', 'unknown')}")
                        lines.append(f"  checkpoint:    {header.get('checkpoint', 'unknown')}")
                        lines.append(f"  laya_version:  {header.get('laya_version', 'unknown')}")
                        lines.append(f"  question_var:  {header.get('question_variant', 'unknown')}")
                        lines.append(f"  total answers: {header.get('total_entries', 'unknown')}")
                    except json.JSONDecodeError:
                        lines.append("REPLAY cache:    (header unreadable)")

    lines.append("=" * 70)
    return "\n".join(lines)


def print_banner(settings: Settings, engine: Any) -> None:
    """Print the engine banner to stdout."""
    print(engine_banner(settings, engine))