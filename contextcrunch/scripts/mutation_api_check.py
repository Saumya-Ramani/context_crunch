"""Mutation-check the API's fail-open guarantees. Not part of the package.

Fail-open is the one property this service cannot get wrong: a host's LLM call
depends on it. Each mutation below breaks one guarantee and confirms a test
catches it, because a passing test proves nothing unless it fails when the
behaviour is removed.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\mutation_api_check.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "src" / "contextcrunch" / "api" / "main.py"

#: (label, old text, new text)
MUTATIONS: list[tuple[str, str, str]] = [
    (
        "fail-open is removed, so errors reach the host",
        "except Exception as exc:  # noqa: BLE001 - fail-open is the whole point",
        "except ZeroDivisionError as exc:",
    ),
    (
        "the deadline is removed, so a slow model stalls the host",
        "timeout=resolved.deadline_s,",
        "timeout=None,",
    ),
    (
        "fail_open is always reported false",
        "fail_open=True,\n        notes=[reason],",
        "fail_open=False,\n        notes=[reason],",
    ),
    (
        "fail-open returns an empty history instead of the original",
        "messages=messages,\n        session_id=session_id,",
        "messages=[],\n        session_id=session_id,",
    ),
    (
        "tokens_after lies on fail-open",
        "tokens_before=total,\n        tokens_after=total,",
        "tokens_before=total,\n        tokens_after=0,",
    ),
    (
        "the traceback is returned to the caller",
        "reason = (\n                f\"timeout after {resolved.deadline_s}s\"",
        "reason = (\n                traceback.format_exc() or (\n                f\"timeout after {resolved.deadline_s}s\"",
    ),
    (
        "the Guardian is skipped, so a session never learns",
        "overrides = app_state.guardian.observe(payload.session_id, messages)",
        "overrides = Overrides()",
    ),
    (
        "the session id is ignored, so stats are per-request",
        'app.get("/v1/sessions/{session_id}/stats", response_model=SessionStats)',
        'app.get("/v1/sessions/x/stats", response_model=SessionStats)',
    ),
]


def run_tests() -> tuple[int, str]:
    """Run the API tests and return (exit code, tail of output)."""
    result = subprocess.run(
        [
            str(ROOT / ".venv" / "Scripts" / "python.exe"),
            "-m",
            "pytest",
            "tests/test_api.py",
            "-q",
            "--no-header",
            "-x",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode, (result.stdout or "")[-300:]


def main() -> None:
    """Apply each mutation in turn and report whether a test caught it."""
    original = TARGET.read_text(encoding="utf-8")
    backup = Path(tempfile.gettempdir()) / "api_main_backup.py"
    shutil.copy(TARGET, backup)

    code, _ = run_tests()
    print(f"baseline (no mutation): {'PASS' if code == 0 else 'FAIL'}")
    print()

    survivors = []
    for label, old, new in MUTATIONS:
        if old not in original:
            print(f"  SKIP      {label}: pattern not found")
            survivors.append(f"{label} (pattern missing)")
            continue
        TARGET.write_text(original.replace(old, new, 1), encoding="utf-8")
        code, _ = run_tests()
        caught = code != 0
        print(f"  {'CAUGHT  ' if caught else 'SURVIVED'}  {label}")
        if not caught:
            survivors.append(label)
        shutil.copy(backup, TARGET)

    print()
    if survivors:
        print(f"{len(survivors)} mutation(s) not caught:")
        for item in survivors:
            print(f"  - {item}")
    else:
        print("every mutation was caught")


if __name__ == "__main__":
    sys.exit(main())