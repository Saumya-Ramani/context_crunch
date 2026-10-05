"""Mutation-check the Worker's critical guarantees. Not part of the package.

Each mutation removes one safety behaviour and confirms a test fails. A passing
test proves nothing unless it fails when the behaviour is removed.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\mutation_check.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "src" / "contextcrunch" / "agents" / "worker.py"

#: (label, old text, new text)
MUTATIONS: list[tuple[str, str, str]] = [
    (
        "VERIFY never runs",
        "if self._verify(messages, compacted):",
        "if False:",
    ),
    (
        "the trigger is ignored",
        "if tokens_before < self.settings.trigger_tokens:",
        "if False:",
    ),
    (
        "recent turns stop being protected",
        "or (total - index) <= profile.K",
        "or False",
    ),
    (
        "a pinned tool is no longer protected",
        "or message.name in ctx.pinned_tools",
        "or False",
    ),
    (
        "a restored item is no longer restored",
        "index in overrides.restore",
        "False",
    ),
    (
        "only tool messages are judged",
        'if message.role != "tool":',
        "if False:",
    ),
    (
        "the store is never written to",
        "self.store.log_decision(",
        "_unused_log_decision(",
    ),
    (
        "empty content is allowed through",
        "elif not new.content.strip():",
        "elif False:",
    ),
    (
        "a lost tool_call_id is allowed",
        "if old.tool_call_id and new.tool_call_id != old.tool_call_id:",
        "if False:",
    ),
]


def run_tests() -> tuple[int, str]:
    """Run the worker tests and return (exit code, tail of output)."""
    result = subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "pytest",
         "tests/test_worker.py", "-q", "--no-header", "-x"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return result.returncode, (result.stdout or "")[-400:]


def main() -> None:
    """Apply each mutation in turn and report whether a test caught it."""
    original = TARGET.read_text(encoding="utf-8")
    backup = Path(tempfile.gettempdir()) / "worker_backup.py"
    shutil.copy(TARGET, backup)

    code, _ = run_tests()
    print(f"baseline (no mutation): {'PASS' if code == 0 else 'FAIL'}")
    print()

    survivors = []
    for label, old, new in MUTATIONS:
        if old not in original:
            print(f"  SKIP  {label}: pattern not found")
            survivors.append(f"{label} (pattern missing)")
            continue
        TARGET.write_text(original.replace(old, new, 1), encoding="utf-8")
        code, tail = run_tests()
        caught = code != 0
        print(f"  {'CAUGHT' if caught else 'SURVIVED'}  {label}")
        if not caught:
            survivors.append(label)
        shutil.copy(backup, TARGET)

    print()
    if survivors:
        print(f"{len(survivors)} mutation(s) not caught by any test:")
        for item in survivors:
            print(f"  - {item}")
    else:
        print("every mutation was caught")


if __name__ == "__main__":
    sys.exit(main())