"""Check the README's host example against a real running service.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\check_readme_example.py [base_url]
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"

BIG_DUMP = ("def round_price(total):\n    return int(total) + 1\n" * 200)
INSTALL_LOG = ("Collecting pytest\nSuccessfully installed pytest-8.2.1\n" * 100)


def main() -> int:
    """Start the service, run the README's example against it, stop the service."""
    repo = Path(__file__).resolve().parent.parent
    python = str(repo / ".venv" / "Scripts" / "python.exe")

    server = subprocess.Popen(
        [
            python, "-m", "uvicorn", "contextcrunch.api.main:create_app",
            "--factory", "--port", "8123", "--log-level", "warning",
        ],
        cwd=repo,
    )
    try:
        # Wait for the port rather than sleeping a fixed amount.
        for _ in range(60):
            try:
                if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(0.25)
        else:
            print("service did not come up")
            return 1

        # --- the README example, verbatim in shape ------------------------- #
        history = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Fix the rounding bug in the billing service."},
            {"role": "tool", "name": "read_file", "tool_call_id": "c1", "content": BIG_DUMP},
            {"role": "assistant", "content": "round_price truncates instead of rounding."},
            {"role": "tool", "name": "run_terminal", "tool_call_id": "c2", "content": INSTALL_LOG},
            {"role": "assistant", "content": "Patched it to use Decimal ROUND_HALF_UP."},
        ]

        response = httpx.post(
            f"{BASE}/v1/compact",
            json={
                "messages": history,
                "session_id": "ticket-4471",
                "profile": None,
                "goal": "Fix the rounding bug in the billing service.",
            },
            timeout=60,
        )
        body = response.json()

        if body["fail_open"]:
            messages = history
            print("fail_open, using what we sent")
        else:
            messages = body["messages"]

        print(
            f"{body['tokens_before']} -> {body['tokens_after']} tokens "
            f"({body['reduction_pct']}% saved) using the {body['profile_used']} profile"
        )
        print(f"messages returned: {len(messages)} of {len(history)}")

        ids = [m.get("tool_call_id") for m in messages if m.get("tool_call_id")]
        print(f"tool_call_ids kept: {ids == ['c1', 'c2']}  ({ids})")

        stats = httpx.get(f"{BASE}/v1/sessions/ticket-4471/stats", timeout=10).json()
        print(f"session stats: {stats}")

        ok = response.status_code == 200 and len(messages) == len(history)
        print("\nREADME example works" if ok else "\nREADME example FAILED")
        return 0 if ok else 1
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())