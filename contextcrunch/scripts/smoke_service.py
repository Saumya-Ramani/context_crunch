"""Smoke-test a running ContextCrunch service. Not part of the package.

Usage:
    .venv\\Scripts\\python.exe scripts\\smoke_service.py [base_url]
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"

CODE = (
    "src/billing/pricing.py\n"
    "def round_price(total):\n"
    "    return int(total) + 1\n\n"
    "class Invoice:\n"
    "    def apply_discount(self, rate):\n"
    "        self.total = round_price(self.total * (1 - rate))\n"
    "    # ... 180 more lines of pricing helpers ...\n"
) * 30
INSTALL_LOG = "Collecting pytest\nDownloading pluggy-1.5.0\nSuccessfully installed pytest-8.2.1\n" * 90

MESSAGES = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Fix the rounding bug in the billing service."},
    {"role": "assistant", "content": "Plan: read the file, patch it, run the tests."},
    {"role": "tool", "name": "read_file", "tool_call_id": "c1", "content": CODE},
    {"role": "assistant", "content": "round_price truncates instead of rounding."},
    {"role": "tool", "name": "run_in_terminal", "tool_call_id": "c2", "content": INSTALL_LOG},
    {"role": "assistant", "content": "Patched round_price to use Decimal."},
    {"role": "tool", "name": "run_tests", "tool_call_id": "c3", "content": "147 passed in 4.05s\n"},
    {"role": "assistant", "content": "All 147 tests pass."},
    {"role": "assistant", "content": "Now update the docs."},
    {"role": "assistant", "content": "Docs updated."},
    {"role": "assistant", "content": "Anything else?"},
]


def get(path: str) -> dict:
    """GET JSON from the service."""
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as response:
        return json.loads(response.read())


def post(path: str, payload: dict) -> dict:
    """POST JSON and return the parsed response."""
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def main() -> None:
    """Exercise health, compaction, stats and the two failure modes."""
    print("health      :", json.dumps(get("/health")))

    body = post(
        "/v1/compact",
        {"messages": MESSAGES, "session_id": "smoke", "profile": "aggressive"},
    )
    print(
        f"compact     : {body['tokens_before']} -> {body['tokens_after']} "
        f"({body['reduction_pct']}%) profile={body['profile_used']} "
        f"fail_open={body['fail_open']}"
    )
    print("counts      :", f"kept={body['kept']} shortened={body['shortened']} removed={body['removed']}")

    sent, original = body["messages"], MESSAGES
    roles_kept = [m["role"] for m in sent] == [m["role"] for m in original]
    ids = [m.get("tool_call_id") for m in sent if m.get("tool_call_id")]
    print("messages    :", len(sent), "of", len(original))
    print("roles kept  :", roles_kept)
    print("ids kept    :", ids == ["c1", "c2", "c3"])
    print("token order :", body["tokens_after"] <= body["tokens_before"])

    print("stats       :", json.dumps(get("/v1/sessions/smoke/stats")))
    print("unknown     :", json.dumps(get("/v1/sessions/nobody/stats")))

    try:
        post("/v1/compact", {"messages": []})
        print("empty       : NOT REJECTED (expected 422)")
    except urllib.error.HTTPError as exc:
        print("empty       : rejected with", exc.code)

    # A second call in the same session must still work after the first.
    again = post(
        "/v1/compact",
        {"messages": MESSAGES, "session_id": "smoke", "profile": "aggressive"},
    )
    print(
        f"second call : {again['tokens_before']} -> {again['tokens_after']} "
        f"fail_open={again['fail_open']}"
    )


if __name__ == "__main__":
    main()