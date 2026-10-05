"""Smoke-test a running ContextCrunch API server. Not part of the package."""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"

CODE = (
    "src/billing/pricing.py\n"
    "def round_price(total):\n"
    "    return int(total) + 1\n\n"
    "class Invoice:\n"
    "    def apply_discount(self, rate):\n"
    "        self.total = round_price(self.total * (1 - rate))\n"
    "    def apply_tax(self, rate):\n"
    "        self.total = round_price(self.total * (1 + rate))\n"
    "    # ... 180 more lines of pricing helpers ...\n"
) * 30
NOISE = "Collecting pytest\nDownloading pluggy-1.5.0\nSuccessfully installed pytest-8.2.1\n" * 90

MESSAGES = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Fix the rounding bug in the billing service."},
    {"role": "assistant", "content": "Plan: read the file, patch it, run the tests."},
    {"role": "tool", "name": "read_file", "tool_call_id": "c1", "content": CODE},
    {"role": "assistant", "content": "round_price truncates instead of rounding. Patching it."},
    {"role": "tool", "name": "run_in_terminal", "tool_call_id": "c2", "content": NOISE},
    {"role": "assistant", "content": "Patched round_price to use Decimal ROUND_HALF_UP."},
    {"role": "tool", "name": "run_tests", "tool_call_id": "c3", "content": "147 passed in 4.05s\n"},
    {"role": "assistant", "content": "All 147 tests pass. The rounding bug is fixed."},
]


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
    """Exercise health, a real compaction, and a bad request."""
    with urllib.request.urlopen(f"{BASE}/health", timeout=30) as response:
        health = json.loads(response.read())
    print("health      :", json.dumps(health))

    result = post(
        "/v1/compact",
        {
            "messages": MESSAGES,
            "profile": "aggressive",
            "goal": "Fix the rounding bug in the billing service.",
        },
    )
    print(
        f"compact     : {result['tokens_before']} -> {result['tokens_after']} "
        f"({result['reduction_pct']}%) degraded={result['degraded']} skipped={result['skipped']}"
    )
    print("actions     :", json.dumps(result["actions"]))
    print("guardian    :", json.dumps(result["guardian"]))
    print("notes       :", result["notes"])

    roles = [m["role"] for m in result["messages"]]
    ids = [m.get("tool_call_id") for m in result["messages"] if m.get("tool_call_id")]
    print("messages    :", len(result["messages"]), roles)
    print("tool ids    :", ids)
    print("roles kept  :", roles == [m["role"] for m in MESSAGES])
    print("ids kept    :", ids == ["c1", "c2", "c3"])
    print("token order :", result["tokens_after"] <= result["tokens_before"])

    # An empty history must be rejected cleanly, not crash the service.
    try:
        post("/v1/compact", {"messages": []})
        print("empty       : NOT REJECTED (expected 422)")
    except urllib.error.HTTPError as exc:
        print("empty       : rejected with", exc.code)

    # A second call must still work, so nothing was left locked or half-written.
    again = post("/v1/compact", {"messages": MESSAGES, "profile": "conservative"})
    print("second call :", f"{again['tokens_before']} -> {again['tokens_after']}",
          "degraded=", again["degraded"])


if __name__ == "__main__":
    main()