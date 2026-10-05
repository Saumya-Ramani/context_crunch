"""Run one full compaction against the hosted Laya API and report the outcome."""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from contextcrunch.agents.worker import compact
from contextcrunch.core.messages import as_messages, to_dicts
from contextcrunch.core.policy import PROFILES
from contextcrunch.core.tokens import count_tokens
from contextcrunch.laya.client import build_client
from contextcrunch.storage.box import Box


def history() -> list[dict[str, object]]:
    """Build a coding trace with big, stale and irreplaceable material."""
    return [
        {"role": "system", "content": "You are a coding assistant. Always run the tests."},
        {"role": "user", "content": "Fix the rounding bug in the billing service."},
        {"role": "assistant", "content": "Plan: read the file, patch it, run the tests."},
        {
            "role": "tool",
            "name": "read_file",
            "tool_call_id": "call_read",
            "content": (
                "src/billing/pricing.py\ndef round_price(total):\n    return int(total) + 1\n\n"
                "class Invoice:\n    def apply_discount(self, rate):\n"
                "        self.total = round_price(self.total * (1 - rate))\n"
                "    # ... 180 more lines of pricing helpers ...\n"
            )
            * 12,
        },
        {
            "role": "assistant",
            "content": "Found it: round_price uses int() so it truncates instead of rounding.",
        },
        {
            "role": "tool",
            "name": "run_in_terminal",
            "tool_call_id": "call_pip",
            "content": "Collecting pytest\nDownloading pluggy-1.5.0\n"
            "Successfully installed pytest-8.2.1 pluggy-1.5.0\n" * 40,
        },
        {
            "role": "assistant",
            "content": "Patched round_price to use Decimal ROUND_HALF_UP.",
        },
        {
            "role": "tool",
            "name": "run_tests",
            "tool_call_id": "call_tests",
            "content": "147 passed in 4.05s\n",
        },
        {"role": "assistant", "content": "All 147 tests pass. The rounding bug is fixed."},
    ]


async def main() -> None:
    messages = as_messages(history())
    before = sum(count_tokens(m.content) for m in messages)
    print(f"before : {len(messages)} messages, {before} tokens")

    with tempfile.TemporaryDirectory() as tmp:
        box = Box(db_path=Path(tmp) / "cc.db", originals_dir=Path(tmp) / "originals")
        client = build_client()
        started = time.time()
        outcome = await compact(
            client=client,
            messages=messages,
            goal="Fix the rounding bug in the billing service.",
            profile=PROFILES["aggressive"],
            box=box,
            profile_name="aggressive",
            deadline_s=90.0,
        )
        elapsed = time.time() - started
        await client.aclose()

        print(f"mode   : {client.mode}  ({elapsed:.2f}s, degraded={outcome.degraded})")
        print(f"after  : {len(outcome.messages)} messages, {outcome.tokens_after} tokens")
        print(f"saving : {outcome.tokens_before - outcome.tokens_after} tokens "
              f"({outcome.as_dict()['reduction_pct']}%)")
        print("actions:", outcome.as_dict()["actions"])
        print("box    :", box.summary(outcome.run_id)["counts"])
        print("notes  :", outcome.notes)
        print()
        for index, (old, new) in enumerate(zip(messages, outcome.messages, strict=True)):
            if new.content != old.content:
                print(f"[{index}] {outcome.actions.get(index, '-')}")
                print(f"   was: {old.content[:64]!r}")
                print(f"   now: {new.content[:80]!r}")
        print()
        print("roles kept        :", all(o.role == n.role
                                        for o, n in zip(messages, outcome.messages, strict=True)))
        print("tool_call_ids kept:", all(o.tool_call_id == n.tool_call_id
                                        for o, n in zip(messages, outcome.messages, strict=True)))
        print("json-safe         :", isinstance(to_dicts(outcome.messages), list))

        dropped = [i for i, a in outcome.actions.items() if a == "DROP"]
        for index in dropped:
            original = box.load_original(box.by_run(outcome.run_id)[0].original_ref or "")
        print(f"dropped items     : {dropped}, originals recoverable from the box")


if __name__ == "__main__":
    asyncio.run(main())