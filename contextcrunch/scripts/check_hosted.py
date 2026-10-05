"""Smoke-check the hosted Laya endpoint with the real key.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\check_hosted.py
"""

from __future__ import annotations

import asyncio
import time

from contextcrunch.core.messages import as_messages
from contextcrunch.core.settings import get_settings
from contextcrunch.core.state import build_state
from contextcrunch.laya.hosted import HostedBackend


async def main() -> None:
    settings = get_settings()
    print("mode      :", settings.laya_mode)
    print("url       :", settings.laya_api_url)
    print("key       :", "present" if settings.laya_api_key else "MISSING")

    messages = as_messages(
        [
            {"role": "user", "content": "Fix the rounding bug in the billing service."},
            {"role": "assistant", "content": "Plan: read the file, then run the tests."},
            {
                "role": "tool",
                "name": "read_file",
                "tool_call_id": "call_1",
                "content": "def round_price(total): return int(total) + 1\n" * 40,
            },
            {"role": "assistant", "content": "Found it: round_price truncates instead of rounding."},
        ]
    )
    state = build_state(
        messages, 2, "Fix the rounding bug in the billing service.", get_settings()
    )

    backend = HostedBackend()
    started = time.time()
    try:
        result = await backend.predict(state, "smoke")
    except Exception as exc:
        print(f"FAILED after {time.time() - started:.2f}s -> {type(exc).__name__}: {exc}")
        return
    finally:
        await backend.aclose()

    print(f"OK in {time.time() - started:.2f}s")
    for question_id, answer in result.answers.items():
        print(f"  {question_id:<12} {answer.type:<7} {answer.label()}")


if __name__ == "__main__":
    asyncio.run(main())