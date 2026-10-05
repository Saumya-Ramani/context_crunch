# ContextCrunch

ContextCrunch sits in front of an existing LLM gateway and shrinks the message history it
sends, using a small decision model to work out what is still worth keeping. It never
rewrites text: what it removes is replaced by a short tombstone that keeps the message's
role and `tool_call_id`, and every original is stored verbatim so any removal can be undone.

It is a Python library and an HTTP service. There is no model to train and nothing to host.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r req.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
Copy-Item .env.example .env      # then put your Laya key in it
```

The editable install matters. Without it `import contextcrunch` fails outside pytest, because
pytest only adds `src` to the path for its own test run.

Configuration is read from the environment with the `CC_` prefix. The one value with no
default is the Laya API key, read as `Laya_Api_key` in `.env`:

```
Laya_Api_key=your-key-here
CC_LAYA_MODE=http
```

If configuration is incomplete the service warns on stderr and falls back to safe defaults
rather than refusing to start.

## The three commands

```powershell
# 1. Tests. Unit tests plus mutation checks on the two safety guarantees.
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\mutation_check.py         # Worker safety rules
.\.venv\Scripts\python.exe scripts\mutation_api_check.py    # API fail-open rules

# 2. Evaluation. Synthetic traces with known answers; the headline number is
#    "dangerous drops", the facts that were lost.
.\.venv\Scripts\python.exe scripts\make_traces.py --mark
.\.venv\Scripts\python.exe scripts\evaluate.py --fake

# 3. Demo. 60 lines, offline, no LLM needed. This is the one to show people.
.\.venv\Scripts\python.exe scripts\demo.py --fake
```

Drop `--fake` from the evaluation to run against the real model, which is roughly 15 seconds
per trace.

## Running the service

```powershell
.\.venv\Scripts\python.exe -m uvicorn contextcrunch.api.main:create_app --factory --port 8080
```

Three endpoints: `GET /health`, `POST /v1/compact`, and `GET /v1/sessions/{id}/stats`.

**Fail-open is mandatory.** No endpoint returns an error for a compaction problem. A model
timeout, a dead backend, an internal bug — all of them return your original messages with
`fail_open: true` and `tokens_before == tokens_after`. Only malformed *input* returns 422,
because that is a bug in the caller and hiding it would hide the bug.

## How a host app calls it

The host keeps its own full history, so a removal costs the host nothing. Send the history,
send back what comes out.

```python
import httpx

BASE = "http://127.0.0.1:8080"

# The history your agent has accumulated so far.
history = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Fix the rounding bug in the billing service."},
    {"role": "tool", "name": "read_file", "tool_call_id": "c1", "content": big_file_dump},
    {"role": "assistant", "content": "round_price truncates instead of rounding."},
    {"role": "tool", "name": "run_terminal", "tool_call_id": "c2", "content": install_log},
    {"role": "assistant", "content": "Patched it to use Decimal ROUND_HALF_UP."},
]

response = httpx.post(
    f"{BASE}/v1/compact",
    json={
        "messages": history,
        "session_id": "ticket-4471",   # lets the Guardian learn across calls
        "profile": None,               # or "aggressive" / "balanced" / "conservative"
        "goal": "Fix the rounding bug in the billing service.",
    },
    timeout=60,
)
body = response.json()

if body["fail_open"]:
    # Compaction did not run or did not help. Use what you sent.
    messages = history
else:
    messages = body["messages"]

print(f"{body['tokens_before']} -> {body['tokens_after']} tokens "
      f"({body['reduction_pct']}% saved) using the {body['profile_used']} profile")

# Send `messages` to your LLM. You still hold `history` in full, so you can
# always fall back to it.
completion = your_llm_client.chat(model="your-model", messages=messages)
```

Send the same `session_id` every time. The Guardian uses it to notice when you ask again for
something it removed, to log that as a mistake, and to switch the session to a safer profile
once it happens twice. Omit it and compaction still works, it just cannot learn.

## What is where

| Path | What it is |
|---|---|
| `src/contextcrunch/agents/scout.py` | Picks the profile and the goal from the tool names present |
| `src/contextcrunch/agents/worker.py` | The Compactor: PLAN, ACT, VERIFY |
| `src/contextcrunch/agents/guardian.py` | Notices repeated removals and stuck loops |
| `src/contextcrunch/core/policy.py` | The thresholds, and nothing else |
| `src/contextcrunch/core/splitter.py` | Splits an oversized item into judgeable pieces |
| `src/contextcrunch/storage/store.py` | The side-car: originals, decisions, mistakes |
| `src/contextcrunch/laya/` | The only place that talks to a model |
| `data/traces/` | Synthetic traces and their answer keys |

`progress.md` is the engineering log: what was measured, what was found, what is still
unresolved.

`md/OPERATIONS.md` is the full operating guide: every command and its output, the request path
step by step, where all the dummy and test data lives, every setting, and the complete API
reference.