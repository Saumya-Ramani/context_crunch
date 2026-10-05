# ContextCrunch — Operating Guide

Everything needed to run, inspect and understand this project. Written from the code as it
stands on 2026-10-05; where a number is quoted it was measured, not estimated.

- **Tests:** 370 passing. `ruff check .` clean.
- **Source:** ~3,200 lines across 6 packages.
- **Demo:** runs offline in about a second, prints 60 lines.

---

## 1. What this thing does

An agent accumulates a long message history: tool outputs, file dumps, install logs, search
results. Every LLM call pays for all of it again. ContextCrunch sits in front of the LLM call,
works out which parts are still worth sending, and returns a shorter history.

Three properties make it safe to put in a live path:

1. **It never rewrites text.** What it removes becomes a short tombstone that keeps the
   message's `role`, `name` and `tool_call_id`. What it keeps is byte-for-byte original.
2. **Nothing is destroyed.** Every original is written to a side-car before it is dropped, so
   any removal can be undone.
3. **Fail-open is mandatory.** If anything goes wrong — model timeout, dead backend, internal
   bug — the caller gets its own history back untouched and the LLM call proceeds.

---

## 2. Setup

```powershell
cd contextcrunch
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r req.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
Copy-Item .env.example .env
```

Then put your key in `.env`:

```
Laya_Api_key=your-key-here
```

Two steps that are easy to miss and cause confusing failures:

- **`pip install -e .` is required.** Without it `import contextcrunch` fails everywhere except
  inside pytest, because pytest adds `src/` to `sys.path` for its own use only. Symptoms:
  `ModuleNotFoundError: No module named 'contextcrunch'` from any script or from uvicorn.
- **`CC_LAYA_API_KEY` has no default anywhere in the code.** That is deliberate — a missing key
  leaves the hosted backend failing open rather than sending a wrong credential. Every other
  setting has an emergency fallback in `core/settings.py::EMERGENCY_DEFAULTS`.

`.env.example` documents every variable with its meaning. The full list also appears in
section 6.

---

## 3. Commands, and what each one prints

### 3.1 Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: `370 passed`. Takes about 12 seconds. No network and no model — every test runs
against a fake backend.

### 3.2 Mutation checks — the ones that matter most

```powershell
.\.venv\Scripts\python.exe scripts\mutation_check.py        # Worker safety rules
.\.venv\Scripts\python.exe scripts\mutation_api_check.py   # API fail-open rules
```

These remove one safety behaviour at a time from the source and assert that a test notices.
A passing test proves nothing on its own; these prove the tests are real.

Expected: `every mutation was caught` — 9 mutations in the Worker, 8 in the API.

### 3.3 Demo — the one to show people

```powershell
.\.venv\Scripts\python.exe scripts\demo.py --fake
```

Runs offline, no LLM, deterministic, 60 lines. Prints, in order:

| Section | What it shows |
|---|---|
| Banner | Engine mode and which trace is being used |
| `SCOUT` | Detected agent type, chosen profile, tools seen, pinned tools, keyword scores |
| `WORKER` | One row per message: index, role, tool, tokens before → after, a coloured action label, and a block-character bar showing the size change |
| `TOTAL` | Tokens before → after, % saved, and the saving extrapolated to a 100-step run |
| `PROOF` | Each planted fact marked `survived` (green) or `LOST` (red) |
| `GUARDIAN` | Mistake logged → session switched to conservative → top 3 items restored |
| Closing | That originals are stored and nothing is lost |

Action label colours: `KEEP` green, `SHORTENED` yellow, `REMOVED` red, `PINNED` blue.

Options: pass a trace path to use a specific one, e.g.
`.\.venv\Scripts\python.exe scripts\demo.py data/traces/coding_03.json --fake`. Drop `--fake`
to use the real model from `.env`.

Note: the demo picks a trace that actually removes something. The default research trace
resolves to the `conservative` profile, which correctly refuses to drop anything — a good
safety property but a poor demo, so it falls back to a coding trace and says so.

### 3.4 Traces and evaluation

```powershell
.\.venv\Scripts\python.exe scripts\make_traces.py --mark     # generate 20 traces
.\.venv\Scripts\python.exe scripts\evaluate.py --fake          # score them
```

`make_traces.py` writes 20 deterministic traces (8 coding, 6 support, 6 research) to
`data/traces/`, each with an answer key. `--mark` inserts the markers the offline fake model
reads (see section 5).

`evaluate.py` prints a per-trace table, a per-archetype summary and totals, and writes
`data/results.csv`. Useful flags: `--archetype coding`, `--verbose` (prints each lost fact),
`--out path.csv`.

The number that matters is **dangerous drops** — planted facts that did not survive. It is
printed in red and counted separately in the totals, never averaged into a percentage. A
compaction that saves 60% and loses one customer note is worse than one that saves nothing,
because a host cannot see that loss in the token count.

Last measured with `--fake`: **0 dangerous drops, 54/54 facts survived.**

### 3.5 The service

```powershell
.\.venv\Scripts\python.exe -m uvicorn contextcrunch.api.main:create_app --factory --port 8080
```

Then, from another shell:

```powershell
# health
Invoke-RestMethod http://127.0.0.1:8080/health

# compact
$r = Invoke-RestMethod http://127.0.0.1:8080/v1/compact -Method Post `
     -ContentType 'application/json' `
     -Body (@{ messages = $history; session_id = 'demo'; goal = 'fix the bug' } | ConvertTo-Json -Depth 6)
$r | ConvertTo-Json -Depth 4

# what has this session done so far
Invoke-RestMethod http://127.0.0.1:8080/v1/sessions/demo/stats
```

### 3.6 Diagnostic scripts

| Script | What it does |
|---|---|
| `scripts/check_hosted.py` | Calls the hosted Laya endpoint once and prints the five answers |
| `scripts/check_pipeline.py` | Runs one full compaction against the real model, end to end |
| `scripts/smoke_service.py` | Exercises all three endpoints of a running service |
| `scripts/smoke_api.py` | Older smoke test against a running service |
| `scripts/check_readme_example.py` | Boots the service and runs the README host example against it |
| `scripts/tune_profiles.py` | Scores profiles against the labelled set; `--search` proposes thresholds |
| `scripts/evalset.py` | The 15 hand-written labelled cases (a library, not a runner) |
| `scripts/probe_laya.py` | One-off exploration of the raw Laya model. Not part of the package |

Example — confirm the real model is reachable:

```powershell
.\.venv\Scripts\python.exe scripts\check_hosted.py
```

```
mode      : http
url       : https://api.impossibl.com/v1/systemone
key       : present
OK in 2.25s
  verdict      choice  keep
  essential    noul    noul=0.55
  ...
```

---

## 4. What runs, in order

This is the path a single `/v1/compact` request takes.

```
POST /v1/compact
  │
  ├─ 1. as_messages()          parse the caller's dicts into Message objects
  │                            (must be first — see section 8)
  │
  ├─ 2. ScoutAgent.run()       pick the profile from the tool names present,
  │                            extract the goal, find side-effecting tools to pin
  │
  ├─ 3. GuardianAgent.observe() look for repeated removals and stuck loops;
  │                            returns Overrides(profile, restore)
  │
  ├─ 4. CompactorAgent.run()   ── wrapped in asyncio.wait_for(settings.deadline_s)
  │      │
  │      ├─ PLAN   count tokens; below CC_TRIGGER_TOKENS → return untouched.
  │      │         Only `tool` messages are eligible. Three things force a KEEP
  │      │         with no model call: listed in overrides.restore, tool is pinned,
  │      │         or within profile.K turns of the end.
  │      │
  │      ├─ ACT    items ≤ CC_ITEM_MAX_CHARS are all judged in ONE parallel batch.
  │      │         Larger items skip the whole-item judgement and go straight to
  │      │         split-first: split into pieces, judge each piece with the same
  │      │         five questions, join whatever survives.
  │      │           DROP     → tombstone_message()
  │      │           KEEP     → unchanged
  │      │           TRUNCATE → rebuild from surviving pieces
  │      │         Every reviewed item is written to the Store with its ORIGINAL text.
  │      │
  │      └─ VERIFY same message count, same roles in order, every tool_call_id
  │                attached, no empty content. Any failure → return the ORIGINALS.
  │
  └─ 5. Build the response     200 with the compacted history, or the originals
                               with fail_open=true. Never an error for a
                               compaction problem.
```

### The five questions asked about every item

Defined once in `core/questions.py`, used everywhere:

| id | type | asks |
|---|---|---|
| `verdict` | choice | keep / truncate / drop |
| `essential` | noul | is it essential |
| `consumed` | noul | has its content already been extracted |
| `superseded` | noul | has a later result made it obsolete |
| `relevance` | score | how relevant, 0–3 |

`consumed` is only answerable because `core/state.py` includes `later_findings` — assistant
messages *after* the item, nearest first. That is what lets the model see that a 14,000-token
fetch was already distilled into a short later message.

### What each decision turns into

| Action | Result |
|---|---|
| `KEEP` | original text, byte for byte |
| `DROP` | `[Removed by ContextCrunch: <tool>. judged no longer relevant. Recoverable via sidecar.]` |
| `TRUNCATE` | surviving pieces rejoined with `\n---\n` |

A tombstone keeps `role`, `name` and `tool_call_id` because an assistant `tool_calls` entry
points at a `tool_call_id`; if the matching result vanished or changed role, the provider
rejects the whole request.

---

## 5. Dummy and test data — where everything lives

### Synthetic traces (the evaluation set)

Generated by `scripts/make_traces.py` into **`data/traces/`**. 20 traces, 8 coding / 6 support
/ 6 research. Each is two files:

- `data/traces/<name>.json` — the message history
- `data/traces/<name>.key.json` — the answer key: `must_survive`, `expected_gone`,
  `planted_kinds`, `tool_messages`

Tool outputs are 3,000–20,000 characters, built from log lines, code dumps, JSON arrays and
boilerplate paragraphs. Generation is seeded, so the same seed gives byte-identical traces.

Five planted cases per trace:

| Case | What it is | Expected |
|---|---|---|
| `used_up` | Large output whose one useful line a later assistant message repeats | Droppable |
| `trap` | Large output holding a fact nothing else mentions | **Must survive** |
| `superseded` | A failing check followed by a passing one | First droppable |
| `noise` | Install logs and boilerplate | Droppable |
| `risky` | `refund_issued` or `send_email` | **Must survive** |

### Markers

`FakeBackend` answers from markers, which is what allows evaluation with no LLM:

- `[[KEEPME]]` — content that must be kept
- `[[NOISE]]` — content that may go

Unmarked content falls back to a stable hash, so it still gets a reproducible verdict.

**Markers must be per paragraph, not per item.** An item over `CC_ITEM_MAX_CHARS` is split
into pieces and each piece is judged separately, so a single marker at the top of a
16,000-character output marks only piece 1 — the rest fall back to the hash and some get
dropped, destroying a planted fact. This cost a full debugging session; `make_traces.py`
marks every paragraph for exactly this reason.

### Test fixtures

**`tests/conftest.py`** holds the shared fixtures:

| Name | What it is |
|---|---|
| `BASE_ENV` | Every `CC_` variable, with test-safe values. The autouse `clean_env` fixture applies these and points the databases at `tmp_path`, so a `.env` in the working tree cannot change a result |
| `settings_factory` | Builds a real `Settings` with chosen overrides |
| `answers(...)` | Builds a scripted `LayaResult`; each argument is one policy signal |
| `ScriptedBackend` | Returns pre-canned answers one call at a time |
| `CODE_OUTPUT` | Small realistic code fixture |
| `REPORT_OUTPUT` | Long structured report, deliberately larger than `CC_PIECE_TARGET_CHARS` |
| `history(...)` | A small but realistic history: goal, plan, tool output, finding |

Per-file fakes: `tests/test_worker.py` defines its own `FakeLaya` (records what it was asked
and how many calls overlapped) plus `NOISE` / `KEEPME` markers; `tests/test_engine.py` defines
`CountingBackend` to prove concurrency; `tests/test_api.py` defines its own `FakeLaya`,
`Exploding` (raises) and a `Broken` store.

### The old labelled set

`scripts/evalset.py` — 15 hand-written cases, 5 per archetype, used by
`scripts/tune_profiles.py`. Answers cache in `data/tune_cache.json` so re-running is free.
This set is too small to tune on; the 20 synthetic traces replaced it for measurement.

### Runtime data

| Path | What |
|---|---|
| `data/contextcrunch.db` | Per-run store (`Box`) |
| `data/originals/` | Originals as files, for the per-run store |
| `data/contextcrunch_store.db` | Session-scoped store (`Store`) |
| `data/store_originals/<sha1>/<index>.txt` | Originals for the session store |
| `data/results.csv` | Evaluation output |

Originals are stored under a **hash** of the session id, never the raw id. Session ids come
from a calling agent, so `../../etc` must not be able to write outside the directory. The two
stores use separate database files on purpose: both create a `decisions` table with different
columns, and SQLite's `CREATE TABLE IF NOT EXISTS` means whichever opens the file first would
define the schema for both.

---

## 6. Configuration

Every variable, read from the environment with the `CC_` prefix.

| Variable | Default | Meaning |
|---|---|---|
| `CC_LAYA_MODE` | `fake` | `http` (hosted, default in `.env`), `serve`, `inprocess`, `fake` |
| `CC_LAYA_MODEL` | `convaiinnovations/laya` | Model id |
| `CC_LAYA_API_URL` | `https://api.impossibl.com/v1/systemone` | Hosted endpoint |
| `Laya_Api_key` | *(none)* | Bearer token. No default anywhere, on purpose |
| `CC_LAYA_MAX_LEN` | `8192` | Token budget for one call |
| `CC_LAYA_CONCURRENCY` | `8` | Model calls allowed in flight at once |
| `CC_LAYA_RETRIES` | `2` | Retries for a transient hosted failure |
| `CC_LAYA_BACKOFF_S` | `0.5` | Base backoff delay |
| `CC_LAYA_HTTP_BASE_URL` | `http://127.0.0.1:8001` | Used when mode is `serve` |
| `CC_TIKTOKEN_ENCODING` | `cl100k_base` | Tokenizer |
| `CC_ITEM_MAX_CHARS` | `12000` | Above this an item is split, not sent whole |
| `CC_FINDINGS_CAP_CHARS` | `4000` | Cap on `later_findings` |
| `CC_GOAL_MAX_CHARS` | `1000` | Cap on the goal |
| `CC_STATE_MAX_TOKENS` | `6000` | Cap on one Laya state |
| `CC_PIECE_TARGET_CHARS` | `2500` | Target piece size |
| `CC_PIECE_MAX_CHARS` | `3500` | Hard maximum piece size |
| `CC_TRIGGER_TOKENS` | `2000` | Below this a history is returned untouched |
| `CC_DEADLINE_S` | `30` | Whole-request deadline before fail-open |
| `CC_DEFAULT_PROFILE` | `conservative` | Used when Scout cannot tell |
| `CC_DB_PATH` | `data/contextcrunch.db` | Per-run store |
| `CC_ORIGINALS_DIR` | `data/originals` | Per-run originals |
| `CC_STORE_DB_PATH` | `data/contextcrunch_store.db` | Session store |
| `CC_STORE_ORIGINALS_DIR` | `data/store_originals` | Session originals |

### Profiles

Three sets of thresholds, and they live in `core/policy.py` and nowhere else.

| Profile | Used for | Character |
|---|---|---|
| `aggressive` | coding | Most aggressive; keeps the last 2 turns, no anchors |
| `balanced` | support | Middle |
| `conservative` | research | Strictest; keeps the last 3 turns, adds source anchors |

Scout maps archetype → profile by counting keyword hits in the tool names
(`read_file`, `kb_`, `search`, …). A tie, or no signal at all, falls back to
`CC_DEFAULT_PROFILE`: guessing wrong in the aggressive direction is the one mistake that can
lose information.

---

## 7. The API

Base URL `http://127.0.0.1:8080`. Three endpoints.

### `GET /health`

```json
{ "status": "ok", "laya_mode": "http" }
```

### `POST /v1/compact`

Request:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `messages` | list of dicts | **yes** | The full history. `min_length=1` |
| `session_id` | string | no | Ties calls together so the Guardian can learn. Omit and it still compacts, it just cannot learn |
| `profile` | string or null | no | `aggressive`, `balanced`, `conservative`. Null lets Scout decide |
| `goal` | string or null | no | Overrides the goal Scout extracts |

Response:

| Field | Type | Meaning |
|---|---|---|
| `messages` | list of dicts | **Always usable.** The originals on any failure |
| `session_id` | string | Echoed back |
| `profile_used` | string | Profile applied, or `"none"` on fail-open |
| `goal` | string | The goal used |
| `tokens_before` | int | Tokens in the input |
| `tokens_after` | int | Tokens in the output |
| `reduction_pct` | float | Percentage saved |
| `fail_open` | bool | True when nothing was changed by a failure |
| `kept` / `shortened` / `removed` | int | Item counts by action |
| `actions` | dict | `"3": "DROP"`, index → action |
| `restored` | list of int | Indexes the Guardian asked back for |
| `notes` | list of string | Human-readable notes |

Python example:

```python
import httpx

BASE = "http://127.0.0.1:8080"

history = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Fix the rounding bug in the billing service."},
    {"role": "tool", "name": "read_file", "tool_call_id": "c1", "content": big_dump},
    {"role": "assistant", "content": "round_price truncates instead of rounding."},
    {"role": "tool", "name": "run_terminal", "tool_call_id": "c2", "content": install_log},
    {"role": "assistant", "content": "Patched it to use Decimal ROUND_HALF_UP."},
]

body = httpx.post(
    f"{BASE}/v1/compact",
    json={
        "messages": history,
        "session_id": "ticket-4471",   # lets the Guardian learn across calls
        "profile": None,               # or "aggressive" / "balanced" / "conservative"
        "goal": "Fix the rounding bug in the billing service.",
    },
    timeout=60,
).json()

messages = history if body["fail_open"] else body["messages"]
print(f"{body['tokens_before']} -> {body['tokens_after']} "
      f"({body['reduction_pct']}% saved), profile {body['profile_used']}")

completion = your_llm_client.chat(model="your-model", messages=messages)
```

The host keeps its own full history throughout, so a removal costs the host nothing — it can
always send the full thing again. That is what makes it safe to be aggressive.

### `GET /v1/sessions/{session_id}/stats`

```json
{ "session_id": "ticket-4471", "decisions": { "KEEP": 1, "TRUNCATE": 2 }, "mistakes": 0 }
```

### Fail-open contract

| Situation | Response |
|---|---|
| Model times out | 200, originals, `fail_open: true` |
| Model backend is down | 200, originals, `fail_open: true` |
| Any internal error | 200, originals, `fail_open: true` |
| `messages` is empty or missing | **422** |
| `messages` is not a list | **422** |

Only malformed *input* returns an error, because that is a bug in the caller and hiding it
would hide the bug. On fail-open, `tokens_before == tokens_after` so the host can see that its
history came back whole. The traceback goes to the log, never to the caller.

---

## 8. Source map

```
src/contextcrunch/
├── agents/
│   ├── scout.py      317  Pick the profile, extract the goal, find pinned tools
│   ├── worker.py     569  CompactorAgent: PLAN / ACT / VERIFY, plus Stats
│   └── guardian.py   269  GuardianAgent: regret and stuck rules
├── api/
│   ├── main.py       269  The service. create_app(), /health, /v1/compact, stats
│   └── app.py        133  SUPERSEDED by main.py. Only test_pipeline_and_api.py uses it
├── core/
│   ├── settings.py   129  Every tunable. EMERGENCY_DEFAULTS for fail-open
│   ├── questions.py   72  The five questions, defined once
│   ├── policy.py     235  decide() and the three profile rows. The only thresholds
│   ├── state.py      157  build_state(): goal, next_step, item, later_findings
│   ├── splitter.py   177  split_text() and clip_head_tail()
│   ├── refiner.py    149  Second pass: judge pieces, rebuild the item
│   ├── messages.py    76  The Message model, fingerprint(), as_messages/to_dicts
│   ├── tokens.py      53  count_tokens(), clip_to_chars()
│   ├── tombstone.py   42  What a dropped message turns into
│   └── pipeline.py   147  Older entry path used by api/app.py
├── laya/
│   ├── client.py     251  Four backends behind one interface, + build_engine()
│   ├── engine.py      91  LayaEngine: batching and the concurrency cap
│   ├── hosted.py     142  The hosted API backend, with retries
│   └── types.py      169  Answer types, parse_result, derive_answer_confidence
└── storage/
    ├── store.py      251  Session-scoped side-car: decisions + mistakes
    └── box.py        202  Per-run side-car
```

### The five questions of "where is X used"

| Thing you want | Where it lives |
|---|---|
| The dummy tool-call data for evaluation | `data/traces/*.json`, generated by `scripts/make_traces.py` |
| The answer key that says what must survive | `data/traces/*.key.json` |
| The old hand-written labelled cases | `scripts/evalset.py` |
| Shared test fixtures and fake backends | `tests/conftest.py` |
| `[[NOISE]]` / `[[KEEPME]]` handling | `FakeBackend` in `src/contextcrunch/laya/client.py` |
| The thresholds | `src/contextcrunch/core/policy.py` |
| Every tunable number | `src/contextcrunch/core/settings.py`, documented in `.env.example` |
| Where originals are written | `store.py::_original_path`, under `sha1(session_id)[:16]` |
| Where the Guardian's rules live | `guardian.py::_apply_regret` and `_apply_stuck` |
| The engineering log | `progress.md` |

---

## 9. Measured results, and what is still open

### Where it stands

| Metric | Offline (`--fake`) | Real hosted model |
|---|---|---|
| Traces | 20 | 6 support traces |
| Dangerous drops | **0** | **0** |
| Facts survived | 54 / 54 | 15 / 15 |
| Tokens saved | 62.5% overall | **3.9%** |
| Time per trace | 0.19s | 17.2s |

Per archetype (offline): coding 81.5%, support 85.0%, research 17.1%. Research is low by
design — it resolves to `conservative`, which refuses to remove anything.

### The one open problem

**Real-world savings are only ~4%, and the cause is known and measured.**

Laya answers `verdict=drop` with `answer_confidence` around **0.38–0.51** on genuinely junk
material. Every shipped profile sets `conf` between **0.70 and 0.90**, so the drop gate vetoes
it. Measured per piece on a live run: 3 of 3 pieces said `drop`, 3 of 3 became `KEEP`.

This is a policy threshold, not a bug. The safety side is working — evidence is not being lost
— but the cost is that almost nothing is being removed. `data/results.csv` is the baseline to
compare any threshold change against.

There is a second, smaller effect: when a rebuild is not genuinely shorter than the original,
it is discarded. Rejoining pieces adds separators and anchor comments, and splitting loses
almost nothing, so a "saving" can evaporate. That guard is tested and deliberate — a compaction
that reports a saving it did not achieve is worse than one that admits it saved nothing.

### Other known gaps

- `core/pipeline.py` still calls the older module-level `guardian.review` against a `Box`, so
  the regret and stuck rules are not in that path. `api/main.py` does use `GuardianAgent`.
- `api/app.py` is superseded by `api/main.py`; only `tests/test_pipeline_and_api.py` imports it.
- The project is **not under version control**. There is no history to recover from when a
  file is damaged in place — which happened on 2026-10-04, when null bytes inside two source
  files made the package unimportable. `git init` is worth doing.
- 20 synthetic traces with known answers are better than 15 hand-written items, but they are
  still synthetic. Real traces labelled at ~100 items per archetype would be better.

---

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'contextcrunch'` | The package is not installed. Run `pip install -e . --no-deps` |
| `SyntaxError: source code string cannot contain null bytes` | A source file is corrupted. Find it with a byte scan; see section 5 of `progress.md` |
| `UnicodeEncodeError` printing `═` or `█` | Windows console is cp1252. Reconfigure stdout to UTF-8, as `scripts/demo.py` does |
| Mojibake in PowerShell output | PowerShell re-decodes piped output. Use `-X utf8` and read it in Python instead |
| API returns `fail_open: true` on every request | Read `notes`. A reason there means a real failure; an empty reason means the trigger check returned early |
| Guardian seems to do nothing | Check the Worker's and the Guardian's `session_id` are the same string. A mismatch is silent |
| Savings are 0% against the real model | Expected. See section 9 — the `conf` floor vetoes drops |
| A test passes but should not | Check the fixture actually reaches the code. See section 11 |

---

## 11. Two lessons that keep recurring

Recorded because both cost real debugging time and both are easy to repeat.

**A safety mechanism that silently does nothing looks exactly like one with nothing to do.**
The demo's Guardian once read a different `session_id` from the one the Worker wrote to. Nothing
crashed, nothing warned, and the output looked like a working safety net. Always confirm a
safety feature actually fired.

**A test only bites if the scenario would actually change without the code under test.**
Three separate times, a test passed while exercising nothing: a trigger test whose fixtures were
all recent turns anyway; a broken-engine test whose only tool message sat inside the protected
window; a marker test whose markers never reached the pieces being judged. When a test asserts
something is safe, confirm it is actually in range before trusting it.

The mutation checks in section 3.2 exist for this reason. Green tests are not evidence on their
own.