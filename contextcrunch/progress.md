# ContextCrunch progress

## 2026-10-01 — first working build
- Built the full package under `src/contextcrunch/`: `core/` (settings, questions, state, policy,
  refiner, tokens, messages, pipeline), `laya/` (types, client, hosted), `storage/box.py`,
  `agents/` (scout, worker, guardian), `api/app.py`. Added `pyproject.toml` with pytest + ruff config.
- Switched Laya from the slow in-process model to the **hosted API**
  (`https://api.impossibl.com/v1/systemone`, bearer auth). Warm calls are ~0.45s vs several seconds
  locally. New `laya/hosted.py` does retries with exponential backoff and fails open.
- Key contract difference found by probing the live API: the hosted response **omits
  `answer_confidence` and `action`**. Added `derive_answer_confidence()` in `laya/types.py`, which
  reconstructs it from `probabilities` / `1 - noul` — verified to match the local model's own value.
- Verified end-to-end against the live API: 2595 → 1052 tokens (59% reduction) in 4.9s, with every
  role and `tool_call_id` preserved. Conservative profile correctly kept all citable material.
- `pytest -q` → 127 passed. `ruff check .` → clean.

## 2026-10-01 — profiles measured, then pulled back
- Added the evaluation harness the spec asks for: `scripts/evalset.py` (15 human-labelled cases,
  5 per archetype) and `scripts/tune_profiles.py` (scores profiles, caches answers, reports how
  well each Laya signal separates junk from must-keep). Answers cache in `data/tune_cache.json`.
- Measured against the live API. Three structural defects found in the original design:
  1. `consumed` does **not** separate junk from keep (40% on two archetypes) yet was a drop gate.
     It now drives TRUNCATE only, where being wrong is cheap.
  2. `superseded` separated better (75-100%) and replaced it as a gate.
  3. The doc's `relevance` ceilings of 1.0-1.5 sat *below* the measured junk range of 1.4-2.1, so
     they could never fire. Ceilings are now set from the measured junk/keep boundaries.
- Thresholds are deliberately pulled back from the maximum-savings search: a wrong drop costs
  evidence, a missed drop only tokens, and 15 cases is far too few to tune on.
- Added `MIN_KEPT_FRACTION` in the refiner: a truncation must keep at least half the text, so
  trimming can never become a quiet delete.
- Measured result, zero dangerous drops everywhere: coding 80.3%, support 46.9%, research 0%.
  Research stays at 0% because Laya says `keep` on all 5 research cases, not because of a gate.
- Live end-to-end: 2045 -> 1326 tokens (35%), stale dump dropped, install log truncated, every
  role and tool_call_id preserved. `pytest -q` 128 passed, `ruff check .` clean.

## 2026-10-01 — policy engine rebuilt to the reference spec
- Rewrote `core/policy.py` to the supplied reference logic: `decide(item, a, profile,
  pinned_tools=frozenset(), is_sub=False)` with a `Profile(ess, rel, con, conf, noise, K, anchors)`
  row per archetype. The three tables are the only numbers in code.
- Hard rules now pin `assistant` as well as `system`/`user`, so only tool output is ever judged.
  New `pinned_tools` escape hatch keeps a configured tool's output regardless of the verdict.
- The drop gate became four vetoes plus a positive **junk reason**: at least one of `consumed`,
  `superseded` or `relevance <= noise`. Any single failed gate blocks the drop.
- `is_sub=True` skips the recency and tiny-item rules (and the 800-token size truncation) so a
  passage inside a filing is judged on its own merits, while the pinned-role rule still applies.
- Added `get_profile()` so an unknown name falls back to the safest row rather than raising.
- Probability and score reads go through small helpers that accept `None` as "no answer", which
  is what makes a null answer a veto instead of a TypeError.
- Updated the two callers to the new signature: `agents/worker.py` and `core/refiner.py`.
- `tests/test_policy.py` rewritten: 42 tests covering every hard rule, every veto, each junk reason
  alone, the `is_sub` exemptions, both truncation paths, and profile divergence.
- `pytest -q` → 152 passed. `ruff check .` → clean. Live pipeline still runs, roles and
  `tool_call_id`s preserved.

### Note on savings after this change
Live pipeline saving moved from 35% to 3% on the same trace. Cause is the reference `conf` floor
(aggressive 0.70) vetoing a real drop: Laya returned `answer_confidence=0.395` on the stale file
dump, so 2 of 4 gates also failed (`relevance 1.63 > 1.5`, `consumed 0.53 < 0.80`). The thresholds
in the spec are stricter than the values measured earlier in the day. Nothing is broken and no
dangerous drop occurs; the gate is simply doing what it was asked to do.

## 2026-10-04 — worker helpers: tombstone, splitter, state
- Added `core/tombstone.py`: `tombstone_message(msg, reason)` keeps the original role, `name` and
  `tool_call_id`, so an assistant `tool_calls` entry never points at a message that vanished. The
  text is plain ASCII so it survives any console or log encoding.
- Added `core/splitter.py`: `Piece`, `split_text(text, target_chars, max_chars)` and
  `clip_head_tail(text, head, tail)`. No defaults for the sizes; callers pass Settings.
  `max_chars` is enforced in one place at the end of every path, so the bound cannot be forgotten
  by a new branch. JSON arrays split one element per piece and fall back to text splitting on a
  parse failure.
- Rewrote `core/state.py`: `later_findings` is now nearest-first and stops *before* exceeding the
  cap, so a finding is never half a finding. `build_state` takes `settings` explicitly instead of
  reaching for a global. Added `fit_state`, which shrinks findings first, then the item content by
  20%, never the goal, and stops at a 200-character floor.
- `item["tokens"]` reports the **full** item size while `item["content"]` is what Laya sees. Laya
  must be able to judge "this is enormous" without being sent all of it.
- Added `Message.fingerprint()`: the tool call behind a message, which is the recoverability signal
  the policy reasons about.
- Moved all splitting out of `core/refiner.py`, which now only judges pieces and rebuilds the item.
- Two bug classes were caught by the tests and fixed: `_fit` returned a list that was joined as a
  single block, and two dead variables that `ruff` flagged as unreachable noise.
- `pytest -q` → 206 passed. `ruff check .` → clean. Live pipeline still compacts; a slow patch of
  the hosted API made one run hit the deadline, and fail-open returned the original history intact.

## 2026-10-04 — session-scoped Storage Box (store.py)
- Added `storage/store.py`: a second, session-scoped side-car alongside the existing
  `storage/box.py`. Keyed on `(session_id, item_index)` rather than a run id, so a re-judged item
  upserts and the latest verdict replaces the earlier one instead of accumulating duplicates.
- Added a `mistakes` table the Guardian fills in when a removal turns out to have been needed.
  That is what makes the system able to learn: a repeated mistake is evidence that a threshold is
  wrong. `log_mistake` uses INSERT OR IGNORE so a Guardian that runs repeatedly cannot inflate it.
- `was_removed` answers the Guardian's question, "have we removed this before?", and needs three
  conditions to hold: the fingerprint matches, the action removed content (DROP or TRUNCATE, since
  a truncation loses its middle), and the earlier index is strictly less than the current one. A
  NULL fingerprint never matches, because an unidentified item cannot be shown to be the same one.
- `top_removed` orders by relevance, putting unscored items last rather than treating NULL as zero
  and losing them in the ordering; ties break on index so the result is deterministic.
- **Path safety**: originals live at `<originals_dir>/<sha1(session_id)[:16]>/<item_index>.txt`.
  Session ids come from the calling agent, so `../x`, `a/b` and `/absolute` are hostile input, not
  a hypothetical. The item index is coerced to `int` for the same reason.
- Verified the traversal tests actually bite: removing the sha1 from the path makes 5 of them fail.
  Restored, all pass.
- Concurrency: one shared connection with `check_same_thread=False` plus a `threading.Lock` held
  across the whole read-modify-write, not just the execute call. Tested with 4 threads and with 8
  racing mistake logs.
- `pytest -q` → 254 passed. `ruff check .` → clean.

## 2026-10-04 — ScoutAgent and GuardianAgent
- Added `ScoutAgent` / `SessionContext` to `agents/scout.py` alongside the existing module-level
  helpers the pipeline still calls. One keyword table serves both entry points, so they cannot
  drift apart. Profile comes from counting keyword hits in `tool_name` of tool messages; the goal
  joins every user message with `" | "` and is deliberately **not** truncated here, because
  `build_state` owns `settings.goal_max_chars` and clipping twice would make the limit untraceable.
- Added `GuardianAgent` / `Overrides` to `agents/guardian.py`, reading the session-scoped `Store`.
  Rule 1 (regret) logs a mistake when a tool message was removed earlier in the same session, and
  forces `conservative` at two mistakes. Rule 2 (stuck) restores the three most relevant removals
  when the same error repeats three times, matching on the first 200 characters so a changing line
  number cannot disguise a repeat.
- `observe()` wraps both rules in one try/except and logs to stderr, returning empty `Overrides`.
  Verified by mutation: removing the `except` makes both broken-store tests fail, so the fail-safe
  is enforced by code rather than only documented.
- The Guardian can only ever add information back or move to a stricter profile. That asymmetry is
  stated in the module docstring and pinned by a test.
- `pytest -q` -> 335 passed. `ruff check .` -> clean.

### Two real limitations of the specified Scout keyword lists, found by tests
1. **`kb_search` is ambiguous.** `kb_` is a support keyword and `search` is a research keyword, so
   the name scores in both and ties. The specified tie rule then sends it to the default profile.
   Documented in a test rather than silently "fixed", because changing the keyword list is a
   domain decision.
2. **camelCase tool names score zero.** Matching is case-insensitive substring, so `ReadFile` does
   not match `read_file`. This fails safe: an unmatched name contributes nothing and the session
   falls back to the default instead of being guessed into the aggressive profile.

## 2026-10-04 — health check: four real defects found and fixed
A full pass was run over the whole project: `pytest`, `ruff`, every script, the live hosted
API, and the FastAPI server end to end. Four genuine defects surfaced, three of which had
been silently breaking the code the user was trying to run.

1. **Null bytes in two source files.** `agents/worker.py` had 1 and `storage/store.py` had 3,
   each sitting *inside* a token rather than between characters, so `import contextcrunch`
   raised `SyntaxError: source code string cannot contain null bytes` and 4 test modules
   failed to collect. This is the single most likely cause of the errors reported earlier:
   nothing in the package could be imported. Repaired by restoring the characters each null
   byte had overwritten (`.run_id`, `` ``CC_STORE_ORIGINALS_DIR`` ``, `~contextcrunch`, and
   `core.settings import`). A byte-level sweep over `src/` and `tests/` is now clean.
2. **`refine()` could make an item longer.** A TRUNCATE splits the item and rejoins the
   surviving pieces with `PIECE_SEPARATOR`, plus per-piece anchors. That overhead can exceed
   the text actually removed: a live run truncated a 3680-char item to **3683 chars**, i.e.
   0% saving where a saving was claimed. Two guards were added — return the original when
   every piece is KEEP, and return the original when the rebuild is not shorter. Both were
   mutation-verified: removing either one fails a test.
3. **`scripts/tune_profiles.py` was dead on arrival.** It called `decide()` with the old
   keyword signature (`role=`, `answers=`), the policy had been rewritten to take an item
   dict; it imported `split_pieces`, which moved to `core/splitter.py` as `split_text`; and
   it passed a `LayaResult` where a `dict[str, Answer]` was expected. Every one of these
   raised on the first real run. All three fixed, and the script now runs across all three
   archetypes, online and `--offline`.
4. **The package was not installed.** Tests passed only because pytest injects `src` via
   `pythonpath`. Every script and `uvicorn contextcrunch.api.app:create_app` failed with
   `ModuleNotFoundError: No module named 'contextcrunch'`. Installed with
   `pip install -e . --no-deps`. **This was the other cause of the reported run errors.**

Verified after the fixes:
- `pytest -q` -> **338 passed**; `ruff check .` -> clean.
- Live hosted API: `check_hosted.py` OK; `check_pipeline.py` compacts safely, roles and
  `tool_call_id`s preserved, fail-open intact.
- FastAPI end to end via a real uvicorn server: `/health` OK, `/v1/compact` **4602 -> 3940
  tokens (14.4%)** with `degraded=false`, Guardian escalated aggressive -> balanced, an empty
  history rejected with 422, and a second call succeeded.
- `tune_profiles.py` completes for coding/support/research with **zero dangerous drops**.

### Note on current savings
`check_pipeline.py` now shows 0% on its own trace. That is not the growth bug returning — it
is the `conf` floor (0.70) vetoing drops, since Laya reports `answer_confidence` around
0.4-0.6 on genuine drops. A history large enough to clear `CC_TRIGGER_TOKENS=2000` does save
(14.4% via the API). The `conf` floor remains uncalibrated, as noted under Next.

## 2026-10-04 — Worker rewritten as CompactorAgent (PLAN / ACT / VERIFY)
- `agents/worker.py` now leads with `CompactorAgent(engine, store, settings)` and
  `run(session_id, messages, ctx, overrides) -> (messages, Stats, profile_name)`.
  Collaborators are injected rather than looked up, so a test can drive a policy path with no
  network call. The module docstring explains all three phases in plain English.
- **PLAN** returns the history untouched below `trigger_tokens` with zero-change stats. Only
  `tool` messages are eligible. Three things make an item a KEEP with no model call: it is in
  `overrides.restore`, its tool is in `ctx.pinned_tools`, or it is within `profile.K` of the end.
  Items over `item_max_chars` are marked split-first instead of being sent whole.
- **ACT** judges every eligible item in ONE batch, then applies `policy.decide`. Split-first
  items go straight to the rebuild, bypassing the whole-item judgement: an item too big for the
  window cannot be judged by summarising its head. Pieces are judged with `is_sub=True`, joined
  with `"\n---\n"`, anchored as `"[<anchor>] "` only when `profile.anchors`. If every piece is
  dropped the item becomes one tombstone rather than a filing of tombstone lines.
- **VERIFY** checks count, roles, `tool_call_id`s and empty content, and returns the ORIGINALS
  on any problem. `run()` has no `try/except` and no deadline: fail-open belongs to the API.
  Proven by a test that asserts `RuntimeError` escapes.
- Added `laya/engine.py` (`LayaEngine`) so batching and the concurrency cap live in one place
  instead of inside the Worker. It judges many states at once, returns answers positionally
  aligned, and never swallows an error.
- `compact()` / `Outcome` kept as a compatibility shim so `core/pipeline.py` and the API keep
  working. `box` is still accepted; dropping it broke `test_pipeline_shrinks_a_history`, which
  is how the parameter's removal was caught.

### Two bugs the tests caught
1. **`action=None` written to the store.** `_rebuild` returns `(text, action)` but `_act`
   unpacked it as `(action, text)`, so every split-first item logged `action=None` and SQLite
   raised `NOT NULL constraint failed: decisions.action`.
2. **A clip that could not clip.** The head/tail sizes came from `piece_max_chars` alone, so
   when that was large relative to the content, `head + tail` exceeded the text and
   `clip_head_tail` correctly returned it unchanged — the item was marked TRUNCATE and saved
   nothing. `_clip_sizes` now caps the sum at half the text.

`pytest -q` -> **354 passed**, `ruff check .` -> clean.

### The tests are mutation-verified, not just green
`scripts/mutation_check.py` removes one safety behaviour at a time and asserts a test fails.
All 9 are caught: VERIFY skipped, trigger ignored, recent turns unprotected, pinned tool
unprotected, restore ignored, non-tool messages judged, store never written, empty content
allowed, lost `tool_call_id` allowed. The first run left one **survivor** — the trigger test
passed even with the trigger check deleted, because every item in its fixture was recent and so
kept anyway. The fixture now has several deep tool messages that *would* change if judged.

### Why the live run saves 0%
Worth stating plainly, because it looks like a regression and is not. Against the live model the
compactor returns `KEEP` on genuine junk: Laya answers `verdict=drop` but with
`answer_confidence` **0.38-0.51**, and the aggressive profile's `conf` floor is **0.70**, so the
gate vetoes it. Measured per piece: all three said `drop`, all three became `KEEP`. The drop path
itself is verified working end to end against the fake backend: 2171 -> 32 tokens, tombstone
written, original recoverable from the store. The `conf` floor is still uncalibrated and is the
single biggest lever on real-world savings.

## 2026-10-05 — the HTTP service (api/main.py)
- `create_app(settings=None, engine=None)` builds the Store, Scout, Guardian and Compactor once
  and holds them on `app.state`. `settings` and `engine` are injectable so a test can drive a
  policy path with no network call; **nothing is constructed at import time**, which is what makes
  `uvicorn --factory` and importing the module in a test work without a valid environment.
- Run command, documented in the module docstring and verified live:
  `uvicorn contextcrunch.api.main:create_app --factory --port 8080`
- `POST /v1/compact` takes `messages`, `session_id`, `profile`, `goal`. Order is Scout -> Guardian
  -> Compactor under `asyncio.wait_for(..., settings.deadline_s)`. Response carries
  `profile_used`, `fail_open`, `tokens_before/after`, `reduction_pct`, `kept/shortened/removed`,
  `actions`, `restored` and `notes`.
- **Fail-open is mandatory and total.** One `except Exception` around the whole handler returns the
  caller's original messages with `tokens_before == tokens_after`, `profile_used="none"` and
  `fail_open=True`. The traceback goes to `LOG.warning(..., exc_info=True)` and never to the
  caller. Only malformed *input* returns 422, because that is the host's bug and hiding it would
  hide the bug.
- Added `GET /v1/sessions/{session_id}/stats` -> decisions and mistake count, and `logging
  .basicConfig(level=INFO)` inside `create_app` as specified.

### A real ordering bug the tests caught
`as_messages(...)` ran *after* `scout.run(...)`, so Scout was handed raw dicts and raised
`AttributeError: 'dict' object has no attribute 'role'`. Fail-open then swallowed it and every
request returned `fail_open=True` with a misleading reason — the service looked healthy while
compiling nothing. Parsing now happens first, so a malformed message is a clear error rather than
a disguised compaction failure.

### A fixture bug that made three tests vacuous
`trace()` ended with the tool message, so it sat inside the profile's protected window of recent
turns and **nothing was ever eligible for judgement**. The broken-engine and timeout tests
"passed" without the engine ever being called. The fixture now appends trailing assistant turns.
This is the same class of mistake as the earlier trigger-test survivor: a test only bites if the
scenario would actually change without the code under test.

### Mutation-verified fail-open
`scripts/mutation_api_check.py` breaks one guarantee at a time; **all 8 are caught**: fail-open
removed, deadline removed, `fail_open` hardcoded false, empty history returned instead of the
original, `tokens_after` lying, traceback returned to the caller, Guardian skipped, session id
ignored.

`pytest -q` -> **370 passed**, `ruff check .` -> clean.

### Live service verified
Against the hosted API on port 8080: `/health` OK; `/v1/compact` 3940 tokens with all 12 messages,
roles and `tool_call_id`s preserved and `fail_open=false`; `/v1/sessions/smoke/stats` returned
`{"KEEP": 1, "TRUNCATE": 2}`; an unknown session returned zeroes rather than an error; an empty
history returned 422; a second call in the same session succeeded. Server logs show the real
hosted endpoint being called with several requests in flight at once.

`shortened=0` on that run is the no-growth guard, not a bug: the item split into 3 pieces losing
only 5 characters, so rejoining them with separators would have been *longer*, and the rebuild was
correctly discarded. Real savings remain blocked by the uncalibrated `conf` floor below.

## 2026-10-05 — synthetic traces and a real measurement
There were no real traces, so nothing could be measured against ground truth. These
are full histories where the right answer was *planted*, so it is known in advance.

- `scripts/make_traces.py` writes 20 seeded traces (8 coding, 6 support, 6 research) to
  `data/traces/`, each with an answer key beside it (`<name>.key.json`). Deterministic: each
  trace gets its own RNG seeded from `f"{seed}:{archetype}:{index}"`, so adding or removing a
  trace does not change the others. Verified byte-identical across runs.
- Planted cases: **used-up dump** (big output, key line repeated by a later assistant message),
  **trap** (big output, fact never mentioned again — the dangerous one), **superseded** (failing
  run then passing run), **noise** (install logs), **risky tool** (`refund_issued` /
  `send_email`, must survive). Filler is log lines, code dumps, JSON arrays and boilerplate,
  sized 3,000-20,000 characters to match the real limits.
- `scripts/evaluate.py` runs Scout then CompactorAgent per trace, checks the answer key against
  the text that would actually be sent, prints per-trace and per-archetype tables plus totals, and
  saves `data/results.csv`. **Trap facts LOST is the headline number**, printed in red and never
  averaged into a percentage, because a compaction that saves 60% and loses one customer note is
  worse than one that saves nothing.
- Added `build_engine(settings)` to `laya/client.py` and taught `FakeBackend` to answer from
  `[[KEEPME]]` / `[[NOISE]]` markers, falling back to the old hash when neither is present.

### The 31 dangerous drops that were not the compactor's fault
The first `--fake` run reported **31 LOST of 54**, every one a planted trap. Traces and markers
both looked correct, so the compactor was the suspect. It was not: `mark()` put a single
`[[KEEPME]]` at the *top* of a 16,000-character item, but an item over `CC_ITEM_MAX_CHARS` is
**split first** and each piece is judged on its own. Only piece 1 carried the marker; the other
six fell back to the hash and some were dropped, destroying the fact planted in them.
**Markers must be per piece, not per item**, because the offline model can only answer per piece.
After the fix: **0 LOST, 54/54 survived, 83.7% saved, 306/306 noise blocks removed.**
Confirmed the tool can still see a real loss by injecting a fact that was never planted
(it reported `4/5, lost=1`) — otherwise a green run would mean nothing.

### Measured against the real hosted model (unmarked traces, support archetype)
**0 dangerous drops, 15/15 facts survived, but only 3.9% saved** (130,569 -> 125,434, 17.2s per
trace). This is the `conf` floor again, now measured properly rather than on one hand-made trace:
Laya reports `answer_confidence` around 0.4-0.6 on correct drops, every shipped profile requires
0.70-0.90, so drops are vetoed. **The safety side is genuinely working — evidence is not being
lost — and the cost is that almost nothing is being removed.** With 20 traces there is finally a
measurement to tune against instead of 15 hand-written items.

`pytest -q` -> **370 passed**. `ruff check .` -> clean.

## 2026-10-05 — demo and README
- `scripts/demo.py` walks the real pipeline on a real trace and prints, in order: banner, Scout's
  verdict, a per-message Worker table (index, role, tool, tokens before -> after, a coloured
  action label and a block-character size bar), totals extrapolated to a 100-step run, a PROOF
  section checking each planted fact, the Guardian's two rules, and a closing note. Stdlib only,
  no LLM needed with `--fake`, **exactly 60 lines** on the default trace.
- Uses a `tempfile` Store so a demo can never write to real data. Colour is dropped when stdout
  is not a terminal, so piping to a file does not fill it with escape codes.

### Three bugs the demo surfaced
1. **`UnicodeEncodeError` on Windows.** A cp1252 console cannot encode `═` or `█`, so the script
   died on the first banner. Fixed by reconfiguring stdout to UTF-8. *Any* script here that
   prints non-ASCII needs this; `make_traces.py` does not, because it writes files.
2. **The Guardian silently did nothing.** The Worker wrote its decisions under session `demo`
   while the Guardian read `demo-session`, so it was staring at an empty side-car. Nothing
   crashed and nothing warned — it just looked exactly like the safety net working. Fixed with
   one `SESSION_ID` constant. **This is the most dangerous class of bug in this project: a
   safety mechanism that no-ops looks exactly like one that has nothing to do.**
3. **`_has_removals` swallowed a `TypeError`.** It indexed `raw["messages"]` when `load()`
   already returns the list, and the bare `except` turned that into "this trace is not a good
   demo", so *every* trace looked removal-free. Now the skip reason is printed.

### A trace-generator defect the demo exposed
The spec's example line is `research -> profile: conservative`, but the demo printed
`aggressive`. Checking all 20 traces: **all 20 were detected as `coding`**, because
`used_up_dump` and `superseded_run` hardcoded `read_file` and `run_tests` regardless of
archetype, and those two coding keywords outscored everything. The archetype column of the
evaluation was therefore meaningless. Fixed with per-archetype tool names, findings, trap
phrasing and check output. Now **0 of 20 mismatches**, and the demo prints
`Detected agent type: research -> profile: conservative` as specified.

Re-measured after the fix: **0 dangerous drops, 54/54 facts survived**, and the per-archetype
split now shows what it should — coding 81.5%, support 85.0%, research 17.1%, because research
resolves to `conservative` and that profile is deliberately reluctant to remove anything. The
old 83.7% headline was inflated by coding traces wearing a research label.

- The demo also picks a trace that actually removes something. The default research trace is
  compacted by `conservative` and removes nothing, which is correct behaviour but a poor demo,
  so it falls back to a coding trace and says so rather than showing a table of KEEPs.
- Added `README.md` and `.env.example` (which did not exist, so the documented setup step would
  have failed). The README's host example is verified by `scripts/check_readme_example.py`,
  which starts the real service and runs it: **6 of 6 messages returned, `tool_call_id`s
  preserved, stats recorded**, against the live hosted API.

`pytest -q` -> **370 passed**. `ruff check .` -> clean.

## Next
- **Calibrate the `conf` floor.** Now measurable across three archetypes: `data/results.csv`
  gives the safety/saving trade-off per trace. The floor is what stands between ~4% and a real
  saving against the real model.
- Run the full 20-trace evaluation against the hosted model and keep it as the baseline.
- Wire `GuardianAgent` into `core/pipeline.py`: it uses the older module-level `guardian.review`
  against a `Box`, so the regret and stuck rules are not in that path. (`api/main.py` does use
  `GuardianAgent.observe`.)
- `api/app.py` is superseded by `api/main.py`; only `tests/test_pipeline_and_api.py` imports it.
- Consider `git init`. The project is not under version control, so there is no history to
  recover from when a file is damaged in place, as happened on 2026-10-04.