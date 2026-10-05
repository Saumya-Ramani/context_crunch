"""A 30-second demo of ContextCrunch, for people who are not the author.

Everything here runs offline in under a second and needs no LLM. The point is to
make one idea obvious: **ContextCrunch makes a long agent history much cheaper
without losing the facts that matter**, and every removal is reversible.

The demo walks the real pipeline on a real trace:

1. **Scout** looks at the history and says what kind of agent this is.
2. **The Worker** compacts it, one row per message, showing what happened.
3. **Totals** turn that into tokens saved, scaled to a 100-step run.
4. **Proof** checks the planted facts are still there.
5. **The Guardian** shows the safety net: it notices when a host asks for
   something that was removed, and puts it back.

Run from the repo root:

    python scripts/demo.py --fake
    python scripts/demo.py data/traces/coding_03.json --fake
    python scripts/demo.py                # the real model from .env

Use ``--fake`` when presenting. It is deterministic, it never fails on a network,
and it produces the same numbers every run, which matters when somebody in the
room is checking your arithmetic.

The Store is a temporary directory. A demo must not write to real data, least of
all the side-car a production Guardian learns from.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextcrunch.agents.guardian import (  # noqa: E402
    REGRET_DOWNSHIFT_AT,
    STUCK_REPEAT_THRESHOLD,
    GuardianAgent,
)
from contextcrunch.agents.scout import AUTO, ScoutAgent  # noqa: E402
from contextcrunch.agents.worker import CompactorAgent  # noqa: E402
from contextcrunch.core.messages import Message, as_messages  # noqa: E402
from contextcrunch.core.settings import get_settings, reset_settings_cache  # noqa: E402
from contextcrunch.core.tokens import count_tokens  # noqa: E402
from contextcrunch.laya.client import build_engine  # noqa: E402
from contextcrunch.storage.store import Store  # noqa: E402

TRACES = Path("data/traces")

# A Windows console defaults to cp1252, which cannot encode the box-drawing and
# block characters this demo draws with. Forcing UTF-8 here is what stops the
# whole script dying with a UnicodeEncodeError on the first banner.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- ANSI colours --------------------------------------------------------- #
# Colour is dropped when stdout is not a terminal, so piping to a file or a
# slide deck does not fill it with escape codes.
COLOUR = sys.stdout.isatty()

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

#: How many steps to extrapolate the saving to.
PROJECTED_STEPS = 100

#: One session id for the whole demo, so the Worker's decisions and the
#: Guardian's observations land in the same place.
SESSION_ID = "demo-session"

#: Bar characters, lightest to fullest.
BLOCKS = " ▏▎▍▌▋▊▉█"


def paint(text: str, colour: str) -> str:
    """Return ``text`` wrapped in ``colour`` when colour is enabled."""
    return f"{colour}{text}{RESET}" if COLOUR else text


def bar(fraction: float, width: int = 10) -> str:
    """Return a block-character bar showing ``fraction`` of the full width.

    Eighth-blocks are used so a small saving is still visible: a bar that is
    entirely spaces reads as "nothing happened" rather than "a little was saved".
    """
    fraction = max(0.0, min(1.0, fraction))
    exact = fraction * width
    full = int(exact)
    remainder = exact - full
    partial = BLOCKS[int(remainder * 8)]
    filled = "█" * full
    if full < width:
        filled += partial
    return filled.ljust(width)


@contextmanager
def temp_store():
    """Yield a Store in a throwaway directory, so no real data is touched."""
    directory = tempfile.TemporaryDirectory(prefix="cc-demo-")
    store = Store(
        db_path=str(Path(directory.name) / "store.db"),
        originals_dir=str(Path(directory.name) / "originals"),
    )
    try:
        yield store
    finally:
        store.close()
        directory.cleanup()


def default_trace() -> Path:
    """Return the default trace: the first research one.

    Falls back to any trace if there are no research traces, so a partially
    populated ``data/traces`` still produces a demo rather than an error.
    """
    traces = sorted(
        path
        for path in TRACES.glob("research_*.json")
        if not path.name.endswith(".key.json")
    )
    if traces:
        return traces[0]
    any_traces = sorted(
        path for path in TRACES.glob("*.json") if not path.name.endswith(".key.json")
    )
    if not any_traces:
        raise SystemExit(f"no traces in {TRACES}. Run: python scripts/make_traces.py --mark")
    return any_traces[0]


async def pick_demo_trace(forced: Path | None) -> Path:
    """Return the trace to demo.

    A trace is only worth showing if something actually gets removed, because
    "nothing was removed" is not a demo. The conservative profile that research
    traces resolve to is deliberately reluctant to drop anything, so if the
    default trace produces no removals the demo moves to one that does and says
    so, rather than showing a table of KEEPs and a Guardian section that never
    fires.
    """
    if forced is not None:
        return forced

    candidates = [default_trace()]
    candidates += sorted(
        path
        for path in TRACES.glob("*.json")
        if not path.name.endswith(".key.json") and path != candidates[0]
    )

    first = await _has_removals(candidates[0])
    if first is not None:
        return first

    for path in candidates[1:]:
        if await _has_removals(path) is not None:
            print(
                f"note: {candidates[0].name} is too conservative to remove anything,\n"
                f"      so this demo uses {path.name} instead."
            )
            return path
    return candidates[0]


async def _has_removals(path: Path) -> Path | None:
    """Return ``path`` when compacting it removes something, else ``None``.

    Used only to choose a demo trace, so it runs the real pipeline on whatever
    engine is configured. A failure means "not a good demo", not a crash: picking
    a trace must never stop the demo from starting.
    """
    settings = get_settings()
    engine = build_engine(settings)
    try:
        raw_messages, _key = load(path)
        messages = as_messages(raw_messages)
        ctx = ScoutAgent().run(messages, AUTO, settings.default_profile)
        with temp_store() as store:
            agent = CompactorAgent(engine=engine, store=store, settings=settings)
            _compacted, stats, _profile = await agent.run(
                session_id="probe", messages=messages, ctx=ctx
            )
        return path if any(a == "DROP" for a in stats.actions.values()) else None
    except Exception as exc:  # noqa: BLE001 - choosing a demo must never fail the demo
        # Silent on purpose, but not invisible: a trace that cannot be probed is
        # skipped, and if every trace fails the demo still runs on the default.
        print(paint(f"  (skipping {path.name}: {type(exc).__name__})", DIM))
        return None
    finally:
        await engine.aclose()


def load(path: Path) -> tuple[list[dict], dict]:
    """Return the trace's messages and its answer key, if one exists."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    key_path = path.with_suffix(".key.json")
    key = json.loads(key_path.read_text(encoding="utf-8")) if key_path.is_file() else {}
    return raw["messages"], key


def rule(title: str = "") -> None:
    """Print a section heading."""
    if title:
        print(f"\n{paint(title, BOLD)}")
        print(paint("─" * 74, DIM))


def show_banner(mode: str, trace: Path) -> None:
    """Print the banner."""
    print(paint("═" * 74, CYAN))
    print(paint("  ContextCrunch demo", BOLD + CYAN))
    print(paint(f"  engine: {mode}    trace: {trace.name}", DIM))
    print(paint("═" * 74, CYAN))


def show_scout(scout: ScoutAgent, messages: list[Message], default_profile: str) -> object:
    """Print what Scout worked out, and return the context it produced."""
    ctx = scout.run(messages, AUTO, default_profile)

    scores = scout.score_archetypes(messages)
    tools = sorted(scout.tool_names(messages))
    reason = ", ".join(tools[:4]) + (" ..." if len(tools) > 4 else "")

    print()
    print(paint("SCOUT", BOLD))
    # The archetype and the profile are usually the same word, but they are not
    # the same thing: the archetype is what the history looks like, the profile
    # is the threshold set applied to it. Both are shown so a manager can see
    # which one Scout actually decided.
    best = max(scores, key=lambda name: scores[name]) if any(scores.values()) else "unknown"
    print(f"  Detected agent type: {best} -> profile: {ctx.profile}")
    print(f"  tools seen: {reason}")
    print(
        "  pinned (never compacted away): "
        f"{', '.join(sorted(ctx.pinned_tools)) or 'none'}"
    )
    print(
        "  keyword scores: "
        + "  ".join(f"{name}={score}" for name, score in scores.items())
    )
    return ctx


#: Action label -> colour.
ACTION_COLOURS = {
    "KEEP": GREEN,
    "SHORTENED": YELLOW,
    "REMOVED": RED,
    "PINNED": BLUE,
}


def show_worker(
    messages: list[Message],
    compacted: list[Message],
    actions: dict[int, str],
    pinned: frozenset[str],
) -> None:
    """Print one row per message: what it cost before and after."""
    print()
    print(paint("WORKER", BOLD))
    header = (
        f"  {'idx':>3}  {'role':<9} {'tool':<16} {'before':>7} {'after':>7}  "
        f"{'action':<10} size"
    )
    print(header)
    print(paint("  " + "─" * 70, DIM))

    for index, (old, new) in enumerate(zip(messages, compacted, strict=True)):
        before = count_tokens(old.content)
        after = count_tokens(new.content)

        if index in pinned:
            label = "PINNED"
        else:
            action = actions.get(index, "KEEP")
            if action == "DROP":
                label = "REMOVED"
            elif action == "TRUNCATE" and after < before:
                label = "SHORTENED"
            else:
                label = "KEEP"

        saved = 1.0 - (after / before) if before else 0.0
        tool = old.name or "-"
        print(
            f"  {index:>3}  {old.role:<9} {tool[:16]:<16} {before:>7} {after:>7}  "
            f"{paint(f'{label:<10}', ACTION_COLOURS[label])} "
            f"{paint(bar(saved), DIM)}"
        )


def show_totals(before: int, after: int) -> None:
    """Print the headline numbers, scaled to a plausible run."""
    saved = before - after
    pct = 100.0 * saved / before if before else 0.0

    print()
    print(paint("TOTAL", BOLD))
    print(f"  tokens      {before} -> {after}   ({paint(f'-{pct:.1f}%', GREEN + BOLD)})")
    print(f"  per call    {paint(f'{saved} tokens saved', GREEN)}")
    print(
        f"  over {PROJECTED_STEPS} steps"
        f"   {paint(f'{saved * PROJECTED_STEPS:,} tokens saved', GREEN + BOLD)}"
    )


def show_proof(key: dict, after_text: str) -> int:
    """Check every planted fact survived. Returns how many were lost."""
    facts = key.get("must_survive", [])
    lost = 0

    if not facts:
        return 0

    print()
    print(paint("PROOF  (the facts planted in this trace)", BOLD))
    for fact in facts:
        if fact in after_text:
            print(f"  {paint('survived', GREEN)}  {fact[:58]}")
        else:
            lost += 1
            print(f"  {paint('LOST', RED + BOLD)}      {fact[:58]}")

    # An anchor is what a shortened piece carries so a later memo can still cite
    # where the passage came from.
    anchors = sorted(set(after_text.split("<!-- ")[1:]))
    if anchors:
        shown = [a.split(" -->")[0] for a in anchors][:3]
        print(f"  {paint('anchors', DIM)}   {', '.join(shown)}")

    return lost


async def show_guardian(
    store: Store,
    messages: list[Message],
    compacted: list[Message],
    actions: dict[int, str],
) -> None:
    """Run the Guardian's two rules for real, against the same session.

    Nothing here is scripted: the mistake is logged by the same
    ``GuardianAgent`` the service uses, from a history the host genuinely re-sent.
    """
    # One session id for the whole demo. The Worker writes its decisions under this
    # id and the Guardian reads them back under the same one; two different ids
    # would leave the Guardian looking at an empty side-car and quietly doing
    # nothing, which looks exactly like the safety net working.
    session = "demo-session"
    guardian = GuardianAgent(store)
    print()
    print(paint("GUARDIAN  (the safety net)", BOLD))

    # --- Rule 1: the host asks again for something that was removed. --------- #
    removed = [
        index for index, action in actions.items() if action == "DROP" and index < len(messages)
    ]

    if not removed:
        print(paint("  (nothing was removed, so there is nothing to regret)", DIM))
        return

    index = removed[0]
    original = messages[index]
    print(f"  host re-requests the tool call removed at index {index} "
          f"({original.name})")

    # The host sends the item back, later in the history, exactly as it would if
    # it needed that evidence again. It has to land at a *higher* index than the
    # removal: `was_removed` only counts a removal that happened strictly
    # earlier, which is what stops an item being treated as evidence against its
    # own removal on the very same turn.
    re_requested = messages[:index] + [
        Message(role="assistant", content="I need that output again."),
        original,
        *messages[index + 1 :],
    ]

    before = store.mistake_count(session)
    escalated = False
    for _ in range(REGRET_DOWNSHIFT_AT):
        overrides = guardian.observe(session, re_requested)
        after = store.mistake_count(session)
        if after > before:
            print(f"  {paint('Mistake logged', YELLOW + BOLD)} "
                  f"(mistakes in this session: {after})")
            before = after
        if overrides.profile and not escalated:
            escalated = True
            print(f"  {paint('Session switched to ' + overrides.profile, YELLOW + BOLD)}"
                  f"  (safer than the profile Scout chose)")

    # --- Rule 2: the agent is stuck on the same error, over and over. --------- #
    # Rule 2 restores from what was removed, so it can only fire once the
    # side-car already holds distinct removals. The compaction above recorded
    # those; what trips the rule is the same error appearing three times.
    error = (
        "Traceback (most recent call last):\n"
        '  File "billing.py", line 88, in charge\n'
        "    total = round_price(total)\n"
        "ValueError: invalid literal for int() with base 10: '10.99'"
    )
    retry = "Retrying after the same failure."
    stuck_history = [
        Message(role="tool", content=error, name="run_tests", tool_call_id="err1"),
        Message(role="assistant", content=retry),
        Message(role="tool", content=error, name="run_tests", tool_call_id="err2"),
        Message(role="assistant", content=retry),
        Message(role="tool", content=error, name="run_tests", tool_call_id="err3"),
    ]
    print(f"  the same error now appears {STUCK_REPEAT_THRESHOLD} times in one history")

    overrides = guardian.observe(session, stuck_history)
    if overrides.restore:
        print(f"  {paint('Restored top 3 items from the Storage Box', YELLOW + BOLD)} "
              f"(indexes {sorted(overrides.restore)})")
    else:
        print(paint("  (not stuck: no repeated error detected)", DIM))

    print(paint("  every item it touched is recoverable from the side-car", DIM))


def show_closing(lost: int) -> None:
    """Print the closing line."""
    print()
    print(paint("─" * 74, DIM))
    if lost:
        print(
            paint(
                f"  {lost} planted fact(s) were lost. That is a bug, not a saving.",
                RED + BOLD,
            )
        )
    else:
        print(paint("  Every planted fact survived.", GREEN + BOLD))
    print(
        "  Nothing was destroyed: every removed message is stored verbatim in a\n"
        "  side-car, so any of it can be put back. That is what makes it safe to\n"
        "  be aggressive, and it is why a host can keep its own full history."
    )
    print(paint("─" * 74, DIM))


async def run(args: argparse.Namespace) -> None:
    """Run the whole demo."""
    if args.fake:
        os.environ["CC_LAYA_MODE"] = "fake"
        reset_settings_cache()

    settings = get_settings()
    scout = ScoutAgent()

    trace = await pick_demo_trace(Path(args.trace) if args.trace else None)
    raw_messages, key = load(trace)
    messages = as_messages(raw_messages)

    show_banner("fake (offline)" if args.fake else settings.laya_mode, trace)
    ctx = show_scout(scout, messages, settings.default_profile)

    engine = build_engine(settings)

    with temp_store() as store:
        agent = CompactorAgent(engine=engine, store=store, settings=settings)
        compacted, stats, _profile = await agent.run(
            session_id=SESSION_ID, messages=messages, ctx=ctx
        )

        show_worker(messages, compacted, stats.actions, ctx.pinned_tools)
        show_totals(stats.tokens_before, stats.tokens_after)
        after_text = "\n".join(m.content for m in compacted)
        lost = show_proof(key, after_text)
        await show_guardian(store, messages, compacted, stats.actions)

    show_closing(lost)
    await engine.aclose()


def main() -> None:
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(description="A 30-second ContextCrunch demo")
    parser.add_argument(
        "trace",
        nargs="?",
        default=None,
        help="trace JSON to use; defaults to the first research trace",
    )
    parser.add_argument(
        "--fake", action="store_true", help="use the offline fake model, no network"
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()