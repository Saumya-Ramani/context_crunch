"""Score ContextCrunch against traces with known answers.

This is the measurement the whole project has been missing. The earlier
evaluation set had fifteen hand-written items; these are full histories, sized
the way real ones are, where the right answer was planted rather than guessed.

The number that matters is **trap facts LOST**. A compaction that saves 60% and
loses one customer note is worse than one that saves nothing, because a host
cannot tell the difference from the token count alone. A LOST above zero is
printed in red and counted separately in the totals, never averaged away into a
percentage.

Everything else is secondary: tokens before and after, the reduction, how much
noise went, and how long each trace took.

Run from the repo root:

    python scripts/evaluate.py --fake     # offline, needs traces made with --mark
    python scripts/evaluate.py            # the real hosted model
    python scripts/evaluate.py --fake --archetype coding --verbose

Results are written to ``data/results.csv``, one row per trace, so two runs can
be compared with a diff rather than by eye.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ``scripts`` is a plain directory, so the repo root has to be importable for
# `from scripts...` and for the installed package when it is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextcrunch.agents.scout import AUTO, ScoutAgent  # noqa: E402
from contextcrunch.agents.worker import CompactorAgent  # noqa: E402
from contextcrunch.core.messages import as_messages  # noqa: E402
from contextcrunch.core.settings import get_settings, reset_settings_cache  # noqa: E402
from contextcrunch.core.tokens import count_tokens  # noqa: E402
from contextcrunch.laya.client import build_engine  # noqa: E402
from contextcrunch.storage.store import Store  # noqa: E402

TRACES = Path("data/traces")
RESULTS = Path("data/results.csv")

# ANSI colours, disabled when the output is piped to a file.
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


def colour(text: str, code: str, enabled: bool) -> str:
    """Wrap ``text`` in an ANSI colour when colouring is on."""
    return f"{code}{text}{RESET}" if enabled else text


@dataclass
class Outcome:
    """What one trace scored."""

    name: str
    archetype: str
    #: Defaults to empty so a trace that fails before the profile is known is
    #: still recorded, rather than crashing the whole run.
    profile: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    survived: int = 0
    lost: int = 0
    lost_facts: list[str] = field(default_factory=list)
    noise_removed: int = 0
    noise_total: int = 0
    seconds: float = 0.0
    degraded: bool = False
    note: str = ""

    @property
    def reduction_pct(self) -> float:
        """Return the percentage of tokens removed."""
        if not self.tokens_before:
            return 0.0
        return round(100.0 * (self.tokens_before - self.tokens_after) / self.tokens_before, 1)

    def row(self) -> dict:
        """Return the flat record written to the CSV."""
        return {
            "trace": self.name,
            "archetype": self.archetype,
            "profile": self.profile,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "reduction_pct": self.reduction_pct,
            "facts_survived": self.survived,
            "facts_lost": self.lost,
            "noise_removed": self.noise_removed,
            "noise_total": self.noise_total,
            "seconds": round(self.seconds, 2),
            "degraded": self.degraded,
            "note": self.note,
        }


def load_traces(directory: Path, archetype: str) -> list[tuple[Path, dict, dict]]:
    """Return (trace path, messages, answer key) for every trace in ``directory``."""
    if not directory.is_dir():
        raise SystemExit(f"no traces in {directory}. Run: python scripts/make_traces.py --mark")

    found = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".key.json"):
            continue
        key_path = path.with_suffix(".key.json")
        if not key_path.is_file():
            continue
        key = json.loads(key_path.read_text(encoding="utf-8"))
        if archetype != "all" and key.get("archetype") != archetype:
            continue
        found.append((path, json.loads(path.read_text(encoding="utf-8")), key))
    return found


async def score_one(
    path: Path,
    raw: dict,
    key: dict,
    agent: CompactorAgent,
    scout: ScoutAgent,
    default_profile: str,
    verbose: bool,
) -> Outcome:
    """Compact one trace and check the answer key against the result."""
    # Scout, Guardian and the Worker all take Message objects. Handing Scout raw
    # dicts raises an AttributeError, which would be recorded as a degraded trace
    # rather than as the wiring bug it actually is, so parsing happens first.
    messages = as_messages(raw["messages"])
    ctx = scout.run(messages, key.get("profile", AUTO), default_profile)

    outcome = Outcome(name=path.stem, archetype=key.get("archetype", "?"))
    before_text = "\n".join(m.content for m in messages)

    started = time.perf_counter()
    try:
        compacted, stats, profile_used = await agent.run(
            session_id=f"eval:{path.stem}", messages=messages, ctx=ctx
        )
        outcome.seconds = time.perf_counter() - started
        outcome.profile = profile_used
        outcome.tokens_before = stats.tokens_before or sum(
            count_tokens(m.content) for m in messages
        )
        outcome.tokens_after = stats.tokens_after
        outcome.degraded = stats.tokens_after == 0 and not stats.changed
        after_text = "\n".join(m.content for m in compacted)
    except Exception as exc:  # noqa: BLE001 - a failed trace is a data point
        outcome.seconds = time.perf_counter() - started
        outcome.degraded = True
        outcome.note = f"{type(exc).__name__}: {exc}"
        outcome.profile = ctx.profile
        outcome.tokens_before = sum(count_tokens(m.content) for m in messages)
        outcome.tokens_after = outcome.tokens_before
        after_text = before_text

    # The answer key, checked against the text that would actually be sent on.
    for fact in key.get("must_survive", []):
        if fact in after_text:
            outcome.survived += 1
        else:
            outcome.lost += 1
            outcome.lost_facts.append(fact)
            if verbose:
                print(f"    LOST: {fact[:70]}", file=sys.stderr)

    outcome.noise_total = before_text.count("[[NOISE]]")
    outcome.noise_removed = outcome.noise_total - after_text.count("[[NOISE]]")
    return outcome


async def run(args: argparse.Namespace) -> pd.DataFrame:
    """Score every trace and return the results frame."""
    settings = get_settings()
    if args.fake:
        # Force the offline backend regardless of what .env says. The setting is
        # overridden on the object AND in the environment, because
        # ``build_engine`` reads ``get_settings()`` again on the way in.
        settings = settings.model_copy(update={"laya_mode": "fake"})
        os.environ["CC_LAYA_MODE"] = "fake"
        reset_settings_cache()
        settings = get_settings()

    engine = build_engine(settings)
    scout = ScoutAgent()

    traces = load_traces(args.traces, args.archetype)
    if not traces:
        raise SystemExit(f"no traces matched archetype={args.archetype}")

    rows = []
    with tempfile_store() as store:
        agent = CompactorAgent(engine=engine, store=store, settings=settings)
        for path, raw, key in traces:
            outcome = await score_one(
                path, raw, key, agent, scout, settings.default_profile, args.verbose
            )
            rows.append(outcome.row())
            if args.verbose:
                print(f"  {outcome.name}: {outcome.reduction_pct}% lost={outcome.lost}")

    await engine.aclose()
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("trace").reset_index(drop=True)
    return frame


@contextmanager
def tempfile_store():
    """Return a Store in a throwaway directory, so evaluation writes nothing.

    The evaluation must never touch the real side-car: a run over twenty traces
    would otherwise fill the production database with decisions from synthetic
    histories, and the Guardian would learn from them.
    """
    directory = tempfile.TemporaryDirectory(prefix="cc-eval-")
    store = Store(
        db_path=str(Path(directory.name) / "store.db"),
        originals_dir=str(Path(directory.name) / "originals"),
    )
    try:
        yield store
    finally:
        store.close()
        directory.cleanup()


def render(frame: pd.DataFrame, coloured: bool) -> str:
    """Return the per-trace table."""
    header = (
        f"{'trace':<14}{'profile':<13}{'before':>8}{'after':>8}{'saved':>8}"
        f"{'traps':>7}{'lost':>6}{'noise':>7}{'sec':>7}"
    )
    lines = [colour(header, BOLD, coloured), "-" * len(header)]

    for _, row in frame.iterrows():
        traps = f"{row.facts_survived}/{row.facts_survived + row.facts_lost}"
        lost_cell = str(int(row.facts_lost))
        if row.facts_lost > 0:
            # A lost fact is the one number worth stopping for.
            lost_cell = colour(lost_cell, RED, coloured)
        note = f"  {row.note}" if row.note else ""
        lines.append(
            f"{row.trace:<14}{row.profile:<13}{int(row.tokens_before):>8}"
            f"{int(row.tokens_after):>8}{row.reduction_pct:>7}%"
            f"{traps:>7}{lost_cell:>6}{int(row.noise_removed):>4}/{int(row.noise_total):<2}"
            f"{row.seconds:>7.2f}{note}"
        )
    return "\n".join(lines)


def render_totals(frame: pd.DataFrame, coloured: bool) -> str:
    """Return the overall totals, led by the dangerous-drop count."""
    total_before = int(frame.tokens_before.sum())
    total_after = int(frame.tokens_after.sum())
    survived = int(frame.facts_survived.sum())
    lost = int(frame.facts_lost.sum())
    reduction = 100.0 * (total_before - total_after) / total_before if total_before else 0.0

    lines = ["", colour("TOTALS", BOLD, coloured), "-" * 40]
    lines.append(f"  traces          {len(frame)}")
    lines.append(f"  tokens          {total_before} -> {total_after}  ({reduction:.1f}% saved)")
    total_seconds = frame.seconds.sum()
    mean_seconds = frame.seconds.mean()
    lines.append(f"  time            {total_seconds:.1f}s total, {mean_seconds:.2f}s mean")

    verdict = colour(f"{lost} LOST", RED, coloured) if lost else "0 LOST"
    lines.append(f"  dangerous drops {verdict}   (facts that had to survive)")
    lines.append(f"  facts survived  {survived} of {survived + lost}")
    if frame.noise_total.sum():
        removed = int(frame.noise_removed.sum())
        total = int(frame.noise_total.sum())
        lines.append(f"  noise removed   {removed} of {total} blocks")
    degraded = int(frame.degraded.sum())
    if degraded:
        lines.append(f"  degraded        {degraded} traces returned nothing")

    if lost:
        lines += [
            "",
            colour("A lost fact is a wrong compaction, not a cheap one.", RED, coloured),
            colour("The host cannot see this loss in the token count.", RED, coloured),
        ]
    return "\n".join(lines)


def by_archetype(frame: pd.DataFrame, coloured: bool) -> str:
    """Return one summary line per archetype."""
    if frame.empty:
        return ""
    lines = ["", colour("BY ARCHETYPE", BOLD, coloured), "-" * 40]
    for archetype, group in frame.groupby("archetype"):
        before = int(group.tokens_before.sum())
        after = int(group.tokens_after.sum())
        saved = 100.0 * (before - after) / before if before else 0.0
        lost = int(group.facts_lost.sum())
        lost_cell = colour(str(lost), RED, coloured) if lost else "0"
        lines.append(
            f"  {archetype:<10}{before:>8} -> {after:<8} {saved:>5.1f}%   lost={lost_cell}"
        )
    return "\n".join(lines)


def main() -> None:
    """Score every trace, print the table and save the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fake",
        action="store_true",
        help="force the offline FakeBackend instead of the configured model",
    )
    parser.add_argument("--traces", type=Path, default=TRACES)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument(
        "--archetype",
        default="all",
        choices=["all", "coding", "support", "research"],
    )
    parser.add_argument("--verbose", action="store_true", help="print each lost fact")
    args = parser.parse_args()

    coloured = sys.stdout.isatty()
    mode = "fake (offline)" if args.fake else get_settings().laya_mode
    print(f"evaluating with: {mode}")
    print()

    frame = asyncio.run(run(args))
    if frame.empty:
        raise SystemExit("nothing was scored")

    print(render(frame, coloured))
    print(by_archetype(frame, coloured))
    print(render_totals(frame, coloured))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"\nsaved {len(frame)} rows to {args.out}")


if __name__ == "__main__":
    main()