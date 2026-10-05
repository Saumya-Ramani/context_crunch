"""Generate synthetic agent traces with known answers.

There are no real agent traces yet, so nothing can be measured against ground
truth. This script manufactures the next best thing: histories where the right
answer is known in advance, because we planted it.

Every trace mixes two kinds of material. **Planted** content carries a fact that
either must survive or should go:

- **used-up dump** - a large output whose one useful line a *later* assistant
  message repeats. Safe to drop, because the insight is already in the history.
- **trap** - a large output holding a fact that is never mentioned again. This is
  the dangerous case: dropping it loses the only copy.
- **superseded** - a failing test run followed by a passing one.
- **noise** - install logs and boilerplate. Free to remove.
- **risky tool** - a ``refund_issued`` or ``send_email`` result. A side-effecting
  call is the only record that it happened, so it must never be dropped.

**Filler** is realistic bulk: log lines, file dumps, JSON arrays, boilerplate
paragraphs, sized between roughly 3,000 and 20,000 characters, because that is
the range the real limits are set for.

Each trace is written twice: the history itself, and an answer key beside it
naming the exact strings that must still be present after compaction and the
strings expected to be gone. The key is the whole point — without a known answer
a compaction result is just a number.

Generation is seeded, so the same seed produces the same traces and a change in
results can only come from a change in the code.

Run from the repo root:

    python scripts/make_traces.py            # 20 traces into data/traces/
    python scripts/make_traces.py --mark     # also insert [[NOISE]] / [[KEEPME]]
    python scripts/make_traces.py --seed 99 --out data/other

``--mark`` exists so the offline fake model can score a trace. The fake backend
answers from the markers alone, which makes a full evaluation possible with no
network and no Laya. It is a testing aid, not part of the traces' realism, so a
marked trace is not a realistic one.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OUT = Path("data/traces")
DEFAULT_SEED = 20261005

#: Marker the fake model reads as "this may go".
NOISE_MARK = "[[NOISE]]"
#: Marker the fake model reads as "this must stay".
KEEP_MARK = "[[KEEPME]]"

#: (archetype, how many traces) — 20 traces in total.
MIX: tuple[tuple[str, int], ...] = (("coding", 8), ("support", 6), ("research", 6))

#: Size range for one tool output, in characters. The low end is a modest read,
#: the high end is the filing that forced split-first in the first place.
MIN_CHARS = 3_000
MAX_CHARS = 20_000


# --------------------------------------------------------------------------- #
# Filler
# --------------------------------------------------------------------------- #

_LOG_LEVELS = ("INFO", "DEBUG", "WARN")
_MODULES = (
    "billing.pricing",
    "billing.invoice",
    "ledger.writer",
    "http.client",
    "cache.store",
    "auth.session",
    "search.index",
    "pdf.extract",
)


def log_lines(rng: random.Random, count: int) -> str:
    """Return plausible log output."""
    out = []
    for _ in range(count):
        out.append(
            f"2026-10-05T{rng.randrange(24):02d}:{rng.randrange(60):02d}:"
            f"{rng.randrange(60):02d}Z {_LOG_LEVELS[rng.randrange(3)]} "
            f"{_MODULES[rng.randrange(len(_MODULES))]}: "
            f"request {rng.randrange(10_000)} completed in {rng.randrange(900)}ms"
        )
    return "\n".join(out) + "\n"


def code_dump(rng: random.Random, lines: int) -> str:
    """Return a plausible source file."""
    body = [
        "import logging",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "",
        "def apply_discount(total, rate):",
        '    """Apply a percentage discount to a total."""',
        "    return round(total * (1 - rate), 2)",
        "",
        "",
        "def apply_tax(total, rate):",
        '    """Apply tax and return the new total."""',
        "    return round(total * (1 + rate), 2)",
        "",
    ]
    for index in range(lines):
        body.append(f"def helper_{index}(value): return value + {index}")
    return "\n".join(body) + "\n"


def json_array(rng: random.Random, count: int) -> str:
    """Return a JSON array of records, one per line-ish, as tools often emit."""
    rows = [
        {
            "invoice_id": f"INV-{rng.randrange(10_000):05d}",
            "customer": f"cust_{rng.randrange(1000):04d}",
            "total_cents": rng.randrange(100, 900_000),
            "currency": "USD",
            "status": ("paid", "open", "refunded")[rng.randrange(3)],
        }
        for _ in range(count)
    ]
    return json.dumps(rows, indent=2)


def boilerplate(rng: random.Random, paragraphs: int) -> str:
    """Return the kind of text nobody reads but everybody pays for."""
    templates = (
        "This section describes the internal process used by the team and is "
        "maintained for reference. It does not change between releases.",
        "The following overview is provided for context. Historical versions of "
        "this document are retained in the archive for audit purposes only.",
        "Reviewers should note that the figures below are illustrative and must "
        "not be quoted externally without written approval from the owner.",
        "Please direct any questions about this material to the service desk. "
        "Support is available during business hours in the primary region.",
    )
    return "\n\n".join(templates[index % len(templates)] for index in range(paragraphs))


def filler(rng: random.Random, kind: str) -> str:
    """Return one block of bulk sized inside the realistic range."""
    if kind == "logs":
        return log_lines(rng, rng.randrange(60, 200))
    if kind == "code":
        return code_dump(rng, rng.randrange(40, 160))
    if kind == "json":
        return json_array(rng, rng.randrange(12, 40))
    return boilerplate(rng, rng.randrange(4, 12))


def bulk(rng: random.Random, kinds: list[str]) -> str:
    """Return several filler blocks joined, padded to the realistic size range."""
    parts = [filler(rng, kind) for kind in kinds]
    text = "\n\n".join(parts)
    target = rng.randrange(MIN_CHARS, MAX_CHARS)
    if len(text) < target:
        # Pad with a repeating but non-uniform block so the text still resists
        # naive token compression, the way real output does.
        pad = log_lines(rng, 40)
        while len(text) < target:
            text += "\n" + pad
    return text


# --------------------------------------------------------------------------- #
# Planted cases
# --------------------------------------------------------------------------- #


@dataclass
class Planted:
    """One tool output with a known answer attached."""

    #: Tool that produced it.
    tool_name: str
    tool_args: dict
    #: The output itself, markers included when ``--mark`` is on.
    content: str
    #: What this case is testing.
    kind: str
    #: Strings that must still be present after compaction.
    must_survive: list[str] = field(default_factory=list)
    #: Strings expected to be gone afterwards. Being present is not a failure.
    expected_gone: list[str] = field(default_factory=list)


def mark(text: str, mark_it: bool, *, noise: bool) -> str:
    """Mark every paragraph of ``text`` for the offline fake model.

    Marking only the whole item is not enough, and the reason is worth stating
    because it cost a whole debugging session once. An item over
    ``CC_ITEM_MAX_CHARS`` is **split first** into pieces and each piece is judged
    on its own, so a single marker at the top of a 16000-character output marks
    only piece 1. The remaining pieces fall back to the fake backend's hash,
    which drops some of them, and a planted fact sitting in one of those pieces
    is destroyed. The offline model can only answer per piece, so the marker has
    to be per piece too.
    """
    if not mark_it:
        return text
    tag = NOISE_MARK if noise else KEEP_MARK
    return "\n\n".join(f"{tag} {part.strip()}" for part in text.split("\n\n") if part.strip())


#: The tools each archetype actually uses. Without this every trace is built from
#: ``read_file`` and ``run_tests``, Scout scores them all as coding, and the
#: archetype column in the evaluation means nothing. The names below match Scout's
#: own keyword table, so the detected profile is the declared one.
ARCHETYPE_TOOLS: dict[str, dict[str, str]] = {
    "coding": {
        "dump": "read_file",
        "search": "grep",
        "verify": "run_tests",
        "shell": "run_in_terminal",
        "path": "src/{arch}/module_{n}.py",
    },
    "support": {
        "dump": "kb_search",
        "search": "crm_lookup",
        "verify": "run_in_terminal",
        "shell": "run_in_terminal",
        "path": "kb/article_{n}",
    },
    "research": {
        "dump": "read_document",
        "search": "search",
        "verify": "run_in_terminal",
        "shell": "run_in_terminal",
        "path": "filings/{arch}_{n}.pdf",
    },
}

#: The finding a used-up dump yields, per archetype. It has to read like the
#: domain or the repeated line looks absurd in the transcript.
ARCHETYPE_FINDINGS: dict[str, str] = {
    "coding": "root cause: round_price uses int() so a price of 10.99 truncates to 10",
    "support": (
        "root cause: the refund was issued against the original charge, not the "
        "replacement, so the customer was debited twice"
    ),
    "research": (
        "root cause: supplier 4 holds 38% of regional volume, above the 25% "
        "disclosure threshold in the filing"
    ),
}


def used_up_dump(rng: random.Random, arch: str, mark_it: bool) -> tuple[Planted, str]:
    """A large output whose one useful line a later message repeats.

    The finding is what makes it droppable: the value has already been extracted,
    so the bulk that produced it earns nothing.
    """
    finding = ARCHETYPE_FINDINGS[arch]
    tools = ARCHETYPE_TOOLS[arch]
    filler = ["logs", "code"] if arch == "coding" else ["logs", "boilerplate"]
    content = bulk(rng, filler) + f"\n{finding}\n"
    return (
        Planted(
            tool_name=tools["dump"],
            tool_args={"path": tools["path"].format(arch=arch, n=rng.randrange(100))},
            content=mark(content, mark_it, noise=True),
            kind="used_up",
            expected_gone=[],
        ),
        f"Found it: {finding}.",
    )


def trap(rng: random.Random, arch: str, mark_it: bool) -> Planted:
    """A large output holding a fact nothing else mentions.

    This is the case that must not be dropped. The whole evaluation is worth
    having because this one can be measured.
    """
    facts = {
        "coding": (
            f"CONFIG NOTE {rng.randrange(10000)}: the rounding mode is set in "
            "billing.toml and must not be changed by the migration"
        ),
        "support": (
            f"CUSTOMER NOTE {rng.randrange(10000)}: account must stay open, do "
            "not close the dispute"
        ),
        "research": (
            f"FILING NOTE {rng.randrange(10000)}: supplier 7 restated segment "
            "revenue in the notes to the accounts"
        ),
    }
    fact = facts[arch]
    tools = ARCHETYPE_TOOLS[arch]
    content = bulk(rng, ["json", "boilerplate"]) + f"\n{fact}\n"
    return Planted(
        tool_name=tools["search"],
        tool_args={"query": f"{arch} record {rng.randrange(10000)}"},
        content=mark(content, mark_it, noise=False),
        kind="trap",
        must_survive=[fact],
    )


def superseded_run(rng: random.Random, arch: str, mark_it: bool) -> tuple[Planted, Planted]:
    """A failing check and the passing run that makes it obsolete."""
    tools = ARCHETYPE_TOOLS[arch]
    failing = {
        "coding": "\nFAILED tests/test_price.py::test_rounding - assert 10 == 11\n",
        "support": "\nFAILED check_ticket.py::test_dispute_open - ticket already closed\n",
        "research": "\nFAILED validate_citations.py::test_refs - reference [3] not found\n",
    }
    passing = {
        "coding": "\n147 passed in 4.05s\n",
        "support": "\n12 checks passed in 1.90s\n",
        "research": "\n48 citations validated in 3.40s\n",
    }
    failure = bulk(rng, ["logs"]) + failing[arch]
    success = log_lines(rng, 30) + passing[arch]
    return (
        Planted(
            tool_name=tools["verify"],
            tool_args={"target": f"{arch}_check_1"},
            content=mark(failure, mark_it, noise=True),
            kind="superseded",
        ),
        Planted(
            tool_name=tools["verify"],
            tool_args={"target": f"{arch}_check_full"},
            content=mark(success, mark_it, noise=False),
            kind="final",
            must_survive=[passing[arch].strip()],
        ),
    )


def noise_block(rng: random.Random, arch: str, mark_it: bool) -> Planted:
    """An install log: pure process, no information worth keeping."""
    commands = {
        "coding": ("run_in_terminal", {"command": "pip install -r requirements.txt"},
                   "Collecting pytest\nDownloading pluggy-1.5.0\n"),
        "support": ("run_in_terminal", {"command": "pip install -r support-requirements.txt"},
                    "Collecting httpx\nDownloading h11-0.14.0\n"),
        "research": ("run_in_terminal", {"command": "pip install -r research-requirements.txt"},
                     "Collecting pdfplumber\nDownloading pdfminer.six-20231228\n"),
    }
    tool_name, args, head = commands[arch]
    content = head + f"Successfully installed in {rng.randrange(2, 9)}s\n" * 40
    content += log_lines(rng, 60)
    return Planted(
        tool_name=tool_name,
        tool_args=args,
        content=mark(content, mark_it, noise=True),
        kind="noise",
    )


def risky_tool(rng: random.Random, mark_it: bool) -> Planted:
    """A side-effecting call whose output is the only record it happened."""
    if rng.random() < 0.5:
        receipt = f"refund_issued receipt_id=RFD-{rng.randrange(100000):05d} amount=42.50"
        name, args = "refund_issued", {"order_id": f"ORD-{rng.randrange(10000):05d}"}
    else:
        receipt = f"send_email queued id=MSG-{rng.randrange(100000):05d} to=customer@example.com"
        name, args = "send_email", {"to": "customer@example.com"}
    content = json.dumps(
        {"status": "ok", "receipt": receipt, "timestamp": "2026-10-05T09:15:00Z"}, indent=2
    )
    return Planted(
        tool_name=name,
        tool_args=args,
        content=mark(content, mark_it, noise=False),
        kind="risky",
        must_survive=[receipt],
    )


# --------------------------------------------------------------------------- #
# Trace assembly
# --------------------------------------------------------------------------- #


GOALS = {
    "coding": "Fix the rounding bug in the billing service and make the tests pass.",
    "support": "Resolve the double-charge dispute for customer {n} and apply the credit.",
    "research": "Summarise the supplier concentration risk for the {s} filing.",
}

SYSTEM = {
    "coding": "You are a coding assistant. Read before you edit, and always run the tests.",
    "support": "You are a support agent. Resolve the ticket and record what you did.",
    "research": "You are a research assistant. Cite the source of every claim.",
}


@dataclass
class Trace:
    """One synthetic history plus the answer key that scores it."""

    name: str
    archetype: str
    goal: str
    messages: list[dict]
    key: dict

    def to_json(self) -> str:
        """Return the history as the JSON written to disk."""
        return json.dumps({"messages": self.messages}, indent=2)

    def key_json(self) -> str:
        """Return the answer key as the JSON written beside it."""
        return json.dumps(self.key, indent=2)


def build_trace(rng: random.Random, index: int, archetype: str, mark_it: bool) -> Trace:
    """Build one trace, planting every kind of case at least once."""
    goal = GOALS[archetype].format(n=1000 + rng.randrange(9000), s="Q3")
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM[archetype]},
        {"role": "user", "content": goal},
        {"role": "assistant", "content": "Plan: gather context, act, then verify."},
    ]

    must_survive: list[str] = []
    expected_gone: list[str] = []
    planted_kinds: list[str] = []
    tool_count = rng.randrange(4, 9)

    for position in range(tool_count):
        # Rotate through the cases so every trace has a spread of them, then
        # fill any remaining slots with noise.
        choice = ["trap", "used_up", "noise", "risky"][position % 4]
        if position == tool_count - 1:
            choice = "superseded"

        if choice == "trap":
            planted = trap(rng, archetype, mark_it)
            messages.append(
                {"role": "assistant", "content": f"Looking into that ({position})."}
            )
            messages.append(_as_tool(planted, position))
        elif choice == "used_up":
            planted, later = used_up_dump(rng, archetype, mark_it)
            messages.append(_as_tool(planted, position))
            # The repeat is what makes it droppable, so it must come after.
            messages.append({"role": "assistant", "content": later})
        elif choice == "superseded":
            first, second = superseded_run(rng, archetype, mark_it)
            messages.append(_as_tool(first, position))
            messages.append(_as_tool(second, position + 1))
        else:
            planted = (
                noise_block(rng, archetype, mark_it)
                if position % 2
                else risky_tool(rng, mark_it)
            )
            messages.append(_as_tool(planted, position))

        if planted.kind not in ("final",):
            planted_kinds.append(planted.kind)
        must_survive.extend(planted.must_survive)
        expected_gone.extend(planted.expected_gone)

    # Every trace ends with assistant turns so the last tool message sits inside
    # the profile's protected window and the earlier ones are actually eligible.
    for extra in range(3):
        messages.append({"role": "assistant", "content": f"Follow-up step {extra} done."})

    messages.append({"role": "assistant", "content": "All done. Summary follows."})

    key = {
        "archetype": archetype,
        "goal": goal,
        "must_survive": sorted(set(must_survive)),
        "expected_gone": sorted(set(expected_gone)),
        "planted_kinds": sorted(set(planted_kinds)),
        "tool_messages": sum(1 for m in messages if m.get("role") == "tool"),
    }
    return Trace(
        name=f"{archetype}_{index:02d}",
        archetype=archetype,
        goal=goal,
        messages=messages,
        key=key,
    )


def _as_tool(planted: Planted, position: int) -> dict:
    """Return one tool message in the shape an agent framework would send."""
    return {
        "role": "tool",
        "name": planted.tool_name,
        "tool_call_id": f"call_{position:02d}",
        "tool_args": planted.tool_args,
        "content": planted.content,
    }


def generate(seed: int, mark_it: bool) -> list[Trace]:
    """Return the whole trace set, deterministically from ``seed``."""
    traces: list[Trace] = []
    for archetype, count in MIX:
        for index in range(1, count + 1):
            # One RNG per trace, seeded from the master seed and the name, so a
            # trace is unchanged when other traces are added or removed.
            rng = random.Random(f"{seed}:{archetype}:{index}")
            traces.append(build_trace(rng, index, archetype, mark_it))
    return traces


def main() -> None:
    """Write the traces and their answer keys."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--mark",
        action="store_true",
        help="insert [[NOISE]] / [[KEEPME]] so the offline fake model can score this",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    traces = generate(args.seed, args.mark)

    marked = " (marked)" if args.mark else ""
    print(f"writing {len(traces)} traces{marked} to {args.out}")
    for trace in traces:
        (args.out / f"{trace.name}.json").write_text(trace.to_json(), encoding="utf-8")
        (args.out / f"{trace.name}.key.json").write_text(trace.key_json(), encoding="utf-8")

    per_arch: dict[str, int] = {}
    for trace in traces:
        per_arch[trace.archetype] = per_arch.get(trace.archetype, 0) + 1

    print()
    for archetype, count in MIX:
        print(f"  {archetype:<10} {count:>2} traces")
    print()
    for trace in traces[:3]:
        sizes = [len(m.get("content", "")) for m in trace.messages if m.get("role") == "tool"]
        print(
            f"  {trace.name}: {len(trace.messages)} messages, "
            f"{len(sizes)} tool outputs {min(sizes)}-{max(sizes)} chars, "
            f"{len(trace.key['must_survive'])} facts that must survive"
        )
    print(f"\n  archetypes: {per_arch}")
    print("  answer keys written alongside as <name>.key.json")


if __name__ == "__main__":
    main()