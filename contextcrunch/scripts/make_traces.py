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
paragraphs, sized between roughly 1,500 and 15,000 characters, because that is
the range the real limits are set for.

Each trace is written twice: the history itself, and an answer key beside it
naming the exact strings that must still be present after compaction and the
strings expected to be gone. The key is the whole point — without a known answer
a compaction result is just a number.

Generation is seeded, so the same seed produces the same traces and a change in
results can only come from a change in the code.

Run from the repo root:

    python scripts/make_traces.py            # 30 traces into data/traces/
    python scripts/make_traces.py --seed 99 --out data/other
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OUT = Path("data/traces")
DEFAULT_SEED = 20261005

#: (archetype, how many traces) — 30 traces in total (70% train, 30% test).
MIX: tuple[tuple[str, int], ...] = (("coding", 12), ("support", 9), ("research", 9))

#: Size range for one tool output, in characters.
MIN_CHARS = 1_500
MAX_CHARS = 15_000


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
    #: The output itself.
    content: str
    #: What this case is testing.
    kind: str
    #: Strings that must still be present after compaction.
    must_survive: list[str] = field(default_factory=list)
    #: Strings expected to be gone afterwards. Being present is not a failure.
    expected_gone: list[str] = field(default_factory=list)
    #: Character spans of planted facts for piece-level labeling.
    fact_spans: list[tuple[int, int]] = field(default_factory=list)


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
    "coding": "root cause: round_price uses int() so it truncates instead of rounding",
    "support": (
        "root cause: the refund was issued against the original charge, not the "
        "replacement, so the customer was debited twice"
    ),
    "research": (
        "root cause: supplier 4 holds 38% of regional volume, above the 25% "
        "disclosure threshold in the filing"
    ),
}


def used_up_dump(rng: random.Random, arch: str) -> tuple[Planted, str]:
    """A large output whose one useful line a later message repeats.

    The finding is what makes it droppable: the value has already been extracted,
    so the bulk that produced it earns nothing.
    """
    finding = ARCHETYPE_FINDINGS[arch]
    tools = ARCHETYPE_TOOLS[arch]
    filler = ["logs", "code"] if arch == "coding" else ["logs", "boilerplate"]
    content = bulk(rng, filler) + f"\n{finding}\n"
    # Record the span of the finding for piece-level labeling
    fact_start = len(content) - len(finding) - 1
    fact_end = len(content) - 1
    return (
        Planted(
            tool_name=tools["dump"],
            tool_args={"path": tools["path"].format(arch=arch, n=rng.randrange(100))},
            content=content,
            kind="used_up",
            expected_gone=[],
            fact_spans=[(fact_start, fact_end)],
        ),
        f"Found it: {finding}.",
    )


def trap(rng: random.Random, arch: str) -> Planted:
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
    fact_start = len(content) - len(fact) - 1
    fact_end = len(content) - 1
    return Planted(
        tool_name=tools["search"],
        tool_args={"query": f"{arch} record {rng.randrange(10000)}"},
        content=content,
        kind="trap",
        must_survive=[fact],
        fact_spans=[(fact_start, fact_end)],
    )


def superseded_run(rng: random.Random, arch: str) -> tuple[Planted, Planted]:
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
            content=failure,
            kind="superseded",
        ),
        Planted(
            tool_name=tools["verify"],
            tool_args={"target": f"{arch}_check_full"},
            content=success,
            kind="final",
            must_survive=[passing[arch].strip()],
        ),
    )


def noise_block(rng: random.Random, arch: str) -> Planted:
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
        content=content,
        kind="noise",
    )


def risky_tool(rng: random.Random) -> Planted:
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
        content=content,
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


def build_trace(rng: random.Random, index: int, archetype: str) -> Trace:
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
    tool_kinds: list[str] = []  # per-message kind for the key file
    tool_count = rng.randrange(6, 13)  # 6-12 tool messages

    for position in range(tool_count):
        # Rotate through the cases so every trace has a spread of them, then
        # fill any remaining slots with noise.
        choice = ["trap", "used_up", "noise", "risky"][position % 4]
        if position == tool_count - 1:
            choice = "superseded"

        if choice == "trap":
            planted = trap(rng, archetype)
            messages.append(
                {"role": "assistant", "content": f"Looking into that ({position})."}
            )
            messages.append(_as_tool(planted, position))
            tool_kinds.append(planted.kind)
        elif choice == "used_up":
            planted, later = used_up_dump(rng, archetype)
            messages.append(_as_tool(planted, position))
            # The repeat is what makes it droppable, so it must come after.
            messages.append({"role": "assistant", "content": later})
            tool_kinds.append(planted.kind)
        elif choice == "superseded":
            first, second = superseded_run(rng, archetype)
            messages.append(_as_tool(first, position))
            messages.append(_as_tool(second, position + 1))
            tool_kinds.append(first.kind)
            tool_kinds.append(second.kind)
        else:
            planted = (
                noise_block(rng, archetype)
                if position % 2
                else risky_tool(rng)
            )
            messages.append(_as_tool(planted, position))
            tool_kinds.append(planted.kind)

        if planted.kind not in ("final",):
            planted_kinds.append(planted.kind)
        must_survive.extend(planted.must_survive)
        expected_gone.extend(planted.expected_gone)

    # Every trace ends with assistant turns so the last tool message sits inside
    # the profile's protected window and the earlier ones are actually eligible.
    for extra in range(3):
        messages.append({"role": "assistant", "content": f"Follow-up step {extra} done."})

    messages.append({"role": "assistant", "content": "All done. Summary follows."})

    # Assign split: 70% train, 30% test (by whole trace, stratified by type)
    split = "train" if rng.random() < 0.7 else "test"

    key = {
        "archetype": archetype,
        "goal": goal,
        "split": split,
        "must_survive": sorted(set(must_survive)),
        "expected_gone": sorted(set(expected_gone)),
        "planted_kinds": sorted(set(planted_kinds)),
        "tool_kinds": tool_kinds,  # per-message kind
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


def generate(seed: int) -> list[Trace]:
    """Return the whole trace set, deterministically from ``seed``."""
    traces: list[Trace] = []
    for archetype, count in MIX:
        for index in range(1, count + 1):
            # One RNG per trace, seeded from the master seed and the name, so a
            # trace is unchanged when other traces are added or removed.
            rng = random.Random(f"{seed}:{archetype}:{index}")
            traces.append(build_trace(rng, index, archetype))
    return traces


def main() -> None:
    """Write the traces and their answer keys."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    traces = generate(args.seed)

    print(f"writing {len(traces)} traces to {args.out}")
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