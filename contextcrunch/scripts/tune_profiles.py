"""Measure the profiles against human-labelled cases.

Run from the repo root:

    .venv\\Scripts\\python.exe scripts\\tune_profiles.py              # all archetypes
    .venv\\Scripts\\python.exe scripts\\tune_profiles.py --archetype coding
    .venv\\Scripts\\python.exe scripts\\tune_profiles.py --search     # propose thresholds

Answers are cached in ``data/tune_cache.json`` so re-running is free and the same
answers are always compared against the same thresholds.

The tuner reports, per archetype:

- **dangerous drops**: irreplaceable material the policy deleted. Must stay zero.
- **recovered**: junk tokens removed, counting both whole drops and truncations.
- **separability**: how well each Laya signal alone tells junk from must-keep.

The separability table is the part worth reading. A signal whose junk and keep
ranges overlap cannot carry a threshold, however the threshold is tuned.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

# ``scripts`` is a plain directory, not a package, so ``from scripts.evalset import
# ...`` only resolves when the repo root is importable. Running this file as
# ``python scripts\tune_profiles.py`` puts ``scripts`` on the path instead, so the
# root is added here rather than left as a step the caller has to remember.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextcrunch.core.policy import (
    DROP as ACTION_DROP,
    KEEP,
    PROFILES,
    TRUNCATE,
    Profile,
    decide,
)
from contextcrunch.core.questions import QUESTION_TYPES
from contextcrunch.core.tokens import count_tokens
from contextcrunch.laya.types import parse_result
from scripts.evalset import DROP, Case, cases_for

CACHE = Path("data/tune_cache.json")

#: Which built-in profile each labelled archetype is compared against.
ARCHETYPE_PROFILE = {
    "coding": "aggressive",
    "support": "balanced",
    "research": "conservative",
}

#: Grid for --search. Wider than the range the shipped profiles use on purpose:
#: the point is to show what the data allows before deciding what to accept.
GRID = {
    "essential_ceiling": [0.35, 0.45, 0.50, 0.55, 0.60, 0.70],
    "relevance_ceiling": [1.2, 1.4, 1.55, 1.7, 1.9, 2.1, 2.4],
    "superseded_floor": [0.0, 0.15, 0.25, 0.35, 0.45, 0.60],
    "confidence_floor": [0.0, 0.20, 0.35, 0.45, 0.50, 0.60],
}


@dataclass
class Judgement:
    """Laya's answers for one case, plus what the policy engine did with them."""

    case: Case
    answers: object
    action: str = ""


def state_for(case: Case) -> dict:
    """Build the Laya state for one case, exactly as production would."""
    return {
        "goal": case.goal,
        "next_step": case.later_findings[-1] if case.later_findings else "",
        "item": {
            "index": 0,
            "role": "tool",
            "tool_call": case.tool,
            "content": case.content,
            "tokens": count_tokens(case.content),
            "age_in_turns": case.age,
        },
        "later_findings": list(case.later_findings),
    }


async def judge(cases, *, offline: bool) -> list[Judgement]:
    """Get Laya's answers for every case, reusing the cache when possible.

    A case the API cannot judge is skipped and reported, never fatal: one flaky
    call must not throw away an entire measurement run.
    """
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    out: list[Judgement] = []
    failures: list[str] = []

    client = None
    if not offline:
        from contextcrunch.laya.client import build_client

        client = build_client()
    try:
        for case in cases:
            key = f"{case.name}|{case.content[:64]}"
            if key not in cache:
                if offline:
                    cache[key] = None
                else:
                    try:
                        result = await client.predict(state_for(case), case.name)
                        cache[key] = result.model_dump(mode="json")
                    except Exception as exc:  # noqa: BLE001 - report and move on
                        failures.append(f"{case.name}: {type(exc).__name__}: {exc}")
                        continue
            payload = cache[key]
            answers = (
                _fake_answers(case)
                if payload is None
                else parse_result(payload, dict(QUESTION_TYPES))
            )
            out.append(Judgement(case=case, answers=answers))
    finally:
        if client is not None:
            await client.aclose()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=2))

    for failure in failures:
        print(f"  skipped {failure}", file=sys.stderr)
    return out


def score(judgements: list[Judgement], profile: Profile) -> dict:
    """Score one profile against the labelled cases."""
    dangerous = matched = saved = missed = truncated_saved = total = 0

    for item in judgements:
        action = decide(
            {
                "role": "tool",
                "tokens": count_tokens(item.case.content),
                "age_in_turns": item.case.age,
                "tool_name": item.case.tool,
            },
            item.answers.answers,
            profile,
        )
        item.action = action
        before = count_tokens(item.case.content)
        total += before
        if action == ACTION_DROP:
            if item.case.label == DROP:
                matched += 1
                saved += before
            else:
                dangerous += 1
        elif item.case.label == DROP:
            missed += before
        elif action == item.case.label.upper():
            matched += 1
        if action == TRUNCATE:
            truncated_saved += _truncation_saving(item.case.content)

    recovered = saved + truncated_saved
    return {
        "matched": matched,
        "dangerous": dangerous,
        "junk": sum(1 for i in judgements if i.case.label == DROP),
        "missed": missed,
        "saved": saved,
        "trunc": truncated_saved,
        "recovered": recovered,
        "total": total,
        "pct": round(100.0 * recovered / total, 1) if total else 0.0,
    }


def _truncation_saving(content: str) -> int:
    """Estimate the tokens a truncation would remove from one item."""
    from contextcrunch.core.settings import get_settings
    from contextcrunch.core.splitter import split_text

    settings = get_settings()
    pieces = split_text(
        content, settings.piece_target_chars, settings.piece_max_chars
    )
    if len(pieces) < 2:
        return 0
    size = settings.piece_target_chars
    rebuilt = sum(count_tokens(piece.text[:size] + piece.text[-size:]) for piece in pieces)
    return max(count_tokens(content) - rebuilt, 0)


def diagnose(judgements: list[Judgement]) -> None:
    """Report how well each Laya signal separates junk from must-keep material."""
    junk = [i for i in judgements if i.case.label == DROP]
    keep = [i for i in judgements if i.case.label != DROP]
    if not junk or not keep:
        print("\n(need both junk and keep cases to diagnose)")
        return

    def best_split(junk_vals, keep_vals, higher_is_junk):
        best = (0.0, None)
        for threshold in sorted(set(junk_vals) | set(keep_vals)):
            hits = sum(
                (v > threshold) if higher_is_junk else (v < threshold) for v in junk_vals
            )
            hits += sum(
                not ((v > threshold) if higher_is_junk else (v < threshold)) for v in keep_vals
            )
            rate = hits / (len(junk_vals) + len(keep_vals))
            if rate > best[0]:
                best = (rate, threshold)
        return best

    signals = [
        ("confidence", lambda i: i.answers.answer("verdict").answer_confidence, True),
        ("essential", lambda i: i.answers.noul("essential"), False),
        ("consumed", lambda i: i.answers.noul("consumed"), True),
        ("superseded", lambda i: i.answers.noul("superseded"), True),
        ("relevance", lambda i: i.answers.answer("relevance").score, False),
    ]
    print(f"\nseparability over {len(junk)} junk / {len(keep)} must-keep")
    print(f"  {'signal':<12} {'junk':<12} {'keep':<12} {'split':<7} threshold")
    for name, get, higher_is_junk in signals:
        junk_vals = [get(i) for i in junk if get(i) is not None]
        keep_vals = [get(i) for i in keep if get(i) is not None]
        if not junk_vals or not keep_vals:
            continue
        rate, threshold = best_split(junk_vals, keep_vals, higher_is_junk)
        print(
            f"  {name:<12} {min(junk_vals):.2f}-{max(junk_vals):.2f}".ljust(26)
            + f"{min(keep_vals):.2f}-{max(keep_vals):.2f}".ljust(14)
            + f"{rate:<7.0%} {threshold:.2f}"
        )

    drops = sum(1 for i in judgements if i.answers.answer("verdict").choice == "drop")
    print(f"  {'verdict':<12} {drops} of {len(judgements)} cases say drop")


def search(judgements: list[Judgement], base: Profile) -> Profile:
    """Find the most permissive profile that still causes no dangerous drop.

    Reported, not applied: with 15 labelled cases a maximum-savings row is
    overfitted, and a wrong drop costs evidence while a missed drop costs only
    tokens. Read the result as the outer bound of what the data allows.
    """
    best, best_score = base, score(judgements, base)
    for values in itertools.product(*GRID.values()):
        candidate = replace(base, **dict(zip(GRID, values, strict=True)))
        result = score(judgements, candidate)
        if result["dangerous"] > 0:
            continue
        if result["recovered"] > best_score["recovered"]:
            best, best_score = candidate, result
    return best


def _fake_answers(case: Case):
    """Offline stand-in: junk scores as junk, everything else as keep."""
    junk = case.label == DROP
    return parse_result(
        {
            "answers": {
                "verdict": {
                    "type": "choice",
                    "choice": "drop" if junk else "keep",
                    "probabilities": {"keep": 0.1, "truncate": 0.1, "drop": 0.8}
                    if junk
                    else {"keep": 0.7, "truncate": 0.2, "drop": 0.1},
                },
                "essential": {"type": "noul", "noul": 0.1 if junk else 0.8},
                "consumed": {"type": "noul", "noul": 0.9 if junk else 0.1},
                "superseded": {"type": "noul", "noul": 0.9 if junk else 0.1},
                "relevance": {
                    "type": "score",
                    "score": 0.2 if junk else 2.5,
                    "probabilities": {"0": 0.8, "1": 0.1, "2": 0.1, "3": 0.0}
                    if junk
                    else {"0": 0.0, "1": 0.0, "2": 0.4, "3": 0.6},
                },
            }
        },
        dict(QUESTION_TYPES),
    )


def report(archetype: str, judgements: list[Judgement], *, do_search: bool) -> None:
    """Print the full report for one archetype."""
    print(f"\n{'=' * 66}\n{archetype}: {len(judgements)} labelled cases\n{'=' * 66}")

    print("\nLaya's answers:")
    for item in judgements:
        verdict = item.answers.answer("verdict")
        relevance = item.answers.answer("relevance")
        print(
            f"  {item.case.name[:38]:<38} {item.case.label:<9}"
            f" {verdict.choice:<9} conf={verdict.answer_confidence:.2f}"
            f" ess={item.answers.noul('essential'):.2f}"
            f" con={item.answers.noul('consumed'):.2f}"
            f" sup={item.answers.noul('superseded'):.2f}"
            f" rel={relevance.score:.2f}"
        )

    print("\nprofiles (shipped):")
    for name, profile in PROFILES.items():
        result = score(judgements, profile)
        print(
            f"  {name:<13} dangerous={result['dangerous']:<3}"
            f" recovered={result['recovered']:<6} ({result['pct']:>5}%)"
            f" matched={result['matched']}/{len(judgements)}"
        )

    diagnose(judgements)

    if do_search:
        base = PROFILES[ARCHETYPE_PROFILE[archetype]]
        tuned = search(judgements, base)
        result = score(judgements, tuned)
        print(f"\nmost permissive row the data allows (NOT shipped):\n  {tuned}")
        print(f"  dangerous={result['dangerous']} recovered={result['recovered']} ({result['pct']}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archetype", choices=["coding", "support", "research", "all"],
                        default="all")
    parser.add_argument("--offline", action="store_true", help="use the fake backend")
    parser.add_argument("--search", action="store_true", help="also report the loosest row")
    args = parser.parse_args()

    archetypes = ["coding", "support", "research"] if args.archetype == "all" else [args.archetype]
    for archetype in archetypes:
        report(archetype, asyncio.run(judge(cases_for(archetype), offline=args.offline)),
               do_search=args.search)

    if not args.offline:
        print(f"\nanswers cached in {CACHE}")


if __name__ == "__main__":
    main()