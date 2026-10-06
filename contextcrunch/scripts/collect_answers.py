"""Collect real Laya answers for all traces and cache them for replay.

This script runs the real Laya model (inprocess or http) on every tool message
of the selected traces and stores the answers in a JSONL cache. The cache can
then be replayed anywhere without calling the model again.

Run from the repo root:

    python scripts/collect_answers.py --split all
    python scripts/collect_answers.py --split train --questions-variant v2
    python scripts/collect_answers.py --limit-traces 5 --split test

The cache is append-only and resumable: keys already present are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextcrunch.core.banner import print_banner  # noqa: E402
from contextcrunch.core.messages import as_messages  # noqa: E402
from contextcrunch.core.settings import get_settings  # noqa: E402
from contextcrunch.core.splitter import split_text  # noqa: E402
from contextcrunch.core.state import build_state, fit_state  # noqa: E402
from contextcrunch.laya.cache import LayaCache, create_cache, cache_key  # noqa: E402
from contextcrunch.laya.client import build_engine  # noqa: E402
from contextcrunch.core.questions import QUESTION_SET, QUESTION_VARIANTS, get_question_set  # noqa: E402


TRACES = Path("data/traces")


def load_traces(split: str) -> list[tuple[Path, dict, dict]]:
    """Return (trace path, messages, key) for traces matching the split."""
    if not TRACES.is_dir():
        raise SystemExit(f"no traces in {TRACES}. Run: python scripts/make_traces.py")

    found = []
    for path in sorted(TRACES.glob("*.json")):
        if path.name.endswith(".key.json"):
            continue
        key_path = path.with_suffix(".key.json")
        if not key_path.is_file():
            continue
        key = json.loads(key_path.read_text(encoding="utf-8"))
        if split != "all" and key.get("split") != split:
            continue
        found.append((path, json.loads(path.read_text(encoding="utf-8")), key))
    return found


async def collect_for_trace(
    path: Path,
    raw: dict,
    key: dict,
    engine,
    cache: LayaCache,
    settings,
    question_variant: str,
) -> tuple[int, int]:
    """Collect answers for one trace. Returns (new_entries, skipped_entries)."""
    messages = as_messages(raw["messages"])
    tool_messages = [m for m in messages if m.role == "tool"]
    tool_kinds = key.get("tool_kinds", [])

    # Get the question set for this variant
    question_set = dict(get_question_set(question_variant))

    new_entries = 0
    skipped_entries = 0

    for idx, msg in enumerate(tool_messages):
        kind = tool_kinds[idx] if idx < len(tool_kinds) else "unknown"

        # (a) Judge the whole item if it fits
        if len(msg.content) <= settings.item_max_chars:
            state = build_state(messages, idx, key["goal"], settings)
            state = fit_state(state, settings)
            state_key = cache_key(state, question_set)

            if not cache.has(state_key):
                start = time.perf_counter()
                results = await engine.decide([state], [f"{path.stem}:item:{idx}"])
                elapsed = time.perf_counter() - start

                cache.add(
                    key=state_key,
                    answers=results[0],
                    raw=results[0].model_dump(mode="json"),
                    seconds=elapsed,
                    meta={
                        "trace": path.stem,
                        "index": idx,
                        "piece": "whole",
                        "kind": kind,
                    },
                )
                new_entries += 1
            else:
                skipped_entries += 1

        # (b) ALWAYS split and judge every piece
        pieces = split_text(msg.content, settings.piece_target_chars, settings.piece_max_chars)
        if len(pieces) > 1:
            for piece_idx, piece in enumerate(pieces):
                piece_state = build_state(messages, idx, key["goal"], settings)
                piece_state["item"] = {
                    "index": idx,
                    "role": "tool",
                    "tool_call": msg.name,
                    "content": piece.text,
                    "tokens": 0,  # will be filled by fit_state
                    "age_in_turns": 0,  # pieces are judged independently
                }
                piece_state = fit_state(piece_state, settings)
                piece_key = cache_key(piece_state, question_set)

                if not cache.has(piece_key):
                    start = time.perf_counter()
                    results = await engine.decide([piece_state], [f"{path.stem}:item:{idx}:piece:{piece_idx}"])
                    elapsed = time.perf_counter() - start

                    cache.add(
                        key=piece_key,
                        answers=results[0],
                        raw=results[0].model_dump(mode="json"),
                        seconds=elapsed,
                        meta={
                            "trace": path.stem,
                            "index": idx,
                            "piece": piece_idx,
                            "kind": kind,
                        },
                    )
                    new_entries += 1
                else:
                    skipped_entries += 1

    return new_entries, skipped_entries


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="which split to collect answers for",
    )
    parser.add_argument(
        "--limit-traces",
        type=int,
        default=0,
        help="limit number of traces (0 = all)",
    )
    parser.add_argument(
        "--questions-variant",
        choices=["v1", "v2"],
        default="v1",
        help="question variant to use",
    )
    args = parser.parse_args()

    settings = get_settings()
    # Override question variant
    settings = settings.model_copy(update={"question_variant": args.questions_variant})

    engine = build_engine(settings)
    print_banner(settings, engine)

    traces = load_traces(args.split)
    if args.limit_traces:
        traces = traces[:args.limit_traces]

    if not traces:
        raise SystemExit(f"no traces matched split={args.split}")

    print(f"Collecting answers for {len(traces)} traces (split={args.split}, variant={args.questions_variant})")
    print(f"Cache directory: {settings.replay_dir or 'data/replay'}")

    # Create cache
    cache = create_cache(
        replay_dir=settings.replay_dir or "data/replay",
        checkpoint=settings.laya_model,
        laya_version="unknown",  # will be filled by banner
        question_variant=args.questions_variant,
    )

    total_new = 0
    total_skipped = 0
    start_time = time.perf_counter()

    for i, (path, raw, key) in enumerate(traces):
        trace_start = time.perf_counter()
        new, skipped = await collect_for_trace(
            path, raw, key, engine, cache, settings, args.questions_variant
        )
        trace_elapsed = time.perf_counter() - trace_start
        total_new += new
        total_skipped += skipped

        elapsed = time.perf_counter() - start_time
        avg_per_trace = elapsed / (i + 1)
        remaining = len(traces) - i - 1
        eta = avg_per_trace * remaining

        print(
            f"  [{i+1}/{len(traces)}] {path.stem}: "
            f"+{new} new, {skipped} skipped, "
            f"{trace_elapsed:.1f}s (ETA: {eta:.1f}s)"
        )

    total_elapsed = time.perf_counter() - start_time
    print(f"\nDone: {total_new} new entries, {total_skipped} skipped in {total_elapsed:.1f}s")
    print(f"Cache written to: {settings.replay_dir or 'data/replay'}")

    await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())