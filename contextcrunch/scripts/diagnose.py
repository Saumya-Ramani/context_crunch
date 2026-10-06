"""Diagnose Laya's signal quality from cached answers.

This script reads the cached real-Laya answers, joins them with the trace labels,
and prints a detailed diagnostic report showing how well each signal separates
removable content from must-keep content.

Run from the repo root:

    python scripts/diagnose.py --split all
    python scripts/diagnose.py --split train --questions-variant v2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextcrunch.core.banner import print_banner  # noqa: E402
from contextcrunch.core.policy import PROFILES, decide  # noqa: E402
from contextcrunch.core.questions import get_question_set  # noqa: E402
from contextcrunch.core.settings import get_settings  # noqa: E402
from contextcrunch.core.state import build_state, fit_state  # noqa: E402
from contextcrunch.core.messages import as_messages  # noqa: E402
from contextcrunch.core.tokens import count_tokens  # noqa: E402
from contextcrunch.laya.cache import LayaCache, cache_key  # noqa: E402
from contextcrunch.laya.client import build_engine  # noqa: E402
from contextcrunch.laya.types import LayaResult  # noqa: E402


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


def mann_whitney_auc(junk_vals: list[float], keep_vals: list[float], higher_is_junk: bool) -> float:
    """Compute AUC using Mann-Whitney U test (pure Python, no scipy needed)."""
    if not junk_vals or not keep_vals:
        return 0.5

    # Combine and rank
    combined = [(v, 1) for v in junk_vals] + [(v, 0) for v in keep_vals]
    combined.sort(key=lambda x: x[0])

    # Handle ties by assigning average rank
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum ranks for junk class
    junk_ranks = sum(ranks[k] for k, (v, label) in enumerate(combined) if label == 1)
    n1 = len(junk_vals)
    n2 = len(keep_vals)

    # U statistic
    u = junk_ranks - n1 * (n1 + 1) / 2
    auc = u / (n1 * n2)

    # If higher_is_junk is False, flip
    if not higher_is_junk:
        auc = 1 - auc

    return auc


def signal_strength_label(auc: float) -> str:
    """Return a label for AUC strength."""
    if auc >= 0.85:
        return "strong"
    elif auc >= 0.70:
        return "usable"
    else:
        return "weak"


def percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def distribution_stats(values: list[float]) -> dict:
    """Return distribution statistics for a list of values."""
    if not values:
        return {"n": 0, "mean": 0, "p10": 0, "median": 0, "p90": 0}
    return {
        "n": len(values),
        "mean": np.mean(values),
        "p10": percentile(values, 10),
        "median": np.median(values),
        "p90": percentile(values, 90),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="which split to diagnose",
    )
    parser.add_argument(
        "--questions-variant",
        choices=["v1", "v2"],
        default="v1",
        help="question variant to use",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path("data/replay"),
        help="replay cache directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/diagnose_report.md"),
        help="output report file",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings = settings.model_copy(update={
        "replay_dir": str(args.replay_dir),
        "question_variant": args.questions_variant,
    })

    engine = build_engine(settings)
    print_banner(settings, engine)

    # Load cache
    cache = LayaCache(args.replay_dir)
    if len(cache) == 0:
        raise SystemExit(f"Cache is empty at {args.replay_dir}. Run collect_answers.py first.")

    print(f"Loaded {len(cache)} cached answers from {args.replay_dir}")

    # Load traces
    traces = load_traces(args.split)
    if not traces:
        raise SystemExit(f"no traces matched split={args.split}")

    # Collect data per kind
    # Kinds: noise, used_up, superseded, trap, risky, filler_keep
    removable_kinds = {"noise", "used_up", "superseded"}
    must_keep_kinds = {"trap", "risky", "filler_keep"}

    # Per-kind signal distributions
    kind_signals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    # For veto funnel
    veto_counts = defaultdict(int)
    any_veto_counts = defaultdict(int)

    # For danger check
    danger_drops = 0
    danger_total = 0

    question_set = dict(get_question_set(args.questions_variant))

    for path, raw, key in traces:
            messages = as_messages(raw["messages"])
            tool_messages = [m for m in messages if m.role == "tool"]
            tool_kinds = key.get("tool_kinds", [])

            for idx, msg in enumerate(tool_messages):
                kind = tool_kinds[idx] if idx < len(tool_kinds) else "unknown"

                # Check whole item
                if len(msg.content) <= settings.item_max_chars:
                    state = build_state(messages, idx, key["goal"], settings)
                    state = fit_state(state, settings)
                    state_key = cache_key(state, question_set)
                    record = cache.get(state_key)
                    if record:
                        answers = LayaResult.model_validate(record["answers"])
                        # Extract signals
                        verdict_conf = answers.answer("verdict").answer_confidence
                        p_drop = answers.answer("verdict").probabilities.get("drop", 0)
                        p_keep = answers.answer("verdict").probabilities.get("keep", 0)
                        p_truncate = answers.answer("verdict").probabilities.get("truncate", 0)
                        essential = answers.noul("essential")
                        consumed = answers.noul("consumed")
                        superseded = answers.noul("superseded")
                        relevance = answers.answer("relevance").score

                        # Store per kind
                        signals = {
                            "verdict_confidence": verdict_conf,
                            "p_drop": p_drop,
                            "p_keep": p_keep,
                            "p_truncate": p_truncate,
                            "essential": essential,
                            "consumed": consumed,
                            "superseded": superseded,
                            "relevance": relevance,
                        }
                        for sig_name, val in signals.items():
                            if val is not None:
                                kind_signals[kind][sig_name].append(val)

                        # Veto funnel on removable items
                        if kind in removable_kinds:
                            item_dict = {
                                "role": "tool",
                                "tokens": count_tokens(msg.content),
                                "age_in_turns": 6,
                                "tool_name": msg.name,
                            }
                            profile = PROFILES["aggressive"]  # Use aggressive for funnel
                            action = decide(item_dict, answers.answers, profile)

                            # Check each gate in order
                            gates = [
                                ("hard_rules", action != "DROP"),  # If not DROP, hard rules blocked
                                ("verdict_drop", answers.answer("verdict").choice != "drop"),
                                ("confidence", verdict_conf <= profile.conf),
                                ("essential", essential >= profile.ess),
                                ("relevance", relevance > profile.rel),
                                ("junk_reason", not (
                                    consumed >= profile.con or
                                    superseded >= profile.con or
                                    relevance <= profile.noise
                                )),
                            ]

                            first_veto = None
                            any_veto = False
                            for gate_name, blocked in gates:
                                if blocked:
                                    any_veto = True
                                    any_veto_counts[gate_name] += 1
                                    if first_veto is None:
                                        first_veto = gate_name
                            if first_veto:
                                veto_counts[first_veto] += 1

                        # Danger check on must_keep items
                        if kind in must_keep_kinds:
                            danger_total += 1
                            item_dict = {
                                "role": "tool",
                                "tokens": count_tokens(msg.content),
                                "age_in_turns": 6,
                                "tool_name": msg.name,
                            }
                            profile = PROFILES["aggressive"]
                            action = decide(item_dict, answers.answers, profile)
                            if action == "DROP":
                                danger_drops += 1

                # Check pieces
                from contextcrunch.core.splitter import split_text
                pieces = split_text(msg.content, settings.piece_target_chars, settings.piece_max_chars)
                if len(pieces) > 1:
                    for piece_idx, piece in enumerate(pieces):
                        piece_state = build_state(messages, idx, key["goal"], settings, content=piece.text)
                        piece_state = fit_state(piece_state, settings)
                        piece_state["item"]["age_in_turns"] = 0  # pieces are judged independently
                        piece_key = cache_key(piece_state, question_set)
                        record = cache.get(piece_key)
                        if record:
                            answers = LayaResult.model_validate(record["answers"])
                            # Determine piece kind
                            piece_kind = kind
                            if kind == "trap":
                                # Check if piece contains a must_survive string
                                for fact in key.get("must_survive", []):
                                    if fact in piece.text:
                                        piece_kind = "trap_piece"
                                        break

                            signals = {
                                "verdict_confidence": answers.answer("verdict").answer_confidence,
                                "p_drop": answers.answer("verdict").probabilities.get("drop", 0),
                                "p_keep": answers.answer("verdict").probabilities.get("keep", 0),
                                "p_truncate": answers.answer("verdict").probabilities.get("truncate", 0),
                                "essential": answers.noul("essential"),
                                "consumed": answers.noul("consumed"),
                                "superseded": answers.noul("superseded"),
                                "relevance": answers.answer("relevance").score,
                            }
                            for sig_name, val in signals.items():
                                if val is not None:
                                    kind_signals[piece_kind][sig_name].append(val)

    # Print report
    report_lines = []
    report_lines.append("# ContextCrunch Diagnostic Report")
    report_lines.append("")
    report_lines.append(f"**Split:** {args.split}  |  **Question variant:** {args.questions_variant}")
    report_lines.append(f"**Cache:** {args.replay_dir}  |  **Entries:** {len(cache)}")
    report_lines.append("")

    # 1) Per-kind distribution table
    report_lines.append("## 1. Per-Kind Signal Distributions")
    report_lines.append("")
    report_lines.append("| Kind | Signal | n | Mean | P10 | Median | P90 |")
    report_lines.append("|------|--------|---|------|-----|--------|-----|")

    signal_order = [
        "verdict_confidence", "p_drop", "p_keep", "p_truncate",
        "essential", "consumed", "superseded", "relevance"
    ]

    for kind in sorted(kind_signals.keys()):
        for sig in signal_order:
            if sig in kind_signals[kind]:
                stats = distribution_stats(kind_signals[kind][sig])
                report_lines.append(
                    f"| {kind} | {sig} | {stats['n']} | "
                    f"{stats['mean']:.3f} | {stats['p10']:.3f} | "
                    f"{stats['median']:.3f} | {stats['p90']:.3f} |"
                )
    report_lines.append("")

    # 2) Separation (AUC)
    report_lines.append("## 2. Signal Separation (AUC: removable vs must_keep)")
    report_lines.append("")
    report_lines.append("| Signal | AUC | Strength |")
    report_lines.append("|--------|-----|----------|")

    # Combine removable and must_keep values
    for sig in signal_order:
        removable_vals = []
        keep_vals = []
        for kind in removable_kinds:
            removable_vals.extend(kind_signals[kind].get(sig, []))
        for kind in must_keep_kinds:
            keep_vals.extend(kind_signals[kind].get(sig, []))

        if removable_vals and keep_vals:
            # Determine orientation
            higher_is_junk = sig in ["p_drop", "consumed", "superseded", "verdict_confidence"]
            if sig in ["essential", "relevance", "p_keep"]:
                higher_is_junk = False
            auc = mann_whitney_auc(removable_vals, keep_vals, higher_is_junk)
            strength = signal_strength_label(auc)
            report_lines.append(f"| {sig} | {auc:.3f} | {strength} |")

    report_lines.append("")

    # 3) Veto funnel
    report_lines.append("## 3. Veto Funnel (removable items, aggressive profile)")
    report_lines.append("")
    report_lines.append("| Gate | First Failing | Any Failing |")
    report_lines.append("|------|---------------|-------------|")

    gate_order = ["hard_rules", "verdict_drop", "confidence", "essential", "relevance", "junk_reason"]
    for gate in gate_order:
        report_lines.append(f"| {gate} | {veto_counts.get(gate, 0)} | {any_veto_counts.get(gate, 0)} |")

    report_lines.append("")

    # 4) Danger check
    report_lines.append("## 4. Danger Check (must_keep items wrongly dropped)")
    report_lines.append("")
    report_lines.append(f"**Must-keep items evaluated:** {danger_total}")
    report_lines.append(f"**Wrongly dropped:** {danger_drops}")
    if danger_total > 0:
        report_lines.append(f"**Danger rate:** {100 * danger_drops / danger_total:.1f}%")
    report_lines.append("")

    # 5) Summary
    report_lines.append("## 5. Summary")
    report_lines.append("")

    # Find strongest signals
    signal_aucs = {}
    for sig in signal_order:
        removable_vals = []
        keep_vals = []
        for kind in removable_kinds:
            removable_vals.extend(kind_signals[kind].get(sig, []))
        for kind in must_keep_kinds:
            keep_vals.extend(kind_signals[kind].get(sig, []))
        if removable_vals and keep_vals:
            higher_is_junk = sig in ["p_drop", "consumed", "superseded", "verdict_confidence"]
            if sig in ["essential", "relevance", "p_keep"]:
                higher_is_junk = False
            auc = mann_whitney_auc(removable_vals, keep_vals, higher_is_junk)
            signal_aucs[sig] = auc

    strong_signals = [s for s, a in signal_aucs.items() if a >= 0.85]
    usable_signals = [s for s, a in signal_aucs.items() if 0.70 <= a < 0.85]
    weak_signals = [s for s, a in signal_aucs.items() if a < 0.70]

    report_lines.append(f"**Strong signals (AUC >= 0.85):** {', '.join(strong_signals) if strong_signals else 'none'}")
    report_lines.append(f"**Usable signals (0.70 <= AUC < 0.85):** {', '.join(usable_signals) if usable_signals else 'none'}")
    report_lines.append(f"**Weak signals (AUC < 0.70):** {', '.join(weak_signals) if weak_signals else 'none'}")
    report_lines.append("")

    # Main blocker
    if veto_counts:
        main_blocker = max(veto_counts.items(), key=lambda x: x[1])
        report_lines.append(f"**Main veto gate:** {main_blocker[0]} ({main_blocker[1]} items blocked first)")
    report_lines.append("")

    # Recommendation
    if "confidence" in weak_signals:
        report_lines.append("**Recommendation:** The verdict confidence signal is weak. Consider lowering the confidence threshold in profiles, but only after verifying no facts are lost on the test split.")
    elif "essential" in weak_signals and "relevance" in weak_signals:
        report_lines.append("**Recommendation:** Both essential and relevance signals are weak. The model struggles to identify irreplaceable content. Consider improving question wording (v2) or adding more training examples.")
    else:
        report_lines.append("**Recommendation:** Signals show usable separation. Profile thresholds can be tuned based on the veto funnel analysis.")

    # Write report
    report_text = "\n".join(report_lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report_text, encoding="utf-8")
    print(f"\nReport written to {args.out}")
    # Print with safe encoding
    try:
        print("\n" + report_text)
    except UnicodeEncodeError:
        # Fallback: print ASCII-safe version
        safe_text = report_text.encode('ascii', 'replace').decode('ascii')
        print("\n" + safe_text)

    await engine.aclose()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())