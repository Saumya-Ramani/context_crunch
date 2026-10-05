"""Tests for the Storage Box: originals and decision rows."""

from __future__ import annotations

import pytest
from tests.conftest import answers

from contextcrunch.storage.box import Box, DecisionRow, build_box, dump_row


@pytest.fixture
def box(tmp_path) -> Box:
    """Return a Storage Box backed by a temporary directory."""
    return Box(db_path=tmp_path / "box.db", originals_dir=tmp_path / "originals")


def test_original_round_trips_verbatim(box: Box) -> None:
    """Restoring a dropped item must return the exact original bytes."""
    content = "def round_price(x):\n    return int(x) + 1\n"
    ref = box.save_original(content)
    assert box.load_original(ref) == content


def test_original_reference_is_content_addressed(box: Box) -> None:
    """Saving the same text twice maps to one row."""
    assert box.save_original("same") == box.save_original("same")


def test_missing_original_returns_empty(box: Box) -> None:
    """An unknown reference must not raise."""
    assert box.load_original("nope") == ""


def test_records_and_reads_back_a_run(box: Box) -> None:
    """Every decision must be retrievable for a run."""
    for index in (3, 4):
        box.record(
            DecisionRow(
                run_id="run1",
                item_index=index,
                role="tool",
                label="tool:read_file",
                item_tokens=1000,
                action="DROP",
                profile="aggressive",
                verdict_choice="drop",
                verdict_conf=0.9,
                essential_noul=0.1,
                consumed_noul=0.9,
                superseded_noul=0.5,
                relevance_score=0.4,
                original_ref=box.save_original(f"body {index}"),
            ),
            answers=answers(),
        )
    rows = box.by_run("run1")
    assert len(rows) == 2
    assert [row.item_index for row in rows] == [3, 4]
    assert rows[0].verdict_choice == "drop"


def test_dropped_sorted_by_relevance(box: Box) -> None:
    """Recovery restores the most relevant drops first."""
    for index, score in ((3, 0.1), (4, 0.9)):
        box.record(
            DecisionRow(
                run_id="run1",
                item_index=index,
                role="tool",
                label="tool",
                item_tokens=10,
                action="DROP",
                profile="aggressive",
                relevance_score=score,
            )
        )
    assert [row.item_index for row in box.dropped("run1")] == [4, 3]


def test_summary_reports_savings(box: Box) -> None:
    """The API needs counts and a reduction percentage."""
    box.record(
        DecisionRow(
            run_id="run1",
            item_index=3,
            role="tool",
            label="tool",
            item_tokens=1000,
            action="DROP",
            profile="aggressive",
        )
    )
    summary = box.summary("run1")
    assert summary["counts"] == {"DROP": 1}
    assert summary["tokens_before"] == 1000
    assert summary["tokens_after"] == 0
    assert summary["reduction_pct"] == 100.0


def test_summary_of_an_unknown_run_is_empty(box: Box) -> None:
    """An unknown run must not raise."""
    assert box.summary("nope")["counts"] == {}


def test_dump_row_is_json(box: Box) -> None:
    """Rows must be loggable."""
    row = DecisionRow(
        run_id="r",
        item_index=1,
        role="tool",
        label="t",
        item_tokens=1,
        action="KEEP",
        profile="balanced",
    )
    assert "KEEP" in dump_row(row)


def test_build_box_uses_the_configured_paths() -> None:
    """build_box must honour CC_DB_PATH and CC_ORIGINALS_DIR."""
    from contextcrunch.core.settings import get_settings

    settings = get_settings()
    built = build_box()
    assert str(built.db_path) == settings.db_path
    assert str(built.originals_dir) == settings.originals_dir
