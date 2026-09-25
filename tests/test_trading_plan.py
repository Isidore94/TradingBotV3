"""P1-7 7a: the trader's trading plan file, its parser and its history.

Every test points `project_paths` at `tmp_path`; the live data dir is never read.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import project_paths  # noqa: E402
import trading_plan  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "trading_plan_v1.md"
NOW = datetime(2026, 9, 24, 21, 0, tzinfo=timezone.utc)


@pytest.fixture()
def plan_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(project_paths, "TRADING_PLAN_FILE", tmp_path / "trading_plan.md")
    monkeypatch.setattr(project_paths, "TRADING_PLAN_HISTORY_DIR", tmp_path / "trading_plan_history")
    return tmp_path


def test_the_parser_reads_the_fixture():
    parsed = trading_plan.parse_plan(FIXTURE.read_text(encoding="utf-8"))

    assert parsed["missing"] == []
    assert parsed["unknown_sections"] == ["Notes to self"]
    ids = {row["id"]: row["text"] for row in parsed["lines"]}
    assert ids == {
        "plan:goals:1": "Make 2R a week on swing trades.",
        "plan:goals:2": "Journal every trade the same day.",
        "plan:rules:1": "No trades in the first 15 minutes.",
        "plan:rules:2": "Respect the stop.",
        "plan:setups_i_trade:1": "D1 AVWAP first-deviation bounce, long only.",
        "plan:risk:1": "Max 1% of the account per trade.",
        "plan:what_i_am_testing:1": "Recap rule for 2026-09-23: hold winners to 1R.",
        "plan:decisions:1": "2026-09-20: Stop trading the open.",
        "plan:decisions:2": "no date on this one",
    }
    assert parsed["decisions"] == [
        {"day": "2026-09-20", "text": "Stop trading the open.", "dated": True},
        {"day": "", "text": "no date on this one", "dated": False},
    ]


def test_the_template_has_the_six_headings_and_no_citable_line():
    parsed = trading_plan.parse_plan(trading_plan.TEMPLATE)

    assert parsed["missing"] == []
    assert parsed["lines"] == []
    assert trading_plan.HEADINGS == (
        "Goals", "Rules", "Setups I trade", "Risk", "What I am testing", "Decisions",
    )


def test_a_missing_plan_is_created_from_the_template_only_once(plan_dir):
    first = trading_plan.read_plan(now=NOW)
    assert first["created"] is True
    assert Path(project_paths.TRADING_PLAN_FILE).read_text(encoding="utf-8") == trading_plan.TEMPLATE

    Path(project_paths.TRADING_PLAN_FILE).write_text("## Goals\n- mine\n", encoding="utf-8")
    second = trading_plan.read_plan(now=NOW)
    assert second["created"] is False
    assert second["text"] == "## Goals\n- mine\n"


def test_create_false_never_writes_a_plan(plan_dir):
    result = trading_plan.read_plan(create=False, now=NOW)

    assert result["exists"] is False
    assert not Path(project_paths.TRADING_PLAN_FILE).exists()
    assert not Path(project_paths.TRADING_PLAN_HISTORY_DIR).exists()


def test_every_change_is_snapshotted_once_and_append_only(plan_dir):
    plan = Path(project_paths.TRADING_PLAN_FILE)
    plan.write_text("## Goals\n- one\n", encoding="utf-8")
    trading_plan.read_plan(now=NOW)
    trading_plan.read_plan(now=NOW + timedelta(minutes=1))  # unchanged: no new snapshot
    assert len(trading_plan.snapshots()) == 1

    plan.write_text("## Goals\n- two\n", encoding="utf-8")
    trading_plan.read_plan(now=NOW + timedelta(minutes=2))
    kept = trading_plan.snapshots()
    assert [path.read_text(encoding="utf-8") for path in kept] == ["## Goals\n- one\n", "## Goals\n- two\n"]

    plan.write_text("## Goals\n- one\n", encoding="utf-8")  # back to the old text is a change too
    trading_plan.read_plan(now=NOW + timedelta(minutes=3))
    assert len(trading_plan.snapshots()) == 3
    assert kept[0].read_text(encoding="utf-8") == "## Goals\n- one\n"


def test_a_failed_snapshot_is_logged_and_retried_on_the_next_read(plan_dir, monkeypatch, caplog):
    plan = Path(project_paths.TRADING_PLAN_FILE)
    plan.write_text("## Rules\n- keep it\n", encoding="utf-8")
    real_open = Path.open

    def refuse(self, mode="r", *args, **kwargs):
        if "x" in mode and self.parent == Path(project_paths.TRADING_PLAN_HISTORY_DIR):
            raise PermissionError("disk says no")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse)
    with caplog.at_level("WARNING"):
        result = trading_plan.read_plan(now=NOW)
    assert result["text"] == "## Rules\n- keep it\n"
    assert result["snapshot"] == ""
    assert "not snapshotted" in caplog.text

    monkeypatch.setattr(Path, "open", real_open)
    again = trading_plan.read_plan(now=NOW + timedelta(minutes=1))
    assert again["snapshot"]
    assert [p.read_text(encoding="utf-8") for p in trading_plan.snapshots()] == ["## Rules\n- keep it\n"]


def test_append_decision_adds_a_dated_line_and_snapshots(plan_dir):
    trading_plan.read_plan(now=NOW)
    trading_plan.append_decision("Stop trading the open.", now=NOW, day="2026-09-24")
    trading_plan.append_decision("Size down in chop.", now=NOW + timedelta(seconds=5), day="2026-09-25")

    parsed = trading_plan.parse_plan(Path(project_paths.TRADING_PLAN_FILE).read_text(encoding="utf-8"))
    assert parsed["decisions"] == [
        {"day": "2026-09-24", "text": "Stop trading the open.", "dated": True},
        {"day": "2026-09-25", "text": "Size down in chop.", "dated": True},
    ]
    assert parsed["missing"] == []
    assert len(trading_plan.snapshots()) == 3  # template, then two decisions


def test_the_recap_rule_line_is_replaced_and_the_traders_lines_kept(plan_dir):
    Path(project_paths.TRADING_PLAN_FILE).write_text(
        "## What I am testing\n\n- my own test\n\n## Decisions\n", encoding="utf-8"
    )
    trading_plan.set_testing_rule("hold winners", for_day="2026-09-24", now=NOW)
    trading_plan.set_testing_rule("respect the stop", for_day="2026-09-25", now=NOW + timedelta(seconds=5))

    parsed = trading_plan.parse_plan(Path(project_paths.TRADING_PLAN_FILE).read_text(encoding="utf-8"))
    assert parsed["sections"]["What I am testing"] == [
        "my own test",
        "Recap rule for 2026-09-25: respect the stop",
    ]
    history = [p.read_text(encoding="utf-8") for p in trading_plan.snapshots()]
    assert any("Recap rule for 2026-09-24: hold winners" in text for text in history)


def test_a_failed_plan_write_raises(plan_dir, monkeypatch):
    trading_plan.read_plan(now=NOW)

    def boom(*_a, **_k):
        raise OSError("read-only")

    monkeypatch.setattr(trading_plan.os, "replace", boom)
    with pytest.raises(trading_plan.PlanWriteError):
        trading_plan.append_decision("x", now=NOW)


def test_snapshot_at_picks_the_newest_before_the_moment(plan_dir):
    plan = Path(project_paths.TRADING_PLAN_FILE)
    plan.write_text("## Goals\n- a\n", encoding="utf-8")
    trading_plan.read_plan(now=NOW)
    plan.write_text("## Goals\n- b\n", encoding="utf-8")
    trading_plan.read_plan(now=NOW + timedelta(days=2))

    chosen = trading_plan.snapshot_at(NOW + timedelta(days=1))
    assert chosen is not None and chosen.read_text(encoding="utf-8") == "## Goals\n- a\n"
    assert trading_plan.snapshot_at(NOW - timedelta(days=1)) is None
