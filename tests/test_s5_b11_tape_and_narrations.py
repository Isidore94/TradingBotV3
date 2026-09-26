"""S5 side-by-tape line and B11 unread slot narrations.

S5: `setup_grades.side_by_tape` counts the 5-session horizon rows vs SPY over the
last 20 scan sessions; Setup Tracker and Day Review print its line and compute
nothing. B11: `slot_narration` reads the `daily_digest` narration (Day Review)
and the `setup_research` narration (Weekend Prep); absent and stale say so.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import setup_grades  # noqa: E402
import slot_narration  # noqa: E402


# ---------------------------------------------------------------------------
# S5 fixture: the numbers
# ---------------------------------------------------------------------------


def _row(symbol, side, scan_day, target_day, side_return, *, mature=True, measured=True):
    return {
        "symbol": symbol,
        "side": side,
        "scan_date": scan_day,
        "target_session": target_day,
        "horizon_sessions": "5",
        "side_return_pct": str(side_return),
        "measured": "True" if measured else "False",
        "maturity": "mature" if mature else "immature",
        "outcome_kind": setup_grades.TAPE_OUTCOME_KIND,
    }


def _days(count: int) -> list[str]:
    start = date(2026, 8, 3)
    return [(start + timedelta(days=i)).isoformat() for i in range(count)]


def test_side_by_tape_counts_the_last_20_sessions_vs_spy():
    days = _days(44)
    rows = []
    spy = {}
    # Scan days use even indexes; targets use odd indexes, so SPY can be +1%
    # on every pair without one close serving two roles.
    for n in range(21):
        scan_day, target_day = days[2 * n], days[2 * n + 1]
        spy[scan_day], spy[target_day] = 100.0, 101.0
        rows.append(_row("AAA", "LONG", scan_day, target_day, 2.0))   # +2 vs +1: win, +1
        rows.append(_row("BBB", "LONG", scan_day, target_day, 0.0))   # 0 vs +1: loss, -1
        rows.append(_row("CCC", "SHORT", scan_day, target_day, 1.0))  # +1 vs -1: win, +2
    # Unknowns never count: immature, unmeasured, no SPY close, target after as_of.
    spy[days[42]], spy[days[43]] = 100.0, 101.0
    rows.append(_row("DDD", "LONG", days[42], days[43], 9.0, mature=False))
    rows.append(_row("EEE", "LONG", days[42], days[43], 9.0, measured=False))
    rows.append(_row("FFF", "SHORT", days[42], "2026-12-31", 9.0))
    rows.append(_row("GGG", "SHORT", days[42], "2026-12-30", 9.0))  # no SPY close
    as_of = days[43]

    summary = setup_grades.side_by_tape(rows, spy, as_of=as_of)

    assert summary["sessions"] == 20
    assert summary["first"] == days[2] and summary["last"] == days[40]
    assert summary["long"]["n"] == 40 and summary["long"]["wins"] == 20
    assert summary["long"]["beat_pct"] == pytest.approx(50.0)
    assert summary["long"]["excess_pct"] == pytest.approx(0.0)  # (+1 + -1) / 2
    assert summary["short"]["n"] == 20 and summary["short"]["wins"] == 20
    assert summary["short"]["excess_pct"] == pytest.approx(2.0)
    assert setup_grades.side_by_tape_line(summary) == (
        "Last 20 sessions, tape-relative: longs beat SPY 50% (excess +0.00%), "
        f"shorts 100% (+2.00%). n 40 long / 20 short, scan dates {days[2]} to {days[40]}."
    )


def test_a_target_after_as_of_is_unknown_point_in_time():
    rows = [_row("AAA", "LONG", "2026-09-01", "2026-09-08", 2.0)]
    spy = {"2026-09-01": 100.0, "2026-09-08": 101.0}

    assert setup_grades.side_by_tape(rows, spy, as_of="2026-09-07")["sessions"] == 0
    assert setup_grades.side_by_tape(rows, spy, as_of="2026-09-08")["sessions"] == 1


def test_missing_data_is_unknown_never_a_number():
    assert setup_grades.side_by_tape_line(None) == setup_grades.SIDE_BY_TAPE_UNKNOWN
    assert setup_grades.side_by_tape_line(setup_grades.side_by_tape([], {})) == (
        setup_grades.SIDE_BY_TAPE_UNKNOWN
    )
    # Rows but no SPY: every row is unknown.
    rows = [_row("AAA", "LONG", "2026-09-01", "2026-09-08", 2.0)]
    assert setup_grades.side_by_tape(rows, {})["sessions"] == 0
    # One side measured, the other not.
    spy = {"2026-09-01": 100.0, "2026-09-08": 101.0}
    line = setup_grades.side_by_tape_line(setup_grades.side_by_tape(rows, spy))
    assert "longs beat SPY 100% (excess +1.00%)" in line
    assert "shorts unknown" in line


def test_the_worker_reader_is_cached_and_unknown_without_a_file(tmp_path, monkeypatch):
    from ui.services import working_lately_service as service

    horizon = tmp_path / "horizon.csv"
    monkeypatch.setattr(service, "_horizon_outcomes_path", lambda: horizon)
    monkeypatch.setattr(service, "_spy_bars_path", lambda: tmp_path / "SPY.parquet")
    service._LOOKING_BACK_CACHE.clear()

    assert service.read_side_by_tape("2026-09-25")["sessions"] == 0

    import csv

    with horizon.open("w", newline="", encoding="utf-8") as handle:
        row = _row("AAA", "LONG", "2026-09-01", "2026-09-08", 2.0)
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    monkeypatch.setattr(
        service, "read_spy_closes", lambda: {"2026-09-01": 100.0, "2026-09-08": 101.0}
    )
    calls = []
    real = setup_grades.side_by_tape
    monkeypatch.setattr(
        setup_grades, "side_by_tape", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    first = service.read_side_by_tape("2026-09-25")
    second = service.read_side_by_tape("2026-09-25")
    assert first["sessions"] == 1 and first["long"]["wins"] == 1
    assert second is first and len(calls) == 1
    service._LOOKING_BACK_CACHE.clear()


# ---------------------------------------------------------------------------
# B11 fixture: the two narration readers
# ---------------------------------------------------------------------------


def _narration(summary_text="Summary words.", **extra):
    body = {
        "executive_summary": summary_text,
        "what_is_working": [{"statement": "Longs held."}],
        "what_is_not_working": [{"statement": "Shorts chopped."}],
        "lessons_for_tomorrow": [],
        "risk_notes": [{"statement": "Thin data."}],
    }
    return {
        "generated_at": "2026-09-26T01:01:42-04:00",
        "model": "gemma3:12b",
        "narration": body,
        **extra,
    }


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload, encoding="utf-8")


def test_digest_narration_present_takes_the_newest_rerun(tmp_path):
    folder = tmp_path / "narration" / "2026"
    _write(folder / "2026-09-25.json", _narration("First run."))
    _write(folder / "2026-09-25.1.json", _narration("Second run."))
    _write(folder / "2026-09-24.json", _narration("Yesterday."))

    read = slot_narration.read_digest_narration("2026-09-25", tmp_path)

    assert read["state"] == "present"
    assert read["text"] == (
        "Night digest for 2026-09-25 (written 2026-09-26 01:01, gemma3:12b):\n"
        "Second run.\n- Working: Longs held.\n- Not working: Shorts chopped.\n"
        "- Risks: Thin data."
    )


def test_digest_narration_absent_says_so_and_names_the_newest(tmp_path):
    _write(tmp_path / "narration" / "2026" / "2026-09-24.json", _narration())

    read = slot_narration.read_digest_narration("2026-09-25", tmp_path)

    assert read["state"] == "absent"
    assert read["text"] == (
        "Night digest: no narration for 2026-09-25. The newest is for 2026-09-24."
    )
    empty = slot_narration.read_digest_narration("2026-09-25", tmp_path / "nothing")
    assert empty["text"] == "Night digest: no narration for 2026-09-25."


def test_digest_narration_unreadable_is_plain(tmp_path):
    _write(tmp_path / "narration" / "2026" / "2026-09-25.json", "{not json")

    read = slot_narration.read_digest_narration("2026-09-25", tmp_path)

    assert read["state"] == "unreadable"
    assert "could not be read" in read["text"]


def test_setup_research_narration_present_stale_and_absent(tmp_path):
    folder = tmp_path / "2026"
    assert slot_narration.read_setup_research_narration(tmp_path)["text"] == (
        "Setup research: no narration yet."
    )
    _write(folder / "2026-09-24.json", {"pack": 1})
    _write(folder / "2026-09-24.narration.json", _narration("Research words."))

    read = slot_narration.read_setup_research_narration(tmp_path)
    assert read["state"] == "present"
    assert read["text"].startswith(
        "Setup research for 2026-09-24 (written 2026-09-26 01:01, gemma3:12b):\n"
        "Research words."
    )

    _write(folder / "2026-09-25.json", {"pack": 2})
    stale = slot_narration.read_setup_research_narration(tmp_path)
    assert stale["state"] == "stale"
    assert "STALE: the 2026-09-25 research has no narration" in stale["text"]

    (folder / "2026-09-24.narration.json").unlink()
    absent = slot_narration.read_setup_research_narration(tmp_path)
    assert absent["state"] == "absent"
    assert absent["text"] == (
        "Setup research: no narration yet. The newest research pack is 2026-09-25."
    )


# ---------------------------------------------------------------------------
# The pages format only
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _boom(*_args, **_kwargs):
    raise AssertionError("the page must not read or compute this")


_SUMMARY = {
    "sessions": 20,
    "first": "2026-08-21",
    "last": "2026-09-18",
    "long": {"n": 10, "wins": 4, "beat_pct": 40.0, "excess_pct": -0.5},
    "short": {"n": 5, "wins": 3, "beat_pct": 60.0, "excess_pct": 1.25},
}
_LINE = (
    "Last 20 sessions, tape-relative: longs beat SPY 40% (excess -0.50%), shorts 60% "
    "(+1.25%). n 10 long / 5 short, scan dates 2026-08-21 to 2026-09-18."
)


@pytest.mark.qt
def test_day_review_prints_the_tape_line_and_digest_it_was_handed(qapp, monkeypatch):
    from ui.panels.day_review_panel import DayReviewPanel
    from ui.services import working_lately_service
    from ui.services.day_review_service import empty_payload

    class _Stub:
        def read_day(self, session_date, **_kwargs):
            return empty_payload(session_date)

    panel = DayReviewPanel(service=_Stub(), clock=lambda: datetime(2026, 9, 26, 8, 0))
    monkeypatch.setattr(panel, "reload", lambda: None)
    monkeypatch.setattr(setup_grades, "side_by_tape", _boom)
    monkeypatch.setattr(working_lately_service, "read_side_by_tape", _boom)
    monkeypatch.setattr(slot_narration, "read_digest_narration", _boom)
    try:
        payload = empty_payload("2026-09-25")
        payload["side_by_tape"] = dict(_SUMMARY)
        payload["digest_narration"] = {"state": "present", "text": "Night digest for 2026-09-25:\nWords."}
        panel.render(payload)
        assert panel.tape_side_note.text() == _LINE
        assert panel.digest_note.text() == "Night digest for 2026-09-25:\nWords."

        panel.render(empty_payload("2026-09-25"))
        assert panel.tape_side_note.text() == setup_grades.SIDE_BY_TAPE_UNKNOWN
        assert panel.digest_note.text() == "Night digest: not read."
    finally:
        panel.shutdown()
        panel.deleteLater()


def test_day_review_read_day_fills_both_on_the_worker(monkeypatch):
    from ui.services import day_review_service, working_lately_service

    assert {"side_by_tape", "digest_narration"} <= set(day_review_service.PAYLOAD_KEYS)
    empty = day_review_service.empty_payload("2026-09-25")
    assert empty["side_by_tape"] == {} and empty["digest_narration"] == {}

    seen = {}
    monkeypatch.setattr(
        working_lately_service,
        "read_side_by_tape",
        lambda as_of="": seen.setdefault("tape", as_of) and dict(_SUMMARY),
    )
    monkeypatch.setattr(
        slot_narration,
        "read_digest_narration",
        lambda session: seen.setdefault("digest", session) and {"text": "x"},
    )

    class _Broken:
        def entries_about(self, _session):
            raise OSError("no journal")

        def __getattr__(self, _name):
            raise OSError("no journal")

    payload = day_review_service.DayReviewService(journal_service=_Broken()).read_day(
        "2026-09-25", now=datetime(2026, 9, 26, 8, 0)
    )

    assert seen == {"tape": "2026-09-25", "digest": "2026-09-25"}
    assert payload["side_by_tape"] == _SUMMARY
    assert payload["digest_narration"] == {"text": "x"}


@pytest.mark.qt
def test_setup_tracker_prints_the_tape_line_it_was_handed(qapp, monkeypatch):
    from ui.panels import setup_tracker_panel as module
    from ui.services import working_lately_service

    panel = module.SetupTrackerPanel()
    monkeypatch.setattr(setup_grades, "side_by_tape", _boom)
    monkeypatch.setattr(working_lately_service, "read_side_by_tape", _boom)
    try:
        panel._on_exports_loaded({"signatures": {}, "ranked": {}, "raw": {}, "side_by_tape": dict(_SUMMARY)})
        assert panel.tape_side_label.text() == _LINE
        panel._on_exports_loaded({"signatures": {}, "ranked": {}, "raw": {}, "min_closed": 1})
        assert panel.tape_side_label.text() == setup_grades.SIDE_BY_TAPE_UNKNOWN
    finally:
        panel.shutdown()
        panel.deleteLater()


def test_setup_tracker_worker_read_carries_the_summary(monkeypatch):
    from ui.panels import setup_tracker_panel as module
    from ui.services import working_lately_service

    monkeypatch.setattr(working_lately_service, "read_side_by_tape", lambda as_of="": dict(_SUMMARY))
    assert module._read_tracker_exports(1)["side_by_tape"] == _SUMMARY

    monkeypatch.setattr(working_lately_service, "read_side_by_tape", _boom)
    assert module._read_tracker_exports(1)["side_by_tape"] == {}


@pytest.mark.qt
def test_weekend_prep_prints_the_setup_research_text(qapp, monkeypatch):
    from ui.panels import weekend_prep_panel as module

    panel = module.WeekendPrepPanel()
    monkeypatch.setattr(slot_narration, "read_setup_research_narration", _boom)
    try:
        panel._on_setup_research_ready({"state": "stale", "text": "Setup research for X - STALE"})
        assert panel.setup_research_note.text() == "Setup research for X - STALE"
        panel._on_setup_research_ready(None)
        assert panel.setup_research_note.text() == "Setup research: no narration yet."
        panel._on_setup_research_failed("share offline")
        assert "share offline" in panel.setup_research_note.text()
    finally:
        panel.shutdown()
        panel.deleteLater()


def test_weekend_prep_worker_read_never_raises(monkeypatch):
    from ui.panels import weekend_prep_panel as module

    monkeypatch.setattr(slot_narration, "read_setup_research_narration", _boom)
    read = module._read_setup_research_narration()
    assert read["state"] == "unreadable" and "could not be read" in read["text"]

    monkeypatch.setattr(
        slot_narration, "read_setup_research_narration", lambda: {"state": "present", "text": "ok"}
    )
    assert module._read_setup_research_narration() == {"state": "present", "text": "ok"}
