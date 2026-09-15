"""Red contract tests for packet DR-REPAIR (2026-09-15).

The small stores here model the append-only M5 file as it is actually written:
an event first receives an empty ``registered`` row, then later update rows;
empty event ids are real rows, not an invitation to join unrelated alerts.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime, time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import daily_recap_schedule as schedule  # noqa: E402

SESSION = "2026-09-14"
NOW = datetime(2026, 9, 15, 7, 30)

INTRADAY_COLUMNS = (
    "event_id",
    "logged_at",
    "trade_date",
    "symbol",
    "direction",
    "entry_time",
    "status",
    "mfe_pct",
    "mae_pct",
    "eod_move_pct",
)
HORIZON_COLUMNS = (
    "scan_row_id",
    "scan_date",
    "target_session",
    "symbol",
    "side",
    "horizon_sessions",
    "measured",
    "maturity",
    "side_return_pct",
)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _outcome(
    event_id: str,
    symbol: str,
    *,
    side: str = "LONG",
    entry: str = "09:30:00",
    logged: str = "09:31:00",
    status: str = "closed",
    mfe: str = "",
    mae: str = "",
    eod: str = "",
) -> dict:
    return {
        "event_id": event_id,
        "logged_at": f"{SESSION}T{logged}-07:00",
        "trade_date": SESSION,
        "symbol": symbol,
        "direction": side,
        "entry_time": f"{SESSION}T{entry}-07:00",
        "status": status,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "eod_move_pct": eod,
    }


def _horizon(
    symbol: str,
    *,
    side: str = "LONG",
    scan_date: str = SESSION,
    horizon: int = 3,
    measured: str = "True",
    maturity: str = "mature",
    result: str = "",
) -> dict:
    return {
        "scan_row_id": f"{symbol}-{horizon}",
        "scan_date": scan_date,
        "target_session": "2026-09-17",
        "symbol": symbol,
        "side": side,
        "horizon_sessions": str(horizon),
        "measured": measured,
        "maturity": maturity,
        "side_return_pct": result,
    }


def _sources(tmp_path: Path, *, outcomes=None, horizons=None, annotations=None):
    """A complete, private RecapSources set.  No test reads a live store."""
    from daily_recap_reader import RecapSources

    outcomes_path = tmp_path / "intraday_bounce_outcomes.csv"
    horizon_path = tmp_path / "session_horizon_outcomes.csv"
    _write_csv(outcomes_path, INTRADAY_COLUMNS, list(outcomes or ()))
    _write_csv(horizon_path, HORIZON_COLUMNS, list(horizons or ()))

    empty_csv = tmp_path / "empty.csv"
    _write_csv(empty_csv, ("when",), [])
    annotations_path = tmp_path / "trader_annotations.jsonl"
    annotations_path.write_text(
        "".join(json.dumps(row) + "\n" for row in (annotations or ())),
        encoding="utf-8",
    )
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("", encoding="utf-8")
    staged = tmp_path / "staged.json"
    staged.write_text(json.dumps({"pending": {"long": {}, "short": {}}}), encoding="utf-8")
    working = tmp_path / "working.json"
    working.write_text("{}", encoding="utf-8")

    return RecapSources(
        intraday_outcomes=outcomes_path,
        tier_outcomes=empty_csv,
        session_horizon_outcomes=horizon_path,
        annotations=annotations_path,
        pick_feedback=empty_jsonl,
        swing_favorites=empty_jsonl,
        human_focus_outcomes=empty_csv,
        review_events=empty_jsonl,
        preference_report=empty_csv,
        staged_picks=staged,
        environment_labels=empty_jsonl,
        working_lately=working,
    )


def _read(sources, *, lookback: int = 3):
    import daily_recap_reader

    return daily_recap_reader.read_session(
        SESSION, lookback_sessions=lookback, now=NOW, sources=sources
    )


def _annotation(symbol: str, timeframe: str, *, kind: str = "veto") -> dict:
    return {
        "created_at": f"{SESSION}T10:00:00-07:00",
        "event_id": f"{symbol}-{timeframe}-{kind}",
        "event_type": kind,
        "session_date": SESSION,
        "symbol": symbol,
        "side": "LONG",
        "timeframe": timeframe,
        "reason_code": "timing",
    }


def _decision(session, symbol: str):
    rows = [row for row in session.my_decisions.rows if row.symbol == symbol]
    assert len(rows) == 1
    return rows[0]


def test_intraday_reader_streams_latest_event_states_and_keeps_blank_ids_distinct(
    tmp_path, monkeypatch
):
    """The 400+ MB append log is covered without ``read_text`` or a blank-id join."""
    sources = _sources(
        tmp_path,
        outcomes=[
            _outcome("event-a", "AAA", status="registered"),
            _outcome("event-a", "AAA", logged="10:00:00", mfe="4.00", eod="2.00"),
            _outcome("event-b", "BBB", mfe="3.00", eod="1.00"),
            _outcome("", "EMPTYA", mfe="2.00"),
            _outcome("", "EMPTYB", mfe="1.00"),
        ],
    )
    original = Path.read_text

    def forbid_intraday_read_text(path, *args, **kwargs):
        if Path(path) == sources.intraday_outcomes:
            raise AssertionError("the intraday append log was materialised with Path.read_text")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", forbid_intraday_read_text)
    session = _read(sources)

    assert session.coverage["intraday_outcomes"].rows == 5
    assert {(row.symbol, row.capture_id) for row in session.worked_today.rows} == {
        ("AAA", "event-a"),
        ("BBB", "event-b"),
        ("EMPTYA", ""),
        ("EMPTYB", ""),
    }
    aaa = next(row for row in session.worked_today.rows if row.symbol == "AAA")
    assert aaa.measures["mfe_pct"] == pytest.approx(4.0)


def test_worked_today_selects_one_unblended_best_event_per_stock_and_side(tmp_path):
    sources = _sources(
        tmp_path,
        outcomes=[
            _outcome("old", "AAA", entry="09:30:00", mfe="8.00", eod="-3.00"),
            _outcome("new", "AAA", entry="10:00:00", mfe="8.00", eod="5.00"),
            _outcome("lower", "AAA", entry="11:00:00", mfe="2.00", eod="9.00"),
            _outcome("short", "AAA", side="SHORT", mfe="3.00", eod="1.00"),
        ],
    )
    session = _read(sources)

    assert [(row.symbol, row.side) for row in session.worked_today.rows] == [
        ("AAA", "LONG"),
        ("AAA", "SHORT"),
    ]
    long_row = next(row for row in session.worked_today.rows if row.side == "LONG")
    assert long_row.capture_id == "new"
    assert long_row.observed_at.strftime("%H:%M") == "10:00"
    assert long_row.measures == {"mfe_pct": 8.0, "eod_move_pct": 5.0}
    assert "best measured alert per stock/side" in session.worked_today.note.lower()


def test_d1_and_m5_decisions_read_their_own_outcome_sources(tmp_path):
    sources = _sources(
        tmp_path,
        outcomes=[
            _outcome("d1-m5", "D1X", mfe="1.00"),
            _outcome("m5-first", "M5Y", mfe="2.00"),
            _outcome("m5-best", "M5Y", logged="10:30:00", mfe="8.00"),
        ],
        horizons=[_horizon("D1X", result="6.25")],
        annotations=[_annotation("D1X", "D1"), _annotation("M5Y", "M5")],
    )
    session = _read(sources)
    d1 = _decision(session, "D1X")
    m5 = _decision(session, "M5Y")

    assert d1.detail["timeframe"] == "D1"
    assert d1.measures["d1_result_pct"] == pytest.approx(6.25)
    assert d1.measures["day_mfe_pct"] is None
    assert d1.detail["result_state"] == "measured"
    assert m5.detail["timeframe"] == "M5"
    assert m5.measures["day_mfe_pct"] == pytest.approx(8.0)
    rejected = {row.symbol: row for row in session.rejected_that_worked.rows}
    assert rejected["D1X"].measures["favorable_pct"] == pytest.approx(6.25)
    assert rejected["M5Y"].measures["favorable_pct"] == pytest.approx(8.0)


def test_an_immature_d1_decision_is_pending_not_the_m5_days_best(tmp_path):
    sources = _sources(
        tmp_path,
        outcomes=[_outcome("misleading-m5", "D1P", mfe="9.00")],
        horizons=[_horizon("D1P", measured="", maturity="immature", result="")],
        annotations=[_annotation("D1P", "D1")],
    )
    d1 = _decision(_read(sources), "D1P")

    assert d1.measures["d1_result_pct"] is None
    assert d1.measures["day_mfe_pct"] is None
    assert d1.detail["result_state"] == "pending"
    assert d1.unavailable["d1_result_pct"]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_pending_swings_are_visible_in_the_rendered_swing_tab(tmp_path, qapp):
    sources = _sources(
        tmp_path,
        horizons=[
            _horizon("MSFT", scan_date="2026-09-11", horizon=1, result="1.50"),
            _horizon(
                "MSFT",
                scan_date="2026-09-11",
                horizon=3,
                measured="",
                maturity="immature",
                result="",
            ),
        ],
    )
    session = _read(sources)
    assert [row.symbol for row in session.recent_swings.pending] == ["MSFT"]

    from ui.panels.daily_recap_panel import DailyRecapPanel

    page = DailyRecapPanel(clock=lambda: NOW)
    page.render_session(session)
    qapp.processEvents()
    cells = [
        page.recent_swings_table.item(row, column).text()
        for row in range(page.recent_swings_table.rowCount())
        for column in range(page.recent_swings_table.columnCount())
        if page.recent_swings_table.item(row, column) is not None
    ]
    assert page.recent_swings_table.rowCount() == 1
    assert "MSFT" in cells
    assert "Pending" in cells
    page.shutdown()
    page.deleteLater()


def test_reader_summary_reports_factual_counts_and_declared_top_m5_rows(tmp_path):
    sources = _sources(
        tmp_path,
        outcomes=[
            _outcome("aaa", "AAA", status="registered"),
            _outcome("aaa", "AAA", logged="10:00:00", mfe="4.00"),
            _outcome("bbb", "BBB", mfe="2.00"),
            _outcome("ccc", "CCC", status="open"),
        ],
        horizons=[
            _horizon("SWINGOK", scan_date="2026-09-11", result="3.00"),
            _horizon(
                "SWINGWAIT",
                scan_date="2026-09-11",
                measured="",
                maturity="immature",
                result="",
            ),
        ],
        annotations=[_annotation("AAA", "M5"), _annotation("D1DONE", "D1")],
    )
    session = _read(sources)
    summary = session.summary

    assert summary.raw_m5_update_rows == 4
    assert summary.latest_m5_event_count == 3
    assert summary.m5_stock_side_count == 3
    assert (summary.m5_measured_count, summary.m5_unmeasured_count) == (2, 1)
    assert [row.symbol for row in summary.top_measured_m5] == ["AAA", "BBB"]
    assert summary.decision_counts["M5"] == {"measured": 1, "pending": 0, "unmeasured": 0}
    assert summary.swing_counts == {"measured": 1, "pending": 1}


def test_summary_m5_counts_are_scoped_to_the_selected_session(tmp_path):
    """Coverage is full-file, but the recap sentence is about the chosen day."""
    prior = _outcome("prior", "OLD", mfe="7.00")
    prior["trade_date"] = "2026-09-11"
    prior["logged_at"] = "2026-09-11T10:00:00-07:00"
    other = _outcome("other", "BBB", mfe="3.00")
    other["trade_date"] = "2026-09-10"
    other["logged_at"] = "2026-09-10T10:00:00-07:00"
    sources = _sources(
        tmp_path,
        outcomes=[
            _outcome("selected", "AAA", status="registered"),
            _outcome("selected", "AAA", logged="10:00:00", mfe="4.00"),
            prior,
            other,
        ],
    )

    session = _read(sources)

    assert session.coverage["intraday_outcomes"].rows == 4
    assert session.summary.raw_m5_update_rows == 2
    assert session.summary.latest_m5_event_count == 1


def test_m5_and_d1_decisions_with_one_identity_stay_separate(tmp_path):
    """Timeframe is part of a decision identity, never a display-only label."""
    sources = _sources(
        tmp_path,
        outcomes=[_outcome("m5", "DUAL", mfe="4.00")],
        horizons=[_horizon("DUAL", result="6.25")],
        annotations=[_annotation("DUAL", "M5"), _annotation("DUAL", "D1")],
    )

    rows = [row for row in _read(sources).my_decisions.rows if row.symbol == "DUAL"]

    assert [row.detail["timeframe"] for row in rows] == ["M5", "D1"]
    assert rows[0].measures["day_mfe_pct"] == pytest.approx(4.0)
    assert rows[1].measures["d1_result_pct"] == pytest.approx(6.25)


def test_rendered_recap_places_the_readers_compact_summary_on_the_page(tmp_path, qapp):
    sources = _sources(
        tmp_path,
        outcomes=[
            _outcome("aaa", "AAA", status="registered"),
            _outcome("aaa", "AAA", logged="10:00:00", mfe="4.00"),
            _outcome("bbb", "BBB", mfe="2.00"),
            _outcome("ccc", "CCC", status="open"),
        ],
    )
    session = _read(sources)
    summary = session.summary

    from PySide6.QtWidgets import QLabel
    from ui.panels.daily_recap_panel import DailyRecapPanel

    page = DailyRecapPanel(clock=lambda: NOW)
    page.render_session(session)
    qapp.processEvents()
    page_text = "\n".join(label.text() for label in page.findChildren(QLabel))
    assert summary.text in page_text
    page.shutdown()
    page.deleteLater()


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _automatic_page(qapp, clock, *, auto_time="12:00"):
    from ui.panels.daily_recap_panel import DailyRecapPanel

    page = DailyRecapPanel(clock=clock, auto_time_reader=lambda: auto_time)
    page.reads = []
    page.reload = lambda: page.reads.append(page.session_date())
    page.show()
    qapp.processEvents()
    return page


@pytest.mark.qt
def test_automatic_reads_run_at_noon_then_once_after_the_regular_close(qapp):
    clock = _Clock(datetime(2026, 9, 14, 12, 0, tzinfo=schedule.PACIFIC))
    page = _automatic_page(qapp, clock)

    assert page.poll_auto_read() == "2026-09-14"
    clock.now = datetime(2026, 9, 14, 13, 1, tzinfo=schedule.PACIFIC)
    assert page.poll_auto_read() == "2026-09-14"
    clock.now = datetime(2026, 9, 14, 13, 2, tzinfo=schedule.PACIFIC)
    assert page.poll_auto_read() is None
    assert page.reads == ["2026-09-14", "2026-09-14"]
    page.shutdown()
    page.deleteLater()


@pytest.mark.qt
def test_automatic_post_close_read_uses_the_exchange_early_close(qapp):
    import market_early_close

    early = date(2026, 11, 27)
    close = market_early_close.session_close(early).astimezone(schedule.PACIFIC)
    assert close.time() == time(10, 0), close
    clock = _Clock(datetime(2026, 11, 27, 9, 30, tzinfo=schedule.PACIFIC))
    page = _automatic_page(qapp, clock, auto_time="09:30")

    assert page.poll_auto_read() == early.isoformat()
    clock.now = datetime(2026, 11, 27, 10, 1, tzinfo=schedule.PACIFIC)
    assert page.poll_auto_read() == early.isoformat()
    assert page.reads == [early.isoformat(), early.isoformat()]
    page.shutdown()
    page.deleteLater()
