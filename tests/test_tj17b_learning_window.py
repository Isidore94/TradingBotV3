"""TJ-17B — selected completed-session learning windows.  RED before build.

The fixture covers 21 real exchange sessions ending Monday 2026-09-21.  Labor
Day (2026-09-07) is deliberately absent.  The final five sessions are
09-15, 09-16, 09-17, 09-18 and 09-21, with three correct calls and two wrong;
the preceding five have one correct and four wrong.  Thus the 5-session window
is 3/5 and the 10-session window is 4/10.  A pending grade is superseded on
09-21, so the stored file has both rows while the reader may count only the
current right row.

The trades are real JournalStore rows, then are read through the real journal
model.  One opens before the 09-18 close: its entry context must be the prior
session's label, while inclusion in the selected result window remains based
on its authoritative ``closed_at``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tj16_support as fx  # noqa: E402


SESSIONS_21 = (
    "2026-08-21",
    "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28",
    "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
    "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11",
    "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
    "2026-09-21",
)
# Tuesday before the New York close: Monday 09-21 is the last completed
# exchange session, so a future requested end must clamp there.
LAST_COMPLETED = datetime(2026, 9, 22, 12, 0, tzinfo=fx.PACIFIC)
LAST_FIVE = list(SESSIONS_21[-5:])
LAST_TEN = list(SESSIONS_21[-10:])
LAST_TWENTY = list(SESSIONS_21[-20:])


def _rows_for_window_fixture() -> list[dict[str, Any]]:
    """Real clicked grade rows, with a CURRENT-row supersession at the end."""
    import market_read_grades as grades

    rows: list[dict[str, Any]] = []
    # The 20 sessions the selector can show: 14 right, 6 wrong.  The final
    # five are 3 right / 2 wrong; their prior five are 1 right / 4 wrong.
    right_sessions = set(SESSIONS_21[1:11]) | {
        "2026-09-08", "2026-09-15", "2026-09-16", "2026-09-21"
    }
    for session in SESSIONS_21[:-1]:
        direction = "up" if session in right_sessions else "down"
        label = "unknown" if session == "2026-09-08" else "trending_up"
        rows.extend(
            fx.graded_session(
                session,
                [{"hour": 9, "direction": direction, "confidence": "high",
                  "last_hour_spy": "up", "d1_environment": label}],
                rising=True,
            )
        )

    # The old pending row stays on disk.  Only the new, measured right row is
    # current, so the last-five fraction is still over five calls, never six.
    session = SESSIONS_21[-1]
    entry = fx.click_entry(session=session, hour=9, direction="up", confidence="high")
    read = grades.read_rows([entry], session=session)[0]
    context = fx.context(
        hour=9, direction="up", confidence="high", last_hour_spy="up",
        d1_environment="trending_up",
    )
    pending = grades.grade_read(read, m5_bars=(), atr=None, now=fx.stamp_at(session, 11), context=context)
    current = grades.grade_read(
        read,
        m5_bars=fx.session_tape(session, rising=True),
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
        now=fx.morning_after(session),
        supersedes=pending["grade_id"],
        context=context,
    )
    rows.extend((pending, current))

    # An older five-session call has finished; the newest one is still open.
    # Neither may be pooled into a rest-of-day percentage.
    rows.extend(fx.five_session_clicks(LAST_TEN, count=1))
    rows.extend(fx.five_session_clicks(LAST_FIVE, count=1))
    return rows


def _ledger(tmp_path: Path) -> Path:
    root = tmp_path / "day_review"
    fx.store_ledger(root, _rows_for_window_fixture())
    return root


def _current_trades(store) -> list[Any]:
    """Use JournalStore's actual rows, then the normal UI model conversion."""
    from ui.models.journal import JournalTrade

    return [JournalTrade.from_mapping(row) for row in store.list_trades()]


def _closed_trade(store, trade_id: str, *, opened_at: str, closed_at: str, net_pnl: float, currency: str = "USD") -> None:
    with store.connection() as conn:
        conn.execute(
            """
            INSERT INTO trades(
                trade_id, broker, account_number, account_label, symbol, security_type,
                currency, direction, status, opened_at, closed_at, trade_date,
                quantity_opened, quantity_closed, average_entry_price, average_exit_price,
                gross_pnl, commission, fees, net_pnl, pnl_usd, updated_at
            ) VALUES(?, 'IBKR', 'U1', 'MAIN', ?, 'STOCK', ?, 'LONG', 'CLOSED', ?, ?, ?,
                     100, 100, 10, 11, ?, 2, 1, ?, ?, '2026-09-22T17:00:00-07:00')
            """,
            (trade_id, trade_id, currency, opened_at, closed_at, closed_at[:10], net_pnl + 3,
             net_pnl, net_pnl if currency == "USD" else None),
        )


def _ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("trade_id") or "") for row in rows if str(row.get("trade_id") or "")}


def test_learning_window_uses_completed_exchange_sessions_and_current_clicked_rows(tmp_path):
    """The selector walks exchange sessions, skips Labor Day and clamps future ends.

    It must read the real append-only ledger through prediction_ledger, not a
    hand-made count.  The last stored grade has a pending sibling, so 3/5
    proves current-grades filtering as well as session slicing.
    """
    import market_calendar
    import session_review

    assert all(market_calendar.is_session(datetime.fromisoformat(day).date()) for day in SESSIONS_21)
    assert not market_calendar.is_session(datetime(2026, 9, 7).date())
    root = _ledger(tmp_path)

    five = session_review.read_learning_window(
        end_session="2030-01-01", sessions=5, root=root, trades=[],
        environment_labels={}, now=LAST_COMPLETED,
    )
    ten = session_review.read_learning_window(
        end_session="2026-09-21", sessions=10, root=root, trades=[],
        environment_labels={}, now=LAST_COMPLETED,
    )
    twenty = session_review.read_learning_window(
        end_session="2026-09-21", sessions=20, root=root, trades=[],
        environment_labels={}, now=LAST_COMPLETED,
    )

    assert five["window"]["requested"] == 5
    assert five["window"]["sessions"] == LAST_FIVE
    assert five["window"]["start"] == "2026-09-15"
    assert five["window"]["end"] == "2026-09-21"
    assert ten["window"]["sessions"] == LAST_TEN
    assert twenty["window"]["sessions"] == LAST_TWENTY
    assert "2026-09-07" not in twenty["window"]["sessions"]
    day = five["reads"]["horizons"]["rest_of_day"]["accuracy"]
    assert (day["right"], day["wrong"], day["pending"], day["n"], day["rate"]) == (3, 2, 0, 5, pytest.approx(0.60))
    with pytest.raises(ValueError):
        session_review.read_learning_window(
            end_session="2026-09-21", sessions=6, root=root, trades=[],
            environment_labels={}, now=LAST_COMPLETED,
        )


def test_learning_window_keeps_horizons_and_group_baselines_on_the_same_stamps(tmp_path):
    """The 10-session selected population contains 10 day calls and 1 D1 call.

    A reader that pools horizons would print 6.  Group rows must carry their
    owner-produced accuracy/baselines, supporting read ids and distinct sessions;
    the unknown D1 context stays a named coverage fact rather than a direction.
    """
    import market_read_grades as grades
    import session_review

    payload = session_review.read_learning_window(
        end_session="2026-09-21", sessions=10, root=_ledger(tmp_path), trades=[],
        environment_labels={}, now=LAST_COMPLETED,
    )
    horizons = payload["reads"]["horizons"]
    assert horizons["rest_of_day"]["accuracy"]["n"] == 10
    assert horizons["next_5_sessions"]["accuracy"]["n"] == 1
    assert horizons["next_5_sessions"]["accuracy"]["pending"] == 1
    assert all(cell.get("n") != 11 for cell in (horizons["rest_of_day"]["accuracy"], horizons["next_5_sessions"]["accuracy"]))

    grouped = [*payload["by_hour"], *payload["by_environment"]]
    assert grouped
    for row in grouped:
        assert {"horizon", "key", "accuracy", "baselines", "session_count", "read_ids", "sessions"} <= set(row), row
        assert row["session_count"] == len(set(row["sessions"])), row
        assert set(row["baselines"]) == set(grades.BASELINES), row
        assert set(row["read_ids"]), row
    assert any("America/New_York" in str(row["key"]) for row in payload["by_hour"])
    assert any("unknown" in str(row["key"]).lower() for row in payload["by_environment"])


def test_learning_window_uses_journal_closed_dates_and_prior_entry_environment(tmp_path):
    """Actual JournalStore rows keep closed-window membership and point-in-time labels.

    T-OLD closes before the five-session start and is excluded even though it is
    otherwise a valid closed trade.  T-DAY opens before the 09-18 close, so its
    entry D1 environment is 09-17's label, never the later same-day label.
    T-CAD keeps its currency and provisional tag visible; no setup is made
    proven by a machine-owned tag.
    """
    from journal_store import JournalStore
    import session_review

    store = JournalStore(tmp_path / "journal.sqlite3")
    _closed_trade(store, "T-DAY", opened_at="2026-09-18T09:45:00-04:00", closed_at="2026-09-18T14:10:00-04:00", net_pnl=100)
    _closed_trade(store, "T-CROSS", opened_at="2026-09-14T10:00:00-04:00", closed_at="2026-09-16T15:30:00-04:00", net_pnl=-40)
    _closed_trade(store, "T-CAD", opened_at="2026-09-15T10:00:00-04:00", closed_at="2026-09-21T15:30:00-04:00", net_pnl=75, currency="CAD")
    _closed_trade(store, "T-OLD", opened_at="2026-09-11T10:00:00-04:00", closed_at="2026-09-11T15:30:00-04:00", net_pnl=999)
    store.save_trade_annotation("T-CAD", setup_tags="machine-guess", notes="")
    # The normal owner uses this state to distinguish the machine proposal.
    with store.connection() as conn:
        conn.execute("UPDATE trade_annotations SET tag_status = 'provisional' WHERE trade_id = 'T-CAD'")

    payload = session_review.read_learning_window(
        end_session="2026-09-21", sessions=5, root=_ledger(tmp_path),
        trades=_current_trades(store),
        environment_labels={"2026-09-17": "trending_up", "2026-09-18": "trending_down"},
        now=LAST_COMPLETED,
    )
    rows = list(payload["trade_rows"])
    by_id = {str(row["trade_id"]): row for row in rows}
    assert _ids(rows) == {"T-DAY", "T-CROSS", "T-CAD"}
    assert by_id["T-DAY"]["closed_session"] == "2026-09-18"
    assert by_id["T-DAY"]["environment"] == "trending_up"
    assert by_id["T-CAD"]["currency"] == "CAD"
    assert by_id["T-CAD"]["tag_status"] == "provisional"
    assert "proven" not in str(by_id["T-CAD"]).lower()
    assert payload["coverage"]["trades"]["unknown_context"] >= 1


pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QAbstractButton, QLabel  # noqa: E402

_app = QApplication.instance() or QApplication([])


class _WindowReader:
    """A worker reader whose first answer waits until a newer choice exists."""

    def __init__(self):
        self.calls: list[int] = []
        self.first_entered = threading.Event()
        self.release_first = threading.Event()
        self.second_entered = threading.Event()

    def __call__(self, **kwargs):
        choice = int(kwargs.get("window_sessions") or 5)
        self.calls.append(choice)
        if len(self.calls) == 1:
            self.first_entered.set()
            assert self.release_first.wait(5), "test did not release the first worker"
        else:
            self.second_entered.set()
        from ui.services.weekend_prep_service import empty_week_payload

        body = empty_week_payload("2026-W39")
        body["learning"] = {
            "schema": "session_learning_window_v1",
            "window": {"requested": choice, "sessions": [f"choice-{choice}"], "start": f"choice-{choice}", "end": f"choice-{choice}"},
            "reads": {"horizons": {}}, "by_hour": [], "by_environment": [],
            "trade_groups": [], "trade_rows": [], "coverage": {}, "error": "",
        }
        return body


def _select_window(page, count: int) -> None:
    wanted = str(count)
    for combo in page.findChildren(QComboBox):
        if any(combo.itemText(index).startswith(wanted) for index in range(combo.count())):
            combo.setCurrentIndex(next(index for index in range(combo.count()) if combo.itemText(index).startswith(wanted)))
            return
    for button in page.findChildren(QAbstractButton):
        if button.text().strip().startswith(wanted):
            button.click()
            return
    raise AssertionError("Week Review has no visible 5/10/20 session selector")


def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Qt worker did not reach the expected state")


def test_week_review_selector_reloads_latest_choice_and_rejects_stale_worker_payload(tmp_path, monkeypatch):
    """A visible 20-session selection must never display the earlier 5 payload."""
    from ui.services import weekend_prep_service as prep
    from ui.services.weekend_prep_service import WeekendPrepService
    from ui.panels.weekend_prep_panel import WeekReviewPage

    reader = _WindowReader()
    monkeypatch.setattr(prep, "read_week_review", reader)
    service = WeekendPrepService(state_path=tmp_path / "state.json", now=LAST_COMPLETED.replace(tzinfo=None))
    page = WeekReviewPage(service)
    try:
        page.reload()
        assert reader.first_entered.wait(5), "the first learning reader never started"
        _select_window(page, 20)
        reader.release_first.set()
        _until(reader.second_entered.is_set)
        _until(lambda: not getattr(page, "_reading", True))
        shown = "\n".join(widget.text() for widget in page.findChildren(QAbstractButton))
        shown += "\n".join(widget.toPlainText() for widget in page.findChildren(type(page.summary)))
        assert reader.calls == [5, 20]
        assert "choice-20" in shown
        assert "choice-5" not in shown
    finally:
        reader.release_first.set()
        page.shutdown()
        page.deleteLater()
        service.shutdown()


def test_week_day_card_click_emits_the_exact_day_review_session(tmp_path):
    """The navigation is a real card click, not a method-existence assertion."""
    from ui.services.weekend_prep_service import WeekendPrepService, empty_week_payload
    from ui.panels.weekend_prep_panel import WeekendPrepPanel

    service = WeekendPrepService(state_path=tmp_path / "state.json", now=LAST_COMPLETED.replace(tzinfo=None))
    panel = WeekendPrepPanel(service=service, focus_service=None)
    seen: list[str] = []
    try:
        panel.openSessionRequested.connect(seen.append)
        payload = empty_week_payload("2026-W39")
        payload.update({"sessions": ["2026-09-18"], "cards": [{"session": "2026-09-18", "has_facts": False}]})
        panel.week_review._render(payload)
        card = panel.week_review.day_cards[0]
        QTest.mouseClick(card, Qt.LeftButton)
        _app.processEvents()
        assert seen == ["2026-09-18"]
    finally:
        panel.shutdown()
        panel.deleteLater()
        service.shutdown()


def test_week_strip_keeps_a_shaped_read_line_per_horizon():
    """A's separated read cells must not fall back to an old mixed scalar rate."""
    from ui.panels.weekend_prep_panel import week_strip_cell

    line = {
        "key": "your_reads", "n": 4, "measured": 4, "rate": 1.0,
        "horizons": {
            "rest_of_day": {
                "label": "Rest of day", "accuracy": {"n": 4, "right": 3, "rate": 0.75},
                "meets_floor": False,
            },
            "next_5_sessions": {
                "label": "Next 5 sessions", "accuracy": {"n": 1, "right": 1, "rate": 1.0},
                "meets_floor": False,
            },
        },
    }

    shown = week_strip_cell(line)
    assert "Rest of day: n 4 · right 3 · waiting 0 · too few to call" in shown
    assert "Next 5 sessions: n 1 · right 1 · waiting 0 · too few to call" in shown
    assert "100%" not in shown


def test_learning_tables_show_owner_counts_and_read_drilldown(tmp_path):
    """The actual page renders the selected owner values and links a read to its day."""
    import session_review
    from ui.panels.weekend_prep_panel import WeekReviewPage
    from ui.services.weekend_prep_service import WeekendPrepService, empty_week_payload

    learning = session_review.read_learning_window(
        end_session="2026-09-21", sessions=5, root=_ledger(tmp_path), trades=[],
        environment_labels={}, now=LAST_COMPLETED,
    )
    service = WeekendPrepService(state_path=tmp_path / "state.json", now=LAST_COMPLETED.replace(tzinfo=None))
    page = WeekReviewPage(service)
    seen: list[str] = []
    wrong_trade: list[str] = []
    try:
        page.openSessionRequested.connect(seen.append)
        page.openTradeRequested.connect(wrong_trade.append)
        payload = empty_week_payload("2026-W39")
        payload["learning"] = learning
        page._render(payload)
        assert page.learning_reads.item(0, 0).text() == "Rest of day"
        assert page.learning_reads.item(0, 1).text() == "5"
        assert page.learning_reads.item(0, 7).text() == "too few to call"
        assert "Confidence" in page.learning_reads.horizontalHeaderItem(10).text()
        assert page.learning_hours.rowCount() > 0
        assert page.learning_environments.rowCount() > 0
        assert page.learning_trades.rowCount() > 0
        page.learning_hours.cellClicked.emit(0, 0)
        _app.processEvents()
        buttons = page.learning_drilldown.findChildren(QAbstractButton)
        assert buttons
        buttons[0].click()
        assert seen == [learning["by_hour"][0]["sessions"][0]]
        assert not wrong_trade, "a read ID must never open as a Journal trade"
        assert any("Read IDs" in label.text() for label in page.learning_drilldown.findChildren(QLabel))
    finally:
        page.shutdown()
        page.deleteLater()
        service.shutdown()
