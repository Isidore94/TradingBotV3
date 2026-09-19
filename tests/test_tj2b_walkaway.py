"""TJ-2B — the four Day Review walk-away tables, RED before the build.

These fixtures are deliberately plain dictionaries.  They model only durable
evidence already read by the Day Review worker; no test may fetch a quote or
read a live store.
"""

from __future__ import annotations

from datetime import datetime
import os
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


SESSION = "2026-09-16"
EXIT_SESSION = "2026-09-18"


@pytest.fixture(scope="module")
def qapp():
    """This file owns its offscreen Qt application, like the TJ-1 page tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6", reason="the Day Review page uses PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _decision(
    *,
    stamp: str,
    verdict: str = "like",
    source: str = "annotations",
    symbol: str = "AAA",
    side: str = "LONG",
    category: str = "chart_review",
    timeframe: str = "M5",
    capture_id: str = "",
) -> dict:
    return {
        "session_date": SESSION,
        "symbol": symbol,
        "side": side,
        "category": category,
        "verdict": verdict,
        "source": source,
        "timeframe": timeframe,
        "stamp": stamp,
        "capture_id": capture_id,
    }


def _bars(*, session: str = SESSION, high: float = 115.0) -> dict[str, list[dict]]:
    return {
        "AAA": [
            {"dt": f"{session}T09:55:00-07:00", "open": 100, "high": 110, "low": 99, "close": 105},
            {"dt": f"{session}T10:00:00-07:00", "open": 105, "high": high, "low": 104, "close": 110},
            {"dt": f"{session}T15:55:00-07:00", "open": 110, "high": high - 1, "low": 108, "close": 109},
        ]
    }


def _build(*, decisions=(), bars=None, trades=(), claims=(), preference=(), outcomes=(), now=None):
    from walkaway_day import build

    return build(
        SESSION,
        sources={"decisions": tuple(decisions), "preference": tuple(preference), "outcomes": tuple(outcomes)},
        bars=bars if bars is not None else _bars(),
        trades=tuple(trades),
        claims=tuple(claims),
        now=now or datetime(2026, 9, 19, 8, 0),
    )


def _all_rows(day):
    return tuple(day.liked_not_traded + day.rejected + day.traded_left_early + day.claimed_d1)


def test_decision_identity_keeps_two_times_and_deduplicates_only_duplicate_sources():
    """A like at 08:00 and pass at 10:00 are two decisions, not one name.

    The duplicate pick-feedback like has the same decision identity as the
    annotation and therefore contributes no third row.
    """
    day = _build(
        decisions=(
            _decision(stamp="2026-09-16T08:00:00-07:00"),
            _decision(stamp="2026-09-16T08:00:00-07:00", source="pick_feedback"),
            _decision(stamp="2026-09-16T10:00:00-07:00", verdict="pass"),
        )
    )
    rows = _all_rows(day)
    assert len(rows) == 2
    assert {row.decision_id for row in rows} == {
        (SESSION, "AAA", "LONG", "chart_review", "like", "M5", "2026-09-16T08:00:00-07:00"),
        (SESSION, "AAA", "LONG", "chart_review", "pass", "M5", "2026-09-16T10:00:00-07:00"),
    }


def test_ran_after_uses_best_move_from_first_completed_bar_after_stamp():
    day = _build(decisions=(_decision(stamp="2026-09-16T09:57:00-07:00", verdict="pass"),))
    row = day.rejected[0]
    # The 09:55 bar is before the decision.  The first eligible completed bar
    # opens at 105 and the best later high is 115, so 10/105 not 15/100.
    assert row.ran_after_pct == pytest.approx(10 / 105 * 100)
    assert row.held_at_close_pct is None
    assert row.state == "measured"


def test_later_matched_trade_moves_like_to_c_and_uses_exit_day_bars():
    trade = {
        "trade_id": "t-1",
        "symbol": "AAA",
        "direction": "LONG",
        "status": "closed",
        "opened_at": "2026-09-18T10:00:00-07:00",
        "closed_at": "2026-09-18T11:00:00-07:00",
        "last_closing_leg_at": "2026-09-18T11:00:00-07:00",
        "net_pnl": 42.0,
    }
    # The enormous 11:00 high belongs to the exit bar, so it is unavailable
    # after that close.  The first completed bar strictly AFTER the last
    # closing leg is 11:05, opening at 120.
    exit_bars = {
        "AAA": [
            {
                "dt": "2026-09-18T10:55:00-07:00",
                "open": 100,
                "high": 200,
                "low": 99,
                "close": 150,
            },
            {
                "dt": "2026-09-18T11:00:00-07:00",
                "open": 150,
                "high": 250,
                "low": 149,
                "close": 160,
            },
            {
                "dt": "2026-09-18T11:05:00-07:00",
                "open": 120,
                "high": 130,
                "low": 119,
                "close": 125,
            },
            {
                "dt": "2026-09-18T15:55:00-07:00",
                "open": 125,
                "high": 129,
                "low": 124,
                "close": 128,
            },
        ]
    }
    day = _build(
        decisions=(_decision(stamp="2026-09-16T08:00:00-07:00"),),
        trades=(trade,),
        preference=({"match_state": "matched", "symbol": "AAA", "side": "LONG", "trade_id": "t-1"},),
        bars={SESSION: _bars()["AAA"], EXIT_SESSION: exit_bars["AAA"]},
    )
    assert not day.liked_not_traded
    row = day.traded_left_early[0]
    assert "liked 09-16, entered 09-18" in row.what_you_did
    assert row.left_on_table_pct == pytest.approx(10 / 120 * 100)
    assert row.state == "measured"

    missing = _build(
        decisions=(_decision(stamp="2026-09-16T08:00:00-07:00"),),
        trades=(trade,),
        preference=({"match_state": "matched", "symbol": "AAA", "side": "LONG", "trade_id": "t-1"},),
        bars=_bars(),
    ).traded_left_early[0]
    assert missing.state == "unmeasured no_bars (exit 2026-09-18)"


def test_left_on_table_never_falls_back_to_a_pre_exit_bar():
    trade = {
        "trade_id": "t-pre-exit",
        "symbol": "AAA",
        "direction": "LONG",
        "status": "closed",
        "opened_at": "2026-09-18T10:00:00-07:00",
        "closed_at": "2026-09-18T11:00:00-07:00",
        "last_closing_leg_at": "2026-09-18T11:00:00-07:00",
    }
    pre_exit_only = {
        EXIT_SESSION: [
            {
                "dt": "2026-09-18T10:55:00-07:00",
                "open": 100,
                "high": 300,
                "low": 99,
                "close": 200,
            },
            {
                "dt": "2026-09-18T11:00:00-07:00",
                "open": 200,
                "high": 400,
                "low": 199,
                "close": 300,
            },
        ]
    }
    row = _build(
        decisions=(_decision(stamp="2026-09-16T08:00:00-07:00"),),
        trades=(trade,),
        preference=({"match_state": "matched", "symbol": "AAA", "side": "LONG", "trade_id": "t-pre-exit"},),
        bars=pre_exit_only,
    ).traded_left_early[0]
    assert row.left_on_table_pct is None
    assert row.state == "unmeasured no_bars (exit 2026-09-18)"


def test_open_later_trade_is_pending_c_while_window_open_like_stays_a():
    open_trade = {
        "trade_id": "open",
        "symbol": "AAA",
        "direction": "LONG",
        "status": "open",
        "opened_at": "2026-09-18T10:00:00-07:00",
    }
    day = _build(
        decisions=(_decision(stamp="2026-09-16T08:00:00-07:00"),),
        trades=(open_trade,),
        preference=({"match_state": "matched", "symbol": "AAA", "side": "LONG", "trade_id": "open"},),
    )
    assert day.traded_left_early[0].state == "pending trade open"
    assert day.traded_left_early[0].left_on_table_pct is None

    waiting = _build(
        decisions=(_decision(stamp="2026-09-16T08:00:00-07:00"),),
        preference=({"match_state": "window_open", "symbol": "AAA", "side": "LONG"},),
    ).liked_not_traded[0]
    assert waiting.traded == "window"


def test_blank_legacy_preference_trade_id_never_claims_an_unrelated_trade():
    trade = {"trade_id": "later", "symbol": "AAA", "direction": "LONG", "status": "closed", "opened_at": "2026-09-18T10:00:00-07:00"}
    day = _build(decisions=(_decision(stamp="2026-09-16T08:00:00-07:00"),), trades=(trade,), preference=({"match_state": "matched", "symbol": "AAA", "side": "LONG"},))
    assert len(day.liked_not_traded) == 1
    assert not day.traded_left_early


def test_claim_history_replays_by_claim_key_and_claimed_annotation_routes_only_d():
    claim = {
        "action": "claim",
        "session_date": SESSION,
        "symbol": "AAA",
        "side": "LONG",
        "claimed_setup_id": "breakout",
        "horizon": "d1",
        "annotation_ref": "like-1",
    }
    dropped = {**claim, "action": "drop", "session_date": "2026-09-17"}
    day = _build(
        decisions=(_decision(stamp="2026-09-16T08:00:00-07:00", capture_id="like-1"),),
        claims=(claim, dropped),
        outcomes=(
            {
                "scan_date": SESSION,
                "symbol": "AAA",
                "side": "LONG",
                "horizon_sessions": 5,
                "measured": False,
                "maturity_date": "2026-09-23",
            },
        ),
    )
    assert not day.liked_not_traded, "a claimed D1 annotation is D, never also A"
    row = day.claimed_d1[0]
    assert row.category == "breakout"
    assert row.state == "dropped 2026-09-17; pending 2026-09-23"
    assert row.decision_id[-1] == "2026-09-16T08:00:00-07:00"


def test_claim_horizon_m5_and_blank_are_not_coerced_to_d1():
    claims = (
        {"action": "claim", "session_date": SESSION, "symbol": "AAA", "side": "LONG", "horizon": "m5"},
        {"action": "claim", "session_date": SESSION, "symbol": "BBB", "side": "SHORT", "horizon": ""},
    )
    day = _build(
        claims=claims,
        outcomes=({"session_date": SESSION, "symbol": "AAA", "side": "LONG", "eod_move_pct": 6.0},),
        bars={},
    )
    m5, blank = day.claimed_d1
    assert m5.state == "measured"
    assert m5.held_at_close_pct == 6.0
    assert blank.category == "claim"
    assert blank.state == "unmeasured no_horizon"


def test_service_and_page_use_one_worker_payload_for_four_tables_and_emit_chart_signal(qapp, monkeypatch):
    """The page consumes the already-built four tables; it starts no extra read."""
    from ui.services.day_review_service import DayReviewService
    from ui.panels.day_review_panel import DayReviewPanel

    expected = _build(decisions=(_decision(stamp="2026-09-16T09:57:00-07:00", verdict="pass"),))
    monkeypatch.setattr("walkaway_day.build", lambda *_a, **_k: expected)
    payload = DayReviewService().read_day(SESSION, now=datetime(2026, 9, 17, 8, 0))
    assert payload["walkaway"] is expected
    assert {"liked_not_traded", "rejected", "traded_left_early", "claimed_d1"} <= set(payload["walkaway"])

    panel = DayReviewPanel(service=object(), clock=lambda: datetime(2026, 9, 17, 8, 0))
    seen: list[tuple[str, str]] = []
    panel.chartRequested.connect(lambda symbol, side: seen.append((symbol, side)))
    panel.render(payload)
    assert {name: table.rowCount() for name, table in panel.walkaway_tables.items()} == {
        "rejected": 1,
        "liked_not_traded": 0,
        "traded_left_early": 0,
        "claimed_d1": 0,
        # TJ-11's fifth population; this fixture supplies no earlier decisions.
        "earlier_calls": 0,
    }
    panel.walkaway_tables["rejected"].itemActivated.emit(panel.walkaway_tables["rejected"].item(0, 1))
    assert seen == [("AAA", "LONG")]
    panel.shutdown()
    panel.deleteLater()


def test_each_walkaway_table_activates_its_own_row(qapp):
    """A liked row must not accidentally chart the rejected table's row."""
    from ui.panels.day_review_panel import DayReviewPanel

    day = _build(decisions=(_decision(stamp="2026-09-16T08:00:00-07:00", verdict="like"),))
    panel = DayReviewPanel(service=object(), clock=lambda: datetime(2026, 9, 17, 8, 0))
    seen = []
    panel.chartRequested.connect(lambda symbol, side: seen.append((symbol, side)))
    panel.render({"session_date": SESSION, "walkaway": day})
    table = panel.walkaway_tables["liked_not_traded"]
    table.itemActivated.emit(table.item(0, 1))
    assert seen == [("AAA", "LONG")]
    panel.shutdown()
    panel.deleteLater()
