"""P8 P7: the Movers board's rally state (Rip-strong / Rip-weak) and the Pop outcome log.

Synthetic bars only. SPY bars are naive Los Angeles wall time (06:30 LA = 09:30 NY),
the shape `bot.m5_chart_bars` hands out.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import movers_scan as ms  # noqa: E402

LA = ZoneInfo("America/Los_Angeles")
NY = ZoneInfo("America/New_York")
PRIOR = date(2026, 9, 21)
TODAY = date(2026, 9, 22)
FLAT_BASELINE = {offset: 100_000.0 for offset in range(78)}


def _bars(day, closes, *, volume=100_000.0, first_open=None, wick=0.05):
    out = []
    previous = closes[0] if first_open is None else first_open
    start = datetime(day.year, day.month, day.day, 6, 30)
    for index, close in enumerate(closes):
        out.append({
            "dt": start + timedelta(minutes=5 * index), "open": previous,
            "high": max(previous, close) + wick, "low": min(previous, close) - wick,
            "close": close, "volume": volume,
        })
        previous = close
    return out


def _series(closes, *, prior_close=100.0):
    return _bars(PRIOR, [prior_close] * 78) + _bars(TODAY, closes, first_open=prior_close)


def _today(bars):
    return [b for b in bars if b["dt"].date() == TODAY]


def _now(n):
    return datetime(TODAY.year, TODAY.month, TODAY.day, 6, 30, tzinfo=LA) + timedelta(minutes=5 * n)


def _state(bars, n):
    return ms.market_state(ms.normalize_bars(bars, now=_now(n), local_tz=LA))


def _plain_rally():
    """Up day, no pullback: the session low is bar 2 (closed above VWAP), then a steady climb."""
    closes = [401.0, 401.2, 401.0, 401.5, 402.0, 402.5, 403.0]
    bars = _series(closes, prior_close=400.0)
    today = _today(bars)
    today[0]["low"] = 400.0
    today[2]["low"] = 399.8  # the swing low, after the first two bars
    return bars, len(closes)


def _pullback_then_rally(bottom=402.0, now_close=403.5):
    """Rally to a 10:15 high, pull back to `bottom` at bar 12, then rally to `now_close`."""
    closes = [400.0, 400.0] + [405.0] * 8 + [404.0, 403.0, bottom, now_close - 0.5, now_close]
    bars = _series(closes, prior_close=400.0)
    today = _today(bars)
    today[0]["volume"] = today[1]["volume"] = 1_000_000.0
    today[9]["high"] = 405.6  # the session high, above VWAP
    return bars, len(closes)


# ------------------------------------------------------------------ market state
def test_plain_rally_lights_rally_from_the_session_low():
    bars, n = _plain_rally()
    state = _state(bars, n)
    assert state.rally is True
    assert state.pullback is False and state.bounce is False
    assert state.state == "up_day"
    assert state.extreme_time == "09:40"  # bar 2
    assert state.extreme_price == pytest.approx(399.8)
    assert state.start_dt is not None and state.start_dt.strftime("%H:%M") == "09:40"
    assert state.spy_from_extreme_pct >= ms.PULLBACK_MIN_PCT
    assert state.to_dict()["rally"] is True


def test_rally_below_the_threshold_lights_nothing():
    closes = [401.0, 401.2, 401.0, 401.1, 401.0, 401.0]
    bars = _series(closes, prior_close=400.0)
    _today(bars)[0]["low"] = 400.9
    _today(bars)[2]["low"] = 400.5  # last 401.0 is only +0.12% off it
    state = _state(bars, len(closes))
    assert state.rally is False and state.start_dt is None


def test_a_low_in_the_first_two_bars_waits_for_the_open_low_bar_count():
    # Lead 2026-09-25: an open low may start a rally only after RALLY_OPEN_LOW_MIN_BARS bars.
    closes = [401.0, 401.2, 401.3, 401.5, 402.0, 402.5]  # bar 5: 5 bars after the low
    bars = _series(closes, prior_close=400.0)
    _today(bars)[0]["low"] = 399.0
    state = _state(bars, len(closes))
    assert state.rally is False and state.start_dt is None
    bars = _series(closes + [403.0], prior_close=400.0)  # bar 6: 6 bars after the low
    _today(bars)[0]["low"] = 399.0
    state = _state(bars, len(closes) + 1)
    assert state.rally is True and state.extreme_time == "09:30"


def _open_low_grind(last_bar):
    """SPY opens at its low (bar 0) and grinds up 0.3 a bar through `last_bar`."""
    closes = [400.0 + 0.3 * i for i in range(last_bar + 1)]
    bars = _series(closes, prior_close=400.0)
    _today(bars)[0]["low"] = 399.8
    return bars, len(closes)


def test_an_open_low_grind_lights_rally_at_bar_8_not_bar_5():
    bars, n = _open_low_grind(5)
    state = _state(bars, n)
    assert (401.5 / 399.8 - 1) * 100 >= ms.PULLBACK_MIN_PCT  # far enough, too soon
    assert state.rally is False and state.start_dt is None
    bars, n = _open_low_grind(8)
    state = _state(bars, n)
    assert state.rally is True and state.pullback is False
    assert state.extreme_time == "09:30" and state.extreme_price == pytest.approx(399.8)
    assert ms.RALLY_OPEN_LOW_MIN_BARS == 6


def test_rally_after_a_pullback_starts_at_the_pullback_low_and_the_later_turn_wins():
    bars, n = _pullback_then_rally(bottom=402.0, now_close=403.5)
    state = _state(bars, n)
    # Both qualify: 403.5 is > 0.30% under the 405.6 high AND > 0.30% over the 401.95 low.
    assert (403.5 / 405.6 - 1) * 100 <= -ms.PULLBACK_MIN_PCT
    assert state.rally is True and state.pullback is False
    assert state.extreme_time == "10:30"  # bar 12, the pullback's low
    assert state.extreme_price == pytest.approx(401.95)


def test_rally_ends_when_a_new_down_leg_makes_the_last_bar_the_swing_low():
    closes = [401.0, 401.2, 401.0, 401.5, 402.0, 402.5, 403.0, 401.6]
    bars = _series(closes, prior_close=400.0)
    _today(bars)[0]["low"] = 400.0
    _today(bars)[2]["low"] = 399.8
    state = _state(bars, len(closes))
    assert state.rally is False


def test_a_bounce_off_the_same_low_stays_a_bounce():
    closes = [400.0, 400.0] + [395.0] * 8
    closes += [395.0 + 395.0 * 0.004 * k / 3 for k in (1, 2, 3)]
    bars = _series(closes, prior_close=400.0)
    today = _today(bars)
    today[0]["volume"] = today[1]["volume"] = 1_000_000.0
    today[9]["low"] = 394.4
    state = _state(bars, len(closes))
    assert state.bounce is True and state.rally is False


def test_a_pullback_still_live_after_a_small_bounce_is_not_a_rally():
    bars, n = _pullback_then_rally(bottom=402.0, now_close=402.8)  # +0.21% off the low
    state = _state(bars, n)
    assert state.pullback is True and state.rally is False


# ------------------------------------------------------------------ rip lists
def _rip_board(extra=None):
    spy, n = _pullback_then_rally(bottom=402.0, now_close=403.5)
    base = [100.0] * 13  # through the rally start (bar 12)
    series = {
        "LEAD": _series(base + [100.8, 101.5]),  # beats SPY's +0.37% since the start
        "LAG": _series(base + [100.0, 100.0]),  # flat while SPY rips
        "SINK": _series(base + [99.5, 99.0]),  # falling into the rip
    }
    series.update(extra or {})
    return ms.build_movers_board(
        series, spy, now=_now(n), baselines={s: FLAT_BASELINE for s in series}, local_tz=LA,
    )


def test_rally_fills_rip_lists_by_excess_vs_spy_and_leaves_dip_empty():
    board = _rip_board()
    assert board["state"]["rally"] is True
    assert board["dip"] == {"long": [], "short": []}
    assert [r["symbol"] for r in board["rip"]["long"]] == ["LEAD"]
    assert [r["symbol"] for r in board["rip"]["short"]] == ["SINK", "LAG"]
    assert all(r["dip_score"] < 0 for r in board["rip"]["short"])
    lead = board["rip"]["long"][0]
    assert lead["since_start_pct"] == pytest.approx(1.5)


def test_rip_lists_keep_the_floors_and_top_n():
    cheap = _series([1.0] * 13 + [0.9, 0.8], prior_close=1.0)
    board = _rip_board({"CHEAP": cheap})
    assert "CHEAP" not in [r["symbol"] for r in board["rip"]["short"]]
    many = {f"W{i:02d}": _series([100.0] * 13 + [99.9 - i * 0.01, 99.8 - i * 0.01])
            for i in range(ms.MOVERS_TOP_N + 5)}
    board = _rip_board(many)
    assert len(board["rip"]["short"]) == ms.MOVERS_TOP_N


def test_rip_lists_are_empty_outside_a_rally():
    spy, n = _pullback_then_rally(bottom=402.0, now_close=402.8)
    board = ms.build_movers_board({"X": _series([100.0] * 15)}, spy, now=_now(n), local_tz=LA)
    assert board["rip"] == {"long": [], "short": []}


def test_rip_lists_get_persistence_and_group_tags():
    board = _rip_board()
    memory = ms.apply_persistence(board, {}, session=TODAY)
    top = board["rip"]["short"][0]
    assert top["streak"] == 1 and top["rank_change"] is None
    assert "rip:short" in memory["lists"]
    ms.apply_group_tags(board, {"SINK": "Semiconductors", "LAG": "Semiconductors"})
    assert "rip" in board["groups"]


# ------------------------------------------------------------------ board widget
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _row(symbol, **values):
    base = {"symbol": symbol, "move15_pct": 1.0, "move30_pct": 1.5, "day_pct": 2.0,
            "rvol": 2.0, "vs_spy15_pct": 0.8, "pop_score": 1.0, "dip_score": None,
            "since_start_pct": None, "note": "", "stale": False}
    base.update(values)
    return base


def _rally_widget_board(start="2026-09-22T10:30:00-04:00"):
    state = {"state": "up_day", "pullback": False, "bounce": False, "rally": True,
             "extreme_time": "10:30", "start_dt": start, "spy_from_extreme_pct": 0.38,
             "spy_day_pct": 0.9}
    return {
        "as_of": "2026-09-22T11:00:00-04:00",
        "state": state,
        "pop": {"long": [_row("AAA")], "short": []},
        "dip": {"long": [], "short": []},
        "rip": {"long": [_row("LEAD", dip_score=1.1, since_start_pct=1.5)],
                "short": [_row("SINK", dip_score=-1.4, since_start_pct=-1.0, streak=1,
                               rank_change=None),
                          _row("LAG", dip_score=-0.6, since_start_pct=0.0)]},
        "mine": {"long": [], "short": []},
    }


def _widget(app):
    from ui.widgets.movers_board import MoversBoard

    board = MoversBoard(persist=False)
    board.resize(420, 400)
    return board


def _names(section):
    return [row["symbol"] for row in section.visible_rows()]


def test_rally_shows_rip_tables_under_pop_with_a_rally_banner(app):
    widget = _widget(app)
    widget.update_board(_rally_widget_board())
    widget.flush_pending_refresh()
    assert not widget.strong.isHidden() and not widget.weak.isHidden()
    assert widget.dip_hint.isHidden()
    assert _names(widget.strong) == ["LEAD"]
    assert _names(widget.weak) == ["SINK", "LAG"]
    assert widget.strong.title_label.text().startswith("Rip-strong")
    assert widget.weak.title_label.text().startswith("Rip-weak")
    assert "low" in widget.weak.title_label.text()
    banner = widget.banner.text()
    assert "RALLY" in banner and "+0.38%" in banner and "low" in banner
    assert "●" in widget.mode_buttons["pop"].text()


def test_rip_weak_rows_are_shorts_and_sort_hide_plus_focus_and_new_cell_work(app):
    from PySide6.QtCore import Qt

    widget = _widget(app)
    widget.update_board(_rally_widget_board())
    widget.flush_pending_refresh()
    clicked, asked = [], []
    widget.symbolActivated.connect(lambda s, side: clicked.append((s, side)))
    widget.focusAddRequested.connect(lambda s, side: asked.append((s, side)))
    widget._on_clicked(widget.weak.proxy.index(0, 0))
    assert clicked == [("SINK", "SHORT")]
    # The first tick on the list tints the Sym cell.
    assert widget.weak.model.data(widget.weak.model.index(0, 0),
                                  Qt.ItemDataRole.BackgroundRole) is not None
    # Header click sorts (xSPY biggest first flips the weak list), third click restores.
    column = [k for k, _h in widget.weak.columns()].index("dip_score")
    widget.weak._on_header_clicked(column)
    assert _names(widget.weak) == ["LAG", "SINK"]
    widget.weak._on_header_clicked(column)
    widget.weak._on_header_clicked(column)
    assert _names(widget.weak) == ["SINK", "LAG"]
    widget.weak.table.selectRow(0)
    widget.add_focus_button.click()
    assert asked == [("SINK", "short")]
    widget._hide_selected()
    assert _names(widget.weak) == ["LAG"]


def test_a_rally_does_not_pull_the_trader_off_my_names(app):
    widget = _widget(app)
    widget.set_mode("mine")
    widget.update_board(_rally_widget_board())
    widget.flush_pending_refresh()
    assert widget.mode == "mine"


def test_no_turn_hint_names_the_rally_too(app):
    widget = _widget(app)
    widget.update_board({"state": {"state": "up_day"}, "pop": {}, "dip": {}, "rip": {},
                         "mine": {}})
    widget.flush_pending_refresh()
    assert "rally" in widget.dip_hint.text()


def test_plus_focus_on_a_rip_only_row_goes_through_the_gate(tmp_path, app):
    from ui.panels.alert_center_panel import AlertCenterPanel

    class Focus:
        def __init__(self):
            self.added = []

        def add(self, symbol, side, category="m5", *, origin="", context=""):
            self.added.append((symbol, side))
            return True

    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    panel.focus_service = Focus()
    board = _rally_widget_board()
    board["rip"]["short"][0].update(last=95.0, prev_high=101.0, prev_low=97.0,
                                    session_vwap=96.0)
    panel.movers_board.update_board(board)
    panel.movers_board.flush_pending_refresh()
    panel.movers_board.focusAddRequested.emit("SINK", "short")
    assert "no longer on the board" not in panel.movers_board.status_label.text()
    assert panel.focus_service.added == [("SINK", "short")]


# ------------------------------------------------------------------ outcome logs
OPEN_NY = datetime(2026, 9, 22, 9, 30, tzinfo=NY)


def _ny_bars(closes, *, start=OPEN_NY, highs=None, lows=None):
    out, prev = [], closes[0]
    for i, close in enumerate(closes):
        out.append({"dt": start + timedelta(minutes=5 * i), "open": prev,
                    "high": (highs or {}).get(i, max(prev, close) + 0.1),
                    "low": (lows or {}).get(i, min(prev, close) - 0.1),
                    "close": close, "volume": 1000.0})
        prev = close
    return out


def _pop_board(long_rows=(), short_rows=()):
    def row(symbol, **extra):
        base = {"symbol": symbol, "move15_pct": 1.2, "rvol": 2.5, "from_vwap_atr": 1.1,
                "atr": 0.5, "last": 101.0, "pop_score": 2.0}
        base.update(extra)
        return base

    return {"state": {"state": "up_day"},
            "pop": {"long": [row(s) for s in long_rows],
                    "short": [row(s, move15_pct=-1.0, pop_score=-2.0) for s in short_rows]}}


RUN = _ny_bars([100.0] * 10 + [101.0] + [101.2, 101.5, 101.0, 101.4, 101.6, 102.0,
                                         101.8, 102.2, 102.4, 102.1, 102.5, 102.6],
               highs={12: 102.3}, lows={13: 100.6})


def _pop_tick(tracker, upto, board, series=None):
    series = series if series is not None else {"RUN": RUN[: upto + 1]}
    now = RUN[upto]["dt"] + timedelta(minutes=5, seconds=20)
    return tracker.observe(board, series, RUN[: upto + 1], now=now)


def test_pop_tracker_flags_once_and_writes_15_30_60_minute_outcomes_in_atr():
    import movers_outcomes as mo

    tracker = mo.PopOutcomeTracker()
    rows = _pop_tick(tracker, 10, _pop_board(["RUN"]))
    assert [r["kind"] for r in rows] == ["flag"]
    flag = rows[0]
    assert flag["symbol"] == "RUN" and flag["side"] == "long" and flag["rank"] == 1
    assert flag["flagged_bar"] == RUN[10]["dt"].isoformat()
    for key in ("move15_pct", "rvol", "from_vwap_atr"):
        assert flag[key] is not None, key
    outcomes = []
    for upto in range(11, 23):
        got = _pop_tick(tracker, upto, _pop_board(["RUN"]))
        assert all(r["kind"] == "outcome" for r in got)  # never re-flagged while listed
        outcomes += got
    assert [r["horizon_min"] for r in outcomes] == [15, 30, 60]
    base, atr = RUN[10]["close"], 0.5
    fifteen = outcomes[0]
    assert fifteen["move_pct"] == pytest.approx((RUN[13]["close"] / base - 1) * 100)
    assert fifteen["move_atr"] == pytest.approx((RUN[13]["close"] - base) / atr)
    assert fifteen["mfe_atr"] == pytest.approx((102.3 - base) / atr)  # bar 12's high
    assert fifteen["mae_atr"] == pytest.approx((base - 100.6) / atr)  # bar 13's low
    sixty = outcomes[2]
    assert sixty["move_atr"] == pytest.approx((RUN[22]["close"] - base) / atr)
    assert tracker.pending == {}


def test_pop_short_side_measures_in_the_short_direction():
    import movers_outcomes as mo

    tracker = mo.PopOutcomeTracker()
    _pop_tick(tracker, 10, _pop_board(short_rows=["RUN"]))
    out = []
    for upto in range(11, 14):
        out += _pop_tick(tracker, upto, _pop_board())
    assert out[0]["side"] == "short"
    assert out[0]["move_atr"] == pytest.approx((RUN[10]["close"] - RUN[13]["close"]) / 0.5)
    assert out[0]["mfe_atr"] == pytest.approx((RUN[10]["close"] - 100.6) / 0.5)


def test_pop_reflags_only_after_leaving_the_list_and_resolving():
    import movers_outcomes as mo

    tracker = mo.PopOutcomeTracker()
    assert len(_pop_tick(tracker, 10, _pop_board(["RUN"]))) == 1
    _pop_tick(tracker, 11, _pop_board())  # off the list
    # Back on while its +60 is still pending: same episode, no new flag.
    assert [r["kind"] for r in _pop_tick(tracker, 12, _pop_board(["RUN"]))] == []
    for upto in range(13, 23):
        _pop_tick(tracker, upto, _pop_board())
    assert tracker.pending == {}
    long_run = RUN + _ny_bars([102.7, 102.8], start=RUN[-1]["dt"] + timedelta(minutes=5))
    now = long_run[23]["dt"] + timedelta(minutes=5, seconds=20)
    rows = tracker.observe(_pop_board(["RUN"]), {"RUN": long_run[:24]}, long_run[:24], now=now)
    assert [r["kind"] for r in rows] == ["flag"]


def test_pop_session_close_writes_unknown_for_missing_horizons():
    import movers_outcomes as mo

    late = OPEN_NY.replace(hour=15, minute=40)
    bars = _ny_bars([100.0, 100.5, 101.0, 101.2], start=late)  # last bar 15:55
    tracker = mo.PopOutcomeTracker()
    now = bars[1]["dt"] + timedelta(minutes=5, seconds=20)
    tracker.observe(_pop_board(["RUN"]), {"RUN": bars[:2]}, bars[:2], now=now)
    now = bars[3]["dt"] + timedelta(minutes=5, seconds=20)
    rows = tracker.observe(_pop_board(["RUN"]), {"RUN": bars}, bars, now=now)
    assert [r["horizon_min"] for r in rows] == [15, 30, 60]
    assert rows[0]["complete"] is False and rows[0]["move_atr"] is None  # only 2 bars after
    assert rows[1]["move_atr"] is None and rows[2]["complete"] is False


def test_pop_restart_does_not_reflag_and_still_resolves():
    import movers_outcomes as mo

    first = mo.PopOutcomeTracker()
    flags = _pop_tick(first, 10, _pop_board(["RUN"]))
    fresh = mo.PopOutcomeTracker()
    fresh.restore(flags, session=OPEN_NY.date(), now=OPEN_NY)
    out = []
    for upto in range(11, 23):
        out += _pop_tick(fresh, upto, _pop_board(["RUN"]))
    assert [r["kind"] for r in out] == ["outcome"] * 3


def test_pop_rows_are_timezone_stamped_and_the_log_is_a_project_paths_constant():
    import movers_outcomes as mo
    import project_paths

    rows = _pop_tick(mo.PopOutcomeTracker(), 10, _pop_board(["RUN"]))
    assert rows[0]["recorded_at"].endswith("-04:00")
    assert project_paths.MOVERS_POP_OUTCOMES_FILE.name == "movers_pop_outcomes.jsonl"


# Dip tracker learns the rally lists.
SPY_R = _ny_bars([400.0, 400.0, 400.5, 401.0, 401.5, 402.0, 402.2, 402.4, 402.6, 402.8,
                  403.0, 403.2], lows={2: 399.5})
R_START = SPY_R[2]["dt"]


def _rally_board(on=True):
    state = {"state": "up_day", "pullback": False, "bounce": False, "rally": on,
             "start_dt": R_START.isoformat() if on else "", "extreme_price": 399.5,
             "spy_from_extreme_pct": 0.4}
    rip = {"long": [{"symbol": "LEAD", "dip_score": 1.0}],
           "short": [{"symbol": "SINK", "dip_score": -1.2}]} if on else {"long": [], "short": []}
    return {"state": state, "dip": {"long": [], "short": []}, "rip": rip}


def test_dip_tracker_flags_both_rip_lists_with_state_rally():
    import movers_outcomes as mo

    tracker = mo.DipOutcomeTracker()
    series = {"LEAD": SPY_R, "SINK": SPY_R}
    now = SPY_R[4]["dt"] + timedelta(minutes=5, seconds=20)
    rows = tracker.observe(_rally_board(), {k: v[:5] for k, v in series.items()}, SPY_R[:5],
                           now=now)
    assert sorted((r["side"], r["symbol"], r["state"]) for r in rows) == [
        ("long", "LEAD", "rally"), ("short", "SINK", "rally")]
    assert all(r["kind"] == "flag" for r in rows)
    out = []
    for upto in range(5, 12):
        now = SPY_R[upto]["dt"] + timedelta(minutes=5, seconds=20)
        out += tracker.observe(_rally_board(), {k: v[: upto + 1] for k, v in series.items()},
                               SPY_R[: upto + 1], now=now)
    outcomes = [r for r in out if r["kind"] == "outcome"]
    assert sorted(r["symbol"] for r in outcomes) == ["LEAD", "SINK"]
    assert all(r["state"] == "rally" for r in outcomes)
    assert all(r["end_reason"] == "six_bars" for r in outcomes)


def test_a_rally_episode_ends_when_spy_loses_the_rally_low():
    import movers_outcomes as mo

    tracker = mo.DipOutcomeTracker()
    now = SPY_R[4]["dt"] + timedelta(minutes=5, seconds=20)
    tracker.observe(_rally_board(), {"LEAD": SPY_R[:5], "SINK": SPY_R[:5]}, SPY_R[:5], now=now)
    spy = SPY_R[:5] + _ny_bars([399.0], start=SPY_R[4]["dt"] + timedelta(minutes=5))
    now = spy[-1]["dt"] + timedelta(minutes=5, seconds=20)
    tracker.observe(_rally_board(), {"LEAD": spy, "SINK": spy}, spy, now=now)
    reasons = {ep["end_reason"] for ep in tracker.episodes.values()}
    assert reasons == {"spy_lost_rally_low"}


def test_restore_keeps_the_rally_state():
    import movers_outcomes as mo

    tracker = mo.DipOutcomeTracker()
    now = SPY_R[4]["dt"] + timedelta(minutes=5, seconds=20)
    flags = tracker.observe(_rally_board(), {"LEAD": SPY_R[:5], "SINK": SPY_R[:5]},
                            SPY_R[:5], now=now)
    fresh = mo.DipOutcomeTracker()
    fresh.restore(flags, session=R_START.date(), now=now)
    assert {ep["state"] for ep in fresh.episodes.values()} == {"rally"}


def test_summary_keeps_rally_apart_and_the_cli_covers_both_files(tmp_path, capsys):
    import json

    import movers_outcomes as mo

    dip_rows = [
        {"kind": "outcome", "session": "2026-09-22", "episode": "a", "side": "long",
         "excess3_pct": 0.2, "excess6_pct": 0.5},
        {"kind": "outcome", "session": "2026-09-22", "episode": "b", "side": "short",
         "state": "rally", "excess3_pct": -0.1, "excess6_pct": -0.3},
    ]
    summary = mo.summarize(dip_rows)
    assert summary["all"]["outcomes"] == 1  # the pullback/bounce lists only
    assert summary["rally"]["all"]["outcomes"] == 1
    assert summary["rally"]["short"]["hit_rate"] == 0.0
    pop_rows = [
        {"kind": "flag", "session": "2026-09-22", "symbol": "RUN", "side": "long"},
        {"kind": "outcome", "session": "2026-09-22", "symbol": "RUN", "side": "long",
         "horizon_min": 15, "move_atr": 0.8, "mfe_atr": 1.0, "mae_atr": 0.2, "complete": True},
        {"kind": "outcome", "session": "2026-09-22", "symbol": "RUN", "side": "long",
         "horizon_min": 30, "move_atr": -0.4, "mfe_atr": 1.0, "mae_atr": 0.9, "complete": True},
        {"kind": "outcome", "session": "2026-09-22", "symbol": "RUN", "side": "long",
         "horizon_min": 60, "move_atr": None, "complete": False},
    ]
    pop = mo.summarize_pop(pop_rows)
    assert pop["flags"] == 1
    assert pop["by_horizon"]["15"]["hit_rate"] == 1.0
    assert pop["by_horizon"]["30"]["avg_mae_atr"] == pytest.approx(0.9)
    assert pop["by_horizon"]["60"]["graded"] == 0
    dip_path = tmp_path / "movers_dip_outcomes.jsonl"
    mo.append_records(dip_path, dip_rows)
    mo.append_records(tmp_path / "movers_pop_outcomes.jsonl", pop_rows)
    assert mo.main(["--summary", "--path", str(dip_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["all"]["outcomes"] == 1
    assert printed["pop"]["flags"] == 1
    assert printed["pop"]["path"].endswith("movers_pop_outcomes.jsonl")


def test_service_appends_pop_rows_and_a_failed_pop_write_keeps_the_board(tmp_path, monkeypatch):
    from datetime import datetime as dt

    from ui.services import movers_service as svc

    now = dt(2026, 9, 22, 10, 40, 20, tzinfo=NY)

    class Bot:
        is_process_proxy = False

        def get_scan_symbol_set(self):
            return []

        def m5_chart_bars(self, symbol, max_sessions=2):
            return [dict(b) for b in RUN[:12]] if symbol == "SPY" else []

    service = svc.MoversService(
        bot_provider=lambda: Bot(), downloader=lambda *a, **k: {},
        universe_provider=lambda: [], clock=lambda: now, autostart=False,
        outcomes_path=tmp_path / "movers_dip_outcomes.jsonl", scanner=lambda: {},
        industry_provider=dict, earnings_provider=lambda _d: set(),
    )
    assert service._pop_outcomes_path == tmp_path / "movers_pop_outcomes.jsonl"
    monkeypatch.setattr(service._pop_tracker, "observe",
                        lambda *a, **k: [{"kind": "flag", "symbol": "X"}])
    service._run_once({"long": [], "short": []})
    text = (tmp_path / "movers_pop_outcomes.jsonl").read_text(encoding="utf-8")
    assert text.count('"flag"') == 1
    service._pop_outcomes_path = tmp_path  # a directory: the write fails
    service._run_once({"long": [], "short": []})
    assert service.board() and "state" in service.board()
