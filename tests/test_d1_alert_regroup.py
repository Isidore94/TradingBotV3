"""D1 alert menu regroup (trader's word 2026-09-24).

Golden fixtures for the three new D1 event kinds (`d1_line_pullback`,
`range_breakout`, `line_break`), the legacy kinds that left the menu but must
still load / evaluate / fire, the grouped menu itself, and the Focus auto
lane, which must not move.
"""

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import chart_watch  # noqa: E402
from chart_watch import (  # noqa: E402
    D1_EVENT_KINDS,
    D1_EXTENSION_KINDS,
    D1_PULLBACK_KINDS,
    D1EventWatch,
    d1_event_watch_from_dict,
    d1_event_watch_to_dict,
    evaluate_d1_event_watch,
    load_d1_event_watches,
    save_d1_event_watches,
)

DAY = datetime(2026, 7, 24)
ARMED = DAY.replace(hour=10, minute=0)
NOW = DAY.replace(hour=10, minute=15)

LEGACY_KINDS = (
    "ema15_reject",
    "new_5d_high",
    "new_5d_low",
    "new_20d_high",
    "new_20d_low",
    "sma_break",
    "avwape_bounce",
    "avwape_break",
    "avwape_dev1_bounce",
    "avwape_dev1_break",
)


def _m5(hour, minute, *, h, low, c, day=DAY):
    return {
        "dt": day.replace(hour=hour, minute=minute),
        "open": float(c),
        "high": float(h),
        "low": float(low),
        "close": float(c),
        "volume": 1000.0,
    }


def _daily(offset, *, h, low, c):
    return {
        "dt": DAY - timedelta(days=offset),
        "open": float(c),
        "high": float(h),
        "low": float(low),
        "close": float(c),
        "volume": 1000.0,
    }


def _flat_daily(sessions=30, *, h=101.0, low=99.0, c=100.0):
    """A tight base: every session 99-101, close 100 (ATR14 = 2, EMA15 = 100)."""
    return [_daily(offset, h=h, low=low, c=c) for offset in range(sessions, 0, -1)]


def _stretched_daily(sessions=30):
    """A stock running 2 points a day: its 20-session range is ~15x its ATR."""
    return [
        _daily(offset, h=100.0 + (sessions - offset) * 2.0 + 0.5,
               low=100.0 + (sessions - offset) * 2.0 - 0.5,
               c=100.0 + (sessions - offset) * 2.0)
        for offset in range(sessions, 0, -1)
    ]


def _ramping_daily():
    """Anchor 6 sessions back; AVWAPE = 105, +1σ ≈ 108.03, prev close 110."""
    return [
        {
            "dt": DAY - timedelta(days=offset),
            "open": 100.0 + (6 - offset) * 2.0,
            "high": 101.0 + (6 - offset) * 2.0,
            "low": 99.0 + (6 - offset) * 2.0,
            "close": 100.0 + (6 - offset) * 2.0,
            "volume": 1000.0,
        }
        for offset in range(6, 0, -1)
    ]


def _watch(kind, armed_at=ARMED):
    return D1EventWatch(symbol="NVDA", kind=kind, armed_at=armed_at)


def _levels_pairs(daily, anchor):
    levels = chart_watch.d1_event_levels(daily, session=DAY.date(), avwape_anchor=anchor)
    return {label: value for label, value in levels["avwape_levels"]}


# ---------------------------------------------------------------- kind sets
def test_new_kinds_are_labelled_and_legacy_kinds_stay_evaluable():
    assert D1_EVENT_KINDS["d1_line_pullback"] == "Pullback to D1 line"
    assert D1_EVENT_KINDS["range_breakout"] == "Range breakout"
    assert D1_EVENT_KINDS["line_break"] == "Line break"
    for kind in LEGACY_KINDS:
        assert D1_EVENT_KINDS[kind], kind


def test_focus_auto_lane_set_is_unchanged():
    assert D1_PULLBACK_KINDS == frozenset(
        {"ema15_reject", "avwape_bounce", "avwape_dev1_bounce"}
    )
    assert "d1_line_pullback" not in D1_PULLBACK_KINDS
    assert {"range_breakout", "line_break"} <= D1_EXTENSION_KINDS
    assert "d1_line_pullback" not in D1_EXTENSION_KINDS


def test_menu_groups_are_the_three_principles_in_order():
    groups = chart_watch.D1_MENU_GROUPS
    assert [title for title, _kinds in groups] == [
        "PULLBACK — it ran, let it calm down",
        "BREAKOUT — it was tight, let it go",
        "LINE BREAK — it crossed a big line",
    ]
    assert [kinds for _title, kinds in groups] == [
        ("pullback", "d1_line_pullback"),
        ("range_breakout",),
        ("line_break", "sma_break_retest", "trendline_break", "trendline_break_retest"),
    ]
    menu_d1 = set(chart_watch.D1_MENU_KINDS)
    assert menu_d1 == {
        "d1_line_pullback",
        "range_breakout",
        "line_break",
        "sma_break_retest",
        "trendline_break",
        "trendline_break_retest",
    }
    assert set(chart_watch.D1_LEGACY_KINDS) == set(LEGACY_KINDS)


def test_avwape_anchor_is_needed_by_the_kinds_that_read_the_line():
    needs = chart_watch.d1_kind_needs_avwape
    for kind in ("d1_line_pullback", "line_break", "avwape_bounce", "avwape_dev1_break"):
        assert needs(kind), kind
    for kind in ("range_breakout", "ema15_reject", "sma_break", "trendline_break"):
        assert not needs(kind), kind


# ---------------------------------------------------------- d1_line_pullback
def test_line_pullback_fires_on_a_15ema_reject_and_names_the_line():
    daily = _flat_daily()
    bar = _m5(10, 5, h=100.4, low=99.8, c=100.3)
    hit = evaluate_d1_event_watch(_watch("d1_line_pullback"), [bar], daily, now=NOW)
    assert hit is not None and hit.resolved_side == "long"
    assert hit.message.startswith("Pullback to D1 line:")
    assert "15EMA" in hit.message


def test_line_pullback_fires_on_a_1sigma_bounce_and_names_the_band():
    daily = _ramping_daily()
    anchor = daily[0]["dt"].date()
    upper_1 = _levels_pairs(daily, anchor)["+1σ"]
    bar = _m5(10, 5, h=110.0, low=upper_1 - 0.05, c=upper_1 + 0.3)
    hit = evaluate_d1_event_watch(
        _watch("d1_line_pullback"), [bar], daily, now=NOW, avwape_anchor=anchor
    )
    assert hit is not None and hit.resolved_side == "long"
    assert "AVWAPE +1σ bounce" in hit.message


def test_line_pullback_is_quiet_without_a_tag():
    daily = _flat_daily()
    bar = _m5(10, 5, h=103.0, low=102.0, c=102.5)  # nowhere near the 15EMA
    assert evaluate_d1_event_watch(_watch("d1_line_pullback"), [bar], daily, now=NOW) is None


def test_line_pullback_forming_bar_never_fires():
    daily = _flat_daily()
    bar = _m5(10, 5, h=100.4, low=99.8, c=100.3)
    early = DAY.replace(hour=10, minute=8)
    assert evaluate_d1_event_watch(_watch("d1_line_pullback"), [bar], daily, now=early) is None


# -------------------------------------------------------------- line_break
def test_line_break_fires_on_an_sma_close_through_and_names_it():
    # 49 closes at 100 then 99: SMA50 = 99.98, prev close 99 below it.
    daily = [
        _daily(offset, h=101.0, low=98.0, c=(99.0 if offset == 1 else 100.0))
        for offset in range(50, 0, -1)
    ]
    bar = _m5(10, 5, h=100.6, low=99.0, c=100.5)
    hit = evaluate_d1_event_watch(_watch("line_break"), [bar], daily, now=NOW)
    assert hit is not None and hit.resolved_side == "long"
    assert hit.message.startswith("Line break:")
    assert "SMA50 break up" in hit.message


def test_line_break_names_the_avwape_line_or_the_band():
    daily = _ramping_daily()
    anchor = daily[0]["dt"].date()
    through_line = _m5(10, 5, h=110.0, low=103.5, c=104.0)
    hit = evaluate_d1_event_watch(
        _watch("line_break"), [through_line], daily, now=NOW, avwape_anchor=anchor
    )
    assert hit is not None and hit.resolved_side == "short"
    assert "AVWAPE break down" in hit.message

    through_band = _m5(10, 5, h=110.0, low=107.0, c=107.5)  # under +1σ, over the line
    hit = evaluate_d1_event_watch(
        _watch("line_break"), [through_band], daily, now=NOW, avwape_anchor=anchor
    )
    assert hit is not None and "AVWAPE +1σ break down" in hit.message


def test_line_break_ignores_a_bounce():
    daily = _ramping_daily()
    anchor = daily[0]["dt"].date()
    upper_1 = _levels_pairs(daily, anchor)["+1σ"]
    bounce = _m5(10, 5, h=110.0, low=upper_1 - 0.05, c=upper_1 + 0.3)
    assert (
        evaluate_d1_event_watch(
            _watch("line_break"), [bounce], daily, now=NOW, avwape_anchor=anchor
        )
        is None
    )


# ----------------------------------------------------------- range_breakout
def test_range_breakout_fires_out_of_a_tight_base_and_says_how_tight():
    daily = _flat_daily()
    bar = _m5(10, 5, h=101.5, low=100.5, c=101.2)
    hit = evaluate_d1_event_watch(_watch("range_breakout"), [bar], daily, now=NOW)
    assert hit is not None and hit.resolved_side == "long"
    assert hit.price == pytest.approx(101.5)
    assert "Range breakout (long)" in hit.message
    assert "20-session range 2.00" in hit.message
    assert "1.0x ATR14" in hit.message


def test_range_breakout_short_mirrors():
    daily = _flat_daily()
    bar = _m5(10, 5, h=99.5, low=98.5, c=98.8)
    hit = evaluate_d1_event_watch(_watch("range_breakout"), [bar], daily, now=NOW)
    assert hit is not None and hit.resolved_side == "short"
    assert "Range breakout (short)" in hit.message


def test_range_breakout_does_not_fire_on_a_stretched_stock():
    daily = _stretched_daily()
    top = max(bar["high"] for bar in daily[-20:])
    bar = _m5(10, 5, h=top + 1.0, low=top - 1.0, c=top + 0.5)
    # The same evidence IS a plain 20-day high: only the tightness gate says no.
    assert evaluate_d1_event_watch(_watch("new_20d_high"), [bar], daily, now=NOW) is not None
    assert evaluate_d1_event_watch(_watch("range_breakout"), [bar], daily, now=NOW) is None


def test_range_breakout_missing_history_is_no_fire():
    daily = _flat_daily(sessions=19)  # one session short of a 20-day base
    bar = _m5(10, 5, h=105.0, low=100.5, c=104.0)
    assert evaluate_d1_event_watch(_watch("range_breakout"), [bar], daily, now=NOW) is None
    assert evaluate_d1_event_watch(_watch("range_breakout"), [bar], [], now=NOW) is None


def test_range_breakout_threshold_is_inclusive_and_pinned():
    assert chart_watch.RANGE_BREAKOUT_TIGHT_ATR == 4.0
    assert chart_watch.RANGE_BREAKOUT_BASE_SESSIONS == 20
    assert chart_watch.RANGE_BREAKOUT_RULE_VERSION == "range_breakout_v1"


def test_range_breakout_fires_off_a_completed_daily_bar_after_the_arm_day():
    base = _flat_daily()
    breakout_day = {
        "dt": DAY,
        "open": 100.5,
        "high": 102.0,
        "low": 100.2,
        "close": 101.8,
        "volume": 1000.0,
    }
    watch = _watch("range_breakout", armed_at=(DAY - timedelta(days=1)).replace(hour=15))
    later = DAY + timedelta(days=1, hours=9)
    hit = evaluate_d1_event_watch(watch, [], base + [breakout_day], now=later)
    assert hit is not None and "D1 bar" in hit.message
    # The forming (today's) daily bar is never evidence.
    assert evaluate_d1_event_watch(watch, [], base + [breakout_day], now=NOW) is None


# ------------------------------------------------------------ legacy kinds
@pytest.mark.parametrize("kind", LEGACY_KINDS + ("d1_line_pullback", "range_breakout", "line_break"))
def test_every_kind_round_trips_the_store(tmp_path, kind):
    path = tmp_path / "d1_event_watches.json"
    save_d1_event_watches([_watch(kind)], path)
    loaded = load_d1_event_watches(path)
    assert [watch.kind for watch in loaded] == [kind]
    assert d1_event_watch_from_dict(d1_event_watch_to_dict(_watch(kind))).kind == kind


def test_a_saved_legacy_15ema_reject_still_fires():
    daily = _flat_daily()
    bar = _m5(10, 5, h=100.4, low=99.8, c=100.3)
    hit = evaluate_d1_event_watch(_watch("ema15_reject"), [bar], daily, now=NOW)
    assert hit is not None
    assert hit.message.startswith("D1 15EMA rejection (long)")


# ------------------------------------------------------------------ the menu
def _qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _visible_menu_rows(menu, headers=()):
    rows = []
    for action in menu.actions():
        if not action.isVisible():
            continue
        if action.isSeparator():
            rows.append("--")
        elif action in headers:
            assert not action.isEnabled()
            rows.append(f"== {action.text()}")
        else:
            rows.append(action.text())
    return rows


def _compact_bar():
    _qt_app()
    from ui.widgets.arm_bar import ArmBar

    bar = ArmBar()
    bar.set_compact(True)
    bar.set_enabled_for_symbol(True)
    return bar


def test_compact_d1_menu_order_and_labels():
    bar = _compact_bar()
    assert _visible_menu_rows(bar.d1_menu_button.menu(), bar.d1_menu_headers) == [
        "== PULLBACK — it ran, let it calm down",
        "Pullback (fast)",
        "Pullback to D1 line",
        "--",
        "== BREAKOUT — it was tight, let it go",
        "Range breakout",
        "--",
        "== LINE BREAK — it crossed a big line",
        "Line break",
        "SMA break + 15EMA retest",
        "Trendline break",
        "Trendline break + retest",
        "--",
        "Open in TradingView",
    ]


def test_pullback_moved_off_the_m5_menu_onto_the_d1_menu():
    bar = _compact_bar()
    assert "pullback" not in bar.m5_actions
    m5_rows = _visible_menu_rows(bar.m5_menu_button.menu())
    assert not any("Pullback" in row for row in m5_rows), m5_rows
    watched = []
    bar.watchToggled.connect(watched.append)
    bar.d1_actions["pullback"].trigger()
    assert watched == ["pullback"]
    # Armed, the D1 menu counts it and ticks it.
    bar.set_armed_kinds(["pullback"])
    assert bar.d1_actions["pullback"].isChecked()
    assert bar.d1_menu_button.text() == "D1 alert (1) ▾"
    assert bar.m5_menu_button.text() == "M5 alert ▾"


def test_new_d1_actions_emit_their_kind():
    bar = _compact_bar()
    fired = []
    bar.d1EventToggled.connect(fired.append)
    for kind in ("d1_line_pullback", "range_breakout", "line_break"):
        bar.d1_actions[kind].trigger()
    assert fired == ["d1_line_pullback", "range_breakout", "line_break"]


def test_an_armed_legacy_kind_shows_so_it_can_be_disarmed():
    bar = _compact_bar()
    assert not bar.d1_actions["ema15_reject"].isVisible()
    bar.set_armed_d1_events(["ema15_reject"])
    action = bar.d1_actions["ema15_reject"]
    assert action.isVisible() and action.isChecked()
    assert action.text() == "15EMA reject ✓"
    fired = []
    bar.d1EventToggled.connect(fired.append)
    action.trigger()
    assert fired == ["ema15_reject"]
    bar.set_armed_d1_events([])
    assert not bar.d1_actions["ema15_reject"].isVisible()


def test_any_bounce_left_the_menu_but_an_armed_one_can_still_be_disarmed():
    bar = _compact_bar()
    assert not bar.d1_actions["any_bounce"].isVisible()
    bar.set_any_bounce_armed(True)
    assert bar.d1_actions["any_bounce"].isVisible()
    clicks = []
    bar.anyBounceToggled.connect(lambda: clicks.append(1))
    bar.d1_actions["any_bounce"].trigger()
    assert clicks == [1]


def test_classic_rows_carry_the_same_grouping():
    _qt_app()
    from ui.widgets.arm_bar import ArmBar

    bar = ArmBar()
    bar.set_enabled_for_symbol(True)
    top = bar._classic_top_widgets()
    d1 = bar._classic_d1_widgets()
    pullback = bar.watch_buttons["pullback"]
    assert pullback not in top and pullback in d1
    assert pullback.text() == "Pullback (fast)"
    shown = [widget.text() for widget in d1 if hasattr(widget, "text") and widget.text()]
    assert shown[:1] == ["D1:"]
    for label in (
        "Pullback (fast)",
        "Pullback to D1 line",
        "Range breakout",
        "Line break",
        "SMA break + 15EMA retest",
        "Trendline break",
        "Trendline break + retest",
        "Open in TradingView",
    ):
        assert label in shown, (label, shown)
    assert "15EMA reject" not in shown and "Any bounce" not in shown
    bar.set_armed_d1_events(["ema15_reject"])
    assert bar.d1_event_buttons["ema15_reject"] in bar._classic_d1_widgets()


# ------------------------------------------------------------- panel wiring
def test_panel_poll_hands_the_avwape_anchor_to_the_new_line_kinds(tmp_path, monkeypatch):
    _qt_app()
    from ui.panels import alert_center_panel as panel_mod
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.services import chart_data_service

    anchors: dict[str, object] = {}

    def _fake_evaluate(watch, m5, d1, *, now=None, avwape_anchor=None, levels_cache=None):
        anchors[watch.kind] = avwape_anchor
        return None

    class _Service:
        def cached_earnings_anchor(self, symbol):
            return date(2026, 5, 1)

    monkeypatch.setattr(panel_mod, "evaluate_d1_event_watch", _fake_evaluate)
    monkeypatch.setattr(chart_data_service, "shared_service", lambda: _Service())
    panel = AlertCenterPanel(parked_symbols_path=tmp_path / "parked.json")
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: _flat_daily())
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol: [])
    monkeypatch.setattr(panel, "_m5_unknown", lambda symbol: False)
    monkeypatch.setattr(panel, "_poll_pullback_watches", lambda now=None: None)
    monkeypatch.setattr(panel, "_save_d1_event_watches", lambda: None)
    now = datetime.now()
    panel._d1_event_watches = [
        D1EventWatch(symbol="NVDA", kind=kind, armed_at=now)
        for kind in ("d1_line_pullback", "line_break", "range_breakout")
    ]
    panel._poll_d1_event_watches(now=now)
    assert anchors == {
        "d1_line_pullback": date(2026, 5, 1),
        "line_break": date(2026, 5, 1),
        "range_breakout": None,
    }


def test_panel_arms_each_new_kind(tmp_path, monkeypatch):
    _qt_app()
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel(parked_symbols_path=tmp_path / "parked.json")
    monkeypatch.setattr(panel, "_save_d1_event_watches", lambda: None)
    for kind in ("d1_line_pullback", "range_breakout", "line_break"):
        assert panel.arm_d1_event_watch("NVDA", kind) is True, kind
    assert set(panel.armed_d1_event_kinds("NVDA")) == {
        "d1_line_pullback",
        "range_breakout",
        "line_break",
    }


def test_journal_evidence_reads_the_new_kinds_as_d1():
    import journal_setup_evidence

    for kind in ("d1_line_pullback", "range_breakout", "line_break"):
        assert journal_setup_evidence._event_horizon({}, kind) == "d1", kind


def test_range_breakout_fire_carries_its_rule_version_and_measure():
    hit = evaluate_d1_event_watch(
        _watch("range_breakout"), [_m5(10, 5, h=101.5, low=100.5, c=101.2)], _flat_daily(), now=NOW
    )
    assert hit is not None
    assert hit.details["rule_version"] == "range_breakout_v1"
    assert hit.details["base_range_20d"] == pytest.approx(2.0)
    assert hit.details["range_atr_ratio"] == pytest.approx(1.0, abs=0.05)


def test_the_fired_record_keeps_the_trigger_details():
    hit = evaluate_d1_event_watch(
        _watch("range_breakout"), [_m5(10, 5, h=101.5, low=100.5, c=101.2)], _flat_daily(), now=NOW
    )
    detail = chart_watch.d1_event_fired_detail(hit)
    assert detail["kind"] == "range_breakout"
    assert detail["message"] == hit.message
    assert detail["rule_version"] == "range_breakout_v1"
