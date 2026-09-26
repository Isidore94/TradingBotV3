"""p9: the strength shadow filter on LONG swing rows. It changes nothing live.

The trader, 2026-09-26 ("Yes"): "strength: close >= 2 ATR above the 50-day SMA, only when SPY is
above a rising 20-day". A LONG scan row carries `perm_strength_filter` (yes / no / unknown), the
Long leaders rows carry the distance and the verdict, the horizon file copies the verdict and the
Setup Tracker grades kept vs dropped longs as one shadow line.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import long_setups as ls  # noqa: E402
import setup_grades  # noqa: E402
import setup_permutations as sp  # noqa: E402
from master_avwap_lib import session_horizon_outcomes as sho  # noqa: E402

COLUMN = "perm_strength_filter"


# --- the verdict

def test_the_rule_is_two_atr_over_the_50_day():
    assert sp.STRENGTH_MIN_SMA50_ATR == 2.0
    assert sp.STRENGTH_COLUMNS == (COLUMN,)


@pytest.mark.parametrize("dist, vs, slope, verdict", [
    (2.0, 0.5, 0.1, "yes"),        # exactly 2 ATR, SPY above a rising 20-day
    (3.5, 1.0, 0.2, "yes"),
    (1.99, 1.0, 0.2, "no"),        # not strong enough
    (3.0, -0.1, 0.2, "no"),        # SPY under its 20-day
    (3.0, 0.5, -0.1, "no"),        # SPY's 20-day falling
    (3.0, 0.5, 0.0, "no"),         # flat is not rising
    (1.0, None, None, "no"),       # weak is no whatever SPY is
    (None, -1.0, 0.1, "no"),       # SPY not rising is no whatever the name is
    (None, 0.5, 0.1, "unknown"),   # missing distance
    (3.0, None, 0.1, "unknown"),   # missing SPY
    ("", "", "", "unknown"),
])
def test_the_verdict(dist, vs, slope, verdict):
    assert sp.strength_filter_verdict(dist, vs, slope) == verdict


def test_the_column_is_written_on_longs_only():
    rows = [
        {"side": "LONG", "perm_dist_sma50_atr": 2.5, "perm_spy_vs_sma20_pct": 1.0, "perm_spy_sma20_slope_pct": 0.3},
        {"side": "LONG", "perm_dist_sma50_atr": 0.5, "perm_spy_vs_sma20_pct": 1.0, "perm_spy_sma20_slope_pct": 0.3},
        {"side": "LONG", "perm_dist_sma50_atr": 2.5},
        {"side": "SHORT", "perm_dist_sma50_atr": 2.5, "perm_spy_vs_sma20_pct": 1.0, "perm_spy_sma20_slope_pct": 0.3},
    ]
    assert sp.strength_columns(rows) == 3
    assert [row[COLUMN] for row in rows] == ["yes", "no", "unknown", None]


def test_the_column_is_off_the_permutation_key():
    row = {"side": "LONG", "setup_family": "avwap_band_bounce", "perm_dist_sma50_atr": 2.5,
           "perm_spy_vs_sma20_pct": 1.0, "perm_spy_sma20_slope_pct": 0.3}
    before = sp.stamp_fields(row)
    sp.strength_columns([row])
    assert row[COLUMN] == "yes"
    assert sp.stamp_fields(row) == before


# --- the scan: the column on the feature history; every existing column is proved identical by
# `test_setup_permutation_scan_parity` (the hook is switched off there with the others)

def _parity_module():
    path = Path(__file__).with_name("test_setup_permutation_scan_parity.py")
    spec = importlib.util.spec_from_file_location("_strength_scan_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan_runs(tmp_path_factory):
    parity = _parity_module()
    base = tmp_path_factory.mktemp("strength-scan")
    return parity, parity._run(base, "on"), parity._run(base, "off")


def test_the_scan_writes_the_strength_verdict_last(scan_runs):
    _parity, stamped, plain = scan_runs
    row = stamped["history"][-1]
    assert list(row)[-len(sp.STRENGTH_COLUMNS):] == list(sp.STRENGTH_COLUMNS)
    assert row["side"] == "LONG"
    # The child's SPY rises; the verdict is the row's own distance + regime columns.
    expected = sp.strength_filter_verdict(row["perm_dist_sma50_atr"], row["perm_spy_vs_sma20_pct"],
                                          row["perm_spy_sma20_slope_pct"])
    assert expected in ("yes", "no")
    assert row[COLUMN] == expected
    assert plain["history"][-1][COLUMN] == ""


def test_the_scan_output_is_identical_with_and_without_the_strength_column(scan_runs):
    parity, stamped, plain = scan_runs
    assert parity._strip(stamped["priority_row"]) == parity._strip(plain["priority_row"])
    assert parity._strip(stamped["ai_state_entry"]) == parity._strip(plain["ai_state_entry"])
    for on_row, off_row in zip(stamped["history"], plain["history"], strict=True):
        assert parity._strip(on_row) == parity._strip(off_row)
        assert list(on_row) == list(off_row)


# --- the Long leaders rows

def _days(count: int) -> list[str]:
    out, day = [], date(2025, 1, 2)
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def _bars(closes, volumes):
    return [{"date": d, "open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": v}
            for d, c, v in zip(_days(len(closes)), closes, volumes, strict=True)]


def _leader():
    base = 230
    closes = [50.0 + 0.1 * i for i in range(base)] + [50.0 + 0.1 * base + 0.5 * k for k in range(1, 61)]
    peak = closes[-1]
    closes += [peak * (1 - 0.012 * k) for k in range(1, 11)]
    return _bars(closes, [2_000_000] * 290 + [1_200_000] * 10)


def _spy(rising: bool, count: int = 300):
    closes = [400.0 + (0.5 * i if rising else -0.5 * i) for i in range(count)]
    return _bars(closes, [1_000_000] * count)


def _scan(bars, spy, atr):
    feature = {"symbol": "LEAD", "side": "LONG", "perm_regime_working": "yes",
               "perm_regime_working_rule": "trader", "perm_market_cap_m": 5000.0}
    return ls.build_rows(bars_by_symbol={"LEAD": bars}, spy_bars=spy, feature_rows=[feature],
                         atr_by_symbol={"LEAD": atr}, as_of=bars[-1]["date"])


def _sma50_atr(bars, atr):
    closes = [bar["close"] for bar in bars]
    return (closes[-1] - sum(closes[-50:]) / 50) / atr


def test_long_leader_rows_carry_the_distance_and_the_verdict():
    bars = _leader()
    rows = _scan(bars, _spy(True), 2.0)["rows"]
    assert rows
    row = rows[0]
    assert row["strength_sma50_atr"] == pytest.approx(_sma50_atr(bars, 2.0), abs=1e-4)
    # A 12% pullback sits under its 50-day: dropped by the shadow, still a live row.
    assert row["strength_filter"] == "no"
    assert row["status"] == ls.STATUS_LISTED  # the live gate's answer (no RS read, no leader bonus)


def test_a_strong_name_in_a_rising_spy_is_kept():
    bars = _leader()
    assert _sma50_atr(bars, 2.0) < 0  # the 12% pullback sits under its 50-day
    # Lower the older part of the 50-day window (never the swing high) so the close is far above it.
    lifted = [dict(bar) for bar in bars]
    for bar in lifted[-50:-12]:
        for key in ("open", "high", "low", "close"):
            bar[key] -= 15.0
    row = ls.leader_pullback(lifted, atr=2.0)
    assert row is not None
    assert _sma50_atr(lifted, 2.0) >= 2.0
    shadow = ls.strength_shadow(lifted, 2.0, 1.0, 0.2)
    assert shadow["strength_filter"] == "yes"
    assert ls.strength_shadow(lifted, 2.0, -1.0, 0.2)["strength_filter"] == "no"


def test_the_shadow_changes_nothing_live():
    bars = _leader()
    rising, falling = _scan(bars, _spy(True), 2.0)["rows"], _scan(bars, _spy(False), 2.0)["rows"]
    strip = lambda rows: [{k: v for k, v in row.items() if not k.startswith("strength_")} for row in rows]  # noqa: E731
    assert strip(rising) == strip(falling)


def test_a_stale_or_short_spy_is_unknown_unless_the_name_is_weak():
    bars = _leader()
    stale = _spy(True)[:-1]
    row = _scan(bars, stale, 2.0)["rows"][0]
    assert row["strength_filter"] == "no"  # weak is no whatever SPY is
    assert ls.strength_shadow(bars[-40:], 2.0, 1.0, 0.2) == {"strength_sma50_atr": None,
                                                             "strength_filter": "unknown"}


# --- the horizon file and the Setup Tracker grade

def test_the_horizon_file_copies_the_verdict_last():
    assert sho.SESSION_HORIZON_OUTCOME_COLUMNS[-1] == "strength_filter"
    days = pd.bdate_range("2026-03-02", periods=12)
    history = pd.DataFrame([
        {"symbol": "AAA", "side": "LONG", "last_trade_date": days[0].date().isoformat(), "last_close": 10.0,
         "run_id": "r1", COLUMN: "yes"},
        {"symbol": "BBB", "side": "LONG", "last_trade_date": days[0].date().isoformat(), "last_close": 10.0,
         "run_id": "r1", COLUMN: None},
    ])
    closes = {day.date(): 10.0 + index for index, day in enumerate(days)}
    built = sho.build_session_horizon_observation_rows(
        history, lambda symbol: closes, horizons=(5,), last_completed_session=days[-1].date(), window_sessions=None)
    by_symbol = {row["symbol"]: row["strength_filter"] for row in built.rows}
    assert by_symbol == {"AAA": "yes", "BBB": ""}


def _horizon(symbol, day, target, side_return, verdict, side="LONG"):
    return {"symbol": symbol, "side": side, "scan_date": day, "target_session": target, "horizon_sessions": "5",
            "side_return_pct": str(side_return), "measured": "True", "maturity": "mature",
            "outcome_kind": setup_grades.TAPE_OUTCOME_KIND, "strength_filter": verdict}


def test_the_grade_counts_kept_vs_the_rest():
    spy = {"2026-03-02": 100.0, "2026-03-09": 101.0}  # SPY +1%
    rows = [
        _horizon("A", "2026-03-02", "2026-03-09", 3.0, "yes"),    # kept, beat
        _horizon("B", "2026-03-02", "2026-03-09", 0.5, "yes"),    # kept, lost
        _horizon("C", "2026-03-02", "2026-03-09", 2.0, "no"),     # rest, beat
        _horizon("D", "2026-03-02", "2026-03-09", -1.0, "no"),    # rest, lost
        _horizon("E", "2026-03-02", "2026-03-09", -2.0, "no"),    # rest, lost
        _horizon("F", "2026-03-02", "2026-03-09", 5.0, "unknown"),  # left out
        _horizon("G", "2026-03-02", "2026-03-09", 5.0, ""),         # left out (older row)
        _horizon("H", "2026-03-02", "2026-03-09", 5.0, "yes", side="SHORT"),  # never a short
        _horizon("I", "2026-03-02", "2026-03-10", 5.0, "yes"),    # SPY unknown on the target: left out
    ]
    cell = setup_grades.strength_filter_cell(rows, spy, as_of="2026-03-20")
    assert (cell["kept_n"], cell["kept_wins"], cell["rest_n"], cell["rest_wins"]) == (2, 1, 3, 1)
    line = setup_grades.strength_filter_line(cell)
    assert line == ("strength filter (shadow): kept 40% of longs; kept rows beat SPY 50% vs 33% for the rest, "
                    "n 5 (kept 2, rest 3) (scan dates 2026-03-02 to 2026-03-02)")


def test_the_grade_waits_for_the_target_session():
    spy = {"2026-03-02": 100.0, "2026-03-09": 101.0}
    rows = [_horizon("A", "2026-03-02", "2026-03-09", 3.0, "yes")]
    assert setup_grades.strength_filter_cell(rows, spy, as_of="2026-03-06")["kept_n"] == 0
    assert setup_grades.strength_filter_line(
        setup_grades.strength_filter_cell(rows, spy, as_of="2026-03-06")) == \
        "strength filter (shadow): no graded longs yet."


def test_the_tracker_worker_shows_the_shadow_line(tmp_path, monkeypatch):
    import csv

    from ui.services import working_lately_service as wls

    path = tmp_path / "horizons.csv"
    rows = [_horizon("A", "2026-03-02", "2026-03-09", 3.0, "yes"),
            _horizon("C", "2026-03-02", "2026-03-09", -2.0, "no")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(wls, "_horizon_outcomes_path", lambda: path)
    monkeypatch.setattr(wls, "read_spy_closes", lambda: {"2026-03-02": 100.0, "2026-03-09": 101.0})
    monkeypatch.setattr(wls, "_cached", lambda name, key, build, keep=True: build())
    line = wls.read_strength_filter_line("2026-03-20")
    assert line.startswith("strength filter (shadow): kept 50% of longs; kept rows beat SPY 100% vs 0%")


def test_the_setup_tracker_worker_carries_the_line_and_survives_a_failure(monkeypatch):
    from ui.panels import setup_tracker_panel as module
    from ui.services import working_lately_service as wls

    monkeypatch.setattr(wls, "read_strength_filter_line", lambda as_of="": "strength filter (shadow): x")
    assert module._read_tracker_exports(1)["strength_filter_line"] == "strength filter (shadow): x"

    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(wls, "read_strength_filter_line", _boom)
    exports = module._read_tracker_exports(1)
    assert exports["strength_filter_line"] == ""
    assert "study_family_lines" in exports


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.qt
def test_the_tracker_prints_the_line_under_the_study_lines(qapp):
    from ui.panels import setup_tracker_panel as module

    panel = module.SetupTrackerPanel()
    try:
        panel._on_exports_loaded({"signatures": {}, "ranked": {}, "raw": {}, "study_family_lines": ["x", "y"],
                                  "strength_filter_line": "strength filter (shadow): z"})
        assert panel.study_family_label.text() == "x\ny\nstrength filter (shadow): z"
    finally:
        panel.shutdown()
        panel.deleteLater()
