"""S15 items 3 + 9: swing path facts (MFE/MAE in ATR) and the fill-model columns."""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_calendar  # noqa: E402
import swing_path_facts as spf  # noqa: E402
from research_warehouse import retest_entry as rt  # noqa: E402

SCAN = date(2026, 9, 1)  # Tuesday; Labor Day 2026-09-07 sits inside the path
ROW_ID = "AAA:2026-09-01:run1"
LATE = date(2026, 12, 31)


def _sessions(count: int) -> list[date]:
    out, cursor = [], SCAN
    for _ in range(count):
        cursor = market_calendar.next_session(cursor)
        out.append(cursor)
    return out


def _horizon_row(side="LONG", family="avwap_breakout", entry=100.0, row_id=ROW_ID, symbol="AAA"):
    return {"outcome_kind": spf.SOURCE_OUTCOME_KIND, "scan_row_id": row_id, "symbol": symbol,
            "side": side, "scan_date": SCAN.isoformat(), "entry_close": str(entry),
            "setup_family": family, "horizon_sessions": "5"}


def _bars(path: list[tuple[float, float, float, float]], *, scan_bar=(100.0, 100.0, 100.0, 100.0)):
    bars = {SCAN: scan_bar}
    for day, bar in zip(_sessions(len(path)), path, strict=True):
        bars[day] = bar
    return bars


FLAT = (100.0, 100.5, 99.5, 100.0)


def _build(bars, *, side="LONG", family="avwap_breakout", facts=None, last=LATE, horizons=(5,)):
    facts = facts if facts is not None else {ROW_ID: spf.ScanFacts(atr20=2.0)}
    build = spf.build_path_fact_rows(
        [_horizon_row(side=side, family=family)], lambda symbol: bars, facts,
        last_completed_session=last, horizons=horizons,
    )
    return {row["horizon_sessions"]: row for row in build.rows}


def test_stop_first_path_reports_adverse_first_then_the_later_run():
    # Session 1 drops 1.5 ATR (low 97), then the name runs to +2.5 ATR (high 105).
    path = [(100.0, 100.2, 97.0, 98.0), (98.0, 101.0, 97.5, 100.5), FLAT,
            (100.5, 105.0, 100.0, 104.0), (104.0, 104.5, 103.0, 103.5)]
    row = _build(_bars(path))[5]
    assert row["measured"] is True
    assert row["mfe_atr"] == pytest.approx(2.5)
    assert row["mae_atr"] == pytest.approx(-1.5)
    assert row["mfe_session"] == 4 and row["mae_session"] == 1
    assert row["first_1atr"] == "adverse"
    assert row["side_return_pct"] == pytest.approx(3.5)


def test_short_side_is_mirrored_and_one_bar_touching_both_is_same_session():
    path = [(100.0, 102.5, 97.5, 100.0)] + [FLAT] * 4
    row = _build(_bars(path), side="SHORT")[5]
    assert row["first_1atr"] == "same_session"
    assert row["mfe_atr"] == pytest.approx(1.25)
    assert row["mae_atr"] == pytest.approx(-1.25)


def test_scan_day_bar_is_never_part_of_the_path():
    # A huge scan-day high must not become the MFE: only bars after entry count.
    bars = _bars([FLAT] * 5, scan_bar=(100.0, 130.0, 80.0, 100.0))
    row = _build(bars)[5]
    assert row["mfe_atr"] == pytest.approx(0.25)
    assert row["mae_atr"] == pytest.approx(-0.25)
    assert row["first_1atr"] == "neither"


def test_gap_next_open_reprices_and_pullback_fills_at_the_gapped_open():
    # Session 1 gaps down to 98 (below the 99.5 limit = 100 - 0.25 * 2 ATR).
    path = [(98.0, 99.0, 97.8, 98.5)] + [FLAT] * 3 + [(100.0, 101.0, 99.8, 101.0)]
    facts = {ROW_ID: spf.ScanFacts(atr20=2.0, pct_from_current_vwap=-5.0, sector="Technology")}
    row = _build(_bars(path), facts=facts)[5]
    assert row["next_open"] == 98.0
    assert row["next_open_side_return_pct"] == pytest.approx((101.0 / 98.0 - 1) * 100)
    assert row["side_return_pct"] == pytest.approx(1.0)
    assert row["leader_pullback"] is True
    assert row["pullback_limit"] == pytest.approx(99.5)
    assert row["pullback_status"] == spf.PULLBACK_FILLED
    assert row["pullback_fill"] == 98.0  # the gapped open, never the better limit
    assert row["pullback_side_return_pct"] == pytest.approx((101.0 / 98.0 - 1) * 100)


def test_pullback_no_fill_is_no_trade_not_zero():
    path = [(100.2, 101.0, 99.8, 100.5)] + [FLAT] * 4
    facts = {ROW_ID: spf.ScanFacts(atr20=2.0, pct_from_current_vwap=-5.0, sector="Energy")}
    row = _build(_bars(path), family="top_pattern_tracking", facts=facts)[5]
    assert row["leader_pullback"] is True
    assert row["pullback_status"] == spf.PULLBACK_NO_FILL
    assert row["pullback_side_return_pct"] == ""


@pytest.mark.parametrize(
    ("side", "family", "facts", "expected"),
    [
        ("SHORT", "top_pattern_tracking", spf.ScanFacts(2.0, -5.0, "Technology"), False),
        ("LONG", "avwap_breakout", spf.ScanFacts(2.0, -2.0, "Technology"), False),
        ("LONG", "avwap_breakout", spf.ScanFacts(2.0, -5.0, "Energy"), False),
        ("LONG", "avwap_breakout", spf.ScanFacts(2.0, -10.0, "Technology"), True),
        ("LONG", "avwap_breakout", spf.ScanFacts(2.0, None, "Technology"), None),
        ("LONG", "avwap_breakout", spf.ScanFacts(2.0, -5.0, ""), None),
        ("LONG", "avwap_breakout", None, None),
    ],
)
def test_leader_pullback_key(side, family, facts, expected):
    assert spf.leader_pullback(side, family, facts) is expected


def test_missing_bar_mid_path_is_unknown_never_zero():
    bars = _bars([FLAT] * 5)
    del bars[_sessions(3)[2]]
    rows = _build(bars, horizons=(1, 3, 5))
    assert rows[1]["measured"] is True
    for horizon in (3, 5):
        assert rows[horizon]["measured"] is False
        assert rows[horizon]["unmeasured_reason"] == spf.REASON_MISSING_BAR
        assert rows[horizon]["mfe_atr"] == "" and rows[horizon]["side_return_pct"] == ""


def test_no_atr_keeps_the_close_return_and_leaves_excursions_unknown():
    facts = {ROW_ID: spf.ScanFacts(atr20=None, pct_from_current_vwap=-5.0, sector="Technology")}
    row = _build(_bars([FLAT] * 5), facts=facts)[5]
    assert row["measured"] is True and row["side_return_pct"] == pytest.approx(0.0)
    assert row["mfe_atr"] == "" and row["mae_atr"] == "" and row["first_1atr"] == ""
    assert row["pullback_status"] == spf.PULLBACK_NO_ATR


def test_no_bars_for_symbol_is_unknown():
    row = _build(None)[5]
    assert row["measured"] is False and row["unmeasured_reason"] == spf.REASON_NO_BARS


def test_horizon_20_and_immature_rows_ignore_bars_after_the_last_completed_session():
    sessions = _sessions(20)
    bars = _bars([FLAT] * 20)
    rows = _build(bars, horizons=spf.PATH_HORIZONS, last=sessions[9])
    assert sorted(rows) == [1, 3, 5, 10, 20]
    assert rows[10]["measured"] is True
    assert rows[20]["measured"] is False and rows[20]["maturity"] == "immature"
    assert rows[20]["target_session"] == sessions[19].isoformat()
    assert rows[20]["observation_id"] == f"{ROW_ID}:20"
    # Labor Day is skipped: session 4 after 2026-09-01 is 2026-09-08.
    assert rows[5]["target_session"] == "2026-09-09"


def test_target_close_matches_the_session_horizon_builder():
    """The sidecar's close-to-close number is the horizons file's number, same bars."""
    import pandas as pd

    from master_avwap_lib.session_horizon_outcomes import build_session_horizon_observation_rows

    path = [(100.0, 101.0, 99.0, 100.0 + i) for i in range(1, 11)]
    bars = _bars(path)
    history = pd.DataFrame([{"symbol": "AAA", "side": "LONG", "last_trade_date": SCAN.isoformat(),
                             "run_id": "run1", "last_close": 100.0}])
    v2 = build_session_horizon_observation_rows(
        history, lambda s: {d: b[3] for d, b in bars.items()}, last_completed_session=LATE,
        window_sessions=None,
    )
    ours = spf.build_path_fact_rows(
        v2.rows, lambda s: bars, {ROW_ID: spf.ScanFacts(atr20=2.0)}, last_completed_session=LATE,
    )
    by_id = {row["observation_id"]: row for row in ours.rows}
    assert v2.rows
    for row in v2.rows:
        assert by_id[row["observation_id"]]["target_close"] == row["target_close"]
        assert by_id[row["observation_id"]]["side_return_pct"] == pytest.approx(row["side_return_pct"])


def test_limit_fill_shared_with_the_retest_study():
    assert rt.limit_fill({"open": 100.0, "high": 101.0, "low": 99.6}, 99.5, True) is None
    assert rt.limit_fill({"open": 100.0, "high": 101.0, "low": 99.0}, 99.5, True) == 99.5
    assert rt.limit_fill({"open": 99.0, "high": 101.0, "low": 98.0}, 99.5, True) == 99.0
    assert rt.limit_fill({"open": 101.0, "high": 102.0, "low": 100.0}, 100.5, False) == 101.0


def test_loaders_and_cli_on_scratch(tmp_path):
    import pandas as pd

    bars_dir = tmp_path / "daily_bars"
    bars_dir.mkdir()
    days = [SCAN, *_sessions(5)]
    pd.DataFrame({"datetime": pd.to_datetime(days), "open": 100.0, "high": 101.0, "low": 99.0,
                  "close": [100.0, 100.0, 100.0, 100.0, 100.0, 102.0]}).to_parquet(
        bars_dir / "AAA.parquet")
    horizons = tmp_path / "h.csv"
    with horizons.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_horizon_row()))
        writer.writeheader()
        writer.writerow(_horizon_row())
    features = tmp_path / "f.csv"
    pd.DataFrame([{"run_id": "run1", "run_date": SCAN.isoformat(),
                   "last_trade_date": SCAN.isoformat(), "symbol": "AAA", "atr20": "2.0",
                   "pct_from_current_vwap": "-4", "sector": "Technology", "other": "x"}]).to_csv(
        features, index=False)
    out = tmp_path / "out.csv"
    assert spf.main(["--horizons", str(horizons), "--daily-bars", str(bars_dir), "--features",
                     str(features), "--out", str(out), "--last-completed", "2026-09-30"]) == 0
    with out.open(newline="", encoding="utf-8") as handle:
        rows = {r["horizon_sessions"]: r for r in csv.DictReader(handle)}
    assert list(rows) == ["1", "3", "5", "10", "20"]
    assert float(rows["5"]["side_return_pct"]) == pytest.approx(2.0)
    assert float(rows["5"]["mfe_atr"]) == pytest.approx(0.5)  # high 101 vs 100, ATR 2
    assert rows["5"]["leader_pullback"] == "True"
    assert rows["10"]["unmeasured_reason"] == spf.REASON_MISSING_BAR


def test_cli_refuses_a_live_output(tmp_path):
    from setup_permutation_backfill import LiveStoreRefused

    with pytest.raises(LiveStoreRefused):
        spf.main(["--horizons", str(tmp_path / "h.csv"), "--daily-bars", str(tmp_path),
                  "--features", str(tmp_path / "f.csv"),
                  "--out", r"C:\TradingBotData\swing_path_facts.csv"])
