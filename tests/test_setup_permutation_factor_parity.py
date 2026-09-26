"""P1-4 / 4a review blocker 1: the stamp columns never reach a legacy reader.

`dist_sma50_atr` / `dist_sma200_atr` are scan-factor fields, so filling them on the
scan row added leaderboard factors and moved the tier list's "Factor Hits",
catch rates and the AI summary. The 4a columns are `perm_*` now; this pins the
scan-factor and tier exports byte-identical with the hooks on and off, on a
history with matured horizons, and checks no legacy name list knows a 4a column.
"""

from __future__ import annotations

import csv
import io
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutation_context as spc  # noqa: E402
import setup_permutations as sp  # noqa: E402
from master_avwap_lib import legacy  # noqa: E402

SYMBOLS = [f"PF{n}" for n in range(10)]


def _sessions(count: int) -> list[date]:
    day = date.today() - timedelta(days=1)
    out = []
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day -= timedelta(days=1)
    return sorted(out)


def _daily_ohlc(close: float, n: int) -> list[dict]:
    """300 sessions ending at ``close`` with a changing range, enough for every P11 column."""
    start = date(2025, 1, 2)
    bars = []
    for index in range(300):
        level = close - (299 - index) * 0.01
        width = 0.5 + ((index + n) % 11) * 0.1
        bars.append({"date": (start + timedelta(days=index)).isoformat(), "open": level,
                     "high": level + width, "low": level - width, "close": level, "volume": 1.0})
    return bars


def _history(hooks_on: bool) -> pd.DataFrame:
    rows = []
    sessions = _sessions(30)
    # P8b: a tracker view that first saw every symbol/side/family on the first session.
    tracker = {"setups": {
        f"{symbol}:{side}:{family}": {"symbol": symbol, "side": side, "setup_family": family,
                                      "scan_date": sessions[0].isoformat()}
        for symbol in SYMBOLS for side in ("LONG", "SHORT")
        for family in ("avwap_band_bounce", "post_earnings_candle_break")
    }}
    for s_index, session in enumerate(sessions):
        feature_rows = []
        for n, symbol in enumerate(SYMBOLS):
            close = 50.0 + n * 3 + s_index * (0.4 if n % 2 else -0.3) + (s_index % 4) * 0.2
            row = {
                "feature_history_schema_version": 1,
                "run_id": f"r{session.isoformat()}",
                "run_timestamp": f"{session.isoformat()}T13:05:00",
                "run_date": session.isoformat(),
                "watchlist_label": "w",
                "symbol": symbol,
                "side": "LONG" if n % 3 else "SHORT",
                "last_trade_date": session.isoformat(),
                "last_close": close,
                "atr20": 1.5 + n * 0.1,
                "relvol": 0.5 + (n % 5) * 0.6,
                "priority_score": 90 + n * 5,
                "priority_bucket": "favorite_setup" if n % 2 else "near_favorite_zone",
                "setup_family": "avwap_band_bounce" if n % 2 else "post_earnings_candle_break",
                "current_band_zone": "VWAP to UPPER_1",
                "trend_ma_alignment": bool(n % 2),
                "market_regime_label": "mixed",
            }
            if hooks_on:
                # Exactly what runner.py's hooks add, in the order it adds them.
                snapshot = {"sma20": close - 1, "sma50": close - 3, "sma100": close + 2, "sma200": close - 9,
                            "ema8": close + 0.5, "ema15": close - 0.4, "ema21": close - 0.8}
                row.update(sp.ma_distance_columns(close, row["atr20"], snapshot))
                # P11: every D1 history column filled, so none can hide behind a blank.
                row.update(sp.d1_history_columns(
                    _daily_ohlc(close, n), side=row["side"], level=close - 1.0, atr=row["atr20"],
                    zone_arm={"side": "LONG", "zone": 1} if n % 2 else None, zone_arm_evaluated=True,
                ))
                row["weekly_ema8_hold_weeks"] = n  # on the in-memory feature row only
                # S15: the earnings gap in the loop, as runner.py does (up, down and small gaps).
                row.update(sp.earnings_gap_columns({
                    "gap_date": "2025-01-02", "gap_atr_multiple": 0.5 + n * 0.3,
                    "gap_open": 50.0 + (1 if n % 3 else -1), "pre_gap_close": 50.0}))
            feature_rows.append(row)
        if hooks_on:
            sp.setup_age_columns(feature_rows, tracker, [day.isoformat() for day in sessions[: s_index + 1]])
            # S6: every trendline column filled (break, nearby line and no line across the symbols).
            for n, row in enumerate(feature_rows):
                refined = {"trendline_break_recent": n % 3 == 0, "trendline_within_alert_range": n % 3 == 1,
                           "trendline_break_candidate": {"type": "H-break" if n % 2 else "L-break"},
                           "trendline_candidate": {"type": "H-" if n % 2 else "L+"}}
                row.update(sp.trendline_columns(refined, frame_bars=250, last_close=row["last_close"],
                                                atr=row["atr20"]))
            # S15: dollar volume, market cap and a sector rank on every row, after the trendline hook.
            for n, row in enumerate(feature_rows):
                bars = [{"close": 40.0 + n, "volume": 2e6 * (n + 1)} for _ in range(25)]
                row.update(sp.liquidity_columns(bars, market_cap_m=1_500.0 * (n + 1) ** 2))
            as_of = session.isoformat()
            sectors = {f"S{k}{m}": f"sector{k}" for k in range(6) for m in range(5)}
            closes = {symbol: [(as_of, 100.0 + int(symbol[1]) + int(symbol[2]))] * 21 for symbol in sectors}
            for pairs in closes.values():
                pairs[0] = ("2000-01-01", 100.0)  # a 20-session return that differs by sector
            sectors.update({symbol: f"sector{n % 6}" for n, symbol in enumerate(SYMBOLS)})
            sp.sector_rank_columns(feature_rows, closes, sectors, as_of=as_of)
            spc.stamp_scan_rows(feature_rows, session=session, context=spc.SessionContext())
            for row in feature_rows:
                row.pop("weekly_ema8_hold_weeks", None)  # not in the runner's CSV allowlist
        rows.extend(feature_rows)
    return pd.DataFrame(rows)


def _export(tmp_path: Path, frame: pd.DataFrame) -> dict[str, bytes]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    history = tmp_path / "d1_features_history.csv"
    frame.to_csv(history, index=False)
    paths = {name: tmp_path / f"{name}.csv" for name in (
        "observations", "leaderboard", "tier_list", "tier_outcomes", "tier_performance", "tier_catch", "horizons")}
    legacy.export_scan_factor_views(history, paths["observations"], paths["leaderboard"])
    legacy.export_bot_tier_tracker_views(
        history, paths["tier_list"], paths["tier_outcomes"], paths["tier_performance"], paths["tier_catch"],
        session_horizon_path=paths["horizons"],
    )
    return {name: _without_clock(path.read_bytes()) for name, path in paths.items()}


#: Wall-clock columns: they say WHEN the export ran, so two runs a second apart differ.
CLOCK_COLUMNS = ("generated_at", "updated_at", "exported_at")


def _without_clock(data: bytes) -> bytes:
    lines = data.decode("utf-8").splitlines()
    if not lines:
        return data
    header = lines[0].split(",")
    drop = {index for index, name in enumerate(header) if name in CLOCK_COLUMNS}
    if not drop:
        return data
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for row in csv.reader(io.StringIO(data.decode("utf-8"))):
        writer.writerow([value for index, value in enumerate(row) if index not in drop])
    return out.getvalue().encode("utf-8")


def test_scan_factor_and_tier_exports_are_byte_identical_with_the_hooks_on(tmp_path):
    plain = _history(hooks_on=False)
    stamped = _history(hooks_on=True)
    assert set(sp.SCAN_ROW_COLUMNS) <= set(stamped.columns)
    assert stamped[list(sp.D1_HISTORY_COLUMNS)].notna().all().all()
    assert stamped[sp.SETUP_AGE_COLUMN].notna().all()
    assert stamped[list(sp.TRENDLINE_COLUMNS[:2])].notna().all().all()
    assert set(stamped["perm_trendline_direction"].dropna()) == {"up", "down"}
    assert stamped[list(sp.S15_COLUMNS)].notna().all().all()
    assert stamped["permutation_rule_version"].eq(sp.PERMUTATION_RULE_VERSION).all()
    off = _export(tmp_path / "off", plain)
    on = _export(tmp_path / "on", stamped)
    # Matured horizons really are in the fixture, so the leaderboard is not empty.
    assert off["observations"].count(b"\n") > 100
    assert off["leaderboard"].count(b"\n") > 1
    # Byte-identical apart from the export's own wall-clock stamp column.
    for name in off:
        assert on[name] == off[name], f"{name} changed when the 4a columns were added"


def test_no_scan_factor_row_gains_a_factor_from_the_4a_columns():
    row = _history(hooks_on=True).iloc[-1].to_dict()
    bare = {key: value for key, value in row.items() if key not in sp.SCAN_ROW_COLUMNS}
    assert legacy._scan_factor_items_from_row(row) == legacy._scan_factor_items_from_row(bare)


def test_no_legacy_name_list_knows_a_4a_column():
    from ai_jobs import miss_contrast
    from research_warehouse import features, queries, schemas

    known = set(legacy.SCAN_FACTOR_NUMERIC_FIELDS) | set(legacy.SCAN_FACTOR_BOOL_FIELDS)
    known |= set(legacy.SCAN_FACTOR_CATEGORICAL_FIELDS)
    known |= set(getattr(legacy, "SCAN_FACTOR_LIST_FIELDS", {}) or {})
    known |= set(miss_contrast.IDENTITY_COLUMNS)
    for module in (features, queries, schemas):
        source = Path(module.__file__).read_text(encoding="utf-8")
        known |= {column for column in sp.SCAN_ROW_COLUMNS if f'"{column}"' in source}
    assert not known & set(sp.SCAN_ROW_COLUMNS), known & set(sp.SCAN_ROW_COLUMNS)
    assert all(column.startswith(("perm_", "permutation_")) for column in sp.SCAN_ROW_COLUMNS)
