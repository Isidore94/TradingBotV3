"""S14 golden parity: the session-horizon export's existing columns never move.

Pinned on the code BEFORE the S14 study tag was added (branch
`claude/p9-s14-long-study-2026-09-26`, first commit). The study tag may only
APPEND a column; every original column, in order, on every row, must hash the
same as it did before.
"""

from __future__ import annotations

import csv
import hashlib
import io
import sys
from datetime import date
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_calendar  # noqa: E402
from master_avwap_lib import session_horizon_outcomes as sho  # noqa: E402

#: The header as it stood before S14.
ORIGINAL_COLUMNS = (
    "observation_id", "scan_row_id", "symbol", "side", "scan_date", "target_session",
    "horizon_sessions", "sessions_spanned", "entry_close", "entry_close_source", "target_close",
    "side_return_pct", "favorable", "measured", "maturity", "unmeasured_reason", "outcome_kind",
    "knowledge_basis", "tier", "tier_source", "priority_bucket", "setup_family", "favorite_zone",
    "collapsed_same_session",
)
LAST_COMPLETED = date(2026, 9, 4)
FIRST_SCAN = date(2026, 8, 17)
SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")
SECTORS = ("Technology", "Healthcare", "Energy", "Technology", "Financial Services", "Healthcare", "")
FAMILIES = ("top_pattern_tracking", "avwap_band_bounce", "avwap_breakout", "avwap_band_bounce",
            "favorite_zone_watch", "top_pattern_tracking", "avwap_band_bounce")


def _sessions() -> list[date]:
    out, cursor = [FIRST_SCAN], FIRST_SCAN
    while cursor < date(2026, 9, 3):
        cursor = market_calendar.next_session(cursor)
        out.append(cursor)
    return out


def golden_history() -> pd.DataFrame:
    """Two scans a session, both sides, every S14 input present, blank or odd somewhere."""
    rows = []
    for s_index, session in enumerate(_sessions()):
        for run in (1, 2):
            for n, symbol in enumerate(SYMBOLS):
                close = 40.0 + n * 7 + s_index * (0.6 if n % 2 else -0.4) + run * 0.05
                rows.append({
                    "run_id": f"r{session.isoformat()}-{run}",
                    "run_timestamp": f"{session.isoformat()}T1{run}:05:00",
                    "run_date": session.isoformat(),
                    "last_trade_date": session.isoformat(),
                    "symbol": symbol,
                    "side": "SHORT" if (n + s_index) % 4 == 0 else "LONG",
                    "last_close": round(close, 4),
                    "priority_bucket": "favorite_setup" if n % 2 else "near_favorite_zone",
                    "setup_family": FAMILIES[n],
                    "favorite_zone": "VWAP to UPPER_1" if n % 3 else "",
                    "tier": "A" if n % 2 else "",
                    "sector": SECTORS[n],
                    "pct_from_current_vwap": -12.0 + ((n * 3 + s_index) % 15),
                    "rs_vs_industry": None if n == 6 else round(-3.0 + n * 1.1 - s_index * 0.05, 3),
                    "spy_above_sma20": s_index % 3 != 0,
                    "trend_20d": ("UP", "DOWN", "SIDEWAYS")[(n + s_index) % 3],
                    "htf_trend_4h": ("UP", "NEUTRAL", "")[(n + s_index) % 3],
                })
    return pd.DataFrame(rows)


def golden_closes() -> dict[str, dict[date, float]]:
    days = []
    cursor = FIRST_SCAN
    while cursor <= LAST_COMPLETED:
        days.append(cursor)
        cursor = market_calendar.next_session(cursor)
    out: dict[str, dict[date, float]] = {}
    for n, symbol in enumerate(SYMBOLS):
        if symbol == "FFF":
            continue  # no bars: unmeasured rows with a reason
        out[symbol] = {day: round(40.0 + n * 7 + i * (0.5 if n % 2 else -0.3), 4)
                       for i, day in enumerate(days) if not (symbol == "EEE" and i == 7)}
    return out


def golden_rows() -> list[dict]:
    closes = golden_closes()
    return sho.build_session_horizon_observation_rows(
        golden_history(), lambda symbol: closes.get(symbol), last_completed_session=LAST_COMPLETED,
    ).rows


def original_projection(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(ORIGINAL_COLUMNS), lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in ORIGINAL_COLUMNS})
    return buffer.getvalue()


#: Recorded on the pre-S14 code (59707d8b) from `original_projection(golden_rows())`.
GOLDEN_ROW_COUNT = 392
GOLDEN_SHA256 = "23d402f5387f572c60856aaf48d663ae40118a29bd04e6a71cdbbc9ecf44b257"


def test_the_original_header_is_unchanged_and_new_columns_only_append():
    assert tuple(sho.SESSION_HORIZON_OUTCOME_COLUMNS[: len(ORIGINAL_COLUMNS)]) == ORIGINAL_COLUMNS


def test_every_original_column_on_every_row_hashes_as_before_s14():
    rows = golden_rows()
    assert len(rows) == GOLDEN_ROW_COUNT
    digest = hashlib.sha256(original_projection(rows).encode("utf-8")).hexdigest()
    assert digest == GOLDEN_SHA256
