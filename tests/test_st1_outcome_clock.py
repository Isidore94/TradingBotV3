"""Packet ST1 - each outcome gets its true meaning and its true clock.

Ten tests, one per packet item's behaviour. Every one of them drives the real
builder, the real reader or the real `Headline`; none of them asserts on source
text and none of them hand-writes the dict the code under test is supposed to
produce.

The three facts these tests pin, in the trader's words (2026-09-06):

1. *"Name and version favorable price-direction observations separately from
   simulated trade outcomes."* The tier outcomes file's `win` column is the SIGN
   OF A CLOSE-TO-CLOSE PERCENT MOVE between two of that symbol's own scan rows.
   It is not a stop-rule verdict and it is not R. `outcome_kind` says so and the
   `Headline` prints "favorable", never "win rate", for it.
2. *"Define exact exchange-session horizons from the entry session and
   completed-bar data, independent of later scan membership."* v1 walks
   `idx + horizon` into the SYMBOL'S OWN scan rows, so "5 sessions later" lands
   wherever the next six watchlist appearances happen to be. v2 walks the
   exchange calendar and reads the bar ON the target session, or says why it
   could not.
3. *"Centralize eligible-row reading across family docs, tier reports, desk, and
   Away: one declared outcome definition, horizon, knowledge basis, maturity
   rule, window, and missingness policy."* Today `setup_docs._all_family_outcomes`
   and `autopilot_core.swing_family_records` drop `stale_horizon` rows and
   `build_bot_tier_performance_rows` does not, so one reader counts what another
   throws away, off the same file.

Nothing here fetches from a network, opens a live store or writes anywhere but
`tmp_path`. `conftest.py` has already pointed `TRADINGBOTV3_DATA_DIR` at a temp
directory before any of these imports run.
"""

from __future__ import annotations

import csv
import io
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import market_calendar  # noqa: E402
from master_avwap_lib import legacy  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
GOLDEN_PATH = FIXTURE_DIR / "st1_tier_outcomes_golden.csv"

#: The tier outcome header EXACTLY as `main` wrote it on 2026-09-06, pinned as a
#: literal rather than read from `legacy.TIER_OUTCOME_COLUMNS` - the packet
#: APPENDS `outcome_kind` to that list, so reading it would make the golden
#: comparison shrink to fit the fix instead of proving it additive.
GOLDEN_V1_COLUMNS = (
    "observation_id",
    "scan_row_id",
    "run_id",
    "run_timestamp",
    "run_date",
    "watchlist_label",
    "scan_date",
    "future_scan_date",
    "horizon_sessions",
    "tier",
    "tier_source",
    "symbol",
    "side",
    "priority_bucket",
    "priority_score",
    "setup_family",
    "favorite_zone",
    "entry_close",
    "future_close",
    "raw_return_pct",
    "side_return_pct",
    "win",
    "spy_forward_return_pct",
    "spy_relative_side_return_pct",
    "sessions_spanned",
    "stale_horizon",
    "positive_scan_factor_match_count",
    "positive_scan_factor_matches",
)

OUTCOME_KIND_SCANROW_V1_TEXT = "favorable_direction_scanrow_v1"
OUTCOME_KIND_SESSION_V2_TEXT = "favorable_direction_session_v2"


# ---------------------------------------------------------------------------
# Synthetic scan history. No live store, no network, no clock.
# ---------------------------------------------------------------------------
def _scan_row(
    symbol: str,
    scan_date: str,
    close: float,
    *,
    side: str = "LONG",
    family: str = "avwap_breakout",
    bucket: str = "favorite_setup",
    assigned_tier: str = "",
    score: float = 120.0,
) -> dict:
    """One row of the scan feature history, shaped like the real file.

    `assigned_tier` is PRESENT AND EMPTY on an old row - that is the value the
    live file carries for every row written before P4 widened it, and the one
    `tier_for_tracker_row` has to read as "absent" rather than as a tier.
    """
    return {
        "run_id": f"run-{scan_date}",
        "run_timestamp": f"{scan_date}T13:00:00",
        "run_date": scan_date,
        "watchlist_label": "swing_longs",
        "symbol": symbol,
        "side": side,
        "last_trade_date": scan_date,
        "last_close": float(close),
        "priority_bucket": bucket,
        "priority_score": float(score),
        "setup_family": family,
        "favorite_zone": "AVWAPE to UPPER_1",
        "current_band_zone": "AVWAPE to UPPER_1",
        "trend_20d": "UP",
        "assigned_tier": assigned_tier,
    }


#: Ten consecutive June 2026 sessions. 2026-06-06/07 and 06-13/14 are weekends,
#: which is exactly the drift v1 cannot see and v2 must.
JUNE_SESSIONS = (
    "2026-06-01",
    "2026-06-02",
    "2026-06-03",
    "2026-06-04",
    "2026-06-05",
    "2026-06-08",
    "2026-06-09",
    "2026-06-10",
    "2026-06-11",
    "2026-06-12",
)


def _closes_map(pairs: dict[str, float]) -> dict[date, float]:
    return {date.fromisoformat(day): float(close) for day, close in pairs.items()}


# ---------------------------------------------------------------------------
# 1. The v1 clock counts scan rows; the v2 clock counts exchange sessions.
# ---------------------------------------------------------------------------
def test_v1_compares_scan_rows_while_v2_walks_five_exchange_sessions():
    """An irregularly scanned symbol: v1's "5 sessions" is 11 sessions away.

    IRREG is scanned on 2026-06-01..06-05 and then not again until 2026-06-16.
    v1's horizon-5 row from 06-01 is `idx + 5` into IRREG's OWN scan rows, which
    is 06-16 - eleven exchange sessions later. That number is UNCHANGED by this
    packet (the v1 file keeps every row and every value); what changes is that a
    second, versioned builder answers the question the column claims to answer:
    the close on the FIFTH SESSION after the entry session, 2026-06-08.

    FALLBACK is the same shape with no bar on its entry session, so its
    `entry_close` falls back to the scan row's own close and SAYS it did.
    """
    from master_avwap_lib.session_horizon_outcomes import (
        OUTCOME_KIND_SESSION_V2,
        build_session_horizon_observation_rows,
    )

    scan_days = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-16"]
    rows = []
    for symbol in ("IRREG", "FALLBACK"):
        for index, day in enumerate(scan_days):
            close = 150.0 if day == "2026-06-16" else 100.0 + index
            rows.append(_scan_row(symbol, day, close))
    history = pd.DataFrame(rows)

    # Bars on EVERY session, including the five nobody scanned.
    bars = _closes_map(
        {
            "2026-06-01": 100.0,
            "2026-06-02": 101.0,
            "2026-06-03": 102.0,
            "2026-06-04": 103.0,
            "2026-06-05": 104.0,
            "2026-06-08": 110.0,
            "2026-06-09": 111.0,
            "2026-06-10": 112.0,
            "2026-06-11": 113.0,
            "2026-06-12": 114.0,
            "2026-06-15": 140.0,
            "2026-06-16": 150.0,
        }
    )
    fallback_bars = {day: close for day, close in bars.items() if day != date(2026, 6, 1)}

    def closes_for(symbol: str):
        return bars if symbol == "IRREG" else fallback_bars

    # --- v1 is unchanged: it still compares two SCAN ROWS eleven sessions apart.
    v1 = legacy.build_scan_factor_observation_rows(history, horizons=(5,))
    v1_row = next(
        row for row in v1 if row["symbol"] == "IRREG" and row["scan_date"] == "2026-06-01"
    )
    assert v1_row["future_scan_date"] == "2026-06-16"
    assert v1_row["future_close"] == 150.0
    assert v1_row["side_return_pct"] == pytest.approx(50.0)

    # --- v2 walks the calendar and reads the bar ON the fifth session.
    built = build_session_horizon_observation_rows(
        history,
        closes_for,
        horizons=(5,),
        last_completed_session=date(2026, 6, 30),
    )
    v2_row = next(
        row
        for row in built.rows
        if row["symbol"] == "IRREG" and row["scan_date"] == "2026-06-01"
    )
    assert v2_row["target_session"] == "2026-06-08"
    assert v2_row["target_close"] == 110.0
    assert v2_row["entry_close"] == 100.0
    assert v2_row["entry_close_source"] != "scan_row"
    assert v2_row["side_return_pct"] == pytest.approx(10.0)
    assert v2_row["sessions_spanned"] == 5
    assert v2_row["horizon_sessions"] == 5
    assert v2_row["favorable"] is True
    assert v2_row["measured"] is True
    assert v2_row["maturity"] == "mature"
    assert v2_row["outcome_kind"] == OUTCOME_KIND_SESSION_V2 == OUTCOME_KIND_SESSION_V2_TEXT
    assert v2_row["knowledge_basis"] == "entry_session_close_to_target_session_close"

    # The two files join 1:1 on `observation_id`.
    assert v2_row["observation_id"] == v1_row["observation_id"]

    # --- the entry-close fallback names itself.
    fb_row = next(
        row
        for row in built.rows
        if row["symbol"] == "FALLBACK" and row["scan_date"] == "2026-06-01"
    )
    assert fb_row["entry_close_source"] == "scan_row"
    assert fb_row["entry_close"] == 100.0
    assert fb_row["target_close"] == 110.0


# ---------------------------------------------------------------------------
# 2. Labor Day is skipped, and the span is counted in SESSIONS.
# ---------------------------------------------------------------------------
def test_labor_day_is_skipped_and_the_span_is_counted_in_sessions():
    """Friday 2026-09-04 + 1 session is Tuesday 2026-09-08, not Monday.

    Monday 2026-09-07 is Labor Day. `numpy.busday_count` counts it, so today's
    `sessions_between` (which the export always calls WITHOUT a calendar) answers
    2 for that pair and the basis string it hands back says "business days" -
    a number nobody measured, presented under the name of one that was.
    """
    from setup_tracker_ledger import horizon_drift, sessions_between

    assert market_calendar.is_session(date(2026, 9, 7)) is False

    # --- the ledger, handed the exchange calendar, counts SESSIONS and says so.
    calendar_days = [
        date(2026, 9, day) for day in range(1, 31) if market_calendar.is_session(date(2026, 9, day))
    ]
    assert sessions_between("2026-09-04", "2026-09-08") == 2  # business days, unchanged
    assert sessions_between("2026-09-04", "2026-09-08", calendar_days) == 1

    drift = horizon_drift("2026-09-04", "2026-09-08", 1, calendar=calendar_days)
    assert drift["sessions_spanned"] == 1
    assert drift["stale_horizon"] is False
    assert "session" in drift["basis"].lower()
    assert "business day" not in drift["basis"].lower()

    # Without a calendar the fallback stays and still names itself.
    plain = horizon_drift("2026-09-04", "2026-09-08", 1)
    assert plain["sessions_spanned"] == 2
    assert "business day" in plain["basis"].lower()

    # --- and the v2 builder lands the target on the session after the holiday.
    from master_avwap_lib.session_horizon_outcomes import (
        build_session_horizon_observation_rows,
    )

    history = pd.DataFrame([_scan_row("LABOR", "2026-09-04", 100.0)])
    bars = _closes_map({"2026-09-04": 100.0, "2026-09-08": 106.0})
    built = build_session_horizon_observation_rows(
        history,
        lambda symbol: bars,
        horizons=(1,),
        last_completed_session=date(2026, 9, 11),
    )
    row = built.rows[0]
    assert row["target_session"] == "2026-09-08"
    assert row["target_close"] == 106.0
    assert row["sessions_spanned"] == 1
    assert row["side_return_pct"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# 3. A missing target-session bar is unmeasured, never the next bar.
# ---------------------------------------------------------------------------
def test_a_missing_target_session_bar_is_unmeasured_not_the_next_bar():
    """No bar on 2026-06-08 -> unmeasured. NOT 2026-06-09's 111.0.

    *"Missing target-session data remains unmeasured with a reason. Do not
    forward-fill to the next scan and call it the original horizon."*
    """
    from master_avwap_lib.session_horizon_outcomes import (
        build_session_horizon_observation_rows,
    )

    history = pd.DataFrame([_scan_row("GAP", "2026-06-01", 100.0)])
    bars = _closes_map(
        {
            "2026-06-01": 100.0,
            "2026-06-02": 101.0,
            "2026-06-03": 102.0,
            "2026-06-04": 103.0,
            "2026-06-05": 104.0,
            # 2026-06-08 - the target session - is MISSING.
            "2026-06-09": 111.0,
            "2026-06-10": 112.0,
        }
    )
    built = build_session_horizon_observation_rows(
        history,
        lambda symbol: bars,
        horizons=(5,),
        last_completed_session=date(2026, 6, 30),
    )
    row = built.rows[0]
    assert row["target_session"] == "2026-06-08"
    assert row["measured"] is False
    assert row["unmeasured_reason"] == "no_bar_for_target_session"
    assert row["favorable"] in ("", None)
    assert row["side_return_pct"] in ("", None)
    assert row["target_close"] in ("", None)
    # The forward-fill this rule exists to forbid.
    assert row["target_close"] != 111.0
    assert row["sessions_spanned"] == 5


# ---------------------------------------------------------------------------
# 4. An immature horizon is pending, never eligible.
# ---------------------------------------------------------------------------
def test_a_target_session_after_the_last_completed_one_is_pending_not_eligible():
    """Entry 2026-06-01, horizon 5, last completed session 2026-06-05.

    The target session (2026-06-08) has not closed. That is `immature` and
    `pending` - never a row in `rows`, and never an unmeasured EXCLUSION either:
    a horizon that has not arrived yet is a different fact from one that arrived
    and could not be read.
    """
    from master_avwap_lib.session_horizon_outcomes import (
        build_session_horizon_observation_rows,
    )
    from swing_evidence import POLICY_SESSION_V2, read_eligible_rows

    history = pd.DataFrame(
        [
            _scan_row("MATURE", "2026-06-01", 100.0),
            _scan_row("MATURE", "2026-06-02", 100.0),
        ]
    )
    bars = _closes_map(
        {
            "2026-06-01": 100.0,
            "2026-06-02": 100.0,
            "2026-06-03": 102.0,
            "2026-06-04": 103.0,
            "2026-06-05": 104.0,
            "2026-06-08": 110.0,
            "2026-06-09": 111.0,
        }
    )
    built = build_session_horizon_observation_rows(
        history,
        lambda symbol: bars,
        horizons=(5,),
        last_completed_session=date(2026, 6, 5),
    )
    assert len(built.rows) == 2
    for row in built.rows:
        assert row["maturity"] == "immature"
        assert row["measured"] is False
        assert row["unmeasured_reason"] == "target_session_not_complete"

    read = read_eligible_rows(built.rows, POLICY_SESSION_V2)
    assert len(read.rows) == 0
    assert len(read.pending) == 2
    assert read.source_rows == 2
    assert read.reconciles is True
    assert sum(read.excluded.values()) == 0


# ---------------------------------------------------------------------------
# 5. One file, three readers, one eligible n.
# ---------------------------------------------------------------------------
def _write_tier_outcomes_csv(path: Path, rows: list[dict]) -> None:
    """Write a tier-outcomes CSV with EVERY v1 column present.

    Unknown cells are PRESENT AND EMPTY, which is how the live file carries
    them; read back through `csv.DictReader` they are `""` and through pandas
    they are NaN.
    """
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GOLDEN_V1_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in GOLDEN_V1_COLUMNS})


def _outcome_csv_row(
    index: int,
    *,
    win: bool,
    stale: str,
    scan_date: str = "2026-06-03",
    tier_source: str = "assigned",
) -> dict:
    return {
        "observation_id": f"OBS{index}:5",
        "scan_row_id": f"OBS{index}",
        "run_id": f"run-{index}",
        "run_timestamp": f"{scan_date}T13:00:00",
        "run_date": scan_date,
        "watchlist_label": "swing_longs",
        "scan_date": scan_date,
        "future_scan_date": "2026-06-10",
        "horizon_sessions": 5,
        "tier": "S",
        "tier_source": tier_source,
        "symbol": f"SYM{index}",
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "priority_score": 120.0,
        "setup_family": "avwap_breakout",
        "favorite_zone": "AVWAPE to UPPER_1",
        "entry_close": 100.0,
        "future_close": 105.0 if win else 95.0,
        "raw_return_pct": 5.0 if win else -5.0,
        "side_return_pct": 5.0 if win else -5.0,
        "win": "True" if win else "False",
        "spy_forward_return_pct": "",
        "spy_relative_side_return_pct": "",
        "sessions_spanned": 18 if stale == "True" else 5,
        "stale_horizon": stale,
        "positive_scan_factor_match_count": 0,
        "positive_scan_factor_matches": "",
    }


def test_the_three_readers_report_the_same_eligible_count_on_one_file(tmp_path, monkeypatch):
    """Four horizon-5 rows, one flagged stale. Every reader must say 3.

    `setup_docs._all_family_outcomes` and `autopilot_core.swing_family_records`
    drop an explicit `stale_horizon == True`; `build_bot_tier_performance_rows`
    reads the same rows and drops nothing, so the tier report today counts a row
    the two trader-facing surfaces threw away. Same file, same question, two
    answers.
    """
    import setup_docs

    csv_rows = [
        _outcome_csv_row(1, win=True, stale="False"),
        _outcome_csv_row(2, win=True, stale="False"),
        _outcome_csv_row(3, win=False, stale="False"),
        _outcome_csv_row(4, win=True, stale="True"),
    ]
    path = tmp_path / "master_avwap_tier_outcomes.csv"
    _write_tier_outcomes_csv(path, csv_rows)

    window = ("2026-06-01", "2026-06-30")

    # --- reader 1: the setup docs' family record.
    setup_docs.clear_family_outcome_cache()
    monkeypatch.setattr(setup_docs, "_family_outcomes_path", lambda: path)
    monkeypatch.setattr(setup_docs, "_family_outcomes_window", lambda: window)
    docs_rows = setup_docs.family_headline_rows()
    setup_docs.clear_family_outcome_cache()
    assert docs_rows["avwap_breakout"]["n"] == 3

    # --- reader 2: the AWAY digest's ranking record.
    import autopilot_core

    records = autopilot_core.swing_family_records(path, window=window)
    assert records["avwap_breakout"]["wins"] + records["avwap_breakout"]["losses"] == 3

    # --- reader 3: the tier performance export, off the SAME rows.
    with open(path, newline="", encoding="utf-8-sig") as handle:
        file_rows = [dict(row) for row in csv.DictReader(handle)]
    for row in file_rows:
        row["horizon_sessions"] = int(row["horizon_sessions"])
        row["side_return_pct"] = float(row["side_return_pct"])
    performance = legacy.build_bot_tier_performance_rows(
        file_rows,
        file_rows,
        lookback_days=365,
        reference_date="2026-06-30",
    )
    cell = next(
        row
        for row in performance
        if row["tier"] == "S" and row["side"] == "LONG" and row["horizon_sessions"] == 5
    )
    assert cell["observation_count"] == 3


# ---------------------------------------------------------------------------
# 6. A favorable move is never printed as a win rate.
# ---------------------------------------------------------------------------
def test_a_favorable_move_is_never_printed_as_a_win_rate():
    """A close-to-close percent move is a DIRECTION, not a stop-rule verdict.

    The scenario the wording hides: a long entered at 100 with its D1 support at
    96 traded down to 94 on day two - the stop-at-a-level rule the docstring
    claims this column implements would have been out at a loss - and then
    closed at 108 on the target session. The tier outcomes file records
    `win = True`, because all it ever measured was the sign of the close-to-close
    move. Both v1 and v2 rows carry `outcome_kind`, and the `Headline` built from
    them says "favorable", never "win rate".
    """
    import swing_headline

    # Both writers stamp it. The stop was breached at 94 and the close was 108.
    history = pd.DataFrame(
        [
            _scan_row("STOPPED", day, close)
            for day, close in zip(
                ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08"),
                (100.0, 94.0, 97.0, 99.0, 101.0, 108.0),
            )
        ]
    )
    observations = legacy.build_scan_factor_observation_rows(history, horizons=(5,))
    assert observations[0]["win"] is True
    assert observations[0]["outcome_kind"] == OUTCOME_KIND_SCANROW_V1_TEXT
    outcomes = legacy.build_bot_tier_outcome_rows(history, observations)
    assert outcomes[0]["win"] is True
    assert outcomes[0]["outcome_kind"] == OUTCOME_KIND_SCANROW_V1_TEXT
    assert "outcome_kind" in legacy.TIER_OUTCOME_COLUMNS

    # 56 favorable out of 90: 62%, Wilson lower bound 52%.
    tracker_rows = [
        {"win": "True" if index < 56 else "False", "side_return_pct": 1.0 if index < 56 else -1.0}
        for index in range(90)
    ]
    favorable = swing_headline.headline_from_tracker_rows("avwap_breakout", tracker_rows)
    assert favorable.n == 90
    assert favorable.win_rate == pytest.approx(56 / 90)
    assert favorable.win_rate_lb == pytest.approx(0.518997640581835)
    assert favorable.outcome_kind == "favorable_direction"

    row = favorable.as_row()
    rendered = swing_headline.format_win_rate(row)
    assert "62%" in rendered
    assert "n=90" in rendered
    assert ">=52%" in rendered
    assert "favorable" in rendered.lower()

    sentence = favorable.sentence()
    assert "favorable" in sentence.lower()
    assert "win rate" not in sentence.lower()

    # An R-graded headline keeps the words it earned.
    traded = swing_headline.headline_from_outcomes(
        "recipe_grid",
        [{"close_r": 1.0}] * 56 + [{"close_r": -1.0}] * 34,
    )
    assert traded.outcome_kind == "trade_r"
    assert "win rate" in traded.sentence().lower()
    assert "favorable" not in traded.sentence().lower()

    # The trader's main swing screen carries the relabelled header.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ui.models.setup_table_model import SetupTableModel

    app = QApplication.instance() or QApplication([])
    assert app is not None
    model = SetupTableModel([])
    column = [key for key, _label in SetupTableModel.COLUMNS].index("family_win_rate")
    header = model.headerData(column, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
    assert "favorable" in str(header).lower()
    assert "win" not in str(header).lower()

    # One helper answers "what kind of outcome is this row", for every reader.
    from swing_evidence import OUTCOME_KIND_SCANROW_V1, outcome_kind_of

    assert OUTCOME_KIND_SCANROW_V1 == OUTCOME_KIND_SCANROW_V1_TEXT
    # An OLD row: the key is PRESENT AND EMPTY, and reads as the v1 kind.
    assert outcome_kind_of({"win": "True", "outcome_kind": ""}) == OUTCOME_KIND_SCANROW_V1_TEXT
    assert outcome_kind_of({"win": "True"}) == OUTCOME_KIND_SCANROW_V1_TEXT
    assert (
        outcome_kind_of({"outcome_kind": OUTCOME_KIND_SESSION_V2_TEXT})
        == OUTCOME_KIND_SESSION_V2_TEXT
    )


# ---------------------------------------------------------------------------
# 7. A duplicate (scan_row_id, horizon) makes one row and is counted.
# ---------------------------------------------------------------------------
def test_a_duplicate_scan_row_and_horizon_yields_one_row_and_is_counted():
    """Seven scan rows, one of them present twice: seven v2 rows, one dropped.

    A silently de-duplicated row is a row nobody can reconcile. The builder
    returns the count so the export can state it.
    """
    from master_avwap_lib.session_horizon_outcomes import (
        build_session_horizon_observation_rows,
    )

    scan_days = JUNE_SESSIONS[:7]
    rows = [_scan_row("DUPE", day, 100.0 + index) for index, day in enumerate(scan_days)]
    rows.append(dict(rows[0]))  # the SAME symbol, scan date, run id and timestamp
    history = pd.DataFrame(rows)

    bars = _closes_map(
        {
            day: 100.0 + index
            for index, day in enumerate(
                (
                    "2026-06-01",
                    "2026-06-02",
                    "2026-06-03",
                    "2026-06-04",
                    "2026-06-05",
                    "2026-06-08",
                    "2026-06-09",
                    "2026-06-10",
                    "2026-06-11",
                    "2026-06-12",
                    "2026-06-15",
                    "2026-06-16",
                    "2026-06-17",
                )
            )
        }
    )
    built = build_session_horizon_observation_rows(
        history,
        lambda symbol: bars,
        horizons=(5,),
        last_completed_session=date(2026, 6, 30),
    )
    assert len(built.rows) == 7
    assert built.dropped_duplicates == 1
    keys = [(row["scan_row_id"], row["horizon_sessions"]) for row in built.rows]
    assert len(set(keys)) == 7


# ---------------------------------------------------------------------------
# 8. A derived tier never validates the assigned ones.
# ---------------------------------------------------------------------------
def test_a_derived_tier_never_validates_the_assigned_tier_cells():
    """Six S-tier outcome rows: four recorded at the decision, two reconstructed.

    *"Keep bucket-derived historic tiers separate from tiers actually assigned at
    the decision time. Never use reconstructed labels to validate shipped S/A
    performance."* Today the S/LONG performance cell counts all six and says
    nothing about the split.
    """
    rows = []
    for index in range(1, 7):
        assigned = "S" if index <= 4 else ""  # PRESENT AND EMPTY on the old rows
        for day, close in zip(
            ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08"),
            (100.0, 101.0, 102.0, 103.0, 104.0, 110.0),
        ):
            rows.append(_scan_row(f"T{index}", day, close, assigned_tier=assigned))
    history = pd.DataFrame(rows)

    observations = legacy.build_scan_factor_observation_rows(history, horizons=(5,))
    outcomes = legacy.build_bot_tier_outcome_rows(history, observations)
    assert len(outcomes) == 6
    assert sorted(row["tier_source"] for row in outcomes) == (
        ["assigned"] * 4 + ["derived_from_bucket"] * 2
    )

    default_rows = legacy.build_bot_tier_performance_rows(
        outcomes, observations, lookback_days=365, reference_date="2026-06-30"
    )
    default_cell = next(
        row
        for row in default_rows
        if row["tier"] == "S" and row["side"] == "LONG" and row["horizon_sessions"] == 5
    )
    assert default_cell["observation_count"] == 6
    assert default_cell["n_assigned_tier"] == 4
    assert default_cell["n_derived_tier"] == 2

    assigned_rows = legacy.build_bot_tier_performance_rows(
        outcomes,
        observations,
        lookback_days=365,
        reference_date="2026-06-30",
        assigned_only=True,
    )
    assigned_cell = next(
        row
        for row in assigned_rows
        if row["tier"] == "S" and row["side"] == "LONG" and row["horizon_sessions"] == 5
    )
    assert assigned_cell["observation_count"] == 4
    assert assigned_cell["n_derived_tier"] == 0

    from swing_evidence import tier_split

    assert tier_split(outcomes) == {"assigned": 4, "derived": 2, "unknown": 0}
    # A row whose source cell is present and empty is UNKNOWN, never assigned.
    assert tier_split([{"tier_source": ""}, {"tier_source": "assigned"}]) == {
        "assigned": 1,
        "derived": 0,
        "unknown": 1,
    }


# ---------------------------------------------------------------------------
# 9. The eligible read reconciles and names every exclusion.
# ---------------------------------------------------------------------------
def test_the_eligible_read_reconciles_and_names_every_exclusion_reason():
    """Every source row lands in exactly one bucket, and the bucket has a name.

    A reader that quietly drops rows publishes a number nobody can check. Eight
    v1 rows in, three eligible, five excluded for five different reasons; then
    six v2 rows in, two eligible, one pending, three excluded.
    """
    from evidence_stats import LATELY_SESSIONS, SWING_HORIZON_SESSIONS
    from swing_evidence import (
        POLICY_SCANROW_V1,
        POLICY_SESSION_V2,
        describe,
        read_eligible_rows,
    )

    assert POLICY_SCANROW_V1.outcome_kind == OUTCOME_KIND_SCANROW_V1_TEXT
    assert POLICY_SCANROW_V1.horizon_sessions == SWING_HORIZON_SESSIONS
    assert POLICY_SCANROW_V1.window_sessions == LATELY_SESSIONS
    assert POLICY_SESSION_V2.outcome_kind == OUTCOME_KIND_SESSION_V2_TEXT

    window = ("2026-06-01", "2026-06-30")
    v1_rows = [
        _outcome_csv_row(1, win=True, stale="False"),
        _outcome_csv_row(2, win=False, stale="False"),
        _outcome_csv_row(3, win=True, stale="False"),
        _outcome_csv_row(4, win=True, stale="True"),  # stale_horizon
        _outcome_csv_row(5, win=True, stale="False", scan_date="2026-05-01"),  # outside_window
        _outcome_csv_row(6, win=True, stale="False"),  # wrong_horizon, set below
        _outcome_csv_row(7, win=True, stale="False"),  # unreadable, set below
        dict(_outcome_csv_row(3, win=True, stale="False")),  # duplicate observation_id
    ]
    v1_rows[5]["horizon_sessions"] = 10
    v1_rows[6]["horizon_sessions"] = ""  # present and empty - unreadable, not zero

    read = read_eligible_rows(v1_rows, POLICY_SCANROW_V1, end=window[1])
    assert read.source_rows == 8
    assert len(read.rows) == 3
    assert len(read.pending) == 0
    assert dict(read.excluded) == {
        "wrong_horizon": 1,
        "outside_window": 1,
        "stale_horizon": 1,
        "unreadable": 1,
        "duplicate": 1,
    }
    assert read.reconciles is True
    assert len(read.rows) + len(read.pending) + sum(read.excluded.values()) == read.source_rows

    line = describe(POLICY_SCANROW_V1, read)
    assert OUTCOME_KIND_SCANROW_V1_TEXT in line
    assert "5" in line  # the declared horizon
    assert "3" in line  # eligible

    v2_rows = [
        {
            "observation_id": "V2A:5",
            "scan_row_id": "V2A",
            "scan_date": "2026-06-01",
            "target_session": "2026-06-08",
            "horizon_sessions": 5,
            "sessions_spanned": 5,
            "favorable": True,
            "measured": True,
            "maturity": "mature",
            "unmeasured_reason": "",
            "outcome_kind": OUTCOME_KIND_SESSION_V2_TEXT,
            "setup_family": "avwap_breakout",
        },
        {
            "observation_id": "V2B:5",
            "scan_row_id": "V2B",
            "scan_date": "2026-06-02",
            "target_session": "2026-06-09",
            "horizon_sessions": 5,
            "sessions_spanned": 5,
            "favorable": False,
            "measured": True,
            "maturity": "mature",
            "unmeasured_reason": "",
            "outcome_kind": OUTCOME_KIND_SESSION_V2_TEXT,
            "setup_family": "avwap_breakout",
        },
        {
            "observation_id": "V2C:5",
            "scan_row_id": "V2C",
            "scan_date": "2026-06-25",
            "target_session": "2026-07-02",
            "horizon_sessions": 5,
            "sessions_spanned": 5,
            "favorable": "",
            "measured": False,
            "maturity": "immature",
            "unmeasured_reason": "target_session_not_complete",
            "outcome_kind": OUTCOME_KIND_SESSION_V2_TEXT,
            "setup_family": "avwap_breakout",
        },
        {
            "observation_id": "V2D:5",
            "scan_row_id": "V2D",
            "scan_date": "2026-06-03",
            "target_session": "2026-06-10",
            "horizon_sessions": 5,
            "sessions_spanned": 5,
            "favorable": "",
            "measured": False,
            "maturity": "mature",
            "unmeasured_reason": "no_bar_for_target_session",
            "outcome_kind": OUTCOME_KIND_SESSION_V2_TEXT,
            "setup_family": "avwap_breakout",
        },
        {
            "observation_id": "V2E:10",
            "scan_row_id": "V2E",
            "scan_date": "2026-06-03",
            "target_session": "2026-06-17",
            "horizon_sessions": 10,
            "sessions_spanned": 10,
            "favorable": True,
            "measured": True,
            "maturity": "mature",
            "unmeasured_reason": "",
            "outcome_kind": OUTCOME_KIND_SESSION_V2_TEXT,
            "setup_family": "avwap_breakout",
        },
        {
            "observation_id": "V2F:5",
            "scan_row_id": "V2F",
            "scan_date": "2026-05-01",
            "target_session": "2026-05-08",
            "horizon_sessions": 5,
            "sessions_spanned": 5,
            "favorable": True,
            "measured": True,
            "maturity": "mature",
            "unmeasured_reason": "",
            "outcome_kind": OUTCOME_KIND_SESSION_V2_TEXT,
            "setup_family": "avwap_breakout",
        },
    ]
    v2_read = read_eligible_rows(v2_rows, POLICY_SESSION_V2, end=window[1])
    assert v2_read.source_rows == 6
    assert len(v2_read.rows) == 2
    assert len(v2_read.pending) == 1
    assert dict(v2_read.excluded) == {
        "unmeasured:no_bar_for_target_session": 1,
        "wrong_horizon": 1,
        "outside_window": 1,
    }
    assert v2_read.reconciles is True


# ---------------------------------------------------------------------------
# 10. The golden. Pinned from the OLD code before any fix existed.
# ---------------------------------------------------------------------------
GOLDEN_SCAN_DAYS = JUNE_SESSIONS
GOLDEN_FAMILIES = (
    # (family, symbol prefix, count, wins)
    ("avwap_breakout", "B", 10, 7),
    ("favorite_zone_watch", "F", 10, 5),
)


def build_golden_history() -> pd.DataFrame:
    """The synthetic scan history the golden fixture was generated from.

    Deterministic and self-contained: no clock, no store, no network. SPY is in
    the frame with real closes so `build_scan_factor_observation_rows` never
    falls back to `_cached_spy_closes()` and never touches a parquet store.

    `GAPPY` is scanned five times in June and then not until 2026-07-15, which
    is what a `stale_horizon` row actually looks like: its declared horizon is 5
    and it spans 31 business days.
    """
    rows: list[dict] = []
    for index, day in enumerate(GOLDEN_SCAN_DAYS):
        rows.append(
            _scan_row("SPY", day, 400.0 + index, family="benchmark", bucket="", score=0.0)
        )
    for family, prefix, count, wins in GOLDEN_FAMILIES:
        for symbol_index in range(count):
            up = symbol_index < wins
            symbol = f"{prefix}{symbol_index}"
            bucket = "favorite_setup" if symbol_index % 2 == 0 else "near_favorite_zone"
            assigned = "S" if symbol_index % 3 == 0 else ""
            for index, day in enumerate(GOLDEN_SCAN_DAYS):
                close = 100.0 + index * (1.5 if up else -1.5)
                rows.append(
                    _scan_row(
                        symbol,
                        day,
                        close,
                        family=family,
                        bucket=bucket,
                        assigned_tier=assigned,
                        score=130.0 - symbol_index,
                    )
                )
    gappy_days = ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-07-15")
    for index, day in enumerate(gappy_days):
        rows.append(
            _scan_row(
                "GAPPY",
                day,
                100.0 + index * 4.0,
                family="favorite_zone_watch",
                bucket="favorite_setup",
                assigned_tier="S",
            )
        )
    return pd.DataFrame(rows)


def _serialize_v1(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(GOLDEN_V1_COLUMNS), lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in GOLDEN_V1_COLUMNS})
    return buffer.getvalue()


def test_regenerate_the_st1_golden_fixture():
    """Writes the golden. Runs ONLY with `ST1_WRITE_GOLDEN=1`, and only on OLD code.

    A fixture generated by the code it is meant to pin is a self-portrait, so
    this refuses to run once `outcome_kind` exists. The committed fixture was
    written on `main` before any ST1 code was written.
    """
    if os.environ.get("ST1_WRITE_GOLDEN") != "1":
        pytest.skip("set ST1_WRITE_GOLDEN=1 to regenerate (old code only)")
    if "outcome_kind" in legacy.TIER_OUTCOME_COLUMNS:
        pytest.fail(
            "the ST1 golden is pinned from the code BEFORE the fix; regenerating "
            "it after `outcome_kind` landed would be a self-portrait"
        )
    history = build_golden_history()
    observations = legacy.build_scan_factor_observation_rows(history)
    outcomes = legacy.build_bot_tier_outcome_rows(history, observations)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(_serialize_v1(outcomes), encoding="utf-8")
    print(f"wrote {GOLDEN_PATH} with {len(outcomes)} rows")


#: Recorded from `main` @ 84ee24d6 on the golden fixture, BEFORE any ST1 code
#: existed. Filled in by the generation run; see the commit message.
GOLDEN_EXPECTED_RANKS = ("avwap_breakout", "favorite_zone_watch")
GOLDEN_EXPECTED_BOUNDS = {
    "avwap_breakout": 0.562496495355466,
    "favorite_zone_watch": 0.3664451431682858,
}
GOLDEN_EXPECTED_N = {"avwap_breakout": 50, "favorite_zone_watch": 50}
GOLDEN_EXPECTED_DIGEST_RECORDS = {
    "avwap_breakout": {"wins": 35, "losses": 15},
    "favorite_zone_watch": {"wins": 25, "losses": 25},
}
#: 429 outcome rows over horizons 1/3/5 (horizon 10 never lands inside a
#: ten-session history), 101 of them at horizon 5, exactly one flagged stale -
#: GAPPY's, which spans 31 business days under a declared horizon of 5.
GOLDEN_ROW_COUNT = 429


def test_the_v1_tier_outcomes_and_their_ranks_are_byte_identical_to_the_golden(
    tmp_path, monkeypatch
):
    """The v1 export and both readers are unchanged; the new column is additive.

    Three separate claims, checked separately:

    * every ORIGINAL column, in the original order, reproduces byte-identical
      from the same synthetic history (this passes on `main` and must keep
      passing after the fix - it is the "prove production scoring unchanged"
      half of the trader's instruction);
    * `outcome_kind` is present on every row with the v1 value (RED on `main`);
    * `setup_docs.family_headline_rows` and `autopilot_core.swing_family_records`
      give the same ranks and the same Wilson bounds on the golden file, and the
      headline declares its outcome kind (the bound half passes on `main`, the
      outcome-kind half is RED).
    """
    import autopilot_core
    import setup_docs
    import swing_headline

    assert GOLDEN_PATH.exists(), f"missing golden fixture {GOLDEN_PATH}"
    golden_text = GOLDEN_PATH.read_text(encoding="utf-8")

    history = build_golden_history()
    observations = legacy.build_scan_factor_observation_rows(history)
    outcomes = legacy.build_bot_tier_outcome_rows(history, observations)
    assert len(outcomes) == GOLDEN_ROW_COUNT

    # --- claim 1: the v1 columns are byte-identical.
    assert _serialize_v1(outcomes) == golden_text

    # --- claim 2: the new column is there, additive, at the END of the header.
    assert list(legacy.TIER_OUTCOME_COLUMNS)[: len(GOLDEN_V1_COLUMNS)] == list(GOLDEN_V1_COLUMNS)
    assert legacy.TIER_OUTCOME_COLUMNS[-1] == "outcome_kind"
    assert {row["outcome_kind"] for row in outcomes} == {OUTCOME_KIND_SCANROW_V1_TEXT}
    assert {row["outcome_kind"] for row in observations} == {OUTCOME_KIND_SCANROW_V1_TEXT}

    # --- claim 3: both readers give the same ranks and bounds off the golden.
    window = ("2026-06-01", "2026-06-30")
    setup_docs.clear_family_outcome_cache()
    monkeypatch.setattr(setup_docs, "_family_outcomes_path", lambda: GOLDEN_PATH)
    monkeypatch.setattr(setup_docs, "_family_outcomes_window", lambda: window)
    docs_rows = setup_docs.family_headline_rows()
    setup_docs.clear_family_outcome_cache()

    for family, expected_n in GOLDEN_EXPECTED_N.items():
        assert docs_rows[family]["n"] == expected_n
        assert docs_rows[family]["win_rate_lb"] == pytest.approx(GOLDEN_EXPECTED_BOUNDS[family])
    ranked = sorted(
        GOLDEN_EXPECTED_N, key=lambda name: -docs_rows[name]["win_rate_lb"]
    )
    assert tuple(ranked) == GOLDEN_EXPECTED_RANKS

    records = autopilot_core.swing_family_records(GOLDEN_PATH, window=window)
    for family, expected in GOLDEN_EXPECTED_DIGEST_RECORDS.items():
        assert records[family]["wins"] == expected["wins"]
        assert records[family]["losses"] == expected["losses"]
    assert (
        autopilot_core.swing_pick_rank({"family": "avwap_breakout"}, records)
        < autopilot_core.swing_pick_rank({"family": "favorite_zone_watch"}, records)
    )

    # The rate is the same number; it is no longer called a win rate.
    grouped = setup_docs._all_family_outcomes()
    headline = swing_headline.headline_from_tracker_rows(
        "avwap_breakout", grouped["avwap_breakout"]
    )
    setup_docs.clear_family_outcome_cache()
    assert headline.win_rate_lb == pytest.approx(GOLDEN_EXPECTED_BOUNDS["avwap_breakout"])
    assert headline.outcome_kind == "favorable_direction"
    assert "favorable" in headline.sentence().lower()
