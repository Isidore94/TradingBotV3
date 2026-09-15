"""WS-TH, builder addition: the overnight slot's BODY and the bar reader.

`tests/test_ws_th_theta_tracker.py` (the tester's 27) pins the slot's POSITION
in the slate and the grader's arithmetic. Neither drives
`ai_jobs.theta_grading.run_theta_pick_grading` itself, and neither reads a real
parquet off disk through `theta_pick_tracker.closes_from_daily_bars` - so the
two pieces that actually run unattended every night had no test at all.

These are COVERAGE additions, not behaviour proofs: the code they exercise did
not exist before this packet, so "prove it fails first" has nothing to fail
against beyond the ImportError the tester's file already records. They exist
because a nightly job nobody drives in a test is a nightly job that fails in
the morning.

Nothing here touches a live store: every path is a `tmp_path`.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import market_calendar  # noqa: E402


SCAN_DATE = date(2026, 6, 1)
#: The 20th exchange session after the scan. Juneteenth (2026-06-19) is not a
#: session, so this is 06-30 and not the 06-29 a weekday walk returns.
SESSION_20 = date(2026, 6, 30)


def _pick_row() -> dict:
    """One recorded pick, in the shape `record_theta_picks` writes."""
    return {
        "symbol": "BBB",
        "scan_date": SCAN_DATE.isoformat(),
        "bar_date": SCAN_DATE.isoformat(),
        "play_type": "sold_put",
        "rank": 1,
        "score": 87,
        "base_score": 35,
        "supports": [
            {"label": "SMA_50", "source": "sma", "level": 97.0, "distance_atr": 1.5, "held": True},
            {"label": "SMA_100", "source": "sma", "level": 95.0, "distance_atr": 2.5, "held": True},
            {"label": "SMA_200", "source": "sma", "level": 94.0, "distance_atr": 3.0, "held": True},
        ],
        "support_combo": "SMA_100+SMA_200+SMA_50",
        "supports_held": 3,
        "strike": 95.0,
        "short_strike": None,
        "long_strike": None,
        "expiry": "2026-07-17",
        "premium": 1.25,
        "atr": 2.0,
        "close": 100.0,
        "first_seen_scan_date": SCAN_DATE.isoformat(),
    }


def _write_store(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_daily_parquet(bars_dir: Path, symbol: str, *, through: date) -> None:
    """A per-symbol parquet in the durable store's own shape."""
    bars_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cursor = SCAN_DATE
    while cursor <= through:
        cursor = date.fromordinal(cursor.toordinal() + 1)
        if not market_calendar.is_session(cursor):
            continue
        rows.append(
            {
                "datetime": pd.Timestamp(cursor),
                "open": 101.0,
                "high": 102.0,
                "low": 100.5,
                "close": 101.0,
                "volume": 1000,
            }
        )
    pd.DataFrame(rows).to_parquet(bars_dir / f"{symbol}.parquet")


# ---------------------------------------------------------------------------
# the bar reader
# ---------------------------------------------------------------------------


def test_the_bar_reader_hands_back_the_low_as_well_as_the_close(tmp_path):
    """MAE and `first_support_broken` are questions the closes cannot answer.

    The grader's `closes_for` contract is `{date: {"close", "low", "high"}}`,
    so this reader is only correct if it carries all three off the parquet.
    """
    from theta_pick_tracker import closes_from_daily_bars

    bars_dir = tmp_path / "daily_bars"
    _write_daily_parquet(bars_dir, "BBB", through=SESSION_20)

    closes_for = closes_from_daily_bars(bars_dir)
    bars = closes_for("BBB")
    assert bars, "the durable parquet was not read"
    day, entry = next(iter(sorted(bars.items())))
    assert isinstance(day, date)
    assert entry["close"] == pytest.approx(101.0)
    assert entry["low"] == pytest.approx(100.5)
    assert entry["high"] == pytest.approx(102.0)
    # A symbol with no file is "we have no bars", never a break.
    assert closes_from_daily_bars(bars_dir)("ZZZ") == {}


def test_the_bar_reader_never_fetches_and_never_raises(tmp_path):
    """An unreadable store is an EMPTY one. An export never makes a network call."""
    from theta_pick_tracker import closes_from_daily_bars

    bars_dir = tmp_path / "daily_bars"
    bars_dir.mkdir()
    (bars_dir / "BBB.parquet").write_text("this is not a parquet", encoding="utf-8")
    assert closes_from_daily_bars(bars_dir)("BBB") == {}


# ---------------------------------------------------------------------------
# the slot's body
# ---------------------------------------------------------------------------


def test_a_store_that_was_never_written_is_skipped_and_not_a_failure(tmp_path):
    """The first night after this packet lands has nothing to grade, and says so."""
    from ai_jobs.theta_grading import run_theta_pick_grading

    result = run_theta_pick_grading(
        picks_path=tmp_path / "theta_picks.jsonl",
        outcomes_path=tmp_path / "out.csv",
        daily_bars_dir=tmp_path / "daily_bars",
        now=datetime(2026, 7, 1, 2, 0),
    )
    assert result["status"] == "skipped"
    assert result["picks"] == 0
    assert "theta_picks.jsonl" in result["reason"]
    assert not (tmp_path / "out.csv").exists(), "a skipped slot wrote an export"


def test_the_slot_grades_the_store_against_the_durable_bars(tmp_path):
    """End to end: store + parquet -> `master_avwap_theta_outcomes.csv`.

    `as_of` is the slot's own `last_completed_session`, so the run is pinned to
    a clock rather than to a date the caller passed: a night that runs at 02:00
    on 2026-07-01 is about 2026-06-30, which IS the 20-session endpoint.
    """
    from ai_jobs.theta_grading import run_theta_pick_grading

    picks = tmp_path / "theta_picks.jsonl"
    outcomes = tmp_path / "master_avwap_theta_outcomes.csv"
    bars_dir = tmp_path / "daily_bars"
    _write_store(picks, [_pick_row()])
    _write_daily_parquet(bars_dir, "BBB", through=SESSION_20)

    result = run_theta_pick_grading(
        session_date="2026-06-30",
        picks_path=picks,
        outcomes_path=outcomes,
        daily_bars_dir=bars_dir,
        # 02:00 ET on 07-01: the last COMPLETED session is 06-30.
        now=datetime(2026, 7, 1, 2, 0, tzinfo=market_calendar.MARKET_TZ),
    )
    assert result["status"] == "ok"
    assert result["as_of"] == SESSION_20.isoformat()
    assert result["picks"] == 1
    assert outcomes.exists()

    with outcomes.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "BBB"
    # Every bar closes at 101.00 against a 95.00 strike, so all three marks held.
    assert row["held_5"] == "True"
    assert row["held_10"] == "True"
    assert row["held_20"] == "True"
    assert row["status"] == "measured"
    # 2026-07-17 has not happened at `as_of`: no verdict, not a loss.
    assert row["held_at_expiry"] == ""
    # The lows never reach SMA_50 at 97.00.
    assert row["first_support_broken"] == "none"
    assert float(row["mae_atr"]) == pytest.approx(0.0)
    assert row["rs_flag"] == "not_measured"
    assert result["measured"] == 1


def test_running_the_slot_twice_produces_the_same_file(tmp_path):
    """Idempotent, because the arithmetic is: a settled endpoint cannot move."""
    from ai_jobs.theta_grading import run_theta_pick_grading

    picks = tmp_path / "theta_picks.jsonl"
    outcomes = tmp_path / "master_avwap_theta_outcomes.csv"
    bars_dir = tmp_path / "daily_bars"
    _write_store(picks, [_pick_row()])
    _write_daily_parquet(bars_dir, "BBB", through=SESSION_20)

    moment = datetime(2026, 7, 1, 2, 0, tzinfo=market_calendar.MARKET_TZ)
    kwargs = dict(
        picks_path=picks, outcomes_path=outcomes, daily_bars_dir=bars_dir, now=moment
    )
    run_theta_pick_grading(**kwargs)
    first = outcomes.read_bytes()
    run_theta_pick_grading(**kwargs)
    assert outcomes.read_bytes() == first
