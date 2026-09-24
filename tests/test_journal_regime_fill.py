"""Auto market environment for the journal's `regimes` table.

Point in time (bars completed before the moment described), missing data is
`unknown`, and a row the trader wrote is never overwritten.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from research_warehouse import exchange_calendar as xcal  # noqa: E402

TRADE_DAY = date(2026, 8, 20)  # a Thursday session


def _sessions_before(day: date, count: int) -> list[date]:
    found: list[date] = []
    cursor = day - timedelta(days=1)
    while len(found) < count:
        if xcal.trading_session(cursor) is not None:
            found.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(found))


def _d1(symbol: str, days: list[date], *, start: float = 500.0, step: float = 1.0) -> list[dict]:
    rows = []
    for index, day in enumerate(days):
        base = start + index * step
        rows.append(
            {
                "symbol": symbol,
                "session_date": day,
                "open": base,
                "high": base + abs(step) + 0.5,
                "low": base - 0.5,
                "close": base + step,
                "volume": 10_000_000,
            }
        )
    return rows


def _m5(day: date, *, base: float, step: float, count: int = 78) -> list[dict]:
    session = xcal.trading_session(day)
    rows = []
    for index in range(count):
        close = base + index * step
        rows.append(
            {
                "symbol": "SPY",
                "interval_start": session.rth_open_at + timedelta(minutes=5 * index),
                "interval_end": session.rth_open_at + timedelta(minutes=5 * (index + 1)),
                "open": close,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1000,
                "is_complete": True,
            }
        )
    return rows


def _bars(step: float = 1.0, count: int = 30):
    days = _sessions_before(TRADE_DAY, count)
    return {symbol: _d1(symbol, days, step=step) for symbol in ("SPY", "QQQ", "IWM")}


def _entry(hour: int, minute: int):
    from datetime import datetime

    return datetime(TRADE_DAY.year, TRADE_DAY.month, TRADE_DAY.day, hour, minute, tzinfo=xcal.EXCHANGE_TZ)


# ------------------------------------------------------------------ reads --


def test_trend_labels_come_from_the_champion_env_keys():
    from journal_regime_fill import trend_label

    assert trend_label("bullish_strong") == "up"
    assert trend_label("bullish_weak") == "up"
    assert trend_label("bearish_weak") == "down"
    assert trend_label("neutral_chop") == "range"
    assert trend_label("unknown") == "unknown"
    assert trend_label(None) == "unknown"


def test_a_rising_tape_reads_up_and_a_falling_tape_reads_down():
    from journal_regime_fill import read_session_regime

    up = read_session_regime(TRADE_DAY.isoformat(), None, d1_by_symbol=_bars(step=1.0))
    down = read_session_regime(TRADE_DAY.isoformat(), None, d1_by_symbol=_bars(step=-1.0))

    assert (up.mid_term_regime, up.short_term_regime) == ("up", "up")
    assert (down.mid_term_regime, down.short_term_regime) == ("down", "down")
    assert "QQQ D1" in up.notes and "IWM D1" in up.notes


def test_the_session_own_d1_bar_is_never_read():
    """A crash printed ON the trade date was not known at the open."""
    from journal_regime_fill import read_session_regime

    bars = _bars(step=1.0)
    with_today = {symbol: rows + [{**rows[-1], "session_date": TRADE_DAY, "close": 1.0}] for symbol, rows in bars.items()}

    before = read_session_regime(TRADE_DAY.isoformat(), None, d1_by_symbol=bars)
    after = read_session_regime(TRADE_DAY.isoformat(), None, d1_by_symbol=with_today)

    assert before == after


def test_too_few_bars_is_unknown_never_a_guess():
    from journal_regime_fill import read_session_regime

    thin = _bars(count=8)
    reading = read_session_regime(TRADE_DAY.isoformat(), None, d1_by_symbol=thin)
    assert reading.mid_term_regime == "unknown"
    assert reading.short_term_regime == "up"

    empty = read_session_regime(TRADE_DAY.isoformat(), None, d1_by_symbol={})
    assert (empty.mid_term_regime, empty.short_term_regime, empty.intraday_regime) == ("unknown",) * 3


def test_intraday_reads_only_bars_completed_by_the_first_entry():
    from journal_regime_fill import read_session_regime

    bars = _bars()
    rising_then_crash = _m5(TRADE_DAY, base=530.0, step=0.3, count=24) + [
        {**row, "close": row["close"] - 60.0, "open": row["open"] - 60.0, "low": row["low"] - 60.0, "high": row["high"] - 60.0}
        for row in _m5(TRADE_DAY, base=537.0, step=-0.5, count=78)[24:]
    ]
    at_eleven = read_session_regime(
        TRADE_DAY.isoformat(), _entry(11, 0), d1_by_symbol=bars, spy_m5=rising_then_crash
    )
    assert at_eleven.intraday_regime == "up"
    assert "first entry 11:00 ET" in at_eleven.notes
    # The same tape read late in the day has seen the crash.
    late = read_session_regime(TRADE_DAY.isoformat(), _entry(15, 30), d1_by_symbol=bars, spy_m5=rising_then_crash)
    assert late.intraday_regime == "down"


def test_a_date_only_entry_or_missing_m5_is_unknown_intraday():
    from journal_regime_fill import read_session_regime

    bars = _bars()
    m5 = _m5(TRADE_DAY, base=530.0, step=0.3)
    midnight = _entry(0, 0)
    assert read_session_regime(TRADE_DAY.isoformat(), midnight, d1_by_symbol=bars, spy_m5=m5).intraday_regime == "unknown"
    assert read_session_regime(TRADE_DAY.isoformat(), _entry(11, 0), d1_by_symbol=bars).intraday_regime == "unknown"


# ------------------------------------------------------------------ store --


def _store(tmp_path):
    from journal_store import JournalStore

    return JournalStore(tmp_path / "journal.sqlite3")


def _trade(store, trade_id, opened_at):
    with store.connection() as conn:
        conn.execute(
            """
            INSERT INTO trades(trade_id, broker, account_number, symbol, direction, status,
                opened_at, closed_at, trade_date, updated_at)
            VALUES(?, 'QUESTRADE', '1', 'AAA', 'LONG', 'CLOSED', ?, ?, ?, ?)
            """,
            (trade_id, opened_at, opened_at, opened_at[:10], opened_at[:10]),
        )


def _loader(bars=None, m5=None):
    return lambda: (bars if bars is not None else _bars(), list(m5 or []), "test")


def test_the_fill_writes_auto_rows_keyed_like_list_trades(tmp_path):
    from journal_regime_fill import fill_regimes

    store = _store(tmp_path)
    _trade(store, "T1", f"{TRADE_DAY.isoformat()}T11:00:00-04:00")
    _trade(store, "T2", f"{TRADE_DAY.isoformat()}T10:00:00-04:00")

    dry = fill_regimes(store, apply=False, loader=_loader(m5=_m5(TRADE_DAY, base=530.0, step=0.3)))
    assert dry["planned"] == 1 and store.list_regime_rows() == []

    summary = fill_regimes(store, apply=True, loader=_loader(m5=_m5(TRADE_DAY, base=530.0, step=0.3)))
    assert summary["written"] == 1
    (row,) = store.list_regime_rows()
    assert row["source"] == "auto"
    assert "first entry 10:00 ET" in row["notes"]  # the day's FIRST entry
    trade = next(item for item in store.list_trades() if item["trade_id"] == "T1")
    assert trade["mid_term_regime"] == "up"
    assert trade["intraday_regime"] == "up"


def test_a_second_run_over_the_same_bars_writes_nothing(tmp_path):
    from journal_regime_fill import fill_regimes

    store = _store(tmp_path)
    _trade(store, "T1", f"{TRADE_DAY.isoformat()}T11:00:00-04:00")
    fill_regimes(store, apply=True, loader=_loader())
    again = fill_regimes(store, apply=True, loader=_loader())
    assert again["written"] == 0 and again["unchanged"] == 1


def test_a_trader_row_is_never_overwritten(tmp_path):
    from journal_regime_fill import fill_regimes

    store = _store(tmp_path)
    _trade(store, "T1", f"{TRADE_DAY.isoformat()}T11:00:00-04:00")
    store.upsert_regime(TRADE_DAY, mid_term_regime="my read", notes="mine")

    summary = fill_regimes(store, apply=True, loader=_loader())

    assert summary["trader_owned"] == 1 and summary["written"] == 0
    (row,) = store.list_regime_rows()
    assert row["mid_term_regime"] == "my read" and row["source"] == ""
    assert store.upsert_auto_regime(
        TRADE_DAY, mid_term_regime="up", short_term_regime="up", intraday_regime="up"
    ) is False


def test_a_trader_edit_of_an_auto_row_takes_it_over(tmp_path):
    from journal_regime_fill import fill_regimes

    store = _store(tmp_path)
    _trade(store, "T1", f"{TRADE_DAY.isoformat()}T11:00:00-04:00")
    fill_regimes(store, apply=True, loader=_loader())
    store.upsert_regime(TRADE_DAY, intraday_regime="range")

    fill_regimes(store, apply=True, loader=_loader(bars=_bars(step=-1.0)))

    (row,) = store.list_regime_rows()
    assert row["source"] == ""
    assert row["intraday_regime"] == "range"
    assert row["mid_term_regime"] == "up"  # the trader kept the auto value they did not edit


def test_no_readable_bars_writes_nothing(tmp_path):
    from journal_regime_fill import fill_regimes

    store = _store(tmp_path)
    _trade(store, "T1", f"{TRADE_DAY.isoformat()}T11:00:00-04:00")
    summary = fill_regimes(store, apply=True, loader=lambda: ({}, [], "none"))
    assert summary["status"] == "no_bars"
    assert store.list_regime_rows() == []


def test_the_nightly_slot_owns_the_fill(tmp_path):
    from ai_jobs.journal_auto_tag import run_journal_auto_tag

    store = _store(tmp_path)
    _trade(store, "T1", f"{TRADE_DAY.isoformat()}T11:00:00-04:00")

    result = run_journal_auto_tag(db_path=tmp_path / "journal.sqlite3", regime_loader=_loader())

    assert result["status"] == "ok", result
    assert "regimes: 1 written" in result["reason"]
    assert store.list_regime_rows()[0]["source"] == "auto"

    empty = run_journal_auto_tag(
        db_path=tmp_path / "journal.sqlite3", regime_loader=lambda: ({}, [], "none")
    )
    assert "no benchmark bars readable" in empty["reason"]


def test_the_cli_is_a_dry_run_by_default(tmp_path, monkeypatch, capsys):
    import journal_regime_fill

    store = _store(tmp_path)
    _trade(store, "T1", f"{TRADE_DAY.isoformat()}T11:00:00-04:00")
    monkeypatch.setattr(journal_regime_fill, "load_benchmark_bars", _loader())

    assert journal_regime_fill.main(["--db", str(tmp_path / "journal.sqlite3")]) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert store.list_regime_rows() == []
