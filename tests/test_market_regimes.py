"""S17 half 1: the deterministic multi-timeframe regime table.

The champion Auto Market Bias env_key on M5, M30, H1, H4, D1 and W per session
and symbol, plus the S16 structure facts; append-only and point in time.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from research_warehouse import exchange_calendar as xcal  # noqa: E402

DAY = date(2026, 9, 24)  # a Thursday session
ENV_KEYS = {"bullish_strong", "bullish_weak", "neutral_chop", "bearish_weak", "bearish_strong", "unknown"}


def _sessions(start: date, end: date) -> list[date]:
    return [session.session_date for session in xcal.sessions_between(start, end)]


def _d1(symbol: str, *, step: float = 1.0, end: date = DAY, count: int = 200) -> list[dict]:
    days = [day for day in _sessions(date(2025, 1, 1), end)][-count:]
    rows = []
    for index, day in enumerate(days):
        base = 300.0 + index * step + (3.0 if index % 7 == 0 else 0.0)
        rows.append(
            {
                "symbol": symbol, "session_date": day, "open": base, "high": base + abs(step) + 1.0,
                "low": base - 1.0, "close": base + step, "volume": 1_000_000,
            }
        )
    return rows


def _m5(symbol: str, days: list[date], *, step: float, base: float = 500.0) -> list[dict]:
    rows = []
    price = base
    for day in days:
        session = xcal.trading_session(day)
        for index in range(78):
            start = session.rth_open_at + timedelta(minutes=5 * index)
            close = price + step
            rows.append(
                {
                    "symbol": symbol, "interval_start": start, "interval_end": start + timedelta(minutes=5),
                    "open": price, "high": max(price, close) + 0.05, "low": min(price, close) - 0.05,
                    "close": close, "volume": 1000, "is_complete": True,
                }
            )
            price = close
    return rows


def _history_days(count: int = 16) -> list[date]:
    return _sessions(date(2026, 8, 1), DAY)[-count:]


# ---------------------------------------------------------------- reads --


def test_the_d1_read_is_the_one_the_journal_uses():
    import journal_regime_fill
    import market_regimes

    assert journal_regime_fill._d1_read is market_regimes.d1_env_key
    assert journal_regime_fill._completed_before is market_regimes.completed_before
    assert journal_regime_fill._session_day is market_regimes.session_day


def test_a_rising_tape_is_bullish_on_d1_and_w_and_a_falling_one_bearish():
    import market_regimes as mr

    up = mr.completed_before(_d1("SPY", step=1.0), DAY)
    down = mr.completed_before(_d1("SPY", step=-1.0), DAY)
    assert mr.d1_env_key(up, mr.D1_WINDOW).startswith("bullish")
    assert mr.weekly_env_key(up, DAY).startswith("bullish")
    assert mr.d1_env_key(down, mr.D1_WINDOW).startswith("bearish")
    assert mr.weekly_env_key(down, DAY).startswith("bearish")


def test_weekly_needs_twenty_one_finished_weeks():
    import market_regimes as mr

    thin = mr.completed_before(_d1("SPY", count=90), DAY)
    assert mr.weekly_env_key(thin, DAY) == "unknown"


def test_intraday_reads_use_only_bars_completed_by_each_snapshot():
    import market_regimes as mr

    history = _history_days()
    rising = _m5("SPY", history[:-1], step=0.05)
    today = _m5("SPY", [DAY], step=0.08, base=rising[-1]["close"])
    # A crash after 11:00 ET: the 10:00 snapshot must not see it.
    crash_from = datetime(2026, 9, 24, 11, 0, tzinfo=xcal.EXCHANGE_TZ)
    today = [
        {**row, **{key: row[key] - 40.0 for key in ("open", "high", "low", "close")}}
        if row["interval_start"] >= crash_from else row
        for row in today
    ]
    d1 = _d1("SPY")
    stamp = datetime(2026, 9, 25, 1, 0, tzinfo=xcal.EXCHANGE_TZ)
    row = mr.session_row("SPY", DAY, d1_rows=d1, m5_rows=rising + today, computed_at=stamp)
    calm = mr.session_row("SPY", DAY, d1_rows=d1, m5_rows=rising + _m5("SPY", [DAY], step=0.08, base=rising[-1]["close"]), computed_at=stamp)
    assert row["snapshots"]["10:00"] == calm["snapshots"]["10:00"]
    assert row["snapshots"]["10:00"]["M5"].startswith("bullish")
    assert row["snapshots"]["close"]["M5"] != calm["snapshots"]["close"]["M5"]
    assert row["snapshots"]["10:00"]["as_of"] == "2026-09-24T10:00:00-04:00"
    assert row["snapshots"]["close"]["as_of"] == "2026-09-24T16:00:00-04:00"
    assert row["timeframes"]["M5"] == row["snapshots"]["close"]["M5"]
    # A later bar changes nothing already stamped: the same row from a tape cut at the close.
    later = today + _m5("SPY", [date(2026, 9, 25)], step=-1.0, base=100.0)
    again = mr.session_row("SPY", DAY, d1_rows=d1, m5_rows=rising + later, computed_at=datetime(2026, 9, 25, 1, 0, tzinfo=xcal.EXCHANGE_TZ))
    assert again == row


def test_a_row_has_six_timeframes_three_snapshots_and_the_structure_facts():
    import market_regimes as mr

    history = _history_days()
    row = mr.session_row(
        "SPY", DAY, d1_rows=_d1("SPY"), m5_rows=_m5("SPY", history, step=0.02),
        computed_at=datetime(2026, 9, 25, 1, 0, tzinfo=xcal.EXCHANGE_TZ),
    )
    assert row["rule"] == mr.RULE
    assert (row["session_date"], row["symbol"]) == ("2026-09-24", "SPY")
    assert tuple(row["timeframes"]) == mr.TIMEFRAMES == ("M5", "M30", "H1", "H4", "D1", "W")
    assert set(row["timeframes"].values()) <= ENV_KEYS
    assert "unknown" not in row["timeframes"].values()
    assert tuple(row["snapshots"]) == ("10:00", "12:00", "close")
    assert set(row["structure"]) == {"weekly", "daily_channel", "atr", "sma20"}
    assert row["computed_at"].endswith("-04:00")
    json.dumps(row)  # the row is plain JSON


def test_missing_bars_are_unknown_never_a_guess():
    import market_regimes as mr

    row = mr.session_row("QQQ", DAY, d1_rows=[], m5_rows=[], computed_at=datetime(2026, 9, 25, tzinfo=xcal.EXCHANGE_TZ))
    assert set(row["timeframes"].values()) == {"unknown"}
    assert row["structure"]["sma20"]["status"] == "unknown"


def test_the_table_covers_the_indexes_and_the_desk_sector_etfs():
    import group_rrs
    import market_regimes as mr

    symbols = mr.table_symbols()
    assert symbols[:3] == ("SPY", "QQQ", "IWM")
    assert set(group_rrs.SECTOR_ETFS.values()) <= set(symbols)


# ---------------------------------------------------------------- table --


def test_the_table_is_append_only_and_never_relabels(tmp_path):
    import market_regimes as mr

    path = tmp_path / "market_regime_table.jsonl"
    first = {"session_date": "2026-09-24", "symbol": "SPY", "timeframes": {"D1": "bullish_weak"}}
    assert mr.append_rows(path, [first]) == 1
    relabel = {**first, "timeframes": {"D1": "bearish_strong"}}
    other = {"session_date": "2026-09-24", "symbol": "QQQ", "timeframes": {"D1": "neutral_chop"}}
    assert mr.append_rows(path, [relabel, other, other]) == 1
    rows = mr.read_table(path)
    assert [(row["symbol"], row["timeframes"]["D1"]) for row in rows] == [("SPY", "bullish_weak"), ("QQQ", "neutral_chop")]


def test_a_torn_last_line_does_not_swallow_the_next_row(tmp_path):
    import market_regimes as mr

    path = tmp_path / "market_regime_table.jsonl"
    path.write_text('{"session_date": "2026-09-23", "symbol": "SPY"}\n{"session_da', encoding="utf-8")
    assert mr.append_rows(path, [{"session_date": "2026-09-24", "symbol": "SPY"}]) == 1
    assert [row["session_date"] for row in mr.read_table(path)] == ["2026-09-23", "2026-09-24"]


# ------------------------------------------------------------ night job --


def _loader(sessions_in_d1_end: date = DAY):
    history = _history_days()

    def load(d1_symbols, m5_symbols, **_kwargs):
        d1 = {symbol: _d1(symbol, end=sessions_in_d1_end, count=220) for symbol in d1_symbols}
        m5 = {symbol: _m5(symbol, history, step=0.02) for symbol in m5_symbols if symbol == "SPY"}
        return d1, m5, "fixture"

    return load


def test_the_night_job_appends_the_session_for_every_symbol(tmp_path):
    from ai_jobs import market_regime_table as job

    import market_regimes as mr

    out = tmp_path / "table.jsonl"
    result = job.run_market_regime_table(
        session_date=DAY.isoformat(), out_path=out, loader=_loader(), sessions=4, symbols=("SPY", "QQQ", "IWM"),
        day_review_root=tmp_path / "no_day_review",
    )
    assert result["status"] == "ok", result
    rows = mr.read_table(out)
    # QQQ/IWM have no M5 anywhere: their two freshest sessions are held (advisory 2, 2026-09-26).
    assert {(row["session_date"], row["symbol"]) for row in rows} == {
        (day, "SPY") for day in ("2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24")
    } | {(day, symbol) for day in ("2026-09-21", "2026-09-22") for symbol in ("QQQ", "IWM")}
    qqq = next(row for row in rows if row["symbol"] == "QQQ" and row["session_date"] == "2026-09-22")
    assert qqq["timeframes"]["M5"] == "unknown"  # no QQQ M5 in the lake
    assert qqq["timeframes"]["D1"] != "unknown"
    assert "holding 4 rows" in result["reason"]
    again = job.run_market_regime_table(
        session_date=DAY.isoformat(), out_path=out, loader=_loader(), sessions=2, symbols=("SPY",),
    )
    assert again["status"] == "ok" and "nothing new" in again["reason"]
    assert len(mr.read_table(out)) == 8


def test_a_non_spy_row_waits_for_its_m5_source_then_settles_unknown(tmp_path):
    from ai_jobs import market_regime_table as job

    import market_regimes as mr

    out = tmp_path / "table.jsonl"
    kwargs = dict(out_path=out, loader=_loader(), sessions=1, symbols=("SPY", "QQQ"),
                  day_review_root=tmp_path / "no_day_review")
    job.run_market_regime_table(session_date=DAY.isoformat(), **kwargs)
    assert [(row["session_date"], row["symbol"]) for row in mr.read_table(out)] == [(DAY.isoformat(), "SPY")]

    # Its source arrives: the held row is written with a real M5 read.
    def with_qqq(d1_symbols, m5_symbols, **_kw):
        d1, m5, source = _loader()(d1_symbols, m5_symbols)
        m5["QQQ"] = _m5("QQQ", _history_days(), step=0.02)
        return d1, m5, source

    job.run_market_regime_table(session_date=DAY.isoformat(), **{**kwargs, "loader": with_qqq})
    qqq = next(row for row in mr.read_table(out) if row["symbol"] == "QQQ")
    assert qqq["timeframes"]["M5"] != "unknown" and qqq["m5_source"] == "lake"

    # Never arrives: two sessions later the row is written unknown.
    other = tmp_path / "other.jsonl"
    later = _sessions(DAY, date(2026, 10, 9))[2]
    job.run_market_regime_table(
        session_date=later.isoformat(), out_path=other, loader=_loader(sessions_in_d1_end=later), sessions=3,
        symbols=("SPY", "QQQ"), day_review_root=tmp_path / "no_day_review",
    )
    held = {(row["session_date"], row["symbol"]) for row in mr.read_table(other)}
    assert (DAY.isoformat(), "QQQ") in held
    assert all((day.isoformat(), "QQQ") not in held for day in _sessions(DAY, later)[1:])


def test_a_session_not_yet_in_the_d1_store_waits(tmp_path):
    from ai_jobs import market_regime_table as job

    import market_regimes as mr

    out = tmp_path / "table.jsonl"
    result = job.run_market_regime_table(
        session_date=DAY.isoformat(), out_path=out, loader=_loader(sessions_in_d1_end=date(2026, 9, 23)),
        sessions=2, symbols=("SPY",),
    )
    assert result["status"] == "ok"
    assert [row["session_date"] for row in mr.read_table(out)] == ["2026-09-23"]


def test_a_failure_keeps_the_file(tmp_path):
    from ai_jobs import market_regime_table as job

    out = tmp_path / "table.jsonl"
    out.write_text('{"session_date": "2026-09-22", "symbol": "SPY"}\n', encoding="utf-8")
    before = out.read_bytes()

    def broken(*_args, **_kwargs):
        raise OSError("lake offline")

    result = job.run_market_regime_table(session_date=DAY.isoformat(), out_path=out, loader=broken, sessions=2)
    assert result["status"] == "failed" and "last file kept" in result["reason"]
    assert out.read_bytes() == before


def test_the_slot_is_deterministic_market_read_after_exit_windows():
    from ai_jobs import runner

    slots = runner.default_slots()
    names = [slot.name for slot in slots]
    slot = slots[names.index("market_regime_table")]
    assert slot.goal == "market_read" and slot.uses_model is False
    assert names[names.index("market_regime_table") - 1] == "exit_windows"


def test_the_table_lives_at_a_project_paths_constant():
    import project_paths as pp

    assert Path(pp.MARKET_REGIME_TABLE_FILE).name == "market_regime_table.jsonl"
