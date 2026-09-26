"""S17 follow-up: QQQ/IWM M5 regimes from the cached Day Review Yahoo store.

The lake has no QQQ/IWM M5. The night table and the desk strip fill a
(symbol, session) the lake lacks from `day_review_bars`' closed-session files;
the lake wins when both exist, a missing file stays `unknown`, nothing is fetched.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from research_warehouse import exchange_calendar as xcal  # noqa: E402
from tests.test_market_regimes import DAY, _d1, _history_days, _m5  # noqa: E402

INTRADAY = ("M5", "M30", "H1", "H4")


def _write_day_review(tmp_path, monkeypatch, symbol_steps: dict[str, float], days: list[date]) -> Path:
    """Write Day Review files with the production writer, one per session."""
    import day_review_bars as dbr

    monkeypatch.setattr(dbr, "DAY_REVIEW_DIR", tmp_path)
    tapes = {symbol: _m5(symbol, days, step=step) for symbol, step in symbol_steps.items()}
    for day in days:
        bars = {
            symbol: [
                {"dt": row["interval_start"].astimezone(dbr.MARKET_ZONE), **{k: row[k] for k in ("open", "high", "low", "close", "volume")}}
                for row in tape
                if row["interval_start"].astimezone(xcal.EXCHANGE_TZ).date() == day
            ]
            for symbol, tape in tapes.items()
        }
        dbr.write_session_bars(day.isoformat(), bars)
    return tmp_path


def _loader(d1_symbols, m5_symbols, **_kwargs):
    d1 = {symbol: _d1(symbol, count=220) for symbol in d1_symbols}
    m5 = {symbol: _m5(symbol, _history_days(), step=0.02) for symbol in m5_symbols if symbol == "SPY"}
    return d1, m5, "fixture"


#: Two sessions before DAY: old enough that a row with no M5 source is written unknown.
SETTLED = _history_days()[-3]


def _run(tmp_path, root, *, sessions=1, session=DAY):
    from ai_jobs import market_regime_table as job

    import market_regimes as mr

    out = tmp_path / "table.jsonl"
    result = job.run_market_regime_table(
        session_date=DAY.isoformat(), out_path=out, loader=_loader, sessions=sessions,
        symbols=("SPY", "QQQ", "IWM"), day_review_root=root,
    )
    assert result["status"] == "ok", result
    return {row["symbol"]: row for row in mr.read_table(out) if row["session_date"] == session.isoformat()}


def test_qqq_m5_is_filled_from_the_day_review_store(tmp_path, monkeypatch):
    root = _write_day_review(tmp_path / "dr", monkeypatch, {"QQQ": 0.02}, _history_days())
    rows = _run(tmp_path, root)
    qqq = rows["QQQ"]
    assert qqq["m5_source"] == "day_review_yahoo"
    assert "unknown" not in {qqq["timeframes"][tf] for tf in INTRADAY}, qqq["timeframes"]
    assert qqq["session_m5_bars"] == 78
    # Same bars through the same classifier give the same reads as the lake's SPY.
    assert {tf: qqq["timeframes"][tf] for tf in INTRADAY} == {tf: rows["SPY"]["timeframes"][tf] for tf in INTRADAY}


def test_the_lake_wins_when_both_stores_hold_the_session(tmp_path, monkeypatch):
    baseline = _run(tmp_path / "a", None)["SPY"]
    root = _write_day_review(tmp_path / "dr", monkeypatch, {"SPY": -0.08}, _history_days())
    spy = _run(tmp_path / "b", root)["SPY"]
    assert spy["m5_source"] == "lake"
    assert {k: v for k, v in spy.items() if k != "computed_at"} == {k: v for k, v in baseline.items() if k != "computed_at"}


def test_a_session_in_neither_store_stays_unknown(tmp_path, monkeypatch):
    # Held while fresh (advisory 2, 2026-09-26); written unknown once two sessions old.
    root = _write_day_review(tmp_path / "dr", monkeypatch, {"QQQ": 0.02}, _history_days())
    assert "IWM" not in _run(tmp_path / "fresh", root)
    iwm = _run(tmp_path / "settled", root, sessions=3, session=SETTLED)["IWM"]
    assert iwm["m5_source"] == "none"
    assert {iwm["timeframes"][tf] for tf in INTRADAY} == {"unknown"}
    assert iwm["session_m5_bars"] == 0


def test_every_row_names_its_m5_source(tmp_path):
    rows = _run(tmp_path, tmp_path / "empty", sessions=3, session=SETTLED)
    assert {symbol: row["m5_source"] for symbol, row in rows.items()} == {"SPY": "lake", "QQQ": "none", "IWM": "none"}


def test_a_bar_not_completed_by_now_is_not_used(tmp_path, monkeypatch):
    import market_regimes as mr

    root = _write_day_review(tmp_path / "dr", monkeypatch, {"QQQ": 0.02}, [DAY])
    close = datetime.combine(DAY, datetime.min.time(), tzinfo=mr.MARKET_TZ).replace(hour=16)
    m5: dict = {}
    sources = mr.fill_m5_from_day_review(m5, ("QQQ",), [DAY], root=root, now=close - timedelta(minutes=2))
    assert sources == {(DAY.isoformat(), "QQQ"): "day_review_yahoo"}
    assert len(m5["QQQ"]) == 77  # the 15:55 bar ends after `now`
    assert all(row["interval_start"].tzinfo is not None for row in m5["QQQ"])


def test_the_desk_strip_reads_qqq_from_the_day_review_store(tmp_path, monkeypatch):
    from tests.test_regime_strip import ET, _service

    root = _write_day_review(tmp_path / "dr", monkeypatch, {"QQQ": 0.02}, _history_days())
    service = _service()
    seen: dict = {}
    service._current_bot = lambda: None
    service._regime_strip_loader = _loader
    service._regime_strip_day_review_root = root
    service._emit = lambda signal, *args: seen.setdefault("payload", args[0]) if signal is service._regimeStripReady else None
    service._load_regime_strip_worker(datetime(2026, 9, 24, 12, 0, tzinfo=ET))
    symbols = seen["payload"]["symbols"]
    assert "unknown" not in {symbols["QQQ"][tf] for tf in INTRADAY}, symbols["QQQ"]
    assert {symbols["IWM"][tf] for tf in INTRADAY} == {"unknown"}
