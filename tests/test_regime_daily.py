"""P10: the point-in-time daily regime dataset ``market_regime_daily``.

Pinned here: every label on a row uses only bars completed by that session's close
(the same row from a history cut at that close); the D1/W env keys are the champion
reads of ``market_regimes``; fixed buckets; auto_structural_v1 rules and causal
hysteresis; missing inputs are unknown; the lake write is idempotent and never
rewrites a row; the newest sessions wait for ^VIX; the reader keys by session.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import market_regimes as mr  # noqa: E402
from research_warehouse import exchange_calendar as xcal  # noqa: E402
from research_warehouse import regime_daily as rd  # noqa: E402
from research_warehouse.store import ResearchStore  # noqa: E402

UTC = timezone.utc
NOW = datetime(2019, 10, 1, 22, 0, tzinfo=UTC)  # after the 2019-10-01 close


def _days(start=date(2018, 1, 2), end=date(2019, 10, 1)) -> list[date]:
    return [s.session_date for s in xcal.sessions_between(start, end)]


def _d1(symbol="SPY", seed=7, days=None) -> list[dict]:
    """Up 300 sessions, a sharp drop, then a rebound: deterministic."""
    days = days or _days()
    rng = np.random.default_rng(seed)
    price, rows = 100.0, []
    for i, day in enumerate(days):
        drift = 0.001 if i < 300 else (-0.012 if i < 325 else 0.004)
        move = drift + rng.normal(0, 0.008)
        open_ = price
        price = price * (1 + move)
        rows.append({
            "symbol": symbol, "session_date": day, "open": open_,
            "high": max(open_, price) * 1.004, "low": min(open_, price) * 0.996,
            "close": price, "volume": 1_000_000,
        })
    return rows


def _vix(days=None, level=18.0) -> list[dict]:
    return [{"session_date": d, "open": level, "high": level, "low": level, "close": level} for d in days or _days()]


def _h1(days: list[date], symbol="SPY") -> list[dict]:
    rows, price = [], 100.0
    for day in days:
        session = xcal.trading_session(day)
        for k in range(7):
            start = session.rth_open_at + timedelta(minutes=60 * k)
            price += 0.1
            rows.append({"symbol": symbol, "interval_start": start, "open": price - 0.1,
                         "high": price + 0.05, "low": price - 0.15, "close": price, "volume": 1000})
    return rows


def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "computed_at"}


# ---------------------------------------------------------------- pure labels
def test_buckets_use_fixed_cut_points_and_missing_is_unknown():
    assert rd.bucket(14.99, rd.VIX_CUTS, rd.VIX_TOP) == "vix_lt15"
    assert rd.bucket(15.0, rd.VIX_CUTS, rd.VIX_TOP) == "vix_15_20"
    assert rd.bucket(29.9, rd.VIX_CUTS, rd.VIX_TOP) == "vix_20_30"
    assert rd.bucket(30.0, rd.VIX_CUTS, rd.VIX_TOP) == "vix_30_plus"
    assert rd.bucket(None, rd.VIX_CUTS, rd.VIX_TOP) == "unknown"
    assert rd.bucket(2.9, rd.DD_CUTS, rd.DD_TOP) == "dd_0_3"
    assert rd.bucket(15.0, rd.DD_CUTS, rd.DD_TOP) == "dd_15_plus"
    assert rd.bucket(0.95, rd.RV_CUTS, rd.RV_TOP) == "extreme"
    assert rd.bucket(0.5, rd.RV_CUTS, rd.RV_TOP) == "normal"


def test_trend50_200_rules():
    assert rd.trend50_200(110, 105, 100, 99) == "uptrend"
    assert rd.trend50_200(90, 95, 100, 101) == "downtrend"
    assert rd.trend50_200(104, 105, 100, 99) == "transition"  # below the 50
    assert rd.trend50_200(90, 95, 100, 99) == "transition"  # below a rising 200
    assert rd.trend50_200(None, 95, 100, 99) == "unknown"
    assert rd.composite("uptrend", "low") == "uptrend|low"
    assert rd.composite("uptrend", "unknown") == "unknown"


def test_hysteresis_needs_three_sessions_except_capitulation():
    raw = ["range"] * 4 + ["bull_run"] * 2 + ["range"] + ["bull_run"] * 3 + ["capitulation"] + ["recovery"] * 2
    assert rd.smooth_labels(raw) == (
        ["unknown", "unknown"] + ["range"] * 2 + ["range"] * 2 + ["range"]
        + ["range", "range", "bull_run"] + ["capitulation"] + ["capitulation"] * 2
    )


def test_hysteresis_is_causal():
    raw = ["range"] * 5 + ["bull_run"] * 5 + ["range"] * 2
    full = rd.smooth_labels(raw)
    for cut in range(1, len(raw) + 1):
        assert rd.smooth_labels(raw[:cut]) == full[:cut]


def _weeks(highs, lows):
    start = date(2019, 1, 7)
    return [{"week_start": start + timedelta(weeks=i), "high": h, "low": lo} for i, (h, lo) in enumerate(zip(highs, lows, strict=True))]


def _facts(**over):
    rising = _weeks([100 + i for i in range(30)], [95 + i for i in range(30)])
    base = dict(close=130.0, sma50=125.0, sma50_prior=124.0, sma200=115.0, sma200_prior=112.0,
                drawdown=0.01, max_drawdown_recent=0.02, bounce=0.05, ret5=0.01, weekly=rising)
    base.update(over)
    return rd.StructuralFacts(**base)


def test_structural_rules_cover_the_trader_vocabulary():
    from structural_regime import VOCABULARY

    assert rd.structural_raw(_facts()) == "bull_run"
    assert rd.structural_raw(_facts(drawdown=0.09, ret5=-0.06)) == "capitulation"
    assert rd.structural_raw(
        _facts(close=100.0, sma50=105.0, max_drawdown_recent=0.15, bounce=0.08, drawdown=0.07)
    ) == "recovery"
    falling = _weeks([130 - i for i in range(30)], [125 - i for i in range(30)])
    assert rd.structural_raw(
        _facts(close=98.0, sma50=103.0, sma50_prior=105.0, weekly=falling, drawdown=0.2)
    ) == "bear_channel_lower_highs"
    # A 26-week high 3 weeks back, none since, ranges contracted to a third.
    highs = [100 + i for i in range(27)] + [126.5, 126.4, 126.3]
    lows = [90 + i for i in range(27)] + [125.0, 125.0, 125.0]
    assert rd.structural_raw(_facts(weekly=_weeks(highs, lows))) == "weekly_hh_then_compression"
    flat = _weeks([100 + (i % 2) for i in range(30)], [95 - (i % 2) for i in range(30)])
    assert rd.structural_raw(_facts(close=100.0, sma50=100.5, sma200=100.0, sma200_prior=100.2, weekly=flat)) == "range"
    assert rd.structural_raw(_facts(sma200=None)) == "unknown"
    assert rd.structural_raw(_facts(weekly=_weeks([1] * 10, [0.5] * 10))) == "unknown"
    labels = {"bull_run", "capitulation", "recovery", "bear_channel_lower_highs", "weekly_hh_then_compression", "range"}
    assert labels == set(VOCABULARY)


# ---------------------------------------------------------------- rows
@pytest.fixture(scope="module")
def spy_rows():
    days = _days()
    return rd.build_rows("SPY", _d1(), vix=_vix(), h1=_h1(days[-40:-5]), computed_at=NOW)


def test_rows_start_after_the_warm_up_and_carry_every_axis(spy_rows):
    days = _days()
    assert spy_rows[0]["session_date"] == days[rd.WARMUP_SESSIONS - 1]
    assert [r["session_date"] for r in spy_rows] == days[rd.WARMUP_SESSIONS - 1:]
    for row in spy_rows:
        assert row["rule_version"] == rd.RULE_VERSION
        assert row["structural_rule_version"] == "auto_structural_v1"
        assert row["next_session_date"] > row["session_date"]
        for axis in ("env_d1", "env_w", "trend20", "trend50_200", "vol_rv", "vol_vix", "drawdown"):
            assert row[axis] != "unknown", (axis, row["session_date"])
    assert spy_rows[-1]["next_session_date"] == date(2019, 10, 2)
    assert {row["structural"] for row in spy_rows} - {"unknown"}


def test_a_row_uses_only_bars_completed_by_its_close(spy_rows):
    """The row for day D is identical when every bar after D is removed."""
    days = _days()
    for cut in (rd.WARMUP_SESSIONS + 3, 330, 360, len(days) - 3):
        day = days[cut]
        later = [d for d in days if d > day]
        truncated = rd.build_rows(
            "SPY", [r for r in _d1() if r["session_date"] <= day], vix=_vix(), h1=_h1(days[-40:-5]),
            computed_at=NOW,
        )
        again = _strip(truncated[-1])
        original = _strip(next(r for r in spy_rows if r["session_date"] == day))
        # With no later bar the next session comes from the calendar, which agrees here.
        assert again == original, day
        assert later  # the full run did see later bars


def test_env_keys_are_the_champion_reads(spy_rows):
    bars = _d1()
    for row in spy_rows[::37]:
        nxt = row["next_session_date"]
        assert row["env_d1"] == mr.d1_env_key(mr.completed_before(bars, nxt), mr.D1_WINDOW)
        assert row["env_w"] == mr.weekly_env_key(mr.completed_before(bars, nxt), nxt)


def test_intraday_env_keys_need_bars_on_that_session(spy_rows):
    days = _days()
    by_day = {row["session_date"]: row for row in spy_rows}
    assert by_day[days[-10]]["env_h1"] != "unknown"
    assert by_day[days[-2]]["env_h1"] == "unknown"  # the H1 feed stopped 5 sessions ago
    assert by_day[days[-60]]["env_h1"] == "unknown"  # before the H1 feed
    assert all(row["env_h4"] == "unknown" for row in spy_rows)


def test_missing_vix_is_unknown_never_carried():
    days = _days()
    vix = [row for row in _vix() if row["session_date"] != days[-20]]
    rows = {r["session_date"]: r for r in rd.build_rows("SPY", _d1(), vix=vix, computed_at=NOW)}
    assert rows[days[-20]]["vol_vix"] == "unknown" and rows[days[-20]]["vix_close"] is None
    assert rows[days[-21]]["vol_vix"] == "vix_15_20"


def test_too_little_history_writes_nothing():
    assert rd.build_rows("SPY", _d1()[: rd.WARMUP_SESSIONS - 1], vix=_vix()) == []


# ---------------------------------------------------------------- lake
def _loaders(vix_days=None):
    days = _days()

    def d1_loader(symbols, start, end):
        out = {s: [r for r in _d1(s, seed=len(s)) if (start or date.min) <= r["session_date"] <= (end or date.max)] for s in symbols if s != "^VIX"}
        out["^VIX"] = _vix(vix_days or days)
        return out

    def intraday_loader(timeframe, symbols, start, end):
        return {}

    return d1_loader, intraday_loader


def test_run_build_is_idempotent_and_the_reader_keys_by_session(tmp_path):
    store = ResearchStore(tmp_path / "lake")
    d1_loader, intraday_loader = _loaders()
    dry = rd.run_build(store, apply=False, now=NOW, d1_loader=d1_loader, intraday_loader=intraday_loader)
    assert dry["new_rows"] > 0 and not store.read_rows(rd.DATASET)
    first = rd.run_build(store, apply=True, now=NOW, d1_loader=d1_loader, intraday_loader=intraday_loader)
    expected = len(_days()) - rd.WARMUP_SESSIONS + 1
    assert first["by_symbol"] == {"SPY": expected, "QQQ": expected, "IWM": expected}
    assert first["rows_published"] == 3 * expected
    again = rd.run_build(store, apply=True, now=NOW, d1_loader=d1_loader, intraday_loader=intraday_loader)
    assert again["new_rows"] == 0 and "rows_published" not in again
    frame = rd.read_regimes("SPY", store=store)
    assert len(frame) == expected and list(frame["session_date"]) == sorted(frame["session_date"])
    assert set(frame["rule_version"]) == {rd.RULE_VERSION}
    nxt = rd.read_regimes("SPY", store=store, as_of="next_open")
    assert list(nxt["trade_session_date"]) == list(nxt["next_session_date"])
    assert rd.read_regimes("DIA", store=store).empty


def test_the_newest_sessions_wait_for_vix_and_never_before_the_close(tmp_path):
    store = ResearchStore(tmp_path / "lake")
    days = _days()
    d1_loader, intraday_loader = _loaders(vix_days=days[:-1])
    report = rd.run_build(store, apply=True, now=NOW, symbols=("SPY",), d1_loader=d1_loader,
                          intraday_loader=intraday_loader)
    assert report["held"] == [f"SPY {days[-1].isoformat()}"]
    written = {row["session_date"] for row in store.read_rows(rd.DATASET)}
    assert days[-1] not in written and days[-2] in written
    # Mid-session on 2019-10-01: that day is not settled, so it is not even computed.
    midday = datetime(2019, 10, 1, 17, 0, tzinfo=UTC)
    report = rd.run_build(store, apply=False, now=midday, symbols=("SPY",), d1_loader=_loaders()[0],
                          intraday_loader=intraday_loader)
    assert report["last_session"] == days[-2].isoformat() and report["new_rows"] == 0


def test_reader_shaped_frames_are_accepted_and_stub_bars_dropped():
    import pandas as pd

    frame = pd.DataFrame(_d1()[:5])
    assert [b["session_date"] for b in rd.daily_bars(frame)] == _days()[:5]
    h1 = pd.DataFrame(_h1(_days()[:1]))
    h1["interval_start"] = pd.to_datetime(h1["interval_start"], utc=True).dt.tz_convert("America/New_York")
    h1["is_stub"] = [False] * 6 + [True]
    bars = rd.intraday_bars(h1)
    assert len(bars) == 6 and all(b["session_date"] == _days()[0] for b in bars)


# ---------------------------------------------------------------- report
def test_forward_facts_and_segments():
    from research_warehouse import regime_report as rr

    days = _days()[:30]
    d1 = [{"session_date": d, "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100 + i}
          for i, d in enumerate(days)]
    fwd = rr.forward_facts(d1)
    assert fwd[days[0]]["fwd_1"] == pytest.approx(0.01)
    assert fwd[days[0]]["fwd_20"] == pytest.approx(0.20)
    assert fwd[days[0]]["fwd_mdd"] == 0.0  # every later low is above the close
    assert fwd[days[-1]]["fwd_1"] is None and fwd[days[10]]["fwd_mdd"] is None
    pairs = [(days[i], lab) for i, lab in enumerate(["a", "a", "b", "a", "a", "a"])]
    out = rr.axis_report(pairs, fwd)
    assert [s["sessions"] for s in out["timeline"]] == [2, 1, 3]
    assert out["transitions"] == {"a": {"b": 1}, "b": {"a": 1}}
    assert out["labels"]["a"]["segments"] == 2 and out["labels"]["a"]["segment_sessions_max"] == 3


def test_trader_agreement_counts_known_sessions_only():
    from research_warehouse import regime_report as rr

    days = _days()[:6]
    typed = [{"segment_id": 1, "start_date": days[2].isoformat(), "regime": "bull_run"}]
    pairs = list(zip(days, ["range", "range", "bull_run", "unknown", "range", "bull_run"], strict=True))
    out = rr.trader_agreement(pairs, typed)
    assert out["sessions_compared"] == 3 and out["sessions_agree"] == 2
    assert out["confusion_trader_by_auto"] == {"bull_run": {"bull_run": 2, "unknown": 1, "range": 1}}
    assert rr.trader_agreement(pairs, [])["status"] == "no typed segments"


def test_trader_segments_are_read_from_a_copy(tmp_path):
    import sqlite3

    from research_warehouse import regime_report as rr

    db = tmp_path / "journal.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE structural_regime (segment_id INTEGER, start_date TEXT, regime TEXT)")
    conn.execute("INSERT INTO structural_regime VALUES (1, '2026-03-01', 'bull_run')")
    conn.commit()
    conn.close()
    before = db.read_bytes()
    assert rr.read_trader_segments(db) == [{"segment_id": 1, "start_date": "2026-03-01", "regime": "bull_run"}]
    assert db.read_bytes() == before
    assert rr.read_trader_segments(tmp_path / "missing.sqlite3") == []


def test_regime_report_cli_path_writes_every_axis(tmp_path):
    import json

    from research_warehouse import cli

    store = ResearchStore(tmp_path / "lake")
    d1_loader, intraday_loader = _loaders()
    cli.run_build_regimes(store, apply=True, now=NOW, lock_path=tmp_path / "lock",
                          d1_loader=d1_loader, intraday_loader=intraday_loader)
    out = tmp_path / "report.json"
    result = cli.run_regime_report(store, symbol="SPY", out=out, journal=tmp_path / "none.sqlite3",
                                   d1_loader=d1_loader)
    assert result["status"] == "OK"
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["axes"]) == set(rd.AXES)
    assert report["span"]["sessions"] == len(_days()) - rd.WARMUP_SESSIONS + 1
    assert report["trader_agreement"]["status"] == "no typed segments"
    assert report["baseline"]["fwd_1"]["n"] == report["span"]["sessions"] - 1
