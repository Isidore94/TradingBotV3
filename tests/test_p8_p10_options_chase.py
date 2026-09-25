"""P10 options chase: the pure picker, the service with a fake IB, the Opt column, the log."""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import options_chase as oc  # noqa: E402

NY = ZoneInfo("America/New_York")
# Monday 2026-09-28; weeklies Fri 10/02 (4 sessions) and Fri 10/09.
TODAY = date(2026, 9, 28)
NOW = datetime(2026, 9, 28, 10, 40, 20, tzinfo=NY)


def _pop(**values):
    row = {"symbol": "ABC", "side": "long", "last": 25.0, "atr": 0.20, "rvol": 3.0, "move15": 2.4}
    row.update(values)
    return row


def _q(expiry, strike, right, bid, ask, delta, iv=0.62):
    return {"expiry": expiry, "strike": strike, "right": right, "bid": bid, "ask": ask,
            "delta": delta, "iv": iv}


def _chain():
    quotes = [
        _q("2026-10-02", 26.0, "C", 1.40, 1.50, 0.38),
        _q("2026-10-02", 27.0, "C", 0.85, 0.95, 0.26),
        _q("2026-10-02", 28.0, "C", 0.40, 0.46, 0.14),
        _q("2026-10-02", 24.0, "C", 1.9, 2.0, 0.60),  # ITM call: never picked
        _q("2026-10-02", 23.0, "P", 0.70, 0.78, -0.24),
        _q("2026-10-02", 22.0, "P", 0.30, 0.34, -0.12),
        _q("2026-10-09", 27.0, "C", 1.10, 1.20, 0.25),
    ]
    return {"expiries": ["20260930", "20261002", "20261009"],
            "strikes": [22.0, 23.0, 24.0, 26.0, 27.0, 28.0], "quotes": quotes}


# ---------------------------------------------------------------- pure picker
def test_picks_nearest_weekly_otm_call_nearest_quarter_delta():
    result = oc.pick_candidate(_pop(), _chain(), today=TODAY, hv=0.41)
    assert result["status"] == oc.STATUS_CANDIDATE
    assert (result["expiry"], result["strike"], result["right"]) == ("2026-10-02", 27.0, "C")
    assert result["delta"] == pytest.approx(0.26)
    assert result["mid"] == pytest.approx(0.90)
    assert result["spread_pct"] == pytest.approx(0.10 / 0.90 * 100)
    assert result["iv_vs_hv"] == pytest.approx(0.62 / 0.41)
    assert oc.cell_text(result) == "27C 10/02 · 0.85x0.95 · 11% · IV 62 (HV 41)"


def test_short_pop_picks_the_otm_put_with_abs_delta():
    result = oc.pick_candidate(_pop(side="short"), _chain(), today=TODAY, hv=0.41)
    assert (result["status"], result["strike"], result["right"]) == ("candidate", 23.0, "P")
    assert result["delta"] == pytest.approx(0.24)


def test_the_wednesday_daily_is_not_a_weekly():
    assert oc.weekly_expiries(["20260930", "20261002", "20261009"]) == [
        date(2026, 10, 2), date(2026, 10, 9)]


def test_expiry_under_two_sessions_rolls_to_the_next_weekly():
    # Thursday 10/01: Friday 10/02 has one session to go, so 10/09 is the pick.
    result = oc.pick_candidate(_pop(), _chain(), today=date(2026, 10, 1), hv=0.41)
    assert result["expiry"] == "2026-10-09"
    assert result["strike"] == 27.0 and result["status"] == "candidate"
    expiry, sessions, _ = oc.pick_expiry(["20261002"], date(2026, 10, 1))
    assert expiry is None and sessions is None


def test_refuses_rvol_under_two_and_unknown_rvol():
    assert oc.pick_candidate(_pop(rvol=1.8), _chain(), today=TODAY)["reason"] == "RVOL 1.8 < 2"
    unknown = oc.pick_candidate(_pop(rvol=None), _chain(), today=TODAY)
    assert (unknown["status"], unknown["reason"]) == ("refused", "RVOL unknown")


def test_refuses_empty_chain():
    for chain in (None, {}, {"expiries": ["20261002"], "quotes": []}):
        result = oc.pick_candidate(_pop(), chain, today=TODAY)
        assert (result["status"], result["reason"]) == ("refused", "empty chain")


def test_refuses_when_no_delta_in_range():
    chain = _chain()
    chain["quotes"] = [_q("2026-10-02", 26.0, "C", 1.4, 1.5, 0.40),
                       _q("2026-10-02", 28.0, "C", 0.4, 0.46, 0.10)]
    result = oc.pick_candidate(_pop(), chain, today=TODAY)
    assert (result["status"], result["reason"]) == ("refused", "no delta in 0.15-0.35")
    chain["quotes"] = [_q("2026-10-02", 27.0, "C", 0.85, 0.95, None)]
    assert oc.pick_candidate(_pop(), chain, today=TODAY)["reason"] == "no delta quoted"


def test_refuses_a_wide_spread_but_still_names_the_contract():
    chain = _chain()
    chain["quotes"][1] = _q("2026-10-02", 27.0, "C", 0.80, 1.00, 0.26)
    result = oc.pick_candidate(_pop(), chain, today=TODAY, hv=0.41)
    assert result["status"] == "refused"
    assert result["reason"] == "spread 22% of mid > 15%"
    assert result["strike"] == 27.0 and result["mid"] == pytest.approx(0.90)
    assert oc.cell_text(result) == "no chase (spread 22% of mid > 15%)"
    assert "refused: spread 22% of mid > 15%" in oc.detail_text(result)


def test_missing_quote_parts_are_unknown_never_guessed():
    chain = _chain()
    chain["quotes"][1] = _q("2026-10-02", 27.0, "C", None, 0.95, 0.26, iv=None)
    result = oc.pick_candidate(_pop(), chain, today=TODAY, hv=None)
    assert (result["status"], result["reason"]) == ("refused", "no two-sided quote")
    assert result["mid"] is None and result["iv"] is None and result["iv_vs_hv"] is None
    assert oc.pick_candidate(_pop(last=None), _chain(), today=TODAY)["reason"] == "last price unknown"
    ok = oc.pick_candidate(_pop(), _chain(), today=TODAY, hv=None)
    assert ok["status"] == "candidate" and ok["iv_vs_hv"] is None
    assert oc.cell_text(ok).endswith("IV 62 (HV —)")


def test_realized_vol_is_annualised_close_to_close_over_20_sessions():
    closes = [100.0 * (1.01 if i % 2 else 0.99) for i in range(21)]
    returns = [math.log(b / a) for a, b in zip(closes[:-1], closes[1:], strict=True)]
    mean = sum(returns) / len(returns)
    expected = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)) * math.sqrt(252)
    assert oc.realized_vol(closes) == pytest.approx(expected)
    assert oc.realized_vol(closes[:20]) is None  # 19 returns: unknown


def test_read_daily_closes_drops_today_and_later(tmp_path):
    path = tmp_path / "ABC.csv"
    path.write_text("datetime,open,high,low,close,volume\n"
                    "2026-09-24,1,1,1,10,1\n2026-09-25,1,1,1,11,1\n2026-09-28,1,1,1,99,1\n",
                    encoding="utf-8")
    assert oc.read_daily_closes(path, before=TODAY) == [10.0, 11.0]
    assert oc.read_daily_closes(tmp_path / "missing.csv", before=TODAY) == []


def test_quote_plan_spreads_otm_strikes_over_the_delta_window():
    strikes = [20 + 0.5 * i for i in range(40)]  # 20.0 .. 39.5
    plan = oc.quote_plan(25.0, "long", strikes, sessions=4, hv=0.60)
    assert plan and all(s > 25.0 for s in plan) and len(plan) <= oc.QUOTE_STRIKES
    puts = oc.quote_plan(25.0, "short", strikes, sessions=4, hv=0.60)
    assert puts and all(s < 25.0 for s in puts)
    assert oc.quote_plan(None, "long", strikes, sessions=4, hv=0.6) == []


# ---------------------------------------------------------------- log rows
def _at(hour, minute):
    return datetime(2026, 9, 28, hour, minute, 20, tzinfo=NY)


def test_tracker_flags_once_per_answer_and_owes_30_60_and_close_rows():
    tracker = oc.ChaseOutcomeTracker()
    candidate = oc.pick_candidate(_pop(), _chain(), today=TODAY, hv=0.41)
    rows = tracker.flag([candidate], now=_at(10, 40))
    assert len(rows) == 1 and rows[0]["kind"] == "flag" and rows[0]["mid"] == pytest.approx(0.90)
    assert tracker.flag([candidate], now=_at(10, 45)) == []  # same answer: no new flag
    assert tracker.observe({"ABC": 25.2}, now=_at(11, 5)) == []
    mids = {"ABC": 1.08}
    thirty = tracker.observe({"ABC": 25.2}, now=_at(11, 10),
                             option_mid=lambda flag: mids.get(flag["symbol"]))
    assert [r["horizon"] for r in thirty] == ["+30m"]
    assert thirty[0]["move_atr"] == pytest.approx(1.0)
    assert thirty[0]["move_atr_chase"] == pytest.approx(1.0)
    assert thirty[0]["option_mid"] == pytest.approx(1.08)
    assert thirty[0]["option_mid_change_pct"] == pytest.approx(20.0)
    sixty = tracker.observe({"ABC": 24.9}, now=_at(11, 40))
    assert [r["horizon"] for r in sixty] == ["+60m"]
    assert sixty[0]["option_mid"] is None and sixty[0]["option_mid_change_pct"] is None
    close = tracker.observe({"ABC": 25.4}, now=_at(16, 0))
    assert [r["horizon"] for r in close] == ["close"]
    assert close[0]["move_atr"] == pytest.approx(2.0)
    assert tracker.observe({"ABC": 25.4}, now=_at(16, 5)) == []


def test_a_short_flag_measures_the_move_in_the_chase_direction_and_refusals_are_flagged():
    tracker = oc.ChaseOutcomeTracker()
    refused = oc.pick_candidate(_pop(side="short", rvol=1.5), _chain(), today=TODAY)
    rows = tracker.flag([refused], now=_at(15, 50))
    assert rows[0]["status"] == "refused" and rows[0]["reason"] == "RVOL 1.5 < 2"
    # The close comes before +30: only the close row is written.
    out = tracker.observe({"ABC": 24.8}, now=_at(16, 0))
    assert [r["horizon"] for r in out] == ["close"]
    assert out[0]["move_atr"] == pytest.approx(-1.0)
    assert out[0]["move_atr_chase"] == pytest.approx(1.0)
    assert out[0]["option_mid"] is None


def test_no_data_rows_are_never_logged():
    tracker = oc.ChaseOutcomeTracker()
    assert tracker.flag([oc.no_data(_pop(), "IB not connected")], now=_at(10, 40)) == []


def test_log_append_summary_and_cli(tmp_path, capsys):
    path = tmp_path / "options_chase_log.jsonl"
    tracker = oc.ChaseOutcomeTracker()
    candidate = oc.pick_candidate(_pop(), _chain(), today=TODAY, hv=0.41)
    refused = oc.pick_candidate(_pop(symbol="XYZ", rvol=1.2), _chain(), today=TODAY)
    assert oc.append_records(path, tracker.flag([candidate, refused], now=_at(10, 40)))
    assert oc.append_records(path, tracker.observe(
        {"ABC": 25.2, "XYZ": 25.0}, now=_at(11, 10), option_mid=lambda _f: 1.08))
    rows = oc.load_records(path)
    assert [r["kind"] for r in rows] == ["flag", "flag", "outcome", "outcome"]
    summary = oc.summarize(rows)
    assert (summary["flags"], summary["candidates"], summary["refused"]) == (2, 1, 1)
    assert summary["horizons"]["+30m"]["outcomes"] == 2
    assert summary["horizons"]["+30m"]["with_option_mid"] == 1
    assert summary["horizons"]["+30m"]["median_option_mid_change_pct"] == pytest.approx(20.0)
    assert oc.main(["--summary", "--path", str(path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["candidates"] == 1


def test_a_failed_log_write_returns_false_and_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert oc.append_records(blocker / "sub" / "log.jsonl", [{"kind": "flag"}]) is False


def test_log_path_is_a_project_paths_constant():
    import project_paths

    assert Path(project_paths.OPTIONS_CHASE_LOG_FILE).name == "options_chase_log.jsonl"
