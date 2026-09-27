"""p9: strong Long leaders that dipped under the earnings AVWAP are promoted first.

The trader, 2026-09-27: "Yes do it". The long-lab study (1,997 names, 2025-12-17..2026-09-25,
SPY above a rising 20-day): leader_pullback + strength + close under the earnings AVWAP beat
SPY 58% at 10 sessions (+5.4% avg), strength alone 53% (+2.1%), the rest ~46% (0%). Every
leader_pullback row carries the earnings AVWAP (anchored the session before the latest
earnings reaction day, 2-120 sessions back, `calc_anchored_vwap_bands`' sigma) and a
`setup_tier`; promotion takes strong + under first, then strong, then the rest, RS-first
inside each group. Grading is shadow only.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
for entry in (SCRIPTS_DIR, TESTS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import long_setups as ls  # noqa: E402


def _days(count: int) -> list[str]:
    out, day = [], date(2026, 1, 5)
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def _bars(count=60, gaps=None, *, drift=0.2):
    """Bars rising `drift` a session; ``gaps`` = {index: open jump} (the whole bar moves up)."""
    bars, level = [], 100.0
    for index, day in enumerate(_days(count)):
        level += drift + (gaps or {}).get(index, 0.0)
        bars.append({"date": day, "open": level - 0.3, "high": level + 1.0, "low": level - 1.0,
                     "close": level, "volume": 1_000_000.0 + 10_000.0 * (index % 7)})
    return bars


# --- the AVWAP and its sigma: `calc_anchored_vwap_bands`' formula, never a new one

def test_the_bands_match_the_legacy_champion():
    from master_avwap_lib.legacy import calc_anchored_vwap_bands

    bars = _bars(40, {10: 3.0, 25: -2.0})
    bars[15]["volume"] = 0.0
    bars[16]["volume"] = None
    frame = pd.DataFrame([{**bar, "volume": float("nan") if bar["volume"] is None else bar["volume"]}
                          for bar in bars])
    for anchor in (0, 9, 14, 30):
        vwap, sigma, _bands = calc_anchored_vwap_bands(frame, anchor)
        got = ls.avwap_bands(bars, anchor)
        assert got == pytest.approx((vwap, sigma), rel=1e-12)


def test_no_volume_is_no_bands():
    bars = _bars(10)
    for bar in bars:
        bar["volume"] = 0.0
    assert ls.avwap_bands(bars, 2) is None
    assert ls.avwap_bands(bars, 99) is None


# --- the earnings AVWAP: the study's anchor

def test_the_reaction_day_is_the_bigger_gap_of_the_date_and_the_next_session():
    bars = _bars(60, {30: 0.5, 31: 4.0})
    day = bars[30]["date"]
    assert ls.earnings_reaction_index(bars, day) == 31  # after the close: the next session gaps
    bars = _bars(60, {30: 4.0, 31: 0.5})
    assert ls.earnings_reaction_index(bars, day) == 30  # before the open: the day itself gaps
    # A weekend date falls on the next session.
    assert ls.earnings_reaction_index(bars, "2026-01-10") == 5


def test_the_earnings_avwap_is_anchored_the_session_before_the_reaction():
    from master_avwap_lib.legacy import calc_anchored_vwap_bands

    bars = _bars(60, {31: 4.0})
    got = ls.earnings_avwap(bars, earnings_dates=[bars[30]["date"]])
    vwap, sigma, _bands = calc_anchored_vwap_bands(pd.DataFrame(bars), 30)
    close = bars[-1]["close"]
    assert got["avwape"] == pytest.approx(vwap, abs=1e-4)
    assert got["avwape_z"] == pytest.approx((close - vwap) / sigma, abs=1e-4)
    assert got["under_avwape"] == "no"  # a rising tape sits over its AVWAP


def test_a_close_under_the_earnings_avwap_is_under():
    bars = _bars(60, {31: 4.0, 55: -12.0})
    got = ls.earnings_avwap(bars, earnings_dates=[bars[30]["date"]])
    assert got["under_avwape"] == "yes" and got["avwape_z"] < 0
    assert bars[-1]["close"] < got["avwape"]


def test_the_latest_past_reaction_wins_and_the_future_is_never_read():
    bars = _bars(60, {10: 4.0, 40: 4.0})
    both = ls.earnings_avwap(bars, earnings_dates=[bars[9]["date"], bars[39]["date"], "2026-12-01"])
    latest = ls.earnings_avwap(bars, earnings_dates=[bars[39]["date"]])
    assert both == latest and both["avwape"] is not None
    # Cut the bars before the second report: only the first one exists.
    early = ls.earnings_avwap(bars[:39], earnings_dates=[bars[9]["date"], bars[39]["date"]])
    assert early == ls.earnings_avwap(bars[:39], earnings_dates=[bars[9]["date"]])


@pytest.mark.parametrize("reaction_back", [0, 1, 121])
def test_a_reaction_too_fresh_or_too_old_is_unknown(reaction_back):
    count = 150
    index = count - 1 - reaction_back
    bars = _bars(count, {index: 4.0})
    got = ls.earnings_avwap(bars, earnings_dates=[bars[index]["date"]])
    assert got == {"avwape": None, "avwape_z": None, "under_avwape": "unknown"}


def test_two_and_one_hundred_twenty_sessions_back_count():
    for back in (2, 120):
        bars = _bars(150, {149 - back: 4.0})
        assert ls.earnings_avwap(bars, earnings_dates=[bars[149 - back]["date"]])["under_avwape"] in ("yes", "no")


def test_no_earnings_is_unknown():
    bars = _bars(60)
    unknown = {"avwape": None, "avwape_z": None, "under_avwape": "unknown"}
    assert ls.earnings_avwap(bars) == unknown
    assert ls.earnings_avwap(bars, earnings_dates=[], gap_date="") == unknown
    assert ls.earnings_avwap(bars, gap_date="2031-01-01") == unknown


def test_the_scan_gap_date_is_the_reaction_only_without_earnings_dates():
    bars = _bars(60, {31: 4.0})
    from_gap = ls.earnings_avwap(bars, gap_date=bars[31]["date"])
    assert from_gap == ls.earnings_avwap(bars, earnings_dates=[bars[30]["date"]])
    # With earnings dates the study's reaction day wins over the scan's gap date.
    assert ls.earnings_avwap(bars, earnings_dates=[bars[30]["date"]], gap_date=bars[20]["date"]) == from_gap


# --- the tier and the promotion order

def _row(symbol, rs, strength="no", under="unknown", setup=ls.LEADER_PULLBACK, leader=False):
    row = {"symbol": symbol, "setup": setup, "rs_percentile": rs, "leader": leader, "strength": 0.0,
           "exit": "hold", "strength_filter": strength, "under_avwape": under}
    row["setup_tier"] = ls.setup_tier(row)
    return row


def test_the_tier():
    assert ls.setup_tier(_row("A", 0.9, "yes", "yes")) == "strength_under_avwape"
    assert ls.setup_tier(_row("A", 0.9, "yes", "no")) == "strength"
    assert ls.setup_tier(_row("A", 0.9, "yes", "unknown")) == "strength"
    assert ls.setup_tier(_row("A", 0.9, "no", "yes")) == ""
    assert ls.setup_tier(_row("A", 0.9, "unknown", "yes")) == ""  # unknown never counts as yes
    assert ls.setup_tier(_row("A", 0.9, "yes", "yes", setup=ls.POST_EARNINGS_DRIFT)) == ""


def test_promotion_takes_strong_under_avwape_first_then_strong_then_the_rest():
    rows = [_row(f"R{i:02d}", 0.99 - i * 0.001) for i in range(8)]           # the rest, top RS
    rows += [_row(f"S{i}", 0.85 - i * 0.001, "yes", "no") for i in range(3)]  # strong
    rows += [_row(f"U{i}", 0.82 - i * 0.001, "yes", "yes") for i in range(2)]  # strong + under
    rows += [_row("LOW", 0.5, "yes", "yes")]                                  # not promotable
    gated = ls.apply_market_gate(ls.rank(rows), "yes", "trader")
    assert [row["symbol"] for row in gated][:6] == ["U0", "U1", "LOW", "S0", "S1", "S2"]
    promoted = [row["symbol"] for row in gated if row["promoted"]]
    assert promoted == ["U0", "U1", "S0", "S1", "S2", "R00", "R01", "R02", "R03", "R04"]
    assert len(promoted) == ls.PROMOTE_MAX


def test_a_market_not_working_still_promotes_nothing():
    gated = ls.apply_market_gate(ls.rank([_row("U", 0.9, "yes", "yes")]), "no", "trader")
    assert gated[0]["promoted"] is False and gated[0]["status"] == ls.STATUS_WAITING


# --- the golden: only the new fields and the promotion order differ

NEW_KEYS = {"avwape", "avwape_z", "under_avwape", "setup_tier"}


def _golden():
    from conftest import load_fixture_contract

    return load_fixture_contract("long_setups_golden_v1")


def _build(raw):
    from test_long_setups_golden import build

    return build(raw)


def test_the_golden_differs_only_in_the_new_fields_and_the_order():
    golden = _golden()
    old, new = golden["expected"], _build(golden["raw"])
    assert {k: v for k, v in new.items() if k != "rows"} == {k: v for k, v in old.items() if k != "rows"}

    def strip(row):
        out = {k: v for k, v in row.items() if k not in NEW_KEYS | {"promoted", "status"}}
        out["reasons"] = [text for text in row["reasons"] if text != ls.REASON_STRENGTH_UNDER_AVWAPE]
        return out

    key = lambda row: (row["symbol"], row["setup"])  # noqa: E731
    assert {key(row): strip(row) for row in new["rows"]} == {key(row): strip(row) for row in old["rows"]}
    assert len(new["rows"]) == len(old["rows"])
    for row in new["rows"]:
        assert NEW_KEYS <= set(row) if row["setup"] == ls.LEADER_PULLBACK else "setup_tier" in row
    tiers = {row["symbol"]: row["setup_tier"] for row in new["rows"] if row["setup_tier"]}
    assert tiers == {
        "GTLB": "strength_under_avwape", "ANF": "strength_under_avwape", "CRM": "strength_under_avwape",
        "CMBT": "strength_under_avwape",
        "NIQ": "strength", "AMPL": "strength", "FIVN": "strength", "BLSH": "strength", "OKTA": "strength",
        "PURR": "strength", "PSNL": "strength",
    }
    # The new order = the old rank order, tier first.
    order = {"strength_under_avwape": 0, "strength": 1, "": 2}
    old_index = {key(row): index for index, row in enumerate(old["rows"])}
    expected_order = sorted(old_index, key=lambda k: (order[tiers.get(k[0], "")], old_index[k]))
    assert [key(row) for row in new["rows"]] == expected_order
    # CMBT is under the 0.8 RS floor in this 86-name fixture but a leader (promotable); PSNL is 11th.
    assert [row["symbol"] for row in new["rows"] if row["promoted"]] == [
        "GTLB", "ANF", "CRM", "CMBT", "NIQ", "AMPL", "FIVN", "BLSH", "OKTA", "PURR"]
    old_promoted = [row["symbol"] for row in old["rows"] if row["promoted"]]
    assert old_promoted == ["NIQ", "AMPL", "FIVN", "BLSH", "GTLB", "OKTA", "ESTC", "ZETA", "WDAY", "ANF"]
    for row in new["rows"]:
        first_reason = row["reasons"][0] == ls.REASON_STRENGTH_UNDER_AVWAPE
        assert first_reason is (row["setup_tier"] == "strength_under_avwape")


def test_the_golden_avwape_is_the_studys():
    new = _build(_golden()["raw"])
    by_symbol = {row["symbol"]: row for row in new["rows"]}
    for symbol in ("GTLB", "ANF", "CRM", "CMBT"):
        assert by_symbol[symbol]["under_avwape"] == "yes" and by_symbol[symbol]["avwape_z"] < 0
    assert by_symbol["ESTC"]["strength_filter"] == "no"  # live ATR20: 1.91 ATR over the 50-day


# --- history and the shadow grade

def test_the_history_keeps_the_tier_fields():
    row = {"symbol": "A", "as_of": "2026-09-25", "setup": ls.LEADER_PULLBACK, "setup_tier": "strength",
           "strength_filter": "yes", "under_avwape": "no", "avwape_z": 0.4, "avwape": 10.0}
    (kept,) = ls.upsert_history([], [row])
    assert {k: kept[k] for k in ("setup_tier", "strength_filter", "under_avwape", "avwape_z")} == {
        "setup_tier": "strength", "strength_filter": "yes", "under_avwape": "no", "avwape_z": 0.4}


def _hist(tier, ret, spy, outcome="filled", day="2026-09-01", setup=ls.LEADER_PULLBACK):
    row = {"setup": setup, "as_of": day, "outcome": outcome, "return_pct": ret, "spy_return_pct": spy}
    if tier is not None:
        row["setup_tier"] = tier
    return row


def test_the_tier_grade():
    import setup_grades

    rows = [
        _hist("strength_under_avwape", 5.0, 1.0),   # beat, +4
        _hist("strength_under_avwape", 0.0, 1.0),   # lost, -1
        _hist("strength", 3.0, 1.0),                # beat, +2
        _hist("", -1.0, 1.0),                       # lost, -2
        _hist("strength_under_avwape", None, None, outcome="no_fill"),  # never a loss
        _hist("strength_under_avwape", 2.0, None),  # SPY unknown: left out
        _hist("strength_under_avwape", 2.0, 1.0, outcome=""),  # unsettled: left out
        _hist(None, 9.0, 1.0),                      # recorded before the tier: left out
        _hist("", 9.0, 1.0, setup=ls.POST_EARNINGS_DRIFT),  # not a leader pullback
    ]
    cells = setup_grades.long_setup_tier_cells(rows)
    assert (cells["strength_under_avwape"]["n"], cells["strength_under_avwape"]["wins"]) == (2, 1)
    assert cells["strength_under_avwape"]["mean_vs_spy"] == pytest.approx(1.5)
    assert (cells["strength"]["n"], cells[""]["n"]) == (1, 1)
    assert setup_grades.long_setup_tier_line(cells) == (
        "strong + under AVWAPE (shadow): n 2, beat SPY 50%, avg vs SPY +1.50%"
        " · strong only: n 1, beat SPY 100%, avg vs SPY +2.00%"
        " · the rest: n 1, beat SPY 0%, avg vs SPY -2.00% (filled limits, 10 sessions)")
    assert setup_grades.long_setup_tier_line(setup_grades.long_setup_tier_cells([])) == \
        "strong + under AVWAPE (shadow): no graded Long leaders yet."


def test_the_setup_tracker_section_shows_the_tier_line(tmp_path, monkeypatch):
    import project_paths
    from diagnostics.artifact_io import atomic_write_json
    from ui.services import working_lately_service as service

    current, history = tmp_path / "long_setups.json", tmp_path / "long_setups_history.json"
    atomic_write_json(current, {"as_of": "2026-09-25", "market_working": "yes", "rows": []})
    atomic_write_json(history, {"rows": [_hist("strength_under_avwape", 5.0, 1.0)]})
    monkeypatch.setattr(project_paths, "LONG_SETUPS_FILE", current)
    monkeypatch.setattr(project_paths, "LONG_SETUPS_HISTORY_FILE", history)
    service._LOOKING_BACK_CACHE.clear()
    try:
        lines = service.read_long_leader_lines()
    finally:
        service._LOOKING_BACK_CACHE.clear()
    assert "strong + under AVWAPE (shadow): n 1, beat SPY 100%, avg vs SPY +4.00%" in lines[-1]


# --- the store reads the earnings dates the study anchored on

def test_the_store_reads_the_earnings_dates_cache(tmp_path):
    import json

    import long_setups_store

    path = tmp_path / "earnings_dates_cache.json"
    path.write_text(json.dumps({"symbols": {"aaa": {"dates": ["2026-08-27", "bad", "2026-05-01"]},
                                            "BBB": {"dates": []}}}), encoding="utf-8")
    assert long_setups_store.read_earnings_dates(path) == {"AAA": ["2026-05-01", "2026-08-27"]}
    assert long_setups_store.read_earnings_dates(tmp_path / "missing.json") == {}
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert long_setups_store.read_earnings_dates(tmp_path / "bad.json") == {}


def test_publish_passes_the_earnings_dates_to_the_rule(tmp_path, monkeypatch):
    import long_setups_store

    seen = {}

    def _build(**kwargs):
        seen.update(kwargs)
        return {"as_of": "", "market_working": "unknown", "market_rule": "unknown", "rows": []}

    monkeypatch.setattr(long_setups_store.long_setups, "build_rows", _build)
    monkeypatch.setattr(long_setups_store, "read_earnings_dates", lambda path=None: {"AAA": ["2026-08-27"]})
    long_setups_store.publish_long_setups(bars_by_symbol={}, spy_bars=[], feature_rows=[], as_of="",
                                          path=tmp_path / "a.json", history_path=tmp_path / "b.json")
    assert seen["earnings_dates_by_symbol"] == {"AAA": ["2026-08-27"]}
