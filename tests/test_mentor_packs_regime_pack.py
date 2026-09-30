"""Trade Mentor regime pack (the tape): golden fixtures, unique tz-aware ids, no live path, the push line."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import regime_pack  # noqa: E402

NOW = datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc)  # 07:00 PT, 10:00 ET

DESK_GOLDEN = """## regime_pack
[tape:asof] Tape as of Tue 2026-09-29 07:00 PT (10:00 ET)
[tape:mode] Auto mode: DESK
[tape:d1env] D1 environment (SPY, 2026-09-29): bearish_trend
[tape:regime] Trader's regime: weak since 2026-09-01 (day 29)
[tape:night:1] (night read, 2026-09-28) SPY stayed below its D1 line.
[tape:night:2] (night read, 2026-09-28) QQQ held the weekly trend.
[tape:night:3] (night read, 2026-09-28) IWM was weakest.
[tape:econ:t1] Econ 2026-09-29 10:00 ET: ISM Manufacturing
[tape:econ:w1] Econ 2026-10-02 08:30 ET: Nonfarm payrolls
[tape:rrs:SPY] SPY rolling RRS: +0.00
[tape:rrs:QQQ] QQQ rolling RRS: +0.42
[tape:rrs:IWM] IWM rolling RRS: -1.30
[tape:breadth] Sector board (2026-09-29T13:50): strongest Technology (XLK, 5d -0.9%), Health Care (XLV, 5d +0.5%), \
Utilities (XLU, 5d +0.2%); weakest Materials (XLB, 5d -3.0%), Energy (XLE, 5d -2.4%), Financials (XLF, 5d -1.1%)
[tape:spy:pause] SPY pause observations today (updated 10:55 desk time): 2 long and 1 short names held up through \
a SPY pause"""


@pytest.fixture(autouse=True)
def _no_live_sources(monkeypatch):
    """A fixture build must never fall back to the live readers."""
    monkeypatch.setattr(regime_pack, "live_sources", lambda: pytest.fail("a live source was used"))


def _off_day() -> regime_pack.Sources:
    return replace(
        regime_pack.fixture_sources(),
        auto_state=lambda: {"mode": "OFF", "profile": "AWAY"},
        night_read=lambda session: None,
        index_rrs=lambda: {},
        sector_board=lambda: {},
        spy_pause=lambda session: None,
    )


def test_desk_day_with_a_night_read_is_golden():
    assert regime_pack.fixture().as_text() == DESK_GOLDEN


def test_off_day_without_a_night_read_says_so():
    pack = regime_pack.build(now=NOW, sources=_off_day())
    rows = {row["id"]: row for row in pack.rows}
    assert rows["tape:mode"]["text"] == "Auto mode: OFF (profile AWAY)"
    assert rows["tape:night:none"]["text"] == "Night read: none on file"
    assert not any(row_id.startswith(("tape:rrs:", "tape:breadth", "tape:spy:")) for row_id in rows), \
        "absent sources leave no rows, never zeros"


def test_an_empty_econ_file_is_one_row_with_the_brief_note():
    import econ_brief

    sources = replace(regime_pack.fixture_sources(), econ=lambda session: econ_brief.today_view(session, forecasts=[]))
    pack = regime_pack.build(now=NOW, sources=sources)
    econ = [row for row in pack.rows if row["id"].startswith("tape:econ:")]
    assert [row["id"] for row in econ] == ["tape:econ:none"]
    assert econ_brief.NO_BRIEF_TEXT in econ[0]["text"]


@pytest.mark.parametrize("sources", [regime_pack.fixture_sources(), _off_day()])
def test_ids_are_unique_and_the_stamp_is_tz_aware(sources):
    pack = regime_pack.build(now=NOW, sources=sources)
    assert len(pack.ids) == len(set(pack.ids)) == len(pack.rows)
    asof = next(row for row in pack.rows if row["id"] == "tape:asof")
    assert datetime.fromisoformat(asof["at_utc"]).tzinfo is not None
    assert datetime.fromisoformat(pack.built_utc).tzinfo is not None
    assert "TradingBotData" not in pack.as_json() and "AppData" not in pack.as_json()


def test_a_failing_source_is_unknown_and_the_rest_still_render():
    broken = replace(regime_pack.fixture_sources(), night_read=lambda session: (_ for _ in ()).throw(OSError("x")),
                     auto_state=lambda: (_ for _ in ()).throw(ValueError("bad")))
    rows = {row["id"]: row for row in regime_pack.build(now=NOW, sources=broken).rows}
    assert rows["tape:mode"]["kind"] == "unknown" and rows["tape:night:none"]["kind"] == "unknown"
    assert rows["tape:d1env"]["kind"] == "d1env"


def test_a_stale_spy_pause_file_is_left_out():
    stale = replace(regime_pack.fixture_sources(),
                    spy_pause=lambda session: {"date": "2026-09-28", "sides": {"long": {"AI": {}}}})
    assert "tape:spy:pause" not in regime_pack.build(now=NOW, sources=stale).ids


def test_the_push_line_is_pack_facts_only_and_short():
    line = regime_pack.push_line(regime_pack.fixture())
    assert line == "Tape: Auto DESK | D1 bearish_trend | next ISM Manufacturing 09-29 10:00 | RRS SPY +0.0 QQQ +0.4 IWM -1.3"
    off = regime_pack.push_line(regime_pack.build(now=NOW, sources=_off_day()))
    assert off.startswith("Tape: Auto OFF | D1 bearish_trend") and "RRS" not in off
    assert "night read" not in line.lower(), "the night read is model text; it never goes in the push"
    long_pack = regime_pack.build(now=NOW, sources=replace(
        regime_pack.fixture_sources(), d1_env=lambda day: "x" * 400))
    assert len(regime_pack.push_line(long_pack)) <= 200


def test_live_readers_read_files_only(tmp_path, monkeypatch):
    import project_paths

    sectors = tmp_path / "sector_indexes.csv"
    sectors.write_text("etf,sector,return_5d_pct,rs_rank\nXLK,Technology,1.0,1\nXLE,Energy,-2.0,2\n", encoding="utf-8")
    snapshot = tmp_path / "industry_board_snapshot.json"
    snapshot.write_text(json.dumps({"sector_path": str(sectors), "last_success_at": "2026-09-29T13:50:19"}),
                        encoding="utf-8")
    state = tmp_path / "autopilot_state.json"
    state.write_text(json.dumps({"enabled": False, "profile": "away"}), encoding="utf-8")
    monkeypatch.setattr(project_paths, "INDUSTRY_BOARD_STATE_FILE", snapshot)
    monkeypatch.setattr(project_paths, "AUTOPILOT_STATE_FILE", state)
    board = regime_pack._live_sector_board()
    assert board["as_of"].startswith("2026-09-29") and [row["etf"] for row in board["rows"]] == ["XLK", "XLE"]
    assert regime_pack._live_auto_state() == {"mode": "OFF", "profile": "AWAY"}
    assert regime_pack._live_index_rrs() == {}, "no file-based index RRS: nothing, never computed"


def test_night_read_lines_lose_the_night_models_own_ids():
    """The live read cites its own ids on every line; the tape model would copy them and be rejected."""
    live_shaped = replace(regime_pack.fixture_sources(), night_read=lambda session: {
        "session_date": "2026-09-28",
        "read": {"paragraph": "SPY stayed in a bear channel [regime:1] [structure:2026-09-28:SPY]. "
                              "QQQ held its weekly trend [structure:2026-09-28:QQQ]."}})
    night = [row for row in regime_pack.build(now=NOW, sources=live_shaped).rows if row["kind"] == "night"]
    assert [row["id"] for row in night] == ["tape:night:1", "tape:night:2"]
    assert [row["text"] for row in night] == ["(night read, 2026-09-28) SPY stayed in a bear channel.",
                                             "(night read, 2026-09-28) QQQ held its weekly trend."]


def test_the_night_read_comes_from_market_regimes_latest(tmp_path):
    import market_regimes

    (tmp_path / "2026-09-28.json").write_text(json.dumps(
        {"session_date": "2026-09-28", "read": {"paragraph": "SPY closed weak. QQQ held."}}), encoding="utf-8")
    (tmp_path / "2026-09-30.json").write_text(json.dumps(
        {"session_date": "2026-09-30", "read": {"paragraph": "From the future."}}), encoding="utf-8")
    sources = replace(regime_pack.fixture_sources(),
                      night_read=lambda session: market_regimes.latest_regime_read(on=session, root=tmp_path))
    night = [row for row in regime_pack.build(now=NOW, sources=sources).rows if row["kind"] == "night"]
    assert [row["text"] for row in night] == ["(night read, 2026-09-28) SPY closed weak.",
                                             "(night read, 2026-09-28) QQQ held."]
