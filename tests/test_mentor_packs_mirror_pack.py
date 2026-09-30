"""Mentor P9: the mirror pack. Weekly cuts of the trader's own record, each with n and weeks=, a
Wilson LB on every rate, "too few" under the floor, spreads paired, no live path, fast."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import journal_read, mirror_pack, registry  # noqa: E402

NOW = mirror_pack.FIXTURE_NOW


@pytest.fixture
def world(tmp_path):
    return mirror_pack.write_fixture_world(tmp_path)


def _rows(pack):
    return {row["id"]: row for row in pack.rows}


def test_golden_fixture_covers_every_cut(world):
    rows = _rows(mirror_pack.build(now=NOW, paths=world))
    assert rows["mirror:weeks"]["text"].startswith("Window weeks=6 (2026-08-17 to 2026-09-28)")
    # 1. liked vs scan by side at 5 and 10 sessions (27 of 36 LONG picks won at +1.5%, 9 lost at -2.0%).
    liked = rows["mirror:liked:LONG:5"]
    assert (liked["n"], liked["wins"]) == (36, 27) and not liked["too_few"]
    assert liked["text"] == (
        "Liked LONG (claimed picks) at 5 sessions, daily closes, weeks=6: n=36, win 75% (LB 0.59), "
        "avg side return +0.62%; scan LONG baseline n=236, win 54% (LB 0.47), avg side return +0.05%")
    assert rows["mirror:liked:LONG:10"]["n"] == 36
    short = rows["mirror:liked:SHORT:5"]
    assert short["too_few"] and "n=3: too few (floor 30)" in short["text"]
    assert "n=0: too few" in rows["mirror:liked:SHORT:10"]["text"] and "3 with no scan row" in rows["mirror:liked:SHORT:10"]["text"]
    # 2. veto reason x side at 10 sessions (the follow-up note is not a second veto).
    veto = rows["mirror:veto:compressed:LONG"]
    assert (veto["n"], veto["wins"]) == (40, 32)
    assert "Vetoed LONG for compressed, what the name did 10 sessions later, weeks=6: n=40, the name won 80%" in veto["text"]
    assert rows["mirror:veto:volume_dry:SHORT"]["too_few"] and "(5 with no outcome yet)" in rows["mirror:veto:volume_dry:SHORT"]["text"]
    # 3. journal: kind (the spread is ONE option decision), hold, hour, weekday, account tax class.
    assert rows["mirror:journal:kind:day"]["n"] == 36
    assert rows["mirror:journal:kind:option"]["n"] == 1 and rows["mirror:journal:kind:option"]["too_few"]
    assert rows["mirror:journal:kind:swing"]["n"] == 4
    day = rows["mirror:journal:kind:day"]["text"]
    assert "n=36, win 33% (LB 0.20), expectancy -$6.67 per trade; net R" in day and "(R unknown for 16)" in day
    assert rows["mirror:journal:hold:5-30-min"]["n"] == 36
    assert "mirror:journal:hold:overnight-1-5-days" in rows
    assert rows["mirror:journal:hour:09"]["n"] == 36 and "entry hour (ET) 09" in rows["mirror:journal:hour:09"]["text"]
    assert "mirror:journal:weekday:Mon" in rows
    assert rows["mirror:journal:tax:margin"]["n"] == 37 and rows["mirror:journal:tax:tax_free"]["n"] == 4
    # 4. by the trader's structural regime on the entry day.
    assert rows["mirror:regime:bear_channel_lower_highs"]["n"] + rows["mirror:regime:weekly_hh_then_compression"]["n"] == 41
    assert "Journal in your regime bear channel, lower highs," in rows["mirror:regime:bear_channel_lower_highs"]["text"]
    # 5. the caveats, one line each.
    assert len(rows["mirror:caveats"]["lines"]) == len(mirror_pack.CAVEATS) and "3 min" in rows["mirror:caveats"]["text"]


def test_every_cut_prints_weeks_and_n_and_no_rate_under_the_floor(world):
    pack = mirror_pack.build(now=NOW, paths=world)
    for row in pack.rows:
        if row["kind"] in ("asof", "weeks", "caveats"):
            continue
        assert "weeks=6" in row["text"] and f"n={row['n']}" in row["text"], row["id"]
        if row["too_few"]:
            assert "too few (floor 30)" in row["text"] and "LB" not in row["text"].split(";")[0], row["id"]
        else:
            assert "LB" in row["text"], row["id"]


def test_ids_unique_citable_tz_aware_and_the_pack_is_registered(world):
    from mentor_packs.citations import CITATION_RE

    pack = mirror_pack.build(now=NOW, paths=world)
    assert len(pack.ids) == len(set(pack.ids))
    for row_id in pack.ids:
        assert CITATION_RE.fullmatch(f"[{row_id}]"), row_id
    assert datetime.fromisoformat(_rows(pack)["mirror:asof"]["asof_utc"]).tzinfo is not None
    assert "mirror_pack" in registry.names()


def test_weeks_argument_narrows_the_window_and_bad_weeks_is_an_empty_pack(world):
    rows = _rows(mirror_pack.build(2, now=NOW, paths=world))
    assert rows["mirror:weeks"]["text"].startswith("Window weeks=2 (2026-09-14 to 2026-09-28)")
    assert rows["mirror:liked:LONG:5"]["n"] < 36
    assert mirror_pack.build("99", now=NOW, paths=world).rows == ()
    assert "between 1 and 52" in mirror_pack.build("x", now=NOW, paths=world).empty_text


def test_the_hash_ignores_the_clock_but_not_the_facts(world):
    from datetime import timedelta

    one = mirror_pack.build(now=NOW, paths=world)
    two = mirror_pack.build(now=NOW + timedelta(minutes=5), paths=world)
    assert mirror_pack.pack_hash(one) == mirror_pack.pack_hash(two)
    assert mirror_pack.pack_hash(one) != mirror_pack.pack_hash(mirror_pack.build(3, now=NOW, paths=world))


def test_spreads_pair_within_three_minutes_only():
    base = {"account_number": "M1", "security_type": "OPT", "status": "CLOSED", "net_pnl": 10.0}
    trades = [
        {**base, "trade_id": "a", "symbol": "AMD261016C00150000", "opened_at": "2026-09-15T10:00:00-04:00",
         "closed_at": "2026-09-16T10:00:00-04:00"},
        {**base, "trade_id": "b", "symbol": "AMD261016C00160000", "opened_at": "2026-09-15T10:02:59.500000-04:00",
         "closed_at": "2026-09-16T10:00:00-04:00"},
        {**base, "trade_id": "c", "symbol": "AMD261016P00140000", "opened_at": "2026-09-15T10:09:00-04:00",
         "closed_at": "2026-09-16T10:00:00-04:00"},
    ]
    units = journal_read.units(trades)
    assert sorted(u.ids for u in units) == [["a", "b"], ["c"]]
    assert all(u.kind == "option" for u in units) and next(u for u in units if u.ids == ["a", "b"]).pnl == 20.0


def test_naive_and_fractional_iso_times_parse_as_new_york():
    assert journal_read.parse_time("2026-09-29T15:49:25.773000-04:00").hour == 15
    assert journal_read.parse_time("2026-09-29T10:00:00").utcoffset().total_seconds() == -4 * 3600
    assert journal_read.parse_time("") is None and journal_read.parse_time("not a time") is None


def test_no_live_path_and_read_only(world, monkeypatch):
    import project_paths

    for name in ("JOURNAL_DB_FILE", "CLAIMED_PICKS_FILE", "TRADER_ANNOTATIONS_FILE", "MASTER_AVWAP_TIER_OUTCOMES_FILE",
                 "VETO_COHORT_OUTCOMES_FILE"):
        monkeypatch.setattr(project_paths, name, Path("Z:/must/not/read"))
    assert mirror_pack.fixture().rows
    for name in ("mentor_packs/mirror_pack.py", "mentor_packs/journal_read.py"):
        source = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        assert "JournalStore(" not in source and "FocusPickStore(" not in source and "PySide6" not in source
        assert "import ui" not in source and "from ui" not in source
    assert "mode=ro" in (SCRIPTS_DIR / "mentor_packs" / "journal_read.py").read_text(encoding="utf-8")


def test_no_rule_proposal_words(world):
    text = mirror_pack.build(now=NOW, paths=world).as_text().lower()
    for word in ("you should", "rule change", "stop trading", "size up", "order"):
        assert word not in text, word


def test_builds_fast(world):
    mirror_pack.build(now=NOW, paths=world)
    started = time.perf_counter()
    mirror_pack.build(now=NOW, paths=world)
    assert time.perf_counter() - started < 2.0


# ---------------------------------------------------------------- review fixes
def test_a_later_drop_ends_the_claim_it_names(world):
    base = mirror_pack.build(now=NOW, paths=world)
    n = _rows(base)["mirror:liked:LONG:5"]["n"]
    with world.claimed_picks.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"action": "claim", "symbol": "B000", "side": "LONG", "session_date": "2026-08-17",
                                 "claimed_setup_id": "general"}) + "\n")
    assert _rows(mirror_pack.build(now=NOW, paths=world))["mirror:liked:LONG:5"]["n"] == n + 1
    with world.claimed_picks.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"action": "drop", "symbol": "B000", "side": "LONG", "session_date": "2026-08-18",
                                 "claimed_setup_id": "other_setup"}) + "\n")
    assert _rows(mirror_pack.build(now=NOW, paths=world))["mirror:liked:LONG:5"]["n"] == n + 1,         "a drop ends only the claim it names"
    with world.claimed_picks.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"action": "drop", "symbol": "B000", "side": "LONG", "session_date": "2026-08-18",
                                 "claimed_setup_id": "general"}) + "\n")
    assert _rows(mirror_pack.build(now=NOW, paths=world))["mirror:liked:LONG:5"]["n"] == n


def test_no_scan_row_is_told_apart_from_not_matured_yet(world):
    with world.claimed_picks.open("a", encoding="utf-8") as handle:
        for symbol, day in (("NEW1", "2026-09-24"), ("NEW2", "2026-09-25")):
            handle.write(json.dumps({"action": "claim", "symbol": symbol, "side": "SHORT", "session_date": day,
                                     "claimed_setup_id": "x"}) + "\n")
    text = _rows(mirror_pack.build(now=NOW, paths=world))["mirror:liked:SHORT:10"]["text"]
    assert "(3 with no scan row, 2 not matured yet)" in text


def test_spreads_pair_by_the_first_leg_not_a_chain():
    base = {"account_number": "M1", "security_type": "OPT", "status": "CLOSED", "net_pnl": 10.0,
            "closed_at": "2026-09-16T10:00:00-04:00"}
    trades = [{**base, "trade_id": t, "symbol": f"AMD261016C0015{i}000", "opened_at": f"2026-09-15T10:0{m}-04:00"}
              for i, (t, m) in enumerate((("a", "0:00"), ("b", "2:30"), ("c", "5:00")))]
    assert sorted(u.ids for u in journal_read.units(trades)) == [["a", "b"], ["c"]]
