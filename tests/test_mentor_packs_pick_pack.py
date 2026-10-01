"""Trade Mentor pick pack: golden fixtures, unique ids, floors out loud, unknown never guessed, read-only."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import pick_pack, registry  # noqa: E402

NOW = pick_pack.FIXTURE_NOW

NVDA_GOLDEN = """## pick_pack
[pick:NVDA:asof] NVDA as of market date 2026-09-29
[pick:NVDA:membership] Focus: swing long (pick clock from 2026-09-22, added)
[pick:NVDA:claim:LONG:avwap_breakout] Claim: LONG avwap_breakout since session 2026-09-29 (d1)
[pick:NVDA:fb:2026-09-28T06:50:00] Verdict 2026-09-28: like LONG (swing, from d1)
[pick:NVDA:side] Side assessed: LONG
[pick:NVDA:cell] Setup cell avwap_breakout LONG (setup from the trader's claim), 5-session D1 outcomes: n=40 (floor 30), win rate 60%, Wilson LB 0.45, avg side return +0.60%
[pick:NVDA:cell_r] Leaderboard avwap_breakout LONG: avg closed R +0.06 over n=100 closed setups (floor 30)
[pick:NVDA:earn] Own earnings: Thu 2026-11-19, in 51 days (long and short alike)
[pick:NVDA:industry] Industry: Semiconductors (5 peers); 2 report within +-7 calendar days, nearest 2 listed
[pick:NVDA:peer:AVGO] Peer AVGO earnings Wed 2026-09-30, tomorrow
[pick:NVDA:peer:MU] Peer MU earnings Thu 2026-09-24, 5 days ago
[pick:NVDA:plan:risk:1] Plan [plan:risk:1]: Max 3 open shorts at once.
[pick:NVDA:plan:risk:2] Plan [plan:risk:2]: No new entries after 12:30 PT.
[pick:NVDA:cohort:1] Cohort human_focus_swing LONG 1d: n=101 (floor 30), win rate 52%, avg side return +0.40%
[pick:NVDA:cohort:3] Cohort human_focus_swing LONG 3d: n=103 (floor 30), win rate 52%, avg side return +0.40%
[pick:NVDA:cohort:5] Cohort human_focus_swing LONG 5d: n=105 (floor 30), win rate 52%, avg side return +0.40%
[pick:NVDA:cohort:10] Cohort human_focus_swing LONG 10d: n=110 (floor 30), win rate 52%, avg side return +0.40%
[pick:NVDA:cohort:d1:1] Cohort human_focus_swing_d1 LONG 1d: too few, n=11 (floor 30)
[pick:NVDA:cohort:d1:3] Cohort human_focus_swing_d1 LONG 3d: too few, n=13 (floor 30)
[pick:NVDA:cohort:d1:5] Cohort human_focus_swing_d1 LONG 5d: too few, n=15 (floor 30)
[pick:NVDA:cohort:d1:10] Cohort human_focus_swing_d1 LONG 10d: too few, n=20 (floor 30)
[pick:NVDA:news:12] Headline "Broadcom reports tomorrow; chips mixed" (finance.yahoo.com, Tue 09-29 05:10 PT) https://finance.yahoo.com/news/avgo
[pick:NVDA:news:11] Headline "Nvidia adds $150 billion buyback" (nytimes.com, Mon 09-28 06:49 PT) https://www.nytimes.com/nvda-buyback
[pick:NVDA:brief] Night brief (2026-09-28): NVDA held its AVWAP into the close. | Higher low on D1 above the anchor. (src: scan.tier_list)"""


@pytest.fixture()
def world(tmp_path, monkeypatch):
    # Any reach for a live path fails the test.
    monkeypatch.setattr(pick_pack, "live_paths", lambda: pytest.fail("the pack reached for live paths"))
    return pick_pack.write_fixture_world(tmp_path / "desk")


def _rows(pack):
    return {row["id"]: row for row in pack.rows}


def test_registered_as_a_tool_with_a_symbol_argument():
    assert "pick_pack" in registry.names()
    schema = registry.modules()["pick_pack"].SCHEMA
    assert schema["function"]["parameters"]["required"] == ["symbol"]


def test_long_golden(world):
    pack = pick_pack.build("nvda", now=NOW, paths=world)
    assert pack.as_text() == NVDA_GOLDEN


def test_short_golden(world):
    rows = _rows(pick_pack.build("TSLA", now=NOW, paths=world))
    assert rows["pick:TSLA:membership"]["text"] == "Focus: swing short (pick clock from 2026-09-25, added)"
    assert rows["pick:TSLA:side"]["side"] == "SHORT"
    assert rows["pick:TSLA:cell"]["text"] == (
        "Setup cell avwap_band_bounce SHORT (setup from the latest D1 scan row, 2026-09-26), 5-session D1 "
        "outcomes: n=35 (floor 30), win rate 40%, Wilson LB 0.26, avg side return -0.20%"
    )
    assert rows["pick:TSLA:claim"]["text"] == "Claim: no standing claim"
    assert rows["pick:TSLA:earn"]["date"] == "2026-10-21"
    assert "0 report within" in rows["pick:TSLA:industry"]["text"]
    assert rows["pick:TSLA:cohort:5"]["text"].startswith("Cohort human_focus_swing SHORT 5d: n=85")
    assert "pick:TSLA:cohort:manual:1" not in rows, "an origin sub-cohort with no rows is left out"


def test_a_peer_reporting_tomorrow_is_listed_first_with_its_date(world):
    pack = pick_pack.build("NVDA", now=NOW, paths=world)
    peers = [row for row in pack.rows if row["kind"] == "peer_earnings"]
    assert [row["peer"] for row in peers] == ["AVGO", "MU"]  # AMAT (+21 d) is outside +-7
    assert peers[0]["date"] == "2026-09-30" and peers[0]["days"] == 1 and "tomorrow" in peers[0]["text"]


def test_peers_are_capped_at_twelve_nearest_first(world):
    from datetime import date, timedelta

    day = date(2026, 9, 29)
    many = {f"P{i:02d}": {"events": [{"earnings_date": (day + timedelta(days=i - 7)).isoformat()}]} for i in range(15)}
    members = ["NVDA", *many]
    import json

    world.earnings_history.write_text(json.dumps({"symbols": many}), encoding="utf-8")
    paths = replace(world, industry_map=lambda: {"NVDA": {"industry": "Semis", "industry_member_symbols": members}})
    peers = [row for row in pick_pack.build("NVDA", now=NOW, paths=paths).rows if row["kind"] == "peer_earnings"]
    assert len(peers) == pick_pack.PEER_MAX_ROWS
    assert [abs(row["days"]) for row in peers] == sorted(abs(row["days"]) for row in peers)
    assert all(abs(row["days"]) <= 7 for row in peers)


def test_below_the_n_floor_says_too_few_and_carries_n(world):
    rows = _rows(pick_pack.build("AMD", now=NOW, paths=world))
    cell = rows["pick:AMD:cell"]
    assert cell["n"] == 12
    assert "too few, n=12 (floor 30)" in cell["text"]
    assert "win rate" not in cell["text"] and "Wilson" not in cell["text"]


def test_an_empty_plan_is_one_row_and_no_rule_ids(tmp_path):
    paths = pick_pack.write_fixture_world(tmp_path / "desk", plan_text="# Trading plan\n")
    pack = pick_pack.build("NVDA", now=NOW, paths=paths)
    plan_rows = [row for row in pack.rows if row["id"].startswith("pick:NVDA:plan")]
    assert [row["id"] for row in plan_rows] == ["pick:NVDA:plan"]
    assert plan_rows[0]["text"] == "Plan: no plan lines"
    assert pick_pack.plan_ids(pack) == set()


def test_unknown_earnings_is_unknown_never_a_guess(world):
    rows = _rows(pick_pack.build("ZZZ", now=NOW, paths=world))
    assert rows["pick:ZZZ:earn"]["text"] == "Own earnings: unknown (no upcoming date in the earnings calendar)"
    assert "date" not in rows["pick:ZZZ:earn"]
    assert rows["pick:ZZZ:peers"]["text"] == "Peer earnings: industry unknown"
    assert rows["pick:ZZZ:cell"]["text"] == "Setup cell: no setup known for ZZZ LONG"


def test_a_missing_calendar_is_unknown_and_does_not_blank_the_rest(world):
    world.earnings_history.unlink()
    rows = _rows(pick_pack.build("NVDA", now=NOW, paths=world))
    assert rows["pick:NVDA:earn"]["kind"] == "unknown" and "unknown" in rows["pick:NVDA:earn"]["text"]
    assert rows["pick:NVDA:peers"]["kind"] == "unknown"
    assert rows["pick:NVDA:cell"]["kind"] == "cell"


@pytest.mark.parametrize("symbol", ["NVDA", "TSLA", "AMD", "ZZZ", "QQQ"])
def test_ids_are_unique_stable_and_all_prefixed(world, symbol):
    first = pick_pack.build(symbol, now=NOW, paths=world)
    again = pick_pack.build(symbol, now=NOW, paths=world)
    assert len(first.ids) == len(set(first.ids)) == len(first.rows)
    assert first.ids == again.ids
    assert all(row_id.startswith(f"pick:{symbol}:") for row_id in first.ids)
    assert all(f"[{row_id}]" in first.as_text() for row_id in first.ids)


def test_dates_are_market_dates_and_the_stamp_is_tz_aware(world):
    # 23:30 PT on the 28th is already the 29th in New York: the market date wins.
    late = datetime.fromisoformat("2026-09-29T06:30:00+00:00")
    rows = _rows(pick_pack.build("NVDA", now=late, paths=world))
    assert datetime.fromisoformat(rows["pick:NVDA:asof"]["at_utc"]).tzinfo is not None
    assert "2026-09-29" in rows["pick:NVDA:asof"]["text"]
    assert rows["pick:NVDA:peer:AVGO"]["days"] == 1


def test_the_hash_ignores_the_stamp_but_not_the_evidence(world):
    a = pick_pack.build("NVDA", now=NOW, paths=world)
    b = pick_pack.build("NVDA", now=NOW.replace(minute=30), paths=world)
    assert a.built_utc or b.built_utc
    assert pick_pack.pack_hash(a) == pick_pack.pack_hash(b)
    world.focus_swing_longs.write_text("ZZZ\n", encoding="utf-8")  # NVDA left Focus
    c = pick_pack.build("NVDA", now=NOW, paths=world)
    assert pick_pack.pack_hash(c) != pick_pack.pack_hash(a)


def test_the_big_csv_is_reread_when_it_changes(world):
    before = _rows(pick_pack.build("NVDA", now=NOW, paths=world))["pick:NVDA:cell"]["n"]
    with world.tier_outcomes.open("a", encoding="utf-8") as handle:
        handle.write("X99,LONG,avwap_breakout,2026-09-01,5,True,1.0,False\n")
    after = _rows(pick_pack.build("NVDA", now=NOW, paths=world))["pick:NVDA:cell"]["n"]
    assert (before, after) == (40, 41)


def test_the_pack_writes_nothing(world):
    files = sorted(p for p in world.focus_longs.parent.iterdir())
    before = {p.name: p.stat().st_mtime_ns for p in files}
    pick_pack.build("NVDA", now=NOW, paths=world)
    after = {p.name: p.stat().st_mtime_ns for p in sorted(world.focus_longs.parent.iterdir())}
    assert after == before


def test_every_path_in_the_fixture_is_under_the_fixture_root(world):
    root = world.focus_longs.parent
    for name in ("focus_longs", "focus_shorts", "focus_swing_longs", "focus_swing_shorts", "pick_clocks",
                 "claimed_picks", "pick_feedback", "tier_outcomes", "leaderboard", "earnings_history",
                 "cohort_performance", "plan"):
        assert getattr(world, name).parent == root, name


def test_a_bad_ticker_is_an_empty_pack():
    assert pick_pack.build("", now=NOW).ids == ()
    assert pick_pack.build("NV DA; drop", now=NOW).ids == ()


def test_the_pack_never_constructs_the_writing_stores():
    source = (SCRIPTS_DIR / "mentor_packs" / "pick_pack.py").read_text(encoding="utf-8")
    assert "FocusPickStore(" not in source and "JournalStore(" not in source
    assert "from ui" not in source and "import ui" not in source and "PySide6" not in source


# ---------------------------------------------------------------- news (P7) and the plan hash
def test_news_section_max_five_from_the_last_three_days_each_with_its_url(world):
    from mentor_packs import news_pack

    many = [{"id": 100 + i, "symbol": "NVDA", "title": f"Story {i}", "url": f"https://example.com/{i}",
             "source": "example.com", "published_utc": f"2026-09-29T{6 + i:02d}:00:00+00:00",
             "fetched_utc": "2026-09-29T13:00:00+00:00", "feed": "yahoo"} for i in range(7)]
    old = {"id": 1, "symbol": "NVDA", "title": "Too old", "url": "https://example.com/old", "source": "x",
           "published_utc": "2026-09-25T12:00:00+00:00", "fetched_utc": "2026-09-25T12:00:00+00:00", "feed": "yahoo"}
    paths = replace(world, news=news_pack.list_reader(many + [old]))
    news = [row for row in pick_pack.build("NVDA", now=NOW, paths=paths).rows if row["kind"] == "news"]
    assert [row["id"] for row in news] == [f"pick:NVDA:news:{106 - i}" for i in range(5)], "newest first, max 5"
    assert all(row["url"] in row["text"] and row["url"].startswith("https://") for row in news)
    assert all(row["news_id"] == f"news:NVDA:{row['id'].rsplit(':', 1)[1]}" for row in news)


def test_no_headline_is_a_row_that_says_so(world):
    from mentor_packs import news_pack

    rows = _rows(pick_pack.build("NVDA", now=NOW, paths=replace(world, news=news_pack.list_reader([]))))
    assert rows["pick:NVDA:news"]["text"] == "News: no stored headlines in the last 3 days"
    no_source = _rows(pick_pack.build("NVDA", now=NOW, paths=replace(world, news=None)))
    assert "not read" in no_source["pick:NVDA:news"]["text"]


def test_a_new_headline_changes_the_hash(world):
    from mentor_packs import news_pack

    a = pick_pack.build("NVDA", now=NOW, paths=world)
    fresh = {"id": 20, "symbol": "NVDA", "title": "Fresh", "url": "https://x.com/f", "source": "x.com",
             "published_utc": "2026-09-29T13:59:00+00:00", "fetched_utc": "2026-09-29T14:00:00+00:00", "feed": "google"}
    b = pick_pack.build("NVDA", now=NOW, paths=replace(world, news=news_pack.list_reader(
        news_pack.FIXTURE_HEADLINES + (fresh,))))
    assert pick_pack.pack_hash(a) != pick_pack.pack_hash(b)


def test_any_plan_edit_changes_the_hash_even_one_that_adds_no_line(world):
    a = pick_pack.build("NVDA", now=NOW, paths=world)
    text = world.plan.read_text(encoding="utf-8")
    world.plan.write_text(text.replace("## Risk", "Rewritten on Sunday by the plan session.\n\n## Risk"), encoding="utf-8")
    b = pick_pack.build("NVDA", now=NOW, paths=world)
    assert a.ids == b.ids and a.as_text() == b.as_text(), "the edit adds no citable line"
    assert pick_pack.pack_hash(a) != pick_pack.pack_hash(b), "a changed plan re-narrates the card"


# ---------------------------------------------------------------- P13: the M5 branch
def test_an_m5_focus_pick_takes_its_setup_cell_from_the_desk_s_day_trade_grades(world):
    rows = {row["id"]: row for row in pick_pack.build("AMD", now=NOW, paths=world).rows}
    assert rows["pick:AMD:branch"]["branch"] == "m5" and "M5 bounce cell" in rows["pick:AMD:branch"]["text"]
    band = rows["pick:AMD:m5cell:vwap_lower_band"]
    assert band["setup"] == "vwap_lower_band" and band["n"] == 40
    assert "latest M5 alert, 2026-09-25 07:05 PT" in band["text"] and "grade B" in band["text"]
    assert "low bound 0.40" in band["text"] and "as of 2026-09-28" in band["text"]
    thin = rows["pick:AMD:m5cell:10_candle"]
    assert "too few, n=12 (floor 30)" in thin["text"] and "grade New" not in thin["text"], "a thin cell carries n only"
    assert "pick:AMD:m5cell:ema_8" not in rows, "only the latest alert's types"
    assert "D1 context only" in rows["pick:AMD:cell"]["text"]
    assert rows["pick:AMD:cohort:1"]["text"].startswith("Cohort human_focus_m5 SHORT")


def test_an_m5_pick_with_no_m5_files_says_not_stored_and_never_guesses(world):
    rows = {row["id"]: row for row in pick_pack.build("AMD", now=NOW, paths=replace(world, m5_grades=None)).rows}
    assert rows["pick:AMD:m5cell"]["text"].startswith("M5 setup cell: not stored")
    missing = replace(world, m5_alerts=world.focus_longs.with_name("nope.csv"))
    rows = {row["id"]: row for row in pick_pack.build("AMD", now=NOW, paths=missing).rows}
    assert "not stored" in rows["pick:AMD:m5cell"]["text"]


def test_an_m5_pick_with_no_recent_alert_of_its_own_says_so(world):
    world.focus_shorts.write_text("AMD\nQQQX\n", encoding="utf-8")
    rows = {row["id"]: row for row in pick_pack.build("QQQX", now=NOW, paths=world).rows}
    assert "no M5 alert for QQQX SHORT in the last 60 days" in rows["pick:QQQX:m5cell"]["text"]
    later = NOW.replace(month=12)
    rows = {row["id"]: row for row in pick_pack.build("AMD", now=later, paths=world).rows}
    assert "no M5 alert for AMD SHORT in the last 60 days" in rows["pick:AMD:m5cell"]["text"]


def test_the_650_mb_m5_outcome_store_is_never_opened(monkeypatch, tmp_path):
    import builtins
    import io

    import project_paths

    outcomes = tmp_path / "intraday_bounce_outcomes.csv"
    outcomes.write_text("event_id\n", encoding="utf-8")
    monkeypatch.setattr(project_paths, "INTRADAY_BOUNCE_OUTCOMES_FILE", outcomes)
    opened: list[str] = []

    def audit(real):
        def wrapper(file, *args, **kwargs):
            if "intraday_bounce_outcomes" in str(file):
                opened.append(str(file))
                raise AssertionError(f"pick_pack opened {file}")
            return real(file, *args, **kwargs)
        return wrapper

    real_path_open = Path.open
    monkeypatch.setattr(builtins, "open", audit(builtins.open))
    monkeypatch.setattr(io, "open", audit(io.open))
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: audit(lambda f, *x, **y: real_path_open(self, *x, **y))(
        self, *a, **k))
    live = pick_pack.live_paths()  # builds the live path set only; nothing is read
    assert outcomes not in {value for value in vars(live).values() if isinstance(value, Path)}
    world = pick_pack.write_fixture_world(tmp_path / "desk")
    for symbol in ("AMD", "NVDA", "TSLA"):
        pick_pack.build(symbol, now=NOW, paths=world)
    assert opened == []


def test_the_pick_carries_the_newest_night_brief_with_its_date(world):
    """P15a: <= 3 lines of the symbol's newest ticker brief, found through the manifests."""
    brief = _rows(pick_pack.build("NVDA", now=NOW, paths=world))["pick:NVDA:brief"]
    assert brief["session"] == "2026-09-28" and brief["evidence_refs"] == ["scan.tier_list"]
    assert brief["text"].startswith("Night brief (2026-09-28): ") and brief["text"].count(" | ") <= 2


def test_no_brief_is_a_none_row_never_silence(world):
    rows = _rows(pick_pack.build("TSLA", now=NOW, paths=world))  # membership only that night: no brief
    assert "pick:TSLA:brief" not in rows and "none for TSLA" in rows["pick:TSLA:brief:none"]["text"]
    unread = _rows(pick_pack.build("NVDA", now=NOW, paths=replace(world, briefs=None)))
    assert unread["pick:NVDA:brief:none"]["text"] == "Night brief: not read (no briefs store)"


def test_a_brief_from_after_the_market_date_is_never_read(world):
    folder = world.briefs / "2026" / "2026-10-01"
    folder.mkdir(parents=True)
    (folder / "ticker_briefs_manifest.jsonl").write_text(
        '{"symbol": "NVDA", "status": "briefed", "result": {"summary": {"executive_summary": "from the future"}}}\n',
        encoding="utf-8")
    assert "future" not in _rows(pick_pack.build("NVDA", now=NOW, paths=world))["pick:NVDA:brief"]["text"]


def test_a_reused_or_older_brief_is_still_shown_never_none(world):
    """Review advisory 2: a reused brief writes no manifest row, and 10 sessions was too short a look-back."""
    import json

    from ai_jobs import week_names

    def manifest(day, rows):
        folder = world.briefs / "2026" / day
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "ticker_briefs_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                                             encoding="utf-8")

    brief = {"status": "briefed", "result": {"summary": {"executive_summary": "TSLA based above its anchor."}}}
    manifest("2026-09-08", [{**brief, "symbol": "TSLA", "session_date": "2026-09-08"}])
    for day in range(9, 26):  # 15+ later brief nights that did not re-brief TSLA or AMD
        if datetime(2026, 9, day).weekday() < 5:
            manifest(f"2026-09-{day:02d}", [{"symbol": "ZZZ", "status": "membership_only"}])
    week_names.append_evidence_cache(week_names.evidence_cache_path(world.briefs), {
        "symbol": "AMD", "session_date": "2026-09-10", "status": "briefed",
        "result": {"summary": {"executive_summary": "AMD faded into its band."}}}, "hash-amd")
    tsla = _rows(pick_pack.build("TSLA", now=NOW, paths=world))
    assert tsla["pick:TSLA:brief"]["text"] == "Night brief (2026-09-08): TSLA based above its anchor."
    amd = _rows(pick_pack.build("AMD", now=NOW, paths=world))
    assert "pick:AMD:brief:none" not in amd
    assert amd["pick:AMD:brief"]["text"] == "Night brief from 2026-09-10, evidence unchanged since: AMD faded into its band."
    assert amd["pick:AMD:brief"]["reused"] is True


def test_a_swing_or_claimed_pick_stays_on_the_d1_branch(world):
    for symbol in ("NVDA", "TSLA"):
        ids = pick_pack.build(symbol, now=NOW, paths=world).ids
        assert f"pick:{symbol}:branch" not in ids and f"pick:{symbol}:m5cell" not in ids
    assert pick_pack.is_m5_branch([("m5", "SHORT")], "SHORT", []) is True
    assert pick_pack.is_m5_branch([("m5", "SHORT")], "SHORT", [{"side": "SHORT"}]) is False, "a D1 claim wins"
    assert pick_pack.is_m5_branch([("m5", "SHORT"), ("swing", "SHORT")], "SHORT", []) is False
