"""P14 earnings pack: own + nearest peer earnings for many names, one row each, never a guess. Read-only."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import earnings_pack, pick_pack, registry  # noqa: E402

NOW = pick_pack.FIXTURE_NOW

GOLDEN = """## earnings_pack
[earn:asof] Earnings for 3 name(s) as of market date 2026-09-29: own next report and nearest industry peer within 14 days
[earn:NVDA] NVDA: no own report within 14 days (next 2026-11-19); nearest peer AVGO Wed 2026-09-30, tomorrow
[earn:TSLA] TSLA: no own report within 14 days (next 2026-10-21); no peer reports within 14 days
[earn:ZZZ] ZZZ: own report unknown (no date in the calendar); peers unknown (no industry)"""


def test_the_golden_fixture():
    assert earnings_pack.fixture().as_text() == GOLDEN
    assert "earnings_pack" in registry.names()


def test_a_name_reporting_inside_14_days_says_so(tmp_path):
    paths = pick_pack.write_fixture_world(tmp_path)
    rows = {row["id"]: row["text"] for row in earnings_pack.build("AVGO, amd $MU", now=NOW, paths=paths).rows}
    assert rows["earn:AVGO"].startswith("AVGO: REPORTS Wed 2026-09-30, tomorrow")
    assert rows["earn:AMD"].startswith("AMD: no own report within 14 days (next 2026-11-03)")
    assert "nearest peer AVGO Wed 2026-09-30, tomorrow" in rows["earn:AMD"], "a string of tickers is read too"
    assert rows["earn:MU"].startswith("MU: own report unknown"), "MU reported 5 days ago: nothing upcoming"


def test_at_most_forty_names_and_the_rest_are_counted(tmp_path):
    paths = pick_pack.write_fixture_world(tmp_path)
    names = [f"Z{index:02d}" for index in range(45)]
    pack = earnings_pack.build(names, now=NOW, paths=paths)
    assert len(pack.rows) == 42 and "5 more name(s) not listed" in pack.rows[0]["text"]
    more = pack.rows[1]
    assert more["id"] == "earn:more" and more["text"].endswith(": Z40, Z41, Z42, Z43, Z44"), "named, second, never cut"
    held = earnings_pack.build(names, book=["Z44", "Z43"], now=NOW, paths=paths)
    assert held.ids[2:4] == ("earn:Z44", "earn:Z43") and "earn:Z38" not in held.ids, "the book first, the tail cut"


def test_an_unreadable_calendar_is_unknown_for_every_name_never_none(tmp_path):
    paths = replace(pick_pack.write_fixture_world(tmp_path), earnings_history=tmp_path / "missing.json")
    pack = earnings_pack.build(["NVDA", "TSLA"], now=NOW, paths=paths)
    texts = [row["text"] for row in pack.rows[1:]]
    assert texts and all("earnings unknown" in text for text in texts)


def test_no_tickers_is_an_empty_pack():
    pack = earnings_pack.build([], now=NOW)
    assert pack.ids == () and "needs tickers" in pack.as_text()
