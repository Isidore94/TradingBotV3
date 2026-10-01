"""P15b fundamentals pack: the pasted morning brief as cited rows; the latest paste wins; none is a none row."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import fundamentals_pack, registry  # noqa: E402

NOW = fundamentals_pack.FIXTURE_NOW  # Wed 2026-09-30 07:00 PT

GOLDEN = """## fundamentals_pack
[fund:2026-09-30:asof] Morning brief (brief for 2026-09-30): pasted 2026-09-30 05:50 PT, source claude, written 2026-09-30T05:45:00-07:00; the trader's outside commentary, not his own view.
[fund:2026-09-30:bottom:1] Bottom line: Softer core PCE weakens the case for an October hike. Rates lead today; oil is the risk to any rally.
[fund:2026-09-30:signal:1] Ranked signal 1: 10Y yield
[fund:2026-09-30:signal:2] Ranked signal 2: Brent
[fund:2026-09-30:signal:3] Ranked signal 3: NVDA/SMH
[fund:2026-09-30:signal:4] Ranked signal 4: DXY
[fund:2026-09-30:bull:1] Playbook, bullish: The bullish continuation needs the 10-year back below 5.25% and SPY holding its opening range.
[fund:2026-09-30:bear:1] Playbook, bearish: The bearish reversal: a new 10-year high above 5.30% with oil up 2% sends small caps lower.
[fund:2026-09-30:turb:1] Turbulence: ~6/10 today, 7–8/10 into Friday's payrolls.
[fund:2026-09-30:event:1] Upcoming: 2026-09-30 09:45 ET Chicago PMI (Sep)
[fund:2026-09-30:event:2] Upcoming: 2026-10-01 10:00 ET ISM Manufacturing PMI (Sep)
[fund:2026-09-30:event:3] Upcoming: 2026-10-02 08:30 ET Employment Situation / Nonfarm Payrolls (Sep)
[fund:2026-09-30:p1] Morning Brief — September 30, 2026: As of 8:45 ET.
[fund:2026-09-30:p2] The setup: - Rates: the 10-year is near 5.28%, its highest since 2007. - Oil: WTI about $90; Hormuz headlines keep a bid under crude. - Dollar: the DXY eased to 101 after the data.
[fund:2026-09-30:p3] 10Y yield → Brent → NVDA/SMH → DXY
[fund:2026-09-30:p4] Intraday playbook: The bullish continuation needs the 10-year back below 5.25% and SPY holding its opening range.
[fund:2026-09-30:p5] The bearish reversal: a new 10-year high above 5.30% with oil up 2% sends small caps lower.
[fund:2026-09-30:p6] Turbulence: ~6/10 today, 7–8/10 into Friday's payrolls.
[fund:2026-09-30:p7] Bottom line: Softer core PCE weakens the case for an October hike. Rates lead today; oil is the risk to any rally.
[fund:2026-09-30:p8] NEXT 7 DAYS — ECONOMIC CALENDAR: 2026-09-30 | 06:45 PT / 09:45 ET | Chicago PMI (Sep) 2026-10-01 | 07:00 PT / 10:00 ET | ISM Manufacturing PMI (Sep) 2026-10-02 | 05:30 PT / 08:30 ET | Employment Situation / Nonfarm Payrolls (Sep)"""


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(fundamentals_pack, "live_paths", lambda: pytest.fail("the pack reached for live paths"))
    return fundamentals_pack.write_fixture_world(tmp_path / "fund")


def test_registered_with_day_and_section():
    assert "fundamentals_pack" in registry.names()
    params = registry.modules()["fundamentals_pack"].SCHEMA["function"]["parameters"]["properties"]
    assert set(params) == {"day", "section"}
    assert params["section"]["enum"] == ["all", "bottom_line", "signals", "playbook", "events", "text", "compact"]


def test_golden_the_latest_paste_wins_and_every_part_is_a_cited_row(world):
    pack = fundamentals_pack.build("today", now=NOW, paths=world)
    assert pack.as_text() == GOLDEN
    assert "SUPERSEDED" not in pack.as_text(), "a re-paste supersedes the first paste"
    assert len(pack.ids) == len(set(pack.ids))


def test_a_session_with_no_paste_names_the_last_brief_with_its_own_date(world):
    pack = fundamentals_pack.build("2026-09-29", now=NOW, paths=world)
    rows = {row["id"]: row for row in pack.rows}
    assert pack.ids[0] == "fund:2026-09-29:none"
    assert "last brief: 2026-09-28" in rows["fund:2026-09-28:asof"]["text"]
    # The Claude-style brief has bold headings, no Markdown: the bottom line says which heading it came from.
    assert rows["fund:2026-09-28:bottom:1"]["text"].startswith("Bottom line (from 'Why it matters'): Yields")
    assert rows["fund:2026-09-28:play:2"]["text"].endswith("Base case: chop into the afternoon.")


def test_nothing_on_or_before_the_session_is_one_none_row_never_quiet(world):
    pack = fundamentals_pack.build("2026-09-25", now=NOW, paths=world)
    assert pack.ids == ("fund:2026-09-25:none",)
    assert "unknown, not quiet" in pack.as_text()
    empty = fundamentals_pack.build("today", now=NOW, paths=fundamentals_pack.FundPaths())
    assert empty.ids == ("fund:2026-09-30:none",)


def test_compact_is_the_bottom_line_and_playbook_in_at_most_eight_rows(world):
    pack = fundamentals_pack.build("today", "compact", now=NOW, paths=world)
    assert pack.ids == ("fund:2026-09-30:asof", "fund:2026-09-30:bottom:1", "fund:2026-09-30:bull:1",
                        "fund:2026-09-30:bear:1", "fund:2026-09-30:turb:1")
    assert fundamentals_pack.bottom_line_sentence(fundamentals_pack.build("today", now=NOW, paths=world)) == (
        "Softer core PCE weakens the case for an October hike.")


def test_one_section_and_a_bad_section(world):
    events = fundamentals_pack.build("today", "events", now=NOW, paths=world)
    assert [i for i in events.ids if ":event:" in i] == [f"fund:2026-09-30:event:{n}" for n in (1, 2, 3)]
    assert events.ids[0] == "fund:2026-09-30:asof"
    bad = fundamentals_pack.build("today", "vibes", now=NOW, paths=world)
    assert bad.ids == () and "not 'vibes'" in bad.empty_text


def test_a_long_brief_is_cut_into_at_most_forty_paragraphs_of_four_hundred_chars(tmp_path):
    long_text = "\n\n".join(f"Paragraph {n}. " + "word " * 150 for n in range(60))
    chunks = fundamentals_pack._chunks(long_text)
    rows = fundamentals_pack.brief_rows({"_session": "2026-09-30", "text": long_text})["text"]
    assert len(chunks) > 40 and len(rows) == 40
    assert all(len(row["text"]) <= 400 for row in rows)
    assert [row["id"] for row in rows] == [f"fund:2026-09-30:p{n}" for n in range(1, 41)]


def test_read_only_and_tz_aware(world):
    files = sorted(Path(world.ledger_dir).glob("*.jsonl")) + [Path(world.theses)]
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files}
    fundamentals_pack.build("today", now=NOW, paths=world)
    assert {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files} == before
    assert not any(name for name in os.listdir(world.ledger_dir) if not name.endswith(".jsonl"))
    # The same instant seen from a naive-free UTC clock: 12:50 UTC is 05:50 PT.
    late = fundamentals_pack.build("today", now=datetime(2026, 10, 1, 3, 0, tzinfo=timezone.utc), paths=world)
    assert late.ids[0] == "fund:2026-09-30:asof", "23:00 ET is still the 30th in New York"


def test_today_is_the_traders_own_date_and_a_weekend_reads_ahead_to_monday():
    assert fundamentals_pack.market_day(NOW).isoformat() == "2026-09-30"
    late = datetime(2026, 10, 1, 4, 30, tzinfo=timezone.utc)  # Wed 21:30 PT = Thu 00:30 ET
    assert fundamentals_pack.market_day(late).isoformat() == "2026-09-30", "his evening is still his day"
    assert fundamentals_pack.market_day(datetime(2026, 10, 3, 16, 0, tzinfo=timezone.utc)).isoformat() == "2026-10-05"
    assert not hasattr(fundamentals_pack, "paste_session"), "the filing rule lives in ONE place, the journal service"


def test_embed_rows_are_one_per_paragraph_and_stable_per_entry(world):
    pack = fundamentals_pack.build("today", now=NOW, paths=world)
    rows = fundamentals_pack.embed_rows(pack)
    assert len(rows) == len([i for i in pack.ids if ":p" in i.split("2026-09-30")[1]])
    assert rows == fundamentals_pack.embed_rows(fundamentals_pack.build("today", now=NOW, paths=world))
    assert rows[0][1].startswith("[fund:2026-09-30:p1] ")
