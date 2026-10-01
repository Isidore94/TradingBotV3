"""P15b recaps pack: day recaps as cited rows; the recurrence table is computed (2+ sessions), never guessed."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import recaps_pack, registry  # noqa: E402

NOW = recaps_pack.FIXTURE_NOW  # Wed 2026-09-30 07:00 PT

GOLDEN = """## recaps_pack
[recap:asof] Recaps for 3 session(s), 2026-09-25 to 2026-09-29; 5 recurring issue(s) (a theme counts at 2+ sessions).
[recap:issues:missed:compressed] Vetoes sharing the reason 'compressed': 2 sessions (2026-09-28, 2026-09-25), first 2026-09-25
[recap:issues:stop:chas_extend_name] The same 'stop' lesson: 2 sessions (2026-09-28, 2026-09-25), first 2026-09-25; e.g. Chasing extended names, chased an extended name
[recap:issues:clue:volume_dry_up] Clue you marked 'volume dry up': 2 sessions (2026-09-29, 2026-09-28), first 2026-09-28; e.g. NVDA, AMD
[recap:issues:rule_broken:wait_for_confirmation] Rule not kept 'wait for confirmation': 2 sessions (2026-09-29, 2026-09-28), first 2026-09-28; e.g. no, partly
[recap:issues:wrong_reads] Reads graded wrong: 2 sessions (2026-09-29, 2026-09-28), first 2026-09-28
[recap:2026-09-29:card:did_well] 2026-09-29 report card did well: Did well: You liked 4 you did not trade (2026-09-29).
[recap:2026-09-29:card:missed] 2026-09-29 report card missed: Missed: You vetoed 30. 2 were real misses; 12 share the reason overhead.
[recap:2026-09-29:verdict:1] 2026-09-29 day review: you said "SPY tests the 50sma later this week" -> wrong
[recap:2026-09-29:reads] 2026-09-29 graded reads: pending 1, wrong 1
[recap:2026-09-29:clue:volume_dry_up] 2026-09-29 clue volume_dry_up on AMD M5 @ 150.0: it dried up before the drop
[recap:2026-09-29:rule_check:1] 2026-09-29 kept the rule (wait_for_confirmation)? partly
[recap:2026-09-29:answer:1] 2026-09-29 miss card ALL: real_miss: I passed and it ran
[recap:2026-09-29:env] 2026-09-29 environment: you corrected the auto label neutral_chop to bearish_weak
[recap:2026-09-29:lesson:1] 2026-09-29 lesson: keep: patience; mood 3
[recap:2026-09-28:card:did_well] 2026-09-28 report card did well: Did well: You liked 4 you did not trade (2026-09-28).
[recap:2026-09-28:card:missed] 2026-09-28 report card missed: Missed: You vetoed 30. 2 were real misses; 12 share the reason compressed.
[recap:2026-09-28:reads] 2026-09-28 graded reads: right 1, wrong 1
[recap:2026-09-28:rule_check:1] 2026-09-28 kept the rule (wait_for_confirmation)? no: jumped in early
[recap:2026-09-28:lesson:1] 2026-09-28 lesson: stop: chased an extended name; try: alerts
[recap:2026-09-28:clue:volume_dry_up] 2026-09-28 clue volume_dry_up on NVDA M5 @ 120.5
[recap:2026-09-25:card:did_well] 2026-09-25 report card did well: Did well: You liked 4 you did not trade (2026-09-25).
[recap:2026-09-25:card:missed] 2026-09-25 report card missed: Missed: You vetoed 30. 2 were real misses; 12 share the reason compressed.
[recap:2026-09-25:rule:wait_for_confirmation] 2026-09-25 rule for next session (wait_for_confirmation): Wait for the 5-min close
[recap:2026-09-25:lesson:1] 2026-09-25 lesson: keep: sizing; stop: Chasing extended names; mood 2"""


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(recaps_pack, "live_paths", lambda: pytest.fail("the pack reached for live paths"))
    return recaps_pack.write_fixture_world(tmp_path / "recaps")


def _rows(pack):
    return {row["id"]: row for row in pack.rows}


def test_registered_with_days_and_section():
    assert "recaps_pack" in registry.names()
    params = registry.modules()["recaps_pack"].SCHEMA["function"]["parameters"]["properties"]
    assert set(params) == {"days", "section"}
    assert params["section"]["enum"] == ["all", "cards", "words", "issues"]


def test_golden_newest_first_with_the_recurrence_table_leading(world):
    pack = recaps_pack.build(10, now=NOW, paths=world)
    assert pack.as_text() == GOLDEN
    assert len(pack.ids) == len(set(pack.ids))
    assert "how_fresh" not in pack.as_text(), "the machine's freshness line is not the trader's day"
    assert "old words" not in pack.as_text(), "a superseded recap row is folded away"


def test_a_theme_needs_two_sessions_to_be_an_issue(world):
    issues = {row["key"]: row for row in recaps_pack.build(10, "issues", now=NOW, paths=world).rows
              if row.get("kind") == "issue"}
    assert "answer:miss_real_miss" not in issues, "one real-miss answer is one session, not an issue"
    assert "env:corrected" not in issues and "missed:overhead" not in issues
    assert issues["rule_broken:wait_for_confirmation"]["sessions"] == ["2026-09-29", "2026-09-28"]
    assert all(row["count"] >= recaps_pack.ISSUE_MIN_SESSIONS for row in issues.values())
    one_day = recaps_pack.build(1, "issues", now=NOW, paths=world)
    assert one_day.ids == ("recap:asof", "recap:issues:none")


def test_sections_split_the_cards_from_his_own_words(world):
    cards = recaps_pack.build(10, "cards", now=NOW, paths=world)
    words = recaps_pack.build(10, "words", now=NOW, paths=world)
    assert all(":card:" in i or ":verdict:" in i or i.endswith(":reads") for i in cards.ids[1:])
    assert "recap:2026-09-29:env" in words.ids and not any(":card:" in i for i in words.ids)
    assert recaps_pack.build(10, "vibes", now=NOW, paths=world).ids == ()


def test_nothing_on_file_is_a_none_row():
    pack = recaps_pack.build(10, now=NOW, paths=recaps_pack.RecapPaths())
    assert pack.ids == ("recap:none",) and "unknown" in pack.as_text()


def test_all_reads_every_session_and_a_future_session_is_never_read(world):
    folder = Path(world.day_review) / "sessions" / "2026-10-01"
    folder.mkdir(parents=True)
    (folder / "pack.json").write_text(json.dumps({"report_card": {"lines": [
        {"key": "did_well", "text": "FUTURE"}]}}), encoding="utf-8")
    pack = recaps_pack.build("all", now=NOW, paths=world)
    assert "FUTURE" not in pack.as_text() and "recap:2026-09-25:card:missed" in pack.ids


def test_read_only(world):
    files = [p for p in Path(world.events).parent.rglob("*") if p.is_file()]
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files}
    recaps_pack.build("all", now=NOW, paths=world)
    recaps_pack.issue_rows(world, today=date(2026, 9, 30))
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files} == before
    assert sorted(p for p in Path(world.events).parent.rglob("*") if p.is_file()) == sorted(files)


def test_keywords_are_a_fixed_stem():
    assert recaps_pack.keywords("Chasing extended names") == recaps_pack.keywords("chased an extended name")
    # Review advisory 2: the most frequent words in text order, not the first four alphabetically.
    one = "buying dips, chasing adds into losers, more losers"
    two = "buying dips, chasing adds into winners, more winners"
    assert recaps_pack.keywords(one) != recaps_pack.keywords(two), "two different long lines, two keys"
    assert recaps_pack.keywords(one) == recaps_pack.keywords(one)
    assert recaps_pack.keywords(one) == "buy_dip_chas_loser"


def test_a_tagged_stop_lesson_is_keyed_by_its_tag(world):
    import json

    with open(world.events, "a", encoding="utf-8") as handle:
        for n, day in ((20, "2026-09-28"), (21, "2026-09-29")):
            handle.write(json.dumps({"schema": "day_recap_event_v1", "id": f"rc-{n}", "kind": "lesson",
                                     "session_date": day, "recorded_at": f"{day}T18:00:00-07:00", "supersedes": "",
                                     "keep": "", "stop": f"different words {n}", "try": "", "mood": None,
                                     "tag": "respect_stop"}) + "\n")
    keys = [row["key"] for row in recaps_pack.build(10, "issues", now=NOW, paths=world).rows if row.get("key")]
    assert "stop:respect_stop" in keys
    assert recaps_pack.keywords("the a of") == ""
    assert recaps_pack.slug("Wait For Confirmation!") == "wait_for_confirmation"


def test_issue_rows_and_the_markdown(world):
    rows = recaps_pack.issue_rows(world, today=date(2026, 9, 30), days=10)
    assert [row["id"] for row in rows][:2] == ["recap:issues:missed:compressed", "recap:issues:stop:chas_extend_name"]
    text = recaps_pack.issues_markdown(rows)
    assert "[recap:issues:clue:volume_dry_up]" in text and "never rules" in text
    assert recaps_pack.issues_markdown([]) == ""


def test_embed_rows_are_the_session_rows_only(world):
    pack = recaps_pack.build("all", now=NOW, paths=world)
    rows = recaps_pack.embed_rows(pack)
    assert rows and all(text.startswith("[recap:2026-") for _ref, text in rows)
    assert len(rows) == len([i for i in pack.ids if i.startswith("recap:2026-")])
