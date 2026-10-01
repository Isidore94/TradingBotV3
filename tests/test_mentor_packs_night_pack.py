"""P15a night pack: golden fixture, ids embed the night's own source ids, missing is a none row, read-only."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import night_pack, registry  # noqa: E402

NOW = night_pack.FIXTURE_NOW  # Wed 2026-09-30 07:00 PT

GOLDEN = """## night_pack
[night:asof] Night reads as of market date 2026-09-30: day review 2026-09-29; ideas 2026-09-29, 2026-09-25; miss contrast 2026-09-29; prediction contrast 2026-09-29; week review 2026-W39; market story 2026-09-29; daily digest 2026-09-29
[night:day_review:2026-09-29:0] Day review 2026-09-29: Choppy session testing support. (graded 2 of 3 reads) (src: day_review:2026-09-29)
[night:day_review:2026-09-29:1] You said "we are compressed, an H- below holds us up" -> right (src: said:mj-2026-09-29-f3f8:prediction; read:rd-5fa9)
[night:day_review:2026-09-29:2] You said "SPY tests the 50sma later this week" -> wrong (src: said:mj-2026-09-29-a81d:prediction; read:rd-03cd)
[night:ideas:2026-09-29:1] Idea (process, measured by report_card_did_well_rate, seen 2x): Liked names you did not trade ran well; ask why. (src: 2026-09-28/report_card:did_well)
[night:ideas:2026-09-29:2] Idea (program, seen 1x): Measure the congruence checks. (src: 2026-09-28/report_card:congruence)
[night:ideas:2026-09-25:1] Idea (process, seen 1x): An old idea. (src: 2026-09-25/report_card:did_well)
[night:miss:2026-09-29:0] Miss contrast 2026-09-29 over 20 sessions: observational, not causal: top 2 of 3 group(s). (src: miss_contrast-2026-09-29)
[night:miss:2026-09-29:1] Miss contrast 'incoming_trendline': real-miss rate 21% of 90 measured (n=105, 19 misses); top feature atr_pct AUC 0.66 (real_run n=19 median 3.1 vs dud n=41 median 2.2) (src: miss_contrast-2026-09-29:group:incoming_trendline)
[night:miss:2026-09-29:2] Miss contrast 'overhead_horizontal': real-miss rate 47% of 30 measured (n=41, 14 misses) (src: miss_contrast-2026-09-29:group:overhead_horizontal)
[night:prediction:2026-09-29:0] Prediction contrast 2026-09-29: observational, not causal: 41 clicked read(s), right against wrong. (src: prediction_contrast-2026-09-29)
[night:prediction:2026-09-29:1] Your reads, Rest of day: right 19, wrong 1, flat 5, pending 0; right rate 76% (LB 0.57, n=25, under the floor) (src: prediction_contrast-2026-09-29:horizon:rest_of_day)
[night:week:2026-W39:0] Week review 2026-W39: Choppy week with a slight downward bias. (src: week_review:2026-W39)
[night:week:2026-W39:1] Week reads: right 14, wrong 2, unresolved 14 (src: week_review:2026-W39)
[night:week:2026-W39:2] Tendency: You call bounces early in a bear channel (n=6) (src: 2026-09-24/read:rd-1dbc)
[night:week:2026-W39:3] Watch next week: The 50 SMA on SPY (src: week_review:2026-W39)
[night:story:2026-09-29:0] Market story 2026-09-29: The market is in a bear channel, lower highs, day 2. (src: market_story:2026-09-29; rollup:weekly:2026-W40)
[night:story:2026-09-29:1] Changed: SPY is 0.06% above its 20-day SMA. (src: market_story:2026-09-29)
[night:story:2026-09-29:2] Changed: Monthly rollups are incomplete. (src: market_story:2026-09-29)
[night:digest:2026-09-29:0] what is working: The D1 scan found several high-tier shorts. (src: scan.tier_list in narration/2026/2026-09-29.json)
[night:digest:2026-09-29:1] what is not working: M5 alerts closed mixed, mean close_r -0.05. (src: outcomes.intraday_finals in narration/2026/2026-09-29.json)"""


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(night_pack, "live_paths", lambda: pytest.fail("the pack reached for live paths"))
    return night_pack.write_fixture_world(tmp_path / "night")


def _rows(pack):
    return {row["id"]: row for row in pack.rows}


def test_registered_as_a_tool_with_section_and_days():
    assert "night_pack" in registry.names()
    params = registry.modules()["night_pack"].SCHEMA["function"]["parameters"]["properties"]
    assert set(params) == {"section", "days"}
    assert params["section"]["enum"] == ["all", *night_pack.SECTIONS]


def test_golden(world):
    assert night_pack.build(now=NOW, paths=world).as_text() == GOLDEN


def test_ids_are_unique_stable_and_every_row_names_its_origin(world):
    first = night_pack.build(now=NOW, paths=world)
    again = night_pack.build(now=NOW, paths=world)
    assert first.ids == again.ids and len(first.ids) == len(set(first.ids))
    for row in first.rows[1:]:
        assert row["id"].startswith(f"night:{row['section']}:{row['date']}:")
        assert row["src"] and row["text"].endswith(f"(src: {row['src']})")


def test_a_day_review_row_keeps_the_original_source_and_evidence_ids(world):
    row = _rows(night_pack.build("day_review", now=NOW, paths=world))["night:day_review:2026-09-29:2"]
    assert row["source_id"] == "said:mj-2026-09-29-a81d:prediction" and row["evidence_id"] == "read:rd-03cd"
    assert row["verdict"] == "wrong"


def test_dismissed_ideas_are_left_out_and_at_most_five(world):
    rows = night_pack.build("ideas", now=NOW, paths=world).rows
    assert all("Dismissed" not in row["text"] for row in rows)
    lines = [json.dumps({"idea_id": f"idea:2026-09-29:{i:03d}", "session_date": "2026-09-29", "text": f"idea {i}",
                         "evidence": [f"2026-09-29/x:{i}"]}) for i in range(9)]
    world.ideas.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert len(night_pack.build("ideas", now=NOW, paths=world).rows) == 1 + night_pack.MAX_IDEAS


def test_contrast_rows_are_capped_at_five_each(world):
    groups = [{"name": f"g{i}", "reportable": True, "rate": 0.1 * i, "measured": 40, "n": 50} for i in range(9)]
    (world.digests / "miss_contrast-2026-09-29.json").write_text(
        json.dumps({"statement": "s", "groups": groups, "leaders": []}), encoding="utf-8")
    rows = [row for row in night_pack.build("contrast", now=NOW, paths=world).rows if row.get("section") == "miss"]
    assert len(rows) == night_pack.MAX_CONTRAST_ROWS
    assert rows[1]["group"] == "g8", "highest real-miss rate first when no leaders are named"


@pytest.mark.parametrize(("when", "week"), [
    (datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc), "2026-W39"),  # Wednesday: last week's
    (datetime(2026, 10, 2, 14, 0, tzinfo=timezone.utc), "2026-W40"),  # Friday: this week's
])
def test_the_week_review_is_last_weeks_early_in_the_week_and_this_weeks_later(world, when, week):
    (world.day_review / "week" / "2026-W40.json").write_text(json.dumps({"narration": {"headline": "This week."}}),
                                                            encoding="utf-8")
    assert f"night:week:{week}:0" in night_pack.build("week", now=when, paths=world).ids


def test_a_missing_artifact_is_one_none_row_and_the_asof_says_none(tmp_path):
    pack = night_pack.build(now=NOW, paths=night_pack.NightPaths(digests=tmp_path / "nothing"))
    assert [row["id"] for row in pack.rows[1:]] == [f"night:{kind}:none" for kind in (
        "day_review", "ideas", "miss", "prediction", "week", "story", "digest")]
    assert "day review none" in pack.rows[0]["text"] and "daily digest none" in pack.rows[0]["text"]
    assert all("unknown" in row["text"] for row in pack.rows[1:])


def test_stale_reads_show_their_age(world):
    later = datetime(2026, 10, 5, 14, 0, tzinfo=timezone.utc)  # Monday, six days on
    assert "day review 2026-09-29, 6 days old" in night_pack.build("day_review", now=later, paths=world).rows[0]["text"]


def test_files_dated_after_the_market_date_are_never_read(world):
    (world.day_review / "narration" / "2026-10-01.json").write_text(
        json.dumps({"narration": {"headline": "future", "were_you_right": []}}), encoding="utf-8")
    assert "night:day_review:2026-10-01:0" not in night_pack.build("day_review", now=NOW, paths=world).ids


def test_a_naive_clock_is_read_as_local_and_the_pack_is_still_built(world):
    pack = night_pack.build(now=datetime(2026, 9, 30, 9, 0), paths=world)
    assert pack.rows[0]["id"] == "night:asof" and len(pack.rows) > 1


def test_an_unknown_section_is_empty_not_a_raise(world):
    pack = night_pack.build("vibes", now=NOW, paths=world)
    assert not pack.rows and "vibes" in pack.empty_text


def test_days_reads_more_nights(world):
    one = night_pack.build("day_review", days=1, now=NOW, paths=world).ids
    two = night_pack.build("day_review", days=2, now=NOW, paths=world).ids
    assert "night:day_review:2026-09-28:0" in two and "night:day_review:2026-09-28:0" not in one


def test_digest_facts_are_headline_values_with_their_pointers(world):
    rows, day = night_pack.digest_fact_rows(world, "2026-09-29")
    assert day == "2026-09-29" and 0 < len(rows) <= night_pack.MAX_FACT_ROWS
    assert rows[0]["text"].startswith("Daily digest 2026-09-29: settled outcomes mean close_r -0.0511 (n=57)")
    assert rows[0]["src"] == "outcomes.intraday_finals in facts/2026/2026-09-29.json"


def test_latest_briefs_take_the_newest_briefed_row_per_symbol(world):
    briefs = night_pack.latest_briefs(world.briefs, NOW.date())
    assert briefs["NVDA"]["session"] == "2026-09-29" and briefs["TSLA"]["session"] == "2026-09-28"
    assert all(not line.startswith("[system]") for line in briefs["NVDA"]["lines"])
    assert len(briefs["NVDA"]["lines"]) <= night_pack.BRIEF_LINES
    rows = night_pack.brief_rows(world.briefs, NOW.date())
    assert [row["id"] for row in rows] == ["brief:NVDA:2026-09-29", "brief:TSLA:2026-09-28"]


def test_the_pack_writes_nothing(world):
    root = Path(world.digests).parent

    def snapshot():
        return sorted((str(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file())

    before = snapshot()
    night_pack.build(now=NOW, paths=world)
    night_pack.latest_briefs(world.briefs, NOW.date())
    assert snapshot() == before


def test_one_unreadable_artifact_never_blanks_the_others(world, monkeypatch):
    def broken(*_a, **_k):
        raise PermissionError("locked")

    monkeypatch.setitem(night_pack._READERS, "week", broken)
    rows = _rows(night_pack.build(now=NOW, paths=world))
    assert rows["night:week:none"]["kind"] == "unknown" and "PermissionError" in rows["night:week:none"]["text"]
    assert "night:day_review:2026-09-29:1" in rows


def test_a_superseding_sibling_wins_and_is_the_file_cited(world):
    """Review blocker: the writers never edit a pack (D6); the correction is `<name>.1.json`."""
    night_pack.write_fixture_corrections(world)
    rows = _rows(night_pack.build(now=NOW, paths=world))
    assert rows["night:miss:2026-09-29:0"]["text"].startswith("Miss contrast 2026-09-29 over 20 sessions: CORRECTED")
    assert rows["night:miss:2026-09-29:1"]["src"] == "miss_contrast-2026-09-29.1:group:corrected_group"
    assert "CORRECTED" in rows["night:prediction:2026-09-29:0"]["text"]
    assert rows["night:prediction:2026-09-29:0"]["src"] == "prediction_contrast-2026-09-29.1"
    assert rows["night:digest:2026-09-29:0"]["text"].startswith("what is working: CORRECTED statement.")
    assert rows["night:digest:2026-09-29:0"]["file"] == "narration/2026/2026-09-29.1.json"
    facts, _day = night_pack.digest_fact_rows(world, "2026-09-29")
    assert "close_r 9.99 (n=99)" in facts[0]["text"] and facts[0]["file"] == "facts/2026/2026-09-29.1.json"
    texts = " ".join(row["text"] for row in night_pack.build(now=NOW, paths=world).rows)
    assert "high-tier shorts" not in texts and "incoming_trendline" not in texts, "the corrected files are not read"
