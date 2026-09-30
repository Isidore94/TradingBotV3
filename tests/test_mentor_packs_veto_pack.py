"""Trade Mentor veto pack: golden fixture, the challenge rule (n >= 30 and LB above the side
baseline), stable unique ids, tz-aware stamps, point-in-time outcomes, read-only, no live path."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import registry, veto_pack  # noqa: E402

NOW = veto_pack.FIXTURE_NOW

GOLDEN = """## veto_pack
[veto:2026-09-29:AAA:1] VETO AAA LONG D1: Compressed (compressed); setup avwap_breakout; at 2026-09-29T09:05:00-07:00; note: coiled under the 50
[veto:2026-09-29:AAA:1:slice] Slice avwap_breakout LONG vetoed for compressed (past vetoes, 5-session D1 outcomes known by 2026-09-29): n=40, wins 34 (85%), LB=0.71 vs the LONG baseline LB=0.45 (n=250): clears the baseline, candidate challenge; veto cohort side return 1d +1.00% (n=40), 3d +2.00% (n=40), 5d +3.00% (n=40), 10d +4.00% (n=40); weeks=9
[veto:2026-09-29:BBB:1] VETO BBB SHORT D1: Too extended below base (too_extended_from_base); setup avwap_band_bounce; at 2026-09-29T09:10:00-07:00
[veto:2026-09-29:BBB:1:slice] Slice avwap_band_bounce SHORT vetoed for too_extended_from_base (past vetoes, 5-session D1 outcomes known by 2026-09-29): n=35, wins 15 (43%), LB=0.28 vs the SHORT baseline LB=0.43 (n=235): does not clear the baseline, no challenge; veto cohort side return 1d too few (n=0), 3d too few (n=0), 5d too few (n=0), 10d too few (n=0); weeks=9
[veto:2026-09-29:CCC:1] VETO CCC LONG D1: Volume dry (volume_dry); setup top_pattern; at 2026-09-29T09:20:00-07:00
[veto:2026-09-29:CCC:1:slice] Slice top_pattern LONG vetoed for volume_dry (past vetoes, 5-session D1 outcomes known by 2026-09-29): too few (n=10, floor 30); never a challenge; veto cohort side return 1d too few (n=0), 3d too few (n=0), 5d too few (n=0), 10d too few (n=0); weeks=9
[veto:2026-09-29:DDD:1] PASS DDD LONG M5: Low rvol; setup unknown; at 2026-09-29T10:00:00-07:00
[veto:2026-09-29:DDD:1:slice] A pass (the trader liked the day trade and passed on one issue): no D1 slice, never a challenge
[veto:2026-09-29:EEE:1] VETO EEE LONG D1: Compressed (compressed); setup unknown; at 2026-09-29T11:00:00-07:00
[veto:2026-09-29:EEE:1:slice] Setup unknown (no D1 scan row for EEE LONG within 4 days up to 2026-09-29): no slice, never a challenge
[veto:2026-09-29:AAA:2] VETO AAA SHORT D1: no reason code (not today) (uncoded); setup unknown; at 2026-09-29T12:30:00-07:00
[veto:2026-09-29:AAA:2:slice] Setup unknown (no D1 scan row for AAA SHORT within 4 days up to 2026-09-29): no slice, never a challenge
[veto:2026-09-29:summary] Session 2026-09-29: 5 vetoes, 1 pass; 1 candidate challenge (a challenge needs n>=30 and an LB above the side's baseline LB)
[veto:2026-09-29:weeks] Decision data: weeks=9 (2026-08-03 to 2026-09-29); a thin record, read every n"""


@pytest.fixture()
def world(tmp_path, monkeypatch):
    # Any reach for a live path fails the test.
    monkeypatch.setattr(veto_pack, "live_paths", lambda: pytest.fail("the pack reached for live paths"))
    return veto_pack.write_fixture_world(tmp_path / "desk")


def _slices(pack):
    return {row["id"]: row for row in pack.rows if row.get("kind") == "slice"}


def test_registered_as_a_tool_with_an_optional_date():
    assert "veto_pack" in registry.names()
    params = veto_pack.SCHEMA["function"]["parameters"]
    assert "date" in params["properties"] and params["required"] == []


def test_golden(world):
    assert veto_pack.build(now=NOW, paths=world).as_text() == GOLDEN


def test_the_floor_is_evidence_stats(world):
    from evidence_stats import MIN_REPORTABLE_N

    assert veto_pack.min_reportable_n() == MIN_REPORTABLE_N == 30


def test_only_a_slice_above_the_floor_that_clears_the_baseline_is_a_challenge(world):
    pack = veto_pack.build(now=NOW, paths=world)
    slices = _slices(pack)
    assert slices["veto:2026-09-29:AAA:1:slice"]["challenge"] is True
    assert slices["veto:2026-09-29:BBB:1:slice"]["challenge"] is False, "n>=30 but no edge over the baseline"
    assert slices["veto:2026-09-29:CCC:1:slice"]["challenge"] is False, "below the floor"
    assert all(not row["challenge"] for key, row in slices.items() if "AAA:1" not in key)
    assert [item["veto"]["symbol"] for item in veto_pack.candidates(pack)] == ["AAA"]
    aaa = slices["veto:2026-09-29:AAA:1:slice"]
    assert (aaa["n"], aaa["wins"], aaa["baseline_n"]) == (40, 34, 250)
    assert aaa["lb"] > aaa["baseline_lb"]


def test_every_challenge_and_every_measured_slice_says_n_and_lb_and_weeks(world):
    for row in _slices(veto_pack.build(now=NOW, paths=world)).values():
        if row.get("n", 0) >= 30:
            assert f"n={row['n']}" in row["text"] and "LB=" in row["text"] and "weeks=9" in row["text"]
        elif "n" in row:
            assert f"too few (n={row['n']}" in row["text"]


def test_the_baseline_is_every_clean_row_of_the_side_vetoed_or_not(world):
    lines = world.tier_outcomes.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not line.startswith("BL")]
    world.tier_outcomes.write_text("\n".join(kept) + "\n", encoding="utf-8")
    pack = veto_pack.build(now=NOW, paths=world)
    # LONG baseline is now the 50 vetoed rows only (n>=30): still measured.
    assert _slices(pack)["veto:2026-09-29:AAA:1:slice"]["baseline_n"] == 50


def test_ids_are_unique_stable_and_prefixed(world):
    first = veto_pack.build(now=NOW, paths=world)
    again = veto_pack.build(now=NOW, paths=world)
    assert first.ids == again.ids and len(first.ids) == len(set(first.ids))
    assert all(row_id.startswith("veto:2026-09-29:") for row_id in first.ids)
    assert veto_pack.pack_hash(first) == veto_pack.pack_hash(again)


def test_stamps_are_tz_aware(world):
    for row in veto_pack.build(now=NOW, paths=world).rows:
        if row.get("at"):
            assert datetime.fromisoformat(row["at"]).tzinfo is not None


def test_an_outcome_not_known_by_the_session_is_not_counted(world):
    # A veto slice row whose 5-session outcome lands after the session is invisible that day.
    lines = world.tier_outcomes.read_text(encoding="utf-8").splitlines()
    lines = [line.replace(",2026-08-10,5,", ",2026-09-30,5,") if line.startswith("L00,") else line for line in lines]
    world.tier_outcomes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pack = veto_pack.build(now=NOW, paths=world)
    assert _slices(pack)["veto:2026-09-29:AAA:1:slice"]["n"] == 39


def test_an_explicit_date_reads_that_session(world):
    pack = veto_pack.build("2026-08-03", now=NOW, paths=world)
    assert pack.ids[0].startswith("veto:2026-08-03:")
    summary = next(row for row in pack.rows if row["kind"] == "summary")
    assert summary["vetoes"] > 0 and summary["challenges"] == 0, "no history before the first session"


def test_a_session_with_nothing_still_has_a_summary_and_weeks(world):
    pack = veto_pack.build("2026-09-28", now=NOW, paths=world)
    assert pack.ids == ("veto:2026-09-28:summary", "veto:2026-09-28:weeks")
    assert "0 vetoes, 0 passes; 0 candidate challenges" in pack.as_text()


def test_the_default_session_is_the_last_one_before_today():
    assert veto_pack.target_session(now=NOW).isoformat() == "2026-09-29"
    monday = datetime(2026, 9, 28, 13, 45, tzinfo=NOW.tzinfo)
    assert veto_pack.target_session(now=monday).isoformat() == "2026-09-25", "Monday reads Friday"


def test_a_bad_date_is_an_empty_pack(world):
    pack = veto_pack.build("yesterday-ish", now=NOW, paths=world)
    assert pack.ids == () and "needs a date" in pack.as_text()


def test_missing_files_are_unknown_not_a_raise(world, tmp_path):
    gone = replace(world, tier_outcomes=tmp_path / "nope.csv", tier_list=tmp_path / "nope2.csv",
                   veto_outcomes=tmp_path / "nope3.csv")
    pack = veto_pack.build(now=NOW, paths=gone)
    assert "tier outcomes unknown" in pack.as_text()
    assert not veto_pack.candidates(pack)


def test_the_pack_writes_nothing(world):
    root = world.annotations.parent
    before = {path: path.stat().st_mtime_ns for path in root.iterdir()}
    veto_pack.build(now=NOW, paths=world)
    assert {path: path.stat().st_mtime_ns for path in root.iterdir()} == before


def test_every_path_in_the_fixture_is_under_the_fixture_root(world):
    root = world.annotations.parent
    for value in (world.annotations, world.tier_outcomes, world.tier_list, world.veto_outcomes):
        assert Path(value).parent == root


def test_the_pack_imports_no_ui_and_constructs_no_store():
    source = (SCRIPTS_DIR / "mentor_packs" / "veto_pack.py").read_text(encoding="utf-8")
    reader = (SCRIPTS_DIR / "annotations_reader.py").read_text(encoding="utf-8")
    for text in (source, reader):
        assert "from ui" not in text and "import ui" not in text and "PySide6" not in text
        assert "FocusPickStore(" not in text and "JournalStore(" not in text


def test_the_store_still_exports_the_lifted_session_seam():
    import annotations_reader
    from ui.annotations import store

    assert store.row_decision_session is annotations_reader.row_decision_session
    assert store.DECISION_SESSION_RULE == annotations_reader.DECISION_SESSION_RULE


def test_a_torn_line_and_other_event_types_are_ignored(world):
    text = veto_pack.build(now=NOW, paths=world).as_text()
    assert "FFF" not in text and "torn" not in text
    assert json.loads(world.annotations.read_text(encoding="utf-8").splitlines()[0])["event_type"] == "veto"
