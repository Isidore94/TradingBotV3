"""S17 half 2 and S16 item 4: the AI reads the regime table; the story opens in the regime.

* `regime_read`: one grounded paragraph, every timeframe word, date, number and
  regime word checked against the table; a breach rejects the whole read, the
  slot is `degraded` and the last verified read stays byte-identical.
* the market story opens with the trader's regime and its day count, then the
  trader's own notes in that regime, then the structure facts, then the prose.
* the read joins the market story and the Day Review Show; the research pack
  gains a `market_regimes` section.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for entry in (str(SCRIPTS_DIR), str(ROOT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

FIXTURE = ROOT_DIR / "tests" / "fixtures" / "market_regime_week_golden_v1.jsonl"
PACIFIC = ZoneInfo("America/Los_Angeles")
TYPED = datetime(2026, 9, 1, 9, 0, tzinfo=PACIFIC)
SESSION = "2026-09-25"
NIGHT = datetime(2026, 9, 26, 3, 0, tzinfo=timezone.utc)

GOOD = (
    "Your regime is the bear channel, lower highs since 2026-08-01, day 56. "
    "SPY daily is a lower-high channel with its last lower high on 09-03. "
    "SPY D1 turned bullish on 09-22 while W stayed bullish. "
    "H1 was bearish on 09-23 and 09-24 and bullish again on 09-25. "
    "IWM D1 is bearish."
)


def _table():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _journal(tmp_path):
    from journal_store import JournalStore

    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    for start, regime in (
        ("2026-03-01", "bull_run"),
        ("2026-06-01", "weekly_hh_then_compression"),
        ("2026-08-01", "bear_channel_lower_highs"),
    ):
        store.append_structural_regime(start_date=start, regime=regime, entered_at=TYPED)
    return store


def _inputs(tmp_path):
    from ai_jobs import regime_read

    return regime_read.build_inputs(SESSION, _table(), _journal(tmp_path).list_structural_regime())


def _reply(paragraph=GOOD, sources=("table:2026-09-25:SPY", "structure:2026-09-25:SPY", "regime:3")):
    return {"paragraph": paragraph, "sources": list(sources)}


# -- the verifier ----------------------------------------------------------------
def test_a_grounded_read_passes(tmp_path):
    from ai_jobs import regime_read

    inputs = _inputs(tmp_path)
    assert inputs["window"] == ["2026-09-21", "2026-09-25"]
    assert inputs["trader_regime"]["day_count"] == 56
    read = regime_read.verify_read(_reply(), inputs)
    assert read["paragraph"] == GOOD
    # A month-day that is a pivot date, and the 20-day the facts are measured on, pass.
    extra = GOOD + " The last lower high came on September 3 with SPY under its 20-day."
    assert regime_read.verify_read(_reply(extra), inputs)["paragraph"] == extra


@pytest.mark.parametrize(
    "paragraph, reason",
    [
        ("SPY D1 was bearish on 09-23.", "D1 bearish"),
        ("M15 flipped bullish on 09-25.", "M15"),
        ("SPY printed a lower high on 09-18.", "09-18"),
        ("SPY sits 47.25 points under its high.", "47.25"),
        ("M30 flipped bullish twice this week.", "twice"),
        ("This looks like capitulation.", "capitulation"),
        ("XLK D1 is bullish.", "XLK"),
        ("The trend began in October.", "October"),
        # Reviewer sneaks (2026-09-26): digits from keys, ids and date parts are not sources.
        ("SPY D1 turned bullish on 09-22 and the monthly trend is bullish.", "monthly"),
        ("SPY will retest its high by September 30.", "September 30"),
        ("SPY dropped 25 points in the H1 window.", "25"),
        ("The 15-minute chart and the quarterly trend are bearish.", "15-minute"),
        ("SPY sits 2026 points under its high.", "2026"),
        ("Yesterday SPY D1 turned bullish; last month W was bullish.", "Yesterday"),
        ("SPY D1 turned bullish on 09-22; last month W was bullish.", "last month"),
        ("SPY could rally 3 percent next week.", "could"),
        ("H4 is the 4-hour read.", "4-hour"),
        ("SPY held 30 points above the low.", "30"),
    ],
)
def test_a_word_the_table_does_not_hold_rejects_the_whole_read(tmp_path, paragraph, reason):
    from ai_jobs import day_review_narration, regime_read

    with pytest.raises(day_review_narration.NarrationRejected) as caught:
        regime_read.verify_read(_reply(GOOD + " " + paragraph), _inputs(tmp_path))
    assert reason in str(caught.value)


def test_a_source_outside_the_inputs_rejects_through_the_day_story_check(tmp_path):
    from ai_jobs import day_review_narration, regime_read

    with pytest.raises(day_review_narration.NarrationRejected, match="does not carry"):
        regime_read.verify_read(_reply(sources=("table:2026-09-30:SPY",)), _inputs(tmp_path))
    with pytest.raises(day_review_narration.NarrationRejected, match="cited nothing"):
        regime_read.verify_read(_reply(sources=()), _inputs(tmp_path))


# -- the slot --------------------------------------------------------------------
class _Model:
    def __init__(self, paragraph=GOOD):
        self.paragraph = paragraph
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        assert kwargs["provider"] == "local"
        assert "table:2026-09-25:SPY" in kwargs["evidence"]["allowed_source_ids"]
        return {"model": "local-test", "summary": _reply(self.paragraph, sources=("table:2026-09-25:SPY",))}


def _run(tmp_path, table_path, rows, model, session=SESSION):
    from ai_jobs import regime_read

    return regime_read.run_regime_read(
        session_date=session, now=NIGHT, table_path=table_path, journal_rows=rows,
        out_dir=tmp_path / "reads", request=model,
    )


def test_the_slot_writes_a_verified_read_and_skips_an_unchanged_night(tmp_path):
    import market_regimes

    table_path = tmp_path / "market_regime_table.jsonl"
    market_regimes.append_rows(table_path, _table())
    rows = _journal(tmp_path).list_structural_regime()
    model = _Model()
    out = _run(tmp_path, table_path, rows, model)
    assert out["status"] == "ok" and model.calls == 1
    saved = json.loads((tmp_path / "reads" / f"{SESSION}.json").read_text(encoding="utf-8"))
    assert saved["read"]["paragraph"] == GOOD
    assert saved["source_id"] == f"regime_read:{SESSION}"
    assert saved["trader_regime"] == "bear channel, lower highs"
    again = _run(tmp_path, table_path, rows, model)
    assert again["status"] == "ok" and model.calls == 1


def test_a_rejected_read_is_degraded_and_keeps_the_last_verified_file(tmp_path):
    import market_regimes
    from ai_jobs import ledger

    table_path = tmp_path / "market_regime_table.jsonl"
    market_regimes.append_rows(table_path, _table())
    store = _journal(tmp_path)
    assert _run(tmp_path, table_path, store.list_structural_regime(), _Model())["status"] == "ok"
    target = tmp_path / "reads" / f"{SESSION}.json"
    before = target.read_bytes()

    # New inputs (a correction in the journal) and a reply that invents a date.
    store.append_structural_regime(
        start_date="2026-08-01", regime="bear_channel_lower_highs",
        structure_note="weekly HH, daily LH/LL channel", entered_at=TYPED,
    )
    out = _run(tmp_path, table_path, store.list_structural_regime(), _Model(GOOD + " A lower high on 09-18."))
    assert out["status"] == ledger.STATUS_DEGRADED
    assert "09-18" in out["reason"]
    assert target.read_bytes() == before

    # A later session that is rejected writes nothing; the last good read is still found.
    later = dict(_table()[-3], session_date="2026-09-28")
    market_regimes.append_rows(table_path, [later])
    out = _run(tmp_path, table_path, store.list_structural_regime(), _Model("M15 is bullish."), session="2026-09-28")
    assert out["status"] == ledger.STATUS_DEGRADED
    assert not (tmp_path / "reads" / "2026-09-28.json").exists()
    kept = market_regimes.latest_regime_read("2026-09-28", root=tmp_path / "reads")
    assert kept["session_date"] == SESSION


def test_no_table_skips_before_any_model(tmp_path):
    model = _Model()
    out = _run(tmp_path, tmp_path / "missing.jsonl", [], model)
    assert out["status"] == "skipped" and model.calls == 0


def test_the_slot_is_a_market_read_model_slot_after_the_market_story():
    from ai_jobs import runner

    slots = runner.default_slots()
    names = [slot.name for slot in slots]
    assert names[names.index("market_story_narration") + 1] == "regime_read"
    slot = next(slot for slot in slots if slot.name == "regime_read")
    story = next(slot for slot in slots if slot.name == "market_story_narration")
    assert slot.goal == "market_read" and slot.uses_model is True
    assert slot.reserve_minutes == story.reserve_minutes and slot.max_attempts == 3
    assert "regime_read" in [slot.name for slot in runner.slots_for("weeknight")]


# -- S16.4: the market story opens inside the regime -------------------------------
def _pack(root, session, text, source_id):
    folder = root / "sessions" / session
    folder.mkdir(parents=True, exist_ok=True)
    pack = {
        "session_date": session,
        "trader_said": [
            {"kind": "observation", "text": text, "at": f"{session}T10:00:00-04:00", "source_id": source_id},
            {"kind": "prediction", "text": "", "direction": "up", "source_id": source_id + "p"},
        ],
    }
    (folder / "pack.json").write_text(json.dumps(pack), encoding="utf-8")


def _frame(tmp_path):
    from ai_jobs import market_story_narration as story

    root = tmp_path / "day_review"
    _pack(root, "2026-09-24", "Lower high again, sellers own 775", "said-24")
    _pack(root, "2026-09-22", "Bounce into the channel top", "said-22")
    _pack(root, "2026-07-15", "Before the regime began", "said-07")
    rows = _journal(tmp_path).list_structural_regime()
    return story.regime_frame(SESSION, journal_rows=rows, table_rows=_table(), day_review_root=root)


def test_the_frame_is_the_traders_regime_their_words_and_the_facts(tmp_path):
    frame = _frame(tmp_path)
    assert frame["regime_line"] == "bear channel, lower highs, day 56 (since 2026-08-01)"
    assert [note["text"] for note in frame["trader_words"]] == [
        "Lower high again, sellers own 775", "Bounce into the channel top",
    ]
    assert frame["facts"][0] == "SPY daily lower high on 2026-09-03 (lower-high / lower-low channel)"
    assert frame["facts_as_of"] == SESSION


def test_the_story_opens_with_the_regime_then_words_then_facts_then_prose(tmp_path):
    from ai_jobs import market_story_narration as story
    from test_market_story_narration import _packs

    rollups, output = tmp_path / "rollups", tmp_path / "narration"
    _packs(rollups)
    frame = _frame(tmp_path)
    seen = {}

    def request(**kwargs):
        seen.update(kwargs["evidence"])
        return {"model": "local-test", "summary": {
            "summary": "The trader expected SPY to hold 5400.", "changes": [], "open_questions": [],
            "mentor_question": "What breaks the channel?", "sources": ["journal:note-1"],
        }}

    out = story.run_market_story_narration(
        session_date=SESSION, now=NIGHT, rollups_dir=rollups, out_dir=output,
        request=request, frame_reader=lambda _day: frame,
    )
    assert out["status"] == "ok"
    assert seen["regime_frame"]["regime_line"] == frame["regime_line"]
    assert seen["evidence_hash"] != story._evidence({"weekly": {}}, None)["evidence_hash"]
    saved = json.loads((output / f"{SESSION}.json").read_text(encoding="utf-8"))
    lines = story.story_lines(saved)
    assert lines[0] == "bear channel, lower highs, day 56 (since 2026-08-01)"
    assert lines[1] == 'You, 2026-09-24: "Lower high again, sellers own 775"'
    assert lines[3].startswith("SPY daily lower high on 2026-09-03")
    assert lines[-1] == "The trader expected SPY to hold 5400."

    # The regime read of that night joins the story.
    reads = tmp_path / "reads"
    reads.mkdir()
    (reads / f"{SESSION}.json").write_text(json.dumps({
        "session_date": SESSION, "source_id": f"regime_read:{SESSION}", "read": {"paragraph": GOOD, "sources": []},
    }), encoding="utf-8")
    joined = story.read_story(SESSION, out_dir=output, reads_dir=reads)
    assert joined["lines"] == lines
    assert joined["regime_read"]["read"]["paragraph"] == GOOD


# -- the Day Review Show -------------------------------------------------------------
def test_the_show_carries_the_regime_read_as_a_read_slide_source():
    import day_review_show

    read = {"session_date": SESSION, "source_id": f"regime_read:{SESSION}", "read": {"paragraph": GOOD}}
    chosen = day_review_show.desk_deck(None, {"session_date": SESSION}, session_date=SESSION, regime_read=read)
    slides = chosen["deck"]["slides"]
    read_slides = [slide for slide in slides if slide["kind"] == "read"]
    assert len(read_slides) == 1
    assert read_slides[0]["source_ids"] == [f"regime_read:{SESSION}"]
    assert GOOD in read_slides[0]["lines"][0]
    assert slides[-1]["kind"] == "close"

    with_reads = {"session_date": SESSION, "reads": [{"source_id": "read-1", "verdict": "right"}]}
    chosen = day_review_show.desk_deck(None, with_reads, session_date=SESSION, regime_read=read)
    read_slides = [slide for slide in chosen["deck"]["slides"] if slide["kind"] == "read"]
    assert len(read_slides) == 1
    assert read_slides[0]["source_ids"] == ["read-1", f"regime_read:{SESSION}"]

    unchanged = day_review_show.desk_deck(None, {"session_date": SESSION}, session_date=SESSION)
    assert not [slide for slide in unchanged["deck"]["slides"] if slide["kind"] == "read"]


# -- the research pack ---------------------------------------------------------------
def test_the_research_pack_carries_the_table_the_journal_and_names_missing_grades(tmp_path):
    from scripts import research_pack as rp

    table_path = tmp_path / "market_regime_table.jsonl"
    table_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    _journal(tmp_path)
    sources = rp.Sources(
        lake_root=None, journal_db=tmp_path / "trade_journal.sqlite3", bounce_outcomes_csv=None,
        bounces_csv=None, extra_files={}, ai_store_root=None, protected=(),
        market_regime_table=table_path, regime_grades=None,
    )
    manifest = rp.export_pack(sources, tmp_path / "out")
    section = manifest["market_regimes"]
    assert section["table"]["rows"] == len(_table())
    assert section["table"]["date_min"] == "2026-09-21" and section["table"]["date_max"] == SESSION
    assert section["journal"]["rows"] == 3
    assert section["grades"]["status"] == "missing" and "S16.3" in section["notes"]["grades"]
    journal = json.loads((tmp_path / "out" / "regime_journal.json").read_text(encoding="utf-8"))
    assert [segment["regime"] for segment in journal["timeline"]] == [
        "bull_run", "weekly_hh_then_compression", "bear_channel_lower_highs",
    ]
    copied = (tmp_path / "out" / "market_regime_table.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(copied) == len(_table())

    grades = tmp_path / "grades.json"
    grades.write_text('{"schema": "setup_grades_by_regime_v1"}', encoding="utf-8")
    sources.regime_grades = grades
    manifest = rp.export_pack(sources, tmp_path / "out2")
    assert manifest["market_regimes"]["grades"]["status"] == "ok"

    # --as-of hides rows computed later and journal rows typed later.
    early = rp.export_pack(sources, tmp_path / "out3", as_of=datetime(2026, 8, 15, tzinfo=timezone.utc))
    assert early["market_regimes"]["table"]["rows"] == 0
    assert early["market_regimes"]["journal"]["rows"] == 0
