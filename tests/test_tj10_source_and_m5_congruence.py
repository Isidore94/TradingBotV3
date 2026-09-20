"""TJ-10 fix round - say WHICH read you are showing, and pair M5 with M5.

Reviewer, 2026-09-20.

**Blocker 2.** Live clicks are 0, every live read row is `source="extracted"`,
and the page printed *"your D1 read is up; the desk reads trending_down"* and a
bare `right` / `wrong` chip. On 2026-09-17 that `up` was inferred from one note
while ANOTHER note in the same session read `down`. A stated call and an
inferred stance are different evidence (decision 0021 answer 29), so every line
and every chip names its source in plain words, a CLICK always wins over an
extraction, and contradictory extracted stances are not used at all - the line
says they contradict.

**Blocker 3.** Lead decision 6's M5 half was not built: only a D1 read was
selected and only D1 picks were counted, so 2026-09-18's five M5 reads produced
no M5 line at all. The rest-of-day read is now paired with the session's M5
likes and not-todays, through the real `read_day`.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tj10_support as fx  # noqa: E402

NOW = datetime(2026, 9, 19, 8, 0)


def _extracted(text: str, *, timeframe: str = "D1", minutes: int = 0):
    return fx.old_entry(
        "empty", text=text, timeframe=timeframe,
        created_at=fx.STAMP + timedelta(minutes=minutes),
    )


def _decision(symbol: str, side: str, *, timeframe: str, verdict: str = "like"):
    return {
        "session_date": fx.SESSION, "symbol": symbol, "side": side,
        "category": "chart_review", "verdict": verdict, "source": "annotations",
        "timeframe": timeframe, "stamp": fx.STAMP.isoformat(),
        "capture_id": f"e-{symbol}-{timeframe}", "reason": "",
        "decision_session": fx.SESSION,
    }


# -- blocker 2: the source is named -----------------------------------------


def test_an_extracted_stance_is_never_printed_as_the_traders_own_call():
    import market_read_grades as grader

    rows = grader.read_rows([_extracted("D1 SPY is downtrending")], session=fx.SESSION)
    read, note = grader.select_read(rows, timeframe="D1")
    assert note == ""

    line = {
        row["kind"]: row for row in grader.congruence_lines(
            session=fx.SESSION, d1_read=read, d1_label="trending_down",
        )
    }["desk_d1_label"]

    assert "we read your note as down" in line["text"]
    assert "your D1 read is" not in line["text"]
    assert line["verdict"] == "agrees"


def test_a_clicked_call_says_it_was_clicked_and_when():
    import market_read_grades as grader

    entry = fx.mentor_entry(
        direction="down", horizon="next_5_sessions", timeframe="D1",
    )
    read = grader.read_rows([entry], session=fx.SESSION)[0]

    line = {
        row["kind"]: row for row in grader.congruence_lines(
            session=fx.SESSION, d1_read=read, d1_label="trending_down",
        )
    }["desk_d1_label"]

    assert line["text"].startswith("your call: down (clicked 07:02)")


def test_a_clicked_read_always_wins_over_an_extracted_one():
    """Even when the extraction is later, and even when they disagree."""
    import market_read_grades as grader

    clicked = fx.mentor_entry(
        direction="down", horizon="next_5_sessions", timeframe="D1",
    )
    noted = _extracted("SPY is uptrending off the lows", minutes=90)
    rows = grader.read_rows([clicked, noted], session=fx.SESSION)

    read, note = grader.select_read(rows, timeframe="D1")

    assert note == ""
    assert read["source"] == grader.SOURCE_CLICK
    assert read["direction"] == "down"


def test_two_notes_that_read_both_ways_are_not_one_view():
    """The reviewer's 2026-09-17 shape: one note up, another down, no click."""
    import market_read_grades as grader

    rows = grader.read_rows(
        [
            _extracted("SPY is uptrending off the lows"),
            _extracted("D1 SPY is downtrending", minutes=120),
        ],
        session=fx.SESSION,
    )

    read, note = grader.select_read(rows, timeframe="D1")

    assert read is None
    assert note == "your notes read both ways (1 up, 1 down) - no single read to compare"

    lines = {
        row["kind"]: row for row in grader.congruence_lines(
            session=fx.SESSION, d1_read=None, d1_note=note,
            d1_label="trending_down",
            decisions=[_decision("AAA", "LONG", timeframe="D1")],
        )
    }
    assert lines["desk_d1_label"]["verdict"] == grader.UNMEASURED
    # UPDATED (TJ-10 follow-up item 2, widened not weakened): the line leads
    # with its OWN content (the desk's label / the picks' own mix) and only
    # then the reason there is nothing to compare it with, rather than
    # dropping its own content and printing the note verbatim.
    assert lines["desk_d1_label"]["text"].startswith("the desk reads trending_down")
    assert lines["desk_d1_label"]["text"].endswith(note)
    assert lines["picks_side_mix"]["verdict"] == grader.UNMEASURED
    assert lines["picks_side_mix"]["text"].startswith(
        "1 of 1 D1 likes and claims were LONG"
    )
    assert lines["picks_side_mix"]["text"].endswith(note)


def test_every_line_names_a_timeframe():
    """Including the fills line, which printed an empty one."""
    import market_read_grades as grader

    entry = fx.mentor_entry(
        direction="down", horizon="next_5_sessions", timeframe="D1",
    )
    read = grader.read_rows([entry], session=fx.SESSION)[0]

    for line in grader.congruence_lines(
        session=fx.SESSION, d1_read=read, d1_label="trending_down",
    ):
        assert line["timeframe"], line["kind"]


def test_a_tie_in_the_side_mix_names_no_side():
    import market_read_grades as grader

    entry = fx.mentor_entry(
        direction="down", horizon="next_5_sessions", timeframe="D1",
    )
    read = grader.read_rows([entry], session=fx.SESSION)[0]
    decisions = [
        _decision("AAA", "LONG", timeframe="D1"),
        _decision("BBB", "SHORT", timeframe="D1"),
    ]

    line = {
        row["kind"]: row for row in grader.congruence_lines(
            session=fx.SESSION, d1_read=read, decisions=decisions,
        )
    }["picks_side_mix"]

    assert "no lean" in line["text"]
    assert "LONG" not in line["text"] and "SHORT" not in line["text"]
    assert line["verdict"] == grader.UNMEASURED
    # ...and the line still restates the read it was compared with.
    assert "your call: down" in line["text"]


# -- blocker 3: the M5 half ---------------------------------------------------


def test_an_m5_read_is_paired_with_m5_picks_and_never_with_d1_ones():
    import market_read_grades as grader

    m5 = fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")
    m5_read = grader.read_rows([m5], session=fx.SESSION)[0]
    decisions = [
        _decision("AAA", "SHORT", timeframe="M5"),
        _decision("BBB", "SHORT", timeframe="M5"),
        _decision("CCC", "LONG", timeframe="D1"),
        _decision("DDD", "LONG", timeframe="D1"),
        _decision("EEE", "LONG", timeframe="M5", verdict="focus__m5_not_today"),
    ]

    lines = {
        row["kind"]: row for row in grader.congruence_lines(
            session=fx.SESSION, m5_read=m5_read, decisions=decisions,
        )
    }
    line = lines[grader.CONGRUENCE_M5_KIND]

    assert line["timeframe"] == "M5"
    assert set(line["source_ids"]) == {"e-AAA-M5", "e-BBB-M5"}
    assert line["counts"] == {"long": 0, "short": 2, "not_today": 1}
    assert line["counts"]["long"] + line["counts"]["short"] == len(line["source_ids"])
    # UPDATED (TJ-10 follow-up item 1, widened not weakened): n=2 is under
    # `evidence_stats.MIN_REPORTABLE_N` (30) - this is the reviewer's live
    # 2026-09-17 shape, where a 2-of-2 side mix printed `disagrees` at full
    # weight. The verdict is now `too_few`; the counts and floor note stand.
    assert line["verdict"] == grader.VERDICT_TOO_FEW
    assert "too few to call" in line["text"]
    assert "not today" in line["text"]


def test_the_three_d1_lines_are_unchanged_when_no_m5_read_exists():
    import market_read_grades as grader

    lines = grader.congruence_lines(session=fx.SESSION)

    assert tuple(line["kind"] for line in lines) == grader.CONGRUENCE_KINDS


# -- both blockers, through the REAL read_day ---------------------------------


pytestmark_qt = pytest.mark.qt


class _Journal:
    def __init__(self, entries=()):
        self._entries = list(entries)

    def entries_about(self, _session):
        return [dict(row) for row in self._entries]

    def daily_story(self, _session):
        return None

    def theses_for(self, _session):
        return []


def _service(monkeypatch, entries, decisions=()):
    import chart_snapshot
    import claimed_picks
    import d1_environment_store
    import daily_recap_reader
    import day_review_bars
    import journal_store
    from ui.services.day_review_service import DayReviewService

    class _Store:
        def __init__(self, rows):
            self.rows = list(rows)

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl", lambda *a, **k: _Store([]))
    monkeypatch.setattr(daily_recap_reader, "_read_csv", lambda *a, **k: _Store([]))

    class _Decision:
        def __init__(self, row):
            self.symbol = row["symbol"]
            self.side = row["side"]
            self.category = row["category"]
            self.verdict = row["verdict"]
            self.source = row["source"]
            self.timeframe = row["timeframe"]
            self.capture_id = row["capture_id"]
            self.reason = row["reason"]
            self.observed_at = fx.STAMP

    monkeypatch.setattr(
        daily_recap_reader, "_decisions",
        lambda stamped, *a, **k: (
            [_Decision(row) for row in decisions] if stamped == fx.SESSION else []
        ),
    )
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        journal_store, "JournalStore",
        lambda *a, **k: type("_J", (), {"list_trades": lambda self: []})(),
    )
    monkeypatch.setattr(day_review_bars, "read_session_bars",
                        lambda *a, **k: {"SPY": fx.session_tape()})
    monkeypatch.setattr(day_review_bars, "session_is_closed", lambda *a, **k: True)
    monkeypatch.setattr(day_review_bars, "session_is_backfillable", lambda *a, **k: False)
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _symbol: [])
    monkeypatch.setattr(d1_environment_store, "_cached_daily_bars", lambda _symbol: [])
    monkeypatch.setattr(
        d1_environment_store, "labels_by_session", lambda **_k: {}
    )
    monkeypatch.setattr(
        d1_environment_store, "label_for_session", lambda *a, **k: "unknown"
    )

    service = DayReviewService(journal_service=_Journal(entries))
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())
    monkeypatch.setattr(service, "_trades", lambda *a, **k: [])
    monkeypatch.setattr(
        service, "_decisions_and_claims", lambda _session: (list(decisions), [])
    )
    return service


def test_read_day_builds_an_m5_line_for_an_m5_read(monkeypatch):
    """2026-09-18's shape: M5 reads and M5 picks, and a line that pairs them."""
    import market_read_grades as grader

    service = _service(
        monkeypatch,
        entries=[fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")],
        decisions=[
            _decision("AAA", "LONG", timeframe="M5"),
            _decision("BBB", "LONG", timeframe="M5"),
            _decision("CCC", "SHORT", timeframe="D1"),
        ],
    )

    payload = service.read_day(fx.SESSION, now=NOW)

    kinds = [line["kind"] for line in payload["congruence"]]
    assert grader.CONGRUENCE_M5_KIND in kinds, payload.get("error")
    line = next(
        row for row in payload["congruence"] if row["kind"] == grader.CONGRUENCE_M5_KIND
    )
    assert line["timeframe"] == "M5"
    assert line["counts"]["long"] == 2
    # UPDATED (TJ-10 follow-up item 1, widened not weakened): n=2 is under the
    # reporting floor, so the verdict is `too_few` rather than `agrees`.
    assert line["verdict"] == grader.VERDICT_TOO_FEW
    assert "your call: up" in line["text"]


def test_read_day_says_when_the_notes_read_both_ways(monkeypatch):
    """The reviewer's 09-17 shape, through the page's own read."""
    import market_read_grades as grader

    service = _service(
        monkeypatch,
        entries=[
            _extracted("SPY is uptrending off the lows"),
            _extracted("D1 SPY is downtrending", minutes=120),
        ],
    )

    payload = service.read_day(fx.SESSION, now=NOW)

    line = next(
        row for row in payload["congruence"] if row["kind"] == "desk_d1_label"
    )
    assert line["verdict"] == grader.UNMEASURED
    assert "both ways" in line["text"]
    # ...and both notes are still RECORDED as reads, with their source named.
    assert {row["source"] for row in payload["reads"]} == {grader.SOURCE_EXTRACTED}


def test_the_payload_carries_the_source_so_the_page_can_name_it(monkeypatch):
    service = _service(
        monkeypatch,
        entries=[
            fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5"),
            _extracted("the market is leaking all afternoon", timeframe="M5", minutes=180),
        ],
    )

    payload = service.read_day(fx.SESSION, now=NOW)

    assert {row["source"] for row in payload["reads"]} == {"click", "extracted"}


def test_the_page_names_the_source_on_every_chip(monkeypatch):
    pytest.importorskip("PySide6", reason="the Day Review page uses PySide6")
    from PySide6.QtWidgets import QApplication

    from ui.panels.day_review_panel import DayReviewPanel
    from ui.services.day_review_service import empty_payload

    app = QApplication.instance() or QApplication([])
    clicked = fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")
    noted = _extracted("the market is leaking", timeframe="M5", minutes=60)
    payload = empty_payload(fx.SESSION)
    payload["entries"] = [clicked, noted]
    payload["reads"] = [
        {"read_id": "rd-1", "entry_id": clicked["entry_id"], "source": "click",
         "direction": "up", "horizon": "rest_of_day", "verdict": "right"},
        {"read_id": "rd-2", "entry_id": noted["entry_id"], "source": "extracted",
         "direction": "down", "horizon": "rest_of_day", "verdict": "wrong"},
    ]

    panel = DayReviewPanel(service=type("_S", (), {
        "read_day": lambda self, session, **k: dict(payload)
    })(), clock=lambda: NOW)
    monkeypatch = None  # noqa: F841 - the panel is driven directly
    try:
        panel.render(payload)
        labels = [
            panel.entries.item(index).text() for index in range(panel.entries.count())
        ]
        assert any("your call: up — right" in text for text in labels), labels
        assert any(
            "we read your note as down — wrong" in text for text in labels
        ), labels
        assert panel.verdict_chips()[clicked["entry_id"]] == "right"
    finally:
        try:
            panel.shutdown()
        except Exception:  # noqa: BLE001
            pass
        panel.deleteLater()
        app.processEvents()
