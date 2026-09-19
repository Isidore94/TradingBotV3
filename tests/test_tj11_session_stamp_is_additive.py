"""TJ-11 blocker 1 - the new session field is ADDITIVE and nothing live moves.

Reviewer NO-GO, 2026-09-19: the first build of TJ-11 changed what
`session_date` MEANS - the writer stamped a Friday-evening veto with Monday's
date and `load_annotations`' default matched rows by the forward-mapped
session. Reproduced on copies of live data: asking for 2026-09-21, base hid 0
symbols and the branch hid 12 while marking 18 "Reviewed today".

Lead design, binding: `session_date` keeps EXACTLY its base meaning and value
for every writer and every reader. The session a decision BELONGS to is a NEW
ADDITIVE field, `decision_session`, read only by TJ-11's own readers. This file
pins both halves:

* the WRITER adds a key and changes nothing else;
* every live reader named in the review - `pick_feedback.decisions_today` and
  `reviewed_symbols_today` (the setups-table hide, the chart-cycling skip, the
  review queue's Reviewed-today mark), `review_learning.attach_annotation_veto_
  reasons` (the veto cohort behind `review_policy.json`), the three cohort
  graders and `daily_recap_reader._decisions` - returns for a NEW row exactly
  what it returns for the same row written the base way.

Nothing here touches a live store: every path is a `tmp_path`.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PACIFIC = ZoneInfo("America/Los_Angeles")

#: Friday 2026-09-18, 21:04 Pacific - the live case. New York's calendar date
#: for that moment is the Saturday; the session it belongs to is Monday's.
FRIDAY_EVENING = datetime(2026, 9, 18, 21, 4, 28, tzinfo=PACIFIC)
SATURDAY = "2026-09-19"
MONDAY = "2026-09-21"


def _veto_row(**overrides) -> dict:
    """One annotation row exactly as the desk wrote it BEFORE TJ-11."""
    row = {
        "schema_version": 1,
        "event_id": "veto-1",
        "event_type": "veto",
        "symbol": "HLIT",
        "side": "SHORT",
        "session_date": SATURDAY,
        "timeframe": "D1",
        "created_at": FRIDAY_EVENING.isoformat(),
        "source": "chart_review",
        "reason_code": "extended",
        "vocab_version": 3,
    }
    row.update(overrides)
    return row


def _a_veto_code() -> str:
    """A code the shipped vocabulary really has. Never a literal version."""
    from ui.annotations.vocabulary import load_veto_vocabulary

    return list(load_veto_vocabulary().codes)[0]


def _write(path: Path, rows) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


# -- the writer --------------------------------------------------------------


def test_the_new_field_is_additive_and_session_date_is_byte_for_byte_base(monkeypatch):
    """The only difference between a base row and a TJ-11 row is ONE key."""
    import market_session
    from ui.annotations import store

    monkeypatch.setattr(
        market_session, "get_market_session_window",
        lambda *a, **k: type("_W", (), {"market_date": date(2026, 9, 19)})(),
    )

    row = store.build_annotation(
        store.EVENT_VETO,
        symbol="HLIT",
        side="SHORT",
        reason_code=_a_veto_code(),
        timeframe="D1",
        created_at=FRIDAY_EVENING,
    )

    # What base wrote, unchanged: the market session WINDOW's date, pinned
    # above to the Saturday the live rows carry.
    assert row["session_date"] == SATURDAY
    # What TJ-11 adds, beside it.
    assert row[store.DECISION_SESSION_FIELD] == MONDAY
    # And nothing else appeared: the row minus the new key is the base row's
    # own key set.
    assert set(row) - {store.DECISION_SESSION_FIELD} == {
        "schema_version", "event_id", "event_type", "symbol", "session_date",
        "created_at", "source", "reason_code", "vocab_version", "side",
        "timeframe",
    }


def test_the_added_field_is_empty_rather_than_guessed_when_it_cannot_be_computed(monkeypatch):
    """A calendar that cannot answer writes nothing, never today's date."""
    import market_calendar
    from ui.annotations import store

    monkeypatch.setattr(
        market_calendar, "decision_session", lambda *_a, **_k: None
    )

    row = store.build_annotation(
        store.EVENT_VETO, symbol="HLIT", side="SHORT", reason_code=_a_veto_code(),
        session_date=SATURDAY, created_at=FRIDAY_EVENING,
    )

    assert row[store.DECISION_SESSION_FIELD] == ""
    assert row["session_date"] == SATURDAY


# -- the readers -------------------------------------------------------------


def test_load_annotations_default_matches_session_date_exactly(tmp_path):
    """The default is the join `pick_feedback` and the graders have always used."""
    from ui.annotations import store

    target = _write(tmp_path / "a.jsonl", [_veto_row(), _veto_row(
        event_id="veto-2", symbol="MKC", session_date=MONDAY,
        created_at="2026-09-21T13:00:00-04:00",
    )])

    assert [row["symbol"] for row in store.load_annotations(target, session_date=MONDAY)] == ["MKC"]
    assert [row["symbol"] for row in store.load_annotations(target, session_date=SATURDAY)] == ["HLIT"]


def test_the_forward_mapping_is_an_explicit_opt_in(tmp_path):
    from ui.annotations import store

    target = _write(tmp_path / "a.jsonl", [_veto_row()])

    assert store.load_annotations(target, session_date=MONDAY) == []
    assert [
        row["symbol"]
        for row in store.load_annotations(
            target, session_date=MONDAY, by_decision_session=True
        )
    ] == ["HLIT"]


def test_a_new_row_reads_the_same_to_pick_feedback_as_the_base_row_did(tmp_path):
    """`decisions_today` feeds the setups-table hide and Reviewed-today."""
    import pick_feedback
    from ui.annotations import store

    base_row = _veto_row()
    new_row = _veto_row() | {store.DECISION_SESSION_FIELD: MONDAY}
    empty = _write(tmp_path / "empty.jsonl", [])

    def _decisions(rows, target):
        # One file per (row shape, target): `_read_day_ledgers` memoizes on the
        # path plus its mtime and size, so two shapes must not share a name.
        kind = "new" if rows[0].get(store.DECISION_SESSION_FIELD) else "base"
        path = _write(tmp_path / f"a-{kind}.jsonl", rows)
        return pick_feedback.decisions_today(
            market_date=target,
            pick_feedback_path=empty,
            review_events_path=empty,
            annotations_path=path,
        )

    for target in (SATURDAY, MONDAY):
        base = _decisions([base_row], target)
        new = _decisions([new_row], target)
        assert sorted(new.rejected) == sorted(base.rejected), target
        assert sorted(new.liked) == sorted(base.liked), target

    # And the value itself: the Friday-evening veto is Saturday's row to this
    # reader, exactly as on base. Monday's setups table hides nothing for it.
    assert "HLIT" in _decisions([new_row], SATURDAY).rejected
    assert _decisions([new_row], MONDAY).rejected == {}


def test_a_new_row_reads_the_same_to_review_learnings_veto_join(tmp_path):
    """The veto cohort behind `review_policy.json` must not move."""
    import review_learning
    from ui.annotations import store

    def _episodes():
        return [
            review_learning.Episode(
                trade_date=day, symbol="HLIT", side="SHORT",
                opportunity_id=f"op-{day}", resolution="reject", timeframe="D1",
            )
            for day in (SATURDAY, MONDAY)
        ]

    def _annotated(rows):
        path = _write(
            tmp_path / f"veto-{rows[0].get('decision_session', 'base')}.jsonl", rows
        )
        episodes = _episodes()
        review_learning.attach_annotation_veto_reasons(episodes, path)
        return {episode.trade_date: episode.dislike_reasons for episode in episodes}

    base = _annotated([_veto_row()])
    new = _annotated([_veto_row() | {store.DECISION_SESSION_FIELD: MONDAY}])

    assert new == base
    # The join is on `session_date`, so it is the SATURDAY episode that carries
    # the code - unchanged by TJ-11.
    assert base[SATURDAY] == "extended"
    assert base[MONDAY] == ""


def test_a_new_row_reads_the_same_to_the_three_cohort_graders(tmp_path):
    """A cohort pick is built from named fields; an extra key adds no column."""
    from ui.annotations import like_cohort, pass_cohort, veto_cohort
    from ui.annotations import store

    base_rows = [_veto_row()]
    new_rows = [_veto_row() | {store.DECISION_SESSION_FIELD: MONDAY}]

    for module, picker in (
        (veto_cohort, "veto_pick_rows"),
        (like_cohort, "like_pick_rows"),
        (pass_cohort, "pass_pick_rows"),
    ):
        build = getattr(module, picker, None)
        if build is None:  # pragma: no cover - a renamed grader is a real failure
            pytest.fail(f"{module.__name__} has no {picker}")
        base = build(base_rows)
        new = build(new_rows)
        assert new == base, module.__name__


def test_a_new_row_reads_the_same_to_daily_recap_readers_decisions(tmp_path):
    """`_decisions` filters `session_date` by exact match and keeps doing so."""
    import daily_recap_reader
    from ui.annotations import store

    class _Store:
        def __init__(self, rows):
            self.rows = list(rows)

    empty = _Store([])

    def _for(rows, target):
        return daily_recap_reader._decisions(target, _Store(rows), empty, empty, empty)

    base_row = _veto_row()
    new_row = _veto_row() | {store.DECISION_SESSION_FIELD: MONDAY}

    for target in (SATURDAY, MONDAY):
        assert _for([new_row], target) == _for([base_row], target), target
    assert [d.symbol for d in _for([new_row], SATURDAY)] == ["HLIT"]
    assert _for([new_row], MONDAY) == ()


# -- TJ-11's own reader ------------------------------------------------------


def test_only_tj11s_reader_moves_the_row_forward():
    """The walk-away builder reads the additive field, or maps the stamp."""
    from walkaway_day import build

    def _rejected(session, decision):
        day = build(
            session,
            sources={"decisions": (decision,), "preference": (), "outcomes": ()},
            bars={},
            now=datetime(2026, 9, 22, 8, 0),
            daily_bars={},
        )
        return [row.symbol for row in day.rejected]

    stored = {
        "session_date": SATURDAY, "symbol": "HLIT", "side": "SHORT",
        "verdict": "veto", "source": "annotations", "timeframe": "D1",
        "stamp": FRIDAY_EVENING.isoformat(), "category": "chart_review",
        "decision_session": MONDAY,
    }
    older = dict(stored)
    older.pop("decision_session")

    # The new row says which session it belongs to; the old one is mapped from
    # its own stamp. Both land on Monday, and neither lands on Friday.
    assert _rejected(MONDAY, stored) == ["HLIT"]
    assert _rejected(MONDAY, older) == ["HLIT"]
    assert _rejected("2026-09-18", stored) == []
    assert _rejected("2026-09-18", older) == []
