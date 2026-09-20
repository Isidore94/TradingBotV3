"""TJ-10, added by the builder: the WRITE side of the read ledger.

The tester's `test_a_gradable_grade_can_never_be_stored_without_its_context`
pins the store's refusal. Three of its other tests hand `append_grades` a grade
built with no context at all, so `grade_read` fills the field with a NAMED
absence (`market_read_grades.CONTEXT_UNMEASURED`) rather than leaving it blank -
otherwise those two assertions cannot both hold (see the handoff's QUESTIONS).

That resolution only stays honest if the production path never relies on it, so
these tests pin the guarantee TJ-16 item 1 actually wants: **the seam that
WRITES a grade builds a real point-in-time context first**, and the ledger it
writes is append-only.

Nothing here touches a live store: the service's stores are wired to plain
dicts and the ledger root is a `tmp_path`.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj10_support as fx  # noqa: E402

NOW = datetime(2026, 9, 19, 8, 0)


class _Journal:
    def __init__(self, entries=()):
        self._entries = list(entries)

    def entries_about(self, _session):
        return [dict(row) for row in self._entries]

    def daily_story(self, _session):
        return None

    def theses_for(self, _session):
        return []


def _service(monkeypatch, tmp_path, entries):
    """A service whose every store is a dict and whose ledger is `tmp_path`."""
    import chart_snapshot
    import claimed_picks
    import daily_recap_reader
    import day_review_bars
    import market_read_grades as grader
    from ui.services.day_review_service import DayReviewService

    class _Store:
        def __init__(self, rows):
            self.rows = list(rows)

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl", lambda *a, **k: _Store([]))
    monkeypatch.setattr(daily_recap_reader, "_read_csv", lambda *a, **k: _Store([]))
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        day_review_bars, "read_session_bars",
        lambda *a, **k: {"SPY": fx.session_tape()},
    )
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _symbol: [])
    monkeypatch.setattr(grader, "_default_root", lambda: Path(tmp_path))

    service = DayReviewService(journal_service=_Journal(entries))
    monkeypatch.setattr(service, "_trades", lambda *a, **k: [])
    return service


def test_the_seam_that_writes_a_grade_builds_a_real_context_first(monkeypatch, tmp_path):
    """TJ-16 item 1, where it actually binds: a stored grade carries the
    internals block the ONE builder made, never the named absence."""
    import market_read_grades as grader
    import trade_mentor_context

    clicked = fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")
    service = _service(monkeypatch, tmp_path, [clicked])

    written = service.build_reads_for(fx.SESSION, now=NOW)

    assert len(written) == 1, "the post-close seam graded nothing"
    context = written[0]["context"]
    assert context != grader.CONTEXT_UNMEASURED
    assert context["internals"]["schema"] == trade_mentor_context.SCHEMA
    assert context["direction"] == "up"
    stored = grader.read_grades(fx.SESSION, root=tmp_path)
    assert len(stored) == 1
    assert stored[0]["context"]["internals"]["schema"] == trade_mentor_context.SCHEMA


def test_a_second_pass_over_an_unchanged_verdict_writes_nothing(monkeypatch, tmp_path):
    """The tick runs every 60 seconds. An append-only ledger that appended the
    same fact twice would make every later count wrong."""
    import market_read_grades as grader

    clicked = fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")
    service = _service(monkeypatch, tmp_path, [clicked])

    first = service.build_reads_for(fx.SESSION, now=NOW)
    second = service.build_reads_for(fx.SESSION, now=NOW)

    assert len(first) == 1
    assert second == []
    assert len(grader.read_grades(fx.SESSION, root=tmp_path)) == 1


def test_a_moved_verdict_is_a_new_row_naming_the_old_one(monkeypatch, tmp_path):
    """The rest-of-day read was `pending` mid-session and measured after the
    close. The first answer stays on disk exactly as it was written."""
    import market_read_grades as grader

    clicked = fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")
    service = _service(monkeypatch, tmp_path, [clicked])

    service.build_reads_for(fx.SESSION, now=datetime(2026, 9, 18, 10, 0))
    moved = service.build_reads_for(fx.SESSION, now=NOW)

    stored = grader.read_grades(fx.SESSION, root=tmp_path)
    assert len(stored) == 2
    assert stored[0]["verdict"] == f"pending {fx.SESSION}"
    assert stored[1]["supersedes"] == stored[0]["grade_id"]
    assert moved and moved[0]["supersedes"] == stored[0]["grade_id"]


def test_the_page_read_writes_no_grade_at_all(monkeypatch, tmp_path):
    """A page open is a READ. Only the post-close seam writes the ledger."""
    import market_read_grades as grader

    clicked = fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")
    service = _service(monkeypatch, tmp_path, [clicked])

    payload = service.read_day(fx.SESSION, now=NOW)

    assert list(payload["reads"]), payload.get("error")
    assert grader.read_grades(fx.SESSION, root=tmp_path) == []
    assert not (Path(tmp_path) / "reads").exists()


@pytest.mark.parametrize("verdict", ["right", "pending 2026-09-25"])
def test_the_store_still_refuses_a_gradable_grade_with_a_blank_context(
    tmp_path, verdict
):
    """The tester's guard, restated over both gradable shapes: a row written
    blank can never be given a context afterwards."""
    import market_read_grades as grader

    with pytest.raises(grader.ContextMissingError):
        grader.append_grades(
            fx.SESSION,
            [{"grade_id": "gr-1", "read_id": "rd-1", "verdict": verdict, "context": {}}],
            root=tmp_path,
        )

    assert grader.read_grades(fx.SESSION, root=tmp_path) == []
