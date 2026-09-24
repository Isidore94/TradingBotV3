"""TJ-13A item 2: three slates, and which one a night runs.

plan.md §12.4 TJ-13 items 6 and 8, decision 0021 answer 19.

* **Weeknights** run the deterministic stage plus the short trader-facing
  narration slots and ``ticker_briefs``.
* **Saturday night** carries the weekly slate: ``ai_summary`` once a week rather
  than nightly, and ``weekly_synthesis`` - which has NEVER run, in 476 ledger
  rows, because it needs a typed ``--weekly-synthesis`` (measured 2026-09-19).
* **Sunday night** is the backlog: the deterministic stage, plus a retry of any
  slot that failed that week and is still inside its cap. A slate that does not
  finish resumes on the next night of the same weekend.

``EXPECTED_SLOT_ORDER`` in ``tests/test_ai_jobs_runner.py`` stays the order
WITHIN a night, so every slate here is a subsequence of it (decision 0018's stage
boundaries do not move).

Contract pinned by these tests, for the builder:

    ai_jobs.runner.night_kind(now=None) -> "weeknight" | "saturday" | "sunday"
    ai_jobs.runner.slots_for(kind, *, summary_scopes=None,
                             session_date="", ledger_path=None) -> list[JobSlot]
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

from zoneinfo import ZoneInfo  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")

#: Friday 2026-09-18 is the session both weekend nights are keyed to; the ledger
#: rows for a weekend firing carry it (measured on the live ledger, 2026-09-19).
WEEKEND_SESSION = "2026-09-18"


def _names(slots):
    return [slot.name for slot in slots]


def _slate(kind, **kwargs):
    from ai_jobs import runner

    return _names(runner.slots_for(kind, **kwargs))


def _is_subsequence(names, order):
    it = iter(order)
    return all(name in it for name in names)


def test_a_weeknight_slate_runs_the_deterministic_stage_and_the_short_narration():
    names = _slate("weeknight")

    # the deterministic stage, whole
    for slot in ("journal_import", "journal_auto_tag", "daily_digest", "measured_report"):
        assert slot in names, f"a weeknight must still do {slot}"
    # the short trader-facing narration; the briefs moved to Saturday
    # (trader decision 2026-09-24, WISHLIST P1-3 3b)
    assert "market_story_narration" in names
    assert "ticker_briefs" not in names
    assert "ticker_briefs" in _slate("saturday")
    # the model-gated stage is unchanged
    assert "journal_enrichment" in names


def test_ai_summary_is_absent_from_a_weeknight_slate():
    """Item 8: it leaves the weeknight slate entirely.

    Measured on the live ledger: ``ai_summary`` ran 12,453-18,540 s a night and
    ended ``degraded_no_narrative`` on 09-15, 09-16, 09-17 and 09-18. It is the
    slot the night cannot afford five times a week.
    """
    assert "ai_summary" not in _slate("weeknight")
    assert "weekly_synthesis" not in _slate("weeknight")


def test_ai_summary_runs_on_the_saturday_slate():
    assert "ai_summary" in _slate("saturday")


def test_weekly_synthesis_runs_on_the_saturday_slate_without_a_typed_command():
    """0 rows in 476: the only way to reach it was ``--weekly-synthesis``."""
    assert "weekly_synthesis" in _slate("saturday")


def test_the_sunday_slate_is_the_deterministic_stage_and_nothing_heavy(tmp_path):
    """Item 6: Sunday night is the backlog, not a second weekly slate."""
    led = tmp_path / "ledger.jsonl"
    names = _slate("sunday", session_date=WEEKEND_SESSION, ledger_path=led)

    assert "journal_import" in names
    assert "measured_report" in names
    # Nothing failed this week, so neither heavy slot is retried.
    assert "ai_summary" not in names
    assert "weekly_synthesis" not in names


def test_an_unfinished_saturday_slate_resumes_on_sunday_night(tmp_path):
    """The Saturday slate did not finish, so Sunday night picks it up.

    Both weekend nights key to the same session date - Friday's - which is what
    makes the ledger the honest record of what the weekend still owes.
    """
    from ai_jobs import ledger

    led = tmp_path / "ledger.jsonl"
    ledger.record(
        job="ai_summary",
        status=ledger.STATUS_DEGRADED,
        session_date=WEEKEND_SESSION,
        reason=(
            "summary for 2026-09-18 from 19 usable source(s); "
            "completion=unsynthesized_fallback"
        ),
        path=led,
    )

    names = _slate("sunday", session_date=WEEKEND_SESSION, ledger_path=led)
    assert "ai_summary" in names, "an unfinished weekend slot resumes the next night"


def test_a_finished_saturday_slate_is_not_run_again_on_sunday(tmp_path):
    """The other half of the same rule: a slot that answered is done for the
    weekend. Without this, "resume" would simply mean "run it twice"."""
    from ai_jobs import ledger

    led = tmp_path / "ledger.jsonl"
    ledger.record(
        job="ai_summary",
        status=ledger.STATUS_OK,
        session_date=WEEKEND_SESSION,
        reason="summary for 2026-09-18; completion=synthesized",
        path=led,
    )

    assert "ai_summary" not in _slate("sunday", session_date=WEEKEND_SESSION, ledger_path=led)


def test_a_capped_slot_is_not_retried_on_sunday_night(tmp_path):
    """"Still inside its cap" is part of the rule, not decoration.

    ``journal_enrichment`` burned its three attempts on 2026-09-17 and the
    runner wrote a terminal marker; a backlog slate that re-offered it would
    spend the night re-earning the same marker.
    """
    from ai_jobs import ledger

    led = tmp_path / "ledger.jsonl"
    ledger.record(
        job="journal_enrichment",
        status=ledger.STATUS_FAILED,
        session_date=WEEKEND_SESSION,
        reason="local provider returned invalid summary JSON after 2 attempt(s)",
        path=led,
    )
    ledger.mark_terminal(
        job="journal_enrichment",
        session_date=WEEKEND_SESSION,
        reason="3 attempt(s) already made for 2026-09-18; the per-session cap is 3",
        path=led,
    )

    names = _slate("sunday", session_date=WEEKEND_SESSION, ledger_path=led)
    assert "journal_enrichment" not in names


def test_every_slate_keeps_the_order_within_a_night():
    """Stage boundaries do not move: decision 0018 orders a night, item 2 picks
    which slots that night holds."""
    from tests.test_ai_jobs_runner import EXPECTED_SLOT_ORDER

    for kind in ("weeknight", "saturday", "sunday"):
        names = [name for name in _slate(kind) if name in EXPECTED_SLOT_ORDER]
        assert _is_subsequence(names, EXPECTED_SLOT_ORDER), f"{kind} reorders the night"


def test_a_slate_never_holds_the_same_slot_twice():
    for kind in ("weeknight", "saturday", "sunday"):
        names = _slate(kind)
        assert len(names) == len(set(names)), f"{kind} repeats a slot"


def test_an_unknown_night_kind_raises_rather_than_running_a_guess():
    from ai_jobs import runner

    with pytest.raises(ValueError):
        runner.slots_for("tuesday-ish")


# ---------------------------------------------------------------------------
# the CLI picks the slate; nothing needs typing
# ---------------------------------------------------------------------------


def _capture_run_slots(monkeypatch, seen):
    from ai_jobs import runner

    def _fake(slots, **kwargs):
        seen.extend(_names(slots))
        return runner.RunReport(session_date="2026-09-18", started_at=datetime.now(PACIFIC))

    monkeypatch.setattr(runner, "run_slots", _fake)


def test_the_scheduled_run_takes_its_slate_from_the_night_kind(monkeypatch):
    """``run_ai_jobs.py`` with no flags: the night decides, not the operator."""
    import run_ai_jobs
    from ai_jobs import runner

    seen: list[str] = []
    _capture_run_slots(monkeypatch, seen)
    monkeypatch.setattr(runner, "night_kind", lambda now=None: "saturday", raising=False)

    assert run_ai_jobs.main([]) == 0
    assert "weekly_synthesis" in seen, "Saturday night needs no typed command"
    assert "ai_summary" in seen


def test_a_weeknight_run_with_no_flags_offers_no_heavy_slot(monkeypatch):
    import run_ai_jobs
    from ai_jobs import runner

    seen: list[str] = []
    _capture_run_slots(monkeypatch, seen)
    monkeypatch.setattr(runner, "night_kind", lambda now=None: "weeknight", raising=False)

    assert run_ai_jobs.main([]) == 0
    assert seen, "the weeknight slate is not empty"
    assert "ai_summary" not in seen
    assert "weekly_synthesis" not in seen


def test_the_typed_weekly_synthesis_command_still_works(monkeypatch):
    """Guard: item 2 adds a path, it does not remove the operator's."""
    import run_ai_jobs
    from ai_jobs import runner

    seen: list[str] = []
    _capture_run_slots(monkeypatch, seen)
    monkeypatch.setattr(runner, "night_kind", lambda now=None: "weeknight", raising=False)

    assert run_ai_jobs.main(["--weekly-synthesis"]) == 0
    assert seen == ["weekly_synthesis"]


def test_the_help_text_no_longer_offers_to_skip_the_window():
    """Item 1: "say which flag does what in ``--help``".

    This reads the CLI's own rendered ``--help``, which is what an operator at
    2 p.m. on a Saturday actually sees. Today it promises "skip the off-hours
    window timing", which after item 1 is a promise the runner will not keep for
    a slot that calls a model.
    """
    import contextlib
    import io

    import run_ai_jobs

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        with pytest.raises(SystemExit):
            run_ai_jobs.main(["--help"])
    text = " ".join(buffer.getvalue().split()).lower()

    assert "--force" in text
    assert "skip the off-hours window timing" not in text
