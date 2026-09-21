"""TJ-12 review round 2 - a session older than the ledger tail is UNKNOWN.

Reviewer, 2026-09-20, against a read-only copy of the live ledger (1,256,082
bytes, 483 rows): the 256 KB / 500-row tail holds only **173 of those rows**,
reaching back to 2026-09-11. `day_review_panel.PICKER_SESSIONS` is 15, so **9
of the 15 sessions the picker offers today** produced

    "0 overnight slot(s) read, 0 finished ok, none reported trouble"

while the ledger really holds 10-16 slots for each of them; 2026-09-11 was
half-visible and said 5 finished ok when 16 did. That is unmeasured presented
as fine - the thing `How fresh` exists to prevent and the same defect class as
round 1's (plan.md sec 5: missing data is uncertainty, never confirmation).

The lead's rule, 2026-09-20, with NO second read and no whole-file fallback:

* the tail reports whether it TRUNCATED and the oldest ``session_date`` it
  returned;
* truncated AND the asked session is older than **or equal to** that oldest
  session -> ``night_status: "unknown"``, no slot counts printed, no failed
  slots. Equal counts as unknown because the window may have cut that night in
  half, which is exactly what 2026-09-11 did;
* inside the window with zero rows -> ``night_status: "no_rows"``;
* **"none reported trouble" is said only when at least one slot was read.**
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj12_support as fx  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")

#: Sessions in the order the live ledger holds them: the oldest is the one the
#: tail cannot reach, the newest is the one the trader opens.
OLD_SESSION = "2026-09-04"
EDGE_SESSION = "2026-09-11"
NEW_SESSION = fx.SESSION  # 2026-09-18


def _row(job: str, status: str, session: str, minute: int) -> dict:
    import ai_jobs.ledger as ledger

    moment = datetime(2026, 9, 18, 22, 0, tzinfo=PACIFIC) + timedelta(minutes=minute)
    return {
        "schema": ledger.LEDGER_SCHEMA,
        "job": job,
        "status": status,
        "session_date": session,
        "model": "",
        "started_at": moment.isoformat(timespec="seconds"),
        "finished_at": (moment + timedelta(minutes=1)).isoformat(timespec="seconds"),
        "duration_seconds": 60.0,
        # The live rows carry a real error string on a failure and a reason on a
        # skip; the padding is what makes the byte window bite, exactly as it
        # does on the 1.2 MB file.
        "reason": "the window was closed" + ("." * 400),
        "outputs": [],
        "tokens": {},
        "error": "",
    }


#: Rows per session in the big fixture. Sized so the byte window lands INSIDE
#: `EDGE_SESSION`, which is the live shape: 2026-09-11 was half-visible.
ROWS_PER_SESSION = 300


def _write_rows(path: Path, rows) -> Path:
    """The ledger's OWN writer, in ONE call.

    `ai_jobs.ledger.append_row` fsyncs per row, which is right for a nightly job
    and 900 fsyncs for a fixture about a byte window. This is the same
    `diagnostics.artifact_io` writer it calls, handed every row at once.
    """
    from diagnostics.artifact_io import append_jsonl_rows

    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl_rows(path, [dict(row) for row in rows], fsync=False)
    return path


def _big_ledger(tmp_path: Path) -> Path:
    """A ledger LARGER than the byte window, oldest session first.

    The window has to START inside `EDGE_SESSION`: `NEW_SESSION`'s rows alone
    fit, `NEW` plus `EDGE` do not. Both are plain facts about the FILE, checked
    here so the fixture cannot drift into a regime where the test would pass for
    the wrong reason.
    """
    import ai_jobs.ledger as ledger
    import day_report_card

    rows = []
    for session in (OLD_SESSION, EDGE_SESSION, NEW_SESSION):
        for index in range(ROWS_PER_SESSION):
            rows.append(_row(f"slot_{index:03d}", ledger.STATUS_OK, session, index))
    path = _write_rows(tmp_path / "ai_store" / "logs" / "ai_job_ledger.jsonl", rows)
    assert path.stat().st_size > day_report_card.LEDGER_TAIL_BYTES, (
        "the fixture has to be bigger than the window it is about"
    )
    lines = path.read_bytes().splitlines(keepends=True)
    newest = sum(len(line) for line in lines[-ROWS_PER_SESSION:])
    two = sum(len(line) for line in lines[-2 * ROWS_PER_SESSION:])
    assert newest < day_report_card.LEDGER_TAIL_BYTES < two, (
        "the window must start inside the middle session, as it does live"
    )
    return path


def _small_ledger(tmp_path: Path) -> Path:
    """A ledger the reader takes WHOLE - nothing is out of sight."""
    import ai_jobs.ledger as ledger
    import day_report_card

    rows = [
        _row("daily_digest", ledger.STATUS_OK, OLD_SESSION, 1),
        _row("ticker_briefs", ledger.STATUS_FAILED, OLD_SESSION, 2),
        _row("daily_digest", ledger.STATUS_OK, NEW_SESSION, 3),
    ]
    path = fx.write_ledger_file(
        tmp_path / "small" / "logs" / "ai_job_ledger.jsonl", rows
    )
    assert path.stat().st_size <= day_report_card.LEDGER_TAIL_BYTES
    return path


def _fresh(path: Path, session: str) -> dict:
    return fx.freshness_facts(ledger_path=path, session=session)


@pytest.fixture(scope="module")
def big(tmp_path_factory) -> Path:
    """Built ONCE: it is a fixed file, and every test asks it a question."""
    return _big_ledger(tmp_path_factory.mktemp("big_ledger"))


@pytest.fixture(scope="module")
def small(tmp_path_factory) -> Path:
    return _small_ledger(tmp_path_factory.mktemp("small_ledger"))


# ---------------------------------------------------------------------------
# the blocker
# ---------------------------------------------------------------------------
def test_a_session_older_than_the_tail_is_unknown_not_a_clean_night(big, small):
    """The nine picker sessions that said "none reported trouble"."""
    import day_report_card

    line = day_report_card.how_fresh(_fresh(big, OLD_SESSION))

    assert line["night_status"] == "unknown"
    assert line["failed_slots"] == ()
    lowered = line["text"].lower()
    assert "night status unknown for this session" in lowered, line["text"]
    assert "older than the ledger tail" in lowered, line["text"]
    assert "none reported trouble" not in lowered, line["text"]
    assert "finished ok" not in lowered, "no slot counts may be printed"


def test_the_boundary_session_is_unknown_because_the_window_may_have_halved_it(big, small):
    """2026-09-11 said 5 slots finished ok when 16 really did.

    The oldest session the tail returned is the one the window CUT, so it is
    the one answer the card cannot stand behind.
    """
    import day_report_card

    line = day_report_card.how_fresh(_fresh(big, EDGE_SESSION))

    assert line["night_status"] == "unknown"
    assert line["failed_slots"] == ()
    assert line["n"] == 0 and line["measured"] == 0
    assert "older than the ledger tail" in line["text"].lower(), line["text"]


def test_a_session_inside_the_window_still_counts_normally(big, small):
    """Today and the last few sessions - what the trader opens 95% of the time."""
    import day_report_card

    line = day_report_card.how_fresh(_fresh(big, NEW_SESSION))

    assert line["night_status"] == "read"
    assert line["n"] == ROWS_PER_SESSION
    assert line["slots_ok"] == ROWS_PER_SESSION
    assert "none reported trouble" in line["text"].lower(), line["text"]


def test_a_whole_file_read_never_calls_an_old_session_unknown(big, small):
    """Nothing was out of sight, so the counts are real however old the session.

    `truncated` is a fact about the READ, never about the date.
    """
    import day_report_card

    line = day_report_card.how_fresh(_fresh(small, OLD_SESSION))

    assert line["night_status"] == "read"
    assert line["n"] == 2
    assert line["slots_ok"] == 1
    assert line["failed_slots"] == ("ticker_briefs",)
    assert "failed: ticker_briefs" in line["text"], line["text"]


def test_a_session_the_window_covers_but_holds_no_rows_for_says_so(big, small):
    """Inside the window and genuinely empty is a THIRD answer.

    "0 read, none reported trouble" reads as a clean night; "no overnight rows
    for this session" is what actually happened.
    """
    import day_report_card

    line = day_report_card.how_fresh(_fresh(small, "2026-09-17"))

    assert line["night_status"] == "no_rows"
    assert line["n"] == 0 and line["measured"] == 0
    assert line["failed_slots"] == ()
    assert "no overnight rows for this session" in line["text"].lower(), line["text"]
    assert "none reported trouble" not in line["text"].lower(), line["text"]


def test_none_reported_trouble_is_only_ever_said_over_at_least_one_slot(big, small, tmp_path):
    """The wording rule, over every state this line has."""
    import day_report_card

    lines = [
        day_report_card.how_fresh(_fresh(big, OLD_SESSION)),
        day_report_card.how_fresh(_fresh(big, EDGE_SESSION)),
        day_report_card.how_fresh(_fresh(big, NEW_SESSION)),
        day_report_card.how_fresh(_fresh(small, OLD_SESSION)),
        day_report_card.how_fresh(_fresh(small, "2026-09-17")),
        day_report_card.how_fresh(fx.freshness_facts(ledger_path=None)),
        day_report_card.how_fresh(
            fx.freshness_facts(ledger_path=tmp_path / "gone" / "ledger.jsonl")
        ),
    ]

    for line in lines:
        if "none reported trouble" in line["text"].lower():
            assert line["n"] >= 1, line["text"]
            assert line["night_status"] == "read", line["text"]
        if line["night_status"] != "read":
            assert line["failed_slots"] == (), line["text"]
            assert line["n"] == 0, line["text"]


def test_the_tail_reports_its_own_truncation_and_oldest_session(big, small):
    """The seam the rule rests on, asserted directly - and NO second read.

    A reader that answered "unknown" by opening the file again would have paid
    for the whole 1.2 MB to say it had not read it.
    """
    import day_report_card

    windowed = day_report_card._tail_rows(big, day_report_card.LEDGER_TAIL_ROWS)
    whole = day_report_card._tail_rows(small, day_report_card.LEDGER_TAIL_ROWS)

    assert windowed.truncated is True
    assert windowed.oldest_session == EDGE_SESSION, windowed.oldest_session
    assert whole.truncated is False
    assert whole.oldest_session == OLD_SESSION
    assert len(windowed.rows) <= day_report_card.LEDGER_TAIL_ROWS


def test_the_unknown_answer_opens_the_file_exactly_once(big, monkeypatch):
    import day_report_card

    path = big
    opened: list[str] = []
    real_open = open

    def _counting_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _counting_open)
    line = day_report_card.how_fresh(_fresh(path, OLD_SESSION))

    assert line["night_status"] == "unknown"
    assert [name for name in opened if name == str(path)] == [str(path)]
