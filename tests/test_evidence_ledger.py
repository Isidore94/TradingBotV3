"""R10.A - the append-only ledger the outcome store will be believed from.

The properties under test are the ones that make a store an authority rather
than a log:

* a caller **cannot** overwrite the schema name, the timestamps or the writer
  identity - a row that can lie about who wrote it is not evidence;
* a torn line is **counted**, never skipped, so a gap can never read as an
  absence of events;
* every row carries UTC **and** the market session, because one without the
  other cannot answer "which session was this?" across a 20:30-local write;
* segments are monthly, and naming what is cold never deletes it.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evidence_ledger as el  # noqa: E402

# 2026-08-21 20:30 PT = 2026-08-22 03:30 UTC. The session is still the 21st.
LATE_WRITE = datetime(2026, 8, 22, 3, 30, tzinfo=timezone.utc)
NOON = datetime(2026, 8, 21, 16, 0, tzinfo=timezone.utc)


def _ledger(tmp_path: Path, **kwargs) -> el.EvidenceLedger:
    return el.intraday_outcome_ledger(tmp_path, **kwargs)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
def test_the_schema_is_a_name_and_it_is_v1():
    assert el.SCHEMA_INTRADAY_OUTCOME_EVENT == "intraday_outcome_event_v1"


def test_a_ledger_refuses_to_exist_without_a_stream_and_a_schema(tmp_path):
    with pytest.raises(ValueError):
        el.EvidenceLedger(stream="", schema="x_v1", directory=tmp_path)
    with pytest.raises(ValueError):
        el.EvidenceLedger(stream="s", schema="", directory=tmp_path)


def test_a_caller_cannot_overwrite_the_ledgers_own_fields(tmp_path):
    """A row that can lie about who wrote it is not evidence."""
    ledger = _ledger(tmp_path)
    row = ledger.append(
        {
            "event_id": "a",
            "schema": "something_else",
            "event_at": "1999-01-01T00:00:00+00:00",
            "session_date": "1999-01-01",
            "writer_pid": 1,
        },
        now=NOON,
    )
    assert row["schema"] == "intraday_outcome_event_v1"
    assert row["event_at"] == "2026-08-21T16:00:00+00:00"
    assert row["session_date"] == "2026-08-21"
    assert row["writer_pid"] != 1


def test_the_callers_mapping_is_not_mutated(tmp_path):
    event = {"event_id": "a"}
    _ledger(tmp_path).append(event, now=NOON)
    assert event == {"event_id": "a"}


def test_every_row_says_who_wrote_it(tmp_path):
    """Two desks ran concurrently on 2026-08-20 and nothing could say so."""
    row = _ledger(tmp_path, run_id="run-7").append({"event_id": "a"}, now=NOON)
    assert row["writer_host"] and row["writer_pid"]
    assert row["run_id"] == "run-7"


def test_a_ledger_with_no_run_id_omits_the_field_rather_than_faking_one(tmp_path):
    row = _ledger(tmp_path).append({"event_id": "a"}, now=NOON)
    assert "run_id" not in row


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------
def test_the_session_is_market_local_not_utc(tmp_path):
    """20:30 PT on the 21st is 03:30 UTC on the 22nd. The session is the 21st."""
    row = _ledger(tmp_path).append({"event_id": "a"}, now=LATE_WRITE)
    assert row["event_at"] == "2026-08-22T03:30:00+00:00"
    assert row["session_date"] == "2026-08-21"


def test_a_naive_clock_is_read_as_utc_rather_than_rejected(tmp_path):
    row = _ledger(tmp_path).append({"event_id": "a"}, now=datetime(2026, 8, 21, 16, 0))
    assert row["event_at"].endswith("+00:00")


def test_the_segment_follows_the_session_not_the_utc_month(tmp_path):
    """A 20:30-local write on 31 August belongs to August."""
    ledger = _ledger(tmp_path)
    ledger.append({"event_id": "a"}, now=datetime(2026, 9, 1, 3, 30, tzinfo=timezone.utc))
    assert [path.name for path in ledger.segments()] == ["intraday_outcome_events-202608.jsonl"]


def test_segments_are_monthly(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.append({"event_id": "a"}, now=datetime(2026, 7, 15, 16, 0, tzinfo=timezone.utc))
    ledger.append({"event_id": "b"}, now=NOON)
    assert [path.name for path in ledger.segments()] == [
        "intraday_outcome_events-202607.jsonl",
        "intraday_outcome_events-202608.jsonl",
    ]


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def test_rows_come_back_in_the_order_they_were_written(tmp_path):
    ledger = _ledger(tmp_path)
    for index in range(5):
        ledger.append({"event_id": f"e{index}"}, now=NOON)
    assert [row["event_id"] for row in ledger.read()] == [f"e{i}" for i in range(5)]


def test_a_torn_line_is_counted_not_skipped(tmp_path):
    """Power loss mid-append. A silently dropped row makes a gap look empty."""
    ledger = _ledger(tmp_path)
    ledger.append({"event_id": "a"}, now=NOON)
    segment = ledger.segments()[0]
    with segment.open("a", encoding="utf-8") as handle:
        handle.write('{"event_id": "b", "sess\n')
    ledger.append({"event_id": "c"}, now=NOON)
    result = ledger.read()
    assert [row["event_id"] for row in result] == ["a", "c"]
    assert result.unreadable == 1
    assert "1 unreadable" in result.coverage_note
    assert "a gap, not an absence" in result.coverage_note


def test_a_clean_read_says_so_without_a_caveat(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.append({"event_id": "a"}, now=NOON)
    assert ledger.read().coverage_note == "n=1"


def test_a_window_selects_by_session_date(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.append({"event_id": "jul"}, now=datetime(2026, 7, 15, 16, 0, tzinfo=timezone.utc))
    ledger.append({"event_id": "aug"}, now=NOON)
    got = ledger.read(start="2026-08-01", end="2026-08-31")
    assert [row["event_id"] for row in got] == ["aug"]


def test_a_row_that_cannot_say_its_session_is_excluded_from_a_window_and_counted(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.append({"event_id": "a"}, now=NOON)
    segment = ledger.segments()[0]
    with segment.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_id": "nodate"}) + "\n")
    windowed = ledger.read(start="2026-08-01", end="2026-08-31")
    assert [row["event_id"] for row in windowed] == ["a"]
    assert windowed.unreadable == 1
    # ...but an unwindowed read keeps it: it is a real row, just an unplaceable one.
    assert len(ledger.read()) == 2


def test_reading_by_event_type(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.append({"event_id": "a", "event_type": "registered"}, now=NOON)
    ledger.append({"event_id": "a", "event_type": "final"}, now=NOON)
    assert [row["event_type"] for row in ledger.read(event_types=["final"])] == ["final"]


def test_an_empty_directory_reads_as_empty_rather_than_raising(tmp_path):
    result = _ledger(tmp_path / "nothing").read()
    assert len(result) == 0 and result.unreadable == 0


# ---------------------------------------------------------------------------
# append-only
# ---------------------------------------------------------------------------
def test_appending_never_rewrites_what_is_already_there(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.append({"event_id": "a", "status": "open"}, now=NOON)
    first = ledger.segments()[0].read_text(encoding="utf-8")
    ledger.append({"event_id": "a", "status": "closed"}, now=NOON)
    after = ledger.segments()[0].read_text(encoding="utf-8")
    assert after.startswith(first), "a correction is a new row, never an edit"
    assert len(ledger.read()) == 2


def test_a_correction_is_a_superseding_event_and_both_survive(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.append({"event_id": "a", "close_r": 0.0, "event_type": "final"}, now=NOON)
    ledger.append(
        {"event_id": "a", "close_r": -1.0, "event_type": "final",
         "supersedes": "the fabricated zero"},
        now=NOON,
    )
    rows = list(ledger.read())
    assert [row["close_r"] for row in rows] == [0.0, -1.0]
    assert rows[1]["supersedes"]


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------
def test_cold_segments_are_named_and_not_touched(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.append({"event_id": "old"}, now=datetime(2025, 1, 15, 16, 0, tzinfo=timezone.utc))
    ledger.append({"event_id": "new"}, now=NOON)
    cold = ledger.cold_segments(today=date(2026, 8, 21))
    assert [path.name for path in cold] == ["intraday_outcome_events-202501.jsonl"]
    assert all(path.exists() for path in cold), "naming what is cold never deletes it"


def test_the_hot_window_is_thirteen_months():
    assert el.HOT_MONTHS == 13


def test_a_segment_with_an_unreadable_stamp_is_left_out_of_the_cold_list(tmp_path):
    ledger = _ledger(tmp_path)
    (tmp_path / "intraday_outcome_events-notamonth.jsonl").write_text("", encoding="utf-8")
    assert ledger.cold_segments(today=date(2026, 8, 21)) == ()


def test_the_default_directory_is_the_runtime_ledger_dir():
    from project_paths import RUNTIME_DATA_DIR

    assert el.default_ledger_dir() == Path(RUNTIME_DATA_DIR) / "evidence_ledgers"


def test_the_default_directory_is_what_the_cold_push_covers():
    """`push_cold_to_das.ps1` must carry the directory this writes to."""
    script = (ROOT_DIR / "scripts" / "ops" / "push_cold_to_das.ps1").read_text(encoding="utf-8")
    assert "evidence_ledgers" in script
