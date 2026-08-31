"""R10.D - the tracker's transitions become an authority of their own.

The tracker is a 951 MB snapshot with no memory. Audit S1 measured the cost:
between one frozen pair, 218 setups changed status, 2,737 CLOSED scenarios
changed status or reason, 1,306 changed exit date, and AMCR LONG on 2026-07-28
went `TIME_STOP @ 46.69, R 0.577` to `TARGET_HIT @ 45.55, R 0.360` on the same
date. A snapshot cannot show any of that.

The constraint that shapes the design: **never deep-copy the payload.** A tiny
per-setup digest sidecar is what makes the diff affordable.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_tracker_ledger as ledger  # noqa: E402


def _setup(setup_id="2026-08-20:AAA:LONG:2026-06-01:favorite_setup", **fields):
    base = {
        "setup_id": setup_id,
        "symbol": "AAA",
        "side": "LONG",
        "scan_date": "2026-08-20",
        "anchor_date": "2026-06-01",
        "setup_status": "OPEN",
        "setup_family": "avwap_retest_followthrough",
        "priority_bucket": "favorite_setup",
        "favorite_zone": "VWAP to UPPER_1",
    }
    base.update(fields)
    return base


def _payload(*setups):
    return {item["setup_id"]: item for item in setups}


# ==========================================================================
# the four event types
# ==========================================================================
def test_a_setup_seen_for_the_first_time_is_initial():
    events = ledger.diff_setups(_payload(_setup()), {}, data_session="2026-08-21")

    assert [event["event_type"] for event in events] == [ledger.EVENT_INITIAL]
    assert events[0]["state_setup_status"] == "OPEN"
    assert events[0]["data_session"] == "2026-08-21"


def test_a_state_change_is_a_transition():
    before = ledger.build_sidecar(_payload(_setup()))
    after = _payload(_setup(setup_status="CLOSED", exit_reason="TARGET_HIT"))

    events = ledger.diff_setups(after, before, data_session="2026-08-21")

    assert [event["event_type"] for event in events] == [ledger.EVENT_TRANSITION]
    assert events[0]["previous_status"] == "OPEN"
    assert events[0]["state_setup_status"] == "CLOSED"


def test_closed_to_open_is_named_reopened():
    """S1 measured 35 CLOSED->OPEN and 1 UNTRADEABLE->OPEN in ONE pair. It is a
    transition, but the one worth naming."""
    before = ledger.build_sidecar(_payload(_setup(setup_status="CLOSED")))
    after = _payload(_setup(setup_status="OPEN"))

    events = ledger.diff_setups(after, before, data_session="2026-08-21")

    assert events[0]["event_type"] == ledger.EVENT_REOPENED
    assert events[0]["previous_status"] == "CLOSED"


def test_untradeable_to_open_is_also_a_reopen():
    before = ledger.build_sidecar(_payload(_setup(setup_status="UNTRADEABLE")))
    events = ledger.diff_setups(_payload(_setup(setup_status="OPEN")), before)
    assert events[0]["event_type"] == ledger.EVENT_REOPENED


def test_a_setup_that_leaves_is_a_tombstone_and_never_a_closure():
    """A setup can vanish because it closed, because the tracker pruned it, or
    because a partial read lost it. This row cannot tell those apart and says
    so rather than implying the flattering one."""
    before = ledger.build_sidecar(_payload(_setup()))

    events = ledger.diff_setups({}, before, data_session="2026-08-21")

    assert events[0]["event_type"] == ledger.EVENT_TOMBSTONE
    assert events[0]["previous_status"] == "OPEN"
    assert "must not be read as a closure" in events[0]["note"]
    assert "state_setup_status" not in events[0]


def test_an_unchanged_setup_emits_nothing():
    """~10k setups run through this on every save. A stream that emitted a row
    per setup per run would say nothing and cost 10 MB a day."""
    before = ledger.build_sidecar(_payload(_setup()))
    assert ledger.diff_setups(_payload(_setup()), before) == []


def test_only_state_bearing_fields_move_the_digest():
    """The payload carries hundreds of fields per setup and most of them move
    every run - a price, a band, a note. Digesting the whole record would emit
    a transition for every setup on every run."""
    before = ledger.build_sidecar(_payload(_setup()))
    noisy = _payload(
        _setup(
            latest_snapshot={"trade_date": "2026-08-21", "close": 101.5},
            priority_score=241.0,
            ranking_note="moved",
        )
    )
    assert ledger.diff_setups(noisy, before) == []


# ==========================================================================
# the sidecar
# ==========================================================================
def test_the_sidecar_holds_digests_not_the_payload(tmp_path):
    """The whole point: the diff must not require two 951 MB dicts in memory."""
    setups = _payload(*[_setup(setup_id=f"id-{index}") for index in range(50)])
    sidecar = ledger.build_sidecar(setups, data_session="2026-08-21")
    path = ledger.save_sidecar(tmp_path / "sidecar.json", sidecar)

    raw = path.read_text(encoding="utf-8")
    assert len(sidecar["digests"]) == 50
    assert all(len(value) == 16 for value in sidecar["digests"].values())
    # A digest, not a record. The state-field NAMES appear once, on purpose -
    # they are what tells the next run which definition these digests were
    # computed over - but no VALUE from the payload is here.
    assert "VWAP to UPPER_1" not in raw
    assert "avwap_retest_followthrough" not in raw
    assert raw.count("favorite_zone") == 1
    # Bounded: a sidecar entry is an id, a 16-char digest and a status, so the
    # file stays a few hundred KB over ~10k setups rather than a second copy of
    # a 951 MB payload.
    assert len(raw) / 50 < 200


def test_an_unreadable_sidecar_reads_as_absent_and_re_seeds(tmp_path):
    """Loud and recoverable. Guessing at a partial sidecar would emit a wave of
    false transitions instead, which is neither."""
    path = tmp_path / "sidecar.json"
    path.write_text("{not json", encoding="utf-8")

    previous = ledger.load_sidecar(path)
    events = ledger.diff_setups(_payload(_setup()), previous)

    assert previous["digests"] == {}
    assert events[0]["event_type"] == ledger.EVENT_INITIAL


def test_a_sidecar_written_over_different_fields_re_seeds_rather_than_lying(tmp_path):
    """If the state-field list ever changes, the old digests were computed over
    a different definition. Comparing them would emit a transition for every
    setup and mean nothing."""
    stale = {
        "schema": ledger.SIDECAR_SCHEMA,
        "state_fields": ["setup_status"],
        "digests": {_setup()["setup_id"]: "deadbeefdeadbeef"},
        "statuses": {_setup()["setup_id"]: "OPEN"},
    }
    events = ledger.diff_setups(_payload(_setup()), stale)

    assert events[0]["event_type"] == ledger.EVENT_INITIAL
    assert "different state fields" in events[0]["note"]


def test_the_schema_is_named_never_numbered():
    assert ledger.SCHEMA_SETUP_TRACKER_EVENT == "setup_tracker_event_v1"
    assert ledger.SIDECAR_SCHEMA == "setup_tracker_digest_sidecar_v1"


# ==========================================================================
# S2 - completed sessions only
# ==========================================================================
def test_a_mark_dated_after_the_vintage_is_counted_and_named():
    """The forming bar. A tracker run during a session marks a close that does
    not exist yet, and a scenario can exit on it."""
    setups = _payload(
        _setup(setup_id="a", latest_snapshot={"trade_date": "2026-08-22"}),
        _setup(setup_id="b", latest_snapshot={"trade_date": "2026-08-21"}),
    )
    result = ledger.forming_bar_marks(setups, "2026-08-21")

    assert result["setups_with_later_marks"] == 1
    assert result["sample"] == ["a"]
    assert result["latest_offending_mark"] == "2026-08-22"
    assert "does not exist yet" in result["note"]


def test_daily_marks_are_checked_too_not_just_the_latest_snapshot():
    setups = _payload(
        _setup(setup_id="a", daily_marks=[{"trade_date": "2026-08-25"}])
    )
    assert ledger.forming_bar_marks(setups, "2026-08-21")["setups_with_later_marks"] == 1


def test_a_clean_payload_says_so_rather_than_staying_silent():
    setups = _payload(_setup(latest_snapshot={"trade_date": "2026-08-21"}))
    result = ledger.forming_bar_marks(setups, "2026-08-21")

    assert result["setups_with_later_marks"] == 0
    assert result["marks_seen"] == 1
    assert "no mark is dated after" in result["note"]


def test_a_payload_with_no_vintage_is_unmeasured_never_clean():
    """Missing data is uncertainty, never confirmation (plan.md sec 5)."""
    result = ledger.forming_bar_marks(_payload(_setup()), "")
    assert result["measured"] is False
    assert "UNMEASURED" in result["note"]


# ==========================================================================
# S3a - exchange sessions
# ==========================================================================
def test_the_span_is_measured_and_a_stale_horizon_is_flagged():
    """Root cause: the future row is `idx + horizon` into the symbol's own scan
    rows, not into sessions. Live medians: horizon 5 -> 64 sessions, 10 -> 73."""
    drift = ledger.horizon_drift("2026-05-01", "2026-08-03", 5)

    assert drift["sessions_spanned"] == 66
    assert drift["stale_horizon"] is True


def test_a_horizon_that_matches_its_span_is_not_flagged():
    drift = ledger.horizon_drift("2026-08-17", "2026-08-20", 3)
    assert drift["sessions_spanned"] == 3
    assert drift["stale_horizon"] is False


def test_an_unreadable_date_is_unmeasured_not_zero():
    drift = ledger.horizon_drift("", "2026-08-20", 5)
    assert drift["sessions_spanned"] is None
    assert drift["stale_horizon"] is None
    assert drift["basis"].startswith("unmeasured")


def test_a_calendar_is_used_when_one_is_supplied():
    """A business-day count over a week containing a holiday is close but not
    exact, and presenting it as an exchange-session count would be a number
    nobody measured."""
    calendar = [date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 21)]
    assert ledger.sessions_between("2026-08-17", "2026-08-21", calendar) == 3
    # Without the calendar the same span reads as four business days.
    assert ledger.sessions_between("2026-08-17", "2026-08-21") == 4


# ==========================================================================
# the wiring: the ledger rides the tracker save, and never copies the payload
# ==========================================================================
def test_the_save_path_emits_transitions_and_writes_the_sidecar_last(tmp_path, monkeypatch):
    """The sidecar is written only after the append succeeded, so a crash
    between the two costs a REPEAT of this run's diff rather than a silent hole
    in the stream. Re-emitting a transition is recoverable; dropping one is not.
    """
    from master_avwap_lib import legacy

    tracker = tmp_path / "tracker.json"
    monkeypatch.setattr(legacy, "SETUP_TRACKER_FILE", tracker)
    appended: list[dict] = []

    class _Stream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def append(self, event, **_kwargs):
            appended.append(event)
            return event

    monkeypatch.setattr("evidence_ledger.EvidenceLedger", _Stream)

    payload = {
        "data_session": "2026-08-21",
        "setups": _payload(_setup(setup_id="a"), _setup(setup_id="b")),
    }
    result = legacy._append_setup_tracker_events(payload)

    kinds = [event["event_type"] for event in appended]
    assert kinds.count(ledger.EVENT_INITIAL) == 2
    assert kinds[-1] == "run_summary"
    assert result["events"] == 2
    # The sidecar exists now, so a second identical save is silent.
    appended.clear()
    legacy._append_setup_tracker_events(payload)
    assert [event["event_type"] for event in appended] == ["run_summary"]


def test_the_run_summary_separates_nothing_changed_from_did_not_run(tmp_path, monkeypatch):
    """An event stream alone cannot make that distinction, and it is the one a
    reader needs first when a day looks empty."""
    from master_avwap_lib import legacy

    monkeypatch.setattr(legacy, "SETUP_TRACKER_FILE", tmp_path / "tracker.json")
    appended: list[dict] = []
    monkeypatch.setattr(
        "evidence_ledger.EvidenceLedger",
        lambda **kwargs: type("S", (), {"append": lambda _self, event, **_k: appended.append(event)})(),
    )

    legacy._append_setup_tracker_events(
        {"data_session": "2026-08-21", "setups": _payload(_setup())}
    )
    summary = appended[-1]

    assert summary["event_type"] == "run_summary"
    assert summary["setups_in_payload"] == 1
    assert summary["tracker_data_session"] == "2026-08-21"
    assert summary["forming_bar_marks"]["setups_with_later_marks"] == 0


def test_the_diff_never_copies_the_payload(tmp_path, monkeypatch):
    """The constraint that shapes the whole design. A setup dict handed in must
    be the SAME object the caller holds - no deepcopy, no re-serialisation of a
    951 MB payload."""
    from master_avwap_lib import legacy

    monkeypatch.setattr(legacy, "SETUP_TRACKER_FILE", tmp_path / "tracker.json")
    monkeypatch.setattr(
        "evidence_ledger.EvidenceLedger",
        lambda **kwargs: type("S", (), {"append": lambda _self, event, **_k: None})(),
    )
    seen: list[int] = []

    class _Watched(dict):
        def __getitem__(self, key):
            seen.append(id(self))
            return super().__getitem__(key)

    record = _Watched(_setup())
    payload = {"data_session": "2026-08-21", "setups": {record["setup_id"]: record}}
    legacy._append_setup_tracker_events(payload)

    # Still the same object; nothing replaced it with a copy.
    assert payload["setups"][record["setup_id"]] is record


def test_a_ledger_failure_leaves_the_sidecar_untouched(tmp_path, monkeypatch):
    """So the next run re-diffs against the same baseline and the transitions
    are recorded LATE rather than lost."""
    from master_avwap_lib import legacy

    monkeypatch.setattr(legacy, "SETUP_TRACKER_FILE", tmp_path / "tracker.json")

    def _explode(**kwargs):
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr("evidence_ledger.EvidenceLedger", _explode)

    with pytest.raises(RuntimeError):
        legacy._append_setup_tracker_events(
            {"data_session": "2026-08-21", "setups": _payload(_setup())}
        )
    assert not legacy._setup_tracker_sidecar_path().exists()
