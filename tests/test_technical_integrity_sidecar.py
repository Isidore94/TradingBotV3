"""Packet 3 item 1: the daily replay stops streaming 618 MB to find a subset.

`technical_integrity_events.jsonl` measured **618 MB** on 2026-08-31 - append-only,
no retention - and the after-close wrap-up replayed it every evening by streaming
and `json.loads`-ing every line to keep the `level_resolved` rows. That is an
hour-class job, and although it runs on a background thread, Python's GIL means an
hour of hot parsing in-process steals GUI-thread time all evening.

The fix is a derived sidecar written as the events happen. What these pin is that
it is exactly that - DERIVED:

* the main log is untouched, and its append happens first;
* a sidecar append failure costs the sidecar line and never the main write;
* the rows the replay gets back are the same rows, in the same order, whether
  they came from the sidecar or the full stream;
* any doubt - missing, torn, or a watermark past the end of the log - falls back
  to the full stream or a rebuild, never to a wrong answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import technical_integrity as ti  # noqa: E402


def _event(index: int, kind: str = "level_resolved") -> dict:
    return {
        "event_type": kind,
        "event_id": f"E{index}",
        "session_date": "2026-08-31",
        "symbol": f"SYM{index}",
        "family": "sma50",
        "resolved_at": f"2026-08-31T09:{index % 60:02d}:00",
    }


def _write_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _mixed_log(path: Path, resolved: int = 6) -> list[dict]:
    rows = []
    for index in range(resolved):
        rows.append(_event(index, "level_test_started"))
        rows.append(_event(index))
        rows.append(_event(index, "post_resolution_followup"))
    _write_log(path, rows)
    return [row for row in rows if row["event_type"] == "level_resolved"]


class TestTheSidecarMatchesTheStream:
    def test_a_rebuild_captures_exactly_the_resolved_rows(self, tmp_path):
        events = tmp_path / "technical_integrity_events.jsonl"
        expected = _mixed_log(events)

        result = ti.rebuild_resolved_sidecar(events)

        assert result["ok"] is True
        assert result["rows"] == result["resolved"] == len(expected)
        assert ti.load_resolved_technical_integrity_events(events) == expected

    def test_the_sidecar_and_the_full_stream_agree(self, tmp_path):
        events = tmp_path / "technical_integrity_events.jsonl"
        _mixed_log(events, resolved=9)
        ti.rebuild_resolved_sidecar(events)

        from_sidecar = ti.load_resolved_technical_integrity_events(events)
        from_stream = ti.load_resolved_technical_integrity_events(events, use_sidecar=False)

        assert from_sidecar == from_stream
        assert len(from_sidecar) == 9

    def test_the_sidecar_lives_beside_the_log_it_derives_from(self, tmp_path):
        events = tmp_path / "technical_integrity_events.jsonl"
        assert ti.technical_integrity_resolved_path(events).parent == events.parent
        assert ti.technical_integrity_resolved_path(events).name.endswith("_resolved.jsonl")

    def test_a_late_mirror_after_a_catch_up_cannot_double_count(self, tmp_path):
        """Review round, packet 3: the interleaving that wrote one event twice.

        The evidence clock appends a resolved row to the MAIN log first and
        mirrors it to the sidecar second; the wrap-up's sync runs on another
        thread. A thread switch between those two steps let sync catch up the
        tail (appending the row) before the clock's own mirror landed (appending
        it again) - and the replay then counted that event twice while the full
        stream counted it once. Both copies carry the same source byte offset,
        which is what the reader now dedupes on. Disk may hold the duplicate
        line; the ANSWER may not.
        """
        events = tmp_path / "technical_integrity_events.jsonl"
        sidecar = ti.technical_integrity_resolved_path(events)

        _write_log(events, [_event(0, "level_test_started"), _event(0)])
        offset_a = events.stat().st_size
        ti.append_resolved_sidecar_row(sidecar, _event(0), offset_a)

        # Clock thread appends B to the log...
        _write_log(events, [_event(1)])
        offset_b = events.stat().st_size
        # ...thread switch: sync catches up the tail before B's mirror lands...
        assert ti.sync_resolved_sidecar(events)["action"] == "caught_up"
        # ...and the clock's own mirror of B arrives late.
        ti.append_resolved_sidecar_row(sidecar, _event(1), offset_b)

        from_sidecar = ti.load_resolved_technical_integrity_events(events)
        from_stream = ti.load_resolved_technical_integrity_events(events, use_sidecar=False)
        assert from_sidecar == from_stream
        assert [row["event_id"] for row in from_sidecar] == ["E0", "E1"]


class TestTheWatermarkCatchesUpRatherThanLying:
    def test_a_stale_sidecar_appends_the_tail_and_answers_completely(self, tmp_path):
        """The desk ran an older build for a week, or the sidecar was deleted
        and half-rebuilt: the answer must still be the whole truth."""
        events = tmp_path / "technical_integrity_events.jsonl"
        first = _mixed_log(events, resolved=3)
        ti.rebuild_resolved_sidecar(events)

        later = _mixed_log(events, resolved=4)

        state = ti.sync_resolved_sidecar(events)
        assert state["action"] == "caught_up"
        assert ti.load_resolved_technical_integrity_events(events) == first + later

    def test_a_current_sidecar_does_no_work(self, tmp_path):
        events = tmp_path / "technical_integrity_events.jsonl"
        _mixed_log(events)
        ti.rebuild_resolved_sidecar(events)

        assert ti.sync_resolved_sidecar(events)["action"] == "current"

    def test_a_tail_with_no_resolved_rows_still_records_how_far_it_read(self, tmp_path):
        """Otherwise every later call rescans the same tail forever."""
        events = tmp_path / "technical_integrity_events.jsonl"
        _mixed_log(events, resolved=2)
        ti.rebuild_resolved_sidecar(events)
        _write_log(events, [_event(99, "level_test_started")])

        assert ti.sync_resolved_sidecar(events)["action"] == "caught_up"
        assert ti.sync_resolved_sidecar(events)["action"] == "current"

    def test_a_watermark_past_the_end_of_the_log_rebuilds(self, tmp_path):
        """The log was replaced under the sidecar - trusting the offset would
        seek past the end and answer with nothing."""
        events = tmp_path / "technical_integrity_events.jsonl"
        _mixed_log(events, resolved=5)
        ti.rebuild_resolved_sidecar(events)

        events.write_text("", encoding="utf-8")
        _mixed_log(events, resolved=1)

        state = ti.sync_resolved_sidecar(events)
        assert state["action"] == "rebuilt"
        assert len(ti.load_resolved_technical_integrity_events(events)) == 1

    def test_a_torn_sidecar_line_rebuilds_rather_than_guesses(self, tmp_path):
        events = tmp_path / "technical_integrity_events.jsonl"
        expected = _mixed_log(events, resolved=4)
        ti.rebuild_resolved_sidecar(events)
        sidecar = ti.technical_integrity_resolved_path(events)
        with sidecar.open("a", encoding="utf-8") as handle:
            handle.write('{"src_offset": 12, "row": {"event_ty')

        assert ti.sync_resolved_sidecar(events)["action"] == "rebuilt"
        assert ti.load_resolved_technical_integrity_events(events) == expected

    def test_a_missing_log_is_not_an_answer(self, tmp_path):
        events = tmp_path / "nothing.jsonl"
        assert ti.load_resolved_technical_integrity_events(events) == []


class TestTheMainLogIsTheAuthority:
    def test_the_collector_mirrors_resolved_rows_as_they_happen(self, tmp_path):
        events = tmp_path / "technical_integrity_events.jsonl"
        collector = ti.TechnicalIntegrityMonitor(
            events_path=events,
            state_path=tmp_path / "state.json",
            snapshot_path=tmp_path / "snapshot.json",
        )

        collector._append_event(_event(1, "level_test_started"))
        collector._append_event(_event(1))
        collector._append_event(_event(2))

        sidecar = ti.technical_integrity_resolved_path(events)
        assert sidecar.exists()
        # No catch-up needed: the sidecar was written alongside the log.
        assert ti.sync_resolved_sidecar(events)["action"] == "current"
        assert [row["event_id"] for row in ti.load_resolved_technical_integrity_events(events)] == [
            "E1",
            "E2",
        ]

    def test_the_main_log_still_holds_every_row(self, tmp_path):
        """Nothing is removed, filtered or rewritten in the authority."""
        events = tmp_path / "technical_integrity_events.jsonl"
        collector = ti.TechnicalIntegrityMonitor(
            events_path=events,
            state_path=tmp_path / "state.json",
            snapshot_path=tmp_path / "snapshot.json",
        )
        kinds = ["level_test_started", "level_resolved", "post_resolution_followup"]
        for index, kind in enumerate(kinds):
            collector._append_event(_event(index, kind))

        written = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
        assert [row["event_type"] for row in written] == kinds

    def test_a_sidecar_failure_never_costs_the_main_append(self, tmp_path, monkeypatch):
        """Evidence rule, applied to a DERIVED file: losing a sidecar line costs
        a catch-up scan later and nothing else."""
        events = tmp_path / "technical_integrity_events.jsonl"
        collector = ti.TechnicalIntegrityMonitor(
            events_path=events,
            state_path=tmp_path / "state.json",
            snapshot_path=tmp_path / "snapshot.json",
        )
        monkeypatch.setattr(
            ti, "append_resolved_sidecar_row", lambda *a, **k: (_ for _ in ()).throw(OSError("full"))
        )

        with pytest.raises(OSError):
            collector._append_event(_event(1))

        # The main write happened FIRST and is on disk regardless.
        written = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
        assert [row["event_id"] for row in written] == ["E1"]

    def test_the_appender_itself_swallows_and_reports(self, tmp_path):
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")

        assert ti.append_resolved_sidecar_row(blocked / "side.jsonl", _event(1), 10) is False
