"""Breadth-ledger gap fill (docs/DURABILITY_CATCHUP_PLAN.md sec 2.4).

``vold_m5.jsonl`` lost 41 of 78 bars on 2026-08-06 and 73 of 78 on 08-07 -
not because the data was unavailable, but because the desk was not up to poll
for it. A completed breadth bar is a pure function of provider history, so
Tier B allows fetching the missing ones afterwards through the same qualified
contract the live poller uses.

The rules the fill must never break: rows carry ``capture_mode: "backfill"``,
bar ends stay unique so nothing a live poll wrote is displaced, and slots that
remain missing keep an explicit ``data_gap`` marker instead of quietly
vanishing from the count.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

LOCAL = ZoneInfo("America/Vancouver")
#: 2026-07-30 06:30 Pacific == the 09:30 ET open.
OPEN = datetime(2026, 7, 30, 6, 30, tzinfo=LOCAL)
SESSION = "2026-07-30"
AFTER_CLOSE = OPEN + timedelta(hours=7)  # 13:30 Pacific / 16:30 ET
NEXT_MORNING = OPEN + timedelta(days=1, hours=-1)


def _row(minutes, *, value=100.0):
    start = OPEN + timedelta(minutes=minutes)
    return {
        "time": start.replace(tzinfo=None).strftime("%Y%m%d  %H:%M:%S"),
        "open": value,
        "high": value + 10.0,
        "low": value - 10.0,
        "close": value + 2.0,
        "volume": 0,
    }


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _recorder(tmp_path):
    from vold_recorder import VoldSessionRecorder

    recorder = VoldSessionRecorder(
        ledger_path=tmp_path / "vold.jsonl",
        state_path=tmp_path / "state.json",
    )
    recorder.activate_contract(
        {
            "con_id": 26718738,
            "symbol": "TICK-NYSE",
            "exchange": "NYSE",
            "proxy_kind": "nyse_tick_proxy",
            "is_exact_vold": False,
        },
        as_of=OPEN,
        now=OPEN,
    )
    return recorder


def _bars(path):
    return [row for row in _events(path) if row["event_type"] == "breadth_bar"]


def _gaps(path):
    return [row for row in _events(path) if row["event_type"] == "data_gap"]


def test_live_bars_are_marked_live_and_absence_still_means_live(tmp_path):
    from diagnostics.artifact_io import CAPTURE_MODE_LIVE, row_capture_mode

    recorder = _recorder(tmp_path)
    assert recorder.observe([_row(0), _row(5)], now=OPEN + timedelta(minutes=12)) == 2

    assert all(row["capture_mode"] == CAPTURE_MODE_LIVE for row in _bars(recorder.ledger_path))
    # Every row written before the field existed must keep reading as live.
    assert row_capture_mode({"event_type": "breadth_bar"}) == CAPTURE_MODE_LIVE


def test_gap_fill_appends_the_missing_bars_marked_backfill(tmp_path):
    from diagnostics.artifact_io import CAPTURE_MODE_BACKFILL, CAPTURE_MODE_LIVE

    recorder = _recorder(tmp_path)
    # The desk polled the first two bars, then went down.
    assert recorder.observe([_row(0), _row(5)], now=OPEN + timedelta(minutes=12)) == 2

    summary = recorder.backfill_session_bars(
        [_row(minutes) for minutes in range(0, 390, 5)],
        session_date=SESSION,
        now=AFTER_CLOSE,
        trigger="close_of_day",
    )

    assert summary["ran"] is True
    assert summary["still_missing"] == 0
    bars = _bars(recorder.ledger_path)
    assert len(bars) == 78  # a full RTH session
    by_mode = {row["bar_end"]: row["capture_mode"] for row in bars}
    assert len(set(by_mode)) == len(bars), "bar ends must stay unique"
    assert sum(mode == CAPTURE_MODE_LIVE for mode in by_mode.values()) == 2
    assert sum(mode == CAPTURE_MODE_BACKFILL for mode in by_mode.values()) == 76


def test_gap_fill_never_rewrites_or_duplicates_a_live_row(tmp_path):
    recorder = _recorder(tmp_path)
    assert recorder.observe([_row(0)], now=OPEN + timedelta(minutes=7)) == 1
    live_row = _bars(recorder.ledger_path)[0]

    recorder.backfill_session_bars(
        [_row(minutes, value=999.0) for minutes in range(0, 390, 5)],
        session_date=SESSION,
        now=AFTER_CLOSE,
        trigger="close_of_day",
    )

    matching = [row for row in _bars(recorder.ledger_path) if row["bar_end"] == live_row["bar_end"]]
    assert matching == [live_row], "the live row must survive untouched and unduplicated"


def test_running_the_fill_twice_changes_nothing(tmp_path):
    recorder = _recorder(tmp_path)
    rows = [_row(minutes) for minutes in range(0, 390, 5)]
    recorder.backfill_session_bars(rows, session_date=SESSION, now=AFTER_CLOSE)
    first = recorder.ledger_path.read_text(encoding="utf-8")

    recorder.backfill_session_bars(rows, session_date=SESSION, now=AFTER_CLOSE)

    assert recorder.ledger_path.read_text(encoding="utf-8") == first


def test_slots_the_provider_cannot_supply_stay_an_explicit_gap(tmp_path):
    from diagnostics.artifact_io import CAPTURE_MODE_BACKFILL

    recorder = _recorder(tmp_path)
    # The provider has the first hour and nothing after it.
    summary = recorder.backfill_session_bars(
        [_row(minutes) for minutes in range(0, 60, 5)],
        session_date=SESSION,
        now=AFTER_CLOSE,
        trigger="close_of_day",
    )

    assert summary["bars_added"] == 12
    assert summary["still_missing"] == 66
    gaps = _gaps(recorder.ledger_path)
    # One contiguous outage becomes one marker, not sixty-six.
    assert len(gaps) == 1
    assert gaps[0]["missing_bar_count"] == 66
    assert gaps[0]["capture_mode"] == CAPTURE_MODE_BACKFILL
    assert gaps[0]["data_gap"] is True


def test_gap_fill_does_not_double_mark_a_slot_the_live_poll_already_flagged(tmp_path):
    recorder = _recorder(tmp_path)
    assert recorder.observe([_row(0)], now=OPEN + timedelta(minutes=7)) == 1
    # The live poller records the 06:40 slot as missing.
    assert recorder.record_data_gap(
        reason="historical request returned no rows",
        now=OPEN + timedelta(minutes=12),
    )
    live_gap_count = len(_gaps(recorder.ledger_path))

    recorder.backfill_session_bars(
        [_row(0)],
        session_date=SESSION,
        now=OPEN + timedelta(minutes=12),
        trigger="close_of_day",
    )

    gap_keys = [row["gap_key"] for row in _gaps(recorder.ledger_path)]
    assert len(gap_keys) == live_gap_count, "the live marker already covers that slot"


def test_only_completed_bars_are_ever_written(tmp_path):
    recorder = _recorder(tmp_path)
    # 06:47 mid-session: the 06:45-06:50 bar is still forming.
    summary = recorder.backfill_session_bars(
        [_row(0), _row(5), _row(10), _row(15)],
        session_date=SESSION,
        now=OPEN + timedelta(minutes=17),
    )

    assert summary["bars_added"] == 3
    assert [row["bar_end"][-14:] for row in _bars(recorder.ledger_path)] == [
        "06:35:00-07:00",
        "06:40:00-07:00",
        "06:45:00-07:00",
    ]


def test_fill_is_disabled_by_setting(tmp_path, monkeypatch):
    import vold_recorder

    recorder = _recorder(tmp_path)
    monkeypatch.setattr(vold_recorder, "get_local_setting", lambda key, default=None: False)

    summary = recorder.backfill_session_bars(
        [_row(minutes) for minutes in range(0, 390, 5)],
        session_date=SESSION,
        now=AFTER_CLOSE,
    )

    assert summary["ran"] is False
    assert "disabled" in summary["reason"]
    assert _bars(recorder.ledger_path) == []


def test_fill_is_due_at_the_close_and_only_once_per_session(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.observe([_row(0)], now=OPEN + timedelta(minutes=7))

    assert recorder.backfill_trigger(now=OPEN + timedelta(minutes=30)) == ""
    assert recorder.backfill_trigger(now=AFTER_CLOSE) == "close_of_day"

    recorder.backfill_session_bars(
        [_row(0)],
        session_date=SESSION,
        now=AFTER_CLOSE,
        trigger="close_of_day",
    )
    assert recorder.backfill_trigger(now=AFTER_CLOSE + timedelta(minutes=30)) == ""


def test_fill_marker_survives_a_restart(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.observe([_row(0)], now=OPEN + timedelta(minutes=7))
    recorder.backfill_session_bars(
        [_row(0)],
        session_date=SESSION,
        now=AFTER_CLOSE,
        trigger="close_of_day",
    )

    restarted = _recorder(tmp_path)
    assert restarted.backfill_trigger(now=AFTER_CLOSE + timedelta(minutes=5)) == ""


def test_fill_is_due_on_the_next_startup_when_the_close_was_missed(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.observe([_row(0)], now=OPEN + timedelta(minutes=7))

    assert recorder.backfill_trigger(now=NEXT_MORNING) == "startup_after_missed_close"


def test_recorder_adapter_widens_the_window_for_an_earlier_session():
    from bounce_bot_lib import legacy

    recorder = Mock()
    recorder.backfill_trigger.return_value = "startup_after_missed_close"
    recorder.session_date = SESSION
    recorder.backfill_session_bars.return_value = {"ran": True}

    bot = object.__new__(legacy.BounceBot)
    bot._vold_recorder = recorder
    bot._vold_contract = object()
    bot._request_historical_contract_bars = Mock(return_value=[_row(0)])

    legacy.BounceBot._backfill_breadth_bars_if_due(bot, NEXT_MORNING)

    _args, kwargs = bot._request_historical_contract_bars.call_args
    assert kwargs["duration"] == "2 D"
    _args, kwargs = recorder.backfill_session_bars.call_args
    assert kwargs["session_date"] == SESSION
    assert kwargs["trigger"] == "startup_after_missed_close"


def test_recorder_adapter_stays_quiet_when_nothing_is_due():
    from bounce_bot_lib import legacy

    recorder = Mock()
    recorder.backfill_trigger.return_value = ""

    bot = object.__new__(legacy.BounceBot)
    bot._vold_recorder = recorder
    bot._request_historical_contract_bars = Mock()

    legacy.BounceBot._backfill_breadth_bars_if_due(bot, OPEN + timedelta(minutes=30))

    bot._request_historical_contract_bars.assert_not_called()
    recorder.backfill_session_bars.assert_not_called()


def test_audit_counts_backfilled_breadth_bars_separately(tmp_path):
    from regime_collection_audit import audit_regime_collection, format_audit

    def _bar(index, capture_mode):
        start = OPEN + timedelta(minutes=5 * index)
        row = {
            "code_version": "vold_session_recorder_v1",
            "event_type": "breadth_bar",
            "session_date": SESSION,
            "bar_start": start.isoformat(timespec="seconds"),
            "bar_end": (start + timedelta(minutes=5)).isoformat(timespec="seconds"),
            "as_of": (start + timedelta(minutes=5)).isoformat(timespec="seconds"),
            "written_at": AFTER_CLOSE.isoformat(timespec="seconds"),
        }
        if capture_mode is not None:
            row["capture_mode"] = capture_mode
        return row

    breadth = tmp_path / "vold.jsonl"
    rows = [_bar(0, None), _bar(1, "live"), _bar(2, "backfill"), _bar(3, "backfill")]
    breadth.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (tmp_path / "technical.jsonl").write_text("", encoding="utf-8")

    report = audit_regime_collection(
        session_date=SESSION,
        technical_events_path=tmp_path / "technical.jsonl",
        breadth_events_path=breadth,
        now=AFTER_CLOSE,
    )

    assert report["breadth_recorder"]["backfilled_bar_count"] == 2
    assert "backfilled=2" in format_audit(report)

# --- bounded retry before the gap becomes permanent ------------------------
#
# The fill writes its data_gap rows *and* the per-session marker in one pass,
# and the marker stops the session from ever being repaired again. Finalising
# both off one failed request turned a transient hiccup into permanently
# missing breadth evidence (checkpoint review 2026-08-08 second review).


def _adapter_bot(fetch):
    from bounce_bot_lib import legacy

    recorder = Mock()
    recorder.backfill_trigger.return_value = "close_of_day"
    recorder.session_date = SESSION
    recorder.backfill_session_bars.return_value = {"ran": True}

    bot = object.__new__(legacy.BounceBot)
    bot._vold_recorder = recorder
    bot._vold_contract = object()
    bot._request_historical_contract_bars = Mock(side_effect=fetch)
    return legacy, bot, recorder


def test_a_transient_fetch_failure_is_retried_and_the_bars_still_arrive(monkeypatch):
    import durability_retry

    monkeypatch.setattr(durability_retry.time, "sleep", lambda seconds: None)
    calls = {"n": 0}

    def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("IBKR pacing violation")
        return [_row(0), _row(5)]

    legacy, bot, recorder = _adapter_bot(_flaky)
    legacy.BounceBot._backfill_breadth_bars_if_due(bot, AFTER_CLOSE)

    assert calls["n"] == 2
    args, _kwargs = recorder.backfill_session_bars.call_args
    assert len(args[0]) == 2, "the retried response must reach the recorder"


def test_exhausted_retries_still_hand_the_recorder_an_honest_empty_result(monkeypatch):
    import durability_retry

    monkeypatch.setattr(durability_retry.time, "sleep", lambda seconds: None)
    calls = {"n": 0}

    def _broken(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("IBKR disconnected")

    legacy, bot, recorder = _adapter_bot(_broken)
    legacy.BounceBot._backfill_breadth_bars_if_due(bot, AFTER_CLOSE)

    assert calls["n"] == 3, "retries must be bounded, never an open loop"
    args, _kwargs = recorder.backfill_session_bars.call_args
    assert args[0] == [], "exhaustion still records the gap honestly"
