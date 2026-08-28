"""The group tape's own data owner (plan.md Phase 0.5 item 11, packet T-2).

The tape's defect was never the formula - it was the clock. So most of what is
pinned here is about WHEN and WHERE the work happens: one batched request per
tick and never a retry inside it (Yahoo rate-limits bursts), the fetch on a
worker and never on the Qt thread, the quiet-hours gate that every automatic
starter obeys, a failed refresh that keeps the last good tape and says it
failed, and a shutdown that is bounded.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the service is a QObject with a QTimer")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

SESSION_OPEN = datetime(2026, 8, 27, 6, 30)


def _frame(closes, *, start=SESSION_OPEN, spread=0.4):
    """A real pandas frame, because `_frame_rows` is the real reader."""
    import pandas as pd

    index = pd.DatetimeIndex(
        [start + timedelta(minutes=5 * i) for i in range(len(closes))]
    )
    return pd.DataFrame(
        {
            "Open": [c - spread / 3 for c in closes],
            "High": [c + spread * (1 + (i % 3) * 0.25) for i, c in enumerate(closes)],
            "Low": [c - spread * (1 + (i % 3) * 0.25) for i, c in enumerate(closes)],
            "Close": list(closes),
            "Volume": [1000.0 + i for i in range(len(closes))],
        },
        index=index,
    )


def _rising(count, base=100.0, step=0.2):
    return [base + step * i for i in range(count)]


def _choppy(count, base=50.0):
    return [base + (i % 5) * 0.3 - (i % 3) * 0.17 for i in range(count)]


class _Downloader:
    """Records every call so "exactly one request per tick" is measurable."""

    def __init__(self, *, bars=24, fail=False, block: threading.Event | None = None):
        self.calls: list[dict] = []
        self.threads: list[str] = []
        self._bars = bars
        self._fail = fail
        self._block = block

    def __call__(self, symbols, *, period, interval):
        self.calls.append({"symbols": list(symbols), "period": period, "interval": interval})
        self.threads.append(threading.current_thread().name)
        if self._block is not None:
            self._block.wait(10.0)
        if self._fail:
            raise RuntimeError("YFRateLimitError: too many requests")
        out = {}
        for index, symbol in enumerate(symbols):
            closes = _choppy(self._bars) if symbol == "SPY" else _rising(
                self._bars, base=80.0 + index, step=0.2 + 0.01 * index
            )
            out[symbol] = _frame(closes)
        return out


@pytest.fixture
def service_factory():
    made = []

    def build(**kwargs):
        from ui.services.group_tape_service import GroupTapeService

        service = GroupTapeService(**kwargs)
        made.append(service)
        return service

    yield build
    for service in made:
        try:
            service.shutdown()
            service.deleteLater()
        except Exception:
            pass


def _run_and_wait(service, timeout=10.0):
    """Kick a refresh and let the worker finish."""
    started = service.refresh_now()
    deadline = time.monotonic() + timeout
    while service.running and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)
    for _ in range(6):
        _app.processEvents()
    return started


# ------------------------------------------------------------------- the build


def test_one_batched_request_carries_spy_and_every_group():
    """61-ish symbols in ONE call. Yahoo rate-limits bursts: a diagnostic run
    hit YFRateLimitError on the 12th single-ticker call."""
    from ui.services.group_tape_service import build_group_tape
    import group_rrs

    downloader = _Downloader()
    build_group_tape(
        downloader=downloader,
        now=SESSION_OPEN + timedelta(minutes=5 * 24),
        industry_etfs=["URA", "XBI", "SMH"],
        industry_note="",
    )

    assert len(downloader.calls) == 1, "one request, and never a retry inside the tick"
    call = downloader.calls[0]
    assert call["period"] == "1d", "today only - a longer period is the overnight gap"
    assert call["interval"] == "5m"
    assert call["symbols"][0] == "SPY"
    assert set(group_rrs.SECTOR_ETFS.values()) <= set(call["symbols"])
    assert {"URA", "XBI", "SMH"} <= set(call["symbols"])
    assert len(call["symbols"]) == len(set(call["symbols"])), "no symbol fetched twice"


def test_the_payload_carries_three_windows_ranked_by_the_thirty_minute_read():
    from ui.services.group_tape_service import build_group_tape

    payload = build_group_tape(
        downloader=_Downloader(bars=30),
        now=SESSION_OPEN + timedelta(minutes=5 * 30),
        industry_etfs=["URA", "XBI"],
        industry_note="",
    )

    groups = payload["group_strength"]
    assert set(groups) == {"30", "60", "90"}
    for label in ("30", "60", "90"):
        assert groups[label]["sectors"], f"{label} should have filled by 30 bars"
        rrs = [row["rrs"] for row in groups[label]["sectors"]]
        assert rrs == sorted(rrs, reverse=True), "strongest first"
        assert all({"group_key", "etf", "rrs"} <= set(row) for row in groups[label]["sectors"])
    assert {row["etf"] for row in groups["30"]["industries"]} == {"URA", "XBI"}
    assert payload["source"] == "yfinance"
    assert payload["as_of_text"], "the as-of has to be visible; a stale tape must not look current"


def test_an_unmeasurable_window_is_absent_rather_than_zero():
    """UNKNOWN never invented. 30 needs 8 bars, 60 needs 14, 90 needs 20."""
    from ui.services.group_tape_service import build_group_tape

    payload = build_group_tape(
        downloader=_Downloader(bars=14),
        now=SESSION_OPEN + timedelta(minutes=5 * 14),
        industry_etfs=[],
        industry_note="",
    )
    groups = payload["group_strength"]
    assert groups["30"]["sectors"], "30 can answer at 14 bars"
    assert groups["60"]["sectors"], "60 can answer at exactly 14 bars"
    assert groups["90"]["sectors"] == [], "90 needs 20 and must stay blank, not 0.0"
    assert "90 min still filling" in payload["status"]


def test_no_spy_bars_today_is_said_out_loud_rather_than_shown_as_an_empty_tape():
    from ui.services.group_tape_service import build_group_tape

    payload = build_group_tape(
        downloader=_Downloader(bars=24),
        # A clock a full day after the bars: nothing is "today" any more.
        now=SESSION_OPEN + timedelta(days=1),
        industry_etfs=["URA"],
        industry_note="",
    )
    assert payload["group_strength"]["30"]["sectors"] == []
    assert "SPY" in payload["status"]
    assert payload["as_of"] is None


def test_a_missing_industry_map_means_sectors_only_and_says_so(tmp_path):
    from ui.services.group_tape_service import build_group_tape, load_industry_etfs

    etfs, note = load_industry_etfs(tmp_path / "not_here.json")
    assert etfs == []
    assert "sectors only" in note

    payload = build_group_tape(
        downloader=_Downloader(bars=24),
        now=SESSION_OPEN + timedelta(minutes=5 * 24),
        industry_etfs=etfs,
        industry_note=note,
    )
    assert payload["group_strength"]["30"]["sectors"], "sectors still work"
    assert payload["group_strength"]["30"]["industries"] == []
    assert "sectors only" in payload["status"], "two thirds of the chips gone must never be silent"


def test_the_industry_map_is_read_the_way_the_real_one_is_shaped(tmp_path):
    """The live map holds 136 industries, 70 of them with `etf: null`, and 49
    distinct proxies - several industries share one ETF. Both of those are
    parsing hazards, so the fixture reproduces them rather than a tidy map.
    (Measured on the desk 2026-08-27; under pytest `conftest` redirects the
    data dir, so the real file is deliberately not what is read here.)"""
    import json

    from ui.services.group_tape_service import load_industry_etfs

    path = tmp_path / "industry_etf_map.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "yahoo_industryKey_to_ref": {
                    "advertising-agencies": {"etf": None, "sectorKey": "communication-services"},
                    "uranium": {"etf": "URA", "sectorKey": "energy"},
                    "biotechnology": {"etf": "XBI", "sectorKey": "healthcare"},
                    "drug-manufacturers-general": {"etf": "XBI", "sectorKey": "healthcare"},
                    "semiconductors": {"etf": " smh ", "sectorKey": "technology"},
                    "broken-row": "not a mapping",
                },
            }
        ),
        encoding="utf-8",
    )

    etfs, note = load_industry_etfs(path)
    assert note == ""
    assert etfs == ["SMH", "URA", "XBI"], "deduped, upper-cased, trimmed, None dropped"


def test_an_unparseable_industry_map_is_sectors_only_rather_than_a_crash(tmp_path):
    from ui.services.group_tape_service import load_industry_etfs

    broken = tmp_path / "industry_etf_map.json"
    broken.write_text("{not json", encoding="utf-8")
    etfs, note = load_industry_etfs(broken)
    assert etfs == []
    assert "sectors only" in note

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"version": 1}), encoding="utf-8")
    etfs, note = load_industry_etfs(empty)
    assert etfs == []
    assert "sectors only" in note


# ----------------------------------------------------------------- the service


@pytest.fixture
def frozen_session_clock(monkeypatch):
    """Freeze the service's clock to the session `SESSION_OPEN` belongs to.

    The tape drops everything outside TODAY's date - that same-date filter is
    the whole point of the rebuild, because a window without it reaches over
    the overnight gap. The fixture bars are pinned to 2026-08-27, so from
    2026-08-28 onwards the service's real `datetime.now()` filtered every one
    of them out and two tests began failing on the calendar rather than on the
    code. Freezing the clock beside the bars is what makes them agree; letting
    the bars float with the real clock cannot work, because near midnight there
    is not yet a session long enough to hold a 30-minute window.

    Patched on the service module only. `build_group_tape` resolves `datetime`
    from its own globals and passes the moment down explicitly, so one seam
    covers the whole path.
    """
    from ui.services import group_tape_service

    frozen = SESSION_OPEN + timedelta(hours=5)  # midday, inside the scan window

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen if tz is None else frozen.astimezone(tz)

    monkeypatch.setattr(group_tape_service, "datetime", _Clock)
    return frozen


def test_the_fetch_never_runs_on_the_qt_thread(service_factory, frozen_session_clock):
    """Hard rule 2. The download is a network read; on the GUI thread it is a
    multi-second freeze every five minutes."""
    downloader = _Downloader()
    service = service_factory(downloader=downloader)
    _run_and_wait(service)

    assert downloader.threads, "the downloader must actually have run"
    assert all(name != threading.main_thread().name for name in downloader.threads)
    assert service.payload()["group_strength"]["30"]["sectors"]


def test_exactly_one_download_per_refresh(service_factory):
    downloader = _Downloader()
    service = service_factory(downloader=downloader)
    _run_and_wait(service)
    assert len(downloader.calls) == 1
    _run_and_wait(service)
    assert len(downloader.calls) == 2, "a second refresh is a second request, not a retry"


def test_a_second_refresh_while_one_is_running_is_refused_not_queued(service_factory):
    gate = threading.Event()
    downloader = _Downloader(block=gate)
    service = service_factory(downloader=downloader)
    try:
        assert service.refresh_now() is True
        deadline = time.monotonic() + 5.0
        while not downloader.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.refresh_now() is False, "single flight"
    finally:
        gate.set()
        deadline = time.monotonic() + 10.0
        while service.running and time.monotonic() < deadline:
            time.sleep(0.01)
    assert len(downloader.calls) == 1


def test_a_failed_download_keeps_the_last_good_tape_and_states_the_failure(
    service_factory, frozen_session_clock
):
    """plan.md sec 5: a failed publish never destroys the last verified one -
    and a stale tape that looks current is worse than no tape."""
    good = _Downloader()
    service = service_factory(downloader=good)
    _run_and_wait(service)
    kept = service.payload()
    assert kept["group_strength"]["30"]["sectors"]

    service._downloader = _Downloader(fail=True)
    _run_and_wait(service)

    assert service.payload()["group_strength"] == kept["group_strength"], "last good survives"
    status = service.status_text()
    assert "FAILED" in status
    assert "YFRateLimitError" in status


def test_the_quiet_hours_gate_holds_and_the_manual_refresh_ignores_it(
    service_factory, monkeypatch
):
    """Hard rule 6: every automatic starter is gated; manual buttons never are."""
    import autopilot_core as core

    downloader = _Downloader()
    service = service_factory(downloader=downloader)
    monkeypatch.setattr(core, "auto_scanning_due", lambda now: (False, "outside session"))

    service._tick()
    assert downloader.calls == [], "the timer must not fetch overnight"

    _run_and_wait(service)
    assert len(downloader.calls) == 1, "the trader asking for it is not automatic work"


def test_the_gate_fails_open_when_the_session_cannot_be_looked_up(
    service_factory, monkeypatch
):
    import autopilot_core as core

    def boom(now):
        raise RuntimeError("no calendar")

    downloader = _Downloader()
    service = service_factory(downloader=downloader)
    monkeypatch.setattr(core, "auto_scanning_due", boom)
    assert service._due(datetime(2026, 8, 27, 8, 0)) is True


def test_the_cadence_is_five_minutes_between_automatic_refreshes(service_factory, monkeypatch):
    import autopilot_core as core

    service = service_factory(downloader=_Downloader())
    monkeypatch.setattr(core, "auto_scanning_due", lambda now: (True, "open"))
    now = datetime(2026, 8, 27, 8, 0)
    service._last_attempt = now
    assert service._due(now + timedelta(minutes=4)) is False
    assert service._due(now + timedelta(minutes=5)) is True


def test_shutdown_is_bounded_even_with_a_fetch_in_flight(service_factory, monkeypatch):
    """A wait with no bound is a hang waiting for a slow day - the lesson the
    GC controller and the 2026-08-26 shutdown freeze both paid for."""
    from ui.services import group_tape_service

    monkeypatch.setattr(group_tape_service, "SHUTDOWN_JOIN_SECONDS", 0.2)
    gate = threading.Event()
    downloader = _Downloader(block=gate)
    service = service_factory(downloader=downloader)
    try:
        service.refresh_now()
        deadline = time.monotonic() + 5.0
        while not downloader.calls and time.monotonic() < deadline:
            time.sleep(0.01)

        started = time.monotonic()
        service.shutdown()
        elapsed = time.monotonic() - started
        assert elapsed < 2.0, f"shutdown waited {elapsed:.1f}s on a stuck fetch"
    finally:
        gate.set()


def test_status_before_the_first_refresh_admits_it_has_no_read(service_factory):
    service = service_factory(downloader=_Downloader())
    assert "never refreshed" in service.status_text()
    assert service.payload()["group_strength"]["30"]["sectors"] == []
