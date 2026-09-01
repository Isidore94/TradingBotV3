"""Packet 3 item 2: the Industry Board obeys quiet hours, and chunks its download.

`industry_board_service` was the ONLY recurring downloader with no
`auto_scanning_due` gate, so its ~1,930-ticker nine-month `yf.download` ran
hourly all night and fired about five seconds after every desk launch, at any
hour. Quiet hours confine automatic starters; the manual "Refresh Board
(yfinance)" button is never gated, and these pin both halves of that.

The download itself handed every ticker to yfinance in ONE call. It is chunked
now, the way the strength board already chunks its own, and a chunk that fails
costs that chunk instead of the whole board. The frames that come back are the
same frames.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import industry_scanner  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _service(tmp_path, runner):
    from ui.services.industry_board_service import IndustryBoardService

    return IndustryBoardService(
        scan_runner=runner,
        sector_path=tmp_path / "sector.csv",
        industry_path=tmp_path / "industry.csv",
        state_path=tmp_path / "state.json",
    )


def _counting_runner(calls):
    def _run(**kwargs):
        calls.append(kwargs)
        return {"sector_rows": [{}], "industry_rows": [{}], "symbol_count": 1}

    return _run


class TestQuietHours:
    def test_the_automatic_tick_downloads_nothing_in_quiet_hours(self, tmp_path, monkeypatch):
        import autopilot_core as core

        calls = []
        service = _service(tmp_path, _counting_runner(calls))
        monkeypatch.setattr(core, "auto_scanning_due", lambda now=None: (False, "quiet hours"))

        assert service.refresh_if_due() is False
        assert calls == [], "a 1,930-ticker download must not run at 3am"

    def test_the_automatic_tick_runs_inside_the_session_window(self, tmp_path, monkeypatch):
        import autopilot_core as core

        calls = []
        service = _service(tmp_path, _counting_runner(calls))
        monkeypatch.setattr(core, "auto_scanning_due", lambda now=None: (True, ""))

        assert service.refresh_if_due() is True
        service._refresh_thread.join(10)
        assert len(calls) == 1

    def test_the_manual_button_is_never_gated(self, tmp_path, monkeypatch):
        """Quiet hours confine automatic starters, never the trader's click."""
        import autopilot_core as core

        calls = []
        service = _service(tmp_path, _counting_runner(calls))
        monkeypatch.setattr(core, "auto_scanning_due", lambda now=None: (False, "quiet hours"))

        assert service.request_refresh(force=True) is True
        service._refresh_thread.join(10)
        assert len(calls) == 1

    def test_the_gate_fails_open(self, tmp_path, monkeypatch):
        """A broken clock must not silently stop the board forever."""
        import autopilot_core as core

        service = _service(tmp_path, _counting_runner([]))
        monkeypatch.setattr(
            core, "auto_scanning_due", lambda now=None: (_ for _ in ()).throw(RuntimeError("clock"))
        )

        assert service._automatic_refresh_allowed(datetime.now()) is True


class TestTheDownloadIsChunked:
    def test_the_chunked_result_equals_the_unchunked_one(self, monkeypatch):
        pd = pytest.importorskip("pandas")

        tickers = [f"SYM{index:03d}" for index in range(450)]

        def _fake_chunk(chunk, *, period):
            return {
                symbol: pd.DataFrame({"datetime": [1], "close": [float(len(symbol))]})
                for symbol in chunk
            }

        seen = []

        def _recording(chunk, *, period):
            seen.append(list(chunk))
            return _fake_chunk(chunk, period=period)

        monkeypatch.setattr(industry_scanner, "_fetch_daily_frames_chunk", _recording)
        frames = industry_scanner.fetch_daily_frames_yf(tickers)

        assert sorted(frames) == sorted(tickers)
        assert len(seen) == 3, "450 symbols at 200 per call"
        assert sum(len(chunk) for chunk in seen) == len(tickers)
        assert all(len(chunk) <= industry_scanner.FETCH_CHUNK_SIZE for chunk in seen)

    def test_a_failed_chunk_costs_that_chunk_and_not_the_board(self, monkeypatch):
        pd = pytest.importorskip("pandas")

        tickers = [f"SYM{index:03d}" for index in range(300)]

        def _flaky(chunk, *, period):
            if chunk[0] == "SYM000":
                raise RuntimeError("yfinance said no")
            return {symbol: pd.DataFrame({"datetime": [1], "close": [1.0]}) for symbol in chunk}

        monkeypatch.setattr(industry_scanner, "_fetch_daily_frames_chunk", _flaky)
        frames = industry_scanner.fetch_daily_frames_yf(tickers)

        assert frames, "the surviving chunk is still returned"
        assert len(frames) == 100

    def test_no_symbols_is_no_call(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            industry_scanner,
            "_fetch_daily_frames_chunk",
            lambda chunk, *, period: called.append(chunk) or {},
        )
        assert industry_scanner.fetch_daily_frames_yf([]) == {}
        assert called == []
