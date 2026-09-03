"""R4 A14 - the one Refresh click reads nothing, and Discovery actually builds.

Two defects under one item:

* `WalkawayPage._reload_review_data` called `journal_feed.week_trades`
  SYNCHRONOUSLY - a journal query over a week of trades, measured at 775 ms of
  the click, on the thread that draws. The V2 packet's 50 ms rule was true of
  every other page and false of this one, and its test could not see it because
  it stubbed the `reload` METHODS away: with every reload replaced by an
  `append`, the click measured the `for` loop.
* `DiscoveryPage` had no `reload`, so `refresh_everything` called the base
  class's no-op, counted the step as started and built nothing - while its six
  per-table Refresh buttons stayed in the layout, which is exactly the
  complaint V2 was answering. The V2 test missed them because it looked for a
  `refresh_button` attribute this page does not have.

The click is measured HERE with the reads stubbed at the WORKER boundary - the
functions the workers call - so every real `reload` runs.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _stub_the_reads(monkeypatch, *, slow_ms: int = 0):
    """Replace what the WORKERS call, never the reload methods themselves."""
    from ui.panels import weekend_prep_panel as panel_module

    def _slow(result):
        def _work(*_args, **_kwargs):
            if slow_ms:
                time.sleep(slow_ms / 1000.0)
            return result

        return _work

    monkeypatch.setattr(
        panel_module.journal_feed,
        "week_trades",
        _slow({"closed": [], "still_open": []}),
    )
    monkeypatch.setattr(panel_module.journal_feed, "week_tag_candidates", _slow([]))
    monkeypatch.setattr(panel_module.journal_feed, "pending_tag_candidates", _slow([]))
    monkeypatch.setattr(panel_module, "_read_like_cohort", _slow([]))
    monkeypatch.setattr(panel_module, "_read_veto_cohort", _slow([]))
    monkeypatch.setattr(panel_module, "_read_week_trades", _slow([]))
    monkeypatch.setattr(panel_module, "_read_awaiting_review", _slow(0))
    monkeypatch.setattr(panel_module, "_read_research_pack", _slow({}))


def test_the_click_reads_nothing_even_with_every_real_reload_running(qapp, monkeypatch):
    """A 200 ms read on any worker must cost the click nothing.

    Each stubbed read sleeps 200 ms. Five pages plus the verdict is over a
    second of work; if any of it were on the Qt thread this click could not come
    in under 50 ms.
    """
    from ui.panels import weekend_prep_panel as panel_module

    _stub_the_reads(monkeypatch, slow_ms=200)
    panel = panel_module.WeekendPrepPanel()
    # The two remaining live starts on this click are a market-history run and
    # the strength boards; both already own threads, and neither is A14's
    # subject.
    monkeypatch.setattr(panel_module, "_WalkawayWorker", lambda *a, **k: _NullWorker())
    monkeypatch.setattr(panel.service, "refresh_board", lambda *a, **k: True)
    monkeypatch.setattr(panel.service, "refresh_week_ahead", lambda *a, **k: True)

    began = time.perf_counter()
    panel.refresh_everything()
    elapsed_ms = (time.perf_counter() - began) * 1000.0

    assert elapsed_ms < 50.0, f"the click itself took {elapsed_ms:.1f} ms"
    assert "Building:" in panel.building_note.text()
    panel.shutdown()


class _NullWorker:
    """Stands in for the walk-away thread: startable, and it does nothing."""

    def __init__(self, *_args, **_kwargs) -> None:
        class _Signal:
            def connect(self, *_a, **_k):
                return None

        self.finished_with = _Signal()
        self.failed = _Signal()

    def isRunning(self) -> bool:
        return False

    def start(self) -> None:
        return None


def test_the_weeks_trades_are_read_on_a_worker(qapp):
    """The 775 ms. Asserted on the seam, because the read is now asynchronous."""
    source = (ROOT / "scripts" / "ui" / "panels" / "weekend_prep_panel.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def _reload_review_data(", 1)[1].split("\n    def ", 1)[0]

    assert "_ReadWorker(" in body
    assert "setText" not in body, "the reload must touch no widget with a read's result"
    # And the widget update happens in the signal seam, not in the reload.
    ready = source.split("def _on_week_trades_ready(", 1)[1].split("\n    def ", 1)[0]
    assert "self.open_note.setText" in ready


def test_a_journal_failure_still_names_itself_on_the_page(qapp, monkeypatch):
    from ui.panels import weekend_prep_panel as panel_module

    panel = panel_module.WeekendPrepPanel()
    page = panel._pages["walkaway"]

    page._on_week_trades_failed("store gone")

    assert "Journal unavailable: store gone" in page.open_note.text()
    panel.shutdown()


def test_the_week_note_renders_from_the_worker_payload(qapp):
    from types import SimpleNamespace

    from ui.panels import weekend_prep_panel as panel_module

    panel = panel_module.WeekendPrepPanel()
    page = panel._pages["walkaway"]

    page._on_week_trades_ready(
        {"closed": [1, 2, 3], "still_open": [SimpleNamespace(symbol="NVDA")]}
    )

    text = page.open_note.text()
    assert "3 closed this week." in text
    assert "still open: NVDA" in text
    assert "flagged, not counted" in text
    panel.shutdown()


def test_discovery_actually_reloads_every_board(qapp, monkeypatch):
    """It had no reload at all, so one Refresh counted it and built nothing."""
    import weekend_strength
    from ui.panels import weekend_prep_panel as panel_module

    panel = panel_module.WeekendPrepPanel()
    page = panel._pages["discovery"]
    asked: list[tuple] = []
    monkeypatch.setattr(
        panel.service,
        "refresh_board",
        lambda timeframe, side=None: asked.append((timeframe, side)) or True,
    )

    page.reload()

    expected = [
        (timeframe.key, side)
        for timeframe in weekend_strength.TIMEFRAMES
        for side in ("long", "short")
    ]
    assert asked == expected
    assert len(asked) == 6
    panel.shutdown()


def test_discovery_is_in_the_building_list_because_it_now_builds(qapp, monkeypatch):
    from ui.panels import weekend_prep_panel as panel_module

    _stub_the_reads(monkeypatch)
    panel = panel_module.WeekendPrepPanel()
    monkeypatch.setattr(panel_module, "_WalkawayWorker", lambda *a, **k: _NullWorker())
    asked: list[tuple] = []
    monkeypatch.setattr(
        panel.service,
        "refresh_board",
        lambda timeframe, side=None: asked.append((timeframe, side)) or True,
    )
    monkeypatch.setattr(panel.service, "refresh_week_ahead", lambda *a, **k: True)

    panel.refresh_everything()

    assert len(asked) == 6, "one Refresh must build Discovery, not just name it"
    assert panel_module.STEP_LABELS["discovery"] in panel.building_note.text()
    panel.shutdown()


def test_the_six_per_table_refresh_buttons_are_out_of_the_layout(qapp):
    """Six buttons on one page is what the trader complained about.

    Checked against the BUTTONS this page builds rather than against a
    `refresh_button` attribute - looking for an attribute Discovery does not
    have is how the V2 test passed while six buttons sat on the screen.
    """
    from ui.panels import weekend_prep_panel as panel_module

    panel = panel_module.WeekendPrepPanel()
    page = panel._pages["discovery"]

    assert len(page._refresh_buttons) == 6
    for button in page._refresh_buttons:
        assert button.parent() is None, button.text()
    # The Adopt button is a verb, not a refresh, and it stays on the page.
    assert any(
        isinstance(child, panel_module.QPushButton) and "Adopt" in child.text()
        for child in page.findChildren(panel_module.QPushButton)
    )
    panel.shutdown()
