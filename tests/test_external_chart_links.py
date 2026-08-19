"""The external chart deep link (WISHLIST item, built 2026-08-18).

Its wishlist prerequisite was explicit: confirm the URL scheme and the FAILURE
behaviour, with no scraping or browser-automation dependency. So the tests are
mostly about refusing:

- a symbol that is not a ticker never becomes a URL (the desk's symbol box is
  free text, and building a link out of unvalidated input is its own bug class);
- a template that cannot carry the symbol is a misconfiguration, not a reason
  to open last week's chart;
- an unknown timeframe falls back to the daily view rather than emitting an
  interval token nobody can predict.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import external_chart_links as links  # noqa: E402


class TestTheLink:
    def test_a_symbol_and_timeframe_become_a_chart_url(self):
        url = links.chart_url("aapl", "M5")
        assert url == "https://www.tradingview.com/chart/?symbol=AAPL&interval=5"

    def test_the_daily_view_is_the_default_and_the_fallback(self):
        assert links.chart_url("AAPL", "D1").endswith("interval=D")
        assert links.chart_url("AAPL", "sideways").endswith("interval=D")
        assert links.chart_url("AAPL").endswith("interval=D")

    def test_an_exchange_prefixed_symbol_is_allowed(self):
        assert "NASDAQ:AAPL" in links.chart_url("NASDAQ:AAPL", "D1")

    def test_free_text_never_becomes_a_url(self):
        for junk in ("", "   ", "not a ticker", "AAPL; rm -rf /", "<script>", "A" * 40):
            assert links.chart_url(junk, "D1") == ""

    def test_a_template_that_cannot_carry_the_symbol_is_refused(self):
        assert links.chart_url("AAPL", "D1", "https://example.com/chart") == ""

    def test_a_typo_in_the_template_is_the_traders_typo_not_a_guess(self):
        assert links.chart_url("AAPL", "D1", "https://x/{symbol}/{nope}") == ""

    def test_a_custom_template_is_honoured(self):
        url = links.chart_url("AAPL", "H1", "myapp://chart/{symbol}?tf={interval}")
        assert url == "myapp://chart/AAPL?tf=60"


class TestTheConfiguredTemplate:
    def test_it_falls_back_to_the_built_in_one(self, monkeypatch):
        import project_paths

        monkeypatch.setattr(project_paths, "get_local_setting", lambda *a, **k: "")
        assert links.configured_template() == links.DEFAULT_CHART_URL_TEMPLATE

    def test_the_traders_setting_wins(self, monkeypatch):
        import project_paths

        monkeypatch.setattr(
            project_paths, "get_local_setting", lambda *a, **k: "tc2000://chart/{symbol}"
        )
        assert links.configured_template() == "tc2000://chart/{symbol}"


class TestFailureIsReported:
    def test_an_unlinkable_symbol_says_so_and_opens_nothing(self, monkeypatch):
        opened, message = links.open_chart("not a ticker")
        assert opened is False
        assert "No external chart link" in message

    def test_a_refused_open_is_reported_rather_than_swallowed(self, monkeypatch):
        from PySide6.QtGui import QDesktopServices

        monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda _url: False))
        opened, message = links.open_chart("AAPL", "D1")
        assert opened is False
        assert "refused" in message

    def test_a_successful_open_names_what_it_opened(self, monkeypatch):
        from PySide6.QtGui import QDesktopServices

        monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda _url: True))
        opened, message = links.open_chart("AAPL", "D1")
        assert opened is True
        assert "tradingview.com" in message
