"""Today's swing picks: the service's two writes, the strip, and where it lives.

Trader, 2026-08-31: *"at the end of the day I have a list of my top swing
targets. I want a place to put them in so the bot knows my personal favourite
picks. They will usually become focus picks too but these ones get special
standing because I picked them by hand... put it at the very bottom of the M5
alerts tab, the tab is so long and I never use all of it."*

What these defend: an add writes ONE evidence row and a swing Focus entry with
NO auto-adoption marker (absence of a marker is what keeps automatic removal
off the trader's own names); a removal appends a retraction and drops the Focus
entry while the original row stays; the "taken" mark is display-only and shows
nothing when it cannot measure; the strip is DIFFED rather than rebuilt; and
the M5 alert bar keeps its column position, its routing and its behaviour.
"""

from __future__ import annotations

import json
import os
import sys
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

import swing_favorites  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def focus(tmp_path):
    from focus_picks import FocusPickStore
    from ui.services.focus_service import FocusService

    return FocusService(
        FocusPickStore(
            focus_longs_path=tmp_path / "focus_longs.txt",
            focus_shorts_path=tmp_path / "focus_shorts.txt",
            longs_path=tmp_path / "longs.txt",
            shorts_path=tmp_path / "shorts.txt",
            membership_path=tmp_path / "membership.json",
        )
    )


@pytest.fixture
def service(tmp_path, focus):
    from ui.services.swing_favorites_service import SwingFavoritesService

    made = SwingFavoritesService(
        focus,
        path=tmp_path / "swing_favorites.jsonl",
        trades_provider=lambda session_date, days: [],
    )
    yield made
    made.shutdown()


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestTheTwoWrites:
    def test_an_add_writes_one_row_and_one_trader_owned_focus_entry(self, service, focus, tmp_path):
        assert service.add("NVDA", "long") == ["NVDA"]

        rows = _rows(tmp_path / "swing_favorites.jsonl")
        assert len(rows) == 1
        assert (rows[0]["symbol"], rows[0]["side"], rows[0]["action"]) == ("NVDA", "long", "add")
        assert rows[0]["origin"] == "trader"

        assert focus.is_focus("NVDA", "long", "swing") is True
        assert focus.is_auto_adopted("NVDA", "long", "swing") is False, (
            "absence of a marker is what makes the entry the trader's"
        )
        assert focus.store.auto_pick_marker("NVDA", "long", "swing") is None
        assert focus.store.auto_pick_markers() == {}

    def test_the_pick_carries_its_own_like_origin(self, service, focus, monkeypatch):
        """So the human-focus tracker grades these as `human_focus_swing_vetted`
        rather than mixing them with every other hand-typed swing name - which
        is what makes "how do MY picks do?" answerable at all."""
        from ui.services import swing_favorites_service as module

        seen: list[dict] = []
        original = focus.add
        monkeypatch.setattr(
            focus,
            "add",
            lambda symbol, side, category="m5", **kwargs: (
                seen.append({"category": category, **kwargs})
                or original(symbol, side, category, **kwargs)
            ),
        )
        service.add("NVDA", "long")
        assert seen == [{"category": "swing", "origin": module.FOCUS_LIKE_ORIGIN}]
        assert module.FOCUS_LIKE_ORIGIN == "vetted"

    def test_the_like_origin_is_a_documented_one(self):
        import pick_feedback
        from ui.services import swing_favorites_service as module

        assert module.FOCUS_LIKE_ORIGIN in pick_feedback.PICK_ORIGINS

    def test_the_pick_does_not_land_in_the_m5_category(self, service, focus):
        service.add("NVDA", "long")
        assert focus.is_focus("NVDA", "long", "m5") is False

    def test_a_paste_places_every_name_in_order(self, service, tmp_path):
        assert service.add("nvda amd\ntsla", "short") == ["NVDA", "AMD", "TSLA"]
        rows = _rows(tmp_path / "swing_favorites.jsonl")
        assert [row["symbol"] for row in rows] == ["NVDA", "AMD", "TSLA"]
        assert {row["side"] for row in rows} == {"short"}

    def test_a_repeat_add_writes_no_second_row(self, service, tmp_path):
        service.add("NVDA", "long")
        assert service.add("NVDA", "long") == []
        assert len(_rows(tmp_path / "swing_favorites.jsonl")) == 1

    def test_a_removal_retracts_and_leaves_the_original_row(self, service, focus, tmp_path):
        service.add("NVDA", "long")
        first = _rows(tmp_path / "swing_favorites.jsonl")[0]

        assert service.remove("NVDA", "long") is True

        rows = _rows(tmp_path / "swing_favorites.jsonl")
        assert len(rows) == 2
        assert rows[0] == first, "nothing is rewritten"
        assert rows[1]["action"] == "remove"
        assert focus.is_focus("NVDA", "long", "swing") is False
        assert service.favorites() == []

    def test_removing_a_name_that_is_not_on_the_list_writes_nothing(self, service, tmp_path):
        assert service.remove("NVDA", "long") is False
        assert _rows(tmp_path / "swing_favorites.jsonl") == []

    def test_a_sideless_add_does_nothing(self, service, tmp_path):
        assert service.add("NVDA", "sideways") == []
        assert _rows(tmp_path / "swing_favorites.jsonl") == []

    def test_a_lost_evidence_row_never_costs_the_pick(self, tmp_path, focus):
        """An evidence store is never allowed to cost the thing it records."""
        from ui.services.swing_favorites_service import SwingFavoritesService

        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")
        made = SwingFavoritesService(
            focus,
            path=blocked / "swing_favorites.jsonl",
            trades_provider=lambda session_date, days: [],
        )
        complaints: list[str] = []
        made.statusChanged.connect(complaints.append)
        try:
            assert made.add("NVDA", "long") == ["NVDA"]
            assert focus.is_focus("NVDA", "long", "swing") is True, "the pick stands"
            assert complaints and "favorites log" in complaints[0]
        finally:
            made.shutdown()


class TestTheTakenMark:
    def test_a_journal_trade_on_the_pick_day_marks_it(self, tmp_path, focus):
        from ui.services.swing_favorites_service import SwingFavoritesService

        today = swing_favorites.current_session_date()
        made = SwingFavoritesService(
            focus,
            path=tmp_path / "swing_favorites.jsonl",
            trades_provider=lambda session_date, days: [
                {"symbol": "NVDA", "opened_at": f"{today}T09:41:00-04:00"}
            ],
        )
        try:
            made.add("NVDA AMD", "long")
            made.refresh_taken(blocking=True)
            assert made.taken() == {("NVDA", "long")}
        finally:
            made.shutdown()

    def test_an_unreadable_journal_marks_nothing(self, tmp_path, focus):
        """Unmeasurable shows nothing; it never raises into the desk."""
        from ui.services.swing_favorites_service import SwingFavoritesService

        def _boom(session_date, days):
            raise RuntimeError("journal is mid-migration")

        made = SwingFavoritesService(
            focus, path=tmp_path / "swing_favorites.jsonl", trades_provider=_boom
        )
        try:
            made.add("NVDA", "long")
            made.refresh_taken(blocking=True)
            assert made.taken() == set()
        finally:
            made.shutdown()

    def test_the_journal_is_asked_for_a_bounded_window(self, tmp_path, focus):
        from ui.services.swing_favorites_service import SwingFavoritesService

        asked: list[tuple[str, int]] = []

        def _record(session_date, days):
            asked.append((session_date, days))
            return []

        made = SwingFavoritesService(
            focus, path=tmp_path / "swing_favorites.jsonl", trades_provider=_record
        )
        try:
            made.add("NVDA", "long")
            made.refresh_taken(blocking=True)
            assert asked and asked[-1][1] == swing_favorites.TAKEN_LOOKBACK_DAYS
        finally:
            made.shutdown()

    def test_an_empty_list_asks_the_journal_nothing(self, tmp_path, focus):
        from ui.services.swing_favorites_service import SwingFavoritesService

        asked: list[tuple[str, int]] = []
        made = SwingFavoritesService(
            focus,
            path=tmp_path / "swing_favorites.jsonl",
            trades_provider=lambda session_date, days: asked.append((session_date, days)) or [],
        )
        try:
            made.refresh_taken(blocking=True)
            assert asked == []
        finally:
            made.shutdown()

    def test_the_default_provider_never_prepares_the_journal(self, monkeypatch):
        """A display badge must not be the thing that migrates a schema."""
        from ui.services import swing_favorites_service

        calls: list[str] = []
        fake = type(
            "Feed",
            (),
            {
                "store_is_initialized": staticmethod(lambda: False),
                "store_needs_preparation": staticmethod(lambda: True),
                "load_trades": staticmethod(
                    lambda **kwargs: calls.append("loaded") or []
                ),
            },
        )
        monkeypatch.setitem(sys.modules, "ui.services.journal_feed", fake)
        assert swing_favorites_service.default_trades_provider("2026-08-31", 10) == []
        assert calls == [], "the store was never opened"


class TestTheStrip:
    def _bar(self):
        from ui.widgets.swing_favorites_bar import SwingFavoritesBar

        return SwingFavoritesBar()

    def test_it_shows_the_picks_in_order(self):
        bar = self._bar()
        bar.set_favorites([
            {"symbol": "NVDA", "side": "long"},
            {"symbol": "AMD", "side": "short"},
        ])
        assert bar.symbols() == [("NVDA", "long"), ("AMD", "short")]
        assert bar.count_label.text() == "2"

    def test_an_unchanged_list_does_no_layout_work(self):
        """Diff, never rebuild - the chips must be the SAME widgets."""
        bar = self._bar()
        payload = [{"symbol": "NVDA", "side": "long"}, {"symbol": "AMD", "side": "long"}]
        bar.set_favorites(payload)
        before = bar._current_chips()
        bar.set_favorites(payload)
        assert bar._current_chips() == before

    def test_one_arrival_keeps_the_existing_chips(self):
        bar = self._bar()
        bar.set_favorites([{"symbol": "NVDA", "side": "long"}])
        first = bar._current_chips()[0]
        bar.set_favorites([
            {"symbol": "NVDA", "side": "long"},
            {"symbol": "AMD", "side": "long"},
        ])
        assert bar._current_chips()[0] is first
        assert bar.symbols() == [("NVDA", "long"), ("AMD", "long")]

    def test_a_departure_removes_only_that_chip(self):
        bar = self._bar()
        bar.set_favorites([
            {"symbol": "NVDA", "side": "long"},
            {"symbol": "AMD", "side": "long"},
        ])
        kept = bar._current_chips()[1]
        bar.set_favorites([{"symbol": "AMD", "side": "long"}])
        assert bar._current_chips() == [kept]

    def test_the_taken_mark_is_display_only_and_moves_no_chip(self):
        bar = self._bar()
        bar.set_favorites([
            {"symbol": "NVDA", "side": "long"},
            {"symbol": "AMD", "side": "long"},
        ])
        before = bar._current_chips()
        bar.set_taken({("NVDA", "long")})
        assert bar._current_chips() == before
        assert before[0].taken_label.isHidden() is False
        assert before[1].taken_label.isHidden() is True
        assert before[0].property("taken") == "true"
        assert before[1].property("taken") == "false"

    def test_a_new_chip_arrives_already_marked(self):
        bar = self._bar()
        bar.set_taken({("NVDA", "long")})
        bar.set_favorites([{"symbol": "NVDA", "side": "long"}])
        assert bar._current_chips()[0].property("taken") == "true"

    def test_the_side_toggle_is_exclusive(self):
        bar = self._bar()
        assert bar.side() == "long"
        bar.set_side("short")
        assert (bar.side(), bar.long_button.isChecked(), bar.short_button.isChecked()) == (
            "short", False, True
        )

    def test_enter_emits_the_typed_text_with_the_current_side(self):
        bar = self._bar()
        seen: list[tuple[str, str]] = []
        bar.addRequested.connect(lambda text, side: seen.append((text, side)))
        bar.set_side("short")
        bar.input.setText("nvda amd")
        bar._emit_add()
        assert seen == [("nvda amd", "short")]
        assert bar.input.text() == "", "the box clears so the next paste is clean"

    def test_blank_input_emits_nothing(self):
        bar = self._bar()
        seen: list[tuple[str, str]] = []
        bar.addRequested.connect(lambda text, side: seen.append((text, side)))
        bar.input.setText("   ")
        bar._emit_add()
        assert seen == []

    def test_the_x_asks_for_that_pick_to_go(self):
        bar = self._bar()
        seen: list[tuple[str, str]] = []
        bar.removeRequested.connect(lambda symbol, side: seen.append((symbol, side)))
        bar.set_favorites([{"symbol": "NVDA", "side": "long"}])
        bar._current_chips()[0].removed.emit("NVDA", "long")
        assert seen == [("NVDA", "long")]

    def test_copy_puts_the_tickers_on_the_clipboard_one_per_line(self):
        """The trader pastes this straight into a TC2000 watchlist."""
        bar = self._bar()
        bar.set_favorites([
            {"symbol": "NVDA", "side": "long"},
            {"symbol": "AMD", "side": "short"},
        ])
        assert bar.copy_all() == "NVDA\nAMD"
        assert QApplication.clipboard().text() == "NVDA\nAMD"

    def test_copy_lists_a_ticker_once_even_on_both_sides(self):
        bar = self._bar()
        bar.set_favorites([
            {"symbol": "NVDA", "side": "long"},
            {"symbol": "NVDA", "side": "short"},
        ])
        assert bar.copy_all() == "NVDA"

    def test_copy_with_nothing_to_copy_says_so(self):
        bar = self._bar()
        assert bar.copy_all() == ""
        assert "Nothing to copy" in bar.status_label.text()

    def test_paste_adds_the_clipboard_on_the_selected_side(self):
        bar = self._bar()
        seen: list[tuple[str, str]] = []
        bar.addRequested.connect(lambda text, side: seen.append((text, side)))
        bar.set_side("short")
        QApplication.clipboard().setText("nvda\namd\ntsla")
        bar.paste()
        assert seen == [("nvda\namd\ntsla", "short")]

    def test_paste_with_an_empty_clipboard_asks_for_nothing(self):
        bar = self._bar()
        seen: list[tuple[str, str]] = []
        bar.addRequested.connect(lambda text, side: seen.append((text, side)))
        QApplication.clipboard().setText("   ")
        bar.paste()
        assert seen == []
        assert "Clipboard is empty" in bar.status_label.text()

    def test_no_chip_carries_a_stylesheet_of_its_own(self):
        """Variants live in theme.qss; a per-widget stylesheet is a CSS parse
        on the GUI thread (fluidity rules, 2026-08-21)."""
        bar = self._bar()
        bar.set_favorites([{"symbol": "NVDA", "side": "long"}])
        chip = bar._current_chips()[0]
        assert chip.styleSheet() == ""
        assert chip.name_label.styleSheet() == ""
        assert chip.taken_label.styleSheet() == ""


class TestTheStripIsStyledByTheTheme:
    def test_every_object_name_the_strip_sets_is_answered_by_the_qss(self):
        qss = (SCRIPTS_DIR / "ui" / "theme.qss").read_text(encoding="utf-8")
        for name in (
            "SwingFavoriteChip",
            "SwingFavoriteName",
            "SwingFavoriteTaken",
        ):
            assert f"#{name}" in qss, name


class TestWhereItLives:
    def test_the_strip_is_the_bottom_of_the_m5_column_in_both_modes(self):
        """Trader: "the very bottom of the M5 alerts tab". The M5 surface is a
        tab in tabs mode and the left column in workspace mode; the strip is
        the bottom of it either way, and the alert bar is always above it."""
        from PySide6.QtWidgets import QTabWidget
        from ui.panels.trading_desk import TradingDeskPanel

        desk = TradingDeskPanel(workspace_mode="workspace")
        try:
            assert desk.m5_column.count() == 2
            assert desk.m5_column.widget(0) is desk.m5_alert_bar
            assert desk.m5_column.widget(1) is desk.swing_favorites_bar
            assert desk.desk_splitter.widget(0) is desk.m5_column

            desk.set_mode("tabs")
            tabs = desk._mode_widget
            assert isinstance(tabs, QTabWidget)
            index = tabs.indexOf(desk.m5_column)
            assert index >= 0 and tabs.tabText(index) == "M5 alerts"
            assert desk.m5_alert_bar.parent() is desk.m5_column
            assert desk.swing_favorites_bar.parent() is desk.m5_column
        finally:
            desk.close()

    def test_the_alert_bar_still_posts_and_charts_across_a_mode_switch(self):
        """The strip must not cost the bar its routing (CLAUDE.md "Intraday
        alerts"): alerts still arrive, a click still charts."""
        from ui.models.bounce import BounceAlert
        from ui.panels.trading_desk import TradingDeskPanel

        alert = BounceAlert(
            time_text="07:09:19",
            symbol="NVDA",
            side="LONG",
            trigger="[S-TIER] VWAP reclaim",
            timeframe="5m",
            tag="green",
            raw_text="[S-TIER] VWAP reclaim NVDA (long)",
        )
        desk = TradingDeskPanel(workspace_mode="workspace")
        try:
            desk.set_mode("tabs")
            desk.alert_center.m5AlertPosted.emit(alert)
            assert desk.m5_alert_bar.alerts() == [alert]
            charted: list[str] = []
            desk.m5_alert_bar.alertActivated.connect(lambda a: charted.append(a.symbol))
            desk.m5_alert_bar.list.itemClicked.emit(desk.m5_alert_bar.list.item(0))
            assert charted == ["NVDA"]
        finally:
            desk.close()

    def test_the_two_share_a_draggable_split_that_neither_can_collapse(self):
        """Trader, 2026-08-31: "the tab needs to be resizable relative to the M5
        alerts tab, I should be able to drag it up to see more"."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QSplitter
        from ui.panels.trading_desk import TradingDeskPanel

        desk = TradingDeskPanel(workspace_mode="workspace")
        try:
            assert isinstance(desk.m5_column, QSplitter)
            assert desk.m5_column.orientation() == Qt.Orientation.Vertical
            assert desk.m5_column.childrenCollapsible() is False
            desk.m5_column.resize(240, 800)
            desk.m5_column.setSizes([400, 400])
            assert min(desk.m5_column.sizes()) > 0
        finally:
            desk.close()

    def test_the_column_drag_has_its_own_settings_key(self):
        """Dragging the strip must never overwrite the desk's column split."""
        from ui.panels import trading_desk

        assert trading_desk.M5_COLUMN_SPLIT_KEY != trading_desk.DESK_SPLIT_KEY
        assert trading_desk.M5_COLUMN_SPLIT_KEY == "qt_m5_column_split_sizes_v1"

    def test_the_chip_area_has_a_floor_and_no_ceiling(self):
        """A ceiling would make "drag it up to see more" do nothing."""
        from ui.widgets.swing_favorites_bar import MIN_CHIP_HEIGHT, SwingFavoritesBar

        bar = SwingFavoritesBar()
        assert bar.chip_scroll.minimumHeight() >= MIN_CHIP_HEIGHT
        assert bar.chip_scroll.maximumHeight() >= 16777215, "no ceiling"

    def test_a_day_roll_re_derives_the_strip_and_clears_the_bar(self):
        from ui.panels.trading_desk import TradingDeskPanel

        desk = TradingDeskPanel(workspace_mode="workspace")
        try:
            refreshed: list[int] = []
            desk.swing_favorites_bar.set_favorites([{"symbol": "NVDA", "side": "long"}])
            desk.alert_center.m5AlertsDayRolled.connect(lambda: refreshed.append(1))
            desk.alert_center.m5AlertsDayRolled.emit()
            assert refreshed == [1]
            assert desk.swing_favorites_bar.symbols() == [], "a new session starts empty"
        finally:
            desk.close()
