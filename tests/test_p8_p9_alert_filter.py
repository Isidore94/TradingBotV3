"""P9 (trader's go 2026-09-25): the Alert Center's "Show" filter for M5 rows.

All / Grade B and up (default) / Best right now. Display only: a hidden row
makes no sound and has no row, but is still recorded and still reaches the
review-queue door (the M5 bar's backing list). Typed names, Focus names, armed
watches and regime-pause rows always show. It composes with the sector hide.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

CSV_TEXT = (
    "symbol,sectorKey,industryKey,sector,industry,updated_utc\n"
    "APA,energy,oil-gas-e-p,Energy,Oil & Gas E&P,2026-03-13T14:00:47Z\n"
    "NVDA,technology,semiconductors,Technology,Semiconductors,2026-03-13T14:00:47Z\n"
)

# bounce type -> grade; "newtype" is never graded (New, n < 30).
GRADES = {"provtype": "PROVEN", "beetype": "B", "ceetype": "C", "deetype": "D"}


def _grades_payload():
    import setup_grades

    return {
        "daytrade": [
            {"key": setup_grades.daytrade_key(kind, side), "bounce_type": kind, "side": side, "grade": grade}
            for kind, grade in GRADES.items()
            for side in ("LONG", "SHORT")
        ]
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tmp classification CSV, tmp typed lists and an in-memory settings store."""
    import alert_show_filter
    import project_paths
    import sector_exclusion

    path = tmp_path / "symbol_classification.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    monkeypatch.setattr(project_paths, "SYMBOL_CLASSIFICATION_CACHE_FILE", path)
    longs = tmp_path / "longs.txt"
    shorts = tmp_path / "shorts.txt"
    longs.write_text("TYPED\n", encoding="utf-8")
    shorts.write_text("", encoding="utf-8")
    monkeypatch.setattr(project_paths, "LONGS_FILE", longs)
    monkeypatch.setattr(project_paths, "SHORTS_FILE", shorts)
    settings: dict[str, object] = {}
    monkeypatch.setattr(
        project_paths, "get_local_setting", lambda key, default=None: settings.get(key, default)
    )
    monkeypatch.setattr(
        project_paths, "save_local_setting", lambda key, value: settings.__setitem__(key, value)
    )
    sector_exclusion.clear_cache()
    alert_show_filter.clear_cache()
    yield settings
    sector_exclusion.clear_cache()
    alert_show_filter.clear_cache()


def _m5(symbol, kind, *, side="LONG", trigger="Bounce confirmed"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="09:31:00",
        symbol=symbol,
        side=side,
        trigger=trigger,
        timeframe="M5",
        tag="green",
        raw_text=f"[S-TIER] {symbol}: {trigger}",
        payload={"feedback": {"bounce_types": kind}},
    )


def _make_panel(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    import alert_repetition
    from ui.panels import alert_center_panel as panel_mod
    from ui.panels.alert_center_panel import AlertCenterPanel

    monkeypatch.setattr(
        alert_repetition.RepetitionLedger, "_in_digest_window", lambda self, now: False
    )
    beeps: list[str] = []
    monkeypatch.setattr(panel_mod.QApplication, "beep", lambda: beeps.append("beep"))
    panel = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
    )
    monkeypatch.setattr(panel, "_alerts_may_sound", lambda: True)
    monkeypatch.setattr(panel, "_auto_mode_now", lambda: "DESK")
    monkeypatch.setattr(panel, "_alert_is_focus", lambda alert: alert.symbol == "FOC")
    posted: list = []
    panel.m5AlertPosted.connect(posted.append)
    statuses: list[str] = []
    panel.statusChanged.connect(statuses.append)
    panel.test_beeps = beeps
    panel.test_posted = posted
    panel.test_statuses = statuses
    return panel


@pytest.fixture
def panel(env, tmp_path, monkeypatch):
    made = _make_panel(tmp_path, monkeypatch)
    made.set_setup_grades(_grades_payload())
    yield made
    made.deleteLater()


FEED = [
    ("PRV", "provtype"),
    ("BEE", "beetype"),
    ("CEE", "ceetype"),
    ("NEW", "newtype"),
    ("TYPED", "ceetype"),  # in longs.txt
    ("FOC", "deetype"),  # a Focus name
]


def _feed_symbols(panel) -> set[str]:
    return {key[0] for key in panel._feed_row_registry()}


def _post_feed(panel):
    for symbol, kind in FEED:
        panel.add_alert(_m5(symbol, kind))


def _choose(panel, value):
    panel.show_filter_input.setCurrentIndex(panel.show_filter_input.findData(value))


def test_the_default_is_grade_b_and_up_and_it_hides_c_d_and_new(panel):
    import alert_show_filter

    assert panel.show_filter_mode() == alert_show_filter.GRADE_B_UP
    _post_feed(panel)
    assert _feed_symbols(panel) == {"PRV", "BEE", "TYPED", "FOC"}
    # Every alert is still recorded and still reached the queue door (the M5 bar).
    assert [a.symbol for a in panel._alerts][::-1] == [s for s, _k in FEED]
    assert [a.symbol for a in panel.test_posted] == [s for s, _k in FEED]
    # No sound for a hidden row.
    assert len(panel.test_beeps) == 4


def test_all_shows_every_row(panel):
    import alert_show_filter

    _choose(panel, alert_show_filter.ALL)
    _post_feed(panel)
    assert _feed_symbols(panel) == {s for s, _k in FEED}
    assert panel.show_filter_hidden_counts() == (0, 0)


def test_best_right_now_shows_only_best_rows_plus_privileged(panel):
    import alert_show_filter
    import best_now

    _choose(panel, alert_show_filter.BEST_NOW)
    _post_feed(panel)
    # Not ranked yet: unknown shows.
    assert _feed_symbols(panel) == {s for s, _k in FEED}
    panel.set_best_now_entries(
        [best_now.BestNowEntry(symbol="CEE", side="LONG", tier=1, why="", entry=None, stop=None)]
    )
    assert _feed_symbols(panel) == {"CEE", "TYPED", "FOC"}


def test_regime_pause_and_chart_watch_rows_always_show(panel):
    from ui.models.bounce import REGIME_PAUSE_TRIGGER_PREFIX

    pause = _m5("SPYP", "deetype", trigger=f"{REGIME_PAUSE_TRIGGER_PREFIX} · holding highs")
    assert panel.show_filter_verdict(pause) == (False, False)
    from ui.models.bounce import CHART_WATCH_TAG

    watch = _m5("WAT", "deetype")
    watch.tag = CHART_WATCH_TAG
    assert panel.show_filter_hides(watch) is False


def test_the_choice_persists_as_a_local_setting(env, panel, tmp_path, monkeypatch):
    import alert_show_filter

    assert alert_show_filter.mode() == alert_show_filter.GRADE_B_UP, "first-run default"
    _choose(panel, alert_show_filter.BEST_NOW)
    assert env[alert_show_filter.SETTING_SHOW_FILTER] == alert_show_filter.BEST_NOW
    (tmp_path / "again").mkdir()
    again = _make_panel(tmp_path / "again", monkeypatch)
    try:
        assert again.show_filter_mode() == alert_show_filter.BEST_NOW
    finally:
        again.deleteLater()


def test_the_status_line_counts_hidden_rows_and_new(panel):
    _post_feed(panel)
    assert panel.show_filter_hidden_counts() == (2, 1)  # CEE and NEW; NEW is New
    assert "2 hidden by Show filter (1 New)" in panel.test_statuses[-1]


def test_the_sector_hide_and_the_show_filter_compose(panel):
    panel.add_alert(_m5("APA", "provtype"))  # PROVEN but oil & gas: sector hides it
    panel.add_alert(_m5("NVDA", "ceetype"))  # tech but C: Show hides it
    panel.add_alert(_m5("PRV", "provtype"))
    assert _feed_symbols(panel) == {"PRV"}
    # The sector-hidden row is not counted as a Show-filter hide.
    assert panel.show_filter_hidden_counts() == (1, 0)
    panel.hide_sector_input.setChecked(False)
    assert _feed_symbols(panel) == {"PRV", "APA"}


def test_grades_not_loaded_hide_nothing(env, tmp_path, monkeypatch):
    made = _make_panel(tmp_path, monkeypatch)
    try:
        _post_feed(made)
        assert _feed_symbols(made) == {s for s, _k in FEED}, "unknown grade shows"
    finally:
        made.deleteLater()


def test_the_compact_drawer_carries_the_show_selector(panel):
    panel.set_compact_layout(True)
    try:
        corner = panel._controls_corner
        assert panel.show_filter_input.parent() is corner
    finally:
        panel.set_compact_layout(False)
    assert panel.show_filter_input in panel._control_widgets


# --------------------------------------------------------------------------- M5 bar
def test_the_m5_bar_draws_only_shown_rows_and_titles_the_hidden_count(panel):
    from ui.widgets.m5_alert_bar import M5AlertBar

    bar = M5AlertBar()
    bar.set_show_filter(panel.show_filter_verdict)
    panel.showFilterChanged.connect(bar.refresh_show_filter)
    panel.m5AlertPosted.connect(bar.post)
    try:
        _post_feed(panel)
        assert set(bar.symbols()) == {"PRV", "BEE", "TYPED", "FOC"}
        assert len(bar._arrival) == len(FEED), "the backing list keeps every row"
        assert "2 hidden by Show filter (1 New)" in bar.title_label.text()
        import alert_show_filter

        _choose(panel, alert_show_filter.ALL)
        assert set(bar.symbols()) == {s for s, _k in FEED}
        assert "hidden" not in bar.title_label.text()
    finally:
        bar.deleteLater()


# --------------------------------------------------------------------------- phone report
def test_the_phone_report_drops_hidden_lines_and_prints_the_count(env):
    import autopilot_core as core

    payload = {
        "alerts": ["09:31 PRV", "09:32 CEE", "09:33 NEW", "09:34 APA"],
        "alert_symbols": ["PRV", "CEE", "NEW", "APA"],
        "alert_show_flags": [(False, False), (True, False), (True, True), (False, False)],
    }
    out = core.hide_sector_names(core.hide_show_filtered_alerts(payload))
    assert out["alerts"] == ["09:31 PRV"]
    assert out["show_hidden_count"] == 2
    assert out["show_hidden_new"] == 1
    assert "alert_show_flags" not in out
    text = core.render_away_report(out)
    assert "Hidden: 1 oil & gas / real estate; 2 hidden by Show filter (1 New)" in text


def test_the_phone_report_show_line_alone(env):
    import autopilot_core as core
    import sector_exclusion

    sector_exclusion.set_hide_enabled(False)
    out = core.hide_sector_names(
        core.hide_show_filtered_alerts(
            {"alerts": ["a"], "alert_symbols": ["CEE"], "alert_show_flags": [(True, False)]}
        )
    )
    assert "Hidden: 1 hidden by Show filter (0 New)" in core.render_away_report(out)


def test_the_autopilot_service_records_a_verdict_per_alert_line():
    from collections import deque

    from ui.services.autopilot_service import AutopilotService

    service = AutopilotService.__new__(AutopilotService)
    service._alerts_today = deque(maxlen=60)
    service._alert_symbols_today = deque(maxlen=60)
    service._alert_show_today = deque(maxlen=60)
    service._show_filter = None
    service.set_show_filter(lambda alert: (alert.symbol == "CEE", False))
    service._on_alert(_m5("CEE", "ceetype"))
    service._on_alert(_m5("PRV", "provtype"))
    assert list(service._alert_show_today) == [(True, False), (False, False)]


# --------------------------------------------------------------------------- review round 1
def test_entry_assist_output_always_shows_and_is_never_counted(panel):
    from ui.models.bounce import BounceAlert

    assist = BounceAlert(
        time_text="09:40:00", symbol="", side="WATCH", tag="entry_assist",
        raw_text="ENTRY ASSIST: window open",
    )
    strongest = BounceAlert(
        time_text="09:41:00", symbol="", side="WATCH", raw_text="STRONGEST 5: NVDA, AMD, MU, AVGO, TSM",
    )
    for alert in (assist, strongest):
        assert panel.show_filter_verdict(alert) == (False, False)
        panel.add_alert(alert)
    # Both are symbol-less WATCH rows, which the feed folds to one row by design;
    # the default filter must show exactly what "All" shows.
    rows = [alert for _key, alert, _n in panel._feed_target_rows()]
    assert rows and any(alert is assist for alert in rows)
    assert panel.show_filter_hidden_counts() == (0, 0)
    import alert_show_filter

    _choose(panel, alert_show_filter.ALL)
    assert [alert for _key, alert, _n in panel._feed_target_rows()] == rows


def test_a_verdict_is_cached_until_an_input_changes(panel, monkeypatch):
    alert = _m5("CEE", "ceetype")
    calls = []
    real = panel.show_filter_grade
    monkeypatch.setattr(panel, "show_filter_grade", lambda a: calls.append(a) or real(a))
    assert panel.show_filter_verdict(alert) == (True, False)
    assert panel.show_filter_verdict(alert) == (True, False)
    assert len(calls) == 1, "cached"
    panel.set_setup_grades({"daytrade": []})
    assert panel.show_filter_verdict(alert) == (False, False), "grades changed: re-read"


def test_focus_or_typed_list_change_redraws_the_bar(panel, env, monkeypatch):
    import alert_show_filter
    import project_paths
    from ui.widgets.m5_alert_bar import M5AlertBar

    bar = M5AlertBar()
    bar.set_show_filter(panel.show_filter_verdict)
    panel.showFilterChanged.connect(bar.refresh_show_filter)
    panel.m5AlertPosted.connect(bar.post)
    try:
        panel.add_alert(_m5("CEE", "ceetype"))
        panel.add_alert(_m5("DEE", "deetype"))
        assert bar.symbols() == []
        # CEE joins Focus: the coalesced Focus reaction redraws the bar.
        monkeypatch.setattr(panel, "_alert_is_focus", lambda alert: alert.symbol in ("FOC", "CEE"))
        panel._on_focus_feed_coalesced()
        assert bar.symbols() == ["CEE"]
        # DEE is typed into longs.txt: the next typed-list check redraws.
        Path(project_paths.LONGS_FILE).write_text("TYPED\nDEE\n", encoding="utf-8")
        alert_show_filter.clear_cache()
        panel._check_typed_symbols()
        assert set(bar.symbols()) == {"CEE", "DEE"}
    finally:
        bar.deleteLater()


def test_a_focus_burst_is_one_coalesced_diff_never_a_rebuild(env, tmp_path, monkeypatch):
    """Review round 2: a Focus change reaches the Show filter only through the
    one coalesced Focus reaction (`_sync_feed`), never a rebuild."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from focus_picks import FocusPickStore
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.services.focus_service import FocusService
    from ui.widgets.m5_alert_bar import M5AlertBar

    focus_dir = tmp_path / "focus"
    focus_dir.mkdir()
    service = FocusService(
        FocusPickStore(
            focus_longs_path=focus_dir / "focus_longs.txt",
            focus_shorts_path=focus_dir / "focus_shorts.txt",
            longs_path=focus_dir / "longs.txt",
            shorts_path=focus_dir / "shorts.txt",
            membership_path=focus_dir / "membership.json",
        )
    )
    panel = AlertCenterPanel(
        focus_service=service,
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
    )
    monkeypatch.setattr(panel, "_auto_mode_now", lambda: "DESK")
    panel.set_setup_grades(_grades_payload())
    bar = M5AlertBar()
    bar.set_show_filter(panel.show_filter_verdict)
    panel.showFilterChanged.connect(bar.refresh_show_filter)
    panel.m5AlertPosted.connect(bar.post)
    try:
        panel.add_alert(_m5("CEE", "ceetype"))
        panel.add_alert(_m5("DEE", "deetype"))
        assert bar.symbols() == []
        rebuilds: list[int] = []
        reactions: list[int] = []
        monkeypatch.setattr(panel, "_rebuild_feed", lambda: rebuilds.append(1))
        panel.showFilterChanged.connect(lambda: reactions.append(1))
        for symbol in ("CEE", "AAA", "BBB"):
            service.store.add(symbol, "long", "m5")
        assert rebuilds == [] and reactions == [], "nothing before the coalesced reaction"
        panel.flush_pending_focus_refresh()
        assert rebuilds == [], "a Focus change is a diff, never a rebuild"
        assert reactions == [1], "one reaction per burst"
        assert bar.symbols() == ["CEE"]
    finally:
        bar.deleteLater()
        panel.deleteLater()


def _hidden_rows(tmp_path) -> list[dict]:
    import review_events

    return [
        row
        for row in review_events.load_review_events(tmp_path / "alert_review_events.jsonl")
        if row.get("action") == "hidden_by_show"
    ]


def test_a_hidden_row_writes_one_hidden_by_show_review_event_with_its_grade(panel, tmp_path):
    """B6: every alert the Show filter hides leaves one `hidden_by_show` evidence row."""
    _post_feed(panel)
    rows = _hidden_rows(tmp_path)
    assert sorted((row["symbol"], row["detail"]["grade"]) for row in rows) == [
        ("CEE", "C"),
        ("NEW", "New"),
    ]
    assert all(row["detail"]["show_mode"] == "grade_b_up" for row in rows)
    # A second alert on a hidden name is a second hidden alert: one more row.
    panel.add_alert(_m5("CEE", "ceetype"))
    assert len(_hidden_rows(tmp_path)) == 3


def test_a_failed_hidden_by_show_write_never_costs_the_alert(panel, monkeypatch):
    from ui.panels import alert_center_panel as panel_mod

    def boom(*_args, **_kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(panel_mod, "record_review_event", boom)
    _post_feed(panel)
    assert [a.symbol for a in panel._alerts][::-1] == [s for s, _k in FEED]
    assert _feed_symbols(panel) == {"PRV", "BEE", "TYPED", "FOC"}
