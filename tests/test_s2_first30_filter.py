"""S2 (finding F5): the First-30 Show filter.

A switch, default on, hides M5 rows whose timezone-aware alert time is
09:30-10:00 ET. PROVEN, Focus, typed names and chart-watch hits always show;
every alert is still recorded; the `hidden_by_show` detail says `first30`.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
for _path in (SCRIPTS_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_p8_p9_alert_filter import _grades_payload, _make_panel, env  # noqa: E402,F401

ET = ZoneInfo("America/New_York")


def _at(hh, mm, ss=0):
    return datetime(2026, 9, 28, hh, mm, ss, tzinfo=ET)


def _m5(symbol, kind, when, *, side="LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text=when.strftime("%H:%M:%S"),
        symbol=symbol,
        side=side,
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text=f"[S-TIER] {symbol}: Bounce confirmed",
        payload={"feedback": {"bounce_types": kind}},
        received_at=when,
    )


@pytest.fixture
def panel(env, tmp_path, monkeypatch):  # noqa: F811
    made = _make_panel(tmp_path, monkeypatch)
    made.set_setup_grades(_grades_payload())
    yield made
    made.deleteLater()


def _feed_symbols(panel) -> set[str]:
    return {key[0] for key in panel._feed_row_registry()}


# --------------------------------------------------------------------------- pure
def test_the_window_boundaries_in_et():
    import alert_show_filter as f

    assert f.in_first30(_at(9, 29, 59)) is False
    assert f.in_first30(_at(9, 30)) is True
    assert f.in_first30(_at(9, 59, 59)) is True
    assert f.in_first30(_at(10, 0)) is False
    # The same instant in another zone is still judged in ET.
    assert f.in_first30(_at(9, 45).astimezone(timezone.utc)) is True
    assert f.in_first30(_at(9, 45).astimezone(ZoneInfo("America/Los_Angeles"))) is True
    # A naive or missing time is unknown: never in the window.
    assert f.in_first30(datetime(2026, 9, 28, 9, 45)) is False
    assert f.in_first30(None) is False


def test_the_switch_defaults_on_and_persists(env):  # noqa: F811
    import alert_show_filter as f

    assert f.first30_enabled() is True
    f.set_first30_enabled(False)
    assert env[f.SETTING_FIRST30] is False
    assert f.first30_enabled() is False


def test_the_hidden_text_names_first30():
    import alert_show_filter as f

    assert f.hidden_text(3, 1, first30=2) == "3 hidden by Show filter (1 New, 2 first30)"
    assert f.hidden_text(3, 1) == "3 hidden by Show filter (1 New)"


def test_from_callback_stamps_a_timezone_aware_time():
    from ui.models.bounce import BounceAlert

    alert = BounceAlert.from_callback("NVDA: Bounce confirmed", "green")
    assert alert.received_at is not None and alert.received_at.utcoffset() is not None


# --------------------------------------------------------------------------- panel
def test_the_switch_is_on_by_default_in_the_alert_center(panel):
    assert panel.first30_input.isChecked() is True
    assert panel.first30_input in panel._control_widgets


def test_09_59_59_hides_and_10_00_shows(panel):
    import alert_show_filter

    # Show: All, so only the first-30 switch can hide.
    panel.show_filter_input.setCurrentIndex(
        panel.show_filter_input.findData(alert_show_filter.ALL)
    )
    early = _m5("BEE", "beetype", _at(9, 59, 59))
    late = _m5("BEE", "beetype", _at(10, 0), side="SHORT")
    assert panel.show_filter_hides(early) is True
    assert panel.show_filter_hides(late) is False
    assert panel.show_filter_reason(early) == "first30"


def test_before_the_open_and_unknown_time_show(panel):
    assert panel.show_filter_hides(_m5("BEE", "beetype", _at(9, 29, 59))) is False
    no_time = _m5("BEE", "beetype", _at(9, 45))
    no_time.received_at = None
    assert panel.show_filter_hides(no_time) is False


def test_proven_is_never_hidden_by_first30(panel):
    assert panel.show_filter_hides(_m5("PRV", "provtype", _at(9, 40))) is False


def test_focus_is_never_hidden_by_first30(panel):
    assert panel.show_filter_hides(_m5("FOC", "deetype", _at(9, 40))) is False


def test_typed_names_are_never_hidden_by_first30(panel):
    assert panel.show_filter_hides(_m5("TYPED", "ceetype", _at(9, 40))) is False


def test_chart_watch_hits_are_never_hidden_by_first30(panel):
    from ui.models.bounce import CHART_WATCH_TAG

    watch = _m5("WAT", "deetype", _at(9, 40))
    watch.tag = CHART_WATCH_TAG
    assert panel.show_filter_hides(watch) is False


def test_switch_off_shows_first30_rows(panel, env):  # noqa: F811
    import alert_show_filter

    alert = _m5("BEE", "beetype", _at(9, 40))
    assert panel.show_filter_hides(alert) is True
    panel.first30_input.setChecked(False)
    assert env[alert_show_filter.SETTING_FIRST30] is False
    assert panel.show_filter_hides(alert) is False


def test_hidden_rows_are_still_recorded_and_counted_as_first30(panel, tmp_path):
    import review_events

    for symbol, kind in (("BEE", "beetype"), ("PRV", "provtype"), ("FOC", "deetype")):
        panel.add_alert(_m5(symbol, kind, _at(9, 40)))
    panel.add_alert(_m5("LATE", "beetype", _at(10, 5)))
    assert _feed_symbols(panel) == {"PRV", "FOC", "LATE"}
    assert {a.symbol for a in panel._alerts} == {"BEE", "PRV", "FOC", "LATE"}
    assert {a.symbol for a in panel.test_posted} == {"BEE", "PRV", "FOC", "LATE"}
    assert panel.show_filter_first30_count() == 1
    assert "1 hidden by Show filter (0 New, 1 first30)" in panel.test_statuses[-1]
    panel.flush_show_hidden_writes()
    rows = [
        row
        for row in review_events.load_review_events(tmp_path / "alert_review_events.jsonl")
        if row.get("action") == "hidden_by_show"
    ]
    assert [(row["symbol"], row["detail"]["reason"]) for row in rows] == [("BEE", "first30")]
