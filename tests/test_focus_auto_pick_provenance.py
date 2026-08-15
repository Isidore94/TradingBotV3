"""Per-entry provenance for M5 Focus picks (plan.md Phase 0.5, packet R2 A.3.1).

The focus files are plain text, one ticker per line, so an auto-adopted pick
and a name the trader typed are the same line in the same file. Without an
origin, no removal verb could be written safely at all - which is why the
"Not today" verb and the triple-VWAP desync repair both wait on this sidecar.

The invariant it makes structural (plan.md sec 5): **user-entered watchlist
names are never automatically removed**. Absence of a marker reads as
user-entered, so every focus file written before this packet is protected by
default and a lost marker fails in the safe direction.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _store(tmp_path):
    from focus_picks import FocusPickStore

    return FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "membership.json",
    )


def test_a_hand_typed_name_carries_no_marker(tmp_path):
    store = _store(tmp_path)
    store.add("NVDA", "long", "m5")
    assert store.is_auto_adopted("NVDA", "long", "m5") is False
    assert store.auto_pick_marker("NVDA", "long", "m5") is None


def test_an_auto_adopted_pick_is_marked_and_removable(tmp_path):
    store = _store(tmp_path)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5", staged_at="09:05:00", reason="PDH break")

    marker = store.auto_pick_marker("NVDA", "long", "m5")
    assert marker["reason"] == "PDH break" and marker["staged_at"] == "09:05:00"
    assert marker["session_date"] == date.today().isoformat()

    assert store.remove_if_auto_adopted("NVDA", "long", "m5") is True
    assert store.focus_symbols("long", "m5") == []


def test_the_scoped_removal_refuses_a_name_the_trader_typed(tmp_path):
    """The whole point: no automatic path reaches the trader's own names."""
    store = _store(tmp_path)
    store.add("NVDA", "long", "m5")
    assert store.remove_if_auto_adopted("NVDA", "long", "m5") is False
    assert store.focus_symbols("long", "m5") == ["NVDA"]


def test_the_removal_touches_exactly_one_entry(tmp_path):
    """Trader rule 2026-08-15: never the swing entry, never the other side,
    never another name."""
    store = _store(tmp_path)
    for side in ("long", "short"):
        for category in ("m5", "swing"):
            store.add("NVDA", side, category)
            store.mark_auto_adopted("NVDA", side, category)
    store.add("AMD", "long", "m5")
    store.mark_auto_adopted("AMD", "long", "m5")

    assert store.remove_if_auto_adopted("NVDA", "long", "m5") is True

    assert store.focus_symbols("long", "m5") == ["AMD"], "the other name survives"
    assert store.focus_symbols("short", "m5") == ["NVDA"], "the other side survives"
    assert store.focus_symbols("long", "swing") == ["NVDA"], "the swing entry survives"
    assert store.focus_symbols("short", "swing") == ["NVDA"]


def test_removing_an_entry_forgets_its_marker(tmp_path):
    """Re-adding the name by hand must start it as the trader's again."""
    store = _store(tmp_path)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5")
    store.remove("NVDA", "long", "m5")

    store.add("NVDA", "long", "m5")
    assert store.is_auto_adopted("NVDA", "long", "m5") is False
    assert store.remove_if_auto_adopted("NVDA", "long", "m5") is False


def test_unfavorite_and_clear_forget_markers_too(tmp_path):
    store = _store(tmp_path)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5")
    store.remove_everywhere("NVDA")
    assert store.auto_pick_markers() == {}

    store.add("AMD", "long", "m5")
    store.mark_auto_adopted("AMD", "long", "m5")
    store.clear("long", "m5")
    assert store.auto_pick_markers() == {}


def test_markers_survive_a_restart(tmp_path):
    store = _store(tmp_path)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5", reason="PDH break")

    reopened = _store(tmp_path)
    assert reopened.is_auto_adopted("NVDA", "long", "m5") is True


def test_markers_day_roll_with_the_picks_they_describe(tmp_path):
    """Yesterday's marker must never authorize removing today's typed name."""
    store = _store(tmp_path)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5")

    store.expire_m5_if_new_day(date.today() + timedelta(days=1))
    assert store.auto_pick_markers() == {}
    assert store.focus_symbols("long", "m5") == []

    # The trader types the same name the next morning: it is theirs.
    store.add("NVDA", "long", "m5")
    assert store.remove_if_auto_adopted("NVDA", "long", "m5") is False


def test_a_corrupt_sidecar_reads_as_no_markers(tmp_path):
    """Failing safe means failing toward 'the trader owns it'."""
    store = _store(tmp_path)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5")
    (tmp_path / "focus_auto_picks.json").write_text("{not json", encoding="utf-8")

    reopened = _store(tmp_path)
    assert reopened.auto_pick_markers() == {}
    assert reopened.remove_if_auto_adopted("NVDA", "long", "m5") is False


def test_the_plain_text_focus_files_are_untouched_by_provenance(tmp_path):
    """The sidecar rides beside the format, never inside it."""
    store = _store(tmp_path)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5", reason="PDH break")
    assert (tmp_path / "focus_longs.txt").read_text(encoding="utf-8").strip() == "NVDA"


# ---------------------------------------------------------------------------
# The "Not today" verb in the Alert Center
# ---------------------------------------------------------------------------


def _panel(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - PySide6 is on the desk
        import pytest

        pytest.skip("PySide6 is not installed")
    QApplication.instance() or QApplication([])
    from focus_picks import FocusPickStore
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.services.focus_service import FocusService

    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "membership.json",
    )
    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    panel.focus_service = FocusService(store)
    return panel, store


def _alert(symbol="NVDA", side="LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert.from_callback(
        f"[S-TIER] {symbol}: Bounce confirmed ({side.lower()})", "green"
    )


def test_not_today_drops_an_auto_adopted_pick(tmp_path):
    panel, store = _panel(tmp_path)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5", reason="PDH break")

    panel._remove_review_alert_for_today(_alert())

    assert store.focus_symbols("long", "m5") == []
    # The injected watchlist line goes with it, so it stops alerting entirely.
    assert "NVDA" not in (tmp_path / "longs.txt").read_text(encoding="utf-8")


def test_not_today_leaves_a_name_the_trader_typed_in_focus(tmp_path):
    """Same button, same click, and the trader's own pick survives."""
    panel, store = _panel(tmp_path)
    store.add("NVDA", "long", "m5")

    panel._remove_review_alert_for_today(_alert())

    assert store.focus_symbols("long", "m5") == ["NVDA"]
    assert "NVDA" in panel._ignored_symbols, "it still leaves today's feed"


def test_not_today_on_a_long_chart_never_drops_the_short_entry(tmp_path):
    panel, store = _panel(tmp_path)
    for side in ("long", "short"):
        store.add("NVDA", side, "m5")
        store.mark_auto_adopted("NVDA", side, "m5")

    panel._remove_review_alert_for_today(_alert(side="LONG"))

    assert store.focus_symbols("long", "m5") == []
    assert store.focus_symbols("short", "m5") == ["NVDA"]


def test_the_verdict_is_recorded_as_not_today_not_a_dislike(tmp_path, monkeypatch):
    """A same-day pass is not "this name is bad" - the scoreboard must not
    learn the wrong lesson from it."""
    panel, store = _panel(tmp_path)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5")

    rows: list[dict] = []
    monkeypatch.setattr(
        "ui.services.focus_service.record_pick_feedback",
        lambda **kwargs: rows.append(kwargs),
    )
    monkeypatch.setattr(store, "uses_default_paths", lambda: True)

    panel._remove_review_alert_for_today(_alert())

    assert len(rows) == 1
    assert rows[0]["verdict"] == "not_today"
    assert rows[0]["origin"] == "auto_pick"
    assert rows[0]["category"] == "m5"


def test_the_button_says_which_action_it_will_take(tmp_path):
    panel, store = _panel(tmp_path)
    store.add("NVDA", "long", "m5")
    alert = _alert()

    panel.chart_review.set_alert(alert, in_focus=True, auto_adopted=False)
    assert panel.chart_review.remove_today_button.text() == "✕ Not today"

    store.mark_auto_adopted("NVDA", "long", "m5")
    panel.chart_review.set_alert(alert, in_focus=True, auto_adopted=True)
    assert panel.chart_review.remove_today_button.text() == "✕ Not today - drop pick"
