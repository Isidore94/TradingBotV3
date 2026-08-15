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


# ---------------------------------------------------------------------------
# The triple-VWAP / Focus desync repair (A.3.4)
# ---------------------------------------------------------------------------


def _desync_path(tmp_path, monkeypatch):
    path = tmp_path / "focus_desync_requests.json"
    import autopilot_core as core

    monkeypatch.setattr(core, "FOCUS_DESYNC_REQUEST_FILE", path)
    return path


def test_the_bot_files_a_request_instead_of_touching_the_store(tmp_path, monkeypatch):
    """One owner per mutable store: the bot cannot write Focus itself."""
    import autopilot_core as core

    _desync_path(tmp_path, monkeypatch)
    core.record_focus_desync("NVDA", "long", reason="triple-VWAP invalidation")

    taken = core.take_focus_desync_requests()
    assert [(row["symbol"], row["side"]) for row in taken] == [("NVDA", "long")]
    # Draining is destructive: re-applying a cut after the trader re-added the
    # name by hand would be exactly the automatic removal this design prevents.
    assert core.take_focus_desync_requests() == []


def test_requests_are_day_scoped(tmp_path, monkeypatch):
    from datetime import datetime, timedelta

    import autopilot_core as core

    _desync_path(tmp_path, monkeypatch)
    yesterday = datetime.now() - timedelta(days=1)
    core.record_focus_desync("NVDA", "long", now=yesterday)
    assert core.take_focus_desync_requests() == []


def test_an_invalidated_auto_pick_leaves_focus(tmp_path, monkeypatch):
    import autopilot_core as core

    panel, store = _panel(tmp_path)
    _desync_path(tmp_path, monkeypatch)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5")
    core.record_focus_desync("NVDA", "long")

    panel._drain_focus_desync_requests()
    assert store.focus_symbols("long", "m5") == []


def test_a_trader_typed_pick_is_left_alone_and_surfaced(tmp_path, monkeypatch):
    """Neither silently deleted nor silently trusted."""
    import autopilot_core as core

    panel, store = _panel(tmp_path)
    _desync_path(tmp_path, monkeypatch)
    store.add("NVDA", "long", "m5")  # no marker: the trader's
    core.record_focus_desync("NVDA", "long")

    messages: list[str] = []
    panel.statusChanged.connect(messages.append)
    panel._drain_focus_desync_requests()

    assert store.focus_symbols("long", "m5") == ["NVDA"], "never auto-removed"
    assert any("no longer being scanned" in text for text in messages)


def test_a_cut_for_a_name_not_in_focus_changes_nothing(tmp_path, monkeypatch):
    import autopilot_core as core

    panel, store = _panel(tmp_path)
    _desync_path(tmp_path, monkeypatch)
    store.add("AMD", "long", "m5")
    store.mark_auto_adopted("AMD", "long", "m5")
    core.record_focus_desync("NVDA", "long")

    messages: list[str] = []
    panel.statusChanged.connect(messages.append)
    panel._drain_focus_desync_requests()

    assert store.focus_symbols("long", "m5") == ["AMD"]
    assert messages == []


def test_the_cut_side_is_the_only_side_touched(tmp_path, monkeypatch):
    import autopilot_core as core

    panel, store = _panel(tmp_path)
    _desync_path(tmp_path, monkeypatch)
    for side in ("long", "short"):
        store.add("NVDA", side, "m5")
        store.mark_auto_adopted("NVDA", side, "m5")
    core.record_focus_desync("NVDA", "long")

    panel._drain_focus_desync_requests()
    assert store.focus_symbols("long", "m5") == []
    assert store.focus_symbols("short", "m5") == ["NVDA"]


# ---------------------------------------------------------------------------
# The ownership collision (R2.1 blocker, external review 2026-08-15)
# ---------------------------------------------------------------------------


def test_adoption_never_relabels_a_name_the_trader_added_first(tmp_path, monkeypatch):
    """The exact sequence the review found.

    AWAY stages SYM. The trader adds SYM to M5 Focus by hand. The flip to DESK
    drains the queue and adopts SYM - but `store.add` returns False because the
    name is already there, and the old code wrote the marker anyway. Their entry
    silently became machine-owned, after which "Not today" and the desync repair
    could both delete it.
    """
    panel, store = _panel(tmp_path)
    panel._auto_pick_pending_path = tmp_path / "pending.json"

    # 1. The trader adds it themselves. No marker: it is theirs.
    store.add("NVDA", "long", "m5")
    assert store.is_auto_adopted("NVDA", "long", "m5") is False

    # 2. The drain adopts the staged pick for the same symbol and side.
    resolved = panel._adopt_auto_pick_into_focus(
        "NVDA", "long", {"staged_at": "09:05:00"}, "PDH break"
    )

    # The proposal is resolved (nothing left to review) ...
    assert resolved is True
    # ... but ownership did NOT change hands.
    assert panel._last_adoption_outcome == "already_trader_owned"
    assert store.is_auto_adopted("NVDA", "long", "m5") is False

    # 3. And the verbs that only touch machine picks still refuse it.
    assert store.remove_if_auto_adopted("NVDA", "long", "m5") is False
    assert store.focus_symbols("long", "m5") == ["NVDA"]


def test_a_marker_is_never_written_over_an_unmarked_existing_entry(tmp_path):
    """The general rule behind that sequence: `add()` returning False means
    nothing was added, so nothing may be claimed."""
    panel, store = _panel(tmp_path)
    for side in ("long", "short"):
        store.add("AMD", side, "m5")

    for side in ("long", "short"):
        panel._adopt_auto_pick_into_focus("AMD", side, {}, "whatever")
        assert store.is_auto_adopted("AMD", side, "m5") is False

    assert store.auto_pick_markers() == {}


def test_a_genuinely_new_pick_is_still_adopted_and_marked(tmp_path):
    """The fix must not stop real adoptions - that is the whole feature."""
    panel, store = _panel(tmp_path)
    resolved = panel._adopt_auto_pick_into_focus(
        "TSLA", "long", {"staged_at": "09:05:00"}, "PDH break"
    )
    assert resolved is True
    assert panel._last_adoption_outcome == "adopted"
    assert store.is_auto_adopted("TSLA", "long", "m5") is True
    assert store.remove_if_auto_adopted("TSLA", "long", "m5") is True


def test_re_adopting_our_own_pick_keeps_the_existing_marker(tmp_path):
    """A second drain of the same pick is a no-op, not a downgrade."""
    panel, store = _panel(tmp_path)
    panel._adopt_auto_pick_into_focus("TSLA", "long", {}, "PDH break")
    first = store.auto_pick_marker("TSLA", "long", "m5")

    panel._adopt_auto_pick_into_focus("TSLA", "long", {}, "PDH break")
    assert panel._last_adoption_outcome == "already_auto"
    assert store.auto_pick_marker("TSLA", "long", "m5") == first


def test_the_status_line_does_not_claim_a_trader_owned_name(tmp_path, monkeypatch):
    """"N auto pick(s) added" must count only names this desk actually took."""
    panel, store = _panel(tmp_path)
    panel._auto_pick_pending_path = tmp_path / "pending.json"
    store.add("MINE", "long", "m5")

    from datetime import datetime

    import autopilot_core

    fresh = datetime.now().isoformat(timespec="seconds")
    bar = autopilot_core.latest_completed_m5_end().isoformat()
    monkeypatch.setattr(
        "autopilot_core.load_auto_populate_pending_picks",
        lambda *_a, **_k: {
            "date": "2026-07-02",
            "pending": {
                "long": {
                    "MINE": {"reason": "r", "gate_state": "open",
                             "gate_checked_at": fresh, "gate_bar_end": bar},
                    "THEIRS": {"reason": "r", "gate_state": "open",
                               "gate_checked_at": fresh, "gate_bar_end": bar},
                }
            },
        },
    )
    monkeypatch.setattr("autopilot_core.read_auto_pilot_mode", lambda *_a, **_k: "DESK")
    panel._auto_mode_cached = None

    messages: list[str] = []
    panel.statusChanged.connect(messages.append)
    panel._poll_auto_pick_pending()

    text = " ".join(messages)
    assert "THEIRS" in text
    assert "1 auto pick(s)" in text, "only the genuinely new one was added"
    assert store.is_auto_adopted("MINE", "long", "m5") is False


# ---------------------------------------------------------------------------
# Sidecar hardening (R2.1 item 5)
# ---------------------------------------------------------------------------


def test_a_marker_from_another_session_is_ignored_on_load(tmp_path):
    """Validation on LOAD, not only on write.

    The day-roll clears markers when it runs, but the sidecar is a plain file
    that can outlive that run - restored from a backup, hand-edited, or written
    by a process that died before the roll fired. A stale marker is precisely a
    licence to delete a name the trader typed this morning.
    """
    import json

    store = _store(tmp_path)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5")

    # Backdate the marker in place, leaving the file's own header current.
    path = tmp_path / "focus_auto_picks.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for marker in payload["picks"].values():
        marker["session_date"] = "2020-01-01"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reopened = _store(tmp_path)
    assert reopened.auto_pick_markers() == {}
    assert reopened.remove_if_auto_adopted("NVDA", "long", "m5") is False
    assert reopened.focus_symbols("long", "m5") == ["NVDA"], "the name survives"


def test_a_half_rolled_file_cannot_smuggle_yesterdays_entries(tmp_path):
    """Per-entry validation, not just the file header: a current header over
    stale entries must not make those entries current."""
    import json
    from datetime import date

    store = _store(tmp_path)
    for symbol in ("TODAY", "STALE"):
        store.add(symbol, "long", "m5")
        store.mark_auto_adopted(symbol, "long", "m5")

    path = tmp_path / "focus_auto_picks.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["market_date"] = date.today().isoformat()      # header says today
    payload["picks"]["STALE|long"]["session_date"] = "2020-01-01"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reopened = _store(tmp_path)
    assert reopened.is_auto_adopted("TODAY", "long", "m5") is True
    assert reopened.is_auto_adopted("STALE", "long", "m5") is False


def test_a_malformed_marker_entry_is_ignored_not_trusted(tmp_path):
    import json

    store = _store(tmp_path)
    store.add("NVDA", "long", "m5")
    store.mark_auto_adopted("NVDA", "long", "m5")

    path = tmp_path / "focus_auto_picks.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["picks"]["NVDA|long"] = "not a marker"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reopened = _store(tmp_path)
    assert reopened.auto_pick_markers() == {}
    assert reopened.remove_if_auto_adopted("NVDA", "long", "m5") is False
