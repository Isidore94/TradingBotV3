import sys

import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def _queue_mechanics_only(monkeypatch):
    """Routing off: these tests are about what the QUEUE does with a row.

    Since 2026-08-27 an ordinary intraday alert lists in the M5 alert bar
    instead of queueing a chart (trader rule; `test_qt_m5_alert_bar.py` owns
    that routing and its exemptions). The mechanics below - filters, expiry,
    verbs, badges - are the same for any row the queue holds, so they are
    exercised with the routing switched off rather than rewritten around D1
    fixtures that would drag the D1 feed into every assertion.
    """
    from ui.panels.alert_center_panel import AlertCenterPanel

    monkeypatch.setattr(
        AlertCenterPanel, "_is_m5_review_alert", staticmethod(lambda alert: False)
    )


def _alert(text, tag="green"):
    from ui.models.bounce import BounceAlert

    return BounceAlert.from_callback(text, tag)


def _pump_until(predicate, timeout=10.0):
    """Spin the event loop until ``predicate`` holds.

    Chart snapshots are built on worker threads (Part C rule C3), so the
    bars a panel shows arrive on a later turn of the loop, not inside the
    call that asked for them.
    """
    import os
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def test_tier_extraction():
    try:
        from ui.panels import alert_center_panel
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise
    extract_alert_tier = alert_center_panel.extract_alert_tier

    assert extract_alert_tier(_alert("[S-TIER] AAOI: Bounce confirmed (short)")) == "S"
    assert extract_alert_tier(_alert("[b-tier] X: Bounce confirmed")) == "B"
    assert extract_alert_tier(_alert("MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA")) == ""
    # BANGER retired 2026-09-01: the matcher is gone, not renamed.
    assert not hasattr(alert_center_panel, "is_banger_alert")


def test_min_tier_filter_policy():
    try:
        from ui.panels.alert_center_panel import alert_passes_min_tier
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    s_alert = _alert("[S-TIER] AAA: Bounce confirmed (long)")
    b_alert = _alert("[B-TIER] BBB: Bounce confirmed (long)")
    d_alert = _alert("[D-TIER] DDD: Bounce confirmed (short)")
    untiered = _alert("MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA (long) Favorite setup upgrade", "d1_flag_long")

    assert all(alert_passes_min_tier(a, "all") for a in (s_alert, b_alert, d_alert, untiered))
    assert alert_passes_min_tier(s_alert, "A")
    assert not alert_passes_min_tier(b_alert, "A")
    assert not alert_passes_min_tier(d_alert, "B")
    # Untiered info passes everything except S-only.
    assert alert_passes_min_tier(untiered, "A")
    assert not alert_passes_min_tier(untiered, "S")


def test_the_banger_token_no_longer_bypasses_the_tier_gate():
    """BANGER retired 2026-09-01 (trader: "We can probably remove this because
    idk what it is"). It was a literal token match with no producer anywhere in
    the tree - 0 of 8,818 review rows carried it - so the tier bypass and the
    always-sound it granted were unreachable privilege. A C-tier alert whose
    text happens to contain the word is now an ordinary C-tier alert.

    Fail-before-fix: on the un-fixed code both assertions below are False.
    """
    try:
        from ui.panels.alert_center_panel import alert_is_loud, alert_passes_min_tier
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    banger = _alert("[C-TIER] RW BANGER CCC (short): SPY paused")
    assert not alert_passes_min_tier(banger, "S")
    assert not alert_is_loud(banger)


def test_proven_bounces_bypass_tier_gate_and_sound():
    try:
        from ui.panels.alert_center_panel import alert_is_loud, alert_passes_min_tier, is_proven_alert
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    proven = _alert(
        "[A-TIER] PROVEN NVDA: Bounce confirmed (long) from dynamic_vwap_upper_band | "
        "proven: dynamic_vwap_upper_band: +0.88R (n=59)",
        "green",
    )
    assert is_proven_alert(proven)
    # The whole point: a proven config is visible and audible in EVERY gate mode.
    assert all(alert_passes_min_tier(proven, mode) for mode in ("all", "B", "A", "S"))
    assert alert_is_loud(proven)

    # Lowercase "proven negative" mute text must not counterfeit the stamp.
    muted_note = _alert("[B-TIER] AAA: Bounce confirmed (long) | why: midday long -0.40R proven negative", "green")
    assert not is_proven_alert(muted_note)
    assert not alert_passes_min_tier(muted_note, "S")


def test_entry_assist_output_bypasses_tier_gate_and_parses_clean():
    try:
        from ui.panels.alert_center_panel import alert_passes_min_tier, is_entry_assist_alert
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    window_open = _alert("ENTRY WINDOW OPEN (long): tracking RS holders while SPY counter-moves [manual].", "blue")
    window_ranked = _alert("ENTRY WINDOW (long): SPY -0.85% since 10:05 - held strongest through it: AAA +0.10% (x+0.95) [manual]", "green")
    strongest = _alert("STRONGEST 30M (long): NVDA +1.20%, AMD +0.90% [manual]", "green")
    weakest = _alert("WEAKEST 30M (short): CCC -1.10% [manual]", "red")
    failure_note = _alert("ENTRY ASSIST: No SPY session bars yet - cannot open an entry window.", "entry_assist")

    for alert in (window_open, window_ranked, strongest, weakest, failure_note):
        assert is_entry_assist_alert(alert)
        # The trader clicked for this output: it must survive every gate mode.
        assert all(alert_passes_min_tier(alert, mode) for mode in ("all", "B", "A", "S"))
        # List-style output, not a single-symbol alert: no bogus "(LONG)" symbol.
        assert alert.symbol == ""

    ordinary = _alert("[B-TIER] AAA: Bounce confirmed (long)")
    assert not is_entry_assist_alert(ordinary)
    assert ordinary.symbol == "AAA"


def test_loud_alerts_are_sa_proven_or_ready_d1():
    try:
        from ui.panels.alert_center_panel import alert_is_loud
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    assert alert_is_loud(_alert("[S-TIER] AAA: Bounce confirmed"))
    assert alert_is_loud(_alert("[A-TIER] AAA: Bounce confirmed"))
    assert alert_is_loud(_alert("[D-TIER] PROVEN MSTR (long): Bounce confirmed"))
    assert not alert_is_loud(_alert("[B-TIER] AAA: Bounce confirmed"))
    # A level-cross trigger is developing evidence; only the final bucket
    # upgrade is a D1 Focus/loud moment.
    assert not alert_is_loud(_alert("MASTER_AVWAP_D1_UPGRADE_TRIGGER: AAPL (long) 1st-dev break UPPER_1@314.57", "d1_flag_long"))
    assert alert_is_loud(_alert("MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA (long) Favorite setup upgrade", "d1_flag_long"))
    assert not alert_is_loud(_alert("MASTER_AVWAP_D1_UPGRADE_WATCH: AAPL (long) AVWAPE retest", "d1_flag_long"))
    # A confirmed tier flip is THE D1 Focus moment - always loud.
    assert alert_is_loud(
        _alert(
            "MASTER_AVWAP_D1_TIER_FLIP: HOMB (long) non-S/A -> A/S predicted "
            "(next scan confirms) 1st-dev break [@102.00; px=102.30]",
            "d1_flag_long",
        )
    )
    # The pause-watch summary line stays quiet by design.
    assert not alert_is_loud(_alert("REGIME PAUSE WATCH (short): SPY paused (+0.15% window) - 3 swing shorts still pressing lows: A, B, C", "red"))


def test_developing_d1_crossings_are_research_only():
    try:
        from ui.panels.alert_center_panel import is_developing_d1_alert
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    developing = (
        _alert(
            "MASTER_AVWAP_D1_RESEARCH: DDOG (long) Developing level observation: "
            "2nd-dev break UPPER_2@270.86",
            "d1_flag_long",
        ),
        # Defensive compatibility: old bot processes can still have one of
        # these messages queued while the GUI is being upgraded.
        _alert(
            "MASTER_AVWAP_D1_UPGRADE_TRIGGER: AAPL (long) A/S upgrade: "
            "1st-dev break UPPER_1@314.57",
            "d1_flag_long",
        ),
        _alert(
            "MASTER_AVWAP_D1_UPGRADE_WATCH: AAPL (long) AVWAPE retest AVWAPE@309.38",
            "d1_flag_long",
        ),
    )
    assert all(is_developing_d1_alert(alert) for alert in developing)
    assert not is_developing_d1_alert(
        _alert(
            "MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA (long) Favorite setup upgrade",
            "d1_flag_long",
        )
    )
    assert not is_developing_d1_alert(
        _alert("MASTER_AVWAP_D1_FLAG: MSFT (short) 15EMA break", "d1_flag_short")
    )


def test_actionable_feeds_exclude_developing_d1_crossings(monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    panel = AlertCenterPanel()
    upgrade = _alert("MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA (long) Favorite setup upgrade [score=245]", "d1_flag_long")
    trigger = _alert("MASTER_AVWAP_D1_UPGRADE_TRIGGER: AAPL (long) A/S upgrade: 1st-dev break UPPER_1@314.57", "d1_flag_long")
    watch = _alert("MASTER_AVWAP_D1_UPGRADE_WATCH: AAPL (long) AVWAPE retest AVWAPE@309.38", "d1_flag_long")
    research = _alert(
        "MASTER_AVWAP_D1_RESEARCH: DDOG (long) Developing level observation: "
        "2nd-dev break UPPER_2@270.86",
        "d1_flag_long",
    )
    generic = _alert("MASTER_AVWAP_D1_FLAG: MSFT (short) 15EMA break [score=88]", "d1_flag_short")
    for alert in (upgrade, trigger, watch, research, generic):
        panel.add_alert(alert)

    # Only a completed rescan's become-a-favorite moment is actionable.
    d1_texts = [a.raw_text for a in panel._d1_alerts]
    assert [t.split(":", 1)[0] for t in d1_texts] == ["MASTER_AVWAP_D1_BUCKET_UPGRADE"]
    # Developing observations are research evidence, not live alerts. Generic
    # champion D1 flags retain their existing live-stream behavior.
    live_prefixes = {a.raw_text.split(":", 1)[0] for a in panel._alerts}
    assert live_prefixes == {"MASTER_AVWAP_D1_FLAG"}


def test_tier_flip_and_zone_alerts_route_to_d1_focus_feed(monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel, is_ready_d1_alert
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    tier_flip = _alert(
        "MASTER_AVWAP_D1_TIER_FLIP: HOMB (long) non-S/A -> A/S predicted "
        "(next scan confirms) 1st-dev break [@102.00; px=102.30; bar=10:35; "
        "was: bucket none, score 62; ctx B-tier; rvol 1.50]",
        "d1_flag_long",
    )
    zone = _alert(
        "MASTER_AVWAP_D1_ZONE: NVDA (long) zone1 bounce off AVWAPE [@100.00; px=102.00]",
        "d1_flag_long",
    )
    assert is_ready_d1_alert(tier_flip)
    assert is_ready_d1_alert(zone)

    panel = AlertCenterPanel()
    panel.add_alert(tier_flip)
    panel.add_alert(zone)
    d1_prefixes = [a.raw_text.split(":", 1)[0] for a in panel._d1_alerts]
    # Feed renders newest-first.
    assert d1_prefixes == ["MASTER_AVWAP_D1_ZONE", "MASTER_AVWAP_D1_TIER_FLIP"]
    # Neither leaks into the live bounce feed.
    assert not panel._alerts


def test_entry_assist_board_renders_all_sections():
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.widgets.entry_assist_board import EntryAssistBoard
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    board = EntryAssistBoard()
    assert "fills automatically" in board.view.toPlainText()

    board.update_board(
        {
            "env_key": "bullish_strong",
            "env_label": "Bullish Strong",
            "bar_time": "10:35",
            "movers_minutes": 30,
            "pause": {"trend_side": "long", "detected": True, "since": "10:20"},
            "window": {
                "active": True,
                "sides": ["long"],
                "started": "10:20",
                "source": "auto",
                "spy_pct": -0.42,
                "rankings": {"long": [{"symbol": "AAA", "window_pct": 0.15, "excess": 0.57}]},
            },
            "movers": {
                "long": [{"symbol": "NVDA", "change_pct": 1.2, "excess": 1.1}],
                "short": [{"symbol": "CCC", "change_pct": -1.4, "excess": 1.3}],
            },
        }
    )
    text = board.view.toPlainText()
    assert "PULLBACK DETECTED" in text
    assert "AAA" in text and "NVDA" in text and "CCC" in text
    assert "Live window (long) since 10:20" in text
    assert "Bullish Strong" in board.title_label.text()

    # Pause with no window shows the preview ranking instead.
    board.update_board(
        {
            "env_key": "bullish_strong",
            "env_label": "Bullish Strong",
            "bar_time": "10:40",
            "movers_minutes": 30,
            "pause": {"trend_side": "long", "detected": True, "since": "10:20"},
            "window": {"active": False},
            "pause_preview": {
                "side": "long",
                "since": "10:20",
                "spy_pct": -0.3,
                "rows": [{"symbol": "BBB", "window_pct": 0.05, "excess": 0.35}],
            },
            "movers": {"long": [], "short": []},
        }
    )
    text = board.view.toPlainText()
    assert "Pause preview (long)" in text and "BBB" in text

    board.update_board({})
    assert "fills automatically" in board.view.toPlainText()


def test_rs_rw_boards_emit_snapshot_symbols(monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QUrl
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.entry_assist_board import EntryAssistBoard, _board_html as entry_html
        from ui.widgets.rrs_snapshot import RrsSnapshotWidget, _board_html as rrs_html
        import ui.widgets.symbol_snapshot_dialog as snapshot_dialog
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    entry_payload = {
        "movers": {
            "long": [{"symbol": "NVDA", "change_pct": 1.2}],
            "short": [{"symbol": "TSLA", "change_pct": -1.4}],
        }
    }
    assert "snapshot://long/NVDA" in entry_html(entry_payload)
    assert "snapshot://short/TSLA" in entry_html(entry_payload)
    entry = EntryAssistBoard()
    entry_calls = []
    entry.symbolActivated.connect(lambda symbol, side: entry_calls.append((symbol, side)))
    entry._on_anchor_clicked(QUrl("snapshot://long/NVDA"))
    assert entry_calls == [("NVDA", "LONG")]

    rrs_payload = {
        "threshold": 0.5,
        "results": [("RS", "AMD", 2.1), ("RW", "META", -1.8)],
    }
    html = rrs_html(rrs_payload)
    assert "snapshot://long/AMD" in html
    assert "snapshot://short/META" in html
    snapshot = RrsSnapshotWidget()
    snapshot_calls = []
    snapshot.symbolActivated.connect(
        lambda symbol, side: snapshot_calls.append((symbol, side))
    )
    snapshot._on_anchor_clicked(QUrl("snapshot://short/META"))
    assert snapshot_calls == [("META", "SHORT")]

    bot = object()

    class _Service:
        def current_bot(self):
            return bot

    panel = AlertCenterPanel()
    panel._bounce_service = _Service()
    popup_calls = []
    monkeypatch.setattr(
        snapshot_dialog,
        "show_symbol_snapshot",
        lambda owner, symbol, **kwargs: popup_calls.append(
            (owner, symbol, kwargs.get("bot"), kwargs.get("side"))
        ),
    )
    panel.entry_board.symbolActivated.emit("NVDA", "LONG")
    panel.rrs_snapshot.symbolActivated.emit("META", "SHORT")
    assert popup_calls == [
        (panel, "NVDA", bot, "LONG"),
        (panel, "META", bot, "SHORT"),
    ]


def test_liked_focus_picks_skip_tier_gate_and_always_sound():
    try:
        from ui.panels.alert_center_panel import alert_passes_feed_gate, alert_should_sound
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    quiet_b = _alert("[B-TIER] AAA: Bounce confirmed (long)")
    # Not liked: obeys the tier gate and stays quiet.
    assert not alert_passes_feed_gate(quiet_b, "A", is_focus=False)
    assert not alert_should_sound(quiet_b, is_focus=False)
    # Liked (focus) picks surface through every gate, even S-only, and sound.
    # Since 2026-08-05 `is_focus` means "a Focus pick that has broken
    # yesterday's extreme in its direction" - see the panel's
    # _alert_has_focus_privilege and test_focus_privilege_waits_for_the_
    # previous_day_extreme below. The policy these two functions encode is
    # unchanged.
    assert alert_passes_feed_gate(quiet_b, "A", is_focus=True)
    assert alert_passes_feed_gate(quiet_b, "S", is_focus=True)
    assert alert_should_sound(quiet_b, is_focus=True)


def _pin_repetition_clock(monkeypatch):
    """Pin the repetition ledger's clock outside the open-digest window.

    Between the open and open+30 an ordinary alert is folded into the open
    digest - no feed row - so a test asserting that an S-tier alert reaches the
    feed on its own merit fails inside that half hour on correct behaviour.
    Found 2026-08-23 at 06:46 PT. The digest stays enabled; only the clock is
    fixed, to a moment outside any session's first half hour.
    """
    import alert_repetition
    from datetime import datetime as _datetime

    class _PinnedClock(_datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401 - matches datetime.now
            return cls(2026, 8, 21, 12, 0, 0)

    monkeypatch.setattr(alert_repetition, "datetime", _PinnedClock)


def test_focus_privilege_waits_for_the_previous_day_extreme(tmp_path, monkeypatch):
    """Trader rule 2026-08-05: a Focus long inside yesterday's range is noise.

    It loses the Focus privileges (tier-gate bypass, always-sound) until it
    trades above the previous day's high - but it is not blacked out: an
    S-tier bounce on it still reaches the feed on its own merit.
    """
    try:
        import os
        from datetime import datetime

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    _pin_repetition_clock(monkeypatch)

    class _FocusService:
        def is_focus(self, symbol, side=None, category=None):
            return symbol == "NVDA"

        def focus_category(self, symbol):
            return "m5" if symbol == "NVDA" else None

        def focus_side(self, symbol, category=None):
            return "long" if symbol == "NVDA" else None

        def all_focus(self, category=None):
            return {"long": ["NVDA"], "short": []}

    def alert_for(trigger):
        return BounceAlert(
            time_text="09:35:00",
            symbol="NVDA",
            side="LONG",
            trigger=trigger,
            timeframe="5m",
            tag="green",
            raw_text=f"{trigger} NVDA",
        )

    panel = AlertCenterPanel(parked_symbols_path=tmp_path / "parked.json")
    panel.focus_service = _FocusService()
    # S tier / PROVEN only. Set directly rather than through the combo box:
    # the widget persists the choice to machine-local settings, which would
    # leak this filter into every panel a later test builds.
    monkeypatch.setattr(panel, "_min_tier_mode", lambda: "S")

    quiet = alert_for("[B-TIER] Bounce confirmed")
    loud = alert_for("[S-TIER] Bounce confirmed")

    # Unmeasured (the poll has not reached it): membership yes, privilege no.
    assert panel._alert_is_focus(quiet)
    assert not panel._alert_has_focus_privilege(quiet)
    panel.add_alert(quiet)
    assert quiet not in _feed_alerts(panel)

    # Measured and still inside yesterday's range: same answer.
    panel._focus_break_state["NVDA|long"] = "closed"
    assert not panel._alert_has_focus_privilege(quiet)
    # ...but the strong bounce is never swallowed - it passes on tier.
    panel.add_alert(loud)
    assert loud in _feed_alerts(panel)

    # Above yesterday's high: the Focus privilege is back and the quiet
    # B-tier surfaces through the S-only gate again.
    panel._focus_break_state["NVDA|long"] = "open"
    panel._focus_break_open_at["NVDA|long"] = datetime(2026, 8, 5, 11, 4)
    assert panel._alert_has_focus_privilege(quiet)
    again = alert_for("[B-TIER] Bounce confirmed again")
    panel.add_alert(again)
    assert again in _feed_alerts(panel)

    # A long that only broke yesterday's LOW earns nothing: the gate is
    # directional, and the state is keyed per side.
    panel._focus_break_state["NVDA|long"] = "closed"
    panel._focus_break_state["NVDA|short"] = "open"
    assert not panel._alert_has_focus_privilege(quiet)


def test_one_extension_flag_per_pick_then_only_pullbacks(tmp_path, monkeypatch):
    """FRPT's case (2026-08-05): it printed a new 20-day high and then simply
    stayed extended. The first extension event spends the whole extension set
    for the day; the pullback set stays live so the pick can still speak when
    it comes back to a level."""
    try:
        import os
        from datetime import datetime

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from chart_watch import D1_EXTENSION_KINDS, D1_PULLBACK_KINDS
        from ui.panels import alert_center_panel as panel_mod
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(
        SymbolSnapshotWidget, "set_symbol", lambda self, symbol, **kwargs: None
    )

    class _FocusService:
        def is_focus(self, symbol, side=None, category=None):
            return symbol == "FRPT"

        def focus_category(self, symbol):
            return "m5" if symbol == "FRPT" else None

        def focus_side(self, symbol, category=None):
            return "long"

        def all_focus(self, category=None):
            return {"long": ["FRPT"], "short": []}

    d1_bars = [
        {
            "dt": datetime(2026, 8, day, 0, 0),
            "high": 63.59 if day == 4 else 62.0,
            "low": 60.0,
            "close": 61.5,
        }
        for day in (3, 4)
    ]
    m5_bars = [
        {"dt": datetime(2026, 8, 5, 6, 30), "high": 68.68, "low": 62.0, "close": 68.0},
    ]

    fires: set[str] = set()
    evaluated: list[str] = []

    class _Hit:
        message = "event"

    def _fake_evaluate(watch, m5, d1, *, now=None, avwape_anchor=None, levels_cache=None):
        evaluated.append(watch.kind)
        return _Hit() if watch.kind in fires else None

    monkeypatch.setattr(panel_mod, "evaluate_d1_event_watch", _fake_evaluate)

    panel = AlertCenterPanel(
        parked_symbols_path=tmp_path / "parked.json",
        focus_d1_flags_path=tmp_path / "focus_flags.json",
    )
    panel.focus_service = _FocusService()
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: list(d1_bars))
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol: list(m5_bars))

    # 06:40 - the breakout prints a new 20-day high.
    fires.add("new_20d_high")
    panel._poll_focus_d1_interest(now=datetime(2026, 8, 5, 6, 40))
    assert [a.trigger for a in panel._d1_alerts] == ["Focus D1 · event"]
    assert panel._focus_extension_spent("FRPT")

    # Later: EVERY kind would hit. Only the pullback set is even measured -
    # no second "still breaking out" flag on a name that is now extended.
    evaluated.clear()
    fires.update(D1_EXTENSION_KINDS | D1_PULLBACK_KINDS)
    panel._poll_focus_d1_interest(now=datetime(2026, 8, 5, 9, 40))
    assert set(evaluated) <= D1_PULLBACK_KINDS
    assert not (set(evaluated) & D1_EXTENSION_KINDS)
    assert evaluated, "the pullback events must still be live"
    fired_kinds = {
        flag.split("|", 1)[1] for flag in panel._focus_d1_flags if flag.startswith("FRPT|")
    }
    assert fired_kinds & D1_PULLBACK_KINDS  # it can still speak on a bounce
    assert fired_kinds & D1_EXTENSION_KINDS == {"new_20d_high"}


def test_a_focus_picks_own_chart_can_remove_it(tmp_path, monkeypatch):
    """2026-08-05: "there's no way of removing this pick from the focus picks".
    On a chart for a name already in Focus the primary verb IS the removal,
    and it takes the focus-injected watchlist line with it."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import FOCUS_D1_EVENT_TAG, BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
        from test_qt_focus_panel import _service
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(
        SymbolSnapshotWidget, "set_symbol", lambda self, symbol, **kwargs: None
    )
    service = _service(tmp_path)
    service.add("FRPT", "long", "m5", origin="test", context="")
    assert "FRPT" in (tmp_path / "longs.txt").read_text(encoding="utf-8")

    panel = AlertCenterPanel(
        service,
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "review_events.jsonl",
    )
    panel.add_alert(
        BounceAlert(
            time_text="06:30:00",
            symbol="FRPT",
            side="LONG",
            trigger="Focus D1 · New 20-day high: 68.68 > 63.59",
            timeframe="D1",
            tag=FOCUS_D1_EVENT_TAG,
            raw_text="FOCUS D1 FRPT (LONG): New 20-day high",
            is_d1=True,
        )
    )
    assert panel._current_review_alert is not None
    # The verb that used to read "Add to Swing Focus" on a name the trader
    # already owns is now the removal.
    assert panel.chart_review.focus_button.text() == "✕ Remove from Focus"

    panel.chart_review.focus_button.click()
    assert not service.is_focus("FRPT")
    assert "FRPT" not in (tmp_path / "longs.txt").read_text(encoding="utf-8")
    # Removing the pick is not the same as muting the symbol for the day.
    assert "FRPT" not in panel._ignored_symbols

    # A name that is NOT a Focus pick keeps the add verb.
    panel.add_alert(
        BounceAlert(
            time_text="09:35:00",
            symbol="AMD",
            side="LONG",
            trigger="[S-TIER] VWAP reclaim",
            timeframe="5m",
            raw_text="[S-TIER] AMD: VWAP reclaim",
        )
    )
    assert panel.chart_review.focus_button.text() == "Add to M5 Focus"


def _feed_alerts(panel) -> list:
    """The alerts actually rendered in the live feed (not merely recorded)."""
    layout = panel.feed_layout
    items = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        alert = getattr(widget, "alert", None)
        if alert is not None:
            items.append(alert)
    return items


def test_focus_d1_flags_hold_until_the_break_and_then_start_there(tmp_path, monkeypatch):
    """The open-time flood fix: no automatic D1 flag while the name is inside
    yesterday's range, and when it breaks out the event window starts AT the
    break - the 09:35 reject it printed while still below never fires."""
    try:
        import os
        from datetime import datetime

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels import alert_center_panel as panel_mod
        from ui.panels.alert_center_panel import AlertCenterPanel
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    class _FocusService:
        def is_focus(self, symbol, side=None, category=None):
            return symbol == "NVDA"

        def focus_category(self, symbol):
            return "m5" if symbol == "NVDA" else None

        def focus_side(self, symbol, category=None):
            return "long" if symbol == "NVDA" else None

        def all_focus(self, category=None):
            return {"long": ["NVDA"], "short": []}

    # Yesterday: high 100.00, low 95.00.
    d1_bars = [
        {
            "dt": datetime(2026, 8, day, 0, 0),
            "high": 100.0 if day == 4 else 99.0,
            "low": 95.0 if day == 4 else 94.0,
            "close": 98.0,
        }
        for day in (3, 4)
    ]
    # Today: still under yesterday's high through 09:40, breaks it at 11:00.
    m5_bars = [
        {"dt": datetime(2026, 8, 5, 9, 35), "high": 99.5, "low": 98.0, "close": 99.0},
        {"dt": datetime(2026, 8, 5, 9, 40), "high": 99.8, "low": 98.5, "close": 99.4},
        {"dt": datetime(2026, 8, 5, 11, 0), "high": 101.5, "low": 99.5, "close": 101.2},
    ]

    evaluated = []  # (kind, armed_at) per D1 event actually measured

    class _Hit:
        message = "5d high"

    def _fake_evaluate(watch, m5, d1, *, now=None, avwape_anchor=None, levels_cache=None):
        evaluated.append((watch.kind, watch.armed_at))
        return _Hit() if watch.kind == "new_5d_high" else None

    monkeypatch.setattr(panel_mod, "evaluate_d1_event_watch", _fake_evaluate)

    panel = AlertCenterPanel(
        parked_symbols_path=tmp_path / "parked.json",
        focus_d1_flags_path=tmp_path / "focus_flags.json",
    )
    panel.focus_service = _FocusService()
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: list(d1_bars))
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol: list(m5_bars))

    # 09:45, price 99.40 vs yesterday's 100.00 high: nothing is even evaluated,
    # nothing is consumed, and the trader can see WHY the feed is quiet.
    panel._poll_focus_d1_interest(now=datetime(2026, 8, 5, 9, 45))
    assert evaluated == []
    assert panel._focus_d1_flags == set()
    assert not panel._d1_alerts
    assert panel._focus_gate_held == 1
    assert panel.focus_break_state("NVDA", "long") == "closed"

    # 11:06, the 11:00 bar has completed above 100.00: the window opens HERE,
    # not at midnight, so the morning's events stay unflagged.
    moment = datetime(2026, 8, 5, 11, 6)
    breakout_bar = datetime(2026, 8, 5, 11, 0)
    panel._poll_focus_d1_interest(now=moment)
    # The window opens at the breaking BAR, not at the poll tick - so the
    # breakout bar's own events count, while 09:35/09:40 (which end at 09:40
    # and 09:45, before 11:00) stay outside it.
    assert evaluated and {armed_at for _kind, armed_at in evaluated} == {breakout_bar}
    assert ("new_5d_high", breakout_bar) in evaluated
    assert panel.focus_break_state("NVDA", "long") == "open"
    assert panel._focus_gate_held == 0
    assert "NVDA|new_5d_high" in panel._focus_d1_flags
    assert [alert.symbol for alert in panel._d1_alerts] == ["NVDA"]
    assert panel._d1_alerts[0].tag == panel_mod.FOCUS_D1_EVENT_TAG

    # The flag is once-per-session: the fired kind is no longer even measured,
    # and the window keeps its original 11:00 open stamp.
    evaluated.clear()
    panel._poll_focus_d1_interest(now=datetime(2026, 8, 5, 11, 30))
    assert "new_5d_high" not in {kind for kind, _armed_at in evaluated}
    assert {armed_at for _kind, armed_at in evaluated} == {breakout_bar}
    assert len(panel._d1_alerts) == 1


def test_visual_alert_review_queues_skips_and_adds_focus(tmp_path, monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
        from test_qt_focus_panel import _service
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    rendered = []
    monkeypatch.setattr(
        SymbolSnapshotWidget,
        "set_symbol",
        lambda self, symbol, **kwargs: rendered.append(symbol),
    )
    service = _service(tmp_path)
    panel = AlertCenterPanel(
        service,
        ignored_symbols_path=tmp_path / "alert_center_ignored.txt",
    )
    first = BounceAlert(
        time_text="09:35:00",
        symbol="NVDA",
        side="LONG",
        trigger="[S-TIER] VWAP reclaim",
        timeframe="5m",
        raw_text="[S-TIER] NVDA: VWAP reclaim",
    )
    second = BounceAlert(
        time_text="09:40:00",
        symbol="TSLA",
        side="SHORT",
        trigger="[S-TIER] EMA rejection",
        timeframe="5m",
        raw_text="[S-TIER] TSLA: EMA rejection",
    )

    panel.add_alert(first)
    panel.add_alert(second)

    assert panel._current_review_alert is first
    assert [alert.symbol for alert in panel._review_queue] == ["TSLA"]
    assert panel.chart_review.alert is first
    assert rendered == ["NVDA"]

    panel._skip_review_alert(first)
    assert panel._current_review_alert is second
    assert panel.chart_review.alert is second
    assert rendered == ["NVDA", "TSLA"]
    assert not service.is_focus("TSLA")

    panel._add_review_alert_to_focus(second)
    assert service.is_focus("TSLA", "short", "m5")
    assert panel._current_review_alert is None
    assert panel.chart_review.alert is None


def test_list_callback_posts_every_m5_name_to_review(tmp_path, monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
        from test_qt_focus_panel import _service
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    panel = AlertCenterPanel(
        _service(tmp_path),
        ignored_symbols_path=tmp_path / "alert_center_ignored.txt",
    )
    alerts = BounceAlert.from_callback_many(
        "STRONGEST 30M (long): NVDA +1.20%, AMD +0.90%, META +0.70% [manual]",
        "green",
    )

    for alert in alerts:
        panel.add_alert(alert)

    assert [alert.symbol for alert in panel._alerts] == ["META", "AMD", "NVDA"]
    assert panel._current_review_alert.symbol == "NVDA"
    assert [alert.symbol for alert in panel._review_queue] == ["AMD", "META"]
    assert all(alert.timeframe == "M5" for alert in alerts)


def test_visual_remove_suppresses_for_today_and_restore(tmp_path, monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
        from test_qt_focus_panel import _service
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    ignored_path = tmp_path / "alert_center_ignored.txt"
    service = _service(tmp_path)
    panel = AlertCenterPanel(service, ignored_symbols_path=ignored_path)
    disliked = BounceAlert(
        time_text="09:35:00",
        symbol="NVDA",
        side="LONG",
        trigger="[S-TIER] visual rejection",
        timeframe="D1",
        raw_text="[S-TIER] NVDA: visual rejection",
        is_d1=True,
    )
    next_alert = BounceAlert(
        time_text="09:40:00",
        symbol="TSLA",
        side="SHORT",
        trigger="[S-TIER] next",
        timeframe="5m",
        raw_text="[S-TIER] TSLA: next",
    )
    panel.add_alert(disliked)
    panel.add_alert(next_alert)

    panel._remove_review_alert_for_today(disliked)

    import json

    stored = json.loads(ignored_path.read_text(encoding="utf-8"))
    assert stored["market_date"] == panel._ignored_market_date
    assert stored["symbols"] == ["NVDA"]
    assert panel._ignored_symbols == {"NVDA"}
    assert all(alert.symbol != "NVDA" for alert in panel._alerts + panel._d1_alerts)
    assert panel._current_review_alert is next_alert

    # Future alerts stay off both the feed and review queue for this date.
    panel.add_alert(disliked)
    assert all(alert.symbol != "NVDA" for alert in panel._alerts + panel._d1_alerts)
    assert all(alert.symbol != "NVDA" for alert in panel._review_queue)

    panel._restore_ignored_symbol("NVDA")
    assert json.loads(ignored_path.read_text(encoding="utf-8"))["symbols"] == []
    panel.add_alert(disliked)
    assert any(alert.symbol == "NVDA" for alert in panel._alerts)
    assert any(alert.symbol == "NVDA" for alert in panel._review_queue)


def test_not_today_preserves_and_fires_trader_armed_d1_watches(tmp_path, monkeypatch):
    """Trader rule: dismissal cannot cancel, defer, mute, or hide an alarm they armed."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from datetime import datetime, timedelta
        import dataclasses

        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels import alert_center_panel as panel_mod
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    beeps = []
    monkeypatch.setattr(panel_mod.QApplication, "beep", lambda: beeps.append("beep"))

    noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)

    def m5_bar(minute, high, low, close):
        return {
            "dt": noon.replace(hour=11, minute=minute),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000.0,
        }

    prior_daily = [
        {
            "dt": (noon - timedelta(days=offset)).replace(hour=0, minute=0),
            "open": 100.0,
            "high": 105.0 + (6 - offset),
            "low": 95.0,
            "close": 100.0,
            "volume": 1_000.0,
        }
        for offset in range(6, 0, -1)
    ]
    bars = [m5_bar(20, 108.0, 102.0, 106.0)]

    panel = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        chart_watches_path=tmp_path / "chart_watches.json",
        d1_level_watches_path=tmp_path / "d1_levels.json",
        d1_event_watches_path=tmp_path / "d1_events.json",
    )
    panel._alerts_may_sound = lambda: True
    monkeypatch.setattr(panel, "_m5_bars_for", lambda _symbol: list(bars))
    monkeypatch.setattr(panel, "_d1_bars_for", lambda _symbol: list(prior_daily))

    assert panel.arm_chart_watch_for("NVDA", "LONG", "new_hod")
    assert panel.arm_d1_level_watch("NVDA", "above", 110.0)
    assert panel.arm_d1_event_watch("NVDA", "new_5d_high")
    armed_at = noon.replace(hour=11, minute=40)
    panel._d1_level_watches[0] = dataclasses.replace(
        panel._d1_level_watches[0], armed_at=armed_at
    )
    panel._d1_event_watches[0] = dataclasses.replace(
        panel._d1_event_watches[0], armed_at=armed_at
    )

    panel._ignore_alert_symbol("NVDA")
    assert [watch.kind for watch in panel._chart_watches] == ["new_hod"]
    assert len(panel._d1_level_watches) == len(panel._d1_event_watches) == 1

    bars.append(m5_bar(45, 111.0, 106.0, 110.5))
    panel._poll_d1_level_watches(now=noon)
    panel._poll_d1_event_watches(now=noon)

    assert panel._d1_level_watches == []
    assert panel._d1_event_watches == []
    fired = [alert for alert in panel._alerts if alert.symbol == "NVDA"]
    assert len(fired) == 2
    assert all(alert.tag == panel_mod.CHART_WATCH_TAG for alert in fired)
    assert {alert.payload.get("chart_watch_kind") for alert in fired} == {
        "d1_level_above",
        "new_5d_high",
    }
    assert panel.feed_layout.count() >= 2
    assert len(beeps) == 2


def test_ignored_focus_derived_d1_interest_still_does_not_fire(tmp_path, monkeypatch):
    """The automatic Focus carve-out remains ignored; it is not trader-armed."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from datetime import datetime

        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert, FOCUS_D1_EVENT_TAG
        from ui.panels import alert_center_panel as panel_mod
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    class _FocusService:
        def all_focus(self):
            return {"long": ["NVDA"], "short": []}

        def is_focus(self, symbol, side=None, category=None):
            return symbol == "NVDA"

        def focus_category(self, symbol):
            return "swing" if symbol == "NVDA" else None

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    beeps = []
    evaluated = []
    monkeypatch.setattr(panel_mod.QApplication, "beep", lambda: beeps.append("beep"))
    monkeypatch.setattr(
        panel_mod,
        "evaluate_d1_event_watch",
        lambda *args, **kwargs: evaluated.append(args[0].kind),
    )

    panel = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        focus_d1_flags_path=tmp_path / "focus_flags.json",
    )
    panel.focus_service = _FocusService()
    panel._alerts_may_sound = lambda: True
    panel._ignored_symbols.add("NVDA")
    monkeypatch.setattr(panel, "_m5_bars_for", lambda _symbol: [{"dt": datetime.now()}])
    monkeypatch.setattr(panel, "_d1_bars_for", lambda _symbol: [{"dt": datetime.now()}])

    panel._poll_focus_d1_interest(now=datetime.now())
    assert evaluated == []
    assert panel._focus_d1_flags == set()
    assert panel._alerts == panel._d1_alerts == []
    assert beeps == []

    # Even a late automatic delivery cannot use the trader-armed tag exemption.
    panel.add_alert(
        BounceAlert(
            time_text="12:00:00",
            symbol="NVDA",
            side="LONG",
            trigger="Focus D1 automatic interest",
            timeframe="D1",
            tag=FOCUS_D1_EVENT_TAG,
            raw_text="FOCUS D1 NVDA automatic interest",
            is_d1=True,
        )
    )
    assert panel._alerts == panel._d1_alerts == []
    assert beeps == []


def test_chart_watch_hits_bypass_tier_gate_and_sound():
    try:
        from ui.models.bounce import BounceAlert, CHART_WATCH_TAG, is_chart_watch_alert
        from ui.panels.alert_center_panel import alert_is_loud, alert_passes_min_tier
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    hit = BounceAlert(
        time_text="10:35:00",
        symbol="NVDA",
        side="LONG",
        trigger="New HOD 111.00 > armed day high 110.00 (bar 10:30)",
        timeframe="M5",
        tag=CHART_WATCH_TAG,
        raw_text="CHART WATCH NVDA (LONG): New HOD 111.00 > armed day high 110.00 (bar 10:30)",
        payload={"chart_watch_kind": "new_hod"},
    )
    assert is_chart_watch_alert(hit)
    # The trader armed this exact condition: visible and audible in EVERY mode.
    assert all(alert_passes_min_tier(hit, mode) for mode in ("all", "B", "A", "S"))
    assert alert_is_loud(hit)
    assert not is_chart_watch_alert(_alert("[B-TIER] AAA: Bounce confirmed (long)"))


def test_review_watch_buttons_arm_trigger_and_flag_red(monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        # Probe, not a use: this whole block exists so a missing PySide6 skips
        # the test instead of erroring. Dropping the import would drop the probe.
        from ui import theme  # noqa: F401
        from ui.models.bounce import BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.alert_feed_item import AlertFeedItem
        from ui.widgets.badge import Badge
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    import dataclasses
    from datetime import datetime

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)

    # Anchor every bar to fixed clock times on TODAY's date so completion
    # math is deterministic regardless of when the suite runs.
    noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)

    def bar(minute, high, low):
        mid = (high + low) / 2
        return {
            "dt": noon.replace(hour=11, minute=minute),
            "open": mid,
            "high": high,
            "low": low,
            "close": mid,
            "volume": 1000.0,
        }

    class _Bot:
        def __init__(self):
            self.bars = []

        def m5_chart_bars(self, symbol, max_sessions=2):
            return list(self.bars)

    class _Service:
        def __init__(self, bot):
            self._bot = bot

        def current_bot(self):
            return self._bot

    bot = _Bot()
    bot.bars = [bar(20, 110.0, 99.0), bar(25, 108.0, 100.0)]
    panel = AlertCenterPanel()
    # The show-time filter is off here for the same reason the routing is:
    # this test is about the WATCH BUTTONS, not about which rows reach the
    # pane. Leaving it on made the test a clock bomb - the fixture's last bar
    # starts at 11:25, so before 11:30 local it was still forming and the VWAP
    # side read UNKNOWN (which shows), while after 11:30 both bars complete,
    # the fixture's long sits under its own session VWAP, and the 2026-08-27
    # rule correctly hid the chart. The bars' comment above claims the timing
    # is deterministic; with the filter on it was not.
    panel._review_movers_only = False
    panel._bounce_service = _Service(bot)
    alert = BounceAlert(
        time_text="11:30:00",
        symbol="NVDA",
        side="LONG",
        trigger="[S-TIER] VWAP reclaim",
        timeframe="5m",
        raw_text="[S-TIER] NVDA: VWAP reclaim",
    )
    panel.add_alert(alert)
    assert panel._current_review_alert is alert

    button = panel.chart_review.watch_buttons["new_hod"]
    assert button.isEnabled()
    button.click()
    assert [watch.kind for watch in panel._chart_watches] == ["new_hod"]
    assert panel._chart_watches[0].baseline == 110.0
    assert panel._chart_watches[0].symbol == "NVDA"
    # The armed button stays clickable, showing its armed state.
    assert button.isEnabled() and button.isChecked()
    assert "armed" in button.text()

    # It is a TOGGLE: a second click disarms (the stuck-armed bug).
    button.click()
    assert panel._chart_watches == []
    assert not button.isChecked()
    assert button.text() == "New HOD"

    # Re-arm for the trigger flow below.
    button.click()
    assert [watch.kind for watch in panel._chart_watches] == ["new_hod"]

    # Backdate the arm, then complete a bar that breaks the armed day high.
    panel._chart_watches[0] = dataclasses.replace(
        panel._chart_watches[0], armed_at=noon.replace(hour=11, minute=40)
    )
    bot.bars = bot.bars + [bar(45, 111.0, 104.0)]
    panel._poll_chart_watches(now=noon)

    # One-shot: the watch retires, the red alert leads the live feed, and the
    # button unlocks for a re-arm.
    assert panel._chart_watches == []
    fired = panel._alerts[0]
    assert fired.tag == "chart_watch"
    assert fired.symbol == "NVDA"
    assert "New HOD 111.00" in fired.trigger
    assert panel.chart_review.watch_buttons["new_hod"].isEnabled()
    assert not panel.chart_review.watch_buttons["new_hod"].isChecked()

    # The requested red-font flag: red trigger text plus a red kind badge.
    # The colour moved into theme.qss on 2026-08-21 (a per-row setStyleSheet is
    # a CSS parse per row), so the object name is what carries it - and the
    # object name is the contract between the widget and the stylesheet.
    item = AlertFeedItem(fired)
    assert item.trigger_label.objectName() == "AlertTriggerWatch"
    assert "NEW HOD" in [badge.text() for badge in item.findChildren(Badge)]


def test_review_chart_auto_refresh_pulls_new_bars(monkeypatch):
    """The review chart renders when an alert LANDS; the 30s tick (invoked
    directly here) must redraw it with the bars of NOW, so a trader who gets
    to the alert minutes later is not reading a stale M5 pane."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    from datetime import datetime

    import chart_snapshot

    # No daily store: this test is about the M5 pane and the refresh wiring.
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _s: [])

    noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)

    def bar(minute, price):
        return {
            "dt": noon.replace(hour=11, minute=minute),
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000.0,
        }

    class _Bot:
        def __init__(self):
            self.bars = [bar(20, 100.0), bar(25, 101.0)]

        def m5_chart_bars(self, symbol, max_sessions=2):
            return list(self.bars)

    class _Service:
        def __init__(self, bot):
            self._bot = bot

        def current_bot(self):
            return self._bot

    bot = _Bot()
    panel = AlertCenterPanel()
    panel._bounce_service = _Service(bot)
    panel.add_alert(
        BounceAlert(
            time_text="11:30:00",
            symbol="NVDA",
            side="LONG",
            trigger="[S-TIER] VWAP reclaim",
            timeframe="5m",
            raw_text="[S-TIER] NVDA: VWAP reclaim",
        )
    )
    review = panel.chart_review
    # Charts build off the GUI thread, so the bars land on a later turn of
    # the event loop rather than inside add_alert.
    assert _pump_until(lambda: review.snapshot.m5_chart.bar_count() == 2)

    # Two more bars completed while the alert sat unreviewed.
    bot.bars += [bar(30, 102.0), bar(35, 103.0)]
    panel._refresh_review_chart()
    assert _pump_until(lambda: review.snapshot.m5_chart.bar_count() == 4)

    # Unchanged cache: the next tick re-renders nothing.
    renders: list[str] = []
    review.snapshot.snapshotRendered.connect(renders.append)
    panel._refresh_review_chart()
    _pump_until(lambda: False, timeout=0.4)
    assert renders == []

    # No current review alert: the tick is a no-op, never a crash.
    panel._current_review_alert = None
    panel._refresh_review_chart()


def test_review_setup_text_is_large_and_red_only_for_live_alerts(monkeypatch):
    """The setup line uses the big ReviewSetupText style, and the alertLive
    property (red via QSS) is on for queue-fed alerts but off for a typed
    manual chart and for the cleared pane."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)

    panel = AlertCenterPanel()
    text = panel.chart_review.alert_text
    assert text.objectName() == "ReviewSetupText"
    assert not bool(text.property("alertLive"))

    panel.add_alert(
        BounceAlert(
            time_text="11:30:00",
            symbol="NVDA",
            side="LONG",
            trigger="[S-TIER] VWAP reclaim",
            timeframe="5m",
            raw_text="[S-TIER] NVDA: VWAP reclaim",
        )
    )
    assert bool(text.property("alertLive"))
    assert "VWAP reclaim" in text.text()

    # Typing a ticker charts on demand: same pane, muted setup line.
    assert panel.chart_symbol("MSFT")
    assert not bool(text.property("alertLive"))

    panel.chart_review.clear()
    assert not bool(text.property("alertLive"))

    # The style sheet actually carries the big/red rules the property drives.
    from ui import theme

    qss = theme.build_stylesheet()
    assert "QLabel#ReviewSetupText" in qss
    assert 'QLabel#ReviewSetupText[alertLive="true"]' in qss


def test_skip_with_armed_d1_alert_parks_chart_for_the_day(tmp_path, monkeypatch):
    """Arming a D1 alert then hitting Skip = decision made: ordinary alerts
    stop re-occupying the chart that day. The armed watch firing still shows,
    a Focus name still shows, and typing the ticker un-parks it."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert, CHART_WATCH_TAG
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)

    class _FocusService:
        def __init__(self):
            self.symbols = set()

        def is_focus(self, symbol, side=None, category=None):
            return symbol in self.symbols

        def focus_category(self, symbol):
            return "m5" if symbol in self.symbols else None

        def focus_side(self, symbol, category=None):
            return None

        def all_focus(self, category=None):
            return {"long": sorted(self.symbols), "short": []}

    def alert_for(symbol, trigger="[S-TIER] VWAP reclaim", tag="green"):
        return BounceAlert(
            time_text="11:30:00",
            symbol=symbol,
            side="LONG",
            trigger=trigger,
            timeframe="5m",
            tag=tag,
            raw_text=f"{trigger} {symbol}",
        )

    focus = _FocusService()
    panel = AlertCenterPanel(
        parked_symbols_path=tmp_path / "parked.json",
        d1_event_watches_path=tmp_path / "events.json",
    )
    panel.focus_service = focus

    panel.add_alert(alert_for("NVDA"))
    assert panel._current_review_alert.symbol == "NVDA"

    # Arm a D1 alert on the chart, then Skip: the symbol parks for the day.
    panel.arm_d1_event_watch("NVDA", "new_5d_high")
    panel._skip_review_alert(panel._current_review_alert)
    assert "NVDA" in panel._parked_symbols
    assert panel._current_review_alert is None

    # Ordinary alerts for the parked name no longer occupy the chart...
    panel.add_alert(alert_for("NVDA"))
    assert panel._current_review_alert is None
    # ...and the parking survives a panel restart within the day.
    second = AlertCenterPanel(parked_symbols_path=tmp_path / "parked.json")
    assert "NVDA" in second._parked_symbols

    # The armed watch firing is exactly what was asked for: it still shows.
    panel.add_alert(alert_for("NVDA", trigger="New 5-day high", tag=CHART_WATCH_TAG))
    assert panel._current_review_alert is not None
    assert panel._current_review_alert.tag == CHART_WATCH_TAG
    panel._skip_review_alert(panel._current_review_alert)

    # A Focus name is the trader's own: parked or not, it shows.
    focus.symbols.add("NVDA")
    panel.add_alert(alert_for("NVDA"))
    assert panel._current_review_alert is not None
    panel._skip_review_alert(panel._current_review_alert)
    focus.symbols.discard("NVDA")

    # Typing the ticker re-engages: un-parked, alerts occupy the chart again.
    panel.chart_symbol("NVDA")
    assert "NVDA" not in panel._parked_symbols

    # A skip WITHOUT an armed D1 alert stays a plain skip (no parking).
    panel2 = AlertCenterPanel(parked_symbols_path=tmp_path / "parked2.json")
    panel2.add_alert(alert_for("TSLA"))
    panel2._skip_review_alert(panel2._current_review_alert)
    assert "TSLA" not in panel2._parked_symbols
    panel2.add_alert(alert_for("TSLA"))
    assert panel2._current_review_alert.symbol == "TSLA"


def test_dock_d1_event_buttons_arm_poll_and_fire_red(monkeypatch):
    """The dock's D1 row: toggling arms a persistent event watch, the 60s
    poll fires it off the daily-store-derived level, and the one-shot retires
    with a red chart-watch alert leading the feed."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    import dataclasses
    from datetime import datetime, timedelta


    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)

    noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    # Six completed sessions before today: prior 5-session high = 110.
    daily = [
        {
            "dt": (noon - timedelta(days=offset)).replace(hour=0, minute=0),
            "open": 100.0,
            "high": 105.0 + offset if offset <= 5 else 100.0,
            "low": 95.0,
            "close": 100.0,
            "volume": 1000.0,
        }
        for offset in range(6, 0, -1)
    ]

    def bar(minute, high, low):
        mid = (high + low) / 2
        return {
            "dt": noon.replace(hour=11, minute=minute),
            "open": mid,
            "high": high,
            "low": low,
            "close": mid,
            "volume": 1000.0,
        }

    class _Bot:
        def __init__(self):
            self.bars = [bar(20, 108.0, 104.0)]

        def m5_chart_bars(self, symbol, max_sessions=2):
            return list(self.bars)

    class _Service:
        def __init__(self, bot):
            self._bot = bot

        def current_bot(self):
            return self._bot

    bot = _Bot()
    panel = AlertCenterPanel()
    monkeypatch.setattr(panel, "_d1_bars_for", lambda _symbol: list(daily))
    panel._bounce_service = _Service(bot)
    # This test is about the D1 event buttons, not about the movers-only review
    # filter (trader rule 2026-08-19), and NVDA here sits inside yesterday's
    # range by construction. Turning the filter off keeps the test measuring
    # what it was written to measure; the filter has its own tests.
    panel._review_movers_only = False
    panel.add_alert(
        BounceAlert(
            time_text="11:30:00",
            symbol="NVDA",
            side="LONG",
            trigger="[S-TIER] VWAP reclaim",
            timeframe="5m",
            raw_text="[S-TIER] NVDA: VWAP reclaim",
        )
    )

    button = panel.chart_review.arm_bar.d1_event_buttons["new_5d_high"]
    assert button.isEnabled()
    button.click()
    assert [watch.kind for watch in panel._d1_event_watches] == ["new_5d_high"]
    assert button.isChecked() and "✓" in button.text()

    # It is a TOGGLE: a second click disarms.
    button.click()
    assert panel._d1_event_watches == []
    assert not button.isChecked()

    # Re-arm, backdate, and complete a bar breaking the prior 5-day high.
    button.click()
    panel._d1_event_watches[0] = dataclasses.replace(
        panel._d1_event_watches[0], armed_at=noon.replace(hour=11, minute=40)
    )
    bot.bars = bot.bars + [bar(45, 110.6, 108.0)]
    panel._poll_d1_event_watches(now=noon)

    # One-shot: the watch retires and the red alert leads the live feed.
    assert panel._d1_event_watches == []
    fired = panel._alerts[0]
    assert fired.tag == "chart_watch"
    assert fired.symbol == "NVDA"
    assert fired.timeframe == "D1"
    assert "New 5-day high: 110.60 > 110.00" in fired.trigger
    assert not panel.chart_review.arm_bar.d1_event_buttons["new_5d_high"].isChecked()


def test_m5_pick_cross_toggle_adds_swing_focus_and_pins_without_advancing(tmp_path, monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
        from test_qt_focus_panel import _service
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    service = _service(tmp_path)
    panel = AlertCenterPanel(
        service,
        ignored_symbols_path=tmp_path / "alert_center_ignored.txt",
    )
    first = BounceAlert(
        time_text="09:35:00",
        symbol="NVDA",
        side="LONG",
        trigger="[S-TIER] VWAP reclaim",
        timeframe="5m",
        raw_text="[S-TIER] NVDA: VWAP reclaim",
    )
    second = BounceAlert(
        time_text="09:40:00",
        symbol="TSLA",
        side="SHORT",
        trigger="[S-TIER] EMA rejection",
        timeframe="5m",
        raw_text="[S-TIER] TSLA: EMA rejection",
    )
    panel.add_alert(first)
    panel.add_alert(second)
    assert panel._current_review_alert is first
    # M5 pick: the cross-promote files it as a swing name.
    assert panel.chart_review.cross_focus_button.text() == "Add to D1 Focus"

    panel.chart_review.cross_focus_button.click()

    # The pick lands in the Focus Picks store (Swing bucket) AND the feed pin.
    assert service.is_focus("NVDA", "long", "swing")
    assert any(
        alert.symbol == "NVDA" and alert.tag == "d1_focus_pin"
        for alert in panel._d1_alerts
    )
    # A toggle never advances the review chart.
    assert panel._current_review_alert is first
    assert panel.chart_review.cross_focus_button.text() == "✓ In D1 Focus"
    assert panel.chart_review.cross_focus_button.isChecked()

    # Second click removes both.
    panel.chart_review.cross_focus_button.click()
    assert not service.is_focus("NVDA", "long", "swing")
    assert panel._d1_alerts == []
    assert panel._current_review_alert is first
    assert panel.chart_review.cross_focus_button.text() == "Add to D1 Focus"
    assert not panel.chart_review.cross_focus_button.isChecked()


def test_swing_pick_cross_toggle_adds_to_m5_focus(tmp_path, monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
        from test_qt_focus_panel import _service
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    service = _service(tmp_path)
    panel = AlertCenterPanel(
        service,
        ignored_symbols_path=tmp_path / "alert_center_ignored.txt",
    )
    swing = BounceAlert(
        time_text="10:05:00",
        symbol="NVDA",
        side="LONG",
        trigger="[S-TIER] 15EMA break",
        timeframe="D1",
        raw_text="[S-TIER] NVDA: 15EMA break",
        is_d1=True,
    )
    panel.add_alert(swing)
    assert panel._current_review_alert is swing
    # Swing pick: the cross-promote is the M5 day-trade list.
    assert panel.chart_review.cross_focus_button.text() == "Add to M5 Focus"
    assert panel.chart_review.focus_button.text() == "Add to Swing Focus"

    panel.chart_review.cross_focus_button.click()
    assert service.is_focus("NVDA", "long", "m5")
    # A toggle never advances the review chart.
    assert panel._current_review_alert is swing
    assert panel.chart_review.cross_focus_button.text() == "✓ In M5 Focus"

    panel.chart_review.cross_focus_button.click()
    assert not service.is_focus("NVDA", "long", "m5")
    assert panel._current_review_alert is swing
    assert panel.chart_review.cross_focus_button.text() == "Add to M5 Focus"


def test_junk_pseudo_symbols_never_occupy_the_visual_review(monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    panel = AlertCenterPanel()
    # The old AUTO WATCHLIST line parsed "(BULLISH_STRONG)" as its symbol.
    # Even if such a summary reaches the panel (old bot process, future
    # message shapes), a non-ticker must never occupy the review chart.
    junk = BounceAlert.from_callback(
        "AUTO WATCHLIST (BULLISH_STRONG): longs 37 auto (+8/-12), shorts 3 auto "
        "(+0/-3) from 1195 universe names.",
        "blue",
    )
    assert junk.symbol == "(BULLISH_STRONG)"
    panel.add_alert(junk)
    assert panel._current_review_alert is None
    assert panel._review_queue == []

    real = _alert("[S-TIER] NVDA: Bounce confirmed (long)")
    panel.add_alert(real)
    assert panel._current_review_alert is real


def test_armed_watches_survive_gui_restart_within_the_day(tmp_path, monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    watches_path = tmp_path / "alert_chart_watches.json"
    d1_path = tmp_path / "d1_level_watches.json"

    panel = AlertCenterPanel(
        chart_watches_path=watches_path, d1_level_watches_path=d1_path
    )
    assert panel.arm_chart_watch_for("NVDA", "LONG", "new_hod")
    assert panel.arm_chart_watch_for("NVDA", "LONG", "band_bounce")
    assert panel.arm_d1_level_watch("TSLA", "above", 250.0, candle_date="2026-07-20")

    # "Restart": a fresh panel on the same files re-arms everything.
    reborn = AlertCenterPanel(
        chart_watches_path=watches_path, d1_level_watches_path=d1_path
    )
    assert reborn.armed_watch_kinds("NVDA") == {"new_hod", "band_bounce"}
    assert [(w.symbol, w.direction, w.level) for w in reborn._d1_level_watches] == [
        ("TSLA", "above", 250.0)
    ]

    # Disarm persists too.
    reborn.disarm_chart_watch_for("NVDA", "new_hod")
    third = AlertCenterPanel(
        chart_watches_path=watches_path, d1_level_watches_path=d1_path
    )
    assert third.armed_watch_kinds("NVDA") == {"band_bounce"}

    # A new trading day starts clean (the file itself is day-scoped).
    from chart_watch import load_chart_watches
    from datetime import date, timedelta

    assert load_chart_watches(watches_path, market_date=date.today() + timedelta(days=1)) == []
    # ...while the D1 level watch is deliberately NOT day-scoped.
    from chart_watch import load_d1_level_watches

    assert len(load_d1_level_watches(d1_path)) == 1


def test_d1_candle_click_arms_persistent_level_alert_and_flags(tmp_path, monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        # Probe, not a use - see the note on the other guarded import above.
        import chart_snapshot  # noqa: F401
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    from datetime import datetime, timedelta

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    panel = AlertCenterPanel(
        chart_watches_path=tmp_path / "watches.json",
        d1_level_watches_path=tmp_path / "d1.json",
    )

    # The review pane's embedded snapshot: click a candle, choose "break above".
    widget = panel.chart_review.snapshot
    widget._symbol = "NVDA"
    yesterday = datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=1)
    widget.d1_chart.set_data(
        [
            {
                "dt": yesterday,
                "open": 48.0,
                "high": 50.0,
                "low": 47.0,
                "close": 49.0,
                "volume": 1000.0,
            }
        ],
        [],
        timeframe="d1",
    )
    widget.request_d1_level_alert("above", 0)
    assert [(w.symbol, w.direction, w.level) for w in panel._d1_level_watches] == [
        ("NVDA", "above", 50.0)
    ]
    # Re-clicking the same candle level does not double-arm.
    widget.request_d1_level_alert("above", 0)
    assert len(panel._d1_level_watches) == 1

    # Not scanned (no bot): the durable daily store provides the evidence.
    # Backdate the arm so a completed later session can exist.
    import dataclasses

    panel._d1_level_watches[0] = dataclasses.replace(
        panel._d1_level_watches[0], armed_at=datetime.now() - timedelta(days=3)
    )
    breaking_day = datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=1)
    monkeypatch.setattr(
        panel,
        "_d1_bars_for",
        lambda symbol: [
            {
                "dt": breaking_day,
                "open": 49.0,
                "high": 51.2,
                "low": 48.5,
                "close": 50.8,
                "volume": 1000.0,
            }
        ],
    )
    panel._poll_d1_level_watches()

    assert panel._d1_level_watches == []
    fired = panel._alerts[0]
    assert fired.tag == "chart_watch"
    assert fired.symbol == "NVDA"
    assert fired.side == "LONG"
    assert "D1 level break above 50.00" in fired.trigger
    assert fired.payload.get("chart_watch_kind") == "d1_level_above"
    # The retired watch is gone from the persistent store as well.
    from chart_watch import load_d1_level_watches

    assert load_d1_level_watches(tmp_path / "d1.json") == []


def test_auto_watchlist_populate_summary_stays_out_of_alert_center():
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.services.bounce_service import BounceService
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    service = BounceService()
    received = []
    service.alertReceived.connect(received.append)
    callback = service._make_callback()

    # The auto-populate engine's housekeeping summary is silent by design:
    # the longs/shorts.txt adds just happen, with no Alert Center entry.
    callback(
        "AUTO WATCHLIST (bearish_strong anchor, live neutral_chop): longs 4 auto "
        "(+2/-1), shorts 6 auto (+3/-0) from 900 universe names.",
        "blue",
    )
    assert received == []

    # Ordinary alerts still flow.
    callback("[S-TIER] NVDA: Bounce confirmed (long)", "green")
    assert [alert.symbol for alert in received] == ["NVDA"]


def test_snapshot_popup_buttons_route_to_alert_center(monkeypatch):
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotDialog, SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_args, **_kwargs: None)
    panel = AlertCenterPanel()
    dialog = SymbolSnapshotDialog()

    # Without a watch host the popup stays a pure quick look.
    dialog.show_symbol("NVDA", side="LONG")
    assert not dialog.action_row.isVisibleTo(dialog)

    # With the Alert Center as host, the chart-only actions appear.
    dialog.show_symbol("NVDA", side="LONG", watch_host=panel)
    assert dialog.action_row.isVisibleTo(dialog)

    dialog.watch_buttons["new_lod"].click()
    assert [watch.kind for watch in panel._chart_watches] == ["new_lod"]
    assert panel._chart_watches[0].symbol == "NVDA"
    assert panel._chart_watches[0].side == "LONG"
    assert dialog.watch_buttons["new_lod"].isEnabled()
    assert dialog.watch_buttons["new_lod"].isChecked()
    assert "armed" in dialog.watch_buttons["new_lod"].text()
    # Re-arming through the panel API cannot double-arm.
    assert panel.arm_chart_watch_for("NVDA", "LONG", "new_lod") is False
    assert len(panel._chart_watches) == 1

    # The D1 event alerts ride the same action row: arm, reflect, disarm.
    event_button = dialog.d1_event_buttons["new_5d_high"]
    assert event_button.isVisibleTo(dialog)
    event_button.click()
    assert [watch.kind for watch in panel._d1_event_watches] == ["new_5d_high"]
    assert panel._d1_event_watches[0].symbol == "NVDA"
    assert event_button.isChecked() and "✓" in event_button.text()
    event_button.click()
    assert panel._d1_event_watches == []
    assert not event_button.isChecked()
    # A second click on the toggle disarms.
    dialog.watch_buttons["new_lod"].click()
    assert panel._chart_watches == []
    assert not dialog.watch_buttons["new_lod"].isChecked()
    assert dialog.watch_buttons["new_lod"].text() == "New LOD"

    dialog.d1_focus_button.click()
    assert [alert.symbol for alert in panel._d1_alerts] == ["NVDA"]
    assert panel._d1_alerts[0].tag == "d1_focus_pin"
    assert dialog.d1_focus_button.isEnabled()
    assert dialog.d1_focus_button.text() == "✓ In D1 Focus"
    # Second click unpins.
    dialog.d1_focus_button.click()
    assert panel._d1_alerts == []
    assert dialog.d1_focus_button.text() == "Add to D1 Focus"

    # No focus service on this panel: the M5 Focus toggle stays a no-op.
    dialog.m5_focus_button.click()
    assert not dialog.m5_focus_button.isChecked()

    # Reopening re-reads live state from the host.
    dialog.watch_buttons["new_hod"].click()
    dialog.show_symbol("NVDA", side="LONG", watch_host=panel)
    assert "armed" in dialog.watch_buttons["new_hod"].text()

    # A different symbol starts clean.
    dialog.show_symbol("TSLA", side="SHORT", watch_host=panel)
    assert not dialog.watch_buttons["new_hod"].isChecked()
    assert dialog.watch_buttons["new_hod"].text() == "New HOD"
    dialog.close()



def _measured_profiles(symbols):
    """Profiles shaped like a real staging pass: a completed close above
    yesterday's high, above session VWAP, stamped with the current bar."""
    import autopilot_core as core

    bar_end = core.latest_completed_m5_end().isoformat()
    return {
        symbol: {
            "last": 103.0,
            "last_complete": 103.0,
            "completed_session_vwap": 101.0,
            "as_of": bar_end,
        }
        for symbol in symbols
    }


def _measured_context(symbols):
    return {symbol: {"prev_high": 100.5, "prev_low": 98.0} for symbol in symbols}


def test_desk_auto_picks_land_in_m5_focus_for_today(tmp_path, monkeypatch):
    """2026-08-05: staged auto picks are ADOPTED into M5 Focus, not queued for
    one-at-a-time approval - "quicker than adding them in and then seeing
    their alerts". Focus owns the watchlist line, so pruning one removes it."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        import autopilot_core as core
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
        from test_qt_focus_panel import _service
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise
    import json

    monkeypatch.setattr(
        SymbolSnapshotWidget, "set_symbol", lambda self, symbol, **kwargs: None
    )
    pending_path = tmp_path / "auto_populate_pending.json"
    membership_path = tmp_path / "membership.json"
    core.stage_auto_populate_candidates(
        {
            "longs": [{"symbol": "NVDA", "score": 2.1, "reason": "PDH break"}],
            "shorts": [{"symbol": "TSLA", "score": 1.6, "reason": "PDL break"}],
        },
        "neutral_chop",
        # Production always stages from a measured pass; without profiles the
        # picks would carry no measured bar and adoption would (correctly)
        # refuse them, which is a different test than this one.
        profiles=_measured_profiles(("NVDA", "TSLA")),
        daily_context=_measured_context(("NVDA", "TSLA")),
        pending_path=pending_path,
        membership_path=membership_path,
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
    )

    service = _service(tmp_path)
    liked = []
    monkeypatch.setattr(
        service, "record_feedback", lambda *a, **k: liked.append((a, k))
    )
    panel = AlertCenterPanel(
        service,
        ignored_symbols_path=tmp_path / "ignored.txt",
        review_events_path=tmp_path / "review_events.jsonl",
        auto_pick_pending_path=pending_path,
    )
    panel._poll_auto_pick_pending()

    # Both picks are M5 Focus names for today, on the correct sides.
    assert service.is_focus("NVDA", "long", "m5")
    assert service.is_focus("TSLA", "short", "m5")
    assert service.focus_symbols("long", "swing") == []  # day-trade list only
    # ...which means BounceBot's intraday watchlists carry them.
    assert "NVDA" in (tmp_path / "longs.txt").read_text(encoding="utf-8")
    assert "TSLA" in (tmp_path / "shorts.txt").read_text(encoding="utf-8")
    # No approval chart: that is the entire point of the change.
    assert panel._current_review_alert is None
    assert panel._review_queue == []
    # A machine adding names is not the trader liking them.
    assert liked == []
    # The proposal is retired as adopted - never re-proposed today - and
    # auto-populate did NOT also claim the watchlist line it does not own.
    decided = json.loads(pending_path.read_text(encoding="utf-8"))["decided"]
    assert decided["long"]["NVDA"]["decision"] == "auto_focus"
    assert decided["short"]["TSLA"]["decision"] == "auto_focus"
    assert not membership_path.exists() or "NVDA" not in membership_path.read_text(
        encoding="utf-8"
    )

    # A second tick adopts nothing new.
    panel._poll_auto_pick_pending()
    assert service.focus_symbols("long", "m5") == ["NVDA"]

    # Pruning is the trader's half of the deal: it takes the watchlist line
    # with it, so a pruned pick stops alerting instead of just losing its star.
    service.remove("NVDA", "long", "m5")
    assert not service.is_focus("NVDA", "long", "m5")
    assert "NVDA" not in (tmp_path / "longs.txt").read_text(encoding="utf-8")


def test_desk_auto_picks_chart_for_approval_without_a_focus_service(tmp_path, monkeypatch):
    """Fallback path only (no Focus store): staged picks still
    occupy the review chart with Approve/Pass verbs and route through
    resolve_auto_populate_pick, rather than vanishing silently."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        import autopilot_core as core
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    rendered = []
    monkeypatch.setattr(
        SymbolSnapshotWidget,
        "set_symbol",
        lambda self, symbol, **kwargs: rendered.append(symbol),
    )
    pending_path = tmp_path / "auto_populate_pending.json"
    core.stage_auto_populate_candidates(
        {
            "longs": [{"symbol": "NVDA", "score": 2.1, "reason": "PDH break"}],
            "shorts": [{"symbol": "TSLA", "score": 1.6, "reason": "PDL break"}],
        },
        "neutral_chop",
        profiles=_measured_profiles(("NVDA", "TSLA")),
        daily_context=_measured_context(("NVDA", "TSLA")),
        pending_path=pending_path,
        membership_path=tmp_path / "membership.json",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
    )

    resolved = []
    monkeypatch.setattr(
        core,
        "resolve_auto_populate_pick",
        lambda symbol, side, approved, **kwargs: resolved.append(
            (symbol, side, approved)
        )
        or {"written": approved, "already_listed": False, "was_pending": True},
    )

    panel = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.txt",
        review_events_path=tmp_path / "review_events.jsonl",
        auto_pick_pending_path=pending_path,
    )
    panel._poll_auto_pick_pending()

    current = panel._current_review_alert
    assert current is not None and current.symbol == "NVDA"
    assert current.tag == "auto_pick"
    assert current.side == "LONG"
    assert "PDH break" in current.trigger and "score 2.10" in current.trigger
    # Unified verb row (2026-07-31 user rule): the SAME buttons in the SAME
    # spots, with adapted labels - no dedicated auto-pick buttons.
    assert panel.chart_review.focus_button.text() == "✓ Add to watchlist"
    assert panel.chart_review.remove_today_button.text() == "✕ Not today"
    assert [alert.symbol for alert in panel._review_queue] == ["TSLA"]

    # The add slot approves the pick (watchlist, not Focus).
    panel.chart_review.focus_button.click()
    assert resolved == [("NVDA", "long", True)]
    assert panel._current_review_alert.symbol == "TSLA"

    # The not-today slot declines it.
    panel.chart_review.remove_today_button.click()
    assert resolved[-1] == ("TSLA", "short", False)
    assert panel._current_review_alert is None
    # A declined pick's symbol is NOT day-ignored - only the proposal dies.
    assert "TSLA" not in panel._ignored_symbols
    # An ordinary alert gets the ordinary labels back in the same spots.
    from ui.models.bounce import BounceAlert

    panel.add_alert(
        BounceAlert(
            time_text="09:35:00",
            symbol="AMD",
            side="LONG",
            trigger="[S-TIER] VWAP reclaim",
            timeframe="5m",
            raw_text="[S-TIER] AMD: VWAP reclaim",
        )
    )
    assert panel.chart_review.focus_button.text() == "Add to M5 Focus"
    assert panel.chart_review.remove_today_button.text() == "✕ Not today"

    # A poll tick never re-queues decided or already-enqueued picks.
    panel._poll_auto_pick_pending()
    assert [alert.symbol for alert in panel._review_queue] == []


def test_focus_review_button_queues_every_pick(tmp_path, monkeypatch):
    """2026-07-31: the strength board's Review button walks every Focus pick
    through the review chart - swing first, one chart per symbol, muted setup
    text (a review, not a live alert)."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
        from test_qt_focus_panel import _service
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    monkeypatch.setattr(
        SymbolSnapshotWidget, "set_symbol", lambda self, symbol, **kwargs: None
    )
    service = _service(tmp_path)
    service.add("NVDA", "long", "swing", origin="test", context="")
    service.add("TSLA", "short", "m5", origin="test", context="")
    panel = AlertCenterPanel(
        service,
        ignored_symbols_path=tmp_path / "ignored.txt",
        review_events_path=tmp_path / "review_events.jsonl",
    )

    panel.focus_strength.review_button.click()

    current = panel._current_review_alert
    assert current is not None and current.symbol == "NVDA"  # swing first
    assert current.tag == "focus_review"
    assert [alert.symbol for alert in panel._review_queue] == ["TSLA"]
    assert not panel.chart_review.alert_text.property("alertLive")

    # Skip walks to the M5 pick; the queue then runs dry.
    panel._skip_review_alert(current)
    assert panel._current_review_alert.symbol == "TSLA"
    assert panel._current_review_alert.side == "SHORT"

    # The verb row adapts for the walkthrough: keep / skip / delete-pick.
    assert panel.chart_review.focus_button.text() == "★ Keep in Focus"
    assert panel.chart_review.remove_today_button.text() == "✕ Remove from Focus"

    # Remove DELETES the pick from the Focus store and advances...
    panel.chart_review.remove_today_button.click()
    assert not service.is_focus("TSLA")
    assert panel._current_review_alert is None
    # ...but never day-ignores the symbol (alerts still show).
    assert "TSLA" not in panel._ignored_symbols
    # NVDA was only skipped, so it is still a Focus pick.
    assert service.is_focus("NVDA", "long", "swing")


def test_focus_picks_flag_on_any_d1_interest_once_per_day(tmp_path, monkeypatch):
    """2026-07-31: Focus picks are auto-watched for the whole D1 event set;
    each (symbol, event) flags once per session into the D1 Focus feed."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget
        from test_qt_focus_panel import _service
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise
    from datetime import datetime, timedelta

    monkeypatch.setattr(
        SymbolSnapshotWidget, "set_symbol", lambda self, symbol, **kwargs: None
    )
    service = _service(tmp_path)
    service.add("NVDA", "long", "swing", origin="test", context="")
    panel = AlertCenterPanel(
        service,
        ignored_symbols_path=tmp_path / "ignored.txt",
        review_events_path=tmp_path / "review_events.jsonl",
        focus_d1_flags_path=tmp_path / "focus_d1_flags.json",
    )

    moment = datetime(2026, 7, 31, 12, 0)
    day = moment.date()
    d1_bars = [
        {
            "dt": datetime(2026, 7, 20, 0, 0) + timedelta(days=i),
            "open": 100.0,
            "high": 100.0 + i,
            "low": 95.0,
            "close": 99.0 + i,
            "volume": 1_000.0,
        }
        for i in range(8)
    ]
    # Today's completed M5 bar prints above every prior session high.
    m5_bars = [
        {
            "dt": datetime(day.year, day.month, day.day, 9, 30),
            "open": 108.0,
            "high": 120.0,
            "low": 107.5,
            "close": 119.0,
            "volume": 5_000.0,
        }
    ]
    panel._d1_bars_for = lambda symbol: list(d1_bars)
    panel._m5_bars_for = lambda symbol: list(m5_bars)

    panel._poll_focus_d1_interest(now=moment)

    flagged = [alert for alert in panel._d1_alerts if alert.tag == "focus_d1_event"]
    assert flagged, "a new multi-session high must flag the Focus pick"
    assert all(alert.symbol == "NVDA" for alert in flagged)
    assert any("NVDA|new_5d_high" == flag or flag.startswith("NVDA|") for flag in panel._focus_d1_flags)
    assert "NVDA|new_5d_high" in panel._focus_d1_flags

    # Second poll: everything already flagged today - nothing new.
    before = len(panel._d1_alerts)
    panel._poll_focus_d1_interest(now=moment)
    assert len(panel._d1_alerts) == before

    # A fresh panel on the same day reloads the fired registry from disk.
    rebuilt = AlertCenterPanel(
        service,
        ignored_symbols_path=tmp_path / "ignored.txt",
        review_events_path=tmp_path / "review_events.jsonl",
        focus_d1_flags_path=tmp_path / "focus_d1_flags.json",
    )
    assert "NVDA|new_5d_high" in rebuilt._focus_d1_flags


def test_d1_watch_read_is_memory_only_and_prefetches_off_thread(monkeypatch):
    from datetime import datetime

    try:
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.services import chart_data_service
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    bars = [{"dt": datetime.now(), "close": 101.0}]

    class _Series:
        def as_bar_dicts(self):
            return list(bars)

    class _Service:
        def __init__(self):
            self.requests = []

        def cached_series(self, symbol):
            return _Series()

        def cached_bar_dicts(self, symbol):
            # The service memoizes the materialization now (2026-08-21): the
            # panel polls this for every armed and every Focus symbol on a
            # 60-second timer, and as_bar_dicts is documented worker-only.
            return _Series().as_bar_dicts()

        def prefetch(self, symbols):
            self.requests.append(list(symbols))

    service = _Service()
    monkeypatch.setattr(chart_data_service, "shared_service", lambda: service)
    panel = AlertCenterPanel()

    assert panel._d1_bars_for(" nvda ") == bars
    # Snappiness packet 2, item 1c: the prefetch is QUEUED, then issued as ONE
    # batch on the next event-loop turn, so ~105 single-element tasks no longer
    # queue ahead of the snapshot for the chart the trader just clicked.
    assert service.requests == []
    panel._flush_d1_prefetch()
    assert service.requests == [["NVDA"]]
