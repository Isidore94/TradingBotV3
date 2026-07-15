import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _alert(text, tag="green"):
    from ui.models.bounce import BounceAlert

    return BounceAlert.from_callback(text, tag)


def test_tier_extraction_and_banger_detection():
    try:
        from ui.panels.alert_center_panel import extract_alert_tier, is_banger_alert
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    assert extract_alert_tier(_alert("[S-TIER] AAOI: Bounce confirmed (short)")) == "S"
    assert extract_alert_tier(_alert("[b-tier] X: Bounce confirmed")) == "B"
    assert extract_alert_tier(_alert("MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA")) == ""
    assert is_banger_alert(_alert("[B-TIER] RW BANGER AAOI (short): SPY paused"))
    assert not is_banger_alert(_alert("[S-TIER] AAOI: Bounce confirmed"))


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
    banger = _alert("[C-TIER] RW BANGER CCC (short): SPY paused")
    untiered = _alert("MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA (long) Favorite setup upgrade", "d1_flag_long")

    assert all(alert_passes_min_tier(a, "all") for a in (s_alert, b_alert, d_alert, banger, untiered))
    assert alert_passes_min_tier(s_alert, "A")
    assert not alert_passes_min_tier(b_alert, "A")
    assert not alert_passes_min_tier(d_alert, "B")
    # Bangers always pass; untiered info passes everything except S-only.
    assert alert_passes_min_tier(banger, "S")
    assert alert_passes_min_tier(untiered, "A")
    assert not alert_passes_min_tier(untiered, "S")


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


def test_loud_alerts_are_sa_bangers_or_ready_d1():
    try:
        from ui.panels.alert_center_panel import alert_is_loud
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    assert alert_is_loud(_alert("[S-TIER] AAA: Bounce confirmed"))
    assert alert_is_loud(_alert("[A-TIER] AAA: Bounce confirmed"))
    assert alert_is_loud(_alert("[D-TIER] RS BANGER MSTR (long)"))
    assert not alert_is_loud(_alert("[B-TIER] AAA: Bounce confirmed"))
    # A level-cross trigger is developing evidence; only the final bucket
    # upgrade is a D1 Focus/loud moment.
    assert not alert_is_loud(_alert("MASTER_AVWAP_D1_UPGRADE_TRIGGER: AAPL (long) 1st-dev break UPPER_1@314.57", "d1_flag_long"))
    assert alert_is_loud(_alert("MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA (long) Favorite setup upgrade", "d1_flag_long"))
    assert not alert_is_loud(_alert("MASTER_AVWAP_D1_UPGRADE_WATCH: AAPL (long) AVWAPE retest", "d1_flag_long"))
    # The pause-watch summary line stays quiet by design.
    assert not alert_is_loud(_alert("REGIME PAUSE WATCH (short): SPY paused (+0.15% window) - 3 swing shorts still pressing lows: A, B, C", "red"))


def test_d1_focus_feed_only_gets_favorite_bucket_transitions():
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    panel = AlertCenterPanel()
    upgrade = _alert("MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA (long) Favorite setup upgrade [score=245]", "d1_flag_long")
    trigger = _alert("MASTER_AVWAP_D1_UPGRADE_TRIGGER: AAPL (long) A/S upgrade: 1st-dev break UPPER_1@314.57", "d1_flag_long")
    watch = _alert("MASTER_AVWAP_D1_UPGRADE_WATCH: AAPL (long) AVWAPE retest AVWAPE@309.38", "d1_flag_long")
    generic = _alert("MASTER_AVWAP_D1_FLAG: MSFT (short) 15EMA break [score=88]", "d1_flag_short")
    for alert in (upgrade, trigger, watch, generic):
        panel.add_alert(alert)

    # Only the become-a-favorite moments live in the D1 Focus feed...
    d1_texts = [a.raw_text for a in panel._d1_alerts]
    assert [t.split(":", 1)[0] for t in d1_texts] == ["MASTER_AVWAP_D1_BUCKET_UPGRADE"]
    # ...while WATCH/context flags stay visible in the live stream.
    live_prefixes = {a.raw_text.split(":", 1)[0] for a in panel._alerts}
    assert live_prefixes == {
        "MASTER_AVWAP_D1_UPGRADE_TRIGGER",
        "MASTER_AVWAP_D1_UPGRADE_WATCH",
        "MASTER_AVWAP_D1_FLAG",
    }


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
    assert alert_passes_feed_gate(quiet_b, "A", is_focus=True)
    assert alert_passes_feed_gate(quiet_b, "S", is_focus=True)
    assert alert_should_sound(quiet_b, is_focus=True)
