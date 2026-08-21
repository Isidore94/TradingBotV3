import os
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_bounce_alert_parses_structured_payload():
    from ui.models.bounce import BounceAlert

    alert = BounceAlert.from_callback(
        {
            "kind": "bounce_alert",
            "text": "AAPL: Bounce confirmed (long) from ema_15, vwap",
            "feedback": {
                "symbol": "AAPL",
                "direction": "long",
                "bounce_types": "ema_15;vwap",
            },
        },
        "green",
        timestamp=datetime(2026, 1, 2, 9, 35, 0),
    )

    assert alert.time_text == "09:35:00"
    assert alert.symbol == "AAPL"
    assert alert.side == "LONG"
    assert alert.trigger == "ema_15, vwap"
    assert not alert.is_d1


def test_bounce_alert_marks_d1_flags():
    from ui.models.bounce import BounceAlert

    alert = BounceAlert.from_callback(
        "MASTER_AVWAP_D1_FLAG: MSFT (short) 15EMA break [score=88]",
        "d1_flag_short",
        timestamp=datetime(2026, 1, 2, 10, 0, 0),
    )

    assert alert.symbol == "MSFT"
    assert alert.side == "SHORT"
    assert alert.is_d1
    assert alert.context == "score=88"


def test_entry_assist_lists_expand_to_one_m5_alert_per_symbol():
    from ui.models.bounce import BounceAlert

    alerts = BounceAlert.from_callback_many(
        "STRONGEST 30M (long): NVDA +1.20%, AMD +0.90% [manual]",
        "green",
        timestamp=datetime(2026, 1, 2, 10, 0, 0),
    )

    assert [alert.symbol for alert in alerts] == ["NVDA", "AMD"]
    assert [alert.side for alert in alerts] == ["LONG", "LONG"]
    assert [alert.timeframe for alert in alerts] == ["M5", "M5"]
    assert [alert.trigger for alert in alerts] == [
        "M5 move +1.20%",
        "M5 move +0.90%",
    ]
    assert [alert.payload["list_rank"] for alert in alerts] == [1, 2]


def test_entry_window_list_expands_with_spy_excess():
    from ui.models.bounce import BounceAlert

    alerts = BounceAlert.from_callback_many(
        "ENTRY WINDOW (short): SPY +0.60% since 10:05 - stayed weakest through it: "
        "TSLA -0.40% (x+1.00), META -0.10% (x+0.70) [auto]",
        "red",
        timestamp=datetime(2026, 1, 2, 10, 30, 0),
    )

    assert [alert.symbol for alert in alerts] == ["TSLA", "META"]
    assert all(alert.side == "SHORT" for alert in alerts)
    assert alerts[0].trigger == "M5 move -0.40% · vs SPY x+1.00"


def test_regime_pause_list_expands_without_a_bogus_regime_symbol():
    from ui.models.bounce import BounceAlert

    alerts = BounceAlert.from_callback_many(
        "REGIME PAUSE WATCH (short): SPY paused (+0.15% window) - "
        "3 swing shorts still pressing lows: AAOI, TSLA, META (3 today). "
        "Recorded as swing-scan evidence, not an entry signal.",
        "red",
    )

    assert [alert.symbol for alert in alerts] == ["AAOI", "TSLA", "META"]
    assert all(alert.side == "SHORT" and alert.timeframe == "M5" for alert in alerts)
    assert alerts[0].trigger == "M5 regime-pause watch · pressing lows"


def test_non_list_callback_stays_one_alert():
    from ui.models.bounce import BounceAlert

    alerts = BounceAlert.from_callback_many(
        "[A-TIER] NVDA: Bounce confirmed (long) from ema_15",
        "green",
    )

    assert len(alerts) == 1
    assert alerts[0].symbol == "NVDA"


def test_ready_d1_alerts_are_final_bucket_upgrades_only():
    from ui.models.bounce import BounceAlert
    try:
        from ui.panels.alert_center_panel import is_ready_d1_alert
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    bucket_alert = BounceAlert.from_callback(
        "MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA (long) Favorite setup upgrade [score=245]",
        "d1_flag_long",
        timestamp=datetime(2026, 1, 2, 10, 1, 0),
    )
    trigger_alert = BounceAlert.from_callback(
        "MASTER_AVWAP_D1_UPGRADE_TRIGGER: AAPL (long) A/S upgrade: 1st-dev break UPPER_1@314.57 [price=314.80]",
        "d1_flag_long",
        timestamp=datetime(2026, 1, 2, 10, 1, 30),
    )
    watch_alert = BounceAlert.from_callback(
        "MASTER_AVWAP_D1_UPGRADE_WATCH: AAPL (long) AVWAPE retest AVWAPE@309.38",
        "d1_flag_long",
        timestamp=datetime(2026, 1, 2, 10, 1, 45),
    )
    generic_alert = BounceAlert.from_callback(
        "MASTER_AVWAP_D1_FLAG: MSFT (short) 15EMA break [score=88]",
        "d1_flag_short",
        timestamp=datetime(2026, 1, 2, 10, 2, 0),
    )

    assert bucket_alert.symbol == "NVDA"
    assert bucket_alert.side == "LONG"
    assert is_ready_d1_alert(bucket_alert)
    assert not is_ready_d1_alert(trigger_alert)
    assert not is_ready_d1_alert(watch_alert)
    assert not is_ready_d1_alert(generic_alert)


def test_a_regime_pause_row_carries_its_own_measure():
    """The line stopped being a batch label (trader, 2026-08-21).

    Every symbol used to inherit the same phrase - "holding highs" - whatever
    its own distance from the high was. Each row now says where IT is.
    """
    from ui.models.bounce import BounceAlert

    alerts = BounceAlert.from_callback_many(
        "REGIME PAUSE WATCH (long): SPY paused (-0.24% window) - "
        "3 swing longs still holding highs: HTFL (new HOD), MRK (0.7 ATR), "
        "TEM (0.2 ATR) (3 today). Recorded as swing-scan evidence, not an "
        "entry signal.",
        "green",
    )

    assert [alert.symbol for alert in alerts] == ["HTFL", "MRK", "TEM"]
    assert alerts[0].trigger == "M5 regime-pause watch · new HOD"
    assert alerts[1].trigger == "M5 regime-pause watch · 0.7 ATR off HOD"
    assert alerts[2].trigger == "M5 regime-pause watch · 0.2 ATR off HOD"


def test_a_short_regime_pause_row_measures_against_the_low():
    from ui.models.bounce import BounceAlert

    alerts = BounceAlert.from_callback_many(
        "REGIME PAUSE WATCH (short): SPY paused (+0.24% window) - "
        "2 swing shorts still pressing lows: GFS (new LOD), WDC (0.9 ATR) "
        "(2 today). Recorded as swing-scan evidence, not an entry signal.",
        "red",
    )

    assert [alert.symbol for alert in alerts] == ["GFS", "WDC"]
    assert alerts[0].trigger == "M5 regime-pause watch · new LOD"
    assert alerts[1].trigger == "M5 regime-pause watch · 0.9 ATR off LOD"


def test_a_measureless_row_still_reads_as_the_batch_phrase():
    """Lines written before per-symbol measures existed, and any symbol whose
    hold could not be measured, keep the old wording rather than acquiring a
    claim this parser invented."""
    from ui.models.bounce import BounceAlert

    alerts = BounceAlert.from_callback_many(
        "REGIME PAUSE WATCH (long): SPY paused (-0.15% window) - "
        "2 swing longs still holding highs: AAOI, TSLA (2 today). "
        "Recorded as swing-scan evidence, not an entry signal.",
        "green",
    )
    assert [alert.trigger for alert in alerts] == [
        "M5 regime-pause watch · holding highs",
        "M5 regime-pause watch · holding highs",
    ]


def test_every_regime_pause_row_is_recognised_as_one():
    """The expiry sweep finds these rows by their trigger prefix, so the two
    have to stay in step whatever the measure says."""
    from ui.models.bounce import BounceAlert, is_regime_pause_alert

    alerts = BounceAlert.from_callback_many(
        "REGIME PAUSE WATCH (long): SPY paused (-0.24% window) - "
        "2 swing longs still holding highs: HTFL (new HOD), MRK (0.7 ATR) "
        "(2 today). Recorded as swing-scan evidence, not an entry signal.",
        "green",
    )
    assert alerts and all(is_regime_pause_alert(alert) for alert in alerts)
