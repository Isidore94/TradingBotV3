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


def test_bucket_upgrade_alert_is_the_only_actionable_d1_focus_message():
    from ui.models.bounce import BounceAlert
    try:
        from ui.panels.alert_center_panel import _is_actionable_d1_alert
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    bucket_alert = BounceAlert.from_callback(
        "MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA (long) Favorite setup upgrade [score=245]",
        "d1_flag_long",
        timestamp=datetime(2026, 1, 2, 10, 1, 0),
    )
    generic_alert = BounceAlert.from_callback(
        "MASTER_AVWAP_D1_FLAG: MSFT (short) 15EMA break [score=88]",
        "d1_flag_short",
        timestamp=datetime(2026, 1, 2, 10, 2, 0),
    )

    assert bucket_alert.symbol == "NVDA"
    assert bucket_alert.side == "LONG"
    assert _is_actionable_d1_alert(bucket_alert)
    assert not _is_actionable_d1_alert(generic_alert)
