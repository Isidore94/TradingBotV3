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


def test_loud_alerts_are_sa_or_bangers():
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
