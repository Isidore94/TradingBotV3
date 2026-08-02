"""Alert popup payload round-trip: main-side capture → JSON → satellite render input.

Uses the real chart_snapshot builders on synthetic bars so the payload
exercises the same snapshot shape the popup widgets consume, then proves
the JSON wire round-trip preserves it (datetimes, timezones, preview
flags, overlay alignment) through the actual desk_link envelope.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import chart_snapshot
from desk_link import protocol
from desk_link.popup_payload import (
    PAYLOAD_SCHEMA,
    build_alert_popup_payload,
    restore_alert_popup_payload,
)
from ui.models.bounce import BounceAlert

EASTERN = ZoneInfo("America/New_York")


def _m5_bars(count: int = 30) -> list[dict]:
    start = datetime(2026, 7, 30, 9, 30, tzinfo=EASTERN)
    bars = []
    price = 100.0
    for index in range(count):
        price += 0.25 if index % 3 else -0.15
        bars.append(
            {
                "dt": start + timedelta(minutes=5 * index),
                "open": price,
                "high": price + 0.4,
                "low": price - 0.4,
                "close": price + 0.1,
                "volume": 10_000 + 37 * index,
            }
        )
    return bars


def _payload():
    bars = _m5_bars()
    m5 = chart_snapshot.build_m5_snapshot("NVDA", bars)
    alert = BounceAlert(
        time_text="06:45:00",
        symbol="NVDA",
        side="LONG",
        trigger="VWAP bounce",
        timeframe="M5",
        tag="chart_watch",
        payload={"chart_watch_kind": "vwap_bounce"},
    )
    return build_alert_popup_payload(
        alert,
        d1_snapshot={"symbol": "NVDA", "timeframe": "D1", "bars": bars[-5:], "overlays": [], "note": ""},
        m5_snapshot=m5,
        armed_kinds=["vwap_bounce"],
        armed_levels=[{"price": 101.5, "label": "prior high"}],
        guidance_text="Focus name — reviewed 2x this week.",
    )


def test_payload_survives_json_and_desk_link_envelope():
    payload = _payload()
    # The full trip a popup actually takes: payload -> envelope -> wire -> back.
    wire = protocol.encode_message(protocol.make_message(protocol.TYPE_ALERT_POPUP, payload))
    received = protocol.decode_message(wire.rstrip(b"\n"))
    restored = restore_alert_popup_payload(received["payload"])

    assert restored["alert"]["symbol"] == "NVDA"
    assert restored["alert"]["payload"]["chart_watch_kind"] == "vwap_bounce"
    assert restored["armed"]["kinds"] == ["vwap_bounce"]
    assert restored["armed"]["levels"] == [{"price": 101.5, "label": "prior high"}]
    assert restored["guidance_text"].startswith("Focus name")


def test_bar_datetimes_round_trip_with_timezone():
    restored = restore_alert_popup_payload(json.loads(json.dumps(_payload())))
    original = _m5_bars()
    for source, back in zip(original, restored["m5"]["bars"], strict=True):
        assert isinstance(back["dt"], datetime)
        assert back["dt"] == source["dt"]
        assert back["dt"].utcoffset() == source["dt"].utcoffset()


def test_overlays_stay_aligned_with_bars():
    restored = restore_alert_popup_payload(json.loads(json.dumps(_payload())))
    m5 = restored["m5"]
    assert m5["overlays"], "expected VWAP/EMA overlays from build_m5_snapshot"
    for overlay in m5["overlays"]:
        assert len(overlay["values"]) == len(m5["bars"])


def test_unknown_schema_is_a_visible_error():
    payload = _payload()
    payload["schema"] = "desk_link.alert_popup.v999"
    with pytest.raises(ValueError):
        restore_alert_popup_payload(payload)
    assert PAYLOAD_SCHEMA.endswith(".v1")


def test_plain_dict_alert_is_accepted():
    payload = build_alert_popup_payload(
        {"symbol": "AMD", "side": "SHORT"},
        d1_snapshot={"symbol": "AMD", "timeframe": "D1", "bars": [], "overlays": [], "note": "no store"},
        m5_snapshot={"symbol": "AMD", "timeframe": "M5", "bars": [], "overlays": [], "note": "no cached M5 bars"},
    )
    restored = restore_alert_popup_payload(json.loads(json.dumps(payload)))
    assert restored["alert"]["symbol"] == "AMD"
    assert restored["d1"]["note"] == "no store"
