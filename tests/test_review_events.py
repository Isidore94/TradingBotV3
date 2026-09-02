"""Review-decision capture: the writer module + the Alert Center's hooks."""

import json
import sys

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def _queue_mechanics_only(monkeypatch):
    """Routing off: these tests are about what the QUEUE does with a row.

    Since 2026-08-27 an ordinary intraday alert lists in the M5 alert bar
    instead of queueing a chart (trader rule; `test_qt_m5_alert_bar.py` owns
    that routing and its exemptions). The mechanics below are the same for
    any row the queue holds, so they are exercised with the routing switched
    off rather than rewritten around D1 fixtures.
    """
    from ui.panels.alert_center_panel import AlertCenterPanel

    monkeypatch.setattr(
        AlertCenterPanel, "_is_m5_review_alert", staticmethod(lambda alert: False)
    )

from review_events import (
    REVIEW_EVENTS_SCHEMA,
    alert_context_fields,
    get_review_installation_id,
    load_review_events,
    record_review_event,
    review_event_shard_path,
    setup_context_fields,
)


def _fake_alert(**overrides):
    context = {
        "rrs_spy": 1.42,
        "rrs_sector": 0.8,
        "rrs_industry": -0.3,
        "session_rvol": 2.1,
        "market_environment": "BULLISH_WEAK",
        "internals_tape": "risk_on",
        "sector": "Technology",
        "industry": "Semiconductors",
    }
    fields = dict(
        symbol="NVDA",
        side="LONG",
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text="[A-TIER] PROVEN NVDA: Bounce confirmed (long) from dynamic_vwap_upper_band",
        is_d1=False,
        payload={
            "feedback": {
                "event_id": "evt-123",
                "bounce_types": "dynamic_vwap_upper_band",
                "entry_price": "181.55",
                "stop_price": "180.10",
                "risk_per_share": "1.45",
                "score": "42",
                "is_focus_pick": False,
                "context_json": json.dumps(context),
            }
        },
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


# ---------------------------------------------------------------------------
# Writer module
# ---------------------------------------------------------------------------
def test_the_banger_column_is_retired_but_still_written(tmp_path):
    """BANGER retired 2026-09-01 (trader: "We can probably remove this because
    idk what it is"). The column survives so every reader of the 8,818
    historical rows keeps working and the row shape does not move - but it is
    now the constant False, even when the alert text carries the word.

    Fail-before-fix: on the un-fixed code this row read True.
    """
    path = tmp_path / "events.jsonl"
    row = record_review_event(
        "skip",
        alert=_fake_alert(raw_text="[C-TIER] RW BANGER AAOI (short): SPY paused"),
        dwell_ms=1,
        queue_len=0,
        now=datetime(2026, 9, 1, 10, 15),
        path=path,
    )
    assert row is not None
    assert "banger" in row
    assert row["banger"] is False


def test_record_review_event_snapshots_structured_alert_context(tmp_path):
    path = tmp_path / "events.jsonl"
    row = record_review_event(
        "skip",
        alert=_fake_alert(),
        dwell_ms=4200,
        queue_len=3,
        now=datetime(2026, 7, 28, 10, 15),
        path=path,
    )
    assert row is not None
    assert row["schema"] == REVIEW_EVENTS_SCHEMA
    assert row["action"] == "skip"
    assert row["symbol"] == "NVDA"
    assert row["side"] == "LONG"
    # The decision-relevant numbers land as real fields, not a text blob.
    assert row["tier"] == "A"
    assert row["proven"] is True
    # RETIRED 2026-09-01: the column stays for the historical rows, always False.
    assert row["banger"] is False
    assert row["event_id"] == "evt-123"
    assert row["bounce_types"] == "dynamic_vwap_upper_band"
    assert row["entry_price"] == 181.55
    assert row["stop_price"] == 180.10
    assert row["rrs_spy"] == 1.42
    assert row["session_rvol"] == 2.1
    assert row["market_environment"] == "BULLISH_WEAK"
    assert row["dwell_ms"] == 4200
    assert row["queue_len"] == 3
    # And it round-trips off disk.
    assert load_review_events(path) == [row]


def test_record_review_event_requires_action_and_symbol(tmp_path):
    path = tmp_path / "events.jsonl"
    assert record_review_event("", alert=_fake_alert(), path=path) is None
    assert record_review_event("skip", symbol="", path=path) is None
    assert not path.exists()


def test_record_review_event_survives_malformed_alerts(tmp_path):
    path = tmp_path / "events.jsonl"
    # No payload, no raw_text - a manual chart or a stand-in object.
    bare = SimpleNamespace(symbol="AAPL", side="WATCH")
    row = record_review_event("shown", alert=bare, path=path)
    assert row is not None
    assert row["tier"] == ""
    assert row["event_id"] == ""
    # context_json that fails to parse is dropped, not fatal.
    broken = _fake_alert()
    broken.payload["feedback"]["context_json"] = "{not json"
    row = record_review_event("skip", alert=broken, path=path)
    assert row is not None
    assert "rrs_spy" not in row


def test_alert_context_fields_reads_chart_watch_payload():
    alert = SimpleNamespace(
        symbol="TSLA",
        side="SHORT",
        raw_text="CHART WATCH TSLA (SHORT): -1σ bounce",
        tag="chart_watch",
        timeframe="M5",
        is_d1=False,
        trigger="-1σ bounce",
        payload={"chart_watch_kind": "band_bounce", "armed_at": "2026-07-28T09:40:00"},
    )
    fields = alert_context_fields(alert)
    assert fields["chart_watch_kind"] == "band_bounce"
    assert fields["tag"] == "chart_watch"


def _setup_row(**overrides):
    fields = dict(
        symbol="LNG",
        side="LONG",
        score=245.0,
        bucket="favorite_setup",
        setup_tags=["AVWAP_BREAKOUT", "D1_RS"],
        expected_r=0.85,
        days_to_earnings=21,
        sector="Energy",
        industry="Oil & Gas Midstream",
        d1_vs_sector=1.8,
        d1_vs_industry=2.4,
        raw={"setup_family": "avwap_breakout"},
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_setup_context_fields_snapshots_the_swing_row(tmp_path):
    fields = setup_context_fields(_setup_row())
    assert fields["surface"] == "setups"
    assert fields["is_d1"] is True and fields["timeframe"] == "D1"
    assert fields["bucket"] == "favorite_setup"
    assert fields["setup_family"] == "avwap_breakout"
    assert fields["setup_tags"] == "AVWAP_BREAKOUT;D1_RS"
    assert fields["expected_r"] == 0.85
    assert fields["d1_vs_industry"] == 2.4

    path = tmp_path / "events.jsonl"
    row = record_review_event(
        "dislike",
        symbol="LNG",
        side="LONG",
        detail={"reason": "too extended", "origin": "setups"},
        context_fields=fields,
        path=path,
    )
    assert row["surface"] == "setups"
    assert row["bucket"] == "favorite_setup"
    assert row["setup_family"] == "avwap_breakout"
    assert row["detail"]["reason"] == "too extended"
    # A bare row still records the surface markers.
    assert setup_context_fields(None) == {
        "surface": "setups",
        "is_d1": True,
        "timeframe": "D1",
    }


def test_load_review_events_skips_bad_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    good = record_review_event("skip", alert=_fake_alert(), path=path)
    path.open("a", encoding="utf-8").write("{broken\n[1,2]\n")
    assert load_review_events(path) == [good]


def test_installation_identity_is_machine_local_stable_and_not_a_hostname(tmp_path):
    identity_path = tmp_path / "local" / "review_installation_id"
    first = get_review_installation_id(identity_path)
    second = get_review_installation_id(identity_path)

    assert len(first) == 32
    assert first == second
    assert identity_path.read_text(encoding="ascii").strip() == first
    assert review_event_shard_path(first, shards_dir=tmp_path / "shared").name == (
        f"review-events-{first}.jsonl"
    )


def test_partitioned_writers_never_modify_the_legacy_shared_file(tmp_path):
    legacy = tmp_path / "alert_review_events.jsonl"
    legacy_row = {
        "schema": "review_events_v1",
        "ts": "2026-07-28T10:00:00",
        "trade_date": "2026-07-28",
        "machine": "MainPC",
        "action": "shown",
        "symbol": "CLMT",
        "side": "LONG",
    }
    legacy_bytes = (json.dumps(legacy_row) + "\n").encode()
    legacy.write_bytes(legacy_bytes)
    shards = tmp_path / "alert_review_events"
    desk_identity = "1" * 32
    mini_identity = "2" * 32
    desk_id_file = tmp_path / "desk-local-id"
    mini_id_file = tmp_path / "mini-local-id"
    desk_id_file.write_text(desk_identity, encoding="ascii")
    mini_id_file.write_text(mini_identity, encoding="ascii")

    desk = record_review_event(
        "skip",
        alert=_fake_alert(symbol="NVDA"),
        now=datetime(2026, 7, 29, 10, 0),
        path=legacy,
        shards_dir=shards,
        installation_id_path=desk_id_file,
        partitioned=True,
    )
    mini = record_review_event(
        "shown",
        alert=_fake_alert(symbol="AMD"),
        now=datetime(2026, 7, 29, 10, 1),
        path=legacy,
        shards_dir=shards,
        installation_id_path=mini_id_file,
        partitioned=True,
    )

    assert legacy.read_bytes() == legacy_bytes
    assert desk["installation_id"] == desk_identity
    assert mini["installation_id"] == mini_identity
    assert review_event_shard_path(desk_identity, shards_dir=shards).exists()
    assert review_event_shard_path(mini_identity, shards_dir=shards).exists()
    rows = load_review_events(legacy, shards_dir=shards, include_shards=True)
    assert [row["symbol"] for row in rows] == ["CLMT", "NVDA", "AMD"]


def test_hostname_change_keeps_writing_the_same_installation_shard(tmp_path, monkeypatch):
    import review_events

    legacy = tmp_path / "alert_review_events.jsonl"
    shards = tmp_path / "alert_review_events"
    identity = "a" * 32
    identity_path = tmp_path / "local-id"
    identity_path.write_text(identity, encoding="ascii")
    names = iter(["OLD-NAME", "NEW-NAME"])
    monkeypatch.setattr(review_events, "_machine_name", lambda: next(names))

    for symbol in ("NVDA", "AMD"):
        assert record_review_event(
            "shown",
            symbol=symbol,
            path=legacy,
            shards_dir=shards,
            installation_id_path=identity_path,
            partitioned=True,
        )

    shard = review_event_shard_path(identity, shards_dir=shards)
    assert [row["machine"] for row in load_review_events(shard)] == [
        "OLD-NAME",
        "NEW-NAME",
    ]
    assert list(shards.glob("*.jsonl")) == [shard]


def test_malformed_existing_installation_identity_fails_closed(tmp_path):
    legacy = tmp_path / "alert_review_events.jsonl"
    identity_path = tmp_path / "local-id"
    identity_path.write_text("half-synced-or-corrupt", encoding="ascii")

    row = record_review_event(
        "shown",
        symbol="NVDA",
        path=legacy,
        shards_dir=tmp_path / "shards",
        installation_id_path=identity_path,
        partitioned=True,
    )

    assert row is None
    assert not legacy.exists()
    assert not (tmp_path / "shards").exists()


# ---------------------------------------------------------------------------
# Alert Center hooks (offscreen Qt; skipped when PySide6 is unavailable)
# ---------------------------------------------------------------------------
def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return None
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _bounce_alert(**overrides):
    from ui.models.bounce import BounceAlert

    fake = _fake_alert(**overrides)
    return BounceAlert(
        time_text="10:15:00",
        symbol=fake.symbol,
        side=fake.side,
        trigger=fake.trigger,
        timeframe=fake.timeframe,
        tag=fake.tag,
        raw_text=fake.raw_text,
        is_d1=fake.is_d1,
        payload=fake.payload,
    )


def _actions(path):
    return [row["action"] for row in load_review_events(path)]


def test_panel_logs_shown_and_skip_with_dwell_and_queue(tmp_path):
    if _qt_app() is None:
        return
    from ui.panels.alert_center_panel import AlertCenterPanel

    path = tmp_path / "events.jsonl"
    panel = AlertCenterPanel(review_events_path=path)
    first = _bounce_alert()
    second = _bounce_alert(symbol="AMD", raw_text="[B-TIER] AMD: Bounce confirmed (long)")
    panel._enqueue_review_alert(first)
    panel._enqueue_review_alert(second)

    rows = load_review_events(path)
    assert [row["action"] for row in rows] == ["shown"]
    assert rows[0]["symbol"] == "NVDA"
    assert rows[0]["tier"] == "A"
    assert rows[0]["queue_len"] == 0  # AMD enqueued after NVDA was shown

    panel._skip_review_alert(first)
    rows = load_review_events(path)
    # Skipping NVDA logs the skip and advances to AMD (a second impression).
    assert [row["action"] for row in rows] == ["shown", "skip", "shown"]
    skip = rows[1]
    assert skip["symbol"] == "NVDA"
    assert skip["dwell_ms"] >= 0
    assert skip["queue_len"] == 1
    assert rows[2]["symbol"] == "AMD"


def test_panel_logs_remove_today_and_restore(tmp_path):
    if _qt_app() is None:
        return
    from ui.panels.alert_center_panel import AlertCenterPanel

    path = tmp_path / "events.jsonl"
    panel = AlertCenterPanel(review_events_path=path)
    alert = _bounce_alert()
    panel._enqueue_review_alert(alert)
    panel._remove_review_alert_for_today(alert)
    panel._restore_ignored_symbol("NVDA")
    assert _actions(path) == ["shown", "remove_today", "restore_today"]


def test_panel_logs_watch_arm_disarm_and_expiry(tmp_path):
    if _qt_app() is None:
        return
    from chart_watch import ChartWatch
    from ui.panels.alert_center_panel import AlertCenterPanel

    path = tmp_path / "events.jsonl"
    panel = AlertCenterPanel(review_events_path=path)
    assert panel.arm_chart_watch_for("NVDA", "LONG", "band_bounce")
    assert panel.disarm_chart_watch_for("NVDA", "band_bounce")

    # A watch armed yesterday is stale: the poll must log its expiry so the
    # log can tell fired / disarmed / expired apart.
    panel._chart_watches.append(
        ChartWatch(
            kind="new_hod",
            symbol="AMD",
            side="LONG",
            armed_at=datetime.now() - timedelta(days=1),
        )
    )
    panel._poll_chart_watches()

    rows = load_review_events(path)
    assert [row["action"] for row in rows] == ["arm_watch", "disarm_watch", "watch_expired"]
    assert rows[0]["detail"]["kind"] == "band_bounce"
    assert rows[2]["symbol"] == "AMD"
    assert rows[2]["detail"]["kind"] == "new_hod"


def test_panel_logs_level_arm_with_fill_source(tmp_path):
    if _qt_app() is None:
        return
    from ui.panels.alert_center_panel import AlertCenterPanel

    path = tmp_path / "events.jsonl"
    panel = AlertCenterPanel(review_events_path=path)
    assert panel.arm_d1_level_watch(
        "NVDA", "above", 181.55, candle_date="2026-07-25", fill_source="candle"
    )
    assert panel.disarm_d1_level_watch("NVDA", "above", 181.55)
    rows = load_review_events(path)
    assert [row["action"] for row in rows] == ["arm_level", "disarm_level"]
    assert rows[0]["detail"] == {
        "direction": "above",
        "level": 181.55,
        "candle_date": "2026-07-25",
        "fill_source": "candle",
    }


def test_setups_panel_logs_swing_star_and_dislike(tmp_path):
    if _qt_app() is None:
        return
    from focus_picks import FocusPickStore
    from ui.models.setup import SetupRow
    from ui.panels.master_avwap_panel import MasterAvwapPanel
    from ui.services.focus_service import FocusService

    service = FocusService(
        FocusPickStore(
            focus_longs_path=tmp_path / "focus_longs.txt",
            focus_shorts_path=tmp_path / "focus_shorts.txt",
            longs_path=tmp_path / "longs.txt",
            shorts_path=tmp_path / "shorts.txt",
            membership_path=tmp_path / "focus_pick_membership.json",
        )
    )
    path = tmp_path / "events.jsonl"
    panel = MasterAvwapPanel(service, review_events_path=path)
    row = SetupRow(
        symbol="LNG",
        side="LONG",
        score=245.0,
        bucket="favorite_setup",
        setup_tags=["AVWAP_BREAKOUT"],
        expected_r=0.85,
        raw={"setup_family": "avwap_breakout"},
    )
    panel._record_review_event(
        "favorite", row, {"on": True, "origin": "setups", "category": "swing"}
    )
    panel._record_dislike(row, "chasing; too far from the level")

    rows = load_review_events(path)
    assert [r["action"] for r in rows] == ["favorite", "dislike"]
    star, dislike = rows
    assert star["surface"] == "setups"
    assert star["bucket"] == "favorite_setup"
    assert star["setup_family"] == "avwap_breakout"
    assert star["is_d1"] is True
    assert dislike["detail"]["reason"] == "chasing; too far from the level"

    # A bare test panel (no default store, no explicit path) must stay silent.
    silent = MasterAvwapPanel(None)
    assert silent._review_events_path is None


def test_arm_bar_tracks_the_quick_fill_source():
    if _qt_app() is None:
        return
    from ui.widgets.arm_bar import ArmBar

    bar = ArmBar()
    bar.set_quick_fill_source(lambda source: {"vwap": 100.5, "upper_1": 101.9}.get(source))
    assert bar.apply_quick_fill("vwap")
    assert bar.last_fill_source() == "vwap"
    assert bar.apply_quick_fill("upper_1")
    assert bar.last_fill_source() == "upper_1"
    bar.set_level(99.25)
    assert bar.last_fill_source() == "chart_click"
    # The trader typing over the fill overrides the remembered source.
    bar.level_input.setValue(97.10)
    assert bar.last_fill_source() == "manual"
