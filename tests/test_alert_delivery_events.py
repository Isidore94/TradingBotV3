"""Phase 1 delivery capture: identity, storage class, and never costing an alert.

Three properties matter here and each is asserted rather than assumed:

* delivery rows stay MACHINE-LOCAL - the trader's decision, because one row per
  alert is far higher volume than a decision row and the review store is
  Drive-synced;
* identity is typed and NOT time-anchored, or every re-fire becomes a distinct
  alert and the duplicate rate collapses to zero;
* recording is subordinate to delivering - a failing write returns None and
  never raises into the caller.
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import alert_delivery_events as ade
from alert_delivery_events import (
    DELIVERED,
    DELIVERY_EVENTS_SCHEMA,
    TYPE_CHART_WATCH,
    TYPE_D1_EVENT,
    TYPE_ENTRY_ASSIST,
    TYPE_FOCUS_PICK,
    TYPE_M5_BOUNCE,
    TYPE_STATUS,
    WATCH_DELIVERED,
    alert_event_id,
    alert_type_for,
    delivery_store_path,
    load_delivery_events,
    record_delivery,
    record_watch_delivery,
    thesis_anchor_for,
)


class FakeAlert:
    """BounceAlert-shaped stand-in; the module is duck-typed on purpose."""

    def __init__(
        self,
        symbol="AAPL",
        side="LONG",
        tag="",
        raw_text="",
        trigger="reclaim",
        is_d1=False,
        payload=None,
        timeframe="M5",
    ):
        self.symbol = symbol
        self.side = side
        self.tag = tag
        self.raw_text = raw_text
        self.trigger = trigger
        self.is_d1 = is_d1
        self.payload = payload or {}
        self.timeframe = timeframe


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    """Redirect the machine-local store into a temp dir for every test here.

    ``autouse`` on purpose. conftest.py's rule is that tests must never append
    synthetic events to the running application's evidence, and an opt-in
    fixture makes that a thing each new test has to remember - one that forgot
    is exactly how this module first leaked rows into the real diagnostics
    directory. Making isolation the default removes the chance to forget.
    """

    monkeypatch.setattr(ade, "get_diagnostics_dir", lambda: tmp_path)
    return tmp_path / "alert_delivery_events"


# --- storage class ----------------------------------------------------------


def test_delivery_rows_never_land_in_the_drive_home_folder(store, monkeypatch):
    import project_paths

    record_delivery(FakeAlert(), loud=True, sounded=True)
    written = list(store.glob("*.jsonl"))
    assert written, "expected a machine-local shard"

    home = Path(project_paths.PERSISTENT_DATA_DIR).resolve()
    for path in written:
        assert home not in path.resolve().parents


def test_store_is_partitioned_by_month(store):
    record_delivery(FakeAlert(), loud=True, sounded=True, now=datetime(2026, 8, 10))
    record_delivery(FakeAlert(), loud=True, sounded=True, now=datetime(2026, 9, 2))
    names = sorted(path.name for path in store.glob("*.jsonl"))
    assert names == ["alert-deliveries-2026-08.jsonl", "alert-deliveries-2026-09.jsonl"]


def test_store_path_follows_a_relocated_diagnostics_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ade, "get_diagnostics_dir", lambda: tmp_path / "a")
    first = delivery_store_path(date(2026, 8, 10))
    monkeypatch.setattr(ade, "get_diagnostics_dir", lambda: tmp_path / "b")
    second = delivery_store_path(date(2026, 8, 10))
    assert first != second, "root must be read per call, not frozen at import"


# --- typed identity ---------------------------------------------------------


def test_alert_type_reads_the_surface_before_the_d1_flag():
    chart_watch_d1 = FakeAlert(tag="chart_watch", is_d1=True)
    assert alert_type_for(chart_watch_d1) == TYPE_CHART_WATCH

    assert alert_type_for(FakeAlert(tag="entry_assist")) == TYPE_ENTRY_ASSIST
    assert alert_type_for(FakeAlert(tag="auto_pick")) == TYPE_FOCUS_PICK
    assert alert_type_for(FakeAlert(tag="focus_review")) == TYPE_FOCUS_PICK
    assert alert_type_for(FakeAlert(is_d1=True)) == TYPE_D1_EVENT
    assert alert_type_for(FakeAlert()) == TYPE_M5_BOUNCE
    assert alert_type_for(FakeAlert(symbol="")) == TYPE_STATUS


def test_tag_literals_still_match_the_source_of_truth():
    """The literals are copied to keep this module out of the ui package."""

    from ui.models import bounce

    assert ade.CHART_WATCH_TAG == bounce.CHART_WATCH_TAG
    assert ade.MANUAL_CHART_TAG == bounce.MANUAL_CHART_TAG
    assert ade.AUTO_PICK_TAG == bounce.AUTO_PICK_TAG
    assert ade.FOCUS_REVIEW_TAG == bounce.FOCUS_REVIEW_TAG
    assert ade.FOCUS_D1_EVENT_TAG == bounce.FOCUS_D1_EVENT_TAG


def test_chart_watch_anchor_uses_the_payload_key_the_panel_writes():
    alert = FakeAlert(tag="chart_watch", payload={"chart_watch_kind": "RECLAIM"})
    assert thesis_anchor_for(alert) == "reclaim"


def test_identity_is_stable_across_redeliveries_of_the_same_thesis():
    alert = FakeAlert()
    first = alert_event_id(alert, trade_date="2026-08-10")
    second = alert_event_id(alert, trade_date="2026-08-10")
    assert first == second


def test_identity_is_not_anchored_on_time():
    """Time in the key would make every re-fire unique and the metric useless."""

    alert = FakeAlert()
    morning = record_delivery(
        alert, loud=True, sounded=True, now=datetime(2026, 8, 10, 9, 31)
    )
    afternoon = record_delivery(
        alert, loud=True, sounded=True, now=datetime(2026, 8, 10, 14, 5)
    )
    assert morning["alert_event_id"] == afternoon["alert_event_id"]
    assert morning["ts"] != afternoon["ts"]


def test_different_sides_and_families_are_different_alerts():
    long_alert = FakeAlert(side="LONG")
    short_alert = FakeAlert(side="SHORT")
    d1_alert = FakeAlert(is_d1=True)
    ids = {
        alert_event_id(long_alert, trade_date="2026-08-10"),
        alert_event_id(short_alert, trade_date="2026-08-10"),
        alert_event_id(d1_alert, trade_date="2026-08-10"),
    }
    assert len(ids) == 3


# --- escalation inputs ------------------------------------------------------


def test_row_stores_escalation_inputs_rather_than_a_verdict(store):
    alert = FakeAlert(raw_text="AAPL [A-TIER] reclaim")
    row = record_delivery(alert, loud=True, sounded=True)
    assert row["tier"] == "A"
    assert row["loud"] is True
    assert row["is_armed_fire"] is False
    assert "is_duplicate" not in row
    assert "is_escalation" not in row


def test_loud_is_recorded_not_rederived(store):
    """The panel owns the loudness rule; a second implementation would drift."""

    loud_looking = FakeAlert(raw_text="AAPL [S-TIER] BANGER")
    row = record_delivery(loud_looking, loud=False, sounded=False)
    assert row["loud"] is False


def test_muted_loud_alert_is_recorded_as_loud_but_not_sounded(store):
    row = record_delivery(FakeAlert(), loud=True, sounded=False)
    assert row["loud"] is True
    assert row["sounded"] is False


def test_context_never_overwrites_identity_or_escalation_fields(store):
    hostile = FakeAlert(
        raw_text="AAPL [D-TIER] reclaim",
        payload={"feedback": {"tier": "S", "loud": True, "alert_event_id": "spoofed"}},
    )
    row = record_delivery(hostile, loud=False, sounded=False)
    assert row["tier"] == "D"
    assert row["loud"] is False
    assert row["alert_event_id"] != "spoofed"


# --- watch delivery ---------------------------------------------------------


def test_watch_delivery_carries_the_latency_that_makes_the_bound_checkable(store):
    row = record_watch_delivery(
        FakeAlert(tag="chart_watch"), watch_id="w-1", fired_to_delivered_ms=250
    )
    assert row["action"] == WATCH_DELIVERED
    assert row["watch_id"] == "w-1"
    assert row["fired_to_delivered_ms"] == 250
    assert row["is_armed_fire"] is True


def test_watch_delivery_without_an_id_is_refused(store):
    assert record_watch_delivery(FakeAlert(), watch_id="") is None


def test_negative_latency_is_clamped_rather_than_stored(store):
    row = record_watch_delivery(FakeAlert(), watch_id="w-1", fired_to_delivered_ms=-5)
    assert row["fired_to_delivered_ms"] == 0


# --- recording never costs an alert -----------------------------------------


def test_symbol_less_alert_is_refused_without_raising(store):
    assert record_delivery(FakeAlert(symbol=""), loud=True, sounded=True) is None


def test_unwritable_store_returns_none_instead_of_raising(store, monkeypatch):
    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", explode)
    assert record_delivery(FakeAlert(), loud=True, sounded=True) is None


def test_malformed_alert_cannot_break_the_write(store):
    class Hostile:
        symbol = "AAPL"
        side = "LONG"

        @property
        def raw_text(self):
            raise RuntimeError("boom")

        @property
        def payload(self):
            raise RuntimeError("boom")

    row = record_delivery(Hostile(), loud=True, sounded=True)
    assert row is not None
    assert row["symbol"] == "AAPL"


# --- reading back -----------------------------------------------------------


def test_rows_load_back_oldest_first_across_months(store):
    record_delivery(FakeAlert(symbol="BBB"), loud=True, sounded=True, now=datetime(2026, 9, 1))
    record_delivery(FakeAlert(symbol="AAA"), loud=True, sounded=True, now=datetime(2026, 8, 1))
    rows = load_delivery_events(store)
    assert [row["symbol"] for row in rows] == ["AAA", "BBB"]
    assert all(row["schema"] == DELIVERY_EVENTS_SCHEMA for row in rows)
    assert all(row["action"] == DELIVERED for row in rows)


def test_a_truncated_line_does_not_cost_the_rest_of_the_store(store):
    record_delivery(FakeAlert(symbol="AAA"), loud=True, sounded=True, now=datetime(2026, 8, 1))
    shard = next(store.glob("*.jsonl"))
    with shard.open("a", encoding="utf-8") as handle:
        handle.write('{"schema": "alert_delivery_ev\n')
        handle.write(json.dumps({"schema": DELIVERY_EVENTS_SCHEMA, "symbol": "BBB", "ts": "z"}) + "\n")
    rows = load_delivery_events(store)
    assert sorted(row["symbol"] for row in rows) == ["AAA", "BBB"]


def test_foreign_schema_rows_are_ignored(store):
    record_delivery(FakeAlert(symbol="AAA"), loud=True, sounded=True, now=datetime(2026, 8, 1))
    shard = next(store.glob("*.jsonl"))
    with shard.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": "review_events_v2", "symbol": "XXX"}) + "\n")
    assert [row["symbol"] for row in load_delivery_events(store)] == ["AAA"]


def test_missing_store_reads_as_empty(tmp_path):
    assert load_delivery_events(tmp_path / "nope") == []
