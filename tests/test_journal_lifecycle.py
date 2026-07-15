import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from journal_importers import manual_execution_from_fields  # noqa: E402
from journal_store import JOURNAL_SCHEMA_VERSION, JournalStore  # noqa: E402


def _execution(execution_id: str, side: str, price: float, timestamp: str):
    return manual_execution_from_fields(
        {
            "broker": "MANUAL",
            "account_number": "A",
            "symbol": "NVDA",
            "side": side,
            "quantity": 10,
            "price": price,
            "timestamp": timestamp,
            "security_type": "STK",
            "currency": "USD",
            "execution_id": execution_id,
        }
    )


def test_opportunity_lifecycle_is_append_only_and_filterable(tmp_path):
    store = JournalStore(tmp_path / "journal.sqlite3")
    first = store.record_opportunity_event(
        opportunity_id="opp-1",
        lifecycle_id="life-1",
        event_type="SEEN",
        symbol="nvda",
        side="long",
        occurred_at="2026-07-14T09:31:00",
        payload={"bucket": "favorite"},
    )
    second = store.record_opportunity_event(
        opportunity_id="opp-1",
        lifecycle_id="life-1",
        event_type="SKIPPED",
        symbol="NVDA",
        side="LONG",
        occurred_at="2026-07-14T09:40:00",
        reason="too extended",
    )

    rows = store.list_opportunity_events(lifecycle_id="life-1")
    assert [row["event_type"] for row in rows] == ["SEEN", "SKIPPED"]
    assert rows[0]["payload"] == {"bucket": "favorite"}
    assert rows[1]["reason"] == "too extended"
    assert first["event_id"] != second["event_id"]

    with pytest.raises(ValueError):
        store.record_opportunity_event(opportunity_id="opp-1", event_type="MADE_UP")
    assert JOURNAL_SCHEMA_VERSION == 2


def test_trade_rebuild_adds_taken_and_closed_once(tmp_path):
    store = JournalStore(tmp_path / "journal.sqlite3")
    store.upsert_executions(
        [
            _execution("open", "BUY", 100.0, "2026-07-14T09:31:00"),
            _execution("close", "SELL", 105.0, "2026-07-14T11:00:00"),
        ]
    )
    assert store.rebuild_trades(refresh_tags=False) == 1
    trade = store.list_trades()[0]
    assert [row["event_type"] for row in store.list_opportunity_events(trade_id=trade["trade_id"])] == [
        "TAKEN",
        "CLOSED",
    ]

    # Rebuild is idempotent for deterministic broker-derived events.
    store.rebuild_trades(refresh_tags=False)
    assert len(store.list_opportunity_events(trade_id=trade["trade_id"])) == 2


def test_structured_trade_review_preserves_each_review(tmp_path):
    store = JournalStore(tmp_path / "journal.sqlite3")
    store.record_opportunity_event(
        opportunity_id="trade:t1",
        event_type="REVIEWED",
        trade_id="t1",
        reason="clean confirmation",
        payload={"review_outcome": "Followed plan"},
        occurred_at="2026-07-14T12:00:00",
    )
    store.record_opportunity_event(
        opportunity_id="trade:t1",
        event_type="REVIEWED",
        trade_id="t1",
        reason="on second review, exit was early",
        payload={"review_outcome": "Poor exit discipline"},
        occurred_at="2026-07-14T13:00:00",
    )

    latest = store.latest_trade_review("t1")
    assert latest is not None
    assert latest["payload"]["review_outcome"] == "Poor exit discipline"
    assert len(store.list_opportunity_events(trade_id="t1", event_type="REVIEWED")) == 2
