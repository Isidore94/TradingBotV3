"""Freeze today's ``rebuild_trades`` behaviour before R7 changes any of it.

``docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`` §9 step 0. The golden fixture
records what the assembler does now - **including six known defects**, listed in
``test_the_golden_still_contains_the_defects_r7_is_here_to_fix`` below. Those
values are wrong on purpose: a characterization fixture that quietly recorded
the *intended* behaviour would prove nothing when the fix lands.

When a later step changes assembly, this file goes red. That is the design. The
fix is to regenerate the golden with ``tests/journal_characterization.py``, write
the change into the fixture's ``intentional_difference`` field, and narrow the
defect list below - never to loosen the comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from conftest import load_fixture_contract  # noqa: E402
from journal_characterization import (  # noqa: E402
    CHARACTERIZATION_EXECUTIONS,
    FIXTURE_NAME,
    build_fixture_store,
    capture_rebuild_output,
)
from journal_store import JournalStore  # noqa: E402


@pytest.fixture
def golden():
    return load_fixture_contract(FIXTURE_NAME)


@pytest.fixture
def rebuilt(tmp_path):
    store = build_fixture_store(tmp_path / "trade_journal.sqlite3")
    return capture_rebuild_output(store)


def test_the_fixture_inputs_are_the_frozen_executions(golden):
    """The golden's declared input is the row list this module ships, verbatim."""
    assert golden["executions"] == CHARACTERIZATION_EXECUTIONS
    assert golden.raw_input_digest() == golden["raw_input_sha256"]


@pytest.mark.parametrize("section", ["trades", "legs", "opportunity_events", "summary"])
def test_rebuild_trades_output_is_bit_for_bit_unchanged(golden, rebuilt, section):
    """Bit-for-bit, not within a tolerance - the fixture declares 0.0."""
    assert golden.tolerance == 0.0
    assert rebuilt[section] == golden[section]


def test_rebuild_is_idempotent(tmp_path):
    """Rebuilding twice over the same rows may not change the answer."""
    store = build_fixture_store(tmp_path / "trade_journal.sqlite3")
    first = capture_rebuild_output(store)
    second = capture_rebuild_output(store)
    assert second == first


def test_refreshing_auto_tags_does_not_change_assembly(tmp_path, golden):
    """The default ``refresh_tags=True`` path leaves every assembled number alone.

    The golden is captured with tags off (``AutoTagger`` reads machine-local
    files). This is what keeps that exclusion honest: the default call must
    agree with the golden on everything except the two tag columns.
    """
    store = build_fixture_store(tmp_path / "trade_journal.sqlite3")
    assert store.rebuild_trades() == golden["summary"]["returned_trade_count"]
    with store.connection() as conn:
        trades = [dict(row) for row in conn.execute("SELECT * FROM trades ORDER BY trade_id")]
    tag_columns = {"auto_tag_summary", "tag_confidence", "updated_at"}
    stripped = [{k: v for k, v in sorted(t.items()) if k not in tag_columns} for t in trades]
    expected = [{k: v for k, v in t.items() if k not in tag_columns} for t in golden["trades"]]
    assert stripped == expected


def test_the_golden_still_contains_the_defects_r7_is_here_to_fix(golden):
    """Name each frozen defect, so the fix has something specific to turn green.

    Every assertion here is a statement about *broken* behaviour. Each one is
    expected to be inverted by the step named beside it; when that happens,
    change the assertion in the same commit as the fix and say so in the
    fixture's ``intentional_difference``.
    """
    trades = {(t["broker"], t["symbol"], t["security_type"]): t for t in golden["trades"]}

    # B4 (§9 step 2) - one IBKR fill arriving from both the socket and Flex is
    # counted twice, so a 10-share position opens as 20 and never closes.
    nvda = trades[("IBKR", "NVDA", "STK")]
    assert nvda["quantity_opened"] == 20.0, "socket+Flex duplicate no longer doubles"
    assert nvda["status"] == "OPEN"

    # B2 (§9 step 4) - a closing sell with no opening buy fabricates a short.
    amd = trades[("QUESTRADE", "AMD", "STOCK")]
    assert amd["direction"] == "SHORT" and amd["quantity_opened"] == 100.0
    assert amd["status"] == "OPEN"

    # B3 (§9 step 3) - the `listingExchange` fallback splits one AMZN position
    # into two groups that can never net against each other.
    assert ("QUESTRADE", "AMZN", "STOCK") in trades
    assert ("QUESTRADE", "AMZN", "NASDAQ") in trades

    # B3 (§9 step 3) - a manual fill is keyed broker="MANUAL" and so orphans
    # itself from the Questrade AAPL position it belongs to.
    assert ("MANUAL", "AAPL", "STK") in trades

    # B1 (§9 step 4) - CLOSED_PARTIAL does not exist; a partially exited trade
    # is indistinguishable from an untouched one.
    assert {t["status"] for t in golden["trades"]} <= {"OPEN", "CLOSED"}
    aapl = trades[("QUESTRADE", "AAPL", "STOCK")]
    assert aapl["quantity_closed"] == 120.0 and aapl["status"] == "OPEN"

    # B8 (§9 step 8) - no CAD conversion exists, so a CAD trade carries no
    # comparable P&L at all.
    shop = trades[("QUESTRADE", "SHOP.TO", "STOCK")]
    assert shop["currency"] == "CAD" and shop["pnl_usd"] is None
    assert "net_pnl_cad" not in shop


def test_annotations_are_orphaned_by_a_rebuild(tmp_path):
    """B6 (§9 step 4): today a rebuild deletes trades and strands their notes.

    This is the invariant I4 exists to create. It does not hold yet, and the
    golden would not show it - annotations are not part of assembly output - so
    it gets its own test that says plainly what is true today.
    """
    db_path = tmp_path / "trade_journal.sqlite3"
    store = build_fixture_store(db_path)
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        trade_id = conn.execute("SELECT trade_id FROM trades WHERE symbol = 'AMD'").fetchone()[0]
    store.save_trade_annotation(trade_id, setup_tags="avwap-reclaim", notes="worked")

    # A backfill inserts an earlier opening fill for the AMD phantom short,
    # which re-sequences that group and re-keys its trade ids.
    store.upsert_executions(
        [
            dict(
                execution_uid="QT:51234567:e-amd-8:AMD:2026-08-04T09:40:00-07:00",
                broker="QUESTRADE",
                account_number="51234567",
                account_label="Margin",
                account_type="",
                symbol="AMD",
                security_type="STOCK",
                currency="USD",
                side="BUY",
                quantity=100,
                price=90.00,
                timestamp="2026-08-04T09:40:00-07:00",
                trade_date="2026-08-04",
                commission=4.95,
                fees=0.0,
                gross_amount=None,
                net_amount=None,
                order_id="",
                exchange_exec_id="",
                raw_json="{}",
            )
        ]
    )
    store.rebuild_trades(refresh_tags=False)

    with store.connection() as conn:
        orphans = conn.execute(
            """
            SELECT a.trade_id FROM trade_annotations a
            LEFT JOIN trades t ON t.trade_id = a.trade_id
            WHERE t.trade_id IS NULL
            """
        ).fetchall()
        amd_rows = conn.execute(
            "SELECT status, direction, quantity_opened FROM trades WHERE symbol = 'AMD'"
        ).fetchall()

    # The backfilled buy pairs with the sell, so AMD is now one closed long.
    assert [tuple(row) for row in amd_rows] == [("CLOSED", "LONG", 100.0)]
    # And the annotation is stranded, because nothing re-keys it. I4 will make
    # this list empty and keep it empty (permanent SQL test, spec §10 gate 4).
    assert len(orphans) == 1


def test_a_fresh_store_starts_at_schema_v2(tmp_path):
    """The migration in §9 step 2 starts from here: v2, no coverage ledger."""
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    with store.connection() as conn:
        version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert version == "2"
    assert not tables & {"import_coverage", "fx_rates", "trade_adjustments", "trade_aliases", "cash_transactions"}
