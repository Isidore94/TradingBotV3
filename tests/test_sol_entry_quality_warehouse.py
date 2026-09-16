"""Phase 0.32 reviewer regressions: warehouse rows must reach the shared report.

Every store below is a synthetic recording fake.  These tests never resolve a
project data directory or fetch a bar.
"""

from __future__ import annotations

import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


NOW = datetime(2026, 9, 15, 21, 0, tzinfo=timezone.utc)


def _entry() -> dict:
    return {
        "occurrence_id": "p8-AAA-2026-09-15",
        "opportunity_id": "p8-AAA-2026-09-15",
        "attempt_id": "p8-AAA-2026-09-15|p8_immediate_v1",
        "symbol": "AAA",
        "side": "LONG",
        "entry_selector_id": "p8_immediate_v1",
        "entry_rule": "p8_immediate",
        "entry_rule_version": "v1",
        "trigger_knowledge_time": "2026-09-15T09:30:00-04:00",
        "feasible_entry_time": "2026-09-15T09:30:00-04:00",
        "feasible_entry_price": 100.0,
        "risk_price": 98.0,
        "entry_atr": 2.0,
        "source_knowledge_basis": "observed",
        "anchor_knowledge_basis": "observed",
    }


def _bars() -> list[dict]:
    # The first whole bar trips the hypothetical stop; the later rally is the
    # crucial gross MFE regression and must stay visible in the flat 30m row.
    return [
        {"symbol": "AAA", "interval_start": "2026-09-15T09:30:00-04:00", "end_time": "2026-09-15T09:35:00-04:00", "high": 100.0, "low": 97.0, "close": 98.0},
        {"symbol": "AAA", "interval_start": "2026-09-15T09:35:00-04:00", "end_time": "2026-09-15T09:40:00-04:00", "high": 110.0, "low": 98.0, "close": 109.0},
        {"symbol": "AAA", "interval_start": "2026-09-15T09:40:00-04:00", "end_time": "2026-09-15T09:45:00-04:00", "high": 109.0, "low": 107.0, "close": 108.0},
        {"symbol": "AAA", "interval_start": "2026-09-15T09:45:00-04:00", "end_time": "2026-09-15T09:50:00-04:00", "high": 108.0, "low": 106.0, "close": 107.0},
        {"symbol": "AAA", "interval_start": "2026-09-15T09:50:00-04:00", "end_time": "2026-09-15T09:55:00-04:00", "high": 107.0, "low": 105.0, "close": 106.0},
        {"symbol": "AAA", "interval_start": "2026-09-15T09:55:00-04:00", "end_time": "2026-09-15T10:00:00-04:00", "high": 106.0, "low": 104.0, "close": 105.0},
    ]


def _flat_row(*, state: str = "complete") -> dict:
    return {
        "schema_version": "entry_quality_window_v1",
        "opportunity_id": "p8-AAA-2026-09-15",
        "attempt_id": "p8-AAA-2026-09-15|p8_immediate_v1",
        "window": "30_trading_minutes",
        "entry_at": "2026-09-15T09:30:00-04:00",
        "symbol": "AAA",
        "side": "LONG",
        "state": state,
        "mfe_pct": 10.0 if state == "complete" else None,
        "mae_pct": -3.0 if state == "complete" else None,
        "close_pct": 5.0 if state == "complete" else None,
        "mfe_atr": 5.0 if state == "complete" else None,
        "mae_atr": -1.5 if state == "complete" else None,
        "mfe_r": 5.0 if state == "complete" else None,
        "mae_r": -1.5 if state == "complete" else None,
        "first_touch_order": "adverse_first" if state == "complete" else "not_measured",
        "coverage": {"expected_bars": 6, "observed_bars": 6, "missing_bars": 0},
        "source_knowledge_basis": "observed",
        "anchor_knowledge_basis": "observed",
        "entry_rule": "p8_immediate",
        "entry_rule_version": "v1",
        "exit_policy": "gross_excursion_no_exit",
    }


def test_entry_quality_window_is_a_versioned_additive_warehouse_dataset_at_attempt_window_grain():
    from research_warehouse import schemas

    spec = schemas.dataset_spec("entry_quality_window")
    names = set(spec.schema.names)

    assert spec.grain == ("opportunity_id", "attempt_id", "window")
    assert {
        "schema_version", "opportunity_id", "attempt_id", "window", "state",
        "mfe_pct", "mae_pct", "close_pct", "mfe_atr", "mae_atr", "mfe_r", "mae_r",
        "first_touch_order", "coverage", "source_knowledge_basis", "anchor_knowledge_basis",
    }.issubset(names)


def test_p8_window_builder_uses_completed_store_bars_and_preserves_gross_post_stop_rally_without_mutating_outcomes():
    from research_warehouse import outcomes

    old_outcomes = [{"occurrence_id": _entry()["occurrence_id"], "recipe_id": "p8_old_exit_v1", "net_r": -1.0}]
    old_before = repr(copy.deepcopy(old_outcomes))

    class Store:
        def __init__(self):
            self.reads: list[tuple] = []
            self.published: list[tuple] = []

        def read_rows(self, dataset, partition=None, **kwargs):
            self.reads.append((dataset, partition, kwargs))
            assert dataset == "bar_m5"
            return _bars()

        def publish(self, dataset, rows, *, job_id=""):
            materialized = [dict(row) for row in rows]
            self.published.append((dataset, materialized, job_id))
            return {"written": len(materialized)}

    store = Store()
    rows = outcomes.build_entry_quality_windows(
        store=store,
        p8_occurrences=[_entry()],
        declared_entry_selector_ids={"p8_immediate_v1"},
        as_of=NOW,
        job_id="test-entry-quality",
    )

    thirties = [row for row in rows if row["window"] == "30_trading_minutes"]
    assert len(thirties) == 1
    row = thirties[0]
    assert row["state"] == "complete"
    assert row["mfe_pct"] == pytest.approx(10.0)
    assert row["first_touch_order"] == "adverse_first"
    assert store.reads and store.reads[0][0] == "bar_m5"
    assert store.published and store.published[0][0] == "entry_quality_window"
    assert repr(old_outcomes) == old_before


def test_entry_comparison_export_is_built_from_flat_window_rows_and_carries_window_and_cells_together():
    import entry_comparison

    rows = [_flat_row(), {**_flat_row(state="no_trigger"), "attempt_id": "p8-BBB|p8_immediate_v1", "opportunity_id": "p8-BBB"}]
    payload = entry_comparison.build_export(rows, as_of="2026-09-15")

    assert payload["schema"] == "entry_quality_comparison_export_v1"
    assert payload["window"] == "30_trading_minutes"
    assert payload["cells"]
    assert payload["cells"]["p8_immediate_v1"]["opportunity_count"] == 2
    assert payload["cells"]["p8_immediate_v1"]["exclusions"]["no_trigger"] == 1


def test_default_warehouse_reader_uses_session_month_scoped_entry_quality_rows_for_the_report(monkeypatch):
    from ai_jobs import measured_report_publish
    from research_warehouse.store import ResearchStore

    class Store:
        root = Path("C:/synthetic/research")

        def __init__(self):
            self.calls: list[tuple] = []

        def partition_dir(self, dataset, partition):
            assert dataset == "entry_quality_window"
            assert partition == "month=2026-09"
            return Path("C:/synthetic/research/entry_quality_window/month=2026-09")

        def read_rows(self, dataset, partition, **kwargs):
            self.calls.append((dataset, partition, kwargs))
            return [_flat_row()]

    store = Store()
    monkeypatch.setattr(ResearchStore, "open", classmethod(lambda _cls: store))
    warehouse = measured_report_publish.default_warehouse()
    rows = warehouse.read_entry_quality("2026-09-15", now=NOW)

    assert rows == [_flat_row()]
    assert len(store.calls) == 1
    dataset, partition, kwargs = store.calls[0]
    assert (dataset, partition) == ("entry_quality_window", "month=2026-09")
    start, end = kwargs["interval_start_range"]
    assert start.date().isoformat() == "2026-09-15"
    assert end.date().isoformat() == "2026-09-16"
    assert kwargs["time_column"] == "entry_at"


def test_unavailable_entry_quality_cell_keeps_the_shared_report_population_window_version_and_exit_policy_honest():
    import measured_report

    report = measured_report.build_report(
        "2026-09-15", now=NOW, sources=measured_report.ReportSources()
    )
    cells = report.as_dict()["entry_quality"]["cells"]

    assert len(cells) == 1
    cell = cells[0]
    assert cell["state"] == "unknown"
    assert cell["value"] is None
    assert cell["population"] == "all_scanner"
    assert cell["window"] == ["entry_quality_window", "not_published"]
    assert cell["version"] == "entry_quality_window_v1"
    assert cell["exit_policy"] == "gross_excursion_no_exit"
