"""The measured report's swing examples name the ticker, not a hash.

Trader 2026-09-23: `measured_report_2026-09-22.md` listed
"2a7c985bb0b8f1fd0a410996e0921a76: 8.2522 R" under "Biggest swing movement in
R". A row must read as symbol, side, setup, date and R; the occurrence id
stays only as a short trailing reference.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import measured_report  # noqa: E402

SESSION = "2026-09-22"
NOW = datetime(2026, 9, 23, 2, 30)
OCC = "2a7c985bb0b8f1fd0a410996e0921a76"


class _Warehouse:
    source_paths = ("research_lake/outcome_path",)

    def __init__(self, rows):
        self._rows = rows

    def read_outcomes(self, session_date, *, now=None):
        return [dict(row) for row in self._rows]

    def read_entry_quality(self, session_date, *, now=None):
        return []


def _swing_table(rows):
    report = measured_report.build_report(
        SESSION, now=NOW, sources=measured_report.ReportSources(),
        warehouse=_Warehouse(rows),
    )
    tables = {table["table_id"]: table for table in report.example_tables}
    return report, tables["swing.best_moves"]


def _row(occurrence_id, value, **extra):
    row = {
        "occurrence_id": occurrence_id,
        "recipe_id": "house",
        "outcome_definition_id": "house_default_v1",
        "entry_at": "2026-09-22T13:35:00+00:00",
        "mfe_r": value,
    }
    row.update(extra)
    return row


def test_a_swing_example_names_symbol_side_setup_and_date():
    _report, table = _swing_table([
        _row(OCC, 8.2522, symbol="VKTX", side="LONG", canonical_setup_id="avwap_band_bounce"),
        _row("b" * 32, 4.2603, symbol="FSLY", side="SHORT", canonical_setup_id="avwap_breakout"),
    ])
    first = table["rows"][0]
    assert first["symbol"] == "VKTX"
    assert first["side"] == "LONG"
    assert first["setup"] == "avwap_band_bounce"
    assert first["date"] == "2026-09-22"
    assert first["value"] == 8.2522
    assert first["name"].startswith("VKTX LONG avwap_band_bounce")
    assert "2026-09-22" in first["name"]
    # The id survives only as a short trailing reference.
    assert OCC not in first["name"]
    assert first["occurrence_id"] == OCC
    assert first["name"].endswith(f"(occ {OCC[:8]})")


def test_a_swing_row_without_a_symbol_says_unknown_rather_than_printing_the_hash():
    _report, table = _swing_table([_row(OCC, 3.0)])
    name = table["rows"][0]["name"]
    assert name.startswith("unknown symbol")
    assert OCC not in name


def test_one_occurrence_measured_by_two_recipes_is_one_example():
    _report, table = _swing_table([
        _row(OCC, 5.0, symbol="VKTX", side="LONG", canonical_setup_id="x", recipe_id="a"),
        _row(OCC, 4.0, symbol="VKTX", side="LONG", canonical_setup_id="x", recipe_id="b"),
        _row("c" * 32, 1.0, symbol="ABSI", side="LONG", canonical_setup_id="x"),
    ])
    assert [row["symbol"] for row in table["rows"]] == ["VKTX", "ABSI"]
    assert table["denominator"] == 3


def test_the_published_markdown_line_leads_with_the_ticker():
    report, _table = _swing_table([
        _row(OCC, 8.2522, symbol="VKTX", side="LONG", canonical_setup_id="avwap_band_bounce"),
    ])
    block = measured_report.render_markdown(report).split("**Biggest swing movement in R**", 1)[1]
    line = next(item for item in block.splitlines() if item.startswith("  - "))
    assert line.strip().startswith("- VKTX LONG avwap_band_bounce")
    assert line.strip().endswith(": 8.2522 R")
    assert f"- {OCC}" not in block


def test_the_lake_reader_joins_symbol_side_and_setup_from_the_occurrence(tmp_path, monkeypatch):
    """outcome_path carries no symbol; the reader joins setup_occurrence."""
    from ai_jobs import measured_report_publish
    from research_warehouse.store import ResearchStore

    store = ResearchStore.open(tmp_path / "research")
    assert store is not None
    entry = datetime(2026, 9, 22, 14, 0, tzinfo=timezone.utc)
    store.publish("setup_occurrence", [{
        "occurrence_id": OCC,
        "symbol": "VKTX",
        "canonical_setup_id": "avwap_band_bounce",
        "side": "LONG",
        "structural_timeframe": "D1",
        "revision_id": "r1",
        "event_at": entry,
        "computed_at": entry,
        "run_id": "test",
    }], job_id="test_occ")
    store.publish("outcome_path", [{
        "occurrence_id": OCC,
        "recipe_id": "house",
        "outcome_definition_id": "house_default_v1",
        "entry_at": entry,
        "mfe_r": 8.2522,
        "computed_at": entry,
        "run_id": "test",
    }], job_id="test_outcome")
    monkeypatch.setattr(ResearchStore, "open", classmethod(lambda _cls: store))

    rows = measured_report_publish.default_warehouse().read_outcomes(SESSION, now=NOW)

    assert len(rows) == 1
    assert rows[0]["symbol"] == "VKTX"
    assert rows[0]["side"] == "LONG"
    assert rows[0]["canonical_setup_id"] == "avwap_band_bounce"
