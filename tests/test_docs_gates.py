"""Regression guards for the active live-gate summary."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_live_gate_summary_has_no_truncations_or_closed_rows():
    """Check only the summary integrity rules; gate details remain in notes."""
    text = (ROOT_DIR / "docs" / "GATES.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("- #")]
    gate_ids = {int(line.split()[1][1:]) for line in rows}

    assert len(rows) == len(gate_ids)
    assert "\u2026" not in text
    assert "..." not in text
    assert {61, 62, 78}.isdisjoint(gate_ids)
    gate_63 = next(line for line in rows if line.startswith("- #63 "))
    assert "journal_enrichment" in gate_63 and "refused" in gate_63
    gate_77 = next(line for line in rows if line.startswith("- #77 "))
    assert "st3_execution_compare" in gate_77
