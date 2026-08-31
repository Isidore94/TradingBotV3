"""Phase 0.11 T5: the report and the Qt panel say the new premium things.

A credit means nothing without the strike it was written against, so the report
now states, per quoted sold put: the credit as a percent of the strike, the
weekly yield that ranks it, the spread it would fill in, the source of the
credit, and how many of SMA 50/100/200 are still above the strike (with the
2-or-more ranking boost visible).

The contract these pin is the ROUND TRIP: whatever `write_theta_put_report`
emits, `extract_theta_rows_from_report` reads back, and `ThetaRow`/
`ThetaTableModel` display. A field the writer emits and the extractor drops is
a field the desk never sees.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap  # noqa: E402


def _quoted_sold_put_row():
    """One theta row carrying a real quoted option, built through the ranker so
    the premium fields are the ranker's own rather than hand-written."""
    row = {
        "symbol": "ABC",
        "last_close": 105.0,
        "score": 80,
        "base_score": 80,
        "support_count": 3,
        "support_summary": "SMA_50 @ 98.00",
        "strike_zone": "95.00-98.00",
        "notes": "theta setup",
        "top_score_drivers": "trend",
        "risk_flags": [],
    }
    quote_row = {
        "strike": 95.0,
        "expiration": "20260508",
        "expiration_date": date(2026, 5, 8),
        "market_days": 5,
        "covered_support_count": 3,
        "covered_major_sma_support_count": 2,
        "covered_avwap_support_count": 1,
        "total_support_count": 3,
        "surrendered_support_count": 0,
        "support_quality_score": 3.0,
        "quote": {"bid": 1.19, "ask": 1.21},
    }
    ranked = master_avwap._rank_sold_put_option_recommendations(row, [quote_row])
    master_avwap._apply_best_option_to_theta_row(row, ranked)
    return row


def _report_text(rows):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "master_avwap_theta_puts.txt"
        master_avwap.write_theta_put_report(path, rows, [])
        return path.read_text(encoding="utf-8")


class TestTheReportStatesThePremium:
    def test_a_quoted_row_carries_credit_percent_yield_spread_source_and_smas(self):
        text = _report_text([_quoted_sold_put_row()])

        premium_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("premium=")]
        assert len(premium_lines) == 1
        line = premium_lines[0]
        for key in ("credit_pct=", "yield_pct_wk=", "spread_pct=", "source=", "sma_above_strike="):
            assert key in line, f"{key} missing from {line!r}"
        assert "(boost)" in line, "two SMAs above the strike is a ranking boost and says so"

    def test_the_header_states_the_percent_rules_not_the_old_dollar_target(self):
        text = _report_text([_quoted_sold_put_row()])

        assert "% of the strike" in text
        assert "information only" in text, "the $100/4-contract framing is display info now"

    def test_a_support_only_row_states_no_premium_it_does_not_have(self):
        """Inventing zeros for a row with no quote would read as a measurement."""
        row = {
            "symbol": "ABC",
            "last_close": 105.0,
            "score": 80,
            "base_score": 80,
            "support_count": 3,
            "support_summary": "SMA_50 @ 98.00",
            "strike_zone": "95.00-98.00",
            "notes": "",
            "risk_flags": [],
        }
        master_avwap._apply_best_option_to_theta_row(
            row,
            [{"play_type": "sold_put", "status": "support_only", "strike": 95.0, "credit": None}],
        )

        text = _report_text([row])

        assert "premium=" not in text


class TestTheExtractorReadsBackWhatWasWritten:
    def test_the_premium_line_round_trips(self):
        row = _quoted_sold_put_row()
        best = row["best_option"]
        text = _report_text([row])

        parsed = master_avwap.extract_theta_rows_from_report(text)

        assert len(parsed) == 1
        got = parsed[0]
        assert got["symbol"] == "ABC"
        assert got["credit_pct_of_strike"] == pytest.approx(best["credit_pct_of_strike"], abs=0.01)
        assert got["credit_pct_per_week"] == pytest.approx(best["credit_pct_per_week"], abs=0.01)
        assert got["spread_pct"] == pytest.approx(best["spread_pct"], abs=0.05)
        assert got["major_sma_above_strike"] == best["covered_major_sma_support_count"]
        assert got["recommended_credit_source"] == best["credit_source"]

    def test_a_row_without_a_premium_line_reads_back_as_unmeasured(self):
        text = _report_text([_quoted_sold_put_row()])
        stripped = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("premium=")
        )

        parsed = master_avwap.extract_theta_rows_from_report(stripped)

        assert len(parsed) == 1
        for key in ("credit_pct_of_strike", "credit_pct_per_week", "spread_pct", "major_sma_above_strike"):
            assert parsed[0][key] is None, key


@pytest.mark.qt
class TestTheDeskShowsThem:
    @staticmethod
    def _model_rows():
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.theta import ThetaRow
        from ui.models.theta_table_model import ThetaTableModel

        text = _report_text([_quoted_sold_put_row()])
        rows = [ThetaRow.from_mapping(row) for row in master_avwap.extract_theta_rows_from_report(text)]
        return ThetaTableModel(rows), rows

    def test_the_table_has_a_column_for_each_new_fact(self):
        model, _rows = self._model_rows()
        keys = [key for key, _label in model.COLUMNS]
        for key in ("credit_pct_of_strike", "credit_pct_per_week", "spread_pct", "major_sma_above_strike"):
            assert key in keys, key

    def test_the_display_strings_are_sane(self):
        from PySide6.QtCore import Qt

        model, rows = self._model_rows()
        by_key = {key: column for column, (key, _label) in enumerate(model.COLUMNS)}

        def shown(key):
            return model.data(model.index(0, by_key[key]), Qt.ItemDataRole.DisplayRole)

        assert shown("credit_pct_of_strike").endswith("%")
        assert shown("credit_pct_per_week").endswith("%")
        assert shown("spread_pct").endswith("%")
        # Two SMAs above the strike is the ranking boost; the board marks it.
        assert shown("major_sma_above_strike") == "2+"
        assert rows[0].credit_pct_of_strike is not None

    def test_an_unmeasured_row_shows_nothing_rather_than_zero(self):
        from PySide6.QtCore import Qt

        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.theta import ThetaRow
        from ui.models.theta_table_model import ThetaTableModel

        model = ThetaTableModel([ThetaRow.from_mapping({"symbol": "ABC"})])
        by_key = {key: column for column, (key, _label) in enumerate(model.COLUMNS)}
        for key in ("credit_pct_of_strike", "credit_pct_per_week", "spread_pct", "major_sma_above_strike"):
            assert model.data(model.index(0, by_key[key]), Qt.ItemDataRole.DisplayRole) == ""
