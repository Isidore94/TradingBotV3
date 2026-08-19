"""R7's deferred visuals, built 2026-08-18: per-group charts and the year heatmap.

Both were retained as future scope by the R7/R8 governance close-out. The
tests here are about honesty rather than pixels, because that is where a chart
lies most easily:

- a bucket whose total cannot be converted is EXCLUDED from the bar chart, not
  drawn as zero, and the exclusion is counted so the caller can say it;
- every bar carries its n, and a thin sample says so on its own label;
- a day with no trading is BLANK on the year heatmap, never a break-even
  colour, and the colour scale is centred on zero so a good year and a bad one
  are drawn on the same footing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from ui.panels.journal.analytics_tab import (  # noqa: E402
    GROUP_CHART_MAX_BARS,
    THIN_SAMPLE_TRADES,
    group_breakdown_rows,
    group_chart_series,
)
from ui.panels.journal.calendar_tab import year_heatmap_matrix  # noqa: E402


def _bucket(label, closed, net):
    return {"label": label, "trades": closed, "closed": closed, "net_pnl": net}


class TestTheGroupChart:
    def test_every_bar_carries_its_n(self):
        rows = [_bucket("bull flag", 12, 340.0), _bucket("gap fill", 7, -80.0)]
        labels, values, dropped = group_chart_series(rows)
        assert labels == ["bull flag (n=12)", "gap fill (n=7)"]
        assert values == [340.0, -80.0]
        assert dropped == 0

    def test_a_thin_sample_says_so(self):
        rows = [_bucket("new idea", THIN_SAMPLE_TRADES - 1, 900.0)]
        labels, _values, _dropped = group_chart_series(rows)
        assert labels[0].endswith("thin")

    def test_an_unconvertible_bucket_is_excluded_not_zeroed(self):
        """None means 'mixed currencies, unconverted' - a zero bar would claim
        the setup broke even."""
        rows = [_bucket("bull flag", 12, 340.0), _bucket("mixed", 4, None)]
        labels, values, dropped = group_chart_series(rows)
        assert len(labels) == 1
        assert values == [340.0]
        assert dropped == 1

    def test_the_bar_cap_counts_what_it_dropped(self):
        rows = [_bucket(f"s{index}", 6, float(index)) for index in range(GROUP_CHART_MAX_BARS + 5)]
        labels, _values, dropped = group_chart_series(rows)
        assert len(labels) == GROUP_CHART_MAX_BARS
        assert dropped == 5

    def test_no_rows_is_an_empty_chart_not_an_error(self):
        assert group_chart_series([]) == ([], [], 0)

    def test_the_breakdown_reads_the_summary_the_table_already_uses(self):
        summary = {"groups": {"account": [_bucket("TFSA", 3, 12.0)]}}
        assert group_breakdown_rows(summary, "account")[0]["label"] == "TFSA"
        assert group_breakdown_rows(summary, "nope") == []
        assert group_breakdown_rows({}, "account") == []


class TestTheYearHeatmap:
    def test_days_land_on_their_month_and_day(self):
        matrix, scale = year_heatmap_matrix(
            {"2026-03-05": 120.0, "2026-03-06": -50.0}, 2026
        )
        assert matrix[2][4] == 120.0
        assert matrix[2][5] == -50.0
        assert scale == 120.0

    def test_a_day_with_no_trading_is_blank_not_zero(self):
        matrix, _scale = year_heatmap_matrix({"2026-03-05": 120.0}, 2026)
        assert matrix[0][0] is None
        # And a real zero is a real value, distinct from blank.
        matrix, _scale = year_heatmap_matrix({"2026-01-02": 0.0}, 2026)
        assert matrix[0][1] == 0.0

    def test_the_scale_is_the_largest_absolute_day(self):
        """Centred on zero, so a bad day and a good one are drawn alike."""
        _matrix, scale = year_heatmap_matrix(
            {"2026-02-02": 10.0, "2026-02-03": -400.0}, 2026
        )
        assert scale == 400.0

    def test_other_years_and_unreadable_rows_are_ignored(self):
        matrix, scale = year_heatmap_matrix(
            {"2025-03-05": 900.0, "not-a-date": 1.0, "2026-03-05": "nope"}, 2026
        )
        assert scale == 0.0
        assert all(value is None for row in matrix for value in row)

    def test_an_empty_year_is_all_blank(self):
        matrix, scale = year_heatmap_matrix({}, 2026)
        assert scale == 0.0
        assert len(matrix) == 12 and all(len(row) == 31 for row in matrix)
