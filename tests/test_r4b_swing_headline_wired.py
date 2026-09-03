"""R4 Part B item B3 - `swing_headline` has production callers, and one Wilson.

V3 built `scripts/swing_headline.py` and then wired it to nothing: the module
existed, the docs said win rate led every trader-facing swing surface, and no
surface called it. R4 A11 gave it its first caller (the AWAY digest's ranking)
and B2 its second (the setup docs' record sentence). This wires the tables.

**One Wilson.** `swing_headline.WILSON_Z` (1.96, 95% two-sided) is the z for
every trader-facing win rate. `master_avwap_lib/expected_r.py` carries a
DIFFERENT z - 1.28, ~90% one-sided - and that one is a parameter of the
Expected-R model's own confident-win-rate term, not a column anybody reads. Two z
values on one screen is the failure; two z values in two different jobs, one of
them named and fenced, is not. This file asserts no trader-facing surface reaches
for the other one.

Offline and pure: rows in, numbers out.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------------------
# One Wilson
# ---------------------------------------------------------------------------


def test_every_trader_facing_win_rate_uses_the_same_z():
    """The surfaces wired by this packet all resolve to `swing_headline`'s z."""
    import swing_headline

    assert swing_headline.WILSON_Z == pytest.approx(1.959963984540054)

    offenders: dict[str, list[str]] = {}
    for name in (
        "scripts/ui/panels/weekend_prep_panel.py",
        "scripts/ui/panels/setup_tracker_panel.py",
        "scripts/ui/models/setup_table_model.py",
        "scripts/ui/panels/master_avwap_panel.py",
        "scripts/setup_docs.py",
        "scripts/autopilot_core.py",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        hits = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"expected_r\s+import\s+.*wilson", line)
            or re.search(r"wilson_lower_bound\s*\([^)]*z\s*=", line)
        ]
        if hits:
            offenders[name] = hits
    assert not offenders, offenders


def test_the_other_z_is_named_and_stays_where_it_is():
    """Not a stray literal: `expected_r`'s 1.28 is a declared model parameter."""
    from master_avwap_lib import expected_r

    assert expected_r.DEFAULT_PQS_CONFIG["wilson_z"] == 1.28


def test_a_rate_and_an_n_are_enough_to_build_a_headline():
    """Several stores keep a rate and a count rather than the rows behind them."""
    from swing_headline import headline_from_rate

    headline = headline_from_rate("cohort", win_rate=0.6, n=90)
    assert headline.n == 90
    assert headline.wins == 54
    assert headline.win_rate == pytest.approx(0.6)
    assert headline.win_rate_lb == pytest.approx(
        __import__("swing_headline").wilson_lower_bound(54, 90)
    )


def test_a_rate_without_an_n_is_no_headline_at_all():
    from swing_headline import headline_from_rate

    assert headline_from_rate("cohort", win_rate=1.0, n=0).win_rate is None
    assert headline_from_rate("cohort", win_rate=None, n=40).n == 0


# ---------------------------------------------------------------------------
# The Master AVWAP setups table
# ---------------------------------------------------------------------------


def test_the_setups_table_shows_the_familys_record_and_sorts_by_the_bound():
    pytest.importorskip("PySide6", reason="the setups table is a Qt model")
    from ui.models.setup import SetupRow
    from ui.models.setup_table_model import SORT_ROLE, SetupTableModel

    thin = SetupRow(symbol="AAA", side="LONG", raw={"setup_family": "thin_family"})
    thick = SetupRow(symbol="BBB", side="LONG", raw={"setup_family": "thick_family"})
    model = SetupTableModel([thin, thick])
    model.set_family_records(
        {
            # 100% on three and 62% on ninety - the exact pair the module exists
            # to order correctly.
            "thin_family": {"win_rate": 1.0, "win_rate_lb": 0.4385, "n": 3},
            "thick_family": {"win_rate": 0.62, "win_rate_lb": 0.5166, "n": 90},
        }
    )

    column = [key for key, _label in model.COLUMNS].index("family_win_rate")
    assert model.COLUMNS[column][1] == "Family Win %"

    def _cell(row, role):
        return model.data(model.index(row, column), role)

    from PySide6.QtCore import Qt

    assert "100%" in _cell(0, Qt.ItemDataRole.DisplayRole)
    assert "n=3" in _cell(0, Qt.ItemDataRole.DisplayRole)
    assert "62%" in _cell(1, Qt.ItemDataRole.DisplayRole)
    # SORTED BY THE BOUND, never the raw rate: the 62%-on-ninety must outrank
    # the 100%-on-three.
    assert _cell(1, SORT_ROLE) > _cell(0, SORT_ROLE)


def test_a_family_with_no_record_sorts_below_every_family_that_has_one():
    pytest.importorskip("PySide6", reason="the setups table is a Qt model")
    from PySide6.QtCore import Qt

    from ui.models.setup import SetupRow
    from ui.models.setup_table_model import SORT_ROLE, SetupTableModel

    model = SetupTableModel(
        [
            SetupRow(symbol="AAA", raw={"setup_family": "known"}),
            SetupRow(symbol="BBB", raw={"setup_family": "never_graded"}),
        ]
    )
    model.set_family_records({"known": {"win_rate": 0.4, "win_rate_lb": 0.2, "n": 20}})
    column = [key for key, _label in model.COLUMNS].index("family_win_rate")

    assert model.data(model.index(1, column), Qt.ItemDataRole.DisplayRole) == "-"
    assert model.data(model.index(1, column), SORT_ROLE) < model.data(
        model.index(0, column), SORT_ROLE
    )


# ---------------------------------------------------------------------------
# The Setup Tracker's recent-types tab
# ---------------------------------------------------------------------------


def test_the_recent_types_tab_leads_with_the_win_rate_and_ranks_by_the_bound():
    pytest.importorskip("PySide6", reason="the tracker is a Qt panel")
    from ui.panels.setup_tracker_panel import (
        RECENT_TYPE_COLUMNS,
        _rank_recent_types,
        _recent_type_headline_rows,
    )

    keys = [key for key, _label in RECENT_TYPE_COLUMNS]
    assert "win_rate_headline" in keys
    assert keys.index("win_rate_headline") < keys.index("avg_closed_r"), (
        "win rate leads; mean R stays beside it, never in front of it"
    )

    rows = _recent_type_headline_rows(
        [
            {"setup_family": "thin", "status": "NEW", "closed_setups": "3",
             "win_rate_closed": "1.0", "avg_closed_r": "2.0", "tracked_setups": "3"},
            {"setup_family": "thick", "status": "NEW", "closed_setups": "90",
             "win_rate_closed": "0.62", "avg_closed_r": "0.4", "tracked_setups": "90"},
        ]
    )
    by_family = {row["setup_family"]: row for row in rows}
    assert "n=3" in by_family["thin"]["win_rate_headline"]
    assert by_family["thick"]["win_rate_lb"] > by_family["thin"]["win_rate_lb"]

    ranked = _rank_recent_types(rows)
    assert [row["setup_family"] for row in ranked] == ["thick", "thin"]


# ---------------------------------------------------------------------------
# The Weekend Prep cohort tables
# ---------------------------------------------------------------------------


def test_a_cohort_row_carries_the_bound_beside_its_rate():
    pytest.importorskip("PySide6", reason="Weekend Prep is a Qt panel")
    from ui.panels.weekend_prep_panel import _cohort_headline_fields

    fields = _cohort_headline_fields({"win_rate": "0.62", "sample_count": "90"})
    assert "62" in fields["win_rate"]
    assert "n=90" in fields["win_rate"]
    assert fields["win_rate_lb"] == pytest.approx(0.5166, abs=0.01)


def test_a_cohort_with_no_sample_shows_a_dash_and_sorts_last():
    pytest.importorskip("PySide6", reason="Weekend Prep is a Qt panel")
    from ui.panels.weekend_prep_panel import _cohort_headline_fields, _rank_cohort_rows

    blank = _cohort_headline_fields({"win_rate": "", "sample_count": ""})
    assert blank["win_rate"] == ""
    assert blank["win_rate_lb"] is None

    ranked = _rank_cohort_rows(
        [
            {"cohort": "unmeasured", "win_rate_lb": None},
            {"cohort": "thin", "win_rate_lb": 0.44},
            {"cohort": "thick", "win_rate_lb": 0.52},
        ]
    )
    assert [row["cohort"] for row in ranked] == ["thick", "thin", "unmeasured"]
