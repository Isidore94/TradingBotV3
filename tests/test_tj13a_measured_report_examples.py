"""TJ-13A item 6: the measured report's examples name three different things.

The published file, verbatim
(``\\\\MINI-PC\\Trading Bot Data\\ai_store\\digests\\measured_report_2026-09-17.md``)::

    **Biggest intraday moves after the observation** - retrospective,
    result-selected, 3 of 19045 measured
      - ABCL: 6.1619 percent
      - ABCL: 6.1619 percent
      - ABCL: 6.1619 percent

    **Worst intraday adverse moves after the observation** - retrospective,
    result-selected, 3 of 19045 measured
      - ERAS: -7.5875 percent
      - ERAS: -7.5875 percent
      - ERAS: -7.5875 percent

    **Biggest swing movement in R** - retrospective, result-selected,
    3 of 309 measured
      - d2c475597830f05c74743f79a29f6756: 4.4529 R
      - d2c475597830f05c74743f79a29f6756: 4.4529 R
      - d2c475597830f05c74743f79a29f6756: 4.4529 R

Nineteen thousand measured observations and the examples name ONE symbol, three
times, at the same number: ``measured_report._example_tables`` sorts and takes
``[:3]`` ROWS, and one symbol carries several observations in a session.

The fixtures below model that exactly - the repeated symbol appears three times
with the SAME value, because that is what the store holds - and the answer is
asserted as a number: the best three are ABCL 6.1619, BBIO 5.0, CCL 4.0. The
DENOMINATOR does not move: de-duplicating the examples must not shrink the
population the label is about.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import measured_report  # noqa: E402

SESSION = "2026-09-17"
NOW = datetime(2026, 9, 18, 2, 30)

#: (symbol, mfe_pct, mae_pct). ABCL and ERAS each carry three observations of
#: the SAME session at the SAME number, which is the live shape.
INTRADAY = [
    ("ABCL", 6.1619, -1.0),
    ("ABCL", 6.1619, -1.0),
    ("ABCL", 6.1619, -1.0),
    ("BBIO", 5.0, -1.1),
    ("CCL", 4.0, -1.2),
    ("DDOG", 3.0, -1.3),
    ("ERAS", 0.5, -7.5875),
    ("ERAS", 0.5, -7.5875),
    ("ERAS", 0.5, -7.5875),
    ("FFIV", 0.4, -6.0),
    ("GGG", 0.3, -5.0),
]

SWING_ID = "d2c475597830f05c74743f79a29f6756"
SWING = [
    (SWING_ID, 4.4529),
    (SWING_ID, 4.4529),
    (SWING_ID, 4.4529),
    ("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6", 3.2),
    ("b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7", 2.1),
]


def _intraday_csv(tmp_path: Path) -> Path:
    """Written and read back as a CSV, the way the slot really gets it."""
    path = tmp_path / "intraday_bounce_outcomes.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trade_date", "symbol", "side", "mfe_pct", "mae_pct"])
        for symbol, mfe, mae in INTRADAY:
            writer.writerow([SESSION, symbol, "LONG", mfe, mae])
    return path


class _Warehouse:
    """The `build_report` warehouse seam: two members and nothing else."""

    source_paths = ("research_lake/outcome_path",)

    def read_outcomes(self, session_date, *, now=None):
        return [
            {
                "occurrence_id": occurrence_id,
                "symbol": "SWG",
                "side": "long",
                "mfe_r": value,
                "outcome_definition_id": "house_default_v1",
            }
            for occurrence_id, value in SWING
        ]

    def read_entry_quality(self, session_date, *, now=None):
        return []


def _tables(tmp_path):
    report = measured_report.build_report(
        SESSION,
        now=NOW,
        sources=measured_report.ReportSources(intraday_outcomes=_intraday_csv(tmp_path)),
        warehouse=_Warehouse(),
    )
    return report, {table["table_id"]: table for table in report.example_tables}


def test_the_best_intraday_example_list_names_three_different_symbols(tmp_path):
    _report, tables = _tables(tmp_path)
    table = tables["day.best_moves"]
    names = [row["name"] for row in table["rows"]]

    assert names == ["ABCL", "BBIO", "CCL"]
    assert len(set(names)) == 3
    assert [row["value"] for row in table["rows"]] == [6.1619, 5.0, 4.0]
    # The population the label is about is untouched: 11 measured observations.
    assert table["denominator"] == 11


def test_the_worst_intraday_example_list_names_three_different_symbols(tmp_path):
    _report, tables = _tables(tmp_path)
    table = tables["day.worst_moves"]
    names = [row["name"] for row in table["rows"]]

    assert names == ["ERAS", "FFIV", "GGG"]
    assert len(set(names)) == 3
    assert [row["value"] for row in table["rows"]] == [-7.5875, -6.0, -5.0]
    assert table["denominator"] == 11


def test_the_swing_example_list_names_three_different_occurrences(tmp_path):
    """A swing row is keyed by its occurrence id and shown by its ticker.

    Trader 2026-09-23: the id is only a short trailing reference.
    """
    _report, tables = _tables(tmp_path)
    table = tables["swing.best_moves"]
    names = [row["name"] for row in table["rows"]]

    assert len(set(names)) == 3
    assert table["rows"][0]["occurrence_id"] == SWING_ID
    assert names[0].startswith("SWG LONG") and SWING_ID not in names[0]
    assert [row["value"] for row in table["rows"]] == [4.4529, 3.2, 2.1]
    assert table["denominator"] == 5


def test_a_short_example_list_is_short_rather_than_padded(tmp_path):
    """Two distinct symbols means two rows, never a third repeat to reach three.

    The de-duplication must not become a reason to re-admit a duplicate when
    fewer than three names exist.
    """
    path = tmp_path / "intraday_bounce_outcomes.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trade_date", "symbol", "side", "mfe_pct", "mae_pct"])
        for symbol, mfe in (("AAA", 2.0), ("AAA", 1.9), ("BBB", 1.5), ("AAA", 1.0)):
            writer.writerow([SESSION, symbol, "LONG", mfe, -0.5])

    report = measured_report.build_report(
        SESSION,
        now=NOW,
        sources=measured_report.ReportSources(intraday_outcomes=path),
        warehouse=None,
    )
    table = {t["table_id"]: t for t in report.example_tables}["day.best_moves"]

    assert [row["name"] for row in table["rows"]] == ["AAA", "BBB"]
    assert [row["value"] for row in table["rows"]] == [2.0, 1.5]
    assert table["denominator"] == 4
    assert "2 of 4 measured observation(s)" in table["note"]


def test_the_published_markdown_shows_three_different_names(tmp_path):
    """The file the trader opens, not only the payload behind it."""
    report, _tables_by_id = _tables(tmp_path)
    text = measured_report.render_markdown(report)

    block = text.split("## examples", 1)[1]
    best = block.split("**Biggest intraday moves after the observation**", 1)[1]
    best = best.split("**", 1)[0]
    assert best.count("ABCL:") == 1
    assert "BBIO:" in best and "CCL:" in best
