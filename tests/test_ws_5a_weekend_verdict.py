"""WS-5A - the weekend verdict card reads NUMBERS, not formatted strings.

WISHLIST item 5, block A. Reproduced by Codex 2026-09-09 and re-verified by
recon 2026-09-12: `weekend_verdict.best_cohort_line` read `avg_r_h3` and
`n_h3`, and the panel readers that feed it published
`cohort / side / horizon / n / avg_return`, where `avg_return` was already a
FORMATTED PERCENT STRING (`+1.23%`). No row on either side of the seam has ever
carried an `avg_r_*` key, so every cell was skipped and BOTH cohort lines of the
card printed "nothing with enough behind it yet" on a desk holding 115 graded
veto rows and 129 graded like rows. Had a row matched, a PERCENT return would
have been printed with an `R` after it.

These tests drive the REAL readers over a CSV written with the live header
(`C:\\TradingBotData\\data\\runtime\\{veto,like}_cohort_performance.csv`,
read read-only 2026-09-12), so a contract change on either side of the seam
fails here rather than on a Saturday.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

#: The live header, byte-for-byte, of both cohort rollups (2026-09-12).
LIVE_HEADER = (
    "cohort,side,horizon_sessions,sample_count,win_rate,avg_side_return,"
    "profit_factor,updated_at,median_return,trimmed_mean_return,p10_return,"
    "p90_return,symbols,sessions,top_symbol_share,ci_low,ci_high,ci_basis,"
    "evidence_label,meets_n_floor"
)

_STAMP = "2026-09-12T04:10:00-04:00"


def _row(
    cohort: str,
    side: str,
    horizon: str,
    n: str,
    avg: str,
    *,
    win_rate: str = "0.5200",
    profit_factor: str = "1.3100",
    median: str = "0.004000",
    trimmed: str = "0.005000",
    meets: str = "1",
) -> str:
    """One rollup row in the live column order.

    Old rows have every key PRESENT and EMPTY rather than absent, which is why
    the unmeasured row below carries empty cells instead of missing ones.
    """
    return ",".join(
        [
            cohort, side, horizon, n, win_rate, avg, profit_factor, _STAMP,
            median, trimmed, "-0.010000", "0.012000", "9", "14", "0.2100",
            "-0.002000", "0.011000", "block bootstrap on the mean",
            "graded", meets,
        ]
    )


#: Both sides, the POOLED side, a mature and a thin cell, positive and negative
#: returns on both sides, another horizon, and a row whose return was never
#: measured. `ALL` is the pooled value the live CSV writes.
LIKE_ROWS = [
    # The pooled row is the biggest n AND the best return at h3. It must never
    # lead a ranking of reasons: it is both sides added together.
    _row("human_focus_like", "ALL", "3", "94", "0.031000"),
    _row("human_focus_like", "LONG", "3", "52", "-0.002908"),
    _row("human_focus_like", "SHORT", "3", "42", "0.007306"),
    # Best return at h3 on a real side, and the row the like line must name.
    _row("like_avwap_reclaim", "LONG", "3", "21", "0.019011", win_rate="0.8095",
         profit_factor="6.5473"),
    # Thin: a flattering number on four observations, never ranked.
    _row("like_lucky_streak", "SHORT", "3", "4", "0.250000", meets="0"),
    # A better return at a DIFFERENT horizon. The card reads h3 only.
    _row("like_ten_session_star", "LONG", "10", "77", "0.099000"),
    # Present and EMPTY: measured n, unmeasured return.
    _row("like_unmeasured", "LONG", "3", "31", "", win_rate="", profit_factor="",
         median="", trimmed=""),
]

VETO_ROWS = [
    _row("human_focus_veto", "ALL", "3", "309", "0.044000"),
    _row("human_focus_veto", "LONG", "3", "163", "-0.011103"),
    _row("human_focus_veto", "SHORT", "3", "146", "0.007035"),
    # The HIGHEST side-adjusted return among the rejected names: the reason
    # most worth another look, and the row the veto line must name.
    _row("veto_v2_compressed", "SHORT", "3", "46", "0.020648"),
    # The most NEGATIVE. The old card named this one ("weakest veto reason"),
    # which is the rejection that was RIGHT - the opposite of a second look.
    _row("veto_v1_too_extended_from_base", "LONG", "3", "65", "-0.014063"),
    _row("veto_thin_sample", "LONG", "3", "3", "0.310000", meets="0"),
]


def _write(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([LIVE_HEADER, *rows]) + "\n", encoding="utf-8")


@pytest.fixture
def cohort_files(tmp_path, monkeypatch):
    """Both rollups, at their NAMED constants, in a sandbox."""
    import project_paths

    runtime = tmp_path / "data" / "runtime"
    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        project_paths, "VETO_COHORT_PERFORMANCE_FILE",
        runtime / "veto_cohort_performance.csv",
    )
    monkeypatch.setattr(
        project_paths, "LIKE_COHORT_PERFORMANCE_FILE",
        runtime / "like_cohort_performance.csv",
    )
    _write(runtime / "veto_cohort_performance.csv", VETO_ROWS)
    _write(runtime / "like_cohort_performance.csv", LIKE_ROWS)
    return runtime


def _card(cohort_files):
    """The ACTUAL readers, through the ACTUAL card builder."""
    import weekend_verdict
    from ui.panels.weekend_prep_panel import _read_like_cohort, _read_veto_cohort

    return weekend_verdict.build_verdict(
        like_rows=_read_like_cohort(),
        veto_rows=_read_veto_cohort(),
    )


def _line(card, key):
    return next(item for item in card.lines if item.key == key)


def _cohort_lines(card):
    keys = {line.key for line in card.lines}
    like = _line(card, "best_like")
    veto_key = "veto_second_look" if "veto_second_look" in keys else "worst_veto"
    return like, _line(card, veto_key)


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------


def test_both_cohort_lines_name_a_cohort_off_the_real_readers(cohort_files):
    """The whole defect in one assertion: 13 graded rows, two empty lines."""
    like, veto = _cohort_lines(_card(cohort_files))

    assert like.measured, like.text
    assert veto.measured, veto.text
    assert "nothing with enough behind it yet" not in like.text
    assert "nothing with enough behind it yet" not in veto.text
    assert like.n and veto.n


def test_the_like_line_names_the_best_side_row_at_the_card_horizon(cohort_files):
    """`like_avwap_reclaim LONG` at h3, not the h10 star and not the pooled row."""
    like, _veto = _cohort_lines(_card(cohort_files))

    assert "like_avwap_reclaim" in like.text, like.text
    assert like.n == 21
    assert "+1.90%" in like.text, like.text
    assert "over 3 sessions" in like.text, like.text
    assert "like_ten_session_star" not in like.text


def test_the_rejection_line_names_the_highest_side_adjusted_return(cohort_files):
    """The reasons worth questioning are the ones that WORKED after the veto.

    `min(return)` named the rejection that was right, which is the one reading
    a trader never has to act on.
    """
    _like, veto = _cohort_lines(_card(cohort_files))

    assert "veto_v2_compressed" in veto.text, veto.text
    assert veto.n == 46
    assert "+2.06%" in veto.text, veto.text
    assert "side-adjusted" in veto.text, veto.text
    assert "veto_v1_too_extended_from_base" not in veto.text


def test_the_pooled_side_never_leads_either_line(cohort_files):
    """`ALL` is both sides added together - never a reason to act on."""
    like, veto = _cohort_lines(_card(cohort_files))

    for line in (like, veto):
        assert line.measured, line.text
        assert " ALL" not in line.text, line.text
        assert "+3.10%" not in line.text, line.text
        assert "+4.40%" not in line.text, line.text
    assert like.n != 94 and veto.n != 309


def test_a_percent_is_never_printed_as_R(cohort_files):
    """A side-adjusted percent return is not an R multiple."""
    like, veto = _cohort_lines(_card(cohort_files))

    for line in (like, veto):
        rendered = line.rendered()
        assert "%" in rendered, rendered
        assert not re.search(r"[\d.]\s*R\b", rendered), rendered


def test_the_thin_sentence_names_the_best_n_and_the_floor(tmp_path, monkeypatch):
    """Under the floor is not a weak finding; it is not a finding."""
    import project_paths
    import weekend_verdict
    from ui.panels.weekend_prep_panel import _read_like_cohort

    runtime = tmp_path / "data" / "runtime"
    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        project_paths, "LIKE_COHORT_PERFORMANCE_FILE",
        runtime / "like_cohort_performance.csv",
    )
    _write(
        runtime / "like_cohort_performance.csv",
        [
            _row("like_thin_a", "LONG", "3", "4", "0.090000", meets="0"),
            _row("like_thin_b", "SHORT", "3", "2", "0.120000", meets="0"),
        ],
    )

    line = _line(
        weekend_verdict.build_verdict(like_rows=_read_like_cohort()), "best_like"
    )

    assert line.measured is False
    assert "nothing with enough behind it yet" in line.text
    assert f"floor of {weekend_verdict.MIN_COHORT_N}" in line.text, line.text
    assert "best n was 4" in line.text, line.text


def test_no_rows_at_all_says_no_cohorts_measured_yet():
    """"Never graded" and "graded and thin" are different absences."""
    import weekend_verdict

    card = weekend_verdict.build_verdict()
    like, veto = _cohort_lines(card)

    assert "no like cohorts measured yet" in like.text.lower(), like.text
    assert "no veto cohorts measured yet" in veto.text.lower(), veto.text
    assert like.measured is False and veto.measured is False


# ---------------------------------------------------------------------------
# The typed contract, and the display edge that still prints the same cell
# ---------------------------------------------------------------------------


def test_the_readers_publish_numbers_not_formatted_strings(cohort_files):
    """A formatted string is what let a percent be read as an R multiple."""
    from ui.panels.weekend_prep_panel import _read_like_cohort, _read_veto_cohort

    for rows in (_read_like_cohort(), _read_veto_cohort()):
        assert rows
        for row in rows:
            assert isinstance(row["horizon_sessions"], int), row
            assert isinstance(row["n"], int), row
            value = row["avg_side_return_pct"]
            assert value is None or isinstance(value, float), row
            assert not isinstance(value, str), row

    row = next(
        item for item in _read_like_cohort() if item["cohort"] == "like_avwap_reclaim"
    )
    assert row["horizon_sessions"] == 3
    assert row["n"] == 21
    assert row["avg_side_return_pct"] == pytest.approx(1.9011)


def test_an_unmeasured_return_is_none_never_a_substituted_zero(cohort_files):
    """A blank is an absent measurement; a zero is a claim."""
    from ui.panels.weekend_prep_panel import _read_like_cohort

    row = next(
        item for item in _read_like_cohort() if item["cohort"] == "like_unmeasured"
    )
    assert row["avg_side_return_pct"] is None
    assert row["n"] == 31


def test_an_unmeasured_row_is_never_the_headline(cohort_files):
    """A `None` return must not sort as a zero and must not be named."""
    like, _veto = _cohort_lines(_card(cohort_files))
    assert like.measured, like.text
    assert "like_unmeasured" not in like.text


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def test_the_table_cell_text_is_unchanged(qapp, cohort_files):
    """The trader's two tables read exactly as they did: `+1.90%`, `21`."""
    from PySide6.QtWidgets import QTableWidget

    from ui.panels.weekend_prep_panel import (
        COHORT_TABLE_COLUMNS,
        _cohort_view,
        _fill_cohort_table,
        _read_like_cohort,
    )

    table = QTableWidget(0, len(COHORT_TABLE_COLUMNS))
    _fill_cohort_table(table, _cohort_view(_read_like_cohort(), "3"))

    avg_column = COHORT_TABLE_COLUMNS.index("avg_return")
    n_column = COHORT_TABLE_COLUMNS.index("n")
    cells = {
        table.item(index, 0).text(): (
            table.item(index, n_column).text(),
            table.item(index, avg_column).text(),
        )
        for index in range(table.rowCount())
    }

    assert cells["like_avwap_reclaim"] == ("21", "+1.90%")
    assert cells["human_focus_like"][1] in {"+3.10%", "-0.29%", "+0.73%"}
    assert cells["like_unmeasured"] == ("31", "")
    assert "like_ten_session_star" not in cells
