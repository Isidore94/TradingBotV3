"""Packet WS-EF1 - the exit-framework comparison, split by setup family.

The trader's question (2026-09-08) is "does taking profit at band 3 beat band 2
for the 1st-dev breakout?" and the tracker cannot answer it: the exit-framework
export pools every scan row by `(framework_family, exit_template_id, side,
priority_bucket)` and never by SETUP, so "full at band 3 loses" is a
whole-population answer. A 1st-dev breakout starts one band from its target; an
AVWAPE bounce starts two. The right exit is probably per setup.

This file is the RED half of the packet. It pins, before any fix exists:

* a SECOND export, `master_avwap_exit_framework_by_family.csv`, built by the
  SAME builder with the grouping key as a parameter, so the two files can never
  disagree on a rate;
* the reconciliation - for every (template, side, bucket) the family rows' `n`,
  `n_closed`, `wins`, `losses`, `n_expired_unmeasured` and
  `n_filtered_by_experiment` SUM to the pooled row;
* `unlabelled` for a setup with no family, never a dropped row;
* `population` (`champion` / `study` / `control`) read from the record's own
  `is_study` / `is_control` flag, never guessed from the family NAME;
* the pooled file byte-identical beside the new one, pinned by a golden
  generated from the OLD code on a fixture tracker;
* a raising by-family export costing neither the tracker save nor the pooled
  file;
* the Exit frameworks tab's ONE new control - a family picker whose first entry
  is `All setups (pooled)` and renders today's table unchanged, whose family
  views filter and rank by the SAME Wilson lower bound, print `below floor`
  when every row is under `evidence_stats.MIN_REPORTABLE_N` while still showing
  those rows, and are never truncated by the table's 300-row cap.

Shadow only: nothing here scores, ranks, gates, alerts or promotes a template.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import legacy  # noqa: E402


BY_FAMILY_FILENAME = "master_avwap_exit_framework_by_family.csv"
BY_FAMILY_CONSTANT = "EXIT_FRAMEWORK_BY_FAMILY_STATS_FILE"
POOLED_PICKER_LABEL = "All setups (pooled)"

TEMPLATE = "full_band2"
PE_FAMILY = "post_earnings_52w_break"
SMA_FAMILY = "sma_breakout_50"

#: The champion exports `export_setup_tracker_views` writes before the guarded
#: shadow block. Named here rather than imported so this file still redirects
#: everything when the packet adds a constant to the exporter.
CHAMPION_EXPORT_CONSTANTS = (
    "SETUP_SCENARIOS_FILE",
    "SETUP_DAILY_FILE",
    "SETUP_STATS_FILE",
    "SETUP_TYPE_STATS_FILE",
    "SETUP_TYPE_RECENT_STATS_FILE",
    "SETUP_PLAYBOOKS_FILE",
    "SETUP_SHORT_HORIZON_FILE",
    "SETUP_ATTRIBUTES_FILE",
    "SETUP_ATTRIBUTE_LEADERBOARD_FILE",
    "SETUP_BAND_VARIANT_STATS_FILE",
    "CONTROL_DISCOVERY_STATS_FILE",
    "STUDY_DISCOVERY_STATS_FILE",
    "EXIT_FRAMEWORK_STATS_FILE",
)


# ---------------------------------------------------------------------------
# Fixture tracker - one template, one side, one bucket, two families
# ---------------------------------------------------------------------------


def _scenario(*, status, total_r, filtered=False):
    scenario = {
        "stop_reference_label": "LOWER_1",
        "stop_source_type": "band",
        "exit_template_id": TEMPLATE,
        "exit_template_label": "Full at band2",
        "framework_family": "baseline",
        "framework_version": "baseline",
        "experimental": False,
        "tradeable": not filtered,
        "status": "FILTERED" if filtered else status,
        "total_r": total_r,
    }
    if filtered:
        # The experiment's own rule skipping a scenario - counted, never
        # dropped, and never confused with an expired record.
        scenario["cohort_filter_reason"] = "Filtered by experiment: skip SMA_50 stop"
    return scenario


def _setup(
    symbol,
    family,
    *,
    status="TARGET_HIT",
    total_r=1.0,
    expired=False,
    filtered=False,
    role=None,
):
    record = {
        "setup_id": f"{symbol}:2026-01-03",
        "symbol": symbol,
        "side": "LONG",
        "scan_date": "2026-01-03",
        "anchor_date": "2026-01-02",
        "priority_bucket": "favorite_setup",
        "scenarios": {"baseline": _scenario(status=status, total_r=total_r, filtered=filtered)},
    }
    if family is not None:
        record["setup_family"] = family
    if role == "study":
        # The flag `record_setup_tracker_snapshot` stamps on a study record
        # (legacy.py :13857) - this, not the family name, is what says study.
        record["is_study"] = True
        record["study_kind"] = str(family or "study")
        record["setup_id"] = f"study:{record['setup_id']}"
    elif role == "control":
        record["is_control"] = True
        record["control_reason"] = "random"
        record["setup_id"] = f"control:{record['setup_id']}"
    if expired:
        record["setup_status"] = legacy.SETUP_STATUS_EXPIRED_UNMEASURED
    return record


def _setups(*records):
    return {record["setup_id"]: record for record in records}


def _two_family_tracker():
    """Five graded setups in two families, plus one expired and one filtered.

    The true answers, computed by hand and confirmed against the pooled builder
    on the pre-packet code:

    * pooled (baseline / full_band2 / LONG / favorite_setup):
      n=5, n_closed=5, wins=3, losses=2, win_rate=0.6, avg_closed_r=0.32,
      n_expired_unmeasured=1, n_filtered_by_experiment=1;
    * `post_earnings_52w_break`: n=2, wins=1, losses=1, win_rate=0.5,
      avg_closed_r=0.3, expired=1, filtered=0;
    * `sma_breakout_50`: n=3, wins=2, losses=1, win_rate=2/3,
      avg_closed_r=1/3, expired=0, filtered=1.

    A split that re-used the pooled rate would print 0.6 on both family rows and
    fail here.
    """
    return _setups(
        _setup("AAA", PE_FAMILY, status="TARGET_HIT", total_r=1.6),
        _setup("BBB", PE_FAMILY, status="STOPPED", total_r=-1.0),
        _setup("CCC", SMA_FAMILY, status="TARGET_HIT", total_r=1.2),
        _setup("DDD", SMA_FAMILY, status="TARGET_HIT", total_r=0.8),
        _setup("EEE", SMA_FAMILY, status="STOPPED", total_r=-1.0),
        _setup("FFF", PE_FAMILY, status="OPEN", total_r=0.0, expired=True),
        _setup("GGG", SMA_FAMILY, status="OPEN", total_r=0.0, filtered=True),
    )


def _by_family(setups, **kwargs):
    return legacy.build_exit_framework_stats_rows(setups, by_family=True, **kwargs)


def _read_rows(path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _header(path) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.reader(handle))


# ---------------------------------------------------------------------------
# The rows: the split, and what it must reconcile to
# ---------------------------------------------------------------------------

SUM_COLUMNS = (
    "n",
    "n_closed",
    "wins",
    "losses",
    "n_expired_unmeasured",
    "n_filtered_by_experiment",
)


def test_two_setups_in_two_families_split_into_rows_that_sum_to_the_pooled_row():
    setups = _two_family_tracker()
    pooled = legacy.build_exit_framework_stats_rows(setups)
    assert len(pooled) == 1, "the fixture is one pooled group by construction"
    pooled_row = pooled[0]

    family_rows = _by_family(setups)
    by_family = {str(row["setup_family"]): row for row in family_rows}
    assert set(by_family) == {PE_FAMILY, SMA_FAMILY}

    for column in SUM_COLUMNS:
        assert sum(int(row[column]) for row in family_rows) == int(pooled_row[column]), column

    # The numbers themselves, not just the shape: a family row that re-used the
    # pooled rate would print 0.6 twice.
    assert by_family[PE_FAMILY]["n"] == 2
    assert by_family[PE_FAMILY]["n_closed"] == 2
    assert by_family[PE_FAMILY]["wins"] == 1
    assert by_family[PE_FAMILY]["win_rate"] == pytest.approx(0.5)
    assert by_family[PE_FAMILY]["avg_closed_r"] == pytest.approx(0.3)
    assert by_family[PE_FAMILY]["n_expired_unmeasured"] == 1
    assert by_family[PE_FAMILY]["n_filtered_by_experiment"] == 0

    assert by_family[SMA_FAMILY]["n"] == 3
    assert by_family[SMA_FAMILY]["wins"] == 2
    assert by_family[SMA_FAMILY]["win_rate"] == pytest.approx(2 / 3)
    assert by_family[SMA_FAMILY]["avg_closed_r"] == pytest.approx(1 / 3)
    assert by_family[SMA_FAMILY]["n_expired_unmeasured"] == 0
    assert by_family[SMA_FAMILY]["n_filtered_by_experiment"] == 1


def test_the_by_family_rows_keep_the_pooled_identity_and_lead_with_the_family():
    """Same columns, `setup_family` FIRST, `population` beside it.

    The identity of the pooled grain has to survive the split: a family row
    that could not say which template, side and bucket it describes cannot be
    read against the pooled row it came from.
    """
    rows = _by_family(_two_family_tracker())
    columns = tuple(legacy.EXIT_FRAMEWORK_BY_FAMILY_STATS_COLUMNS)
    assert columns[0] == "setup_family"
    assert "population" in columns
    assert (
        tuple(name for name in columns if name not in {"setup_family", "population"})
        == legacy.EXIT_FRAMEWORK_STATS_COLUMNS
    )
    for row in rows:
        assert set(row) == set(columns)
        assert row["framework_family"] == "baseline"
        assert row["exit_template_id"] == TEMPLATE
        assert row["side"] == "LONG"
        assert row["priority_bucket"] == "favorite_setup"


def test_the_pooled_call_is_the_default_and_carries_no_family_column():
    """The grouping key is a PARAMETER of the same builder, default pooled."""
    setups = _two_family_tracker()
    pooled = legacy.build_exit_framework_stats_rows(setups)
    assert tuple(pooled[0]) == legacy.EXIT_FRAMEWORK_STATS_COLUMNS
    assert "setup_family" not in pooled[0]
    assert pooled == legacy.build_exit_framework_stats_rows(setups, by_family=False)


def test_a_setup_with_no_family_lands_in_unlabelled_and_is_never_dropped():
    """An old tracker record has the key PRESENT and EMPTY; a newer one omits it.

    Both are the same thing to a reader and both must be COUNTED, because a
    dropped row would break the reconciliation silently - the pooled row would
    be bigger than the sum of its families and nothing on the page would say so.
    """
    setups = _setups(
        _setup("HHH", "", status="TARGET_HIT", total_r=1.1),
        _setup("III", None, status="STOPPED", total_r=-1.0),
        _setup("JJJ", PE_FAMILY, status="TARGET_HIT", total_r=1.4),
    )
    rows = _by_family(setups)
    by_family = {str(row["setup_family"]): row for row in rows}
    assert set(by_family) == {"unlabelled", PE_FAMILY}
    assert by_family["unlabelled"]["n"] == 2
    assert by_family["unlabelled"]["wins"] == 1
    pooled_row = legacy.build_exit_framework_stats_rows(setups)[0]
    assert sum(int(row["n"]) for row in rows) == int(pooled_row["n"]) == 3


def test_the_population_comes_from_the_record_never_from_the_family_name():
    """`champion` / `study` / `control` from `is_study` / `is_control`.

    The negative half is the point: a champion setup whose family is literally
    called `study_1stdev_breakout_probe` is still `champion`, and a study record
    whose family carries no such word is still `study`. A label inferred from
    the name would get both backwards.
    """
    setups = _setups(
        _setup("KKK", "study_1stdev_breakout_probe", status="TARGET_HIT", total_r=1.5),
        _setup("LLL", "1stdev_breakout", status="TARGET_HIT", total_r=1.2, role="study"),
        _setup("MMM", "control_pool", status="STOPPED", total_r=-1.0, role="control"),
    )
    by_family = {str(row["setup_family"]): row for row in _by_family(setups)}
    assert by_family["study_1stdev_breakout_probe"]["population"] == "champion"
    assert by_family["1stdev_breakout"]["population"] == "study"
    assert by_family["control_pool"]["population"] == "control"


def test_a_study_family_keeps_its_own_row_and_never_merges_into_the_champion():
    """Two records, one family NAME, two populations - never pooled into one.

    A study record and a champion record that happen to share a family name are
    different evidence; merging them would put an unpromoted idea's outcomes
    inside the champion's own record.
    """
    setups = _setups(
        _setup("NNN", "1stdev_breakout", status="TARGET_HIT", total_r=1.3),
        _setup("OOO", "1stdev_breakout", status="STOPPED", total_r=-1.0, role="study"),
    )
    rows = _by_family(setups)
    populations = sorted(str(row["population"]) for row in rows)
    assert populations == ["champion", "study"]
    assert all(int(row["n"]) == 1 for row in rows)


def test_an_empty_tracker_exports_no_by_family_rows():
    assert _by_family({}) == []


# ---------------------------------------------------------------------------
# The export: both files, the pooled one byte-identical
# ---------------------------------------------------------------------------

#: The pooled export for `_two_family_tracker()`, captured from the code as it
#: stood BEFORE this packet (branch `claude/wishlist-sweep-2026-09-12`,
#: 2026-09-12). Nothing in this packet may move a byte of it.
POOLED_GOLDEN = (
    b"framework_family,exit_template_id,exit_template_label,side,priority_bucket,"
    b"experimental,framework_version,n,n_closed,wins,losses,win_rate,win_rate_lb,"
    b"meets_n_floor,avg_closed_r,stop_out_rate,target_hit_rate,n_expired_unmeasured,"
    b"n_filtered_by_experiment,tracker_saved_at,tracker_saved_by\r\n"
    b"baseline,full_band2,Full at band2,LONG,favorite_setup,False,baseline,5,5,3,2,"
    b"0.6,0.23072428127601285,False,0.32,0.4,0.6,1,1,2026-09-11T18:30:00,tester\r\n"
)

GOLDEN_PAYLOAD_CLOCK = {"saved_at": "2026-09-11T18:30:00", "saved_by": "tester"}


def _redirect_exports(monkeypatch, tmp_path) -> None:
    for name in (*CHAMPION_EXPORT_CONSTANTS, BY_FAMILY_CONSTANT):
        monkeypatch.setattr(
            legacy, name, tmp_path / f"{name.lower()}.csv", raising=False
        )


def test_the_by_family_export_has_its_own_file_constant_beside_the_pooled_one():
    assert Path(getattr(legacy, BY_FAMILY_CONSTANT)).name == BY_FAMILY_FILENAME
    assert (
        Path(getattr(legacy, BY_FAMILY_CONSTANT)).parent
        == Path(legacy.EXIT_FRAMEWORK_STATS_FILE).parent
    ), "the two exports sit beside each other"


def test_the_save_pass_writes_both_files_and_the_pooled_one_is_byte_identical(
    tmp_path, monkeypatch
):
    _redirect_exports(monkeypatch, tmp_path)
    legacy.export_setup_tracker_views(
        {"setups": _two_family_tracker(), "control_setups": {}, "study_setups": {}, **GOLDEN_PAYLOAD_CLOCK}
    )

    pooled_path = Path(legacy.EXIT_FRAMEWORK_STATS_FILE)
    family_path = Path(getattr(legacy, BY_FAMILY_CONSTANT))
    assert pooled_path.exists()
    assert family_path.exists(), "the by-family export was not written"
    assert pooled_path.read_bytes() == POOLED_GOLDEN, "the pooled export moved"

    header = _header(family_path)
    assert header[0] == "setup_family"
    assert "population" in header
    # Same clock the other shadow exports carry, so the page can answer
    # "as of when?" from the export rather than a file mtime.
    assert header[-2:] == [legacy.TRACKER_SAVED_AT_COLUMN, legacy.TRACKER_SAVED_BY_COLUMN]

    rows = _read_rows(family_path)
    assert {row["setup_family"] for row in rows} == {PE_FAMILY, SMA_FAMILY}
    assert {row["population"] for row in rows} == {"champion"}
    assert sum(int(row["n"]) for row in rows) == 5


def test_the_save_pass_labels_the_study_and_control_namespaces_in_the_by_family_file(
    tmp_path, monkeypatch
):
    """The study and control records are IN the file, labelled.

    They live in their own namespaces precisely so no champion aggregate can
    see them; the by-family export is the first surface that reads their exit
    scenarios, and it must say which population each row came from.
    """
    _redirect_exports(monkeypatch, tmp_path)
    legacy.export_setup_tracker_views(
        {
            "setups": _setups(_setup("AAA", PE_FAMILY, status="TARGET_HIT", total_r=1.6)),
            "control_setups": _setups(
                _setup("CTL", "control_pool", status="STOPPED", total_r=-1.0, role="control")
            ),
            "study_setups": _setups(
                _setup("STD", "1stdev_breakout", status="TARGET_HIT", total_r=1.2, role="study")
            ),
            **GOLDEN_PAYLOAD_CLOCK,
        }
    )
    rows = _read_rows(getattr(legacy, BY_FAMILY_CONSTANT))
    population_by_family = {row["setup_family"]: row["population"] for row in rows}
    assert population_by_family == {
        PE_FAMILY: "champion",
        "control_pool": "control",
        "1stdev_breakout": "study",
    }
    # And the pooled file is STILL the champion's population only - the study
    # and control namespaces never reach it.
    assert Path(legacy.EXIT_FRAMEWORK_STATS_FILE).read_bytes().count(b"\r\n") == 2


def test_a_raising_by_family_export_costs_neither_the_save_nor_the_pooled_file(
    tmp_path, monkeypatch, caplog
):
    """R10 in this export: the evidence store never costs the thing it records.

    Only the BY-FAMILY call raises here - the pooled call goes through - so this
    is the real failure mode: a bug in the new grouping must leave the tracker
    save and the pooled file exactly as they were.
    """
    _redirect_exports(monkeypatch, tmp_path)
    real_builder = legacy.build_exit_framework_stats_rows
    calls: list[bool] = []

    def _only_the_family_call_raises(setups, *args, **kwargs):
        by_family = bool(kwargs.get("by_family") or (args and args[0]))
        calls.append(by_family)
        if by_family:
            raise ValueError("by-family grouping is broken in this run")
        return real_builder(setups, *args, **kwargs)

    monkeypatch.setattr(
        legacy, "build_exit_framework_stats_rows", _only_the_family_call_raises
    )

    with caplog.at_level(logging.WARNING):
        legacy.export_setup_tracker_views(
            {
                "setups": _two_family_tracker(),
                "control_setups": {},
                "study_setups": {},
                **GOLDEN_PAYLOAD_CLOCK,
            }
        )

    assert True in calls, (
        "the save pass never asked for the by-family grouping, so nothing raised "
        f"and the guard was never exercised (calls: {calls})"
    )
    assert False in calls, "the pooled export must still be built by the same pass"
    assert not Path(getattr(legacy, BY_FAMILY_CONSTANT)).exists()
    assert Path(legacy.SETUP_SCENARIOS_FILE).exists(), "the tracker save paid for it"
    assert Path(legacy.EXIT_FRAMEWORK_STATS_FILE).read_bytes() == POOLED_GOLDEN
    assert any(
        "family" in record.getMessage().lower() for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


# ---------------------------------------------------------------------------
# The Exit frameworks tab: ONE picker above the table
# ---------------------------------------------------------------------------


def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - headless install
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


POOLED_FIELDS = tuple(legacy.EXIT_FRAMEWORK_STATS_COLUMNS)
BY_FAMILY_FIELDS = ("setup_family", "population", *POOLED_FIELDS)

POOLED_CSV = "exit_framework.csv"
BY_FAMILY_CSV = "exit_framework_by_family.csv"


def _pooled_row(
    *,
    framework_family="baseline",
    template,
    side="LONG",
    bucket="favorite_setup",
    win_rate,
    win_rate_lb,
    avg_closed_r,
    n="40",
    n_closed="30",
):
    wins = str(round(float(win_rate) * int(n_closed)))
    return {
        "framework_family": framework_family,
        "exit_template_id": template,
        "exit_template_label": template.replace("_", " "),
        "side": side,
        "priority_bucket": bucket,
        "experimental": "True" if framework_family != "baseline" else "False",
        "framework_version": "baseline" if framework_family == "baseline" else "2026-04-14",
        "n": n,
        "n_closed": n_closed,
        "wins": wins,
        "losses": str(int(n_closed) - int(wins)),
        "win_rate": str(win_rate),
        "win_rate_lb": str(win_rate_lb),
        "meets_n_floor": "True" if int(n_closed) >= 30 else "False",
        "avg_closed_r": str(avg_closed_r),
        "stop_out_rate": "0.4",
        "target_hit_rate": "0.6",
        "n_expired_unmeasured": "0",
        "n_filtered_by_experiment": "0",
    }


#: Today's Exit frameworks table, on a fixture whose ranked order is written out
#: by hand below. `_rank_exit_frameworks` orders the (side, bucket) BLOCKS by
#: their best bound, puts the baseline framework above its comparison twin
#: inside a block, and orders templates by the bound. The SHORT block's 0.600
#: beats the LONG block's best 0.512, so it leads.
POOLED_ROWS = [
    _pooled_row(template="full_band2", win_rate="0.6", win_rate_lb="0.423", avg_closed_r="0.95"),
    _pooled_row(template="full_band3", win_rate="0.7", win_rate_lb="0.512", avg_closed_r="0.10"),
    _pooled_row(
        framework_family="comparison_apr2026",
        template="exp_full_band2_hard_stop_125r",
        win_rate="0.5",
        win_rate_lb="0.331",
        avg_closed_r="0.12",
    ),
    _pooled_row(
        template="full_band2",
        side="SHORT",
        bucket="near_favorite_zone",
        win_rate="0.75",
        win_rate_lb="0.600",
        avg_closed_r="0.20",
    ),
]

#: The pooled table, in order, as `(side, framework_family, exit_template_id)`.
POOLED_GOLDEN_ORDER = [
    ("SHORT", "baseline", "full_band2"),
    ("LONG", "baseline", "full_band3"),
    ("LONG", "baseline", "full_band2"),
    ("LONG", "comparison_apr2026", "exp_full_band2_hard_stop_125r"),
]


def _family_row(family, population, **kwargs):
    return {"setup_family": family, "population": population, **_pooled_row(**kwargs)}


#: One champion family with four baseline templates in ONE (side, bucket) block,
#: and one study family entirely under `MIN_REPORTABLE_N`.
#:
#: `full_band2` carries the HIGHEST mean R (0.95) and only the third-best bound,
#: so a table sorted by mean R would lead with it and fail the ordering pin.
BY_FAMILY_ROWS = [
    _family_row(PE_FAMILY, "champion", template="full_band2", win_rate="0.6",
                win_rate_lb="0.423", avg_closed_r="0.95", n_closed="30"),
    _family_row(PE_FAMILY, "champion", template="full_band3", win_rate="0.7",
                win_rate_lb="0.512", avg_closed_r="0.10", n_closed="33"),
    _family_row(PE_FAMILY, "champion", template="half_band2_trail_band1", win_rate="0.55",
                win_rate_lb="0.387", avg_closed_r="0.22", n_closed="31"),
    _family_row(PE_FAMILY, "champion", template="half_band2_band3_trail_band1",
                win_rate="0.65", win_rate_lb="0.455", avg_closed_r="0.18", n_closed="32"),
    _family_row("1stdev_breakout", "study", template="full_band2", win_rate="0.5",
                win_rate_lb="0.250", avg_closed_r="0.05", n="14", n_closed="12"),
    _family_row("1stdev_breakout", "study", template="full_band3", win_rate="0.5833",
                win_rate_lb="0.300", avg_closed_r="0.31", n="14", n_closed="12"),
]

#: `post_earnings_52w_break`, in order, by the bound alone - one block, one
#: framework, four templates.
PE_FAMILY_GOLDEN_ORDER = [
    "full_band3",
    "half_band2_band3_trail_band1",
    "full_band2",
    "half_band2_trail_band1",
]


def _write(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def panel_module(monkeypatch, tmp_path):
    if _qt_app() is None:  # pragma: no cover - headless install
        pytest.skip("PySide6 is not installed")
    from ui.panels import setup_tracker_panel

    setup_tracker_panel.clear_setup_tracker_csv_cache()
    monkeypatch.setattr(
        setup_tracker_panel, "EXIT_FRAMEWORK_STATS_FILE", tmp_path / POOLED_CSV
    )
    monkeypatch.setattr(
        setup_tracker_panel,
        BY_FAMILY_CONSTANT,
        tmp_path / BY_FAMILY_CSV,
        raising=False,
    )
    return setup_tracker_panel


def _drain() -> None:
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance()
    for _ in range(5):
        if application is not None:
            application.processEvents()


def _rendered_rows(model) -> list[dict]:
    from ui.models.tracker_table_model import ROW_ROLE

    return [
        model.data(model.index(row, 0), ROW_ROLE) for row in range(model.rowCount())
    ]


def _loaded_panel(panel_module, tmp_path, *, by_family_rows=None):
    from tests.conftest import refresh_setup_tracker

    _write(tmp_path / POOLED_CSV, POOLED_FIELDS, POOLED_ROWS)
    _write(
        tmp_path / BY_FAMILY_CSV,
        BY_FAMILY_FIELDS,
        BY_FAMILY_ROWS if by_family_rows is None else by_family_rows,
    )
    panel = panel_module.SetupTrackerPanel()
    refresh_setup_tracker(panel)
    _drain()
    return panel


def _picker(panel):
    combo = panel.exit_framework_family_combo
    return combo, [combo.itemText(index) for index in range(combo.count())]


def _pick(panel, family) -> None:
    combo, labels = _picker(panel)
    assert family in labels, f"{family!r} is not in the picker: {labels}"
    combo.setCurrentIndex(labels.index(family))
    _drain()


def test_the_picker_sits_above_the_table_with_the_pooled_view_first(
    panel_module, tmp_path
):
    panel = _loaded_panel(panel_module, tmp_path)
    try:
        _combo, labels = _picker(panel)
        assert labels[0] == POOLED_PICKER_LABEL
        # Then every family in the by-family export, sorted by NAME - never by
        # a result.
        assert labels[1:] == sorted({"1stdev_breakout", PE_FAMILY})
    finally:
        panel.deleteLater()


def test_the_first_entry_renders_todays_pooled_rows_unchanged(panel_module, tmp_path):
    """The golden: `All setups (pooled)` is the table as it ships today."""
    panel = _loaded_panel(panel_module, tmp_path)
    try:
        rendered = _rendered_rows(panel.exit_framework_model)
        assert [
            (row["side"], row["framework_family"], row["exit_template_id"])
            for row in rendered
        ] == POOLED_GOLDEN_ORDER
        # And it comes BACK unchanged after a family view - the picker filters a
        # presentation, it never consumes the pooled rows.
        _pick(panel, PE_FAMILY)
        _pick(panel, POOLED_PICKER_LABEL)
        assert [
            (row["side"], row["framework_family"], row["exit_template_id"])
            for row in _rendered_rows(panel.exit_framework_model)
        ] == POOLED_GOLDEN_ORDER
    finally:
        panel.deleteLater()


def test_picking_a_family_filters_to_it_and_ranks_by_the_wilson_bound(
    panel_module, tmp_path
):
    panel = _loaded_panel(panel_module, tmp_path)
    try:
        _pick(panel, PE_FAMILY)
        rendered = _rendered_rows(panel.exit_framework_model)
        assert {row["setup_family"] for row in rendered} == {PE_FAMILY}
        assert [row["exit_template_id"] for row in rendered] == PE_FAMILY_GOLDEN_ORDER
        # Mean R is beside the headline, never the sort: `full_band2` has the
        # highest Avg R in this family and is third.
        assert rendered[0]["exit_template_id"] != "full_band2"
    finally:
        panel.deleteLater()


def test_a_family_view_names_the_family_and_its_n_closed(panel_module, tmp_path):
    panel = _loaded_panel(panel_module, tmp_path)
    try:
        _pick(panel, PE_FAMILY)
        sentence = panel.exit_framework_status_label.text()
        assert PE_FAMILY in sentence
        assert "33" in sentence, sentence
        # The four templates measure the SAME setups, so their n_closed is never
        # summed - 30+33+31+32 is a claim about 126 setups that do not exist.
        assert "126" not in sentence, sentence
    finally:
        panel.deleteLater()


def test_a_below_floor_family_says_below_floor_and_still_shows_its_rows(
    panel_module, tmp_path
):
    """`MIN_REPORTABLE_N` is 30 and every `1stdev_breakout` row closed 12.

    Below the floor the rows are still SHOWN - hiding them is how a study never
    gets looked at - and the sentence says the number is not yet reportable.
    """
    from evidence_stats import MIN_REPORTABLE_N

    panel = _loaded_panel(panel_module, tmp_path)
    try:
        _pick(panel, "1stdev_breakout")
        rendered = _rendered_rows(panel.exit_framework_model)
        assert len(rendered) == 2, "a below-floor family had its rows hidden"
        assert all(int(row["n_closed"]) < MIN_REPORTABLE_N for row in rendered)
        sentence = panel.exit_framework_status_label.text()
        assert "below floor" in sentence.lower(), sentence
        assert "1stdev_breakout" in sentence
        assert "12" in sentence, sentence
    finally:
        panel.deleteLater()


def test_one_familys_view_is_never_truncated_by_the_three_hundred_row_cap(
    panel_module, tmp_path
):
    """The cap exists for the pooled table; a family view must filter FIRST.

    Sixty families of six rows is 360 rows, and `zzz_last_family` carries the
    six LOWEST bounds, so every one of its rows sits past position 300 in the
    pooled ordering. A view that filtered after the cap would show zero.
    """
    rows = []
    for index in range(59):
        for position, template in enumerate(
            ("full_band2", "full_band3", "half_band2_trail_band1",
             "half_band2_band3_trail_band1", "exp_a", "exp_b")
        ):
            bound = 0.900 - (index * 6 + position) * 0.001
            rows.append(
                _family_row(
                    f"family_{index:02d}",
                    "champion",
                    template=template,
                    win_rate="0.6",
                    win_rate_lb=f"{bound:.3f}",
                    avg_closed_r="0.20",
                )
            )
    for position, template in enumerate(
        ("full_band2", "full_band3", "half_band2_trail_band1",
         "half_band2_band3_trail_band1", "exp_a", "exp_b")
    ):
        rows.append(
            _family_row(
                "zzz_last_family",
                "champion",
                template=template,
                win_rate="0.6",
                win_rate_lb=f"{0.001 * (position + 1):.3f}",
                avg_closed_r="0.20",
            )
        )
    assert len(rows) == 360

    panel = _loaded_panel(panel_module, tmp_path, by_family_rows=rows)
    try:
        _pick(panel, "zzz_last_family")
        rendered = _rendered_rows(panel.exit_framework_model)
        assert len(rendered) == 6
        assert {row["setup_family"] for row in rendered} == {"zzz_last_family"}
    finally:
        panel.deleteLater()


def test_the_tab_reads_the_file_the_exporter_writes():
    from ui.panels import setup_tracker_panel

    assert (
        Path(getattr(setup_tracker_panel, BY_FAMILY_CONSTANT)).name
        == Path(getattr(legacy, BY_FAMILY_CONSTANT)).name
        == BY_FAMILY_FILENAME
    )


def test_there_is_still_exactly_one_scenario_walker():
    """`test_band_variant_fence_guard.py`'s rule, restated for this packet.

    The split is a KEY change, not a new reader: every row it groups still comes
    from `_flatten_tracker_scenarios`, which is where the band-variant fence
    lives. A second walk of `setup["scenarios"]` written for the by-family
    export would be an eighth unfenced reader.
    """
    import ast

    source = (SCRIPTS_DIR / "master_avwap_lib" / "legacy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    builder = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_exit_framework_stats_rows"
    )
    walkers = [
        node.func.id
        for node in ast.walk(builder)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert walkers.count("_flatten_tracker_scenarios") >= 1
    assert "scenarios" not in {
        node.value
        for node in ast.walk(builder)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }, "the by-family builder walks setup['scenarios'] itself"
