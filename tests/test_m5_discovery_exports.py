"""Packet M5.1 - the control and study populations reach a file a reader opens.

`build_control_discovery_rows` and `build_study_discovery_rows` have graded the
two shadow namespaces since B4. On the live tracker that is 401 control records
and 3,992 study records - **4,393 graded records whose only readers were two
test files.** The functions were never wrong; nothing ever called them outside a
`.txt` report the desk does not show.

This file pins the export that fixes it:

* it rides the SAME `export_setup_tracker_views` pass as every other stats CSV,
  so it cannot drift to its own clock or its own tracker snapshot;
* every row carries `n`, wins, losses, `win_rate` and its **Wilson lower bound
  through `swing_headline`** - the one z for every trader-facing win rate - with
  mean R beside it, never instead of it;
* the window is named in SESSIONS, and the all-history block is kept beside a
  `lately` block rather than replaced by it;
* a failure in either export is logged and never costs the tracker save, which
  is the rule the band-variant export already follows.

Nothing here scores, ranks, gates or alerts. The control namespace is setups the
scan REJECTED and the study namespace is ideas that have never been promoted;
both are evidence about the gate, and a row from either must never be mistaken
for a pick.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import legacy  # noqa: E402
from swing_headline import wilson_lower_bound  # noqa: E402


#: Every export path `export_setup_tracker_views` writes, champion first, with
#: the three shadow files this packet adds at the end.
EXPORT_FILES = (
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
NEW_SHADOW_FILES = (
    "CONTROL_DISCOVERY_STATS_FILE",
    "STUDY_DISCOVERY_STATS_FILE",
    "EXIT_FRAMEWORK_STATS_FILE",
)
CHAMPION_FILES = tuple(
    name
    for name in EXPORT_FILES
    if name not in {*NEW_SHADOW_FILES, "SETUP_BAND_VARIANT_STATS_FILE"}
)


def _redirect_exports(monkeypatch, tmp_path) -> None:
    for name in EXPORT_FILES:
        monkeypatch.setattr(legacy, name, tmp_path / f"{name.lower()}.csv", raising=False)


def _graded_record(
    symbol,
    family,
    closed_r,
    *,
    side="LONG",
    reason=None,
    anchor="2026-01-02",
    scan="2026-01-03",
):
    record = {
        "symbol": symbol,
        "side": side,
        "anchor_date": anchor,
        "scan_date": scan,
        "setup_family": family,
        "scenarios": {
            "s1": {
                "tradeable": True,
                "status": "TARGET_HIT" if closed_r > 0 else "STOPPED",
                "total_r": closed_r,
                "stop_reference_label": "LOWER_1" if side == "LONG" else "UPPER_1",
            }
        },
    }
    if reason is not None:
        record["is_control"] = True
        record["control_reason"] = reason
    return record


def _tracker_with_three_and_three():
    """Three control records and three study records, all graded.

    The control three are one family split 2 wins / 1 loss so the win rate is a
    real fraction rather than 0 or 1 - a Wilson bound on 100% is the one number
    a reader most needs to be non-trivial.
    """
    control = {
        "control:a": _graded_record("AAA", "post_earnings_52w_break", 1.4, reason="near_miss"),
        "control:b": _graded_record(
            "BBB", "post_earnings_52w_break", 0.9, reason="near_miss", anchor="2026-01-05"
        ),
        "control:c": _graded_record(
            "CCC", "post_earnings_52w_break", -0.8, reason="random", anchor="2026-01-09"
        ),
    }
    study = {
        "study:a": _graded_record("DDD", "hv_level_break", 2.1, anchor="2026-01-11"),
        "study:b": _graded_record("EEE", "hv_level_break", -1.0, anchor="2026-01-12"),
        "study:c": _graded_record("FFF", "compression_break", 0.6, anchor="2026-01-13"),
    }
    return {"setups": {}, "control_setups": control, "study_setups": study}


def _read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _family_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row.get("row_kind")) == "family"]


def _all_history(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row.get("window")) == "all"]


# ---------------------------------------------------------------------------
# The two files exist, in the tracker's own save pass
# ---------------------------------------------------------------------------


def test_the_save_pass_writes_both_discovery_csvs(tmp_path, monkeypatch):
    _redirect_exports(monkeypatch, tmp_path)

    legacy.export_setup_tracker_views(_tracker_with_three_and_three())

    assert legacy.CONTROL_DISCOVERY_STATS_FILE.exists()
    assert legacy.STUDY_DISCOVERY_STATS_FILE.exists()
    # The file the desk already had is a .txt report nothing renders; the CSV is
    # a different artifact and must not overwrite it.
    assert legacy.CONTROL_DISCOVERY_STATS_FILE.name.endswith(".csv")
    assert legacy.CONTROL_DISCOVERY_STATS_FILE != legacy.CONTROL_DISCOVERY_FILE


def test_the_control_export_carries_the_cohorts_and_the_families(tmp_path, monkeypatch):
    _redirect_exports(monkeypatch, tmp_path)
    legacy.export_setup_tracker_views(_tracker_with_three_and_three())

    rows = _all_history(_read_rows(legacy.CONTROL_DISCOVERY_STATS_FILE))
    cohorts = {str(row["cohort"]) for row in rows if row["row_kind"] == "cohort"}
    assert cohorts == {"promoted", "near_miss", "random"}

    families = _family_rows(rows)
    assert len(families) == 1
    row = families[0]
    assert row["setup_family"] == "post_earnings_52w_break"
    assert row["side"] == "LONG"
    assert int(row["n"]) == 3
    assert int(row["wins"]) == 2
    assert int(row["losses"]) == 1


def test_the_study_export_ranks_the_study_namespace(tmp_path, monkeypatch):
    _redirect_exports(monkeypatch, tmp_path)
    legacy.export_setup_tracker_views(_tracker_with_three_and_three())

    rows = _all_history(_read_rows(legacy.STUDY_DISCOVERY_STATS_FILE))
    families = {row["setup_family"]: row for row in _family_rows(rows)}
    assert set(families) == {"hv_level_break", "compression_break"}
    assert int(families["hv_level_break"]["n"]) == 2
    assert int(families["hv_level_break"]["wins"]) == 1
    assert int(families["compression_break"]["n"]) == 1
    # The study namespace has no promoted/near-miss/random split - it is one
    # population, and inventing a cohort column for it would read as one.
    assert not [row for row in rows if row["row_kind"] == "cohort"]


# ---------------------------------------------------------------------------
# Win rate leads, ONE Wilson, sessions not days
# ---------------------------------------------------------------------------


def test_every_row_carries_the_one_wilson_lower_bound(tmp_path, monkeypatch):
    _redirect_exports(monkeypatch, tmp_path)
    legacy.export_setup_tracker_views(_tracker_with_three_and_three())

    for path in (legacy.CONTROL_DISCOVERY_STATS_FILE, legacy.STUDY_DISCOVERY_STATS_FILE):
        for row in _read_rows(path):
            n = int(row["n"])
            if not n:
                assert row["win_rate"] == ""
                assert row["win_rate_lb"] == ""
                continue
            wins = int(row["wins"])
            assert int(row["losses"]) == n - wins
            assert float(row["win_rate"]) == pytest.approx(wins / n)
            # `swing_headline`'s z (1.96), never `expected_r`'s 1.28.
            assert float(row["win_rate_lb"]) == pytest.approx(wilson_lower_bound(wins, n))


def test_mean_r_stays_beside_the_win_rate(tmp_path, monkeypatch):
    _redirect_exports(monkeypatch, tmp_path)
    legacy.export_setup_tracker_views(_tracker_with_three_and_three())

    families = _family_rows(_all_history(_read_rows(legacy.CONTROL_DISCOVERY_STATS_FILE)))
    assert float(families[0]["avg_closed_r"]) == pytest.approx((1.4 + 0.9 - 0.8) / 3)


def test_the_window_is_stated_in_sessions_and_lately_is_a_second_block(
    tmp_path, monkeypatch
):
    _redirect_exports(monkeypatch, tmp_path)
    legacy.export_setup_tracker_views(_tracker_with_three_and_three())

    rows = _read_rows(legacy.CONTROL_DISCOVERY_STATS_FILE)
    windows = {str(row["window"]) for row in rows}
    assert windows == {"all", "lately"}, "the all-history block is kept, not replaced"

    from evidence_stats import LATELY_SESSIONS, lately_window

    first, last = lately_window()
    for row in rows:
        if row["window"] == "lately":
            assert int(row["window_sessions"]) == LATELY_SESSIONS
            assert row["window_start"] == first
            assert row["window_end"] == last
        else:
            # All history has no session count to state, and stating one would
            # be a claim the file cannot support.
            assert row["window_sessions"] == ""


def test_the_lately_block_excludes_records_scanned_before_the_window(
    tmp_path, monkeypatch
):
    """The whole point of the second block: it is a different population."""
    _redirect_exports(monkeypatch, tmp_path)
    from evidence_stats import lately_window

    _first, last = lately_window()
    tracker = _tracker_with_three_and_three()
    tracker["control_setups"]["control:inside"] = _graded_record(
        "ZZZ", "inside_the_window", 1.1, reason="random", anchor="2026-02-02", scan=last
    )
    legacy.export_setup_tracker_views(tracker)

    rows = _read_rows(legacy.CONTROL_DISCOVERY_STATS_FILE)
    lately_families = {
        row["setup_family"] for row in rows if row["window"] == "lately" and row["row_kind"] == "family"
    }
    assert lately_families == {"inside_the_window"}
    all_families = {
        row["setup_family"] for row in rows if row["window"] == "all" and row["row_kind"] == "family"
    }
    assert all_families == {"post_earnings_52w_break", "inside_the_window"}


# ---------------------------------------------------------------------------
# The evidence store never costs the thing it records
# ---------------------------------------------------------------------------


def _exploding_builder(*_args, **_kwargs):
    raise ValueError("a malformed setup dict reached the discovery export")


@pytest.mark.parametrize(
    "builder,expected_missing,phrase",
    [
        ("build_control_discovery_stats_rows", "CONTROL_DISCOVERY_STATS_FILE", "control discovery"),
        ("build_study_discovery_stats_rows", "STUDY_DISCOVERY_STATS_FILE", "study discovery"),
    ],
)
def test_a_raising_discovery_export_never_costs_the_other_files(
    tmp_path, monkeypatch, caplog, builder, expected_missing, phrase
):
    _redirect_exports(monkeypatch, tmp_path)
    monkeypatch.setattr(legacy, builder, _exploding_builder)

    with caplog.at_level(logging.WARNING):
        legacy.export_setup_tracker_views(_tracker_with_three_and_three())

    for name in CHAMPION_FILES:
        assert getattr(legacy, name).exists(), f"{name} was not written"
    assert legacy.SETUP_BAND_VARIANT_STATS_FILE.exists()
    # The other shadow exports are not taken down with it: one guard each.
    for name in NEW_SHADOW_FILES:
        if name == expected_missing:
            assert not getattr(legacy, name).exists()
        else:
            assert getattr(legacy, name).exists(), f"{name} rode on the failure of another"
    assert any(
        phrase in record.getMessage().lower() for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_an_empty_tracker_writes_a_header_and_no_invented_numbers(tmp_path, monkeypatch):
    """An absent population is an empty table, never a crash and never a lie."""
    _redirect_exports(monkeypatch, tmp_path)
    legacy.export_setup_tracker_views({"setups": {}, "control_setups": {}, "study_setups": {}})

    for path in (legacy.CONTROL_DISCOVERY_STATS_FILE, legacy.STUDY_DISCOVERY_STATS_FILE):
        assert path.exists()
        header = path.read_text(encoding="utf-8").splitlines()[0]
        assert "win_rate_lb" in header

    # The study namespace is one population, so an empty one has no rows at all.
    assert _read_rows(legacy.STUDY_DISCOVERY_STATS_FILE) == []
    # The control comparison always names its three cohorts - that is
    # `build_control_discovery_rows`' own shape and it is the honest one:
    # "promoted: nothing graded" is a fact, and an absent row would leave a
    # reader to guess whether the cohort exists. Every cell is BLANK, never 0.
    rows = _read_rows(legacy.CONTROL_DISCOVERY_STATS_FILE)
    assert {row["cohort"] for row in rows} == {"promoted", "near_miss", "random"}
    assert all(row["row_kind"] == "cohort" for row in rows)
    for row in rows:
        assert int(row["n"]) == 0
        assert row["win_rate"] == ""
        assert row["win_rate_lb"] == ""
