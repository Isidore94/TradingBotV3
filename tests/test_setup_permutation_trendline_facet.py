"""S6 - the scan's own D1 trendline read on the scan row, and the `trendline` facet.

Shadow only. The directional refine already computes `trendline_break_recent` /
`trendline_within_alert_range` for priority rows; the scan now copies them (and the
line's break direction) onto `d1_features_history.csv` as appended `perm_` columns.
A row the refine never looked at, or one it looked at without the bars to say "no
line", is unknown. The scan golden proves every other column and the
detector/scoring output are unchanged with the hook on.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutation_backfill as bf  # noqa: E402
import setup_permutations as sp  # noqa: E402

BREAK, NEAR, DIRECTION = sp.TRENDLINE_COLUMNS


def _refined(broke=False, near=False, break_type="H-break", line_type="H-"):
    return {
        "trendline_break_recent": broke,
        "trendline_within_alert_range": near,
        "trendline_break_candidate": {"type": break_type} if broke else None,
        "trendline_candidate": {"type": line_type} if near else None,
    }


def _columns(row, frame_bars=250, last_close=100.0, atr=2.0):
    return sp.trendline_columns(row, frame_bars=frame_bars, last_close=last_close, atr=atr)


# ---------------------------------------------------------------------------
# the columns: copied from the refined priority row; unknown when the scan cannot say
# ---------------------------------------------------------------------------
def test_a_long_break_is_a_break_up():
    assert _columns(_refined(broke=True)) == {BREAK: True, NEAR: False, DIRECTION: "up"}


def test_a_short_break_is_a_break_down():
    assert _columns(_refined(broke=True, break_type="L-break")) == {BREAK: True, NEAR: False, DIRECTION: "down"}


def test_a_nearby_line_carries_its_direction():
    assert _columns(_refined(near=True)) == {BREAK: False, NEAR: True, DIRECTION: "up"}
    assert _columns(_refined(near=True, line_type="L+")) == {BREAK: False, NEAR: True, DIRECTION: "down"}


def test_a_break_and_a_nearby_line_keep_both_flags_and_the_break_direction():
    assert _columns(_refined(broke=True, near=True)) == {BREAK: True, NEAR: True, DIRECTION: "up"}


def test_no_line_with_the_full_lookback_is_no_line():
    assert _columns(_refined()) == {BREAK: False, NEAR: False, DIRECTION: None}


@pytest.mark.parametrize("row", [None, {}, {"symbol": "AAA"}, "junk"])
def test_a_row_the_refine_never_looked_at_is_unknown(row):
    assert _columns(row) == dict.fromkeys(sp.TRENDLINE_COLUMNS)


@pytest.mark.parametrize(("frame_bars", "last_close", "atr"), [
    (199, 100.0, 2.0), (0, 100.0, 2.0), (None, 100.0, 2.0),
    (250, None, 2.0), (250, 100.0, None), (250, 100.0, 0.0), (250, 100.0, float("nan")),
])
def test_no_line_without_the_bars_or_atr_to_look_is_unknown(frame_bars, last_close, atr):
    assert _columns(_refined(), frame_bars, last_close, atr) == dict.fromkeys(sp.TRENDLINE_COLUMNS)


def test_a_found_line_is_known_whatever_the_cached_bar_count():
    # The refine fetched its own bars when the cache was short; a found line is still a fact.
    assert _columns(_refined(broke=True), frame_bars=0)[BREAK] is True


@pytest.mark.parametrize("candidate", [None, {}, {"type": "junk"}, "H-break"])
def test_a_found_line_without_a_readable_type_is_unknown(candidate):
    row = {"trendline_break_recent": True, "trendline_within_alert_range": False,
           "trendline_break_candidate": candidate}
    assert _columns(row) == dict.fromkeys(sp.TRENDLINE_COLUMNS)


def test_the_known_bar_floor_is_the_scans_trendline_lookback():
    source = (SCRIPTS_DIR / "master_avwap_lib" / "legacy.py").read_text(encoding="utf-8")
    lookback = re.search(r"^PRIORITY_TRENDLINE_LOOKBACK_BARS = (\d+)$", source, re.M)
    assert lookback and int(lookback.group(1)) == sp.TRENDLINE_MIN_KNOWN_BARS


def test_the_columns_append_after_the_setup_age():
    # S15 appends its columns after the trendline set, then the regime columns.
    newest = (*sp.S15_COLUMNS, *sp.REGIME_COLUMNS)
    columns = sp.SCAN_ROW_COLUMNS[:-len(newest)]
    assert sp.SCAN_ROW_COLUMNS[-len(newest):] == newest
    assert columns[-len(sp.TRENDLINE_COLUMNS):] == sp.TRENDLINE_COLUMNS
    assert columns[-len(sp.TRENDLINE_COLUMNS) - 1] == sp.SETUP_AGE_COLUMN
    assert all(column.startswith("perm_") for column in sp.TRENDLINE_COLUMNS)


# ---------------------------------------------------------------------------
# the facet: value and unknown (CSV spellings included)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("broke", "near", "direction", "expected"), [
    (True, False, "up", "trendline_break_up"),
    ("True", "False", "down", "trendline_break_down"),
    (True, True, "up", "trendline_break_up"),
    (False, True, "up", "trendline_near_up"),
    ("False", "True", "down", "trendline_near_down"),
    (False, False, "", "no_trendline"),
    ("False", "False", None, "no_trendline"),
    (1.0, 0.0, "UP", "trendline_break_up"),
    ("", "", "", "unknown"),
    (None, None, None, "unknown"),
    (float("nan"), float("nan"), float("nan"), "unknown"),
    (True, None, "up", "unknown"),
    (True, False, "", "unknown"),
    (False, True, "sideways", "unknown"),
])
def test_trendline_facet(broke, near, direction, expected):
    key = sp.facets_for_row({"side": "LONG", "setup_family": "general", BREAK: broke, NEAR: near,
                             DIRECTION: direction})
    assert key.get("trendline") == expected


def test_no_trendline_stays_out_of_the_label_and_a_break_shows():
    base = {"side": "LONG", "setup_family": "general"}
    assert "no_trendline" not in sp.facets_for_row({**base, BREAK: False, NEAR: False}).label
    assert "trendline_break_up" in sp.facets_for_row({**base, BREAK: True, NEAR: False, DIRECTION: "up"}).label


def test_a_row_without_the_columns_keys_exactly_as_before():
    key = sp.facets_for_row({"side": "LONG", "setup_family": "general", "atr20": 2.0})
    assert key.get("trendline") == sp.UNKNOWN
    assert "trendline" not in key.compact_key
    assert key.permutation_rule_version == "setup_permutations.v1"


def test_the_facet_is_registered_in_its_own_group():
    assert sp.FACETS["trendline"].group == "trendline"


def test_the_backfill_writes_an_f_trendline_column():
    assert "f_trendline" in bf.output_columns()


# ---------------------------------------------------------------------------
# the scan golden: output unchanged, the trendline break on the row
# ---------------------------------------------------------------------------
def _parity_module():
    path = Path(__file__).with_name("test_setup_permutation_scan_parity.py")
    spec = importlib.util.spec_from_file_location("_s6_scan_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan_runs(tmp_path_factory):
    parity = _parity_module()
    base = tmp_path_factory.mktemp("s6-scan")
    # Lower highs, then a two-session pop through the falling line: the refine finds an H-break.
    return (parity, parity._run(base, "on", 300, bars="trendline_break"),
            parity._run(base, "off", 300, bars="trendline_break"))


def test_the_scan_found_the_break_it_carries(scan_runs):
    _parity, stamped, _plain = scan_runs
    assert stamped["priority_row"]["trendline_break_recent"] is True


def test_the_scan_writes_the_trendline_columns(scan_runs):
    _parity, stamped, _plain = scan_runs
    row = stamped["history"][-1]
    tail = len(sp.S15_COLUMNS) + len(sp.REGIME_COLUMNS)  # S15, then the regime, append after the trendline set
    assert list(row)[-len(sp.TRENDLINE_COLUMNS) - tail:-tail] == list(sp.TRENDLINE_COLUMNS)
    assert (row[BREAK], row[NEAR], row[DIRECTION]) == ("True", "False", "up")
    key = dict(part.split("=", 1) for part in row["permutation_key"].split("|")[3].split(";"))
    assert key["trendline"] == "trendline_break_up"


def test_the_scan_output_is_identical_with_and_without_the_trendline_columns(scan_runs):
    parity, stamped, plain = scan_runs
    assert stamped["priority_row"] and plain["priority_row"]
    assert parity._strip(stamped["priority_row"]) == parity._strip(plain["priority_row"])
    assert parity._strip(stamped["ai_state_entry"]) == parity._strip(plain["ai_state_entry"])
    assert len(stamped["history"]) == len(plain["history"])
    for on_row, off_row in zip(stamped["history"], plain["history"], strict=True):
        assert parity._strip(on_row) == parity._strip(off_row)
        assert list(on_row) == list(off_row)  # same header, same order
    assert all(plain["history"][-1][column] == "" for column in sp.TRENDLINE_COLUMNS)
