"""S11 exit-window truth: the pure numbers, the chunked reader, the night slot.

Eight hand-built events (F14). Every number the payload carries for the
`vwap|LONG` cell is checked against arithmetic done by hand in the comments.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import exit_windows as ew  # noqa: E402

COLUMNS = [
    "event_id", "event_type", "trade_date", "direction", "entry_price",
    "risk_per_share", "bars_elapsed", "minutes_elapsed", "close_r", "mfe_r",
    "target_1r_hit", "stop_hit", "eod_close",
]


def _event(event_id, date, side, steps, *, final=True, final_close="use"):
    """`steps` = [(minutes, close_r, mfe_r, t1, stop)]; the last step is the final row."""
    rows = [{
        "event_id": event_id, "event_type": "registered", "trade_date": date,
        "direction": side, "entry_price": "100", "risk_per_share": "1.0",
        "bars_elapsed": "0", "minutes_elapsed": "", "close_r": "", "mfe_r": "",
        "target_1r_hit": "False", "stop_hit": "False", "eod_close": "",
    }]
    for index, (minutes, close_r, mfe_r, t1, stop) in enumerate(steps):
        last = index == len(steps) - 1
        kinds = ["update", "final"] if (last and final) else ["update"]
        for kind in kinds:
            rows.append({
                "event_id": event_id, "event_type": kind, "trade_date": date,
                "direction": side, "entry_price": "100", "risk_per_share": "1.0",
                "bars_elapsed": str(minutes // 5), "minutes_elapsed": str(minutes),
                "close_r": ("" if (kind == "final" and final_close is None) else str(close_r)),
                "mfe_r": str(mfe_r), "target_1r_hit": str(bool(t1)), "stop_hit": str(bool(stop)),
                "eod_close": "101" if kind == "final" else "",
            })
    return rows


def eight_events() -> list[dict]:
    rows: list[dict] = []
    # E1: peaks at 60, +1R at 60, gives back to +0.2 at the close.
    rows += _event("AAA_long_20260921_10_00_00_vwap", "2026-09-21", "long", [
        (30, 0.5, 0.8, 0, 0), (60, 1.0, 1.2, 1, 0), (120, 0.4, 1.2, 1, 0), (180, 0.2, 1.2, 1, 0)])
    # E2: stopped at 60, peak 0.1 at 30.
    rows += _event("BBB_long_20260922_10_00_00_vwap", "2026-09-22", "long", [
        (30, -0.5, 0.1, 0, 0), (60, -1.1, 0.1, 0, 1), (90, -1.5, 0.1, 0, 1)])
    # E3: +1R at 30, stopped at 120, closes -1.2.
    rows += _event("CCC_long_20260922_10_00_00_vwap", "2026-09-22", "long", [
        (30, 1.1, 1.5, 1, 0), (60, 0.2, 1.5, 1, 0), (120, -1.0, 1.5, 1, 1), (180, -1.2, 1.5, 1, 1)])
    # E4: slow grinder, peaks at 180, never +1R, never stopped.
    rows += _event("DDD_long_20260923_10_00_00_vwap", "2026-09-23", "long", [
        (30, 0.3, 0.4, 0, 0), (60, 0.5, 0.6, 0, 0), (120, 0.9, 0.9, 0, 0), (180, 0.7, 0.95, 0, 0)])
    # E5: the one SHORT.
    rows += _event("EEE_short_20260923_10_00_00_vwap", "2026-09-23", "short", [
        (30, 0.2, 0.3, 0, 0), (60, 0.1, 0.3, 0, 0)])
    # E6: still open (no final row) - excluded.
    rows += _event("FFF_long_20260924_10_00_00_vwap", "2026-09-24", "long", [
        (30, 3.0, 3.0, 1, 0)], final=False)
    # E7: final row with a blank close R (unsettled) - excluded.
    rows += _event("GGG_long_20260924_10_00_00_vwap", "2026-09-24", "long", [
        (30, 2.0, 2.0, 1, 0), (60, 2.0, 2.0, 1, 0)], final_close=None)
    # E8: two types; +1R and stop on the SAME row (a loss), closes +0.5.
    rows += _event("HHH_long_20260924_10_00_00_vwap-ema_15", "2026-09-24", "long", [
        (30, 1.0, 1.0, 1, 1), (60, 0.5, 1.0, 1, 1)])
    return rows


def _cell(payload, key):
    return {cell["key"]: cell for cell in payload["cells"]}[key]


def test_every_number_of_the_vwap_long_cell():
    payload = ew.build_payload(eight_events(), as_of="2026-09-24", window=("2026-08-01", "2026-09-24"))
    cell = _cell(payload, "vwap|LONG")
    # E1, E2, E3, E4, E8 (E6 open and E7 unsettled are out).
    assert cell["n"] == 5
    assert cell["sessions"] == 4
    # Peaks: E1 60, E2 30, E3 30, E4 180, E8 30.
    assert cell["peak_within"] == {"30": 0.6, "60": 0.8, "120": 0.8}
    # MFE by 60: 1.2 + 0.1 + 1.5 + 0.6 + 1.0 = 4.4 / 5.
    assert cell["mfe_by"] == {"60": 0.88, "120": 0.94, "close": 0.95}
    # +1R printed: E1 (1.2 - 0.2), E3 (1.5 + 1.2), E8 (1.0 - 0.5).
    assert cell["hit_1r_n"] == 3
    assert cell["give_back_median"] == 1.0
    assert cell["give_back_mean"] == 1.4
    assert cell["hit_1r_closed_le_0"] == pytest.approx(1 / 3, abs=1e-4)
    assert cell["ev"] == {
        "hold_to_close": -0.42,   # 0.2 - 1 - 1 + 0.7 - 1
        "exit_60m": -0.06,        # 1.0 - 1 + 0.2 + 0.5 - 1
        "t1_or_60m": 0.1,         # 1 - 1 + 1 + 0.5 - 1
        "t1_or_120m": 0.18,       # 1 - 1 + 1 + 0.9 - 1
        "bracket_1to1": 0.14,     # 1 - 1 + 1 + 0.7 - 1
    }


def test_multi_type_short_and_all_cells():
    payload = ew.build_payload(eight_events(), as_of="2026-09-24", window=("2026-08-01", "2026-09-24"))
    ema = _cell(payload, "ema_15|LONG")
    assert ema["n"] == 1
    assert ema["ev"]["bracket_1to1"] == -1.0
    assert ema["give_back_median"] == 0.5
    short = _cell(payload, "vwap|SHORT")
    assert short["n"] == 1
    assert short["ev"] == {
        "hold_to_close": 0.1, "exit_60m": 0.1, "t1_or_60m": 0.1,
        "t1_or_120m": 0.1, "bracket_1to1": 0.1,
    }
    assert short["hit_1r_n"] == 0 and short["give_back_median"] is None
    # One alert counts once in the side total, whatever it carries.
    assert _cell(payload, "all|LONG")["n"] == 5
    assert payload["alerts"] == 6
    assert payload["schema"] == ew.SCHEMA


def test_formatting_lines():
    payload = ew.build_payload(eight_events(), as_of="2026-09-24", window=("2026-08-01", "2026-09-24"))
    cell = _cell(payload, "vwap|LONG")
    assert ew.tracker_text(cell) == "peak <= 60 min 80%; +1R/60m +0.10R vs hold -0.42R"
    assert ew.alert_line(cell) == (
        "Exit by: this family usually peaks inside 30 min; "
        "+1R or 60 min has beaten holding by 0.52R (n 5)"
    )
    worse = dict(cell, ev=dict(cell["ev"], t1_or_60m=-0.5))
    assert "holding has beaten +1R or 60 min by 0.08R" in ew.alert_line(worse)
    late = dict(cell, peak_within={"30": 0.1, "60": 0.2, "120": 0.4})
    assert "usually peaks after 120 min" in ew.alert_line(late)
    assert ew.tracker_text(None) == "" and ew.alert_line(None) == ""


def test_cell_for_alert_picks_the_largest_type():
    payload = ew.build_payload(eight_events(), as_of="2026-09-24", window=("2026-08-01", "2026-09-24"))
    lookup = ew.lookup(payload)
    assert ew.cell_for_alert(lookup, "ema_15-vwap", "LONG")["key"] == "vwap|LONG"
    assert ew.cell_for_alert(lookup, "ema_15", "long")["key"] == "ema_15|LONG"
    assert ew.cell_for_alert(lookup, "nothing", "LONG") is None


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS + ["context_json"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, context_json='{"a": 1}'))


def test_chunked_reader_matches_the_pure_build(tmp_path, monkeypatch):
    rows = eight_events()
    # One row outside the window must not count.
    rows += _event("ZZZ_long_20260701_10_00_00_vwap", "2026-07-01", "long", [(30, 1.0, 1.0, 1, 0)])
    path = tmp_path / "outcomes.csv"
    _write_csv(path, rows)
    monkeypatch.setattr(ew, "CHUNK_ROWS", 7)
    read = ew.read_window_rows(path, ("2026-08-01", "2026-09-24"))
    built = ew.build_payload(read, as_of="2026-09-24", window=("2026-08-01", "2026-09-24"))
    direct = ew.build_payload(eight_events(), as_of="2026-09-24", window=("2026-08-01", "2026-09-24"))
    assert built["cells"] == direct["cells"]


def test_night_slot_writes_and_a_failure_keeps_the_last_good_file(tmp_path):
    from ai_jobs import exit_windows_night

    src = tmp_path / "outcomes.csv"
    _write_csv(src, eight_events())
    out = tmp_path / "exit_windows.json"
    result = exit_windows_night.run_exit_windows(
        session_date="2026-09-24", outcomes_path=src, out_path=out,
        window=("2026-08-01", "2026-09-24"),
    )
    assert result["status"] == "ok", result
    good = out.read_text(encoding="utf-8")
    assert _cell(json.loads(good), "vwap|LONG")["n"] == 5

    # A broken read fails the slot and leaves the file byte-identical.
    bad = exit_windows_night.run_exit_windows(
        session_date="2026-09-25", outcomes_path=tmp_path / "missing.csv", out_path=out,
        window=("2026-08-01", "2026-09-25"),
    )
    assert bad["status"] == "failed"
    assert out.read_text(encoding="utf-8") == good

    # An empty window is not a failure but writes nothing either.
    empty = exit_windows_night.run_exit_windows(
        session_date="2026-09-25", outcomes_path=src, out_path=out,
        window=("2025-01-01", "2025-01-31"),
    )
    assert empty["status"] == "ok" and "kept" in empty["reason"]
    assert out.read_text(encoding="utf-8") == good


def test_read_payload_and_the_cache(tmp_path):
    out = tmp_path / "exit_windows.json"
    assert ew.read_payload(out) == {}
    payload = ew.build_payload(eight_events(), as_of="2026-09-24", window=("2026-08-01", "2026-09-24"))
    out.write_text(json.dumps(payload), encoding="utf-8")
    ew.reset_cache_for_tests()
    assert ew.cached_lookup() == {}
    assert ew.warm_cache(out) is True
    assert "vwap|LONG" in ew.cached_lookup()
    assert ew.warm_cache(out) is False
    ew.reset_cache_for_tests()


def test_path_constant_sits_beside_the_setup_grades():
    import project_paths as pp

    assert Path(pp.EXIT_WINDOWS_FILE).name == "exit_windows.json"
    assert Path(pp.EXIT_WINDOWS_FILE).parent == Path(pp.LOCAL_SETTINGS_DIR) / "working_lately"
