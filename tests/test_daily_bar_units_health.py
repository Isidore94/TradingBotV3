"""R10.V step 6 - the nightly unit measurement and the tile that reads it.

Two design decisions the tests pin, because both are the difference between a
signal and a permanent alarm:

* **The tile reads; it never measures.** The measurement takes ~7 s over 1,958
  files, so it rides the nightly evidence-snapshot job. A tile a human waits on
  is a tile nobody opens.
* **`lots_rth` degrades; `unknown` does not.** A round-lot row means something
  got past a write seam that refuses IB volume - the splice starting again. An
  `unknown` row is the known residue Yahoo cannot supply, named in the backfill
  manifest and unclearable by anyone. An alarm nobody can clear is an alarm
  people learn to ignore.

The cliff count is reported and never sets the status: after the backfill, 19
all-`yahoo` files still step >20x because a 20x volume step is a real market
event (DJT at its listing, OKLO's de-SPAC).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap  # noqa: E402
import operations_audit as audit  # noqa: E402
from ops import daily_bar_cliff as cliff  # noqa: E402

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def _frame(days: int = 30, *, unit: str = "shares", splice_at: int | None = None):
    stamps = pd.bdate_range("2026-06-01", periods=days)
    volumes = []
    for index in range(days):
        volume = 20_000_000.0 + index
        if splice_at is not None and index >= splice_at:
            volume /= 100.0
        volumes.append(volume)
    return pd.DataFrame(
        {
            "datetime": stamps,
            "open": [10.0] * days,
            "high": [10.5] * days,
            "low": [9.5] * days,
            "close": [10.2] * days,
            "volume": volumes,
            "source": ["yahoo" if unit == "shares" else "ibkr"] * days,
            "volume_unit": [unit] * days,
        }
    )


def _write(store: Path, symbol: str, frame, *, v2: bool = True) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"{symbol}.parquet"
    if v2:
        master_avwap._write_daily_bar_parquet(path, frame)
    else:
        frame.drop(columns=["source", "volume_unit"]).to_parquet(path, index=False)
    return path


def _health_file(tmp_path: Path, payload: dict) -> Path:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / cliff.HEALTH_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    return diagnostics


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------
def test_the_measurement_counts_rows_by_unit_and_files_by_schema(tmp_path):
    store = tmp_path / "store"
    _write(store, "AAA", _frame(30))
    _write(store, "OLD", _frame(30), v2=False)
    payload = cliff.measure_store_health(store)
    assert payload["files"] == 2
    assert payload["rows_by_volume_unit"]["shares"] == 30
    assert payload["files_by_schema"] == {"v2": 1, "v1": 1}
    assert payload["files_not_all_shares"] == 1, "the v1 file has no unit column at all"


def test_a_file_with_no_unit_column_counts_as_not_all_shares(tmp_path):
    store = tmp_path / "store"
    _write(store, "OLD", _frame(30), v2=False)
    payload = cliff.measure_store_health(store)
    assert payload["files_not_all_shares"] == 1
    assert payload["rows_by_volume_unit"].get("shares", 0) == 0


def test_the_measurement_carries_the_cliff_scan(tmp_path):
    store = tmp_path / "store"
    _write(store, "SPLICED", _frame(40, splice_at=20))
    payload = cliff.measure_store_health(store)
    assert payload["cliff"]["cliffed"] == 1
    assert payload["cliff"]["threshold"] == cliff.CLIFF_RATIO_THRESHOLD


def test_the_writer_stamps_the_caller_s_clock(tmp_path):
    store = tmp_path / "store"
    _write(store, "AAA", _frame(30))
    out = tmp_path / "out" / cliff.HEALTH_FILENAME
    payload = cliff.write_store_health(store, out, measured_at=NOW.isoformat())
    assert payload["measured_at"] == NOW.isoformat()
    assert json.loads(out.read_text(encoding="utf-8"))["files"] == 1


# ---------------------------------------------------------------------------
# the tile
# ---------------------------------------------------------------------------
def test_no_measurement_yet_is_unknown_not_clean(tmp_path):
    check = audit._daily_bar_units_check(NOW, None, tmp_path / "empty")
    assert check["status"] == "unknown"
    assert "not the same as clean" in check["summary"]


def test_an_all_shares_store_is_healthy(tmp_path):
    diagnostics = _health_file(tmp_path, {
        "rows": 1_000, "rows_by_volume_unit": {"shares": 1_000},
        "cliff": {"cliffed": 0, "threshold": 20.0},
        "measured_at": (NOW - timedelta(hours=6)).isoformat(),
    })
    check = audit._daily_bar_units_check(NOW, None, diagnostics)
    assert check["status"] == "healthy"
    assert "No file steps" in check["summary"]


def test_a_round_lot_row_degrades_because_the_write_seam_refuses_them(tmp_path):
    diagnostics = _health_file(tmp_path, {
        "rows": 1_000, "rows_by_volume_unit": {"shares": 990, "lots_rth": 10},
        "cliff": {"cliffed": 0, "threshold": 20.0},
        "measured_at": (NOW - timedelta(hours=6)).isoformat(),
    })
    check = audit._daily_bar_units_check(NOW, None, diagnostics)
    assert check["status"] == "degraded"
    assert "got past it" in check["summary"]


def test_unmeasured_rows_are_reported_without_degrading(tmp_path):
    """188 rows Yahoo has no data for. An alarm nobody can clear is ignored."""
    diagnostics = _health_file(tmp_path, {
        "rows": 1_117_170, "rows_by_volume_unit": {"shares": 1_116_982, "unknown": 188},
        "cliff": {"cliffed": 53, "threshold": 20.0},
        "measured_at": (NOW - timedelta(hours=6)).isoformat(),
    })
    check = audit._daily_bar_units_check(NOW, None, diagnostics)
    assert check["status"] == "healthy"
    assert "188 row(s) remain unmeasured" in check["summary"]
    assert check["details"]["rows_other"] == 188


def test_a_cliff_is_reported_and_never_sets_the_status(tmp_path):
    """A 20x volume step is a real market event in a single-source file."""
    diagnostics = _health_file(tmp_path, {
        "rows": 1_000, "rows_by_volume_unit": {"shares": 1_000},
        "cliff": {"cliffed": 19, "threshold": 20.0},
        "measured_at": (NOW - timedelta(hours=6)).isoformat(),
    })
    check = audit._daily_bar_units_check(NOW, None, diagnostics)
    assert check["status"] == "healthy"
    assert "market event" in check["summary"]


def test_a_stale_measurement_degrades_because_it_cannot_answer_today(tmp_path):
    diagnostics = _health_file(tmp_path, {
        "rows": 1_000, "rows_by_volume_unit": {"shares": 1_000},
        "cliff": {"cliffed": 0, "threshold": 20.0},
        "measured_at": (NOW - timedelta(days=4)).isoformat(),
    })
    check = audit._daily_bar_units_check(NOW, None, diagnostics)
    assert check["status"] == "degraded"
    assert "days old" in check["summary"]


def test_an_unparseable_stamp_does_not_take_the_tile_down(tmp_path):
    diagnostics = _health_file(tmp_path, {
        "rows": 10, "rows_by_volume_unit": {"shares": 10},
        "cliff": {"cliffed": 0, "threshold": 20.0},
        "measured_at": "not a date",
    })
    check = audit._daily_bar_units_check(NOW, None, diagnostics)
    assert check["status"] == "healthy"


def test_the_tile_reads_and_never_measures():
    """It must not open 1,958 parquet files while a human waits."""
    import inspect

    source = inspect.getsource(audit._daily_bar_units_check)
    assert "measure_store_health" not in source
    assert "scan_store" not in source
    assert "read_parquet" not in source


def test_the_nightly_job_is_what_writes_it():
    import inspect

    from ops import evidence_snapshot

    source = inspect.getsource(evidence_snapshot.main)
    assert "write_store_health" in source
    # ...and a failure there must not fail the backup it rides on.
    assert "daily-bar unit health measurement skipped" in source


def test_the_check_is_in_the_operational_set():
    payload = audit.build_operations_audit()
    ids = {item["id"] for item in payload["checks"]}
    assert "daily_bar_units" in ids
