"""The 15 s Health audit must not re-read unchanged evidence files.

Desk snappiness packet 1 item 1. On 2026-08-31 the desk's own stall log showed
`_outcome_claim_coverage_check` re-parsing the 269 MB, 294k-row
`intraday_bounce_outcomes.csv` on every 15 s audit pass (2.29 s each), and the
two shadow checks re-streaming both shadow JSONL logs with no mtime guard.
The fix caches each parse on the file's `(st_mtime_ns, st_size)` stamp - the
`review_events.load_review_events` template - so an unchanged file is parsed
once and an append invalidates. Caching only: the result for identical inputs
must be byte-identical to the uncached parse.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import operations_audit  # noqa: E402
from diagnostics import shadow_log_audit  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Module-level caches survive across tests in one process; start clean."""
    operations_audit._outcome_claim_cache = None
    shadow_log_audit._scan_cache.clear()
    yield
    operations_audit._outcome_claim_cache = None
    shadow_log_audit._scan_cache.clear()


class _OpenCounter:
    """Count `Path.open` calls against one file name, pass everything through."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, filename: str) -> None:
        self.count = 0
        original = Path.open
        counter = self

        def counting_open(path_self, *args, **kwargs):
            if path_self.name == filename:
                counter.count += 1
            return original(path_self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", counting_open)


def _write_outcomes(path: Path, rows: int) -> None:
    lines = ["event_id,outcome"]
    lines += [f"AAPL_long_20260831_06_30_0{i}_bounce_v1,MEASURED" for i in range(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bump_mtime(path: Path) -> None:
    """Guarantee the stamp moves even on a coarse filesystem clock."""
    import os

    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


# ---------------------------------------------------------------------------
# outcome claim coverage (the 269 MB CSV)
# ---------------------------------------------------------------------------


def test_unchanged_outcome_csv_is_parsed_once(tmp_path, monkeypatch):
    csv_path = tmp_path / "intraday_bounce_outcomes.csv"
    _write_outcomes(csv_path, rows=3)
    counter = _OpenCounter(monkeypatch, csv_path.name)

    first = operations_audit._outcome_claim_coverage_check(csv_path)
    second = operations_audit._outcome_claim_coverage_check(csv_path)

    assert counter.count == 1, "an unchanged stamp must not re-open the CSV"
    assert second == first, "the cached check must be byte-identical"


def test_appending_to_the_outcome_csv_invalidates(tmp_path, monkeypatch):
    csv_path = tmp_path / "intraday_bounce_outcomes.csv"
    _write_outcomes(csv_path, rows=3)
    counter = _OpenCounter(monkeypatch, csv_path.name)

    operations_audit._outcome_claim_coverage_check(csv_path)
    with csv_path.open("a", encoding="utf-8") as handle:
        handle.write("MSFT_short_20260831_07_00_00_bounce_v1,MEASURED\n")
    _bump_mtime(csv_path)
    before = counter.count

    refreshed = operations_audit._outcome_claim_coverage_check(csv_path)

    assert counter.count == before + 1, "a moved (mtime, size) stamp must re-parse"
    # And the re-parse must equal a cold parse of the same bytes.
    operations_audit._outcome_claim_cache = None
    assert refreshed == operations_audit._outcome_claim_coverage_check(csv_path)


def test_outcome_csv_error_result_is_not_cached(tmp_path):
    missing = tmp_path / "intraday_bounce_outcomes.csv"
    verdict = operations_audit._outcome_claim_coverage_check(missing)
    assert verdict["status"] == operations_audit.STATUS_UNKNOWN
    assert operations_audit._outcome_claim_cache is None


# ---------------------------------------------------------------------------
# shadow log scans
# ---------------------------------------------------------------------------


def _spy_row(ts: str) -> dict:
    return {
        "schema": "spy_state_shadow_v4",
        "ts": ts,
        "evaluated_at": ts,
        "bar_ts": ts,
        "session_date": ts[:10],
        "timezone": "Pacific Daylight Time",
        "machine": "test-machine",
        "engine_version": "spy_state_v1",
        "config_hash": "spy-config-1",
        "state": "BULL_IMPULSE",
        "usable": True,
        "incomplete_bar": False,
        "complete_bar_ts": ts,
        "stale": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


_NOW = datetime(2026, 8, 31, 20, 0, tzinfo=timezone.utc)


def _scan(log: Path) -> dict:
    return shadow_log_audit.scan_shadow_log(
        log,
        shadow_log_audit.SPY_PROFILE,
        now=_NOW,
        market_date="2026-08-31",
    )


def test_unchanged_shadow_log_is_streamed_once(tmp_path, monkeypatch):
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_jsonl(log, [_spy_row("2026-08-31T06:35:00-07:00")])
    counter = _OpenCounter(monkeypatch, log.name)

    first = _scan(log)
    second = _scan(log)

    assert counter.count == 1, "an unchanged stamp must not re-stream the log"
    assert second == first, "the cached scan must be byte-identical"


def test_appending_to_a_shadow_log_invalidates(tmp_path, monkeypatch):
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_jsonl(log, [_spy_row("2026-08-31T06:35:00-07:00")])
    counter = _OpenCounter(monkeypatch, log.name)

    assert _scan(log)["valid_rows"] == 1
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_spy_row("2026-08-31T06:40:00-07:00")) + "\n")
    _bump_mtime(log)
    before = counter.count

    refreshed = _scan(log)

    assert counter.count == before + 1, "a moved (mtime, size) stamp must re-stream"
    assert refreshed["valid_rows"] == 2


def test_a_different_market_date_is_a_different_scan(tmp_path, monkeypatch):
    """The key carries the query, not just the file: midnight must invalidate."""
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_jsonl(log, [_spy_row("2026-08-31T06:35:00-07:00")])
    counter = _OpenCounter(monkeypatch, log.name)

    today = _scan(log)
    tomorrow = shadow_log_audit.scan_shadow_log(
        log,
        shadow_log_audit.SPY_PROFILE,
        now=_NOW,
        market_date="2026-09-01",
    )

    assert counter.count == 2
    assert today["rows_for_market_date"] == 1
    assert tomorrow["rows_for_market_date"] == 0


def test_damaged_read_is_not_cached(tmp_path):
    """A file that vanishes between stat and open must not poison the cache."""
    log = tmp_path / "spy_state_shadow.jsonl"
    _write_jsonl(log, [_spy_row("2026-08-31T06:35:00-07:00")])
    _scan(log)
    assert (shadow_log_audit.SPY_PROFILE.name, str(log)) in shadow_log_audit._scan_cache
    shadow_log_audit._scan_cache.clear()

    original_open = Path.open

    def failing_open(path_self, *args, **kwargs):
        if path_self.name == log.name and args and args[0] == "rb":
            raise OSError("gone")
        return original_open(path_self, *args, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(Path, "open", failing_open):
        broken = _scan(log)
    assert broken["readable"] is False
    assert not shadow_log_audit._scan_cache, "an error scan must never be cached"
