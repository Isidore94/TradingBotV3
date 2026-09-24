"""P0-2 2d (decision 0017): tracker readers move to the SQLite mirror.

Step 1: the write slot's ``load_setup_tracker_payload`` reads the mirror instead
of parsing the 1.4 GB JSON. The mirror is stamped with the JSON file it copies
(path, size, mtime); when the stamp does not match - no store, an older mirror,
a failed or disabled mirror, a JSON rewritten since - the read falls back to the
JSON with a logged reason, never to an empty tracker.

The golden check runs the write slot's tracker update in two scratch children,
one reading the JSON and one reading the store, with the clock frozen, and
requires every file they publish to be byte-identical.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap_lib.legacy as legacy  # noqa: E402
import tracker_store  # noqa: E402
from master_avwap_lib import runner  # noqa: E402


def _record(symbol: str, scan_date: str, score) -> dict:
    # Keys deliberately NOT sorted: the store must hand back the JSON's key order.
    return {
        "symbol": symbol,
        "setup_id": f"{scan_date}:{symbol}:LONG:2026-06-01:favorite_setup",
        "side": "LONG",
        "scan_date": scan_date,
        "priority_score": score,
        "zeta": {"b": 2, "a": 1, "nan": float("nan"), "tiny": 1e-7, "sum": 0.1 + 0.2, "neg0": -0.0},
        "big": 2**70,
        "note": "ünïcode → ok",
        "setup_tags": ["b", "a"],
        "compression_flag": False,
        "retest_reference_level": None,
    }


def _payload() -> dict:
    return {
        "schema_version": 2,
        "updated_at": "2026-09-23T07:15:53",
        "data_session": "2026-09-22",
        "saved_at": "2026-09-23T07:15:53-04:00",
        "saved_by": "close_slot",
        "daily_watchlists": {},
        "setups": {
            "2026-09-22:MSFT:LONG:2026-06-01:favorite_setup": _record("MSFT", "2026-09-22", np.float64(2.5)),
            "2026-09-22:AAPL:LONG:2026-06-01:favorite_setup": _record("AAPL", "2026-09-22", np.int64(3)),
            "2026-09-21:ZM:LONG:2026-06-01:favorite_setup": _record("ZM", "2026-09-21", 0.5),
            "not-a-dict": "legacy junk row",
        },
        "control_setups": {"control:2026-09-22:XOM:SHORT": _record("XOM", "2026-09-22", 1.0)},
        "study_setups": {"study:2026-09-22:KO:LONG": {"symbol": "KO", "seen": {date(2026, 9, 22)}}},
        "stats": [{"family": "x", "n": np.int64(3)}],
        "setup_type_stats": [],
        "attribute_registry": {"levels.current_band_zone": {"kind": "cat"}},
    }


def _canonical(payload: dict) -> str:
    """The loaded tracker as the JSON save would encode it - order included."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=legacy._json_default)


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    json_path = tmp_path / "tracker.json"
    db_path = tmp_path / "tracker.sqlite"
    monkeypatch.setattr(legacy, "SETUP_TRACKER_FILE", json_path)
    monkeypatch.setattr(legacy, "_setup_tracker_backup_path", lambda: tmp_path / "tracker.json.bak")
    monkeypatch.setattr(legacy, "_append_setup_tracker_events", lambda payload: {})
    monkeypatch.setattr(legacy, "save_setup_tracker_scoring_payload", lambda payload: None)
    monkeypatch.setattr(tracker_store, "default_store_path", lambda: db_path)
    monkeypatch.setattr(tracker_store, "shadow_enabled", lambda: True)
    return json_path, db_path


def _forbid_json_parse(monkeypatch, json_path: Path) -> None:
    real = legacy.load_json

    def guarded(path, default):
        if Path(path) == json_path:
            raise AssertionError("the write slot parsed the tracker JSON")
        return real(path, default)

    monkeypatch.setattr(legacy, "load_json", guarded)


def test_the_write_slot_reads_the_store_not_the_json(tracker, monkeypatch):
    json_path, _db = tracker
    legacy.save_setup_tracker_payload(_payload(), data_session="2026-09-22")
    from_json = legacy.load_setup_tracker_payload()

    _forbid_json_parse(monkeypatch, json_path)
    from_store = runner.load_setup_tracker_payload()

    assert _canonical(from_store) == _canonical(from_json)
    assert list(from_store["setups"]) == list(from_json["setups"])
    assert list(from_store["setups"]["2026-09-22:AAPL:LONG:2026-06-01:favorite_setup"]) == list(
        from_json["setups"]["2026-09-22:AAPL:LONG:2026-06-01:favorite_setup"]
    ), "each record keeps its own key order"


def test_parity_survives_a_reorder_a_delete_and_numpy_values(tracker, monkeypatch):
    json_path, _db = tracker
    payload = _payload()
    legacy.save_setup_tracker_payload(payload, data_session="2026-09-22")
    # The scan pops today's setups and re-adds them: the key moves to the end.
    moved = payload["setups"].pop("2026-09-22:MSFT:LONG:2026-06-01:favorite_setup")
    moved["priority_score"] = np.float32(1.25)
    payload["setups"]["2026-09-22:MSFT:LONG:2026-06-01:favorite_setup"] = moved
    del payload["setups"]["2026-09-21:ZM:LONG:2026-06-01:favorite_setup"]
    payload["setups"]["2026-09-23:NVDA:LONG:2026-06-01:favorite_setup"] = _record("NVDA", "2026-09-23", 4.0)
    legacy.save_setup_tracker_payload(payload, data_session="2026-09-23")
    from_json = legacy.load_setup_tracker_payload()

    _forbid_json_parse(monkeypatch, json_path)
    from_store = legacy.load_setup_tracker_payload(prefer_store=True)

    assert _canonical(from_store) == _canonical(from_json)
    assert list(from_store["setups"])[-2:] == [
        "2026-09-22:MSFT:LONG:2026-06-01:favorite_setup",
        "2026-09-23:NVDA:LONG:2026-06-01:favorite_setup",
    ]
    assert from_store["setups"]["2026-09-22:AAPL:LONG:2026-06-01:favorite_setup"]["priority_score"] == 3


def _explode(*args, **kwargs):
    raise RuntimeError("disk full")


def _second_save_without_mirror(payload, monkeypatch, how: str) -> None:
    payload["setups"]["2026-09-23:NVDA:LONG:2026-06-01:favorite_setup"] = _record("NVDA", "2026-09-23", 4.0)
    if how == "mirror_failed":
        monkeypatch.setattr(tracker_store.TrackerStore, "save_payload", _explode)
    elif how == "shadow_off":
        monkeypatch.setattr(tracker_store, "shadow_enabled", lambda: False)
    legacy.save_setup_tracker_payload(payload, data_session="2026-09-23")


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("missing_db", "no SQLite store"),
        ("json_rewritten", "changed after the last mirror"),
        ("mirror_failed", "changed after the last mirror"),
        ("shadow_off", "changed after the last mirror"),
        ("unstamped_mirror", "no format-2 source stamp"),
        ("other_json", "store mirrors"),
        ("corrupt_db", "store unreadable"),
    ],
)
def test_a_store_that_is_not_current_falls_back_to_the_json_and_says_why(
    tracker, monkeypatch, caplog, case, reason
):
    json_path, db_path = tracker
    payload = _payload()
    legacy.save_setup_tracker_payload(payload, data_session="2026-09-22")
    if case == "missing_db":
        db_path.unlink()
    elif case == "json_rewritten":
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        raw["setups"]["2026-09-23:NVDA:LONG:2026-06-01:favorite_setup"] = _record("NVDA", "2026-09-23", 4.0)
        json_path.write_text(json.dumps(raw), encoding="utf-8")
    elif case in ("mirror_failed", "shadow_off"):
        _second_save_without_mirror(payload, monkeypatch, case)
    elif case == "unstamped_mirror":
        tracker_store.TrackerStore(db_path).save_payload(json.loads(json_path.read_text(encoding="utf-8")))
    elif case == "other_json":
        other = json_path.with_name("other.json")
        monkeypatch.setattr(legacy, "SETUP_TRACKER_FILE", other)
        other.write_bytes(json_path.read_bytes())
        json_path = other
    elif case == "corrupt_db":
        for suffix in ("-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
        db_path.write_bytes(b"this is not a sqlite file" * 100)

    expected = legacy.load_setup_tracker_payload()
    with caplog.at_level(logging.WARNING):
        got = legacy.load_setup_tracker_payload(prefer_store=True)

    assert got["setups"], "never an empty tracker"
    assert _canonical(got) == _canonical(expected)
    messages = [record.getMessage() for record in caplog.records]
    assert any("read from JSON, not the SQLite store" in m and reason in m for m in messages), messages


def test_an_empty_store_still_gets_the_backup_recovery(tracker, caplog):
    json_path, _db = tracker
    legacy.save_setup_tracker_payload(_payload(), data_session="2026-09-22")
    # An empty save (allow_empty) rotates the full JSON to .bak; the mirror is current and empty.
    legacy.save_setup_tracker_payload(
        {"setups": {}, "control_setups": {}, "study_setups": {}}, allow_empty=True
    )
    with caplog.at_level(logging.WARNING):
        got = legacy.load_setup_tracker_payload(prefer_store=True)
    assert len(got["setups"]) == 4, "recovered from the .bak exactly as the JSON path does"
    assert any("store holds no records" in r.getMessage() for r in caplog.records)


GOLDEN_CHILD = r'''
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

SCRATCH = Path(sys.argv[1])
MODE = sys.argv[3]
os.environ["TRADINGBOTV3_DATA_DIR"] = str(SCRATCH / "home")
os.environ["LOCALAPPDATA"] = str(SCRATCH / "localappdata")
os.environ["TRADINGBOT_DIAGNOSTICS_DIR"] = str(SCRATCH / "diag")
os.environ["TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE"] = "1"
sys.path.insert(0, sys.argv[2])

import pandas as pd
import project_paths

for root in (project_paths.DATA_DIR, project_paths.LOCAL_SETTINGS_DIR, project_paths.get_diagnostics_dir()):
    if not str(root).startswith(str(SCRATCH)):
        raise SystemExit(f"ABORT: a project_paths root is outside the scratch dir: {root}")

import master_avwap_lib.legacy as legacy
from master_avwap_lib import runner


class Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        base = cls(2026, 9, 24, 16, 5, 0)
        return base if tz is None else base.replace(tzinfo=tz)


legacy.datetime = Frozen
legacy.fetch_daily_bars = lambda *a, **k: pd.DataFrame()


def record(symbol, scan_date, score, status):
    return {
        "symbol": symbol,
        "setup_id": f"{scan_date}:{symbol}:LONG:2026-06-01:favorite_setup",
        "side": "LONG",
        "scan_date": scan_date,
        "anchor_date": "2026-06-01",
        "setup_status": status,
        "priority_score": score,
        "priority_bucket": "favorite_setup",
        "setup_family": "favorite",
        "favorite_zone": "UPPER_1",
        "compression_flag": score > 2,
        "entry_attributes": {"trend.trend_20d": "up", "levels.current_band_zone": "UPPER_1"},
        "zeta": {"b": 2, "a": 1, "sum": 0.1 + 0.2},
    }


setups = {}
for index, symbol in enumerate(["MSFT", "AAPL", "ZM", "NVDA", "KO"]):
    for day in ("2026-09-18", "2026-09-21", "2026-09-22"):
        item = record(symbol, day, 1.0 + index * 0.7, "CLOSED" if day < "2026-09-22" else "OPEN")
        setups[item["setup_id"]] = item
seed = {
    "setups": setups,
    "control_setups": {"control:x": dict(record("XOM", "2026-09-22", 1.0, "OPEN"), setup_id="control:x")},
    "study_setups": {},
    "stats": [],
    "setup_type_stats": [],
    "attribute_registry": {},
    "daily_watchlists": {},
}
legacy.save_setup_tracker_payload(seed, data_session="2026-09-22", saved_at="2026-09-22T16:05:00-04:00")
tracker_file = Path(legacy.SETUP_TRACKER_FILE)

if MODE == "store":
    real_load_json = legacy.load_json

    def guarded(path, default):
        if Path(path) == tracker_file:
            raise SystemExit("ABORT: the store run parsed the tracker JSON")
        return real_load_json(path, default)

    legacy.load_json = guarded
    tracker = runner.load_setup_tracker_payload()
    legacy.load_json = real_load_json
else:
    tracker = legacy.load_setup_tracker_payload()

legacy.update_setup_tracker_from_scan(
    [], {"symbols": {}}, {}, {}, None, scan_date="2026-09-23", tracker_payload=tracker
)

import re

# Only what differs between two runs by construction is masked: the scratch
# folder's own name, the writer pid, and wall-clock stamps from modules the
# frozen clock above does not reach.
MASK = re.compile(r'"(event_at|writer_pid|generated_at|updated_at)"(\s*):(\s*)("[^"]*"|\d+)')
home = SCRATCH / "home"
out = {}
for path in sorted(home.rglob("*")):
    if not path.is_file() or ".sqlite" in path.name or path.suffix == ".log":
        continue
    text = path.read_bytes().decode("utf-8", errors="surrogateescape")
    for spelling in (str(SCRATCH), str(SCRATCH).replace("\\", "\\\\"), str(SCRATCH).replace("\\", "/")):
        text = text.replace(spelling, "<SCRATCH>")
    text = MASK.sub(r'"\1"\2:\3"<MASKED>"', text)
    data = text.encode("utf-8", errors="surrogateescape")
    out[str(path.relative_to(home)).replace("\\", "/")] = hashlib.sha256(data).hexdigest()
print("GOLDEN::" + json.dumps(out))
'''


def _run_golden_child(tmp_path: Path, mode: str) -> dict:
    scratch = tmp_path / mode
    for name in ("home", "localappdata", "diag"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    child = scratch / "golden_child.py"
    child.write_text(GOLDEN_CHILD, encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(child), str(scratch), str(SCRIPTS_DIR), mode],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=300,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("GOLDEN::")]
    assert completed.returncode == 0 and lines, completed.stderr[-4000:]
    return json.loads(lines[-1][len("GOLDEN::"):])


def test_golden_the_write_slot_publishes_byte_identical_files_from_the_store(tmp_path):
    from_json = _run_golden_child(tmp_path, "json")
    from_store = _run_golden_child(tmp_path, "store")
    assert any(name.endswith("master_avwap_setup_tracker.json") for name in from_json), sorted(from_json)
    assert len(from_json) > 3, sorted(from_json)
    assert from_store == from_json


# ---------------------------------------------------------------------------
# Step 3: journal_analytics context rows (the parse behind the Corrections OK button).


def _journal_payload() -> dict:
    payload = _payload()
    payload["setups"].update(
        {
            "k1": {"symbol": "shop", "side": "short", "entry_trade_date": "2026-09-19",
                   "retest_reference_level": 0, "mid_earnings_primary_trigger_level": 101.5,
                   "compression_flag": "false", "priority_score": "1.5"},
            "k2": {"symbol": "TSLA", "side": "LONG", "scan_date": "2026-09-22", "compression_flag": [],
                   "retest_reference_level": "UPPER_2", "priority_score": float("nan"), "setup_family": ""},
            "k3": {"symbol": "AMD", "compression_flag": {}, "favorite_zone": None, "priority_bucket": 0},
            "k4": {"symbol": "IBM", "compression_flag": 1, "retest_reference_level": {"level": 3},
                   "setup_family": "favorite", "scan_date": "garbage"},
            "k5": 42,
        }
    )
    return payload


def _json_rows(json_path: Path, tmp_path: Path) -> list[dict]:
    import journal_analytics

    tagger = journal_analytics.AutoTagger(
        setup_tracker_path=json_path, setup_tracker_db_path=tmp_path / "no-such.sqlite"
    )
    return tagger._load_tracker_rows()


def _rows_text(rows: list[dict]) -> str:
    return json.dumps(rows, default=str, ensure_ascii=False)


def test_journal_context_rows_come_from_the_store_and_match_the_json(tracker, tmp_path, monkeypatch):
    import journal_analytics

    json_path, db_path = tracker
    legacy.save_setup_tracker_payload(_journal_payload(), data_session="2026-09-22")
    expected = _json_rows(json_path, tmp_path)
    assert len(expected) == 7

    real = journal_analytics._load_json

    def guarded(path):
        if Path(path) == json_path:
            raise AssertionError("the auto-tagger parsed the tracker JSON")
        return real(path)

    monkeypatch.setattr(journal_analytics, "_load_json", guarded)
    journal_analytics.clear_context_row_cache()
    tagger = journal_analytics.AutoTagger(
        setup_tracker_path=json_path,
        setup_tracker_db_path=db_path,
        focus_path=tmp_path / "none.json",
        avwap_signals_path=tmp_path / "none.csv",
        intraday_bounces_path=tmp_path / "none2.csv",
    )
    got = tagger.load_context_rows()
    journal_analytics.clear_context_row_cache()

    assert _rows_text(got) == _rows_text(expected)


def test_journal_context_rows_fall_back_to_the_json_when_the_store_is_stale(tracker, tmp_path, caplog):
    import journal_analytics

    json_path, db_path = tracker
    legacy.save_setup_tracker_payload(_journal_payload(), data_session="2026-09-22")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw["setups"]["late"] = {"symbol": "LATE", "side": "LONG", "scan_date": "2026-09-23"}
    json_path.write_text(json.dumps(raw), encoding="utf-8")

    tagger = journal_analytics.AutoTagger(setup_tracker_path=json_path, setup_tracker_db_path=db_path)
    with caplog.at_level(logging.WARNING):
        got = tagger._load_tracker_rows()

    assert [row["symbol"] for row in got][-1] == "LATE"
    assert _rows_text(got) == _rows_text(_json_rows(json_path, tmp_path))
    assert any(
        "not the SQLite store" in r.getMessage() and "changed after the last mirror" in r.getMessage()
        for r in caplog.records
    )


def test_journal_rows_match_when_the_truthy_fields_hold_nan(tracker, tmp_path, monkeypatch):
    """SQLite's `->` reads NaN as null; the JSON path must treat NaN the same way."""
    import journal_analytics

    json_path, db_path = tracker
    payload = _payload()
    payload["setups"].update(
        {
            "n1": {"symbol": "NANA", "side": "LONG", "scan_date": "2026-09-22", "compression_flag": float("nan"),
                   "retest_reference_level": float("nan"), "mid_earnings_primary_trigger_level": 12.5},
            "n2": {"symbol": "NANB", "compression_flag": [float("nan")],
                   "retest_reference_level": {"level": float("nan")}, "favorite_zone": float("nan")},
        }
    )
    legacy.save_setup_tracker_payload(payload, data_session="2026-09-22")
    expected = _json_rows(json_path, tmp_path)
    got = journal_analytics.AutoTagger(setup_tracker_path=json_path, setup_tracker_db_path=db_path)._load_tracker_rows()
    assert _rows_text(got) == _rows_text(expected)
    nana = next(row for row in got if row["symbol"] == "NANA")
    assert nana["compression"] is False and nana["retest"] == 12.5


# ---------------------------------------------------------------------------
# Reviewer blocker: the stamp must describe the file the mirrored payload came from.


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, default=legacy._json_default), encoding="utf-8")


def test_a_mirror_of_an_older_payload_is_never_stamped_current(tmp_path):
    json_path = tmp_path / "t.json"
    db_path = tmp_path / "t.sqlite"
    j0 = _payload()
    j0.pop("study_setups")
    _write(json_path, j0)
    stamp_j0 = tracker_store.file_stamp(json_path)
    loaded_j0 = json.loads(json_path.read_text(encoding="utf-8"))
    j1 = json.loads(json_path.read_text(encoding="utf-8"))
    j1["setups"]["2026-09-23:NVDA:LONG:2026-06-01:favorite_setup"] = {"symbol": "NVDA"}
    _write(json_path, j1)
    os.utime(json_path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))

    store = tracker_store.TrackerStore(db_path)
    # The reviewer's reproduction: a source path alone never stamps the mirror.
    store.save_payload(loaded_j0, source_path=json_path)
    assert tracker_store.load_fresh_payload(json_path, db_path)[0] is None
    # With the stamp taken while J0 was on disk, the rewrite is caught at commit.
    store.save_payload(loaded_j0, source_path=json_path, source_stamp=stamp_j0)
    payload, reason = tracker_store.load_fresh_payload(json_path, db_path)
    assert payload is None and "no format-2 source stamp" in reason
    # The honest case still stamps.
    store.save_payload(j1, source_path=json_path, source_stamp=tracker_store.file_stamp(json_path))
    payload, reason = tracker_store.load_fresh_payload(json_path, db_path)
    assert reason == "" and "2026-09-23:NVDA:LONG:2026-06-01:favorite_setup" in payload["setups"]


def _rewrite_json_before_mirroring(monkeypatch, json_path: Path) -> None:
    real = tracker_store.TrackerStore.save_payload

    def racing(self, payload, **kwargs):
        newer = json.loads(json_path.read_text(encoding="utf-8"))
        newer["setups"]["2026-09-24:LATE:LONG:2026-06-01:favorite_setup"] = {"symbol": "LATE"}
        _write(json_path, newer)
        os.utime(json_path, ns=(1_800_000_000_000_000_000, 1_800_000_000_000_000_000))
        return real(self, payload, **kwargs)

    monkeypatch.setattr(tracker_store.TrackerStore, "save_payload", racing)


def test_a_writer_between_the_save_and_the_mirror_leaves_the_store_unstamped(tracker, monkeypatch, caplog):
    json_path, db_path = tracker
    _rewrite_json_before_mirroring(monkeypatch, json_path)
    with caplog.at_level(logging.WARNING):
        legacy.save_setup_tracker_payload(_payload(), data_session="2026-09-22")

    assert tracker_store.load_fresh_payload(json_path, db_path)[0] is None
    got = runner.load_setup_tracker_payload()
    assert "2026-09-24:LATE:LONG:2026-06-01:favorite_setup" in got["setups"], "the newer JSON wins"
    assert any("mirror left unstamped" in r.getMessage() for r in caplog.records)


def test_the_cli_mirror_does_not_stamp_a_file_rewritten_during_its_run(tmp_path, monkeypatch):
    json_path = tmp_path / "t.json"
    db_path = tmp_path / "t.sqlite"
    j0 = _payload()
    j0.pop("study_setups")
    _write(json_path, j0)
    _rewrite_json_before_mirroring(monkeypatch, json_path)

    assert tracker_store._main(["mirror", "--json", str(json_path), "--db", str(db_path)]) == 0
    payload, reason = tracker_store.load_fresh_payload(json_path, db_path)
    assert payload is None and reason
