"""The setup tracker archives a sealed record's per-bar detail, verified, before
compaction strips it; an archive failure costs disk space, never data or the save."""

from __future__ import annotations

import copy
import json
import logging
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402
import tracker_detail_archive as tda  # noqa: E402

NAMESPACES = ("setups", "control_setups", "study_setups")
SCAN = "2026-07-01"


def _text(value) -> str:
    """The tracker save's own encoding (save_json)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=m._json_default)


def _mark(day: str, i: int) -> dict:
    return {
        "trade_date": day,
        "is_entry_day": i == 0,
        "close": 100.0 + i,
        "high": 101.5 + i,
        "low": 98.5 + i,
        "note": "Zürich ✓ 日本",
        "nan_value": float("nan"),
        "inf_value": float("inf"),
        "np_value": np.float64(1.25),
        "np_int": np.int64(7),
        "when": date.fromisoformat(day),
        "stamp": datetime(2026, 5, 1, 9, 30),
        "tags": ("a", "b"),
        "current_levels": {"UPPER_1": 101.0},
    }


def _record(status: str = "CLOSED", last_mark: str = "2026-05-01", *, n_marks: int = 4) -> dict:
    last = date.fromisoformat(last_mark)
    marks = [_mark((last - timedelta(days=n_marks - 1 - i)).isoformat(), i) for i in range(n_marks)]
    return {
        "symbol": "AAA",
        "scan_date": marks[0]["trade_date"],
        "side": "LONG",
        "setup_status": status,
        "entry_price": 100.0,
        "latest_snapshot": {"trade_date": last_mark, "close": 103.0},
        "entry_feature_snapshot": {"ema8": 50.0},
        "daily_marks": marks,
        "scenarios": {
            "s1": {
                "status": "STOPPED",
                "total_r": -1.0,
                "tradeable": True,
                "initial_risk_per_share": 2.0,
                "stop_reference_label": "LOWER_1",
                "events": [{"reason": "STOP", "shares": 100, "price": 98.0, "date": last_mark}],
            },
            "s2": {"status": "OPEN", "total_r": 0.0, "events": []},  # nothing to strip
            "s3": {"status": "TARGET_HIT", "total_r": 1.5, "events": [{"reason": "TARGET", "price": float("nan")}]},
        },
        "tail_field": "kept",
    }


def _variants() -> dict[str, dict]:
    empty_marks = _record()
    empty_marks["daily_marks"] = []
    no_events = _record()
    for scenario in no_events["scenarios"].values():
        scenario["events"] = []
    with_short = _record()
    with_short["short_horizon"] = {"complete": False, "bars_available": 1}
    return {"full": _record(), "empty_marks": empty_marks, "no_events": no_events, "with_short": with_short}


@pytest.fixture
def archive_path(tmp_path):
    return tmp_path / "detail_archive.sqlite"


# -- round trip ----------------------------------------------------------------
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("variant", ["full", "empty_marks", "no_events", "with_short"])
def test_archive_compact_restore_reproduces_the_record(archive_path, namespace, variant):
    original = _variants()[variant]
    original_text = _text(original)
    working = copy.deepcopy(original)

    safe = tda.archive_before_compaction([(namespace, "K1", working)], json_default=m._json_default, path=archive_path)
    assert (namespace, "K1") in safe
    assert m._compact_tracker_setup_record(working)
    assert not m._compact_tracker_setup_record(working)  # compacted twice
    assert _text(working) != original_text

    detail = tda.DetailArchive(archive_path).load_detail(namespace, "K1")
    restored = tda.restore_record(working, detail)
    assert _text(restored) == original_text


def test_compaction_adds_short_horizon_and_restore_takes_it_back_out(archive_path):
    original = _record()
    assert "short_horizon" not in original
    working = copy.deepcopy(original)
    tda.archive_before_compaction([("setups", "K", working)], json_default=m._json_default, path=archive_path)
    m._compact_tracker_setup_record(working)
    assert "short_horizon" in working  # the case is really exercised
    restored = tda.restore_record(working, tda.DetailArchive(archive_path).load_detail("setups", "K"))
    assert "short_horizon" not in restored
    assert _text(restored) == _text(original)


def test_restore_is_pure(archive_path):
    working = _record()
    detail = json.loads(tda.encode_detail(tda.extract_detail(working), m._json_default))
    m._compact_tracker_setup_record(working)
    before = _text(working)
    restored = tda.restore_record(working, detail)
    assert _text(working) == before
    restored["daily_marks"][0]["close"] = -1
    assert detail["daily_marks"][0]["close"] != -1


def test_a_record_with_nothing_to_strip_is_safe_and_writes_nothing(archive_path):
    record = _record()
    record["daily_marks"] = []
    for scenario in record["scenarios"].values():
        scenario["events"] = []
    assert tda.extract_detail(record) is None
    safe = tda.archive_before_compaction([("setups", "K", record)], path=archive_path)
    assert safe == {("setups", "K")}
    assert tda.DetailArchive(archive_path).status()["rows"] == 0


# -- idempotence and different content -------------------------------------------
def test_archiving_the_same_content_twice_changes_nothing(archive_path):
    archive = tda.DetailArchive(archive_path)
    first = archive.archive([("setups", "K", _record())], json_default=m._json_default)
    status = archive.status()
    second = archive.archive([("setups", "K", _record())], json_default=m._json_default)
    assert (first.written, second.written, second.already_archived) == (1, 0, 1)
    assert ("setups", "K") in second.safe
    after = archive.status()
    assert (after["rows"], after["stored_bytes"]) == (status["rows"], status["stored_bytes"])


def test_different_content_for_the_same_key_is_kept_beside_never_over(archive_path):
    archive = tda.DetailArchive(archive_path)
    old = _record()
    new = _record()
    new["daily_marks"][0]["close"] = 555.0
    archive.archive([("setups", "K", old)], json_default=m._json_default)
    result = archive.archive([("setups", "K", new)], json_default=m._json_default)
    assert ("setups", "K") in result.safe
    versions = archive.load_versions("setups", "K")
    assert len(versions) == 2
    assert versions[0]["daily_marks"][0]["close"] == 100.0
    assert archive.load_detail("setups", "K")["daily_marks"][0]["close"] == 555.0


def test_namespaces_are_separate_keys(archive_path):
    archive = tda.DetailArchive(archive_path)
    a = _record()
    b = _record()
    b["daily_marks"][0]["close"] = 7.0
    archive.archive([("setups", "K", a), ("study_setups", "K", b)], json_default=m._json_default)
    assert archive.load_detail("setups", "K")["daily_marks"][0]["close"] == 100.0
    assert archive.load_detail("study_setups", "K")["daily_marks"][0]["close"] == 7.0
    assert archive.load_detail("control_setups", "K") is None


# -- the hook in the tracker save --------------------------------------------------
def _tracker() -> dict:
    return {
        "setups": {
            "sealed": _record("CLOSED", "2026-05-01"),
            "recent_closed": _record("CLOSED", "2026-06-28"),
            "open": _record("OPEN", "2026-05-01"),
        },
        "control_setups": {"sealed_c": _record("CLOSED", "2026-05-20")},
        "study_setups": {"sealed_s": _record("CLOSED", "2026-03-01")},
        "daily_watchlists": {},
    }


def test_the_hook_archives_every_record_it_strips(archive_path):
    tracker = _tracker()
    originals = copy.deepcopy(tracker)
    with mock.patch.object(tda, "default_archive_path", return_value=archive_path):
        assert m._compact_sealed_tracker_setups(tracker, SCAN) == 3
    archive = tda.DetailArchive(archive_path)
    for namespace, key in (("setups", "sealed"), ("control_setups", "sealed_c"), ("study_setups", "sealed_s")):
        restored = tda.restore_record(tracker[namespace][key], archive.load_detail(namespace, key))
        assert _text(restored) == _text(originals[namespace][key])
    assert archive.load_detail("setups", "recent_closed") is None


def test_an_unavailable_archive_strips_nothing_and_says_so(tmp_path, caplog):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    tracker = _tracker()
    before = _text(tracker)
    with mock.patch.object(tda, "default_archive_path", return_value=blocker / "archive.sqlite"), \
         caplog.at_level(logging.ERROR):
        assert m._compact_sealed_tracker_setups(tracker, SCAN) == 0
    assert _text(tracker) == before
    assert any("FAILED for 3 sealed record(s)" in r.getMessage() for r in caplog.records)


def test_a_failed_read_back_leaves_the_records_whole(archive_path, caplog):
    tracker = _tracker()
    before = _text(tracker)
    with mock.patch.object(tda, "default_archive_path", return_value=archive_path),          mock.patch.object(tda, "_blob_ok", return_value=False), caplog.at_level(logging.ERROR):
        assert m._compact_sealed_tracker_setups(tracker, SCAN) == 0
    assert _text(tracker) == before
    assert any("FAILED for 3 sealed record(s)" in r.getMessage() for r in caplog.records)


def test_a_write_error_on_one_record_leaves_only_that_record_whole(archive_path):
    tracker = _tracker()
    tracker["study_setups"]["sealed_s"]["daily_marks"][0]["close"] = 9.0
    bad_text = _text(tracker["study_setups"]["sealed_s"])
    real_encode = tda.encode_detail

    def _encode(detail, json_default=None):
        if detail["daily_marks"][0]["close"] == 9.0:
            raise TypeError("boom")
        return real_encode(detail, json_default)

    with mock.patch.object(tda, "default_archive_path", return_value=archive_path), \
         mock.patch.object(tda, "encode_detail", side_effect=_encode):
        assert m._compact_sealed_tracker_setups(tracker, SCAN) == 2
    assert _text(tracker["study_setups"]["sealed_s"]) == bad_text
    assert tracker["setups"]["sealed"]["daily_marks"] == []


def test_a_missing_archive_module_never_costs_the_compaction_call(archive_path, caplog):
    tracker = _tracker()
    before = _text(tracker)
    with mock.patch.dict(sys.modules, {"tracker_detail_archive": None}), caplog.at_level(logging.ERROR):
        assert m._compact_sealed_tracker_setups(tracker, SCAN) == 0
    assert _text(tracker) == before
    assert any("detail archive unavailable" in r.getMessage() for r in caplog.records)


def test_a_locked_archive_fails_closed_and_the_next_save_retries(archive_path):
    tda.DetailArchive(archive_path).archive([], json_default=m._json_default)  # create the file
    tracker = _tracker()
    holder = sqlite3.connect(str(archive_path))
    holder.execute("BEGIN EXCLUSIVE")
    try:
        with mock.patch.object(tda, "default_archive_path", return_value=archive_path), \
             mock.patch.object(tda, "_BUSY_TIMEOUT_SECONDS", 0.1):
            assert m._compact_sealed_tracker_setups(tracker, SCAN) == 0
    finally:
        holder.rollback()
        holder.close()
    with mock.patch.object(tda, "default_archive_path", return_value=archive_path):
        assert m._compact_sealed_tracker_setups(tracker, SCAN) == 3


def _old_compact_sealed(tracker: dict, scan_date: str) -> int:
    """The pre-archive compaction loop, verbatim, as the golden reference."""
    compacted = 0
    for namespace in ("setups", "control_setups", "study_setups"):
        for setup in (tracker.get(namespace) or {}).values():
            if not isinstance(setup, dict):
                continue
            if not m._tracker_setup_recompute_is_sealed(setup, scan_date):
                continue
            if m._compact_tracker_setup_record(setup):
                compacted += 1
    return compacted


def _saved_payload_text(tracker: dict, *, compact=None) -> str:
    patches = [
        mock.patch.object(m, "export_setup_tracker_views"),
        mock.patch.object(m, "write_control_discovery_report"),
        mock.patch.object(m, "write_master_avwap_study_report"),
        mock.patch.object(m, "tracker_save_timestamp", return_value="2026-07-01T16:30:00-04:00"),
        mock.patch.object(m, "fetch_daily_bars", return_value=pd.DataFrame()),
        mock.patch.object(m, "_load_cached_daily_bar_frame", return_value=None),
    ]
    if compact is not None:
        patches.append(mock.patch.object(m, "_compact_sealed_tracker_setups", side_effect=compact))
    with mock.patch.object(m, "save_setup_tracker_payload") as save_mock:
        for patch in patches:
            patch.start()
        try:
            m.update_setup_tracker_from_scan(
                [], {"symbols": {}}, {}, {}, None, scan_date=SCAN, auto_tune=False, tracker_payload=tracker
            )
        finally:
            for patch in patches:
                patch.stop()
    save_mock.assert_called_once()
    saved = save_mock.call_args.args[0]
    for watchlist in saved.get("daily_watchlists", {}).values():
        watchlist["updated_at"] = ""  # wall clock; the two runs may straddle a second
    return _text(saved)


def test_golden_the_saved_payload_is_identical_with_a_healthy_archive(archive_path):
    reference = _saved_payload_text(_tracker(), compact=_old_compact_sealed)
    with mock.patch.object(tda, "default_archive_path", return_value=archive_path):
        archived = _saved_payload_text(_tracker())
    assert archived == reference
    assert '"daily_marks":[]' in archived
    assert tda.DetailArchive(archive_path).status()["records"] == 3


# -- CLI -----------------------------------------------------------------------
def test_cli_status_verify_show(archive_path, capsys):
    tda.DetailArchive(archive_path).archive([("setups", "K", _record())], json_default=m._json_default)
    assert tda._main(["--db", str(archive_path), "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["records"] == 1 and status["stored_bytes"] > 0 and status["last_write"]
    assert tda._main(["--db", str(archive_path), "verify"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert tda._main(["--db", str(archive_path), "show", "setups", "K"]) == 0
    assert "Zürich" in capsys.readouterr().out
    assert tda._main(["--db", str(archive_path), "show", "setups", "missing"]) == 1


def test_cli_verify_catches_a_corrupt_blob(archive_path, capsys):
    tda.DetailArchive(archive_path).archive([("setups", "K", _record())], json_default=m._json_default)
    conn = sqlite3.connect(str(archive_path))
    conn.execute("UPDATE detail SET blob = ?", (b"not zlib",))
    conn.commit()
    conn.close()
    assert tda._main(["--db", str(archive_path), "verify"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


# -- review advisories (2026-10-02) ------------------------------------------------
def _archive_with_compactor(archive, namespace, key, record):
    return archive.archive([(namespace, key, record)], json_default=m._json_default, compact=m._compact_tracker_setup_record)


def _as_saved(record: dict) -> dict:
    """The record as the tracker JSON holds it after compaction and a save/load."""
    compacted = copy.deepcopy(record)
    m._compact_tracker_setup_record(compacted)
    return json.loads(_text(compacted))


def test_restore_picks_the_version_that_belongs_to_the_saved_record(archive_path):
    # Archive X (save failed), Y (save failed), X again (saved): rowid order says Y.
    x = _record()
    y = _record()
    y["daily_marks"][0]["close"] = 555.0
    y["latest_snapshot"]["close"] = 555.0
    archive = tda.DetailArchive(archive_path)
    for record in (x, y, x):
        assert ("setups", "K") in _archive_with_compactor(archive, "setups", "K", record).safe
    restored = archive.restore_saved_record("setups", "K", _as_saved(x), json_default=m._json_default)
    assert _text(restored) == _text(x)
    restored_y = archive.restore_saved_record("setups", "K", _as_saved(y), json_default=m._json_default)
    assert _text(restored_y) == _text(y)


def test_restore_tells_x_from_y_when_only_the_detail_differs(archive_path):
    x = _record()
    y = _record()
    y["daily_marks"][2]["close"] = 555.0  # changes the short_horizon compaction writes, nothing else
    archive = tda.DetailArchive(archive_path)
    for record in (x, y, x):
        _archive_with_compactor(archive, "setups", "K", record)
    assert _as_saved(x) != _as_saved(y)
    assert _text(archive.restore_saved_record("setups", "K", _as_saved(x), json_default=m._json_default)) == _text(x)


def test_restore_says_so_when_no_version_matches(archive_path):
    x = _record()
    archive = tda.DetailArchive(archive_path)
    _archive_with_compactor(archive, "setups", "K", x)
    saved = _as_saved(x)
    saved["scenarios"]["s1"]["total_r"] = 9.9  # an outcome no archived version produced
    with pytest.raises(tda.NoMatchingDetail, match="1 archived version"):
        archive.restore_saved_record("setups", "K", saved, json_default=m._json_default)
    with pytest.raises(tda.NoMatchingDetail, match="0 archived version"):
        archive.restore_saved_record("setups", "OTHER", saved, json_default=m._json_default)


def test_the_tracker_save_records_which_version_each_saved_record_carries(archive_path):
    tracker = _tracker()
    for namespace in NAMESPACES:
        for record in tracker[namespace].values():
            record["expiry_reason"] = ""  # live records carry the sweep's stamp from earlier saves
    originals = copy.deepcopy(tracker)
    with mock.patch.object(tda, "default_archive_path", return_value=archive_path), \
         mock.patch.object(m, "export_setup_tracker_views"), \
         mock.patch.object(m, "write_control_discovery_report"), \
         mock.patch.object(m, "write_master_avwap_study_report"), \
         mock.patch.object(m, "fetch_daily_bars", return_value=pd.DataFrame()), \
         mock.patch.object(m, "_load_cached_daily_bar_frame", return_value=None), \
         mock.patch.object(m, "save_setup_tracker_payload") as save_mock:
        m.update_setup_tracker_from_scan(
            [], {"symbols": {}}, {}, {}, None, scan_date=SCAN, auto_tune=False, tracker_payload=tracker
        )
    saved = json.loads(_text(save_mock.call_args.args[0]))
    archive = tda.DetailArchive(archive_path)
    for namespace, key in (("setups", "sealed"), ("control_setups", "sealed_c"), ("study_setups", "sealed_s")):
        restored = archive.restore_saved_record(namespace, key, saved[namespace][key], json_default=m._json_default)
        assert _text(restored) == _text(originals[namespace][key])


def test_an_archive_written_by_the_first_schema_still_reads_and_says_it_cannot_match(archive_path):
    # The ba99fabb schema: one `detail` table, no record of which saved record a version belongs to.
    x = _record()
    text = tda.encode_detail(tda.extract_detail(x), m._json_default)
    conn = sqlite3.connect(str(archive_path))
    conn.execute(
        "CREATE TABLE detail (namespace TEXT NOT NULL, setup_key TEXT NOT NULL, sha256 TEXT NOT NULL,"
        " format TEXT NOT NULL, raw_bytes INTEGER NOT NULL, blob BLOB NOT NULL,"
        " archived_at TEXT NOT NULL, PRIMARY KEY (namespace, setup_key, sha256))"
    )
    conn.execute(
        "INSERT INTO detail VALUES ('setups', 'OLD', ?, 'tracker_detail_v1', ?, ?, '2026-10-02T00:00:00+00:00')",
        (tda._hash(text), len(text), tda._compress(text)),
    )
    conn.commit()
    conn.close()
    archive = tda.DetailArchive(archive_path)
    assert archive.load_detail("setups", "OLD") == json.loads(text)
    with pytest.raises(tda.NoMatchingDetail):
        archive.restore_saved_record("setups", "OLD", _as_saved(x), json_default=m._json_default)
    assert ("setups", "NEW") in _archive_with_compactor(archive, "setups", "NEW", _record()).safe
    assert archive.verify()["ok"]
    assert archive.status()["records"] == 2


def test_a_locked_archive_costs_the_tracker_save_about_a_second_not_fourteen(archive_path):
    import time

    tda.DetailArchive(archive_path).archive([], json_default=m._json_default)
    tracker = _tracker()
    holder = sqlite3.connect(str(archive_path))
    holder.execute("BEGIN EXCLUSIVE")
    try:
        started = time.monotonic()
        with mock.patch.object(tda, "default_archive_path", return_value=archive_path):
            assert m._compact_sealed_tracker_setups(tracker, SCAN) == 0
        elapsed = time.monotonic() - started
    finally:
        holder.rollback()
        holder.close()
    assert elapsed < 3.0, f"a locked archive held the save for {elapsed:.1f}s"


def _bulky_record(seed: int) -> dict:
    import random

    rng = random.Random(seed)
    record = _record(n_marks=40)
    for mark in record["daily_marks"]:
        mark["noise"] = [rng.random() for _ in range(40)]  # incompressible
    return record


def test_a_disk_full_rollback_is_caught_by_the_read_back(archive_path):
    """SQLITE_FULL rolls back inserts the pass already counted; only the read-back
    keeps those records from being compacted with their detail gone."""
    tda.DetailArchive(archive_path).archive([], json_default=m._json_default)
    real_connect = tda.DetailArchive._connect

    def _capped(self, *, create):
        conn = real_connect(self, create=create)
        if create:
            pages = conn.execute("PRAGMA page_count").fetchone()[0]
            conn.execute(f"PRAGMA max_page_count={pages + 12}")
        return conn

    records = {f"K{i}": _bulky_record(i) for i in range(30)}
    items = [("setups", key, record) for key, record in records.items()]
    with mock.patch.object(tda.DetailArchive, "_connect", _capped):
        result = tda.DetailArchive(archive_path).archive(
            items, json_default=m._json_default, compact=m._compact_tracker_setup_record
        )
    assert result.failed, "the cap did not bite; the test is not exercising a full disk"
    assert result.safe, "nothing fit under the cap; the test is not exercising a partial write"
    archive = tda.DetailArchive(archive_path)
    for namespace, key in result.safe:
        restored = archive.restore_saved_record(namespace, key, _as_saved(records[key]), json_default=m._json_default)
        assert _text(restored) == _text(records[key])
