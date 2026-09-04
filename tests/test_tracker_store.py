"""The setup tracker's SQLite mirror (assessment packet F3, step 1).

Shadow only: the JSON stays authoritative and every reader still loads it. What
this step must prove is parity - the mirror reproduces the payload exactly, a
second save rewrites only what changed, a narrowed read returns the right
records, and a failure in the mirror never reaches the scanner's save.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tracker_store import HEADER_FIELDS, SECTION_FIELDS, TrackerStore, mirror_payload  # noqa: E402


def _record(symbol: str, scan_date: str, score: float = 1.0) -> dict:
    return {
        "setup_id": f"{scan_date}:{symbol}:LONG:2026-06-01:favorite_setup",
        "symbol": symbol,
        "side": "LONG",
        "scan_date": scan_date,
        "priority_score": score,
        "setup_tags": ["a", "b"],
        "scenarios": {"stop_sma": {"r": 1.5, "path": [1, 2, 3]}},
        "feature_row": {"atr": 2.5, "note": "ünïcode"},
        "nested": {"none": None, "flag": True},
    }


def _payload() -> dict:
    return {
        "schema_version": 2,
        "updated_at": "2026-09-03T07:15:53",
        "data_session": "2026-09-02",
        "daily_watchlists": {"2026-09-02": {"symbols": ["AAPL", "MSFT"]}},
        "setups": {
            "2026-09-02:AAPL:LONG:2026-06-01:favorite_setup": _record("AAPL", "2026-09-02"),
            "2026-09-02:MSFT:LONG:2026-06-01:favorite_setup": _record("MSFT", "2026-09-02"),
            "2026-09-01:AAPL:LONG:2026-06-01:favorite_setup": _record("AAPL", "2026-09-01", 0.5),
        },
        "control_setups": {"control:2026-09-02:XOM:SHORT": _record("XOM", "2026-09-02")},
        "study_setups": {},
        "stats": [{"family": "x", "n": 3}],
        "setup_type_stats": [],
        "attribute_registry": {"levels.current_band_zone": {"kind": "cat"}},
    }


def test_round_trip_is_exact(tmp_path):
    store = TrackerStore(tmp_path / "tracker.sqlite")
    payload = _payload()
    report = store.save_payload(payload)
    assert report.records_seen == 4 and report.records_written == 4 and report.records_deleted == 0
    assert report.sections_written == len(SECTION_FIELDS)

    back = store.load_payload()
    assert back is not None
    for name in HEADER_FIELDS + SECTION_FIELDS:
        assert back[name] == payload[name], name
    assert back["setups"] == payload["setups"]
    assert back["control_setups"] == payload["control_setups"]
    assert back["study_setups"] == {}
    assert json.dumps(back, sort_keys=True) == json.dumps(payload, sort_keys=True)
    assert store.verify(payload).ok


def test_a_second_save_rewrites_only_what_changed(tmp_path):
    store = TrackerStore(tmp_path / "tracker.sqlite")
    payload = _payload()
    store.save_payload(payload)

    payload["setups"]["2026-09-02:AAPL:LONG:2026-06-01:favorite_setup"]["priority_score"] = 9.9
    del payload["setups"]["2026-09-01:AAPL:LONG:2026-06-01:favorite_setup"]
    payload["setups"]["2026-09-03:NVDA:LONG:2026-06-01:favorite_setup"] = _record("NVDA", "2026-09-03")
    payload["updated_at"] = "2026-09-04T07:15:53"

    report = store.save_payload(payload)
    assert report.records_seen == 4
    assert report.records_written == 2, "the changed AAPL row and the new NVDA row, nothing else"
    assert report.records_deleted == 1
    assert report.sections_written == 0, "no small section changed"
    assert store.verify(payload).ok
    assert store.load_payload()["updated_at"] == "2026-09-04T07:15:53"


def test_narrowed_reads_return_only_the_asked_for_records(tmp_path):
    store = TrackerStore(tmp_path / "tracker.sqlite")
    store.save_payload(_payload())
    aapl = store.load_records("setups", symbols=["aapl"])
    assert sorted(aapl) == [
        "2026-09-01:AAPL:LONG:2026-06-01:favorite_setup",
        "2026-09-02:AAPL:LONG:2026-06-01:favorite_setup",
    ]
    today = store.load_records("setups", scan_dates=["2026-09-02"])
    assert len(today) == 2 and all(row["scan_date"] == "2026-09-02" for row in today.values())
    both = store.load_records("setups", symbols=["AAPL"], scan_dates=["2026-09-02"])
    assert list(both) == ["2026-09-02:AAPL:LONG:2026-06-01:favorite_setup"]
    assert store.load_records("control_setups") and not store.load_records("study_setups")
    assert store.counts() == {"control_setups": 1, "setups": 3}


def test_verify_names_every_difference(tmp_path):
    store = TrackerStore(tmp_path / "tracker.sqlite")
    payload = _payload()
    store.save_payload(payload)
    truth = _payload()
    truth["setups"]["2026-09-02:MSFT:LONG:2026-06-01:favorite_setup"]["priority_score"] = 3.0
    truth["setups"]["2026-09-03:TSLA:LONG:2026-06-01:favorite_setup"] = _record("TSLA", "2026-09-03")
    del truth["setups"]["2026-09-01:AAPL:LONG:2026-06-01:favorite_setup"]
    truth["data_session"] = "2026-09-03"
    report = store.verify(truth)
    assert not report.ok
    assert report.differing == ["setups:2026-09-02:MSFT:LONG:2026-06-01:favorite_setup"]
    assert report.missing_in_db == ["setups:2026-09-03:TSLA:LONG:2026-06-01:favorite_setup"]
    assert report.extra_in_db == ["setups:2026-09-01:AAPL:LONG:2026-06-01:favorite_setup"]
    assert report.header_differences == ["data_session"]
    assert report.differences == 4


def test_the_mirror_hook_never_raises_and_honours_the_setting(tmp_path, monkeypatch):
    import tracker_store

    # A bad path: the hook logs and returns None, the caller's save is untouched.
    assert mirror_payload(_payload(), path=tmp_path / "missing-dir\0bad") is None
    # The setting turns the mirror off entirely.
    monkeypatch.setattr(tracker_store, "shadow_enabled", lambda: False)
    assert mirror_payload(_payload(), path=tmp_path / "off.sqlite") is None
    assert not (tmp_path / "off.sqlite").exists()
    monkeypatch.setattr(tracker_store, "shadow_enabled", lambda: True)
    report = mirror_payload(_payload(), path=tmp_path / "on.sqlite")
    assert report is not None and report.records_written == 4


def test_the_scanner_save_mirrors_after_the_json_write(tmp_path, monkeypatch):
    """The hook sits right after `save_json(SETUP_TRACKER_FILE, ...)` and reads
    the SAME payload; a mirror failure is a warning, never a failed save."""
    import master_avwap_lib.legacy as legacy
    import tracker_store

    json_path = tmp_path / "tracker.json"
    db_path = tmp_path / "tracker.sqlite"
    monkeypatch.setattr(legacy, "SETUP_TRACKER_FILE", json_path)
    monkeypatch.setattr(legacy, "_setup_tracker_backup_path", lambda: tmp_path / "tracker.json.bak")
    monkeypatch.setattr(legacy, "_append_setup_tracker_events", lambda payload: {})
    monkeypatch.setattr(legacy, "save_setup_tracker_scoring_payload", lambda payload: None)
    monkeypatch.setattr(tracker_store, "default_store_path", lambda: db_path)
    monkeypatch.setattr(tracker_store, "shadow_enabled", lambda: True)

    payload = _payload()
    legacy.save_setup_tracker_payload(payload, data_session="2026-09-03")

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["data_session"] == "2026-09-03"
    mirrored = TrackerStore(db_path).load_payload()
    assert mirrored is not None
    assert json.dumps(mirrored, sort_keys=True) == json.dumps(written, sort_keys=True)

    # The mirror breaking must not break the save.
    def explode(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(tracker_store.TrackerStore, "save_payload", explode)
    payload["setups"]["2026-09-03:NVDA:LONG:2026-06-01:favorite_setup"] = _record("NVDA", "2026-09-03")
    legacy.save_setup_tracker_payload(payload, data_session="2026-09-03")
    assert "NVDA" in json_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("command", ["counts", "verify"])
def test_the_cli_reads_a_named_pair(tmp_path, command, capsys):
    from tracker_store import _main

    json_path = tmp_path / "t.json"
    json_path.write_text(json.dumps(_payload()), encoding="utf-8")
    db_path = tmp_path / "t.sqlite"
    assert _main(["mirror", "--json", str(json_path), "--db", str(db_path)]) == 0
    assert _main([command, "--json", str(json_path), "--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert '"records"' in out or '"ok": true' in out
