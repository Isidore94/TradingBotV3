"""Lossless candidates clean-up (trader 2026-10-02: "lose nothing", "yes to both").

`compact_bounce_candidates_csv` still shrinks the live file by the same rule
(near_miss older than 30 days, everything older than 365, rows with no date),
but a row leaves only after it is packed and proven in the candidates archive,
and `read_history` on the store equals the file as it would have been with no
clean-up ever: every row, every column, exact text, original order.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import d1_feature_history_archive as arc  # noqa: E402
from bounce_bot_lib import learning  # noqa: E402

FIELDS = ["schema_version", "event_id", "event_type", "logged_at", "trade_date", "symbol", "score", "levels_json"]
TODAY = datetime.now().date()


def _day(days_ago: int) -> str:
    return (TODAY - timedelta(days=days_ago)).isoformat()


def _rows(start: int, n: int) -> list[dict]:
    # Interleaved like the real file: near_miss rows dominate, confirmed rows sit between.
    ages = [400, 200, 45, 31, 29, 5, 0]
    types = ["near_miss", "near_miss", "confirmed", "near_miss", "detected", "near_miss", "expired"]
    out = []
    for i in range(start, start + n):
        out.append({
            "schema_version": "2",
            "event_id": f"E{i}",
            "event_type": types[i % len(types)],
            "logged_at": f"T{i:06d}",
            "trade_date": "" if i % 23 == 0 else _day(ages[(i * 3) % len(ages)]),
            "symbol": ["AAA", "NA", "None"][i % 3],
            "score": ["1.50", "", "0.0"][i % 3],
            "levels_json": json.dumps({"vwap": 1.5, "note": 'a "b", c', "i": i}),
        })
    return out


def _append(path: Path, rows: list[dict]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ref(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


@pytest.fixture
def cands(tmp_path):
    path = tmp_path / "intraday_bounce_candidates.csv"
    shadow = tmp_path / "never_cleaned.csv"  # the file as it would be with no clean-up ever
    _append(path, _rows(0, 140))
    _append(shadow, _rows(0, 140))
    return path, shadow


def _compact(path: Path):
    return learning.compact_bounce_candidates_csv(path, min_bytes_to_bother=1)


def _store(path: Path):
    return learning.candidates_archive_store(path)


def _expected_live(shadow: Path) -> pd.DataFrame:
    frame = _ref(shadow)
    near = frame["event_type"].str.strip().str.lower() == "near_miss"
    cutoff = near.map({True: _day(30), False: _day(365)})
    keep = (frame["trade_date"] != "") & (frame["trade_date"] >= cutoff)
    return frame[keep].reset_index(drop=True)


def test_the_clean_up_removes_the_same_rows_and_loses_none(cands):
    path, shadow = cands
    result = _compact(path)
    assert result["compacted"] is True and result["archived"] == 140
    live = _ref(path)
    pd.testing.assert_frame_equal(live, _expected_live(shadow))
    assert result["dropped"] == 140 - len(live) > 0
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), _ref(shadow))
    assert arc.verify(store=_store(path))["ok"] is True
    manifest = arc.load_manifest(_store(path).archive_dir)
    assert manifest.get("live_map")  # rows went from the middle


def test_clean_up_append_clean_up_again_stays_lossless(cands):
    path, shadow = cands
    _compact(path)
    _append(path, _rows(140, 60))
    _append(shadow, _rows(140, 60))
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), _ref(shadow))
    result = _compact(path)
    assert result["archived"] == 60
    pd.testing.assert_frame_equal(_ref(path), _expected_live(shadow))
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), _ref(shadow))
    # A third run with nothing new is a no-op on the live file and the archive.
    before = path.read_bytes()
    third = _compact(path)
    assert third["dropped"] == 0 and third["archived"] == 0
    assert path.read_bytes() == before
    # Narrow and date-bounded reads honour the map too.
    got = arc.read_history(store=_store(path), columns=["event_id", "trade_date"], since=_day(45))
    full = _ref(shadow)
    mask = (full["trade_date"] != "") & (full["trade_date"] >= _day(45))
    pd.testing.assert_frame_equal(got, full.loc[mask, ["event_id", "trade_date"]].reset_index(drop=True))


def test_a_crash_after_the_replace_is_settled_and_loses_nothing(cands, monkeypatch):
    path, shadow = cands
    real = arc._write_manifest
    calls = {"n": 0}

    def die_on_commit(target, manifest):
        if "pending_live_map" not in manifest and calls["pending"]:
            raise OSError("power cut")
        calls["pending"] = "pending_live_map" in manifest
        return real(target, manifest)

    calls["pending"] = False
    monkeypatch.setattr(arc, "_write_manifest", die_on_commit)
    with pytest.raises(OSError):
        _compact(path)
    monkeypatch.setattr(arc, "_write_manifest", real)
    manifest = arc.load_manifest(_store(path).archive_dir)
    assert manifest.get("pending_live_map")
    pd.testing.assert_frame_equal(_ref(path), _expected_live(shadow))  # the replace landed
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), _ref(shadow))
    _append(path, _rows(140, 10))
    _append(shadow, _rows(140, 10))
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), _ref(shadow))
    _compact(path)  # settles the pending map under the locks, then packs and cleans
    manifest = arc.load_manifest(_store(path).archive_dir)
    assert not manifest.get("pending_live_map")
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), _ref(shadow))


def test_a_failed_replace_removes_nothing(cands, monkeypatch):
    path, shadow = cands
    real = arc.os.replace

    def refuse_csv(src, dst):
        if Path(dst) == path:
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr(arc.os, "replace", refuse_csv)
    result = _compact(path)
    monkeypatch.setattr(arc.os, "replace", real)
    assert result["compacted"] is False
    assert path.read_bytes() == shadow.read_bytes()
    assert not arc.load_manifest(_store(path).archive_dir).get("pending_live_map")
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), _ref(shadow))


def test_a_failed_verify_refuses_the_clean_up_and_the_bot_carries_on(cands):
    path, shadow = cands
    _compact(path)
    _append(path, _rows(140, 5))
    store = _store(path)
    manifest = arc.load_manifest(store.archive_dir)
    target = store.archive_dir / next(iter(manifest["files"].values()))["file"]
    target.write_bytes(target.read_bytes() + b"tamper")
    before = path.read_bytes()
    result = _compact(path)  # returns, never raises
    assert result["compacted"] is False and "nothing removed" in result["reason"]
    assert path.read_bytes() == before


def test_a_row_the_archive_does_not_prove_is_never_removed(cands, monkeypatch):
    path, _shadow = cands
    learning.compact_bounce_candidates_csv(path, min_bytes_to_bother=1, max_age_days=10_000,
                                           near_miss_keep_days=10_000)  # packs, removes nothing
    frame = _ref(path)
    victim = frame.index[frame["event_type"] == "near_miss"][0]
    frame.loc[victim, "levels_json"] = "edited after packing"
    with path.open("w", newline="", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False)
    before = path.read_bytes()
    result = _compact(path)
    assert result["compacted"] is False
    assert path.read_bytes() == before


def test_the_night_packs_the_candidates_and_never_shrinks_them(cands, tmp_path, monkeypatch):
    import project_paths as pp
    from ai_jobs import history_pack

    path, shadow = cands
    monkeypatch.setattr(pp, "get_local_setting", lambda key, default=None: True)  # every switch on
    assert "intraday_bounce_candidates" in arc.registered_stores()
    result = history_pack.run_history_pack(specs=[_store(path)], report_path=tmp_path / "r.json", stores={})
    assert result["status"] == "ok", result
    assert path.read_bytes() == shadow.read_bytes()
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    block = report["stores_packed"]["intraday_bounce_candidates"]
    assert block["archive"]["archived_rows"] == 140 and block["trim"]["enabled"] is False


def test_the_live_file_store_uses_the_registered_archive_folder():
    import project_paths as pp

    store = learning.candidates_archive_store(pp.INTRADAY_BOUNCE_CANDIDATES_FILE)
    assert store.archive_dir == pp.INTRADAY_BOUNCE_CANDIDATES_ARCHIVE_DIR
    assert store.writer_lock_key is not None


# ---------------------------------------------------------------------------
# The appender's writer lock (trader's yes covers `_append_learning_row`)
# ---------------------------------------------------------------------------


def _bot():
    from bounce_bot_lib.legacy import BounceBot

    bot = SimpleNamespace()
    bot._learning_csv_header = lambda p, f: BounceBot._learning_csv_header(bot, p, f)
    bot.append = lambda p, f, r: BounceBot._append_learning_row(bot, p, f, r)
    assert BounceBot.LEARNING_ROW_LOCK_TIMEOUT_SECONDS == 0.25
    return bot


def test_the_appender_takes_the_files_writer_lock(tmp_path, monkeypatch):
    import local_writer_lock as lwl

    path = tmp_path / "c.csv"
    keys = []
    real = lwl.local_writer_lock

    @contextmanager
    def spy(key, **kwargs):
        keys.append((key, kwargs.get("timeout_seconds")))
        with real(key, **kwargs) as info:
            yield info

    monkeypatch.setattr(lwl, "local_writer_lock", spy)
    _bot().append(path, FIELDS, _rows(0, 1)[0])
    assert keys == [(lwl.lock_key_for_path(path), 0.25)]
    assert len(_ref(path)) == 1


def test_a_row_that_misses_the_lock_waits_and_keeps_its_order(tmp_path, monkeypatch):
    import local_writer_lock as lwl

    path = tmp_path / "c.csv"
    bot = _bot()
    real = lwl.local_writer_lock
    busy = {"on": True}

    @contextmanager
    def maybe_busy(key, **kwargs):
        if busy["on"]:
            time.sleep(kwargs.get("timeout_seconds", 0))
            raise lwl.LocalLockUnavailable("held by the clean-up")
        with real(key, **kwargs) as info:
            yield info

    monkeypatch.setattr(lwl, "local_writer_lock", maybe_busy)
    rows = _rows(0, 4)
    started = time.monotonic()
    bot.append(path, FIELDS, rows[0])
    bot.append(path, FIELDS, rows[1])
    assert time.monotonic() - started < 2.0  # the bot thread is never stalled
    assert not path.exists()
    busy["on"] = False
    bot.append(path, FIELDS, rows[2])
    assert _ref(path)["event_id"].tolist() == ["E0", "E1", "E2"]


def test_an_append_waits_out_a_clean_up_holding_the_lock(cands):
    """End to end: the clean-up and the appender share the real lock."""
    import threading

    import local_writer_lock as lwl

    path, _shadow = cands
    bot = _bot()
    with lwl.local_writer_lock(lwl.lock_key_for_path(path)):
        worker = threading.Thread(target=bot.append, args=(path, FIELDS, _rows(500, 1)[0]))
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
        assert "E500" not in set(_ref(path)["event_id"])  # held back, not interleaved
    bot.append(path, FIELDS, _rows(501, 1)[0])
    assert _ref(path)["event_id"].tolist()[-2:] == ["E500", "E501"]


def test_the_copy_based_trial_helper_needs_no_live_store(tmp_path, cands):
    # Guard for the real-copy trial: a copied file gets its own archive folder.
    path, _shadow = cands
    copy = tmp_path / "copy.csv"
    shutil.copyfile(path, copy)
    store = learning.candidates_archive_store(copy)
    assert store.archive_dir == tmp_path / "copy_archive"
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == hashlib.sha256(path.read_bytes()).hexdigest()
