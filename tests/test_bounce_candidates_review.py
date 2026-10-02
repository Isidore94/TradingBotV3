"""Review round on the lossless candidates clean-up (2026-10-02).

BLOCKER: an interrupted removal whose removed rows were all at the END left
the old file looking like the new one (the new file is a byte prefix of the
old), so the removed rows came back twice. The removal now records the old
file too and counts as landed only when the live file is provably not it.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
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

FIELDS = ["event_id", "event_type", "logged_at", "trade_date", "symbol", "levels_json"]
TODAY = datetime.now().date()


def _day(days_ago: int) -> str:
    return (TODAY - timedelta(days=days_ago)).isoformat()


def _write(path: Path, rows: list[tuple[str, str]], start: int = 0) -> None:
    """rows: (event_type, trade_date) in file order."""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for i, (event_type, trade_date) in enumerate(rows, start):
            writer.writerow({"event_id": f"E{i}", "event_type": event_type, "logged_at": f"T{i}",
                             "trade_date": trade_date, "symbol": "AAA",
                             "levels_json": json.dumps({"i": i, "s": 'a "b", c'})})


def _ref(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def _store(path: Path):
    return learning.candidates_archive_store(path)


RECENT = [("confirmed", _day(1))] * 5
OLD_NEAR_MISS = [("near_miss", _day(40))] * 3


def _interrupt_the_replace(monkeypatch, path: Path, exc: BaseException):
    real = arc.os.replace

    def replace(src, dst):
        if Path(dst) == path:
            raise exc
        return real(src, dst)

    monkeypatch.setattr(arc.os, "replace", replace)
    return real


@pytest.mark.parametrize("layout", ["removed_at_end", "removed_at_head"])
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(1)])
def test_an_interrupted_replace_never_doubles_or_loses_rows(tmp_path, monkeypatch, layout, interrupt):
    path = tmp_path / "c.csv"
    rows = RECENT + OLD_NEAR_MISS if layout == "removed_at_end" else OLD_NEAR_MISS + RECENT
    _write(path, rows)
    original = _ref(path)
    real = _interrupt_the_replace(monkeypatch, path, interrupt)
    with pytest.raises(type(interrupt)):
        learning.compact_bounce_candidates_csv(path, min_bytes_to_bother=1)
    monkeypatch.setattr(arc.os, "replace", real)
    assert len(_ref(path)) == 8  # the replace never happened
    assert arc.load_manifest(_store(path).archive_dir).get("pending_live_map")
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), original)
    assert arc.verify(store=_store(path), deep=True)["ok"] is True
    # The next clean-up settles the pending entry as "did not land", packs
    # nothing twice, and removes the rows for real.
    result = learning.compact_bounce_candidates_csv(path, min_bytes_to_bother=1)
    assert result["compacted"] is True and result["archived"] == 0 and result["dropped"] == 3
    assert len(_ref(path)) == 5
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), original)


@pytest.mark.parametrize("layout", ["removed_at_end", "removed_at_head"])
def test_a_replace_that_landed_before_the_crash_is_still_recognised(tmp_path, monkeypatch, layout):
    path = tmp_path / "c.csv"
    rows = RECENT + OLD_NEAR_MISS if layout == "removed_at_end" else OLD_NEAR_MISS + RECENT
    _write(path, rows)
    original = _ref(path)
    real = arc.os.replace

    def replace_then_die(src, dst):
        real(src, dst)
        if Path(dst) == path:
            raise KeyboardInterrupt  # the interrupt lands just after the call returned

    monkeypatch.setattr(arc.os, "replace", replace_then_die)
    with pytest.raises(KeyboardInterrupt):
        learning.compact_bounce_candidates_csv(path, min_bytes_to_bother=1)
    monkeypatch.setattr(arc.os, "replace", real)
    assert len(_ref(path)) == 5
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), original)
    _write(path, [("confirmed", _day(0))], start=8)
    expected = pd.concat([original, _ref(path).tail(1)], ignore_index=True)
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), expected)
    learning.compact_bounce_candidates_csv(path, min_bytes_to_bother=1)
    pd.testing.assert_frame_equal(arc.read_history(store=_store(path)), expected)


# ---------------------------------------------------------------------------
# Advisory 2: the kept-rows re-comparison is load-bearing
# ---------------------------------------------------------------------------


def test_a_byte_copy_that_alters_a_kept_row_removes_nothing(tmp_path, monkeypatch):
    path = tmp_path / "c.csv"
    _write(path, OLD_NEAR_MISS + RECENT)
    before = path.read_bytes()
    real = arc._read_record
    seen = {"n": 0}

    def corrupting(handle):
        record = real(handle)
        seen["n"] += 1
        if record is not None and seen["n"] == 7:  # header, 3 removed rows, then a kept one
            record = record.replace(b"AAA", b"AAB")
        return record

    monkeypatch.setattr(arc, "_read_record", corrupting)
    result = learning.compact_bounce_candidates_csv(path, min_bytes_to_bother=1)
    assert result["compacted"] is False and "nothing removed" in result["reason"]
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# Advisory 3: failure paths are refusals, never a broken startup
# ---------------------------------------------------------------------------


def test_a_ragged_row_is_a_refusal_not_an_exception(tmp_path):
    path = tmp_path / "c.csv"
    _write(path, OLD_NEAR_MISS + RECENT)
    with path.open("a", newline="", encoding="utf-8") as handle:
        handle.write("E99,near_miss,T99\r\n")  # fewer cells than the header
    before = path.read_bytes()
    result = learning.compact_bounce_candidates_csv(path, min_bytes_to_bother=1)
    assert result["compacted"] is False
    assert "could not be read" in result["reason"] and "nothing removed" in result["reason"]
    assert path.read_bytes() == before


def test_the_maintenance_still_refreshes_learning_after_a_refused_clean_up(tmp_path, monkeypatch):
    from bounce_bot_lib import legacy

    path = tmp_path / "c.csv"
    _write(path, OLD_NEAR_MISS + RECENT)
    with path.open("a", newline="", encoding="utf-8") as handle:
        handle.write("E99,near_miss,T99\r\n")
    refreshed = []
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_CANDIDATES_CSV", path)
    real = learning.compact_bounce_candidates_csv
    monkeypatch.setattr(learning, "compact_bounce_candidates_csv",
                        lambda p, **k: real(p, **{**k, "min_bytes_to_bother": 1}))
    monkeypatch.setattr(learning, "refresh_bounce_learning_if_stale", lambda: refreshed.append(True) or False)
    legacy.BounceBot._run_bounce_learning_maintenance(SimpleNamespace())
    assert refreshed == [True]


def test_a_byte_order_mark_is_named_in_the_reason(tmp_path):
    path = tmp_path / "c.csv"
    _write(path, OLD_NEAR_MISS + RECENT)
    path.write_bytes("﻿".encode("utf-8") + path.read_bytes())
    before = path.read_bytes()
    result = learning.compact_bounce_candidates_csv(path, min_bytes_to_bother=1)
    assert result["compacted"] is False and "byte-order mark" in result["reason"]
    assert path.read_bytes() == before


def _bot():
    from bounce_bot_lib.legacy import BounceBot

    bot = SimpleNamespace()
    bot._learning_csv_header = lambda p, f: BounceBot._learning_csv_header(bot, p, f)
    bot.append = lambda p, f, r: BounceBot._append_learning_row(bot, p, f, r)
    return bot


def _row(i: int) -> dict:
    return {"event_id": f"E{i}", "event_type": "confirmed", "logged_at": f"T{i}", "trade_date": _day(0),
            "symbol": "AAA", "levels_json": "{}"}


def test_broken_lock_infrastructure_falls_back_to_the_unlocked_write(tmp_path, monkeypatch, caplog):
    import local_writer_lock as lwl

    def broken(key, **kwargs):
        raise PermissionError("cannot create the lock directory in %TEMP%")

    monkeypatch.setattr(lwl, "local_writer_lock", broken)
    path = tmp_path / "c.csv"
    bot = _bot()
    with caplog.at_level(logging.WARNING):
        bot.append(path, FIELDS, _row(0))
        bot.append(path, FIELDS, _row(1))
    assert _ref(path)["event_id"].tolist() == ["E0", "E1"]
    warnings = [r for r in caplog.records if "lock" in r.getMessage().lower()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING


def test_the_first_park_is_a_warning_with_the_reason(tmp_path, monkeypatch, caplog):
    import local_writer_lock as lwl

    def busy(key, **kwargs):
        raise lwl.LocalLockUnavailable("held by the candidates clean-up")

    monkeypatch.setattr(lwl, "local_writer_lock", busy)
    path = tmp_path / "c.csv"
    bot = _bot()
    with caplog.at_level(logging.WARNING):
        bot.append(path, FIELDS, _row(0))
        bot.append(path, FIELDS, _row(1))
    parked = [r for r in caplog.records if r.levelno == logging.WARNING and "held by the candidates clean-up" in r.getMessage()]
    assert len(parked) == 1
    assert not path.exists()


def test_parked_rows_are_flushed_at_interpreter_exit(tmp_path, monkeypatch):
    import atexit

    import local_writer_lock as lwl

    registered = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.append((fn, a, k)))
    real = lwl.local_writer_lock
    state = {"busy": True}

    def maybe_busy(key, **kwargs):
        if state["busy"]:
            raise lwl.LocalLockUnavailable("held by the candidates clean-up")
        return real(key, **kwargs)

    monkeypatch.setattr(lwl, "local_writer_lock", maybe_busy)
    path = tmp_path / "c.csv"
    bot = _bot()
    bot.append(path, FIELDS, _row(0))
    bot.append(path, FIELDS, _row(1))
    assert len(registered) == 1 and not path.exists()
    state["busy"] = False
    fn, args, kwargs = registered[0]
    fn(*args, **kwargs)  # what interpreter exit runs
    assert _ref(path)["event_id"].tolist() == ["E0", "E1"]
