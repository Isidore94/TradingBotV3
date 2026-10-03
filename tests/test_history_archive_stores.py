"""Spec-driven history archive: one core, several append-only CSV stores.

The bounce outcomes CSV is the first store whose writer takes NO lock
(`bounce_bot_lib.legacy._append_learning_row`: `open("a")` + one
`csv.DictWriter.writerow`). For such a store the archive reads a length
snapshot cut at the last complete record, never a torn last line, and trim
refuses, always.
"""

from __future__ import annotations

import csv
import json
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import d1_feature_history_archive as arc  # noqa: E402

FIELDS = ["schema_version", "event_id", "event_type", "logged_at", "trade_date", "symbol", "close_r", "context_json"]


def _row(i: int, trade_date: str, event_type: str = "milestone") -> dict:
    return {
        "schema_version": "3",
        "event_id": f"E{i // 3}",
        "event_type": event_type,
        "logged_at": f"{trade_date}T10:{i % 60:02d}:00",
        "trade_date": trade_date,
        "symbol": ["AAA", "NA", "None"][i % 3],
        "close_r": ["1.50", "", "-0.25"][i % 3],
        "context_json": json.dumps({"note": 'say "hi", ok', "levels": [1.5, 2], "i": i}),
    }


def _append(path: Path, rows: list[dict], fields: list[str] = FIELDS) -> None:
    """Exactly what `_append_learning_row` does: open("a") + DictWriter.writerow."""
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _widen(path: Path, fields: list[str]) -> None:
    """Exactly what `_learning_csv_header` does when the code gains a column."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, restkey="_extra"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _ref(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


DATES = ["2026-07-30", "2026-07-31", "2026-08-03", "2026-09-28", "2026-10-01", "2026-09-30", "2026-10-02"]


@pytest.fixture
def bounce(tmp_path):
    csv_path = tmp_path / "intraday_bounce_outcomes.csv"
    rows = [_row(i, DATES[i % len(DATES)]) for i in range(40)]  # dates deliberately out of order
    _append(csv_path, rows)
    spec = arc.registered_stores()["intraday_bounce_outcomes"].with_paths(csv_path, tmp_path / "arc")
    return spec


def test_the_registry_holds_d1_the_outcomes_and_the_candidates():
    # 2026-10-02 (trader's yes): the candidates joined once their startup
    # clean-up went through proven removal; this pin said "not the candidates".
    import project_paths as pp

    stores = arc.registered_stores()
    assert set(stores) == {"d1_features_history", "intraday_bounce_outcomes", "intraday_bounce_candidates"}
    d1 = stores["d1_features_history"]
    assert d1.csv_path == pp.D1_FEATURES_HISTORY_FILE and d1.archive_dir == pp.D1_FEATURES_HISTORY_ARCHIVE_DIR
    assert d1.date_column == "run_date" and d1.key_columns == ("run_id", "symbol", "side")
    assert d1.writer_lock_key is not None and d1.trim_setting == "d1_history_trim_enabled"
    out = stores["intraday_bounce_outcomes"]
    assert out.csv_path == pp.INTRADAY_BOUNCE_OUTCOMES_FILE
    assert out.archive_dir == pp.INTRADAY_BOUNCE_OUTCOMES_ARCHIVE_DIR
    assert out.archive_dir.parent == out.csv_path.parent
    assert out.date_column == "trade_date"
    assert out.key_columns == ("event_id", "event_type", "logged_at")
    assert out.writer_lock_key is None and out.trim_setting == ""
    cand = stores["intraday_bounce_candidates"]
    assert cand.csv_path == pp.INTRADAY_BOUNCE_CANDIDATES_FILE
    assert cand.archive_dir == pp.INTRADAY_BOUNCE_CANDIDATES_ARCHIVE_DIR
    assert cand.date_column == "trade_date" and cand.key_columns == ("event_id", "event_type", "logged_at")
    assert cand.writer_lock_key is not None and cand.trim_setting == ""


def test_a_no_lock_store_archives_and_reads_back_exactly(bounce):
    original = _ref(bounce.csv_path)
    result = arc.archive(store=bounce)
    assert result["archived_rows"] == 40
    manifest = arc.load_manifest(bounce.archive_dir)
    assert sorted(manifest["files"]) == ["2026-07", "2026-08", "2026-09", "2026-10"]
    pd.testing.assert_frame_equal(arc.read_history(store=bounce), original)
    got = arc.read_history(store=bounce, columns=["event_id", "context_json"], since="2026-09-30")
    mask = original["trade_date"] >= "2026-09-30"
    pd.testing.assert_frame_equal(got, original.loc[mask, ["event_id", "context_json"]].reset_index(drop=True))
    assert arc.verify(store=bounce)["ok"] is True
    assert arc.archive(store=bounce)["archived_rows"] == 0


def test_a_torn_last_line_is_never_archived_or_read(bounce):
    original = _ref(bounce.csv_path)
    complete = _row(99, "2026-10-02")
    buffer = []
    with bounce.csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writerow(complete)
    text = bounce.csv_path.read_bytes()
    last_start = text.rstrip(b"\r\n").rfind(b"\n") + 1
    torn_tail = text[last_start:]
    bounce.csv_path.write_bytes(text[: last_start] + torn_tail[: len(torn_tail) // 2])  # half a row, mid-JSON
    assert arc.archive(store=bounce)["archived_rows"] == 40
    pd.testing.assert_frame_equal(arc.read_history(store=bounce), original)
    assert arc.verify(store=bounce)["ok"] is True
    with bounce.csv_path.open("ab") as handle:  # the writer finishes its line
        handle.write(torn_tail[len(torn_tail) // 2:])
    assert arc.archive(store=bounce)["archived_rows"] == 1
    pd.testing.assert_frame_equal(arc.read_history(store=bounce), _ref(bounce.csv_path))
    assert buffer == []


def test_an_append_during_the_archive_is_left_for_the_next_run(bounce, monkeypatch):
    real = arc._archive_table
    calls = {"n": 0}

    def append_mid_pack(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            _append(bounce.csv_path, [_row(500, "2026-10-02"), _row(501, "2026-10-02")])
        return real(*args, **kwargs)

    monkeypatch.setattr(arc, "_archive_table", append_mid_pack)
    assert arc.archive(store=bounce)["archived_rows"] == 40
    monkeypatch.setattr(arc, "_archive_table", real)
    assert arc.verify(store=bounce)["ok"] is True
    assert arc.archive(store=bounce)["archived_rows"] == 2
    pd.testing.assert_frame_equal(arc.read_history(store=bounce), _ref(bounce.csv_path))


def test_a_no_lock_store_takes_only_the_archive_lock(bounce, monkeypatch):
    from local_writer_lock import lock_key_for_path

    keys = []
    real = arc.local_writer_lock

    @contextmanager
    def spy(key, **kwargs):
        keys.append(key)
        with real(key, **kwargs) as info:
            yield info

    monkeypatch.setattr(arc, "local_writer_lock", spy)
    arc.archive(store=bounce)
    assert lock_key_for_path(bounce.csv_path) not in keys
    assert keys == [lock_key_for_path(bounce.archive_dir / arc.MANIFEST_NAME)]


def test_trim_always_refuses_a_store_whose_writer_takes_no_lock(bounce):
    arc.archive(store=bounce)
    before = bounce.csv_path.read_bytes()
    for apply in (False, True):
        with pytest.raises(arc.ArchiveRefused, match="no lock"):
            arc.trim(store=bounce, today=date(2027, 1, 1), apply=apply, keep_days=0)
    assert bounce.csv_path.read_bytes() == before


def test_the_writers_in_place_header_widening_stays_lossless(bounce):
    arc.archive(store=bounce)
    wider = FIELDS + ["outcome_mode", "eod_close"]
    _widen(bounce.csv_path, wider)
    _append(bounce.csv_path, [{**_row(700, "2026-10-02"), "outcome_mode": "eod", "eod_close": "12.30"}], wider)
    widened = _ref(bounce.csv_path)
    assert list(widened.columns) == wider
    assert arc.verify(store=bounce)["ok"] is True
    assert arc.archive(store=bounce)["archived_rows"] == 1
    # The csv-module rewrite keeps every cell's text, so the read equals the file exactly.
    pd.testing.assert_frame_equal(arc.read_history(store=bounce), widened)


def test_the_cli_takes_a_store_name_and_refuses_trim_for_a_no_lock_store(bounce, capsys):
    base = ["--store", bounce.name, "--csv", str(bounce.csv_path), "--archive-dir", str(bounce.archive_dir)]
    assert arc.main(["archive", *base]) == 0
    assert arc.main(["verify", *base]) == 0
    assert arc.main(["status", *base]) == 0
    assert '"store": "intraday_bounce_outcomes"' in capsys.readouterr().out
    before = bounce.csv_path.read_bytes()
    assert arc.main(["trim", *base, "--apply"]) == 2
    assert "no lock" in capsys.readouterr().err
    assert bounce.csv_path.read_bytes() == before
    assert arc.main(["status", "--store", "no_such_store"]) == 2


def test_the_cli_default_store_is_still_the_d1_history(tmp_path, capsys):
    csv_path = tmp_path / "h.csv"
    pd.DataFrame([{"run_id": "r", "run_date": "2026-09-01", "symbol": "A", "side": "LONG"}]).to_csv(
        csv_path, index=False
    )
    assert arc.main(["archive", "--csv", str(csv_path), "--archive-dir", str(tmp_path / "a")]) == 0
    capsys.readouterr()
    assert arc.main(["status", "--csv", str(csv_path), "--archive-dir", str(tmp_path / "a")]) == 0
    assert '"store": "d1_features_history"' in capsys.readouterr().out
