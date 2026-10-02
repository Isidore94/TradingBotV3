"""Second review round on the history archive (2026-10-02).

1. The key check has its own narrow equivalence: exact text, or live "" for
   archived pandas-NA text (the writer's rewrite direction only). Float-equal
   keys such as '0' / '0.0' or '1' / ' 1' are refused.
2. A no-lock store whose file changes under the read (the bounce writer's
   non-atomic `open("w")` header rewrite) is refused with nothing written,
   never an ArrowInvalid traceback; a locked store's corrupt file still fails
   loudly as before.
3. `read_history` on a no-lock store notices a change under the read, retries,
   and raises rather than return a short frame.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import d1_feature_history_archive as arc  # noqa: E402

FIELDS = ["event_id", "event_type", "logged_at", "trade_date", "symbol", "context_json"]


def _ref(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def _snapshot(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}


# ---------------------------------------------------------------------------
# 1. narrow key equivalence
# ---------------------------------------------------------------------------


@pytest.fixture
def d1(tmp_path):
    csv_path = tmp_path / "d1.csv"
    pd.DataFrame(
        [
            {"run_id": "r1", "run_date": "2026-08-03", "symbol": sym, "side": "LONG", "x": "1"}
            for sym in ("0", "1", "AAA")
        ]
    ).to_csv(csv_path, index=False)
    return {"csv_path": csv_path, "archive_dir": tmp_path / "arc"}


@pytest.mark.parametrize("old,new", [("0", "0.0"), ("1", " 1")])
def test_a_float_equal_key_is_refused(d1, old, new):
    arc.archive(**d1)
    frame = _ref(d1["csv_path"])
    frame.loc[frame["symbol"] == old, "symbol"] = new
    frame.to_csv(d1["csv_path"], index=False)
    assert arc.verify(**d1)["ok"] is False
    with pytest.raises(arc.VerifyFailed):
        arc.archive(**d1)


def test_the_key_equivalence_is_one_directional():
    # live "" for archived NA text: the writer's rewrite. Nothing else.
    assert arc._keys_match("", "NA") and arc._keys_match("", "None") and arc._keys_match("", None)
    assert arc._keys_match("AAA", "AAA")
    for live, archived in [("0.0", "0"), (" 1", "1"), ("1e0", "1"), ("inf", "INF"), ("NaN", "nan"),
                           ("None", ""), ("NA", "")]:
        assert not arc._keys_match(live, archived), (live, archived)
    # The trim proof keeps its wider equivalence.
    assert arc._cells_prove("0.0", "0") and arc._cells_prove("1.5", "1.50")


# ---------------------------------------------------------------------------
# 2 and 3. a no-lock store whose file changes under the read
# ---------------------------------------------------------------------------


def _write(path: Path, rows: int, fields=FIELDS) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(rows):
            writer.writerow({"event_id": f"E{i}", "event_type": "final", "logged_at": f"t{i}",
                             "trade_date": "2026-09-0" + str(1 + i % 5), "symbol": "AAA",
                             "context_json": '{"a": "b, c"}'})


@pytest.fixture
def bounce(tmp_path):
    csv_path = tmp_path / "outcomes.csv"
    _write(csv_path, 60)
    return arc.registered_stores()["intraday_bounce_outcomes"].with_paths(csv_path, tmp_path / "arc")


def _rewrite_in_progress(path: Path) -> bytes:
    """Leave the file as `open("w")` would mid-rewrite: a new header, half the rows."""
    full = path.read_bytes()
    lines = full.split(b"\r\n")
    partial = b"\r\n".join([lines[0] + b",outcome_mode", *lines[1:30]]) + b"\r\nE30,fin"
    path.write_bytes(partial)
    return full


def test_a_rewrite_under_the_archive_is_refused_with_nothing_written(bounce, monkeypatch, capsys):
    arc.archive(store=bounce)
    _write(bounce.csv_path, 80)  # 20 new rows
    before = _snapshot(bounce.archive_dir)
    real = arc._archive_table
    state = {"done": False}

    def rewrite_mid_pack(*args, **kwargs):
        if not state["done"]:
            state["done"] = True
            _rewrite_in_progress(bounce.csv_path)
        return real(*args, **kwargs)

    monkeypatch.setattr(arc, "_archive_table", rewrite_mid_pack)
    with pytest.raises(arc.ArchiveRefused, match="changed during the read"):
        arc.archive(store=bounce)
    assert _snapshot(bounce.archive_dir) == before


def test_an_arrow_parse_error_on_a_no_lock_store_is_a_refusal_in_the_cli(bounce, monkeypatch, capsys):
    arc.archive(store=bounce)
    _write(bounce.csv_path, 80)
    before = _snapshot(bounce.archive_dir)

    def torn(*args, **kwargs):
        raise pa.ArrowInvalid("CSV parse error: Expected 6 columns, got 3")
        yield  # pragma: no cover

    monkeypatch.setattr(arc, "_iter_source", torn)
    base = ["--store", bounce.name, "--csv", str(bounce.csv_path), "--archive-dir", str(bounce.archive_dir)]
    assert arc.main(["archive", *base]) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "changed during the read" in err
    assert _snapshot(bounce.archive_dir) == before
    report = arc.verify(store=bounce)
    assert report["ok"] is False and any("changed during the read" in p for p in report["problems"])


def test_a_corrupt_file_on_a_locked_store_still_fails_loudly(tmp_path):
    csv_path = tmp_path / "d1.csv"
    csv_path.write_text("run_id,run_date,symbol,side\nr1,2026-08-03,AAA,LONG\nr1,2026-08-03,BBB,LONG,extra\n",
                        encoding="utf-8")
    with pytest.raises(pa.ArrowInvalid):
        arc.archive(csv_path=csv_path, archive_dir=tmp_path / "arc")


def test_read_history_retries_a_read_that_ran_into_a_rewrite(bounce, monkeypatch):
    arc.archive(store=bounce)
    original = _ref(bounce.csv_path)
    real = arc._iter_csv
    calls = {"n": 0}

    def rewrite_during_first_read(path, header, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            full = _rewrite_in_progress(path)
            try:
                batches = list(real(path, header, *args, **kwargs))
            except pa.ArrowInvalid:
                batches = []
            path.write_bytes(full)
            yield from batches
            return
        yield from real(path, header, *args, **kwargs)

    monkeypatch.setattr(arc, "_iter_csv", rewrite_during_first_read)
    pd.testing.assert_frame_equal(arc.read_history(store=bounce), original)
    assert calls["n"] >= 2


def test_read_history_raises_rather_than_return_a_short_frame(bounce):
    arc.archive(store=bounce)
    _write(bounce.csv_path, 25)  # the live file lost rows the archive says it holds
    with pytest.raises(arc.ArchiveError):
        arc.read_history(store=bounce)
