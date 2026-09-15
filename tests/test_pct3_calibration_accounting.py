r"""PCT-3: the calibration report counts every veto and names its population.

Added by the BUILDER in the fix round. The first review ran the report against
copies of the live stores and measured what it was actually doing:

> 213 unique compression vetoes (symbol, session, side) over 12 sessions; the
> CLI joined 86, because `build_rows` joins on tracker records whose session is
> `scan_date` - the session the setup was ENTERED - so a veto on a row shown
> days later finds nothing. The whole 2026-09-15 session (23 vetoes) vanished
> while the header printed "..09-14". The three anchor measures printed
> `n = 0 / nan` because no live record carries them yet.

A report that silently discards 60 % of its own evidence is worse than no
report: it turns "the measure disagrees with the eye" into a number nobody can
check. So this file pins the five rules that came out of that:

1. every veto lands in exactly one of `joined` / `untracked` / `pending`, and
   all three counts are printed;
2. a session's population is the records ACTIVE on it, not the records entered
   on it;
3. a session that is not complete is EXCLUDED and named, never silently
   dropped and never measured (`plan.md` sec 5: completed bars only);
4. an anchor measure the record does not carry is RECOMPUTED from the bar cache
   through `summarize_anchor_compression` - the same function, never a second
   copy - and a row whose anchor cannot be had says `anchor unknown` and still
   carries the four fixed-window measures;
5. the tracker is STREAMED: 1.26 GB live, and `read_text` on it peaked at
   3.79 GB.

`tests/conftest.py` points `project_paths` at a test directory, and every path
below is a `tmp_path`; nothing here reads or writes a live store.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


#: A session the trader vetoed on, complete.
SESSION = "2026-08-20"
#: The session the setups were ENTERED on - a week earlier. This gap is the
#: whole defect: joining on it dropped every veto on a name the trader had been
#: carrying.
ENTRY = "2026-08-13"
#: A session that has not closed yet.
PENDING_SESSION = "2026-09-15"
SINCE = "2026-08-01"


def _veto_row(symbol: str, session: str, code: str, version: int) -> dict:
    return {
        "schema_version": 1,
        "event_id": f"evt-{symbol}-{session}",
        "event_type": "veto",
        "symbol": symbol,
        "session_date": session,
        "created_at": f"{session}T09:40:00-07:00",
        "source": "chart_review",
        "reason_code": code,
        "vocab_version": version,
        "timeframe": "D1",
        "side": "LONG",
        "surface": "chart_review",
    }


def _veto_version(code: str) -> int:
    """Asked of the vocabulary, never typed in (`CLAUDE.md`)."""
    from ui.annotations.vocabulary import available_veto_versions, load_veto_vocabulary

    for version in sorted(available_veto_versions()):
        vocabulary = load_veto_vocabulary(version=version)
        if code in vocabulary.codes:
            return int(vocabulary.vocab_version)
    raise AssertionError(f"the build no longer carries {code}")


def _record(symbol: str, *, entry: str, last: str, measured: bool, anchor: str = "") -> dict:
    record = {
        "setup_id": f"{symbol}-LONG-{entry}",
        "symbol": symbol,
        "side": "LONG",
        "scan_date": entry,
        "entry_trade_date": entry,
        "last_replayed_session": last,
        "anchor_date": anchor or "2026-07-06",
        "entry_price": 103.0,
        "priority_bucket": "near_favorite_zone",
        "setup_status": "OPEN",
        "compression_flag": symbol == "AAA",
        "compression_penalty": 10 if symbol == "AAA" else 0,
        "compression_note": "",
    }
    if measured:
        record.update(
            {
                "compression_score": 3 if symbol == "AAA" else 1,
                "compression_stdev_atr_ratio": 0.40,
                "compression_range_atr_ratio": 1.0,
                "compression_close_range_atr_ratio": 0.5,
                "compression_rule_version": "anchor_compression_v1",
            }
        )
    return record


def _bar_rows(symbol: str, *, tight: bool, through: str) -> list[dict]:
    import pandas as pd

    sessions = pd.bdate_range(end=pd.Timestamp(through), periods=220)
    rows = []
    for index, stamp in enumerate(sessions):
        base = 100.0 + (index % 7) * 0.2
        half = 0.25 if (tight and index >= 190) else 2.5
        rows.append(
            {
                "datetime": stamp.date().isoformat(),
                "open": base,
                "high": base + half,
                "low": base - half,
                "close": base + half / 4.0,
                "volume": 1_000_000,
            }
        )
    return rows


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A scratch home with vetoes, a tracker and a bar cache. Three veto fates.

    * `AAA` - vetoed on `SESSION`, entered a week earlier, still carried -> joined
    * `BBB` - vetoed on `SESSION`, entered a week earlier, carries NO measure
      (an old record) -> joined, anchor recomputed
    * `CCC` - vetoed on `SESSION`, no record active at all -> untracked
    * `DDD` - never vetoed, active on `SESSION` -> the "rest" side
    * `EEE` - vetoed on `PENDING_SESSION`, which is not complete -> pending
    """
    import project_paths

    import compression_calibration

    home = tmp_path / "home"
    bars_dir = tmp_path / "bars"
    out_dir = tmp_path / "out"
    for directory in (home, bars_dir, out_dir):
        directory.mkdir(parents=True, exist_ok=True)

    annotations = home / "trader_annotations.jsonl"
    compressed = _veto_version("compressed")
    cluttered = _veto_version("support_resistance_cluttered")
    rows = [
        _veto_row("AAA", SESSION, "compressed", compressed),
        _veto_row("BBB", SESSION, "support_resistance_cluttered", cluttered),
        _veto_row("CCC", SESSION, "compressed", compressed),
        _veto_row("EEE", PENDING_SESSION, "compressed", compressed),
    ]
    # An UNCODED veto, exactly as `ui/annotations/store.py` writes one: the key
    # is present and empty. It is a veto and it is NOT a compression veto.
    uncoded = _veto_row("DDD", SESSION, "", compressed)
    uncoded["reason_code"] = ""
    uncoded.pop("vocab_version")
    rows.append(uncoded)
    annotations.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    tracker = home / "master_avwap_setup_tracker.json"
    tracker.write_text(
        json.dumps(
            {
                "saved_at": f"{SESSION}T13:05:00-07:00",
                "setups": {
                    record["setup_id"]: record
                    for record in (
                        _record("AAA", entry=ENTRY, last=SESSION, measured=True),
                        _record("BBB", entry=ENTRY, last=SESSION, measured=False, anchor=ENTRY),
                        _record("DDD", entry=ENTRY, last=SESSION, measured=True),
                        # Closed BEFORE the vetoed session: not in its population.
                        _record("ZZZ", entry="2026-07-01", last="2026-08-05", measured=True),
                    )
                },
                "control_setups": {},
                "study_setups": {},
                "stats": [],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    import csv as _csv

    for symbol, tight in (("AAA", True), ("BBB", True), ("CCC", True), ("DDD", False)):
        path = bars_dir / f"{symbol}.csv"
        rows_out = _bar_rows(symbol, tight=tight, through=SESSION)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = _csv.DictWriter(handle, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)

    monkeypatch.setattr(project_paths, "TRADER_ANNOTATIONS_FILE", annotations, raising=False)
    monkeypatch.setattr(
        project_paths, "MASTER_AVWAP_SETUP_TRACKER_FILE", tracker, raising=False
    )
    monkeypatch.setattr(project_paths, "DAILY_BARS_CACHE_DIR", bars_dir, raising=False)
    monkeypatch.setattr(
        compression_calibration, "session_is_complete", lambda session: session == SESSION
    )
    return out_dir


# ---------------------------------------------------------------------------
# Rule 1 and 3 - every veto is counted, and a pending session is named
# ---------------------------------------------------------------------------


def test_every_veto_lands_in_exactly_one_bucket_and_all_three_are_printed(scratch):
    import compression_calibration

    rows, summary = compression_calibration.build_rows(since=SINCE)
    assert summary["veto_count"] == 4, "the uncoded veto is not a compression veto"
    assert summary["joined"] == 2, "AAA and BBB have records active on the session"
    assert summary["untracked"] == 1, "CCC was vetoed with nothing tracking it"
    assert summary["pending"] == 1, "EEE's session has not closed"
    assert summary["joined"] + summary["untracked"] + summary["pending"] == summary["veto_count"], (
        "a veto fell out of the accounting - that is the 127 the review found"
    )

    text = compression_calibration.render_report(rows, summary, since=SINCE)
    assert "4 vetoes: 2 joined, 1 untracked, 1 pending" in text, text
    assert PENDING_SESSION in text, "the excluded session is named"
    assert "not yet complete" in text, "and WHY it was excluded"


def test_the_window_printed_is_the_window_measured(scratch):
    """The header said "..09-14" while the trader had vetoed on 09-15."""
    import compression_calibration

    rows, summary = compression_calibration.build_rows(since=SINCE)
    text = compression_calibration.render_report(rows, summary, since=SINCE)
    assert summary["sessions_measured"] == [SESSION]
    assert summary["sessions_pending"] == [PENDING_SESSION]
    assert f"{SESSION} .. {SESSION}" in text
    assert {row["session_date"] for row in rows} == {SESSION}, (
        "no row may be measured on a session that has not closed"
    )


def test_a_real_future_session_is_never_complete():
    """The fixture patches the clock; this pins the function it patches."""
    import compression_calibration

    future = (date.today() + timedelta(days=30)).isoformat()
    assert compression_calibration.session_is_complete(future) is False
    assert compression_calibration.session_is_complete("not-a-date") is False


# ---------------------------------------------------------------------------
# Rule 2 - the population is what was ACTIVE, not what was entered
# ---------------------------------------------------------------------------


def test_a_setup_entered_a_week_earlier_is_in_the_session_s_population(scratch):
    """The defect, stated as a test. Every record here was entered on 08-13 and
    vetoed on 08-20; joining on the entry date finds none of them."""
    import compression_calibration

    rows, _summary = compression_calibration.build_rows(since=SINCE)
    joined = {row["symbol"] for row in rows if row["population_role"] == "joined"}
    assert {"AAA", "BBB", "DDD"} <= joined
    assert "ZZZ" not in {row["symbol"] for row in rows}, (
        "a record that stopped being carried before the session is not its population"
    )


def test_the_header_prints_the_population_definition_and_never_says_shown(scratch):
    import compression_calibration

    rows, summary = compression_calibration.build_rows(since=SINCE)
    text = compression_calibration.render_report(rows, summary, since=SINCE)
    assert "ACTIVE on it" in text
    assert "shown row" not in text, "the old word claimed something the report never knew"


def test_the_active_window_is_read_off_the_record_s_own_dates():
    import compression_calibration

    assert compression_calibration.record_active_window(
        {"entry_trade_date": "2026-08-13", "last_replayed_session": "2026-08-20"}
    ) == ("2026-08-13", "2026-08-20")
    # Never replayed: live for its entry session alone, which is the honest
    # answer rather than "still open forever".
    assert compression_calibration.record_active_window({"scan_date": "2026-08-13"}) == (
        "2026-08-13",
        "2026-08-13",
    )


def test_an_untracked_veto_still_carries_the_four_fixed_window_measures(scratch):
    """It joins the vetoed side; only the anchor is unavailable."""
    import compression_calibration

    rows, _summary = compression_calibration.build_rows(since=SINCE)
    untracked = [row for row in rows if row["symbol"] == "CCC"]
    assert len(untracked) == 1
    row = untracked[0]
    assert row["population_role"] == "untracked"
    assert row["compressed_veto"] is True
    assert row["anchor_source"] == "anchor unknown"
    for key in compression_calibration.FIXED_WINDOW_MEASURE_KEYS:
        assert row[key] is not None, f"{key} needs no anchor and must be measured"
    for key in compression_calibration.ANCHOR_MEASURE_KEYS:
        assert row[key] is None, f"{key} cannot be measured without an anchor"


# ---------------------------------------------------------------------------
# Rule 4 - a missing anchor measure is recomputed, not printed as nan
# ---------------------------------------------------------------------------


def test_a_record_with_no_measure_has_its_anchor_measures_recomputed(scratch):
    """`n = 0 / nan` on the three anchor measures was the live run's answer,
    because no record on disk carries them yet. They are recomputable."""
    import compression_calibration

    rows, _summary = compression_calibration.build_rows(since=SINCE)
    from_record = [row for row in rows if row["symbol"] == "AAA"][0]
    recomputed = [row for row in rows if row["symbol"] == "BBB"][0]

    assert from_record["anchor_source"] == "record"
    assert from_record["compression_stdev_atr_ratio"] == pytest.approx(0.40)

    assert recomputed["anchor_source"] == "recomputed"
    for key in compression_calibration.ANCHOR_MEASURE_KEYS:
        assert recomputed[key] is not None, f"{key} was left unmeasured"
        assert recomputed[key] >= 0.0


def test_every_measure_block_carries_an_n(scratch):
    """The point of the recompute: seven blocks, seven counts, no empty table."""
    import compression_calibration

    rows, summary = compression_calibration.build_rows(since=SINCE)
    text = compression_calibration.render_report(rows, summary, since=SINCE)
    lines = text.splitlines()
    for key in compression_calibration.MEASURE_KEYS:
        index = next(i for i, line in enumerate(lines) if key in line)
        block = "\n".join(lines[index : index + 4])
        assert "n = " in block, block
        assert " / 0" not in block.split("n = ")[1].splitlines()[0] or key in (
            compression_calibration.ANCHOR_MEASURE_KEYS
        ), block


def test_the_recompute_is_point_in_time(scratch, tmp_path, monkeypatch):
    """A bar after the session may not reach the recomputed anchor either."""
    import project_paths

    import compression_calibration

    before = compression_calibration.build_rows(since=SINCE)[0]

    bars_dir = Path(project_paths.DAILY_BARS_CACHE_DIR)
    import csv as _csv

    for path in sorted(bars_dir.glob("*.csv")):
        rows_out = list(_csv.DictReader(path.open("r", encoding="utf-8", newline="")))
        stamp = date.fromisoformat(SESSION)
        for step in range(1, 11):
            stamp += timedelta(days=1)
            rows_out.append(
                {
                    "datetime": stamp.isoformat(),
                    "open": str(300.0 + step * 12.0),
                    "high": str(325.0 + step * 12.0),
                    "low": str(275.0 + step * 12.0),
                    "close": str(320.0 + step * 12.0),
                    "volume": "1000000",
                }
            )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = _csv.DictWriter(handle, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)

    after = compression_calibration.build_rows(since=SINCE)[0]
    assert before == after, "a bar dated after the session moved a measure"


# ---------------------------------------------------------------------------
# Rule 5 - the 1.26 GB tracker is streamed
# ---------------------------------------------------------------------------


class _BoundedReader:
    """A file object that refuses to hand over more than `limit` at a time.

    `read_text` and `json.load` both ask for everything; a streaming reader asks
    for a window. This is the difference, made into a failure.
    """

    def __init__(self, path: Path, limit: int):
        self._handle = path.open("r", encoding="utf-8")
        self._limit = limit
        self.largest_request = 0

    def read(self, size: int = -1) -> str:
        self.largest_request = max(self.largest_request, size)
        if size is None or size < 0 or size > self._limit:
            raise AssertionError(
                f"the reader asked for {size} bytes at once; the cap is {self._limit}"
            )
        return self._handle.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self._handle.close()
        return False


def test_the_tracker_is_streamed_and_never_read_whole(tmp_path, monkeypatch):
    import compression_calibration

    payload = {
        "saved_at": "2026-08-20T13:05:00-07:00",
        # A large non-record section: it must be WALKED PAST, not decoded.
        "stats": [{"group": f"g{index}", "blob": "x" * 2000} for index in range(400)],
        "setups": {
            f"SYM{index}-LONG-2026-08-20": {
                "setup_id": f"SYM{index}-LONG-2026-08-20",
                "symbol": f"SYM{index}",
                "side": "LONG",
                "scan_date": "2026-08-20",
                "entry_trade_date": "2026-08-20",
                "padding": "y" * 1000,
            }
            for index in range(300)
        },
        "control_setups": {},
        "study_setups": {},
    }
    path = tmp_path / "tracker.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    assert path.stat().st_size > 1_000_000, "the fixture has to be bigger than the window"

    limit = 64 * 1024
    readers: list[_BoundedReader] = []
    real_open = compression_calibration.open if hasattr(compression_calibration, "open") else open

    def _bounded_open(target, *args, **kwargs):  # noqa: ANN001
        if str(target) == str(path):
            reader = _BoundedReader(path, limit)
            readers.append(reader)
            return reader
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _bounded_open)
    records = list(compression_calibration.iter_tracker_records(path, chunk_size=limit))
    monkeypatch.undo()

    assert len(records) == 300, f"streamed {len(records)} of 300 records"
    assert {record["symbol"] for record in records} == {f"SYM{index}" for index in range(300)}
    assert readers and readers[0].largest_request <= limit


def test_the_streamed_reader_survives_a_truncated_file(tmp_path):
    """A half-written tracker yields what it can and stops - never raises."""
    import compression_calibration

    path = tmp_path / "truncated.json"
    path.write_text(
        '{"setups": {"A": {"symbol": "A", "scan_date": "2026-08-20"}, "B": {"symbol": "B"',
        encoding="utf-8",
    )
    records = list(compression_calibration.iter_tracker_records(path))
    assert [record["symbol"] for record in records] == ["A"]


def test_the_refusal_names_the_bar_cache_too(tmp_path, capsys):
    r"""The message named only the home folder; the run also reads the
    machine-local daily-bar cache, read-only."""
    import project_paths

    import compression_calibration

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(project_paths, "DATA_DIR", Path(r"C:\TradingBotData\data"), raising=False)
        monkey.setattr(
            project_paths, "TRADER_ANNOTATIONS_FILE", tmp_path / "none.jsonl", raising=False
        )
        monkey.setattr(
            project_paths, "MASTER_AVWAP_SETUP_TRACKER_FILE", tmp_path / "none.json", raising=False
        )
        monkey.setattr(project_paths, "DAILY_BARS_CACHE_DIR", tmp_path / "none", raising=False)
        code = compression_calibration.main(["--since", SINCE, "--out", str(tmp_path)])
    finally:
        monkey.undo()

    assert code not in (0, None)
    said = "".join(capsys.readouterr())
    assert "--live" in said
    assert "TradingBotData" in said
    assert "LOCALAPPDATA" in said, "the bar cache is read too, and read-only"
    assert "READ-ONLY" in said
