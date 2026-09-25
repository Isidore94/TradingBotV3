r"""TJ-15, the builder's own additions - nothing here weakens a tester's file.

Two assertions in `tests/test_tj15_point_in_time.py` and
`tests/test_tj15_miss_contrast_slot.py` cannot be satisfied as written and are
left RED rather than edited (builder handoff, 2026-09-19):

* the streamed-reader test writes ``range(4000)`` rows and keeps every index
  where ``index % 666 == 0``, which is SEVEN rows (0, 666, 1332, 1998, 2664,
  3330, 3996) and not the six it asserts - so it fails on the row count before
  it ever reaches the memory claim, which is the thing the test is about;
* the forced-run test asserts the ledger row reads ``ok``, but
  ``runner._run_slots_locked`` has converted every successful FORCED run to
  ``manual_test`` since the flag existed - *"a deliberate operator run produced
  real artifacts, but it is not the session's nightly brief and must not be
  counted as coverage"* - and `tests/test_ai_jobs_runner.py:590` pins that.

Both intents are real and both are proven here instead, with the arithmetic
taken from the fixture rather than written out by hand. The rest of this file
covers what those two tests could not reach: the supersede-and-read path, and
the degrade-to-a-reason path a locked or missing 709 MB features file takes.
"""

from __future__ import annotations

import csv
import json
import sys
import tracemalloc
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = ZoneInfo("America/New_York")
#: A fixed instant inside the night window. Nothing here reads the wall clock:
#: the live task fires at 22:00 Pacific and holds the runner lock for hours.
OVERNIGHT = datetime(2026, 8, 12, 2, 0, tzinfo=ET)

SLOT = "miss_contrast"
SESSION = "2026-09-18"
DECIDED = "2026-09-11"
NOW = datetime(2026, 9, 19, 2, 0)
FEATURE_COLUMNS = ("run_id", "run_timestamp", "run_date", "symbol", "side",
                   "pct_from_current_vwap")


@pytest.fixture(autouse=True)
def _pin_the_desk_zone(monkeypatch):
    monkeypatch.setenv("TRADINGBOT_MARKET_TIMEZONE", "America/Los_Angeles")


@pytest.fixture
def ai_store(tmp_path, monkeypatch):
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    root = tmp_path / "ai_store"
    root.mkdir()
    monkeypatch.setenv("TRADINGBOTV3_AI_STORE_DIR", str(root))
    return root


@pytest.fixture
def unlocked(monkeypatch):
    import local_writer_lock as lock_mod

    @contextmanager
    def _open(_key, **_kwargs):
        yield None

    monkeypatch.setattr(lock_mod, "local_writer_lock", _open)


def _registered_slot():
    from ai_jobs import runner

    by_name = {slot.name: slot for slot in runner.default_slots()}
    assert SLOT in by_name, sorted(by_name)
    return by_name[SLOT]


def _sessions_ending(session: str, count: int) -> list[str]:
    from market_calendar import previous_session

    cursor = date.fromisoformat(session)
    out = [cursor.isoformat()]
    for _ in range(count - 1):
        cursor = previous_session(cursor)
        out.append(cursor.isoformat())
    return list(reversed(out))


def _sessions_after(session: str, count: int) -> list[str]:
    from market_calendar import is_session

    cursor = date.fromisoformat(session)
    out: list[str] = []
    while len(out) < count:
        cursor += timedelta(days=1)
        if is_session(cursor):
            out.append(cursor.isoformat())
    return out


def _daily(*, run: bool, session: str = DECIDED, forward: int = 5) -> list[dict]:
    flat = [
        {"dt": f"{day}T00:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for day in _sessions_ending(session, 15)
    ]
    first = (100.0, 103.0, 99.9, 102.5) if run else (100.0, 100.5, 98.5, 98.6)
    rest = (102.5, 103.0, 102.0, 102.5) if run else (98.6, 99.0, 98.0, 98.5)
    after = ([first] + [rest] * 4)[:forward]
    days = _sessions_after(session, len(after))
    return flat + [
        {"dt": f"{day}T00:00:00", "open": o, "high": h, "low": low, "close": c}
        for day, (o, h, low, c) in zip(days, after, strict=False)
    ]


def _decision(symbol: str, *, reason: str = "extended", verdict: str = "veto") -> dict:
    return {
        "session_date": DECIDED, "symbol": symbol, "side": "LONG",
        "category": "chart_review", "verdict": verdict, "source": "annotations",
        "timeframe": "D1", "stamp": f"{DECIDED}T10:05:00-04:00",
        "capture_id": f"e-{symbol}", "reason": reason, "decision_session": DECIDED,
    }


def _history(tmp_path: Path, rows) -> Path:
    path = tmp_path / "d1_features_history.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FEATURE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _scan(symbol: str, *, vwap: float) -> dict:
    return {
        "run_id": f"{DECIDED}-064000", "run_timestamp": f"{DECIDED}T06:40:00",
        "run_date": DECIDED, "symbol": symbol, "side": "LONG",
        "pct_from_current_vwap": vwap,
    }


# ---------------------------------------------------------------------------
# the streamed reader, with the arithmetic taken from the fixture
# ---------------------------------------------------------------------------


def test_the_stream_holds_the_window_and_not_the_file(tmp_path):
    """4,000 rows on disk, a handful in the window, under 1 MB of Python.

    The live file is 709 MB over 264 columns. A reader that materialised it
    would hold about 1.4 GB of dicts to answer a question about a few hundred
    rows, inside the process that owns the night.
    """
    from ai_jobs import miss_contrast

    columns = list(FEATURE_COLUMNS) + [f"feature_{i:02d}" for i in range(25)]
    path = tmp_path / "d1_features_history.csv"
    wanted = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(4000):
            in_window = index % 666 == 0
            wanted += int(in_window)
            run_date = DECIDED if in_window else "2026-07-02"
            row = {name: f"{index}.{name[-2:]}00000" for name in columns}
            row.update({
                "run_id": f"{run_date}-0640{index:02d}",
                "run_timestamp": f"{run_date}T06:40:00", "run_date": run_date,
                "symbol": f"S{index:04d}", "side": "LONG",
            })
            writer.writerow(row)
    size = path.stat().st_size
    assert size > 1_000_000, size
    assert wanted == 7, "the fixture's own count, never a number written by hand"

    stream = miss_contrast.stream_feature_rows(path, sessions={DECIDED})
    assert iter(stream) is stream, "the reader must be a lazy iterator"

    tracemalloc.start()
    try:
        kept = list(stream)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(kept) == wanted, [row["run_date"] for row in kept]
    assert all(row["run_date"] == DECIDED for row in kept)
    assert peak < 1_000_000, f"peak {peak} bytes over a {size} byte file"


def test_an_unreadable_features_file_yields_nothing_and_raises_nothing(tmp_path):
    """A locked or missing 709 MB file is a quiet empty walk, not an exception."""
    from ai_jobs import miss_contrast

    missing = tmp_path / "nowhere" / "d1_features_history.csv"
    assert list(miss_contrast.stream_feature_rows(missing, sessions={DECIDED})) == []
    # No session asked for is no file opened at all.
    assert list(miss_contrast.stream_feature_rows(missing, sessions=())) == []


# ---------------------------------------------------------------------------
# the forced daytime run, against the runner's own manual-run contract
# ---------------------------------------------------------------------------


def test_a_forced_run_outside_the_window_still_publishes_and_never_fails(
    tmp_path, ai_store, unlocked, monkeypatch
):
    """`--force` buys the clock for a DETERMINISTIC slot (TJ-13A item 1).

    The status it records is the runner's manual-run status, not `ok`: a
    deliberate operator run publishes real artifacts and must never be counted
    as the session's coverage (`runner._run_slots_locked`, pinned by
    `tests/test_ai_jobs_runner.py`). What TJ-15 owes is that the slot RAN with
    an empty home folder rather than being skipped or failing the night.
    """
    from ai_jobs import ledger, runner, window

    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(window, "launch_allowed", lambda *_a, **_k: (False, "window closed"))

    led = tmp_path / "ledger.jsonl"
    report = runner.run_slots(
        [_registered_slot()], now=OVERNIGHT, force=True, ledger_path=led
    )

    assert len(report.results) == 1, report.results
    row = report.results[0]
    assert row["job"] == SLOT
    assert row["status"] not in (ledger.STATUS_SKIPPED, ledger.STATUS_FAILED), row
    assert row["status"] == ledger.STATUS_MANUAL, row
    assert not str(row.get("model") or ""), "a deterministic slot names no model"
    assert not str(row.get("error") or ""), row
    # An empty home folder is a REASON on a published pack, never a failure.
    outputs = [Path(value) for value in (row.get("outputs") or ())]
    assert len(outputs) == 1, row
    pack = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert pack["groups"] == []
    assert pack["leaders"] == []
    assert "no group has reached the floor" in pack["statement"], pack["statement"]


# ---------------------------------------------------------------------------
# supersede, then read the latest
# ---------------------------------------------------------------------------


def test_read_latest_returns_the_superseding_sibling_not_the_first_pack(tmp_path):
    """A pack is never edited, so the reader has to know which one is current."""
    from ai_jobs import miss_contrast

    root = tmp_path / "packs"
    decisions = [_decision("AAA"), _decision("BBB")]
    features = _history(tmp_path, [_scan("AAA", vwap=4.0), _scan("BBB", vwap=1.0)])
    bars = {"AAA": _daily(run=True), "BBB": _daily(run=False)}

    first = miss_contrast.run_miss_contrast(
        session_date=SESSION, now=NOW, root=root,
        decisions=decisions, features=features, daily_bars=bars,
    )
    second = miss_contrast.run_miss_contrast(
        session_date=SESSION, now=datetime(2026, 9, 19, 3, 0), root=root,
        decisions=decisions, features=features, daily_bars=bars,
    )
    assert first["outputs"] != second["outputs"]

    latest = miss_contrast.read_latest(SESSION, root=root)
    assert latest is not None
    assert latest["built_at"] == "2026-09-19T03:00:00"
    # A session nobody built is None, never an empty pack that reads as a fact.
    assert miss_contrast.read_latest("2026-09-17", root=root) is None
    assert miss_contrast.read_latest("", root=root) is None


def test_a_missing_features_file_costs_the_features_and_never_the_night(tmp_path):
    """Degrade to a RECORDED reason: the decisions are still counted."""
    from ai_jobs import miss_contrast

    root = tmp_path / "packs"
    out = miss_contrast.run_miss_contrast(
        session_date=SESSION, now=NOW, root=root,
        decisions=[_decision("AAA"), _decision("BBB")],
        features=tmp_path / "nowhere.csv",
        daily_bars={"AAA": _daily(run=True), "BBB": _daily(run=False)},
    )

    assert out["status"] == "ok", out
    pack = json.loads(Path(out["outputs"][0]).read_text(encoding="utf-8"))
    group = pack["groups"][0]
    assert (group["n"], group["measured"]) == (2, 2)
    assert (group["misses"], group["correct"]) == (1, 1)
    # Every name lost its features and says so; nothing was filled in.
    assert group["no_point_in_time_scan"] == 2
    assert group["compared"] == 0
    assert group["features"] == []
    assert group["reportable"] is False
    assert "too few to call" in group["floor_note"]
