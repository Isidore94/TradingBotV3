"""TJ-15 item 2 - the features are the ones the scan had AT the decision.

plan.md sec 5 (hard invariant): *"Point-in-time research uses only information
available at the simulated decision time; timestamps carry explicit
timezones."* plan.md §12.4 TJ-15: *"features taken from the scan AT OR BEFORE
the decision's session - never a later scan"*, and the 709 MB
`d1_features_history.csv` is *"streamed by session, never loaded whole"*.

Measured on this branch, 2026-09-19: the live file is 708,957,529 bytes with
264 columns; `run_id` / `run_timestamp` / `run_date` / `symbol` / `side` are its
first columns, and `run_timestamp` is written NAIVE
(`master_avwap_lib/runner.py:922-925`: `datetime.now().isoformat()`), i.e. desk
wall time. An annotation's `created_at` is explicitly ZONED
(`ui/annotations/store.py::_created_at_text`). So the join is aware-against-
naive, and the house rule applies: ATTACH the desk zone to the naive side,
never strip the aware one.

The contract these tests pin:

    miss_contrast.stream_feature_rows(path, *, sessions) -> iterator of dicts
    miss_contrast.build_pack(session_date, *, now, decisions,
                             daily_bars=None, features=None) -> pack dict

``features`` is the path to a file in `d1_features_history.csv` shape (or an
iterable of row mappings). ``decisions`` are Day Review's own decision rows.
"""

from __future__ import annotations

import csv
import sys
import tracemalloc
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

#: The session the decisions were made on, and the night that reads them.
DECIDED = "2026-09-11"
SESSION = "2026-09-18"
NOW = datetime(2026, 9, 22, 8, 0)
#: Flat pre-history: every bar spans exactly 2.00 around a 100.00 close, so
#: Wilder ATR(14) is exactly 2.00 and one ATR is exactly 2%.
_RUN_DAY = (100.0, 103.0, 99.9, 102.5)
_QUIET_DAY = (102.5, 103.0, 102.0, 102.5)
_NO_RUN_DAY = (100.0, 100.5, 98.5, 98.6)
_FLAT_DAY = (98.6, 99.0, 98.0, 98.5)

FEATURE_COLUMNS = ("run_id", "run_timestamp", "run_date", "symbol", "side",
                   "pct_from_current_vwap")


@pytest.fixture(autouse=True)
def _pin_the_desk_zone(monkeypatch):
    """A test that resolves the zone from the machine is only a test there."""
    monkeypatch.setenv("TRADINGBOT_MARKET_TIMEZONE", "America/Los_Angeles")


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


def _daily(*, run: bool, session: str = DECIDED) -> list[dict]:
    flat = [
        {"dt": f"{day}T00:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for day in _sessions_ending(session, 15)
    ]
    first = _RUN_DAY if run else _NO_RUN_DAY
    rest = _QUIET_DAY if run else _FLAT_DAY
    after = [first] + [rest] * 4
    days = _sessions_after(session, len(after))
    return flat + [
        {"dt": f"{day}T00:00:00", "open": o, "high": h, "low": low, "close": c}
        for day, (o, h, low, c) in zip(days, after)
    ]


def _decision(symbol: str, *, stamp: str, reason: str = "extended") -> dict:
    return {
        "session_date": DECIDED, "symbol": symbol, "side": "LONG",
        "category": "chart_review", "verdict": "veto", "source": "annotations",
        "timeframe": "D1", "stamp": stamp, "capture_id": f"e-{symbol}",
        "reason": reason, "decision_session": DECIDED,
    }


def _history(tmp_path: Path, rows) -> Path:
    path = tmp_path / "d1_features_history.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FEATURE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _scan(symbol: str, *, at: str, vwap: float, run_date: str = DECIDED) -> dict:
    return {
        "run_id": f"{run_date}-{at.replace(':', '')}", "run_timestamp": f"{run_date}T{at}",
        "run_date": run_date, "symbol": symbol, "side": "LONG",
        "pct_from_current_vwap": vwap,
    }


def _group(pack, reason: str = "extended") -> dict:
    groups = [g for g in pack["groups"] if g.get("reason_code") == reason]
    assert len(groups) == 1, pack["groups"]
    return groups[0]


def _feature(group, name: str = "pct_from_current_vwap") -> dict:
    rows = [row for row in group["features"] if row["feature"] == name]
    assert len(rows) == 1, group["features"]
    return rows[0]


# -- the point-in-time rule --------------------------------------------------


def test_the_features_come_from_the_last_scan_at_or_before_the_decision(tmp_path):
    """Two scans that day: 06:40 desk-local, and 07:30 after the 07:05 veto.

    The 07:30 row did not exist when the trader clicked. A join on the SESSION
    alone takes it, and the medians then read 9.9 - the number the decision
    could not have been made on.
    """
    from ai_jobs import miss_contrast

    decisions = [
        _decision("AAA", stamp=f"{DECIDED}T10:05:00-04:00"),   # 07:05 desk-local
        _decision("BBB", stamp=f"{DECIDED}T10:05:00-04:00"),
    ]
    features = _history(tmp_path, [
        _scan("AAA", at="06:40:00", vwap=3.0), _scan("AAA", at="07:30:00", vwap=9.9),
        _scan("BBB", at="06:40:00", vwap=1.0), _scan("BBB", at="07:30:00", vwap=9.9),
    ])

    pack = miss_contrast.build_pack(
        SESSION, now=NOW, decisions=decisions, features=features,
        daily_bars={"AAA": _daily(run=True), "BBB": _daily(run=False)},
    )

    group = _group(pack)
    assert (group["misses"], group["correct"]) == (1, 1)
    row = _feature(group)
    assert row["median_a"] == pytest.approx(3.0)
    assert row["median_b"] == pytest.approx(1.0)


def test_a_decision_with_no_scan_before_it_is_unmeasured_not_filled_from_a_later_one(tmp_path):
    """The miss was decided at 07:05 and the only scan it has ran at 07:30.

    Missing data is uncertainty, never confirmation: the name is COUNTED as
    unmeasured and contributes no feature at all.
    """
    from ai_jobs import miss_contrast

    decisions = [
        _decision("AAA", stamp=f"{DECIDED}T10:05:00-04:00"),
        _decision("BBB", stamp=f"{DECIDED}T10:05:00-04:00"),
    ]
    features = _history(tmp_path, [
        _scan("AAA", at="07:30:00", vwap=9.9),
        _scan("BBB", at="06:40:00", vwap=1.0),
    ])

    pack = miss_contrast.build_pack(
        SESSION, now=NOW, decisions=decisions, features=features,
        daily_bars={"AAA": _daily(run=True), "BBB": _daily(run=False)},
    )

    group = _group(pack)
    assert (group["misses"], group["correct"]) == (1, 1)
    assert group["features_unmeasured"] == 1
    assert group["compared"] == 0
    assert group["features"] == [] or all(
        row["median_a"] is None for row in group["features"]
    ), group["features"]


def test_the_same_decision_written_in_another_zone_picks_the_same_scan_row(tmp_path):
    """07:05-07:00 and 14:05+00:00 are ONE instant and must read alike.

    `run_timestamp` is naive desk-local. Strip the aware side instead of
    attaching the naive one and the UTC spelling reads 14:05, which puts the
    07:30 scan before the decision.
    """
    from ai_jobs import miss_contrast

    rows = [
        _scan("AAA", at="06:40:00", vwap=3.0), _scan("AAA", at="07:30:00", vwap=9.9),
        _scan("BBB", at="06:40:00", vwap=1.0), _scan("BBB", at="07:30:00", vwap=9.9),
    ]
    features = _history(tmp_path, rows)
    bars = {"AAA": _daily(run=True), "BBB": _daily(run=False)}

    def _pack(stamp: str):
        return miss_contrast.build_pack(
            SESSION, now=NOW, features=features, daily_bars=bars,
            decisions=[_decision("AAA", stamp=stamp), _decision("BBB", stamp=stamp)],
        )

    local = _feature(_group(_pack(f"{DECIDED}T07:05:00-07:00")))
    utc = _feature(_group(_pack(f"{DECIDED}T14:05:00+00:00")))

    assert local["median_a"] == pytest.approx(3.0)
    assert utc["median_a"] == pytest.approx(local["median_a"])
    assert utc["median_b"] == pytest.approx(local["median_b"])


def test_the_history_file_is_streamed_by_session_and_never_materialised(tmp_path):
    """The live file is 709 MB. The reader holds the window, not the file.

    A row-count-bounded reader: 4,000 rows on disk, 6 of them in the window,
    and the peak Python allocation while the whole stream is consumed stays
    under 1 MB. Materialising the file costs about 12 MB at this size and about
    7 GB at the live one.
    """
    from ai_jobs import miss_contrast

    columns = list(FEATURE_COLUMNS) + [f"feature_{i:02d}" for i in range(25)]
    path = tmp_path / "d1_features_history.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(4000):
            run_date = DECIDED if index % 666 == 0 else "2026-07-02"
            row = {name: f"{index}.{name[-2:]}00000" for name in columns}
            row.update({
                "run_id": f"{run_date}-0640{index:02d}",
                "run_timestamp": f"{run_date}T06:40:00", "run_date": run_date,
                "symbol": f"S{index:04d}", "side": "LONG",
            })
            writer.writerow(row)
    size = path.stat().st_size
    assert size > 1_000_000, size

    stream = miss_contrast.stream_feature_rows(path, sessions={DECIDED})
    assert iter(stream) is stream, "the reader must be a lazy iterator"

    tracemalloc.start()
    try:
        kept = list(stream)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(kept) == 6, [row["run_date"] for row in kept]
    assert all(row["run_date"] == DECIDED for row in kept)
    assert peak < 1_000_000, f"peak {peak} bytes over a {size} byte file"
