"""P2-8 8a - breadth from the local daily cache, stored beside d1_environment,
recorded by the nightly read-grades slot, and graded like the reads."""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_axes  # noqa: E402
import market_breadth_store as store  # noqa: E402
import project_paths  # noqa: E402
from indicators.breadth import RULE_VERSION, compute_breadth, name_facts  # noqa: E402

ET = ZoneInfo("America/New_York")


def _sessions(count: int, last: str = "2026-09-24") -> list[str]:
    import market_calendar

    day = date.fromisoformat(last)
    out = [day]
    while len(out) < count:
        day = market_calendar.previous_session(day)
        out.append(day)
    return [d.isoformat() for d in reversed(out)]


def _bars(closes: list[float], last: str = "2026-09-24") -> list[dict]:
    days = _sessions(len(closes), last)
    return [{"dt": d, "close": c} for d, c in zip(days, closes, strict=False)]


# -- the pure rule -----------------------------------------------------------
def test_a_name_counts_up_and_above_both_smas():
    bars = _bars([10.0] * 59 + [12.0])
    facts = name_facts(bars, session="2026-09-24", prior_session="2026-09-23")
    assert facts == {"change": "up", "above_sma20": True, "above_sma50": True}


def test_a_missing_session_bar_is_unknown_not_a_decliner():
    bars = _bars([10.0] * 60, last="2026-09-23")
    facts = name_facts(bars, session="2026-09-24", prior_session="2026-09-23")
    assert facts == {"change": "unknown", "above_sma20": None, "above_sma50": None}


def test_a_gap_before_the_session_leaves_advance_decline_unknown():
    bars = _bars([10.0] * 30)
    del bars[-2]  # the prior session's bar is missing
    facts = name_facts(bars, session="2026-09-24", prior_session="2026-09-23")
    assert facts["change"] == "unknown"
    assert facts["above_sma20"] is not None
    assert facts["above_sma50"] is None  # only 29 closes


def test_bars_after_the_session_are_never_read():
    bars = _bars([10.0] * 60 + [20.0], last="2026-09-25")
    facts = name_facts(bars, session="2026-09-24", prior_session="2026-09-23")
    assert facts["change"] == "flat"
    assert facts["above_sma20"] is False


def test_the_reading_counts_unknown_apart():
    reading = compute_breadth(
        {
            "UP": _bars([10.0] * 59 + [12.0]),
            "DN": _bars([10.0] * 59 + [8.0]),
        },
        universe=["UP", "DN", "GONE"],
        session="2026-09-24",
        prior_session="2026-09-23",
    )
    row = reading.as_row()
    assert row["rule_version"] == RULE_VERSION
    assert (row["advancers"], row["decliners"], row["ad_unknown"]) == (1, 1, 1)
    assert (row["above_sma20"], row["sma20_known"], row["pct_above_sma20"]) == (1, 2, 50.0)
    assert row["names_total"] == 3


# -- the store ---------------------------------------------------------------
#: Four more names that close up, so a fixture with one MISSING name stays above
#: the 80% coverage floor.
_UP4 = {f"UP{i}": [10.0] * 59 + [12.0] for i in range(4)}


def _cache(tmp_path, names_closes: dict[str, list[float]], *, written: datetime):
    folder = tmp_path / "daily_bars"
    folder.mkdir()
    for name, closes in names_closes.items():
        target = folder / f"{name}.csv"
        lines = ["datetime,open,high,low,close,volume"]
        for bar in _bars(closes):
            lines.append(f"{bar['dt']},1,1,1,{bar['close']},100")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.utime(target, (written.timestamp(), written.timestamp()))
    universe = tmp_path / "universe_all.txt"
    universe.write_text("\n".join(names_closes) + "\nMISSING\n", encoding="utf-8")
    return folder, universe


def test_the_store_is_append_only_and_keyed_by_session(tmp_path):
    path = tmp_path / "market_breadth.jsonl"
    reading = compute_breadth(
        {"A": _bars([10.0] * 59 + [12.0])}, universe=["A"],
        session="2026-09-24", prior_session="2026-09-23",
    )
    assert store.append_breadth(reading, path=path) is True
    assert store.append_breadth(reading, path=path) is False
    assert len(store.read_rows(path)) == 1
    row = store.row_for_session("2026-09-24", path=path)
    assert row["universe"] == store.UNIVERSE_ALL
    assert row["pct_above_sma20"] == 100.0
    assert row["ad_unknown"] == 0


def test_a_file_written_before_the_close_is_unknown_for_that_session(tmp_path):
    closes = {"AAA": [10.0] * 59 + [12.0]}
    before_close = datetime(2026, 9, 24, 11, 0, tzinfo=ET)
    folder, universe = _cache(tmp_path, closes, written=before_close)
    reading, _as_of = store.compute_for_session("2026-09-24", universe_path=universe, directory=folder)
    assert reading.advancers == 0
    assert reading.ad_unknown == 2  # AAA (forming bar) + MISSING


def test_the_night_records_the_last_completed_session(tmp_path, monkeypatch):
    # 6 of 7 names known (86%), above the coverage floor.
    closes = {"AAA": [10.0] * 59 + [12.0], "BBB": [10.0] * 59 + [9.0], **_UP4}
    after_close = datetime(2026, 9, 24, 17, 0, tzinfo=ET)
    folder, universe = _cache(tmp_path, closes, written=after_close)
    path = tmp_path / "market_breadth.jsonl"
    monkeypatch.setattr(project_paths, "MARKET_BREADTH_FILE", path)
    monkeypatch.setattr(project_paths, "DAILY_BARS_CACHE_DIR", folder)
    monkeypatch.setattr(project_paths, "UNIVERSE_ALL_FILE", universe)
    night = datetime(2026, 9, 24, 23, 30, tzinfo=ET)

    first = store.record_last_session(night)
    again = store.record_last_session(night)

    assert first["written"] is True and first["session"] == "2026-09-24"
    assert again["written"] is False
    row = store.row_for_session("2026-09-24")
    assert (row["advancers"], row["decliners"], row["ad_unknown"]) == (5, 1, 1)
    assert row["source"] == store.SOURCE_NIGHT


def test_the_read_grades_slot_records_breadth(tmp_path, monkeypatch):
    from ai_jobs import read_grades_mature

    closes = {"AAA": [10.0] * 59 + [12.0], **_UP4}
    folder, universe = _cache(tmp_path, closes, written=datetime(2026, 9, 24, 17, 0, tzinfo=ET))
    monkeypatch.setattr(project_paths, "MARKET_BREADTH_FILE", tmp_path / "b.jsonl")
    monkeypatch.setattr(project_paths, "DAILY_BARS_CACHE_DIR", folder)
    monkeypatch.setattr(project_paths, "UNIVERSE_ALL_FILE", universe)

    outcome = read_grades_mature.run_read_grades_mature(
        now=datetime(2026, 9, 24, 23, 30, tzinfo=ET), root=tmp_path / "reads"
    )

    assert outcome["status"] == "ok"
    assert outcome["reason"].endswith("breadth 2026-09-24 recorded")
    assert store.row_for_session("2026-09-24") is not None


def test_the_backfill_is_dry_by_default(tmp_path, monkeypatch, capsys):
    closes = {"AAA": [10.0] * 59 + [12.0], **_UP4}
    folder, universe = _cache(tmp_path, closes, written=datetime(2026, 9, 24, 17, 0, tzinfo=ET))
    path = tmp_path / "b.jsonl"
    monkeypatch.setattr(project_paths, "MARKET_BREADTH_FILE", path)
    monkeypatch.setattr(project_paths, "DAILY_BARS_CACHE_DIR", folder)
    monkeypatch.setattr(project_paths, "UNIVERSE_ALL_FILE", universe)
    monkeypatch.setattr(store, "_sessions_since", lambda _since, _now: ["2026-09-24"])

    assert store.main(["backfill", "--since", "2026-09-24"]) == 0
    assert not path.exists()
    assert "DRY RUN" in capsys.readouterr().out
    assert store.main(["backfill", "--since", "2026-09-24", "--apply"]) == 0
    assert store.row_for_session("2026-09-24")["source"] == store.SOURCE_BACKFILL


# -- graded like the reads -----------------------------------------------------
def _spy(closes_by_day: dict[str, float]) -> list[dict]:
    base = _bars([100.0 + (i % 3) for i in range(30)], last="2026-09-22")
    for bar in base:
        bar.update(open=bar["close"], high=bar["close"] + 1.0, low=bar["close"] - 1.0)
    for day, close in closes_by_day.items():
        base.append({"dt": day, "open": close, "high": close + 1.0, "low": close - 1.0, "close": close})
    return base


def test_a_weak_breadth_read_is_right_when_spy_falls_and_mixed_is_no_call():
    read = market_axes.morning_read(
        "2026-09-23",
        d1_label="mixed",
        breadth_row={"pct_above_sma20": 35.0, "pct_above_sma50": 41.0, "advancers": 400, "decliners": 900},
        internals_snapshot=None,
    )
    grades = market_axes.grade_axes(
        read,
        spy_daily_bars=_spy({"2026-09-23": 100.0, "2026-09-24": 95.0}),
        target_session="2026-09-24",
        now=datetime(2026, 9, 24, 17, 0, tzinfo=ET),
    )
    by_axis = {grade["axis"]: grade for grade in grades}
    assert [axis["axis"] for axis in read["axes"]] == ["spy", "breadth"]
    assert by_axis["breadth"]["verdict"] == "right"
    assert by_axis["spy"]["verdict"] == market_axes.VERDICT_NO_CALL
    assert by_axis["breadth"]["flat_band_rule"] == "atr_0.25_v1"
    assert "breadth weak (35% > SMA20, 41% > SMA50, A/D 400/900)" in market_axes.read_line(read)


def test_an_open_session_is_pending_and_a_missing_close_is_unmeasured():
    read = market_axes.morning_read(
        "2026-09-23", d1_label="trending_up", breadth_row=None, internals_snapshot=None
    )
    bars = _spy({"2026-09-23": 100.0})
    pending = market_axes.grade_axes(
        read, spy_daily_bars=bars, target_session="2026-09-24",
        now=datetime(2026, 9, 24, 12, 0, tzinfo=ET),
    )
    missing = market_axes.grade_axes(
        read, spy_daily_bars=bars, target_session="2026-09-24",
        now=datetime(2026, 9, 24, 17, 0, tzinfo=ET) + timedelta(hours=1),
    )
    assert pending[0]["verdict"].startswith("pending")
    assert missing[0]["verdict"].startswith("unmeasured")
    assert read["axes"][1]["text"] == "breadth unknown"
    assert market_axes.grade_summary(missing)["unmeasured"] == 1


# -- coverage floor (review follow-up) -------------------------------------------
def test_the_axis_names_the_unknown_count():
    row = {"names_total": 1467, "advancers": 490, "decliners": 750, "unchanged": 13,
           "ad_unknown": 214, "sma20_known": 1253, "pct_above_sma20": 27.0, "pct_above_sma50": 29.0}
    axis = market_axes.breadth_axis(row)
    assert axis["state"] == "weak" and axis["lean"] == "down"
    assert "A/D 490/750 (214 unknown)" in axis["text"]


def test_a_thin_row_is_thin_has_no_lean_and_is_never_graded():
    row = {"names_total": 1467, "advancers": 39, "decliners": 131, "unchanged": 0,
           "ad_unknown": 1297, "sma20_known": 170, "pct_above_sma20": 20.0}
    axis = market_axes.breadth_axis(row)
    assert axis["state"] == "thin" and axis["lean"] == ""
    assert axis["text"] == "breadth thin (170 known)"
    grades = market_axes.grade_axes(
        {"basis": "2026-09-23", "axes": [axis]}, spy_daily_bars=[],
        target_session="2026-09-24", now=datetime(2026, 9, 24, 17, 0, tzinfo=ET),
    )
    assert grades[0]["verdict"] == market_axes.VERDICT_NO_CALL


def test_a_thin_session_is_never_written_so_a_later_retry_can(tmp_path, monkeypatch):
    closes = {name: [10.0] * 59 + [12.0] for name in ("AAA", "BBB")}
    folder, universe = _cache(tmp_path, closes, written=datetime(2026, 9, 24, 17, 0, tzinfo=ET))
    universe.write_text("AAA\nBBB\nX1\nX2\nX3\n", encoding="utf-8")  # 2 of 5 known
    path = tmp_path / "market_breadth.jsonl"
    monkeypatch.setattr(project_paths, "MARKET_BREADTH_FILE", path)
    monkeypatch.setattr(project_paths, "DAILY_BARS_CACHE_DIR", folder)
    monkeypatch.setattr(project_paths, "UNIVERSE_ALL_FILE", universe)
    night = datetime(2026, 9, 24, 23, 30, tzinfo=ET)

    thin = store.record_last_session(night)
    assert thin["written"] is False and "thin" in thin["reason"]
    assert not path.exists()

    universe.write_text("AAA\nBBB\n", encoding="utf-8")
    good = store.record_last_session(night)
    assert good["written"] is True
    assert store.row_for_session("2026-09-24")["ad_unknown"] == 0


def test_the_backfill_never_writes_a_thin_row(tmp_path, monkeypatch):
    closes = {"AAA": [10.0] * 59 + [12.0]}
    folder, universe = _cache(tmp_path, closes, written=datetime(2026, 9, 24, 17, 0, tzinfo=ET))
    path = tmp_path / "b.jsonl"  # universe = AAA + MISSING -> 1 of 2 known
    monkeypatch.setattr(project_paths, "MARKET_BREADTH_FILE", path)
    monkeypatch.setattr(project_paths, "DAILY_BARS_CACHE_DIR", folder)
    monkeypatch.setattr(project_paths, "UNIVERSE_ALL_FILE", universe)
    monkeypatch.setattr(store, "_sessions_since", lambda _since, _now: ["2026-09-24"])
    assert store.main(["backfill", "--since", "2026-09-24", "--apply"]) == 0
    assert not path.exists()
