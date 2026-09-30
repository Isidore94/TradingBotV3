"""Every care name briefed nightly, reuse by evidence hash, briefs last (trader 2026-09-30)."""

from __future__ import annotations

import copy
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = ZoneInfo("America/New_York")
OVERNIGHT = datetime(2026, 8, 12, 2, 0, tzinfo=ET)


def _journal(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trades (symbol TEXT, status TEXT, opened_at TEXT, closed_at TEXT, "
        "trade_date TEXT)"
    )
    conn.executemany(
        "INSERT INTO trades VALUES (?, ?, ?, ?, ?)",
        [
            ("TSLA", "CLOSED", "2026-09-22T10:00:00", "2026-09-22T11:00:00", "2026-09-22"),
            # opened three weeks ago, still open: a care name though not traded this week
            ("HOLD", "OPEN", "2026-09-01T10:00:00", "", "2026-09-01"),
            ("HALF", "CLOSED_PARTIAL", "2026-09-02T10:00:00", "", "2026-09-02"),
            ("GONE", "CLOSED", "2026-09-02T10:00:00", "2026-09-03T10:00:00", "2026-09-02"),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _sources(tmp_path: Path) -> dict:
    (tmp_path / "claimed.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "feedback.jsonl").write_text(
        json.dumps({"verdict": "like", "symbol": "OLDLIKE", "side": "long",
                    "trade_date": "2026-09-10", "ts": "2026-09-10T10:00:00"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "favorites.jsonl").write_text(
        json.dumps({"action": "add", "symbol": "FAVE", "side": "long",
                    "session_date": "2026-09-15", "event_at": "2026-09-15T16:00:00"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "swing_longs.txt").write_text("MSFT\n", encoding="utf-8")
    (tmp_path / "swing_shorts.txt").write_text("SWSH\n", encoding="utf-8")
    (tmp_path / "m5_longs.txt").write_text("M5L\n", encoding="utf-8")
    # a Focus short never picked, alerted or traded
    (tmp_path / "m5_shorts.txt").write_text("FSHORT\n", encoding="utf-8")
    (tmp_path / "alerts.csv").write_text(
        "time_local,trade_date,symbol,direction\n"
        "09:40:00,2026-09-24,AMD,long\n"
        "09:45:00,2026-09-24,PLTR,long\n"
        "10:40:00,2026-09-25,PLTR,short\n",
        encoding="utf-8",
    )
    return {
        "journal": _journal(tmp_path / "journal.sqlite3"),
        "claimed": tmp_path / "claimed.jsonl",
        "feedback": tmp_path / "feedback.jsonl",
        "favorites": tmp_path / "favorites.jsonl",
        "swing_focus_longs": tmp_path / "swing_longs.txt",
        "swing_focus_shorts": tmp_path / "swing_shorts.txt",
        "m5_focus_longs": tmp_path / "m5_longs.txt",
        "m5_focus_shorts": tmp_path / "m5_shorts.txt",
        "alerts": tmp_path / "alerts.csv",
    }


def test_the_roster_holds_every_care_name_including_a_focus_short_nobody_touched(tmp_path, monkeypatch):
    from ai_jobs import week_names
    from mentor_app import settings

    monkeypatch.setattr(settings, "liked_sources", lambda: frozenset({"claims", "likes", "favorites"}))
    week = week_names.load_week_names("2026-09-25", sources=_sources(tmp_path))

    assert week.unreadable == []
    assert week.ordered == [
        "TSLA", "HOLD", "HALF", "FAVE", "OLDLIKE", "MSFT", "SWSH", "M5L", "FSHORT", "PLTR", "AMD",
    ]
    assert "GONE" not in week.reasons, "a closed trade from an earlier week is not a care name"
    assert week.reasons["HOLD"] == "open_position"
    assert week.reasons["FAVE"] == "liked"
    assert week.reasons["FSHORT"] == "m5_focus"
    assert week.reasons["SWSH"] == "swing_focus"


def test_care_names_are_uncapped_and_alert_only_names_keep_the_cap():
    from ai_jobs.week_names import WeekNames

    week = WeekNames(week="2026-W39")
    for index in range(60):
        week.add(f"F{index}", "swing_focus")
    for symbol in ("AL1", "AL2", "AL3"):
        week.add(symbol, "alerted")

    symbols, over = week.roster(2)
    assert len(symbols) == 62 and symbols[-2:] == ["AL1", "AL2"] and over == 1
    assert week.roster(0) == (week.ordered, 0)


def _base(rows: dict[str, str]) -> dict:
    from tests.test_ai_ticker_briefs import _base_evidence

    base = _base_evidence()
    base["sources"][0]["content"] = [
        {"symbol": symbol, "setup_id": f"{symbol.lower()}-1", "state": state}
        for symbol, state in rows.items()
    ]
    return base


def _patch(monkeypatch, calls: list[str], base_ref: dict):
    import ai_summary
    from ai_jobs import window
    from tests.test_ai_ticker_briefs import _model_result

    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")
    monkeypatch.setattr(window, "in_offhours_window", lambda now=None: True)
    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier: f"{tier}-model")
    monkeypatch.setattr(
        ai_summary, "build_evidence_package", lambda *a, **k: copy.deepcopy(base_ref["base"])
    )

    def endpoint(**kwargs):
        symbol = kwargs["evidence"]["brief_symbol"]
        calls.append(symbol)
        return _model_result(symbol, "setups.rows")

    monkeypatch.setattr(ai_summary, "request_ai_summary", endpoint)


def _week(names: dict, key: str):
    from ai_jobs.week_names import WeekNames

    week = WeekNames(week=key)
    for symbol, reason in names.items():
        week.add(symbol, reason)
    return week


def test_the_run_briefs_every_care_name_past_the_cap(tmp_path, monkeypatch):
    from ai_jobs import briefs

    calls: list[str] = []
    ref = {"base": _base({s: "ready" for s in ("A", "B", "C", "X", "Y")})}
    _patch(monkeypatch, calls, ref)
    outcome = briefs.run_ticker_briefs(
        session_date="2026-08-11",
        now=OVERNIGHT,
        watchlist_paths={"focus_longs": tmp_path / "none.txt"},
        output_root=tmp_path / "briefs",
        morning_path=tmp_path / "morning.txt",
        week=_week({"A": "traded", "B": "swing_focus", "C": "m5_focus",
                    "X": "alerted", "Y": "alerted"}, "2026-W33"),
        name_cap=1,
    )
    assert calls == ["A", "B", "C", "X"]
    assert "roster 4 name(s) (1 traded, 1 swing_focus, 1 m5_focus, 1 alerted)" in outcome["reason"]
    assert "1 alert-only name(s) skipped: over the 1-name cap" in outcome["reason"]


def test_an_unchanged_hash_reuses_the_brief_and_a_changed_one_rebriefs(tmp_path, monkeypatch):
    from ai_jobs import briefs, week_names

    calls: list[str] = []
    ref = {"base": _base({"MSFT": "watch", "NVDA": "ready"})}
    _patch(monkeypatch, calls, ref)
    common = dict(
        now=OVERNIGHT,
        watchlist_paths={"focus_longs": tmp_path / "none.txt"},
        output_root=tmp_path / "briefs",
        morning_path=tmp_path / "morning.txt",
        name_cap=10,
    )
    names = {"MSFT": "swing_focus", "NVDA": "swing_focus"}
    briefs.run_ticker_briefs(session_date="2026-08-11", week=_week(names, "2026-W33"), **common)
    assert calls == ["MSFT", "NVDA"]
    cached = week_names.read_evidence_cache(week_names.evidence_cache_path(tmp_path / "briefs"))
    assert set(cached) == {"MSFT", "NVDA"} and all(row["content_hash"] for row in cached.values())

    # next WEEK, same evidence: no model call at all
    again = briefs.run_ticker_briefs(session_date="2026-08-18", week=_week(names, "2026-W34"), **common)
    assert calls == ["MSFT", "NVDA"], "unchanged evidence is never re-briefed"
    assert again["tokens"]["tickers_evidence_cache_reused"] == 2
    assert "2 reused: evidence unchanged" in again["reason"]
    assert again["status"] == "ok"

    # NVDA's evidence moves, MSFT's does not: only NVDA is re-briefed, even in the same week
    ref["base"] = _base({"MSFT": "watch", "NVDA": "triggered"})
    briefs.run_ticker_briefs(session_date="2026-08-19", week=_week(names, "2026-W34"), **common)
    assert calls == ["MSFT", "NVDA", "NVDA"]


def test_the_content_hash_ignores_read_stamps_and_membership():
    from ai_jobs import briefs, week_names

    base = _base({"NVDA": "ready"})
    first = briefs.build_ticker_evidence(base, "NVDA", [{"list": "focus_longs", "path": "a"}])
    later = copy.deepcopy(base)
    later["generated_at"] = "2026-08-19T02:00:00-04:00"
    later["session_date"] = "2026-08-18"
    later["sources"][0]["as_of"] = "2026-08-18T16:00:00-04:00"
    second = briefs.build_ticker_evidence(later, "NVDA", [{"list": "week_traded", "path": ""}])
    assert week_names.content_hash(first) == week_names.content_hash(second)
    moved = _base({"NVDA": "triggered"})
    third = briefs.build_ticker_evidence(moved, "NVDA", [])
    assert week_names.content_hash(third) != week_names.content_hash(first)


def test_ticker_briefs_run_last_on_the_weeknight_and_before_weekly_synthesis_on_saturday():
    from ai_jobs import runner

    weeknight = [slot.name for slot in runner.slots_for("weeknight")]
    assert weeknight[-1] == "ticker_briefs"
    assert weeknight[-2] == "improvement_ideas"
    saturday = [slot.name for slot in runner.slots_for("saturday")]
    assert saturday[-2:] == ["ticker_briefs", "weekly_synthesis"]
    assert saturday[-3] == "improvement_ideas"
    for story in ("day_review_narration", "market_story_narration", "econ_brief", "setup_research"):
        assert weeknight.index(story) < weeknight.index("ticker_briefs"), story


def test_the_reserve_grows_with_the_roster_and_never_drops_below_120(monkeypatch):
    from ai_jobs import briefs, runner, week_names

    assert briefs.ticker_briefs_reserve_minutes(0) == 120.0
    assert briefs.ticker_briefs_reserve_minutes(100) == 120.0
    assert briefs.ticker_briefs_reserve_minutes(350) == 140.0
    monkeypatch.setattr(week_names, "roster_size", lambda session_date=None: 500)
    slot = {s.name: s for s in runner.default_slots()}["ticker_briefs"]
    assert slot.reserve_minutes == 200.0
