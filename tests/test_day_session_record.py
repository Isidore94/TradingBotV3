"""Day Recap coach: one complete point-in-time record per session."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

ET = timezone(timedelta(hours=-4))
SESSION = "2026-09-22"
BUILT = datetime(2026, 9, 23, 1, 0, tzinfo=ET)


def _walk_row(symbol, *, ran=None, category="d1_like", reason=""):
    return {
        "decision_id": (SESSION, symbol, "LONG", category, "like", "annotations", "cap-" + symbol),
        "time": f"{SESSION}T10:0{len(symbol)}:00-04:00",
        "symbol": symbol, "side": "LONG", "category": category, "what_you_did": "liked",
        "ran_after_pct": ran, "held_at_close_pct": None, "traded": "no",
        "state": "measured" if ran is not None else "unmeasured", "reason": reason,
        "horizon_moves": ((1, ran),),
    }


def _inputs(**overrides):
    import recap_store

    recap_rows = [
        {"schema": recap_store.SCHEMA, "id": "rc-1", "kind": "lesson", "session_date": SESSION,
         "recorded_at": "2026-09-22T17:00:00-04:00", "recorded_after_close": True, "supersedes": "",
         "keep": "waited", "stop": "chasing", "try": "size down", "mood": 3},
        {"schema": recap_store.SCHEMA, "id": "rc-2", "kind": "environment_verdict", "session_date": SESSION,
         "recorded_at": "2026-09-22T17:01:00-04:00", "recorded_after_close": True, "supersedes": "",
         "auto_label": "bearish_strong", "verdict": "neutral_chop", "clue_ids": [], "text": "chop"},
        {"schema": recap_store.SCHEMA, "id": "rc-3", "kind": "rule", "session_date": SESSION,
         "recorded_at": "2026-09-22T17:02:00-04:00", "recorded_after_close": True, "supersedes": "",
         "text": "hold winners", "tag": "hold_winners"},
    ]
    base = {
        "payload": {
            "session_date": SESSION,
            "provisional": False,
            "entries": [
                {"entry_id": "mj-1", "created_at": "2026-09-22T09:40:00-04:00", "text": "choppy open",
                 "origin": "desk", "timeframe": "M5"},
            ],
            "trades": [
                {"trade_id": "t-open", "symbol": "DRAM", "direction": "LONG", "status": "OPEN",
                 "opened_at": "2026-09-22T12:33:41-04:00", "closed_at": "", "net_pnl": 0.0,
                 "net_pnl_cad": 0.0, "planned_risk": None, "currency": "USD", "setup_tags": ""},
                {"trade_id": "t-win", "symbol": "AMD", "direction": "LONG", "status": "CLOSED",
                 "opened_at": "2026-09-22T09:45:00-04:00", "closed_at": "2026-09-22T10:30:00-04:00",
                 "net_pnl": 120.0, "net_pnl_cad": 168.0, "planned_risk": 84.0, "currency": "USD",
                 "setup_tags": "vwap_bounce; morning",
                 "exit_fields": {"why": {"code": "target_hit", "span": [], "quote": ""}}},
            ],
            "trade_reviews": [
                {"trade_id": "t-win", "entry_raw": {"text": "saw the reclaim", "recorded_at": "x"},
                 "entry_answers": {}, "exit_raw": {"text": "hit my target"}, "exit_fields": {}},
            ],
            "walkaway": {
                "liked_not_traded": [_walk_row("AAA", ran=3.0), _walk_row("BB", ran=1.0),
                                     _walk_row("C", ran=2.0), _walk_row("DDDD")],
                "rejected": [_walk_row("EEE", ran=-1.0, category="veto", reason="extended")],
                "traded_left_early": [],
                "claimed_d1": [_walk_row("FF", ran=0.5, category="claim")],
                "earlier_calls": [_walk_row("OLD", ran=9.0)],
            },
            "reads": [
                {"read_id": "rd-1", "stamp": "2026-09-22T07:43:24-07:00", "direction": "chop",
                 "horizon": "rest_of_day", "verdict": "right", "move_atr": 0.02, "because": "big day"},
            ],
            "congruence": [],
            "mood": {"n": 1, "recorded": [{"entry_id": "mj-2", "score": 4, "at": "2026-09-22T16:10:00-04:00"}]},
            "ideas": [],
            "pack_sources_unread": (),
        },
        "pack": {
            "environment": [
                {"kind": "regime_shift", "event_at": "2026-09-22T13:57:26+00:00", "from_regime": "bullish_strong",
                 "to_regime": "bullish_weak", "source": "auto", "source_id": "env:regime_shift:0"},
                {"kind": "regime_shift", "event_at": "2026-09-22T15:00:28+00:00", "from_regime": "bullish_weak",
                 "to_regime": "bearish_strong", "source": "auto", "source_id": "env:regime_shift:1"},
                {"kind": "d1_label", "label": "uptrend", "source_id": "env:d1_label"},
            ],
            "measured": [
                {"symbol": "SPY", "close": 774.54, "change_pct": 0.13, "status": "measured", "bars_through": SESSION},
                {"symbol": "IWM", "close": None, "change_pct": None, "status": "unmeasured", "reason": "no bars"},
            ],
        },
        "recap": recap_rows,
        "alert_reviews": [
            {"review_record_id": "rv-1", "ts": "2026-09-22T10:01:00", "action": "dismiss", "symbol": "AMD",
             "side": "LONG", "installation_id": "i", "machine": "m", "pid": 1, "grade": "B"},
        ],
        "env_annotations": [],
        "trade_legs": {"t-win": [{"leg_id": 1, "role": "OPEN", "timestamp": "2026-09-22T09:45:00-04:00",
                                  "quantity": 10, "price": 150.0, "raw_json": "{\"huge\": 1}"}]},
        "mentor_answers": {"t-win": [{"event_id": "ev-9", "occurred_at": "2026-09-22T16:30:00-04:00",
                                      "payload": {"mentor_question_kind": "trade_origin",
                                                  "trade_origin": "an_alert"}}]},
        "ideas": [
            {"idea_id": "id-1", "session_date": SESSION, "text": "wait 15m", "status": "kept"},
            {"idea_id": "id-2", "session_date": SESSION, "text": "trade more", "status": "dismissed"},
        ],
        "rule_for_session": {"id": "rc-0", "text": "no trade first 15m", "session_date": "2026-09-21"},
        "rule_streak": 2,
        "read_errors": "",
    }
    base.update(overrides)
    return base


def test_the_record_has_every_section_and_every_item_names_its_source():
    import day_session_record as dsr
    import project_paths

    record = dsr.build_record(SESSION, _inputs(), built_at=BUILT)
    assert record["schema"] == project_paths.DAY_SESSION_RECORD_SCHEMA == "day_session_record_v1"
    for section in dsr.SECTIONS:
        assert section in record, section
    items = dsr.iter_items(record)
    assert items
    for item in items:
        source = item.get("source")
        assert isinstance(source, dict) and source.get("store") and source.get("id"), item


def test_missing_is_unknown_never_zero():
    import day_session_record as dsr

    record = dsr.build_record(SESSION, _inputs(), built_at=BUILT)
    trades = {row["trade_id"]: row for row in record["trades"]["rows"]}
    assert trades["t-open"]["net_pnl"] is None  # an open trade's 0.0 is not a result
    assert trades["t-open"]["r_multiple"] is None
    assert trades["t-open"]["origin"] == "unknown"
    assert trades["t-open"]["exit_reason"] == "unknown"
    assert trades["t-win"]["net_pnl"] == 120.0
    assert trades["t-win"]["r_multiple"] == pytest.approx(2.0)
    assert trades["t-win"]["origin"] == "an_alert"
    assert trades["t-win"]["exit_reason"] == "target_hit"
    assert trades["t-win"]["legs"] and "raw_json" not in trades["t-win"]["legs"][0]
    assert trades["t-win"]["entry_words"]["text"] == "saw the reclaim"
    iwm = next(row for row in record["market_context"]["index_closes"] if row["symbol"] == "IWM")
    assert iwm["close"] is None and iwm["status"] == "unknown"


def test_every_pick_is_kept_not_the_top_three_and_outcomes_are_marked_later():
    import day_session_record as dsr

    record = dsr.build_record(SESSION, _inputs(), built_at=BUILT)
    picks = record["picks"]["rows"]
    assert {row["symbol"] for row in picks} == {"AAA", "BB", "C", "DDDD", "EEE", "FF"}
    assert record["outcomes_measured_at"] == BUILT.isoformat()
    for row in picks:
        assert row["outcome"]["measured_later"] is True
    veto = next(row for row in picks if row["symbol"] == "EEE")
    assert veto["reason"] == "extended"
    assert veto["population"] == "rejected"
    unmeasured = next(row for row in picks if row["symbol"] == "DDDD")
    assert unmeasured["outcome"]["ran_after_pct"] is None
    call = record["calls"]["rows"][0]
    assert call["outcome"]["verdict"] == "right" and call["outcome"]["measured_later"] is True
    assert call["stamp"] == "2026-09-22T07:43:24-07:00"


def test_market_context_holds_auto_labels_and_the_trader_verdict():
    import day_session_record as dsr

    record = dsr.build_record(SESSION, _inputs(), built_at=BUILT)
    context = record["market_context"]
    assert [row["label"] for row in context["auto_environment"]] == ["bullish_weak", "bearish_strong"]
    assert context["d1_label"] == "uptrend"
    assert context["trader_verdict"]["verdict"] == "neutral_chop"
    assert context["session_label"] == {"label": "neutral_chop", "source": "trader_corrected", "verdict_id": "rc-2"}


def test_recap_inputs_alerts_ideas_and_grades_are_carried():
    import day_session_record as dsr

    record = dsr.build_record(SESSION, _inputs(), built_at=BUILT)
    assert {row["kind"] for row in record["recap"]["rows"]} == {"lesson", "environment_verdict", "rule"}
    assert record["recap"]["rule_checked_today"]["id"] == "rc-0"
    assert record["recap"]["rule_streak"] == 2
    alert = record["alerts"]["rows"][0]
    assert alert["source"]["id"] == "rv-1" and "installation_id" not in alert
    assert {row["status"] for row in record["ai_ideas"]["rows"]} == {"kept", "dismissed"}
    assert record["setup_grades"]["status"] == "recorded"
    assert record["setup_grades"]["rows"][0]["grade"] == "B"
    assert record["notes"]["rows"][0]["text"] == "choppy open"


def test_no_grades_on_disk_says_so():
    import day_session_record as dsr

    inputs = _inputs(alert_reviews=[])
    record = dsr.build_record(SESSION, inputs, built_at=BUILT)
    assert record["setup_grades"]["status"] == "not_recorded"
    assert record["setup_grades"]["rows"] == []


def test_write_is_atomic_idempotent_and_writes_a_markdown_summary(tmp_path):
    import day_session_record as dsr

    first = dsr.build_record(SESSION, _inputs(), built_at=BUILT)
    result = dsr.write_record(first, root=tmp_path)
    path = tmp_path / f"{SESSION}.json"
    assert result["changed"] is True and Path(result["path"]) == path
    md = (tmp_path / f"{SESSION}.md").read_text(encoding="utf-8")
    assert SESSION in md and "neutral_chop" in md
    original = path.read_bytes()
    again = dsr.build_record(SESSION, _inputs(), built_at=BUILT + timedelta(days=1))
    assert dsr.write_record(again, root=tmp_path)["changed"] is False
    assert path.read_bytes() == original
    assert json.loads(original)["content_hash"] == again["content_hash"]


def test_a_failed_write_keeps_the_last_good_file(tmp_path, monkeypatch):
    import day_session_record as dsr

    dsr.write_record(dsr.build_record(SESSION, _inputs(), built_at=BUILT), root=tmp_path)
    path = tmp_path / f"{SESSION}.json"
    good = path.read_bytes()
    changed = dsr.build_record(SESSION, _inputs(rule_streak=5), built_at=BUILT)

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(dsr.os, "replace", _boom)
    with pytest.raises(OSError):
        dsr.write_record(changed, root=tmp_path)
    assert path.read_bytes() == good


def test_a_degraded_build_never_replaces_a_good_record(tmp_path):
    import day_session_record as dsr

    dsr.write_record(dsr.build_record(SESSION, _inputs(), built_at=BUILT), root=tmp_path)
    path = tmp_path / f"{SESSION}.json"
    good = path.read_bytes()
    degraded = dsr.build_record(SESSION, _inputs(unread=["trade legs (locked)"]), built_at=BUILT)
    result = dsr.write_record(degraded, root=tmp_path)
    assert result["changed"] is False and "could not read" in result["reason"]
    assert path.read_bytes() == good
    # With no prior record, the degraded one is written and names what it missed.
    fresh = dsr.write_record(degraded, root=tmp_path / "empty")
    assert fresh["changed"] is True
    stored = json.loads(Path(fresh["path"]).read_text(encoding="utf-8"))
    assert stored["session_facts"]["sources_unread"] == ["trade legs (locked)"]


def test_records_are_never_pruned_by_the_session_index(tmp_path):
    import day_review_index
    import day_session_record as dsr

    dsr.write_record(dsr.build_record(SESSION, _inputs(), built_at=BUILT), root=tmp_path / "records")
    for index in range(3):
        (tmp_path / "sessions" / f"2026-01-0{index + 1}").mkdir(parents=True)
    day_review_index._prune(tmp_path, keep=0)
    assert (tmp_path / "records" / f"{SESSION}.json").is_file()
    assert Path(dsr.records_dir()).name == "records"


def test_week_rollup_groups_with_sample_sizes_and_too_few_flag():
    import day_session_record as dsr

    monday = dsr.build_record("2026-09-21", _inputs(recap=[]), built_at=BUILT)
    tuesday = dsr.build_record(SESSION, _inputs(), built_at=BUILT)
    week = dsr.build_week("2026-W39", [monday, tuesday], built_at=BUILT)
    assert week["week"] == "2026-W39"
    assert week["sessions"] == ["2026-09-21", SESSION]
    env = {row["key"]: row for row in week["by_environment"]}
    # Tuesday's trades carry the trader-corrected label; Monday falls back to auto at open.
    assert env["neutral_chop"]["n"] == 2
    assert env["neutral_chop"]["too_few_to_tell"] is True
    family = {row["key"]: row for row in week["by_setup_family"]}
    assert family["vwap_bounce"]["n"] == 2
    assert family["vwap_bounce"]["pnl_cad"] == pytest.approx(336.0)
    assert family["unknown"]["pnl_unknown_n"] == 2
    for name in ("by_grade", "by_time_of_day", "by_origin", "by_exit_reason"):
        assert week[name], name
    assert week["rule_kept"]["n"] == 0 and week["rule_kept"]["rate"] is None
    assert week["lesson_recurrence"]["stop"][0] == {"text": "chasing", "n": 1}
    assert week["rule_recurrence"]["by_tag"][0] == {"tag": "hold_winners", "n": 1}


def test_week_key_is_iso():
    import day_session_record as dsr

    assert dsr.week_key("2026-09-22") == "2026-W39"
    assert dsr.week_key("2027-01-01") == "2026-W53"


def test_rebuild_dry_run_writes_nothing_and_refuses_an_open_session(tmp_path, monkeypatch):
    import day_session_record as dsr

    monkeypatch.setattr(dsr, "collect_inputs", lambda session, **_kwargs: _inputs())
    result = dsr.rebuild(SESSION, now=BUILT, root=tmp_path, dry_run=True)
    assert result["status"] == "ok" and result["written"] is False
    assert not any(tmp_path.iterdir())
    refused = dsr.rebuild("2026-09-23", now=BUILT, root=tmp_path)
    assert refused["status"] == "skipped"


def test_rebuild_recent_writes_records_and_the_week_and_survives_one_failure(tmp_path, monkeypatch):
    import day_session_record as dsr

    def _collect(session, **_kwargs):
        if session == "2026-09-18":
            raise RuntimeError("journal locked")
        return _inputs(payload={**_inputs()["payload"], "session_date": session})

    monkeypatch.setattr(dsr, "collect_inputs", _collect)
    prior = tmp_path / "2026-09-18.json"
    prior.write_bytes(b'{"last":"good"}\n')
    result = dsr.rebuild_recent(SESSION, sessions=3, now=BUILT, root=tmp_path)
    assert sorted(result["written"] + result["unchanged"]) == ["2026-09-21", SESSION]
    assert [row["session"] for row in result["failed"]] == ["2026-09-18"]
    assert prior.read_bytes() == b'{"last":"good"}\n'
    assert (tmp_path / "week-2026-W39.json").is_file()
    # Friday's week has no valid record (its build failed), so no rollup is invented.
    assert not (tmp_path / "week-2026-W38.json").exists()


def test_the_night_facts_slot_refreshes_records_without_changing_its_own_status(tmp_path, monkeypatch):
    import day_session_record as dsr
    from ai_jobs.day_review_facts import run_day_review_facts

    calls: list[str] = []
    monkeypatch.setattr(
        dsr, "rebuild_recent",
        lambda end_session, **kwargs: calls.append(end_session) or {
            "written": [end_session], "unchanged": [], "failed": [{"session": "x", "reason": "y"}], "weeks": [],
        },
    )

    class Service:
        def build_index_for(self, session_date, **_kwargs):
            return {"session": session_date}

        def build_session_bars_for(self, *_args, **_kwargs):
            return {"SPY": [{"close": 1.0}]}

        def build_reads_for(self, *_args, **_kwargs):
            return []

        def build_pack_for(self, session_date, **_kwargs):
            return {"session_date": session_date, "inputs_hash": "h"}

    outcome = run_day_review_facts(
        session_date="2026-09-21", now=datetime(2026, 9, 22, 1, 0), service=Service(), root=tmp_path,
    )
    assert calls == ["2026-09-21"]
    assert outcome["status"] == "ok", outcome
    assert outcome["records"]["written"] == ["2026-09-21"]
    assert "records" in outcome["reason"]


def test_cli_dry_run_prints_a_summary(monkeypatch, capsys):
    import day_session_record as dsr

    monkeypatch.setattr(dsr, "collect_inputs", lambda session, **_kwargs: _inputs())
    assert dsr.main(["--date", SESSION, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and SESSION in out and "trades 2" in out
