"""Mentor app P6: /check - parse, one cited narration (advice only), a `gate` challenge row, and its grading."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import challenge, commands, gate, settings  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import gate_pack  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
REQ = gate.CheckRequest("SHORT", "NVDA", 400.0, 3.2, 3.05)


@pytest.fixture
def pack(tmp_path):
    return gate_pack.build("SHORT", "NVDA", 400, 3.2, 3.05, now=gate_pack.FIXTURE_NOW,
                           sources=gate_pack.fixture_sources(tmp_path))


@pytest.fixture
def store(tmp_path):
    return MentorChatStore(tmp_path / "mentor_chat.sqlite3")


def _reply(verdict="wait", bullets=None, flags=None):
    return {"summary": {
        "verdict": verdict,
        "bullets": bullets if bullets is not None else [
            {"text": "$60 at risk is 0.6x your setting.", "evidence_refs": ["gate:NVDA:risk"]}],
        "rule_flags": flags if flags is not None else [],
    }, "model": "m"}


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("text, expected", [
    ("short NVDA 400 stop 3.20 entry 3.05", ("SHORT", "NVDA", 400.0, 3.2, 3.05)),
    ("SHORT nvda", ("SHORT", "NVDA", None, None, None)),
    ("long amd entry 150 stop 148.5", ("LONG", "AMD", None, 148.5, 150.0)),
    ("nvda short 100 stop $3.2", ("SHORT", "NVDA", 100.0, 3.2, None)),
    ("buy MU 50sh @ 90.1 stop 88", ("LONG", "MU", 50.0, 88.0, 90.1)),
])
def test_parse_check_is_tolerant(text, expected):
    got = gate.parse_check(text)
    assert (got.side, got.symbol, got.size, got.stop, got.entry) == expected


@pytest.mark.parametrize("text", ["", "short", "sideways NVDA", "short NVDA stop", "short NVDA 1 2", "short NVDA stop x"])
def test_parse_check_refuses_what_it_cannot_read(text):
    assert gate.parse_check(text) is None


def test_the_command_returns_a_check_request_or_a_hint():
    result = commands.handle("/check short NVDA 400 stop 3.20 entry 3.05")
    assert result.action == "check" and result.arg == REQ
    assert commands.handle("/check nonsense").action == "error"
    assert "/check" in commands.HELP_TEXT


def test_the_request_is_in_the_cache_key():
    other = gate.CheckRequest("SHORT", "NVDA", 300.0, 3.2, 3.05)
    assert gate.request_hash(REQ, "h") != gate.request_hash(other, "h")
    assert gate.request_hash(REQ, "h") == gate.request_hash(gate.CheckRequest("SHORT", "NVDA", 400, 3.2, 3.05), "h")


# ---------------------------------------------------------------- narration and citations
def test_one_600_token_call_with_the_gate_schema(pack):
    seen, posted = {}, {}

    def request(**kwargs):
        seen.update(kwargs)
        kwargs["post"]("u", json={"max_tokens": 4000})
        return _reply()

    card = gate.narrate(pack, symbol="NVDA", pack_hash="h", model="gemma3:12b", endpoint="http://x/",
                        request=request, post=lambda url, **kw: posted.update(kw["json"]))
    assert posted["max_tokens"] == 600 and seen["schema"] is gate.SCHEMA and seen["endpoint"] == "http://x/v1"
    assert gate.SCHEMA["properties"]["verdict"]["enum"] == ["go", "wait", "breaks a rule"]
    assert card.narrated and card.verdict == "wait"


def test_foreign_id_rejects_uncited_dropped_unknown_plan_flag_dropped(pack):
    rejected = gate.narrate(pack, symbol="NVDA", pack_hash="h", model="m", endpoint="http://x", request=lambda **_: _reply(
        bullets=[{"text": "Tape is weak.", "evidence_refs": ["tape:d1env"]}]))
    assert not rejected.narrated and "rejected" in rejected.error
    card = gate.narrate(pack, symbol="NVDA", pack_hash="h", model="m", endpoint="http://x", request=lambda **_: _reply(
        verdict="breaks a rule",
        bullets=[{"text": "Risk is fine.", "evidence_refs": ["gate:NVDA:risk"]}, {"text": "Feels late.", "evidence_refs": []}],
        flags=[{"plan_id": "plan:risk:1", "breaks": True, "text": "4th short"},
               {"plan_id": "plan:made:up", "breaks": True, "text": "invented"}]))
    assert card.narrated and [b["text"] for b in card.bullets] == ["Risk is fine."]
    assert [f["plan_id"] for f in card.rule_flags] == ["plan:risk:1"]
    reasons = sorted(d["reason"] for d in card.dropped)
    assert reasons == ["uncited", "unknown plan id"]
    text = gate.card_markdown(card, REQ, pack)
    assert "**breaks a rule**" in text and "[gate:NVDA:risk]" in text and "Plan breaks [plan:risk:1]" in text
    assert text.rstrip().endswith(gate.FOOTER) and "never orders" in gate.FOOTER


def test_an_unknown_verdict_is_no_verdict(pack):
    card = gate.narrate(pack, symbol="NVDA", pack_hash="h", model="m", endpoint="http://x",
                        request=lambda **_: _reply(verdict="buy now"))
    assert not card.narrated


def test_brain_down_card_is_the_pack_with_no_verdict_and_the_footer(pack):
    from mentor_app.assess import Assessment

    text = gate.card_markdown(Assessment(symbol="NVDA", pack_hash="h", error="the brain is off: x"), REQ, pack)
    assert "no verdict" in text and "[gate:NVDA:risk]" in text and "[gate:NVDA:book:industry]" in text
    assert text.rstrip().endswith(gate.FOOTER)


# ---------------------------------------------------------------- the challenge row and its grading
def _narrated(pack):
    return gate.narrate(pack, symbol="NVDA", pack_hash="h", model="m", endpoint="http://x",
                        request=lambda **_: _reply(), now=lambda: datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc))


def _journal(path: Path, trades):
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE trades (trade_id TEXT, symbol TEXT, direction TEXT, status TEXT, opened_at TEXT, closed_at TEXT,"
        " quantity_closed REAL, average_entry_price REAL, average_exit_price REAL);"
        "CREATE TABLE trade_annotations (trade_id TEXT, planned_stop REAL);")
    for trade in trades:
        conn.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)", trade[:9])
        conn.execute("INSERT INTO trade_annotations VALUES (?,?)", (trade[0], trade[9]))
    conn.commit()
    conn.close()
    return path


def test_a_narrated_check_is_one_gate_challenge_with_the_outcome_seed(pack, store):
    card = _narrated(pack)
    assert gate.record(store, card, REQ, "d1")
    (row,) = store.challenges(kind="gate")
    seed = json.loads(row["outcome_json"])
    assert row["symbol"] == "NVDA" and row["claim"].startswith("wait: $60 at risk")
    assert json.loads(row["evidence_ids_json"]) == ["gate:NVDA:risk"]
    assert (seed["side"], seed["entry"], seed["stop"], seed["size"]) == ("SHORT", 3.05, 3.2, 400.0)
    assert datetime.fromisoformat(row["issued_utc"]).tzinfo is not None
    from mentor_app.assess import Assessment

    assert not gate.record(store, Assessment(symbol="NVDA", pack_hash="h", error="off"), REQ, "d2")


def test_grading_hit_on_a_closed_winner_opened_next_session(pack, store, tmp_path):
    gate.record(store, _narrated(pack), REQ, "d1")
    db = _journal(tmp_path / "j.sqlite3", [
        ("T1", "NVDA", "SHORT", "CLOSED", "2026-09-30T07:00:00-07:00", "2026-09-30T09:00:00-07:00", 400, 3.05, 2.90, 3.20)])
    assert challenge.grade_open(store, datetime(2026, 10, 1, 8, 0, tzinfo=PT), veto_outcomes=tmp_path / "none.csv",
                                journal=db) == 1
    (row,) = store.challenges(kind="gate")
    outcome = json.loads(row["outcome_json"])
    assert outcome["hit"] is True and outcome["r"] == pytest.approx(1.0) and outcome["result"] == "taken"
    assert row["graded_utc"]


def test_grading_waits_on_an_open_trade_and_a_loser_is_no_hit(pack, store, tmp_path):
    gate.record(store, _narrated(pack), REQ, "d1")
    db = _journal(tmp_path / "j.sqlite3", [
        ("T1", "NVDA", "SHORT", "OPEN", "2026-09-29T07:00:00-07:00", "", 0, 3.05, 0, 3.20)])
    gate.grade_open(store, datetime(2026, 10, 1, 8, 0, tzinfo=PT), journal=db)
    (row,) = store.challenges(kind="gate")
    assert not row["graded_utc"] and json.loads(row["outcome_json"])["reason"] == "trade still open"
    db2 = _journal(tmp_path / "j2.sqlite3", [
        ("T1", "NVDA", "SHORT", "CLOSED", "2026-09-29T07:00:00-07:00", "x", 400, 3.05, 3.20, 3.20)])
    gate.grade_open(store, datetime(2026, 10, 1, 8, 0, tzinfo=PT), journal=db2)
    outcome = json.loads(store.challenges(kind="gate")[0]["outcome_json"])
    assert outcome["hit"] is False and outcome["r"] == pytest.approx(-1.0)


def test_no_trade_stays_open_then_is_not_taken_after_5_sessions(pack, store, tmp_path):
    gate.record(store, _narrated(pack), REQ, "d1")
    late = _journal(tmp_path / "j.sqlite3", [  # opened 3 sessions later: not this gate's trade
        ("T1", "NVDA", "SHORT", "CLOSED", "2026-10-02T07:00:00-07:00", "x", 400, 3.05, 2.9, 3.2)])
    gate.grade_open(store, datetime(2026, 10, 2, 8, 0, tzinfo=PT), journal=late)
    (row,) = store.challenges(kind="gate")
    assert not row["graded_utc"] and json.loads(row["outcome_json"])["reason"] == "no trade yet"
    gate.grade_open(store, datetime(2026, 10, 6, 8, 0, tzinfo=PT), journal=late)  # 5 sessions after Tue 09-29
    (row,) = store.challenges(kind="gate")
    outcome = json.loads(row["outcome_json"])
    assert row["graded_utc"] and outcome["result"] == "not taken" and "hit" not in outcome


def test_the_scorecard_lists_the_gate_kind(pack, store):
    gate.record(store, _narrated(pack), REQ, "d1")
    assert "**Scorecard: gate challenges**" in challenge.scorecard(store, floor=30)


# ---------------------------------------------------------------- the window
@pytest.fixture
def win(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    calls: list = []
    sources = gate_pack.fixture_sources(tmp_path / "world")

    def request(**kwargs):
        calls.append(kwargs)
        return _reply()

    window = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(),
        stream_post=lambda *a, **k: [], post=lambda *a, **k: {}, mentor_enabled=False,
        now=lambda: datetime(2026, 9, 29, 7, 0, tzinfo=PT),
        gate_builder=lambda req: gate_pack.build(req.side, req.symbol, req.size, req.stop, req.entry,
                                                 now=gate_pack.FIXTURE_NOW, sources=sources),
        gate_request=request,
    )
    window.calls = calls
    yield window
    window.shutdown()
    window.deleteLater()


def _drain(window):
    while window.queue.run_one():
        pass


def test_check_with_the_brain_off_is_the_pack_and_no_verdict(win):
    win._brain_reason = "the night AI owns the GPU"
    win.send("/check short NVDA 400 stop 3.20 entry 3.05")
    _drain(win)
    text = win.transcript.toPlainText()
    assert "no verdict" in text and "gate:NVDA:risk" in text and "never orders" in text and win.calls == []
    assert win.store.challenges(kind="gate") == []


def test_check_narrates_each_different_request_and_records_it(win):
    win._brain_ok, win._endpoint, win._model = True, "http://127.0.0.1:11436", "gemma3:12b"
    win.send("/check short NVDA 400 stop 3.20 entry 3.05")
    _drain(win)
    win.send("/check short NVDA 300 stop 3.20 entry 3.05")
    _drain(win)
    assert len(win.calls) == 2
    assert len(win.store.challenges(kind="gate")) == 2
    text = win.transcript.toPlainText()
    assert "wait" in text and "never orders" in text


# ---------------------------------------------------------------- review advisories (2026-09-30)
def test_a_trade_opened_before_the_check_never_grades_it(pack, store, tmp_path):
    # Check at 10:00 ET (14:00 UTC); T0 opened 09:35 and lost, T1 opened 11:00 and won.
    gate.record(store, _narrated(pack), REQ, "d1")
    db = _journal(tmp_path / "j.sqlite3", [
        ("T0", "NVDA", "SHORT", "CLOSED", "2026-09-29T09:35:00-04:00", "x", 400, 3.05, 3.20, 3.20),
        ("T1", "NVDA", "SHORT", "CLOSED", "2026-09-29T11:00:00-04:00", "x", 400, 3.05, 2.90, 3.20)])
    gate.grade_open(store, datetime(2026, 9, 30, 8, 0, tzinfo=PT), journal=db)
    outcome = json.loads(store.challenges(kind="gate")[0]["outcome_json"])
    assert outcome["trade_id"] == "T1" and outcome["hit"] is True


def test_the_scorecard_shows_gate_before_the_first_check_with_its_own_label(store):
    text = challenge.scorecard(store, floor=30)
    gate_block = text.split("**Scorecard: gate challenges**", 1)[1].split("**", 1)[0]
    assert "win rate of taken trades (R > 0): too few (n=0" in gate_block
    assert "session hit rate" not in gate_block
    assert "5-session hit rate" in text.split("**Scorecard: gate challenges**", 1)[0]  # veto keeps its label


def test_a_failed_check_card_carries_the_advice_only_footer(win):
    def broken(req):
        raise RuntimeError("disk gone")

    win._gate_builder = broken
    win.send("/check short NVDA")
    _drain(win)
    text = win.transcript.toPlainText()
    assert "could not be built" in text and "advice only; you click, it never orders" in text
