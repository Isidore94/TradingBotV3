"""Day Recap coach: Week Review edge/leaks/month, Ask the AI, and its night slot."""

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
NIGHT = datetime(2026, 9, 18, 23, 30, tzinfo=ET)


def _row(key, *, n, pnl=None, r=None, r_n=0, wins=0, losses=0):
    return {"key": key, "n": n, "pnl_known_n": n if pnl is not None else 0, "pnl_unknown_n": 0 if pnl is not None else n,
            "pnl_cad": pnl, "wins": wins, "losses": losses, "r_n": r_n, "avg_r": r, "too_few_to_tell": n < 10}


def _week_body(week="2026-W38", **groups):
    body = {"schema": "day_session_week_v1", "week": week, "sessions": [], "trades_n": 0,
            "lesson_recurrence": {"keep": [], "stop": [], "try": []},
            "rule_recurrence": {"by_tag": [], "by_text": []},
            "rule_kept": {"n": 0, "yes": 0, "partly": 0, "no": 0, "rate": None, "too_few_to_tell": True}}
    for name in ("by_setup_family", "by_grade", "by_time_of_day", "by_origin", "by_exit_reason", "by_environment"):
        body[name] = groups.get(name, [])
    return body


# ---------------------------------------------------------------------------
# edge and leaks
# ---------------------------------------------------------------------------
def test_edge_and_leaks_only_use_rows_with_n_ten_or_more():
    import week_coach

    body = _week_body(
        by_time_of_day=[
            _row("open", n=12, pnl=300.0, r=0.8, r_n=12, wins=8, losses=4),
            _row("midday", n=9, pnl=900.0, r=2.0, r_n=9, wins=9),  # too few to tell
            _row("close", n=11, pnl=-400.0, r=-0.6, r_n=11, wins=3, losses=8),
            _row("unknown", n=30, pnl=-999.0, r=-3.0, r_n=30),  # unknown is not a finding
        ],
        by_setup_family=[
            _row("bounce", n=10, pnl=120.0, wins=6, losses=4),  # no R: ranked by P&L
            _row("avwap", n=15, pnl=-50.0, wins=7, losses=8),
        ],
    )
    out = week_coach.edge_and_leaks(body)
    assert [row["key"] for row in out["edge"]] == ["open", "bounce"]
    assert out["edge"][0]["metric"] == "avg_r" and out["edge"][1]["metric"] == "pnl_cad"
    assert [row["key"] for row in out["leaks"]] == ["close", "avwap"]
    assert all(row["pnl_known_n"] >= 10 for row in out["edge"] + out["leaks"])
    assert "midday" not in {row["key"] for row in out["edge"]}
    assert out["thin_rows"] == 1
    assert week_coach.row_line(out["edge"][0]).endswith("8 won / 4 lost")
    assert "(n 12)" in week_coach.row_line(out["edge"][0])


def test_edge_is_capped_at_three_and_unknown_pnl_is_not_zero():
    import week_coach

    rows = [_row(f"s{i}", n=10, pnl=float(10 * (i + 1))) for i in range(5)]
    rows.append(_row("nopnl", n=20))  # P&L unknown on every trade: never ranked
    out = week_coach.edge_and_leaks(_week_body(by_setup_family=rows))
    assert [row["key"] for row in out["edge"]] == ["s4", "s3", "s2"]
    assert out["leaks"] == []
    assert week_coach.fmt_money(None) == "unknown"
    assert week_coach.fmt_rate(0.5, 4) == "50% (n 4), too few to tell"
    assert week_coach.fmt_rate(None, 0) == "unknown (n 0)"


# ---------------------------------------------------------------------------
# month
# ---------------------------------------------------------------------------
def test_month_rollup_sums_counts_and_weights_r():
    import week_coach

    one = _week_body("2026-W37", by_time_of_day=[_row("open", n=6, pnl=100.0, r=1.0, r_n=6, wins=4, losses=2)])
    one["rule_kept"] = {"n": 4, "yes": 3, "partly": 1, "no": 0}
    one["lesson_recurrence"]["stop"] = [{"text": "chasing", "n": 1}]
    two = _week_body("2026-W38", by_time_of_day=[_row("open", n=6, pnl=-40.0, r=-0.5, r_n=4, wins=2, losses=4),
                                                  _row("close", n=3)])
    two["rule_kept"] = {"n": 6, "yes": 3, "partly": 0, "no": 3}
    two["lesson_recurrence"]["stop"] = [{"text": "chasing", "n": 1}]
    month = week_coach.month_rollup([two, one], month="2026-09")
    rows = {row["key"]: row for row in month["by_time_of_day"]}
    assert rows["open"]["n"] == 12 and rows["open"]["pnl_known_n"] == 12
    assert rows["open"]["pnl_cad"] == pytest.approx(60.0)
    assert rows["open"]["avg_r"] == pytest.approx((6 * 1.0 + 4 * -0.5) / 10)
    assert rows["open"]["too_few_to_tell"] is False
    assert rows["close"]["pnl_cad"] is None  # unknown stays unknown
    assert month["rule_kept"]["n"] == 10 and month["rule_kept"]["rate"] == pytest.approx(0.6)
    assert month["weeks"] == ["2026-W37", "2026-W38"]
    assert month["lesson_recurrence"]["stop"] == [{"text": "chasing", "n": 2}]
    # The merged row now has n >= 10, so it can be an edge; each week alone could not.
    assert [row["key"] for row in week_coach.edge_and_leaks(month)["edge"]] == ["open"]
    assert week_coach.edge_and_leaks(one)["edge"] == []


def test_month_weeks_are_the_calendar_month_of_the_chosen_week():
    import week_coach

    available = ["2026-W35", "2026-W36", "2026-W37", "2026-W38", "2026-W39", "2026-W40", "2026-W41"]
    # September 2026 by ISO Thursday: W36 (Sep 3) .. W40 (Oct 1 is W40's Thursday -> October).
    assert week_coach.month_weeks("2026-W38", available) == ["2026-W36", "2026-W37", "2026-W38", "2026-W39"]
    assert week_coach.month_of("2026-W40") == "2026-10"


# ---------------------------------------------------------------------------
# records on disk
# ---------------------------------------------------------------------------
def _trade(trade_id, *, bucket, pnl, r=None, family="bounce"):
    return {"trade_id": trade_id, "symbol": "AMD", "direction": "LONG", "net_pnl_cad": pnl, "net_pnl": pnl,
            "r_multiple": r, "setup_family": family, "grade": "unknown", "time_bucket": bucket,
            "origin": "unknown", "exit_reason": "unknown", "auto_environment_at_open": "unknown",
            "source": {"store": "trade_journal.sqlite3:trades", "id": trade_id}}


def _write_records(root: Path, days: dict[str, list[dict]], *, calls=None, recap=None):
    import day_session_record as dsr

    root.mkdir(parents=True, exist_ok=True)
    for session, trades in days.items():
        record = {
            "schema": dsr.SCHEMA, "session_date": session,
            "market_context": {"session_label": {"label": "unknown", "source": "none"}},
            "trades": {"n": len(trades), "rows": trades},
            "calls": {"n": 0, "rows": list((calls or {}).get(session, ()))},
            "recap": {"n": 0, "rows": list((recap or {}).get(session, ()))},
            "content_hash": "h-" + session,
        }
        (root / f"{session}.json").write_text(json.dumps(record), encoding="utf-8")
    for week in sorted({dsr.week_key(day) for day in days}):
        dsr.write_week(week, root=root, built_at=NIGHT)


def _afternoon_week(root: Path):
    days = {}
    for index, session in enumerate(("2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18")):
        days[session] = [
            _trade(f"pm-{index}-{k}", bucket="close", pnl=-20.0, r=-0.5) for k in range(2)
        ] + [_trade(f"am-{index}", bucket="open", pnl=30.0, r=0.4)]
    calls = {"2026-09-15": [{"source": {"id": "rd-1"}, "outcome": {"verdict": "right"}},
                            {"source": {"id": "rd-2"}, "outcome": {"verdict": "wrong"}},
                            {"source": {"id": "rd-3"}, "outcome": {"verdict": "pending 2026-09-22"}}]}
    recap = {
        "2026-09-14": [{"id": "rc-r1", "kind": "rule", "text": "Hold winners", "tag": "hold_winners"},
                       {"id": "rc-l1", "kind": "lesson", "keep": "waited", "stop": "chasing"}],
        "2026-09-15": [{"id": "rc-c1", "kind": "rule_check", "answer": "yes", "rule_id": "rc-r1", "text": ""},
                       {"id": "rc-l2", "kind": "lesson", "keep": "", "stop": "Chasing "}],
        "2026-09-16": [{"id": "rc-c2", "kind": "rule_check", "answer": "no", "rule_id": "", "text": "hold winners"}],
    }
    _write_records(root, days, calls=calls, recap=recap)


def test_read_view_from_records_on_disk(tmp_path):
    import week_coach

    root = tmp_path / "records"
    _afternoon_week(root)
    view = week_coach.read_view("2026-W38", root=root, questions_path=tmp_path / "q.jsonl",
                                answers_path=tmp_path / "a.jsonl")
    assert view["recorded"] and view["trades_n"] == 15
    assert [row["key"] for row in view["leaks"]] == ["close", "bounce"]  # both avg R < 0, n >= 10
    assert view["leaks"][0]["metric"] == "avg_r" and view["leaks"][0]["r_n"] == 10
    assert view["edge"] == []  # "open" has n 5: too few to tell
    assert view["calls"] == {"right": 1, "wrong": 1, "flat": 0, "open": 1, "n": 2, "rate": 0.5,
                             "too_few_to_tell": True}
    lessons = view["repeats"]["lessons"]
    assert {"part": "stop", "text": "chasing", "n": 2} in lessons
    rule = view["repeats"]["rules"][0]
    assert rule["text"] == "hold winners" and rule["checked_n"] == 2 and rule["rate"] == 0.5
    assert rule["too_few_to_tell"] is True
    trend = view["trend"]
    assert [row["week"] for row in trend] == ["2026-W35", "2026-W36", "2026-W37", "2026-W38"]
    assert [row["recorded"] for row in trend] == [False, False, False, True]
    assert trend[-1]["pnl_cad"] == pytest.approx(-50.0) and trend[-1]["pnl_known_n"] == 15
    month = week_coach.read_view("2026-W38", month=True, root=root, questions_path=tmp_path / "q.jsonl",
                                 answers_path=tmp_path / "a.jsonl")
    assert month["month"] == "2026-09" and month["covered_weeks"] == ["2026-W38"]
    # Month view picked from an earlier week of the month: the trend ends at its last recorded week.
    early = week_coach.read_view("2026-W36", month=True, root=root, questions_path=tmp_path / "q.jsonl",
                                 answers_path=tmp_path / "a.jsonl")
    assert early["covered_weeks"] == ["2026-W38"] and early["trend"][-1]["week"] == "2026-W38"


# ---------------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------------
def test_a_question_is_saved_pending_with_a_timezone_and_never_rewritten(tmp_path):
    import week_coach

    path = tmp_path / "q.jsonl"
    row = week_coach.record_question("  Do my afternoon   trades lose? ", week="2026-W38", now=NIGHT, path=path)
    assert row["text"] == "Do my afternoon trades lose?"
    assert row["status"] == "pending" and row["id"].startswith("q-")
    assert datetime.fromisoformat(row["asked_at"]).utcoffset() is not None
    before = path.read_bytes()
    week_coach.record_question("second", now=NIGHT, path=path)
    assert path.read_bytes().startswith(before)  # append-only
    with pytest.raises(week_coach.QuestionError):
        week_coach.record_question("   ", path=path)
    with pytest.raises(week_coach.QuestionError):
        week_coach.record_question("naive", now=datetime(2026, 9, 18, 12, 0), path=path)
    with pytest.raises(week_coach.QuestionError):
        week_coach.record_question("x" * 501, path=path)
    assert len(week_coach.pending_questions(questions_path=path, answers_path=tmp_path / "a.jsonl")) == 2


def test_citation_validation_drops_uncited_and_invented_ids():
    import week_coach

    allowed = {
        "week:2026-W38:by_time_of_day:close": {"session": "", "n": 10},
        "week:2026-W38:by_time_of_day:open": {"session": "", "n": 5},
        "trade:pm-1-0": {"session": "2026-09-15", "n": 1},
    }
    claims = [
        {"text": "Afternoon trades lost.", "citations": ["week:2026-W38:by_time_of_day:close", "trade:pm-1-0"]},
        {"text": "Mornings won.", "citations": ["week:2026-W38:by_time_of_day:open"]},
        {"text": "You always win on Fridays.", "citations": []},
        {"text": "An invented row.", "citations": ["week:2026-W38:by_grade:A+"]},
        {"text": "", "citations": ["trade:pm-1-0"]},
        "not a claim",
    ]
    checked = week_coach.validate_claims(claims, allowed)
    assert [claim["text"] for claim in checked["shown"]] == ["Afternoon trades lost.", "Mornings won."]
    assert checked["shown"][0]["citations"][1] == {"id": "trade:pm-1-0", "session": "2026-09-15"}
    assert checked["shown"][0]["flag"] == ""
    assert checked["shown"][1]["flag"].startswith("too few to tell")
    assert len(checked["dropped"]) == 3
    assert all(item["note"] == "uncited — not shown" for item in checked["dropped"])


class _FakeModel:
    def __init__(self, claims=None, *, fail=False):
        self.claims = claims
        self.fail = fail
        self.evidence: list[dict] = []

    def __call__(self, **kwargs):
        self.evidence.append(kwargs["evidence"])
        if self.fail:
            raise RuntimeError("model down")
        return {"summary": {"claims": self.claims}, "model": "fake-medium"}


def test_night_answers_pending_questions_with_only_cited_claims(tmp_path):
    import week_coach
    from ai_jobs import week_questions

    root = tmp_path / "records"
    _afternoon_week(root)
    paths = {"questions_path": tmp_path / "q.jsonl", "answers_path": tmp_path / "a.jsonl"}
    question = week_coach.record_question("Do my afternoon trades lose?", week="2026-W38", now=NIGHT,
                                          path=paths["questions_path"])
    fake = _FakeModel([
        {"text": "Yes: close trades lost on average.", "citations": ["week:2026-W38:by_time_of_day:close"]},
        {"text": "One of them was on the 15th.", "citations": ["trade:pm-1-0"]},
        {"text": "You should size up.", "citations": []},
        {"text": "Made up.", "citations": ["trade:nope"]},
    ])
    out = week_questions.run_week_questions(session_date="2026-09-18", now=NIGHT, request=fake, root=root, **paths)
    assert out["status"] == "ok", out

    # Only the relevant, small context went to the model.
    evidence = fake.evidence[0]
    assert "_allowed" not in evidence
    assert [item["id"] for item in evidence["weeks_read"]] == ["week:2026-W38"]
    assert set(evidence["weeks_read"][0]) >= {"by_time_of_day"}
    assert "by_grade" not in evidence["weeks_read"][0]
    assert evidence["trades_total"] == 15 and len(evidence["trades"]) == 15
    assert "trade:pm-1-0" in evidence["allowed_source_ids"]

    [asked] = week_coach.read_questions(**paths)
    assert asked["id"] == question["id"] and asked["status"] == "answered"
    answer = asked["answer"]
    assert answer["model"] == "fake-medium"
    assert datetime.fromisoformat(answer["answered_at"]).utcoffset() is not None
    assert [claim["text"] for claim in answer["claims"]] == [
        "Yes: close trades lost on average.", "One of them was on the 15th."]
    assert answer["dropped_n"] == 2
    assert {"id": "trade:pm-1-0", "session": "2026-09-15"} in answer["citations"]
    assert week_coach.pending_questions(**paths) == []

    # Frontier digest for the week, plain file with record ids.
    digest = (root / "week-2026-W38-frontier.md").read_text(encoding="utf-8")
    assert "[week:2026-W38:by_time_of_day:close]" in digest
    assert "[session:2026-09-15]" in digest
    assert "## Leaks" in digest and "## Rules" in digest and "## Questions" in digest

    # A second night asks the model nothing.
    again = week_questions.run_week_questions(session_date="2026-09-18", now=NIGHT, request=fake, root=root, **paths)
    assert again["reason"].startswith("no pending questions") and len(fake.evidence) == 1


def test_a_failed_or_missing_model_leaves_the_question_pending(tmp_path, monkeypatch):
    import ai_summary
    import week_coach
    from ai_jobs import week_questions

    root = tmp_path / "records"
    _afternoon_week(root)
    paths = {"questions_path": tmp_path / "q.jsonl", "answers_path": tmp_path / "a.jsonl"}
    week_coach.record_question("Do I keep my rules?", week="2026-W38", now=NIGHT, path=paths["questions_path"])

    out = week_questions.run_week_questions(session_date="2026-09-18", now=NIGHT, request=_FakeModel(fail=True),
                                            root=root, **paths)
    assert out["status"] == "degraded_no_narrative"
    assert len(week_coach.pending_questions(**paths)) == 1
    assert not paths["answers_path"].exists()

    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: False)
    out = week_questions.run_week_questions(session_date="2026-09-18", now=NIGHT, root=root, **paths)
    assert out["status"] == "degraded_no_narrative" and "stay pending" in out["reason"]

    # The forced-by-day half writes the digest and asks nothing.
    out = week_questions.run_week_questions(session_date="2026-09-18", now=NIGHT, answer=False,
                                            request=_FakeModel(fail=True), root=root, **paths)
    assert out["status"] == "ok" and "stay pending" in out["reason"]
    assert (root / "week-2026-W38-frontier.md").is_file()
    assert len(week_coach.pending_questions(**paths)) == 1


def test_rule_questions_send_rule_rows_and_no_records_means_no_cited_answer(tmp_path):
    import week_coach
    from ai_jobs import week_questions

    root = tmp_path / "records"
    _afternoon_week(root)
    paths = {"questions_path": tmp_path / "q.jsonl", "answers_path": tmp_path / "a.jsonl"}
    week_coach.record_question("Do I keep my rule?", week="2026-W38", now=NIGHT, path=paths["questions_path"])
    week_coach.record_question("What about 2025-01-06?", now=NIGHT + timedelta(seconds=1),
                               path=paths["questions_path"])
    fake = _FakeModel([{"text": "Kept once of two.", "citations": ["recap:rc-c1", "recap:rc-c2"]}])
    week_questions.run_week_questions(session_date="2026-09-18", now=NIGHT, request=fake, root=root, **paths)
    assert len(fake.evidence) == 1  # the 2025 question had no records: the model is not asked
    assert "rule_kept" in fake.evidence[0]["weeks_read"][0]
    by_text = {row["text"]: row for row in week_coach.read_questions(**paths)}
    assert by_text["Do I keep my rule?"]["status"] == "answered"
    assert by_text["Do I keep my rule?"]["answer"]["citations"][0]["session"] == "2026-09-15"
    empty = by_text["What about 2025-01-06?"]
    assert empty["status"] == "no_cited_answer" and empty["answer"]["claims"] == []


# ---------------------------------------------------------------------------
# the night slot
# ---------------------------------------------------------------------------
def test_the_night_slot_is_registered_after_day_facts_at_the_end_of_stage_two():
    from ai_jobs import runner

    names = [slot.name for slot in runner.default_slots()]
    assert names.index("week_questions") > names.index("day_review_facts")
    assert names[names.index("week_questions") - 1] == "market_story_narration"
    assert names[names.index("week_questions") + 1] == "journal_enrichment"
    slot = next(slot for slot in runner.default_slots() if slot.name == "week_questions")
    assert slot.uses_model is True and slot.max_attempts == 3
    assert slot.model_free_kwargs == {"answer": False}
    assert "week_questions" in [slot.name for slot in runner.slots_for("weeknight")]
    assert "week_questions" in [slot.name for slot in runner.slots_for("saturday")]
