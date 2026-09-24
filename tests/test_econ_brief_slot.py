"""The econ morning brief: the night slot, its validation, and the Mentor's view.

The night reads the newest pasted brief and asks the local model for 3-6 lines.
Every time and event in a kept reply is in the deterministic pack; anything
else keeps the last good file and the Mentor shows the brief's own lines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "day_review"
BRIEF_0924 = (FIXTURES / "forecast_2026-09-24.md").read_text(encoding="utf-8")
BRIEF_0924_CAL = (FIXTURES / "forecast_2026-09-24_calendar.md").read_text(encoding="utf-8")
BRIEF_0917 = (FIXTURES / "forecast_2026-09-17.md").read_text(encoding="utf-8")


def _forecasts(*pairs):
    return [
        {"session": session, "text": text, "entry_id": f"e-{session}", "created_at": f"{session}T08:55:00-04:00"}
        for session, text in pairs
    ]


def _pack(target="2026-09-25", brief=BRIEF_0924):
    import econ_brief

    return econ_brief.build_pack(_forecasts(("2026-09-24", brief)), target_session=target)


def _ids(pack, needle):
    rows = list(pack["today"]) + list(pack["week"])
    return [row["id"] for row in rows if needle.casefold() in row["label"].casefold()]


def _good_reply(pack):
    durable = _ids(pack, "durable")[0]
    michigan = _ids(pack, "michigan")[0]
    jolts = _ids(pack, "jolts")[0]
    return {
        "lines": [
            {"text": "8:30 a.m.: durable goods orders. Rates may jump on a hot print.", "event_ids": [durable]},
            {"text": "10:00: Michigan sentiment and inflation expectations.", "event_ids": [michigan]},
            {"text": "Next week: JOLTS on Tuesday. Keep size small into it.", "event_ids": [jolts]},
        ]
    }


def _request_returning(reply):
    calls = []

    def _request(**kwargs):
        calls.append(kwargs)
        return {"summary": reply, "model": "local-test"}

    _request.calls = calls
    return _request


# ---------------------------------------------------------------------------
# the pack
# ---------------------------------------------------------------------------
def test_the_pack_holds_the_next_sessions_events_and_the_week():
    pack = _pack()
    assert pack["brief_session"] == "2026-09-24"
    today = [(row["time_et"], row["label"]) for row in pack["today"]]
    assert ("08:30", "August durable goods orders") in today
    assert ("10:00", "Michigan sentiment") in today
    week_dates = {row["date"] for row in pack["week"]}
    assert {"2026-09-29", "2026-09-30", "2026-10-02"} <= week_dates
    # The 09-24 events are yesterday's for a 09-25 pack.
    assert all(row["date"] >= "2026-09-25" for row in pack["today"] + pack["week"])
    assert pack["playbook_bullish"].startswith("Bullish reversal:")
    assert pack["turbulence_lines"]


def test_forecasts_are_one_per_session_newest_first_and_never_after_up_to():
    import econ_brief

    rows = [
        {"origin": "external_forecast", "session_date": "2026-09-17", "written_session_date": "2026-09-17",
         "text": BRIEF_0917, "created_at": "2026-09-17T09:00:00-04:00", "entry_id": "a"},
        {"origin": "external_forecast", "session_date": "2026-09-24", "written_session_date": "2026-09-24",
         "text": BRIEF_0924, "created_at": "2026-09-24T09:00:00-04:00", "entry_id": "b"},
        {"origin": "desk_tab", "session_date": "2026-09-24", "written_session_date": "2026-09-24",
         "text": "my note", "created_at": "2026-09-24T09:10:00-04:00", "entry_id": "c"},
    ]
    out = econ_brief.forecasts_from_rows(rows, up_to="2026-09-24")
    assert [item["entry_id"] for item in out] == ["b", "a"]
    assert [item["entry_id"] for item in econ_brief.forecasts_from_rows(rows, up_to="2026-09-20")] == ["a"]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_a_grounded_reply_is_kept():
    from ai_jobs import econ_brief_narration as job

    pack = _pack()
    lines = job.validate(_good_reply(pack), pack)
    assert len(lines) == 3


def test_a_time_the_pack_does_not_hold_is_rejected():
    from ai_jobs import econ_brief_narration as job

    pack = _pack()
    reply = _good_reply(pack)
    reply["lines"][0]["text"] = "9:15 a.m.: durable goods orders."
    with pytest.raises(ValueError, match="time"):
        job.validate(reply, pack)


def test_a_time_with_no_cited_event_is_rejected():
    from ai_jobs import econ_brief_narration as job

    pack = _pack()
    reply = _good_reply(pack)
    reply["lines"][2] = {"text": "Something big lands at 2 p.m.", "event_ids": []}
    with pytest.raises(ValueError):
        job.validate(reply, pack)


def test_an_event_id_outside_the_pack_is_rejected():
    from ai_jobs import econ_brief_narration as job

    pack = _pack()
    reply = _good_reply(pack)
    reply["lines"][1]["event_ids"] = ["x9"]
    with pytest.raises(ValueError, match="outside"):
        job.validate(reply, pack)


def test_a_release_the_brief_does_not_hold_is_rejected():
    from ai_jobs import econ_brief_narration as job

    pack = _pack()
    reply = _good_reply(pack)
    reply["lines"][2] = {"text": "CPI is the big one this week.", "event_ids": []}
    with pytest.raises(ValueError, match="CPI|cpi"):
        job.validate(reply, pack)


def test_too_few_or_too_many_lines_are_rejected():
    from ai_jobs import econ_brief_narration as job

    pack = _pack()
    reply = _good_reply(pack)
    with pytest.raises(ValueError):
        job.validate({"lines": reply["lines"][:2]}, pack)
    with pytest.raises(ValueError):
        job.validate({"lines": reply["lines"] * 3}, pack)


# ---------------------------------------------------------------------------
# the slot
# ---------------------------------------------------------------------------
def test_the_slot_writes_the_next_sessions_file(tmp_path):
    from ai_jobs import econ_brief_narration as job

    request = _request_returning(_good_reply(_pack()))
    outcome = job.run_econ_brief(
        session_date="2026-09-24",
        out_dir=tmp_path,
        forecasts=_forecasts(("2026-09-24", BRIEF_0924)),
        request=request,
    )
    assert outcome["status"] == "ok", outcome
    payload = json.loads((tmp_path / "2026-09-25.json").read_text(encoding="utf-8"))
    assert payload["target_session"] == "2026-09-25"
    assert len(payload["summary_lines"]) == 3
    assert request.calls and request.calls[0]["provider"] == "local"


def test_an_invalid_reply_keeps_the_last_good_file(tmp_path):
    from ai_jobs import econ_brief_narration as job

    forecasts = _forecasts(("2026-09-24", BRIEF_0924))
    job.run_econ_brief(
        session_date="2026-09-24", out_dir=tmp_path, forecasts=forecasts,
        request=_request_returning(_good_reply(_pack())),
    )
    before = (tmp_path / "2026-09-25.json").read_text(encoding="utf-8")
    bad = {"lines": [{"text": "CPI at 9:15 a.m.", "event_ids": []}] * 3}
    # A different pack (the older brief added) so the unchanged-inputs shortcut is not taken.
    outcome = job.run_econ_brief(
        session_date="2026-09-24", out_dir=tmp_path,
        forecasts=_forecasts(("2026-09-24", BRIEF_0924 + "\n\nExtra line.")),
        request=_request_returning(bad),
    )
    assert outcome["status"] == "degraded_no_narrative"
    assert (tmp_path / "2026-09-25.json").read_text(encoding="utf-8") == before


def test_a_failed_model_call_writes_nothing(tmp_path):
    from ai_jobs import econ_brief_narration as job

    def _boom(**_kwargs):
        raise RuntimeError("endpoint down")

    outcome = job.run_econ_brief(
        session_date="2026-09-24", out_dir=tmp_path,
        forecasts=_forecasts(("2026-09-24", BRIEF_0924)), request=_boom,
    )
    assert outcome["status"] == "degraded_no_narrative"
    assert not list(tmp_path.glob("*.json"))


def test_no_brief_pasted_calls_no_model_and_says_so(tmp_path):
    from ai_jobs import econ_brief_narration as job

    request = _request_returning(_good_reply(_pack()))
    outcome = job.run_econ_brief(session_date="2026-09-24", out_dir=tmp_path, forecasts=[], request=request)
    assert outcome["status"] == "skipped"
    assert "No brief pasted" in outcome["reason"]
    assert request.calls == []


def test_the_slot_is_registered_honestly():
    from ai_jobs import runner

    slot = {slot.name: slot for slot in runner.default_slots()}["econ_brief"]
    assert slot.uses_model is True
    assert slot.model_free_kwargs is None
    assert slot.max_attempts == 3


# ---------------------------------------------------------------------------
# the Mentor's view
# ---------------------------------------------------------------------------
def test_the_view_uses_the_night_summary_made_from_the_same_brief(tmp_path):
    import econ_brief
    from ai_jobs import econ_brief_narration as job

    forecasts = _forecasts(("2026-09-24", BRIEF_0924))
    job.run_econ_brief(
        session_date="2026-09-24", out_dir=tmp_path, forecasts=forecasts,
        request=_request_returning(_good_reply(_pack())),
    )
    view = econ_brief.today_view("2026-09-25", forecasts=forecasts, out_dir=tmp_path)
    assert view["origin"] == econ_brief.ORIGIN_NIGHT
    assert len(view["summary_lines"]) == 3
    assert [row["time_et"] for row in view["today"] if row["time_et"]] == ["08:30", "10:00"]


def test_the_view_falls_back_to_the_briefs_own_lines_with_no_night_file(tmp_path):
    import econ_brief

    view = econ_brief.today_view(
        "2026-09-25", forecasts=_forecasts(("2026-09-24", BRIEF_0924)), out_dir=tmp_path
    )
    assert view["origin"] == econ_brief.ORIGIN_LAST_BRIEF
    assert view["summary_lines"] and view["summary_lines"][0].startswith("Watch, in order:")
    assert view["today"]


def test_todays_paste_replaces_the_night_summary_with_todays_brief(tmp_path):
    import econ_brief
    from ai_jobs import econ_brief_narration as job

    job.run_econ_brief(
        session_date="2026-09-24", out_dir=tmp_path,
        forecasts=_forecasts(("2026-09-24", BRIEF_0924)),
        request=_request_returning(_good_reply(_pack())),
    )
    todays = BRIEF_0924_CAL.replace("Thursday, September 24, 2026", "Friday, September 25, 2026")
    view = econ_brief.today_view(
        "2026-09-25",
        forecasts=_forecasts(("2026-09-25", todays), ("2026-09-24", BRIEF_0924)),
        out_dir=tmp_path,
    )
    assert view["origin"] == econ_brief.ORIGIN_TODAY_BRIEF
    assert view["origin_text"] == "from today's brief"
    assert view["brief_session"] == "2026-09-25"
    assert view["unread_lines"] == 1


def test_no_brief_pasted_is_said_plainly(tmp_path):
    import econ_brief

    view = econ_brief.today_view("2026-09-25", forecasts=[], out_dir=tmp_path)
    assert view["origin"] == econ_brief.ORIGIN_NONE
    assert view["note"] == "No brief pasted."
    assert view["today"] == [] and view["summary_lines"] == []
