"""P2 pick assessment: the structured call, the citation drops, /pick and the Focus chips,
the 06:15 / hourly prefetch, and guardrail 2's grey numbers."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import assess, commands, grounding, pick_jobs  # noqa: E402
from mentor_packs import pick_pack  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
NOW = pick_pack.FIXTURE_NOW

GOOD_REPLY = {
    "verdict": "worth a look",
    "bullets": [
        {"text": "The breakout cell wins 60% over n=40.", "evidence_refs": ["pick:NVDA:cell"]},
        {"text": "AVGO reports tomorrow.", "evidence_refs": ["pick:NVDA:peer:AVGO"]},
        {"text": "It just feels strong.", "evidence_refs": []},
    ],
    "rule_flags": [
        {"plan_id": "plan:risk:2", "breaks": False, "text": "entry is before 12:30"},
        {"plan_id": "plan:made:up", "breaks": True, "text": "invented rule"},
    ],
}


class _Response:
    status_code = 200

    def __init__(self, reply):
        self._body = {"id": "r1", "choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}]}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class _FakePost:
    def __init__(self, reply=GOOD_REPLY):
        self.reply = reply
        self.payloads: list[dict] = []

    def __call__(self, url, **kwargs):
        self.payloads.append({"url": url, **kwargs["json"]})
        return _Response(self.reply)


@pytest.fixture()
def world(tmp_path):
    return pick_pack.write_fixture_world(tmp_path / "desk")


@pytest.fixture()
def nvda(world):
    pack = pick_pack.build("NVDA", now=NOW, paths=world)
    return pack, pick_pack.pack_hash(pack)


# ---------------------------------------------------------------- the structured call
def test_the_call_goes_through_request_ai_summary_with_the_schema_and_a_600_token_cap(nvda):
    pack, digest = nvda
    post = _FakePost()
    out = assess.assess(pack, symbol="NVDA", pack_hash=digest, model="gpt-oss:20b",
                        endpoint="http://127.0.0.1:11436", post=post)
    sent = post.payloads[-1]
    assert sent["url"] == "http://127.0.0.1:11436/v1/chat/completions"
    assert sent["max_tokens"] <= 600
    assert sent["response_format"]["json_schema"]["name"] == assess.SCHEMA_NAME
    assert sent["reasoning_effort"] == "high", "a background assessment thinks hard"
    assert '"source_id": "pick:NVDA:cell"' in sent["messages"][1]["content"], "the model sees every row by its id"
    assert out.narrated and out.verdict == "worth a look" and out.effort == "high"


def test_a_live_ask_uses_medium_effort_and_gemma_gets_none(nvda):
    pack, digest = nvda
    post = _FakePost()
    assess.assess(pack, symbol="NVDA", pack_hash=digest, model="gpt-oss:20b", endpoint="http://h", post=post, live=True)
    assert post.payloads[-1]["reasoning_effort"] == "medium"
    assess.assess(pack, symbol="NVDA", pack_hash=digest, model="gemma3:12b", endpoint="http://h", post=post)
    assert "reasoning_effort" not in post.payloads[-1]


def test_uncited_bullets_and_unknown_plan_ids_are_dropped_and_logged(nvda, caplog):
    pack, digest = nvda
    caplog.set_level("INFO")
    out = assess.assess(pack, symbol="NVDA", pack_hash=digest, model="gpt-oss:20b", endpoint="http://h", post=_FakePost())
    assert [b["evidence_refs"] for b in out.bullets] == [["pick:NVDA:cell"], ["pick:NVDA:peer:AVGO"]]
    assert out.rule_flags == [{"plan_id": "plan:risk:2", "breaks": False, "text": "entry is before 12:30"}]
    reasons = sorted((d["kind"], d["reason"]) for d in out.dropped)
    assert reasons == [("bullet", "uncited"), ("rule_flag", "unknown plan id")]
    assert "dropped" in caplog.text and "plan:made:up" in caplog.text


def test_a_plan_flag_may_cite_the_pick_row_id_too(nvda):
    pack, _ = nvda
    reply = {"verdict": "wait", "bullets": [{"text": "x", "evidence_refs": ["pick:NVDA:cell"]}],
             "rule_flags": [{"plan_id": "pick:NVDA:plan:risk:1", "breaks": True, "text": "fourth short"}]}
    _, flags, _ = assess.check_reply(reply, pack)
    assert flags == [{"plan_id": "plan:risk:1", "breaks": True, "text": "fourth short"}]


def test_an_invented_evidence_id_rejects_the_whole_reply(nvda):
    pack, digest = nvda
    bad = {**GOOD_REPLY, "bullets": [{"text": "Win rate 90%", "evidence_refs": ["pick:NVDA:made_up"]}]}
    out = assess.assess(pack, symbol="NVDA", pack_hash=digest, model="m", endpoint="http://h", post=_FakePost(bad))
    assert not out.narrated and out.bullets == [] and "rejected" in out.error


def test_with_an_empty_plan_every_rule_flag_is_dropped(tmp_path):
    paths = pick_pack.write_fixture_world(tmp_path / "desk", plan_text="# Trading plan\n")
    pack = pick_pack.build("NVDA", now=NOW, paths=paths)
    _, flags, drops = assess.check_reply(GOOD_REPLY, pack)
    assert flags == [] and sum(d["kind"] == "rule_flag" for d in drops) == 2


def test_a_failed_call_is_an_error_card_that_still_links_the_evidence(nvda):
    pack, digest = nvda

    def down(url, **kwargs):
        raise ConnectionError("tunnel down")

    out = assess.assess(pack, symbol="NVDA", pack_hash=digest, model="m", endpoint="http://h", post=down)
    assert not out.narrated and out.error
    card = assess.card_markdown(out)
    assert f"(evidence:NVDA:{digest})" in card and "no assessment" in card


def test_the_card_cites_every_bullet_and_says_how_many_were_dropped(nvda):
    pack, digest = nvda
    out = assess.assess(pack, symbol="NVDA", pack_hash=digest, model="m", endpoint="http://h", post=_FakePost())
    card = assess.card_markdown(out, side="LONG")
    assert card.startswith("**Pick NVDA LONG**: **worth a look**")
    assert "[pick:NVDA:cell]" in card and "[plan:risk:2]" in card and "1 uncited point dropped" in card
    assert Assessment_round_trip(out) == out


def Assessment_round_trip(out):  # noqa: N802 - helper named for what it checks
    return assess.Assessment.from_json(out.to_json())


# ---------------------------------------------------------------- commands and jobs
def test_pick_command_parses_symbol_and_side():
    assert commands.handle("/pick nvda").arg == ("NVDA", "")
    assert commands.handle("/pick tsla short").arg == ("TSLA", "SHORT")
    assert commands.handle("/pick").action == "error"
    assert commands.handle("/pick ;drop").action == "error"
    assert "/pick SYM" in commands.handle("/help").reply


def test_the_schedule_runs_at_0615_then_hourly():
    schedule = pick_jobs.PickSchedule()
    at = lambda h, m: datetime(2026, 9, 30, h, m, tzinfo=PT)  # noqa: E731
    assert schedule.due(at(6, 14)) == ""
    assert schedule.due(at(6, 15)) == "morning"
    schedule.mark("morning", at(6, 15))
    assert schedule.due(at(6, 50)) == ""
    assert schedule.due(at(7, 15)) == "hourly"
    schedule.mark("hourly", at(7, 15))
    assert schedule.due(at(8, 0)) == ""
    assert schedule.due(datetime(2026, 10, 1, 6, 20, tzinfo=PT)) == "morning"


def test_focus_names_are_swing_first_and_unique():
    focus = {"swing": {"long": ["nvda"], "short": ["TSLA"]}, "m5": {"long": ["NVDA"], "short": ["AMD"]}}
    assert pick_jobs.focus_names(focus) == [("NVDA", "LONG"), ("TSLA", "SHORT"), ("AMD", "SHORT")]


class _Store:
    def __init__(self):
        self.rows: dict[tuple, str] = {}

    def get_pack(self, name, args):
        text = self.rows.get((name, json.dumps(args, sort_keys=True)))
        return {"pack_json": text} if text else None

    def put_pack(self, name, args, pack_json, built_utc=""):
        self.rows[(name, json.dumps(args, sort_keys=True))] = pack_json


def _job(world, store, calls, *, kind, now=NOW, should_yield=lambda: False, symbol="NVDA"):
    def narrate(pack, digest):
        calls.append(digest)
        return assess.assess(pack, symbol=symbol, pack_hash=digest, model="m", endpoint="http://h", post=_FakePost(),
                             now=lambda: now)

    return pick_jobs.run_pick_job(
        symbol, "LONG", kind=kind, store=store,
        build_pack=lambda sym, side: pick_pack.build(sym, side, now=now, paths=world),
        pack_hash=pick_pack.pack_hash, narrate=narrate, now=lambda: now, should_yield=should_yield,
    )


def test_an_hourly_pass_narrates_only_a_changed_pack(world):
    store, calls = _Store(), []
    assert _job(world, store, calls, kind="morning")["narrated"]
    assert not _job(world, store, calls, kind="hourly")["narrated"], "same hash, cached card reused"
    world.pick_feedback.write_text(
        json.dumps({"ts": "2026-09-29T06:59:00", "trade_date": "2026-09-29", "symbol": "NVDA", "side": "LONG",
                    "verdict": "dislike", "category": "swing", "origin": "d1", "reason": "gap up"}) + "\n",
        encoding="utf-8",
    )
    assert _job(world, store, calls, kind="hourly")["narrated"]
    assert len(calls) == 2


def test_the_morning_pass_renarrates_a_card_from_yesterday(world):
    store, calls = _Store(), []
    yesterday = NOW.replace(day=28)
    _job(world, store, calls, kind="morning", now=yesterday)
    # Same evidence text is impossible across days here (days-to-earnings moves), so pin the rule directly.
    old = assess.Assessment(symbol="NVDA", pack_hash="h", verdict="wait", bullets=[{"text": "x"}],
                            built_utc=yesterday.isoformat())
    assert pick_jobs.needs_narration(old, "morning", NOW.astimezone(PT).date())
    assert not pick_jobs.needs_narration(old, "hourly", NOW.astimezone(PT).date())


def test_a_job_yields_its_model_call_to_a_waiting_chat_turn(world):
    store, calls = _Store(), []
    result = _job(world, store, calls, kind="morning", should_yield=lambda: True)
    assert result["yielded"] and not result["narrated"] and calls == []


# ---------------------------------------------------------------- guardrail 2
def test_a_cited_number_stays_and_an_invented_one_is_grey():
    pack_text = "[pick:NVDA:cell] Setup cell ...: n=40 (floor 30), win rate 60%"
    reply = "The cell wins 60% over n=40 [pick:NVDA:cell], but I'd guess 73% next week."
    shown = grounding.mark_uncited_numbers(reply, [pack_text])
    assert "60%" in shown and '<span class="uncited" style="color:#8a8a8a">73%</span>' in shown
    assert '<span class="uncited" style="color:#8a8a8a">60%</span>' not in shown
    assert "[pick:NVDA:cell]" in shown, "citation digits are never greyed"
    assert shown.replace('<span class="uncited" style="color:#8a8a8a">73%</span>', "73%") == reply


def test_number_matching_ignores_sign_currency_and_grouping():
    assert grounding.grounded_numbers(["avg side return +0.60%, $1,234.50"]) >= {"0.6", "1234.5"}
    assert "grey" not in grounding.mark_uncited_numbers("-0.6% and 1234.5", ["+0.60%, $1,234.50"])
    assert "uncited" not in grounding.mark_uncited_numbers("-0.6% and 1234.5", ["+0.60%, $1,234.50"])


# ---------------------------------------------------------------- the window
Qt = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture()
def app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(app, world, tmp_path, monkeypatch):
    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.store import MentorChatStore
    from mentor_app.window import MentorWindow

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    requests_made: list[dict] = []

    def request(**kwargs):
        requests_made.append(kwargs)
        return {"summary": GOOD_REPLY if kwargs["evidence"]["rows"][0]["source_id"].startswith("pick:NVDA") else {
            "verdict": "wait", "bullets": [{"text": "cell", "evidence_refs": [kwargs["evidence"]["rows"][5]["source_id"]]}],
            "rule_flags": []}}

    clock = {"now": NOW}
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        queue=PrefetchQueue(),
        stream_post=lambda url, payload, cancelled: [],
        post=lambda url, payload, timeout: {},
        now=lambda: clock["now"],
        mentor_enabled=False,
        pick_builder=lambda sym, side: pick_pack.build(sym, side, now=clock["now"], paths=world),
        assess_request=request,
        focus_source=lambda: {"swing": {"long": ["NVDA"], "short": ["TSLA"]}, "m5": {"long": [], "short": []}},
    )
    win.requests_made = requests_made
    win.clock = clock
    yield win
    win.shutdown()
    win.deleteLater()


def _drain(win, app):
    for _ in range(50):
        app.processEvents()
        if not win.queue.run_one():
            app.processEvents()
            if not win.queue.pending():
                break
    win._io.submit(lambda: None).result(5)


def _text(win):
    return win.transcript.toPlainText()


def test_pick_with_the_brain_off_shows_the_evidence_not_a_guess(window, app):
    window._brain_reason = "not connected"
    window.send("/pick NVDA")
    assert "building" in _text(window)
    _drain(window, app)
    assert "no assessment (the brain is off: not connected)" in _text(window)
    assert window.requests_made == []
    link = next(key for key in window._pick_packs if key[0] == "NVDA")
    from PySide6.QtCore import QUrl

    window._on_anchor(QUrl(f"evidence:NVDA:{link[1]}"))
    assert "[pick:NVDA:peer:AVGO]" in _text(window), "show evidence prints the raw pack"


def test_pick_narrates_once_then_answers_from_the_cache_instantly(window, app):
    window._brain_ok, window._endpoint, window._model = True, "http://127.0.0.1:11436", "gpt-oss:20b"
    window.send("/pick nvda")
    _drain(window, app)
    text = _text(window)
    assert "Pick NVDA: worth a look" in text and "[pick:NVDA:cell]" in text and "narrating" not in text
    assert len(window.requests_made) == 1
    assert window.requests_made[0]["endpoint"] == "http://127.0.0.1:11436/v1"
    # The card joined the conversation (follow-ups see it) and the turn logged the drops.
    assert window.chat.turns[-1].text.startswith("**Pick NVDA")
    stored = window.store.turns()[-1]
    assert "unknown plan id" in stored["tool_calls_json"] and "pick:NVDA:cell" in stored["pack_ids_json"]
    before = len(window._blocks)
    window.show_pick("NVDA", "LONG")
    assert len(window._blocks) == before + 1 and "worth a look" in window._blocks[-1], "cached card at once"
    _drain(window, app)
    assert len(window.requests_made) == 1, "an unchanged pack is never narrated twice"


def test_a_changed_pack_is_renarrated(window, app, world):
    window._brain_ok, window._endpoint, window._model = True, "http://h", "gemma3:12b"
    window.show_pick("NVDA")
    _drain(window, app)
    world.focus_swing_longs.write_text("ZZZ\n", encoding="utf-8")  # NVDA left Focus: new evidence
    window.show_pick("NVDA")
    _drain(window, app)
    assert len(window.requests_made) == 2 and "evidence changed" in _text(window)


def test_focus_chips_open_their_pick(window, app):
    from mentor_packs import context_pack

    window._on_context(context_pack.fixture())
    assert list(window.pick_chips) == ["AAPL", "TSLA", "AMD", "NVDA"]
    assert window.pick_chips["TSLA"].text() == "TSLA S"
    window.pick_chips["NVDA"].click()
    assert "Pick NVDA" in _text(window)


def test_the_0615_prefetch_assesses_every_focus_name_quietly(window, app):
    window._brain_ok, window._endpoint, window._model = True, "http://h", "gpt-oss:20b"
    window.clock["now"] = datetime(2026, 9, 29, 6, 10, tzinfo=PT)
    window.maybe_prefetch_picks()
    assert window.queue.pending() == []
    window.clock["now"] = datetime(2026, 9, 29, 6, 15, tzinfo=PT)
    before = _text(window)
    window.maybe_prefetch_picks()
    _drain(window, app)
    assert sorted(window._pick_cards) == ["NVDA", "TSLA"]
    assert _text(window) == before, "prefetched cards never move the transcript"
    assert all(call["post"] is not None for call in window.requests_made) and len(window.requests_made) == 2
    # An hour later nothing changed: no model call.
    window.clock["now"] = datetime(2026, 9, 29, 7, 20, tzinfo=PT)
    window.maybe_prefetch_picks()
    _drain(window, app)
    assert len(window.requests_made) == 2


def test_the_prefetch_waits_for_the_brain(window, app):
    window.clock["now"] = datetime(2026, 9, 29, 6, 20, tzinfo=PT)
    window.maybe_prefetch_picks()
    assert window.queue.pending() == [] and window._pick_schedule.last_morning is None


def test_a_free_chat_reply_greys_an_invented_number(window):
    window._context_text = "[ctx:focus:swing:long] Focus swing longs (2): AAPL, NVDA"
    window._blocks.append("**Mentor:** ")
    window._worker = object()
    window.queue.begin_interactive()
    window._on_done({"text": "You have 2 swing longs [ctx:focus:swing:long] and 7 shorts.",
                     "pack_texts": [], "model": "m"})
    html = window.transcript.toHtml()
    assert "#8a8a8a" in html
    assert window._blocks[-1].count('class="uncited"') == 1 and ">7</span>" in window._blocks[-1]


def test_a_second_click_while_building_strands_no_placeholder(window, app):
    window.show_pick("NVDA")
    window.show_pick("NVDA")
    assert sum("building" in block for block in window._blocks) == 1
    _drain(window, app)
    assert not any("building" in block for block in window._blocks)


def test_a_failed_build_says_so_and_frees_the_symbol(window, app):
    window._pick_builder = lambda sym, side: (_ for _ in ()).throw(OSError("disk gone"))
    window.show_pick("NVDA")
    _drain(window, app)
    assert "could not be built (OSError: disk gone)" in _text(window)
    assert "NVDA" not in window._pick_blocks
