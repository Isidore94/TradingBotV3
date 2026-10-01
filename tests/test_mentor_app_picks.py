"""P2 pick assessment: the structured call, the citation drops, /pick and the liked names,
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
    scope = {"value": "liked"}
    monkeypatch.setattr(settings, "prefetch_scope", lambda: scope["value"])
    gpu = {"reason": ""}
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: gpu["reason"])
    liked = [("NVDA", "LONG"), ("TSLA", "SHORT")]
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
        focus_source=lambda: {"swing": {"long": ["NVDA"], "short": ["TSLA"]}, "m5": {"long": [], "short": ["AMD"]}},
        liked_source=lambda: list(liked),
    )
    win.requests_made = requests_made
    win.scope, win.gpu, win.liked = scope, gpu, liked
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


def test_liked_picks_are_kept_without_ticker_chips(window, app):
    from PySide6.QtWidgets import QPushButton

    # Trader 2026-10-01: the ticker chips gave way to quick buttons; the liked list still feeds
    # auto-attach and the prefetch, and `/pick SYM` still opens a card.
    window.refresh_liked()
    _drain(window, app)
    assert [sym for sym, _side in window._liked_names] == ["NVDA", "TSLA"], "AMD is on Focus but not liked"
    assert not window.findChildren(QPushButton, "MentorPickChip"), "no ticker chips"
    window.send("/pick NVDA")
    assert "Pick NVDA" in _text(window)


def test_the_quick_buttons_run_tape_tilt_mirror_and_scorecard(window, monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(window, "send", lambda text: sent.append(text))
    assert [b.text() for b in window.quick_buttons.values()] == ["Tape", "Tilt", "Mirror", "Scorecard"]
    for button in window.quick_buttons.values():
        assert button.toolTip()
        button.click()
    assert sent == ["/tape", "/tilt", "/mirror", "/scorecard"]


def test_the_0615_prefetch_assesses_the_liked_picks_quietly(window, app):
    window._brain_ok, window._endpoint, window._model = True, "http://h", "gpt-oss:20b"
    window.clock["now"] = datetime(2026, 9, 29, 6, 10, tzinfo=PT)
    window.maybe_prefetch_picks()
    assert window.queue.pending() == []
    window.clock["now"] = datetime(2026, 9, 29, 6, 15, tzinfo=PT)
    before = _text(window)
    window.maybe_prefetch_picks()
    _drain(window, app)
    assert sorted(window._pick_cards) == ["NVDA", "TSLA"], "AMD (Focus, not liked) is not prefetched"
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


# ---------------------------------------------------------------- review fixes
def test_a_chip_tap_mid_stream_never_eats_the_answer(window, app):
    """Reviewer's repro: tokens after a pick placeholder landed in it and were overwritten."""
    window._brain_reason = "not connected"
    window._blocks[:] = ["Hi. Ask me anything.", "**Mentor:** "]
    window._stream_index = 1
    window._worker = object()
    window.queue.begin_interactive()
    window._on_token("The answer is")
    window.show_pick("NVDA")
    window._on_token(" forty-two.")
    window._on_done({"text": "The answer is forty-two.", "pack_texts": [], "model": "m"})
    _drain(window, app)
    text = _text(window)
    assert "The answer is forty-two." in text
    assert "Pick NVDA" in text and "no assessment" in text
    assert window._blocks[1] == "**Mentor:** The answer is forty-two."


def test_the_liked_set_is_claims_likes_and_favourites_newest_first(tmp_path):
    from datetime import date

    claims = tmp_path / "claimed_picks.jsonl"
    claims.write_text(json.dumps({"action": "claim", "symbol": "FORM", "side": "LONG", "claimed_setup_id": "x",
                                  "claim_at": "2026-09-29T12:10:02-07:00", "session_date": "2026-09-29"}) + "\n",
                      encoding="utf-8")
    feedback = tmp_path / "pick_feedback.jsonl"
    feedback.write_text("\n".join(json.dumps(row) for row in (
        {"ts": "2026-09-28T07:00:00", "trade_date": "2026-09-28", "symbol": "NVDA", "side": "LONG", "verdict": "like"},
        {"ts": "2026-08-01T07:00:00", "trade_date": "2026-08-01", "symbol": "OLD", "side": "LONG", "verdict": "like"},
        {"ts": "2026-09-29T08:00:00", "trade_date": "2026-09-29", "symbol": "BAD", "side": "LONG", "verdict": "dislike"},
    )) + "\n", encoding="utf-8")
    favourites = tmp_path / "swing_favorites.jsonl"
    favourites.write_text("\n".join(json.dumps(row) for row in (
        {"action": "add", "symbol": "SHOP", "side": "long", "session_date": "2026-09-29", "event_at": "2026-09-29T13:00:00-07:00"},
        {"action": "add", "symbol": "GONE", "side": "long", "session_date": "2026-09-29", "event_at": "2026-09-29T13:01:00-07:00"},
        {"action": "remove", "symbol": "GONE", "side": "long", "session_date": "2026-09-29", "event_at": "2026-09-29T13:02:00-07:00"},
    )) + "\n", encoding="utf-8")
    got = pick_jobs.liked_picks(today=date(2026, 9, 29), claims_path=claims, feedback_path=feedback,
                                favorites_path=favourites)
    assert got == [("SHOP", "LONG"), ("FORM", "LONG"), ("NVDA", "LONG")]


def test_scope_all_adds_the_rest_of_focus_at_low_effort_when_idle(window, app):
    from mentor_app.prefetch import PRIORITY_IDLE

    window._brain_ok, window._endpoint, window._model = True, "http://h", "gpt-oss:20b"
    window.scope["value"] = "all"
    window.clock["now"] = datetime(2026, 9, 29, 6, 15, tzinfo=PT)
    efforts: dict[str, str] = {}
    real = assess.assess

    def spy(pack, **kwargs):
        efforts[kwargs["symbol"]] = kwargs.get("effort")
        return real(pack, **kwargs)

    import mentor_app.window as window_module

    window_module.pick_assess.assess, saved = spy, real
    try:
        window.maybe_prefetch_picks()
        window.queue.run_one()  # the plan job queues the names
        jobs = {job.name: job.priority for job in window.queue._jobs}
        assert jobs["pick_prefetch AMD"] == PRIORITY_IDLE and jobs["pick_prefetch NVDA"] < PRIORITY_IDLE
        _drain(window, app)
    finally:
        window_module.pick_assess.assess = saved
    assert efforts == {"NVDA": "high", "TSLA": "high", "AMD": "low"}
    assert list(window.queue.ran[-3:]) == ["pick_prefetch NVDA", "pick_prefetch TSLA", "pick_prefetch AMD"]


def test_a_narration_that_never_gets_the_model_shows_the_evidence_and_frees_the_symbol(window, app):
    import time

    window._brain_ok, window._endpoint, window._model = True, "http://h", "m"
    window.assess_wait_ms = 20
    window.show_pick("NVDA")
    window.queue.run_one()  # the pack builds
    app.processEvents()
    window.gpu["reason"] = "the night AI owns the GPU"  # the model job now waits
    assert "narrating" in _text(window)
    deadline = time.monotonic() + 2
    while "NVDA" in window._pick_blocks and time.monotonic() < deadline:
        app.processEvents()
    text = _text(window)
    assert "no assessment (the brain is busy or off" in text and "narrating" not in text
    assert "NVDA" not in window._pick_blocks
    assert "pick-assess:NVDA" not in window.queue.pending_keys()
    assert "evidence:NVDA:" in window._blocks[-1]


def test_a_prefetched_card_replaces_the_waiting_placeholder_once(window, app):
    window._brain_ok, window._endpoint, window._model = True, "http://h", "m"
    window.gpu["reason"] = "busy"  # the live narration will wait
    window.show_pick("NVDA")
    window.queue.run_one()
    app.processEvents()
    built_pack = next(v for (sym, _), v in window._pick_packs.items() if sym == "NVDA")
    digest = pick_pack.pack_hash(built_pack)
    failed = assess.Assessment(symbol="NVDA", pack_hash=digest, error="timeout")
    window._on_pick_card({"symbol": "NVDA", "side": "LONG", "pack": built_pack, "hash": digest,
                          "assessment": failed, "source": "prefetch"})
    assert "narrating" in _text(window) and "timeout" not in _text(window), "a failed prefetch is never a card"
    good = assess.Assessment(symbol="NVDA", pack_hash=digest, verdict="wait",
                             bullets=[{"text": "cell", "evidence_refs": ["pick:NVDA:cell"]}])
    window._on_pick_card({"symbol": "NVDA", "side": "LONG", "pack": built_pack, "hash": digest,
                          "assessment": good, "source": "prefetch"})
    text = _text(window)
    assert "narrating" not in text and text.count("Pick NVDA") == 1 and "wait" in text
    assert "pick-assess:NVDA" not in window.queue.pending_keys(), "the waiting live job is dropped"


def test_only_the_24_newest_liked_picks_get_the_high_effort_pass(window, app):
    from mentor_app.prefetch import PRIORITY_IDLE, PRIORITY_REFRESH

    window._brain_ok, window._endpoint, window._model = True, "http://h", "gpt-oss:20b"
    window.liked[:] = [(f"S{i:02d}", "LONG") for i in range(30)]
    window.clock["now"] = datetime(2026, 9, 29, 6, 15, tzinfo=PT)
    window.maybe_prefetch_picks()
    window.queue.run_one()
    jobs = {job.name: job.priority for job in window.queue._jobs}
    assert all(jobs[f"pick_prefetch S{i:02d}"] == PRIORITY_REFRESH for i in range(24))
    assert all(jobs[f"pick_prefetch S{i:02d}"] == PRIORITY_IDLE for i in range(24, 30))
    assert "pick_prefetch AMD" not in jobs, "scope liked never reaches the rest of Focus"


# ---------------------------------------------------------------- P2 follow-ups (P3 branch)
def _liked_world(tmp_path):
    claims = tmp_path / "claimed_picks.jsonl"
    claims.write_text("\n".join(json.dumps(row) for row in (
        {"action": "claim", "symbol": "FORM", "side": "LONG", "claimed_setup_id": "x",
         "claim_at": "2026-09-29T12:10:02-07:00", "session_date": "2026-09-29"},
        {"action": "claim", "symbol": "FADED", "side": "SHORT", "claimed_setup_id": "x",
         "claim_at": "2026-09-01T09:00:00-07:00", "session_date": "2026-09-01"},
    )) + "\n", encoding="utf-8")
    feedback = tmp_path / "pick_feedback.jsonl"
    feedback.write_text("\n".join(json.dumps(row) for row in (
        {"ts": "2026-09-28T07:00:00", "trade_date": "2026-09-28", "symbol": "NVDA", "side": "LONG",
         "verdict": "like", "origin": "d1"},
        {"ts": "2026-09-29T07:30:00", "trade_date": "2026-09-29", "symbol": "BOARD", "side": "LONG",
         "verdict": "like", "origin": "strength_board"},
    )) + "\n", encoding="utf-8")
    favourites = tmp_path / "swing_favorites.jsonl"
    favourites.write_text("", encoding="utf-8")
    return {"claims_path": claims, "feedback_path": feedback, "favorites_path": favourites}


def test_a_strength_board_like_is_out_by_default_and_in_with_its_token(tmp_path):
    from datetime import date

    paths = _liked_world(tmp_path)
    default = pick_jobs.liked_picks(today=date(2026, 9, 29), sources=pick_jobs.DEFAULT_LIKED_SOURCES, **paths)
    assert [sym for sym, _ in default] == ["FORM", "NVDA"], "a Strength Board like is a browsing click"
    opted = pick_jobs.liked_picks(
        today=date(2026, 9, 29), sources=(*pick_jobs.DEFAULT_LIKED_SOURCES, "likes_strength_board"), **paths
    )
    assert [sym for sym, _ in opted] == ["FORM", "BOARD", "NVDA"]


def test_a_faded_claim_is_not_a_liked_pick(tmp_path):
    from datetime import date

    got = pick_jobs.liked_picks(today=date(2026, 9, 29), sources=("claims",), **_liked_world(tmp_path))
    assert got == [("FORM", "LONG")], "FADED was claimed 20 sessions ago: past the fade"


def test_the_liked_sources_setting_is_the_default(tmp_path, monkeypatch):
    from datetime import date

    from mentor_app import settings

    saved = {"mentor_liked_sources": "claims"}
    monkeypatch.setattr(settings, "_setting", lambda key, default=None: saved.get(key, default))
    assert pick_jobs.liked_picks(today=date(2026, 9, 29), **_liked_world(tmp_path)) == [("FORM", "LONG")]
    saved["mentor_liked_sources"] = "nonsense"
    assert settings.liked_sources() == frozenset(pick_jobs.DEFAULT_LIKED_SOURCES), "unknown tokens fall back"


def test_a_token_after_the_turn_ended_changes_nothing(window, app):
    window._blocks[:] = ["Hi. Ask me anything.", "**Mentor:** "]
    window._stream_index = 1
    window._worker = object()
    window.queue.begin_interactive()
    window._on_token("Done")
    window._on_done({"text": "Done.", "pack_texts": [], "model": "m"})
    window.show_pick("NVDA")
    _drain(window, app)
    before = list(window._blocks)
    window._on_token("STALE")
    assert window._blocks == before, "a late token from a finished stream never lands in a card"
    assert "STALE" not in _text(window)
