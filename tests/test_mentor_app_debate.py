"""Mentor app P10: /debate SYM. Two persona calls on ONE pick pack (bull, bear), each citation-checked;
a rejected side is shown as rejected, never replaced; the app computes the scoreboard line; a clean
pair is cached by (symbol, side, pack hash, prompt version); brain off = no debate + the pack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ai_summary  # noqa: E402
from mentor_app import commands, debate  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import pick_pack  # noqa: E402

BULL = {
    "case": [
        {"text": "Setup cell wins 60% (n=40).", "evidence_refs": ["pick:NVDA:cell"]},
        {"text": "The trader claimed it this morning.", "evidence_refs": ["pick:NVDA:claim:LONG:avwap_breakout"]},
        {"text": "It just feels strong.", "evidence_refs": []},
    ],
    "weakest_point": {"text": "A peer reports tomorrow.", "evidence_refs": ["pick:NVDA:peer:AVGO"]},
}
BEAR = {
    "case": [
        {"text": "A peer reports tomorrow.", "evidence_refs": ["pick:NVDA:peer:AVGO"]},
        {"text": "The same cell loses 40% of the time.", "evidence_refs": ["pick:NVDA:cell"]},
    ],
    "weakest_point": {"text": "No own earnings soon.", "evidence_refs": []},
}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return pick_pack.write_fixture_world(tmp_path_factory.mktemp("debate"))


@pytest.fixture
def pack(world):
    return pick_pack.build("NVDA", "LONG", now=pick_pack.FIXTURE_NOW, paths=world)


@pytest.fixture
def store(tmp_path):
    return MentorChatStore(tmp_path / "mentor_chat.sqlite3")


def _requester(bull=BULL, bear=BEAR, calls=None):
    replies = {"BULL": bull, "BEAR": bear}

    def request(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        role = "BULL" if "the BULL" in kwargs["system_instruction"] else "BEAR"
        reply = replies[role]
        if isinstance(reply, Exception):
            raise reply
        return {"summary": json.loads(json.dumps(reply)), "model": "m"}

    return request


def _run(pack, **kwargs):
    return debate.debate(pack, symbol="NVDA", side="LONG", pack_hash="h", model="gpt-oss:20b",
                         endpoint="http://x/", **kwargs)


# ---------------------------------------------------------------- the two calls
def test_two_persona_calls_on_the_same_pack_capped_at_400_tokens_medium_effort(pack):
    calls, posted = [], []

    def request(**kwargs):
        calls.append(kwargs)
        kwargs["post"]("u", json={"max_tokens": 4000})
        return _requester()(**kwargs)

    _run(pack, request=request, post=lambda url, **kw: posted.append(kw["json"]))
    assert len(calls) == 2
    base = ai_summary._system_instruction()
    assert [c["system_instruction"] for c in calls] == [
        ai_summary.persona_instruction(debate.ROLES["bull"]), ai_summary.persona_instruction(debate.ROLES["bear"])]
    assert all(c["system_instruction"].startswith(base) for c in calls), "personas wrap the citation rules"
    assert calls[0]["evidence"]["rows"] == calls[1]["evidence"]["rows"], "the same pack for both sides"
    assert all(c["schema"] is debate.SCHEMA and c["endpoint"] == "http://x/v1" for c in calls)
    assert set(debate.SCHEMA["properties"]) == {"case", "weakest_point"}
    assert [p["max_tokens"] for p in posted] == [400, 400]
    assert [p["reasoning_effort"] for p in posted] == ["medium", "medium"]


def test_uncited_points_are_dropped_and_the_scoreboard_is_computed(pack):
    result = _run(pack, request=_requester())
    assert result.clean
    assert len(result.bull.case) == 2 and result.bull.weakest_point["evidence_refs"] == ["pick:NVDA:peer:AVGO"]
    assert result.bear.weakest_point is None and [d["kind"] for d in result.bear.dropped] == ["weakest_point"]
    line = debate.scoreboard_line(result, pack)
    import setup_grades

    lb = setup_grades.wilson_lower_bound(24, 40)
    assert line == (f"**Scoreboard:** bull 2 cited, bear 2 cited; cited by both: [pick:NVDA:cell] [pick:NVDA:peer:AVGO]; "
                    f"cell [pick:NVDA:cell]: n=40, LB {lb:.2f}")
    text = debate.card_markdown(result, pack, symbol="NVDA", side="LONG", pack_hash="h")
    assert "| Bull | Bear |" in text and "*Weakest point:* A peer reports tomorrow. [pick:NVDA:peer:AVGO]" in text
    assert "*Weakest point: none cited*" in text and "It just feels strong" not in text
    assert "2 uncited points dropped" in text
    assert debate.FOOTER in text and "you decide" in debate.FOOTER


def test_a_foreign_id_rejects_that_side_only_and_it_is_never_replaced(pack):
    bad_bear = {"case": [{"text": "TSLA is weak.", "evidence_refs": ["pick:TSLA:cell"]}],
                "weakest_point": {"text": "x", "evidence_refs": ["pick:NVDA:cell"]}}
    result = _run(pack, request=_requester(bear=bad_bear))
    assert result.bull.argued and result.bear.rejected and result.bear.case == [] and not result.clean
    text = debate.card_markdown(result, pack, symbol="NVDA", pack_hash="h")
    assert "bear case rejected:" in text and "pick:TSLA:cell" in text
    assert "TSLA is weak" not in text, "a rejected side shows why, never its bullets"
    assert "Setup cell wins 60%" in text
    assert "bull 2 cited, bear rejected; cited by both: none" in text


def test_a_failed_call_is_shown_as_not_argued(pack):
    result = _run(pack, request=_requester(bull=RuntimeError("timeout")))
    assert not result.bull.argued and result.bear.argued
    text = debate.card_markdown(result, pack, symbol="NVDA", pack_hash="h")
    assert "bull case not argued: RuntimeError: timeout" in text and "A peer reports tomorrow." in text


def test_stop_between_the_calls_leaves_the_bear_not_run(pack):
    calls, flag = [], {"stop": False}

    def request(**kwargs):
        calls.append(kwargs)
        flag["stop"] = True
        return _requester()(**kwargs)

    result = _run(pack, request=request, cancelled=lambda: flag["stop"])
    assert len(calls) == 1 and result.bull.argued and result.bear.error == "not run: stopped"


def test_both_sides_down_is_no_debate_plus_the_pack(pack):
    result = _run(pack, request=_requester(bull=RuntimeError("a"), bear=RuntimeError("b")))
    text = debate.card_markdown(result, pack, symbol="NVDA", pack_hash="h")
    assert "no debate (" in text and "[pick:NVDA:cell]" in text and "| Bull |" not in text


def test_brain_off_is_no_debate_and_the_pack(pack):
    text = debate.card_markdown(None, pack, symbol="NVDA", side="LONG", pack_hash="h", brain_reason="")
    assert "**Debate NVDA LONG**: no debate (brain off)" in text
    assert all(f"[{row_id}]" in text for row_id in pack.ids)


# ---------------------------------------------------------------- cache
def test_a_clean_pair_is_cached_by_symbol_side_hash_and_prompt(store, pack):
    ran = []

    def run(p, digest):
        ran.append(digest)
        return debate.debate(p, symbol="NVDA", side="LONG", pack_hash=digest, model="m", endpoint="http://x",
                             request=_requester())

    first = debate.run_debate_job("NVDA", "LONG", store=store, build_pack=lambda s, d: pack,
                                  pack_hash=pick_pack.pack_hash, run=run)
    again = debate.run_debate_job("NVDA", "LONG", store=store, build_pack=lambda s, d: pack,
                                  pack_hash=pick_pack.pack_hash, run=run)
    assert len(ran) == 1 and first["ran"] and not again["ran"]
    assert again["debate"].bull.case == first["debate"].bull.case
    key = debate.cache_key("NVDA", "LONG", first["hash"])
    assert key == {"symbol": "NVDA", "side": "LONG", "hash": first["hash"], "prompt": debate.PROMPT_VERSION}
    assert store.get_pack(debate.CACHE_NAME, key) is not None
    assert debate.cached_debate(store, "NVDA", "SHORT", first["hash"]) is None


def test_a_rejected_pair_is_not_cached_and_brain_off_never_calls(store, pack):
    bad = {"case": [{"text": "x", "evidence_refs": ["nope:1"]}], "weakest_point": {"text": "y", "evidence_refs": []}}
    out = debate.run_debate_job(
        "NVDA", "LONG", store=store, build_pack=lambda s, d: pack, pack_hash=pick_pack.pack_hash,
        run=lambda p, d: debate.debate(p, symbol="NVDA", side="LONG", pack_hash=d, model="m", endpoint="http://x",
                                       request=_requester(bear=bad)))
    assert out["ran"] and debate.cached_debate(store, "NVDA", "LONG", out["hash"]) is None
    off = debate.run_debate_job("NVDA", "LONG", store=store, build_pack=lambda s, d: pack,
                                pack_hash=pick_pack.pack_hash, run=None)
    assert off["debate"] is None and not off["ran"] and off["pack"] is pack


def test_the_turn_log_carries_both_replies_and_drops(pack):
    result = _run(pack, request=_requester())
    calls = debate.turn_tool_calls({"symbol": "NVDA", "side": "LONG", "hash": "h", "debate": result})
    assert [c["name"] for c in calls] == ["pick_pack", "debate_bull", "debate_bear"]
    assert calls[1]["reply"]["case"][2]["text"] == "It just feels strong."
    assert calls[1]["dropped"][0]["reason"] == "uncited" and calls[2]["dropped"][0]["kind"] == "weakest_point"


def test_debate_is_qt_free_and_never_orders():
    source = (SCRIPTS_DIR / "mentor_app" / "debate.py").read_text(encoding="utf-8")
    assert "PySide6" not in source and "place_order" not in source


def test_the_debate_command_parses():
    assert commands.handle("/debate nvda").action == "debate"
    assert commands.handle("/debate nvda").arg == ("NVDA", "")
    assert commands.handle("/debate NVDA short").arg == ("NVDA", "SHORT")
    assert commands.handle("/debate").action == "error"
    assert commands.handle("/debate NVDA sideways").action == "error"
    assert "/debate" in commands.HELP_TEXT


def test_an_unsided_debate_reuses_the_pair_cached_for_the_side_the_pack_chose(store, pack):
    ran = []

    def run(p, digest):
        ran.append(digest)
        return debate.debate(p, symbol="NVDA", side="LONG", pack_hash=digest, model="m", endpoint="http://x",
                             request=_requester())

    debate.run_debate_job("NVDA", "LONG", store=store, build_pack=lambda s, d: pack,
                          pack_hash=pick_pack.pack_hash, run=run)
    again = debate.run_debate_job("NVDA", "", store=store, build_pack=lambda s, d: pack,
                                  pack_hash=pick_pack.pack_hash, run=run)
    assert len(ran) == 1 and again["side"] == "LONG" and again["debate"].clean


# ---------------------------------------------------------------- the window
@pytest.fixture
def win(tmp_path, monkeypatch, world):
    from PySide6.QtWidgets import QApplication

    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    calls: list = []
    hooks: dict = {}

    def request(**kwargs):
        calls.append(kwargs)
        if hooks.get("during"):
            hooks["during"]()
        return _requester()(**kwargs)

    window = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(), news_queue=PrefetchQueue(),
        stream_post=lambda *a, **k: [], post=lambda *a, **k: {}, now=lambda: pick_pack.FIXTURE_NOW,
        mentor_enabled=False,
        pick_builder=lambda sym, side: pick_pack.build(sym, side, now=pick_pack.FIXTURE_NOW, paths=world),
        debate_request=request,
    )
    window.calls, window.hooks = calls, hooks
    yield window
    window.shutdown()
    window.deleteLater()


def _drain(window):
    while window.queue.run_one():
        pass
    window._io.submit(lambda: None).result(5)
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()


def _up(window):
    window._brain_ok, window._endpoint, window._model = True, "http://127.0.0.1:11436", "gpt-oss:20b"


def test_debate_with_the_brain_off_is_the_pack_and_no_model_call(win):
    win._brain_reason = "the night AI owns the GPU"
    win.send("/debate NVDA")
    _drain(win)
    text = win.transcript.toPlainText()
    assert "Debate NVDA LONG: no debate (brain off: the night AI owns the GPU)" in text
    assert "pick:NVDA:cell" in text and win.calls == []


def test_debate_shows_two_columns_the_scoreboard_and_logs_both_replies(win):
    _up(win)
    win.send("/debate NVDA long")
    assert win.stop_button.isEnabled(), "Stop can cancel a debate"
    _drain(win)
    assert len(win.calls) == 2 and not win.stop_button.isEnabled()
    block = win._blocks[-1]
    assert "| Bull | Bear |" in block and "**Scoreboard:** bull 2 cited, bear 2 cited" in block
    assert block.rstrip().endswith(f"(evidence:NVDA:{pick_pack.pack_hash(win._build_pick('NVDA', 'LONG'))})")
    assert debate.FOOTER in block and win.chat.turns[-1].text == block
    stored = [row for row in win.store.turns() if row["role"] == "assistant"][-1]
    calls = json.loads(stored["tool_calls_json"])
    assert [c["name"] for c in calls] == ["pick_pack", "debate_bull", "debate_bear"]
    assert calls[1]["dropped"] and calls[1]["reply"]["case"]
    win.send("/debate NVDA long")
    _drain(win)
    assert len(win.calls) == 2, "an unchanged pack reuses the cached pair"
    assert "| Bull | Bear |" in win._blocks[-1]


def test_stop_before_the_debate_starts_drops_it(win):
    _up(win)
    win.send("/debate NVDA")
    win.stop_turn()
    _drain(win)
    assert win.calls == [] and "stopped before it started" in win._blocks[-1]
    assert not win.stop_button.isEnabled()


def test_stop_between_the_two_calls_skips_the_bear(win):
    _up(win)
    win.hooks["during"] = win.stop_turn
    win.send("/debate NVDA")
    _drain(win)
    assert len(win.calls) == 1
    block = win._blocks[-1]
    assert "bear case not argued: not run: stopped" in block and "Setup cell wins 60%" in block
    assert debate.cached_debate(win.store, "NVDA", "LONG", pick_pack.pack_hash(win._build_pick("NVDA", "LONG"))) is None
