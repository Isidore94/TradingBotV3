"""Mentor app P11: the frontier switch. Off by default, keyed only through the credential store,
capped per PT day in USD from a dated pricing table (guard BEFORE the call, real usage after),
never automatic, and it reads only what the local model read. No network: every call is faked."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ai_summary  # noqa: E402
from mentor_app import commands, frontier, settings  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402

NOW = datetime(2026, 9, 30, 17, 0, tzinfo=timezone.utc)  # 10:00 PT
DAY = "2026-09-30"


@pytest.fixture
def store(tmp_path):
    return MentorChatStore(tmp_path / "mentor_chat.sqlite3")


def _fake(reply, usage=None, calls=None):
    def request(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return {"summary": json.loads(json.dumps(reply)), "model": kwargs["model"],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200} if usage is None else usage}

    return request


def _metered(store, *, cap=2.0, request=None, sink=None, purpose="chat"):
    return frontier.metered_request(store=store, purpose=purpose, model="claude-sonnet-5", api_key="sk-test",
                                    cap_usd=cap, now=lambda: NOW, request=request, spent_sink=sink)


# ---------------------------------------------------------------- defaults and the dated table
def test_off_by_default_sonnet_and_two_dollars():
    assert settings.frontier_enabled() is False
    assert settings.frontier_model() == "claude-sonnet-5" == settings.DEFAULT_FRONTIER_MODEL
    assert settings.frontier_daily_cap_usd() == 2.00
    assert frontier.PRICING_USD_PER_MTOK["claude-sonnet-5"] == (2.00, 10.00)
    assert frontier.PRICING_AS_OF == "2026-06-24"


def test_the_setting_must_be_exactly_true(monkeypatch):
    monkeypatch.setattr(settings, "_setting", lambda key, default=None: "yes")
    assert settings.frontier_enabled() is False
    monkeypatch.setattr(settings, "_setting", lambda key, default=None: -1 if "cap" in key else True)
    assert settings.frontier_enabled() is True and settings.frontier_daily_cap_usd() == 2.00


def test_cost_and_estimate_come_from_the_table():
    assert frontier.cost_usd("claude-sonnet-5", 1_000_000, 100_000) == pytest.approx(2.00 + 1.00)
    assert frontier.estimate_usd("claude-sonnet-5", 4000, 4000) == pytest.approx((1000 * 2 + 4000 * 10) / 1e6)
    with pytest.raises(frontier.FrontierRefused, match="no price for gpt-9"):
        frontier.cost_usd("gpt-9", 1, 1)


# ---------------------------------------------------------------- the status
@pytest.mark.parametrize("kwargs, reason", [
    ({"enabled": False}, "the frontier switch is off"),
    ({"enabled": True, "key_loader": lambda: ""}, "no Anthropic key saved (Credential Manager, anthropic_api_key)"),
    ({"enabled": True, "model": "claude-x"}, "no price for claude-x"),
])
def test_status_says_why_it_cannot_run(store, kwargs, reason):
    kwargs.setdefault("key_loader", lambda: "sk")
    state = frontier.status(store, NOW, cap=2.0, **kwargs)
    assert not state.usable and reason in state.reason


def test_status_at_the_cap_says_the_cap(store):
    store.add_frontier_usage(day_pt=DAY, purpose="chat", model="claude-sonnet-5", input_tokens=1, output_tokens=1,
                             est_usd=2.0)
    state = frontier.status(store, NOW, enabled=True, cap=2.0, key_loader=lambda: "sk")
    assert state.reason == "frontier cap reached ($2.00 of $2.00)"
    assert "today (PT): $2.0000 of $2.00" in frontier.status_text(state)


def test_an_off_switch_never_reads_the_key(store):
    def boom():
        raise AssertionError("the key was read with the switch off")

    assert frontier.status(store, NOW, enabled=False, key_loader=boom).reason.startswith("the frontier switch is off")


# ---------------------------------------------------------------- the metered call
def test_the_guard_refuses_before_any_call_when_the_estimate_passes_the_cap(store):
    store.add_frontier_usage(day_pt=DAY, purpose="chat", model="claude-sonnet-5", input_tokens=1, output_tokens=1,
                             est_usd=1.99)
    calls = []
    run = _metered(store, request=_fake({}, calls=calls))
    with pytest.raises(frontier.FrontierRefused, match=r"frontier cap reached \(\$1.99 of \$2.00\)"):
        run(provider="local", model="m", api_key="", evidence={"rows": []}, schema={})
    assert calls == [] and len(store.frontier_usage(DAY)) == 1


def test_yesterdays_spend_does_not_count(store):
    store.add_frontier_usage(day_pt="2026-09-29", purpose="chat", model="claude-sonnet-5", input_tokens=1,
                             output_tokens=1, est_usd=5.0)
    run = _metered(store, request=_fake({"answer": []}))
    run(provider="local", model="m", api_key="", evidence={}, schema={})
    assert store.frontier_spent(DAY) == pytest.approx(frontier.cost_usd("claude-sonnet-5", 1000, 200))


def test_the_real_usage_is_priced_and_written(store):
    calls, sink = [], []
    run = _metered(store, request=_fake({"answer": []}, calls=calls), sink=sink, purpose="pick")
    run(provider="local", model="gemma3:12b", api_key="", evidence={"a": 1}, schema={"type": "object"},
        endpoint="http://x/v1")
    sent = calls[0]
    assert sent["provider"] == "anthropic" and sent["model"] == "claude-sonnet-5" and sent["api_key"] == "sk-test"
    assert sent["evidence"] == {"a": 1} and sent["schema"] == {"type": "object"}, "the caller's package, unchanged"
    row = store.frontier_usage(DAY)[0]
    assert (row["purpose"], row["model"], row["input_tokens"], row["output_tokens"], row["measured"]) == (
        "pick", "claude-sonnet-5", 1000, 200, 1)
    assert row["est_usd"] == pytest.approx((1000 * 2 + 200 * 10) / 1e6) and sink[0]["measured"] is True
    assert datetime.fromisoformat(row["ts_utc"]).tzinfo is not None


def test_missing_usage_or_a_failed_call_counts_the_estimate(store):
    run = _metered(store, request=_fake({"answer": []}, usage={}))
    run(provider="local", model="m", api_key="", evidence={"x": "y" * 400}, schema={})

    def down(**_):
        raise RuntimeError("anthropic request failed (529)")

    with pytest.raises(RuntimeError):
        _metered(store, request=down)(provider="local", model="m", api_key="", evidence={}, schema={})
    rows = store.frontier_usage(DAY)
    assert [row["measured"] for row in rows] == [0, 0] and rows[1]["purpose"] == "chat:failed"
    assert all(row["est_usd"] > 0 for row in rows)


def test_an_unreadable_spend_refuses(store, monkeypatch):
    monkeypatch.setattr(store, "frontier_spent", lambda day: None)
    with pytest.raises(frontier.FrontierRefused, match="cannot be read"):
        _metered(store, request=_fake({}))(provider="local", model="m", api_key="", evidence={}, schema={})


def test_frontier_post_caps_output_and_sets_effort_only_where_supported():
    sent = []
    frontier.frontier_post(lambda url, **kw: sent.append(kw["json"]), model="claude-sonnet-5")(
        "u", json={"max_tokens": 600, "reasoning_effort": "low", "output_config": {"format": {"type": "json_schema"}}})
    frontier.frontier_post(lambda url, **kw: sent.append(kw["json"]), model="claude-haiku-4-5")("u", json={})
    assert sent[0] == {"max_tokens": frontier.MAX_OUTPUT_TOKENS,
                       "output_config": {"format": {"type": "json_schema"}, "effort": "medium"}}
    assert sent[1] == {"max_tokens": frontier.MAX_OUTPUT_TOKENS}, "Haiku 4.5 rejects effort"


def test_through_ai_summary_the_anthropic_payload_is_capped_and_priced(store):
    """The real request_ai_summary with a fake post: no network, the payload the API would get."""
    posted = []

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"id": "msg_1", "content": [{"type": "text", "text": json.dumps({"answer": []})}],
                    "usage": {"input_tokens": 1500, "output_tokens": 300}}

    def post(url, **kwargs):
        posted.append((url, kwargs))
        return Response()

    result = frontier.think_chat("Is NVDA worth it?", context_text="[ctx:mode] Auto: Balanced",
                                 memory_block="", pack_texts=[], model="claude-sonnet-5",
                                 request=_metered(store), post=frontier.frontier_post(post, model="claude-sonnet-5"))
    url, kwargs = posted[0]
    assert url == ai_summary.ANTHROPIC_MESSAGES_URL and kwargs["headers"]["x-api-key"] == "sk-test"
    assert kwargs["json"]["max_tokens"] == frontier.MAX_OUTPUT_TOKENS and kwargs["json"]["model"] == "claude-sonnet-5"
    assert result.error == "no cited point survived the check"
    assert store.frontier_spent(DAY) == pytest.approx((1500 * 2 + 300 * 10) / 1e6)


# ---------------------------------------------------------------- the same evidence, the same check
CONTEXT = "## context_pack\n[ctx:auto_mode] Auto mode: Balanced\n[ctx:d1_env] D1: weak tape"
MEMORY = "# Memory\nNight digests and the trader's own notes, oldest first. Cite by id.\n[mem:note:3] (2026-09-28) rule: two losses"
PACK = "## pick_pack\n[pick:NVDA:cell] Cell n=40, LB 0.45"


def test_think_chat_sends_the_local_texts_byte_identical_and_checks_citations():
    calls = []
    reply = {"answer": [{"text": "Auto is Balanced.", "evidence_refs": ["ctx:auto_mode"]},
                        {"text": "Cell n=40.", "evidence_refs": ["pick:NVDA:cell", "mem:note:3"]},
                        {"text": "A hunch.", "evidence_refs": []}]}
    out = frontier.think_chat("Is NVDA worth it?", context_text=CONTEXT, memory_block=MEMORY, pack_texts=[PACK],
                              model="claude-sonnet-5", request=_fake(reply, calls=calls), post=lambda *a, **k: None)
    evidence = calls[0]["evidence"]
    assert evidence["desk_context"] == CONTEXT and evidence["memory"] == MEMORY and evidence["packs"] == [PACK]
    assert calls[0]["system_instruction"].startswith(ai_summary._system_instruction())
    assert calls[0]["schema"] is frontier.CHAT_SCHEMA
    assert [p["text"] for p in out.points] == ["Auto is Balanced.", "Cell n=40."] and len(out.dropped) == 1
    card = frontier.card_markdown(out, spend="cost $0.0040; today $0.0040 of $2.00")
    assert card.startswith("**frontier: claude-sonnet-5**") and "[pick:NVDA:cell] [mem:note:3]" in card


def test_a_foreign_id_rejects_the_frontier_answer():
    reply = {"answer": [{"text": "TSLA is weak.", "evidence_refs": ["pick:TSLA:cell"]}]}
    out = frontier.think_chat("q", context_text=CONTEXT, memory_block="", pack_texts=[PACK], model="claude-sonnet-5",
                              request=_fake(reply), post=lambda *a, **k: None)
    assert out.points == [] and out.error.startswith("reply rejected:") and "TSLA" not in frontier.card_markdown(
        out).split("reply rejected")[0]


def test_think_week_reads_four_weeks_of_digests_the_mirror_and_open_hypotheses(tmp_path):
    from mentor_packs import hypothesis_pack, mirror_pack

    digests = [
        {"session_date": "2026-09-29", "digest": [{"text": "Liked three shorts."}], "open_questions": []},
        {"session_date": "2026-08-20", "digest": [{"text": "Too old."}], "open_questions": []},
    ]
    world = hypothesis_pack.write_fixture_world(tmp_path)
    hyps = hypothesis_pack.build(chat_db=world["chat"], history_dir=world["history"], report_file=world["report_file"])
    mirror = mirror_pack.fixture()
    rows = frontier.week_rows(digests=digests, mirror=mirror, hypotheses=hyps, now=NOW)
    ids = [row["source_id"] for row in rows]
    assert "mem:digest:2026092900" in ids and not any(i.startswith("mem:digest:20260820") for i in ids)
    assert set(mirror.ids) <= set(ids)
    assert "hyp:2026-09-28:1:cell" in ids and "hyp:2026-09-21:1" not in ids and "hyp:2026-09-21:1:cell" not in ids
    calls = []
    reply = {"observations": [{"text": "Shorts liked.", "evidence_refs": ["mem:digest:2026092900"]}],
             "questions": [{"text": "Still SMA100?", "evidence_refs": ["hyp:2026-09-28:1"]}]}
    out = frontier.think_week(rows, model="claude-sonnet-5", request=_fake(reply, calls=calls),
                              post=lambda *a, **k: None)
    assert calls[0]["schema"] is frontier.WEEK_SCHEMA and calls[0]["system_instruction"] is None
    assert len(out.points) == 1 and len(out.questions) == 1 and not out.error


# ---------------------------------------------------------------- the key, the commands, never automatic
def test_the_key_lives_only_in_the_credential_store(monkeypatch):
    import ai_credentials
    import secret_store

    backend = ai_credentials.MemoryCredentialBackend()
    monkeypatch.setattr(ai_credentials, "default_backend", lambda: backend)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    assert secret_store.load_secret(secret_store.ANTHROPIC_API_KEY_NAME) == "", "no env var fallback"
    assert secret_store.save_secret(secret_store.ANTHROPIC_API_KEY_NAME, " sk-ant-1 ")
    assert frontier.load_key() == "sk-ant-1"
    assert secret_store.save_secret(secret_store.ANTHROPIC_API_KEY_NAME, "")
    assert frontier.load_key() == ""
    assert frontier.SECRET_NAME == secret_store.ANTHROPIC_API_KEY_NAME


def test_the_think_and_frontier_commands_parse():
    assert commands.handle("/think").arg == ("chat",)
    assert commands.handle("/think week").arg == ("week",)
    assert commands.handle("/think pick nvda").arg == ("pick", "NVDA", "")
    assert commands.handle("/think pick NVDA short").arg == ("pick", "NVDA", "SHORT")
    for bad in ("/think pick", "/think month", "/think pick NVDA up"):
        assert commands.handle(bad).action == "error"
    assert commands.handle("/frontier").action == "frontier"
    assert "/think" in commands.HELP_TEXT and "/frontier" in commands.HELP_TEXT


def test_nothing_automatic_imports_the_frontier():
    """No night job, prefetch, Inbox or pack reaches the frontier; only the window's commands do."""
    users = []
    for path in list((SCRIPTS_DIR / "ai_jobs").glob("*.py")) + list((SCRIPTS_DIR / "mentor_packs").glob("*.py")) + \
            list((SCRIPTS_DIR / "mentor_app").glob("*.py")):
        if path.name == "frontier.py":
            continue
        if "mentor_app import frontier" in path.read_text(encoding="utf-8") or \
                "mentor_app.frontier" in path.read_text(encoding="utf-8"):
            users.append(path.name)
    assert users == ["window.py"]
    source = (SCRIPTS_DIR / "mentor_app" / "frontier.py").read_text(encoding="utf-8")
    assert "PySide6" not in source and "place_order" not in source and "os.environ" not in source


# ---------------------------------------------------------------- the window
@pytest.fixture
def win(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow
    from mentor_packs import mirror_pack, pick_pack

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    world = pick_pack.write_fixture_world(tmp_path / "picks")
    calls: list = []
    posted: list = []
    reply = {"answer": [{"text": "Balanced tape.", "evidence_refs": ["ctx:auto_mode"]}],
             "verdict": "wait", "bullets": [{"text": "Cell n=40.", "evidence_refs": ["pick:NVDA:cell"]}],
             "rule_flags": [], "observations": [], "questions": []}

    def request(**kwargs):
        calls.append(kwargs)
        kwargs["post"]("u", json={"max_tokens": 3500})
        return {"summary": {k: reply[k] for k in kwargs["schema"]["required"]}, "model": kwargs["model"],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 300}}

    window = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(), news_queue=PrefetchQueue(),
        stream_post=lambda *a, **k: [], post=lambda *a, **k: {}, now=lambda: NOW, mentor_enabled=False,
        pick_builder=lambda sym, side: pick_pack.build(sym, side, now=pick_pack.FIXTURE_NOW, paths=world),
        mirror_builder=lambda weeks: mirror_pack.fixture(), memory_root=tmp_path / "ai",
        frontier_request=request, frontier_post=lambda url, **kw: posted.append(kw["json"]),
        frontier_key=lambda: "sk-test",
    )
    window.calls, window.posted = calls, posted
    yield window
    window.shutdown()
    window.deleteLater()


def _settle(window):
    from PySide6.QtWidgets import QApplication

    for thread in list(window._threads):
        thread.join(5)
    QApplication.processEvents()
    window._io.submit(lambda: None).result(5)
    QApplication.processEvents()


def _switch(monkeypatch, on: bool):
    monkeypatch.setattr(settings, "frontier_enabled", lambda: on)


def test_with_the_switch_off_nothing_is_called_and_the_button_is_hidden(win, monkeypatch):
    _switch(monkeypatch, False)
    win._sync_status()
    assert not win.think_button.isVisibleTo(win)
    win._last_turn = {"question": "q", "context_text": CONTEXT, "memory_block": "", "pack_texts": []}
    win.send("/think")
    _settle(win)
    assert win.calls == [] and "not called (the frontier switch is off" in win._blocks[-1]
    win.send("/frontier")
    _settle(win)
    assert "switch: off" in win.transcript.toPlainText()


def test_think_needs_a_last_question(win, monkeypatch):
    _switch(monkeypatch, True)
    win.send("/think")
    assert "Ask a question first" in win.transcript.toPlainText() and win.calls == []


def test_think_re_asks_the_last_turn_labelled_and_logged(win, monkeypatch):
    _switch(monkeypatch, True)
    win._sync_status()
    assert win.think_button.isVisibleTo(win)
    win._last_turn = {"question": "Is it a Balanced day?", "context_text": CONTEXT, "memory_block": MEMORY,
                      "pack_texts": [PACK]}
    win.think_button.click()
    _settle(win)
    assert len(win.calls) == 1 and win.calls[0]["evidence"]["desk_context"] == CONTEXT
    assert win.calls[0]["provider"] == "anthropic" and win.posted[0]["max_tokens"] == frontier.MAX_OUTPUT_TOKENS
    block = win._blocks[-1]
    assert block.startswith("**frontier: claude-sonnet-5**") and "Balanced tape. [ctx:auto_mode]" in block
    cost = (2000 * 2 + 300 * 10) / 1e6
    assert f"cost ${cost:.4f}; today ${cost:.4f} of $2.00" in block
    stored = [row for row in win.store.turns() if row["role"] == "assistant"][-1]
    assert stored["model"] == "claude-sonnet-5" and json.loads(stored["tool_calls_json"])[0]["name"] == "frontier"
    win.send("/frontier")
    _settle(win)
    assert f"today (PT): ${cost:.4f} of $2.00" in win.transcript.toPlainText()


def test_think_pick_sends_the_same_package_the_local_assessment_sees(win, monkeypatch):
    from mentor_app import assess

    _switch(monkeypatch, True)
    win.send("/think pick NVDA long")
    _settle(win)
    assert len(win.calls) == 1
    pack = win._build_pick("NVDA", "LONG")
    assert win.calls[0]["evidence"] == assess.evidence_for(pack) and win.calls[0]["schema"] is assess.SCHEMA
    block = win._blocks[-1]
    assert "**frontier: claude-sonnet-5** · Think harder: pick NVDA" in block and "[pick:NVDA:cell]" in block
    assert win.store.frontier_usage(DAY)[0]["purpose"] == "pick"


def test_think_week_reads_digests_mirror_and_hypotheses(win, monkeypatch):
    _switch(monkeypatch, True)
    win.send("/think week")
    _settle(win)
    rows = win.calls[0]["evidence"]["rows"]
    assert any(row["source_id"] == "hyp:report:asof" for row in rows)
    assert "Think: the last four weeks" in win._blocks[-1]


def test_at_the_cap_the_call_is_refused_with_the_amount(win, monkeypatch):
    _switch(monkeypatch, True)
    win.store.add_frontier_usage(day_pt=DAY, purpose="chat", model="claude-sonnet-5", input_tokens=1,
                                 output_tokens=1, est_usd=2.0)
    win.send("/think week")
    _settle(win)
    assert win.calls == [] and "frontier cap reached ($2.00 of $2.00)" in win._blocks[-1]
    assert not win.think_button.isEnabled() and "frontier cap reached" in win.think_button.toolTip()


def test_a_missing_key_disables_it_with_the_reason(win, monkeypatch):
    _switch(monkeypatch, True)
    win._frontier_key = lambda: ""
    win.send("/think week")
    _settle(win)
    assert win.calls == [] and "no Anthropic key saved" in win._blocks[-1]
