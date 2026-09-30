"""Trade Mentor brain: NDJSON stream parse, tool loop capped at 4, gemma one-shot, payload shape."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import brain  # noqa: E402
from mentor_app.chat_model import SYSTEM_PROMPT, ChatModel  # noqa: E402
from mentor_packs.registry import make_pack  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "context_pack", "description": "desk", "parameters": {"type": "object"}}}
]


def _lines(*chunks):
    return [json.dumps(chunk).encode() for chunk in chunks]


def _answer(*pieces, prompt=100, completion=20):
    chunks = [{"message": {"role": "assistant", "content": piece}, "done": False} for piece in pieces]
    chunks.append({"message": {"role": "assistant", "content": ""}, "done": True,
                   "prompt_eval_count": prompt, "eval_count": completion})
    return _lines(*chunks)


def _fake_pack(name, args):
    return make_pack(name, [{"id": f"ctx:{name}:{len(args)}", "text": "DESK"}])


def test_ndjson_parse_skips_blanks_and_garbage():
    rows = list(brain.iter_ndjson([b"", b'{"a": 1}', b"not json", "data: {\"b\": 2}", b"  "]))
    assert rows == [{"a": 1}, {"b": 2}]


def test_a_plain_answer_streams_tokens_and_counts():
    seen: list[str] = []
    payloads: list[dict] = []

    def stream_post(url, payload, cancelled):
        payloads.append(payload)
        assert url == "http://127.0.0.1:11436/api/chat"
        return _answer("Auto is ", "DESK [ctx:auto_mode].")

    ticks = iter([0.0, 0.4, 1.0])
    result = brain.run_turn(
        [{"role": "user", "content": "mode?"}], model="gpt-oss:20b", endpoint="http://127.0.0.1:11436",
        keep_alive=-1, num_ctx=65536, tools=TOOLS, stream_post=stream_post, on_token=seen.append,
        clock=lambda: next(ticks),
    )
    assert seen == ["Auto is ", "DESK [ctx:auto_mode]."]
    assert result["text"] == "Auto is DESK [ctx:auto_mode]."
    assert result["first_token_ms"] == 400 and result["total_ms"] == 1000
    assert (result["prompt_tokens"], result["completion_tokens"]) == (100, 20)
    payload = payloads[0]
    assert payload["stream"] is True and payload["keep_alive"] == -1
    assert payload["options"]["num_ctx"] == 65536 and payload["tools"] == TOOLS
    assert payload["reasoning_effort"] == "low", "gpt-oss chat turns reason low"


def test_the_tool_loop_stops_at_four_calls_and_then_answers_without_tools():
    requests_seen: list[dict] = []
    built: list[str] = []

    def stream_post(url, payload, cancelled):
        requests_seen.append(payload)
        if payload.get("tools"):
            return _lines({"message": {"role": "assistant", "content": "",
                                       "tool_calls": [{"function": {"name": "context_pack", "arguments": {}}}]},
                           "done": True, "prompt_eval_count": 10, "eval_count": 2})
        return _answer("Done.")

    def build(name, args):
        built.append(name)
        return _fake_pack(name, args)

    calls: list[dict] = []
    result = brain.run_turn(
        [{"role": "user", "content": "go"}], model="gpt-oss:20b", endpoint="http://x", tools=TOOLS,
        stream_post=stream_post, build_pack=build, on_tool_call=calls.append,
    )
    assert len(built) == brain.MAX_TOOL_CALLS == 4
    assert len(calls) == 4 and len(result["tool_calls"]) == 4
    assert "tools" not in requests_seen[-1], "past the cap the model must answer without tools"
    assert result["text"] == "Done."
    tool_messages = [m for m in requests_seen[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 4 and tool_messages[0]["tool_name"] == "context_pack"
    assert result["pack_ids"] == ["ctx:context_pack:0"]


def test_many_calls_in_one_reply_are_still_capped():
    def stream_post(url, payload, cancelled):
        if payload.get("tools"):
            call = {"function": {"name": "context_pack", "arguments": "{}"}}
            return _lines({"message": {"content": "", "tool_calls": [call] * 7}, "done": True})
        return _answer("ok")

    built: list[str] = []
    brain.run_turn([{"role": "user", "content": "go"}], model="qwen3:14b", endpoint="http://x", tools=TOOLS,
                   stream_post=stream_post, build_pack=lambda n, a: built.append(n) or _fake_pack(n, a))
    assert len(built) == 4


def test_gemma_gets_a_one_shot_pack_choice_then_a_streamed_answer():
    posted: list[dict] = []
    streamed: list[dict] = []

    def post(url, payload, timeout):
        posted.append(payload)
        assert payload["stream"] is False and payload["format"]["required"] == ["packs"]
        return {"message": {"content": json.dumps({"packs": [{"name": "context_pack", "args": {}}]})}}

    def stream_post(url, payload, cancelled):
        streamed.append(payload)
        return _answer("Auto is DESK [ctx:context_pack:0].")

    result = brain.run_turn(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "mode?"}],
        model="gemma3:12b-tbv3ctx-64k", endpoint="http://x", tools=TOOLS, post=post, stream_post=stream_post,
        build_pack=_fake_pack,
    )
    assert len(posted) == 1 and len(streamed) == 1
    assert "tools" not in streamed[0]
    roles = [m["role"] for m in streamed[0]["messages"]]
    assert roles == ["system", "system", "user"], "packs go before the newest user turn"
    assert "[ctx:context_pack:0]" in streamed[0]["messages"][1]["content"]
    assert result["tool_calls"] == [{"name": "context_pack", "arguments": {}}]
    assert "reasoning_effort" not in streamed[0]


def test_cancel_stops_at_the_next_chunk():
    flag = {"stop": False}
    seen: list[str] = []

    def on_token(text):
        seen.append(text)
        flag["stop"] = True

    result = brain.run_turn(
        [{"role": "user", "content": "x"}], model="qwen3", endpoint="http://x",
        stream_post=lambda u, p, c: _answer("one", "two", "three"), on_token=on_token,
        cancelled=lambda: flag["stop"],
    )
    assert seen == ["one"] and result["cancelled"] is True


def test_an_ollama_error_line_raises():
    import pytest

    with pytest.raises(brain.BrainError):
        brain.run_turn([{"role": "user", "content": "x"}], model="m", endpoint="http://x",
                       stream_post=lambda u, p, c: _lines({"error": "model not found"}))


def test_unload_hands_the_gpu_back_with_keep_alive_zero():
    sent: list[dict] = []
    brain.unload("http://127.0.0.1:11436", "gpt-oss:20b", post=lambda u, p, t: sent.append((u, p)) or {})
    url, payload = sent[0]
    assert url.endswith("/api/chat") and payload["keep_alive"] == 0


def test_embed_reads_the_native_embeddings_field():
    vectors = brain.embed("http://x", ["a"], model="nomic-embed-text",
                          post=lambda u, p, t: {"embeddings": [[1, 2]]})
    assert vectors == [[1.0, 2.0]]


def test_the_system_prefix_is_byte_stable_across_turns():
    chat = ChatModel()
    chat.add("user", "first")
    one = chat.messages(context_text="[ctx:auto_mode] Auto mode: DESK")
    chat.add("assistant", "reply")
    chat.add("user", "second")
    two = chat.messages(context_text="[ctx:auto_mode] Auto mode: DESK", memory_text="NVDA note")
    assert one[0] == two[0] and one[0]["content"].startswith(SYSTEM_PROMPT)
    assert two[-1] == {"role": "user", "content": "second"}
    assert two[-2]["role"] == "system" and "NVDA note" in two[-2]["content"]


def test_the_budget_drops_the_oldest_turns_but_keeps_the_newest():
    chat = ChatModel()
    for index in range(50):
        chat.add("user" if index % 2 == 0 else "assistant", f"turn {index} " + "x" * 400)
    messages = chat.messages(budget_tokens=3000)
    kept = [m["content"] for m in messages[1:]]
    assert kept[-1].startswith("turn 49")
    assert not any(text.startswith("turn 0 ") for text in kept)
    assert 1 < len(kept) < 50


def test_the_stream_worker_emits_tokens_and_done_off_the_qt_thread():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    worker = brain.StreamWorker(
        [{"role": "user", "content": "x"}], model="qwen3", endpoint="http://x",
        stream_post=lambda u, p, c: _answer("a", "b"),
    )
    tokens: list[str] = []
    done: list[dict] = []
    worker.token.connect(tokens.append)
    worker.done.connect(done.append)
    worker.start()
    assert worker.wait(5000)
    app.processEvents()
    assert tokens == ["a", "b"] and done and done[0]["text"] == "ab"
