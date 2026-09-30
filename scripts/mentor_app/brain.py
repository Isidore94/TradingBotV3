"""The brain: Ollama native ``/api/chat`` streaming with a tool loop over the pack registry.

``run_turn`` is Qt-free and takes injected ``stream_post``/``post`` callables so tests
never touch the network. ``StreamWorker`` runs it on a QThread and re-emits tokens,
tool calls, the result and failures as signals. At most MAX_TOOL_CALLS packs per turn;
after that the model must answer without tools. Models without native tool calling
(gemma) get a one-shot JSON "which packs do you need" step first.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import requests
from PySide6.QtCore import QThread, Signal

MAX_TOOL_CALLS = 4
THINKING_MODEL_PREFIXES = ("gpt-oss",)
#: Families Ollama serves without native tool calling.
NO_NATIVE_TOOLS_MARKERS = ("gemma",)
CHAT_REASONING_EFFORT = "low"
STREAM_TIMEOUT = (10, 300)
SELECT_TIMEOUT = 120

StreamPost = Callable[[str, dict, Callable[[], bool]], Iterable[Any]]
Post = Callable[[str, dict, float], Mapping[str, Any]]


class BrainError(RuntimeError):
    """Ollama answered with an error, or not at all."""


def is_thinking_model(model: str) -> bool:
    return str(model or "").strip().lower().startswith(THINKING_MODEL_PREFIXES)


def has_native_tools(model: str) -> bool:
    lowered = str(model or "").lower()
    return not any(marker in lowered for marker in NO_NATIVE_TOOLS_MARKERS)


def chat_payload(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    stream: bool = True,
    keep_alive: Any = -1,
    num_ctx: int | None = None,
    fmt: Any = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if num_ctx:
        options["num_ctx"] = int(num_ctx)
    if max_tokens:
        options["num_predict"] = int(max_tokens)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "stream": bool(stream),
        "keep_alive": keep_alive,
        "options": options,
    }
    if tools:
        payload["tools"] = [dict(tool) for tool in tools]
    if fmt is not None:
        payload["format"] = fmt
    if is_thinking_model(model):
        # Same shape as the night's gpt-oss call; `think` is the native API's name for it.
        payload["reasoning_effort"] = CHAT_REASONING_EFFORT
        payload["think"] = CHAT_REASONING_EFFORT
    return payload


def iter_ndjson(lines: Iterable[Any]) -> Iterator[dict[str, Any]]:
    """One dict per non-empty NDJSON line; a garbled line is logged and skipped."""
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        text = str(raw or "").strip()
        if not text:
            continue
        if text.startswith("data:"):
            text = text[5:].strip()
        try:
            chunk = json.loads(text)
        except ValueError:
            logging.warning("Trade Mentor: unreadable stream line skipped: %.120s", text)
            continue
        if isinstance(chunk, dict):
            yield chunk


def default_stream_post(url: str, payload: dict, cancelled: Callable[[], bool]) -> Iterator[bytes]:
    response = requests.post(url, json=payload, stream=True, timeout=STREAM_TIMEOUT)
    try:
        if response.status_code != 200:
            raise BrainError(f"HTTP {response.status_code}: {response.text[:200]}")
        for line in response.iter_lines():
            if cancelled():
                return
            yield line
    finally:
        response.close()


def default_post(url: str, payload: dict, timeout: float) -> Mapping[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise BrainError(f"HTTP {response.status_code}: {response.text[:200]}")
    return response.json()


def _arguments(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function") or {}
    name = str(function.get("name") or call.get("name") or "")
    args = function.get("arguments", call.get("arguments")) or {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            args = {}
    return name, dict(args) if isinstance(args, Mapping) else {}


def _default_build(name: str, args: Mapping[str, Any]):
    from mentor_packs import registry

    return registry.build(name, **{key: value for key, value in args.items() if key != "name"})


SELECT_PROMPT = (
    "Before answering, choose the packs (at most {cap}) you need from this list, as JSON "
    '{{"packs": [{{"name": "...", "args": {{}}}}]}}. Choose none if the question needs no desk data.\n{catalog}'
)
SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "packs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "args": {"type": "object"}},
                "required": ["name"],
            },
        }
    },
    "required": ["packs"],
}


def run_turn(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    endpoint: str,
    keep_alive: Any = -1,
    num_ctx: int | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    stream_post: StreamPost = default_stream_post,
    post: Post = default_post,
    build_pack: Callable[[str, Mapping[str, Any]], Any] = _default_build,
    on_token: Callable[[str], None] = lambda text: None,
    on_tool_call: Callable[[dict], None] = lambda call: None,
    cancelled: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """One chat turn, tools included. Returns the result dict ``done`` carries."""
    url = f"{endpoint.rstrip('/')}/api/chat"
    started = clock()
    convo = [dict(message) for message in messages]
    result: dict[str, Any] = {
        "text": "",
        "model": model,
        "first_token_ms": None,
        "total_ms": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tool_calls": [],
        "pack_ids": [],
        "cancelled": False,
    }
    emitted: list[str] = []

    def use_pack(name: str, args: dict[str, Any]) -> Any:
        call = {"name": name, "arguments": args}
        result["tool_calls"].append(call)
        on_tool_call(call)
        pack = build_pack(name, args)
        for pack_id in getattr(pack, "ids", ()):
            if pack_id not in result["pack_ids"]:
                result["pack_ids"].append(pack_id)
        return pack

    native = bool(tools) and has_native_tools(model)
    if tools and not native:
        catalog = "\n".join(
            f"- {tool['function']['name']}: {tool['function'].get('description', '')}" for tool in tools
        )
        ask = convo + [{"role": "user", "content": SELECT_PROMPT.format(cap=MAX_TOOL_CALLS, catalog=catalog)}]
        try:
            reply = post(url, chat_payload(model, ask, stream=False, keep_alive=keep_alive, num_ctx=num_ctx, fmt=SELECT_SCHEMA), SELECT_TIMEOUT)
            chosen = json.loads(str((reply.get("message") or {}).get("content") or "{}")).get("packs") or []
        except Exception as exc:  # noqa: BLE001 - no selection means answer without packs
            logging.info("Trade Mentor: pack selection skipped (%s)", exc)
            chosen = []
        texts = []
        for item in list(chosen)[:MAX_TOOL_CALLS]:
            if isinstance(item, Mapping) and item.get("name"):
                args = item.get("args") if isinstance(item.get("args"), Mapping) else {}
                texts.append(use_pack(str(item["name"]), dict(args)).as_text())
        if texts:
            convo.insert(len(convo) - 1, {"role": "system", "content": "# Packs\n" + "\n\n".join(texts)})

    while True:
        offer = tools if native and len(result["tool_calls"]) < MAX_TOOL_CALLS else None
        payload = chat_payload(model, convo, tools=offer, keep_alive=keep_alive, num_ctx=num_ctx)
        parts: list[str] = []
        calls: list[Mapping[str, Any]] = []
        for chunk in iter_ndjson(stream_post(url, payload, cancelled)):
            if cancelled():
                break
            if chunk.get("error"):
                raise BrainError(str(chunk["error"]))
            message = chunk.get("message") or {}
            piece = str(message.get("content") or "")
            if piece:
                if result["first_token_ms"] is None:
                    result["first_token_ms"] = int((clock() - started) * 1000)
                parts.append(piece)
                emitted.append(piece)
                on_token(piece)
            calls.extend(call for call in message.get("tool_calls") or () if isinstance(call, Mapping))
            if chunk.get("done"):
                result["prompt_tokens"] += int(chunk.get("prompt_eval_count") or 0)
                result["completion_tokens"] += int(chunk.get("eval_count") or 0)
        if cancelled():
            result["cancelled"] = True
            break
        if not calls or offer is None:
            break
        convo.append({"role": "assistant", "content": "".join(parts), "tool_calls": [dict(call) for call in calls]})
        for call in calls:
            if len(result["tool_calls"]) >= MAX_TOOL_CALLS:
                break
            name, args = _arguments(call)
            convo.append({"role": "tool", "content": use_pack(name, args).as_text(), "tool_name": name})
    result["text"] = "".join(emitted)
    result["total_ms"] = int((clock() - started) * 1000)
    return result


def warm(endpoint: str, model: str, keep_alive: Any = -1, *, post: Post = default_post) -> Mapping[str, Any]:
    """Load the model and keep it resident (an empty chat)."""
    return post(f"{endpoint.rstrip('/')}/api/chat", {"model": model, "messages": [], "keep_alive": keep_alive}, 300)


def unload(endpoint: str, model: str, *, post: Post = default_post) -> Mapping[str, Any]:
    """Hand the GPU back: keep_alive 0 unloads the model now."""
    return post(f"{endpoint.rstrip('/')}/api/chat", {"model": model, "messages": [], "keep_alive": 0}, 60)


def embed(endpoint: str, texts: Sequence[str], *, model: str, post: Post = default_post) -> list[list[float]]:
    reply = post(f"{endpoint.rstrip('/')}/api/embed", {"model": model, "input": list(texts)}, 60)
    return [list(map(float, vector)) for vector in reply.get("embeddings") or ()]


class StreamWorker(QThread):
    """Runs one turn off the Qt thread; ``cancel()`` stops it at the next chunk."""

    token = Signal(str)
    tool_call = Signal(dict)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, messages, *, parent=None, **turn_kwargs) -> None:
        super().__init__(parent)
        self._messages = list(messages)
        self._kwargs = dict(turn_kwargs)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            result = run_turn(
                self._messages,
                on_token=self.token.emit,
                on_tool_call=self.tool_call.emit,
                cancelled=lambda: self._cancel,
                **self._kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - any failure goes to the banner
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(result)
