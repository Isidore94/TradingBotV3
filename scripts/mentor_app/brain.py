"""The brain: Ollama native ``/api/chat`` streaming with a tool loop over the pack registry.

``run_turn`` is Qt-free and takes injected ``stream_post``/``post`` callables so tests
never touch the network. ``StreamWorker`` runs it on a QThread and re-emits tokens,
tool calls, the result and failures as signals. At most MAX_TOOL_CALLS packs per turn;
after that the model must answer without tools. Native tool calling is decided by the
model's own capabilities (``/api/show``), never by its tag; a model without ``tools``, or
one whose capabilities are unknown, gets a one-shot JSON "which packs do you need" step.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import requests
from PySide6.QtCore import QThread, Signal

MAX_TOOL_CALLS = 4
THINKING_MODEL_PREFIXES = ("gpt-oss",)
#: How long a tag's probed capabilities are trusted (app_state cache).
CAPS_TTL = timedelta(hours=24)
CAPS_STATE_PREFIX = "model_caps:"
SHOW_TIMEOUT = 20
CHAT_REASONING_EFFORT = "low"
STREAM_TIMEOUT = (10, 300)
SELECT_TIMEOUT = 120
#: How often a blocking one-shot call checks the Stop flag.
CANCEL_POLL_SECONDS = 0.1

StreamPost = Callable[[str, dict, Callable[[], bool]], Iterable[Any]]
Post = Callable[[str, dict, float], Mapping[str, Any]]


class BrainError(RuntimeError):
    """Ollama answered with an error, or not at all."""


def is_thinking_model(model: str) -> bool:
    return str(model or "").strip().lower().startswith(THINKING_MODEL_PREFIXES)


#: Tags whose failed or unknown capability probe was already logged (once per tag per process).
_probe_logged: set[str] = set()


def _log_probe_once(model: str, why: str) -> None:
    if model in _probe_logged:
        return
    _probe_logged.add(model)
    logging.info("Trade Mentor: %s capabilities unknown (%s); tools: fallback", model, why)


def show_model(endpoint: str, model: str, *, post: Post) -> Mapping[str, Any] | None:
    """Ollama ``POST /api/show`` for one tag; None when the host does not describe it."""
    reply = post(f"{endpoint.rstrip('/')}/api/show", {"model": model}, SHOW_TIMEOUT)
    if isinstance(reply, Mapping) and ("capabilities" in reply or "details" in reply):
        return reply
    return None


def _caps_of(reply: Mapping[str, Any] | None) -> tuple[str, ...] | None:
    caps = (reply or {}).get("capabilities")
    if not isinstance(caps, (list, tuple)):
        return None
    return tuple(str(cap).strip().lower() for cap in caps)


def model_capabilities(
    endpoint: str, model: str, *, post: Post, store: Any = None, now: datetime | None = None
) -> tuple[str, ...] | None:
    """The tag's capabilities (e.g. ``("completion", "tools")``), cached 24 h in app_state; None = unknown."""
    moment = now or datetime.now(timezone.utc)
    key = CAPS_STATE_PREFIX + str(model)
    if store is not None:
        try:
            cached = json.loads(store.get_state(key) or "{}")
            at = datetime.fromisoformat(str(cached.get("at_utc") or ""))
            if isinstance(cached.get("caps"), list) and timedelta(0) <= moment - at < CAPS_TTL:
                return tuple(str(cap) for cap in cached["caps"])
        except (TypeError, ValueError, AttributeError):
            pass
    try:
        caps = _caps_of(show_model(endpoint, model, post=post))
    except Exception as exc:  # noqa: BLE001 - a failed probe is "unknown", never "native"
        _log_probe_once(model, f"{type(exc).__name__}: {exc}")
        return None
    if caps is None:
        _log_probe_once(model, "the host listed no capabilities")
        return None
    if store is not None:
        store.set_state(key, json.dumps({"caps": list(caps), "at_utc": moment.astimezone(timezone.utc).isoformat()}))
    return caps


def native_tools_for(caps: Sequence[str] | None) -> bool:
    """Native tool calling only when the model lists ``tools``; unknown = the fallback."""
    return caps is not None and "tools" in caps


def model_present(endpoint: str, model: str, *, post: Post) -> bool:
    """True when the host describes ``model`` (``/api/show`` answers for it); any failure = absent."""
    try:
        return show_model(endpoint, model, post=post) is not None
    except Exception as exc:  # noqa: BLE001
        logging.info("Trade Mentor: %s is not on the host (%s)", model, exc)
        return False


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


def _cancellable_call(fn: Callable[[], Any], cancelled: Callable[[], bool]) -> tuple[bool, Any]:
    """Run a blocking call on a daemon thread, polling ``cancelled``: (False, None) once it is set.

    A cancelled call is abandoned; its thread ends when the request times out.
    """
    box: dict[str, Any] = {}
    finished = threading.Event()

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=target, name="mentor-pack-choice", daemon=True).start()
    while not finished.wait(CANCEL_POLL_SECONDS):
        if cancelled():
            return False, None
    if "error" in box:
        raise box["error"]
    return True, box.get("value")


def run_turn(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    endpoint: str,
    keep_alive: Any = -1,
    num_ctx: int | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    native_tools: bool | None = None,
    stream_post: StreamPost = default_stream_post,
    post: Post = default_post,
    build_pack: Callable[[str, Mapping[str, Any]], Any] = _default_build,
    on_token: Callable[[str], None] = lambda text: None,
    on_tool_call: Callable[[dict], None] = lambda call: None,
    cancelled: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """One chat turn, tools included. Returns the result dict ``done`` carries.

    ``native_tools`` is the probed capability (:func:`native_tools_for`); None (unknown) = the fallback.
    """
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
        #: Every pack text sent this turn (guardrail 2 greys numbers found in none of them).
        "pack_texts": [],
        "cancelled": False,
    }
    emitted: list[str] = []

    def use_pack(name: str, args: dict[str, Any]) -> Any:
        call = {"name": name, "arguments": args}
        result["tool_calls"].append(call)
        on_tool_call(call)
        pack = build_pack(name, args)
        if hasattr(pack, "as_text"):
            result["pack_texts"].append(pack.as_text())
        for pack_id in getattr(pack, "ids", ()):
            if pack_id not in result["pack_ids"]:
                result["pack_ids"].append(pack_id)
        return pack

    native = bool(tools) and native_tools is True
    if tools and not native:
        catalog = "\n".join(
            f"- {tool['function']['name']}: {tool['function'].get('description', '')}" for tool in tools
        )
        ask = convo + [{"role": "user", "content": SELECT_PROMPT.format(cap=MAX_TOOL_CALLS, catalog=catalog)}]
        select_payload = chat_payload(model, ask, stream=False, keep_alive=keep_alive, num_ctx=num_ctx, fmt=SELECT_SCHEMA)
        try:
            finished, reply = _cancellable_call(lambda: post(url, select_payload, SELECT_TIMEOUT), cancelled)
            if not finished:
                result["cancelled"] = True
                result["total_ms"] = int((clock() - started) * 1000)
                return result
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


def unload(endpoint: str, model: str, *, post: Post = default_post, timeout: float = 60) -> Mapping[str, Any]:
    """Hand the GPU back: keep_alive 0 unloads the model now."""
    return post(f"{endpoint.rstrip('/')}/api/chat", {"model": model, "messages": [], "keep_alive": 0}, timeout)


def unload_embedder(endpoint: str, model: str, *, post: Post = default_post, timeout: float = 60) -> Mapping[str, Any]:
    """Unload an embedding model: it has no chat route, so an empty /api/embed carries keep_alive 0."""
    return post(f"{endpoint.rstrip('/')}/api/embed", {"model": model, "input": [], "keep_alive": 0}, timeout)


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
