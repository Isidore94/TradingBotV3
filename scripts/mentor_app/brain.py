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
import re
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
#: The app's auto-attached packs share this many tokens; lower-priority packs are dropped first.
ATTACH_BUDGET_TOKENS = 6000
#: Room below this is not worth a cut pack: the pack is dropped instead.
MIN_TRUNCATED_TOKENS = 300
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
    think: bool | None = None,
) -> dict[str, Any]:
    """One /api/chat body. ``think=False`` turns reasoning off (gemma4 reasons unless told not to)."""
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
    elif think is not None:
        payload["think"] = bool(think)
    return payload


def thinks_unless_told(model: str) -> bool:
    """True for a tag that reasons unless asked not to (gemma4): its reasoning eats ``num_predict``."""
    import ai_summary

    return ai_summary.model_thinking_off(model)


def json_reply(reply: Mapping[str, Any]) -> Any:
    """The JSON in an Ollama chat reply's content: bare, in a code fence, or after a line of prose.

    An empty content raises ValueError naming why (reasoning used the token cap, or nothing came back).
    """
    message = (reply or {}).get("message") or {}
    text = str(message.get("content") or "").strip()
    if not text:
        why = "the reasoning used the token cap" if str(message.get("thinking") or "").strip() else "no content"
        raise ValueError(f"empty reply ({why}, done_reason={(reply or {}).get('done_reason')!r})")
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        start = text.find("{")
        if start < 0:
            raise
        return json.JSONDecoder().raw_decode(text[start:])[0]


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
    attachments: Sequence[Any] = (),
    seen_ids: Iterable[str] = (),
    question: str | None = None,
    attach_budget_tokens: int = ATTACH_BUDGET_TOKENS,
    turn_instruction: str = "",
    max_tokens: int | None = None,
    plain: bool = False,
) -> dict[str, Any]:
    """One chat turn, tools included. Returns the result dict ``done`` carries.

    ``turn_instruction`` (P20, ``attach.turn_shape``) is appended to the newest user message for this turn only;
    ``max_tokens`` caps its replies (``num_predict``), never on a thinking model whose reasoning would eat it.
    ``plain`` (a simple turn) rides on the result so the caller's guard strips bullets, bold and headers.

    ``native_tools`` is the probed capability (:func:`native_tools_for`); None (unknown) = the fallback.
    ``attachments`` (``attach.AttachRequest``) are built here and injected as tool results after the
    newest user turn (native) or as a packs block before it (fallback): never into the system prefix.
    Rows cited in ``seen_ids`` are left out unless ``question`` names their subject.
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
        #: Packs the app attached from the question (``source: auto``), kept or dropped by the budget.
        "attached": [],
        "attach_ms": 0,
        #: "Not covered: ..." for a pre-trade question whose reply skipped checklist sections.
        "appendix": "",
        "cancelled": False,
    }
    emitted: list[str] = []
    gate_packs: list[Any] = []

    def note_pack(name: str, pack: Any, shown_ids: Iterable[str] | None = None) -> None:
        if hasattr(pack, "as_text"):
            result["pack_texts"].append(pack.as_text())
        for pack_id in (getattr(pack, "ids", ()) if shown_ids is None else shown_ids):
            if pack_id not in result["pack_ids"]:
                result["pack_ids"].append(pack_id)
        if name == "gate_pack" and getattr(pack, "rows", ()):
            gate_packs.append(pack)

    def use_pack(name: str, args: dict[str, Any]) -> Any:
        call = {"name": name, "arguments": args}
        result["tool_calls"].append(call)
        on_tool_call(call)
        pack = build_pack(name, args)
        note_pack(name, pack)
        return pack

    native = bool(tools) and native_tools is True
    kept: list[tuple[Any, str]] = []
    if attachments:
        asked = question if question is not None else next(
            (str(m.get("content") or "") for m in reversed(convo) if m.get("role") == "user"), "")
        kept = _attach(list(attachments), build_pack, note_pack, result, set(seen_ids or ()), asked,
                       int(attach_budget_tokens), on_tool_call, cancelled)
        result["attach_ms"] = int((clock() - started) * 1000)
        if cancelled():
            result["cancelled"] = True
            result["total_ms"] = int((clock() - started) * 1000)
            return result
    cap = int(max_tokens) if max_tokens and not is_thinking_model(model) else None
    result["turn_instruction"], result["max_tokens"] = str(turn_instruction or ""), cap
    result["plain"] = bool(plain or turn_instruction)
    if turn_instruction:
        last_user = max((n for n, m in enumerate(convo) if m.get("role") == "user"), default=None)
        if last_user is not None:
            content = str(convo[last_user].get("content") or "").rstrip()
            convo[last_user] = {**convo[last_user], "content": f"{content}\n\n{turn_instruction}"}
    if kept and native:
        convo.append({"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": request.name, "arguments": dict(request.args)}} for request, _ in kept]})
        convo.extend({"role": "tool", "content": text, "tool_name": request.name} for request, text in kept)
    elif kept:
        # No native tools: the app's packs go just before the newest user turn, the prefix untouched.
        convo.insert(len(convo) - 1, {"role": "system", "content": "# Packs (attached by the app)\n"
                                      + "\n\n".join(text for _, text in kept)})
    if tools and not native and not kept:
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
        payload = chat_payload(model, convo, tools=offer, keep_alive=keep_alive, num_ctx=num_ctx, max_tokens=cap)
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
    if not result["cancelled"]:
        _fetch_check(result, convo, attachments, build_pack, note_pack, question, attach_budget_tokens,
                     on_tool_call, on_token, cancelled, model=model, url=url, keep_alive=keep_alive,
                     num_ctx=num_ctx, stream_post=stream_post, max_tokens=cap)
    if not result["cancelled"] and not result["text"].strip():
        # P16: an empty reply never renders empty: the data the app sent, under one plain line.
        logging.warning("Trade Mentor: the model returned no text; showing the %d pack(s) sent",
                        len(result["pack_texts"]))
        result["empty_fallback"] = True
        result["text"] = "\n\n".join([EMPTY_REPLY_LINE, *result["pack_texts"]]) if result["pack_texts"] else (
            f"{EMPTY_REPLY_LINE[:-len('; here is the data)')]}; no data was attached)")
    if gate_packs and not result["cancelled"]:
        from mentor_app import checklist

        result["appendix"] = "\n\n".join(filter(None, (checklist.appendix(result["text"], pack)
                                                       for pack in gate_packs[:1])))
    result["total_ms"] = int((clock() - started) * 1000)
    result["timings"] = {
        "attach_ms": result["attach_ms"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "first_token_ms": result["first_token_ms"],
        "total_ms": result["total_ms"],
        "tool_calls": len(result["tool_calls"]),
        "auto_packs": sum(1 for item in result["attached"] if not item.get("dropped")),
    }
    logging.info("Trade Mentor turn: %s", json.dumps(result["timings"], sort_keys=True))
    return result


#: P16: what an empty model reply shows instead, above the packs the app sent.
EMPTY_REPLY_LINE = "(the model returned no text; here is the data)"


def _fetch_check(result: dict[str, Any], convo: list[dict[str, Any]], attachments: Sequence[Any],
                 build_pack: Callable[[str, Mapping[str, Any]], Any], note_pack: Callable[..., None],
                 question: str | None, budget: int, on_tool_call: Callable[[dict], None],
                 on_token: Callable[[str], None], cancelled: Callable[[], bool], *, model: str, url: str,
                 keep_alive: Any, num_ctx: int | None, stream_post: StreamPost,
                 max_tokens: int | None = None) -> None:
    """P16: a reply that ends announcing a fetch with no tool call made gets ONE re-ask with the planner's packs
    built again (no dedupe); with no packs to fetch, or still announcing, it gets ``style.NO_FETCH_NOTE``."""
    from mentor_app import style

    if not style.announces_fetch(result["text"]) or result["tool_calls"]:
        return
    result["fetch_retry"] = False
    asked = question if question is not None else next(
        (str(m.get("content") or "") for m in reversed(convo) if m.get("role") == "user"), "")
    kept = _attach(list(attachments), build_pack, note_pack, {"attached": []}, set(), asked, int(budget),
                   on_tool_call, cancelled) if attachments else []
    if kept and not cancelled():
        result["fetch_retry"] = True
        ask = convo + [
            {"role": "assistant", "content": result["text"]},
            {"role": "user", "content": style.FETCH_RETRY_PROMPT + "\n\n# Packs (attached by the app)\n"
             + "\n\n".join(text for _, text in kept) + f"\n\nQuestion: {asked}"},
        ]
        parts: list[str] = []
        on_token("\n\n")
        for chunk in iter_ndjson(stream_post(url, chat_payload(model, ask, keep_alive=keep_alive, num_ctx=num_ctx,
                                                                    max_tokens=max_tokens),
                                             cancelled)):
            if cancelled():
                result["cancelled"] = True
                break
            if chunk.get("error"):
                raise BrainError(str(chunk["error"]))
            piece = str((chunk.get("message") or {}).get("content") or "")
            if piece:
                parts.append(piece)
                on_token(piece)
            if chunk.get("done"):
                result["prompt_tokens"] += int(chunk.get("prompt_eval_count") or 0)
                result["completion_tokens"] += int(chunk.get("eval_count") or 0)
        retried = "".join(parts).strip()
        if retried:
            result["first_reply"] = result["text"]
            result["text"] = retried
    if style.announces_fetch(result["text"]) and not result["cancelled"]:
        result["text"] = f"{result['text'].rstrip()} {style.NO_FETCH_NOTE}"


def _render(name: str, rows: Sequence[Mapping[str, Any]], empty_text: str, hidden: int) -> str:
    """A pack as the model sees it: one ``[id] text`` line per shown row."""
    if not rows:
        body = f"## {name}\n{empty_text or 'nothing'}"
    else:
        body = "\n".join([f"## {name}", *(f"[{row['id']}] {row.get('text', '')}" for row in rows)])
    if hidden:
        body += f"\n({hidden} row(s) you cited in the last turns left out; ask again by name to see them)"
    return body


def _attach(
    requests_: list[Any],
    build_pack: Callable[[str, Mapping[str, Any]], Any],
    note_pack: Callable[..., None],
    result: dict[str, Any],
    seen_ids: set[str],
    question: str,
    budget: int,
    on_tool_call: Callable[[dict], None],
    cancelled: Callable[[], bool],
) -> list[tuple[Any, str]]:
    """Build the app's packs, drop recently-cited rows, keep the most important under ``budget`` tokens."""
    from mentor_app.attach import names_subject
    from mentor_app.chat_model import estimate_tokens

    built: list[tuple[Any, Any, list[dict[str, Any]], int]] = []
    for request in sorted(requests_, key=lambda item: getattr(item, "priority", 50)):
        if cancelled():
            return []
        on_tool_call({"name": request.name, "arguments": dict(request.args), "source": "auto"})
        try:
            pack = build_pack(request.name, dict(request.args))
        except Exception as exc:  # noqa: BLE001 - a broken pack is left out, never a guess
            logging.warning("Trade Mentor: auto pack %s failed: %s", request.name, exc)
            continue
        rows = [dict(row) for row in getattr(pack, "rows", ()) or ()]
        shown = [row for row in rows if str(row.get("id")) not in seen_ids or names_subject(question, str(row.get("id")))]
        built.append((request, pack, shown, len(rows) - len(shown)))
    # Strict priority: keep packs in order while they fit; the first that does not fit is cut to the room
    # left (its first rows), and every lower-priority pack after it is dropped, whatever its size.
    kept: list[tuple[Any, str]] = []
    used = 0
    full = False
    for request, pack, shown, hidden in built:
        text = _render(getattr(pack, "name", request.name), shown, getattr(pack, "empty_text", ""), hidden)
        cost = estimate_tokens(text)
        dropped = truncated = False
        room = budget - used
        if full:
            dropped = True
        elif cost > room:
            full = True
            if kept and room < MIN_TRUNCATED_TOKENS:
                dropped = True
            else:
                lines = text.split("\n")
                while lines and estimate_tokens("\n".join(lines)) > room:
                    lines.pop()
                shown = [row for row in shown if f"[{row['id']}]" in "\n".join(lines)]
                text, cost, truncated = "\n".join(lines), estimate_tokens("\n".join(lines)), True
        entry = {"name": request.name, "arguments": dict(request.args), "reason": getattr(request, "reason", ""),
                 "source": "auto", "rows": len(shown), "hidden": hidden, "tokens": cost, "dropped": dropped,
                 "truncated": truncated}
        result["attached"].append(entry)
        if dropped:
            continue
        used += cost
        note_pack(request.name, pack, [str(row["id"]) for row in shown])
        kept.append((request, text))
    return kept


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
