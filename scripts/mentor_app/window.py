"""MentorWindow: the Trade Mentor chat window. The Qt thread only paints.

Streaming runs on a ``brain.StreamWorker`` QThread; the tunnel, warm-up and unload on
plain threads; pack builds and embeddings on the prefetch queue's thread; every store
write on one IO thread. Results come back through queued signals on ``_Bridge``.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from mentor_app import brain, commands, settings
from mentor_app.chat_model import ChatModel
from mentor_app.inbox import Inbox
from mentor_app.prefetch import PRIORITY_EMBED, PRIORITY_REFRESH, PrefetchQueue
from mentor_app.store import MentorChatStore

CONTEXT_REFRESH_MS = 5 * 60 * 1000
GPU_CHECK_MS = 60 * 1000
#: How long after a failed connect the app waits before trying the host again.
RECONNECT_BACKOFF_SECONDS = 10 * 60
CHIP_KINDS = ("auto_mode", "d1_env", "regime")


class _Bridge(QObject):
    """Signals the worker threads emit; Qt queues them onto the window's thread."""

    context_ready = Signal(object)
    brain_state = Signal(dict)
    memory_ready = Signal(str)
    note = Signal(str)


class InputBox(QPlainTextEdit):
    """Enter sends; Shift+Enter is a new line."""

    submitted = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class MentorWindow(QMainWindow):
    def __init__(
        self,
        *,
        store: MentorChatStore | None = None,
        tunnel: Any = None,
        queue: PrefetchQueue | None = None,
        inbox: Inbox | None = None,
        stream_post: Callable[..., Any] | None = None,
        post: Callable[..., Any] | None = None,
        now: Callable[[], datetime] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Trade Mentor")
        self.resize(1100, 800)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.store = store or MentorChatStore()
        self._tunnel = tunnel
        self._stream_post = stream_post or brain.default_stream_post
        self._post = post or brain.default_post
        self._brain_ok = False
        self._brain_reason = "connecting to the GPU host..."
        self._connecting = False
        self._last_connect = 0.0
        self._host = ""
        self._model = ""
        self._endpoint = ""
        self._latency_ms: int | None = None
        self.queue = queue or PrefetchQueue(blocked=self._gpu_reason, model_ready=lambda: self._brain_ok)
        self.inbox = inbox or Inbox(per_day_cap=settings.proactive_per_day(), now=self._now)
        self.chat = ChatModel()
        self._context_pack: Any = None
        self._context_text = ""
        self._memory_text = ""
        self._session_id: int | None = None
        self._worker: Any = None
        self._blocks: list[str] = []
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mentor-store")
        self._threads: list[threading.Thread] = []
        self._focus_server = None
        self._bridge = _Bridge(self)
        self._bridge.context_ready.connect(self._on_context)
        self._bridge.brain_state.connect(self._on_brain_state)
        self._bridge.memory_ready.connect(self._on_memory)
        self._bridge.note.connect(self._add_note)
        self._build_ui()
        self._context_timer = QTimer(self)
        self._context_timer.setInterval(CONTEXT_REFRESH_MS)
        self._context_timer.timeout.connect(self.refresh_context)
        self._gpu_timer = QTimer(self)
        self._gpu_timer.setInterval(GPU_CHECK_MS)
        self._gpu_timer.timeout.connect(self.check_gpu_share)
        self._add_note("Hi. Ask me anything, or type `/help`.")

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.banner = QLabel("")
        self.banner.setObjectName("MentorBanner")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        self.transcript = QTextBrowser()
        self.transcript.setObjectName("MentorTranscript")
        self.transcript.setOpenExternalLinks(False)
        self.chip_row = QHBoxLayout()
        self.chip_row.setSpacing(6)
        self.chips: dict[str, QPushButton] = {}
        for kind in CHIP_KINDS + ("positions", "focus"):
            chip = QPushButton(kind.replace("_", " "))
            chip.setObjectName("MentorChip")
            chip.setFlat(True)
            chip.clicked.connect(lambda _=False, k=kind: self._show_chip(k))
            self.chips[kind] = chip
            self.chip_row.addWidget(chip)
        self.chip_row.addStretch(1)
        self.input = InputBox()
        self.input.setPlaceholderText("Ask the mentor... (Enter sends, Shift+Enter new line, Win+H to dictate)")
        self.input.setFixedHeight(84)
        self.input.submitted.connect(self.send_current)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_current)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_turn)
        buttons = QVBoxLayout()
        buttons.addWidget(self.send_button)
        buttons.addWidget(self.stop_button)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input, 1)
        input_row.addLayout(buttons)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.banner)
        left_layout.addWidget(self.transcript, 1)
        left_layout.addLayout(self.chip_row)
        left_layout.addLayout(input_row)

        self.inbox_header = QLabel("Inbox")
        self.inbox_header.setObjectName("MentorInboxHeader")
        self.inbox_list = QListWidget()
        self.inbox_list.itemClicked.connect(self._open_inbox_item)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.inbox_header)
        right_layout.addWidget(self.inbox_list, 1)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.status_pill = QLabel("")
        self.status_pill.setObjectName("MentorStatusPill")
        self.activity_label = QLabel("")
        self.activity_label.setObjectName("MentorActivity")
        self.statusBar().addWidget(self.activity_label, 1)
        self.statusBar().addPermanentWidget(self.status_pill)
        self._sync_status()
        self.refresh_inbox()

    def _render(self) -> None:
        self.transcript.setMarkdown("\n\n---\n\n".join(self._blocks))
        self.transcript.moveCursor(QTextCursor.MoveOperation.End)

    def _add_block(self, markdown: str) -> None:
        self._blocks.append(markdown)
        self._render()

    def _add_note(self, markdown: str) -> None:
        self._add_block(f"*Mentor (desk):* {markdown}")

    def _sync_status(self) -> None:
        model = self._model or "model?"
        host = self._host or "no host"
        latency = f"{self._latency_ms} ms" if self._latency_ms is not None else "-"
        state = "ready" if self._brain_ok else "off"
        self.status_pill.setText(f"{host} · {model} · {latency} · brain {state}")
        self.banner.setVisible(not self._brain_ok)
        self.banner.setText(f"Brain is off: {self._brain_reason}. Packs still work: try /tape.")

    # ------------------------------------------------------------------ lifecycle
    def start_background(self) -> None:
        """Start the queue, the timers, the focus listener and the first connect."""
        from mentor_app.focus_link import make_focus_server

        self._focus_server = make_focus_server(self, self.bring_to_front)
        self.queue.start()
        self._context_timer.start()
        self._gpu_timer.start()
        self._submit_io(self._open_session)
        self.refresh_context()
        self.connect_brain()

    def shutdown(self) -> None:
        self._context_timer.stop()
        self._gpu_timer.stop()
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
        self.queue.stop()
        if self._tunnel is not None:
            self._tunnel.stop()
        self._io.shutdown(wait=True)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.shutdown()
        super().closeEvent(event)

    def bring_to_front(self) -> None:
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    def _submit_io(self, fn: Callable[[], Any]) -> None:
        try:
            self._io.submit(fn)
        except RuntimeError:  # shut down
            pass

    def _spawn(self, name: str, fn: Callable[[], None]) -> None:
        thread = threading.Thread(target=fn, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _open_session(self) -> None:
        self._session_id = self.store.start_session(self._model)
        notes = self.store.profile_notes()
        self._bridge.memory_ready.emit("\n".join(f"- {row['text']}" for row in notes))

    def _on_memory(self, text: str) -> None:
        self._memory_text = text

    # ------------------------------------------------------------------ brain
    def _gpu_reason(self) -> str:
        try:
            return settings.gpu_block_reason(self._now())
        except Exception as exc:  # noqa: BLE001 - an unreadable clock keeps the model off
            return f"the GPU window could not be read ({type(exc).__name__})"

    def connect_brain(self) -> None:
        if self._connecting:
            return
        self._connecting = True
        self._last_connect = time.monotonic()
        self._spawn("mentor-connect", self._connect_worker)

    def _connect_worker(self) -> None:
        state: dict[str, Any] = {"ok": False, "reason": "", "host": "", "model": "", "endpoint": ""}
        try:
            state["model"] = settings.mentor_model()
            reason = self._gpu_reason()
            if reason:
                state["reason"] = reason
                return
            if self._tunnel is None:
                from mentor_app.tunnel import from_settings

                self._tunnel = from_settings()
            status = self._tunnel.preflight()
            state["host"] = status.host
            if not status.ok:
                state["reason"] = status.reason
                return
            state["endpoint"] = self._tunnel.endpoint
            brain.warm(state["endpoint"], state["model"], settings.keep_alive(), post=self._post)
            self._install_recall(state["endpoint"])
            state["ok"] = True
        except Exception as exc:  # noqa: BLE001 - any failure is "brain off", with the reason
            state["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            self._bridge.brain_state.emit(state)

    def _install_recall(self, endpoint: str) -> None:
        from mentor_packs import recall

        recall.set_searcher(
            recall.make_searcher(
                lambda texts: brain.embed(endpoint, texts, model=settings.EMBED_MODEL, post=self._post),
                lambda: self.store.embeddings(settings.EMBED_MODEL),
            )
        )

    def _on_brain_state(self, state: dict) -> None:
        self._connecting = False
        self._brain_ok = bool(state.get("ok"))
        self._brain_reason = str(state.get("reason") or "")
        self._host = str(state.get("host") or self._host)
        self._model = str(state.get("model") or self._model)
        self._endpoint = str(state.get("endpoint") or self._endpoint)
        self._sync_status()

    def check_gpu_share(self) -> None:
        """Every minute: hand the GPU back before the night, take it again after."""
        reason = self._gpu_reason()
        if reason and self._brain_ok:
            self._brain_ok = False
            self._brain_reason = reason
            endpoint, model = self._endpoint, self._model
            from mentor_packs import recall

            recall.set_searcher(None)
            self._spawn("mentor-unload", lambda: self._unload(endpoint, model))
            self._sync_status()
        elif reason:
            self._brain_reason = reason
            self._sync_status()
        elif not self._brain_ok and not self._connecting:
            if time.monotonic() - self._last_connect >= RECONNECT_BACKOFF_SECONDS or self._last_connect == 0:
                self.connect_brain()

    def _unload(self, endpoint: str, model: str) -> None:
        try:
            brain.unload(endpoint, model, post=self._post)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Trade Mentor: unload failed: %s", exc)

    # ------------------------------------------------------------------ context
    def refresh_context(self) -> None:
        def build():
            from mentor_packs import context_pack

            pack = context_pack.build()
            self.store.put_pack(pack.name, {}, pack.as_json(), pack.built_utc)
            return pack

        self.queue.submit("context_pack", build, priority=PRIORITY_REFRESH, key="context_pack",
                          on_done=self._bridge.context_ready.emit)

    def _on_context(self, pack: Any) -> None:
        self._context_pack = pack
        self._context_text = pack.as_text()
        rows = list(pack.rows)
        by_kind = {row.get("kind"): row for row in rows}
        for kind in CHIP_KINDS:
            row = by_kind.get(kind) or next((r for r in rows if r.get("id") == f"ctx:{kind}"), None)
            if row:
                self.chips[kind].setText(str(row["text"]).split(" (")[0][:48])
        positions = [row for row in rows if row.get("kind") == "position"]
        self.chips["positions"].setText(f"Open: {len(positions)}")
        focus = sum(
            0 if "none" in str(row["text"]) else str(row["text"]).count(",") + 1
            for row in rows if row.get("kind") == "focus"
        )
        self.chips["focus"].setText(f"Focus: {focus}")

    def _show_chip(self, kind: str) -> None:
        if self._context_pack is None:
            self._add_note("The desk context is still loading.")
            return
        prefix = {"positions": "ctx:pos", "focus": "ctx:focus"}.get(kind, f"ctx:{kind}")
        lines = [f"[{row['id']}] {row['text']}" for row in self._context_pack.rows if str(row["id"]).startswith(prefix)]
        self._add_note("\n\n".join(lines) or "nothing")

    # ------------------------------------------------------------------ inbox
    def post_to_inbox(self, kind: str, text: str, **kwargs: Any) -> bool:
        item = self.inbox.add(kind, text, **kwargs)
        self.refresh_inbox()
        return item is not None

    def refresh_inbox(self) -> None:
        self.inbox_list.clear()
        for item in self.inbox.items():
            row = QListWidgetItem(("• " if not item.read else "") + f"{item.kind}: {item.text[:80]}")
            row.setData(Qt.ItemDataRole.UserRole, item.id)
            self.inbox_list.addItem(row)
        badge = self.inbox.badge()
        self.inbox_header.setText(f"Inbox ({badge})" if badge else "Inbox")

    def _open_inbox_item(self, row: QListWidgetItem) -> None:
        item_id = int(row.data(Qt.ItemDataRole.UserRole))
        for item in self.inbox.items():
            if item.id == item_id:
                self._add_note(item.text)
        self.inbox.mark_read(item_id)
        self.refresh_inbox()

    # ------------------------------------------------------------------ sending
    def send_current(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or self._worker is not None:
            return
        self.input.clear()
        self.send(text)

    def send(self, text: str) -> None:
        result = commands.handle(text)
        if result is not None:
            self._add_block(f"**You:** {text}")
            self._run_command(result)
            return
        self._add_block(f"**You:** {text}")
        self._store_turn("user", text)
        if not self._brain_ok:
            self._add_note(f"The brain is off: {self._brain_reason or 'not connected'}. Packs still work: try `/tape`.")
            return
        self.chat.add("user", text)
        from mentor_packs import registry

        messages = self.chat.messages(
            context_text=self._context_text, budget_tokens=settings.context_tokens(), memory_text=self._memory_text
        )
        worker = brain.StreamWorker(
            messages,
            parent=self,
            model=self._model,
            endpoint=self._endpoint,
            keep_alive=settings.keep_alive(),
            num_ctx=settings.context_tokens(),
            tools=registry.tool_schemas(),
            stream_post=self._stream_post,
            post=self._post,
        )
        worker.token.connect(self._on_token)
        worker.tool_call.connect(self._on_tool_call)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        self.queue.begin_interactive()
        self.stop_button.setEnabled(True)
        self.send_button.setEnabled(False)
        self._blocks.append("**Mentor:** ")
        self._render()
        worker.start()

    def _run_command(self, result: commands.CommandResult) -> None:
        if result.action == "quiet":
            until = self.inbox.mute(result.arg)
            self._add_note(f"Inbox muted until {until.astimezone().strftime('%H:%M')}.")
        elif result.action == "remember":
            note = str(result.arg)
            self._memory_text = (self._memory_text + f"\n- {note}").strip()
            self._submit_io(lambda: self.store.add_profile_note(note, "remember"))
            self._add_note(f"Kept: {note}")
        elif result.action == "tape":
            if self._context_pack is None:
                self._add_note("The desk context is still loading; try again in a moment.")
            else:
                self._add_note(self._context_pack.as_text().replace("\n", "\n\n") + "\n\n(Full tape talk comes in Phase 5.)")
        else:
            self._add_note(result.reply)

    def stop_turn(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _on_token(self, text: str) -> None:
        self._blocks[-1] += text
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.transcript.setTextCursor(cursor)

    def _on_tool_call(self, call: dict) -> None:
        # A quiet label in the window's own status bar, cleared when the turn ends.
        self.activity_label.setText(f"reading {call.get('name')}...")

    def _finish_turn(self) -> None:
        self._worker = None
        self.activity_label.setText("")
        self.queue.end_interactive()
        self.stop_button.setEnabled(False)
        self.send_button.setEnabled(True)

    def _on_done(self, result: dict) -> None:
        text = str(result.get("text") or "")
        if result.get("cancelled"):
            text += " *(stopped)*"
        self._blocks[-1] = f"**Mentor:** {text}"
        self._render()
        self.chat.add("assistant", text)
        self._latency_ms = result.get("first_token_ms")
        self._sync_status()
        self._store_turn(
            "assistant",
            text,
            pack_ids=result.get("pack_ids") or (),
            model=str(result.get("model") or self._model),
            latency_ms=result.get("first_token_ms"),
            prompt_tokens=result.get("prompt_tokens"),
            completion_tokens=result.get("completion_tokens"),
            tool_calls=result.get("tool_calls") or (),
        )
        self._finish_turn()
        self._queue_embeddings()

    def _on_failed(self, message: str) -> None:
        self._blocks[-1] = f"**Mentor:** *(failed: {message})*"
        self._render()
        self._brain_reason = message
        self._finish_turn()

    def _store_turn(self, role: str, text: str, **kwargs: Any) -> None:
        self._submit_io(lambda: self.store.add_turn(self._session_id, role, text, **kwargs))

    def _queue_embeddings(self) -> None:
        endpoint = self._endpoint

        def job() -> int:
            done = 0
            for row in self.store.unembedded_turns(settings.EMBED_MODEL, limit=20):
                vectors = brain.embed(endpoint, [row["text"]], model=settings.EMBED_MODEL, post=self._post)
                if vectors:
                    self.store.put_embedding("turn", row["id"], settings.EMBED_MODEL, vectors[0], text=row["text"][:2000])
                    done += 1
            return done

        # Waits for the IO thread's pending turn writes first, then embeds on the queue.
        self._submit_io(
            lambda: self.queue.submit("embed_turns", job, priority=PRIORITY_EMBED, needs_model=True, key="embed_turns")
        )
