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

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from mentor_app import assess as pick_assess
from mentor_app import brain, challenge, commands, grounding, memory, pick_jobs, settings
from mentor_app.chat_model import ChatModel
from mentor_app.inbox import Inbox
from mentor_app.prefetch import (
    PRIORITY_EMBED,
    PRIORITY_IDLE,
    PRIORITY_INTERACTIVE,
    PRIORITY_REFRESH,
    PrefetchQueue,
)
from mentor_app.store import MentorChatStore

CONTEXT_REFRESH_MS = 5 * 60 * 1000
GPU_CHECK_MS = 60 * 1000
PICK_CHECK_MS = 60 * 1000
#: How long a live narration may wait for the model before the card shows the evidence alone.
ASSESS_WAIT_MS = 90 * 1000
#: How long after a failed connect the app waits before trying the host again.
RECONNECT_BACKOFF_SECONDS = 10 * 60
#: On close the app waits at most this long for each model unload.
SHUTDOWN_UNLOAD_SECONDS = 4
CHIP_KINDS = ("auto_mode", "d1_env", "regime")


class _Bridge(QObject):
    """Signals the worker threads emit; Qt queues them onto the window's thread."""

    context_ready = Signal(object)
    brain_state = Signal(dict)
    memory_ready = Signal(object)
    still_true = Signal(object)
    note = Signal(str)
    pick_built = Signal(object)
    pick_card = Signal(object)
    pick_failed = Signal(object)
    liked_ready = Signal(object)
    veto_built = Signal(object)
    veto_card = Signal(object)
    veto_failed = Signal(object)
    veto_morning = Signal(object)


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
        card_host: Any = None,
        mentor_enabled: bool | None = None,
        pick_builder: Callable[[str, str], Any] | None = None,
        assess_request: Callable[..., Any] | None = None,
        focus_source: Callable[[], Any] | None = None,
        liked_source: Callable[[], Any] | None = None,
        veto_builder: Callable[[str], Any] | None = None,
        challenge_request: Callable[..., Any] | None = None,
        veto_outcomes: Any = None,
        memory_root: Any = None,
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
        #: P4 memory: the start-of-day block (system prefix, byte-stable) and this session's new notes (tail).
        self._memory_root = memory_root
        self._memory = memory.Memory()
        self._memory_block = ""
        self._memory_text = ""
        self._still_true_day: Any = None
        self._session_id: int | None = None
        self._worker: Any = None
        self._blocks: list[str] = []
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mentor-store")
        self._threads: list[threading.Thread] = []
        self._focus_server = None
        self._shut = False
        self._bridge = _Bridge(self)
        self._bridge.context_ready.connect(self._on_context)
        self._bridge.brain_state.connect(self._on_brain_state)
        self._bridge.memory_ready.connect(self._on_memory)
        self._bridge.still_true.connect(self._on_still_true)
        self._bridge.note.connect(self._add_note)
        self._bridge.pick_built.connect(self._on_pick_built)
        self._bridge.pick_card.connect(self._on_pick_card)
        self._bridge.pick_failed.connect(self._on_pick_failed)
        self._bridge.liked_ready.connect(self._sync_pick_chips)
        self._bridge.veto_built.connect(self._on_veto_built)
        self._bridge.veto_card.connect(self._on_veto_card)
        self._bridge.veto_failed.connect(self._on_veto_failed)
        self._bridge.veto_morning.connect(self._on_veto_morning)
        # P2 pick assessments: the pack builder, the narration call and the Focus reader are injectable.
        self._pick_builder = pick_builder
        self._assess_request = assess_request
        self._focus_source = focus_source
        self._liked_source = liked_source
        self.assess_wait_ms = ASSESS_WAIT_MS
        self._assess_tokens: dict[str, object] = {}
        #: The block the streaming reply writes to; pick cards and notes never move it.
        self._stream_index: int | None = None
        self._pick_cards: dict[str, Any] = {}
        self._pick_packs: dict[tuple[str, str], Any] = {}
        self._pick_blocks: dict[str, int] = {}
        self._pick_schedule = pick_jobs.PickSchedule()
        # P3 veto challenges: the pack builder, the wording call and the cohort outcomes are injectable.
        self._veto_builder = veto_builder
        self._challenge_request = challenge_request
        self._veto_outcomes = veto_outcomes
        self._veto_block: int | None = None
        self._veto_token: object | None = None
        self._veto_schedule = challenge.VetoSchedule()
        #: (session, PT day queued, inbox line, card markdown); held through quiet hours or a mute, never past the day.
        self._veto_inbox_waiting: list[tuple[str, Any, str, str]] = []
        self._inbox_cards: dict[int, str] = {}
        self._graded_on: Any = None
        # P1: with `mentor_app_enabled` on, this process owns the Trade Mentor card.
        self.card_host = card_host
        if self.card_host is None:
            if mentor_enabled is None:
                from ui.services.mentor_launcher import mentor_app_enabled

                mentor_enabled = mentor_app_enabled()
            if mentor_enabled:
                from mentor_app.card_host import AppMentorHost

                self.card_host = AppMentorHost(self)
        self._build_ui()
        self._context_timer = QTimer(self)
        self._context_timer.setInterval(CONTEXT_REFRESH_MS)
        self._context_timer.timeout.connect(self.refresh_context)
        self._gpu_timer = QTimer(self)
        self._gpu_timer.setInterval(GPU_CHECK_MS)
        self._gpu_timer.timeout.connect(self.check_gpu_share)
        self._pick_timer = QTimer(self)
        self._pick_timer.setInterval(PICK_CHECK_MS)
        self._pick_timer.timeout.connect(self.maybe_prefetch_picks)
        self._pick_timer.timeout.connect(self.maybe_veto_card)
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
        self.transcript.setOpenLinks(False)
        self.transcript.anchorClicked.connect(self._on_anchor)
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
        # One chip per Focus name: its pick card (P2).
        self.pick_chips: dict[str, QPushButton] = {}
        pick_strip = QWidget()
        self.pick_chip_row = QHBoxLayout(pick_strip)
        self.pick_chip_row.setContentsMargins(0, 0, 0, 0)
        self.pick_chip_row.setSpacing(4)
        self.pick_more = QLabel("")
        self.pick_more.setObjectName("MutedLabel")
        self.pick_more.setVisible(False)
        self.pick_chip_row.addWidget(self.pick_more)
        self.pick_chip_row.addStretch(1)
        self.pick_scroll = QScrollArea()
        self.pick_scroll.setWidget(pick_strip)
        self.pick_scroll.setWidgetResizable(True)
        self.pick_scroll.setFixedHeight(38)
        self.pick_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pick_scroll.setVisible(False)
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
        self.read_button = QPushButton("Give a read")
        self.read_button.setToolTip("Write a market read now and file it in the Market Journal.")
        self.read_button.clicked.connect(self.give_a_read)
        self.read_button.setVisible(self.card_host is not None)
        buttons.addWidget(self.read_button)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input, 1)
        input_row.addLayout(buttons)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.banner)
        left_layout.addWidget(self.transcript, 3)
        if self.card_host is not None:
            # The Trade Mentor card sits in the conversation column, under the chat.
            left_layout.addWidget(self.card_host.dock, 2)
            self.card_host.cardShown.connect(self._on_card_shown)
            self.card_host.dock.mentor_card.set_ai_request_provider(self._brain_fill_request)
        left_layout.addLayout(self.chip_row)
        left_layout.addWidget(self.pick_scroll)
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
        if self.card_host is not None:
            self.card_host.start()
        self._context_timer.start()
        self._gpu_timer.start()
        self._pick_timer.start()
        self.install_recall_fallback()
        self._submit_io(self._open_session)
        self.refresh_context()
        self.connect_brain()

    def shutdown(self) -> None:
        if self._shut:
            return
        self._shut = True
        self._context_timer.stop()
        self._gpu_timer.stop()
        self._pick_timer.stop()
        from mentor_packs import recall

        recall.set_fallback(None)
        recall.set_searcher(None)
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
        self.queue.stop()
        if self.card_host is not None:
            self.card_host.shutdown()
        if self._endpoint:
            # Hand both models back before the tunnel closes; a dead host costs at most the timeout.
            endpoint, model = self._endpoint, self._model
            unloader = threading.Thread(
                target=lambda: self._unload(endpoint, model, timeout=SHUTDOWN_UNLOAD_SECONDS),
                name="mentor-unload-exit", daemon=True,
            )
            unloader.start()
            unloader.join(SHUTDOWN_UNLOAD_SECONDS + 1)
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
        self._load_memory()

    def _load_memory(self) -> None:
        """IO thread: the night digests and live notes, rendered once into the byte-stable block."""
        self._bridge.memory_ready.emit(memory.load(self.store, ai_root=self._memory_root))

    def _on_memory(self, loaded: Any) -> None:
        self._memory = loaded
        self._memory_block = loaded.text
        self._memory_text = ""  # notes kept this session are in the new block now
        if self._brain_ok:
            self._queue_memory_embeddings()

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
            single = int(getattr(status, "slots", 2) or 2) == 1
            if single:
                # The night left Ollama running with one slot: background jobs yield fully.
                logging.info("Trade Mentor: Ollama already running: 1 slot, night-started")
            self.queue.set_single_slot(single)
            brain.warm(state["endpoint"], state["model"], settings.keep_alive(), post=self._post)
            self._install_recall(state["endpoint"])
            state["ok"] = True
        except Exception as exc:  # noqa: BLE001 - any failure is "brain off", with the reason
            state["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            self._bridge.brain_state.emit(state)

    def install_recall_fallback(self) -> None:
        """With the brain down, /recall and the recall tool still answer by plain substring."""
        from mentor_packs import recall

        recall.set_fallback(lambda query, k: memory.substring_search(self.store, self._memory, query, k))

    def _install_recall(self, endpoint: str) -> None:
        from mentor_packs import recall

        def rows() -> list[dict]:
            retired = {int(row["id"]) for row in self.store.profile_notes(limit=100_000, include_retired=True)
                       if row.get("retired_utc")}
            return [row for row in self.store.embeddings(settings.EMBED_MODEL)
                    if not (row["kind"] == "note" and int(row["ref_id"]) in retired)]

        recall.set_searcher(
            recall.make_searcher(
                lambda texts: brain.embed(endpoint, texts, model=settings.EMBED_MODEL, post=self._post),
                rows,
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
        if self._brain_ok:
            self._queue_memory_embeddings()

    def _bump_stats(self, **counts: float) -> None:
        """This PT day's service counters (the night's mentor_review reads them)."""
        day = self._now().astimezone(challenge.PT).date().isoformat()
        self._submit_io(lambda: self.store.bump_day_stats(day, **counts))

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
            # Outside the night's hours a down brain is an outage minute (the night hand-back is not).
            self._bump_stats(brain_offline_min=GPU_CHECK_MS / 60_000)
            if time.monotonic() - self._last_connect >= RECONNECT_BACKOFF_SECONDS or self._last_connect == 0:
                self.connect_brain()

    def _unload(self, endpoint: str, model: str, timeout: float = 60) -> None:
        """Unload the chat model and the embedder (keep_alive 0)."""
        for name, unload in ((model, brain.unload), (settings.EMBED_MODEL, brain.unload_embedder)):
            if not name:
                continue
            try:
                unload(endpoint, name, post=self._post, timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Trade Mentor: unload of %s failed: %s", name, exc)

    # ------------------------------------------------------------------ context
    def refresh_context(self) -> None:
        def build():
            from mentor_packs import context_pack

            pack = context_pack.build()
            self.store.put_pack(pack.name, {}, pack.as_json(), pack.built_utc)
            return pack

        self.queue.submit("context_pack", build, priority=PRIORITY_REFRESH, key="context_pack",
                          on_done=self._bridge.context_ready.emit)
        self.refresh_liked()

    def _liked(self) -> list[tuple[str, str]]:
        if self._liked_source is not None:
            return list(self._liked_source())
        return pick_jobs.liked_picks(today=self._now().astimezone(pick_jobs.PT).date())

    def refresh_liked(self) -> None:
        """The chips follow the trader's liked picks (claims, likes, swing favourites), read off-thread."""
        self.queue.submit("liked_picks", self._liked, priority=PRIORITY_REFRESH, key="liked_picks",
                          on_done=self._bridge.liked_ready.emit)

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

    # ------------------------------------------------------------------ Trade Mentor card
    def give_a_read(self) -> None:
        if self.card_host is not None:
            self.card_host.give_a_read()

    def _on_card_shown(self, slot: Any) -> None:
        """A due card is up under the chat; the Inbox gets a quiet line (cap and quiet hours apply)."""
        at = getattr(slot, "scheduled_at", None)
        when = f" {at:%H:%M}" if hasattr(at, "strftime") else ""
        self.post_to_inbox("question", f"Trade Mentor{when}: questions are waiting under the chat.")

    def _brain_fill_request(self) -> Callable[..., Any] | None:
        """The card's AI fill goes to the 5080 while the brain is up; None = the local path."""
        if not self._brain_ok or not self._endpoint:
            return None
        endpoint, model = self._endpoint, self._model

        def request(**kwargs: Any) -> Any:
            import ai_summary

            kwargs["model"] = model or kwargs.get("model")
            return ai_summary.request_ai_summary(endpoint=f"{endpoint}/v1", **kwargs)

        return request

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
                card = self._inbox_cards.get(item_id)
                if card:
                    self._add_block(card)
                else:
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
        self._maybe_still_true()
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
            context_text=self._context_text, budget_tokens=settings.context_tokens(), memory_text=self._memory_text,
            memory_block=self._memory_block,
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
        self._stream_index = len(self._blocks)
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
        elif result.action == "forget":
            note_id = int(result.arg)

            def forget() -> None:
                if self.store.retire_note(note_id):
                    self._bridge.note.emit(f"Retired [mem:note:{note_id}]. It is kept, never deleted.")
                    self._load_memory()
                else:
                    self._bridge.note.emit(f"There is no note {note_id}. `/memory` shows the ids.")

            self._submit_io(forget)
        elif result.action == "keep":
            note_id = int(result.arg)
            self._submit_io(lambda: self._bridge.note.emit(
                f"Still true: [mem:note:{note_id}]. I will ask again in {memory.STILL_TRUE_DAYS} days."
                if self.store.check_note(note_id) else f"There is no note {note_id}. `/memory` shows the ids."))
        elif result.action == "memory":
            self._add_note(memory.as_listing(self._memory))
        elif result.action == "recall":
            from mentor_packs import recall

            query = str(result.arg)
            self.queue.submit("recall", lambda: recall.build(query).as_text().replace("\n", "\n\n"),
                              priority=PRIORITY_INTERACTIVE, key=f"recall:{query}", on_done=self._bridge.note.emit)
        elif result.action in ("read", "pause") and self.card_host is None:
            self._add_note("The Trade Mentor questions are on the desk (`mentor_app_enabled` is off).")
        elif result.action == "read":
            self.give_a_read()
        elif result.action == "pause":
            self.card_host.pause_today()
            self._add_note("No more Trade Mentor questions today.")
        elif result.action == "pick":
            symbol, side = result.arg
            self.show_pick(symbol, side)
        elif result.action == "vetoes":
            self.show_vetoes(str(result.arg or ""))
        elif result.action == "scorecard":
            self.queue.submit("scorecard", lambda: challenge.scorecard(self.store), priority=PRIORITY_INTERACTIVE,
                              key="scorecard", on_done=self._bridge.note.emit)
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

    def _stream_slot(self) -> int:
        index = self._stream_index
        if index is None or index >= len(self._blocks):
            self._stream_index = index = len(self._blocks) - 1
        return index

    def _on_token(self, text: str) -> None:
        if self._stream_index is None:
            return  # the turn already ended: a late token from its stream is dropped
        index = self._stream_slot()
        self._blocks[index] += text
        if index != len(self._blocks) - 1:
            self._render()  # a card or note came after the reply: repaint in place
            return
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.transcript.setTextCursor(cursor)

    def _on_tool_call(self, call: dict) -> None:
        # A quiet label in the window's own status bar, cleared when the turn ends.
        self.activity_label.setText(f"reading {call.get('name')}...")

    def _finish_turn(self) -> None:
        self._worker = None
        self._stream_index = None
        self.activity_label.setText("")
        self.queue.end_interactive()
        self.stop_button.setEnabled(False)
        self.send_button.setEnabled(True)

    def _on_done(self, result: dict) -> None:
        text = str(result.get("text") or "")
        if result.get("cancelled"):
            text += " *(stopped)*"
        # Guardrail 2: a number no pack sent this turn is grey, never hidden.
        shown = grounding.mark_uncited_numbers(text, [self._context_text, *(result.get("pack_texts") or ())])
        numbers = grounding.count_numbers(text)
        if numbers:
            self._bump_stats(numbers=numbers, uncited_numbers=shown.count(f'class="{grounding.UNCITED_CLASS}"'))
        self._blocks[self._stream_slot()] = f"**Mentor:** {shown}"
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
        self._blocks[self._stream_slot()] = f"**Mentor:** *(failed: {message})*"
        self._render()
        self._brain_reason = message
        self._finish_turn()

    # ------------------------------------------------------------------ memory (P4)
    def _maybe_still_true(self) -> None:
        """First turn of a session day: look for ONE old ``rule:`` note to re-check (IO thread)."""
        now = self._now()
        day = now.astimezone(challenge.PT).date()
        if self._still_true_day == day:
            return
        self._still_true_day = day
        if not challenge.is_session_day(now):
            return
        self._submit_io(lambda: self._bridge.still_true.emit(
            memory.still_true_candidate(self.store.profile_notes(limit=memory.NOTE_LIMIT), self._now())))

    def _on_still_true(self, row: Any) -> None:
        """Post the "still true?" item to the Inbox when it may land; marked asked only once it did."""
        if not row:
            return
        item = self.inbox.add("memory", memory.still_true_text(row))
        if item is None:
            logging.info("Trade Mentor: the still-true question waits (%s)", self.inbox.last_refusal)
            return
        note_id = int(row["id"])
        self._submit_io(lambda: self.store.mark_note_asked(note_id))
        self.refresh_inbox()

    def _queue_memory_embeddings(self) -> None:
        """Idle priority: embed night digests and notes the recall search has not seen yet."""
        endpoint, items = self._endpoint, list(self._memory.items)

        def job() -> int:
            done = 0
            have = self.store.embedded_refs("digest", settings.EMBED_MODEL)
            todo = [(item.kind, item.ref_id, item.text) for item in items if item.kind == "digest" and item.ref_id not in have]
            todo += [("note", int(row["id"]), str(row["text"])) for row in self.store.unembedded_notes(settings.EMBED_MODEL)]
            for kind, ref_id, text in todo:
                if self.queue.should_yield():
                    break
                vectors = brain.embed(endpoint, [text], model=settings.EMBED_MODEL, post=self._post)
                if vectors:
                    self.store.put_embedding(kind, ref_id, settings.EMBED_MODEL, vectors[0], text=text[:2000])
                    done += 1
            return done

        self.queue.submit("embed_memory", job, priority=PRIORITY_IDLE, needs_model=True, key="embed_memory")

    def _store_turn(self, role: str, text: str, **kwargs: Any) -> None:
        self._submit_io(lambda: self.store.add_turn(self._session_id, role, text, **kwargs))

    def _queue_embeddings(self) -> None:
        endpoint = self._endpoint

        def job() -> int:
            done = 0
            for row in self.store.unembedded_turns(settings.EMBED_MODEL, limit=20):
                if self.queue.should_yield():
                    break  # one slot: the chat turn goes first; the rest embed after it
                vectors = brain.embed(endpoint, [row["text"]], model=settings.EMBED_MODEL, post=self._post)
                if vectors:
                    self.store.put_embedding("turn", row["id"], settings.EMBED_MODEL, vectors[0], text=row["text"][:2000])
                    done += 1
            return done

        # Waits for the IO thread's pending turn writes first, then embeds on the queue.
        self._submit_io(
            lambda: self.queue.submit("embed_turns", job, priority=PRIORITY_EMBED, needs_model=True, key="embed_turns")
        )

    # ------------------------------------------------------------------ picks (P2)
    def _build_pick(self, symbol: str, side: str) -> Any:
        if self._pick_builder is not None:
            return self._pick_builder(symbol, side)
        from mentor_packs import pick_pack

        return pick_pack.build(symbol, side)

    def _narrator(self, symbol: str, *, live: bool, effort: str | None = None) -> Callable[[Any, str], Any]:
        endpoint, model, request = self._endpoint, self._model, self._assess_request

        def narrate(pack: Any, digest: str) -> Any:
            return pick_assess.assess(
                pack, symbol=symbol, pack_hash=digest, model=model, endpoint=endpoint, live=live,
                effort=effort, request=request,
            )

        return narrate

    def _sync_pick_chips(self, liked: Any) -> None:
        """One chip per liked pick, newest first, at most MAX_CHIPS; the rest are a `/pick` away."""
        everything = [(str(sym).upper(), str(side or "").upper()) for sym, side in liked or ()]
        wanted = everything[: pick_jobs.MAX_CHIPS]
        extra = len(everything) - len(wanted)
        self.pick_more.setText(f"+{extra} more: /pick SYM" if extra > 0 else "")
        self.pick_more.setVisible(extra > 0)
        if [(sym, chip.property("side")) for sym, chip in self.pick_chips.items()] == wanted:
            return
        for chip in self.pick_chips.values():
            self.pick_chip_row.removeWidget(chip)
            chip.deleteLater()
        self.pick_chips = {}
        for index, (symbol, side) in enumerate(wanted):
            chip = QPushButton(f"{symbol} {side[:1]}".strip())
            chip.setObjectName("MentorPickChip")
            chip.setFlat(True)
            chip.setProperty("side", side)
            chip.setToolTip(f"{symbol} {side.lower()}: what the desk knows, narrated")
            chip.clicked.connect(lambda _=False, sym=symbol, sd=side: self.show_pick(sym, sd))
            self.pick_chip_row.insertWidget(index, chip)
            self.pick_chips[symbol] = chip
        self.pick_scroll.setVisible(bool(wanted))

    def show_pick(self, symbol: str, side: str = "") -> None:
        """The pick's card: the cached one at once, then a rebuild off-thread; narrate if the pack changed."""
        symbol = str(symbol or "").upper()
        if symbol in self._pick_blocks:
            self.activity_label.setText(f"{symbol} is still being built...")
            return  # one card per pick at a time; a second click never strands a placeholder
        self._pick_blocks[symbol] = len(self._blocks)
        cached = self._pick_cards.get(symbol)
        if cached is not None:
            self._add_block(pick_assess.card_markdown(cached, side=side))
            self.activity_label.setText(f"checking {symbol} for changes...")
        else:
            self._add_block(f"**Pick {symbol}**: building...")
            self.activity_label.setText(f"building {symbol}...")

        def job() -> dict:
            from mentor_packs import pick_pack

            pack = self._build_pick(symbol, side)
            digest = pick_pack.pack_hash(pack)
            found = pick_jobs.cached_assessment(self.store, symbol, digest)
            return {"symbol": symbol, "side": side, "pack": pack, "hash": digest, "assessment": found}

        self.queue.submit(f"pick_pack {symbol}", job, priority=PRIORITY_INTERACTIVE, key=f"pick-live:{symbol}",
                          on_done=self._bridge.pick_built.emit,
                          on_error=lambda exc: self._bridge.pick_failed.emit((symbol, exc)))

    def _on_pick_built(self, built: dict) -> None:
        symbol, digest = built["symbol"], built["hash"]
        self._pick_packs[(symbol, digest)] = built["pack"]
        if symbol not in self._pick_blocks:
            return  # a prefetched card already answered this click
        found = built.get("assessment")
        shown = self._pick_cards.get(symbol)
        if shown is not None and shown.pack_hash == digest:
            # The card already on screen still fits the evidence.
            self._pick_blocks.pop(symbol, None)
            self.activity_label.setText("")
            return
        if found is not None and found.narrated:
            self._show_pick_card(built)
            return
        if not self._brain_ok:
            offline = pick_assess.Assessment(
                symbol=symbol, pack_hash=digest, pack_json=built["pack"].as_json(),
                error=f"the brain is off: {self._brain_reason or 'not connected'}",
            )
            self._show_pick_card({**built, "assessment": offline})
            return
        if shown is not None:
            self._add_note(f"{symbol}'s evidence changed since that card; re-assessing.")
            self._pick_blocks[symbol] = len(self._blocks)
            self._add_block(f"**Pick {symbol}**: narrating...")
        else:
            self._replace_pick_block(symbol, f"**Pick {symbol}**: narrating...")
        self.activity_label.setText(f"narrating {symbol}...")
        narrate = self._narrator(symbol, live=True)
        pack, side = built["pack"], built["side"]

        def job() -> dict:
            assessment = narrate(pack, digest)
            if assessment.narrated:
                self.store.put_pack(pick_assess.CACHE_NAME, pick_jobs.cache_key(symbol, digest),
                                    assessment.to_json(), assessment.built_utc)
            return {"symbol": symbol, "side": side, "pack": pack, "hash": digest, "assessment": assessment,
                    "source": "live"}

        self.queue.submit(f"pick_assess {symbol}", job, priority=PRIORITY_INTERACTIVE, needs_model=True,
                          max_tokens=pick_assess.MAX_OUTPUT_TOKENS, key=f"pick-assess:{symbol}",
                          on_done=self._bridge.pick_card.emit,
                          on_error=lambda exc: self._bridge.pick_failed.emit((symbol, exc)))
        token = self._assess_tokens[symbol] = object()
        QTimer.singleShot(self.assess_wait_ms, self, lambda: self._assess_deadline(symbol, built, token))

    def _assess_deadline(self, symbol: str, built: dict, token: object | None = None) -> bool:
        """The narration never got the model in time: show the evidence alone and free the symbol."""
        if token is not None and self._assess_tokens.get(symbol) is not token:
            return False
        if not self.queue.cancel(f"pick-assess:{symbol}"):
            return False  # it started (or finished); its own result arrives
        self._assess_tokens.pop(symbol, None)
        busy = pick_assess.Assessment(
            symbol=symbol, pack_hash=built["hash"], pack_json=built["pack"].as_json(),
            error="the brain is busy or off; the model was not free in time",
        )
        self._show_pick_card({**built, "assessment": busy})
        return True

    def _on_pick_card(self, done: dict) -> None:
        """A live or prefetched narration finished."""
        assessment = done.get("assessment")
        if assessment is None:
            return
        symbol = done["symbol"]
        self._pick_packs[(symbol, done["hash"])] = done["pack"]
        if done.get("source") == "prefetch" and not assessment.narrated:
            return  # a failed background narration is never shown as a card
        if symbol in self._pick_blocks:
            if done.get("source") == "prefetch":
                # The background card answers the click: the live job (if still waiting) is dropped.
                self.queue.cancel(f"pick-assess:{symbol}")
                self.queue.cancel(f"pick-live:{symbol}")
            self._assess_tokens.pop(symbol, None)
            self._show_pick_card(done)
        elif assessment.narrated:
            # A prefetched card waits in memory; the transcript only moves when the trader asks.
            self._pick_cards[done["symbol"]] = assessment

    def _on_pick_failed(self, failed: Any) -> None:
        symbol, exc = failed
        self._replace_pick_block(symbol, f"**Pick {symbol}**: could not be built ({type(exc).__name__}: {exc}).")
        self._pick_blocks.pop(symbol, None)
        self.activity_label.setText("")

    def _replace_pick_block(self, symbol: str, markdown: str) -> None:
        index = self._pick_blocks.get(symbol)
        if index is not None and index < len(self._blocks):
            self._blocks[index] = markdown
            self._render()
        else:
            self._pick_blocks[symbol] = len(self._blocks)
            self._add_block(markdown)

    def _show_pick_card(self, done: dict) -> None:
        symbol, assessment = done["symbol"], done["assessment"]
        card = pick_assess.card_markdown(assessment, side=done.get("side") or "")
        self._replace_pick_block(symbol, card)
        self._pick_blocks.pop(symbol, None)
        self.activity_label.setText("")
        if assessment.narrated:
            self._pick_cards[symbol] = assessment
        # The card joins the conversation, so a follow-up question sees it (pick_pack stays a tool).
        self.chat.add("assistant", card)
        pack = done.get("pack")
        self._store_turn(
            "assistant", card, pack_ids=getattr(pack, "ids", ()), model=assessment.model or "",
            tool_calls=[{"name": "pick_pack", "arguments": {"symbol": symbol}, "hash": done["hash"],
                         "effort": assessment.effort, "dropped": assessment.dropped, "error": assessment.error}],
        )

    def _on_anchor(self, url: QUrl) -> None:
        text = url.toString()
        if not text.startswith("evidence:"):
            return
        _, symbol, digest = (text.split(":", 2) + ["", ""])[:3]
        pack = self._pick_packs.get((symbol, digest))
        if pack is None:
            card = self._pick_cards.get(symbol)
            pack = card.pack() if card is not None and card.pack_hash == digest else None
        if pack is None:
            self._add_note(f"The evidence for {symbol} is gone; try `/pick {symbol}` again.")
            return
        self._add_note(pack.as_text().replace("\n", "\n\n"))

    def maybe_prefetch_picks(self) -> None:
        """06:15 PT: the liked picks (then, scope `all`, the rest of Focus when idle); hourly: changed packs only."""
        now = self._now()
        kind = self._pick_schedule.due(now)
        if not kind or not self._brain_ok or self._gpu_reason():
            return
        self._pick_schedule.mark(kind, now)
        source = self._focus_source
        if source is None:
            from mentor_packs import context_pack

            source = context_pack._live_focus

        scope = settings.prefetch_scope()

        def plan() -> int:
            from mentor_packs import pick_pack

            liked = self._liked()
            # The chip set (newest liked picks) at high effort; older liked picks only when idle, at low.
            names = [
                (sym, side, PRIORITY_REFRESH, pick_assess.EFFORT_PREFETCH) if index < pick_jobs.MAX_CHIPS
                else (sym, side, PRIORITY_IDLE, pick_assess.EFFORT_IDLE)
                for index, (sym, side) in enumerate(liked)
            ]
            if scope == "all":
                mine = {sym for sym, _ in liked}
                names += [
                    (sym, side, PRIORITY_IDLE, pick_assess.EFFORT_IDLE)
                    for sym, side in pick_jobs.focus_names(source()) if sym not in mine
                ]
            for symbol, side, priority, effort in names:
                narrate = self._narrator(symbol, live=False, effort=effort)

                def run(symbol=symbol, side=side, narrate=narrate) -> dict:
                    result = pick_jobs.run_pick_job(
                        symbol, side, kind=kind, store=self.store, build_pack=self._build_pick,
                        pack_hash=pick_pack.pack_hash, narrate=narrate, now=self._now,
                        should_yield=self.queue.should_yield,
                    )
                    return {**result, "source": "prefetch"}

                self.queue.submit(f"pick_prefetch {symbol}", run, priority=priority, needs_model=True,
                                  max_tokens=pick_assess.MAX_OUTPUT_TOKENS, key=f"pick-prefetch:{symbol}",
                                  on_done=self._bridge.pick_card.emit)
            return len(names)

        self.queue.submit("pick_prefetch_plan", plan, priority=PRIORITY_REFRESH, key="pick_prefetch_plan")

    # ------------------------------------------------------------------ vetoes (P3)
    def _build_veto(self, day: str) -> Any:
        if self._veto_builder is not None:
            return self._veto_builder(day)
        from mentor_packs import veto_pack

        return veto_pack.build(day)

    def _veto_build_job(self, day: str, source: str) -> Callable[[], dict]:
        """Build the pack; reuse the cached card for (session, hash); a card with no candidate needs no model."""

        def job() -> dict:
            from mentor_packs import veto_pack

            pack = self._build_veto(day)
            digest = veto_pack.pack_hash(pack)
            session = next((str(row["id"]).split(":")[1] for row in pack.rows if row.get("kind") == "summary"), "")
            key = {"date": session, "hash": digest}
            card = None
            row = self.store.get_pack(challenge.CACHE_NAME, key) if session else None
            if row:
                try:
                    card = challenge.VetoCard.from_json(str(row["pack_json"]))
                except (ValueError, TypeError, KeyError):
                    card = None
            if card is None and session and not challenge.candidates(pack):
                card = challenge.word(pack, pack_hash=digest, model="", endpoint="", now=self._now)
                self.store.put_pack(challenge.CACHE_NAME, key, card.to_json(), card.built_utc)
            events = sum(1 for r in pack.rows if r.get("kind") in ("veto", "pass"))
            posted = self.store.get_state(challenge.MORNING_POSTED_KEY) if source == "morning" else None
            return {"pack": pack, "hash": digest, "session": session, "card": card, "events": events, "source": source,
                    "posted": posted}

        return job

    def _veto_word_job(self, built: dict, source: str) -> Callable[[], dict]:
        endpoint, model, request = self._endpoint, self._model, self._challenge_request

        def job() -> dict:
            card = challenge.word(built["pack"], pack_hash=built["hash"], model=model, endpoint=endpoint,
                                  request=request, now=self._now)
            if card.done:
                self.store.put_pack(challenge.CACHE_NAME, {"date": built["session"], "hash": built["hash"]},
                                    card.to_json(), card.built_utc)
                challenge.record(self.store, card)
            return {**built, "card": card, "source": source}

        return job

    def show_vetoes(self, day: str = "") -> None:
        """The /vetoes card: the cached one when the pack is unchanged, else built and worded off-thread."""
        if self._veto_block is not None:
            self.activity_label.setText("the vetoes card is still being built...")
            return
        self._veto_block = len(self._blocks)
        self._add_block("**Vetoes**: building...")
        self.activity_label.setText("building the vetoes card...")
        self.queue.submit("veto_pack", self._veto_build_job(day, "live"), priority=PRIORITY_INTERACTIVE,
                          key="vetoes-live", on_done=self._bridge.veto_built.emit,
                          on_error=self._bridge.veto_failed.emit)

    def _replace_veto_block(self, markdown: str) -> None:
        index = self._veto_block
        if index is not None and index < len(self._blocks):
            self._blocks[index] = markdown
            self._render()
        else:
            self._add_block(markdown)

    def _on_veto_built(self, built: dict) -> None:
        pack, card = built["pack"], built.get("card")
        if not built.get("session"):
            self._finish_veto(f"**Vetoes**: {pack.empty_text or 'nothing to show'}")
            return
        if card is not None and card.done:
            self._show_veto_card(built)
            return
        if not self._brain_ok:
            offline = challenge.VetoCard(session=built["session"], pack_hash=built["hash"], pack_json=pack.as_json(),
                                         error=f"the brain is off: {self._brain_reason or 'not connected'}")
            self._show_veto_card({**built, "card": offline})
            return
        self._replace_veto_block("**Vetoes**: wording the challenges...")
        self.queue.submit("veto_challenges", self._veto_word_job(built, "live"), priority=PRIORITY_INTERACTIVE,
                          needs_model=True, max_tokens=challenge.MAX_OUTPUT_TOKENS, key="vetoes-word",
                          on_done=self._bridge.veto_card.emit, on_error=self._bridge.veto_failed.emit)
        token = self._veto_token = object()
        QTimer.singleShot(self.assess_wait_ms, self, lambda: self._veto_deadline(built, token))

    def _veto_deadline(self, built: dict, token: object) -> bool:
        """The wording never got the model in time: show the slices alone."""
        if self._veto_token is not token or not self.queue.cancel("vetoes-word"):
            return False
        busy = challenge.VetoCard(session=built["session"], pack_hash=built["hash"], pack_json=built["pack"].as_json(),
                                  error="the brain is busy or off; the model was not free in time")
        self._show_veto_card({**built, "card": busy})
        return True

    def _on_veto_card(self, done: dict) -> None:
        if done.get("source") == "morning":
            if done["card"].done:
                self._queue_veto_inbox(done["card"])
            return  # a failed morning wording never posts; /vetoes still answers
        self._show_veto_card(done)

    def _show_veto_card(self, done: dict) -> None:
        card = done["card"]
        markdown = challenge.card_markdown(card)
        self._finish_veto(markdown)
        # The card joins the conversation, so a follow-up question sees it (veto_pack stays a tool).
        self.chat.add("assistant", markdown)
        pack = done.get("pack")
        self._store_turn(
            "assistant", markdown, pack_ids=getattr(pack, "ids", ()), model=card.model or "",
            tool_calls=[{"name": "veto_pack", "arguments": {"date": card.session}, "hash": card.pack_hash,
                         "dropped": card.dropped, "error": card.error}],
        )

    def _finish_veto(self, markdown: str) -> None:
        self._replace_veto_block(markdown)
        self._veto_block = None
        self._veto_token = None
        self.activity_label.setText("")

    def _on_veto_failed(self, exc: Any) -> None:
        self._finish_veto(f"**Vetoes**: could not be built ({type(exc).__name__}: {exc}).")

    # ------------------------------------------------------------------ the 06:45 card and the daily grading
    def maybe_veto_card(self) -> None:
        """Every minute: deliver a waiting card, grade once a day, and from 06:45 PT build the morning card."""
        now = self._now()
        self._deliver_veto_inbox()
        local_day = now.astimezone(challenge.PT).date()
        # 22:00-06:00 PT the night's mentor_review grades; the app waits for 06:00 (one writer at a time).
        if self._graded_on != local_day and not challenge.night_owns_grading(now):
            self._graded_on = local_day
            self.queue.submit(
                "grade_challenges",
                lambda: 0 if challenge.night_owns_grading(self._now())
                else challenge.grade_open(self.store, self._now(), veto_outcomes=self._veto_outcomes),
                priority=PRIORITY_IDLE, key="grade_challenges",
            )
        if not self._veto_schedule.due(now):
            return
        self._veto_schedule.mark(now)
        if not challenge.is_session_day(now):
            return  # weekends and holidays: no morning card
        self.queue.submit("veto_morning_pack", self._veto_build_job("", "morning"), priority=PRIORITY_REFRESH,
                          key="veto-morning", on_done=self._bridge.veto_morning.emit)

    def _on_veto_morning(self, built: dict) -> None:
        if not built.get("session") or not built.get("events"):
            return  # no vetoes or passes last session: nothing to say
        if built.get("posted") == built["session"]:
            return  # this session's card was posted before a restart
        card = built.get("card")
        if card is not None and card.done:
            self._queue_veto_inbox(card)
            return
        self.queue.submit("veto_morning_word", self._veto_word_job(built, "morning"), priority=PRIORITY_REFRESH,
                          needs_model=True, max_tokens=challenge.MAX_OUTPUT_TOKENS, key="veto-morning-word",
                          on_done=self._bridge.veto_card.emit)

    def _queue_veto_inbox(self, card: Any) -> None:
        day = self._now().astimezone(challenge.PT).date()
        self._veto_inbox_waiting.append((card.session, day, challenge.inbox_line(card), challenge.card_markdown(card)))
        self._deliver_veto_inbox()

    def _deliver_veto_inbox(self) -> None:
        """Post the waiting morning card once quiet hours or a mute end; a used daily cap or a new day drops it."""
        today = self._now().astimezone(challenge.PT).date()
        while self._veto_inbox_waiting:
            session, day, line, markdown = self._veto_inbox_waiting[0]
            if day != today:
                logging.info("Trade Mentor: the vetoes card for %s was dropped (held past its day)", session)
                self._veto_inbox_waiting.pop(0)
                continue
            if not self.inbox.refusal():
                # The marker is queued before the item shows: a quit between the two never reposts the card.
                self._submit_io(lambda session=session: self.store.set_state(challenge.MORNING_POSTED_KEY, session))
            item = self.inbox.add("vetoes", line)
            if item is None:
                if "cap" in self.inbox.last_refusal:
                    logging.info("Trade Mentor: the vetoes card was dropped (%s)", self.inbox.last_refusal)
                    self._veto_inbox_waiting.pop(0)
                    continue
                return  # quiet hours or muted: try again next minute
            self._veto_inbox_waiting.pop(0)
            self._inbox_cards[item.id] = markdown
            self.refresh_inbox()
