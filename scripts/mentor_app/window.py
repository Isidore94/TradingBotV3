"""MentorWindow: the Trade Mentor chat window. The Qt thread only paints.

Streaming runs on a ``brain.StreamWorker`` QThread; the tunnel, warm-up and unload on
plain threads; pack builds and embeddings on the prefetch queue's thread; every store
write on one IO thread. Results come back through queued signals on ``_Bridge``.
"""

from __future__ import annotations

import json
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

import ai_pause
from mentor_app import assess as pick_assess
from mentor_app import attach, brain, challenge, commands, grounding, memory, pick_jobs, plan_infer, settings, style, tape
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
#: The /check narration's output cap (gate.MAX_OUTPUT_TOKENS).
MAX_GATE_TOKENS = 600
#: How long a /check waits for the Questrade read it queued (then it uses the journal).
BOOK_WAIT_SECONDS = 10.0
GPU_CHECK_MS = 60 * 1000
#: How often the app looks at the Pause AI switch (the desk may flip it).
PAUSE_CHECK_MS = 5 * 1000
#: A desk-launched app looks at the desk's slot this often; this many free checks in a row = the desk closed.
DESK_CHECK_MS = 30 * 1000
DESK_FREE_CHECKS = 2
PICK_CHECK_MS = 60 * 1000
#: How long a live narration may wait for the model before the card shows the evidence alone.
ASSESS_WAIT_MS = 90 * 1000
#: How long after a failed connect the app waits before trying the host again.
RECONNECT_BACKOFF_SECONDS = 10 * 60
#: On close the app waits at most this long for each model unload.
SHUTDOWN_UNLOAD_SECONDS = 4
CHIP_KINDS = ("auto_mode", "d1_env", "regime")
#: A background tape narration that fails is tried once more for the same pack hash, then not again.
TAPE_MAX_FAILURES_PER_HASH = 2


def _desk_slot_is_free() -> bool | None:
    """True when no desk holds its single-instance slot; None when unknown."""
    from single_instance import DESK_LOCK_KEY, slot_is_free

    return slot_is_free(DESK_LOCK_KEY)


def _quit_qt() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.quit()


def latency_card(rows: list[dict]) -> str:
    """`/latency`: the last answers' timings from the turn log (newest last)."""
    import json

    if not rows:
        return "No answers logged yet."
    lines = ["**Last answers** (attach ms · first token ms · total ms · prompt tokens · model tools + app packs)", ""]
    for row in rows:
        try:
            timing = json.loads(row.get("timings_json") or "{}")
        except ValueError:
            timing = {}
        first = timing.get("first_token_ms", row.get("latency_ms"))
        lines.append(
            f"- {str(row.get('ts_utc') or '')[11:19]} UTC {row.get('model') or '?'}: "
            f"{timing.get('attach_ms', '-')} · {'-' if first is None else first} · {timing.get('total_ms', '-')} · "
            f"{timing.get('prompt_tokens', row.get('prompt_tokens') or '-')} · "
            f"{timing.get('tool_calls', '-')} + {timing.get('auto_packs', '-')}"
        )
    return "\n".join(lines)


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
    tape_ready = Signal(object)
    tape_refreshed = Signal(object)
    check_card = Signal(object)
    desk_state = Signal(object)
    news_card = Signal(object)
    book_card = Signal(object)
    mirror_card = Signal(object)
    mirror_week = Signal(object)
    tilt_ready = Signal(object)
    tilt_card = Signal(object)
    debate_card = Signal(object)
    frontier_card = Signal(object)
    frontier_state = Signal(object)
    journal_symbols = Signal(object)
    habit_item = Signal(object)
    routine_ready = Signal(object)


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


def _ibkr_fetch_book(now: Any) -> Any:
    """The real IBKR read (``ibkr_positions.fetch_book``), imported on the news thread."""
    import ibkr_positions

    return ibkr_positions.fetch_book(now)


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
        tape_builder: Callable[[], Any] | None = None,
        tape_request: Callable[..., Any] | None = None,
        push_send: Callable[..., Any] | None = None,
        gate_builder: Callable[..., Any] | None = None,
        gate_request: Callable[..., Any] | None = None,
        follow_desk: bool = False,
        desk_probe: Callable[[], bool | None] | None = None,
        quit_app: Callable[[], Any] | None = None,
        news_fetcher: Any = None,
        news_open_symbols: Callable[[], Any] | None = None,
        news_queue: PrefetchQueue | None = None,
        book_fetch: Callable[..., Any] | None = None,
        book_sources: Callable[[], Any] | None = None,
        ibkr_fetch: Callable[..., Any] | None = None,
        mirror_builder: Callable[[int], Any] | None = None,
        mirror_request: Callable[..., Any] | None = None,
        tilt_builder: Callable[[], Any] | None = None,
        tilt_journal: Any = None,
        habits_sources: Any = None,
        routines_path: Any = None,
        debate_request: Callable[..., Any] | None = None,
        frontier_request: Callable[..., Any] | None = None,
        frontier_post: Callable[..., Any] | None = None,
        frontier_key: Callable[[], str] | None = None,
        plan_path: Any = None,
        journal_path: Any = None,
        pack_builder: Callable[[str, Any], Any] | None = None,
        forecast_service: Any = None,
        paste_prompt: Callable[[], Any] | None = None,
        fund_builder: Callable[[str], Any] | None = None,
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
        #: The model's probed native tool calling (False = the one-shot fallback, also when unknown).
        self._native_tools = False
        #: P13 auto-attach: the liked picks and the journal's names of the last 60 days (the trader's universe).
        self._liked_names: list[tuple[str, str]] = []
        self._journal_symbols: list[str] = []
        #: The journal /today and the universe read (None = the live one, read-only).
        self._journal_path = journal_path
        #: The chat turn's pack builder (None = the registry); tests inject fixtures.
        self._pack_builder = pack_builder
        #: When the Pause AI switch ends, as last seen; None = AI on.
        self._paused_until: datetime | None = None
        self.queue =queue or PrefetchQueue(blocked=self._gpu_reason, model_ready=lambda: self._brain_ok)
        self.inbox = inbox or Inbox(per_day_cap=settings.proactive_per_day(), now=self._now)
        self.chat = ChatModel()
        self._context_pack: Any = None
        self._context_text = ""
        #: P4 memory: the start-of-day block (system prefix, byte-stable) and this session's new notes (tail).
        self._memory_root = memory_root
        #: P11 /hypotheses: the permutation report paths (None = project_paths' live constants, read-only).
        self.permutation_history: Any = None
        self.permutation_report: Any = None
        #: P15a: the night's artifacts (``night_pack.NightPaths``; None = the live ones, read-only).
        self.night_paths: Any = None
        #: P15b: the pasted brief (``fundamentals_pack.FundPaths``) and the recap stores (``recaps_pack.RecapPaths``);
        #: None = the live ones, read-only.
        self.fund_paths: Any = None
        self.recap_paths: Any = None
        #: /paste: the Market Journal writer (None = the desk's shared service) and the dialog (None = Qt's).
        self._forecast_service = forecast_service
        self._paste_prompt = paste_prompt
        #: /tape: today's brief, compact (None = the live fundamentals pack).
        self._fund_builder = fund_builder
        self._memory = memory.Memory()
        self._memory_block = ""
        self._memory_text = ""
        self._still_true_day: Any = None
        #: Plan inference: the plan file (None = the live one) and the newest trader turn already read.
        self._plan_path = plan_path
        self._plan_after = 0
        self._session_id: int | None = None
        self._worker: Any = None
        self._blocks: list[str] = []
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mentor-store")
        self._threads: list[threading.Thread] = []
        self._focus_server = None
        self._shut = False
        self._bridge = _Bridge(self)
        self._bridge.context_ready.connect(self._on_context)
        self._bridge.journal_symbols.connect(self._on_journal_symbols)
        self._bridge.brain_state.connect(self._on_brain_state)
        self._bridge.memory_ready.connect(self._on_memory)
        # P18: the night's habits; one Inbox item a week at most, only for a habit followed by red days.
        self._habits_sources = habits_sources
        self._bridge.habit_item.connect(self._on_habit_item)
        # P18 D: the night's routine table; a bucket's packs are prefetched once when it starts.
        self._routines_path = routines_path
        self._routine_payload: dict = {}
        self._routine_forgotten: list[str] = []
        self._routine_bucket_done = ""
        self._routine_ready: tuple[str, str] | None = None
        self._bridge.routine_ready.connect(self._on_routine_ready)
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
        # P5 tape talk: the regime pack builder, the narration call and the push sender are injectable.
        self._tape_builder = tape_builder
        self._tape_request = tape_request
        self._gate_builder = gate_builder
        self._gate_request = gate_request
        self._check_blocks: dict[int, int] = {}
        self._check_seq = 0
        self._bridge.check_card.connect(self._on_check_card)
        self._push_send = push_send
        #: {pack, hash, card, at_utc} of the last tape build; /tape answers from it while it is fresh.
        self._tape_last: dict[str, Any] | None = None
        self._tape_block: int | None = None
        self._tape_schedule = tape.TapeSchedule()
        self._push_checked_day: Any = None
        #: pack hash -> failed background narrations; cleared when the brain comes back from down.
        self._tape_failures: dict[str, int] = {}
        self._bridge.tape_ready.connect(self._on_tape_ready)
        self._bridge.tape_refreshed.connect(self._on_tape_refreshed)
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
        self._pause_timer = QTimer(self)
        self._pause_timer.setInterval(PAUSE_CHECK_MS)
        self._pause_timer.timeout.connect(self.check_ai_pause)
        # Follow the desk: only a desk-launched app (--follow-desk) exits when the desk closes.
        self._follow_desk = bool(follow_desk)
        self._desk_probe = desk_probe or _desk_slot_is_free
        self._quit_app = quit_app or _quit_qt
        self._desk_free_checks = 0
        self._bridge.desk_state.connect(self._on_desk_state)
        self._desk_timer = QTimer(self)
        self._desk_timer.setInterval(DESK_CHECK_MS)
        self._desk_timer.timeout.connect(self.check_desk)
        self._pick_timer = QTimer(self)
        self._pick_timer.setInterval(PICK_CHECK_MS)
        self._pick_timer.timeout.connect(self.maybe_prefetch_picks)
        self._pick_timer.timeout.connect(self.maybe_veto_card)
        self._pick_timer.timeout.connect(self.maybe_prefetch_tape)
        self._pick_timer.timeout.connect(self.maybe_prefetch_routine)
        self._pick_timer.timeout.connect(self.maybe_push_brief)
        # P7 news: the fetcher and the open-book reader are injectable; no model, never the Inbox.
        from mentor_app import news_jobs
        from news_feed import NewsFetcher

        self._news_fetcher = news_fetcher or NewsFetcher()
        self._news_open_symbols = news_open_symbols
        self._news_named = news_jobs.NamedSymbols()
        #: News fetches are network-bound: their own one-thread queue, so /pick, /check and /news never wait.
        self.news_queue = news_queue or PrefetchQueue(thread_name="mentor-news")
        self._news_seeded = False
        self._news_schedule = news_jobs.NewsSchedule()
        self._news_blocks: dict[int, int] = {}
        self._news_seq = 0
        self._bridge.news_card.connect(self._on_news_card)
        self._pick_timer.timeout.connect(self.maybe_refresh_news)
        # P8 book: Questrade read on demand (/book, /check, 06:20 PT) on the news thread; injectable.
        # P12: then IBKR, over its own short-lived read-only TWS client (id 9155).
        from mentor_app import book_jobs

        self._book_fetch = book_fetch
        self._ibkr_fetch = ibkr_fetch if ibkr_fetch is not None else _ibkr_fetch_book
        self._book_sources = book_sources
        self._book_schedule = book_jobs.BookSchedule()
        self._book_blocks: dict[int, int] = {}
        self._book_seq = 0
        self._book_event: threading.Event | None = None
        self._book_lock = threading.Lock()
        self._bridge.book_card.connect(self._on_book_card)
        self._pick_timer.timeout.connect(self.maybe_fetch_book)
        # P9 mirror: /mirror and the weekly Inbox card; the pack builder and the narration call are injectable.
        from mentor_app import mirror as mirror_mod

        self._mirror_builder = mirror_builder
        self._mirror_request = mirror_request
        self._mirror_blocks: dict[int, int] = {}
        self._mirror_seq = 0
        self._mirror_schedule = mirror_mod.WeeklySchedule()
        #: (ISO week, PT day queued, card markdown); held through quiet hours or a mute, never past the day.
        self._mirror_waiting: list[tuple[str, Any, str]] = []
        self._bridge.mirror_card.connect(self._on_mirror_card)
        self._bridge.mirror_week.connect(self._on_mirror_week)
        self._pick_timer.timeout.connect(self.maybe_mirror_week)
        # P9 tilt watch: every 2 min in the session on the news thread (no model); one Inbox item per 30 min.
        from mentor_app import tilt_watch

        self._tilt_builder = tilt_builder
        self._tilt_journal = tilt_journal
        self._tilt_schedule = tilt_watch.TiltSchedule()
        self._tilt_blocks: dict[int, int] = {}
        self._tilt_seq = 0
        #: Observations waiting for the Inbox (quiet hours, a mute or the 30-min spacing), with their PT day.
        self._tilt_waiting: list[tuple[Any, dict]] = []
        self._tilt_last_post = ""
        #: P15b: closed trades waiting for their one "how did it feel?" item, and the posted items' trades.
        self._feel_waiting: list[tuple[Any, dict]] = []
        self._inbox_feel: dict[int, dict] = {}
        self._feel_asked: set[str] = set()
        # P18 journal mode: `/journal on` (persisted) keeps every message as a journal line.
        self._journal_on = False

        def load_journal_mode() -> None:
            from mentor_app import journal_mode

            self._journal_on = self.store.get_state(journal_mode.MODE_KEY) == "on"

        self._submit_io(load_journal_mode)
        self._bridge.tilt_ready.connect(self._on_tilt_ready)
        self._bridge.tilt_card.connect(self._on_tilt_card)
        self._tilt_timer = QTimer(self)
        self._tilt_timer.setInterval(tilt_watch.CHECK_MS)
        self._tilt_timer.timeout.connect(self.maybe_watch_tilt)
        # P10 debate: two persona calls on one pick pack, on demand; Stop cancels between the calls.
        self._debate_request = debate_request
        self._debate_blocks: dict[int, int] = {}
        self._debate_stops: dict[int, threading.Event] = {}
        self._debate_seq = 0
        self._bridge.debate_card.connect(self._on_debate_card)
        # P11 frontier: metered, off by default, never automatic; one call at a time on its own thread.
        self._frontier_request = frontier_request
        self._frontier_post = frontier_post
        self._frontier_key = frontier_key
        self._frontier_busy = False
        self._frontier_seq = 0
        self._frontier_blocks: dict[int, int] = {}
        self._frontier_state: Any = None
        #: What the last local chat turn read, verbatim, for /think.
        self._last_turn: dict[str, Any] | None = None
        self._bridge.frontier_card.connect(self._on_frontier_card)
        self._bridge.frontier_state.connect(self._on_frontier_state)
        if settings.frontier_enabled():
            self._spawn("mentor-frontier-status", self._frontier_status)
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
        # P18 D: a quiet chip when the usual read for this half hour is ready (no Inbox, no pop).
        self.routine_chip = QPushButton("")
        self.routine_chip.setObjectName("MentorChip")
        self.routine_chip.setFlat(True)
        self.routine_chip.setVisible(False)
        self.routine_chip.clicked.connect(self._show_routine)
        self.chip_row.addWidget(self.routine_chip)
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

        from ui.widgets.ai_pause_control import AiPauseButton

        # The header: Pause AI (frees the GPU host; the questions keep working).
        self.ai_pause_button = AiPauseButton(self, now=self._now)
        self.ai_pause_button.changed.connect(self.check_ai_pause)
        header = QHBoxLayout()
        header.addStretch(1)
        # P15b: paste the morning brief into the Market Journal (the desk's own writer).
        self.paste_button = QPushButton("Paste brief")
        self.paste_button.setToolTip("Paste today's morning brief; it is saved to the Market Journal for today")
        self.paste_button.clicked.connect(self.open_paste_dialog)
        header.addWidget(self.paste_button)
        # P11: shown only while the frontier switch is on; disabled with the reason when it cannot run.
        self.think_button = QPushButton("Think harder")
        self.think_button.setToolTip("Ask the frontier model the last question again (metered, capped per day)")
        self.think_button.clicked.connect(lambda: self.send("/think"))
        self.think_button.setVisible(settings.frontier_enabled())
        header.addWidget(self.think_button)
        header.addWidget(self.ai_pause_button)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(header)
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
        self.think_button.setVisible(settings.frontier_enabled())
        if self._paused_until is not None:
            # Paused is its own state, not "down": the pill and banner say so.
            until = ai_pause.until_text(self._paused_until, self._now())
            self.status_pill.setText(f"AI paused {until}")
            self.banner.setVisible(True)
            self.banner.setText(
                f"AI is paused {until}. Packs and Trade Mentor questions still work. `/ai on` resumes."
            )
            self.ai_pause_button.refresh()
            return
        model = self._model or "model?"
        host = self._host or "no host"
        latency = f"{self._latency_ms} ms" if self._latency_ms is not None else "-"
        state = "ready" if self._brain_ok else "off"
        tools = "native" if self._native_tools else "fallback"
        self.status_pill.setText(f"{host} · {model} · tools: {tools} · {latency} · brain {state}")
        self.banner.setVisible(not self._brain_ok)
        self.banner.setText(f"Brain is off: {self._brain_reason}. Packs still work: try /tape.")
        self.ai_pause_button.refresh()

    # ------------------------------------------------------------------ lifecycle
    def start_background(self) -> None:
        """Start the queue, the timers, the focus listener and the first connect."""
        from mentor_app.focus_link import make_focus_server

        self._focus_server = make_focus_server(self, self.bring_to_front)
        self.queue.start()
        self.news_queue.start()
        if self.card_host is not None:
            self.card_host.start()
        self._context_timer.start()
        self._gpu_timer.start()
        self._pause_timer.start()
        if self._follow_desk:
            self._desk_timer.start()
        self._pick_timer.start()
        self._tilt_timer.start()
        self.install_recall_fallback()
        self._submit_io(self._open_session)
        self.refresh_context()
        self._paused_until = self._ai_paused_until()
        if self._paused_until is not None:
            # Started while paused: no tunnel, no warm-up, no model.
            self._brain_reason = self._pause_reason()
            self._sync_status()
            return
        self.connect_brain()

    def shutdown(self) -> None:
        if self._shut:
            return
        self._shut = True
        self._context_timer.stop()
        self._gpu_timer.stop()
        self._pause_timer.stop()
        self._desk_timer.stop()
        self._pick_timer.stop()
        self._tilt_timer.stop()
        from mentor_packs import recall

        recall.set_fallback(None)
        recall.set_searcher(None)
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
        for stop in self._debate_stops.values():
            stop.set()
        self.queue.stop()
        self.news_queue.stop(timeout=0.5)  # a fetch stuck on the network is a daemon; never wait 10 s for it
        if self.card_host is not None:
            self.card_host.shutdown()
        if self._endpoint and self._paused_until is None:
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

    # ------------------------------------------------------------------ follow the desk
    def check_desk(self) -> None:
        """Every 30 s (desk-launched only): probe the desk's slot off the Qt thread."""
        if not self._follow_desk or self._shut:
            return
        probe, emit = self._desk_probe, self._bridge.desk_state.emit

        def run() -> None:
            try:
                free = probe()
            except Exception:  # noqa: BLE001 - a broken probe is "unknown", never "closed"
                free = None
            emit(free)

        threading.Thread(target=run, name="mentor-desk-probe", daemon=True).start()

    def _on_desk_state(self, free: Any) -> None:
        """Two free checks in a row: the desk closed, so hand the GPU back and exit."""
        self._desk_free_checks = self._desk_free_checks + 1 if free is True else 0
        if not self._follow_desk or self._shut or self._desk_free_checks < DESK_FREE_CHECKS:
            return
        logging.info("desk closed; Trade Mentor following")
        self.shutdown()
        self._quit_app()

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
        self._bridge.memory_ready.emit(memory.load(self.store, ai_root=self._memory_root, night_paths=self.night_paths,
                                                   now=self._now(), fund_paths=self.fund_paths))
        self._check_habit_inbox()
        self._load_routines()

    def _load_routines(self) -> None:
        """IO thread: the night's routine table and the buckets the trader told to forget."""
        from mentor_app import routines

        try:
            path = self._routines_path if self._routines_path is not None else routines.live_path()
            self._routine_payload = routines.read_routines(path)
            self._routine_forgotten = routines.forgotten_list(self.store.get_state(routines.FORGOTTEN_KEY))
        except Exception:  # noqa: BLE001 - no routine file: nothing is prefetched
            logging.warning("Trade Mentor: the routine table could not be read", exc_info=True)

    def maybe_prefetch_routine(self) -> None:
        """Every minute: when a half hour with a routine starts, build its packs once at refresh priority."""
        from mentor_app import routines

        bucket = routines.bucket_of(self._now())
        if bucket == self._routine_bucket_done:
            return
        row = next((r for r in routines.visible(self._routine_payload, self._routine_forgotten)
                    if r.get("bucket") == bucket), None)
        if row is None:
            return
        self._routine_bucket_done = bucket
        names = [str(p["name"]) for p in row.get("packs") or ()]
        if "regime_pack" in names and self._brain_ok:
            self._queue_tape_narration(PRIORITY_REFRESH, "prefetch")  # the tape card already prefetches its words

        def job() -> tuple[str, str]:
            cards = [self._pack_card(name, {}) for name in names]
            return bucket, "\n\n".join(cards)

        self.queue.submit("routine_prefetch", job, priority=PRIORITY_REFRESH, key=f"routine:{bucket}",
                          on_done=self._bridge.routine_ready.emit)

    def _on_routine_ready(self, ready: Any) -> None:
        bucket, markdown = ready
        self._routine_ready = (bucket, markdown)
        self.routine_chip.setText(f"usual {bucket} read ready")
        self.routine_chip.setVisible(True)

    def _show_routine(self) -> None:
        if self._routine_ready is not None:
            self._add_block(self._routine_ready[1])
        self.routine_chip.setVisible(False)

    def _routine_command(self, action: str, bucket: str = "") -> None:
        """``/routine`` prints the table; ``/forget routine <bucket>`` hides a line (persisted)."""
        from mentor_app import routines

        if action == "forget_routine":
            if bucket not in self._routine_forgotten:
                self._routine_forgotten = [*self._routine_forgotten, bucket]
            forgotten = json.dumps(sorted(self._routine_forgotten))
            self._submit_io(lambda: self.store.set_state(routines.FORGOTTEN_KEY, forgotten))
            self._add_note(f"Forgot the {bucket} routine line; I will not get it ready any more.")
            return
        self._add_note(routines.table_text(self._routine_payload, self._routine_forgotten))

    def _check_habit_inbox(self) -> None:
        """IO thread: the week's one habit Inbox item, when the night marked a habit for it."""
        from mentor_packs import habits_pack

        try:
            pack = habits_pack.build(sources=self._habits_sources)
            found = habits_pack.inbox_line(pack, self.store.get_state(habits_pack.INBOX_WEEK_KEY), self._now())
        except Exception:  # noqa: BLE001 - an unreadable habit file posts nothing
            logging.warning("Trade Mentor: the habit Inbox check failed", exc_info=True)
            return
        if found is not None:
            self._bridge.habit_item.emit(found)

    def _on_habit_item(self, found: Any) -> None:
        from mentor_packs import habits_pack

        week, line = found
        if self.inbox.add("habits", line) is not None:
            self._submit_io(lambda: self.store.set_state(habits_pack.INBOX_WEEK_KEY, week))
            self.refresh_inbox()

    def _on_memory(self, loaded: Any) -> None:
        self._memory = loaded
        self._memory_block = loaded.text
        self._memory_text = ""  # notes kept this session are in the new block now
        if self._brain_ok:
            self._queue_memory_embeddings()

    # ------------------------------------------------------------------ brain
    def _gpu_reason(self) -> str:
        try:
            return self._pause_reason() or settings.gpu_block_reason(self._now())
        except Exception as exc:  # noqa: BLE001 - an unreadable clock keeps the model off
            return f"the GPU window could not be read ({type(exc).__name__})"

    # ------------------------------------------------------------------ Pause AI
    def _ai_paused_until(self) -> datetime | None:
        try:
            return ai_pause.paused_until(self._now())
        except Exception:  # noqa: BLE001 - an unreadable switch reads as "not paused", like the helper
            logging.warning("Trade Mentor: the Pause AI setting could not be read.", exc_info=True)
            return None

    def _pause_reason(self) -> str:
        """``AI paused until HH:MM`` while paused, else ""."""
        until = self._ai_paused_until()
        return f"AI paused {ai_pause.until_text(until, self._now())}" if until is not None else ""

    def _offline_why(self) -> str:
        """Why a card shows its evidence without a narration."""
        return self._pause_reason() or f"the brain is off: {self._brain_reason or 'not connected'}"

    def check_ai_pause(self) -> None:
        """Every 5 s (and on the button or `/ai`): act on a change of the Pause AI switch."""
        until = self._ai_paused_until()
        was, self._paused_until = self._paused_until, until
        if until is not None and was is None:
            self._enter_pause()
        elif until is None and was is not None:
            self._leave_pause()
        elif until != was:
            self._sync_status()

    def _enter_pause(self) -> None:
        """Stop the turn, unload both models through the tunnel, then close the tunnel."""
        self._brain_reason = self._pause_reason()
        if self._worker is not None:
            self._worker.cancel()
        from mentor_packs import recall

        recall.set_searcher(None)
        self._brain_ok = False
        endpoint, tunnel = self._endpoint, self._tunnel
        models = self._pause_unload_models()

        def release() -> None:
            if endpoint:
                self._unload(endpoint, models[0], extra=models[1:])
            if tunnel is not None:
                tunnel.stop()

        self._spawn("mentor-pause", release)
        self._sync_status()

    def _pause_unload_models(self) -> tuple[str, ...]:
        """The app's chat model (the setting when none is known yet), then the night's model
        tags: a pause frees the GPU of all of them (unloading an idle model is harmless)."""
        import ai_summary

        names: list[str] = []
        for read in (lambda: self._model or settings.mentor_model(),
                     lambda: ai_summary.local_model("medium"),
                     lambda: ai_summary.local_model("large")):
            try:
                name = str(read() or "").strip()
            except Exception:  # noqa: BLE001 - an unreadable tag is skipped, never fatal
                logging.debug("Trade Mentor: a model tag could not be read.", exc_info=True)
                continue
            if name and name not in names:
                names.append(name)
        return tuple(names) or ("",)

    def _leave_pause(self) -> None:
        """AI is back: the normal start (tunnel, warm-up, prefetch), subject to the night window."""
        self._brain_reason = "connecting to the GPU host..."
        self._last_connect = 0.0
        self._sync_status()
        reason = self._gpu_reason()
        if reason:
            self._brain_reason = reason
            self._sync_status()
        elif not self._connecting:
            self.connect_brain()

    def set_ai_pause(self, choice: Any) -> None:
        """`/ai off ...`: write the switch and act on it now."""
        until = ai_pause.pause_for(choice, self._now())
        self.check_ai_pause()
        self._add_note(
            f"AI paused {ai_pause.until_text(until, self._now())}. The GPU host is free; "
            "packs and Trade Mentor questions still work. `/ai on` resumes."
        )

    def resume_ai(self) -> None:
        ai_pause.resume()
        self.check_ai_pause()
        self._add_note("AI is on again: reconnecting to the GPU host." if not self._gpu_reason()
                       else f"AI is on again, but {self._gpu_reason()}.")

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
            if not settings.explicit_model() and brain.model_present(state["endpoint"], settings.DAY_MODEL,
                                                                     post=self._post):
                state["model"] = settings.DAY_MODEL
            caps = brain.model_capabilities(state["endpoint"], state["model"], post=self._post, store=self.store,
                                            now=self._now())
            state["native_tools"] = brain.native_tools_for(caps)
            single =int(getattr(status, "slots", 2) or 2) == 1
            if single:
                # The night left Ollama running with one slot: background jobs yield fully.
                logging.info("Trade Mentor: Ollama already running: 1 slot, night-started")
            self.queue.set_single_slot(single)
            if self._paused_while_connecting(state, warmed=False):
                return
            brain.warm(state["endpoint"], state["model"], settings.keep_alive(), post=self._post)
            if self._paused_while_connecting(state, warmed=True):
                return
            self._install_recall(state["endpoint"])
            state["ok"] = True
        except Exception as exc:  # noqa: BLE001 - any failure is "brain off", with the reason
            state["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            self._bridge.brain_state.emit(state)

    def _paused_while_connecting(self, state: dict[str, Any], *, warmed: bool) -> bool:
        """Connect thread: AI was paused mid-connect, so hand back what was taken and stop."""
        reason = self._pause_reason()
        if not reason:
            return False
        state["reason"] = reason
        if warmed:
            self._unload(state["endpoint"], state["model"])
        self._tunnel.stop()
        return True

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
        paused = self._pause_reason()
        if state.get("ok") and paused:
            # Paused in the last instant of a connect: hand the model back now.
            endpoint, model, tunnel = str(state.get("endpoint") or ""), str(state.get("model") or ""), self._tunnel

            def release() -> None:
                self._unload(endpoint, model)
                if tunnel is not None:
                    tunnel.stop()

            self._spawn("mentor-pause", release)
            state = {**state, "ok": False, "reason": paused}
        if state.get("ok") and not self._brain_ok:
            self._tape_failures.clear()  # the brain is back: a failed tape may be read again
        self._brain_ok = bool(state.get("ok"))
        self._brain_reason = str(state.get("reason") or "")
        self._host = str(state.get("host") or self._host)
        self._model = str(state.get("model") or self._model)
        if "native_tools" in state:
            self._native_tools = bool(state["native_tools"])
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
        was_paused = self._paused_until is not None
        self.check_ai_pause()
        if was_paused or self._paused_until is not None:
            return  # paused (or just resumed, already reconnecting): not an outage minute
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

    def _unload(self, endpoint: str, model: str, timeout: float = 60, extra: tuple[str, ...] = ()) -> None:
        """Unload the chat model, any ``extra`` chat models, and the embedder (keep_alive 0)."""
        chats = [(name, brain.unload) for name in (model, *extra)]
        for name, unload in (*chats, (settings.EMBED_MODEL, brain.unload_embedder)):
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
        self.queue.submit("journal_symbols", self._read_journal_symbols, priority=PRIORITY_REFRESH,
                          key="journal_symbols", on_done=self._bridge.journal_symbols.emit)

    def _read_journal_symbols(self) -> list[str]:
        """Queue thread: the journal's names of the last 60 days (part of the auto-attach universe)."""
        from mentor_packs import journal_pack

        try:
            return journal_pack.recent_symbols(self._journal_path, now=self._now())
        except Exception as exc:  # noqa: BLE001 - an unreadable journal only narrows the universe
            logging.info("Trade Mentor: journal symbols not read (%s)", exc)
            return []

    def _on_journal_symbols(self, symbols: Any) -> None:
        self._journal_symbols = [str(sym) for sym in symbols or ()]

    def _today_card(self, day: str) -> str:
        """Queue thread: `/today` as a card from the journal pack (ids intact)."""
        from mentor_packs import journal_pack

        pack = journal_pack.build(day, now=self._now(), journal=self._journal_path, chat_db=self.store.path)
        return "**Journal**\n\n" + pack.as_text().replace("\n", "\n\n")

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
                trade = self._inbox_feel.get(item_id)
                if card:
                    self._add_block(card)
                elif trade:
                    # P15b: the answer box starts with the trade's id; his words follow it.
                    self._add_note(f"{item.text} Type it after the id and press Enter.")
                    self.input.setPlainText(f"/feel {trade['trade_id']} ")
                    self.input.moveCursor(QTextCursor.MoveOperation.End)
                    self.input.setFocus()
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
        # P18: self talk is kept as a journal line with a one-line "Noted"; a question inside it is answered too.
        from mentor_app import journal_mode

        kind = journal_mode.classify(text, forced=self._journal_on)
        if kind.statement:
            self.record_journal(text, asks=kind.asks)
            if not kind.asks:
                return
        self._store_turn("user", text)
        paused = self._ai_paused_until()
        if paused is not None:
            self._add_note(f"AI is paused {ai_pause.until_text(paused, self._now())} — `/ai on` to resume.")
            return
        if not self._brain_ok:
            self._add_note(f"The brain is off: {self._brain_reason or 'not connected'}. Packs still work: try `/tape`.")
            return
        self.chat.add("user", text)
        from mentor_packs import registry

        messages = self.chat.messages(
            context_text=self._context_text, budget_tokens=settings.context_tokens(), memory_text=self._memory_text,
            memory_block=self._memory_block,
        )
        # P13: the app reads the question and attaches the packs it needs; the model may still call more.
        context_rows = self._context_pack.rows if self._context_pack is not None else ()
        known = attach.known_symbols(context_rows, self._liked_names, self._journal_symbols)
        attachments = attach.plan_attachments(text, known, self._now(), book=attach.book_symbols(context_rows),
                                              liked=self._liked_names)
        seen = attach.recent_cited_ids(self.chat.turns[:-1], attach.DEDUPE_TURNS)
        # P14: a book_pack built from the journal (no fresh broker snapshot) queues the broker read on the
        # news thread and says so; the next turn reads the fresh snapshot.
        from mentor_app import book_jobs

        store, now, probe = self.store, self._now, self._desk_probe
        build_pack = book_jobs.chat_pack_builder(
            self._pack_builder or self._chat_pack,
            request_fetch=lambda: self._queue_book_fetch("chat"),
            state=lambda: book_jobs.chat_fetch_state(store, now(), probe),
        )
        worker = brain.StreamWorker(
            messages,
            parent=self,
            model=self._model,
            endpoint=self._endpoint,
            keep_alive=settings.keep_alive(),
            num_ctx=settings.context_tokens(),
            tools=registry.tool_schemas(),
            native_tools=self._native_tools,
            stream_post=self._stream_post,
            post=self._post,
            attachments=attachments,
            seen_ids=seen,
            question=text,
            build_pack=build_pack,
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

    def _chat_pack(self, name: str, args: Any) -> Any:
        """A chat turn's pack (worker thread): the registry's, except the book reads this app's own store."""
        if name == "book_pack":
            from mentor_packs import book_pack

            return book_pack.build(now=self._now(), sources=self._book_pack_sources())
        if name == "journal_pack":
            # P15b: the feelings this app stored ride on their trades (its own chat store, read-only).
            return brain._default_build(name, {**dict(args or {}), "chat_db": self.store.path})
        if name == "habits_pack":
            from mentor_packs import habits_pack

            return habits_pack.build(sources=self._habits_sources)
        if name == "regime_pack":
            # P16: the diff reads earlier snapshots from this app's store; the day's first tape is snapshotted.
            pack = brain._default_build(name, {**dict(args or {}), "chat_db": self.store.path})
            self._snapshot_tape(pack)
            return pack
        return brain._default_build(name, args)

    def _snapshot_tape(self, pack: Any) -> None:
        """Store the PT day's first tape (rows only) under ``tape:snapshot:<day>``; never overwrites (worker)."""
        from mentor_packs import regime_pack

        try:
            if getattr(pack, "name", "") != regime_pack.NAME or not regime_pack.snapshot_clean(pack):
                return  # an unreadable source: try again at the next build
            key = regime_pack.snapshot_key(self._now().astimezone(regime_pack.PT).date())
            if self.store.get_state(key) is None:
                self.store.set_state(key, regime_pack.snapshot_json(pack))
        except Exception:  # noqa: BLE001 - a lost snapshot only costs tomorrow's diff
            logging.warning("Trade Mentor: tape snapshot not stored", exc_info=True)

    def _run_command(self, result: commands.CommandResult) -> None:
        stamp = self._utc_stamp()
        if result.action == "quiet":
            until = self.inbox.mute(result.arg)
            self._add_note(f"Inbox muted until {until.astimezone().strftime('%H:%M')}.")
        elif result.action == "remember":
            note = str(result.arg)
            self._memory_text = (self._memory_text + f"\n- {note}").strip()
            day, now, path = self._pt_day(), self._now(), self._plan_path

            def remember() -> None:
                self.store.add_profile_note(note, "remember", ts_utc=stamp)
                for line in plan_infer.remember_rule(note, store=self.store, day=day, now=now, path=path):
                    self._bridge.note.emit(line)

            self._submit_io(remember)
            self._add_note(f"Kept: {note}")
        elif result.action == "plan":
            path = self._plan_path
            self._submit_io(lambda: self._bridge.note.emit(plan_infer.listing(store=self.store, path=path)))
        elif result.action == "drop":
            plan_id, day, now, path = str(result.arg), self._pt_day(), self._now(), self._plan_path
            self._submit_io(lambda: self._bridge.note.emit(plan_infer.drop(plan_id, store=self.store, day=day, now=now, path=path)))
        elif result.action == "forget":
            note_id = int(result.arg)

            def forget() -> None:
                if self.store.retire_note(note_id, stamp):
                    self._bridge.note.emit(f"Retired [mem:note:{note_id}]. It is kept, never deleted.")
                    self._load_memory()
                else:
                    self._bridge.note.emit(f"There is no note {note_id}. `/memory` shows the ids.")

            self._submit_io(forget)
        elif result.action == "keep":
            note_id = int(result.arg)
            self._submit_io(lambda: self._bridge.note.emit(
                f"Still true: [mem:note:{note_id}]. I will ask again in {memory.STILL_TRUE_DAYS} days."
                if self.store.check_note(note_id, stamp) else f"There is no note {note_id}. `/memory` shows the ids."))
        elif result.action == "memory":
            self._add_note(memory.as_listing(self._memory))
        elif result.action == "feel":
            self.record_feeling(*result.arg)
        elif result.action in ("routine", "forget_routine"):
            self._routine_command(result.action, str(result.arg or ""))
        elif result.action == "journal":
            self._journal_command(str(result.arg or ""))
        elif result.action == "paste":
            text, session = result.arg
            if text:
                self.paste_brief(str(text), session=session)
            else:
                self.open_paste_dialog(session=session)
        elif result.action == "brief":
            # P15a: the night's coach brief, read on the IO thread (a file on the ai_store).
            root, now = self._memory_root, self._now()
            self._submit_io(lambda: self._bridge.note.emit(memory.coach_brief_text(memory.read_coach_brief(root, now))))
        elif result.action == "issues":
            # P15b: the night's issues, then the ones computed from the day recaps (2+ sessions).
            root, now, recap_paths = self._memory_root, self._now(), self.recap_paths

            def issues() -> None:
                from mentor_packs import recaps_pack

                text = memory.issues_text(memory.read_coach_brief(root, now))
                try:
                    recap = recaps_pack.issues_markdown(recaps_pack.build(10, "issues", now=now, paths=recap_paths).rows)
                except Exception:  # noqa: BLE001 - unreadable recaps read as "none computed", never a crash
                    logging.warning("Trade Mentor: the recap issues could not be read", exc_info=True)
                    recap = ""
                self._bridge.note.emit(text + (f"\n\n{recap}" if recap else ""))

            self._submit_io(issues)
        elif result.action == "recaps":
            days, now, recap_paths = result.arg, self._now(), self.recap_paths

            def recaps() -> None:
                from mentor_packs import recaps_pack

                pack = recaps_pack.build(days, now=now, paths=recap_paths)
                self._bridge.note.emit(recaps_pack.card_markdown(pack))

            self._submit_io(recaps)
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
            self._name_for_news(symbol)
            self.show_pick(symbol, side)
        elif result.action == "debate":
            symbol, side = result.arg
            self._name_for_news(symbol)
            self.show_debate(symbol, side)
        elif result.action == "news":
            symbol, days = result.arg
            self.show_news(symbol, days)
        elif result.action == "vetoes":
            self.show_vetoes(str(result.arg or ""))
        elif result.action == "scorecard":
            self.queue.submit("scorecard", lambda: challenge.scorecard(self.store, facts=memory.load_facts(self._memory_root)),
                              priority=PRIORITY_INTERACTIVE, key="scorecard", on_done=self._bridge.note.emit)
        elif result.action == "think":
            self.think(result.arg)
        elif result.action == "frontier":
            self._spawn("mentor-frontier-status", lambda: self._bridge.note.emit(
                self._frontier_status_text()))
        elif result.action == "hypotheses":
            self.queue.submit("hypotheses", self._hypotheses_card, priority=PRIORITY_INTERACTIVE, key="hypotheses",
                              on_done=self._bridge.note.emit)
        elif result.action == "tape":
            self.show_tape()
        elif result.action == "pack":
            # P17 /rs, /alerts: the pack's rows as a card, built off the Qt thread. No model.
            pack_name, pack_args = result.arg
            self.queue.submit(pack_name, lambda: self._pack_card(pack_name, dict(pack_args)),
                              priority=PRIORITY_INTERACTIVE, key=f"{pack_name}:{sorted(pack_args.items())}",
                              on_done=self._bridge.note.emit)
        elif result.action == "check":
            self._name_for_news(result.arg.symbol)
            self.show_check(result.arg)
        elif result.action == "book":
            self.show_book()
        elif result.action == "today":
            day = str(result.arg or "today")
            self.queue.submit("journal_today", lambda: self._today_card(day), priority=PRIORITY_INTERACTIVE,
                              key=f"journal_today:{day}", on_done=self._bridge.note.emit)
        elif result.action == "latency":
            self._submit_io(lambda: self._bridge.note.emit(latency_card(self.store.latency_rows(10))))
        elif result.action == "mirror":
            self.show_mirror(int(result.arg or 6))
        elif result.action == "tilt":
            self.show_tilt()
        elif result.action == "ai_off":
            self.set_ai_pause(result.arg)
        elif result.action == "ai_on":
            self.resume_ai()
        elif result.action == "ai_status":
            paused = self._pause_reason()
            self._add_note(f"{paused}. `/ai on` resumes." if paused else
                           f"AI is on (brain {'ready' if self._brain_ok else 'off: ' + (self._brain_reason or 'not connected')}).")
        else:
            self._add_note(result.reply)

    def stop_turn(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._stop_debates()

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
        self.stop_button.setEnabled(bool(self._debate_stops))
        self.send_button.setEnabled(True)

    def _on_done(self, result: dict) -> None:
        raw = str(result.get("text") or "")
        question = next((turn.text for turn in reversed(self.chat.turns) if turn.role == "user"), "")
        # P14 style guard: measure the model's wrapper, then drop headers and a closing offer; substance stays.
        style_numbers = style.measure(raw, question)
        text, stripped = style.guard(raw)
        style_numbers["stripped"] = stripped
        # The turn log keeps the model's raw words (plus the app's appendix); the transcript shows the guarded ones.
        stored = raw
        if result.get("appendix"):
            # The pre-trade checklist: the app adds the pack's own rows for every section the reply skipped.
            text = f"{text.rstrip()}\n\n{result['appendix']}"
            stored = f"{stored.rstrip()}\n\n{result['appendix']}"
        if result.get("cancelled"):
            text += " *(stopped)*"
            stored += " *(stopped)*"
        # Guardrail 2: a number no pack sent this turn is grey, never hidden.
        grounds = [self._context_text, *(result.get("pack_texts") or ())]
        shown = grounding.mark_uncited_numbers(text, grounds)
        # P14: an earnings claim about a name no pack gave earnings for is grey too.
        shown = grounding.mark_ungrounded_earnings(shown, grounds)
        numbers = grounding.count_numbers(text)
        if numbers:
            self._bump_stats(numbers=numbers, uncited_numbers=shown.count(f'class="{grounding.UNCITED_CLASS}"'))
        self._blocks[self._stream_slot()] = f"**Mentor:** {shown}"
        self._render()
        self.chat.add("assistant", text)
        self._latency_ms = result.get("first_token_ms")
        self._sync_status()
        if not result.get("cancelled"):
            self._last_turn = {"question": question, "context_text": self._context_text,
                               "memory_block": self._memory_block,
                               "pack_texts": list(result.get("pack_texts") or ())}
        self._store_turn(
            "assistant",
            stored,
            pack_ids=result.get("pack_ids") or (),
            model=str(result.get("model") or self._model),
            latency_ms=result.get("first_token_ms"),
            prompt_tokens=result.get("prompt_tokens"),
            completion_tokens=result.get("completion_tokens"),
            tool_calls=[*({**call, "source": "model"} for call in result.get("tool_calls") or ()),
                        *(result.get("attached") or ())],
            timings={**(result.get("timings") or {}), "style": style_numbers},
        )
        self._finish_turn()
        self._queue_embeddings()
        self._queue_plan_inference()

    def _on_failed(self, message: str) -> None:
        self._blocks[self._stream_slot()] = f"**Mentor:** *(failed: {message})*"
        self._render()
        self._brain_reason = message
        self._finish_turn()

    # ------------------------------------------------------------------ memory (P4)
    def _utc_stamp(self) -> str:
        """The window's clock as the store's UTC stamp (tests inject the clock; the wall clock never leaks in)."""
        return self._now().astimezone(timezone.utc).isoformat(timespec="milliseconds")

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
        stamp = self._utc_stamp()
        self._submit_io(lambda: self.store.mark_note_asked(note_id, stamp))
        self.refresh_inbox()

    def _queue_memory_embeddings(self) -> None:
        """Idle priority: embed night digests, the night's rows and ticker briefs (P15a) and notes not seen yet."""
        endpoint, loaded, now, root, night_paths = (self._endpoint, self._memory, self._now(), self._memory_root,
                                                    self.night_paths)
        fund_paths, recap_paths = self.fund_paths, self.recap_paths

        def job() -> int:
            done = 0
            paths = memory.night_paths_for(memory.digests_root(root), night_paths)
            have = self.store.embedded_refs("digest", settings.EMBED_MODEL)
            todo = [(item.kind, item.ref_id, item.text) for item in loaded.items
                    if item.kind == "digest" and item.ref_id not in have]
            try:
                night = memory.embed_candidates(loaded, paths, now, fund_paths=fund_paths, recap_paths=recap_paths)
            except Exception:  # noqa: BLE001 - an unreadable night read is embedded next time
                logging.warning("Trade Mentor: the night rows could not be read for recall", exc_info=True)
                night = []
            seen = {kind: self.store.embedded_refs(kind, settings.EMBED_MODEL) for kind in memory.EMBED_KINDS}
            todo += [(kind, ref, text) for kind, ref, text in night if ref not in seen[kind]]
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

    # ------------------------------------------------------------------ plan inference
    def _pt_day(self) -> str:
        return self._now().astimezone(challenge.PT).date().isoformat()

    def _queue_plan_inference(self) -> None:
        """After a reply: one side call infers plan lines from the trader's new turns (queue thread, never Qt)."""
        endpoint, model, path, post = self._endpoint, self._model, self._plan_path, self._post

        def job() -> list[dict]:
            if not self._brain_ok or self._session_id is None:
                logging.info("Trade Mentor: plan inference skipped (brain off)")
                return []
            turns = [row for row in self.store.turns(self._session_id, limit=4 * plan_infer.TURN_LIMIT)
                     if row.get("role") == "user"]
            if not turns or int(turns[-1]["id"]) <= self._plan_after:
                return []
            after, self._plan_after = self._plan_after, int(turns[-1]["id"])
            return plan_infer.infer(turns, after=after, endpoint=endpoint, model=model, post=post,
                                    keep_alive=settings.keep_alive(), num_ctx=settings.context_tokens(), path=path)

        def written(ops: Any) -> None:
            if ops:
                self._submit_io(lambda: self._apply_plan_ops(ops))

        # Waits for the IO thread's pending turn writes first, then asks on the queue's thread.
        self._submit_io(lambda: self.queue.submit(
            "plan_infer", job, priority=PRIORITY_REFRESH, needs_model=True,
            max_tokens=plan_infer.MAX_OUTPUT_TOKENS, key="plan_infer", on_done=written))

    def _apply_plan_ops(self, ops: list[dict]) -> None:
        """IO thread (the one owner of the day's count): write the ops, show one line per change."""
        for line in plan_infer.apply(ops, store=self.store, day=self._pt_day(), now=self._now(), path=self._plan_path):
            self._bridge.note.emit(line)

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
            # P18: journal lines are recalled too (kind "journal").
            for row in self.store.unembedded_journal(settings.EMBED_MODEL, limit=20):
                if self.queue.should_yield():
                    break
                vectors = brain.embed(endpoint, [row["text"]], model=settings.EMBED_MODEL, post=self._post)
                if vectors:
                    self.store.put_embedding("journal", row["id"], settings.EMBED_MODEL, vectors[0],
                                             text=row["text"][:2000])
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
        self._liked_names = list(everything)
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
                error=self._offline_why(),
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

    # ------------------------------------------------------------------ tape (P5)
    def _build_tape(self) -> Any:
        if self._tape_builder is not None and self._fund_builder is None:
            return self._tape_builder()
        from mentor_packs import regime_pack

        pack = self._tape_builder() if self._tape_builder is not None else regime_pack.build()
        self._snapshot_tape(pack)
        # P15b: the tape also reads today's pasted brief (bottom line + playbook, at most 8 rows).
        try:
            fund = self._build_fund("compact")
        except Exception:  # noqa: BLE001 - an unreadable brief never costs the tape
            logging.warning("Trade Mentor: the pasted brief could not be read for the tape", exc_info=True)
            fund = None
        return tape.with_fundamentals(pack, fund)

    def _build_fund(self, section: str) -> Any:
        if self._fund_builder is not None:
            return self._fund_builder(section)
        from mentor_packs import fundamentals_pack

        return fundamentals_pack.build("today", section, now=self._now(), paths=self.fund_paths)

    # ------------------------------------------------------------------ /paste (P15b)
    def open_paste_dialog(self, session: str = "") -> None:
        """The "Paste brief" button and a bare ``/paste``: a plain-text box; OK saves it."""
        if self._paste_prompt is not None:
            text = self._paste_prompt()
        else:
            from PySide6.QtWidgets import QInputDialog

            text, ok = QInputDialog.getMultiLineText(self, "Paste the morning brief",
                                                     "Paste the brief. It is filed in the Market Journal under its "
                                                     "own title date (else the last closed session), like the desk.")
            text = text if ok else None
        if text is None:
            return
        self.paste_brief(str(text), session=session)

    def paste_brief(self, text: str, *, session: str = "") -> None:
        """Save a pasted brief through the Market Journal's own writer, off the Qt thread, then reload memory.

        The session is the desk's rule (``forecast_session``: the brief's title date, else the last closed
        session) unless the trader named one with ``/paste for <date>``."""
        body = str(text or "").strip()
        if not body:
            self._add_note("Nothing was pasted, so nothing was saved.")
            return
        from mentor_packs import fundamentals_pack
        from ui.services.market_journal_service import forecast_session

        service = self._forecast_service
        if service is None:
            from ui.services.market_journal_service import shared_journal_service

            service = shared_journal_service()  # built here, on the Qt thread; it writes on the IO thread
        now = self._now()
        try:
            session = str(session or "").strip() or forecast_session(body, now)
        except Exception as exc:  # noqa: BLE001 - no calendar answer: the trader names the session
            self._add_note(f"Brief NOT saved: no session could be picked ({exc}). Try `/paste for 2026-09-30 ...`.")
            return
        today = fundamentals_pack.market_day(now).isoformat()
        fund_paths = self.fund_paths
        self._add_note(f"Saving the brief for {session}...")

        def job() -> None:
            try:
                result = service.import_daily_forecast(text=body, target_session=session, now=now)
            except Exception as exc:  # noqa: BLE001 - the trader is told; the text is still in the box he pasted from
                logging.warning("Trade Mentor: the pasted brief was not saved", exc_info=True)
                result = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            if not result.get("ok"):
                self._bridge.note.emit(f"Brief NOT saved: {result.get('reason', 'unknown')}. Paste it again.")
                return
            line = ""
            try:
                built = fundamentals_pack.build(session, "bottom_line", now=now, paths=fund_paths)
                line = fundamentals_pack.bottom_line_sentence(built)
            except Exception:  # noqa: BLE001 - saved is saved; the confirmation just has no bottom line
                logging.warning("Trade Mentor: the saved brief could not be read back", exc_info=True)
            note = f"Brief saved for {session}: {line or 'no bottom line found by its headings'}"
            if session != today:
                note += f" (filed for {session}, not today {today}; `/paste for {today} ...` files it for today)"
            self._bridge.note.emit(note)
            self._load_memory()

        self._submit_io(job)

    def _tape_narrator(self) -> Callable[[Any, str], Any] | None:
        """The narration call while the brain is up; None = the pack alone."""
        if not self._brain_ok or not self._endpoint or self._gpu_reason():
            return None
        endpoint, model, request = self._endpoint, self._model, self._tape_request

        def narrate(pack: Any, digest: str) -> Any:
            return tape.narrate(pack, pack_hash=digest, model=model, endpoint=endpoint, request=request, now=self._now)

        return narrate

    def _tape_fresh(self) -> bool:
        last = self._tape_last
        return last is not None and self._now() - last["at_utc"] < tape.REFRESH_EVERY

    def _tape_job(self, *, narrate: bool, source: str) -> Callable[[], dict]:
        narrator = self._tape_narrator() if narrate else None
        last_hash = str((self._tape_last or {}).get("hash") or "")

        def job() -> dict:
            result = tape.run_tape_job(store=self.store, build_pack=self._build_tape, narrate=narrator,
                                       last_hash=last_hash, should_yield=self.queue.should_yield)
            return {**result, "at_utc": self._now(), "source": source}

        return job

    def show_tape(self) -> None:
        """/tape: the cached read at once while fresh; else the pack off-thread, then one narration."""
        if self._tape_fresh():
            last = self._tape_last
            self._tape_block = len(self._blocks)
            self._add_block(tape.card_markdown(last["pack"], last["card"], brain_reason=self._tape_why()))
            if not (last["card"] is not None and last["card"].narrated):
                self._queue_tape_narration(PRIORITY_INTERACTIVE, "tape")
            return
        self._tape_block = len(self._blocks)
        self._add_block("**Tape**: reading the desk...")
        self.queue.submit("tape", self._tape_job(narrate=False, source="tape"), priority=PRIORITY_INTERACTIVE,
                          key="tape-build", on_done=self._bridge.tape_ready.emit)

    # ------------------------------------------------------------------ /check (P6)
    def _build_gate(self, request: Any) -> Any:
        if self._gate_builder is not None:
            return self._gate_builder(request)
        import dataclasses

        from mentor_packs import gate_pack

        book = self._book_pack_sources()
        sources = dataclasses.replace(gate_pack.live_sources(), book_snapshot=book.snapshot, book_status=book.status,
                                      ibkr_book_snapshot=book.ibkr_snapshot, ibkr_book_status=book.ibkr_status)
        return gate_pack.build(request.side, request.symbol, request.size, request.stop, request.entry,
                               sources=sources)

    def show_check(self, request: Any) -> None:
        """/check: build the gate pack off-thread, narrate once (never cached across requests), store the claim."""
        self._check_seq += 1
        seq = self._check_seq
        self._check_blocks[seq] = len(self._blocks)
        self._add_block(f"**Check {request.side} {request.symbol}**: building...")
        # The book is read on the news thread; the gate waits a little for it only while that thread runs.
        book_done = self._queue_book_fetch("check")
        book_wait = BOOK_WAIT_SECONDS if self.news_queue.running() else 0.0
        live = self._brain_ok and bool(self._endpoint) and not self._gpu_reason()
        paused = self._pause_reason()
        why = "" if live else (paused or f"the brain is off: {self._gpu_reason() or self._brain_reason or 'not connected'}")
        endpoint, model, gate_request, store, now = self._endpoint, self._model, self._gate_request, self.store, self._now

        def job() -> dict:
            from mentor_app import assess as _assess
            from mentor_app import gate
            from mentor_packs import gate_pack

            if book_done is not None and book_wait:
                book_done.wait(book_wait)
            pack = self._build_gate(request)
            digest = gate.request_hash(request, gate_pack.pack_hash(pack))
            if live:
                card = gate.narrate(pack, symbol=request.symbol, pack_hash=digest, model=model, endpoint=endpoint,
                                    request=gate_request, now=now)
                gate.record(store, card, request, digest)
            else:
                card = _assess.Assessment(symbol=request.symbol, pack_hash=digest, pack_json=pack.as_json(),
                                          error=why)
            return {"seq": seq, "markdown": gate.card_markdown(card, request, pack), "pack": pack}

        from mentor_app.gate import FOOTER as gate_footer

        def failed(exc: BaseException) -> None:
            self._bridge.check_card.emit({"seq": seq, "markdown": (
                f"**Check {request.side} {request.symbol}**: could not be built ({type(exc).__name__}: {exc}).\n\n"
                f"{gate_footer}")})

        self.queue.submit(f"gate {request.symbol}", job, priority=PRIORITY_INTERACTIVE, needs_model=live,
                          max_tokens=MAX_GATE_TOKENS, key=f"gate:{seq}", on_done=self._bridge.check_card.emit,
                          on_error=failed)

    def _on_check_card(self, done: dict) -> None:
        index = self._check_blocks.pop(done.get("seq"), None)
        markdown = str(done.get("markdown") or "")
        if index is not None and index < len(self._blocks):
            self._blocks[index] = markdown
            self._render()
        else:
            self._add_block(markdown)
        # The card joins the conversation, so a follow-up sees it (gate_pack stays a tool).
        self.chat.add("assistant", markdown)

    def _tape_why(self) -> str:
        return "" if self._brain_ok else (self._pause_reason() or self._brain_reason or "the brain is off")

    def _queue_tape_narration(self, priority: int, source: str) -> None:
        if self._tape_narrator() is None:
            return
        done = self._bridge.tape_ready.emit if source == "tape" else self._bridge.tape_refreshed.emit
        self.queue.submit("tape_narrate", self._tape_job(narrate=True, source=source), priority=priority,
                          needs_model=True, max_tokens=tape.MAX_OUTPUT_TOKENS, key=f"tape-narrate:{source}",
                          on_done=done)

    def _remember_tape(self, result: dict) -> None:
        old = self._tape_last or {}
        card = result.get("card")
        kept = old.get("card") if old.get("hash") == result["hash"] else None
        if kept is not None and kept.narrated and (card is None or not card.narrated):
            card = kept  # a pack-only rebuild or a failed re-read keeps the read it already has
        self._tape_last = {"pack": result["pack"], "hash": result["hash"], "card": card, "at_utc": result["at_utc"]}

    def _on_tape_ready(self, result: dict) -> None:
        """A /tape answer: replace the /tape block; narrate once when no cached read fits the pack."""
        self._remember_tape(result)
        last = self._tape_last
        markdown = tape.card_markdown(last["pack"], last["card"], brain_reason=self._tape_why())
        index = self._tape_block
        if index is not None and index < len(self._blocks):
            self._blocks[index] = markdown
            self._render()
        else:
            self._tape_block = len(self._blocks)
            self._add_block(markdown)
        card = last["card"]
        if card is not None and card.narrated:
            self._store_turn("assistant", markdown, tool_calls=[{"name": "regime_pack", "arguments": {},
                                                                 "hash": last["hash"]}])
        elif not result.get("narrated") and not result.get("yielded"):
            self._queue_tape_narration(PRIORITY_INTERACTIVE, "tape")

    def _on_tape_refreshed(self, result: dict) -> None:
        """A background rebuild: keep it for /tape; re-narrate only on a hash change. The transcript never moves."""
        changed = bool(result.get("changed"))
        self._remember_tape(result)
        card = self._tape_last["card"]
        digest = str(result.get("hash") or "")
        if result.get("narrated"):
            fresh = result.get("card")
            if fresh is not None and not fresh.narrated:
                self._tape_failures[digest] = self._tape_failures.get(digest, 0) + 1
            return
        if result.get("yielded"):
            return  # yielded to a chat turn: the next rebuild tries again
        if self._tape_failures.get(digest, 0) >= TAPE_MAX_FAILURES_PER_HASH:
            return  # failed twice on this tape: wait for a new hash or the brain coming back
        if (changed or card is None) and not (card is not None and card.narrated):
            self._queue_tape_narration(PRIORITY_REFRESH, "prefetch")

    def maybe_prefetch_tape(self) -> None:
        """Every 30 min in the session (06:00-13:00 PT): rebuild the regime pack off-thread."""
        now = self._now()
        if not self._tape_schedule.due(now):
            return
        self._tape_schedule.mark(now)
        self.queue.submit("tape_prefetch", self._tape_job(narrate=False, source="prefetch"), priority=PRIORITY_REFRESH,
                          key="tape-prefetch", on_done=self._bridge.tape_refreshed.emit)

    def maybe_push_brief(self) -> None:
        """06:30-07:30 PT: the one optional phone line (off by default); gated again inside brief_push."""
        from mentor_app import brief_push

        now = self._now()
        local = now.astimezone(brief_push.PT)
        if self._push_checked_day == local.date() or not (brief_push.SEND_FROM <= local.time() < brief_push.SEND_UNTIL):
            return
        # Read each check, so turning it on inside the window still sends that day.
        if not settings.push_brief_enabled():
            return
        send, day = self._push_send, local.date()

        def job() -> str:
            if self.store.get_state(brief_push.STATE_KEY) == day.isoformat():
                self._push_checked_day = day  # already sent (maybe before a restart)
                return ""
            last = self._tape_last
            pack = last["pack"] if last is not None and now - last["at_utc"] < tape.REFRESH_EVERY else self._build_tape()
            line = brief_push.maybe_send(self.store, pack, now=now, enabled=True, send=send)
            if line:
                self._push_checked_day = day  # sent: no more checks today
            return line

        self.queue.submit("tape_push", job, priority=PRIORITY_REFRESH, key="tape-push")

    # ------------------------------------------------------------------ news (P7)
    def _name_for_news(self, symbol: str) -> None:
        """A symbol the trader typed joins today's news scope (newest first, at most 10, dropped at day end)."""
        self._news_named.add(symbol, self._now())

    def _seed_news(self) -> None:
        """News thread, once: the store's last request stamps keep the 30-min spacing across a restart."""
        if not self._news_seeded:
            from mentor_app import news_jobs

            news_jobs.seed_fetcher(self._news_fetcher, self.store)
            self._news_seeded = True

    def _open_book_symbols(self) -> list[str]:
        if self._news_open_symbols is not None:
            return list(self._news_open_symbols() or ())
        from pathlib import Path

        from mentor_packs import gate_pack
        from project_paths import JOURNAL_DB_FILE

        return [str(row.get("symbol") or "") for row in gate_pack.read_open_trades(Path(JOURNAL_DB_FILE))]

    def show_news(self, symbol: str, days: int = 3) -> None:
        """/news: the stored headlines at once (main queue, no network); a symbol with no good fetch yet is
        then fetched once on the news thread and the card is redrawn. No model."""
        from mentor_app import news_jobs

        self._name_for_news(symbol)
        self._news_seq += 1
        seq = self._news_seq
        self._news_blocks[seq] = len(self._blocks)
        self._add_block(f"**News {symbol}**: reading...")
        store, now = self.store, self._now

        def job() -> dict:
            card = news_jobs.news_card(symbol, days, store=store, now=now())
            return {**card, "seq": seq, "days": days, "need_fetch": news_jobs.needs_first_fetch(store, symbol, now())}

        self.queue.submit(f"news {symbol}", job, priority=PRIORITY_INTERACTIVE, key=f"news-live:{seq}",
                          on_done=self._bridge.news_card.emit, on_error=self._news_failed(seq, symbol))

    def _news_failed(self, seq: int, symbol: str) -> Callable[[BaseException], None]:
        def failed(exc: BaseException) -> None:
            self._bridge.news_card.emit({"seq": seq, "markdown": (
                f"**News {symbol}**: could not be read ({type(exc).__name__}: {exc}).")})

        return failed

    def _first_fetch(self, seq: int, symbol: str, days: int) -> None:
        """News thread, ahead of the refresh cycle's waiting symbols: fetch once, then redraw the card."""
        from mentor_app import news_jobs

        def job() -> dict:
            self._seed_news()
            out = news_jobs.refresh_symbol(symbol, store=self.store, fetcher=self._news_fetcher, now=self._now())
            card = news_jobs.news_card(symbol, days, store=self.store, now=self._now(), note=news_jobs.fetch_note(out))
            return {**card, "seq": seq, "need_fetch": False}

        self.news_queue.submit(f"news_first {symbol}", job, priority=PRIORITY_INTERACTIVE, key=f"news-first:{seq}",
                               on_done=self._bridge.news_card.emit, on_error=self._news_failed(seq, symbol))

    def _on_news_card(self, done: dict) -> None:
        seq = done.get("seq")
        index = self._news_blocks.get(seq)
        markdown = str(done.get("markdown") or "")
        if index is not None and index < len(self._blocks):
            self._blocks[index] = markdown
            self._render()
        else:
            self._news_blocks[seq] = len(self._blocks)
            self._add_block(markdown)
        if done.get("need_fetch") and not self._shut:
            self._first_fetch(seq, str(done.get("symbol") or ""), int(done.get("days") or 3))
            return  # the fetched card replaces this one
        self._news_blocks.pop(seq, None)
        # The card joins the conversation, so a follow-up sees it (news_pack stays a tool).
        self.chat.add("assistant", markdown)
        pack = done.get("pack")
        self._store_turn("assistant", markdown, pack_ids=getattr(pack, "ids", ()),
                         tool_calls=[{"name": "news_pack", "arguments": {"symbol": done.get("symbol")}}])

    def maybe_refresh_news(self) -> None:
        """Every 30 min, 06:00-13:30 PT weekdays: fetch the scope's headlines on the news thread (never the
        main queue, so no /pick waits). Runs while AI is paused (news is not the GPU); skipped while the desk
        is closed. Never the Inbox, never the transcript."""
        from mentor_app import news_jobs

        now = self._now()
        if self._shut or not self._news_schedule.due(now):
            return
        self._news_schedule.mark(now)
        probe, named = self._desk_probe, self._news_named.today(now)

        def plan() -> int:
            try:
                closed = probe() is True
            except Exception:  # noqa: BLE001 - a broken probe is "unknown", never "closed"
                closed = False
            if closed:
                logging.info("Trade Mentor news: the desk is closed; no fetch this cycle")
                return 0
            self._seed_news()
            try:
                book = self._open_book_symbols()
            except Exception as exc:  # noqa: BLE001 - an unreadable journal leaves the other names
                logging.warning("Trade Mentor news: open book unreadable (%s)", exc)
                book = []
            liked = [sym for sym, _ in self._liked()]
            scope = news_jobs.news_scope(named=named, open_book=book, liked=liked)
            queued = 0
            for symbol in scope:
                if not self._news_fetcher.due(symbol, self._now()):
                    continue
                self.news_queue.submit(
                    f"news_fetch {symbol}",
                    lambda symbol=symbol: news_jobs.refresh_symbol(symbol, store=self.store,
                                                                   fetcher=self._news_fetcher, now=self._now()),
                    priority=PRIORITY_REFRESH, key=f"news-fetch:{symbol}",
                )
                queued += 1
            logging.info("Trade Mentor news cycle: %d in scope, %d due and queued (cap %d per 30 min)",
                         len(scope), queued, self._news_fetcher.max_per_cycle)
            return queued

        self.news_queue.submit("news_plan", plan, priority=PRIORITY_REFRESH, key="news-plan")

    # ------------------------------------------------------------------ book (P8)
    def _book_pack_sources(self) -> Any:
        """``book_pack`` sources: the snapshot from this app's store, the journal ``mode=ro``."""
        from mentor_app import book_jobs

        base = self._book_sources() if self._book_sources is not None else None
        return book_jobs.store_sources(self.store, base)

    def _queue_book_fetch(self, why: str, priority: int = PRIORITY_INTERACTIVE) -> threading.Event | None:
        """Queue one Questrade read on the news thread (fresh / backoff / no token / desk closed skip it inside).

        Returns the event set when that read finishes (shared with a read already waiting)."""
        from mentor_app import book_jobs

        if self._shut:
            return None
        with self._book_lock:
            if self._book_event is not None and "book-fetch" in self.news_queue.pending_keys():
                return self._book_event
            event = self._book_event = threading.Event()
        store, now, fetch, probe, ibkr = self.store, self._now, self._book_fetch, self._desk_probe, self._ibkr_fetch

        def job() -> dict:
            try:
                out = book_jobs.ensure_book(store, now(), fetch=fetch, desk_closed=probe, ibkr_fetch=ibkr)
                logging.info("Trade Mentor book (%s): %s", why, out)
                return out
            finally:
                event.set()

        if not self.news_queue.submit(f"book_fetch {why}", job, priority=priority, key="book-fetch",
                                      on_error=lambda exc: event.set()):
            event.set()
        return event

    def maybe_fetch_book(self) -> None:
        """Once a weekday from 06:20 PT: one Questrade read at refresh priority (never a poll)."""
        now = self._now()
        if self._shut or not self._book_schedule.due(now):
            return
        self._book_schedule.mark(now)
        self._queue_book_fetch("06:20", priority=PRIORITY_REFRESH)

    def show_book(self) -> None:
        """/book: read Questrade when due (news thread), then the card from book_pack. No model."""
        from mentor_app import book_jobs

        self._book_seq += 1
        seq = self._book_seq
        self._book_blocks[seq] = len(self._blocks)
        self._add_block("**Book**: reading...")
        store, now, fetch, probe, ibkr = self.store, self._now, self._book_fetch, self._desk_probe, self._ibkr_fetch

        def job() -> dict:
            from mentor_packs import book_pack

            out = book_jobs.ensure_book(store, now(), fetch=fetch, desk_closed=probe, ibkr_fetch=ibkr)
            note = book_jobs.fetch_note(out)
            pack = book_pack.build(now=now(), sources=self._book_pack_sources())
            return {"seq": seq, "pack": pack, "markdown": book_jobs.card_markdown(pack, note=note)}

        def failed(exc: BaseException) -> None:
            self._bridge.book_card.emit({"seq": seq, "markdown": (
                f"**Book**: could not be read ({type(exc).__name__}: {exc}).\n\n{book_jobs.FOOTER}")})

        self.news_queue.submit("book", job, priority=PRIORITY_INTERACTIVE, key=f"book:{seq}",
                               on_done=self._bridge.book_card.emit, on_error=failed)

    def _on_book_card(self, done: dict) -> None:
        index = self._book_blocks.pop(done.get("seq"), None)
        markdown = str(done.get("markdown") or "")
        if index is not None and index < len(self._blocks):
            self._blocks[index] = markdown
            self._render()
        else:
            self._add_block(markdown)
        # The card joins the conversation, so a follow-up sees it (book_pack stays a tool).
        self.chat.add("assistant", markdown)
        pack = done.get("pack")
        self._store_turn("assistant", markdown, pack_ids=getattr(pack, "ids", ()),
                         tool_calls=[{"name": "book_pack", "arguments": {}}])

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
                                         error=self._offline_why())
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

    # ------------------------------------------------------------------ mirror (P9)
    def _build_mirror(self, weeks: int) -> Any:
        if self._mirror_builder is not None:
            return self._mirror_builder(weeks)
        from mentor_packs import mirror_pack

        return mirror_pack.build(weeks)

    def _mirror_narrator(self) -> Callable[[Any, str], Any] | None:
        """The narration call while the brain is up; None = the pack alone."""
        if not self._brain_ok or not self._endpoint or self._gpu_reason():
            return None
        from mentor_app import mirror as mirror_mod

        endpoint, model, request = self._endpoint, self._model, self._mirror_request

        def narrate(pack: Any, digest: str) -> Any:
            return mirror_mod.narrate(pack, pack_hash=digest, model=model, endpoint=endpoint, request=request,
                                      now=self._now)

        return narrate

    def _mirror_job(self, weeks: int, *, narrate: Callable[[Any, str], Any] | None, extra: dict) -> Callable[[], dict]:
        from mentor_app import mirror as mirror_mod

        def job() -> dict:
            result = mirror_mod.run_mirror_job(store=self.store, build_pack=lambda: self._build_mirror(weeks),
                                               narrate=narrate)
            return {**result, **extra, "weeks": weeks}

        return job

    def show_mirror(self, weeks: int = 6) -> None:
        """/mirror: the pack off-thread, narrated once per pack hash (cached); brain down = the pack alone."""
        from mentor_app import mirror as mirror_mod

        self._mirror_seq += 1
        seq = self._mirror_seq
        self._mirror_blocks[seq] = len(self._blocks)
        self._add_block("**Mirror**: reading your record...")
        narrate = self._mirror_narrator()

        def failed(exc: BaseException) -> None:
            self._bridge.mirror_card.emit({"seq": seq, "error": f"{type(exc).__name__}: {exc}"})

        self.queue.submit("mirror", self._mirror_job(weeks, narrate=narrate, extra={"seq": seq}),
                          priority=PRIORITY_INTERACTIVE, needs_model=narrate is not None,
                          max_tokens=mirror_mod.MAX_OUTPUT_TOKENS, key=f"mirror:{seq}",
                          on_done=self._bridge.mirror_card.emit, on_error=failed)

    def _on_mirror_card(self, done: dict) -> None:
        from mentor_app import mirror as mirror_mod

        index = self._mirror_blocks.pop(done.get("seq"), None)
        if done.get("error"):
            markdown = f"**Mirror**: could not be built ({done['error']}).\n\n{mirror_mod.FOOTER}"
        else:
            markdown = mirror_mod.card_markdown(done["pack"], done.get("card"), brain_reason=self._tape_why())
        if index is not None and index < len(self._blocks):
            self._blocks[index] = markdown
            self._render()
        else:
            self._add_block(markdown)
        if done.get("error"):
            return
        # The card joins the conversation, so a follow-up sees it (mirror_pack stays a tool).
        self.chat.add("assistant", markdown)
        card = done.get("card")
        self._store_turn("assistant", markdown, pack_ids=getattr(done.get("pack"), "ids", ()),
                         model=getattr(card, "model", "") or "",
                         tool_calls=[{"name": "mirror_pack", "arguments": {"weeks": done.get("weeks")},
                                      "hash": done.get("hash"), "error": getattr(card, "error", "")}])

    # ------------------------------------------------------------------ frontier (P11)
    def _key(self) -> str:
        from mentor_app import frontier

        return (self._frontier_key or frontier.load_key)()

    def _frontier_status(self) -> Any:
        """The switch, key and cap now (reads the key: never on the Qt thread)."""
        from mentor_app import frontier

        state = frontier.status(self.store, self._now(), key_loader=self._key)
        self._bridge.frontier_state.emit(state)
        return state

    def _frontier_status_text(self) -> str:
        from mentor_app import frontier

        return frontier.status_text(self._frontier_status())

    def _on_frontier_state(self, state: Any) -> None:
        self._frontier_state = state
        self.think_button.setVisible(bool(state.enabled))
        self.think_button.setEnabled(state.usable and not self._frontier_busy)
        self.think_button.setToolTip("Ask the frontier model the last question again (metered, capped per day)"
                                     if state.usable else f"Not available: {state.reason}")

    def think(self, what: Any = None) -> None:
        """/think [pick SYM [side] | week]: one metered frontier call over what the local model read."""
        kind = (what or ("chat",))[0]
        if self._frontier_busy:
            self._add_note("A frontier call is already running.")
            return
        last = dict(self._last_turn) if self._last_turn else None
        if kind == "chat" and not last:
            self._add_note("Ask a question first: `/think` re-asks the last one with the frontier model.")
            return
        self._frontier_busy = True
        self.think_button.setEnabled(False)
        self._frontier_seq += 1
        seq = self._frontier_seq
        self._frontier_blocks[seq] = len(self._blocks)
        self._add_block("**frontier**: checking the switch, the key and today's cap...")

        def job() -> None:
            try:
                self._bridge.frontier_card.emit({"seq": seq, **self._frontier_job(what or ("chat",), last)})
            except Exception as exc:  # noqa: BLE001 - a failed frontier job is a note, never a crash
                self._bridge.frontier_card.emit({"seq": seq, "markdown": f"**frontier**: failed ({exc})."})

        self._spawn("mentor-frontier", job)

    def _frontier_job(self, what: Any, last: dict[str, Any] | None) -> dict[str, Any]:
        """Worker thread: the guard, then the one call, then the card and the turn log."""
        from mentor_app import assess, frontier

        kind = what[0]
        state = self._frontier_status()
        if not state.usable:
            return {"markdown": f"**frontier**: not called ({state.reason})."}
        usage: list[dict[str, Any]] = []
        request = frontier.metered_request(
            store=self.store, purpose=kind, model=state.model, api_key=self._key(), cap_usd=state.cap_usd,
            now=self._now, request=self._frontier_request, spent_sink=usage)
        post = frontier.frontier_post(self._frontier_post, model=state.model)
        pack_ids: tuple[str, ...] = ()
        if kind == "pick":
            from mentor_packs import pick_pack

            symbol, side = what[1], what[2] if len(what) > 2 else ""
            pack = self._build_pick(symbol, side)
            digest = pick_pack.pack_hash(pack)
            self._pick_packs[(symbol, digest)] = pack
            result = assess.assess(pack, symbol=symbol, pack_hash=digest, model=state.model, endpoint="",
                                   live=True, request=request, post=post, now=self._now)
            body = assess.card_markdown(result, side=side)
            markdown = f"**{frontier.label(state.model)}** · Think harder: pick {symbol}\n\n{body}"
            pack_ids, error = pack.ids, result.error
            reply: Any = {"verdict": result.verdict, "bullets": result.bullets, "rule_flags": result.rule_flags}
            dropped = result.dropped
        else:
            if kind == "week":
                from mentor_packs import hypothesis_pack

                rows = frontier.week_rows(
                    digests=frontier.load_digests(self._memory_root), mirror=self._build_mirror(6),
                    hypotheses=hypothesis_pack.build(now=self._now(), chat_db=self.store.path,
                                                     history_dir=self.permutation_history,
                                                     report_file=self.permutation_report),
                    now=self._now())
                answer = frontier.think_week(rows, model=state.model, request=request, post=post)
                pack_ids = tuple(row["source_id"] for row in rows)
            else:
                last = last or {}
                answer = frontier.think_chat(
                    str(last.get("question") or ""), context_text=str(last.get("context_text") or ""),
                    memory_block=str(last.get("memory_block") or ""), pack_texts=last.get("pack_texts") or (),
                    model=state.model, request=request, post=post)
            markdown = frontier.card_markdown(answer)
            error, reply, dropped = answer.error, answer.reply, answer.dropped
        spent = self.store.frontier_spent(frontier.day_pt(self._now()))
        markdown += "\n\n*" + frontier.spend_line(usage, spent, state.cap_usd) + "*"
        self._frontier_status()
        return {"markdown": markdown, "model": state.model, "pack_ids": pack_ids,
                "tool_calls": [{"name": "frontier", "purpose": kind, "model": state.model, "usage": usage,
                                "reply": reply, "dropped": dropped, "error": error}]}

    def _on_frontier_card(self, done: dict) -> None:
        self._frontier_busy = False
        state = self._frontier_state
        self.think_button.setEnabled(bool(state is not None and state.usable))
        markdown = str(done.get("markdown") or "")
        index = self._frontier_blocks.pop(done.get("seq"), None)
        if index is not None and index < len(self._blocks):
            self._blocks[index] = markdown
            self._render()
        else:
            self._add_block(markdown)
        if done.get("model"):
            self.chat.add("assistant", markdown)
            self._store_turn("assistant", markdown, pack_ids=done.get("pack_ids") or (), model=str(done["model"]),
                             tool_calls=done.get("tool_calls") or ())

    # ------------------------------------------------------------------ /hypotheses (P11)
    def _hypotheses_card(self) -> str:
        """The night's hypotheses and their cells (queue thread; reads only)."""
        from mentor_packs import hypothesis_pack

        pack = hypothesis_pack.build(now=self._now(), chat_db=self.store.path, history_dir=self.permutation_history,
                                     report_file=self.permutation_report)
        return hypothesis_pack.card_markdown(pack)

    def _pack_card(self, name: str, args: dict) -> str:
        """P17 /rs and /alerts: one pack as a card with its ids (queue thread; file reads only)."""
        from mentor_packs import registry

        if name == "habits_pack":
            # P18 /habits: the night's habit counts (its file; tests pass their own).
            from mentor_packs import habits_pack

            pack = habits_pack.build(sources=self._habits_sources)
        else:
            try:
                if name in ("rs_pack", "alerts_pack"):
                    args.setdefault("liked", [sym for sym, _side in self._liked()])
            except Exception:  # noqa: BLE001 - no liked marks; the book still marks
                pass
            pack = registry.build(name, **args)
        lines = [f"- [{row['id']}] {row.get('text', '')}" for row in pack.rows]
        title = {"rs_pack": "Relative strength", "alerts_pack": "Alerts", "habits_pack": "Habits"}.get(name, name)
        return f"**{title}**\n\n" + ("\n".join(lines) if lines else (pack.empty_text or "nothing"))

    # ------------------------------------------------------------------ /debate (P10)
    def _debate_runner(self, stop: threading.Event) -> Callable[[Any, str], Any] | None:
        """Both persona calls while the brain is up; None = no debate, the pack alone."""
        if not self._brain_ok or not self._endpoint or self._gpu_reason():
            return None
        from mentor_app import debate

        endpoint, model, request = self._endpoint, self._model, self._debate_request

        def run(pack: Any, digest: str, symbol: str, side: str) -> Any:
            return debate.debate(pack, symbol=symbol, side=side, pack_hash=digest, model=model, endpoint=endpoint,
                                 request=request, cancelled=stop.is_set, now=self._now)

        return run

    def show_debate(self, symbol: str, side: str = "") -> None:
        """/debate: the pick pack once, then bull and bear on the model queue (interactive); brain off = the pack."""
        from mentor_app import debate
        from mentor_packs import pick_pack

        symbol = str(symbol or "").upper()
        self._debate_seq += 1
        seq = self._debate_seq
        self._debate_blocks[seq] = len(self._blocks)
        self._add_block(f"**Debate {symbol}**: building the pack, then bull and bear...")
        stop = threading.Event()
        runner = self._debate_runner(stop)
        why = self._tape_why() or self._gpu_reason()
        if runner is not None:
            self._debate_stops[seq] = stop
            self.stop_button.setEnabled(True)
            self.activity_label.setText(f"debating {symbol}...")

        def job() -> dict:
            result = debate.run_debate_job(
                symbol, side, store=self.store, build_pack=self._build_pick, pack_hash=pick_pack.pack_hash,
                run=None if runner is None else (
                    lambda pack, digest: runner(pack, digest, symbol, debate.pack_side(pack) or side)),
            )
            return {**result, "seq": seq, "why": why}

        def failed(exc: BaseException) -> None:
            self._bridge.debate_card.emit({"seq": seq, "symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})

        self.queue.submit(f"debate {symbol}", job, priority=PRIORITY_INTERACTIVE, needs_model=runner is not None,
                          max_tokens=debate.job_budget(self._model), calls=debate.CALLS, key=f"debate:{seq}",
                          on_done=self._bridge.debate_card.emit, on_error=failed)

    def _stop_debates(self) -> None:
        """Stop: a debate not started yet is dropped; a running one stops before its next call."""
        for seq, stop in list(self._debate_stops.items()):
            stop.set()
            if self.queue.cancel(f"debate:{seq}"):
                self._debate_stops.pop(seq, None)
                self._replace_debate_block(seq, "**Debate**: stopped before it started.")
        if not self._debate_stops and self._worker is None:
            self.stop_button.setEnabled(False)

    def _replace_debate_block(self, seq: int, markdown: str) -> None:
        index = self._debate_blocks.pop(seq, None)
        if index is not None and index < len(self._blocks):
            self._blocks[index] = markdown
            self._render()
        else:
            self._add_block(markdown)

    def _on_debate_card(self, done: dict) -> None:
        from mentor_app import debate

        seq = done.get("seq")
        self._debate_stops.pop(seq, None)
        if not self._debate_stops:
            self.activity_label.setText("")
            if self._worker is None:
                self.stop_button.setEnabled(False)
        symbol = str(done.get("symbol") or "")
        if done.get("error"):
            self._replace_debate_block(seq, f"**Debate {symbol}**: could not be built ({done['error']}).")
            return
        pack, digest, result = done["pack"], done["hash"], done.get("debate")
        self._pick_packs[(symbol, digest)] = pack  # the card's evidence link opens this pack
        markdown = debate.card_markdown(result, pack, symbol=symbol, side=str(done.get("side") or ""),
                                        pack_hash=digest, brain_reason=str(done.get("why") or ""))
        self._replace_debate_block(seq, markdown)
        # The card joins the conversation; a follow-up goes through the normal turn (pick_pack stays a tool).
        self.chat.add("assistant", markdown)
        self._store_turn("assistant", markdown, pack_ids=getattr(pack, "ids", ()),
                         model=getattr(result, "model", "") or "", tool_calls=debate.turn_tool_calls(done))

    def maybe_mirror_week(self) -> None:
        """Every minute: deliver a waiting weekly card; from 06:50 PT on the week's first session day, build it."""
        from mentor_app import mirror as mirror_mod

        now = self._now()
        self._deliver_mirror_inbox()
        if self._shut or not self._mirror_schedule.due(now):
            return
        self._mirror_schedule.mark(now)
        week = mirror_mod.week_key(now)
        narrate = self._mirror_narrator()
        store = self.store

        def job() -> dict:
            posted = store.get_state(mirror_mod.POSTED_KEY)
            if posted == week:
                return {"week": week, "posted": posted}
            result = self._mirror_job(6, narrate=narrate, extra={})()
            return {**result, "week": week, "posted": posted}

        self.queue.submit("mirror_week", job, priority=PRIORITY_REFRESH, needs_model=narrate is not None,
                          max_tokens=mirror_mod.MAX_OUTPUT_TOKENS, key="mirror-week",
                          on_done=self._bridge.mirror_week.emit)

    def _on_mirror_week(self, built: dict) -> None:
        from mentor_app import mirror as mirror_mod

        if built.get("posted") == built.get("week") or "pack" not in built:
            return  # this week's card was posted before a restart
        markdown = mirror_mod.card_markdown(built["pack"], built.get("card"), brain_reason=self._tape_why(),
                                            title=mirror_mod.INBOX_LINE)
        self._mirror_waiting.append((str(built["week"]), self._now().astimezone(challenge.PT).date(), markdown))
        self._deliver_mirror_inbox()

    def _deliver_mirror_inbox(self) -> None:
        """Post the waiting weekly card once quiet hours or a mute end; a used cap or a new day drops it."""
        from mentor_app import mirror as mirror_mod

        today = self._now().astimezone(challenge.PT).date()
        while self._mirror_waiting:
            week, day, markdown = self._mirror_waiting[0]
            if day != today:
                logging.info("Trade Mentor: the mirror card for %s was dropped (held past its day)", week)
                self._mirror_waiting.pop(0)
                continue
            if not self.inbox.refusal():
                # The marker is queued before the item shows: a quit between the two never reposts the card.
                self._submit_io(lambda week=week: self.store.set_state(mirror_mod.POSTED_KEY, week))
            item = self.inbox.add("mirror", mirror_mod.INBOX_LINE)
            if item is None:
                if "cap" in self.inbox.last_refusal:
                    logging.info("Trade Mentor: the mirror card was dropped (%s)", self.inbox.last_refusal)
                    self._mirror_waiting.pop(0)
                    continue
                return  # quiet hours or muted: try again next minute
            self._mirror_waiting.pop(0)
            self._inbox_cards[item.id] = markdown
            self.refresh_inbox()

    # ------------------------------------------------------------------ tilt watch (P9)
    def _build_tilt(self) -> Any:
        if self._tilt_builder is not None:
            return self._tilt_builder()
        from mentor_packs import tilt_pack

        return tilt_pack.build(now=self._now(), journal=self._tilt_journal)

    def _tilt_signature(self) -> Callable[[], Any] | None:
        if self._tilt_builder is not None and self._tilt_journal is None:
            return None
        from mentor_packs import journal_read, tilt_pack

        journal = self._tilt_journal if self._tilt_journal is not None else tilt_pack.live_journal()
        day = self._now().astimezone(journal_read.ET).date().isoformat()
        return lambda: journal_read.leg_signature(journal, day)

    def maybe_watch_tilt(self) -> None:
        """Every 2 min in 06:30-13:00 PT weekdays: run the tilt pack on the news thread (no model, no GPU)."""
        from mentor_app import tilt_watch

        now = self._now()
        self._deliver_tilt()
        if self._shut or not self._tilt_schedule.due(now):
            return
        store, signature, closed = self.store, self._tilt_signature(), self._tilt_closed()
        self.news_queue.submit("tilt_watch", lambda: tilt_watch.run_watch(store, now, build=self._build_tilt,
                                                                          signature=signature, closed=closed),
                               priority=PRIORITY_REFRESH, key="tilt-watch", on_done=self._bridge.tilt_ready.emit)

    def _tilt_closed(self) -> Callable[[str], list] | None:
        """P15b: the day's closed trades from the journal the tilt watch reads (None = no journal to read)."""
        if self._tilt_builder is not None and self._tilt_journal is None:
            return None
        from mentor_app import tilt_watch
        from mentor_packs import tilt_pack

        journal = self._tilt_journal if self._tilt_journal is not None else tilt_pack.live_journal()
        return lambda day: tilt_watch.closed_today(journal, day)

    def _on_tilt_ready(self, result: dict) -> None:
        """New observations wait for the Inbox; they are posted as ONE item at most every 30 min."""
        stored = str(result.get("last_post") or "")
        if stored > self._tilt_last_post:
            self._tilt_last_post = stored
        day = self._now().astimezone(challenge.PT).date()
        self._tilt_waiting.extend((day, row) for row in result.get("new") or [])
        # The watch returns every close still owed its question (held ones persist in app_state).
        held = {row["trade_id"] for _d, row in self._feel_waiting}
        self._feel_waiting.extend((day, row) for row in result.get("closed") or []
                                  if row.get("trade_id") not in held and row.get("trade_id") not in self._feel_asked)
        self._deliver_tilt()

    def _deliver_tilt(self) -> None:
        """Post the waiting observations as one item once the 30 min, quiet hours or a mute allow it.

        A used daily cap or a new day drops them (``/tilt`` still shows them). Never pops, never moves
        the transcript. P15b: a closed trade's one feelings question shares the same spacing, after
        any tilt item (one Inbox item per pass at most)."""
        from mentor_app import tilt_watch

        today = self._now().astimezone(challenge.PT).date()
        self._tilt_waiting = [(day, row) for day, row in self._tilt_waiting if day == today]
        self._feel_waiting = [(day, row) for day, row in self._feel_waiting if day == today]
        if not (self._tilt_waiting or self._feel_waiting) or not tilt_watch.may_post(self._tilt_last_post or None,
                                                                                     self._now()):
            return
        if not self._tilt_waiting:
            self._deliver_feeling()
            return
        item = self.inbox.add("tilt", tilt_watch.inbox_text([row for _, row in self._tilt_waiting]))
        if item is None:
            if "cap" in self.inbox.last_refusal:
                logging.info("Trade Mentor: tilt observations stayed out of the Inbox (%s)", self.inbox.last_refusal)
                self._tilt_waiting = []
                self._mark_feel_asked([row for _d, row in self._feel_waiting])
                self._feel_waiting = []
            return  # quiet hours or muted: try again at the next watch
        self._tilt_waiting = []
        self._stamp_tilt_post()

    def _deliver_feeling(self) -> None:
        """One "how did it feel?" item for the oldest waiting close; the cap drops the rest for today."""
        from mentor_app import tilt_watch

        _day, trade = self._feel_waiting[0]
        item = self.inbox.add("feeling", tilt_watch.feel_text(trade))
        if item is None:
            if "cap" in self.inbox.last_refusal:
                logging.info("Trade Mentor: feelings questions stayed out of the Inbox (%s)", self.inbox.last_refusal)
                self._mark_feel_asked([row for _d, row in self._feel_waiting])
                self._feel_waiting = []
            return  # quiet hours or muted: still held (in app_state too), asked at the next pass
        self._feel_waiting.pop(0)
        self._inbox_feel[item.id] = dict(trade)
        self._mark_feel_asked([trade])
        self._stamp_tilt_post()

    def _mark_feel_asked(self, trades: list[dict]) -> None:
        """Only an accepted (or cap-dropped) question is marked asked; IO thread writes app_state."""
        from mentor_app import tilt_watch

        self._feel_asked.update(str(row.get("trade_id")) for row in trades)
        by_day: dict[str, list[str]] = {}
        for row in trades:
            by_day.setdefault(str(row.get("day") or ""), []).append(str(row.get("trade_id")))
        for day, ids in by_day.items():
            if day:
                self._submit_io(lambda day=day, ids=ids: tilt_watch.mark_asked(self.store, day, ids))

    def record_journal(self, text: str, *, asks: bool = False) -> None:
        """P18: store one journal-mode statement with its tags and context (IO thread; journal read-only)."""
        from mentor_app import journal_mode
        from mentor_packs import tilt_pack

        journal = self._tilt_journal if self._tilt_journal is not None else tilt_pack.live_journal()
        rows = list(self._context_pack.rows) if self._context_pack is not None else []
        now, forced = self._now(), self._journal_on

        def job() -> None:
            fields = journal_mode.entry_fields(text, now, context_rows=rows, journal=journal, forced=forced, asks=asks)
            self._bridge.note.emit(journal_mode.noted_line(fields, self.store.add_journal_entry(fields)))

        self._submit_io(job)
        if self._brain_ok and not asks:
            self._queue_embeddings()  # an answered statement embeds after its reply

    def _journal_command(self, word: str) -> None:
        """``/journal on|off`` sets the mode (persisted); ``/journal`` shows today's entries (IO thread)."""
        from mentor_app import journal_mode

        if word in ("on", "off"):
            self._journal_on = word == "on"
            self._submit_io(lambda: self.store.set_state(journal_mode.MODE_KEY, word))
            self._add_note("Journal mode ON: every message is kept as a journal line (a question in it is still "
                           "answered). `/journal off` to stop." if word == "on"
                           else "Journal mode OFF: I keep self talk and answer questions.")
            return
        day = self._now().astimezone(journal_mode.ET).date().isoformat()
        self._submit_io(lambda: self._bridge.note.emit(
            journal_mode.entries_text(self.store.journal_entries(day, day), day)))

    def record_feeling(self, ref: str, words: str) -> None:
        """``/feel <SYM|trade id> <words>``: one feelings note kept with its trade (IO thread; journal read-only)."""
        from mentor_app import tilt_watch
        from mentor_packs import tilt_pack

        journal = self._tilt_journal if self._tilt_journal is not None else tilt_pack.live_journal()
        now, stamp = self._now(), self._utc_stamp()

        def job() -> None:
            trade = tilt_watch.resolve_trade(journal, ref, now)
            if trade is None or not trade["trade_id"]:
                self._bridge.note.emit(f"I can't find a trade for `{ref}` in the last {tilt_watch.FEEL_LOOKBACK_DAYS} "
                                       "days. Try its trade id (`/today` shows them).")
                return
            note_id = self.store.add_feeling(trade["trade_id"], tilt_watch.feeling_note(trade, words), ts_utc=stamp)
            if note_id is None:
                self._bridge.note.emit("That feeling was NOT saved (the chat store failed). Type it again.")
                return
            self._bridge.note.emit(f"Kept with {trade['symbol']} ({trade['trade_id']}): {words} [mem:note:{note_id}]")

        self._submit_io(job)

    def _stamp_tilt_post(self) -> None:
        from mentor_app import tilt_watch

        stamp = self._now().astimezone(timezone.utc).isoformat(timespec="seconds")
        self._tilt_last_post = stamp
        self._submit_io(lambda: self.store.set_state(tilt_watch.LAST_POST_KEY, stamp))
        self.refresh_inbox()

    def show_tilt(self) -> None:
        """/tilt: today's observations and the base rates, built on the news thread. No model."""
        from mentor_app import tilt_watch

        self._tilt_seq += 1
        seq = self._tilt_seq
        self._tilt_blocks[seq] = len(self._blocks)
        self._add_block("**Tilt**: reading today's journal...")

        def job() -> dict:
            pack = self._build_tilt()
            return {"seq": seq, "pack": pack, "markdown": tilt_watch.card_markdown(pack)}

        def failed(exc: BaseException) -> None:
            self._bridge.tilt_card.emit({"seq": seq, "markdown": f"**Tilt**: could not be read ({type(exc).__name__})."})

        self.news_queue.submit("tilt", job, priority=PRIORITY_INTERACTIVE, key=f"tilt:{seq}",
                               on_done=self._bridge.tilt_card.emit, on_error=failed)

    def _on_tilt_card(self, done: dict) -> None:
        index = self._tilt_blocks.pop(done.get("seq"), None)
        markdown = str(done.get("markdown") or "")
        if index is not None and index < len(self._blocks):
            self._blocks[index] = markdown
            self._render()
        else:
            self._add_block(markdown)
        pack = done.get("pack")
        if pack is not None:
            self.chat.add("assistant", markdown)
            self._store_turn("assistant", markdown, pack_ids=getattr(pack, "ids", ()),
                             tool_calls=[{"name": "tilt_pack", "arguments": {}}])
