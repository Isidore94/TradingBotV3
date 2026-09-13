"""The left-nav Market Journal page — R10.H, plus the chart capture follow-on.

The sit-down review: the entries, the tape each one was written against, the
environment timeline with its auto-vs-manual agreement rate, the calendar strip,
and the machine's own day-context row beside the trader's words.

Two labels that look like a collision and are not: the existing left-nav
**"Journal"** is the trade/tax journal — what you traded. This is **"Market
Journal"** — what you thought. Merging them would turn the tax record into a
diary, so the difference is deliberate and stays.

**What changed after the first live day** (2026-08-27). The trader wrote five
entries through the Desk tab and this page showed nothing, because it loaded
only when "Refresh" was clicked — nothing called `reload()` at construction or
on show, and the desk tab held a *second* service instance, so its
`entryWritten` never reached here. Both are fixed: one shared service, and the
page loads the first time it is shown. The other half of the same report was
that words alone were not worth re-reading — so every entry now carries the M5
and D1 of its symbol and of SPY as they stood when it was written, and this
page draws them (`market_journal_capture`).

Everything expensive is off the Qt thread (ground rule 9): entries, digests and
the stored bar windows all load on workers, and the page renders what it is
handed.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.widgets.candle_chart import CandleChart

#: How many D1 charts the page shows. Six is the trader's own number.
CHART_COUNT = 6

#: The four panes of a capture, in reading order: what you were watching first,
#: then the market it was moving inside.
CAPTURE_PANES = (
    ("symbol_m5", "{symbol} M5", "m5"),
    ("symbol_d1", "{symbol} D1", "d1"),
    ("benchmark_m5", "{benchmark} M5", "m5"),
    ("benchmark_d1", "{benchmark} D1", "d1"),
)

#: How much of a thought the narrow entries list shows (G3.1). The list
#: elides one line whatever it is handed, so putting a 1,200-character
#: thought in there showed the trader a clipped fragment and nothing else;
#: 90 characters is about a sentence, which is enough to FIND an entry. The
#: rest of it is the reader pane's job, not the list's.
EXCERPT_LIMIT = 90

#: How many lines of the composer the page opens with (G3.2). The composer
#: used to take whatever the vertical splitter's stretch gave it, which on a
#: tall screen was half the page for an empty box. Four lines is a paragraph;
#: the splitter handle still drags it as tall as the trader wants.
COMPOSER_LINES = 4

#: The reader's MEASURE, in characters of its own font (G3 fix round). The
#: first cut of G3 gave `thought_view` the whole right half - 2,779 px at
#: 3456 x 2160 - so a 68-character sentence was set as one 400-character
#: line and the eye lost the start of it before it found the end. Typography
#: has one rule here and it is old: a line of running text is readable at
#: about 45-100 characters. The pane keeps its full width; the TEXT is capped
#: and sits at the left of it, with the slack on the right.
READER_MEASURE_CHARS = 100

#: The same measure's ceiling in design pixels, so a large font on a small
#: shell cannot push the column wider than the pane it lives in. Scaled
#: through `theme.px`, and the SMALLER of the two wins.
READER_MEASURE_MAX_PX = 1200

#: The reader still needs a floor: a splitter dragged narrow must shrink the
#: column, never hide it.
READER_MEASURE_MIN_PX = 240

#: How the left (entries + timeline + context) and right (reader + charts)
#: halves of the page OPEN (G3 fix round). Stretch alone gave the entries
#: column 655 px at 3456 x 2160, where a 90-character excerpt clipped after
#: about 35 characters and the list read as nothing. One third is enough for
#: the whole excerpt; the handle still drags it either way.
LOWER_SPLIT_SHARES = (1000, 2000)

#: The Story pane's three headings (WS-10D item 3, WISHLIST 10D step 1). They
#: are CONSTANTS because the three kinds must be visibly distinct wherever the
#: story is rendered and because a test asserts the reading order: the trader's
#: own words, then what the desk measured, then the sources each came from.
STORY_TRADER_HEADING = "You said"
STORY_MEASURED_HEADING = "The market did"
STORY_SOURCES_HEADING = "Sources"
#: Someone else's words, under their own heading and never under "You said"
#: (WISHLIST 10K). An imported forecast is outside commentary; it becomes the
#: trader's view only if they write an entry of their own adopting it.
STORY_FORECAST_HEADING = "External forecast"

#: The link scheme the Story pane's sources use. A source is a LINK - clicking
#: it selects the entry (and therefore its capture) in the list on the left,
#: which is the whole point of naming sources rather than counting them.
STORY_ENTRY_SCHEME = "entry"


def _reader_measure(metrics: QFontMetrics) -> int:
    """The pixel width of `READER_MEASURE_CHARS` characters, capped and floored.

    `averageCharWidth` is the font's own answer, so the measure follows the
    theme instead of guessing at it; `theme.px` scales the ceiling the same
    way every other Python-side pixel budget on this desk is scaled.
    """
    per_char = max(1, int(metrics.averageCharWidth()))
    wanted = min(per_char * READER_MEASURE_CHARS, theme.px(READER_MEASURE_MAX_PX))
    return max(theme.px(READER_MEASURE_MIN_PX), wanted)


def _excerpt(text: str, limit: int = EXCERPT_LIMIT) -> str:
    """The first line of a thought, cut at `limit`, with `…` when there is more.

    The ellipsis is a CLAIM - "there is more text than this" - so it is never
    printed for a short single-line entry that is shown whole.
    """
    body = str(text or "").strip()
    lines = body.splitlines()
    first = lines[0].strip() if lines else ""
    truncated = len(lines) > 1 or len(first) > limit
    if len(first) > limit:
        first = first[:limit].rstrip()
    return f"{first}…" if truncated else first


def _utc_offset_label(moment: datetime) -> str:
    """`UTC-07:00` / `UTC+00:00` - the offset an AWARE moment carries."""
    offset = moment.utcoffset() or timedelta(0)
    total_minutes = int(offset.total_seconds() // 60)
    sign = "-" if total_minutes < 0 else "+"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def _written_line(created_at: Any) -> str:
    """`written HH:MM UTC±HH:MM`, in the DESK's own zone (G3b item 1).

    Every live row stores `created_at` in UTC (`market_journal.py` astimezones
    to `timezone.utc` before writing), so printing the zone the stamp CARRIES
    - what this used to do - read `written 13:36 UTC` for a note typed at
    06:36 Pacific, on every live row. The AWARE side gets `astimezone`d to
    `pass_bars.desk_zone()` (the one desk-zone seam, N1 2026-09-05) so the
    printed time is the trader's own clock; a NAIVE stamp carries no zone to
    convert FROM and still says so rather than being silently given one - the
    Market Journal's whole contract is that an entry is never backdated, and a
    time printed in a zone nobody recorded is a quiet backdating.
    """
    raw = str(created_at or "").strip()
    if not raw:
        return "written at an unrecorded time"
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return f"written {raw[:19]}"
    if moment.tzinfo is None:
        return f"written {moment.strftime('%H:%M')} (no zone recorded)"
    from ui.annotations.pass_bars import desk_zone

    local = moment.astimezone(desk_zone())
    return f"written {local.strftime('%H:%M')} {_utc_offset_label(local)}"


def _html(text: Any) -> str:
    """The trader's own characters, safe inside the one HTML view on this page."""
    from html import escape

    return escape(str(text or ""), quote=True)


def _story_entry_prefix(row: Any) -> str:
    """`13:36 · [written after the session]` - how to read the sentence below it."""
    parts = [str(row.get("created_at") or "")[11:16]]
    if row.get("written_after_the_session"):
        parts.append("[written after the session]")
    elif row.get("predicts_this_session"):
        parts.append("[written during the session]")
    symbols = ", ".join(str(item) for item in (row.get("symbols") or ()))
    if symbols:
        parts.append(symbols)
    return "  ·  ".join(part for part in parts if part)


def _measured_line(cell: Any) -> str:
    """One benchmark's row, or the reason there is none.

    An unmeasured benchmark is NAMED. A cell dropped for having no bars reads
    as "nothing happened there", which is a claim nobody measured.
    """
    symbol = str(cell.get("symbol") or "")
    if str(cell.get("status") or "") != "measured":
        return f"{symbol}: unmeasured — {cell.get('reason') or 'no completed bars'}"
    position = dict(cell.get("position_vs_sma20") or {})
    distance = position.get("distance_atr")
    sma_part = (
        f"{position.get('side', 'unknown')} SMA20 by {distance:.2f} ATR"
        if isinstance(distance, (int, float))
        else f"{position.get('side', 'unknown')} SMA20"
    )
    change = cell.get("change_pct")
    close = cell.get("close")
    span = cell.get("range_atr")
    head = f"{symbol}: close {close:.2f}" if isinstance(close, (int, float)) else f"{symbol}:"
    return (
        head
        + (f", {change:+.2f}%" if isinstance(change, (int, float)) else "")
        + (f", range {span:.2f} ATR" if isinstance(span, (int, float)) else "")
        + f", {sma_part}"
        + f" (bars through {cell.get('bars_through') or 'unknown'})"
    )


def _thesis_label(row: Any) -> str:
    """Claim, horizon, stance, condition, invalidation - `unstated` when unsaid.

    `unstated` is printed, never hidden: a thesis with no stated invalidation
    and a thesis whose invalidation nobody bothered to show look identical if
    the field is simply left out, and only one of those is honest.
    """
    claim = str(row.get("claim") or "").strip() or "(no claim found)"
    horizon = str(row.get("horizon") or "")
    stance = str(row.get("stance") or "")
    condition = str(row.get("condition") or "")
    invalidation = str(row.get("invalidation") or "")
    parts = [f"{stance}: {claim}"]
    if horizon:
        parts.append(f"horizon {horizon}")
    parts.append(f"condition: {condition}")
    parts.append(f"invalidation: {invalidation}")
    return "  ·  ".join(parts)


class _EntriesWorker(QThread):
    """Loads entries, digests, the regime timeline and the day context."""

    loaded = Signal(dict)

    def __init__(self, service, session_date: str, parent=None) -> None:
        super().__init__(parent)
        self._service = service
        self._session = session_date

    def run(self) -> None:  # pragma: no cover - exercised through its signal seam
        payload: dict[str, Any] = {"session_date": self._session}
        try:
            # R4 A16: EVERY session, not one. The picker is gone, so the list is
            # the journal - dated, newest first - and the day context below it
            # follows whichever entry is selected.
            payload["entries"] = self._service.entries_for()
            payload["sessions"] = self._service.sessions_with_entries()
            payload["timeline"] = self._service.regime_timeline()
            payload["context"] = self._service.day_context(self._session)
        except Exception as exc:  # noqa: BLE001
            payload["error"] = str(exc)
        try:
            payload["digests"] = self._service.chart_digests()
        except Exception:  # noqa: BLE001
            # A missing capture store is a quieter page, never a failed one:
            # the entries are the record and they loaded.
            payload["digests"] = {}
        # WS-10D: the story and the theses load HERE, on the worker, with
        # everything else the page reads (G7). Each in its own guard: a story
        # that cannot be built must not blank the entries, which are the record.
        try:
            payload["story"] = self._service.daily_story(self._session)
        except Exception:  # noqa: BLE001
            payload["story"] = None
        try:
            payload["theses"] = self._service.theses_for(self._session)
        except Exception:  # noqa: BLE001
            payload["theses"] = []
        self.loaded.emit(payload)


class _CaptureWorker(QThread):
    """Reads one entry's stored bar window off the GUI thread."""

    loaded = Signal(str, dict)

    def __init__(self, service, entry_id: str, parent=None) -> None:
        super().__init__(parent)
        self._service = service
        self._entry_id = entry_id

    def run(self) -> None:  # pragma: no cover - exercised through its signal seam
        try:
            capture = self._service.chart_capture(self._entry_id) or {}
        except Exception:  # noqa: BLE001
            capture = {}
        self.loaded.emit(self._entry_id, capture)


class MarketJournalPanel(QFrame):
    """What the trader thought, beside what the machine measured."""

    statusChanged = Signal(str)
    symbolActivated = Signal(str)

    def __init__(self, service=None, parent=None) -> None:
        super().__init__(parent)
        if service is None:
            from ui.services.market_journal_service import shared_journal_service

            service = shared_journal_service()
        self.service = service
        self._worker: _EntriesWorker | None = None
        self._capture_worker: _CaptureWorker | None = None
        self._entries: list[dict] = []
        self._digests: dict[str, dict] = {}
        self._theses: list[dict] = []
        self._loaded_once = False

        self.heading = QLabel("Market Journal")
        self.heading.setObjectName("SectionTitle")
        self.subtitle = QLabel(
            "What you thought, beside what the desk measured. The left-nav "
            "“Journal” page is the trade and tax record; this one is not."
        )
        self.subtitle.setObjectName("SectionSubtitle")
        self.subtitle.setWordWrap(True)

        # R4 A16: the picker, the Refresh, the timeframe box, the Save button
        # and the after-the-fact caption are OUT OF THE LAYOUT, not deleted -
        # the V2 idiom, and for the same reason: `reload()` and `_save()` still
        # read them, and nothing leaves the SCHEMA. Decision 0016 answer 11 is
        # "one box, one Enter", and V2 built that on the Desk tab and left this
        # page exactly as it was.
        self.session_picker = QComboBox()
        self.session_picker.setEditable(True)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.reload)

        self.entry_text = QPlainTextEdit()
        self.entry_text.setPlaceholderText(
            "What happened today, and what you make of it. Enter saves; "
            "Shift+Enter starts a new line."
        )
        self.entry_text.installEventFilter(self)
        self.timeframe_picker = QComboBox()
        self.save_button = QPushButton("Save entry")
        self.save_button.clicked.connect(self._save)
        self.after_the_fact = QLabel("")
        self.after_the_fact.setObjectName("CautionLabel")
        self.after_the_fact.setWordWrap(True)

        self.entries = QListWidget()
        self.entries.currentRowChanged.connect(self._on_entry_selected)
        self.timeline = QListWidget()
        self.agreement = QLabel("")
        self.agreement.setWordWrap(True)
        self.context_table = QTableWidget(0, 2)
        self.context_table.setHorizontalHeaderLabels(["Measured", "Value"])
        self.context_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.calendar_strip = QLabel("")
        self.calendar_strip.setWordWrap(True)
        # G3.2: the reader. The page's reason to exist is the WORDS, and until
        # now they were shown nowhere - the whole text went into a one-line
        # list item and was elided. Read-only, wrapped, selectable, styled from
        # `theme.qss` by object name (no `setStyleSheet` on the Qt thread).
        self.thought_meta = QLabel("")
        self.thought_meta.setObjectName("ThoughtMeta")
        self.thought_meta.setWordWrap(True)
        self.thought_meta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.thought_view = QTextBrowser()
        self.thought_view.setObjectName("ThoughtReader")
        self.thought_view.setReadOnly(True)
        self.thought_view.setLineWrapMode(QTextBrowser.WidgetWidth)
        self.thought_view.setOpenExternalLinks(False)
        self.thought_view.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        # WS-10D: the story of the selected session, and the theses read out of
        # it. Read-only; the ONE editable thing here is the interpretation box,
        # and what it writes is a NEW row that supersedes the machine's draft.
        self.story_view = QTextBrowser()
        self.story_view.setObjectName("StoryReader")
        self.story_view.setReadOnly(True)
        self.story_view.setOpenExternalLinks(False)
        self.story_view.setOpenLinks(False)
        self.story_view.anchorClicked.connect(self._on_story_anchor)
        self.theses_list = QListWidget()
        self.theses_list.currentRowChanged.connect(self._on_thesis_selected)
        self.thesis_questions = QLabel("")
        self.thesis_questions.setObjectName("ThesisQuestions")
        self.thesis_questions.setWordWrap(True)
        self.thesis_questions.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.interpretation_box = QPlainTextEdit()
        self.interpretation_box.setPlaceholderText(
            "What you actually meant. Saving writes a NEW row; the machine's "
            "reading and your original note both stay exactly as they are."
        )
        self.save_interpretation_button = QPushButton("Save interpretation")
        self.save_interpretation_button.clicked.connect(self._save_interpretation)

        self.charts_note = QLabel("")
        self.charts_note.setWordWrap(True)
        self.digest_label = QLabel("")
        self.digest_label.setWordWrap(True)
        self.digest_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        import market_journal

        self.timeframe_picker.addItems(list(market_journal.TIMEFRAMES))
        self.timeframe_picker.setCurrentText(market_journal.TIMEFRAME_D1)

        compose = QVBoxLayout()
        compose.addWidget(QLabel("New entry"))
        compose.addWidget(self.entry_text, 1)
        compose_widget = QWidget()
        compose_widget.setLayout(compose)

        review = QVBoxLayout()
        review.addWidget(QLabel("Entries, newest first"))
        review.addWidget(self.entries, 2)
        review.addWidget(QLabel("Environment timeline"))
        review.addWidget(self.agreement)
        review.addWidget(self.timeline, 1)
        review.addWidget(QLabel("What the desk measured that session"))
        review.addWidget(self.context_table, 1)
        review.addWidget(self.calendar_strip)
        review_widget = QWidget()
        review_widget.setLayout(review)

        # The right half: the tape the selected entry was written against.
        # Charts, not a note about charts - the whole point of the capture.
        # G7.3: the four panes are BUILT on the first capture that needs them,
        # not here. Four `CandleChart`s is the page's whole construction cost
        # (299 ms in the G0 baseline), and the trader sees a chart only once
        # they click an entry that has one - most sessions, never.
        self.charts: dict[str, CandleChart] = {}
        self.chart_titles: dict[str, QLabel] = {}
        self.chart_holders: dict[str, QWidget] = {}
        self._charts_layout = QGridLayout()
        self._charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout = self._charts_layout
        charts_widget = QWidget()
        charts_body = QVBoxLayout(charts_widget)
        charts_body.setContentsMargins(0, 0, 0, 0)
        charts_body.addWidget(QLabel("What you were looking at"))
        charts_body.addWidget(self.charts_note)
        charts_body.addWidget(self.digest_label)
        charts_body.addLayout(charts_layout, 1)

        # G3 fix round: the words get a MEASURE. Left-aligned inside the pane
        # with the slack on the right - a stretch after each row, so the pane
        # keeps its full width and the text stops at about 100 characters.
        self.refresh_reader_measure()
        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.addWidget(self.thought_meta, 1)
        meta_row.addStretch(0)
        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.addWidget(self.thought_view, 1)
        view_row.addStretch(0)

        reader_widget = QWidget()
        reader_body = QVBoxLayout(reader_widget)
        reader_body.setContentsMargins(0, 0, 0, 0)
        reader_body.addWidget(QLabel("The thought, in full"))
        reader_body.addLayout(meta_row)
        reader_body.addLayout(view_row, 1)

        # WS-10D: the story beside the theses. The story is the session's
        # sequence - what you said, what the market did, where each came from -
        # and the theses are what a note CLAIMED, with the questions that claim
        # raises. Side by side because one is read and the other is answered.
        story_side = QVBoxLayout()
        story_side.setContentsMargins(0, 0, 0, 0)
        story_side.addWidget(QLabel("The story of this session"))
        story_side.addWidget(self.story_view, 1)
        story_holder = QWidget()
        story_holder.setLayout(story_side)

        thesis_side = QVBoxLayout()
        thesis_side.setContentsMargins(0, 0, 0, 0)
        thesis_side.addWidget(QLabel("Active theses"))
        thesis_side.addWidget(self.theses_list, 1)
        thesis_side.addWidget(self.thesis_questions)
        thesis_side.addWidget(self.interpretation_box, 1)
        thesis_side.addWidget(self.save_interpretation_button)
        thesis_holder = QWidget()
        thesis_holder.setLayout(thesis_side)

        story_widget = QSplitter(Qt.Horizontal)
        story_widget.addWidget(story_holder)
        story_widget.addWidget(thesis_holder)
        story_widget.setSizes([1400, 1000])

        # G3.2: the right half is now READER over CHARTS. The charts are the
        # follow-on evidence; the trader opens this page to read what they
        # wrote, so the words get the top of the column and the 2 x 2 grid
        # keeps the larger share below it. Draggable either way.
        right = QSplitter(Qt.Vertical)
        right.addWidget(reader_widget)
        right.addWidget(story_widget)
        right.addWidget(charts_widget)
        right.setStretchFactor(0, 2)
        right.setStretchFactor(1, 3)
        right.setStretchFactor(2, 3)
        # Stretch alone only governs RESIZES; the opening split comes from the
        # size hints, and an empty chart grid hints far larger than a paragraph
        # of text - which opened the reader at a couple of lines. These are
        # proportions, not pixels: QSplitter scales them to the real height and
        # honours each child's minimum. Measured 796 / 1194 at 3456 x 2160.
        right.setSizes([700, 700, 900])

        # NOTE: this pair is LEFT vs RIGHT and is not the pair above.
        lower = QSplitter(Qt.Horizontal)
        lower.addWidget(review_widget)
        lower.addWidget(right)
        lower.setStretchFactor(0, 2)
        lower.setStretchFactor(1, 3)
        # G3 fix round: 2:3 in STRETCH is only how a resize is shared, and the
        # size hints opened the entries column at 655 px - where the 90-character
        # excerpt clipped after about 35 and the list said nothing. These are
        # proportions, not pixels (QSplitter scales them to the real width), so
        # the column opens at a third of the page. The handle still moves.
        lower.setSizes(list(LOWER_SPLIT_SHARES))

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(compose_widget)
        splitter.addWidget(lower)
        splitter.setStretchFactor(1, 3)
        # An empty box was opening at roughly half the page on a tall screen,
        # because a stretch factor alone gives the composer a share of every
        # resize. Four text lines is the initial size; the handle still moves.
        compose_height = self.entry_text.fontMetrics().lineSpacing() * COMPOSER_LINES + 48
        splitter.setSizes([compose_height, compose_height * 6])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.heading)
        layout.addWidget(self.subtitle)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status)

        self.service.statusChanged.connect(self.status.setText)
        # Both refreshes are gated on the page having been opened at least
        # once. A note typed on the Desk tab must reach this page - that is the
        # wiring the second service instance was breaking - but a page nobody
        # has looked at yet has nothing to refresh, and `showEvent` will load
        # it in full when they do. Otherwise every note would cost a worker
        # thread reading the ledger for a hidden widget.
        self.service.entryWritten.connect(lambda _row: self._refresh_if_loaded())
        capture_signal = getattr(self.service, "chartCaptured", None)
        if capture_signal is not None:
            capture_signal.connect(lambda _result: self._refresh_if_loaded())
        self._sync_after_the_fact()
        self._clear_charts("Select an entry to see the charts it was written against.")

    # -- session ----------------------------------------------------------
    def session_date(self) -> str:
        """The session a note typed NOW is about. COMPUTED (R4 A16).

        The picker is gone, so this is `market_journal.session_date_for` - the
        same function the Desk tab's box uses, which is what makes a note filed
        from either surface land on the same day. A calendar that cannot answer
        falls back to today, because a note that could not be filed is a lost
        thought.
        """
        import market_journal

        try:
            return market_journal.session_date_for()
        except Exception:  # noqa: BLE001
            return date.today().isoformat()

    def eventFilter(self, watched, event):  # noqa: N802 (Qt override)
        """Enter saves, Shift+Enter makes a newline - answer 11's "one Enter".

        An event filter rather than a `QShortcut`, exactly as the Desk tab's box
        does it: a shortcut on Return would fire for every widget in this page's
        scope, and this key means "save" only while the cursor is in this box.
        """
        try:
            from PySide6.QtCore import QEvent

            if (
                watched is self.entry_text
                and event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            ):
                self._save()
                return True
        except Exception:  # noqa: BLE001 - a key handler never breaks the page
            pass
        return super().eventFilter(watched, event)

    def _sync_after_the_fact(self) -> None:
        """Say plainly when the entry being typed is about a past session.

        Decision record §5a: an entry about Friday written on Saturday says so.
        A reader weighing "what did you think at the time?" needs to know it was
        not written at the time, and the trader should see that before they
        write, not after.
        """
        session = self.session_date()
        today = date.today().isoformat()
        if session and session < today:
            self.after_the_fact.setText(
                f"This entry is ABOUT {session} and will be stamped as written "
                f"today ({today}). It is filed under the session, never backdated."
            )
        else:
            self.after_the_fact.setText("")

    # -- loading ----------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Load the first time the page is actually looked at.

        The page shipped with no caller for `reload()` at all, so it was empty
        until "Refresh" was pressed - which read as an empty journal on a day
        with five entries in it. Loading here rather than in `__init__` keeps
        the cost with the page that asked for it: the desk builds every left-nav
        panel at startup and most are never opened.
        """
        super().showEvent(event)
        if not self._loaded_once:
            self._loaded_once = True
            self.reload()

    def _refresh_if_loaded(self) -> None:
        if self._loaded_once:
            self.reload()

    def reload(self) -> None:
        self._sync_after_the_fact()
        if self._worker is not None and self._worker.isRunning():
            return
        self._worker = _EntriesWorker(self.service, self.session_date(), self)
        self._worker.loaded.connect(self._render)
        self._worker.start()

    def _render(self, payload: dict) -> None:
        if payload.get("error"):
            self.status.setText(f"Market journal unavailable: {payload['error']}")
            return
        self._digests = dict(payload.get("digests") or {})
        self._render_sessions(payload.get("sessions") or [])
        self._render_entries(payload.get("entries") or [])
        self._render_timeline(payload.get("timeline") or {})
        self._render_context(payload.get("context") or {})
        self._render_calendar()
        self._render_story(payload.get("story"))
        self._render_theses(payload.get("theses") or [])

    def _render_sessions(self, sessions: list[str]) -> None:
        current = self.session_picker.currentText()
        known = {self.session_picker.itemText(i) for i in range(self.session_picker.count())}
        blocked = self.session_picker.blockSignals(True)
        try:
            for session in sessions:
                if session not in known:
                    self.session_picker.addItem(session)
            # An empty box means "today" everywhere else in this class, so it
            # is filled in rather than left to whatever addItem selected -
            # otherwise adding the first session silently changes which one is
            # being read.
            self.session_picker.setCurrentText(current or self.session_date())
        finally:
            self.session_picker.blockSignals(blocked)

    def _render_entries(self, entries: list[dict]) -> None:
        import market_journal

        previous = self._selected_entry_id()
        # R4 A16: NEWEST FIRST, and every session in one list. The page used to
        # show one session at a time behind a picker, so reading back "what did
        # I think last week" meant knowing the date first.
        self._entries = sorted(
            entries,
            key=lambda row: (
                str(row.get("session_date") or ""),
                str(row.get("created_at") or ""),
            ),
            reverse=True,
        )
        entries = self._entries
        blocked = self.entries.blockSignals(True)
        try:
            self.entries.clear()
            if not entries:
                self.entries.addItem("No entries for this session yet.")
            for entry in entries:
                marker = (
                    " [written after the session]"
                    if entry.get("written_after_the_session")
                    else ""
                )
                hand = " [desk]" if market_journal.is_machine_entry(entry) else ""
                symbols = ", ".join(entry.get("symbols") or ())
                camera = " 📈" if str(entry.get("entry_id") or "") in self._digests else ""
                # DATED by the session it is ABOUT, which is the question the
                # picker used to answer. `created_at` moves to the tooltip.
                session = str(entry.get("session_date") or "")[:10]
                # G3.1: an EXCERPT, not the whole text. The date stays first
                # and the two-space separator stays - both are pinned by
                # `test_the_entries_list_is_dated_and_newest_first`.
                label = (
                    f"{session}  {entry.get('timeframe', '')}{hand}{marker}{camera} "
                    f"{('[' + symbols + '] ') if symbols else ''}"
                    f"{_excerpt(entry.get('text', ''))}"
                )
                item = QListWidgetItem(label)
                item.setToolTip(f"written {str(entry.get('created_at') or '')[:19]}")
                item.setData(Qt.UserRole, str(entry.get("entry_id") or ""))
                self.entries.addItem(item)
        finally:
            self.entries.blockSignals(blocked)
        if not entries:
            # Cleared under blocked signals, so the charts would otherwise keep
            # drawing the previous session's tape under this session's silence.
            # G3.3: and the reader would keep yesterday's words under it.
            self._fill_reader(None)
            self._clear_charts("No entries for this session, so there is nothing to chart.")
            return
        row = self._row_for_entry(previous)
        # Newest first, so the newest entry is row ZERO.
        self.entries.setCurrentRow(row if row is not None else 0)
        # setCurrentRow is a no-op when the row is already current (a reload
        # that changed nothing), and the charts must still be right.
        self._on_entry_selected(self.entries.currentRow())

    def _row_for_entry(self, entry_id: str) -> int | None:
        if not entry_id:
            return None
        for index, entry in enumerate(self._entries):
            if str(entry.get("entry_id") or "") == entry_id:
                return index
        return None

    def _selected_entry_id(self) -> str:
        item = self.entries.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.UserRole) or "")

    def _render_timeline(self, timeline: dict) -> None:
        self.timeline.clear()
        for shift in timeline.get("shifts") or []:
            self.timeline.addItem(
                f"{str(shift.get('event_at') or '')[:19]} "
                f"{shift.get('from_regime', '')} -> {shift.get('to_regime', '')} "
                f"({shift.get('source', '')})"
            )
        agreement = timeline.get("agreement") or {}
        if agreement.get("rate") is None:
            self.agreement.setText(
                f"Auto-vs-manual agreement: UNMEASURED - {agreement.get('note', '')}"
            )
        else:
            self.agreement.setText(
                f"Auto-vs-manual agreement: {agreement['rate'] * 100:.0f}% over "
                f"{agreement.get('sessions_compared', 0)} session(s). "
                f"{agreement.get('note', '')}"
            )

    def _render_context(self, context: dict) -> None:
        self.context_table.setRowCount(0)
        if not context.get("measured"):
            self.context_table.setRowCount(1)
            self.context_table.setItem(0, 0, QTableWidgetItem("(absent)"))
            self.context_table.setItem(
                0, 1, QTableWidgetItem(str(context.get("reason") or "unmeasured"))
            )
            return
        row = context.get("row") or {}
        fields = [
            (name, value)
            for name, value in sorted(row.items())
            if name not in {"schema", "event_type", "writer_host", "writer_pid", "run_id"}
        ]
        self.context_table.setRowCount(len(fields))
        for index, (name, value) in enumerate(fields):
            self.context_table.setItem(index, 0, QTableWidgetItem(str(name)))
            self.context_table.setItem(index, 1, QTableWidgetItem(str(value)))

    def _render_calendar(self) -> None:
        try:
            import market_context_ledger

            overlay = market_context_ledger.load_calendar_overlay()
            coverage = market_context_ledger.calendar_coverage(overlay)
        except Exception as exc:  # noqa: BLE001
            self.calendar_strip.setText(f"Calendar coverage unavailable: {exc}")
            return
        self.calendar_strip.setText(f"Calendar: {coverage.get('note', '')}")

    # -- the reader ---------------------------------------------------------
    def refresh_reader_measure(self) -> None:
        """Recompute the reader's pixel cap from the CURRENT font (G3b item 3).

        `_reader_measure` ran once, in `__init__`, so a scale change re-applied
        every other Python-side pixel budget (`MainWindow._apply_scaled_metrics`
        exists for exactly that) but left this pane's old cap in place. Callable
        on its own so that method can call it too. Polished first: the theme
        sizes fonts in the stylesheet, so an unpolished widget would be
        measured in the default font and the column would not follow the
        theme it claims to follow.
        """
        self.thought_view.ensurePolished()
        reader_measure = _reader_measure(self.thought_view.fontMetrics())
        self.thought_meta.setMaximumWidth(reader_measure)
        self.thought_view.setMaximumWidth(reader_measure)

    def _entry_for_id(self, entry_id: str) -> dict | None:
        """The entry behind an `entry_id`, keyed the same way every row is.

        G3b item 2: this used to be `_entry_for_row`, keyed by the QListWidget's
        ROW INDEX into `self._entries` - which lines up only because every row
        today IS an entry. A future header or grouping row would desync the
        two lists silently and put one entry's words under another's
        selection - the worst failure this page could have (see
        `_fill_reader`'s own docstring). Keying by the `Qt.UserRole` id every
        row already carries removes that dependency entirely.
        """
        index = self._row_for_entry(entry_id)
        return self._entries[index] if index is not None else None

    def _fill_reader(self, entry: dict | None) -> None:
        """G3.2/G3.3 - the full thought, written from the ENTRY.

        Never from the capture worker: `_render_capture` can arrive for a row
        the trader has already left (that is what the late-capture guard is
        for), and one entry's words under another entry's selection is the
        worst failure this page could have.

        `setPlainText`, never `setHtml`: the trader's own words are text, and a
        thought containing `<` is not markup.
        """
        if not entry:
            self.thought_meta.setText("")
            self.thought_view.setPlainText("")
            return
        import market_journal

        parts = [
            part
            for part in (
                str(entry.get("session_date") or "")[:10],
                str(entry.get("timeframe") or ""),
                _written_line(entry.get("created_at")),
            )
            if part
        ]
        if market_journal.is_machine_entry(entry):
            parts.append("[desk]")
        if entry.get("written_after_the_session"):
            parts.append("[written after the session]")
        symbols = ", ".join(entry.get("symbols") or ())
        if symbols:
            parts.append(symbols)
        self.thought_meta.setText("  ·  ".join(parts))
        self.thought_view.setPlainText(str(entry.get("text") or ""))

    # -- the story and the theses (WS-10D) --------------------------------
    def _render_story(self, story: Any) -> None:
        """The three kinds, visibly distinct, in reading order.

        `setHtml` rather than `setPlainText` here, and ONLY here: the sources
        are links the trader clicks to reach the entry they name, which a plain
        block of text cannot be. Everything the trader wrote is escaped before
        it goes in, so a thought containing `<` is still a thought.
        """
        if story is None:
            self.story_view.setHtml(
                f"<p>{_html('No story could be built for this session.')}</p>"
            )
            return
        get = story.get if isinstance(story, dict) else lambda name, default=None: getattr(
            story, name, default
        )
        session = str(get("session_date", "") or "")
        parts: list[str] = [f"<h3>{_html(STORY_TRADER_HEADING)}</h3>"]
        said = list(get("trader_said", ()) or ())
        if said:
            for row in said:
                parts.append(
                    "<p>"
                    + _html(_story_entry_prefix(row))
                    + "<br/>"
                    + _html(str(row.get("text") or "")).replace("\n", "<br/>")
                    + "</p>"
                )
        else:
            parts.append(
                "<p><i>"
                + _html(
                    f"No note was written for {session}. Nothing below is a view "
                    "you held - it is only what the desk measured."
                )
                + "</i></p>"
            )

        forecasts = list(get("external_forecasts", ()) or ())
        if forecasts:
            parts.append(f"<h3>{_html(STORY_FORECAST_HEADING)}</h3>")
            for row in forecasts:
                parts.append(
                    "<p>"
                    + _html("Imported, not yours: ")
                    + _html(str(row.get("text") or "")).replace("\n", "<br/>")
                    + "</p>"
                )

        parts.append(f"<h3>{_html(STORY_MEASURED_HEADING)}</h3>")
        for cell in list(get("measured", ()) or ()):
            parts.append("<p>" + _html(_measured_line(cell)) + "</p>")

        parts.append(f"<h3>{_html(STORY_SOURCES_HEADING)}</h3>")
        sources = dict(get("sources", {}) or {})
        entry_ids = list(sources.get("entry_ids") or ())
        captured = set(sources.get("capture_entry_ids") or ())
        if entry_ids:
            links = [
                f'<a href="{STORY_ENTRY_SCHEME}:{_html(str(entry_id))}">{_html(str(entry_id))}</a>'
                + (_html(" (with charts)") if entry_id in captured else "")
                for entry_id in entry_ids
            ]
            parts.append("<p>" + "<br/>".join(links) + "</p>")
        else:
            parts.append("<p>" + _html("No journal entry was written for this session.") + "</p>")
        context_id = str(sources.get("context_row_id") or "")
        if context_id:
            parts.append("<p>" + _html(f"Desk context row: {context_id}") + "</p>")
        for note in list(get("notes", ()) or ()):
            parts.append("<p><i>" + _html(str(note)) + "</i></p>")
        self.story_view.setHtml("".join(parts))

    def _on_story_anchor(self, url) -> None:
        """A source link selects the entry it names (WS-10D item 3).

        The whole reason the story lists its sources by id is so the trader can
        get from a sentence back to the thing it was built from. Selecting the
        entry is also what draws its capture, so one click answers both "which
        note was that?" and "what did it look like?".
        """
        try:
            raw = url.toString()
        except AttributeError:
            raw = str(url or "")
        _scheme, _, entry_id = raw.partition(":")
        entry_id = entry_id.strip()
        if not entry_id:
            return
        row = self._row_for_entry(entry_id)
        if row is None:
            self.status.setText(f"{entry_id} is not in the list on the left.")
            return
        self.entries.setCurrentRow(row)
        self._on_entry_selected(row)

    def _render_theses(self, theses: list[dict]) -> None:
        self._theses = [dict(row) for row in theses or ()]
        blocked = self.theses_list.blockSignals(True)
        try:
            self.theses_list.clear()
            for row in self._theses:
                item = QListWidgetItem(_thesis_label(row))
                item.setData(Qt.UserRole, str(row.get("thesis_id") or ""))
                self.theses_list.addItem(item)
        finally:
            self.theses_list.blockSignals(blocked)
        if not self._theses:
            self.thesis_questions.setText(
                "No note on this session, so there is no claim to question."
            )
            self.interpretation_box.setPlainText("")
            return
        self.theses_list.setCurrentRow(0)
        self._on_thesis_selected(self.theses_list.currentRow())

    def _current_thesis(self) -> dict | None:
        row = self.theses_list.currentRow()
        if row is None or row < 0 or row >= len(self._theses):
            return None
        return self._theses[row]

    def _on_thesis_selected(self, _row: int) -> None:
        thesis = self._current_thesis()
        if thesis is None:
            self.thesis_questions.setText("")
            self.interpretation_box.setPlainText("")
            return
        questions = [str(item) for item in (thesis.get("questions") or ()) if str(item).strip()]
        self.thesis_questions.setText("  ".join(questions))
        # The trader's own reading when there is one, an empty box when the
        # machine's draft is all there is. Never pre-filled with the draft:
        # editing a machine sentence would make it look like the trader's.
        self.interpretation_box.setPlainText(str(thesis.get("text") or ""))

    def _save_interpretation(self) -> None:
        thesis = self._current_thesis()
        if thesis is None:
            self.status.setText("Select a thesis before saving an interpretation.")
            return
        result = self.service.save_interpretation(
            entry_id=str(thesis.get("entry_id") or ""),
            supersedes=str(thesis.get("thesis_id") or ""),
            text=self.interpretation_box.toPlainText(),
        )
        if result.get("ok"):
            self.status.setText(
                "Interpretation saved. The machine's draft and your note are untouched."
            )
            self._refresh_if_loaded()
        else:
            self.status.setText(f"Interpretation NOT saved: {result.get('reason', '')}")

    # -- the captured charts ----------------------------------------------
    def _on_entry_selected(self, _row: int) -> None:
        # G3.3: the WORDS FIRST, synchronously, at the head of the method -
        # before the no-capture guard below returns early and before any
        # worker is constructed. An entry with no capture is still readable.
        # G3b item 2: keyed by `entry_id` (the same id the row's Qt.UserRole
        # and the charts both use), never by `_row` - a row index into
        # `self._entries` is only right while every row is an entry.
        entry_id = self._selected_entry_id()
        self._fill_reader(self._entry_for_id(entry_id))
        if not entry_id:
            self._clear_charts("Select an entry to see the charts it was written against.")
            return
        digest = self._digests.get(entry_id)
        if digest is None:
            self._clear_charts(
                "No charts were captured with this entry. Entries written "
                "before the capture was built, and entries written when no "
                "bars were cached, have none - it is not a chart that was lost."
            )
            return
        self.digest_label.setText(str(digest.get("digest") or ""))
        self.charts_note.setText("Loading the captured charts…")
        if self._capture_worker is not None and self._capture_worker.isRunning():
            return
        self._capture_worker = _CaptureWorker(self.service, entry_id, self)
        self._capture_worker.loaded.connect(self._render_capture)
        self._capture_worker.start()

    def _ensure_charts(self) -> None:
        """Build the four capture panes, once, the first time one is needed.

        G7.3. A `CandleChart` is a pyqtgraph plot and four of them were the
        page's construction cost; a page with no entry selected shows no chart
        at all, so nothing here belongs in `__init__`. Every reader of
        `self.charts` tolerates the empty dict, which is what "not built yet"
        looks like before the first capture.
        """
        if self.charts:
            return
        for index, (key, _template, _timeframe) in enumerate(CAPTURE_PANES):
            title = QLabel("")
            title.setObjectName("SectionSubtitle")
            chart = CandleChart()
            chart.setMinimumHeight(160)
            self.chart_titles[key] = title
            self.charts[key] = chart
            pane = QVBoxLayout()
            pane.setContentsMargins(0, 0, 0, 0)
            pane.addWidget(title)
            pane.addWidget(chart, 1)
            holder = QWidget()
            holder.setLayout(pane)
            self.chart_holders[key] = holder
            self._charts_layout.addWidget(holder, index // 2, index % 2)

    def _clear_charts(self, note: str) -> None:
        self.charts_note.setText(note)
        self.digest_label.setText("")
        # Nothing built yet is already cleared: the loop over an empty dict is
        # the "not built" case, not a missed reset.
        for key, chart in self.charts.items():
            chart.set_data([])
            self.chart_titles[key].setText("")
            self.chart_holders[key].setVisible(False)

    def _render_capture(self, entry_id: str, capture: dict) -> None:
        if entry_id != self._selected_entry_id():
            # The trader moved on while the file was being read. Drawing it now
            # would put one entry's tape under another entry's words.
            return
        if not capture:
            self._clear_charts(
                "This entry has a capture row but its stored bars could not be "
                "read. The row is on disk; the bar file is not."
            )
            return
        import market_journal_capture

        # G7.3: the first capture that has something to draw is what pays for
        # the four panes.
        self._ensure_charts()
        symbol = str(capture.get("symbol") or "").strip().upper() or "(no symbol)"
        benchmark = str(capture.get("benchmark") or market_journal_capture.BENCHMARK_SYMBOL)
        series = capture.get("series") or {}
        missing = 0
        for key, template, timeframe in CAPTURE_PANES:
            stored = series.get(key) or []
            bars = market_journal_capture.revive_bars(stored)
            missing += len(stored) - len(bars)
            # A pane with nothing stored is HIDDEN, not drawn empty: an
            # auto-mode flip captures SPY alone, and four axes where two of
            # them never had a chart reads as two failed charts.
            self.chart_holders[key].setVisible(bool(bars))
            if not bars:
                continue
            self.chart_titles[key].setText(
                f"{template.format(symbol=symbol, benchmark=benchmark)} — {len(bars)} bars"
            )
            self.charts[key].set_data(bars, timeframe=timeframe)
        reason = str(capture.get("reason") or "")
        note = str(capture.get("note") or "")
        stamp = str(capture.get("captured_at") or "")[:19]
        gap = f" {missing} stored bar(s) had no readable stamp and are not drawn." if missing else ""
        self.charts_note.setText(
            f"Captured {stamp} ({reason}){(' — ' + note) if note else ''}.{gap}"
        )

    def shutdown(self) -> None:
        for worker in (self._worker, self._capture_worker):
            if worker is not None and worker.isRunning():
                worker.wait(2000)
        # A capture killed mid-write leaves a `.tmp` and never a torn sidecar
        # (temp file + replace), so this is politeness rather than correctness -
        # but a note the trader typed seconds before closing the desk should
        # keep its charts.
        waiter = getattr(self.service, "wait_for_captures", None)
        if callable(waiter):
            waiter(2000)

    # -- writing ----------------------------------------------------------
    def _save(self) -> None:
        result = self.service.write_entry(
            text=self.entry_text.toPlainText(),
            session_date=self.session_date(),
            timeframe=self.timeframe_picker.currentText(),
            origin="journal_page",
        )
        if result.get("ok"):
            self.entry_text.clear()
