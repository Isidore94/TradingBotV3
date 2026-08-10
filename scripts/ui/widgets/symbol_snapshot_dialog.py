from __future__ import annotations

"""Click-a-ticker quick look: D1 candles on top, M5 candles below.

D1 comes from the durable daily parquet store (always available offline);
M5 from BounceBot's cached bars (only names in the current scan set). Both
are local reads, so the popup fills synchronously on click. Overlays are
fixed per the desk's reading style: D1 = SMA50/100/200 + EMA8/15/21, M5 =
session VWAP with +/-1 sigma bands + EMA15/21 - just the candles otherwise.
"""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import logging
import math
import threading
from datetime import datetime, timedelta

import chart_levels
from chart_watch import D1_EVENT_KINDS, WATCH_KINDS
from ui import theme
from ui.widgets.candle_chart import CandleChart
from ui.widgets.paint_lines_button import PaintLinesButton

#: Per-symbol backfill cooldown, shared across every snapshot widget so two
#: open charts of one stale symbol cannot double-fetch, and a holiday (which
#: the session calendar cannot see) costs at most one probe per window.
_D1_BACKFILL_ATTEMPTS: dict[str, datetime] = {}
_D1_BACKFILL_COOLDOWN = timedelta(minutes=10)

#: Today's forming daily candle for symbols the bot is not scanning (trader
#: rule 2026-08-05: "I want to ALWAYS see the latest D1 candle as it's
#: forming intraday"). The durable store only gains a session's bar when a
#: scan touches the symbol, so a name typed into the chart box - RY, say -
#: showed a tail that stopped at yesterday with no hint that today's big
#: candle existed. This is a display-only fetch: the bar is cached in memory,
#: marked as a preview, and NEVER written to the durable store, so no partial
#: session can leak into research or detector history. Yahoo-sourced, so the
#: IB budget is untouched; it can lag the tape by a few minutes.
_FORMING_BARS: dict[str, tuple[datetime, dict]] = {}
_FORMING_ATTEMPTS: dict[str, datetime] = {}
_FORMING_REFRESH = timedelta(minutes=2)


def _last_row_as_bar(frame) -> dict | None:
    """Newest row of a daily-bar frame as a chart bar dict, or None.

    Deliberately tolerant: a provider hiccup returns an empty or short frame
    rather than raising, and a missing forming candle must degrade to "no
    preview", never to a broken chart.
    """
    if frame is None or getattr(frame, "empty", True):
        return None
    try:
        row = frame.iloc[-1]
        stamp = row["datetime"]
        stamp = stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp
        if not isinstance(stamp, datetime):
            return None
        return {
            "dt": stamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0) or 0.0),
        }
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return None


def _legend_html(
    title: str,
    overlays: list[dict],
    *,
    missing_reason: str = "needs deeper stored history",
) -> str:
    parts = [f"<b>{title}</b>"]
    seen = set()
    missing = []
    for overlay in overlays:
        label = str(overlay.get("label") or "")
        if not label or label in seen:
            continue
        seen.add(label)
        if not any(value is not None for value in overlay.get("values") or []):
            # e.g. SMA200 while the durable daily store is still shorter than
            # 200 sessions: say why the line is absent instead of lying.
            missing.append(label)
            continue
        color = theme.color(str(overlay.get("color") or "neutral"))
        parts.append(f"<span style='color:{color};'>— {label}</span>")
    if missing:
        parts.append(
            f"<span style='color:{theme.color('text_muted')};'>"
            f"({', '.join(missing)}: {missing_reason})</span>"
        )
    # Keep each marker visually grouped, but leave ordinary spaces between
    # entries so a narrow popup can wrap instead of forcing one giant line.
    return " &nbsp; ".join(parts)


# How often a visible snapshot re-pulls its charts from the local stores /
# the bot's M5 cache. Matches the Alert Center's 30s watch tick: bars change
# at most every 5 minutes, so this bounds display staleness the same way.
REFRESH_INTERVAL_MS = 30_000


def _bars_fingerprint(bars: list) -> tuple | None:
    """Cheap change detector for a rendered bar series.

    (count, last stamp, last close) moves whenever a bar is appended OR the
    forming last bar updates in place - and refresh skips the full re-render
    (which would also reset the trader's pan/zoom) when it has not moved.
    """
    if not bars:
        return None
    last = bars[-1]
    return (len(bars), last.get("dt"), last.get("close"))


def _levels_fingerprint(levels: list) -> tuple:
    """Change detector for the painted levels.

    Needed alongside the bar fingerprint: a scan rewriting the level store
    changes which lines belong on the chart without moving a single candle,
    and the refresh guard would otherwise hold yesterday's lines all session.
    """
    return tuple(
        (str(level.get("id") or ""), level.get("price"))
        for level in levels or ()
    )


class SymbolSnapshotWidget(QWidget):
    """Reusable embedded D1-over-M5 snapshot view.

    Clicking a D1 candle opens a small menu offering a persistent level
    alert off that candle's high or low; the choice is emitted as
    ``d1LevelAlertRequested(symbol, direction, level, candle_date)`` for the
    hosting surface to arm (it stays on across sessions until it flags).
    """

    d1LevelAlertRequested = Signal(str, str, float, str)
    # Click-to-price from either chart, for hosts with a level box to fill.
    pricePicked = Signal(float)
    # (symbol, level_id, family, price) - the trader clicked a PAINTED level
    # on the D1 chart. Forwarded, not acted on: A4's obligation is that the
    # identity of the line is available to whatever wants to record it (the
    # capture rail's ref_level_id / ref_level_family). Nothing here arms,
    # scores, or suppresses anything.
    d1LevelSelected = Signal(str, str, str, float)
    # A background D1 backfill finished for this symbol; re-read the store.
    _d1BackfillDone = Signal(str)
    # (symbol) - charts just repainted. Hosts that gate controls on what the
    # charts hold (e.g. the arm dock's watch buttons, which need M5 bars) wait
    # on this instead of on a return value: the build is off-thread now, so
    # set_symbol/refresh have already returned by the time bars exist.
    snapshotRendered = Signal(str)
    # (symbol, worker meta). Hosts use this for provenance/freshness chrome;
    # it is emitted even when unchanged bars intentionally skip repainting.
    snapshotMetaChanged = Signal(str, object)

    def __init__(
        self,
        parent=None,
        *,
        compact: bool = False,
        d1_sessions: int | None = None,
        allow_alerts: bool = True,
    ) -> None:
        """``compact`` trades legend wrapping for chart height.

        The standalone popup is 1180px wide with height to spare, so its
        legends keep wrapping (a narrow popup showing one long unwrapped line
        was a real complaint). The desk's embedded pane is the opposite: it is
        height-starved, and wrapped legends measured 59px EACH at 2560x1440 -
        43% of the whole snapshot - so there it stays on one line.
        """
        super().__init__(parent)
        self._symbol = ""
        self._bot = None
        self._compact = bool(compact)
        self._d1_sessions = (
            max(1, int(d1_sessions)) if d1_sessions is not None else None
        )
        # Chart Review is a judgement-capture surface and passes False. The
        # shared charts and painted-level selection remain identical; only
        # candle-click alert menus and alert emission are disabled there.
        self._allow_alerts = bool(allow_alerts)
        self._d1_backfill_thread: threading.Thread | None = None
        self._forming_thread: threading.Thread | None = None
        self._d1BackfillDone.connect(self._on_d1_backfill_done)
        # Every chart build runs off the GUI thread (C3: no I/O here). The
        # worker pool is shared desk-wide, but this widget owns its delivery
        # object: a single global signal would offer every result to every
        # chart widget ever constructed, including ones already destroyed,
        # and a queued delivery into a dead receiver is an access violation.
        from ui.services.chart_data_service import ChartDataService, shared_pool

        self._data = ChartDataService(pool=shared_pool())
        self._data.snapshotReady.connect(self._on_snapshot_ready)
        self._data.snapshotFailed.connect(self._on_snapshot_failed)
        # Stop the service the moment this widget goes away, so a build still
        # in flight drops its result instead of emitting into the teardown.
        # The lambda captures the service only - capturing self here would
        # keep the widget alive past its own destruction.
        self.destroyed.connect(
            lambda _obj=None, service=self._data: service.shutdown()
        )
        # Latest snapshot dicts, retained so callers can quick-fill a price from
        # a drawn overlay (VWAP, +/-1 sigma). set_data plots overlays and drops
        # them, so without this the values are unrecoverable after rendering.
        self._d1: dict = {}
        self._m5: dict = {}
        # The whole snapshot must be Expanding: it is the thing that should eat
        # the column's spare height. It previously declared Preferred with
        # verticalStretch 0, so the chart pane's stretch factor never reached
        # the charts themselves.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        legend_v = QSizePolicy.Policy.Fixed if self._compact else QSizePolicy.Policy.Preferred
        self.d1_legend = QLabel()
        self.d1_legend.setTextFormat(Qt.TextFormat.RichText)
        self.d1_legend.setWordWrap(not self._compact)
        self.d1_legend.setSizePolicy(QSizePolicy.Policy.Expanding, legend_v)
        # One control for every line group on the chart, machine-local and
        # defaulting to all-on (A4). It sits on the D1 legend row because the
        # groups it governs are the ones the D1 legend names.
        self.paint_lines_button = PaintLinesButton(compact=self._compact)
        self.paint_lines_button.groupsChanged.connect(self._on_paint_lines_changed)
        self.d1_header = QWidget()
        header_layout = QHBoxLayout(self.d1_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addWidget(self.d1_legend, 1)
        header_layout.addWidget(self.paint_lines_button, 0)

        self.d1_chart = CandleChart()
        if self._allow_alerts:
            self.d1_chart.barClicked.connect(self._on_d1_bar_clicked)
        self.d1_chart.levelSelected.connect(self._on_d1_level_selected)
        self.d1_chart.setMinimumHeight(120)
        self.d1_note = QLabel()
        self.d1_note.setObjectName("MutedLabel")
        self.d1_note.setWordWrap(True)
        self.d1_note.setVisible(False)

        self.m5_legend = QLabel()
        self.m5_legend.setTextFormat(Qt.TextFormat.RichText)
        self.m5_legend.setWordWrap(not self._compact)
        self.m5_legend.setSizePolicy(QSizePolicy.Policy.Expanding, legend_v)
        self.m5_chart = CandleChart()
        # M5 candle clicks used to be inert: only the D1 chart was wired, so an
        # opening-range high, a premarket high, or any intraday level could not
        # be armed by clicking the bar that shows it.
        if self._allow_alerts:
            self.m5_chart.barClicked.connect(self._on_m5_bar_clicked)
        self.m5_chart.setMinimumHeight(120)
        for chart in (self.d1_chart, self.m5_chart):
            chart.priceClicked.connect(
                lambda _index, price: self.pricePicked.emit(price)
            )
        self.m5_note = QLabel()
        self.m5_note.setObjectName("MutedLabel")
        self.m5_note.setWordWrap(True)
        self.m5_note.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self.d1_header)
        layout.addWidget(self.d1_chart, 1)
        layout.addWidget(self.d1_note)
        layout.addWidget(self.m5_legend)
        layout.addWidget(self.m5_chart, 1)
        layout.addWidget(self.m5_note)

    def set_symbol(self, symbol: str, *, bot=None) -> None:
        """Show ``symbol``, painting whatever is already known immediately.

        Nothing here blocks (C3). If this symbol was charted before, its last
        build repaints at once and the worker's fresh one lands on top; if it
        was not, the charts say so until the bars arrive. Either way the GUI
        thread does no reading.
        """
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return
        switched = symbol != self._symbol
        self._symbol = symbol
        # Retained so refresh() can re-pull the M5 cache on a timer tick. The
        # hosting panel passes a fresh bot on its own ticks; this reference
        # only carries the popup between clicks.
        self._bot = bot
        known = self._data.last_snapshot(symbol)
        if known is not None:
            self._render_snapshots(known[0], known[1])
        elif switched:
            self._show_pending(symbol)
        self._request_snapshots()

    def _request_snapshots(self) -> bool:
        """Queue an off-thread rebuild of both charts for the current symbol.

        The M5 bars are read here, on the GUI thread, on purpose:
        ``m5_chart_bars`` is documented as an in-memory read of the bot's
        cache that never triggers a fetch, and passing them in keeps the
        worker from reaching into the bot across threads.
        """
        if not self._symbol:
            return False
        m5_bars = []
        if self._bot is not None:
            try:
                m5_bars = self._bot.m5_chart_bars(self._symbol, max_sessions=2)
            except Exception:
                m5_bars = []
        # The bot's cache is only rewritten when the scan loop reaches this
        # symbol (~28 min), so an alert reached late charts its scan-time bars.
        # Prefer a display-only refetch when one reaches further forward; it
        # never replaces a longer cached series with a shorter fresh one.
        try:
            from ui.services.chart_bar_refresh import shared_refresh_service

            m5_bars = shared_refresh_service().best_bars(self._symbol, m5_bars)
        except Exception:
            logging.debug("Chart M5 refresh lookup failed.", exc_info=True)
        # A symbol the bot is not scanning has no M5 cache to preview today's
        # daily candle from; a separately fetched today-bar stands in. It is a
        # daily bar, so it feeds ONLY the D1 build, never the M5 pane.
        forming = self._forming_bar(self._symbol)
        source = (
            "ibkr-cache"
            if m5_bars
            else "yfinance-fallback"
            if forming
            else "durable-store"
        )
        self._data.request(
            self._symbol,
            m5_bars,
            sessions=self._d1_sessions,
            d1_preview_bars=[forming] if forming else [],
            source=source,
        )
        return True

    def _show_pending(self, symbol: str) -> None:
        """Skeleton state for a symbol with nothing cached to show yet."""
        self.d1_chart.setVisible(False)
        self.m5_chart.setVisible(False)
        self.d1_legend.setText(f"<b>{symbol} · D1</b>")
        self.m5_legend.setText(f"<b>{symbol} · M5</b>")
        self.d1_note.setText(f"Loading {symbol} daily bars…")
        self.d1_note.setVisible(True)
        self.m5_note.setText(f"Loading {symbol} intraday bars…")
        self.m5_note.setVisible(True)

    def _on_snapshot_ready(self, symbol: str, d1: dict, m5: dict, meta: dict) -> None:
        if symbol != self._symbol:
            return  # a stale delivery, or another widget's symbol
        meta = dict(meta or {})
        self.snapshotMetaChanged.emit(symbol, meta)
        self._apply_freshness(symbol, meta)
        # Unchanged bars must not re-render: a repaint resets the trader's
        # pan/zoom, and the 30s tick would otherwise do that on every pass.
        if self._d1 and self._m5 and _bars_fingerprint(
            d1.get("bars") or []
        ) == _bars_fingerprint(self._d1.get("bars") or []) and _bars_fingerprint(
            m5.get("bars") or []
        ) == _bars_fingerprint(self._m5.get("bars") or []) and _levels_fingerprint(
            d1.get("levels") or []
        ) == _levels_fingerprint(self._d1.get("levels") or []):
            return
        self._render_snapshots(d1, m5)

    def _on_snapshot_failed(self, symbol: str) -> None:
        if symbol != self._symbol:
            return
        self.d1_note.setText(
            f"Could not build the {symbol} chart - see the log. "
            "The daily store may be unreachable."
        )
        self.d1_note.setVisible(True)
        self.d1_chart.setVisible(False)

    def _apply_freshness(self, symbol: str, meta: dict) -> None:
        """Act on the worker's freshness verdict.

        The probes themselves ran off-thread (they resolve the market session,
        which reads settings). All that is left here is starting the same
        background fetches as before - painting stays fetch-free, plan.md
        Milestone 8.
        """
        try:
            if meta.get("stale_store"):
                self._start_d1_backfill(symbol)
            elif meta.get("want_forming"):
                self._start_forming_fetch(symbol)
        except Exception:
            logging.debug("D1 freshness follow-up failed.", exc_info=True)

    def show_payload_snapshots(self, symbol: str, d1: dict, m5: dict) -> None:
        """Render prebuilt snapshots (Desk Link satellite path).

        The satellite has no local store, no bot cache, and no TWS: the
        payload IS the data. No backfill is triggered and refresh() has
        nothing to re-read, so the render is a frozen picture of what the
        main saw at alert time.
        """
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return
        self._symbol = symbol
        self._bot = None

        def normalized(snapshot: dict) -> dict:
            snapshot = dict(snapshot or {})
            snapshot.setdefault("bars", [])
            snapshot.setdefault("overlays", [])
            snapshot.setdefault("note", "")
            return snapshot

        self._render_snapshots(normalized(d1), normalized(m5))

    def refresh(self, *, bot=None) -> bool:
        """Queue a re-pull of both charts; render later, only on change.

        Returns whether a rebuild was REQUESTED, not whether one rendered -
        the build is off-thread now. Hosts that need to react to what the
        charts hold connect to ``snapshotRendered`` instead. Unchanged bars
        still skip the repaint, so a 30s tick never disturbs pan/zoom.
        """
        if not self._symbol:
            return False
        if bot is not None:
            self._bot = bot
        return self._request_snapshots()

    @staticmethod
    def _forming_bar(symbol: str) -> dict | None:
        """The cached forming daily bar for ``symbol``, if it is still today's."""
        entry = _FORMING_BARS.get(symbol)
        if entry is None:
            return None
        _fetched_at, bar = entry
        stamp = bar.get("dt")
        if not hasattr(stamp, "date") or stamp.date() != datetime.now().date():
            _FORMING_BARS.pop(symbol, None)  # yesterday's preview is not today's
            return None
        return bar

    def _start_forming_fetch(self, symbol: str) -> None:
        """Fetch today's forming daily bar off the GUI thread.

        Yahoo only (`fetch_daily_bars_from_yahoo`, not the caching
        `fetch_daily_bars` wrapper): the wrapper would merge and PERSIST the
        partial bar into the durable store, which is precisely what must not
        happen - a half-finished session is display material, never stored
        evidence. Painting stays fetch-free (plan.md Milestone 8); the same
        per-symbol cooldown pattern as the stale-store backfill keeps a dead
        provider from becoming a fetch loop.
        """
        if self._forming_thread is not None and self._forming_thread.is_alive():
            return
        now = datetime.now()
        attempted_at = _FORMING_ATTEMPTS.get(symbol)
        if attempted_at is not None and (now - attempted_at) < _FORMING_REFRESH:
            return
        _FORMING_ATTEMPTS[symbol] = now

        def worker() -> None:
            try:
                from ui.services.safe_import import master_avwap_legacy

                legacy = master_avwap_legacy()
                frame = legacy.fetch_daily_bars_from_yahoo(symbol, 5)
                bar = _last_row_as_bar(frame)
                if bar is not None and bar["dt"].date() == datetime.now().date():
                    _FORMING_BARS[symbol] = (datetime.now(), bar)
            except Exception:
                logging.warning(
                    "Forming D1 candle fetch failed for %s.", symbol, exc_info=True
                )
            try:
                self._d1BackfillDone.emit(symbol)
            except RuntimeError:
                pass  # widget deleted while the fetch ran

        self._forming_thread = threading.Thread(
            target=worker, name=f"d1-forming-{symbol}", daemon=True
        )
        self._forming_thread.start()

    def _start_d1_backfill(self, symbol: str) -> None:
        """Refresh the durable daily store for ``symbol`` off the GUI thread.

        Guards: one backfill per widget at a time, and a per-symbol cooldown
        shared across widgets so a dead provider (or a holiday, which the
        session calendar cannot see) cannot turn a stale chart into a fetch
        loop. fetch_daily_bars itself merges + persists to the durable store,
        which busts load_d1_bars' mtime cache, so the next refresh() simply
        sees the new tail.
        """
        if self._d1_backfill_thread is not None and self._d1_backfill_thread.is_alive():
            return
        now = datetime.now()
        attempted_at = _D1_BACKFILL_ATTEMPTS.get(symbol)
        if attempted_at is not None and (now - attempted_at) < _D1_BACKFILL_COOLDOWN:
            return
        _D1_BACKFILL_ATTEMPTS[symbol] = now

        def worker() -> None:
            try:
                from ui.services.safe_import import master_avwap_legacy

                calendar_days = (
                    260
                    if self._d1_sessions is None
                    else int(math.ceil(self._d1_sessions * 365 / 252))
                )
                master_avwap_legacy().fetch_daily_bars(
                    None, symbol, max(260, calendar_days)
                )
            except Exception:
                logging.warning("D1 store backfill failed for %s.", symbol, exc_info=True)
            try:
                self._d1BackfillDone.emit(symbol)
            except RuntimeError:
                pass  # widget deleted while the fetch ran

        self._d1_backfill_thread = threading.Thread(
            target=worker, name=f"d1-backfill-{symbol}", daemon=True
        )
        self._d1_backfill_thread.start()

    def _on_d1_backfill_done(self, symbol: str) -> None:
        if symbol == self._symbol:
            self.refresh()

    def _on_paint_lines_changed(self, _hidden_groups: list) -> None:
        """Re-apply the line filter without rebuilding or re-ranging anything.

        The snapshot already in hand holds every line, so hiding a group is a
        pure display decision: no worker round-trip, and - because it goes
        through set_overlays/set_levels rather than set_data - no reset of the
        pan and zoom the trader had set up before reaching for the control.
        """
        if not self._d1:
            return
        self._paint_d1_lines(self._d1)

    def _visible_d1_lines(self, d1: dict) -> tuple[list, list]:
        hidden = self.paint_lines_button.hidden_groups()
        return (
            chart_levels.visible_overlays(d1.get("overlays") or [], hidden),
            chart_levels.visible_levels(d1.get("levels") or [], hidden),
        )

    def _paint_d1_lines(self, d1: dict) -> None:
        """Push overlays + levels only, leaving the candles and view alone."""
        overlays, levels = self._visible_d1_lines(d1)
        self.d1_chart.set_overlays(overlays)
        self.d1_chart.set_levels(levels)
        self._set_d1_legend(d1, overlays)

    def _set_d1_legend(self, d1: dict, overlays: list) -> None:
        self.d1_legend.setText(_legend_html(f"{self._symbol} · D1", overlays))
        if not d1.get("bars"):
            return
        last_bar = d1["bars"][-1]
        stamp = last_bar["dt"].strftime("%m/%d")
        reach = f"{stamp} forming" if last_bar.get("preview") else f"through {stamp}"
        anchor_iso = str(d1.get("avwape_anchor") or "")
        if anchor_iso:
            # Which earnings the AVWAPE lines hang from - without it the
            # bands are just unexplained curves.
            reach += f" · AVWAPE from {anchor_iso[5:7]}/{anchor_iso[8:10]}"
        prev_iso = str(d1.get("avwape_prev_anchor") or "")
        if prev_iso:
            reach += f" · prev {prev_iso[5:7]}/{prev_iso[8:10]}"
        self.d1_legend.setText(
            self.d1_legend.text()
            + f" &nbsp; <span style='color:{theme.color('text_muted')};'>"
            + f"{reach}</span>"
        )

    @staticmethod
    def _staleness_badge(bars) -> str:
        """Loud age marker for M5 bars that are meaningfully behind the tape.

        Returns "" when the bars are current, so a healthy chart carries no
        extra furniture. Never raises: a legend decoration must not be able to
        cost the trader the chart itself.
        """
        try:
            from ui.services.chart_bar_refresh import STALE_AFTER, bars_age

            age = bars_age(bars)
            if age is None or age < STALE_AFTER:
                return ""
            minutes = int(age.total_seconds() // 60)
            return (
                f" &nbsp; <span style='color:{theme.color('caution')};'>"
                f"● bars {minutes} min behind</span>"
            )
        except Exception:
            logging.debug("Staleness badge failed.", exc_info=True)
            return ""

    def _render_snapshots(self, d1: dict, m5: dict) -> None:
        symbol = self._symbol
        self._d1 = d1
        overlays, levels = self._visible_d1_lines(d1)
        self.d1_chart.set_data(d1["bars"], overlays, timeframe="d1")
        self.d1_chart.set_levels(levels)
        self.d1_chart.setVisible(bool(d1["bars"]))
        self._set_d1_legend(d1, overlays)
        self.d1_note.setVisible(not d1["bars"])
        if not d1["bars"]:
            self.d1_note.setText(
                f"No daily store for {symbol} - it is outside the built universe "
                "(Universe tab rebuilds fill the store)."
            )

        self._m5 = m5
        self.m5_legend.setText(
            _legend_html(
                f"{symbol} · M5",
                m5["overlays"],
                missing_reason="needs positive cached volume",
            )
        )
        self.m5_chart.set_data(m5["bars"], m5["overlays"], timeframe="m5")
        self.m5_chart.setVisible(bool(m5["bars"]))
        self.m5_note.setVisible(not m5["bars"])
        if not m5["bars"]:
            self.m5_note.setText(
                f"No cached M5 bars for {symbol} - it is not in the current scan set, "
                "or the bot has not completed a scan cycle yet."
            )
        else:
            last = m5["bars"][-1]["dt"]
            legend = (
                self.m5_legend.text()
                + f" &nbsp; <span style='color:{theme.color('text_muted')};'>"
                + f"last bar {last.strftime('%m/%d %H:%M')}</span>"
            )
            # A stale chart must never be readable as a current one. The last
            # bar time was already here, but muted and easy to skim past; an
            # age this far behind gets said out loud instead.
            legend += self._staleness_badge(m5["bars"])
            self.m5_legend.setText(legend)
        self.snapshotRendered.emit(symbol)

    def quick_fill(self, source: str) -> float | None:
        """Resolve a quick-fill source against the M5 chart's drawn series.

        HOD/LOD/Last come from the intraday bars; VWAP and the sigma bands come
        from the overlays already plotted on them, so the number that fills is
        the line the trader is looking at.
        """
        from ui.widgets.arm_bar import quick_fill_value

        snapshot = self._m5 or {}
        return quick_fill_value(
            source, snapshot.get("bars") or [], snapshot.get("overlays") or []
        )

    def _on_d1_bar_clicked(self, index: int) -> None:
        self._popup_level_menu(self.d1_chart, index, "%m/%d")

    def _on_d1_level_selected(self, level_id: str, family: str, price: float) -> None:
        """Re-emit a painted-level click with the symbol attached."""
        if not self._symbol:
            return
        self.d1LevelSelected.emit(self._symbol, level_id, family, float(price))

    def selected_d1_level(self) -> dict | None:
        """The painted D1 level currently highlighted, if any.

        A pull-side companion to ``d1LevelSelected``: a capture rail that
        fills ``ref_level_id`` at write time reads it here rather than
        having to have been listening when the click happened.
        """
        chosen = self.d1_chart.selected_level_id()
        if not chosen:
            return None
        for level in self.d1_chart.drawn_levels():
            if str(level.get("id") or "") == chosen:
                return level
        return None

    def _on_m5_bar_clicked(self, index: int) -> None:
        self._popup_level_menu(self.m5_chart, index, "%m/%d %H:%M")

    def _popup_level_menu(self, chart, index: int, stamp_format: str) -> None:
        """Offer a persistent break-above/below alert off a clicked candle.

        Both timeframes route into the same persistent D1 level watch: the
        watch is a price level, not a bar, so an M5 candle's high is as valid
        an anchor as a daily candle's.
        """
        bar = chart.bar_at(index)
        if bar is None or not self._symbol:
            return
        stamp = bar["dt"].strftime(stamp_format)
        menu = QMenu(self)
        above = menu.addAction(f"Alert: break above {bar['high']:.2f} ({stamp} high)")
        above.triggered.connect(
            lambda: self._emit_level_alert(chart, "above", index)
        )
        below = menu.addAction(f"Alert: break below {bar['low']:.2f} ({stamp} low)")
        below.triggered.connect(
            lambda: self._emit_level_alert(chart, "below", index)
        )
        menu.popup(QCursor.pos())

    def request_d1_level_alert(self, direction: str, index: int) -> None:
        """Emit the persistent level alert for a clicked D1 candle's high/low."""
        self._emit_level_alert(self.d1_chart, direction, index)

    def request_m5_level_alert(self, direction: str, index: int) -> None:
        """Emit the persistent level alert for a clicked M5 candle's high/low."""
        self._emit_level_alert(self.m5_chart, direction, index)

    def _emit_level_alert(self, chart, direction: str, index: int) -> None:
        if not self._allow_alerts:
            return
        bar = chart.bar_at(index)
        if bar is None or not self._symbol or direction not in ("above", "below"):
            return
        level = float(bar["high"] if direction == "above" else bar["low"])
        self.d1LevelAlertRequested.emit(
            self._symbol, direction, level, bar["dt"].strftime("%Y-%m-%d")
        )


class SymbolSnapshotDialog(QDialog):
    """Non-modal two-chart snapshot, reused across clicks (one per panel).

    When opened with a ``watch_host`` (the Alert Center panel, or anything
    exposing ``armed_watch_kinds`` / ``arm_chart_watch_for`` /
    ``disarm_chart_watch_for`` / ``is_d1_focus_active`` /
    ``toggle_d1_focus`` / ``is_m5_focus`` / ``toggle_m5_focus``) the popup
    grows the chart-only action row: "Add to D1 Focus" (Swing Focus picks +
    D1 Focus feed pin) and "Add to M5 Focus" (day-trade list) toggles plus
    the one-shot watch toggles whose hits flag red in the Alert Center.
    Everything is a toggle - a second click removes or disarms. Without a
    host the popup stays a pure quick look.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.resize(1180, 760)
        self.watch_host = None
        # Review-flow host (the setups panel): supplies the popup's ✕ dislike
        # and the advance-to-next-chart follow-through, so the trader can
        # work the whole table chart by chart without touching it.
        self.review_host = None
        self._symbol = ""
        self._side = ""

        self.snapshot = SymbolSnapshotWidget(self)
        self.snapshot.d1LevelAlertRequested.connect(self._on_d1_level_alert)
        # Compatibility aliases for existing callers and tests.
        for name in (
            "d1_legend",
            "d1_chart",
            "d1_note",
            "m5_legend",
            "m5_chart",
            "m5_note",
            "paint_lines_button",
        ):
            setattr(self, name, getattr(self.snapshot, name))

        self.d1_focus_button = QPushButton("Add to D1 Focus")
        self.d1_focus_button.setCheckable(True)
        self.d1_focus_button.setToolTip(
            "Toggle this pick into Swing Focus (it lands on the Focus Picks "
            "tab and the swing watchlists) and pin it in the Alert Center's "
            "D1 Focus feed. Click again to remove both."
        )
        self.d1_focus_button.clicked.connect(self._toggle_d1_focus)
        self.m5_focus_button = QPushButton("Add to M5 Focus")
        self.m5_focus_button.setCheckable(True)
        self.m5_focus_button.setToolTip(
            "Toggle this symbol onto the M5 Focus day-trade list (BounceBot "
            "M5-scans it immediately). Click again to remove."
        )
        self.m5_focus_button.clicked.connect(self._toggle_m5_focus)
        self.watch_buttons: dict[str, QPushButton] = {}
        for kind, label in WATCH_KINDS.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(
                f"Toggle a one-shot {label} watch for this symbol. The first "
                "completed M5 bar that meets it fires a red alert in the "
                "Alert Center (bypasses the tier gate and sounds). Click "
                "again to disarm."
            )
            button.clicked.connect(
                lambda _checked=False, k=kind: self._toggle_watch(k)
            )
            self.watch_buttons[kind] = button
        # Persistent D1 event alerts (15EMA reject, 5d/20d extremes, SMA
        # break) - the same toggles the review pane's arm dock carries, so a
        # setups-table chart can arm them without a detour through the dock.
        self.d1_event_buttons: dict[str, QPushButton] = {}
        for kind, label in D1_EVENT_KINDS.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(
                f"Toggle a persistent {label} alert for this symbol. The "
                "reference level re-derives from the daily store every poll; "
                "one-shot, survives restarts and sessions, fires red in the "
                "Alert Center. Click again to disarm."
            )
            button.clicked.connect(
                lambda _checked=False, k=kind: self._toggle_d1_event(k)
            )
            self.d1_event_buttons[kind] = button

        # Review-flow ✕: log a dislike (with the typed reason) and advance to
        # the next visible setup row's chart. Only shown when a review host
        # (the setups panel) opened this popup.
        self.dislike_button = QPushButton("✕ Dislike")
        self.dislike_button.setToolTip(
            "Log a dislike for this setup - you'll be asked why, and the "
            "reason feeds the review-learning loop - then advance to the "
            "next chart in the setups table."
        )
        self.dislike_button.clicked.connect(self._review_dislike)

        self.action_row = QWidget()
        action_layout = QHBoxLayout(self.action_row)
        action_layout.setContentsMargins(10, 0, 10, 8)
        action_layout.setSpacing(6)
        action_layout.addWidget(self.dislike_button)
        action_layout.addWidget(self.d1_focus_button)
        action_layout.addWidget(self.m5_focus_button)
        for button in self.watch_buttons.values():
            action_layout.addWidget(button)
        for button in self.d1_event_buttons.values():
            action_layout.addWidget(button)
        action_layout.addStretch(1)
        self.action_row.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.snapshot)
        layout.addWidget(self.action_row)

        # The popup is non-modal and stays open while the trader works, so
        # its charts go stale exactly like the review pane's did. Re-pull on
        # the same 30s cadence; refresh() only re-renders when a bar changed.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(REFRESH_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self._auto_refresh)
        self._refresh_timer.start()

    def _auto_refresh(self) -> None:
        if not self.isVisible() or not self._symbol:
            return
        try:
            self.snapshot.refresh()
        except Exception:
            # A refresh must never take down the popup; the next tick retries.
            pass

    def show_symbol(
        self, symbol: str, *, bot=None, side: str = "", watch_host=None, review_host=None
    ) -> None:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return
        self._symbol = symbol
        self._side = side if side in ("LONG", "SHORT") else ""
        self.watch_host = watch_host
        self.review_host = review_host
        side_text = f" ({side})" if side in ("LONG", "SHORT") else ""
        self.setWindowTitle(f"{symbol}{side_text} — D1 + M5 snapshot")
        self.snapshot.set_symbol(symbol, bot=bot)
        self._refresh_watch_actions()
        # show + raise only (no activateWindow): the popup must never steal
        # typing focus from a watchlist editor or the live feed.
        self.show()
        self.raise_()

    def _refresh_watch_actions(self) -> None:
        host = self.watch_host
        reviewing = self.review_host is not None
        self.action_row.setVisible(host is not None or reviewing)
        self.dislike_button.setVisible(reviewing)
        # Watch/focus toggles need a watch host to act through; hide them
        # rather than showing dead buttons in a review-only popup.
        for button in (
            self.d1_focus_button,
            self.m5_focus_button,
            *self.watch_buttons.values(),
            *self.d1_event_buttons.values(),
        ):
            button.setVisible(host is not None)
        if host is None:
            return
        try:
            armed = set(host.armed_watch_kinds(self._symbol))
        except Exception:
            armed = set()
        for kind, button in self.watch_buttons.items():
            label = WATCH_KINDS[kind]
            is_armed = kind in armed
            button.setText(f"{label} ✓ armed" if is_armed else label)
            button.setChecked(is_armed)
        try:
            armed_events = set(host.armed_d1_event_kinds(self._symbol))
        except Exception:
            # An older host without the D1-event API: leave the buttons
            # visible-but-unchecked; a click will no-op through the same guard.
            armed_events = set()
        for kind, button in self.d1_event_buttons.items():
            label = D1_EVENT_KINDS[kind]
            is_armed = kind in armed_events
            button.setText(f"{label} ✓" if is_armed else label)
            button.setChecked(is_armed)
        pinned = bool(host.is_d1_focus_active(self._symbol, self._side))
        self.d1_focus_button.setText("✓ In D1 Focus" if pinned else "Add to D1 Focus")
        self.d1_focus_button.setChecked(pinned)
        in_m5 = bool(host.is_m5_focus(self._symbol, self._side))
        self.m5_focus_button.setText("✓ In M5 Focus" if in_m5 else "Add to M5 Focus")
        self.m5_focus_button.setChecked(in_m5)

    def _toggle_watch(self, kind: str) -> None:
        if self.watch_host is None or not self._symbol:
            return
        if kind in set(self.watch_host.armed_watch_kinds(self._symbol)):
            self.watch_host.disarm_chart_watch_for(self._symbol, kind)
        else:
            self.watch_host.arm_chart_watch_for(
                self._symbol,
                self._side or "WATCH",
                kind,
                source_text=f"chart snapshot: {self.windowTitle()}",
            )
        self._refresh_watch_actions()

    def _toggle_d1_event(self, kind: str) -> None:
        host = self.watch_host
        if host is None or not self._symbol:
            return
        if not hasattr(host, "arm_d1_event_watch"):
            return  # older host: the button is inert rather than a crash
        if kind in set(host.armed_d1_event_kinds(self._symbol)):
            host.disarm_d1_event_watch(self._symbol, kind)
        else:
            host.arm_d1_event_watch(self._symbol, kind)
        self._refresh_watch_actions()

    def _review_dislike(self) -> None:
        """The review-flow ✕: the host prompts for the reason, logs it, and
        advances to the next chart; a cancelled prompt leaves this one up."""
        if self.review_host is None or not self._symbol:
            return
        self.review_host.snapshot_review_dislike(self._symbol)

    def _toggle_d1_focus(self) -> None:
        if self.watch_host is None or not self._symbol:
            return
        added = self.watch_host.toggle_d1_focus(
            self._symbol,
            self._side,
            origin="chart",
            context=f"chart snapshot: {self.windowTitle()}",
        )
        self._refresh_watch_actions()
        # Review flow: filing the pick into D1 Focus is a decision made -
        # move on to the next chart. Toggling OFF is a correction, not a
        # decision; it stays on this chart.
        if added and self.review_host is not None:
            self.review_host.snapshot_review_advance()

    def _toggle_m5_focus(self) -> None:
        if self.watch_host is None or not self._symbol:
            return
        self.watch_host.toggle_m5_focus(
            self._symbol,
            self._side,
            origin="chart",
            context=f"chart snapshot: {self.windowTitle()}",
        )
        self._refresh_watch_actions()

    def _on_d1_level_alert(self, symbol: str, direction: str, level: float, candle_date: str) -> None:
        if self.watch_host is None:
            return
        self.watch_host.arm_d1_level_watch(
            symbol, direction, level, candle_date=candle_date
        )


def show_symbol_snapshot(
    owner, symbol: str, *, bot=None, side: str = "", watch_host=None, review_host=None
) -> SymbolSnapshotDialog:
    """Panel helper: lazily create one reusable dialog per owner widget."""
    dialog = getattr(owner, "_symbol_snapshot_dialog", None)
    if dialog is None:
        dialog = SymbolSnapshotDialog(owner)
        owner._symbol_snapshot_dialog = dialog
    dialog.show_symbol(
        symbol, bot=bot, side=side, watch_host=watch_host, review_host=review_host
    )
    return dialog
