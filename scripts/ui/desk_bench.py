"""A repeatable desk workload bench, and the layout-fit check that goes with it.

Packet G0 (2026-09-06), the first step of the Phase 0.22 desk reshape. This
module measures; **it changes nothing**. No panel imports it, nothing on the
desk calls it, and it is never started by a timer.

Why it exists
-------------
`ui/stall_watchdog.py` measures a SESSION on the live desk: it says the GUI
thread was blocked for 1008 s, and it cannot say which click paid for it.
`ui/interaction_trace.py` stamps "which click" onto a watchdog record but only
if a trader is there to click. Neither can be run twice over the same work and
compared, which is exactly what a reshape needs: a number before, the same
number after, over the same inputs.

This bench builds the pages ONE AT A TIME - never `MainWindow`, so no IB, no
autopilot, no timers - drives a fixed workload over a STAGED COPY of the home
folder, and reports p50 / p95 / max per operation across repeats.

Three numbers per operation, because they answer different questions:

* **sync_ms** - the wall time of the call itself on the Qt thread. This is the
  part a trader feels as "the click did not return".
* **settle_ms** - from the call until the page stopped working (its workers
  finished and the queued renders drained), or the deadline. A page that
  returns in 3 ms and fills in 9 s is not fast.
* **longest_iteration_ms** - the longest single `processEvents()` call during
  the settle wait. That is the stall proxy: one slot that runs for 800 ms shows
  up here and nowhere else.

A deadline is a RESULT, not an error. `settled: false` with the deadline in the
table is the honest answer for a page that never finished, and it is counted.

Safety, which is the reason for the shape of `main()`
-----------------------------------------------------
On 2026-09-05 two agent scratch scripts imported something under `scripts/`
before pointing the data directory anywhere, `project_paths` resolved to the
live home folder, and the live 1.2 GB setup tracker and the leaderboard CSVs
were overwritten. So:

* `--data-dir` is REQUIRED and is set into `TRADINGBOTV3_DATA_DIR` **before the
  first import of anything under `scripts/`** - nothing under `scripts/` is
  imported at module top here, and a test asserts that at the source level;
* the resolved `project_paths.DATA_DIR` is PRINTED and the process EXITS 2 if
  it lands under the live home folder or the DAS;
* `LOCALAPPDATA` is redirected into the scratch too, so a panel that writes a
  cache cannot reach the machine-local store either. The one file written
  outside the scratch is this bench's own JSON under the real diagnostics
  folder, and its path is resolved BEFORE the redirect.

`stage` is the other half: it copies an allowlist of the read inputs the panels
open out of the live store, opening the source read-only and refusing a
destination under the live home or the DAS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

#: Roots nothing here may resolve to, write to, or stage into. Compared
#: case-folded against the resolved path, so a `c:/tradingbotdata/scratch`
#: cannot slip through on capitalisation.
LIVE_STORE_ROOTS: tuple[str, ...] = (
    r"c:\tradingbotdata",
    "\\\\mini-pc\\",
    "//mini-pc/",
)

#: The SECOND guard's own literals, deliberately NOT built from
#: `LIVE_STORE_ROOTS`: the two guards are independent so that breaking one -
#: which is exactly what a reviewer does to prove a guard bites - leaves the
#: other standing. Kept as a module constant only so a test can point BOTH at a
#: fake root under `tmp_path`; no test may ever aim a sabotaged guard at the
#: real live paths as a destination.
WRITE_REFUSAL_PREFIXES: tuple[str, ...] = (
    "c:\\tradingbotdata\\",
    "\\\\mini-pc\\",
)

#: The machine-local settings keys the bench carries from the real
#: `%LOCALAPPDATA%\TradingBotV3\local_settings.json` into its redirected one.
#:
#: An ALLOWLIST, and a short one, for two reasons. The real file holds five
#: credentials (`market_prep_openai_api_key`, `journal_questrade_refresh_token`,
#: `journal_ibkr_flex_token`, `push_ntfy_token`, `desk_link_token`) that have no
#: business in a scratch directory, and it holds three PATH keys
#: (`shared_data_dir`, `research_store_dir`, `ai_store_dir`) that point at the
#: live home folder and the DAS - copying those into a bench whose entire point
#: is that it never touches the live store would be the 2026-09-05 incident
#: wearing a lab coat.
#:
#: What is carried is what CHANGES WHAT IS MEASURED: `qt_ui_scale` scales every
#: `theme.px` and therefore every number the layout-fit check prints, and
#: density, theme and mode change what a page builds. `daily_bars_source` is
#: carried because it is the trader's pin and costs nothing (this bench runs no
#: scan and fetches no bar).
MACHINE_SETTINGS_CARRIED: tuple[str, ...] = (
    "daily_bars_source",
    "gui_mode",
    "gui_performance_mode",
    "qt_alert_min_tier",
    "qt_compact_density",
    "qt_explain_mode",
    "qt_nav_collapsed",
    "qt_setups_bucket_filter",
    "qt_theme",
    "qt_ui_scale",
    "qt_workspace_mode",
    "qt_alert_center_split_sizes_v2",
    "qt_alert_tabs_row_split_sizes_v1",
    "qt_desk_split_sizes_v2",
    "qt_desk_split_sizes_v3",
)

#: What `stage` copies, relative to the source home folder. This is an
#: ALLOWLIST on purpose: the live store holds a 1.2 GB tracker JSON, a 622 MB
#: attributes CSV and a 142 MB scenarios CSV that no page on this bench opens,
#: and copying them would make staging the slowest part of a measuring tool.
#:
#: A `*` is a glob and may match many files; a plain name must match exactly.
#: A pattern that matches nothing is REPORTED, never an error - the control and
#: study discovery exports do not exist until the next persisted tracker write
#: (live gate #74), and a bench that refused to run until they did would be
#: useless on the day it is needed.
STAGE_ALLOWLIST: tuple[str, ...] = (
    # -- the Setup Tracker's own reads ---------------------------------
    "data/runtime/master_avwap_setup_stats.csv",
    "data/runtime/master_avwap_setup_type_stats.csv",
    "data/runtime/master_avwap_setup_type_recent_stats.csv",
    "data/runtime/master_avwap_setup_short_horizon.csv",
    "data/runtime/master_avwap_setup_playbooks.csv",
    "data/runtime/master_avwap_tier_list.csv",
    "data/runtime/master_avwap_tier_performance.csv",
    "data/runtime/master_avwap_tier_catch_rate.csv",
    "data/runtime/master_avwap_tier_outcomes.csv",
    "data/runtime/master_avwap_scan_factor_leaderboard.csv",
    "data/runtime/master_avwap_setup_attribute_leaderboard.csv",
    "data/runtime/master_avwap_setup_attribute_leaderboard_by_family.csv",
    "data/runtime/master_avwap_setup_attribute_leaderboard_by_regime.csv",
    "data/runtime/master_avwap_band_variant_stats.csv",
    "data/runtime/master_avwap_control_discovery.csv",
    "data/runtime/master_avwap_study_discovery.csv",
    "data/runtime/master_avwap_exit_framework_stats.csv",
    # -- the Day-trade Tracker -----------------------------------------
    "data/runtime/intraday_bounce_outcomes.csv",
    "data/runtime/intraday_bounce_performance.csv",
    "data/runtime/intraday_bounce_learning_state.json",
    "data/runtime/review_preference_state.json",
    # -- the human-focus and cohort evidence Weekend Prep reads ---------
    "data/runtime/human_focus_daily_picks.csv",
    "data/runtime/human_focus_outcomes.csv",
    "data/runtime/human_focus_performance.csv",
    "data/runtime/veto_cohort_*.csv",
    "data/runtime/like_cohort_*.csv",
    "data/runtime/pass_cohort_*.csv",
    "data/runtime/rejection_cohort_*.csv",
    "data/runtime/weekend_prep_state.json",
    "pick_feedback.jsonl",
    "review_preference_state.json",
    "swing_favorites.jsonl",
    # -- the Market Journal --------------------------------------------
    "data/runtime/evidence_ledgers/market_journal-*.jsonl",
    "data/runtime/evidence_ledgers/market_journal_charts-*.jsonl",
    "data/runtime/evidence_ledgers/market_regime_shifts-*.jsonl",
    # -- capture / review evidence -------------------------------------
    "trader_annotations.jsonl",
    "alert_review_events.jsonl",
    "alert_review_events/*.jsonl",
    # -- the trade journal, as a COPY -----------------------------------
    "data/runtime/trade_journal.sqlite3",
    # NO settings file here on purpose. `local_settings.json` was listed once
    # against the home-folder root, where it does not live: the real file is at
    # `%LOCALAPPDATA%\TradingBotV3\local_settings.json`, so the entry reported
    # `absent` on every staging run and the first baseline was measured against
    # a synthetic one-key file. The machine-local settings are now carried by
    # `machine_settings_seed` at run time, key by key - see
    # `MACHINE_SETTINGS_CARRIED` for why it is an allowlist and not a copy.
)

#: How long one operation may take before the settle wait gives up. A hit is
#: recorded and counted, never raised.
DEFAULT_DEADLINE_S = 20.0

#: After the page's last worker stops, keep draining for this long so the
#: queued render that the worker's signal triggers is inside `settle_ms`.
#: Without it a page whose read finishes in 20 ms and then repaints for 900 ms
#: would measure 20 ms, which is not what the trader waits for.
QUIET_MS = 120.0

#: The p95 line the G packets are held to. Only a label for the printed table.
SLOW_OP_MS = 250.0

_SIZE_DEFAULT = "3456x2160"


# ---------------------------------------------------------------------------
# Pure helpers. No Qt, nothing under `scripts/`. These are what the tests pin.
# ---------------------------------------------------------------------------
def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile. `None` for an empty sample, never 0.0.

    Nearest-rank rather than an interpolating variant on purpose: three repeats
    is a normal sample here, and interpolating between two of three readings
    invents a number that was never measured.
    """
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    if fraction <= 0:
        return ordered[0]
    rank = int(-(-len(ordered) * float(fraction) // 1))  # ceil, no float error
    rank = max(1, min(len(ordered), rank))
    return ordered[rank - 1]


def summarize(values: Sequence[float]) -> dict[str, Any]:
    """`n / p50 / p95 / max` over one operation's readings.

    An empty sample reports `n: 0` and `None` for the three statistics. A zero
    would read as "instant" and be wrong in the one case that matters - the
    operation that never ran.
    """
    sample = [float(v) for v in values]
    return {
        "n": len(sample),
        "p50": percentile(sample, 0.50),
        "p95": percentile(sample, 0.95),
        "max": max(sample) if sample else None,
    }


def is_under_live_store(path: str | os.PathLike[str]) -> bool:
    """True when `path` resolves inside the live home folder or the DAS."""
    try:
        resolved = str(Path(path).expanduser().resolve())
    except OSError:
        resolved = str(path)
    lowered = resolved.replace("/", "\\").casefold()
    for root in LIVE_STORE_ROOTS:
        marker = root.replace("/", "\\").casefold()
        if lowered == marker.rstrip("\\") or lowered.startswith(marker.rstrip("\\") + "\\"):
            return True
        if marker.startswith("\\\\") and lowered.startswith(marker):
            return True
    return False


def fit_verdict(
    *,
    minimum_size_hint: int,
    minimum_size: int,
    size_hint: int,
    available: int,
) -> dict[str, Any]:
    """Does this page fit in `available` pixels of height?

    The requirement is the LARGER of the layout's own minimum hint and any
    explicit floor the page set on itself: a page can raise its minimum past
    what its children ask for, and a check that read only one of the two would
    miss the Weekend Focus Review defect in either direction.
    """
    required = max(int(minimum_size_hint), int(minimum_size))
    return {
        "minimum_size_hint": int(minimum_size_hint),
        "minimum_size": int(minimum_size),
        "size_hint": int(size_hint),
        "required": required,
        "available": int(available),
        "overflow": required > int(available),
        "overflow_px": max(0, required - int(available)),
    }


@dataclass
class StageResult:
    """What `stage` copied, what it could not find, and how much it moved."""

    copied: list[tuple[str, int]] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    total_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "copied": [{"path": name, "bytes": size} for name, size in self.copied],
            "absent": list(self.absent),
            "total_bytes": self.total_bytes,
            "files": len(self.copied),
        }


class StageRefused(RuntimeError):
    """The destination was somewhere this tool will not write."""


def _refuse_to_open_for_writing(target: Path) -> None:
    """The SECOND, independent guard, checked once per file before the write.

    Deliberately not a call to `is_under_live_store`: it compares its own
    literals, so disabling one guard leaves the other standing. That is not
    paranoia about a bug - it is about the reviewer, who proves a guard bites by
    breaking it and re-running, and whose run would otherwise write test files
    into the live store while demonstrating that it must not.

    Discovered exactly that way on 2026-09-06 while building this packet: the
    fail-before-fix proof for the first guard created `C:\\TradingBotData\\scratch`
    and the same folder on the DAS (six synthetic files, both trees removed, no
    live file touched). One guard was one too few.
    """
    text = str(target).replace("/", "\\").casefold()
    for prefix in WRITE_REFUSAL_PREFIXES:
        if text.startswith(prefix.replace("/", "\\").casefold()):
            raise StageRefused(
                f"refusing to open {target} for writing: that is inside the live "
                "home folder or the DAS"
            )


def refuse_live_destination(label: str, path: str | os.PathLike[str], stream) -> int | None:
    """Both guards, run on a path BEFORE anything is created at it. 2, or None.

    This is the fix for the defect the reviewer found on 2026-09-06: `main()`
    called `_prepare_environment` first, which `mkdir`s the data directory and
    writes a settings file into it, and only THEN asked whether that directory
    was the live home folder. `--data-dir C:\\TradingBotData` therefore exited 2
    having already created three directories and a file inside the live store -
    the packet's own invariant broken by the code that states it, and the exact
    shape of the 2026-09-05 incident.

    Both guards are run because they are independent by design: `is_under_live_store`
    resolves the path and compares against `LIVE_STORE_ROOTS`, and
    `_refuse_to_open_for_writing` compares its own literals against the text.
    """
    if is_under_live_store(path):
        print(
            f"REFUSED: {label} {path} is inside the live home folder or the DAS. "
            "desk_bench never writes there, and nothing has been created.",
            file=stream,
        )
        return 2
    try:
        _refuse_to_open_for_writing(Path(path) / "_")
    except StageRefused as exc:
        print(f"REFUSED: {exc}", file=stream)
        return 2
    return None


def stage(
    source: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    patterns: Iterable[str] = STAGE_ALLOWLIST,
    chunk_bytes: int = 4 * 1024 * 1024,
) -> StageResult:
    """Copy the allowlist out of `source` into `dest`, same relative layout.

    The source is opened `rb` and never written; the destination is refused
    outright if it is under the live home folder or the DAS, because a staging
    run that landed back on top of the store it read is the 2026-09-05 incident
    with extra steps.
    """
    source_dir = Path(source).expanduser()
    dest_dir = Path(dest).expanduser()
    if is_under_live_store(dest_dir):
        raise StageRefused(
            f"refusing to stage into {dest_dir}: that is the live home folder or the DAS"
        )
    _refuse_to_open_for_writing(dest_dir / "_")
    dest_dir.mkdir(parents=True, exist_ok=True)

    result = StageResult()
    for pattern in patterns:
        relative = pattern.replace("\\", "/")
        if "*" in relative or "?" in relative:
            matches = sorted(source_dir.glob(relative))
            matches = [m for m in matches if m.is_file()]
            if not matches:
                result.absent.append(relative)
                continue
        else:
            candidate = source_dir / relative
            if not candidate.is_file():
                result.absent.append(relative)
                continue
            matches = [candidate]
        for match in matches:
            rel = match.relative_to(source_dir).as_posix()
            target = dest_dir / rel
            _refuse_to_open_for_writing(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            copied = 0
            with open(match, "rb") as src, open(target, "wb") as dst:
                while True:
                    block = src.read(chunk_bytes)
                    if not block:
                        break
                    dst.write(block)
                    copied += len(block)
            result.copied.append((rel, copied))
            result.total_bytes += copied
    return result


def format_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:,.1f} GB"


def parse_size(text: str) -> tuple[int, int]:
    """`3456x2160` -> `(3456, 2160)`. Raises on anything else."""
    raw = str(text).strip().lower().replace("×", "x")
    if "x" not in raw:
        raise ValueError(f"size must look like 3456x2160, got {text!r}")
    left, _, right = raw.partition("x")
    width, height = int(left), int(right)
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be positive, got {text!r}")
    return width, height


# ---------------------------------------------------------------------------
# The measured run. Everything below imports Qt (and, after `main` has set the
# data directory, the desk's own modules) inside the function that needs it.
# ---------------------------------------------------------------------------
@dataclass
class OpReading:
    """One timed operation, once."""

    op: str
    size: str
    sync_ms: float
    settle_ms: float
    longest_iteration_ms: float
    settled: bool
    error: str = ""


def _widget_workers_running(root) -> bool:
    """Is anything this page owns still reading?

    Two idioms are in use and both are checked: `ui.read_worker.ReadWorker` and
    friends are `QThread`s parented to the widget, and the Day-trade Tracker
    uses plain `threading.Thread` objects held on attributes. A settle wait that
    knew about only one of them would call the other page settled while its read
    was still in flight.
    """
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QWidget

    for worker in root.findChildren(QThread):
        try:
            if worker.isRunning():
                return True
        except RuntimeError:  # pragma: no cover - deleted mid-scan
            continue
    for widget in [root, *root.findChildren(QWidget)]:
        try:
            attributes = list(vars(widget).values())
        except TypeError:  # pragma: no cover - C++-only object
            continue
        for value in attributes:
            if isinstance(value, threading.Thread) and value.is_alive():
                return True
    return False


def settle(app, root, *, deadline_s: float = DEFAULT_DEADLINE_S) -> tuple[float, float, bool]:
    """Drain the event loop until the page stops working, or give up.

    Returns `(settle_ms, longest_iteration_ms, settled)`.

    The longest iteration is the longest single `processEvents()` call, which
    is the stall proxy: a slot that runs 800 ms inside one drain is invisible in
    the total and obvious here.

    **The 2 ms yield is INSIDE `settle_ms` and outside `longest_iteration_ms`,
    and that is the honest description of it.** While a worker is running this
    loop sleeps 2 ms between drains so the worker actually gets the GIL - a
    spin would measure the bench fighting the thing it is measuring - and the
    trader waits through that time too, so it belongs in the settle total. It
    is excluded from the longest-iteration timer, which brackets the
    `processEvents()` call alone and is the stall proxy.

    **Every settle carries the `QUIET_MS` (120 ms) floor**, because settling is
    declared only after the page has been quiet for that long. A reading near
    120-135 ms therefore means "nothing was measured here": the op finished
    before the first drain and the number is the floor, not work. Compare such
    ops on `sync_ms`.
    """
    start = time.perf_counter()
    longest = 0.0
    quiet_since: float | None = None
    settled = False
    while True:
        t0 = time.perf_counter()
        app.processEvents()
        longest = max(longest, (time.perf_counter() - t0) * 1000.0)

        now = time.perf_counter()
        if _widget_workers_running(root):
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        elif (now - quiet_since) * 1000.0 >= QUIET_MS:
            settled = True
            break
        if now - start >= deadline_s:
            break
        if quiet_since is None:
            time.sleep(0.002)
    return (time.perf_counter() - start) * 1000.0, longest, settled


def time_op(app, root, name: str, size_label: str, call: Callable[[], Any], *, deadline_s: float) -> OpReading:
    """Run one operation, time the call, then time the settle."""
    error = ""
    t0 = time.perf_counter()
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - a broken op is a RESULT here
        error = f"{type(exc).__name__}: {exc}"
    sync_ms = (time.perf_counter() - t0) * 1000.0
    settle_ms, longest, settled = settle(app, root, deadline_s=deadline_s)
    return OpReading(
        op=name,
        size=size_label,
        sync_ms=sync_ms,
        settle_ms=settle_ms,
        longest_iteration_ms=longest,
        settled=settled,
        error=error,
    )


# -- the layout-fit check ---------------------------------------------------
def chrome_height(app) -> dict[str, int]:
    """How much vertical room the window shell takes, measured from widgets.

    `ui/app.py` puts a `TopBar` frame above the page stack and a `QStatusBar`
    below it, inside a central layout with `theme.px(8)` margins. This builds
    the same widgets with the same theme constants and asks THEM how tall they
    are, rather than carrying a pixel number that would go stale the first time
    the top bar gains a button.

    `MainWindow` itself is deliberately not constructed: it starts IB, the
    autopilot and a dozen timers, none of which belong in a bench.
    """
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QStatusBar,
    )

    from ui import theme

    top_bar = QFrame()
    top_bar.setObjectName("TopBar")
    top_layout = QHBoxLayout(top_bar)
    top_layout.setContentsMargins(theme.px(12), theme.px(10), theme.px(12), theme.px(10))
    title = QLabel("Trading Desk")
    title.setObjectName("PageTitle")
    top_layout.addWidget(title)
    top_layout.addStretch(1)
    top_layout.addWidget(QPushButton("Workspace"))
    top_layout.addWidget(QPushButton("Tabs"))

    status = QStatusBar()
    status.addPermanentWidget(QPushButton("AUTO-DESK"))

    top_px = int(top_bar.sizeHint().height())
    status_px = int(status.sizeHint().height())
    margins_px = int(theme.px(8))  # central layout: top 8, bottom 0
    top_bar.deleteLater()
    status.deleteLater()
    app.processEvents()
    return {
        "top_bar": top_px,
        "status_bar": status_px,
        "central_margins": margins_px,
        "total": top_px + status_px + margins_px,
    }


def table_floor_sum(root) -> int:
    """Sum of the explicit minimum heights of every table on a page.

    This is the Weekend Focus Review number: nine tables with a 260 px floor
    each in one vertical layout is 2,340 px of floors, and a page cannot be
    shorter than the sum of the floors it stacked.

    A table on a hidden tab still counts. Its floor is what the page will need
    the moment that tab is selected, and a check that only measured what is on
    screen would go green the instant a defect was moved behind a tab.
    """
    total = 0
    for view in root.findChildren(_item_view_type()):
        total += max(0, int(view.minimumHeight()))
    return total


def measure_fit(root, *, page: str, size_label: str, available: int) -> dict[str, Any]:
    """One page's fit record at one window size."""
    verdict = fit_verdict(
        minimum_size_hint=int(root.minimumSizeHint().height()),
        minimum_size=int(root.minimumSize().height()),
        size_hint=int(root.sizeHint().height()),
        available=int(available),
    )
    floors = table_floor_sum(root)
    verdict.update(
        {
            "page": page,
            "size": size_label,
            "table_floor_sum": floors,
            "table_floor_overflow_px": max(0, floors - int(available)),
            "tables": len(root.findChildren(_item_view_type())),
        }
    )
    return verdict


def _item_view_type():
    from PySide6.QtWidgets import QAbstractItemView

    return QAbstractItemView


# -- the panels and their workload ------------------------------------------
def _research_tab_widget(panel):
    """`ResearchPanel` does not keep its tab strip on an attribute.

    Found by identity against a child it DOES name, rather than by taking the
    first `QTabWidget` in the tree - the Setup Tracker and the Day-trade Tracker
    each own one, and "the first one found" is a coin toss that would silently
    drive the wrong widget.
    """
    from PySide6.QtWidgets import QTabWidget

    for tabs in panel.findChildren(QTabWidget):
        if tabs.indexOf(panel.setup_tracker_panel) >= 0:
            return tabs
    return None


def _tab_ops(tabs, prefix: str) -> list[tuple[str, Callable[[], Any]]]:
    ops: list[tuple[str, Callable[[], Any]]] = []
    for index in range(tabs.count()):
        label = tabs.tabText(index).replace("&", "")
        ops.append((f"{prefix}.{label}", lambda i=index: tabs.setCurrentIndex(i)))
    return ops


def build_panel(name: str):
    """Construct one page, or raise. The caller records the failure and names it."""
    if name == "weekend_prep":
        from ui.panels.weekend_prep_panel import WeekendPrepPanel

        return WeekendPrepPanel()
    if name == "market_journal":
        from ui.panels.market_journal_panel import MarketJournalPanel

        return MarketJournalPanel()
    if name == "research":
        from ui.panels.research_panel import ResearchPanel

        return ResearchPanel()
    if name == "away_recap":
        from ui.panels.away_recap_panel import AwayRecapPanel

        return AwayRecapPanel()
    if name == "journal":
        from ui.panels.journal_panel import JournalPanel

        return JournalPanel()
    raise KeyError(name)


PANEL_NAMES: tuple[str, ...] = (
    "weekend_prep",
    "market_journal",
    "research",
    "away_recap",
    "journal",
)

SMOKE_PANEL_NAMES: tuple[str, ...] = ("away_recap", "market_journal")


def workload_ops(name: str, panel) -> list[tuple[str, Callable[[], Any]]]:
    """The fixed click sequence for one page, in the order a trader would.

    Every op is a real call on a real widget. Where a page cannot be driven
    without a seam that does not exist, it is left out and named in the run's
    `not_measurable` list rather than faked.
    """
    ops: list[tuple[str, Callable[[], Any]]] = []
    if name == "weekend_prep":
        ops.append(("weekend.refresh_everything", panel.refresh_everything))
        # The rail's own text is NOT the op name. It carries a status glyph
        # (`○` before the step is done, `✓` after) that changes between runs,
        # and an op whose name depends on the state of the data cannot be
        # compared against a baseline - `--compare` would report every step as
        # NEW and GONE at once. The step IDS are stable and are the rail order.
        for row, step_id in enumerate(panel._pages):
            ops.append((f"weekend.step.{step_id}", lambda r=row: panel.rail.setCurrentRow(r)))
    elif name == "market_journal":
        ops.append(("market_journal.reload", panel.reload))
        for row in range(10):
            ops.append(
                (
                    f"market_journal.entry[{row}]",
                    lambda r=row: panel.entries.setCurrentRow(r),
                )
            )
    elif name == "research":
        tabs = _research_tab_widget(panel)
        if tabs is not None:
            ops.extend(_tab_ops(tabs, "research.tab"))
        tracker = panel.setup_tracker_panel
        # Bring the owning Research tab to the front FIRST. A tab switch inside
        # a hidden page costs almost nothing - Qt does not lay out or paint what
        # is not on screen - so measuring the child tabs while the Research
        # strip sits on some other page would report a fast desk that nobody has.
        if tabs is not None:
            ops.append(("research.select.setup_tracker", lambda: tabs.setCurrentWidget(tracker)))
        ops.append(("setup_tracker.refresh", tracker.refresh))
        ops.extend(_tab_ops(tracker.tabs, "setup_tracker.tab"))
        daytrade = panel.daytrade_tracker_panel
        if tabs is not None:
            ops.append(("research.select.daytrade_tracker", lambda: tabs.setCurrentWidget(daytrade)))
        ops.extend(_tab_ops(daytrade.tabs, "daytrade.tab"))
    elif name == "away_recap":
        ops.append(("away_recap.reload", panel.reload))
    elif name == "journal":
        ops.extend(_tab_ops(panel.tabs, "journal.tab"))
    return ops


def fit_targets(name: str, panel) -> list[tuple[str, Any]]:
    """Every widget whose fit is recorded for this page: it, and its children.

    A Research child and a Weekend step are the pages a trader actually looks
    at; the container's own minimum is the sum of whichever child is showing,
    so recording only the container would hide exactly the defect G1 fixes.
    """
    targets: list[tuple[str, Any]] = [(name, panel)]
    if name == "weekend_prep":
        for step, page in panel._pages.items():
            targets.append((f"weekend_prep.{step}", page))
    elif name == "research":
        for attribute in (
            "market_prep_panel",
            "setup_tracker_panel",
            "setup_docs_panel",
            "move_forensics_panel",
            "daytrade_tracker_panel",
            "ticker_lookup_panel",
            "price_alerts_panel",
            "warehouse_readout_panel",
        ):
            child = getattr(panel, attribute, None)
            if child is not None:
                targets.append((f"research.{attribute}", child))
    return targets


def shutdown_panel(panel) -> None:
    """Give a page its own shutdown, then let Qt take it. Never raises."""
    for method in ("shutdown", "close"):
        call = getattr(panel, method, None)
        if callable(call):
            try:
                call()
            except Exception:  # noqa: BLE001 - teardown must not fail a run
                pass
    try:
        panel.setParent(None)
        panel.deleteLater()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def run_bench(
    *,
    sizes: Sequence[tuple[int, int]],
    repeat: int,
    deadline_s: float,
    panel_names: Sequence[str],
    record_screens: bool,
) -> dict[str, Any]:
    from PySide6.QtWidgets import QApplication

    from ui import theme

    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app, "dark")

    chrome = chrome_height(app)
    readings: list[OpReading] = []
    fits: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for width, height in sizes:
        size_label = f"{width}x{height}"
        available = max(1, height - chrome["total"])
        for name in panel_names:
            for attempt in range(int(repeat)):
                try:
                    t0 = time.perf_counter()
                    panel = build_panel(name)
                    construct_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as exc:  # noqa: BLE001
                    skipped.append(
                        {
                            "panel": name,
                            "size": size_label,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    break
                settle_ms, longest, settled = settle(app, panel, deadline_s=deadline_s)
                readings.append(
                    OpReading(
                        op=f"{name}.construct",
                        size=size_label,
                        sync_ms=construct_ms,
                        settle_ms=settle_ms,
                        longest_iteration_ms=longest,
                        settled=settled,
                    )
                )

                def _show(panel=panel, width=width, available=available):
                    panel.resize(width, available)
                    panel.show()

                readings.append(
                    time_op(app, panel, f"{name}.show", size_label, _show, deadline_s=deadline_s)
                )

                for op_name, call in workload_ops(name, panel):
                    readings.append(
                        time_op(app, panel, op_name, size_label, call, deadline_s=deadline_s)
                    )

                if attempt == 0:
                    for page_name, widget in fit_targets(name, panel):
                        fits.append(
                            measure_fit(
                                widget,
                                page=page_name,
                                size_label=size_label,
                                available=available,
                            )
                        )
                shutdown_panel(panel)
                app.processEvents()

    payload: dict[str, Any] = {
        "schema": "desk_bench_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": os.environ.get("QT_QPA_PLATFORM", "windows"),
        "data_dir": _data_dir_text(),
        "repeat": int(repeat),
        "deadline_s": float(deadline_s),
        "quiet_ms": QUIET_MS,
        "chrome": chrome,
        "sizes": [f"{w}x{h}" for w, h in sizes],
        "panels": list(panel_names),
        "skipped": skipped,
        "not_measurable": list(NOT_MEASURABLE),
        "ops": _aggregate(readings),
        "fit": fits,
    }
    if record_screens:
        payload["screens"] = _screen_records(app)
    return payload


#: Operations the packet asks for that today's code cannot be driven through
#: without adding a seam to a panel - and G0 changes no panel. Each names the
#: seam so G7 can decide whether to add it.
NOT_MEASURABLE: tuple[str, ...] = ()


def _data_dir_text() -> str:
    try:
        import project_paths

        return str(project_paths.DATA_DIR)
    except Exception:  # noqa: BLE001
        return ""


def _screen_records(app) -> list[dict[str, Any]]:
    records = []
    for screen in app.screens():
        geometry = screen.geometry()
        available = screen.availableGeometry()
        records.append(
            {
                "name": screen.name(),
                "geometry": [geometry.x(), geometry.y(), geometry.width(), geometry.height()],
                "available_geometry": [
                    available.x(),
                    available.y(),
                    available.width(),
                    available.height(),
                ],
                "device_pixel_ratio": float(screen.devicePixelRatio()),
                "logical_dpi": float(screen.logicalDotsPerInch()),
            }
        )
    return records


def _aggregate(readings: Sequence[OpReading]) -> list[dict[str, Any]]:
    """One row per (op, size), with the three statistics over the repeats."""
    grouped: dict[tuple[str, str], list[OpReading]] = {}
    order: list[tuple[str, str]] = []
    for reading in readings:
        key = (reading.op, reading.size)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(reading)
    rows = []
    for op, size in order:
        group = grouped[(op, size)]
        errors = sorted({r.error for r in group if r.error})
        rows.append(
            {
                "op": op,
                "size": size,
                "samples": len(group),
                "sync_ms": summarize([r.sync_ms for r in group]),
                "settle_ms": summarize([r.settle_ms for r in group]),
                "longest_iteration_ms": summarize([r.longest_iteration_ms for r in group]),
                "deadline_hits": sum(1 for r in group if not r.settled),
                "errors": errors,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.1f}"


def format_ops_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'op':<44} {'size':<10} {'n':>2} "
        f"{'sync p50':>9} {'sync p95':>9} {'sync max':>9} "
        f"{'settle p50':>11} {'settle p95':>11} {'settle max':>11} "
        f"{'stall p95':>10} {'stall max':>10} {'dl':>3}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        mark = " *" if (row["sync_ms"]["p95"] or 0.0) > SLOW_OP_MS else ""
        lines.append(
            f"{row['op'][:44]:<44} {row['size']:<10} {row['samples']:>2} "
            f"{_ms(row['sync_ms']['p50']):>9} {_ms(row['sync_ms']['p95']):>9} {_ms(row['sync_ms']['max']):>9} "
            f"{_ms(row['settle_ms']['p50']):>11} {_ms(row['settle_ms']['p95']):>11} {_ms(row['settle_ms']['max']):>11} "
            f"{_ms(row['longest_iteration_ms']['p95']):>10} {_ms(row['longest_iteration_ms']['max']):>10} "
            f"{row['deadline_hits']:>3}{mark}"
        )
    lines.append("")
    lines.append(f"* = sync p95 over {SLOW_OP_MS:.0f} ms.  dl = settle deadline hits (a RESULT, not an error).")
    return "\n".join(lines)


def format_fit_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'page':<40} {'size':<10} {'min hint':>9} {'min size':>9} "
        f"{'hint':>8} {'avail':>7} {'over':>7} {'tables':>6} {'floors':>7} {'floor over':>10}  verdict"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        verdict = "OVERFLOW" if row["overflow"] else "fits"
        lines.append(
            f"{row['page'][:40]:<40} {row['size']:<10} {row['minimum_size_hint']:>9} "
            f"{row['minimum_size']:>9} {row['size_hint']:>8} {row['available']:>7} "
            f"{row['overflow_px']:>7} {row['tables']:>6} {row['table_floor_sum']:>7} "
            f"{row['table_floor_overflow_px']:>10}  {verdict}"
        )
    return "\n".join(lines)


def format_compare(current: Sequence[dict], baseline: Sequence[dict]) -> str:
    """Per-op deltas against a saved run. Positive is SLOWER."""
    index = {(row["op"], row["size"]): row for row in baseline}
    header = (
        f"{'op':<44} {'size':<10} {'sync p95':>10} {'was':>10} {'delta':>10} "
        f"{'settle p95':>11} {'was':>11} {'delta':>11}"
    )
    lines = [header, "-" * len(header)]
    for row in current:
        was = index.get((row["op"], row["size"]))
        if was is None:
            lines.append(f"{row['op'][:44]:<44} {row['size']:<10} {'NEW':>10}")
            continue
        now_sync = row["sync_ms"]["p95"]
        was_sync = was["sync_ms"]["p95"]
        now_settle = row["settle_ms"]["p95"]
        was_settle = was["settle_ms"]["p95"]
        d_sync = None if (now_sync is None or was_sync is None) else now_sync - was_sync
        d_settle = None if (now_settle is None or was_settle is None) else now_settle - was_settle
        lines.append(
            f"{row['op'][:44]:<44} {row['size']:<10} {_ms(now_sync):>10} {_ms(was_sync):>10} "
            f"{_ms(d_sync):>10} {_ms(now_settle):>11} {_ms(was_settle):>11} {_ms(d_settle):>11}"
        )
    gone = [key for key in index if key not in {(r["op"], r["size"]) for r in current}]
    for op, size in gone:
        lines.append(f"{op[:44]:<44} {size:<10} {'GONE':>10}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="desk_bench",
        description=(
            "Measure the desk's pages over a staged copy of the home folder, "
            "and check whether each page fits the window. Measures only."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    stage_parser = sub.add_parser("stage", help="copy the read inputs the pages open into a scratch dir")
    stage_parser.add_argument("--from", dest="source", required=True)
    stage_parser.add_argument("--to", dest="dest", required=True)

    parser.add_argument("--data-dir", dest="data_dir", help="the STAGED scratch home folder (required to run)")
    parser.add_argument("--size", default=_SIZE_DEFAULT, help=f"window size (default {_SIZE_DEFAULT})")
    parser.add_argument("--sizes", nargs="*", default=None, help="run several sizes, e.g. 3456x2160 2560x1440")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--deadline", type=float, default=DEFAULT_DEADLINE_S)
    parser.add_argument("--platform", default="offscreen", choices=("offscreen", "windows"))
    parser.add_argument("--panels", nargs="*", default=None, choices=PANEL_NAMES)
    parser.add_argument("--out", default=None)
    parser.add_argument("--compare", default=None, help="a previous desk_bench JSON to diff against")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one repeat over two cheap pages at one size; what the suite runs",
    )
    return parser


def _real_diagnostics_dir() -> Path:
    """The machine's real diagnostics folder, resolved BEFORE LOCALAPPDATA moves."""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "TradingBotV3" / "diagnostics"
    return Path.home() / ".tradingbotv3" / "diagnostics"


def machine_settings_seed(real_settings: dict[str, Any]) -> dict[str, Any]:
    """The settings the bench's redirected `local_settings.json` starts from.

    `MACHINE_SETTINGS_CARRIED` keys only, plus the one setting a non-desk
    process must never default - `qt_autopilot_auto_arm` - forced False and not
    readable from the real file, so a trader who armed the desk cannot arm a
    bench. Anything a credential or a path lives under is not on the list and
    is not copied.
    """
    seed: dict[str, Any] = {}
    for key in MACHINE_SETTINGS_CARRIED:
        if key in real_settings:
            seed[key] = real_settings[key]
    seed["qt_autopilot_auto_arm"] = False
    return seed


def _read_real_machine_settings() -> tuple[dict[str, Any], str]:
    """Read `%LOCALAPPDATA%\\TradingBotV3\\local_settings.json`, read-only.

    Returns the payload and a one-line note for the printed output. A missing
    or unreadable file is a RESULT: the bench runs on the defaults and says so,
    because a measuring tool that refuses to start on a machine that never ran
    the desk is worse than one that names what it could not read.
    """
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return {}, "no LOCALAPPDATA on this machine; bench settings are defaults"
    path = Path(local) / "TradingBotV3" / "local_settings.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, f"{path} absent; bench settings are defaults"
    except (OSError, ValueError) as exc:
        return {}, f"{path} unreadable ({type(exc).__name__}); bench settings are defaults"
    if not isinstance(payload, dict):
        return {}, f"{path} is not an object; bench settings are defaults"
    return payload, f"carried from {path}"


def _prepare_environment(data_dir: Path, platform: str) -> str:
    """Point every store at the scratch BEFORE anything under `scripts/` loads."""
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRADINGBOTV3_DATA_DIR"] = str(data_dir)
    os.environ["TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE"] = "1"
    if platform != "windows":
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    else:
        os.environ.pop("QT_QPA_PLATFORM", None)

    # The machine-local half. A panel that writes a cache must not be able to
    # reach `%LOCALAPPDATA%\TradingBotV3` either, so it moves into the scratch
    # and gets the one setting a non-desk process must never default:
    # `qt_autopilot_auto_arm`. This bench builds no `MainWindow`, so nothing can
    # arm - but the 2026-09-02 failure was a timer nobody expected to exist.
    #
    # The rest of the seed is the TRADER's display settings, carried key by key
    # out of the real machine-local file BEFORE `LOCALAPPDATA` moves. This is
    # not decoration: `qt_ui_scale` scales every `theme.px`, so a bench running
    # on the default scale measures a layout the trader never sees. Credentials
    # and path keys are not on the list (`MACHINE_SETTINGS_CARRIED`).
    real_settings, settings_note = _read_real_machine_settings()
    local_appdata = data_dir / "_localappdata"
    (local_appdata / "TradingBotV3").mkdir(parents=True, exist_ok=True)
    (local_appdata / "TradingBotV3" / "local_settings.json").write_text(
        json.dumps(machine_settings_seed(real_settings), indent=1) + "\n",
        encoding="utf-8",
    )
    os.environ["LOCALAPPDATA"] = str(local_appdata)
    os.environ["TRADINGBOT_DIAGNOSTICS_DIR"] = str(data_dir / "_diagnostics")

    root = Path(__file__).resolve().parents[2]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return settings_note


def _abort_if_live(stream) -> int | None:
    """Print the resolved store and refuse to touch a live one. Exit code 2."""
    import project_paths

    print(f"project_paths.DATA_DIR = {project_paths.DATA_DIR}", file=stream)
    print(f"project_paths.SHARED_HOME_DIR = {project_paths.SHARED_HOME_DIR}", file=stream)
    for label, path in (
        ("DATA_DIR", project_paths.DATA_DIR),
        ("SHARED_HOME_DIR", project_paths.SHARED_HOME_DIR),
    ):
        if is_under_live_store(path):
            print(
                f"REFUSED: {label} resolved to {path}, which is the live home folder "
                "or the DAS. desk_bench never runs against the live store.",
                file=stream,
            )
            return 2
    return None


def main(argv: Sequence[str] | None = None, *, stream=None) -> int:
    stream = stream or sys.stdout
    # A Setup Tracker tab is labelled with a ring character and the Windows
    # console defaults to cp1252, so the FIRST full run died in `print` with
    # the whole measurement already taken and thrown away. The table is
    # diagnostics: a character it cannot encode is worth a replacement glyph,
    # never a lost run.
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - a non-file stream
            pass
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "stage":
        try:
            result = stage(args.source, args.dest)
        except StageRefused as exc:
            print(f"REFUSED: {exc}", file=stream)
            return 2
        for name, size in result.copied:
            print(f"{format_bytes(size):>12}  {name}", file=stream)
        for name in result.absent:
            print(f"{'absent':>12}  {name}", file=stream)
        print(
            f"\nStaged {len(result.copied)} file(s), {format_bytes(result.total_bytes)} "
            f"into {args.dest}. {len(result.absent)} allowlist entr(y/ies) not present at the source.",
            file=stream,
        )
        return 0

    if not args.data_dir:
        parser.error("--data-dir is required: desk_bench never runs against the live store")

    # BEFORE anything is created. `_prepare_environment` mkdirs the data
    # directory and writes a settings file into it, so every refusal has to be
    # decided here, on the ARGUMENT, and not afterwards on the resolved store.
    # `_abort_if_live` still runs below: it answers a different question - what
    # `project_paths` actually resolved to - and a scratch directory whose
    # settings redirect the store elsewhere would pass this check and fail that
    # one.
    for label, candidate in (("--data-dir", args.data_dir), ("--out", args.out)):
        if not candidate:
            continue
        refusal = refuse_live_destination(label, candidate, stream)
        if refusal is not None:
            return refusal

    out_default = _real_diagnostics_dir()
    settings_note = _prepare_environment(Path(args.data_dir), args.platform)
    refusal = _abort_if_live(stream)
    if refusal is not None:
        return refusal
    print(f"machine settings: {settings_note}", file=stream)

    if args.sizes:
        sizes = [parse_size(text) for text in args.sizes]
    else:
        sizes = [parse_size(args.size)]
    panel_names = tuple(args.panels) if args.panels else PANEL_NAMES
    repeat = max(1, int(args.repeat))
    if args.smoke:
        sizes = sizes[:1]
        panel_names = SMOKE_PANEL_NAMES
        repeat = 1

    payload = run_bench(
        sizes=sizes,
        repeat=repeat,
        deadline_s=float(args.deadline),
        panel_names=panel_names,
        record_screens=(args.platform == "windows"),
    )

    print(
        "\nChrome measured from the widgets: "
        f"top bar {payload['chrome']['top_bar']} px + status bar {payload['chrome']['status_bar']} px "
        f"+ margins {payload['chrome']['central_margins']} px = {payload['chrome']['total']} px.",
        file=stream,
    )
    print("\n== OPERATIONS ==", file=stream)
    print(format_ops_table(payload["ops"]), file=stream)
    print("\n== LAYOUT FIT ==", file=stream)
    print(format_fit_table(payload["fit"]), file=stream)

    slow = [row for row in payload["ops"] if (row["sync_ms"]["p95"] or 0.0) > SLOW_OP_MS]
    if slow:
        print(f"\n{len(slow)} op(s) over {SLOW_OP_MS:.0f} ms sync p95:", file=stream)
        for row in slow:
            print(f"  {row['op']} @ {row['size']}: {_ms(row['sync_ms']['p95'])} ms", file=stream)
    overflow = [row for row in payload["fit"] if row["overflow"]]
    if overflow:
        print(f"\n{len(overflow)} page(s) flagged OVERFLOW:", file=stream)
        for row in overflow:
            print(
                f"  {row['page']} @ {row['size']}: needs {row['required']} px, "
                f"has {row['available']} px ({row['overflow_px']} px over)",
                file=stream,
            )
    if payload["skipped"]:
        print("\nPanels skipped (named, not faked):", file=stream)
        for entry in payload["skipped"]:
            print(f"  {entry['panel']} @ {entry['size']}: {entry['reason']}", file=stream)

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_path = out_default / f"desk_bench_{stamp}.json"
    # The default lands under the real `%LOCALAPPDATA%`, which is neither the
    # home folder nor the DAS - but it is resolved from the environment, so it
    # is checked like everything else rather than trusted. A refusal here costs
    # the run's numbers; that is the cheaper of the two mistakes.
    refusal = refuse_live_destination("--out", out_path, stream)
    if refusal is not None:
        return refusal
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", file=stream)

    if args.compare:
        baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print(f"\n== COMPARE vs {args.compare} ==", file=stream)
        print(format_compare(payload["ops"], baseline.get("ops", [])), file=stream)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
