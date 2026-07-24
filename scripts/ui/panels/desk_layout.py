"""Trading Desk split presets and persistence.

The desk's proportions used to be hardcoded in two places and were never
saved, so every settings change reset whatever the trader had dragged. The
opening proportions here are deliberate rather than incidental: the visual
chart column leads, and its share GROWS with the window instead of shrinking,
which is what the old 1:2 stretch did.

Persistence is debounced because `save_local_setting` is a whole-file
read-modify-write with no caching - wiring it straight to `splitterMoved`
would rewrite the settings file on every mouse-move frame of a drag.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, QTimer

from project_paths import get_local_setting, save_local_setting

# Desk column split, as (chart column, setups column) weights. The chart column
# is the larger share and widens further on a big monitor: the setups table
# needs a bounded width to stay readable, while a candle chart uses every pixel
# it is given.
DESK_SPLIT_NARROW = (52, 48)
DESK_SPLIT_WIDE = (58, 42)
# Above this desk content width the wide preset applies.
WIDE_DESK_THRESHOLD = 2000

# Alert Center column, top to bottom: chart pane, feed tabs, hidden detail
# pane. Weights, not pixels - they are scaled to the real column height.
ALERT_COLUMN_WEIGHTS = (62, 33, 0)

SAVE_DEBOUNCE_MS = 400


def desk_split_for(width: int) -> tuple[int, int]:
    """Column weights for a desk of this content width."""
    return DESK_SPLIT_WIDE if int(width or 0) >= WIDE_DESK_THRESHOLD else DESK_SPLIT_NARROW


def scaled_sizes(weights, total: int) -> list[int]:
    """Turn relative weights into pixel sizes summing to `total`."""
    weights = [max(0, int(weight)) for weight in weights]
    span = sum(weights)
    if span <= 0 or total <= 0:
        return [max(0, int(total // max(len(weights), 1)))] * len(weights)
    sizes = [int(total * weight / span) for weight in weights]
    sizes[0] += total - sum(sizes)  # absorb the rounding remainder
    return sizes


def load_sizes(key: str, expected_count: int) -> list[int] | None:
    """Read a persisted split, or None if it is absent or no longer valid.

    A stored split is rejected rather than trusted when the widget count has
    changed (the Alert Center column went from four children to three) or when
    it contains a non-positive entry, which would silently collapse a pane the
    trader cannot then find.
    """
    raw = get_local_setting(key, None)
    if not isinstance(raw, (list, tuple)) or len(raw) != expected_count:
        return None
    sizes: list[int] = []
    for value in raw:
        try:
            size = int(value)
        except (TypeError, ValueError):
            return None
        sizes.append(size)
    if sum(sizes) <= 0 or any(size < 0 for size in sizes):
        return None
    return sizes


def apply_saved_sizes(splitter, key: str, fallback_weights) -> None:
    """Restore a persisted split, falling back to the preset weights.

    Called during construction the splitter has no laid-out extent yet, so the
    preset is applied against whatever it reports and then re-applied on the
    first real resize (see `track_preset`). Without that second pass the
    weights are scaled against a placeholder size and Qt's own distribution
    wins - which is how the chart pane ended up pinned near its size hint.
    """
    count = splitter.count()
    saved = load_sizes(key, count)
    if saved is not None:
        splitter.setSizes(saved)
        return
    apply_weights(splitter, fallback_weights)


def apply_weights(splitter, weights) -> None:
    """Set the split from relative weights against the current extent."""
    count = splitter.count()
    vertical = splitter.orientation() == Qt.Orientation.Vertical
    total = splitter.height() if vertical else splitter.width()
    sized = list(weights)[:count] or [1] * count
    while len(sized) < count:
        sized.append(0)
    splitter.setSizes(scaled_sizes(sized, total or sum(sized) * 10))


class _PresetTracker(QObject):
    """Holds a splitter at its preset weights until the trader drags it.

    An event filter rather than an overridden `resizeEvent`: assigning
    `splitter.resizeEvent = fn` on a plain QSplitter instance does not reach
    Qt's virtual dispatch, so the preset silently never re-applied and the
    children's size hints won the split instead. The setups workspace hints
    1562px wide, so losing this pass costs the chart column ~8% of the desk.
    """

    def __init__(self, owner, splitter, key: str, weights_for) -> None:
        super().__init__(owner)
        self._splitter = splitter
        self._weights_for = weights_for
        self._user_dragged = load_sizes(key, splitter.count()) is not None
        splitter.splitterMoved.connect(self._on_moved)
        splitter.installEventFilter(self)

    def _on_moved(self, *_args) -> None:
        self._user_dragged = True

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        if watched is self._splitter and event.type() == QEvent.Type.Resize:
            self.reapply()
        return False

    def reapply(self) -> None:
        if self._user_dragged:
            return
        vertical = self._splitter.orientation() == Qt.Orientation.Vertical
        extent = self._splitter.height() if vertical else self._splitter.width()
        if extent > 0:
            apply_weights(self._splitter, self._weights_for(extent))


def track_preset(owner, splitter, key: str, weights_for) -> _PresetTracker:
    """Re-apply the preset on resize until the trader drags the splitter.

    `weights_for` is called with the splitter's current extent so a preset can
    widen the chart column on a bigger monitor instead of holding a ratio.
    """
    tracker = _PresetTracker(owner, splitter, key, weights_for)
    trackers = getattr(owner, "_split_trackers", None)
    if trackers is None:
        trackers = {}
        owner._split_trackers = trackers
    trackers[key] = tracker
    return tracker


def persist_sizes(owner, splitter, key: str) -> None:
    """Save this splitter's sizes whenever the trader drags it, debounced.

    The timer is parented to `owner` and stashed on it so it survives as long
    as the panel does.
    """
    timers = getattr(owner, "_split_save_timers", None)
    if timers is None:
        timers = {}
        owner._split_save_timers = timers

    timer = QTimer(owner)
    timer.setSingleShot(True)
    timer.setInterval(SAVE_DEBOUNCE_MS)

    def _save() -> None:
        try:
            save_local_setting(key, [int(size) for size in splitter.sizes()])
        except OSError:
            pass

    timer.timeout.connect(_save)
    timers[key] = timer
    splitter.splitterMoved.connect(lambda *_args: timer.start())
