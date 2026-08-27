from __future__ import annotations

"""Paint-lines for the D1 chart: the levels the scan already knows about.

The master scan maintains every D1 level the trader reads a chart by - the
horizontal S/R stores (``hv_horizontal`` + ``cloud_flat`` JSONs under the
levels directory) and the directional trendline (``priority_trendline_
candidate`` in the ai_state file). :mod:`d1_level_feed` already shapes those
for BounceBot's Technical Integrity monitor; this module shapes the same
artifacts for *drawing*, which needs different things: a stable id per line,
a style, a label, and - for the trendline - the whole projected series rather
than one price.

**Everything here is I/O and must run on a worker.** The level store lives in
the shared home folder and the ai_state file is ~38MB; reading either
on the GUI thread is the exact defect chart-perf-c existed to remove. The
only caller is :meth:`ui.services.chart_data_service.ChartDataService.
build_snapshots`, which runs on the chart pool, and the result rides the
existing ``snapshotReady`` delivery. Both loaders are mtime-cached, so a
session pays for each file once per scan that rewrites it - and the ai_state
parse is :mod:`d1_level_feed`'s single shared one, so a session pays for that
38MB read once in total rather than once per consumer of it.

Decision-support only: nothing here writes state, scores, or influences an
alert. It draws what the scan already decided.

Why this is a separate module and not part of ``chart_snapshot``: the ai_state
trendline record is also detector input (``d1_level_feed`` feeds the Technical
Integrity monitor), and the file-scoped ask-first rule puts any edit to those
files behind a question. Keeping the drawing path in its own module means the
drawing decisions - which lines, what colour, how many - are made where no
detector can read them. The one thing this module takes from the detector
side is the raw ai_state parse itself (``load_ai_state_projection``, approved
by the trader 2026-08-09): the same bytes, read once, shaped separately.

Contract - each level is::

    {"id":      stable across sessions, see level_id() / TRENDLINE ids below
     "family":  d1_horizontal | d1_cloud_flat | prev_day_high | prev_day_low
                | d1_trendline
     "group":   the paint-lines toggle group this line belongs to
     "price":   float - the level, or a sloped line's value at the last bar
     "values":  None for a horizontal; a per-bar series for a sloped line
     "label":   what the legend/tooltip calls it
     "color":   a ui.theme role name (this module stays theme-agnostic)
     "width", "dash":  same style vocabulary as the overlay contract
     "conviction": 0..~2.0 for store levels, else None}

``values``, when present, aligns 1:1 with the snapshot's bars exactly like an
overlay, with None where the line is not defined.
"""

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

_log = logging.getLogger(__name__)

#: Toggle groups. The first three name lines ``build_d1_snapshot`` already
#: paints as overlays (A4 only gains them a switch); the last three are the
#: levels this module adds.
GROUP_SMA = "sma"
GROUP_EMA = "ema"
GROUP_AVWAP = "avwap"
GROUP_HORIZONTAL = "d1_horizontal"
GROUP_PREV_DAY = "prev_day"
GROUP_TRENDLINE = "d1_trendline"
#: R4 section 4: the trader's OWN armed alarms, drawn strictly read-only.
GROUP_ALERTS = "armed_alerts"
#: Phase 0.10: the AVWAP band challenger, drawn beside the champion's bands so
#: the trader can see the difference on a real chart. DEFAULT OFF - it is a
#: candidate under test, not a level the desk has decided anything with, and
#: nothing reads it: no zone arm, no alert, no detector.
GROUP_AVWAP_VARIANT = "avwap_variant"

#: Display order, and the order the paint-lines control lists them in.
LEVEL_GROUPS: tuple[tuple[str, str], ...] = (
    (GROUP_SMA, "Daily SMAs"),
    (GROUP_EMA, "EMAs"),
    (GROUP_AVWAP, "AVWAP bands"),
    (GROUP_HORIZONTAL, "D1 S/R"),
    (GROUP_PREV_DAY, "Prev-day H/L"),
    (GROUP_TRENDLINE, "D1 trendline"),
    (GROUP_ALERTS, "Armed alerts"),
    # Last on purpose: a challenger does not push the trader's own lines around
    # in a menu they read by position.
    (GROUP_AVWAP_VARIANT, "AVWAP σ variant"),
)
GROUP_NAMES: dict[str, str] = dict(LEVEL_GROUPS)

#: Groups that are OFF until the trader switches them on. Every other group in
#: `LEVEL_GROUPS` defaults ON, deliberately, so a group added by a later version
#: appears switched on rather than silently missing. A formula under test is the
#: opposite case: it must not appear on a chart nobody asked for it on.
GROUPS_HIDDEN_BY_DEFAULT: frozenset[str] = frozenset({GROUP_AVWAP_VARIANT})

#: A trendline projects along its slope and goes wrong fast, so the scan's
#: view of it is only honest for a few sessions. Same budget ``d1_level_feed``
#: applies to the same record (TRENDLINE_MAX_AGE_DAYS); still duplicated
#: rather than imported, so that re-tuning a detector's freshness budget can
#: never silently change what the trader sees drawn.
TRENDLINE_MAX_AGE_DAYS = 5

#: Store levels below this strength never earned green-bucket-quality respect.
#: Matches ``d1_level_feed.MIN_HORIZONTAL_STRENGTH``.
MIN_HORIZONTAL_STRENGTH = 1.0

#: Clutter budget per bucket, highest conviction first. A chart with forty
#: lines on it is a chart nobody reads.
MAX_GREEN_HORIZONTALS = 10
MAX_RED_HORIZONTALS = 6
MAX_CLOUD_FLATS = 4

#: Styling (chosen here, 2026-08-09): green solid and weighted by conviction,
#: red faint and dashed, cloud dotted. The trader reads a chart by colour
#: first, and the point of the split is that a level worth respecting looks
#: different from one merely on record.
_GREEN_COLOR = "chart_green"
_RED_COLOR = "chart_grey"
_CLOUD_COLOR = "chart_light_blue"
_PREV_DAY_COLOR = "chart_white"
#: Armed alerts deliberately do NOT take a ``chart_*`` role. All eight of those
#: are fixed overlay assignments the trader reads by colour first (SMA200
#: purple, EMA21 yellow, AVWAPE white, ...), so reusing one would make the
#: trader's own alarm look like a moving average. ``caution`` is a semantic
#: theme role, defined in both themes, and used nowhere else on this chart.
_ALERT_COLOR = "caution"
#: A symbol carries at most two price-alert sides; level watches are armed by
#: hand one at a time. The cap only stops a corrupt store from painting a wall.
MAX_ARMED_ALERT_LEVELS = 12
#: Not chart_yellow: "AVWAPE prev" already owns yellow on this chart.
_TRENDLINE_COLOR = "chart_purple"
#: One colour for all six challenger lines, deliberately unlike the champion's
#: three-colour band set: they read as one alternative reading of the same
#: anchor, not as a second set of levels to trade off.
_VARIANT_COLOR = "chart_pink"

#: This module's key into ``d1_level_feed``'s shared ai_state parse.
_TRENDLINE_PROJECTION = "chart_levels.trendline"

_level_store_cache: dict[str, tuple[int, list[dict[str, Any]]]] = {}


# --------------------------------------------------------------------------
# stable ids
# --------------------------------------------------------------------------
def level_id(family: str, anchor: str, price: float | None) -> str:
    """A level's identity, stable across sessions.

    ``family`` + the date the level was ANCHORED (``first_seen`` for a
    high-volume horizontal, the effective-range start for a cloud flat, the
    session for a prev-day extreme) + the price to the cent.

    The stores carry no id of their own, so this is derived - and derivation
    has one honest limit worth stating: a clustered horizontal's ``price`` is
    a volume-weighted mean of its members, so a level that absorbs a new
    member and shifts by more than a cent gets a NEW id. The capture row also
    records ``ref_level_family`` and the price it referenced, which is what
    any later analysis actually joins on, so a re-clustered level costs a link
    rather than the evidence. If a future writer gives store levels their own
    id field, :func:`_store_levels` prefers it and this rule stops applying.
    """
    price_text = "" if price is None else f"{float(price):.2f}"
    return f"{family}:{anchor or '-'}:{price_text}"


def trendline_id(candidate: Mapping[str, Any]) -> str:
    """A trendline's identity: its type and its two pivots.

    Deliberately no price. The line projects along its slope, so its price
    moves every single session while it stays the same line drawn through the
    same two pivots - a price-bearing id would churn daily and never join.
    """
    kind = str(candidate.get("type") or "trendline")
    start = str(candidate.get("start_date") or "")
    end = str(candidate.get("end_date") or "")
    return f"{GROUP_TRENDLINE}:{kind}:{start}_{end}"


# --------------------------------------------------------------------------
# overlay grouping (the toggle's other half)
# --------------------------------------------------------------------------
def overlay_group(label: object) -> str:
    """Which toggle group an existing snapshot overlay belongs to.

    ``build_d1_snapshot`` already paints the SMAs, EMAs and AVWAPE bands, and
    A4's job for those is the switch, not the painting - so the toggle has to
    be able to name them from what the overlay contract carries, which is a
    label.
    """
    raw = str(label or "").strip()
    text = raw.upper()
    if text.startswith("SMA"):
        return GROUP_SMA
    if text.startswith("EMA"):
        return GROUP_EMA
    # The sigma test runs against the RAW label: str.upper() folds "σ" to "Σ",
    # so the band labels would stop matching the moment they were normalized.
    if text.startswith("AVWAP") or "σ" in raw:  # AVWAPE, AVWAPE prev, ±kσ
        return GROUP_AVWAP
    return ""


def visible_overlays(
    overlays: Sequence[Mapping[str, Any]], hidden_groups: Sequence[str]
) -> list[dict[str, Any]]:
    """``overlays`` minus every one whose group the trader switched off.

    An overlay in no group (nothing today, but the contract is open) is never
    hidden by a group switch - a line the control cannot name is a line the
    control must not silently remove.
    """
    hidden = {str(name) for name in hidden_groups or ()}
    if not hidden:
        return [dict(overlay) for overlay in overlays or ()]
    kept = []
    for overlay in overlays or ():
        group = overlay_group(overlay.get("label"))
        if group and group in hidden:
            continue
        kept.append(dict(overlay))
    return kept


def visible_levels(
    levels: Sequence[Mapping[str, Any]], hidden_groups: Sequence[str]
) -> list[dict[str, Any]]:
    """``levels`` minus every one whose group the trader switched off."""
    hidden = {str(name) for name in hidden_groups or ()}
    return [
        dict(level)
        for level in levels or ()
        if str(level.get("group") or "") not in hidden
    ]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _bar_date(bar: Mapping[str, Any]) -> date | None:
    stamp = bar.get("dt")
    if isinstance(stamp, datetime):
        return stamp.date()
    if isinstance(stamp, date):
        return stamp
    return stamp.date() if hasattr(stamp, "date") else None


def _bar_index_by_date(bars: Sequence[Mapping[str, Any]]) -> dict[date, int]:
    index: dict[date, int] = {}
    for position, bar in enumerate(bars or ()):
        stamp = _bar_date(bar)
        if stamp is not None:
            index.setdefault(stamp, position)
    return index


# --------------------------------------------------------------------------
# loaders (worker threads only - these read the home folder and a 38MB JSON)
# --------------------------------------------------------------------------
def _store_levels(symbol: str, levels_dir: Path) -> list[dict[str, Any]]:
    """Raw hv_horizontal + cloud_flat records for ``symbol``, mtime-cached."""
    from master_avwap_lib.levels import level_store_path

    path = level_store_path(Path(levels_dir), symbol)
    key = str(path)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return []
    cached = _level_store_cache.get(key)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A torn or unreachable store is uncertainty, not "no levels here":
        # keep whatever was last read rather than blanking the chart.
        return cached[1] if cached is not None else []
    records: list[dict[str, Any]] = []
    raw = payload.get("levels") if isinstance(payload, Mapping) else None
    for level in raw or ():
        if isinstance(level, Mapping) and str(level.get("kind") or "") in {
            "hv_horizontal",
            "cloud_flat",
        }:
            records.append(dict(level))
    _level_store_cache[key] = (mtime_ns, records)
    return records


def _trendline_record(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """One symbol's ai_state record -> its trendline sliver, or None.

    Only the trendline record and the symbol's scan date survive the parse:
    the source file is ~38MB and holding a second copy of it in memory for
    the sake of one nested dict per symbol would be its own defect.
    """
    candidate = entry.get("priority_trendline_candidate")
    if not isinstance(candidate, Mapping):
        # The break candidate is the same geometry after the line gave way;
        # it is still the line the trader is looking at.
        candidate = entry.get("priority_trendline_break_candidate")
    if not isinstance(candidate, Mapping):
        return None
    return {
        "candidate": dict(candidate),
        "last_trade_date": str(entry.get("last_trade_date") or ""),
    }


def _ai_state_trendlines(path: Path) -> dict[str, dict[str, Any]]:
    """{symbol: {candidate, last_trade_date}} from ai_state, mtime-cached.

    The 38MB parse itself belongs to :func:`d1_level_feed.
    load_ai_state_projection`, which both consumers of that file share: this
    module used to run an identical second read of the same bytes for the
    sake of a different sliver. Only the shaping below is ours; the caching
    and the read-failure rule (last good result stands) are the shared
    loader's.
    """
    from d1_level_feed import load_ai_state_projection

    return load_ai_state_projection(_TRENDLINE_PROJECTION, _trendline_record, path)


def reset_caches() -> None:
    """Drop the store cache and the shared ai_state parse. Tests only."""
    _level_store_cache.clear()
    from d1_level_feed import reset_ai_state_cache

    reset_ai_state_cache()


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def prev_day_levels(bars: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The previous session's high and low, from the bars already in hand.

    "Previous" is relative to the session the chart is SHOWING, whether that
    last bar is a completed daily bar or today's forming preview: the last bar
    whose date is strictly earlier than the last bar's date. Computed here
    from the snapshot rather than imported, so the paint path never reaches
    into ``bounce_bot_lib``.
    """
    bars = list(bars or ())
    if len(bars) < 2:
        return []
    current = _bar_date(bars[-1])
    if current is None:
        return []
    previous = None
    for bar in reversed(bars[:-1]):
        stamp = _bar_date(bar)
        if stamp is not None and stamp < current:
            previous = bar
            break
    if previous is None:
        return []
    stamp = _bar_date(previous)
    anchor = stamp.isoformat() if stamp else ""
    out: list[dict[str, Any]] = []
    for family, key, label in (
        ("prev_day_high", "high", "PDH"),
        ("prev_day_low", "low", "PDL"),
    ):
        price = _coerce_float(previous.get(key))
        if price is None or price <= 0:
            continue
        out.append(
            {
                "id": level_id(family, anchor, price),
                "family": family,
                "group": GROUP_PREV_DAY,
                "price": price,
                "values": None,
                "label": f"{label} {price:.2f}",
                "color": _PREV_DAY_COLOR,
                "width": 1.0,
                "dash": True,
                "conviction": None,
            }
        )
    return out


def armed_alert_levels(
    symbol: str,
    *,
    price_alerts: Sequence[Mapping[str, Any]] | None = None,
    level_watches: Sequence[Any] | None = None,
    event_watches: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """The trader's armed alarms for ``symbol``, as paint-lines (R4 section 4).

    Read-only display, and that is the whole contract. This function opens no
    file, writes no file, and arms/disarms nothing: it is handed whatever the
    two single-writer stores already hold and turns the armed ones into lines.
    Arming still goes through the one existing writer flow, and clicking a
    painted line only selects it via the ordinary ``levelSelected`` path.

    Three deliberate choices:

    - **A disarmed side paints nothing.** ``price_alerts.json`` keeps the price
      after a disarm so the trader can re-arm it; a line for a disarmed side
      would be an alarm that is not going to ring.
    - **A D1 EVENT watch contributes no line.** ``D1EventWatch`` is a condition
      (a new N-day extreme, an SMA break, a 15EMA rejection) whose reference
      level is re-derived on every poll. It has no armed price, and picking one
      for it would draw a level the trader never chose. Event watches stay
      visible as ``ArmBar`` chips, which is where a condition belongs.
    - **Nothing is dropped for being off-chart.** Painted levels are excluded
      from the y-autoscale, so a far-away alert costs nothing to draw - and
      hiding a live alarm because today's range does not reach it is exactly
      the wrong failure.

    Accepts either dataclass instances (``D1LevelWatch``) or their mapping form,
    so a caller holding one shape never has to convert.
    """
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return []

    def _field(item: Any, name: str) -> Any:
        if isinstance(item, Mapping):
            return item.get(name)
        return getattr(item, name, None)

    def _price(value: Any) -> float | None:
        price = _coerce_float(value)
        return price if price is not None and price > 0 else None

    out: list[dict[str, Any]] = []

    for entry in price_alerts or ():
        if str(_field(entry, "symbol") or "").strip().upper() != symbol:
            continue
        for side, arrow in (("above", "↑"), ("below", "↓")):
            price = _price(_field(entry, side))
            if price is None:
                continue
            # Absent flag means armed: that is normalize_price_alert's own
            # default, and a raw store row written before the flags existed
            # is an armed alert, not a disarmed one.
            armed = _field(entry, f"armed_{side}")
            if armed is None:
                armed = True
            if not armed:
                continue
            family = f"price_alert_{side}"
            out.append(
                {
                    "id": level_id(family, symbol, price),
                    "family": family,
                    "group": GROUP_ALERTS,
                    "price": price,
                    "values": None,
                    "label": f"Alert {arrow} {price:.2f}",
                    "color": _ALERT_COLOR,
                    "width": 1.4,
                    "dash": False,
                    "conviction": None,
                }
            )

    for watch in level_watches or ():
        if str(_field(watch, "symbol") or "").strip().upper() != symbol:
            continue
        direction = str(_field(watch, "direction") or "").strip().lower()
        if direction not in ("above", "below"):
            continue
        price = _price(_field(watch, "level"))
        if price is None:
            continue
        armed_at = _field(watch, "armed_at")
        anchor = ""
        if isinstance(armed_at, datetime):
            anchor = armed_at.date().isoformat()
        elif armed_at:
            anchor = str(armed_at)[:10]
        family = f"d1_level_watch_{direction}"
        arrow = "↑" if direction == "above" else "↓"
        out.append(
            {
                "id": level_id(family, anchor or symbol, price),
                "family": family,
                "group": GROUP_ALERTS,
                "price": price,
                "values": None,
                "label": f"Watch {arrow} {price:.2f}",
                "color": _ALERT_COLOR,
                "width": 1.4,
                "dash": True,
                "conviction": None,
            }
        )

    # Event watches are accepted and deliberately ignored - see the docstring.
    # Taking the argument keeps callers from having to know that, and keeps the
    # omission a stated decision rather than a forgotten store.
    del event_watches

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for level in out:
        if level["id"] in seen:
            continue
        seen.add(level["id"])
        deduped.append(level)
    return deduped[:MAX_ARMED_ALERT_LEVELS]


def trendline_level(
    candidate: Mapping[str, Any],
    bars: Sequence[Mapping[str, Any]],
    *,
    scan_date: date | None = None,
    max_age_days: int = TRENDLINE_MAX_AGE_DAYS,
) -> dict[str, Any] | None:
    """The scan's D1 trendline, projected onto ``bars``. None when it cannot be.

    The stored record is fully projectable: ``slope_log_per_bar`` is the
    line's slope in LOG price space and ``current_line_price`` is its value at
    ``lookback_end``, so for a bar ``i`` sessions from that anchor the line
    sits at ``current_line_price * exp(slope * i)``. Both the scan's frame and
    these bars come from the same durable daily store, so one step of ``i`` is
    one trading day on both sides and the projection is exact rather than
    approximated from a calendar.

    Returns None - never a guess - when any of the following is true, because
    a line drawn from a record that cannot support it is worse than no line
    (plan.md sec 5: missing data is uncertainty, never confirmation):

    * ``slope_log_per_bar`` is absent (a record written before the field
      existed). There is no honest way to recover a slope from one price.
    * ``lookback_end`` names no bar on this chart, so the projection has no
      anchor to count sessions from.
    * the scan that produced it is more than ``max_age_days`` older than the
      last session on the chart.
    """
    bars = list(bars or ())
    if not bars:
        return None
    slope = _coerce_float(candidate.get("slope_log_per_bar"))
    anchor_price = _coerce_float(candidate.get("current_line_price"))
    if slope is None or anchor_price is None or anchor_price <= 0:
        return None
    end_date = _parse_date(candidate.get("lookback_end"))
    if end_date is None:
        return None

    by_date = _bar_index_by_date(bars)
    anchor_index = by_date.get(end_date)
    if anchor_index is None:
        return None

    if scan_date is not None:
        last_date = _bar_date(bars[-1])
        if last_date is not None:
            age = (last_date - scan_date).days
            if age > int(max_age_days) or age < 0:
                return None

    start_date = _parse_date(candidate.get("start_date"))
    start_index = by_date.get(start_date, 0) if start_date is not None else 0

    values: list[float | None] = [None] * len(bars)
    for position in range(start_index, len(bars)):
        exponent = slope * (position - anchor_index)
        # exp() overflows long before a price does; a projection that far out
        # is meaningless anyway, so break the line instead of drawing inf.
        if abs(exponent) > 50:
            continue
        price = anchor_price * math.exp(exponent)
        if math.isfinite(price) and price > 0:
            values[position] = price
    if all(value is None for value in values):
        return None

    last_value = next((value for value in reversed(values) if value is not None), None)
    kind = str(candidate.get("type") or "")
    touches = int(_coerce_float(candidate.get("touch_count")) or 0)
    label = f"D1 trendline {kind}".strip()
    if touches:
        label += f" ({touches} touches)"
    return {
        "id": trendline_id(candidate),
        "family": GROUP_TRENDLINE,
        "group": GROUP_TRENDLINE,
        "price": last_value,
        "values": values,
        "label": label,
        "color": _TRENDLINE_COLOR,
        "width": 1.4,
        "dash": False,
        "conviction": None,
    }


def horizontal_levels(
    records: Sequence[Mapping[str, Any]],
    *,
    as_of: date | None = None,
    price_range: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Store levels shaped for drawing, filtered to what is worth drawing.

    Three filters, in order: effective on ``as_of`` (a cloud flat is only in
    force inside its displaced range), inside the chart's price range (a level
    the trader cannot see is clutter in the payload and nothing else - the
    chart's y-range follows the candles and does not pan), and then a clutter
    budget per bucket taking the highest-conviction levels first.
    """
    from master_avwap_lib.levels import level_conviction, level_is_effective_on

    low, high = price_range if price_range else (None, None)
    green: list[tuple[float, dict]] = []
    red: list[tuple[float, dict]] = []
    cloud: list[tuple[float, dict]] = []

    for record in records or ():
        price = _coerce_float(record.get("price"))
        if price is None or price <= 0:
            continue
        kind = str(record.get("kind") or "")
        is_cloud = kind == "cloud_flat"
        strength = _coerce_float(record.get("strength")) or 0.0
        if not is_cloud and strength < MIN_HORIZONTAL_STRENGTH:
            continue
        if as_of is not None and not level_is_effective_on(dict(record), as_of):
            continue
        if low is not None and not (low <= price <= high):
            continue
        conviction = level_conviction(dict(record))
        if is_cloud:
            anchor = ""
            effective = record.get("effective_range")
            if isinstance(effective, list) and effective:
                anchor = str(effective[0] or "")
            anchor = anchor or str(record.get("first_seen") or "")
            level = {
                "id": _record_id(record, "d1_cloud_flat", anchor, price),
                "family": "d1_cloud_flat",
                "group": GROUP_HORIZONTAL,
                "price": price,
                "values": None,
                "label": f"Cloud flat {price:.2f}",
                "color": _CLOUD_COLOR,
                "width": 1.0,
                "dash": "dot",
                "conviction": conviction,
            }
            cloud.append((conviction, level))
            continue
        bucket = str(record.get("bucket") or "red").lower()
        is_green = bucket == "green"
        anchor = str(record.get("first_seen") or "")
        touches = int(_coerce_float(record.get("touch_count")) or 0)
        level = {
            "id": _record_id(record, "d1_horizontal", anchor, price),
            "family": "d1_horizontal",
            "group": GROUP_HORIZONTAL,
            "price": price,
            "values": None,
            "label": (
                f"{'HV' if is_green else 'hv'} {price:.2f}"
                + (f" ×{touches}" if touches else "")
            ),
            "color": _GREEN_COLOR if is_green else _RED_COLOR,
            # Conviction earns line weight, so the level that has held five
            # times does not read the same as one nobody has tested.
            "width": (1.0 + min(0.8, conviction / 2.5)) if is_green else 0.9,
            "dash": False if is_green else True,
            "conviction": conviction,
        }
        (green if is_green else red).append((conviction, level))

    def _top(entries: list[tuple[float, dict]], budget: int) -> list[dict]:
        entries.sort(key=lambda item: (-item[0], item[1]["price"]))
        return [level for _conviction, level in entries[:budget]]

    out = (
        _top(green, MAX_GREEN_HORIZONTALS)
        + _top(red, MAX_RED_HORIZONTALS)
        + _top(cloud, MAX_CLOUD_FLATS)
    )
    out.sort(key=lambda level: level["price"])
    return out


def avwap_variant_levels(
    bars: Sequence[Mapping[str, Any]],
    anchor_index: int,
) -> list[dict[str, Any]]:
    """The AVWAP band challenger's +/-1/2/3, as six sloped paint-lines.

    SHADOW ONLY (plan.md Phase 0.10 / `docs/AVWAP_BAND_VARIANT_STUDY.md`). An
    anchored HLC/3 centre with a 20-close population Bollinger sigma as its
    half-width, replicated from OneOption / Option Stalker Pro on 2026-08-26.
    The champion's own bands (`calc_anchored_vwap_bands`, frozen by decision
    0008) are drawn by `chart_snapshot` as overlays and are not touched here.

    A bar the challenger cannot measure is `None` in the series, exactly like
    every other sloped line in this module - a warm-up bar is unmeasurable,
    never a band sitting on its centre.

    Worker threads only, like everything else here. It is pure arithmetic over
    the bars already in hand, so it costs no I/O at all - but it runs where the
    rest of the payload is built, never on the paint path.
    """
    bars = list(bars or ())
    if not bars:
        return []
    try:
        anchor = int(anchor_index)
    except (TypeError, ValueError):
        return []
    if not 0 <= anchor < len(bars):
        return []

    try:
        from indicators.avwap_band_variants import oneoption_avwap_band_series

        series = oneoption_avwap_band_series(bars, anchor)
    except Exception:
        _log.debug("AVWAP variant bands unavailable.", exc_info=True)
        return []

    anchor_stamp = _bar_date(bars[anchor])
    anchor_key = anchor_stamp.isoformat() if anchor_stamp else str(anchor)

    out: list[dict[str, Any]] = []
    for multiple, width in ((1, 1.1), (2, 1.0), (3, 0.9)):
        for sign, side in ((1, "upper"), (-1, "lower")):
            values = list(series[f"{side}_{multiple}"])
            last = next((value for value in reversed(values) if value is not None), None)
            if last is None:
                # Not one measurable bar in the whole window: draw nothing
                # rather than an empty line the legend would still list.
                continue
            out.append(
                {
                    "id": f"{GROUP_AVWAP_VARIANT}:{anchor_key}:{sign * multiple:+d}",
                    "family": GROUP_AVWAP_VARIANT,
                    "group": GROUP_AVWAP_VARIANT,
                    "price": float(last),
                    "values": values,
                    "label": f"{'+' if sign > 0 else '-'}{multiple}\u03c3 var",
                    "color": _VARIANT_COLOR,
                    "width": width,
                    "dash": "dot",
                    "conviction": None,
                }
            )
    return out


def _record_id(
    record: Mapping[str, Any], family: str, anchor: str, price: float
) -> str:
    """The store's own id when it has one, else the derived rule."""
    for key in ("id", "level_id"):
        existing = str(record.get(key) or "").strip()
        if existing:
            return existing
    return level_id(family, anchor, price)


def build_d1_levels(
    symbol: str,
    bars: Sequence[Mapping[str, Any]],
    *,
    levels_dir: Path | None = None,
    ai_state_path: Path | None = None,
    store_records: Sequence[Mapping[str, Any]] | None = None,
    trendline_feed: Mapping[str, Mapping[str, Any]] | None = None,
    price_alerts_path: Path | None = None,
    d1_level_watches_path: Path | None = None,
    avwap_anchor: date | str | None = None,
) -> list[dict[str, Any]]:
    """Every paint-line for ``symbol``'s D1 chart. Worker threads only.

    ``store_records`` / ``trendline_feed`` bypass the two file reads (tests,
    and any caller that already holds the data). A failure in one family never
    costs the others: the chart draws what it could load and stays quiet about
    what it could not, which is the same discipline the overlays use.

    ``price_alerts_path`` / ``d1_level_watches_path`` do the same for R4's
    armed-alert family. Both reads are strictly read-only; the single-writer
    rule on each store is untouched.

    ``avwap_anchor`` is the current AVWAPE anchor date the snapshot already
    resolved. Passing it in rather than resolving it again is the point: the
    challenger's centre must be anchored on exactly the bar the champion's is,
    or the two lines on the chart would differ for two reasons at once. Omitted
    (or not a session in ``bars``), the challenger group is simply absent.
    """
    symbol = str(symbol or "").strip().upper()
    bars = list(bars or ())
    if not symbol or not bars:
        return []

    levels: list[dict[str, Any]] = []
    lows = [value for value in (_coerce_float(bar.get("low")) for bar in bars) if value]
    highs = [value for value in (_coerce_float(bar.get("high")) for bar in bars) if value]
    price_range = (min(lows), max(highs)) if lows and highs else None
    as_of = _bar_date(bars[-1])

    try:
        if store_records is None:
            from project_paths import MASTER_AVWAP_LEVELS_DIR

            store_records = _store_levels(
                symbol, Path(levels_dir or MASTER_AVWAP_LEVELS_DIR)
            )
        levels.extend(
            horizontal_levels(store_records, as_of=as_of, price_range=price_range)
        )
    except Exception:
        _log.debug("D1 horizontal levels unavailable for %s.", symbol, exc_info=True)

    try:
        levels.extend(prev_day_levels(bars))
    except Exception:
        _log.debug("Prev-day levels unavailable for %s.", symbol, exc_info=True)

    try:
        if trendline_feed is None:
            from project_paths import MASTER_AVWAP_AI_STATE_FILE

            trendline_feed = _ai_state_trendlines(
                Path(ai_state_path or MASTER_AVWAP_AI_STATE_FILE)
            )
        entry = (trendline_feed or {}).get(symbol)
        if isinstance(entry, Mapping):
            line = trendline_level(
                entry.get("candidate") or {},
                bars,
                scan_date=_parse_date(entry.get("last_trade_date")),
            )
            if line is not None:
                levels.append(line)
    except Exception:
        _log.debug("D1 trendline unavailable for %s.", symbol, exc_info=True)

    try:
        from chart_watch import load_d1_level_watches
        from price_alerts import load_price_alerts
        from project_paths import D1_LEVEL_WATCHES_FILE, PRICE_ALERTS_FILE

        levels.extend(
            armed_alert_levels(
                symbol,
                price_alerts=load_price_alerts(
                    Path(price_alerts_path or PRICE_ALERTS_FILE)
                ),
                level_watches=load_d1_level_watches(
                    Path(d1_level_watches_path or D1_LEVEL_WATCHES_FILE)
                ),
            )
        )
    except Exception:
        _log.debug("Armed alert levels unavailable for %s.", symbol, exc_info=True)

    try:
        anchor_day = _parse_date(avwap_anchor)
        if anchor_day is not None:
            index = _bar_index_by_date(bars).get(anchor_day)
            if index is not None:
                levels.extend(avwap_variant_levels(bars, index))
    except Exception:
        _log.debug("AVWAP variant levels unavailable for %s.", symbol, exc_info=True)

    return levels
