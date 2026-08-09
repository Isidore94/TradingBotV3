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
the Drive-backed home folder and the ai_state file is ~38MB; reading either
on the GUI thread is the exact defect chart-perf-c existed to remove. The
only caller is :meth:`ui.services.chart_data_service.ChartDataService.
build_snapshots`, which runs on the chart pool, and the result rides the
existing ``snapshotReady`` delivery. Both loaders are mtime-cached, so a
session pays for each file once per scan that rewrites it.

Decision-support only: nothing here writes state, scores, or influences an
alert. It draws what the scan already decided.

Why this is a separate module and not part of ``chart_snapshot``: the ai_state
trendline record is also detector input (``d1_level_feed`` feeds the Technical
Integrity monitor), and the file-scoped ask-first rule puts any edit to those
files behind a question. Keeping the drawing path in its own module means the
paint-lines work touches no detector, scoring, or alert file at all.

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

#: Display order, and the order the paint-lines control lists them in.
LEVEL_GROUPS: tuple[tuple[str, str], ...] = (
    (GROUP_SMA, "Daily SMAs"),
    (GROUP_EMA, "EMAs"),
    (GROUP_AVWAP, "AVWAP bands"),
    (GROUP_HORIZONTAL, "D1 S/R"),
    (GROUP_PREV_DAY, "Prev-day H/L"),
    (GROUP_TRENDLINE, "D1 trendline"),
)
GROUP_NAMES: dict[str, str] = dict(LEVEL_GROUPS)

#: A trendline projects along its slope and goes wrong fast, so the scan's
#: view of it is only honest for a few sessions. Same budget ``d1_level_feed``
#: applies to the same record (TRENDLINE_MAX_AGE_DAYS); duplicated rather than
#: imported so the drawing path does not depend on a detector-input module.
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
#: Not chart_yellow: "AVWAPE prev" already owns yellow on this chart.
_TRENDLINE_COLOR = "chart_purple"

_ai_state_trendline_cache: dict[str, tuple[int, dict[str, dict[str, Any]]]] = {}
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
# loaders (worker threads only - these read Drive and a 38MB JSON)
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


def _ai_state_trendlines(path: Path) -> dict[str, dict[str, Any]]:
    """{symbol: {candidate, last_trade_date}} from ai_state, mtime-cached.

    Only the trendline record and the symbol's scan date survive the parse:
    the source file is ~38MB and holding a second copy of it in memory for
    the sake of one nested dict per symbol would be its own defect.
    """
    key = str(path)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return {}
    cached = _ai_state_trendline_cache.get(key)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        _log.warning("Chart levels could not read ai_state: %s", exc)
        return cached[1] if cached is not None else {}
    feed: dict[str, dict[str, Any]] = {}
    symbols = payload.get("symbols") if isinstance(payload, Mapping) else {}
    for symbol, entry in (symbols or {}).items():
        if not isinstance(entry, Mapping):
            continue
        candidate = entry.get("priority_trendline_candidate")
        if not isinstance(candidate, Mapping):
            # The break candidate is the same geometry after the line gave
            # way; it is still the line the trader is looking at.
            candidate = entry.get("priority_trendline_break_candidate")
        if not isinstance(candidate, Mapping):
            continue
        feed[str(symbol).strip().upper()] = {
            "candidate": dict(candidate),
            "last_trade_date": str(entry.get("last_trade_date") or ""),
        }
    _ai_state_trendline_cache[key] = (mtime_ns, feed)
    return feed


def reset_caches() -> None:
    """Drop both mtime caches. Tests only."""
    _ai_state_trendline_cache.clear()
    _level_store_cache.clear()


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
) -> list[dict[str, Any]]:
    """Every paint-line for ``symbol``'s D1 chart. Worker threads only.

    ``store_records`` / ``trendline_feed`` bypass the two file reads (tests,
    and any caller that already holds the data). A failure in one family never
    costs the others: the chart draws what it could load and stays quiet about
    what it could not, which is the same discipline the overlays use.
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

    return levels
