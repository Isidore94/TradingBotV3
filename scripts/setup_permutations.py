"""Setup permutation keys (WISHLIST P1-4 / 4a): family x named facets, pure.

A facet is a small versioned rule over one D1 scan row (a `d1_features_history.csv`
row as a mapping) plus an optional context mapping for data that lives in other
stores. Every facet reads only what the row knew at its own scan date. Missing,
blank or NaN input gives ``unknown`` - never a default.

No I/O, no Qt, and nothing reads the key for ranking. Adding a facet is one
function decorated with ``@facet(...)`` plus its own fixture test.

Column names were checked against the writer (`master_avwap_lib/legacy.py`) and
the live CSV header. Sign conventions on the row:

- ``distance_from_current_<level>`` = close - level (positive: price above).
- ``hv_level_nearest_distance_atr`` / ``cloud_level_nearest_distance_atr`` =
  (level - price) / ATR (positive: level above price), from ``levels.levels_near``.
- ``dist_<ma>_atr`` (not on the row yet) follows the research warehouse:
  (close - ma) / ATR (positive: price above the MA).
- ``current_band_zone`` is "A to B" with the order flipped for shorts, so the zone
  is normalised to low-to-high band order before it becomes a value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Mapping

PERMUTATION_RULE_VERSION = "setup_permutations.v1"
UNKNOWN = "unknown"

FacetFn = Callable[[Mapping[str, Any], Mapping[str, Any], str], str]


@dataclass(frozen=True)
class FacetSpec:
    name: str
    group: str
    fn: FacetFn
    # Values left out of the short display label (``unknown`` is always left out).
    quiet: frozenset[str]
    # False keeps the facet on the key but out of the short label.
    in_label: bool = True


#: name -> spec, in registration order (the order of the key and the label).
FACETS: dict[str, FacetSpec] = {}


def facet(
    name: str,
    group: str,
    *,
    quiet: tuple[str, ...] = (),
    in_label: bool = True,
) -> Callable[[FacetFn], FacetFn]:
    """Register ``fn(row, ctx, side) -> value`` as the facet ``name``."""

    def _register(fn: FacetFn) -> FacetFn:
        if name in FACETS:
            raise ValueError(f"facet {name!r} is already registered")
        FACETS[name] = FacetSpec(name=name, group=group, fn=fn, quiet=frozenset(quiet), in_label=in_label)
        return fn

    return _register


@dataclass(frozen=True)
class PermutationKey:
    family: str
    side: str
    facets: tuple[tuple[str, str], ...]
    permutation_rule_version: str = PERMUTATION_RULE_VERSION

    def as_dict(self) -> dict[str, str]:
        return dict(self.facets)

    def get(self, name: str) -> str:
        return self.as_dict().get(name, UNKNOWN)

    @property
    def key(self) -> str:
        """Full stable key string: every facet, unknowns included."""
        body = ";".join(f"{name}={value}" for name, value in self.facets)
        return f"{self.permutation_rule_version}|{self.family}|{self.side}|{body}"

    @property
    def label(self) -> str:
        """Short display label, e.g. ``sma100_support|weekly_ema15_hold``."""
        parts = []
        for name, value in self.facets:
            spec = FACETS.get(name)
            if value == UNKNOWN or (spec is not None and (not spec.in_label or value in spec.quiet)):
                continue
            parts.append(value)
        return "|".join(parts)


def facets_for_row(
    row: Mapping[str, Any],
    ctx: Mapping[str, Any] | None = None,
) -> PermutationKey:
    """Build the permutation key for one scan row. ``ctx`` holds other-store data."""
    context: Mapping[str, Any] = ctx or {}
    side = _side(row.get("side"))
    family = _text(row.get("setup_family")) or UNKNOWN
    values = []
    for name, spec in FACETS.items():
        try:
            value = spec.fn(row, context, side)
        except (TypeError, ValueError, ArithmeticError):
            value = UNKNOWN
        values.append((name, value or UNKNOWN))
    return PermutationKey(family=family, side=side, facets=tuple(values))


# --- value parsing: blank / NaN / junk -> None (the caller turns None into unknown)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "nat"}:
        return True
    return False


def _text(value: Any) -> str | None:
    if _missing(value):
        return None
    return str(value).strip()


def _num(value: Any) -> float | None:
    if _missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _flag(value: Any) -> bool | None:
    """The CSV holds booleans as True/False and, in older rows, 1.0/0.0."""
    if _missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return None if value not in (0, 1) else bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    return None


def _side(value: Any) -> str:
    text = (_text(value) or "").upper()
    return text if text in {"LONG", "SHORT"} else UNKNOWN


def _band(value: float | None, edges: tuple[float, ...], names: tuple[str, ...]) -> str:
    """Bucket ``value``: ``names[i]`` holds values below ``edges[i]``; the last name the rest."""
    if value is None:
        return UNKNOWN
    for edge, name in zip(edges, names):
        if value < edge:
            return name
    return names[-1]


def _yes_no(value: Any, yes: str, no: str) -> str:
    flag = _flag(value)
    if flag is None:
        return UNKNOWN
    return yes if flag else no


def _date(value: Any) -> date | None:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def _atr_distance(row: Mapping[str, Any], column: str) -> float | None:
    distance = _num(row.get(column))
    atr = _num(row.get("atr20"))
    if distance is None or not atr or atr <= 0:
        return None
    return distance / atr


_ATR_DIST_EDGES = (-2.0, -1.0, 0.0, 1.0, 2.0)
_ATR_DIST_NAMES = ("below_2atr", "below_1to2atr", "below_0to1atr", "above_0to1atr", "above_1to2atr", "above_2atr")


# --- D1 structure: anchor


@facet("anchor_age", "anchor", in_label=False)
def _anchor_age(row, ctx, side):
    anchor = _date(row.get("current_anchor_date"))
    scanned = _date(row.get("run_date"))
    if anchor is None or scanned is None or scanned < anchor:
        return UNKNOWN
    days = (scanned - anchor).days
    return _band(float(days), (11.0, 31.0, 61.0), ("anchor_0_10d", "anchor_11_30d", "anchor_31_60d", "anchor_61d_plus"))


@facet("previous_anchor_level", "anchor", quiet=("prev_anchor_far",))
def _previous_anchor_level(row, ctx, side):
    # The writer leaves both columns blank in every live row so far (2026-09-24).
    level = _text(row.get("previous_anchor_nearest_level"))
    distance = _num(row.get("previous_anchor_nearest_distance_atr"))
    if level is None or distance is None:
        return UNKNOWN
    if abs(distance) > 1.0:
        return "prev_anchor_far"
    return f"prev_anchor_{level.lower()}_near"


# --- D1 structure: band zone

_BAND_RANK = {"LOWER_3": 0, "LOWER_2": 1, "LOWER_1": 2, "VWAP": 3, "UPPER_1": 4, "UPPER_2": 5, "UPPER_3": 6}


def _band_token(name: str) -> str:
    return name.lower().replace("_", "")


@facet("band_zone", "band_zone")
def _band_zone(row, ctx, side):
    text = _text(row.get("current_band_zone"))
    if not text:
        return UNKNOWN
    names = [part.strip().upper() for part in text.split(" to ")]
    if any(name not in _BAND_RANK for name in names) or not 1 <= len(names) <= 2:
        return UNKNOWN
    if len(names) == 1:
        return f"at_{_band_token(names[0])}"
    low, high = sorted(names, key=_BAND_RANK.__getitem__)
    return f"{_band_token(low)}_{_band_token(high)}"


@facet("vwap_distance", "band_zone", in_label=False)
def _vwap_distance(row, ctx, side):
    value = _band(_atr_distance(row, "distance_from_current_vwap"), _ATR_DIST_EDGES, _ATR_DIST_NAMES)
    return value if value == UNKNOWN else f"vwap_{value}"


@facet("upper1_distance", "band_zone", in_label=False)
def _upper1_distance(row, ctx, side):
    value = _band(_atr_distance(row, "distance_from_current_upper_1"), _ATR_DIST_EDGES, _ATR_DIST_NAMES)
    return value if value == UNKNOWN else f"upper1_{value}"


@facet("lower1_distance", "band_zone", in_label=False)
def _lower1_distance(row, ctx, side):
    value = _band(_atr_distance(row, "distance_from_current_lower_1"), _ATR_DIST_EDGES, _ATR_DIST_NAMES)
    return value if value == UNKNOWN else f"lower1_{value}"


# --- D1 structure: moving averages

#: The support-side MA set. Each needs ``dist_<ma>_atr`` on the row; ``ema21`` can
#: also be derived from the ``ema21`` / ``last_close`` / ``atr20`` columns.
SUPPORT_MAS = ("sma20", "sma50", "sma100", "sma200", "ema8", "ema15", "ema21")


def _ma_distance_atr(row: Mapping[str, Any], ma: str) -> float | None:
    distance = _num(row.get(f"dist_{ma}_atr"))
    if distance is not None:
        return distance
    level = _num(row.get(ma))
    close = _num(row.get("last_close"))
    atr = _num(row.get("atr20"))
    if level is None or close is None or not atr or atr <= 0:
        return None
    return (close - level) / atr


@facet("ma_support", "ma_support", quiet=("no_ma_support",))
def _ma_support(row, ctx, side):
    # Needs every MA known: a partial set cannot tell "none" from "multiple".
    if side == UNKNOWN:
        return UNKNOWN
    distances = {ma: _ma_distance_atr(row, ma) for ma in SUPPORT_MAS}
    if any(value is None for value in distances.values()):
        return UNKNOWN
    if side == "LONG":
        hits = [ma for ma, d in distances.items() if 0.0 <= d <= 1.0]
    else:
        hits = [ma for ma, d in distances.items() if -1.0 <= d <= 0.0]
    if not hits:
        return "no_ma_support"
    if len(hits) > 1:
        return "multiple_ma_support"
    return f"{hits[0]}_support"


@facet("trend_ma_alignment", "ma_stack", quiet=("ema15_sma20_not_aligned",))
def _trend_ma_alignment(row, ctx, side):
    # The writer stores False (not blank) when EMA15/SMA20 were missing; only a blank is unknown.
    return _yes_no(row.get("trend_ma_alignment"), "ema15_sma20_aligned", "ema15_sma20_not_aligned")


@facet("price_vs_ema21", "ma_stack", quiet=("above_ema21", "below_ema21"))
def _price_vs_ema21(row, ctx, side):
    distance = _ma_distance_atr(row, "ema21")
    if distance is None:
        return UNKNOWN
    return "above_ema21" if distance >= 0 else "below_ema21"


@facet("ma_order", "ma_stack", in_label=False)
def _ma_order(row, ctx, side):
    # Needs dist_sma50_atr and dist_sma200_atr (or sma50/sma200 values) on the row.
    distances = {"ema21": _ma_distance_atr(row, "ema21"), "sma50": _ma_distance_atr(row, "sma50"),
                 "sma200": _ma_distance_atr(row, "sma200")}
    if any(value is None for value in distances.values()):
        return UNKNOWN
    # A level's height relative to price is -distance; price sits at 0.
    heights = {"price": 0.0, **{ma: -d for ma, d in distances.items()}}
    return ">".join(sorted(heights, key=lambda name: -heights[name]))


# --- D1 structure: weekly


@facet("weekly_ema15_hold", "weekly", quiet=("no_weekly_ema15_hold",))
def _weekly_ema15_hold(row, ctx, side):
    return _yes_no(row.get("top_pattern_weekly_ema15_hold"), "weekly_ema15_hold", "no_weekly_ema15_hold")


@facet("weekly_above_sma100", "weekly", quiet=("weekly_below_sma100",))
def _weekly_above_sma100(row, ctx, side):
    return _yes_no(row.get("top_pattern_weekly_above_sma100"), "weekly_above_sma100", "weekly_below_sma100")


@facet("weekly_sma50_retest", "weekly", quiet=("no_weekly_sma50_retest",))
def _weekly_sma50_retest(row, ctx, side):
    return _yes_no(row.get("top_pattern_weekly_sma50_retest_recent"), "weekly_sma50_retest", "no_weekly_sma50_retest")


@facet("weekly_ema8_streak", "weekly", quiet=("weekly_ema8_hold_0w",))
def _weekly_ema8_streak(row, ctx, side):
    # `weekly_ema8_hold_weeks` is computed by the scan but not written to d1_features_history yet.
    weeks = _num(row.get("weekly_ema8_hold_weeks"))
    if weeks is None or weeks < 0:
        return UNKNOWN
    return _band(weeks, (1.0, 3.0, 6.0), ("weekly_ema8_hold_0w", "weekly_ema8_hold_1_2w",
                                          "weekly_ema8_hold_3_5w", "weekly_ema8_hold_6w_plus"))


# --- D1 structure: levels


def _nearby_level(row, prefix: str, bucket_col: str | None) -> str:
    count = _num(row.get(f"{prefix}_nearby_count"))
    if count is None:
        return UNKNOWN
    blocking = _num(row.get(f"{prefix}_blocking_count")) if prefix == "hv_level" else 0.0
    if count <= 0 and not blocking:
        return f"no_{prefix}"
    distance = _num(row.get(f"{prefix}_nearest_distance_atr"))
    if distance is None:
        return UNKNOWN
    where = "above" if distance > 0 else "below" if distance < 0 else "at"
    bucket = _text(row.get(bucket_col)) if bucket_col else None
    if bucket_col and not bucket:
        return UNKNOWN
    name = f"{prefix}_{bucket.lower()}" if bucket else prefix
    return f"{name}_{where}"


@facet("hv_level", "levels", quiet=("no_hv_level",))
def _hv_level(row, ctx, side):
    return _nearby_level(row, "hv_level", "hv_level_nearest_bucket")


@facet("cloud_level", "levels", quiet=("no_cloud_level",))
def _cloud_level(row, ctx, side):
    return _nearby_level(row, "cloud_level", None)


@facet("prev_day_range_break", "levels", quiet=("no_pdr_break",))
def _prev_day_range_break(row, ctx, side):
    return _yes_no(row.get("previous_day_range_break"), "pdr_break", "no_pdr_break")


@facet("compression", "levels", quiet=("not_compressed",))
def _compression(row, ctx, side):
    broke_today = _flag(row.get("compression_break_today"))
    if broke_today:
        direction = (_text(row.get("compression_break_direction")) or "").lower()
        return f"compression_break_{direction}" if direction in {"up", "down"} else UNKNOWN
    recent = _flag(row.get("compression_break_recent"))
    if recent:
        return "compression_break_recent"
    flag = _flag(row.get("compression_flag"))
    if flag is None or broke_today is None or recent is None:
        return UNKNOWN
    return "compressed" if flag else "not_compressed"


# --- D1 structure: strength


@facet("rs_vs_spy", "strength", in_label=False)
def _rs_vs_spy(row, ctx, side):
    # The writer leaves daily_relative_strength_score blank in every live row so far (2026-09-24).
    score = _num(row.get("daily_relative_strength_score"))
    if score is None:
        return UNKNOWN
    if score >= 1.0:
        return "rs_spy_strong"
    if score <= -1.0:
        return "rs_spy_weak"
    return "rs_spy_neutral"


@facet("rs_vs_industry_5d", "strength", in_label=False)
def _rs_vs_industry_5d(row, ctx, side):
    value = _num(row.get("rs_vs_industry_5d"))
    if value is None:
        return UNKNOWN
    if value > 0:
        return "beats_industry_5d"
    if value < 0:
        return "lags_industry_5d"
    return "industry_5d_flat"


@facet("industry_13w", "strength", in_label=False)
def _industry_13w(row, ctx, side):
    return _band(_num(row.get("industry_13w_return_pct")), (-10.0, 0.0, 10.0),
                 ("industry_13w_below_neg10", "industry_13w_neg10_0", "industry_13w_0_10", "industry_13w_above_10"))


@facet("relvol", "strength", in_label=False)
def _relvol(row, ctx, side):
    value = _num(row.get("relvol"))
    if value is not None and value < 0:
        return UNKNOWN
    return _band(value, (0.8, 1.5, 3.0), ("relvol_below_0_8", "relvol_0_8_1_5", "relvol_1_5_3", "relvol_3_plus"))


# --- D1 structure: earnings


@facet("earnings_phase", "earnings", quiet=("no_earnings_phase",))
def _earnings_phase(row, ctx, side):
    post = _flag(row.get("post_earnings_active"))
    mid = _flag(row.get("mid_earnings_watch"))
    if post:
        return "post_earnings"
    if mid:
        return "mid_earnings"
    if post is None or mid is None:
        return UNKNOWN
    return "no_earnings_phase"


@facet("earnings_gap_age", "earnings", in_label=False)
def _earnings_gap_age(row, ctx, side):
    sessions = _num(row.get("latest_release_sessions_since_gap"))
    if sessions is None or sessions < 0:
        return UNKNOWN
    return _band(sessions, (6.0, 21.0, 41.0), ("gap_0_5s", "gap_6_20s", "gap_21_40s", "gap_41s_plus"))


@facet("next_earnings", "earnings", in_label=False)
def _next_earnings(row, ctx, side):
    days = _num(row.get("days_to_next_earnings"))
    if days is None or days < 0:
        return UNKNOWN
    return _band(days, (6.0, 15.0, 31.0), ("earnings_0_5d", "earnings_6_14d", "earnings_15_30d", "earnings_31d_plus"))


# --- other stores (ctx): unknown until a caller supplies them

DISCOVERY_SLOTS = ("0730", "1000", "1245", "close")


@facet("discovery_slot", "discovery")
def _discovery_slot(row, ctx, side):
    # From scan_replay: which scan slot first showed the name.
    slot = (_text(ctx.get("discovery_slot")) or "").lower().replace(":", "")
    return f"slot_{slot}" if slot in DISCOVERY_SLOTS else UNKNOWN


# --- entry timing (H1/H4)


@facet("htf_trend", "htf", in_label=False)
def _htf_trend(row, ctx, side):
    h1 = (_text(row.get("htf_trend_1h")) or "").upper()
    h4 = (_text(row.get("htf_trend_4h")) or "").upper()
    allowed = {"UP", "DOWN", "NEUTRAL"}
    if h1 not in allowed or h4 not in allowed:
        return UNKNOWN
    return f"h1_{h1.lower()}_h4_{h4.lower()}"


@facet("htf_aligned", "htf", quiet=("htf_not_aligned",))
def _htf_aligned(row, ctx, side):
    return _yes_no(row.get("htf_trend_aligned"), "htf_aligned", "htf_not_aligned")


@facet("htf_retest", "htf", quiet=("no_htf_retest",))
def _htf_retest(row, ctx, side):
    confirmed = _flag(row.get("htf_retest_confirmed"))
    if confirmed is None:
        return UNKNOWN
    if not confirmed:
        return "no_htf_retest"
    smas = _text(row.get("htf_retest_sma"))
    if not smas:
        return UNKNOWN
    parts = []
    for item in smas.split(";"):
        timeframe, _, sma = item.strip().partition(":")
        if not timeframe or not sma:
            return UNKNOWN
        parts.append(f"{timeframe.lower()}_{sma.lower().replace('_', '')}")
    return "htf_retest_" + "+".join(sorted(parts))


@facet("entry_trigger", "entry", quiet=("no_trigger",))
def _entry_trigger(row, ctx, side):
    # From alert_chart_watches.json history / review events: the trigger that fired before entry.
    trigger = _text(ctx.get("entry_trigger"))
    return trigger.lower() if trigger else UNKNOWN


@facet("m5_confirmation", "entry", quiet=("no_m5_confirmation",))
def _m5_confirmation(row, ctx, side):
    # From the M5 alert log: an alert on the same name and side that day. The caller passes
    # "none" only when it checked and found none; absent means unknown.
    raw = ctx.get("m5_bounce_type")
    if isinstance(raw, str) and raw.strip().lower() == "none":
        return "no_m5_confirmation"
    bounce = _text(raw)
    return f"m5_{bounce.lower()}" if bounce else UNKNOWN


# --- market


@facet("d1_environment", "market", in_label=False)
def _d1_environment(row, ctx, side):
    # Not on the scan row: the caller joins d1_environment_store labels by session.
    label = _text(ctx.get("d1_environment"))
    return f"env_{label.lower()}" if label and label.lower() != UNKNOWN else UNKNOWN


@facet("market_regime", "market", in_label=False)
def _market_regime(row, ctx, side):
    label = _text(row.get("market_regime_label"))
    return f"regime_{label.lower()}" if label else UNKNOWN


@facet("spy_trend", "market", in_label=False)
def _spy_trend(row, ctx, side):
    above20 = _flag(row.get("spy_above_sma20"))
    above50 = _flag(row.get("spy_above_sma50"))
    if above20 is None or above50 is None:
        return UNKNOWN
    word = {True: "above", False: "below"}
    return f"spy_{word[above20]}_sma20_{word[above50]}_sma50"


@facet("spy_5d", "market", in_label=False)
def _spy_5d(row, ctx, side):
    return _band(_num(row.get("spy_five_day_return_pct")), (-2.0, 0.0, 2.0),
                 ("spy_5d_below_neg2", "spy_5d_neg2_0", "spy_5d_0_2", "spy_5d_above_2"))


_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@facet("weekday", "market", in_label=False)
def _weekday(row, ctx, side):
    scanned = _date(row.get("run_date"))
    return _WEEKDAYS[scanned.weekday()] if scanned else UNKNOWN


@facet("side_aligned_day", "market", quiet=("side_not_aligned_day",))
def _side_aligned_day(row, ctx, side):
    # The writer leaves side_aligned_day blank in every live row so far (2026-09-24).
    return _yes_no(row.get("side_aligned_day"), "side_aligned_day", "side_not_aligned_day")


# --- stamping (4a): the scan-row columns and the honest input view

#: `dist_<ma>_atr` columns the enrichment step writes: (close - ma) / ATR20.
MA_DISTANCE_COLUMNS = tuple(f"dist_{ma}_atr" for ma in SUPPORT_MAS)
#: The key, its short label and its rule version, as written on a row.
STAMP_COLUMNS = ("permutation_key", "permutation_label", "permutation_rule_version")
#: Every column 4a appends to `d1_features_history.csv`, in order.
SCAN_ROW_COLUMNS = (*MA_DISTANCE_COLUMNS, "weekly_ema8_hold_weeks", *STAMP_COLUMNS)

_WEEKLY_TOP_PATTERN_FLAGS = (
    "top_pattern_weekly_ema15_hold",
    "top_pattern_weekly_above_sma100",
    "top_pattern_weekly_sma50_retest_recent",
)


def ma_distance_columns(close: Any, atr: Any, levels: Mapping[str, Any] | None) -> dict[str, float | None]:
    """``dist_<ma>_atr`` for every support MA; None when close, ATR or the MA is missing."""
    close_value = _num(close)
    atr_value = _num(atr)
    source = levels if isinstance(levels, Mapping) else {}
    out: dict[str, float | None] = {}
    for ma in SUPPORT_MAS:
        level = _num(source.get(ma))
        if close_value is None or level is None or not atr_value or atr_value <= 0:
            out[f"dist_{ma}_atr"] = None
        else:
            out[f"dist_{ma}_atr"] = round((close_value - level) / atr_value, 6)
    return out


def scan_row_view(row: Mapping[str, Any], *, has_ma_columns: bool) -> dict[str, Any]:
    """A copy of ``row`` with yes/no columns blanked where the scan wrote False without computing.

    The CSV keeps those False values (legacy readers take ``bool(value)`` and NaN
    is truthy); only the key's input is made honest. ``has_ma_columns`` says the
    row was written with the ``dist_<ma>_atr`` columns, so a blank one means the
    MA was really missing rather than "row older than the column".
    """
    view = dict(row)
    if has_ma_columns and (
        _num(row.get("last_close")) is None
        or _num(row.get("dist_ema15_atr")) is None
        or _num(row.get("dist_sma20_atr")) is None
    ):
        view["trend_ma_alignment"] = None
    # The weekly flags are only computed when the TOP weekly structure holds; the ratio says so.
    if _num(row.get("top_pattern_weekly_ema15_hold_ratio")) is None:
        for column in _WEEKLY_TOP_PATTERN_FLAGS:
            view[column] = None
    side = _side(row.get("side"))
    level_column = {"LONG": "previous_day_high", "SHORT": "previous_day_low"}.get(side)
    if level_column in row and (_num(row.get(level_column)) is None or _num(row.get("last_close")) is None):
        view["previous_day_range_break"] = None
    return view


def stamp_fields(
    row: Mapping[str, Any],
    ctx: Mapping[str, Any] | None = None,
    *,
    has_ma_columns: bool = True,
) -> dict[str, str]:
    """The three stamp columns for one scan row, through the honest input view."""
    key = facets_for_row(scan_row_view(row, has_ma_columns=has_ma_columns), ctx)
    return {
        "permutation_key": key.key,
        "permutation_label": key.label,
        "permutation_rule_version": key.permutation_rule_version,
    }
