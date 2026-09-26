"""Setup permutation keys (WISHLIST P1-4 / 4a): family x named facets, pure.

A facet is a small versioned rule over one D1 scan row (a `d1_features_history.csv`
row as a mapping) plus an optional context mapping for data that lives in other
stores. Every facet reads only what the row knew at its own scan date. Missing,
blank or NaN input gives ``unknown`` - never a default.

No I/O, no Qt, and nothing reads the key for ranking. Adding a facet is one
function decorated with ``@facet(...)`` plus its own fixture test.

M5-native facets (group ``m5``, P11) live in their own registry, ``M5_FACETS``,
and make their own versioned key (``m5_facets_for``) over what one M5 alert
carried at alert time; the D1 key never includes them.

Column names were checked against the writer (`master_avwap_lib/legacy.py`) and
the live CSV header. Sign conventions on the row:

- ``distance_from_current_<level>`` = close - level (positive: price above).
- ``hv_level_nearest_distance_atr`` / ``cloud_level_nearest_distance_atr`` =
  (level - price) / ATR (positive: level above price), from ``levels.levels_near``.
- ``perm_dist_<ma>_atr`` (written by the enrichment step since 4a; ``dist_<ma>_atr`` is read as a
  fallback) follows the research warehouse:
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
    def compact_key(self) -> str:
        """The key with unknown facets left out (absent means unknown); what a row stores."""
        body = ";".join(f"{name}={value}" for name, value in self.facets if value != UNKNOWN)
        return f"{self.permutation_rule_version}|{self.family}|{self.side}|{body}"

    @property
    def label(self) -> str:
        """Short display label, e.g. ``sma100_support|weekly_ema15_hold``."""
        parts = []
        for name, value in self.facets:
            spec = FACETS.get(name) or M5_FACETS.get(name)
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
    for edge, name in zip(edges, names, strict=False):
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

#: The support-side MA set. Each needs ``perm_dist_<ma>_atr`` (or ``dist_<ma>_atr``) on the row; ``ema21`` can
#: also be derived from the ``ema21`` / ``last_close`` / ``atr20`` columns.
SUPPORT_MAS = ("sma20", "sma50", "sma100", "sma200", "ema8", "ema15", "ema21")


def _ma_distance_atr(row: Mapping[str, Any], ma: str) -> float | None:
    distance = _num(row.get(f"perm_dist_{ma}_atr"))
    if distance is None:
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
    # Needs the SMA50 and SMA200 distances (or sma50/sma200 values) on the row.
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
    # The scan's streak, written as `perm_weekly_ema8_hold_weeks` since 4a (LONG rows only).
    weeks = _num(row.get("perm_weekly_ema8_hold_weeks"))
    if weeks is None:
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


# --- 4f: wider facets (only where the row or its ctx holds the data at scan time)


@facet("earnings_gap_size", "earnings", in_label=False)
def _earnings_gap_size(row, ctx, side):
    # Written only inside the post-earnings window; blank elsewhere is unknown, never "no gap".
    value = _num(row.get("post_earnings_gap_atr_multiple"))
    if value is None:
        return UNKNOWN
    return _band(abs(value), (1.0, 2.0, 4.0), ("gap_below_1atr", "gap_1_2atr", "gap_2_4atr", "gap_4atr_plus"))


@facet("pullback_52w", "weekly", in_label=False)
def _pullback_52w(row, ctx, side):
    # Weekly close vs the 52-week high, in %; the scan writes it only when the TOP weekly structure holds.
    value = _num(row.get("top_pattern_weekly_pullback_from_52w_high_pct"))
    if value is None or value < 0:
        return UNKNOWN
    return _band(value, (5.0, 15.0, 30.0), ("off_52w_high_0_5pct", "off_52w_high_5_15pct",
                                            "off_52w_high_15_30pct", "off_52w_high_30pct_plus"))


@facet("htf_retest_age", "htf", in_label=False)
def _htf_retest_age(row, ctx, side):
    # Intraday bars since the HTF retest; only meaningful when a retest was confirmed.
    if not _flag(row.get("htf_retest_confirmed")):
        return UNKNOWN
    bars = _num(row.get("htf_retest_age_bars"))
    if bars is None or bars < 0:
        return UNKNOWN
    return _band(bars, (3.0, 8.0, 20.0), ("htf_retest_0_2bars", "htf_retest_3_7bars",
                                          "htf_retest_8_19bars", "htf_retest_20bars_plus"))


@facet("industry_rs_consistent", "strength", in_label=False)
def _industry_rs_consistent(row, ctx, side):
    # The writer stores False when the industry ETF or rs_vs_industry is missing; that is unknown.
    if not _text(row.get("industry_etf")) or _num(row.get("rs_vs_industry")) is None:
        return UNKNOWN
    return _yes_no(row.get("industry_rs_consistent"), "industry_rs_consistent", "industry_rs_mixed")


ENTRY_TRIGGER_CHECKPOINTS = ("open", "midday", "final hour", "close")


@facet("entry_trigger_time", "entry", in_label=False)
def _entry_trigger_time(row, ctx, side):
    # From ctx: the exchange-clock checkpoint of the first watch_fired that session.
    checkpoint = (_text(ctx.get("entry_trigger_checkpoint")) or "").lower()
    if checkpoint not in ENTRY_TRIGGER_CHECKPOINTS:
        return UNKNOWN
    return "trigger_" + checkpoint.replace(" ", "_")


# --- P11: D1 facets from the scan's own daily bars (appended `perm_` columns)


@facet("atr_percentile", "volatility", in_label=False)
def _atr_percentile(row, ctx, side):
    # ATR14 placed in its own 252-session range (0 = lowest, 1 = highest).
    value = _num(row.get("perm_atr14_pctile_252"))
    if value is None or not 0.0 <= value <= 1.0:
        return UNKNOWN
    return _band(value, (0.2, 0.5, 0.8), ("atr_pctile_0_20", "atr_pctile_20_50", "atr_pctile_50_80",
                                          "atr_pctile_80_100"))


@facet("low_52w_distance", "weekly", in_label=False)
def _low_52w_distance(row, ctx, side):
    value = _num(row.get("perm_low_52w_dist_atr"))
    if value is None or value < 0:
        return UNKNOWN
    return _band(value, (2.0, 5.0, 10.0), ("off_52w_low_0_2atr", "off_52w_low_2_5atr", "off_52w_low_5_10atr",
                                           "off_52w_low_10atr_plus"))


@facet("closes_vs_level", "levels", in_label=False)
def _closes_vs_level(row, ctx, side):
    # Of the last 5 closes, how many sat on the setup's side of the current AVWAPE.
    count = _num(row.get("perm_closes_right_of_level_5"))
    if count is None or not 0 <= count <= 5:
        return UNKNOWN
    if count >= 5:
        return "closes_right_5of5"
    return "closes_right_3_4of5" if count >= 3 else "closes_right_0_2of5"


@facet("level_respect", "levels", in_label=False)
def _level_respect(row, ctx, side):
    # Sessions in the last 20 that touched the current AVWAPE and closed on the setup's side.
    count = _num(row.get("perm_level_respect_20"))
    if count is None or count < 0:
        return UNKNOWN
    if count >= 4:
        return "level_respect_4_plus"
    if count >= 2:
        return "level_respect_2_3"
    return f"level_respect_{int(count)}"


@facet("d1_zone_arm", "band_zone", quiet=("no_zone_arm",))
def _d1_zone_arm(row, ctx, side):
    # The scan's own D1 zone arm for the name ("LONG_z1"); "not_armed" when evaluated and nothing armed.
    text = (_text(row.get("perm_d1_zone_arm")) or "").upper()
    if text == "NOT_ARMED":
        return "no_zone_arm"
    arm_side, _, zone = text.partition("_Z")
    if arm_side not in {"LONG", "SHORT"} or zone not in {"1", "2", "3"}:
        return UNKNOWN
    return f"zone_arm_{arm_side.lower()}_z{zone}"


# --- P8b: setup age from the setup tracker's first-seen record (appended `perm_` column)


@facet("setup_age", "age", in_label=False)
def _setup_age(row, ctx, side):
    # Completed sessions since the tracker first saw this symbol/side/family; blank = no record.
    value = _num(row.get("perm_setup_age_sessions"))
    if value is None or value < 0 or value != int(value):
        return UNKNOWN
    if value == 0:
        return "setup_age_0"
    return _band(value, (3.0, 6.0, 11.0), ("setup_age_1_2", "setup_age_3_5", "setup_age_6_10", "setup_age_11_plus"))


# --- S6: the scan's own D1 trendline read (appended `perm_trendline_*` columns)


@facet("trendline", "trendline", quiet=("no_trendline",))
def _trendline(row, ctx, side):
    # A recent break wins over a nearby line; the direction is the way a break of the line goes.
    broke = _flag(row.get("perm_trendline_break_recent"))
    near = _flag(row.get("perm_trendline_within_alert_range"))
    if broke is None or near is None:
        return UNKNOWN
    if not broke and not near:
        return "no_trendline"
    direction = (_text(row.get("perm_trendline_direction")) or "").lower()
    if direction not in {"up", "down"}:
        return UNKNOWN
    return f"trendline_break_{direction}" if broke else f"trendline_near_{direction}"


# --- stamping (4a): the scan-row columns and the honest input view

#: `perm_dist_<ma>_atr` columns the enrichment step writes: (close - ma) / ATR20. The `perm_`
#: prefix keeps them out of every legacy reader (`dist_sma50/200_atr` are scan-factor fields).
MA_DISTANCE_COLUMNS = tuple(f"perm_dist_{ma}_atr" for ma in SUPPORT_MAS)
#: The weekly EMA8 hold streak, copied from the scan's own feature-row value under a new name.
WEEKLY_STREAK_COLUMN = "perm_weekly_ema8_hold_weeks"
#: The compact key (unknown facets omitted), its short label and its rule version, as written on a row.
STAMP_COLUMNS = ("permutation_key", "permutation_label", "permutation_rule_version")
#: P11: the D1 facet inputs computed from the scan's own daily bars, appended after the 4a columns.
D1_HISTORY_COLUMNS = (
    "perm_atr14_pctile_252",
    "perm_low_52w_dist_atr",
    "perm_closes_right_of_level_5",
    "perm_level_respect_20",
    "perm_d1_zone_arm",
)
#: P8b: completed sessions since the setup tracker first saw this symbol/side/family.
SETUP_AGE_COLUMN = "perm_setup_age_sessions"
#: S6: the scan's own trendline read for priority rows (break, nearby line, break direction).
TRENDLINE_COLUMNS = (
    "perm_trendline_break_recent",
    "perm_trendline_within_alert_range",
    "perm_trendline_direction",
)
#: Every column P1-4 appends to `d1_features_history.csv`, in order (4a, then P11, then P8b, then S6).
SCAN_ROW_COLUMNS = (*MA_DISTANCE_COLUMNS, WEEKLY_STREAK_COLUMN, *STAMP_COLUMNS, *D1_HISTORY_COLUMNS,
                    SETUP_AGE_COLUMN, *TRENDLINE_COLUMNS)

_WEEKLY_TOP_PATTERN_FLAGS = (
    "top_pattern_weekly_ema15_hold",
    "top_pattern_weekly_above_sma100",
    "top_pattern_weekly_sma50_retest_recent",
)


def ma_distance_columns(close: Any, atr: Any, levels: Mapping[str, Any] | None) -> dict[str, float | None]:
    """``perm_dist_<ma>_atr`` for every support MA; None when close, ATR or the MA is missing."""
    close_value = _num(close)
    atr_value = _num(atr)
    source = levels if isinstance(levels, Mapping) else {}
    out: dict[str, float | None] = {}
    for ma in SUPPORT_MAS:
        level = _num(source.get(ma))
        if close_value is None or level is None or not atr_value or atr_value <= 0:
            out[f"perm_dist_{ma}_atr"] = None
        else:
            out[f"perm_dist_{ma}_atr"] = round((close_value - level) / atr_value, 6)
    return out


def scan_row_view(row: Mapping[str, Any], *, has_ma_columns: bool) -> dict[str, Any]:
    """A copy of ``row`` with yes/no columns blanked where the scan wrote False without computing.

    The CSV keeps those False values (legacy readers take ``bool(value)`` and NaN
    is truthy); only the key's input is made honest. ``has_ma_columns`` says the
    row was written with the ``perm_dist_<ma>_atr`` columns, so a blank one means the
    MA was really missing rather than "row older than the column".
    """
    view = dict(row)
    if has_ma_columns and (
        _num(row.get("last_close")) is None
        or _num(row.get("perm_dist_ema15_atr")) is None
        or _num(row.get("perm_dist_sma20_atr")) is None
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
        # Compact: ~1 KB per row saved on a file that grows by thousands of rows a day.
        "permutation_key": key.compact_key,
        "permutation_label": key.label,
        "permutation_rule_version": key.permutation_rule_version,
    }


# --- P11: the D1 history columns (pure; the scan passes the bars it already holds)

ATR_PERCENTILE_WINDOW = 252
ATR_PERCENTILE_LENGTH = 14
LOW_52W_WINDOW = 252
CLOSES_VS_LEVEL_WINDOW = 5
LEVEL_RESPECT_WINDOW = 20
#: A touch is a low (long) or high (short) within this many ATR of the level.
LEVEL_TOUCH_TOL_ATR = 0.1


def _bar_values(bar: Mapping[str, Any]) -> tuple[float, float, float] | None:
    high, low, close = _num(bar.get("high")), _num(bar.get("low")), _num(bar.get("close"))
    if high is None or low is None or close is None:
        return None
    return high, low, close


def _atr14_percentile(bars: list[tuple[float, float, float]]) -> float | None:
    trs = []
    previous_close = None
    for high, low, close in bars:
        tr = high - low if previous_close is None else max(high - low, abs(high - previous_close),
                                                           abs(low - previous_close))
        trs.append(tr)
        previous_close = close
    length = ATR_PERCENTILE_LENGTH
    atrs = [sum(trs[i - length + 1:i + 1]) / length for i in range(length - 1, len(trs))]
    if len(atrs) < ATR_PERCENTILE_WINDOW:
        return None
    window = atrs[-ATR_PERCENTILE_WINDOW:]
    low, high = min(window), max(window)
    if high - low <= 1e-12:
        return None
    return round((window[-1] - low) / (high - low), 6)


def d1_history_columns(
    daily_ohlc: Any,
    *,
    side: Any,
    level: Any,
    atr: Any,
    as_of: Any = None,
    zone_arm: Mapping[str, Any] | None = None,
    zone_arm_evaluated: bool = False,
) -> dict[str, Any]:
    """The P11 ``perm_`` columns for one scan row; None where the scan's bars cannot say.

    ``daily_ohlc`` is the scan's own list of ``{date, open, high, low, close}``
    dicts in date order; bars after ``as_of`` (ISO date) are never read. ``level``
    is the current AVWAPE; ``zone_arm`` the scan's zone-arm entry for the name.
    """
    as_of_text = str(as_of)[:10] if as_of else None
    bars = []
    for bar in daily_ohlc or ():
        if not isinstance(bar, Mapping):
            continue
        if as_of_text and str(bar.get("date") or "")[:10] > as_of_text:
            continue
        values = _bar_values(bar)
        if values is None:
            return dict.fromkeys(D1_HISTORY_COLUMNS)  # a hole in the bars: nothing is measured
        bars.append(values)
    side_text = _side(side)
    level_value = _num(level)
    atr_value = _num(atr)
    atr_ok = atr_value is not None and atr_value > 0
    out: dict[str, Any] = dict.fromkeys(D1_HISTORY_COLUMNS)
    out["perm_atr14_pctile_252"] = _atr14_percentile(bars)
    if atr_ok and len(bars) >= LOW_52W_WINDOW:
        year_low = min(low for _high, low, _close in bars[-LOW_52W_WINDOW:])
        out["perm_low_52w_dist_atr"] = round((bars[-1][2] - year_low) / atr_value, 6)
    if level_value is not None and side_text != UNKNOWN:
        long_side = side_text == "LONG"
        if len(bars) >= CLOSES_VS_LEVEL_WINDOW:
            closes = [close for _high, _low, close in bars[-CLOSES_VS_LEVEL_WINDOW:]]
            out["perm_closes_right_of_level_5"] = sum(
                1 for close in closes if (close >= level_value if long_side else close <= level_value)
            )
        if atr_ok and len(bars) >= LEVEL_RESPECT_WINDOW:
            tol = LEVEL_TOUCH_TOL_ATR * atr_value
            held = 0
            for high, low, close in bars[-LEVEL_RESPECT_WINDOW:]:
                if long_side and low <= level_value + tol and close >= level_value:
                    held += 1
                elif not long_side and high >= level_value - tol and close <= level_value:
                    held += 1
            out["perm_level_respect_20"] = held
    if zone_arm_evaluated:
        arm_side = _side(zone_arm.get("side")) if zone_arm else UNKNOWN
        zone = zone_arm.get("zone") if zone_arm else None
        out["perm_d1_zone_arm"] = f"{arm_side}_z{zone}" if arm_side != UNKNOWN and zone in (1, 2, 3) else "not_armed"
    return out


# --- P8b: the setup-age column (pure; the scan passes the tracker view and bar dates it holds)


def _setup_key(symbol: Any, side: Any, family: Any) -> tuple[str, str, str] | None:
    symbol_text = (_text(symbol) or "").upper()
    side_text = _side(side)
    family_text = (_text(family) or "").lower()
    if not symbol_text or side_text == UNKNOWN or not family_text:
        return None
    return symbol_text, side_text, family_text


def setup_first_seen_index(tracker_payload: Any) -> dict[tuple[str, str, str], list[str]]:
    """(symbol, side, setup family) -> sorted scan dates the setup tracker recorded for it."""
    setups = tracker_payload.get("setups") if isinstance(tracker_payload, Mapping) else None
    index: dict[tuple[str, str, str], set[str]] = {}
    for setup in setups.values() if isinstance(setups, Mapping) else ():
        if not isinstance(setup, Mapping):
            continue
        key = _setup_key(setup.get("symbol"), setup.get("side"), setup.get("setup_family"))
        scanned = _date(setup.get("scan_date"))
        if key is None or scanned is None:
            continue
        index.setdefault(key, set()).add(scanned.isoformat())
    return {key: sorted(dates) for key, dates in index.items()}


def setup_age_sessions(scan_dates: list[str] | None, as_of: Any, session_dates: list[str]) -> int | None:
    """Sessions in ``session_dates`` after the first scan date on or before ``as_of``, up to ``as_of``.

    None when no scan date is known at ``as_of`` or the calendar does not reach back to it.
    """
    as_of_day = _date(as_of)
    if as_of_day is None or not scan_dates or not session_dates:
        return None
    as_of_text = as_of_day.isoformat()
    known = [day for day in scan_dates if day <= as_of_text]
    if not known:
        return None
    first = known[0]
    if first < session_dates[0]:
        return None
    return sum(1 for day in session_dates if first < day <= as_of_text)


def setup_age_columns(feature_rows: Any, tracker_payload: Any, session_dates: Any) -> int:
    """Write ``perm_setup_age_sessions`` on each scan row in place (None = unknown). Returns rows aged.

    ``session_dates`` is the session calendar (ISO dates), or a callable giving it for one symbol
    (the scan passes each symbol's own daily-bar dates).
    """
    index = setup_first_seen_index(tracker_payload)
    aged = 0
    for row in feature_rows or ():
        if not isinstance(row, dict):
            continue
        key = _setup_key(row.get("symbol"), row.get("side"), row.get("setup_family"))
        scan_dates = index.get(key) if key else None
        age = None
        if scan_dates:
            days = session_dates(key[0]) if callable(session_dates) else session_dates
            calendar = sorted({str(day)[:10] for day in days or () if _date(day) is not None})
            age = setup_age_sessions(scan_dates, row.get("last_trade_date"), calendar)
        row[SETUP_AGE_COLUMN] = age
        aged += age is not None
    return aged


# --- S6: the trendline columns (pure; the scan passes the priority row it already refined)

#: The scan's trendline lookback (`legacy.PRIORITY_TRENDLINE_LOOKBACK_BARS`): with fewer cached bars
#: the scan fetched its own, so "no line found" cannot be told from "too few bars".
TRENDLINE_MIN_KNOWN_BARS = 200
_TRENDLINE_DIRECTIONS = {"H-": "up", "H-break": "up", "L+": "down", "L-break": "down"}


def trendline_columns(priority_row: Any, *, frame_bars: Any, last_close: Any, atr: Any) -> dict[str, Any]:
    """The S6 ``perm_trendline_*`` columns for one refined priority row; None where the scan cannot say.

    ``priority_row`` holds the scan's `find_directional_trendline_candidate` fields (a row the
    directional refine never looked at has none: unknown). ``frame_bars`` is the cached daily bar
    count the refine saw; "no line" is only claimed when it had the full lookback and an ATR.
    """
    out: dict[str, Any] = dict.fromkeys(TRENDLINE_COLUMNS)
    if not isinstance(priority_row, Mapping) or "trendline_break_recent" not in priority_row:
        return out
    broke = bool(priority_row.get("trendline_break_recent"))
    near = bool(priority_row.get("trendline_within_alert_range"))
    if broke or near:
        candidate = priority_row.get("trendline_break_candidate" if broke else "trendline_candidate")
        line_type = candidate.get("type") if isinstance(candidate, Mapping) else None
        direction = _TRENDLINE_DIRECTIONS.get(str(line_type or ""))
        if direction is None:
            return out
        out.update({TRENDLINE_COLUMNS[0]: broke, TRENDLINE_COLUMNS[1]: near, TRENDLINE_COLUMNS[2]: direction})
        return out
    bars = _num(frame_bars)
    atr_value = _num(atr)
    if bars is not None and bars >= TRENDLINE_MIN_KNOWN_BARS and _num(last_close) is not None \
            and atr_value is not None and atr_value > 0:
        out.update({TRENDLINE_COLUMNS[0]: False, TRENDLINE_COLUMNS[1]: False})
    return out


# --- P11: M5-native facets over one alert's own inputs (group ``m5``)

M5_PERMUTATION_RULE_VERSION = "setup_permutations.m5.v1"
#: name -> spec for the M5 key, in registration order. Never part of the D1 key.
M5_FACETS: dict[str, FacetSpec] = {}
#: The alert-time inputs an M5 facet reads (`m5_setup_key_stamp.alert_inputs` builds them).
M5_INPUT_FIELDS = (
    "alert_bar_close", "alert_bar_complete", "session_rvol", "vwap_dist_atr", "spy_state", "spy_side_sign",
    "bounce_type",
    # S6: structure over the cached bars (`m5_setup_key_stamp.structure_inputs`) and SPY's D1 label.
    "alert_price", "m5_ema8", "m5_ema21", "prev_day_high", "prev_day_low", "open_range_high", "open_range_low",
    "m5_range12_atr20", "m5_range12_break", "d1_environment",
)


def m5_facet(name: str, *, quiet: tuple[str, ...] = (), in_label: bool = True) -> Callable[[FacetFn], FacetFn]:
    """Register ``fn(inputs, ctx, side) -> value`` as the M5 facet ``name``."""

    def _register(fn: FacetFn) -> FacetFn:
        if name in M5_FACETS or name in FACETS:
            raise ValueError(f"facet {name!r} is already registered")
        M5_FACETS[name] = FacetSpec(name=name, group="m5", fn=fn, quiet=frozenset(quiet), in_label=in_label)
        return fn

    return _register


def m5_facets_for(inputs: Mapping[str, Any] | None, side: Any) -> PermutationKey:
    """The M5 key for one alert; family = its bounce type. Missing input is unknown."""
    source: Mapping[str, Any] = inputs or {}
    side_text = _side(side)
    family = _text(source.get("bounce_type")) or UNKNOWN
    values = []
    for name, spec in M5_FACETS.items():
        try:
            value = spec.fn(source, {}, side_text)
        except (TypeError, ValueError, ArithmeticError):
            value = UNKNOWN
        values.append((name, value or UNKNOWN))
    return PermutationKey(family=family, side=side_text, facets=tuple(values),
                          permutation_rule_version=M5_PERMUTATION_RULE_VERSION)


#: Minutes after 09:30 ET at which the alert bar CLOSED: (upper bound inclusive, name).
_M5_TIME_BUCKETS = ((30, "first30"), (120, "morning"), (330, "midday"), (390, "last60"))
#: US Eastern offsets (EDT, EST) in hours: the only offsets `entry_time` may carry.
_EXCHANGE_UTC_OFFSETS = (-4.0, -5.0)


@m5_facet("m5_time_bucket")
def _m5_time_bucket(inputs, ctx, side):
    # ``alert_bar_close`` is the alert bar's close in exchange time (`m5_setup_key_stamp.alert_inputs`).
    text = _text(inputs.get("alert_bar_close"))
    if not text:
        return UNKNOWN
    try:
        local = datetime.fromisoformat(text)
    except ValueError:
        return UNKNOWN
    offset = local.utcoffset()
    if offset is None or offset.total_seconds() / 3600.0 not in _EXCHANGE_UTC_OFFSETS:
        return UNKNOWN  # naive or not written in exchange time: never guessed
    minutes = local.hour * 60 + local.minute - (9 * 60 + 30)
    if minutes <= 0:
        return "time_extended"
    for upper, name in _M5_TIME_BUCKETS:
        if minutes <= upper:
            return name
    return "time_extended"


@m5_facet("m5_rvol_bucket")
def _m5_rvol_bucket(inputs, ctx, side):
    value = _num(inputs.get("session_rvol"))
    if value is None or value < 0:
        return UNKNOWN
    return _band(value, (1.0, 2.0, 3.0), ("rvol_below_1", "rvol_1_2", "rvol_2_3", "rvol_3_plus"))


@m5_facet("m5_vwap_dist_atr")
def _m5_vwap_dist_atr(inputs, ctx, side):
    # (alert-bar close - session VWAP) / M5 ATR14, both at the alert bar.
    value = _band(_num(inputs.get("vwap_dist_atr")), _ATR_DIST_EDGES, _ATR_DIST_NAMES)
    return value if value == UNKNOWN else f"m5vwap_{value}"


_SPY_COUNTERMOVE = {"COUNTERMOVE_ARMED", "COUNTERMOVE_ACTIVE", "STABILIZING"}
_SPY_TREND = {"BULL_IMPULSE": 1, "BEAR_IMPULSE": -1, "TREND_RESUMED": 0}
_SPY_QUIET = {"PREOPEN", "OPENING_DISCOVERY", "RANGE", "REGIME_FAILED"}


@m5_facet("m5_spy_state", quiet=("spy_none",))
def _m5_spy_state(inputs, ctx, side):
    # The SPY market-state engine's recorded state at the alert bar (`market_state_bridge` shadow log).
    state = (_text(inputs.get("spy_state")) or "").upper()
    sign = _num(inputs.get("spy_side_sign"))
    if state in _SPY_QUIET:
        return "spy_none"
    if sign not in (1.0, -1.0):
        return UNKNOWN
    if state in _SPY_COUNTERMOVE:
        return "spy_pullback" if sign > 0 else "spy_bounce"
    if state in _SPY_TREND:
        direction = _SPY_TREND[state] or int(sign)
        return "spy_rally" if direction > 0 else "spy_selloff"
    return UNKNOWN


@m5_facet("m5_bounce_type")
def _m5_bounce_type(inputs, ctx, side):
    bounce = _text(inputs.get("bounce_type"))
    return f"bounce_{bounce.lower().replace(' ', '_')}" if bounce else UNKNOWN


# --- S6: M5 structure facets (inputs from `m5_setup_key_stamp.structure_inputs`; never in the label)

#: A 12-bar box no wider than this many M5 ATR20 is a squeeze.
M5_SQUEEZE_RANGE_ATR = 2.5
#: The open-range facet only speaks once the alert bar starts at or after 10:00 ET (closes 10:05+).
_OPEN_RANGE_DONE_MINUTES = 10 * 60 + 5


def _above_inside_below(price: float | None, high: float | None, low: float | None,
                        names: tuple[str, str, str]) -> str:
    if price is None or high is None or low is None or high < low:
        return UNKNOWN
    if price > high:
        return names[0]
    return names[2] if price < low else names[1]


@m5_facet("m5_ema_stack", in_label=False)
def _m5_ema_stack(inputs, ctx, side):
    # EMA 8 vs 21 order, then the alert close against both.
    price, fast, slow = _num(inputs.get("alert_price")), _num(inputs.get("m5_ema8")), _num(inputs.get("m5_ema21"))
    if price is None or fast is None or slow is None:
        return UNKNOWN
    order = "8over21" if fast >= slow else "8under21"
    if price >= max(fast, slow):
        where = "above_both"
    elif price <= min(fast, slow):
        where = "below_both"
    else:
        where = "between"
    return f"m5ema_{order}_{where}"


@m5_facet("m5_pdh_pdl", in_label=False)
def _m5_pdh_pdl(inputs, ctx, side):
    return _above_inside_below(_num(inputs.get("alert_price")), _num(inputs.get("prev_day_high")),
                               _num(inputs.get("prev_day_low")), ("above_pdh", "inside_pd_range", "below_pdl"))


@m5_facet("m5_open_range", in_label=False)
def _m5_open_range(inputs, ctx, side):
    # Only after the first 30 minutes are over; the alert bar close must be in exchange time.
    text = _text(inputs.get("alert_bar_close"))
    try:
        local = datetime.fromisoformat(text) if text else None
    except ValueError:
        local = None
    offset = local.utcoffset() if local is not None else None
    if offset is None or offset.total_seconds() / 3600.0 not in _EXCHANGE_UTC_OFFSETS:
        return UNKNOWN
    if local.hour * 60 + local.minute < _OPEN_RANGE_DONE_MINUTES:
        return UNKNOWN
    return _above_inside_below(_num(inputs.get("alert_price")), _num(inputs.get("open_range_high")),
                               _num(inputs.get("open_range_low")), ("above_or", "inside_or", "below_or"))


@m5_facet("m5_compression", in_label=False)
def _m5_compression(inputs, ctx, side):
    # The 12 bars before the alert as a box in M5 ATR20: a squeeze, then where the alert bar closed.
    ratio = _num(inputs.get("m5_range12_atr20"))
    broke = (_text(inputs.get("m5_range12_break")) or "").lower()
    if ratio is None or ratio < 0 or broke not in {"up", "down", "inside"}:
        return UNKNOWN
    if ratio > M5_SQUEEZE_RANGE_ATR:
        return "no_squeeze"
    return "squeeze_inside" if broke == "inside" else f"squeeze_break_{broke}"


@m5_facet("m5_side_vs_d1_env", in_label=False)
def _m5_side_vs_d1_env(inputs, ctx, side):
    # SPY's D1 environment for the session before the alert, against the alert's side.
    label = (_text(inputs.get("d1_environment")) or "").lower()
    if side == UNKNOWN or label in {"", UNKNOWN}:
        return UNKNOWN
    if label in {"compressed", "mixed"}:
        return f"d1_env_{label}"
    trend = {"trending_up": "LONG", "trending_down": "SHORT"}.get(label)
    if trend is None:
        return UNKNOWN
    return "side_with_d1_trend" if side == trend else "side_against_d1_trend"
