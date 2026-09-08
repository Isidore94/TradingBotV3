"""Rank the Master AVWAP setups by a POINT system (trader, 2026-09-08).

Presentation only - the ST6 pattern. The switch `rank_setups_by_points`
REORDERS the setups table and never hides a row; it is read AT SORT TIME
(`rank_enabled`) and nothing here reaches a detector, a score, an alert or a
file. `legacy.py` is untouched: every input is a field the scan already
writes on a focus row, or the family record the panel already injects.

Four graded inputs, each a named part with its own range so the tooltip can
say WHY a row sits where it sits:

1. ``setup`` (0 to 50) - the FAMILY's record. The Wilson lower bound on the
   family's recent win rate x 40 (the bound, never the raw rate: a 100%-on-
   three must not outrank a 62%-on-ninety), plus the row's expected R clamped
   to +-1 x 10. A family nobody has graded scores 0 here and says so.
2. ``sr`` (-20 to +10) - nearby support / resistance knocks it down. Starts
   at +10 for a clean path and loses points per HV level blocking (4), per HV
   level nearby (2), per cloud level nearby (2), for a trendline in play (4),
   for a moving average inside 1 ATR AHEAD of price (3 each: EMA21 and the
   SMA the breakout tracker names) and for the nearest level inside 0.5 ATR
   (2). Floors at -20.
3. ``rs`` (-15 to +15) - RS/RW in the trade's DIRECTION: vs SPY, vs the
   sector, vs the industry, each clamped to +-5, sign-flipped for a SHORT so
   weakness is the good reading. Unmeasured legs score 0 and are named.
4. ``bounce`` (0 to 15) - a recent bounce: +15 for a bounce event TODAY, +8
   when the setup is a bounce by name (a BOUNCE signal, a bounce family or
   the SMA50 bounce pattern) without today's event.

The total is the plain sum. Ranking is `rank_order`: the rows in
`RANKED_BUCKETS` (favourite, near favourite, high conviction) by total
descending with ties keeping ARRIVAL order, then every other row in arrival
order - so the table shows exactly the same rows either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: The buckets the point ranking reorders; everything else keeps its place after them.
RANKED_BUCKETS = frozenset({"favorite_setup", "near_favorite_zone", "high_conviction"})

SETTING_KEY = "rank_setups_by_points"

SETUP_BOUND_WEIGHT = 40.0
SETUP_EXPECTED_R_WEIGHT = 10.0
SR_CLEAN_PATH = 10.0
SR_FLOOR = -20.0
SR_BLOCKING_LEVEL = 4.0
SR_NEARBY_LEVEL = 2.0
SR_CLOUD_LEVEL = 2.0
SR_TRENDLINE = 4.0
SR_MOVING_AVERAGE = 3.0
SR_TIGHT_LEVEL = 2.0
SR_MA_ATR_REACH = 1.0
SR_TIGHT_ATR = 0.5
RS_LEG_CAP = 5.0
BOUNCE_TODAY = 15.0
BOUNCE_NAMED = 8.0


@dataclass(frozen=True)
class SetupPoints:
    total: float
    setup: float
    sr: float
    rs: float
    bounce: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def text(self) -> str:
        return f"{self.total:+.0f}"

    def tooltip(self) -> str:
        lines = [
            f"Points {self.total:+.1f}",
            f"  setup {self.setup:+.1f}  (family bound x {SETUP_BOUND_WEIGHT:.0f}, expected R x {SETUP_EXPECTED_R_WEIGHT:.0f})",
            f"  S/R {self.sr:+.1f}  (clean path {SR_CLEAN_PATH:+.0f}, levels ahead knock it down)",
            f"  RS/RW {self.rs:+.1f}  (vs SPY / sector / industry, in the trade's direction, +-{RS_LEG_CAP:.0f} each)",
            f"  bounce {self.bounce:+.1f}  (today {BOUNCE_TODAY:+.0f}, named {BOUNCE_NAMED:+.0f})",
        ]
        lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "" or isinstance(value, bool):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def setup_part(record: Mapping[str, Any] | None, expected_r: Any) -> tuple[float, list[str]]:
    """Input 1: the family's Wilson lower bound and the row's expected R."""
    notes: list[str] = []
    bound = _float((record or {}).get("win_rate_lb"))
    if bound is None:
        points = 0.0
        notes.append("family ungraded: no win-rate bound yet, setup part is 0")
    else:
        points = _clamp(bound, 0.0, 1.0) * SETUP_BOUND_WEIGHT
    expected = _float(expected_r)
    if expected is None:
        notes.append("no expected R on the row")
    else:
        points += _clamp(expected, -1.0, 1.0) * SETUP_EXPECTED_R_WEIGHT
    return round(points, 2), notes


def _ahead(level: float | None, close: float | None, atr: float | None, side: str) -> bool:
    """A level inside `SR_MA_ATR_REACH` ATR AHEAD of price in the trade's direction."""
    if level is None or close is None or not atr or atr <= 0:
        return False
    distance = (level - close) if side == "LONG" else (close - level)
    return 0.0 < distance <= SR_MA_ATR_REACH * atr


def sr_part(raw: Mapping[str, Any], side: str) -> tuple[float, list[str]]:
    """Input 2: nearby S/R knocks it down, from the scan's own level counts."""
    notes: list[str] = []
    points = SR_CLEAN_PATH
    blocking = _int(raw.get("hv_level_blocking_count"))
    nearby = _int(raw.get("hv_level_nearby_count"))
    cloud = _int(raw.get("cloud_level_nearby_count"))
    if blocking:
        points -= SR_BLOCKING_LEVEL * blocking
        notes.append(f"{blocking} HV level(s) blocking")
    if nearby:
        points -= SR_NEARBY_LEVEL * nearby
        notes.append(f"{nearby} HV level(s) nearby")
    if cloud:
        points -= SR_CLOUD_LEVEL * cloud
        notes.append(f"{cloud} cloud level(s) nearby")
    if str(raw.get("trendline_note") or "").strip():
        points -= SR_TRENDLINE
        notes.append("a trendline is in play")
    close = _float(raw.get("previous_close"))
    atr = _float(raw.get("atr20"))
    for key, label in (("ema21", "EMA21"), ("sma_breakout_sma_level", str(raw.get("sma_breakout_sma_label") or "SMA"))):
        if _ahead(_float(raw.get(key)), close, atr, side):
            points -= SR_MOVING_AVERAGE
            notes.append(f"{label} inside {SR_MA_ATR_REACH:.0f} ATR ahead")
    nearest = _float(raw.get("hv_level_nearest_distance_atr"))
    if nearest is not None and abs(nearest) < SR_TIGHT_ATR:
        points -= SR_TIGHT_LEVEL
        notes.append(f"nearest level {abs(nearest):.2f} ATR away")
    return round(max(SR_FLOOR, points), 2), notes


def rs_part(raw: Mapping[str, Any], side: str, *, d1_vs_sector: Any = None, d1_vs_industry: Any = None) -> tuple[float, list[str]]:
    """Input 3: RS/RW vs SPY, sector and industry in the trade's direction."""
    notes: list[str] = []
    sign = -1.0 if side == "SHORT" else 1.0
    legs = (
        ("SPY", _float(raw.get("daily_relative_strength_score"))),
        ("sector", _float(d1_vs_sector)),
        ("industry", _float(d1_vs_industry) if _float(d1_vs_industry) is not None else _float(raw.get("rs_vs_industry"))),
    )
    points = 0.0
    for label, value in legs:
        if value is None:
            notes.append(f"RS vs {label} unmeasured")
            continue
        points += _clamp(sign * value, -RS_LEG_CAP, RS_LEG_CAP)
    return round(points, 2), notes


def _names_a_bounce(raw: Mapping[str, Any]) -> bool:
    tags = list(raw.get("favorite_signals") or []) + list(raw.get("setup_tags") or [])
    if any("BOUNCE" in str(tag).upper() for tag in tags):
        return True
    if "bounce" in str(raw.get("setup_family") or "").lower():
        return True
    return bool(raw.get("top_pattern_daily_sma50_bounce"))


def bounce_part(raw: Mapping[str, Any]) -> tuple[float, list[str]]:
    """Input 4: a recent bounce."""
    if raw.get("has_bounce_event_today"):
        return BOUNCE_TODAY, ["bounce event today"]
    if _names_a_bounce(raw):
        return BOUNCE_NAMED, ["a bounce by name, no event today"]
    return 0.0, ["no recent bounce"]


def score_row(
    raw: Mapping[str, Any] | None,
    *,
    side: str = "",
    family_record: Mapping[str, Any] | None = None,
    d1_vs_sector: Any = None,
    d1_vs_industry: Any = None,
) -> SetupPoints:
    """The four parts and their sum for one setup row. Pure."""
    raw = raw or {}
    side = str(side or raw.get("side") or "").strip().upper()
    setup, notes = setup_part(family_record, raw.get("expected_r"))
    sr, sr_notes = sr_part(raw, side)
    rs, rs_notes = rs_part(raw, side, d1_vs_sector=d1_vs_sector, d1_vs_industry=d1_vs_industry)
    bounce, bounce_notes = bounce_part(raw)
    total = round(setup + sr + rs + bounce, 2)
    return SetupPoints(
        total=total,
        setup=setup,
        sr=sr,
        rs=rs,
        bounce=bounce,
        notes=tuple(notes + sr_notes + rs_notes + bounce_notes),
    )


def rank_order(items: Sequence[tuple[str, float | None]]) -> list[int]:
    """Indices in display order: ranked buckets by points, then the rest.

    `items` is `[(bucket, total)]` in ARRIVAL order. A ranked-bucket row with
    no total sorts after every ranked row that has one; ties keep arrival
    order; every index appears exactly once.
    """
    ranked: list[tuple[float, float, int]] = []
    rest: list[int] = []
    for index, (bucket, total) in enumerate(items):
        if str(bucket or "").strip().lower() in RANKED_BUCKETS:
            missing = 1.0 if total is None else 0.0
            ranked.append((missing, -(float(total) if total is not None else 0.0), index))
        else:
            rest.append(index)
    ranked.sort()
    return [index for _missing, _neg, index in ranked] + rest


def rank_enabled() -> bool:
    """The persisted switch, read AT SORT TIME and never at write time. Default OFF."""
    try:
        import project_paths

        return bool(project_paths.get_local_setting(SETTING_KEY, False))
    except Exception:  # noqa: BLE001 - a display preference never costs a list
        return False
