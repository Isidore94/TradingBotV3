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
2. ``sr`` (-20 to +10) - measured nearby support / resistance. V2 starts
   unknown at zero; a proven clear path earns +10. A known HV obstacle uses
   its strongest blocking/nearby/nearest deduction once. Cloud, trendline
   and moving-average obstacles also deduct, even from partial data. Floors
   at -20. V1 retains the original missing-as-clear behavior for replay.
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

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: The buckets the point ranking reorders; everything else keeps its place after them.
#: `claimed_like` joined them in packet D1C-A (trader, 2026-09-14: *"Rank
#: manually claimed picks using the existing setup-ranking system"*) - the SAME
#: four inputs on the same scale, from whatever the claim carried, with a
#: missing input scoring 0 and saying so. A ranked row with no total still
#: sorts after every ranked row that has one, so a claim never jumps a measured
#: pick by being unmeasured.
RANKED_BUCKETS = frozenset(
    {"favorite_setup", "near_favorite_zone", "high_conviction", "claimed_like"}
)

SETTING_KEY = "rank_setups_by_points"
#: The trader's switch that lets the PROPOSED multipliers apply (default OFF).
LEARNED_SETTING_KEY = "setup_points_learned_weights"

# ``points_v1`` was the original presentation calculation.  It remains an
# explicit replay choice so an append-only old log keeps its meaning.  New desk
# observations use v2: an absent S/R reading is uncertainty, never a clean
# path.
POINTS_V1 = "points_v1"
POINTS_V2 = "points_v2"
DEFAULT_POINTS_VERSION = POINTS_V2

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
    #: The parts BEFORE any learned multiplier - what the evidence log records.
    raw_parts: dict[str, float] = field(default_factory=dict)
    #: The multipliers in force when this was computed (all 1.0 unless learned).
    weights: dict[str, float] = field(default_factory=dict)
    points_version: str = DEFAULT_POINTS_VERSION

    def text(self) -> str:
        return f"{self.total:+.0f}"

    def log_row(self, *, scan_date: str, symbol: str, side: str, family: str, bucket: str) -> dict[str, Any]:
        """One evidence-log row: raw parts, the shown total, the weights used."""
        return {
            "scan_date": str(scan_date or ""),
            "symbol": str(symbol or "").upper(),
            "side": str(side or "").upper(),
            "family": str(family or ""),
            "bucket": str(bucket or ""),
            "total": float(self.total),
            **{part: float(self.raw_parts.get(part, 0.0)) for part in ("setup", "sr", "rs", "bounce")},
            "multipliers": dict(self.weights),
            "points_version": self.points_version,
        }

    def tooltip(self) -> str:
        learned = {part: value for part, value in self.weights.items() if value != 1.0}
        lines = [
            f"Points {self.total:+.1f}"
            + (
                "  (learned weights: " + ", ".join(f"{p} x{v:.2f}" for p, v in sorted(learned.items())) + ")"
                if learned
                else ""
            ),
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
    return out if math.isfinite(out) else None


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _count(value: Any) -> int | None:
    """A count is measured only when it is a finite, non-negative integer."""
    number = _float(value)
    if number is None or number < 0 or int(number) != number:
        return None
    return int(number)


def _truth(value: Any) -> bool | None:
    """Read scanner booleans semantically (``bool('False')`` is a lie)."""
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "on"}:
        return True
    if text in {"false", "no", "0", "off", "none", "null", ""}:
        return False
    return None


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


def _trendline_in_play(raw: Mapping[str, Any]) -> bool | None:
    for key in ("trendline_in_play", "trendline_note"):
        value = _truth(raw.get(key))
        if value is not None:
            return value
    text = str(raw.get("trendline_note") or "").strip()
    return True if text else None


def _sr_part_v1(raw: Mapping[str, Any], side: str) -> tuple[float, list[str]]:
    """The original clean-path calculation, retained only for log replay."""
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


def sr_part(raw: Mapping[str, Any], side: str, *, version: str = DEFAULT_POINTS_VERSION) -> tuple[float, list[str]]:
    """Input 2: measured S/R earns the clear-path bonus; unknown starts at zero.

    A partial row may still name a real obstacle.  It starts at zero and keeps
    that deduction.  HV count / nearest-distance fields often describe the
    same level, so v2 takes their strongest one deduction rather than adding a
    level twice.
    """
    if version == POINTS_V1:
        return _sr_part_v1(raw, side)

    notes: list[str] = []
    counts = {key: _count(raw.get(key)) for key in (
        "hv_level_blocking_count", "hv_level_nearby_count", "cloud_level_nearby_count"
    )}
    close = _float(raw.get("previous_close"))
    atr = _float(raw.get("atr20"))
    nearest = _float(raw.get("hv_level_nearest_distance_atr"))
    measured = _truth(raw.get("sr_inputs_measured"))
    malformed = any(raw.get(key) not in (None, "") and value is None for key, value in counts.items())
    malformed = malformed or (raw.get("previous_close") not in (None, "") and close is None)
    malformed = malformed or (raw.get("atr20") not in (None, "") and atr is None)
    complete = (
        bool(measured) and not malformed
        and all(value is not None for value in counts.values())
        and close is not None and atr is not None and atr > 0
        and (nearest is not None or (counts["hv_level_blocking_count"] == 0 and counts["hv_level_nearby_count"] == 0))
    )
    partial = not complete
    points = SR_CLEAN_PATH if complete else 0.0
    if partial:
        notes.append("S/R partial: clear path not measured")
    if malformed:
        notes.append("S/R unmeasured: unreadable required input")

    blocking = counts["hv_level_blocking_count"]
    nearby = counts["hv_level_nearby_count"]
    hv_deductions: list[tuple[float, str]] = []
    if blocking:
        hv_deductions.append((SR_BLOCKING_LEVEL * blocking, f"{blocking} HV level(s) blocking"))
    if nearby:
        hv_deductions.append((SR_NEARBY_LEVEL * nearby, f"{nearby} HV level(s) nearby"))
    if nearest is not None and abs(nearest) < SR_TIGHT_ATR:
        hv_deductions.append((SR_TIGHT_LEVEL, f"nearest level {abs(nearest):.2f} ATR away"))
    if hv_deductions:
        deduction, note = max(hv_deductions, key=lambda item: item[0])
        points -= deduction
        notes.append(note)
    cloud = counts["cloud_level_nearby_count"]
    if cloud:
        points -= SR_CLOUD_LEVEL * cloud
        notes.append(f"{cloud} cloud level(s) nearby")
    if _trendline_in_play(raw) is True:
        points -= SR_TRENDLINE
        notes.append("a trendline is in play")
    for key, label in (("ema21", "EMA21"), ("sma_breakout_sma_level", str(raw.get("sma_breakout_sma_label") or "SMA"))):
        if _ahead(_float(raw.get(key)), close, atr, side):
            points -= SR_MOVING_AVERAGE
            notes.append(f"{label} inside {SR_MA_ATR_REACH:.0f} ATR ahead")
    if partial and not notes[:-1] and not hv_deductions and not cloud and _trendline_in_play(raw) is None and close is None:
        return 0.0, ["S/R unmeasured: required scan facts absent"]
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
    return _truth(raw.get("top_pattern_daily_sma50_bounce")) is True


def bounce_part(raw: Mapping[str, Any], *, version: str = DEFAULT_POINTS_VERSION) -> tuple[float, list[str]]:
    """Input 4: a recent bounce."""
    if (bool(raw.get("has_bounce_event_today")) if version == POINTS_V1 else _truth(raw.get("has_bounce_event_today")) is True):
        return BOUNCE_TODAY, ["bounce event today"]
    if version == POINTS_V1 and bool(raw.get("top_pattern_daily_sma50_bounce")):
        return BOUNCE_NAMED, ["a bounce by name, no event today"]
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
    weights: Mapping[str, float] | None = None,
    version: str = DEFAULT_POINTS_VERSION,
) -> SetupPoints:
    """The four parts and their sum for one setup row. Pure.

    `weights` are the learned multipliers (`setup_points_evidence`), applied
    to each part AFTER it is measured; the raw parts are kept on the result so
    the evidence log records the measurement, never the weighting.
    """
    raw = raw or {}
    side = str(side or raw.get("side") or "").strip().upper()
    setup, notes = setup_part(family_record, raw.get("expected_r"))
    version = POINTS_V1 if version == POINTS_V1 else POINTS_V2
    sr, sr_notes = sr_part(raw, side, version=version)
    rs, rs_notes = rs_part(raw, side, d1_vs_sector=d1_vs_sector, d1_vs_industry=d1_vs_industry)
    bounce, bounce_notes = bounce_part(raw, version=version)
    raw_parts = {"setup": setup, "sr": sr, "rs": rs, "bounce": bounce}
    used = {part: 1.0 for part in raw_parts}
    for part, value in dict(weights or {}).items():
        if part in used:
            try:
                used[part] = float(value)
            except (TypeError, ValueError):
                continue
    weighted = {part: round(value * used[part], 2) for part, value in raw_parts.items()}
    total = round(sum(weighted.values()), 2)
    return SetupPoints(
        total=total,
        setup=weighted["setup"],
        sr=weighted["sr"],
        rs=weighted["rs"],
        bounce=weighted["bounce"],
        notes=tuple(notes + sr_notes + rs_notes + bounce_notes),
        raw_parts=raw_parts,
        weights=used,
        points_version=version,
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


def learned_weights_enabled() -> bool:
    """The trader's switch for the proposed multipliers. Default OFF."""
    try:
        import project_paths

        return bool(project_paths.get_local_setting(LEARNED_SETTING_KEY, False))
    except Exception:  # noqa: BLE001 - a display preference never costs a list
        return False


def active_weights() -> dict[str, float]:
    """The multipliers in force: the proposal's when the switch is ON, else all 1.0.

    Read from the proposal FILE, never recomputed here; the proposal is written
    by the panel's worker (`setup_points_evidence.log_and_grade`) and every
    multiplier in it is floored on n before it is proposed.
    """
    if not learned_weights_enabled():
        return {}
    try:
        import project_paths
        import setup_points_evidence

        proposal = setup_points_evidence.read_proposal(project_paths.SETUP_POINTS_WEIGHTS_FILE)
        return setup_points_evidence.proposal_multipliers(proposal)
    except Exception:  # noqa: BLE001 - no proposal is "defaults"
        return {}


def rank_enabled() -> bool:
    """The persisted switch, read AT SORT TIME and never at write time. Default OFF."""
    try:
        import project_paths

        return bool(project_paths.get_local_setting(SETTING_KEY, False))
    except Exception:  # noqa: BLE001 - a display preference never costs a list
        return False
