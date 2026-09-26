"""The swing table (p9): what today proved, shown in the Master AVWAP setups table.

The trader, 2026-09-26: "I want everything from today that will help me find the best
swing trades in the master avwap table. That is basically my 'swing' table."

Pure: `SetupRow`s, the regime-grades payload and ONE swing-context payload in (built off
the Qt thread by `ui.services.swing_table_context`), rows and text out. No file, no clock.
Nothing here changes a score, a bucket key the scan wrote, the live points or an alert;
"Best swing" is a display order only. Unknown is blank, never 0.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from typing import Any, Iterable, Mapping

import regime_grades
import setup_grades
from ui.models.setup import SetupRow

#: The Long leaders bucket key (a row from `long_setups`, or a scan row it was merged onto).
LONG_LEADER_BUCKET = "long_leader"
#: `row.raw` key holding the merged leader facts.
LEADER_KEY = "long_leader"
#: The row source of a leader-only row.
LEADER_SOURCE = "long_setups"
UNTESTED = regime_grades.UNTESTED
#: `master_avwap_lib.legacy.FAVZONE_LONG_RETIRED` (the scan's raw flag; read, never imported).
FAVZONE_LONG_RETIRED = "favzone_long_retired"
RETIRED_TEXT = "fav-zone long retired"
#: Short families proven on 2026-09-26 (favourite zone SHORT, the retest follow-through SHORT).
PROVEN_SHORT_FAMILIES = frozenset({"avwap_retest_followthrough"})
FAVZONE_BUCKETS = frozenset({"favorite_setup", "near_favorite_zone"})
#: The swing columns, in `SetupTableModel.COLUMNS` order.
SWING_COLUMNS = ("regime_grade", "leader", "sp4", "strength", "study", "universe")
SOURCE_LABELS = {"momentum_scanner": "momentum", "journal_traded": "traded"}
SETUP_LABELS = {"leader_pullback": "leader pullback", "post_earnings_drift": "post-earnings drift"}
STUDY_LABELS = {"leader_pullback_long": "leader pullback (study)",
                "band_bounce_leader_long": "band bounce leader (study)"}
#: A long-setups payload this many calendar days older than the table's scan date is stale.
LEADER_MAX_AGE_DAYS = 4


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(_text(value)[:10])
    except ValueError:
        return None


def _family(row: SetupRow) -> str:
    return _text((row.raw or {}).get("setup_family")).lower()


def side_family_key(side: Any, family: Any) -> str:
    """``SIDE|family`` - how the context keys a family (the SP4 evidence's own key)."""
    return f"{_text(side).upper()}|{_text(family).lower() or 'general'}"


# --------------------------------------------------------------------------- Long leaders
def leaders_current(payload: Mapping[str, Any] | None, data_date: Any) -> bool:
    """True when the long-setups payload belongs with a table dated ``data_date``."""
    as_of = _day((payload or {}).get("as_of"))
    if as_of is None:
        return False
    table = _day(data_date)
    return table is None or (table - as_of).days <= LEADER_MAX_AGE_DAYS


def leaders_by_symbol(payload: Mapping[str, Any] | None, data_date: Any = "") -> dict[str, dict[str, Any]]:
    """``{SYMBOL: leader facts}`` in the payload's rank order; a name with two setups is one entry."""
    if not leaders_current(payload, data_date):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in (payload or {}).get("rows") or ():
        if not isinstance(row, Mapping):
            continue
        symbol = _text(row.get("symbol")).upper()
        setup = _text(row.get("setup") or row.get("setup_family"))
        if not symbol or not setup:
            continue
        info = out.get(symbol)
        if info is None:
            out[symbol] = {
                "setups": [setup],
                "rank": len(out),
                "as_of": _text(row.get("as_of") or (payload or {}).get("as_of"))[:10],
                "sector": _text(row.get("sector")),
                "entry_limit": _num(row.get("entry_limit")),
                "stop": _num(row.get("stop")),
                "exit": _text(row.get("exit")),
                "status": _text(row.get("status")),
                "promoted": bool(row.get("promoted")),
                "market_working": _text(row.get("market_working") or (payload or {}).get("market_working")),
                "rs_percentile": _num(row.get("rs_percentile")),
                "reasons": [_text(reason) for reason in row.get("reasons") or () if _text(reason)],
                "strength_filter": _text(row.get("strength_filter")),
                "strength_sma50_atr": _num(row.get("strength_sma50_atr")),
            }
        elif setup not in info["setups"]:
            info["setups"].append(setup)
            info["promoted"] = info["promoted"] or bool(row.get("promoted"))
            info["reasons"] += [f"{SETUP_LABELS.get(setup, setup)}: {_text(reason)}"
                                for reason in row.get("reasons") or () if _text(reason)]
    return out


def leader_info(row: Any) -> dict[str, Any] | None:
    raw = getattr(row, "raw", None)
    info = raw.get(LEADER_KEY) if isinstance(raw, Mapping) else None
    return info if isinstance(info, Mapping) else None


def _leader_raw(info: Mapping[str, Any]) -> dict[str, Any]:
    # Each setup name as a raw flag: `longs_market_gate.row_is_exempt` reads it (a leader waits on its own).
    return {LEADER_KEY: dict(info), **{setup: True for setup in info["setups"]}}


def leader_row(symbol: str, info: Mapping[str, Any]) -> SetupRow:
    """A table row for a leader the scan table does not already hold."""
    setup = info["setups"][0]
    entry, stop = info.get("entry_limit"), info.get("stop")
    level = " · ".join(part for part in (
        f"limit {entry:.2f}" if entry is not None else "",
        f"stop {stop:.2f}" if stop is not None else "",
    ) if part)
    return SetupRow(
        symbol=symbol,
        side="LONG",
        bucket=LONG_LEADER_BUCKET,
        setup_tags=[SETUP_LABELS.get(name, name) for name in info["setups"]],
        key_level=level,
        sector=_text(info.get("sector")),
        last_trade_date=_text(info.get("as_of")),
        source=LEADER_SOURCE,
        raw={"symbol": symbol, "side": "LONG", "setup_family": setup,
             "bucket_keys": [LONG_LEADER_BUCKET], **_leader_raw(info)},
    )


def merge_long_leaders(rows: Iterable[SetupRow], payload: Mapping[str, Any] | None,
                       data_date: Any = "", *, leaders: Mapping[str, Any] | None = None,
                       memo: dict | None = None) -> list[SetupRow]:
    """The table's rows plus the Long leaders. Never in place, never a duplicate.

    A leader whose symbol already has a LONG row is merged onto the first such row (the
    chip and the facts, its own bucket and score untouched); any other leader is a new
    LONG row after the scan's rows, in the leaders' rank order. ``memo`` (one per
    ``leaders``) hands back the rows built for the same input rows last time.
    """
    rows = list(rows)
    if leaders is None:
        leaders = leaders_by_symbol(payload, data_date)
    if not leaders:
        return rows
    merged: set[str] = set()
    out: list[SetupRow] = []
    for row in rows:
        symbol = _text(row.symbol).upper()
        info = leaders.get(symbol)
        if info is None or symbol in merged or _text(row.side).upper() != "LONG":
            out.append(row)
            continue
        merged.add(symbol)
        hit = memo.get(id(row)) if memo is not None else None
        if hit is None or hit[0] is not row:
            raw = dict(row.raw or {})
            raw["bucket_keys"] = sorted(row.bucket_keys | {LONG_LEADER_BUCKET})
            raw.update(_leader_raw(info))
            hit = (row, dataclasses.replace(row, raw=raw))
            if memo is not None:
                memo[id(row)] = hit
        out.append(hit[1])
    for symbol, info in leaders.items():
        if symbol in merged:
            continue
        built = memo.get(symbol) if memo is not None else None
        if built is None:
            built = leader_row(symbol, info)
            if memo is not None:
                memo[symbol] = built
        out.append(built)
    return out


def _pct(value: Any) -> str:
    number = _num(value)
    return "" if number is None else f"{number * 100:.0f}"


def leader_text(row: Any) -> str:
    """``ready · limit 120.10 · stop 115.20 · RS 92``; "" for a row that is not a leader."""
    info = leader_info(row)
    if info is None:
        return ""
    parts = [_text(info.get("status")) or "status unknown"]
    if info.get("entry_limit") is not None:
        parts.append(f"limit {info['entry_limit']:.2f}")
    if info.get("stop") is not None:
        parts.append(f"stop {info['stop']:.2f}")
    rs = _pct(info.get("rs_percentile"))
    if rs:
        parts.append(f"RS {rs}")
    return " · ".join(parts)


def leader_tooltip(row: Any) -> str:
    info = leader_info(row)
    if info is None:
        return ""
    setups = ", ".join(SETUP_LABELS.get(name, name) for name in info.get("setups") or ())
    status = _text(info.get("status")) or "status unknown"
    lines = [f"Long leader ({setups}), scan {info.get('as_of') or '?'}: {status}"
             + (" - promoted" if info.get("promoted") else "")]
    if info.get("entry_limit") is not None:
        lines.append(f"Entry: buy limit {info['entry_limit']:.2f}")
    if info.get("exit"):
        lines.append(f"Exit: {info['exit']}")
    rs = _pct(info.get("rs_percentile"))
    lines.append(f"63-day RS vs SPY: {rs}th percentile of the scan" if rs else "63-day RS vs SPY: unknown")
    reasons = info.get("reasons") or ()
    if reasons:
        lines.append("Why: " + "; ".join(reasons))
    return "\n".join(lines)


# --------------------------------------------------------------------------- regime grade
def current_regime(payload: Mapping[str, Any] | None) -> str:
    return _text(((payload or {}).get("current") or {}).get("regime"))


def regime_entry(row: SetupRow, payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The row's family entry in the grades-by-regime payload (`regime_grades.build_payload`)."""
    key = setup_grades.swing_key(row.side, row.bucket, (row.raw or {}).get("setup_family"))
    entry = ((payload or {}).get("swing") or {}).get(key)
    return entry if isinstance(entry, Mapping) else None


def regime_cell(row: SetupRow, payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """This family's cell in the CURRENT regime; None when untested (or no regime is typed)."""
    regime = current_regime(payload)
    entry = regime_entry(row, payload)
    cell = ((entry or {}).get("by_regime") or {}).get(regime) if regime else None
    return cell if isinstance(cell, Mapping) and int(cell.get("n") or 0) else None


def regime_grade_text(row: SetupRow, payload: Mapping[str, Any] | None) -> str:
    """``B n42`` / ``untested in this regime``; "" when no regime is typed or nothing is loaded."""
    if not current_regime(payload):
        return ""
    cell = regime_cell(row, payload)
    if cell is None:
        return UNTESTED
    return f"{setup_grades.badge(cell.get('grade'))} n{int(cell.get('n') or 0)}"


def regime_sort_rank(row: SetupRow, payload: Mapping[str, Any] | None) -> int:
    """Best grade first; untested after every graded cell (unknown is not measured-bad)."""
    cell = regime_cell(row, payload)
    if cell is None:
        return setup_grades.UNGRADED_RANK + 10
    return setup_grades.sort_rank(cell.get("grade"))


def path_line(row: SetupRow, context: Mapping[str, Any] | None) -> str:
    cell = ((context or {}).get("path") or {}).get(side_family_key(row.side, _family(row)))
    if not isinstance(cell, Mapping) or _num(cell.get("mfe_atr")) is None:
        return ""
    return (f"Typical path (10 sessions, median of {int(cell.get('n') or 0)}): "
            f"best +{_num(cell['mfe_atr']):.2f} ATR, worst {_num(cell.get('mae_atr')) or 0:+.2f} ATR")


def exit_line(row: SetupRow, context: Mapping[str, Any] | None) -> str:
    cell = ((context or {}).get("exits") or {}).get(side_family_key(row.side, _family(row)))
    if not isinstance(cell, Mapping):
        return ""
    models = [(name, _num(cell.get(f"{key}_r"))) for name, key in
              (("the tracker's target/stop", "current"), ("stop 1 ATR, hold 10", "stop_only"),
               ("1 ATR trail, hold 10", "trail"))]
    models = [(name, value) for name, value in models if value is not None]
    if not models:
        return ""
    name, value = max(models, key=lambda item: item[1])
    return f"Best exit model (S13, n {int(cell.get('n') or 0)}): {name}, {value:+.2f}R"


def regime_grade_tooltip(row: SetupRow, payload: Mapping[str, Any] | None,
                         context: Mapping[str, Any] | None) -> str:
    lines = []
    current = (payload or {}).get("current")
    if not current:
        lines.append("No regime typed yet - answer the Mentor's regime question.")
    else:
        entry = regime_entry(row, payload) or {}
        lines.append(f"Regime now: {current.get('label')} (day {current.get('day_count')}).")
        lines.append(regime_grades.by_regime_text(entry.get("by_regime") or {}, payload,
                                                  pooled=entry.get("all") or {}))
        lines.append("Longs judged raw; shorts vs SPY once 30+ have it, raw beside it.")
    lines += [text for text in (path_line(row, context), exit_line(row, context)) if text]
    return "\n".join(lines)


# --------------------------------------------------------------------------- shadows and tags
def sp4_read(row: SetupRow, evidence: Mapping[str, Any] | None) -> tuple[float, float, str] | None:
    """``(sp4 points, family adjust, basis)``; None when the score or the evidence is unknown."""
    if not (evidence or {}).get("families") or row.score is None:
        return None
    import points_challenger

    cell, basis = points_challenger.effective_cell(row.side, _family(row), evidence)
    adjust = points_challenger.family_adjust(cell) if cell is not None else 0.0
    return round(float(row.score) + adjust, 1), adjust, basis


def sp4_text(row: SetupRow, evidence: Mapping[str, Any] | None) -> str:
    read = sp4_read(row, evidence)
    return "" if read is None else f"{read[0]:.1f} ({read[1]:+.0f})"


def sp4_tooltip(row: SetupRow, evidence: Mapping[str, Any] | None) -> str:
    read = sp4_read(row, evidence)
    if read is None:
        return "SP4 (shadow): unknown - no family evidence or no live score for this row."
    return (f"SP4 SHADOW points {read[0]:.1f} = live score {row.score:.1f} + family adjust {read[1]:+.1f} "
            f"(read from {read[2]}). Shadow only: the live points, buckets, sort and alerts are unchanged.")


def _study_row(row: SetupRow, context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    found = ((context or {}).get("study") or {}).get(f"{_text(row.symbol).upper()}|{_text(row.side).upper()}")
    return found if isinstance(found, Mapping) else {}


def strength_read(row: SetupRow, context: Mapping[str, Any] | None) -> tuple[str, float | None] | None:
    """``(yes|no, distance above the 50-day in ATR)`` for a LONG; None when unknown or a short."""
    if _text(row.side).upper() != "LONG":
        return None
    raw = row.raw or {}
    info = leader_info(row) or {}
    verdict = (_text(info.get("strength_filter")) or _text(raw.get("perm_strength_filter"))
               or _text(_study_row(row, context).get("strength_filter"))).lower()
    if verdict not in {"yes", "no"}:
        return None
    distance = _num(info.get("strength_sma50_atr"))
    if distance is None:
        distance = _num(raw.get("perm_dist_sma50_atr"))
    return verdict, distance


def strength_text(row: SetupRow, context: Mapping[str, Any] | None) -> str:
    read = strength_read(row, context)
    if read is None:
        return ""
    verdict, distance = read
    return verdict if distance is None else f"{verdict} {distance:+.1f} ATR"


def study_tags(row: SetupRow, context: Mapping[str, Any] | None) -> list[str]:
    raw = row.raw or {}
    text = _text(raw.get("study_families")) or _text(_study_row(row, context).get("study_families"))
    tags = [STUDY_LABELS.get(name, name.replace("_", " ")) for name in text.split(";") if name.strip()]
    if _text(raw.get(FAVZONE_LONG_RETIRED)):
        tags.append(RETIRED_TEXT)
    return tags


def study_tooltip(row: SetupRow, context: Mapping[str, Any] | None) -> str:
    lines = []
    tags = [tag for tag in study_tags(row, context) if tag != RETIRED_TEXT]
    if tags:
        lines.append("S14 study families (research only, never scored): " + ", ".join(tags))
    retired = _text((row.raw or {}).get(FAVZONE_LONG_RETIRED))
    if retired:
        lines.append(f"This LONG would have been {retired.replace('_', ' ')}. The favourite zone is SHORT-only "
                     "since 2026-09-26, so it lost that bucket; longs come from Long leaders now.")
    return "\n".join(lines)


def source_badges(row: SetupRow, context: Mapping[str, Any] | None) -> list[str]:
    sources = ((context or {}).get("sources") or {}).get(_text(row.symbol).upper()) or ()
    return [SOURCE_LABELS[name] for name in SOURCE_LABELS if name in sources]


def source_tooltip(row: SetupRow, context: Mapping[str, Any] | None) -> str:
    badges = source_badges(row, context)
    words = {"momentum": "a momentum-scanner name (added to the long universe)",
             "traded": "a name you traded in the last year (journal)"}
    return "; ".join(words[badge] for badge in badges)


def columns_with_values(rows: Iterable[SetupRow], regime_payload: Mapping[str, Any] | None,
                        context: Mapping[str, Any] | None) -> set[str]:
    """The swing columns at least one row has a value in, without formatting any cell."""
    rows = list(rows)
    context = context or {}
    found: set[str] = set()
    if rows and current_regime(regime_payload):
        found.add("regime_grade")  # every row reads a grade or "untested in this regime"
    if any(leader_info(row) is not None for row in rows):
        found.add("leader")
    if (context.get("sp4") or {}).get("families") and any(row.score is not None for row in rows):
        found.add("sp4")
    if any(strength_read(row, context) is not None for row in rows):
        found.add("strength")
    study = context.get("study") or {}
    if any((row.raw or {}).get("study_families") or (row.raw or {}).get(FAVZONE_LONG_RETIRED)
           or (study and f"{_text(row.symbol).upper()}|{_text(row.side).upper()}" in study)
           for row in rows):
        found.add("study")
    sources = context.get("sources") or {}
    if sources and any(source_badges(row, context) for row in rows):
        found.add("universe")
    return found


# --------------------------------------------------------------------------- Best swing
def is_proven_short(row: SetupRow) -> bool:
    """Favourite-zone SHORT, or a SHORT of a proven family (`PROVEN_SHORT_FAMILIES`)."""
    if _text(row.side).upper() != "SHORT":
        return False
    raw = row.raw or {}
    return bool(row.bucket_keys & FAVZONE_BUCKETS or raw.get("favorite_zone")
                or _family(row) in PROVEN_SHORT_FAMILIES)


def best_swing_tier(row: SetupRow, gate_verdict: str) -> int:
    """0 promoted Long leaders and proven shorts, 1 the rest, 2 longs in a market not working."""
    long = _text(row.side).upper() == "LONG"
    if long and gate_verdict == "no":
        return 2
    info = leader_info(row)
    if (long and info is not None and info.get("promoted")) or is_proven_short(row):
        return 0
    return 1


def best_swing_order(rows: Iterable[SetupRow], *, gate_verdict: str = "",
                     regime_payload: Mapping[str, Any] | None = None) -> list[SetupRow]:
    """The "Best swing" display order. Stable: the incoming order breaks every tie.

    The market gate first (longs last while it says "no"), then promoted Long leaders and
    the proven short families by their current-regime grade, then everything else.
    """
    gate = _text(gate_verdict).lower()
    keyed = []
    for index, row in enumerate(rows):
        tier = best_swing_tier(row, gate)
        grade = regime_sort_rank(row, regime_payload) if tier == 0 else 0
        keyed.append(((tier, grade, index), row))
    keyed.sort(key=lambda item: item[0])
    return [row for _key, row in keyed]
