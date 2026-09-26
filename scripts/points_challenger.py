"""Points challenger SP4 (S12 / P13): a SHADOW score beside the live points. Pure.

SP4 = the champion priority score + one per-(setup family, side) adjust read from
the nightly `family_side_evidence.json` (`ai_jobs.family_side_evidence`):

    adjust = 60 x (beat_low_h5 - 0.50) + 20 x mean_move_atr_h10, clamped to [-40, +40]

``beat_low_h5`` is the 95% Wilson low bound of the share of 5-session outcomes
that beat SPY's same-side return; ``mean_move_atr_h10`` is the mean 10-session
side move in ATR20 units. The adjust is 0 unless the cell has n >= 80 tape-decided
5-session outcomes over >= 15 sessions. These numbers were FROZEN on 2026-09-26
before any trial: do not tune them after looking.

Shadow only. Nothing here feeds the live score, the tracker sort, a bucket, an
alert, Focus or `review_policy.json`; the Setup Tracker shows an "SP4" column and
chip and the Saturday report shows one line per side.

SP4 rules, fixed 2026-09-26 (GATES #276):

* Trial: 20 new entry sessions from the first night the evidence is written, then
  5 sessions for the last entries to mature. Each entry session is ranked with the
  adjusts written THAT night (point in time), per side: champion top quartile by
  live points vs SP4 top quartile by SP4 points.
* Success: SP4's top quartile beats the champion's by >= 0.5% mean excess vs SPY at
  5 sessions AND by >= 0.10R mean tracker R, over all 20 matured entry sessions.
* Downside stop: SP4 trails the champion by more than 0.5% excess once 10 entry
  sessions have matured.
* Rollback: delete the SP4 column (and chip, and Saturday line). Nothing else reads it.
* S16 (2026-09-26): the evidence also carries each structural regime's own cells
  (every scan date of that regime up to the night). SP4 reads the current regime's
  cell when it meets the same n / sessions gates, else the pooled cell, and the
  chip says how many fell back. Each night's adjust history records what was read.
* Promotion to the live points is ask-first (`master_avwap_lib/legacy.py`), needs
  golden fixtures and the trader's quoted yes.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable, Mapping

SCHEMA = "family_side_evidence_v1"

# ---------------------------------------------------------------- frozen rule
BEAT_WEIGHT = 60.0
BEAT_CENTRE = 0.50
MOVE_WEIGHT = 20.0
ADJUST_CLAMP = 40.0
MIN_N = 80
MIN_SESSIONS = 15
#: Trailing completed scan sessions the evidence reads.
WINDOW_SESSIONS = 40
#: The evidence file is not written with fewer sessions than this in the window.
MIN_WINDOW_SESSIONS = 15
TAPE_HORIZON = 5
MOVE_HORIZONS = (5, 10)
OUTCOME_KIND = "favorable_direction_session_v2"

# ---------------------------------------------------------------- trial rules
TRIAL_ENTRY_SESSIONS = 20
TRIAL_MATURE_SESSIONS = 5
SUCCESS_EXCESS_PCT = 0.5
SUCCESS_R = 0.10
STOP_TRAIL_PCT = 0.5
STOP_AFTER_SESSIONS = 10
#: Nights of adjusts kept in the file (the trial needs 20; the rest is history).
ADJUST_HISTORY_KEEP = 60

COLLECTING, SUCCESS, NO_EDGE, STOPPED = "collecting", "success", "no_edge", "stopped"


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _r4(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _mean(values: list[float]) -> float | None:
    return _r4(sum(values) / len(values)) if values else None


def _day(value: Any) -> str:
    return str(value or "").strip()[:10]


def family_key(side: Any, family: Any) -> str:
    """``SIDE|family`` - the evidence cell a row reads."""
    fam = str(family or "").strip().lower() or "general"
    return f"{str(side or '').strip().upper()}|{fam}"


# ------------------------------------------------------------------ the adjust
def family_adjust(cell: Mapping[str, Any] | None) -> float:
    """The frozen adjust for one evidence cell; 0 below the n / session floor."""
    cell = cell or {}
    n = int(_num(cell.get("n")) or 0)
    sessions = int(_num(cell.get("sessions")) or 0)
    low = _num(cell.get("beat_low_h5"))
    move = _num(cell.get("mean_move_atr_h10"))
    if n < MIN_N or sessions < MIN_SESSIONS or low is None or move is None:
        return 0.0
    raw = BEAT_WEIGHT * (low - BEAT_CENTRE) + MOVE_WEIGHT * move
    return round(max(-ADJUST_CLAMP, min(ADJUST_CLAMP, raw)), 1)


#: What an SP4 adjust read from the pooled cell says (S16).
ALL_REGIMES = "all regimes"


def _meets_gates(cell: Mapping[str, Any] | None) -> bool:
    cell = cell or {}
    return int(_num(cell.get("n")) or 0) >= MIN_N and int(_num(cell.get("sessions")) or 0) >= MIN_SESSIONS


def effective_cell(
    side: Any, family: Any, evidence: Mapping[str, Any] | None
) -> tuple[Mapping[str, Any] | None, str]:
    """``(cell, basis)``: the current regime's cell when it meets the n / sessions
    gates (S16), else the pooled cell with basis ``all regimes``.

    ``evidence["current_regime"]`` is the trader's regime on the night's as-of
    date; ``families_by_regime`` holds each regime's own cells. Evidence without
    them (before S16) reads the pooled cell exactly as before.
    """
    evidence = evidence or {}
    key = family_key(side, family)
    cells = evidence.get("families") or {}
    pooled = cells.get(key) if isinstance(cells, Mapping) else None
    pooled = pooled if isinstance(pooled, Mapping) else None
    regime = str(evidence.get("current_regime") or "")
    by_regime = evidence.get("families_by_regime") or {}
    if regime and isinstance(by_regime, Mapping):
        regime_cells = by_regime.get(regime) or {}
        cell = regime_cells.get(key) if isinstance(regime_cells, Mapping) else None
        if isinstance(cell, Mapping) and _meets_gates(cell):
            return cell, regime
    return pooled, ALL_REGIMES


def adjust_for(side: Any, family: Any, evidence: Mapping[str, Any] | None) -> float:
    cell, _basis = effective_cell(side, family, evidence)
    return family_adjust(cell) if cell is not None else 0.0


def regime_basis(evidence: Mapping[str, Any] | None) -> dict[str, str]:
    """``{SIDE|family: basis}`` for every family the evidence has a cell for."""
    evidence = evidence or {}
    keys = set((evidence.get("families") or {}).keys())
    regime = str(evidence.get("current_regime") or "")
    keys |= set(((evidence.get("families_by_regime") or {}).get(regime) or {}).keys()) if regime else set()
    out = {}
    for key in sorted(keys):
        side, _, family = key.partition("|")
        out[key] = effective_cell(side, family, evidence)[1]
    return out


def sp4_points(row: Mapping[str, Any], evidence: Mapping[str, Any] | None) -> float | None:
    """Champion priority score + the family/side adjust; None when the score is unknown."""
    score = _num((row or {}).get("priority_score"))
    if score is None:
        return None
    return round(score + adjust_for(row.get("side"), row.get("setup_family"), evidence), 1)


# ---------------------------------------------------------------- the evidence
def _spy_side_return(entry_day: str, target_day: str, side: str,
                     spy_closes: Mapping[str, float]) -> float | None:
    entry, target = _num(spy_closes.get(entry_day)), _num(spy_closes.get(target_day))
    if entry is None or entry <= 0 or target is None:
        return None
    move = (target / entry - 1.0) * 100.0
    return move if side == "LONG" else -move


def tape_excess(row: Mapping[str, Any], spy_closes: Mapping[str, float], *, as_of: str) -> float | None:
    """Side return minus SPY's same-side return over a mature, measured row; else None."""
    if str(row.get("measured") or "").strip().lower() != "true":
        return None
    if str(row.get("maturity") or "").strip().lower() != "mature":
        return None
    side = str(row.get("side") or "").strip().upper()
    target = _day(row.get("target_session"))
    side_return = _num(row.get("side_return_pct"))
    if side not in {"LONG", "SHORT"} or side_return is None or not target:
        return None
    if as_of and target > as_of:
        return None
    spy = _spy_side_return(_day(row.get("scan_date")), target, side, spy_closes)
    return None if spy is None else side_return - spy


def move_atr(row: Mapping[str, Any], atr20: float | None, *, as_of: str) -> float | None:
    """The side move in ATR20 units over a mature, measured row; else None."""
    if str(row.get("measured") or "").strip().lower() != "true":
        return None
    if str(row.get("maturity") or "").strip().lower() != "mature":
        return None
    if as_of and _day(row.get("target_session")) > as_of:
        return None
    side_return, close = _num(row.get("side_return_pct")), _num(row.get("entry_close"))
    if side_return is None or close is None or close <= 0 or atr20 is None or atr20 <= 0:
        return None
    return side_return / 100.0 * close / atr20


def window_days(horizon_rows: Iterable[Mapping[str, Any]], *, as_of: str,
                sessions: int = WINDOW_SESSIONS) -> list[str]:
    """The newest ``sessions`` scan dates at or before ``as_of``."""
    days = {
        _day(row.get("scan_date")) for row in horizon_rows
        if _day(row.get("scan_date")) and (not as_of or _day(row.get("scan_date")) <= as_of)
    }
    return sorted(days)[-int(sessions):] if sessions > 0 else []


def _wilson_low(wins: int, n: int) -> float | None:
    from swing_headline import wilson_lower_bound

    return wilson_lower_bound(wins, n) if n else None


def build_families(
    horizon_rows: list[Mapping[str, Any]],
    spy_closes: Mapping[str, float],
    atr_by_day: Mapping[tuple[str, str], float],
    tracker_r: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    as_of: str,
    sessions: int = WINDOW_SESSIONS,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Per ``SIDE|family`` cell over the trailing ``sessions`` scan dates.

    ``atr_by_day`` is ``{(SYMBOL, day): atr20}``; ``tracker_r`` is
    ``{SIDE|family: {"avg_total_r", "setups"}}`` from the attribute leaderboard.
    Returns ``(cells, window days)``.
    """
    days = window_days(horizon_rows, as_of=as_of, sessions=sessions)
    kept = set(days)
    acc: dict[str, dict[str, Any]] = {}
    for row in horizon_rows:
        if str(row.get("outcome_kind") or OUTCOME_KIND).strip() != OUTCOME_KIND:
            continue
        day = _day(row.get("scan_date"))
        if day not in kept:
            continue
        side = str(row.get("side") or "").strip().upper()
        if side not in {"LONG", "SHORT"}:
            continue
        horizon = int(_num(row.get("horizon_sessions")) or 0)
        key = family_key(side, row.get("setup_family"))
        cell = acc.setdefault(key, {
            "side": side, "family": key.split("|", 1)[1], "wins": 0, "excess": [],
            "days": set(), "moves": {h: [] for h in MOVE_HORIZONS},
        })
        if horizon == TAPE_HORIZON:
            excess = tape_excess(row, spy_closes, as_of=as_of)
            if excess is not None:
                cell["excess"].append(excess)
                cell["wins"] += 1 if excess > 0 else 0
                cell["days"].add(day)
        if horizon in MOVE_HORIZONS:
            atr = atr_by_day.get((str(row.get("symbol") or "").strip().upper(), day))
            move = move_atr(row, atr, as_of=as_of)
            if move is not None:
                cell["moves"][horizon].append(move)
    cells: dict[str, dict[str, Any]] = {}
    for key, cell in sorted(acc.items()):
        n = len(cell["excess"])
        out: dict[str, Any] = {
            "side": cell["side"],
            "family": cell["family"],
            "n": n,
            "sessions": len(cell["days"]),
            "wins_h5": cell["wins"],
            "beat_rate_h5": _r4(cell["wins"] / n) if n else None,
            "beat_low_h5": _r4(_wilson_low(cell["wins"], n)),
            "mean_excess_pct_h5": _mean(cell["excess"]),
        }
        for horizon in MOVE_HORIZONS:
            moves = cell["moves"][horizon]
            out[f"n_move_h{horizon}"] = len(moves)
            out[f"mean_move_atr_h{horizon}"] = _mean(moves)
            out[f"median_move_atr_h{horizon}"] = _r4(median(moves)) if moves else None
        ups = [m for m in cell["moves"][10] if m > 0]
        downs = [-m for m in cell["moves"][10] if m < 0]
        out["payoff_h10"] = _r4((sum(ups) / len(ups)) / (sum(downs) / len(downs))) if ups and downs else None
        tracker = (tracker_r or {}).get(key) or {}
        out["avg_total_r"] = _r4(_num(tracker.get("avg_total_r")))
        out["tracker_setups"] = int(_num(tracker.get("setups")) or 0)
        out["adjust"] = family_adjust(out)
        cells[key] = out
    return cells, days


def tracker_r_by_family(leaderboard_rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The tracker's avg_total_r per ``SIDE|family``: setup-count weighted over buckets."""
    acc: dict[str, list[float]] = {}
    for row in leaderboard_rows:
        if str(row.get("attribute_key") or "").strip() != "setup.setup_family":
            continue
        count, r = _num(row.get("setup_count")), _num(row.get("avg_total_r"))
        if not count or r is None:
            continue
        key = family_key(row.get("side"), row.get("value_label"))
        total = acc.setdefault(key, [0.0, 0.0])
        total[0] += count * r
        total[1] += count
    return {key: {"avg_total_r": total[0] / total[1], "setups": int(total[1])}
            for key, total in acc.items() if total[1]}


# ------------------------------------------------------------------- the trial
def _side_top(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    """The top quartile (ceil(n/4)) by ``score_key``, ties by symbol."""
    if not rows:
        return []
    ranked = sorted(rows, key=lambda r: (-r[score_key], r["symbol"]))
    return ranked[: max(1, math.ceil(len(ranked) / 4))]


def trial_summary(
    horizon_rows: list[Mapping[str, Any]],
    spy_closes: Mapping[str, float],
    scores: Mapping[tuple[str, str, str], float],
    tracker_pick_r: Mapping[tuple[str, str, str], float | None],
    adjust_history: Mapping[str, Mapping[str, float]],
    *,
    as_of: str,
) -> dict[str, Any]:
    """Champion top quartile vs SP4 top quartile, per side, over the trial's entry sessions.

    ``scores`` is ``{(SYMBOL, SIDE, day): live priority score}``; ``tracker_pick_r``
    is ``{(SYMBOL, SIDE, day): representative closed R or None}``;
    ``adjust_history`` is ``{night session: {SIDE|family: adjust}}`` - each entry
    session is ranked with the adjusts written that night.
    """
    nights = sorted(day for day in adjust_history if not as_of or day <= as_of)
    entries = nights[:TRIAL_ENTRY_SESSIONS]
    h5: dict[tuple[str, str], list[dict[str, Any]]] = {}
    wanted = set(entries)
    for row in horizon_rows:
        if int(_num(row.get("horizon_sessions")) or 0) != TAPE_HORIZON:
            continue
        day = _day(row.get("scan_date"))
        if day not in wanted:
            continue
        side = str(row.get("side") or "").strip().upper()
        symbol = str(row.get("symbol") or "").strip().upper()
        score = _num(scores.get((symbol, side, day)))
        if side not in {"LONG", "SHORT"} or score is None:
            continue
        adjust = _num((adjust_history.get(day) or {}).get(family_key(side, row.get("setup_family")))) or 0.0
        h5.setdefault((day, side), []).append({
            "symbol": symbol, "champion": score, "sp4": score + adjust,
            "excess": tape_excess(row, spy_closes, as_of=as_of),
            "r": _num(tracker_pick_r.get((symbol, side, day))),
        })
    out: dict[str, Any] = {
        "first_session": nights[0] if nights else "",
        "entry_sessions": entries,
        "entry_target": TRIAL_ENTRY_SESSIONS,
    }
    for side in ("LONG", "SHORT"):
        picks: dict[str, list[dict[str, Any]]] = {"champion": [], "sp4": []}
        matured = 0
        entered = 0
        for day in entries:
            rows = h5.get((day, side)) or []
            if not rows:
                continue
            entered += 1
            if any(r["excess"] is not None for r in rows):
                matured += 1
            for name in picks:
                picks[name].extend(_side_top(rows, name))
        block: dict[str, Any] = {"entry_sessions": entered, "matured_sessions": matured}
        for name, chosen in picks.items():
            excess = [r["excess"] for r in chosen if r["excess"] is not None]
            rs = [r["r"] for r in chosen if r["r"] is not None]
            block[name] = {"n": len(excess), "excess_pct": _mean(excess),
                           "r_n": len(rs), "tracker_r": _mean(rs)}
        block["verdict"] = trial_verdict(block, entries_done=len(entries))
        out[side.lower()] = block
    return out


def trial_verdict(block: Mapping[str, Any], *, entries_done: int) -> str:
    """The fixed SP4 rule on one side's block."""
    champion, sp4 = block.get("champion") or {}, block.get("sp4") or {}
    c_x, s_x = _num(champion.get("excess_pct")), _num(sp4.get("excess_pct"))
    matured = int(_num(block.get("matured_sessions")) or 0)
    if c_x is None or s_x is None:
        return COLLECTING
    if matured >= STOP_AFTER_SESSIONS and s_x - c_x < -STOP_TRAIL_PCT:
        return STOPPED
    if entries_done >= TRIAL_ENTRY_SESSIONS and matured >= TRIAL_ENTRY_SESSIONS:
        c_r, s_r = _num(champion.get("tracker_r")), _num(sp4.get("tracker_r"))
        if s_x - c_x >= SUCCESS_EXCESS_PCT and c_r is not None and s_r is not None and s_r - c_r >= SUCCESS_R:
            return SUCCESS
        return NO_EDGE
    return COLLECTING


_VERDICT_TEXT = {
    COLLECTING: "collecting",
    SUCCESS: "SUCCESS - promotion is ask-first",
    NO_EDGE: "no edge - roll back (delete the column)",
    STOPPED: "downside stop - roll back (delete the column)",
}


def _pct(value: Any) -> str:
    number = _num(value)
    return "unknown" if number is None else f"{round(number, 2) + 0.0:+.2f}%"


def _r(value: Any) -> str:
    number = _num(value)
    return "unknown" if number is None else f"{round(number, 2) + 0.0:+.2f}R"


def saturday_lines(trial: Mapping[str, Any] | None) -> list[str]:
    """One line per side for the Saturday report. Formatting only."""
    trial = trial or {}
    lines = []
    for side, word in (("long", "longs"), ("short", "shorts")):
        block = trial.get(side) or {}
        if not int(_num(block.get("entry_sessions")) or 0):
            lines.append(f"SP4 shadow, {word}: no entry sessions yet (the trial starts the first "
                         "night the family evidence is written).")
            continue
        champion, sp4 = block.get("champion") or {}, block.get("sp4") or {}
        lines.append(
            f"SP4 shadow, {word}: {block.get('matured_sessions', 0)} of {TRIAL_ENTRY_SESSIONS} entry "
            f"sessions measured ({block.get('entry_sessions', 0)} entered since "
            f"{trial.get('first_session') or '?'}). Top quartile vs SPY: SP4 {_pct(sp4.get('excess_pct'))} "
            f"vs champion {_pct(champion.get('excess_pct'))} (n {sp4.get('n', 0)} / {champion.get('n', 0)}); "
            f"tracker R SP4 {_r(sp4.get('tracker_r'))} vs {_r(champion.get('tracker_r'))}. "
            f"{_VERDICT_TEXT.get(str(block.get('verdict') or COLLECTING), 'collecting')}."
        )
    return lines


def with_sp4(rows: Iterable[Mapping[str, Any]], evidence: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Copies of ``rows`` in the SAME order with ``sp4_score`` added ("" when unknown).

    The input rows and their order are never changed: the live sort is the caller's.
    """
    out = []
    for row in rows:
        points = sp4_points(row, evidence)
        out.append({**row, "sp4_score": "" if points is None else points})
    return out


def chip_text(evidence: Mapping[str, Any] | None) -> str:
    """The Setup Tracker's SP4 chip: what the shadow column is and its biggest moves."""
    evidence = evidence or {}
    cells = evidence.get("families") or {}
    if not isinstance(cells, Mapping) or not cells:
        return "SP4 (shadow): no family evidence yet - the column shows the live score."
    basis = regime_basis(evidence)
    moved = sorted(
        ((adjust, key) for key in basis
         if (adjust := adjust_for(*key.split("|", 1), evidence))),
        key=lambda item: (-abs(item[0]), item[1]),
    )
    tops = ", ".join(
        f"{key.split('|', 1)[1]} {key.split('|', 1)[0]} {adjust:+.0f}" for adjust, key in moved[:4]
    ) or "no family past the n >= 80 / 15-session floor"
    window = evidence.get("window") or {}
    text = (
        f"SP4 (shadow, as of {evidence.get('as_of') or '?'}, {window.get('sessions', '?')} sessions): "
        f"live score + family adjust. Biggest: {tops}. The live sort, buckets and alerts are unchanged."
    )
    regime = str(evidence.get("current_regime") or "")
    if regime:
        label = str(evidence.get("current_regime_label") or regime.replace("_", " "))
        in_regime = sum(1 for value in basis.values() if value == regime)
        fell_back = len(basis) - in_regime
        text += (
            f" Regime {label}: {in_regime} famil{'y' if in_regime == 1 else 'ies'} read this regime's cells; "
            f"{fell_back} fell back to {ALL_REGIMES} (under n {MIN_N} / {MIN_SESSIONS} sessions in this regime)."
        )
    elif "current_regime" in evidence:
        text += f" No regime typed: every family reads {ALL_REGIMES}."
    return text
