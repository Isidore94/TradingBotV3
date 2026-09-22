"""Does the point system rank well? Track it, grade it, propose a correction.

Trader, 2026-09-08: *"ensure that we track how this system performs ... do
higher ranked setups perform better? ... if it's not we should have a way for
the system to correct itself over time so that the highest rated stuff
actually performs."*

Three seams, all shadow, none reaching a detector, a score, an alert or a
watchlist:

1. **The log** (`SETUP_POINTS_LOG_FILE`, append-only JSONL): every distinct
   same-day score snapshot retains its version, inputs, parts, total, shown
   multipliers and bucket. A failed append loses the row, never the table.
2. **The grade** (`grade`): the earliest eligible pre-close v2 snapshot per
   entry session/name/side joins the tracker's session outcome through
   `swing_evidence.read_eligible_rows` and `POLICY_SESSION_V2` after five
   exchange sessions. Whole tied values stay together in the total thirds
   and part halves. V1 logs remain readable but do not join this v2 grade.
3. **The correction** (`propose_weights` -> `SETUP_POINTS_WEIGHTS_FILE`): one
   multiplier per part, `1 + 2 x lift` clamped to [0.5, 1.5], proposed only
   when BOTH halves of that part hold at least `MIN_REPORTABLE_N` graded rows
   from five distinct entry sessions each;
   otherwise 1.0 with the reason. The desk APPLIES the proposal only when the
   trader's `setup_points_learned_weights` switch is ON (default OFF); the
   Points tooltip says which multipliers are in force either way. A weight
   never moves silently, and it never moves on fewer than the floor.

Everything is computed on the panel's worker, never the Qt thread.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

PARTS = ("setup", "sr", "rs", "bounce")
WEIGHT_FLOOR = 0.5
WEIGHT_CEILING = 1.5
LIFT_GAIN = 2.0
WEIGHTS_VERSION = "points_v2"
OUTCOME_POLICY_V1 = "favorable_direction_scanrow_v1"
OUTCOME_POLICY_V2 = "favorable_direction_session_v2"


@dataclass(frozen=True)
class Cell:
    label: str
    n: int
    wins: int

    @property
    def win_rate(self) -> float | None:
        return (self.wins / self.n) if self.n else None

    @property
    def lower_bound(self) -> float | None:
        from swing_headline import wilson_lower_bound

        return wilson_lower_bound(self.wins, self.n) if self.n else None

    def line(self) -> str:
        if not self.n:
            return f"{self.label}: no graded rows"
        return f"{self.label}: {self.win_rate * 100:.0f}% of {self.n} (>= {self.lower_bound * 100:.0f}%)"


@dataclass(frozen=True)
class PointsGrade:
    n_logged: int
    n_joined: int
    horizon_sessions: int
    terciles: tuple[Cell, Cell, Cell]  # top, middle, bottom
    part_lift: dict[str, float | None] = field(default_factory=dict)
    part_halves: dict[str, tuple[Cell, Cell]] = field(default_factory=dict)
    window: tuple[str, str] = ("", "")
    n_pending: int = 0
    outcome_policy: str = OUTCOME_POLICY_V2
    distinct_symbols: int = 0
    distinct_sessions: int = 0
    part_sessions: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def lift(self) -> float | None:
        top, _mid, bottom = self.terciles
        if top.win_rate is None or bottom.win_rate is None:
            return None
        return top.win_rate - bottom.win_rate

    def sentence(self) -> str:
        """The one line the desk shows. Says 'not enough' rather than a number."""
        if not self.n_joined:
            return (
                f"Points grade: {self.n_logged} ranked rows logged, none graded yet "
                f"(outcomes arrive {self.horizon_sessions} sessions after the scan)."
            )
        top, _mid, bottom = self.terciles
        lift = self.lift
        head = (
            f"Points grade over {self.n_joined} graded rows at {self.horizon_sessions} exchange sessions "
            f"(directional %, {self.distinct_symbols} symbols / {self.distinct_sessions} sessions): "
        )
        if lift is None or min(top.n, bottom.n) < _floor():
            return head + f"not enough per third yet ({top.line()}; {bottom.line()})."
        verdict = "higher points DID perform better" if lift > 0 else "higher points did NOT perform better"
        return head + f"{verdict} - {top.line()} vs {bottom.line()}, lift {lift * 100:+.0f} pts."

    def to_payload(self) -> dict[str, Any]:
        out = asdict(self)
        out["lift"] = self.lift
        out["sentence"] = self.sentence()
        return out


def _floor() -> int:
    from evidence_stats import MIN_REPORTABLE_N

    return int(MIN_REPORTABLE_N)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- the log


def log_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Snapshot identity; old logs lacking a snapshot retain their v1 grain."""
    snapshot = str(row.get("snapshot_id") or "").strip()
    return (
        str(row.get("scan_date") or ""),
        str(row.get("symbol") or "").upper(),
        str(row.get("side") or "").upper(),
        str(row.get("points_version") or "points_v1"),
        snapshot,
    )


def _outcome_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("scan_date") or ""), str(row.get("symbol") or "").upper(),
        str(row.get("side") or "").upper(),
    )


def read_log(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except OSError:
        return out
    return out


def append_log(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Append new score snapshots; a later same-day measurement is evidence too."""
    existing = {log_key(row) for row in read_log(path)}
    fresh: list[dict[str, Any]] = []
    for row in rows:
        key = log_key(row)
        if not all(key[:4]) or (key[3] == WEIGHTS_VERSION and not key[4]) or key in existing:
            continue
        existing.add(key)
        record = dict(row)
        record.setdefault("logged_at", _now_iso())
        record.setdefault("points_version", WEIGHTS_VERSION)
        record.setdefault("weights_version", record["points_version"])
        fresh.append(record)
    if not fresh:
        return 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for record in fresh:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        return 0
    return len(fresh)


# ---------------------------------------------------------------- the grade


def _win(value: Any) -> bool | None:
    text = str(value if value is not None else "").strip().lower()
    if text in {"1", "true", "yes", "win"}:
        return True
    if text in {"0", "false", "no", "loss"}:
        return False
    return None


def _cell(label: str, rows: list[tuple[float, bool]]) -> Cell:
    return Cell(label=label, n=len(rows), wins=sum(1 for _value, win in rows if win))


def _lift(top: Cell, bottom: Cell) -> float | None:
    if top.win_rate is None or bottom.win_rate is None:
        return None
    return top.win_rate - bottom.win_rate


def grade(
    log_rows: Iterable[Mapping[str, Any]],
    outcome_rows: Iterable[Mapping[str, Any]],
    *,
    horizon_sessions: int,
    window: tuple[str, str] = ("", ""),
    n_pending: int = 0,
    outcome_policy: str = OUTCOME_POLICY_V2,
) -> PointsGrade:
    """Join the log to graded outcomes and read the terciles. Pure."""
    log_rows = [dict(row) for row in log_rows if isinstance(row, Mapping)]
    outcome_rows = [dict(row) for row in outcome_rows if isinstance(row, Mapping)]
    outcomes: dict[tuple[str, str, str], bool] = {}
    for row in outcome_rows:
        win = _win(row.get("favorable", row.get("win")))
        if win is None:
            continue
        outcomes.setdefault(_outcome_key(row), win)
    joined: list[dict[str, Any]] = []
    logged = 0
    for row in log_rows:
        logged += 1
        win = outcomes.get(_outcome_key(row))
        if win is None:
            continue
        try:
            total = float(row.get("total"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(total):
            continue
        joined.append({"total": total, "win": win, "scan_date": row.get("scan_date"), "symbol": row.get("symbol"), **{part: row.get(part) for part in PARTS}})

    ordered = sorted(joined, key=lambda item: -item["total"])
    top, middle, bottom = _whole_value_groups(ordered, "total", len(ordered) // 3)
    terciles = (
        _cell("top third", [(i["total"], i["win"]) for i in top]),
        _cell("middle third", [(i["total"], i["win"]) for i in middle]),
        _cell("bottom third", [(i["total"], i["win"]) for i in bottom]),
    )
    part_lift: dict[str, float | None] = {}
    part_halves: dict[str, tuple[Cell, Cell]] = {}
    part_sessions: dict[str, tuple[int, int]] = {}
    for part in PARTS:
        scored = []
        for item in joined:
            try:
                value = float(item.get(part))
                if math.isfinite(value):
                    scored.append({"value": value, "win": item["win"], "scan_date": item.get("scan_date")})
            except (TypeError, ValueError):
                continue
        scored.sort(key=lambda item: -item["value"])
        upper_rows, _mid, lower_rows = _whole_value_groups(scored, "value", len(scored) // 2)
        upper = _cell(f"{part} top half", [(item["value"], item["win"]) for item in upper_rows])
        lower = _cell(f"{part} bottom half", [(item["value"], item["win"]) for item in lower_rows])
        part_halves[part] = (upper, lower)
        part_lift[part] = _lift(upper, lower)
        part_sessions[part] = (
            len({str(item.get("scan_date") or "") for item in upper_rows if item.get("scan_date")}),
            len({str(item.get("scan_date") or "") for item in lower_rows if item.get("scan_date")}),
        )
    return PointsGrade(
        n_logged=logged,
        n_joined=len(joined),
        horizon_sessions=int(horizon_sessions),
        terciles=terciles,
        part_lift=part_lift,
        part_halves=part_halves,
        window=window,
        n_pending=int(n_pending),
        outcome_policy=outcome_policy,
        distinct_symbols=len({str(row.get("symbol") or "").upper() for row in joined if str(row.get("symbol") or "").strip()}),
        distinct_sessions=len({str(row.get("scan_date") or "")[:10] for row in joined if str(row.get("scan_date") or "").strip()}),
        part_sessions=part_sessions,
    )


def _whole_value_groups(rows: list[dict[str, Any]], key: str, target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Take whole equal-value groups, or abstain when one value owns a boundary."""
    if not target or len(rows) < 2:
        return [], list(rows), []
    groups: list[list[dict[str, Any]]] = []
    for row in rows:
        if not groups or row[key] != groups[-1][0][key]:
            groups.append([row])
        else:
            groups[-1].append(row)
    if len(groups) < 2:
        return [], list(rows), []
    # Choose the pair of whole-group boundaries closest to the desired size.
    # A binary part with 40 high and 80 low readings must compare 40 against
    # 80, rather than consume the low tie into both halves and abstain.
    prefixes: list[int] = []
    cumulative = 0
    for group in groups[:-1]:
        cumulative += len(group)
        prefixes.append(cumulative)
    suffix_best: list[tuple[int, int]] = [(0, 0)] * len(prefixes)
    best: tuple[int, int] | None = None
    for index in range(len(prefixes) - 1, -1, -1):
        candidate = (abs(len(rows) - prefixes[index] - target), index + 1)
        best = candidate if best is None or candidate < best else best
        suffix_best[index] = best
    _distance, left, right = min(
        (abs(prefixes[index] - target) + suffix_best[index][0], index + 1, suffix_best[index][1])
        for index in range(len(prefixes))
    )
    top = [row for group in groups[:left] for row in group]
    bottom = [row for group in groups[right:] for row in group]
    excluded = {id(item) for item in top + bottom}
    middle = [row for row in rows if id(row) not in excluded]
    return top, middle, bottom


# ---------------------------------------------------------------- the correction


def propose_weights(result: PointsGrade) -> dict[str, Any]:
    """One multiplier per part from its lift, floored on n. Nothing applies it."""
    floor = _floor()
    multipliers: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for part in PARTS:
        upper, lower = result.part_halves.get(part, (Cell(part, 0, 0), Cell(part, 0, 0)))
        lift = result.part_lift.get(part)
        upper_sessions, lower_sessions = result.part_sessions.get(part, (0, 0))
        if lift is None or min(upper.n, lower.n) < floor or min(upper_sessions, lower_sessions) < 5:
            multipliers[part] = 1.0
            reasons[part] = (
                f"kept at 1.0: {min(upper.n, lower.n)} graded per half, floor {floor}; "
                f"{upper_sessions}/{lower_sessions} distinct entry sessions (need 5 each)"
            )
            continue
        value = max(WEIGHT_FLOOR, min(WEIGHT_CEILING, 1.0 + LIFT_GAIN * lift))
        multipliers[part] = round(value, 2)
        reasons[part] = f"lift {lift * 100:+.0f} pts over {upper.n}+{lower.n} graded -> x{value:.2f}"
    return {
        "weights_version": WEIGHTS_VERSION,
        "outcome_policy": result.outcome_policy,
        "as_of": _now_iso(),
        "n_joined": result.n_joined,
        "horizon_sessions": result.horizon_sessions,
        "multipliers": multipliers,
        "reasons": reasons,
        "grade": result.to_payload(),
    }


def write_proposal(path: Path, proposal: Mapping[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(dict(proposal), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return False
    return True


def read_proposal(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def proposal_multipliers(
    proposal: Mapping[str, Any] | None,
    *,
    points_version: str = WEIGHTS_VERSION,
    outcome_policy: str = OUTCOME_POLICY_V2,
) -> dict[str, float]:
    if str((proposal or {}).get("weights_version") or "") != points_version:
        return {}
    if str((proposal or {}).get("outcome_policy") or "") != outcome_policy:
        return {}
    out = {part: 1.0 for part in PARTS}
    for part, value in dict((proposal or {}).get("multipliers") or {}).items():
        if part in out:
            try:
                out[part] = max(WEIGHT_FLOOR, min(WEIGHT_CEILING, float(value)))
            except (TypeError, ValueError):
                continue
    return out


# ---------------------------------------------------------------- the worker's one pass


def log_and_grade(
    ranked_rows: Iterable[Mapping[str, Any]],
    *,
    log_path: Path,
    weights_path: Path,
    outcomes_path: Path,
) -> PointsGrade:
    """Append today's rows, grade the whole log, write the proposal. Off the Qt thread."""
    from swing_evidence import POLICY_SESSION_V2, read_eligible_rows

    append_log(log_path, ranked_rows)
    log_rows = read_log(log_path)
    dates = sorted({str(row.get("scan_date") or "") for row in log_rows if row.get("scan_date")})
    window = (dates[0], dates[-1]) if dates else ("", "")
    policy = POLICY_SESSION_V2
    outcome_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    if dates:
        try:
            # The log declares the sample window.  v2's own clock is the
            # target session, so a scan-date window would exclude every mature
            # five-session result before the join gets to decide its identity.
            read = read_eligible_rows(outcomes_path, policy, window=("0000-01-01", "9999-12-31"))
            outcome_rows = list(read.rows)
            pending_rows = list(read.pending)
        except Exception:  # noqa: BLE001 - an unreadable outcome file is "none graded yet"
            outcome_rows = []
    observed = _declared_pre_close_observations(log_rows)
    outcome_rows = _points_policy_outcomes(outcome_rows, policy.outcome_kind)
    observed_keys = {_outcome_key(row) for row in observed}
    pending = sum(
        1 for row in pending_rows
        if _outcome_key(row) in observed_keys and str(row.get("outcome_kind") or "") == policy.outcome_kind
    )
    result = grade(
        observed, outcome_rows, horizon_sessions=int(policy.horizon_sessions), window=window,
        n_pending=pending, outcome_policy=policy.outcome_kind,
    )
    write_proposal(weights_path, propose_weights(result))
    return result


def _points_policy_outcomes(rows: Iterable[Mapping[str, Any]], outcome_kind: str) -> list[dict[str, Any]]:
    """This scorecard's strict v2 guard, without changing generic readers."""
    out: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if str(row.get("outcome_kind") or "") != outcome_kind:
            continue
        if outcome_kind == OUTCOME_POLICY_V2 and str(row.get("maturity") or "").strip().lower() != "mature":
            continue
        out.append(row)
    return out


def _declared_pre_close_observations(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The earliest v2 snapshot before the entry-session close is the gradeable one."""
    chosen: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        if str(row.get("points_version") or "points_v1") != WEIGHTS_VERSION:
            continue
        stamp = str(row.get("observed_at") or row.get("logged_at") or "")
        day = str(row.get("scan_date") or "")[:10]
        observed = _aware_timestamp(stamp)
        if observed is None or not day or not _is_pre_close_observation(stamp, day):
            continue
        logged = str(row.get("logged_at") or stamp)
        scan_stamp = str(row.get("scan_timestamp") or "")
        logged_at = _aware_timestamp(logged)
        scanned_at = _aware_timestamp(scan_stamp) if scan_stamp else None
        if logged_at is None or logged_at < observed or (scan_stamp and scanned_at is None) or (scanned_at is not None and scanned_at > observed):
            continue
        key = (day, str(row.get("symbol") or "").upper(), str(row.get("side") or "").upper(), WEIGHTS_VERSION)
        if not all(key[:3]):
            continue
        earlier = _aware_timestamp(str(chosen[key].get("observed_at") or chosen[key].get("logged_at") or "")) if key in chosen else None
        if earlier is None or observed < earlier:
            chosen[key] = row
    return list(chosen.values())


def _is_pre_close_observation(stamp: str, session_date: str) -> bool:
    """Compare an aware observation with the real (including half-day) close."""
    try:
        from market_calendar import is_session
        from market_early_close import session_close
        day = date.fromisoformat(session_date)
        if not is_session(day):
            return False
        observed = _aware_timestamp(stamp)
        return observed is not None and observed.astimezone(session_close(day).tzinfo).date() == day and observed < session_close(day)
    except Exception:
        return False


def _aware_timestamp(stamp: str) -> datetime | None:
    """Read instants consistently; an old naive desk stamp gets its own zone."""
    if not stamp:
        return None
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        from market_session import get_market_local_timezone

        desk_zone, _name = get_market_local_timezone()
        moment = moment.replace(tzinfo=desk_zone)
    return moment


def replay_observation(
    legacy_row: Mapping[str, Any], *, reconstructed_row_facts: Mapping[str, Any] | None = None,
    reconstructed_family_record: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Pure comparison only; reconstructed facts are labelled, never prospective evidence."""
    import setup_points

    old = setup_points.score_row(legacy_row, side=str(legacy_row.get("side") or ""), version="points_v1")
    merged = dict(legacy_row)
    merged.update(dict(reconstructed_row_facts or {}))
    rebuilt = setup_points.score_row(merged, side=str(merged.get("side") or ""), family_record=reconstructed_family_record)
    return {
        "legacy": {**old.log_row(scan_date=str(legacy_row.get("scan_date") or ""), symbol=str(legacy_row.get("symbol") or ""), side=str(legacy_row.get("side") or ""), family="", bucket=""), "points_version": "points_v1"},
        "reconstructed": {**rebuilt.log_row(scan_date=str(merged.get("scan_date") or ""), symbol=str(merged.get("symbol") or ""), side=str(merged.get("side") or ""), family="", bucket=""), "points_version": WEIGHTS_VERSION, "reconstructed_row_facts": bool(reconstructed_row_facts), "reconstructed_family_facts": bool(reconstructed_family_record)},
    }
