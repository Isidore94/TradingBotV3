"""Does the point system rank well? Track it, grade it, propose a correction.

Trader, 2026-09-08: *"ensure that we track how this system performs ... do
higher ranked setups perform better? ... if it's not we should have a way for
the system to correct itself over time so that the highest rated stuff
actually performs."*

Three seams, all shadow, none reaching a detector, a score, an alert or a
watchlist:

1. **The log** (`SETUP_POINTS_LOG_FILE`, append-only JSONL): one row per
   ranked-bucket setup per scan date - the RAW four parts (before any learned
   multiplier), the total the desk showed, the multipliers it used and the
   bucket. De-duplicated on `(scan_date, symbol, side)`; a failed append loses
   the row, never the table.
2. **The grade** (`grade`): the log joined to the tracker's own outcome rows
   (`master_avwap_tier_outcomes.csv`, the ONE reader `swing_evidence.
   read_eligible_rows`, the declared horizon) on `(scan_date, symbol, side)`.
   Terciles by total, each with `n`, win rate and the Wilson lower bound; the
   LIFT (top third minus bottom third) is the headline; every part gets the
   same top-half-minus-bottom-half lift so the trader can see WHICH input is
   earning its weight.
3. **The correction** (`propose_weights` -> `SETUP_POINTS_WEIGHTS_FILE`): one
   multiplier per part, `1 + 2 x lift` clamped to [0.5, 1.5], proposed only
   when BOTH halves of that part hold at least `MIN_REPORTABLE_N` graded rows;
   otherwise 1.0 with the reason. The desk APPLIES the proposal only when the
   trader's `setup_points_learned_weights` switch is ON (default OFF); the
   Points tooltip says which multipliers are in force either way. A weight
   never moves silently, and it never moves on fewer than the floor.

Everything is computed on the panel's worker, never the Qt thread.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

PARTS = ("setup", "sr", "rs", "bounce")
WEIGHT_FLOOR = 0.5
WEIGHT_CEILING = 1.5
LIFT_GAIN = 2.0
WEIGHTS_VERSION = "points_v1"


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
        head = f"Points grade over {self.n_joined} graded rows at {self.horizon_sessions} sessions: "
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


def log_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("scan_date") or ""),
        str(row.get("symbol") or "").upper(),
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
    """Append the ranked rows not already logged for their scan date. Returns the count written."""
    existing = {log_key(row) for row in read_log(path)}
    fresh: list[dict[str, Any]] = []
    for row in rows:
        key = log_key(row)
        if not all(key) or key in existing:
            continue
        existing.add(key)
        record = dict(row)
        record.setdefault("logged_at", _now_iso())
        record.setdefault("weights_version", WEIGHTS_VERSION)
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
) -> PointsGrade:
    """Join the log to graded outcomes and read the terciles. Pure."""
    outcomes: dict[tuple[str, str, str], bool] = {}
    for row in outcome_rows:
        win = _win(row.get("win"))
        if win is None:
            continue
        outcomes.setdefault(log_key(row), win)
    joined: list[dict[str, Any]] = []
    logged = 0
    for row in log_rows:
        logged += 1
        win = outcomes.get(log_key(row))
        if win is None:
            continue
        try:
            total = float(row.get("total"))
        except (TypeError, ValueError):
            continue
        joined.append({"total": total, "win": win, **{part: row.get(part) for part in PARTS}})

    ordered = sorted(joined, key=lambda item: -item["total"])
    third = len(ordered) // 3
    top = ordered[:third] if third else []
    bottom = ordered[len(ordered) - third :] if third else []
    middle = ordered[third : len(ordered) - third] if third else ordered
    terciles = (
        _cell("top third", [(i["total"], i["win"]) for i in top]),
        _cell("middle third", [(i["total"], i["win"]) for i in middle]),
        _cell("bottom third", [(i["total"], i["win"]) for i in bottom]),
    )
    part_lift: dict[str, float | None] = {}
    part_halves: dict[str, tuple[Cell, Cell]] = {}
    for part in PARTS:
        scored = []
        for item in joined:
            try:
                scored.append((float(item.get(part)), item["win"]))
            except (TypeError, ValueError):
                continue
        scored.sort(key=lambda pair: -pair[0])
        if scored and scored[0][0] == scored[-1][0]:
            # A CONSTANT part splits by arrival order, not by value, and the
            # "lift" would be whatever the log happened to be sorted by. No
            # halves, no lift, and `propose_weights` keeps it at 1.0.
            scored = []
        half = len(scored) // 2
        upper = _cell(f"{part} top half", scored[:half])
        lower = _cell(f"{part} bottom half", scored[len(scored) - half :] if half else [])
        part_halves[part] = (upper, lower)
        part_lift[part] = _lift(upper, lower)
    return PointsGrade(
        n_logged=logged,
        n_joined=len(joined),
        horizon_sessions=int(horizon_sessions),
        terciles=terciles,
        part_lift=part_lift,
        part_halves=part_halves,
        window=window,
    )


# ---------------------------------------------------------------- the correction


def propose_weights(result: PointsGrade) -> dict[str, Any]:
    """One multiplier per part from its lift, floored on n. Nothing applies it."""
    floor = _floor()
    multipliers: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for part in PARTS:
        upper, lower = result.part_halves.get(part, (Cell(part, 0, 0), Cell(part, 0, 0)))
        lift = result.part_lift.get(part)
        if lift is None or min(upper.n, lower.n) < floor:
            multipliers[part] = 1.0
            reasons[part] = f"kept at 1.0: {min(upper.n, lower.n)} graded per half, floor {floor}"
            continue
        value = max(WEIGHT_FLOOR, min(WEIGHT_CEILING, 1.0 + LIFT_GAIN * lift))
        multipliers[part] = round(value, 2)
        reasons[part] = f"lift {lift * 100:+.0f} pts over {upper.n}+{lower.n} graded -> x{value:.2f}"
    return {
        "weights_version": WEIGHTS_VERSION,
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


def proposal_multipliers(proposal: Mapping[str, Any] | None) -> dict[str, float]:
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
    from evidence_stats import SWING_HORIZON_SESSIONS
    from swing_evidence import POLICY_SCANROW_V1, read_eligible_rows

    append_log(log_path, ranked_rows)
    log_rows = read_log(log_path)
    dates = sorted({str(row.get("scan_date") or "") for row in log_rows if row.get("scan_date")})
    window = (dates[0], dates[-1]) if dates else ("", "")
    outcome_rows: list[dict[str, Any]] = []
    if dates:
        try:
            read = read_eligible_rows(outcomes_path, POLICY_SCANROW_V1, window=window)
            outcome_rows = list(read.rows)
        except Exception:  # noqa: BLE001 - an unreadable outcome file is "none graded yet"
            outcome_rows = []
    result = grade(log_rows, outcome_rows, horizon_sessions=int(SWING_HORIZON_SESSIONS), window=window)
    write_proposal(weights_path, propose_weights(result))
    return result
