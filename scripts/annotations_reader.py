"""Pure, Qt-free readers for the trader's annotation stream and its reason vocabularies.

Lifted from ``ui.annotations.store`` (which re-exports the decision-session pieces
unchanged) so non-UI readers such as the Trade Mentor's veto pack never import
``ui.*``. Everything here reads; nothing writes. The vocabularies are read straight
from their JSON files (no validation: the capture side's ``ui.annotations.vocabulary``
owns that) and the veto cohort's graded forward returns straight from its CSV.
"""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

#: The ADDITIVE field TJ-11 writes beside `session_date`: the exchange session
#: the decision JUDGED, which for a call made after Friday's close is FRIDAY's
#: (TJ-11F, trader 2026-09-19). `session_date` keeps its own meaning and its own
#: value for every writer and every reader; an old row simply lacks this key and
#: is never rewritten, and only TJ-11's own readers look at it.
DECISION_SESSION_FIELD = "decision_session"

#: Which RULE computed :data:`DECISION_SESSION_FIELD` on this row. Rows written
#: between wave 1 going live (2026-09-19 16:03 PDT) and TJ-11F could hold a
#: FORWARD-mapped value, and no row is ever rewritten - so a stored session is
#: believed only when the row also says the judged-session rule wrote it, and a
#: value without the marker is recomputed from the row's own stamp. This is a
#: schema stamp, not a vocabulary version.
DECISION_SESSION_RULE_FIELD = "decision_session_rule"
DECISION_SESSION_RULE = "judged_session_v2"

EVENT_VETO = "veto"
EVENT_PASS = "pass"
VETO_FAMILY = "veto_reasons"
PASS_FAMILY = "pass_reasons"
COHORT_HORIZONS = (1, 3, 5, 10)


def _judged_session(value: Any) -> str:
    """The exchange session a stamp JUDGED, or ``""`` when unanswerable.

    One seam, `market_calendar.decision_session`. A date that IS a session comes
    back unchanged, so this only ever walks a weekend, an evening after the
    close or a holiday BACK to the session it was made on. Answering ``""``
    rather than guessing is the point: a row that cannot be placed carries no
    claim about where it belongs.
    """
    try:
        from market_calendar import decision_session

        answer = decision_session(value)
    except Exception:  # noqa: BLE001 - a calendar that cannot answer never guesses
        return ""
    return answer.isoformat() if answer is not None else ""


def row_decision_session(row: Mapping[str, Any]) -> str:
    """Which exchange session one annotation row's decision JUDGED.

    A stored :data:`DECISION_SESSION_FIELD` is believed only when the row also
    carries :data:`DECISION_SESSION_RULE` - without it the value came from the
    forward rule TJ-11F reversed. Otherwise the row's own stamp (`created_at`,
    else `session_date`) is mapped through the calendar. The row is never
    rewritten either way: a reader maps, it does not repair.
    """
    stored = str(row.get(DECISION_SESSION_FIELD) or "").strip()
    rule = str(row.get(DECISION_SESSION_RULE_FIELD) or "").strip()
    if stored and rule == DECISION_SESSION_RULE:
        return stored[:10]
    return _judged_session(row.get("created_at") or row.get("session_date"))


def read_rows(path: Path | str, *, event_types: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Every readable row in file order; a torn or non-object line is skipped. Missing file = []."""
    wanted = None if event_types is None else {str(kind) for kind in event_types}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if wanted is not None and str(row.get("event_type") or "") not in wanted:
            continue
        rows.append(row)
    return rows


# ---------------------------------------------------------------- vocabularies
def vocabulary_dir() -> Path:
    import project_paths

    return Path(project_paths.ROOT_DIR) / "scripts" / "ui" / "annotations" / "vocabularies"


@lru_cache(maxsize=32)
def _vocabulary(path_text: str) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        str(entry.get("code") or "").strip(): dict(entry)
        for entry in payload.get("reasons") or ()
        if isinstance(entry, dict) and entry.get("code")
    }


def _versions(family: str, directory: Path) -> list[int]:
    pattern = re.compile(rf"^{re.escape(family)}_v(\d+)\.json$")
    try:
        names = [entry.name for entry in directory.iterdir()]
    except OSError:
        return []
    return sorted(int(match.group(1)) for name in names if (match := pattern.fullmatch(name)))


def reason_label(
    code: Any, *, family: str = VETO_FAMILY, version: Any = None, side: str = "", directory: Path | None = None
) -> str:
    """The reason's label as the trader saw it (a SHORT's own words when it has them); "" when unknown.

    The row's stamped version first, else the newest version that carries the code.
    """
    wanted = str(code or "").strip().lower()
    if not wanted:
        return ""
    target = Path(directory) if directory is not None else vocabulary_dir()
    versions = _versions(family, target)
    try:
        stamped = int(str(version).lower().lstrip("v")) if version not in (None, "") else None
    except ValueError:
        stamped = None
    order = ([stamped] if stamped in versions else []) + list(reversed(versions))
    short = str(side or "").strip().upper().startswith("SHORT")
    for number in order:
        entry = _vocabulary(str(target / f"{family}_v{number}.json")).get(wanted)
        if entry:
            label = str(entry.get("short_label") or "").strip() if short else ""
            return label or str(entry.get("label") or "").strip()
    return ""


# ---------------------------------------------------------------- the veto cohort's forward returns
def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def veto_forward_returns(path: Path | str) -> dict[tuple[str, str, str], dict[int, tuple[str, float]]]:
    """``{(trade_date, SYMBOL, SIDE): {horizon: (horizon_date, side_return)}}`` from ``veto_cohort_outcomes.csv``.

    Returns are side-adjusted decimals (``human_focus_tracking``'s outcome math);
    only matured horizons (a date and a number) are present. Missing file = {}.
    """
    out: dict[tuple[str, str, str], dict[int, tuple[str, float]]] = {}
    try:
        handle = Path(path).open(newline="", encoding="utf-8-sig")
    except OSError:
        return out
    with handle:
        for row in csv.DictReader(handle):
            side = str(row.get("side") or "").strip().upper()
            key = (str(row.get("trade_date") or "").strip()[:10], str(row.get("symbol") or "").strip().upper(),
                   "SHORT" if side.startswith("SHORT") else "LONG")
            if not key[0] or not key[1] or key in out:
                continue  # first row per key wins, like the cohort's own picks
            horizons: dict[int, tuple[str, float]] = {}
            for horizon in COHORT_HORIZONS:
                when = str(row.get(f"h{horizon}_date") or "").strip()[:10]
                value = _float(row.get(f"h{horizon}_return"))
                if when and value is not None:
                    horizons[horizon] = (when, value)
            out[key] = horizons
    return out
