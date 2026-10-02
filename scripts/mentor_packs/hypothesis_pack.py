"""Hypothesis pack: the night's hypotheses looked up in the shadow permutation grid. Read-only, no new compute.

A hypothesis is a QUERY into the newest ``setup_permutation_report_v1`` (``permutation_report_history/``
newest file, else the live report): ``{population: swing|m5, horizon, family, side, facets: ["name=value",
... <= 3]}``. :func:`find` validates it against the report's own vocabulary (the facet names and values
its published cells carry; an unknown one is "not in vocabulary", never a guess) and returns the
published cell - a key or a listed rejected cell - with its numbers, or why there is none.

Rows: ``hyp:report:asof`` (which report), then per hypothesis in the chat DB's ``challenges``
(kind ``hypothesis``, read ``mode=ro``): ``hyp:<id>`` (the query and why) and ``hyp:<id>:cell`` (the
cell's numbers or the miss reason, and the grade once a newer report is out).

Grading (:func:`grade_open`, deterministic): when a report with a NEWER data date appears, each open
hypothesis is looked up again: still a key (it clears the report's own hold-out bar) = hit; a
published cell that failed = miss; not found = gone. Nothing here changes a score, a filter or an alert.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mentor_packs.registry import Pack, make_pack

NAME = "hypothesis_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The night's hypotheses about setups, each looked up in the shadow permutation grid (n, win rate, "
            "Wilson lower bound, hold-out result, sessions) or why it is not there, and how each did against "
            "the next weekly report. Cells in a shadow grid, never rules."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}
KIND = "hypothesis"
POPULATIONS = ("swing", "m5")
MAX_FACETS = 3
MAX_SHOWN = 20
REPORT_SCHEMA = "setup_permutation_report_v1"
BUSY_TIMEOUT_MS = 5000
FOOTER = "*A cell in the shadow grid; changes go through fixtures and the ladder.*"
MISS_NOT_IN_GRID = ("not in the grid (depth > 3 or under the floor, or a tested cell the report does not "
                    "list)")


# ---------------------------------------------------------------- the report
@dataclass(frozen=True)
class Report:
    path: str
    asof: str
    body: Mapping[str, Any]


def _paths(history_dir: Path | str | None, report_file: Path | str | None) -> tuple[Path, Path]:
    import project_paths

    return (Path(history_dir) if history_dir is not None else Path(project_paths.SETUP_PERMUTATION_REPORT_HISTORY_DIR),
            Path(report_file) if report_file is not None else Path(project_paths.SETUP_PERMUTATION_REPORT_FILE))


def _read(path: Path) -> Report | None:
    import setup_permutation_search as sps

    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(body, dict) or body.get("schema") != REPORT_SCHEMA:
        return None
    day = sps.report_data_date(body)
    return Report(path=str(path), asof=day.isoformat() if day else "", body=body)


def latest_report(history_dir: Path | str | None = None, report_file: Path | str | None = None) -> Report | None:
    """The newest dated history copy, else the live report; None when neither reads."""
    import setup_permutation_search as sps

    history, live = _paths(history_dir, report_file)
    for _day, path in reversed(sps.history_files(history)):
        found = _read(path)
        if found is not None:
            return found
    return _read(live) if live.exists() else None


# ---------------------------------------------------------------- vocabulary
def _published_cells(family_block: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    cells: list[tuple[str, Mapping[str, Any]]] = [("key", row) for row in family_block.get("keys") or ()]
    cells += [("rejected", row) for row in family_block.get("top_rejected") or ()]
    return [(status, row) for status, row in cells if isinstance(row, Mapping)]


def vocabulary(report: Report | None) -> dict[str, Any]:
    """Per population: horizons, "family SIDE" groups and facet name -> values, from published cells only."""
    out: dict[str, Any] = {}
    if report is None:
        return out
    for population, block in sorted((report.body.get("populations") or {}).items()):
        if population not in POPULATIONS or not isinstance(block, Mapping):
            continue
        horizons: list[str] = []
        families: set[str] = set()
        facets: dict[str, set[str]] = {}
        for key, horizon in (block.get("horizons") or {}).items():
            if not isinstance(horizon, Mapping) or horizon.get("refused"):
                continue
            horizons.append(str(horizon.get("horizon_name") or key))
            for group, fam in (horizon.get("families") or {}).items():
                families.add(str(group))
                for _status, cell in _published_cells(fam or {}):
                    for name, value in (cell.get("facets") or {}).items():
                        facets.setdefault(str(name), set()).add(str(value))
        out[population] = {"horizons": horizons, "families": sorted(families),
                           "facets": {name: sorted(values) for name, values in sorted(facets.items())}}
    return out


# ---------------------------------------------------------------- lookup
@dataclass
class Cell:
    """One published cell with the report's numbers (selection LB is the report's 95%; hold-out LB is 99%)."""

    population: str
    horizon: str
    horizon_name: str
    family: str
    side: str
    facets: dict[str, str]
    label: str
    status: str  # "key" (passed the report's bar) or "rejected" (a listed cell that failed it)
    depth: int
    n: int
    sessions: int
    wins: int
    win_rate: float | None
    wilson_lb: float | None
    holdout_n: int
    holdout_wins: int
    holdout_win_rate: float | None
    holdout_lb99: float | None
    holdout_passed: bool
    holdout_reason: str
    grid_cells: int | None
    report_asof: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def text(self) -> str:
        def pct(value: float | None) -> str:
            return "unknown" if value is None else f"{value:.0%}"

        def lb(value: float | None) -> str:
            return "unknown" if value is None else f"{value:.2f}"

        verdict = "passed" if self.holdout_passed else f"failed: {self.holdout_reason or 'no reason given'}"
        return (f"{self.label} ({self.family} {self.side}, {self.horizon_name}, {self.population}): {self.status}; "
                f"selection n={self.n}, {self.sessions} sessions, win {pct(self.win_rate)}, Wilson LB95 "
                f"{lb(self.wilson_lb)}; hold-out n={self.holdout_n}, win {pct(self.holdout_win_rate)}, LB99 "
                f"{lb(self.holdout_lb99)}, {verdict}; report {self.report_asof}")


def _facet_pairs(raw: Any) -> list[tuple[str, str]] | None:
    """``["name=value", ...]`` or ``{name: value}`` -> pairs; None when unreadable."""
    items: list[tuple[str, str]] = []
    if isinstance(raw, Mapping):
        items = [(str(k).strip(), str(v).strip()) for k, v in raw.items()]
    elif isinstance(raw, (list, tuple)):
        for part in raw:
            if isinstance(part, Mapping) and "name" in part:
                items.append((str(part.get("name")).strip(), str(part.get("value", "")).strip()))
            elif isinstance(part, str) and "=" in part:
                name, _, value = part.partition("=")
                items.append((name.strip(), value.strip()))
            else:
                return None
    else:
        return None
    return items if all(name and value for name, value in items) else None


def resolve_bare_facets(query: Any, report: Report | None) -> tuple[Any, list[str]]:
    """A bare facet value (``"no_trigger"``) becomes ``name=value`` when exactly one facet name in the
    report's vocabulary carries it; else it is dropped with a note. (query, notes); other shapes as given."""
    if not isinstance(query, Mapping) or not isinstance(query.get("facets"), (list, tuple)):
        return query, []
    population = str(query.get("population") or "").strip().lower()
    names = (vocabulary(report).get(population) or {}).get("facets") or {}
    facets: list[Any] = []
    notes: list[str] = []
    for part in query["facets"]:
        if not isinstance(part, str) or "=" in part or not part.strip():
            facets.append(part)
            continue
        token = part.strip()
        owners = [name for name, values in names.items() if token in values]
        if len(owners) == 1:
            facets.append(f"{owners[0]}={token}")
            notes.append(f"facet {token} read as {owners[0]}={token}")
        else:
            why = "names more than one facet" if owners else "is not a value in the report's vocabulary"
            notes.append(f"facet {token} dropped: it {why}")
    if facets == list(query["facets"]):
        return query, notes
    return {**query, "facets": facets}, notes


def normalise(query: Any) -> tuple[dict[str, Any] | None, str]:
    """(clean query, "") or (None, why it cannot be looked up). Shape only; no report needed."""
    if not isinstance(query, Mapping):
        return None, "the query was not an object"
    population = str(query.get("population") or "").strip().lower()
    if population not in POPULATIONS:
        return None, f"population must be one of {', '.join(POPULATIONS)}"
    pairs = _facet_pairs(query.get("facets"))
    if not pairs:
        return None, "facets must be 1 to 3 'name=value' items"
    if len(pairs) > MAX_FACETS:
        return None, f"more than {MAX_FACETS} facets: the grid stops at depth 3"
    if len({name for name, _ in pairs}) != len(pairs):
        return None, "a facet name repeats"
    clean = {
        "population": population,
        "horizon": str(query.get("horizon") or "").strip(),
        "family": str(query.get("family") or "").strip(),
        "side": str(query.get("side") or "").strip().upper(),
        "facets": [f"{name}={value}" for name, value in sorted(pairs)],
    }
    if not clean["horizon"] or not clean["family"] or clean["side"] not in ("LONG", "SHORT"):
        return None, "horizon, family and side (LONG or SHORT) are required"
    return clean, ""


def query_label(query: Mapping[str, Any]) -> str:
    facets = " + ".join(query.get("facets") or ()) or "?"
    return (f"{query.get('population', '?')} {query.get('family', '?')} {query.get('side', '?')} "
            f"{query.get('horizon', '?')}: {facets}")


def _horizon_block(population: Mapping[str, Any], horizon: str) -> tuple[str, Mapping[str, Any]] | None:
    for key, block in (population.get("horizons") or {}).items():
        if isinstance(block, Mapping) and horizon in (str(key), str(block.get("horizon_name") or "")):
            return str(key), block
    return None


def find(query: Any, report: Report | None) -> tuple[Cell | None, str]:
    """(the published cell, "") or (None, the miss reason). Never guesses a cell."""
    import setup_permutation_search as sps

    clean, why = normalise(resolve_bare_facets(query, report)[0])
    if clean is None:
        return None, f"not a valid query: {why}"
    if report is None:
        return None, "no permutation report to look in"
    vocab = vocabulary(report).get(clean["population"])
    if vocab is None:
        return None, f"not in vocabulary: population {clean['population']} is not in the report"
    population = report.body["populations"][clean["population"]]
    found = _horizon_block(population, clean["horizon"])
    if found is None:
        return None, f"not in vocabulary: horizon {clean['horizon']}"
    horizon_key, block = found
    if block.get("refused"):
        return None, f"the report refused this horizon: {block.get('refused_reason', '')}".strip()
    group = f"{clean['family']} {clean['side']}"
    family = (block.get("families") or {}).get(group)
    if not isinstance(family, Mapping):
        return None, f"not in vocabulary: family {group}"
    for part in clean["facets"]:
        name, _, value = part.partition("=")
        if name not in vocab["facets"]:
            return None, f"not in vocabulary: facet {name}"
        if value not in vocab["facets"][name]:
            return None, f"not in vocabulary: {name}={value}"
    wanted = dict(part.partition("=")[::2] for part in clean["facets"])
    for status, row in _published_cells(family):
        if {str(k): str(v) for k, v in (row.get("facets") or {}).items()} != wanted:
            continue
        sel, hold = row.get("selection") or {}, row.get("holdout") or {}
        hold_n, hold_wins = int(hold.get("n") or 0), int(hold.get("wins") or 0)
        return Cell(
            population=clean["population"], horizon=horizon_key,
            horizon_name=str(block.get("horizon_name") or horizon_key), family=clean["family"], side=clean["side"],
            facets=wanted, label=str(row.get("label") or " + ".join(clean["facets"])), status=status,
            depth=int(row.get("depth") or len(wanted)), n=int(sel.get("n") or 0), sessions=int(sel.get("sessions") or 0),
            wins=int(sel.get("wins") or 0), win_rate=sel.get("win_rate"), wilson_lb=sel.get("wilson_lb"),
            holdout_n=hold_n, holdout_wins=hold_wins, holdout_win_rate=hold.get("win_rate"),
            holdout_lb99=(round(sps.wilson_lower_bound(hold_wins, hold_n, sps.Z99), 4) if hold_n else None),
            holdout_passed=status == "key",
            holdout_reason=str((row.get("reason") if status == "rejected" else hold.get("reason")) or ""),
            grid_cells=row.get("grid_cells"), report_asof=report.asof,
        ), ""
    return None, MISS_NOT_IN_GRID


def lookup(query: Any, report: Report | None) -> Cell | None:
    """The published cell for ``query``, or None (:func:`find` says why)."""
    return find(query, report)[0]


def lookup_record(query: Any, report: Report | None) -> dict[str, Any]:
    """What a challenge row keeps: found, the cell numbers or the miss reason, and which report."""
    cell, why = find(query, report)
    return {"found": cell is not None, "cell": cell.as_dict() if cell else None, "reason": why,
            "report_asof": report.asof if report else "", "report_file": Path(report.path).name if report else ""}


def record_text(record: Mapping[str, Any]) -> str:
    cell = record.get("cell")
    if isinstance(cell, Mapping):
        return Cell(**{key: cell[key] for key in Cell.__dataclass_fields__ if key in cell}).text()
    return str(record.get("reason") or "no lookup recorded")


# ---------------------------------------------------------------- chat DB rows (read-only)
def _outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("outcome_json") or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def read_hypotheses(chat_db: Path | str) -> list[dict[str, Any]]:
    """Every ``hypothesis`` challenge, newest first; [] when the store or table is missing."""
    path = Path(chat_db)
    if not path.exists():
        return []
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True,
                                     timeout=BUSY_TIMEOUT_MS / 1000)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM challenges WHERE kind = ? ORDER BY issued_utc DESC, id DESC",
                                (KIND,)).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return []
        raise
    return [dict(row) for row in rows]


def grade_text(outcome: Mapping[str, Any]) -> str:
    result = str(outcome.get("result") or "")
    if not result:
        return "open: waiting for a newer weekly report"
    head = {"hit": "HIT: still clears the report's bar", "miss": "MISS: listed but failed the bar",
            "gone": "GONE: not in the newer report"}.get(result, result)
    later = outcome.get("graded_lookup") or {}
    return f"{head} (report {later.get('report_asof') or '?'}): {record_text(later)}"


def build(*, now: datetime | None = None, chat_db: Path | str | None = None,
          history_dir: Path | str | None = None, report_file: Path | str | None = None) -> Pack:
    """Build the pack: which report, then each hypothesis and its cell (reads only; call it on a worker)."""
    if chat_db is None:
        import project_paths

        chat_db = project_paths.MENTOR_CHAT_DB_FILE
    report = latest_report(history_dir, report_file)
    moment = now or datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    try:
        found = read_hypotheses(chat_db)
    except sqlite3.Error as exc:
        return make_pack(NAME, (), empty_text=f"the chat store could not be read ({type(exc).__name__}); unknown")
    open_rows = [row for row in found if not row.get("graded_utc")]
    report_text = (f"Newest permutation report: data date {report.asof or 'unknown'} ({Path(report.path).name})"
                   if report else "No permutation report found (history and live report both missing)")
    rows.append({"id": "hyp:report:asof", "kind": "asof", "asof": report.asof if report else "",
                 "built_utc": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
                 "text": f"{report_text}; {len(open_rows)} open, {len(found) - len(open_rows)} graded hypotheses"})
    for row in found[:MAX_SHOWN]:
        outcome = _outcome(row)
        query = outcome.get("query") or {}
        status = "graded" if row.get("graded_utc") else "open"
        rows.append({"id": str(row["id"]), "kind": "hypothesis", "status": status,
                     "issued_utc": str(row.get("issued_utc") or ""),
                     "text": f"{status}, issued {str(row.get('issued_utc') or '')[:10]}: {query_label(query)}; "
                             f"why: {str(row.get('claim') or '').strip() or 'none given'}"})
        lookup_at_issue = outcome.get("lookup") or {}
        cell_text = f"at issue (report {lookup_at_issue.get('report_asof') or '?'}): {record_text(lookup_at_issue)}"
        if row.get("graded_utc"):
            cell_text += f" | {grade_text(outcome)}"
        rows.append({"id": f"{row['id']}:cell", "kind": "cell", "found": bool(lookup_at_issue.get("found")),
                     "text": cell_text})
    if not found:
        rows.append({"id": "hyp:none", "kind": "none", "text": "No hypothesis from the night yet"})
    return make_pack(NAME, rows)


def card_markdown(pack: Pack) -> str:
    """The /hypotheses card: every row with its id, then the footer (never a rule proposal)."""
    lines = ["**Hypotheses (the night's queries into the shadow permutation grid)**", ""]
    if not pack.rows:
        lines.append(pack.empty_text or "nothing")
    for row in pack.rows:
        indent = "  - " if row.get("kind") == "cell" else "- "
        lines.append(f"{indent}{row.get('text', '')} [{row['id']}]")
    lines += ["", FOOTER]
    return "\n".join(lines)


# ---------------------------------------------------------------- grading (no model)
def grade_open(store: Any, now: datetime, *, history_dir: Path | str | None = None,
               report_file: Path | str | None = None) -> int:
    """Grade each open hypothesis against a report NEWER than the one it was looked up in. Returns rows updated."""
    report = latest_report(history_dir, report_file)
    if report is None or not report.asof:
        return 0
    moment = now if now.tzinfo else now.astimezone()
    updated = 0
    for row in store.challenges(kind=KIND, open_only=True):
        outcome = _outcome(row)
        issued_asof = str((outcome.get("lookup") or {}).get("report_asof") or outcome.get("report_asof") or "")
        if issued_asof and report.asof <= issued_asof:
            continue
        later = lookup_record(outcome.get("query") or {}, report)
        cell = later.get("cell") or {}
        new = {**outcome, "graded_lookup": later, "status": "graded"}
        if cell.get("status") == "key":
            new.update(result="hit", hit=True)
        elif cell:
            new.update(result="miss", hit=False)
        else:
            new["result"] = "gone"
            new.pop("hit", None)
        graded = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
        if store.update_challenge(row["id"], outcome=new, graded_utc=graded):
            updated += 1
    return updated


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 29, 23, 0, tzinfo=timezone.utc)


def fixture_report(asof: str = "2026-09-26", *, key_passes: bool = True) -> dict[str, Any]:
    """A small report: swing 5-session avwap_breakout LONG with one key and one listed rejected cell."""
    key_cell = {
        "rank": 1, "depth": 2, "facets": {"sma100_support": "held", "spy_trend": "up"},
        "label": "sma100_support=held + spy_trend=up",
        "selection": {"n": 64, "sessions": 22, "wins": 41, "win_rate": 0.6406, "wilson_lb": 0.5184, "mean_r": 0.4},
        "lift_pp": 9.1, "grid_cells": 14,
        "holdout": {"n": 18, "sessions": 9, "wins": 13, "win_rate": 0.7222, "wilson_lb": 0.4913, "mean_r": 0.5,
                    "passed": True, "reason": "passed"},
    }
    rejected = {
        "label": "rs_vs_spy=weak", "depth": 1, "facets": {"rs_vs_spy": "weak"},
        "selection": {"n": 40, "sessions": 15, "wins": 22, "win_rate": 0.55, "wilson_lb": 0.3982, "mean_r": 0.1},
        "holdout": {"n": 12, "sessions": 7, "wins": 5, "win_rate": 0.4167, "wilson_lb": 0.1934, "mean_r": -0.1},
        "reason": "selection: lower bound does not beat the baseline",
    }
    family = {"family": "avwap_breakout", "side": "LONG", "horizon_name": "5d", "verdict": "key_found",
              "baseline": {"n": 300, "sessions": 60, "wins": 165, "win_rate": 0.55, "wilson_lb": 0.49},
              "holdout_baseline": {"n": 60, "sessions": 20, "wins": 31, "win_rate": 0.5167, "wilson_lb": 0.39},
              "keys": [key_cell] if key_passes else [], "top_rejected": [rejected] if key_passes else
              [rejected, {**key_cell, "holdout": {k: v for k, v in key_cell["holdout"].items()
                                                  if k not in ("passed", "reason")},
                          "reason": "hold-out: does not beat the grid median (k > 10)"}]}
    return {
        "schema": REPORT_SCHEMA, "data_date": asof, "generated_at": f"{asof}T20:00:00+00:00",
        "floors": {"min_n": 30, "min_sessions": 10, "k_rule": 10, "max_depth": 3},
        "populations": {
            "swing": {"horizons": {"5": {"horizon_name": "5d", "holdout_window": ["2026-08-28", asof],
                                         "families": {"avwap_breakout LONG": family}, "study_families": {}}}},
            "m5": {"horizons": {"0": {"horizon_name": "held30", "holdout_window": ["2026-08-28", asof],
                                      "refused": True, "refused_reason": "m5 held30: too few sessions",
                                      "families": {}, "study_families": {}}}},
        },
    }


FIXTURE_QUERY = {"population": "swing", "horizon": "5", "family": "avwap_breakout", "side": "LONG",
                 "facets": ["sma100_support=held", "spy_trend=up"]}


def write_fixture_world(root: Path) -> dict[str, Path]:
    """A history dir with one report and a chat DB with one open and one graded hypothesis."""
    from mentor_app.store import MentorChatStore

    root = Path(root)
    history = root / "permutation_report_history"
    history.mkdir(parents=True, exist_ok=True)
    (history / "2026-09-26.json").write_text(json.dumps(fixture_report()), encoding="utf-8")
    chat = root / "mentor_chat.sqlite3"
    store = MentorChatStore(chat)
    report = latest_report(history, root / "missing.json")
    store.add_challenge("hyp:2026-09-28:1", kind=KIND, symbol="", claim="SMA100 holds in an up tape",
                        evidence_ids=["fact:tools"], issued_utc="2026-09-29T05:10:00+00:00",
                        outcome={"status": "open", "query": FIXTURE_QUERY, "lookup": lookup_record(FIXTURE_QUERY, report)})
    missing = {**FIXTURE_QUERY, "facets": ["made_up=yes"]}
    store.add_challenge("hyp:2026-09-21:1", kind=KIND, symbol="", claim="a guess", evidence_ids=["fact:turns"],
                        issued_utc="2026-09-22T05:10:00+00:00",
                        outcome={"status": "open", "query": missing, "lookup": lookup_record(missing, report)})
    store.update_challenge("hyp:2026-09-21:1", outcome={
        "status": "graded", "query": missing, "result": "gone", "lookup": lookup_record(missing, report),
        "graded_lookup": lookup_record(missing, report)}, graded_utc="2026-09-27T06:00:00+00:00")
    return {"history": history, "chat": chat, "report_file": root / "missing.json"}


def fixture() -> Pack:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        world = write_fixture_world(Path(tmp))
        return build(now=FIXTURE_NOW, chat_db=world["chat"], history_dir=world["history"],
                     report_file=world["report_file"])


__all__: Sequence[str] = ("Cell", "Report", "find", "lookup", "lookup_record", "latest_report", "vocabulary",
                          "build", "grade_open", "card_markdown")
