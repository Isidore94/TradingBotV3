"""What I said, what I did, what happened — P6.

Four stores already answer one third of this each. The annotation log knows what
the trader SAID about a name; the journal knows what they TRADED; the cohort
rollups know what the name then DID on paper. Nothing put the three on one row,
so the question the whole capture programme exists to answer —

    *of the setups I liked, which did I actually take, and how did the ones I
    skipped do?*

— could only be answered by opening three files and joining them by eye.

This is that join, and it is deliberately a REPORT rather than a link.

**Every row renders its match confidence, or says "no match".** Nothing here
mints an identifier: `plan.md` P5.3/P5.4 own the canonical opportunity id, and a
second one invented in a nightly report would compete with it while being
weaker. What a row carries instead is a `trade_id` when a trade was found, a
stated confidence, and `match_basis` naming what the match rested on. A reader
can always see how firm the link is.

**Read-only, and it never writes into the journal.** `trade_annotations` are
trader-owned; this module reads `list_trades` and writes one CSV of its own.
Following `journal_walkaway`'s pattern: pure computation over stores something
else fills, publishing a file nothing scores from.

**A missing half is stated, never filled.** A statement with no trade is the
most interesting row in the file — it is the skip — so it is written with an
empty `trade_id` and an explicit `traded` of "no". A trade whose paper grade has
not matured yet carries a blank forward return, never a zero.
"""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from project_paths import OUTPUT_DIR

_log = logging.getLogger(__name__)

SCHEMA = "preference_trade_outcomes_v1"

#: Where the report lands. Beside the other read-only journal reports.
REPORT_FILE = OUTPUT_DIR / "preference_trade_outcomes.csv"

#: How many days back a nightly run considers. Long enough to cover a swing
#: idea's life, short enough that the file stays readable.
DEFAULT_WINDOW_DAYS = 45

#: How many sessions after the statement a trade may open and still count as
#: acting on it. A trade three weeks later is a different decision.
TRADE_WINDOW_DAYS = 10

COLUMNS = [
    "schema",
    "generated_at",
    "session_date",
    "symbol",
    "side",
    # WHAT YOU SAID
    "channel",
    "statement",
    "statement_detail",
    "statement_id",
    # WHAT YOU DID
    "traded",
    "trade_id",
    "trade_opened_at",
    "match_confidence",
    "match_basis",
    # WHAT HAPPENED
    "journal_r",
    "journal_net_pnl",
    "paper_forward_return_h3",
    "paper_forward_return_h5",
    "paper_cohort",
]


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("SHORT") or text in {"SELL", "S"}:
        return "SHORT"
    if text.startswith("LONG") or text in {"BUY", "B"}:
        return "LONG"
    return ""


def _canonical_r(trade) -> str:
    """`net_pnl_cad / |planned_risk|`, blank when either half is missing.

    One definition, stated in `ui/services/journal_feed.r_multiple` and repeated
    here as arithmetic rather than as an import: that module is the Qt feed and
    this is a headless nightly slot. Deliberately CAD - an R computed from a
    native P&L against a risk typed in dollars mixes currencies, which is the
    same defect the journal already fixed once.
    """
    try:
        risk = float(trade.get("planned_risk"))
        pnl = float(trade.get("net_pnl_cad"))
    except (TypeError, ValueError):
        return ""
    if abs(risk) < 1e-9:
        return ""
    return f"{pnl / abs(risk):.4f}"


def _float_or_blank(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if number != number else f"{number:.4f}"


# ---------------------------------------------------------------------------
# what you said
# ---------------------------------------------------------------------------
def collect_statements(
    *,
    since: date,
    until: date,
    annotations_path: Path | None = None,
    feedback_path: Path | None = None,
    favorites_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Every statement the trader made about a name in the window.

    Four channels, each read through the module that owns its store rather than
    a second parser here:

    * ``like_claim`` and ``pass`` from the annotation log;
    * ``swing_favorite`` from the swing favorites store - the trader's own
      end-of-day list;
    * ``like`` from `pick_feedback`.

    A statement with no side is KEPT and marked, unlike the cohorts which
    refuse to grade one: this report is about what was said and whether it was
    acted on, and a sideless statement was still made.
    """
    statements: list[dict[str, Any]] = []
    statements.extend(_annotation_statements(since, until, annotations_path))
    statements.extend(_favorite_statements(since, until, favorites_path))
    statements.extend(_feedback_statements(since, until, feedback_path))
    statements.sort(key=lambda row: (row["session_date"], row["symbol"], row["channel"]))
    return statements


def _annotation_statements(since: date, until: date, path: Path | None) -> list[dict[str, Any]]:
    try:
        from project_paths import TRADER_ANNOTATIONS_FILE
        from ui.annotations.store import EVENT_LIKE_CLAIM, EVENT_PASS, load_annotations

        rows = load_annotations(
            Path(path or TRADER_ANNOTATIONS_FILE),
            event_types=(EVENT_LIKE_CLAIM, EVENT_PASS),
        )
    except Exception as exc:  # noqa: BLE001 - a channel is never worth the report
        _log.debug("Annotation statements unavailable: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        session = _as_date(row.get("session_date"))
        if session is None or not (since <= session <= until):
            continue
        kind = str(row.get("event_type") or "")
        if kind == "like_claim":
            statement = "liked"
            detail = str(row.get("claimed_setup_id") or "")
        else:
            statement = "passed"
            codes = [str(code or "").strip() for code in (row.get("reason_codes") or []) if code]
            detail = ", ".join(codes)
        out.append(
            {
                "session_date": session,
                "symbol": _symbol(row.get("symbol")),
                "side": _side(row.get("side")),
                "channel": f"annotation:{kind}",
                "statement": statement,
                "statement_detail": detail,
                "statement_id": str(row.get("event_id") or ""),
            }
        )
    return out


def _favorite_statements(since: date, until: date, path: Path | None) -> list[dict[str, Any]]:
    try:
        import swing_favorites

        raw = (
            swing_favorites.load_rows(path) if path else swing_favorites.load_rows()
        )
    except Exception as exc:  # noqa: BLE001
        _log.debug("Swing favorites unavailable: %s", exc)
        return []

    # Resolved PER SESSION through the store's own `favorites_for_session`, so a
    # name the trader added and then retracted is not reported as a pick they
    # never took. The append-only log keeps the retraction; the live list for
    # that session is what they actually stood behind.
    sessions = sorted(
        {
            str(row.get("session_date") or "").strip()
            for row in raw
            if str(row.get("session_date") or "").strip()
        }
    )
    out: list[dict[str, Any]] = []
    for session_text in sessions:
        session = _as_date(session_text)
        if session is None or not (since <= session <= until):
            continue
        for row in swing_favorites.favorites_for_session(session_text, rows=raw):
            out.append(
                {
                    "session_date": session,
                    "symbol": _symbol(row.get("symbol")),
                    "side": _side(row.get("side")),
                    "channel": "swing_favorite",
                    "statement": "picked",
                    "statement_detail": str(row.get("origin") or "today's swing list"),
                    "statement_id": "",
                }
            )
    return out


def _feedback_statements(since: date, until: date, path: Path | None) -> list[dict[str, Any]]:
    try:
        import json

        from project_paths import PICK_FEEDBACK_FILE

        target = Path(path or PICK_FEEDBACK_FILE)
        if not target.exists():
            return []
        raw: list[dict[str, Any]] = []
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and str(row.get("verdict") or "") == "like":
                    raw.append(row)
    except OSError as exc:
        _log.debug("Pick feedback unavailable: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    for row in raw:
        session = _as_date(row.get("trade_date"))
        if session is None or not (since <= session <= until):
            continue
        out.append(
            {
                "session_date": session,
                "symbol": _symbol(row.get("symbol")),
                "side": _side(row.get("side")),
                "channel": "pick_feedback:like",
                "statement": "liked",
                "statement_detail": str(row.get("origin") or ""),
                "statement_id": "",
            }
        )
    return out


# ---------------------------------------------------------------------------
# what you did
# ---------------------------------------------------------------------------
def match_trade(statement: Mapping[str, Any], trades: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The trade that acted on this statement, with a stated confidence.

    Never a hard link. The best available evidence is (symbol, side, a trade
    opened on or within ``TRADE_WINDOW_DAYS`` sessions after the statement), and
    that is a JUDGEMENT: the trader could have taken the name for an unrelated
    reason the same week. So the row carries what the match rested on and how
    firm it is, and a reader can discount it.

    * symbol + side + same day -> 0.9, "symbol+side+same_session"
    * symbol + side + inside the window -> 0.7, "symbol+side+window"
    * symbol only (a sideless statement, or the trader took it the other way)
      -> 0.5, "symbol+window_side_unknown"
    * nothing -> confidence 0.0 and "no match", which is the interesting row.
    """
    symbol = statement.get("symbol")
    said_on = statement.get("session_date")
    side = statement.get("side") or ""
    if not symbol or not isinstance(said_on, date):
        return {"trade": None, "confidence": 0.0, "basis": "no match"}

    window_end = said_on + timedelta(days=TRADE_WINDOW_DAYS)
    best: tuple[float, str, Mapping[str, Any]] | None = None
    for trade in trades:
        if _symbol(trade.get("symbol")) != symbol:
            continue
        opened = _as_date(trade.get("opened_at")) or _as_date(trade.get("trade_date"))
        if opened is None or not (said_on <= opened <= window_end):
            continue
        trade_side = _side(trade.get("direction"))
        if side and trade_side and trade_side == side:
            confidence, basis = (
                (0.9, "symbol+side+same_session") if opened == said_on else (0.7, "symbol+side+window")
            )
        elif side and trade_side and trade_side != side:
            # Taken the other way round. Still an action on the name, and the
            # row says so rather than claiming the statement was followed.
            confidence, basis = 0.35, "symbol+window_opposite_side"
        else:
            confidence, basis = 0.5, "symbol+window_side_unknown"
        if best is None or confidence > best[0]:
            best = (confidence, basis, trade)
    if best is None:
        return {"trade": None, "confidence": 0.0, "basis": "no match"}
    return {"trade": best[2], "confidence": best[0], "basis": best[1]}


# ---------------------------------------------------------------------------
# what happened
# ---------------------------------------------------------------------------
def load_paper_grades() -> dict[tuple[str, str, str], dict[str, Any]]:
    """(session, symbol, side) -> the cohort's forward return for that pick.

    Read from the cohort OUTCOME files that already exist - the like, pass and
    rejection trios and the focus rollup - so this report never recomputes a
    forward return. Ground rule 6: reformatted, never derived.

    Absent files are absent grades, and a row with no grade renders blank.
    """
    grades: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        import project_paths
    except Exception:  # noqa: BLE001
        return grades

    for attribute in (
        "LIKE_COHORT_OUTCOMES_FILE",
        "HUMAN_FOCUS_OUTCOMES_FILE",
        "VETO_COHORT_OUTCOMES_FILE",
    ):
        path = getattr(project_paths, attribute, None)
        if path is None or not Path(path).is_file():
            continue
        try:
            with Path(path).open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    key = (
                        str(row.get("trade_date") or "").strip(),
                        _symbol(row.get("symbol")),
                        _side(row.get("side")),
                    )
                    if not key[0] or not key[1]:
                        continue
                    grades.setdefault(
                        key,
                        {
                            "h3": row.get("h3_return"),
                            "h5": row.get("h5_return"),
                            "cohort": str(row.get("source") or ""),
                        },
                    )
        except OSError as exc:
            _log.debug("Paper grade file unreadable (%s): %s", attribute, exc)
    return grades


def build_rows(
    statements: list[dict[str, Any]],
    trades: list[Mapping[str, Any]],
    *,
    grades: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """One row per statement: what was said, whether it was taken, what it did."""
    stamp = (now or datetime.now()).isoformat(timespec="seconds")
    grades = grades if grades is not None else {}
    rows: list[dict[str, Any]] = []
    for statement in statements:
        match = match_trade(statement, trades)
        trade = match["trade"]
        session_text = statement["session_date"].isoformat()
        grade = grades.get((session_text, statement["symbol"], statement["side"])) or {}
        rows.append(
            {
                "schema": SCHEMA,
                "generated_at": stamp,
                "session_date": session_text,
                "symbol": statement["symbol"],
                "side": statement["side"],
                "channel": statement["channel"],
                "statement": statement["statement"],
                "statement_detail": statement["statement_detail"],
                "statement_id": statement["statement_id"],
                # The plainest column in the file, and the one the whole report
                # exists for.
                "traded": "yes" if trade is not None else "no",
                "trade_id": str(trade.get("trade_id") or "") if trade else "",
                "trade_opened_at": str(trade.get("opened_at") or "") if trade else "",
                "match_confidence": f"{float(match['confidence']):.2f}" if trade else "",
                "match_basis": match["basis"],
                # THE CANONICAL R (R1). `r_multiple` is a key that exists
                # nowhere in scripts/: every traded row shipped a blank R, and
                # the test that "covered" it invented the key on its fixture.
                # The journal's one definition is `net_pnl_cad / |planned_risk|`
                # (`ui/services/journal_feed.r_multiple`), computed here from
                # the same two columns rather than imported, because this module
                # is a headless nightly and that one lives behind the Qt feed.
                # Blank when the trader never typed a risk - which is most rows,
                # and is a real answer.
                "journal_r": _canonical_r(trade) if trade else "",
                # CAD too. `net_pnl` is the trade's NATIVE currency, so a column
                # holding both was adding USD to CAD one row apart.
                "journal_net_pnl": _float_or_blank(trade.get("net_pnl_cad")) if trade else "",
                "paper_forward_return_h3": _float_or_blank(grade.get("h3")),
                "paper_forward_return_h5": _float_or_blank(grade.get("h5")),
                "paper_cohort": str(grade.get("cohort") or ""),
            }
        )
    return rows


def write_rows(rows: list[dict[str, Any]], path: Path | None = None) -> bool:
    """Publish the report atomically. Returns whether it was written."""
    import os

    target = Path(path or REPORT_FILE)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in COLUMNS})
        os.replace(tmp, target)
        return True
    except OSError as exc:
        _log.debug("Preference/trade report could not be written: %s", exc)
        return False


def run_preference_trade_outcomes(
    *,
    now: datetime | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    report_path: Path | None = None,
    trades: list[Mapping[str, Any]] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """The nightly slot. Deterministic, read-only, and no model is called."""
    moment = now or datetime.now()
    until = moment.date()
    since = until - timedelta(days=max(1, int(window_days)))

    if trades is None:
        try:
            from journal_store import JournalStore

            trades = list(JournalStore().list_trades())
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "skipped",
                "reason": f"journal unavailable: {exc}",
                "rows": 0,
            }

    statements = collect_statements(since=since, until=until)
    if not statements:
        return {
            "status": "skipped",
            "reason": (
                f"no statements recorded between {since} and {until} - an absent "
                "record, not a window without opinions"
            ),
            "rows": 0,
        }

    rows = build_rows(statements, trades, grades=load_paper_grades(), now=moment)
    written = write_rows(rows, report_path)
    taken = sum(1 for row in rows if row["traded"] == "yes")
    return {
        "status": "ok" if written else "degraded",
        "rows": len(rows),
        "taken": taken,
        "not_taken": len(rows) - taken,
        "reason": (
            f"{len(rows)} statement(s) between {since} and {until}; {taken} were traded, "
            f"{len(rows) - taken} were not"
            + ("" if written else "; the report could not be written")
        ),
    }


__all__ = [
    "COLUMNS",
    "DEFAULT_WINDOW_DAYS",
    "REPORT_FILE",
    "SCHEMA",
    "TRADE_WINDOW_DAYS",
    "build_rows",
    "collect_statements",
    "load_paper_grades",
    "match_trade",
    "run_preference_trade_outcomes",
    "write_rows",
]
