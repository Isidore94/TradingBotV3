"""Append-only log of the trader's pick verdicts, built for AI review.

Every ★ like (with its origin: which alert timeframe or screen it came from),
every ✕ dislike (with the trader's typed reason), and every unfavorite is one
JSON object per line in ``pick_feedback.jsonl`` (shared home, syncs across
machines). Hand the file to an AI with a prompt like "review my dislikes and
suggest scan/scoring changes" - each row carries enough context (origin,
category, raw alert text / setup row summary) to reason about.

Row shape:
    {"ts": "...", "trade_date": "YYYY-MM-DD", "symbol": "NVDA", "side": "LONG",
     "verdict": "like" | "dislike" | "unfavorite" | "not_today",
     "category": "swing" | "m5" | "",
     "origin": "h1" | "d1" | "m5" | "setups" | "manual" | "",
     "reason": "<why the trader disliked it>",
     "context": "<alert text or setup-row summary>"}

The like origins also feed the human-focus tracker: the daily snapshot tags
each pick's source as e.g. ``focus_swing_h1`` so H1-alert swing picks grade as
their own cohort next to D1-sourced ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from project_paths import PICK_FEEDBACK_FILE
from project_paths import ALERT_REVIEW_EVENTS_FILE, TRADER_ANNOTATIONS_FILE


# "not_today" (packet R2) is narrower than "dislike": the trader is throwing
# back ONE auto-adopted M5 pick for ONE session, not saying the name is bad.
# Keeping them distinct matters to the review-learning scoreboard - counting a
# same-day pass as a dislike would teach it the wrong lesson about the name.
PICK_VERDICTS = ("like", "dislike", "unfavorite", "not_today")
# "chart_review" is the Chart Review workspace's capture rail; "auto_pick" is a
# machine-staged pick the trader ruled on. Like every other origin they are
# descriptive only - `record_pick_feedback` accepts any string - but the
# human-focus snapshot turns them into cohort suffixes such as
# `focus_swing_chart_review`, so the list documents what those names mean.
# "vetted" is the trader's own end-of-day swing list (the strip under the M5
# alert bar): distinct from "manual" on purpose, so those picks grade as their
# own `human_focus_swing_vetted` cohort rather than mixing with every other
# hand-typed swing name.
PICK_ORIGINS = ("h1", "d1", "m5", "setups", "manual", "chart_review", "auto_pick", "vetted")

_REVIEWED_TODAY_CACHE: dict[tuple, "_DayLedgerRead"] = {}
_PICK_DECISIONS = {"like", "dislike", "unfavorite", "not_today"}
_REVIEW_EVENT_DECISIONS = {
    "favorite",
    "dislike",
    "remove_today",
    "add_focus",
    "toggle_d1_focus",
    "toggle_m5_focus",
}
_ANNOTATION_DECISIONS = {"veto", "like_claim", "note"}

# WS-SX. `decisions_today` answers a WIDER question than `reviewed_symbols_today`
# and the difference is deliberate. "Reviewed today" means the trader touched the
# name on a review surface; a day-trade PASS and a click away from an M5 alert
# were never in its filters and adding them there would move a shipped badge, so
# the two answers are built from ONE parse and kept apart at the end of it.
#
# Which rows are a LIKE (the ★) and which are a REJECT (the ✕):
LIKE_KINDS = ("quick", "claimed", "like")
REJECT_KINDS = ("veto", "dislike", "not_today", "pass", "m5_click_away", "remove_today")
#: `unfavorite` is in NEITHER map. Taking a name out of Focus is not a verdict on
#: it (CLAUDE.md, P5: "`unfavorite` is never graded").
_LIKE_PICK_VERDICTS = {"like"}
_REJECT_PICK_VERDICTS = {"dislike", "not_today"}
_REJECT_REVIEW_ACTIONS = {"dislike", "remove_today"}
#: A day-trade pass is an annotation event type of its own (`pass_reasons`
#: family), outside `_ANNOTATION_DECISIONS`.
_PASS_ANNOTATION = "pass"
#: A click away from an M5 alert IS a pass (trader, 2026-09-01). It has no verb
#: of its own: it is an `action: "skip"` review event whose detail reason is this
#: string, written by `alert_center_panel._render_current_review`'s skip branch.
#: Never rename it - `review_learning` keys on it too.
M5_CLICK_AWAY_REASON = "clicked_away_from_m5_alert"


@dataclass(frozen=True)
class SymbolDecisions:
    """One symbol's decisions on one trade date. Two independent facts.

    A name can be liked AND vetoed on the same day; neither mark cancels the
    other (lead's answer to WS-SX, 2026-09-12).
    """

    symbol: str = ""
    liked: tuple[tuple[str, str], ...] = ()
    rejected: tuple[tuple[str, str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.liked or self.rejected)


_NO_DECISIONS = SymbolDecisions()


@dataclass(frozen=True)
class DayDecisions:
    """Today's likes and rejects, by symbol, each entry `(kind, timestamp)`.

    Presentation only: nothing here hides, mutes, re-orders or scores a row.
    """

    trade_date: str = ""
    liked: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    rejected: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    # Memoized per-symbol views: the setups delegate asks once per painted cell,
    # so the answer is built once per symbol and handed back, never rebuilt.
    _views: dict[str, SymbolDecisions] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def for_symbol(self, symbol: object) -> SymbolDecisions:
        """The per-symbol view. Never None - an undecided name is an empty one."""
        key = str(symbol or "").strip().upper()
        if not key:
            return _NO_DECISIONS
        view = self._views.get(key)
        if view is None:
            liked = tuple(self.liked.get(key, ()))
            rejected = tuple(self.rejected.get(key, ()))
            view = (
                SymbolDecisions(key, liked, rejected) if (liked or rejected) else _NO_DECISIONS
            )
            self._views[key] = view
        return view

    def symbols(self) -> set[str]:
        return set(self.liked) | set(self.rejected)


@dataclass(frozen=True)
class _DayLedgerRead:
    """What ONE parse of the three ledgers answers: both questions."""

    reviewed: frozenset[str]
    decisions: DayDecisions


def _trade_date_text() -> str:
    try:
        from market_session import get_market_session_window

        return get_market_session_window().market_date.isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def record_pick_feedback(
    *,
    symbol: object,
    side: object = "",
    verdict: str,
    category: str = "",
    origin: str = "",
    reason: str = "",
    context: str = "",
    now: datetime | None = None,
    path: Path = PICK_FEEDBACK_FILE,
) -> dict[str, Any] | None:
    """Append one verdict row. Returns the row, or None for a blank symbol."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    side_text = str(side or "").strip().upper()
    row = {
        "ts": (now or datetime.now()).isoformat(timespec="seconds"),
        "trade_date": _trade_date_text(),
        "symbol": sym,
        "side": "SHORT" if side_text.startswith("SHORT") else "LONG" if side_text.startswith("LONG") else side_text,
        "verdict": str(verdict or "").strip().lower(),
        "category": str(category or "").strip().lower(),
        "origin": str(origin or "").strip().lower(),
        "reason": str(reason or "").strip(),
        "context": str(context or "").strip(),
    }
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        return None  # another process can briefly lock files; best-effort
    return row


def load_pick_feedback(path: Path = PICK_FEEDBACK_FILE) -> list[dict[str, Any]]:
    """All feedback rows in file order (oldest first). Bad lines are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except OSError:
        return []
    return rows


def _file_signature(path: Path) -> tuple[int, int]:
    try:
        stat = Path(path).stat()
        return int(stat.st_mtime_ns), int(stat.st_size)
    except OSError:
        return 0, 0


def clear_reviewed_today_cache() -> None:
    _REVIEWED_TODAY_CACHE.clear()


def reviewed_symbols_today(
    *,
    market_date: str | None = None,
    pick_feedback_path: Path = PICK_FEEDBACK_FILE,
    review_events_path: Path = ALERT_REVIEW_EVENTS_FILE,
    annotations_path: Path = TRADER_ANNOTATIONS_FILE,
) -> set[str]:
    """Union today's explicit decisions across the three review ledgers.

    Presentation only. ``shown`` impressions and hypothesis stops are excluded:
    this marker means the trader made a decision (star/x, veto, like, or note),
    not merely that a row appeared on screen.

    Narrower than :func:`decisions_today` on purpose - see `_PASS_ANNOTATION`.
    """
    return set(
        _read_day_ledgers(
            market_date=market_date,
            pick_feedback_path=pick_feedback_path,
            review_events_path=review_events_path,
            annotations_path=annotations_path,
        ).reviewed
    )


def decisions_today(
    *,
    market_date: str | None = None,
    pick_feedback_path: Path = PICK_FEEDBACK_FILE,
    review_events_path: Path = ALERT_REVIEW_EVENTS_FILE,
    annotations_path: Path = TRADER_ANNOTATIONS_FILE,
) -> DayDecisions:
    """Today's likes and rejects by symbol, with the kind and time of each.

    The same cached, mtime-keyed read :func:`reviewed_symbols_today` uses: one
    parse of the three ledgers answers both questions. Presentation only.
    """
    return _read_day_ledgers(
        market_date=market_date,
        pick_feedback_path=pick_feedback_path,
        review_events_path=review_events_path,
        annotations_path=annotations_path,
    ).decisions


def _add_decision(
    mapping: dict[str, list[tuple[str, str]]], symbol: str, kind: str, stamp: object
) -> None:
    entry = (kind, str(stamp or ""))
    entries = mapping.setdefault(symbol, [])
    if entry not in entries:
        entries.append(entry)


def _read_day_ledgers(
    *,
    market_date: str | None = None,
    pick_feedback_path: Path = PICK_FEEDBACK_FILE,
    review_events_path: Path = ALERT_REVIEW_EVENTS_FILE,
    annotations_path: Path = TRADER_ANNOTATIONS_FILE,
) -> _DayLedgerRead:
    """ONE parse of the three review ledgers, memoized on (paths, mtime, size).

    Both the "Reviewed today" badge and the setups table's ★/✕ are answered from
    the result, because the delegate asks per painted cell and the 2026-08-31
    stall log named that viewport pass as its hottest stack.
    """
    target_date = str(market_date or _trade_date_text())
    pick_path = Path(pick_feedback_path)
    events_path = Path(review_events_path)
    annotation_path = Path(annotations_path)
    if events_path == Path(ALERT_REVIEW_EVENTS_FILE):
        try:
            from review_events import review_event_store_mtime

            events_signature = (int((review_event_store_mtime(events_path) or 0) * 1e9), 0)
        except Exception:
            events_signature = _file_signature(events_path)
    else:
        events_signature = _file_signature(events_path)
    key = (
        target_date,
        str(pick_path),
        _file_signature(pick_path),
        str(events_path),
        events_signature,
        str(annotation_path),
        _file_signature(annotation_path),
    )
    cached = _REVIEWED_TODAY_CACHE.get(key)
    if cached is not None:
        return cached

    symbols: set[str] = set()
    liked: dict[str, list[tuple[str, str]]] = {}
    rejected: dict[str, list[tuple[str, str]]] = {}

    for row in load_pick_feedback(pick_path):
        if str(row.get("trade_date") or "") != target_date:
            continue
        verdict = str(row.get("verdict") or "").strip().lower()
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        if verdict in _PICK_DECISIONS:
            symbols.add(symbol)
        if verdict in _LIKE_PICK_VERDICTS:
            _add_decision(liked, symbol, "like", row.get("ts"))
        elif verdict in _REJECT_PICK_VERDICTS:
            _add_decision(rejected, symbol, verdict, row.get("ts"))

    try:
        from review_events import load_review_events

        review_rows = load_review_events(
            events_path,
            include_shards=events_path == Path(ALERT_REVIEW_EVENTS_FILE),
        )
    except Exception:
        review_rows = []
    for row in review_rows:
        if str(row.get("trade_date") or "") != target_date:
            continue
        action = str(row.get("action") or "").strip().lower()
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        if action in _REVIEW_EVENT_DECISIONS:
            symbols.add(symbol)
        if action in _REJECT_REVIEW_ACTIONS:
            _add_decision(rejected, symbol, action, row.get("ts"))
        elif action == "skip":
            detail = row.get("detail")
            reason = ""
            if isinstance(detail, dict):
                reason = str(detail.get("reason") or "").strip().lower()
            if reason == M5_CLICK_AWAY_REASON:
                _add_decision(rejected, symbol, "m5_click_away", row.get("ts"))

    try:
        from ui.annotations.store import like_mode_of, load_annotations

        annotation_rows = load_annotations(
            annotation_path,
            session_date=target_date,
            event_types=tuple(sorted(_ANNOTATION_DECISIONS | {_PASS_ANNOTATION})),
        )
    except Exception:
        annotation_rows = []
        like_mode_of = None  # type: ignore[assignment]
    for row in annotation_rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        kind = str(row.get("event_type") or "").strip().lower()
        if kind in _ANNOTATION_DECISIONS:
            symbols.add(symbol)
        if kind == "veto":
            _add_decision(rejected, symbol, "veto", row.get("created_at"))
        elif kind == _PASS_ANNOTATION:
            _add_decision(rejected, symbol, "pass", row.get("created_at"))
        elif kind == "like_claim":
            mode = like_mode_of(row) if like_mode_of is not None else "claimed"
            _add_decision(liked, symbol, str(mode), row.get("created_at"))

    read = _DayLedgerRead(
        reviewed=frozenset(symbols),
        decisions=DayDecisions(
            trade_date=target_date,
            liked={symbol: tuple(entries) for symbol, entries in liked.items()},
            rejected={symbol: tuple(entries) for symbol, entries in rejected.items()},
        ),
    )
    # Bound the cache naturally to recent signatures/days.
    if len(_REVIEWED_TODAY_CACHE) >= 16:
        _REVIEWED_TODAY_CACHE.clear()
    _REVIEWED_TODAY_CACHE[key] = read
    return read


def latest_like_origins(
    rows: list[dict[str, Any]] | None = None,
    *,
    path: Path = PICK_FEEDBACK_FILE,
) -> dict[tuple[str, str, str], str]:
    """{(SYMBOL, LONG/SHORT, category): origin} from the most recent like of each pick.

    Used by the daily human-focus snapshot to tag each pick's cohort source
    (e.g. focus_swing_h1) so origins grade separately in the tracker.
    """
    origins: dict[tuple[str, str, str], str] = {}
    for row in rows if rows is not None else load_pick_feedback(path):
        if str(row.get("verdict") or "").lower() != "like":
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        category = str(row.get("category") or "").strip().lower()
        origin = str(row.get("origin") or "").strip().lower()
        if symbol and side in {"LONG", "SHORT"} and origin:
            origins[(symbol, side, category)] = origin
    return origins
