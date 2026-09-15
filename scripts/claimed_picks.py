"""The trader's CLAIMED D1 picks - an append-only evidence store.

Trader, 2026-09-14 (packet D1C-A):

    *"The left side of the Trading Desk is for M5 trades. The right side is for
    D1 trades. When I like and claim a D1 setup, it becomes a ranked pick I can
    follow in Master AVWAP Setups. I should not have to keep reviewing the same
    D1 chart."*

That narrowly supersedes the old "a claimed like places nothing" rule, and only
for the D1 horizon. A claim is still not a placement verb anywhere else: this
module writes no Focus entry, injects nothing into a watchlist, parks no symbol
and carries no suppression field. It records what the trader said about a chart
and two surfaces read it back - the Master AVWAP setups table (one row) and the
chart-review queue gate (this D1 thesis has been answered).

The storage rules are ``swing_favorites``' rules, for the same reasons:

* **Append-only, never rewritten.** A claim is one row; a drop and a machine
  expiry are rows that FOLLOW it. The file is the record of what happened, in
  the order it happened, so "claimed AMD then dropped it" and "never claimed
  AMD" stay different facts.
* **Every row carries its time twice** - ``claim_at`` is tz-aware machine-local
  (machine order), ``claim_at_utc`` travels beside it, and ``session_date`` is
  the market-local session (trading order). Neither clock alone answers "which
  session was this?" across an evening write.
* **The live list is derived, not stored.** :func:`active_claims` replays the
  file in order; the last action per ``(symbol, side, claimed_setup_id)`` wins.
* **A failed append never raises.** An evidence store is never allowed to cost
  the thing it records - but the CALLER is told (``record_claim`` returns
  ``None``), because this one decides whether a chart is retired, and a chart
  retired for a pick that does not exist is the trader losing both.

Expiry and removal (the lifecycle is REUSED, not re-invented)
-------------------------------------------------------------
A claim is ACTIVE from the moment it is made until one of exactly two things
happens:

1. **The trader drops it** - "Drop my claim" on the setups row, or any other
   caller of :func:`record_drop`. A veto, a dislike, a day-trade pass or a "Not
   today" never retracts a claim: verdicts stay separate (P5), and only the
   verb that made the claim can unmake it.
2. **It fades.** ``focus_picks.FADE_TRADING_DAYS`` (10) TRADING days after its
   session, counted by ``market_calendar.trading_days_between`` - the same
   clock and the same constant as a quiet Focus pick's fade, referenced by name
   so the two can never drift apart. :func:`sweep_expired` appends one
   ``expire`` row per faded claim and is idempotent.

**A calendar that cannot answer expires nothing.** ``trading_days_between``
raises outside its validated range, and every caller of it deletes something
when it answers; uncertainty never deletes (plan.md sec 5), so a raising
calendar leaves every claim active and the sweep appends no row.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from project_paths import CLAIMED_PICKS_FILE

#: Schema by NAME, never by number - a changed meaning is a new name.
SCHEMA_CLAIMED_PICK = "claimed_pick_v1"

ACTION_CLAIM = "claim"
ACTION_DROP = "drop"
ACTION_EXPIRE = "expire"
ACTIONS = (ACTION_CLAIM, ACTION_DROP, ACTION_EXPIRE)

#: The only horizon this store ever writes. The M5 side of the desk keeps the
#: route it has always had (a claimed like on an intraday chart advances and
#: places nothing), so a row here is a D1 thesis by construction.
HORIZON_D1 = "d1"
HORIZON_M5 = "m5"

#: The exact field set of a row. Asserted by the tests, and the reason is
#: substantive: a suppression field must never appear here (plan.md sec 5).
ROW_FIELDS = (
    "schema",
    "action",
    "symbol",
    "side",
    "horizon",
    "claimed_setup_id",
    "claim_at",
    "claim_at_utc",
    "session_date",
    "source",
    "annotation_ref",
    "known_at_claim",
    "note",
)


def fade_trading_days() -> int:
    """``focus_picks.FADE_TRADING_DAYS``, read by name rather than copied."""
    from focus_picks import FADE_TRADING_DAYS

    return int(FADE_TRADING_DAYS)


def normalize_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def normalize_side(side: object) -> str:
    """LONG / SHORT / '' - the spelling the setups table and the alerts use."""
    text = str(side or "").strip().upper()
    if text.startswith("LONG") or text in {"L", "BUY"}:
        return "LONG"
    if text.startswith("SHORT") or text in {"S", "SELL"}:
        return "SHORT"
    return ""


def current_session_date(now: datetime | None = None) -> str:
    """Today's market-local session date, as ``YYYY-MM-DD``."""
    try:
        from market_session import get_market_session_window

        return get_market_session_window(now).market_date.isoformat()
    except Exception:  # noqa: BLE001 - a clock never costs the claim
        return (now or datetime.now()).date().isoformat()


def _event_time(now: datetime | None = None) -> datetime:
    moment = now or datetime.now()
    if moment.tzinfo is None:
        # astimezone() on a naive datetime ATTACHES the machine's zone; it never
        # strips one. A row without an offset cannot be ordered against one
        # written on the other side of a DST change.
        moment = moment.astimezone()
    return moment


def build_claim_row(
    *,
    symbol: object,
    side: object,
    horizon: str = HORIZON_D1,
    claimed_setup_id: object = "",
    source: object = "",
    annotation_ref: object = "",
    known_at_claim: Mapping[str, Any] | None = None,
    note: object = "",
    action: str = ACTION_CLAIM,
    session_date: str = "",
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """One store row, or None when the symbol, side or action is unusable."""
    sym = normalize_symbol(symbol)
    side_text = normalize_side(side)
    action_text = str(action or "").strip().lower()
    if not sym or not side_text or action_text not in ACTIONS:
        return None
    moment = _event_time(now)
    from datetime import timezone

    known = dict(known_at_claim or {})
    return {
        "schema": SCHEMA_CLAIMED_PICK,
        "action": action_text,
        "symbol": sym,
        "side": side_text,
        "horizon": str(horizon or "").strip().lower(),
        "claimed_setup_id": str(claimed_setup_id or "").strip(),
        "claim_at": moment.isoformat(timespec="seconds"),
        "claim_at_utc": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "session_date": str(session_date or "").strip() or current_session_date(now),
        "source": str(source or "").strip(),
        "annotation_ref": str(annotation_ref or "").strip(),
        "known_at_claim": known,
        "note": str(note or "").strip(),
    }


def append_row(row: Mapping[str, Any], path: Path = CLAIMED_PICKS_FILE) -> bool:
    """Append one row. Returns False when the write failed - never raises.

    "Never raises" means never, not "never on an OSError". `known_at_claim`
    carries whatever the alert's PAYLOAD held, so a value `json.dumps` refuses
    is a live possibility rather than a hypothetical - and it must cost the
    row, never the click that made it. The row is serialised INSIDE the try for
    exactly that reason, and a refused row leaves nothing half-written behind
    because nothing is opened until the text exists.
    """
    try:
        text = json.dumps(dict(row), sort_keys=True) + "\n"
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(text)
    except (OSError, TypeError, ValueError):
        return False
    return True


def load_rows(path: Path = CLAIMED_PICKS_FILE) -> list[dict[str, Any]]:
    """Every row in file order (oldest first). A torn line is skipped, not raised."""
    target = Path(path)
    if not target.exists():
        return []
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def claim_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """The identity: symbol, side and the setup the trader NAMED.

    The same symbol claimed LONG and SHORT, or under two setups, is two picks -
    they are two theses and they are graded apart.
    """
    return (
        normalize_symbol(row.get("symbol")),
        normalize_side(row.get("side")),
        str(row.get("claimed_setup_id") or "").strip(),
    )


def _faded(session_date: str, as_of: date) -> bool:
    """True only when the calendar SAYS so. A failure is 'not faded'."""
    text = str(session_date or "").strip()
    if not text:
        return False
    try:
        claimed_on = date.fromisoformat(text[:10])
    except ValueError:
        return False
    try:
        import market_calendar

        sessions = market_calendar.trading_days_between(claimed_on, as_of)
    except Exception:  # noqa: BLE001 - uncertainty never deletes (plan.md sec 5)
        return False
    return int(sessions) > fade_trading_days()


def _rows_of(rows_or_path) -> list[dict[str, Any]]:
    if isinstance(rows_or_path, (str, Path)):
        return load_rows(Path(rows_or_path))
    return [row for row in (rows_or_path or []) if isinstance(row, Mapping)]


def standing_claims(rows_or_path=CLAIMED_PICKS_FILE) -> list[dict[str, Any]]:
    """Every claim the trader has not ended, the FADE aside.

    The replay half of :func:`active_claims`, separated so the sweep can ask
    "what is still standing?" without the fade answering first - a sweep that
    read the faded-out list would have nothing left to expire.
    """
    live: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in _rows_of(rows_or_path):
        key = claim_key(row)
        if not key[0] or not key[1]:
            continue
        action = str(row.get("action") or "").strip().lower()
        if action in (ACTION_DROP, ACTION_EXPIRE):
            live.pop(key, None)
        elif action == ACTION_CLAIM:
            live.pop(key, None)
            live[key] = dict(row)
    return list(live.values())


def active_claims(
    rows_or_path=CLAIMED_PICKS_FILE, *, as_of: date | None = None
) -> list[dict[str, Any]]:
    """The live claims, replayed in file order; the last action per key wins.

    A ``drop`` or an ``expire`` ends a claim. A claim whose session is more than
    ``focus_picks.FADE_TRADING_DAYS`` trading days behind ``as_of`` has faded
    and is not returned - but a calendar that cannot answer keeps it.
    """
    moment = as_of or date.today()
    return [
        row
        for row in standing_claims(rows_or_path)
        if not _faded(str(row.get("session_date") or ""), moment)
    ]


def active_keys(
    rows_or_path=CLAIMED_PICKS_FILE, *, as_of: date | None = None
) -> set[tuple[str, str]]:
    """``(symbol, side)`` for every active claim - what the queue gate asks.

    The SETUP is deliberately dropped here. The gate's question is "has the
    trader already answered this D1 chart for this side?", and a second flag on
    the same name and side is the same chart. The claimed LONG says nothing
    about a SHORT thesis, which is why the side stays.
    """
    return {
        (normalize_symbol(row.get("symbol")), normalize_side(row.get("side")))
        for row in active_claims(rows_or_path, as_of=as_of)
    }


def record_claim(
    *,
    symbol: object,
    side: object,
    horizon: str = HORIZON_D1,
    claimed_setup_id: object = "",
    source: object = "",
    annotation_ref: object = "",
    known_at_claim: Mapping[str, Any] | None = None,
    note: object = "",
    session_date: str = "",
    now: datetime | None = None,
    path: Path = CLAIMED_PICKS_FILE,
) -> dict[str, Any] | None:
    """Claim one pick. Returns the row, or None when nothing could be written.

    **A second claim of a key that is already active writes NO row** and returns
    the EXISTING active row with ``duplicate: True`` beside it. Clicking the
    same setup twice is one pick, not two, and the caller still treats it as a
    success - the pick exists, so the chart is still done with.

    ``duplicate`` lives on the returned dict only; it is never a stored field.
    """
    row = build_claim_row(
        symbol=symbol,
        side=side,
        horizon=horizon,
        claimed_setup_id=claimed_setup_id,
        source=source,
        annotation_ref=annotation_ref,
        known_at_claim=known_at_claim,
        note=note,
        action=ACTION_CLAIM,
        session_date=session_date,
        now=now,
    )
    if row is None:
        return None
    as_of = _as_of_for(row, now)
    for existing in active_claims(path, as_of=as_of):
        if claim_key(existing) == claim_key(row):
            return dict(existing, duplicate=True)
    return row if append_row(row, path) else None


def record_drop(
    symbol: object,
    side: object,
    claimed_setup_id: object,
    *,
    source: object = "setups_table",
    note: object = "",
    session_date: str = "",
    now: datetime | None = None,
    path: Path = CLAIMED_PICKS_FILE,
) -> dict[str, Any] | None:
    """The trader ends a claim. One append-only retraction row.

    ``claimed_setup_id`` is REQUIRED and the retraction is exact: a claim is
    identified by ``(symbol, side, setup)`` and a drop ends the one it names.
    PCT-1 briefly gave it a blank default and made a nameless drop end every
    claim on that ``(symbol, side)``; the review sent that back, because a
    wildcard retraction is a wider production semantic than any caller needed
    (lead ruling 2026-09-15).
    """
    row = build_claim_row(
        symbol=symbol,
        side=side,
        horizon=HORIZON_D1,
        claimed_setup_id=claimed_setup_id,
        source=source,
        annotation_ref="",
        known_at_claim={},
        note=note,
        action=ACTION_DROP,
        session_date=session_date,
        now=now,
    )
    if row is None:
        return None
    return row if append_row(row, path) else None


def sweep_expired(
    path: Path = CLAIMED_PICKS_FILE,
    as_of: date | None = None,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Append one ``expire`` row per claim that crossed the fade. Idempotent.

    Returns the CLAIM rows that were retired, so a caller can say what it did.
    Called from the panel's day roll only: the fade is a session clock, and a
    sweep on a timer would burn a file read for a question that can only change
    once a day. A calendar failure expires nothing and appends nothing.
    """
    moment = as_of or date.today()
    rows = load_rows(path)
    live = standing_claims(rows)  # everything still standing, fade aside
    expired: list[dict[str, Any]] = []
    for row in live:
        if not _faded(str(row.get("session_date") or ""), moment):
            continue
        marker = build_claim_row(
            symbol=row.get("symbol"),
            side=row.get("side"),
            horizon=str(row.get("horizon") or HORIZON_D1),
            claimed_setup_id=row.get("claimed_setup_id"),
            source="machine",
            annotation_ref="",
            known_at_claim={},
            note=f"faded after {fade_trading_days()} trading days",
            action=ACTION_EXPIRE,
            session_date=str(row.get("session_date") or ""),
            now=now,
        )
        if marker is None:
            continue
        if append_row(marker, path):
            expired.append(dict(row))
    return expired


def _as_of_for(row: Mapping[str, Any], now: datetime | None) -> date:
    text = str(row.get("session_date") or "").strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return (now or datetime.now()).date()


# ---------------------------------------------------------------------------
# The horizon, resolved ONCE and explicitly
# ---------------------------------------------------------------------------
#: Registry groups whose families are DAILY theses. `setup_docs` has four
#: groups and every one of them is a swing/daily family - there is no day-trade
#: group in the registry, and this packet does not invent one. The `"m5"` answer
#: comes from the alert being an M5 review alert, never from a group name.
D1_REGISTRY_GROUPS = frozenset(
    {
        "Main swing",
        "Earnings cycle",
        "Study (measured only)",
        "Playbook research",
    }
)


def registry_horizon(claimed_setup_id: object) -> str:
    """``"d1"`` when the registry names this setup, ``""`` when it cannot.

    ``none_of_these`` is the honest answer with no family behind it, and an id
    the registry never heard of is the same answer. Both place NOTHING: a pick
    whose horizon cannot be resolved is a pick nobody can rank.

    Deliberately NOT ``setup_docs.resolve_setup_doc``, which falls back to
    ``general`` for anything it does not know - a fallback that would give an
    unknown id a horizon it never earned.
    """
    setup_id = str(claimed_setup_id or "").strip().lower()
    if not setup_id:
        return ""
    try:
        from ui.annotations.setup_claims import setup_claim_groups

        for group_name, claims in setup_claim_groups():
            if group_name not in D1_REGISTRY_GROUPS:
                continue
            for claim in claims:
                if str(claim.setup_id).strip().lower() == setup_id:
                    return HORIZON_D1
    except Exception:  # noqa: BLE001 - an unreadable registry places nothing
        return ""
    return ""


def claim_horizon(alert, claimed_setup_id: object, *, is_m5_review: bool = False) -> str:
    """Which desk a claimed like belongs to: ``"d1"``, ``"m5"`` or ``""``.

    Trader, 2026-09-14: *"Determine the trade horizon explicitly; do not rely on
    a stale chart timeframe."* The risk is real - the review pane's capture rail
    is constructed on "D1" and ``set_alert`` never re-pointed it, so every chart
    in the queue carried the same answer whatever was on screen.

    So the answer comes from the ALERT and the CLAIM, in that order, and there
    is no rail in this signature for a stale value to arrive through:

    1. a D1 scan alert (``BounceAlert.is_d1``) is a D1 thesis;
    2. an alert the panel routes to the M5 bar is an intraday one - the panel's
       own ``_is_m5_review_alert`` answer is handed in as a flag, because the
       widget that calls this never imports the panel;
    3. anything else - a manual chart look, a chart-watch hit, an armed alert -
       takes the horizon of the setup the trader CLAIMED, from the registry.
    """
    if bool(getattr(alert, "is_d1", False)):
        return HORIZON_D1
    if is_m5_review:
        return HORIZON_M5
    return registry_horizon(claimed_setup_id)


def known_at_claim_from_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the desk already knew about the name, from the alert's own payload.

    Honest and possibly EMPTY: a claimed name the scan never carried has no
    measurements, and ``{}`` is what the row says instead of a confident zero.
    """
    source = payload if isinstance(payload, Mapping) else {}
    known: dict[str, Any] = {}
    for key in KNOWN_AT_CLAIM_FIELDS:
        if key in source and source.get(key) not in (None, ""):
            known[key] = source.get(key)
    return known


#: The measurements a claim carries forward when the alert had them. Named, not
#: "whatever the payload held": a claimed row is read by the point system, and a
#: field list that drifted with the payload would change what a pick scores.
KNOWN_AT_CLAIM_FIELDS = (
    "priority_score",
    "score",
    "expected_r",
    "setup_family",
    "priority_bucket",
    "key_level",
    "last_price",
    "d1_vs_sector",
    "d1_vs_industry",
)


def claim_source(surface: object, alert) -> str:
    """``surface:kind`` - which screen made the claim, and off what.

    Two halves because both are asked later: "do I claim better off the chart
    review or the setups table?" needs the surface, and "was this a D1 flag or a
    name I typed?" needs the alert's own tag.
    """
    surface_text = str(surface or "").strip() or "chart_review"
    kind = str(getattr(alert, "tag", "") or "").strip()
    if not kind:
        kind = str(getattr(alert, "timeframe", "") or "").strip().lower()
    return f"{surface_text}:{kind}" if kind else surface_text


def claims_for_symbols(
    symbols: Iterable[str],
    rows_or_path=CLAIMED_PICKS_FILE,
    *,
    as_of: date | None = None,
) -> list[dict[str, Any]]:
    """Active claims restricted to a symbol set. Read-only convenience."""
    wanted = {normalize_symbol(symbol) for symbol in symbols or ()}
    return [
        row
        for row in active_claims(rows_or_path, as_of=as_of)
        if normalize_symbol(row.get("symbol")) in wanted
    ]
