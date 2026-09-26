"""Longs off in a bad market (P9, the trader 2026-09-26). Presentation only.

The verdict is `setup_permutations.long_regime_working` for today: the trader's
regime first, else SPY above a rising 20-day on the last completed session. It
is computed once per day on a worker and cached; the Qt thread only reads the
cache. Unknown never hides: an unknown market shows longs. Nothing here changes
a detector, a score or what is recorded.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from datetime import date, datetime
from typing import Any, Iterable, Mapping

import project_paths

#: Machine-local switch, shared by the Alert Center, the swing table and the phone report.
SETTING = "longs_off_bad_market"
DEFAULT_ON = True
LABEL = "Longs off in a bad market"
#: The `hidden_by_show` detail reason for a longs-off hide.
REASON = "longs_off"
#: Swing rows with their own "waiting for the market" state; never hidden here.
EXEMPT_ROW_KEYS = ("leader_pullback", "post_earnings_drift")
#: How often the worker re-reads open positions (seconds); the verdict itself is per day.
OPEN_REFRESH_SECONDS = 300.0

YES = "yes"
NO = "no"
UNKNOWN = "unknown"
_ET_NAME = "America/New_York"


@dataclasses.dataclass(frozen=True)
class Verdict:
    """Today's long gate. ``verdict`` is yes / no / unknown; ``since`` is an ISO date or ""."""

    day: str
    verdict: str = UNKNOWN
    rule: str = UNKNOWN
    reason: str = ""
    since: str = ""
    open_symbols: frozenset = frozenset()

    @property
    def longs_off(self) -> bool:
        return self.verdict == NO


def enabled() -> bool:
    """The saved switch; unreadable keeps the default (on)."""
    try:
        value = project_paths.get_local_setting(SETTING, DEFAULT_ON)
    except Exception:  # noqa: BLE001 - a preference read never costs a surface
        return DEFAULT_ON
    return value if isinstance(value, bool) else DEFAULT_ON


def set_enabled(value: bool) -> None:
    project_paths.save_local_setting(SETTING, bool(value))


def banner_text(verdict: Verdict | None) -> str:
    """"Longs off: <reason> (since <date>)" when longs are off, else ""."""
    if verdict is None or not verdict.longs_off:
        return ""
    since = f" (since {verdict.since})" if verdict.since else ""
    return f"Longs off: {verdict.reason or 'the market is not on a long side'}{since}"


def _text(value: Any) -> str:
    return str(value or "").strip()


def row_is_exempt(raw: Any) -> bool:
    """A swing row that is a `leader_pullback` / `post_earnings_drift` row (it waits on its own)."""
    if not isinstance(raw, Mapping):
        return False
    for key in EXEMPT_ROW_KEYS:
        value = raw.get(key)
        if value not in (None, "", False, 0, "0", "false", "False"):
            return True
    texts = " ".join(
        _text(raw.get(field)).lower()
        for field in ("setup_family", "bucket", "setup_key", "study_families", "family")
    )
    return any(key in texts for key in EXEMPT_ROW_KEYS)


def is_long(side: Any) -> bool:
    return _text(side).upper() in {"LONG", "BUY", "L"}


def hides_long(verdict: Verdict | None, side: Any, symbol: Any, *, exempt: bool = False) -> bool:
    """True when the gate hides this row: longs off, a LONG, not exempt, no open position."""
    if verdict is None or not verdict.longs_off or exempt or not is_long(side):
        return False
    return _text(symbol).upper() not in verdict.open_symbols


# --- pure verdict -----------------------------------------------------------


def _day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(_text(value)[:10])
    except ValueError:
        return None


def _reference_session(today: date) -> date | None:
    """The last completed session before ``today`` (the verdict is fixed for the day)."""
    try:
        import market_calendar

        return market_calendar.previous_session(today)
    except Exception:  # noqa: BLE001 - outside the calendar is unknown
        return None


def _trader_since(regime_rows: Iterable[Mapping[str, Any]], today: date) -> str:
    """Start of the unbroken run of non-working trader regimes that holds today."""
    import setup_permutations as sp
    import structural_regime

    since = ""
    for segment in reversed(structural_regime.effective_segments(regime_rows)):
        start = _day(segment.get("start_date"))
        if start is None or start > today:
            continue
        if _text(segment.get("regime")) in sp.LONG_WORKING_TRADER_REGIMES:
            break
        since = start.isoformat()
    return since


def _spy_since(days: list[str], values: list[float]) -> str:
    """First day of the unbroken run of SPY-rule "no" days ending on the last close."""
    import setup_permutations as sp

    since = ""
    for end in range(len(values), 0, -1):
        vs, slope = sp.spy_trend(values[:end])
        if vs is None or slope is None or (vs > 0 and slope > 0):
            break
        since = days[end - 1]
    return since


def compute(
    *,
    today: Any,
    regime_rows: Iterable[Mapping[str, Any]] | None,
    spy_closes: Mapping[str, float] | None,
    open_symbols: Iterable[str] = (),
) -> Verdict:
    """Today's verdict from the trader's regime rows and SPY's completed daily closes.

    SPY counts only through the last completed session before ``today``; a stale
    or short SPY is unknown.
    """
    import setup_permutations as sp
    import structural_regime

    day = _day(today) or date.today()
    opens = frozenset(_text(s).upper() for s in open_symbols or () if _text(s))
    rows = list(regime_rows or ())
    segment = structural_regime.regime_at(rows, day) if rows else None
    trader = _text((segment or {}).get("regime"))
    reference = _reference_session(day)
    pairs = sorted(
        (str(key)[:10], float(close))
        for key, close in (spy_closes or {}).items()
        if close is not None and reference is not None and str(key)[:10] <= reference.isoformat()
    )
    current = bool(pairs) and reference is not None and pairs[-1][0] == reference.isoformat()
    days = [key for key, _close in pairs]
    values = [close for _key, close in pairs]
    vs, slope = sp.spy_trend(values) if current else (None, None)
    verdict, rule = sp.long_regime_working(trader, vs, slope)
    reason = since = ""
    if verdict == NO and rule == sp.WORKING_RULE_TRADER:
        reason = f"your regime is {structural_regime.label(trader)}"
        since = _trader_since(rows, day)
    elif verdict == NO:
        reason = "SPY is under its 20-day" if vs is not None and vs <= 0 else "SPY's 20-day is falling"
        since = _spy_since(days, values)
    return Verdict(day=day.isoformat(), verdict=verdict, rule=rule, reason=reason, since=since,
                   open_symbols=opens)


# --- IO (worker only) -------------------------------------------------------


def _today() -> date:
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo(_ET_NAME)).date()


def read_regime_rows() -> list[dict[str, Any]]:
    """The trader's raw regime journal rows, read-only. IO: never on the Qt thread."""
    import regime_join

    return regime_join.read_rows()


def read_spy_closes() -> dict[str, float]:
    """SPY's daily closes from the scan's bar store. IO: never on the Qt thread."""
    from pathlib import Path

    import setup_permutation_backfill

    return setup_permutation_backfill.read_spy_closes(
        Path(project_paths.MASTER_AVWAP_DAILY_BARS_DIR) / "SPY.parquet"
    )


def read_open_symbols() -> frozenset:
    """Symbols with an OPEN journal trade, read-only. IO: never on the Qt thread."""
    import regime_join

    rows = regime_join.select_read_only(None, "SELECT symbol FROM trades WHERE status = 'OPEN'")
    return frozenset(_text(row.get("symbol")).upper() for row in rows if _text(row.get("symbol")))


def load(today: Any = None) -> Verdict:
    """Compute today's verdict from the live stores. IO: never on the Qt thread."""
    day = _day(today) or _today()
    try:
        rows = read_regime_rows()
    except Exception:  # noqa: BLE001 - no regime is unknown
        rows = []
    try:
        closes = read_spy_closes()
    except Exception:  # noqa: BLE001 - no SPY is unknown
        closes = {}
    try:
        opens = read_open_symbols()
    except Exception:  # noqa: BLE001 - unread positions exempt nothing
        opens = frozenset()
    return compute(today=day, regime_rows=rows, spy_closes=closes, open_symbols=opens)


# --- the one cache ----------------------------------------------------------

_lock = threading.Lock()
_state: dict[str, Any] = {"verdict": None, "running": False, "opens_at": 0.0, "generation": 0}


def snapshot() -> Verdict | None:
    """The cached verdict, or None before the first worker read. Never does IO."""
    with _lock:
        return _state["verdict"]


def set_snapshot(verdict: Verdict | None) -> None:
    """Replace the cache (tests, and a caller that computed it on its own worker)."""
    with _lock:
        _state["verdict"] = verdict
        _state["opens_at"] = time.monotonic()
        _state["generation"] += 1


def clear_cache() -> None:
    set_snapshot(None)
    with _lock:
        _state["opens_at"] = 0.0


def _stale(now: float) -> tuple[bool, bool]:
    """``(verdict stale, open positions stale)``; call under the lock."""
    verdict = _state["verdict"]
    day_stale = verdict is None or verdict.day != _today().isoformat()
    return day_stale, day_stale or now - _state["opens_at"] >= OPEN_REFRESH_SECONDS


def current() -> Verdict:
    """Today's verdict, refreshed in place when stale. IO: call from a worker only."""
    now = time.monotonic()
    with _lock:
        day_stale, opens_stale = _stale(now)
        verdict, generation = _state["verdict"], _state["generation"]
    if not opens_stale and verdict is not None:
        return verdict
    if day_stale or verdict is None:
        fresh = load()
    else:
        try:
            fresh = dataclasses.replace(verdict, open_symbols=read_open_symbols())
        except Exception:  # noqa: BLE001 - keep the last positions
            fresh = verdict
    with _lock:
        if _state["generation"] == generation:
            _state["verdict"] = fresh
            _state["opens_at"] = time.monotonic()
            _state["generation"] += 1
        return _state["verdict"] or fresh


def refresh_async() -> bool:
    """Start one worker read when the cache is stale; True when one started. Qt-safe."""
    with _lock:
        if _state["running"]:
            return False
        day_stale, opens_stale = _stale(time.monotonic())
        if not (day_stale or opens_stale):
            return False
        _state["running"] = True

    def run() -> None:
        try:
            current()
        except Exception:  # noqa: BLE001 - unknown shows; the next tick retries
            pass
        finally:
            with _lock:
                _state["running"] = False

    threading.Thread(target=run, name="longs-market-gate", daemon=True).start()
    return True
