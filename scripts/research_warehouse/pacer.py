"""The shared IB pacer: champions are never metered (plan sec 5.3).

One process-wide arbiter decides when **capture** may issue a provider request.
Champion traffic - the master scan, BounceBot, armed watches - is
pass-through: counted so capture knows how much budget is left, never delayed,
never queued, never refused. Metering a champion would change live champion
timing without golden fixtures, which plan.md Section 5 forbids outright.

The three rules, in the order they bite:

1. **Champion activity wins instantly.** While a champion request is in flight,
   or within the quiet period right after one, capture does not go out.
2. **Capture lives inside a token bucket** sized as the published floor minus
   the champions' own observed consumption over the trailing window. The floor
   (60 requests / 10 minutes) is a conservative published number, not measured
   capacity; the pilot measures the real ceiling and only then does the
   allocation grow (LD-02).
3. **A pacing error stops capture, not the desk.** IB error 162/366 backs
   capture off with escalating cool-off. A capture-caused error is tagged as
   capture at this layer and can never reach the champion fetch boundary's
   Yahoo-only circuit breaker (`_IBKR_HISTORICAL_FAILURE_COUNT` in
   `master_avwap_lib/legacy.py`) - that breaker's silent downgrade to Yahoo is
   exactly the BF.B/LC blackout precedent, and risk R1 exists to prevent
   capture from ever triggering it.

Nothing here opens a socket: the pacer decides, the caller acts. That keeps it
fully testable offline and keeps the decision logic in one place instead of
smeared across job code.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# --- client-ID allocation (sec 5.3) ----------------------------------------
# 1003 collided with the M1 dual-scheduler and is retired. The mini-PC range is
# reserved but excluded from Phases 0-8 entirely (LD-01).
CLIENT_ID_CAPTURE_STREAMER = 1010
CLIENT_ID_NIGHTLY_BACKFILL = 1011
CLIENT_ID_MINI_PC_RANGE = range(1020, 1030)
RETIRED_CLIENT_IDS = frozenset({1003})

ROLE_CAPTURE_STREAM = "capture_stream"
ROLE_NIGHTLY_BACKFILL = "nightly_backfill"
ROLE_MINI_PC_BUNDLE = "mini_pc_bundle"

CLIENT_IDS_BY_ROLE = {
    ROLE_CAPTURE_STREAM: CLIENT_ID_CAPTURE_STREAMER,
    ROLE_NIGHTLY_BACKFILL: CLIENT_ID_NIGHTLY_BACKFILL,
}

# --- published floors (sec 5.1); starting allocations, never capacity truth --
PACING_WINDOW_SECONDS = 600
PUBLISHED_REQUESTS_PER_WINDOW = 60
#: Capture's share of the floor before champion consumption is subtracted.
#: Effectively ~10-15 req/10 min during RTH once champions are counted.
CAPTURE_WINDOW_ALLOWANCE = 15
#: Identical (symbol, timeframe, window) requests inside this many seconds are
#: refused: IB's own identical-request cooldown.
IDENTICAL_REQUEST_COOLDOWN_SECONDS = 15
#: How long capture stays out of the way after a champion request.
CHAMPION_QUIET_SECONDS = 1.0
#: Escalating cool-off after a pacing error, capped.
ERROR_BACKOFF_SECONDS = 60.0
ERROR_BACKOFF_MAX_SECONDS = 600.0

PACING_ERROR_CODES = frozenset({162, 366})

DENY_CHAMPION_ACTIVE = "CHAMPION_ACTIVE"
DENY_BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
DENY_PACING_BACKOFF = "PACING_BACKOFF"
DENY_IDENTICAL_COOLDOWN = "IDENTICAL_REQUEST_COOLDOWN"
GRANTED = "GRANTED"


class ClientIdError(RuntimeError):
    """A capture connection asked for an id it must never use."""


def assert_client_id(client_id: int, role: str) -> int:
    """Validate a capture client id at connect time (risk R2).

    The 1003 collision produced a silent Yahoo fallback the first time it
    happened; asserting at connect makes a repeat loud instead of silent.
    """
    value = int(client_id)
    if value in RETIRED_CLIENT_IDS:
        raise ClientIdError(
            f"client id {value} is retired (it collided with the M1 dual scheduler); "
            f"use {CLIENT_IDS_BY_ROLE.get(role, CLIENT_ID_NIGHTLY_BACKFILL)} for role {role!r}."
        )
    if role == ROLE_MINI_PC_BUNDLE:
        if value not in CLIENT_ID_MINI_PC_RANGE:
            raise ClientIdError(f"mini-PC bundle ids are {CLIENT_ID_MINI_PC_RANGE}, not {value}.")
        raise ClientIdError(
            "the mini-PC is excluded from warehouse Phases 0-8 (LD-01); it never runs capture "
            "until the plan.md M1 client-ID/dual-scheduler reconciliation lands."
        )
    expected = CLIENT_IDS_BY_ROLE.get(role)
    if expected is None:
        raise ClientIdError(f"unknown capture role {role!r}; allocated roles: {sorted(CLIENT_IDS_BY_ROLE)}")
    if value != expected:
        raise ClientIdError(f"role {role!r} must connect with client id {expected}, not {value}.")
    return value


def is_pacing_error(code, message: str = "") -> bool:
    try:
        numeric = int(code)
    except (TypeError, ValueError):
        numeric = 0
    if numeric in PACING_ERROR_CODES:
        return True
    text = str(message or "").lower()
    return any(marker in text for marker in ("pacing violation", "rate limit", "query cancelled"))


@dataclass
class PacerDecision:
    granted: bool
    reason: str
    wait_seconds: float = 0.0


@dataclass
class _Event:
    at: datetime
    kind: str  # champion | capture
    family: str = ""
    symbol: str = ""


@dataclass
class PacerSnapshot:
    champion_requests_in_window: int = 0
    capture_requests_in_window: int = 0
    capture_budget: int = 0
    capture_remaining: int = 0
    champion_errors: int = 0
    capture_errors: int = 0
    backoff_until: str = ""
    last_champion_at: str = ""
    grants: int = 0
    denials: dict = field(default_factory=dict)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class IbPacer:
    """Process-wide arbiter. Decides for capture; only observes champions."""

    def __init__(
        self,
        *,
        window_seconds: int = PACING_WINDOW_SECONDS,
        published_per_window: int = PUBLISHED_REQUESTS_PER_WINDOW,
        capture_allowance: int = CAPTURE_WINDOW_ALLOWANCE,
        champion_quiet_seconds: float = CHAMPION_QUIET_SECONDS,
        identical_cooldown_seconds: int = IDENTICAL_REQUEST_COOLDOWN_SECONDS,
        error_backoff_seconds: float = ERROR_BACKOFF_SECONDS,
        error_backoff_max_seconds: float = ERROR_BACKOFF_MAX_SECONDS,
        clock=None,
    ):
        self.window = timedelta(seconds=int(window_seconds))
        self.published_per_window = int(published_per_window)
        self.capture_allowance = int(capture_allowance)
        self.champion_quiet_seconds = float(champion_quiet_seconds)
        self.identical_cooldown = timedelta(seconds=int(identical_cooldown_seconds))
        self.error_backoff_seconds = float(error_backoff_seconds)
        self.error_backoff_max_seconds = float(error_backoff_max_seconds)
        self._clock = clock or _utcnow
        self._lock = threading.RLock()
        self._events: list[_Event] = []
        self._champion_in_flight = 0
        self._last_champion_at: datetime | None = None
        self._backoff_until: datetime | None = None
        self._backoff_seconds = 0.0
        self._recent_keys: dict[str, datetime] = {}
        self._champion_errors = 0
        self._capture_errors = 0
        self._grants = 0
        self._denials: dict[str, int] = {}

    # -- champion side: observation only -----------------------------------
    def note_champion_request(self, family: str = "", symbol: str = "", *, now: datetime | None = None) -> None:
        """Count one champion request. This call NEVER blocks or delays."""
        stamp = now or self._clock()
        with self._lock:
            self._events.append(_Event(at=stamp, kind="champion", family=family, symbol=symbol))
            self._last_champion_at = stamp
            self._trim(stamp)

    def champion_window(self, family: str = "", symbol: str = ""):
        """Context manager marking a champion request as in flight.

        Entering it can never wait: capture yields to the champion, never the
        other way round.
        """
        return _ChampionWindow(self, family, symbol)

    # -- capture side: metered ---------------------------------------------
    def try_acquire(self, *, key: str = "", now: datetime | None = None) -> PacerDecision:
        """Ask whether capture may issue one request right now."""
        stamp = now or self._clock()
        with self._lock:
            self._trim(stamp)
            if self._champion_in_flight > 0:
                return self._deny(DENY_CHAMPION_ACTIVE, self.champion_quiet_seconds)
            if self._last_champion_at is not None:
                quiet_for = (stamp - self._last_champion_at).total_seconds()
                if quiet_for < self.champion_quiet_seconds:
                    return self._deny(DENY_CHAMPION_ACTIVE, self.champion_quiet_seconds - quiet_for)
            if self._backoff_until is not None and stamp < self._backoff_until:
                return self._deny(DENY_PACING_BACKOFF, (self._backoff_until - stamp).total_seconds())
            if key:
                seen = self._recent_keys.get(key)
                if seen is not None and stamp - seen < self.identical_cooldown:
                    wait = (self.identical_cooldown - (stamp - seen)).total_seconds()
                    return self._deny(DENY_IDENTICAL_COOLDOWN, wait)
            budget = self._capture_budget()
            used = sum(1 for event in self._events if event.kind == "capture")
            if used >= budget:
                oldest = next((event.at for event in self._events if event.kind == "capture"), stamp)
                wait = max(0.0, (oldest + self.window - stamp).total_seconds())
                return self._deny(DENY_BUDGET_EXHAUSTED, wait)

            self._events.append(_Event(at=stamp, kind="capture"))
            if key:
                self._recent_keys[key] = stamp
            self._grants += 1
            return PacerDecision(granted=True, reason=GRANTED)

    def acquire(self, *, key: str = "", timeout: float = 0.0, sleep=None, now=None) -> PacerDecision:
        """Blocking form: wait up to ``timeout`` for a capture slot."""
        import time

        sleeper = sleep or time.sleep
        deadline_budget = float(timeout)
        while True:
            decision = self.try_acquire(key=key, now=now() if callable(now) else now)
            if decision.granted or deadline_budget <= 0:
                return decision
            nap = min(max(decision.wait_seconds, 0.01), deadline_budget)
            sleeper(nap)
            deadline_budget -= nap

    # -- errors -------------------------------------------------------------
    def note_error(self, code, message: str = "", *, capture: bool, now: datetime | None = None) -> bool:
        """Record a provider error. Returns True when capture backs off.

        ``capture=True`` marks the error as belonging to capture traffic. That
        tag is the whole isolation mechanism: capture errors are handled here
        and never routed to the champion's failure counter, so capture can
        never push live scans onto the Yahoo-only path (R1).
        """
        stamp = now or self._clock()
        pacing = is_pacing_error(code, message)
        with self._lock:
            if capture:
                self._capture_errors += 1
            else:
                self._champion_errors += 1
            if not pacing:
                return False
            # Any observed pacing signal - champion or capture - means the
            # window is under pressure, so capture yields. The champion is
            # never slowed by this; only capture is.
            self._backoff_seconds = (
                self.error_backoff_seconds
                if self._backoff_seconds <= 0
                else min(self._backoff_seconds * 2, self.error_backoff_max_seconds)
            )
            self._backoff_until = stamp + timedelta(seconds=self._backoff_seconds)
            return True

    def note_capture_success(self, *, now: datetime | None = None) -> None:
        """A clean capture response relaxes the escalating cool-off."""
        with self._lock:
            self._backoff_seconds = 0.0
            self._backoff_until = None

    # -- introspection ------------------------------------------------------
    def snapshot(self, *, now: datetime | None = None) -> PacerSnapshot:
        stamp = now or self._clock()
        with self._lock:
            self._trim(stamp)
            champions = sum(1 for event in self._events if event.kind == "champion")
            captures = sum(1 for event in self._events if event.kind == "capture")
            budget = self._capture_budget()
            return PacerSnapshot(
                champion_requests_in_window=champions,
                capture_requests_in_window=captures,
                capture_budget=budget,
                capture_remaining=max(0, budget - captures),
                champion_errors=self._champion_errors,
                capture_errors=self._capture_errors,
                backoff_until=self._backoff_until.isoformat() if self._backoff_until else "",
                last_champion_at=self._last_champion_at.isoformat() if self._last_champion_at else "",
                grants=self._grants,
                denials=dict(self._denials),
            )

    # -- internals ----------------------------------------------------------
    def _capture_budget(self) -> int:
        champions = sum(1 for event in self._events if event.kind == "champion")
        headroom = self.published_per_window - champions
        return max(0, min(self.capture_allowance, headroom))

    def _trim(self, now: datetime) -> None:
        cutoff = now - self.window
        self._events = [event for event in self._events if event.at > cutoff]
        key_cutoff = now - self.identical_cooldown
        self._recent_keys = {key: at for key, at in self._recent_keys.items() if at > key_cutoff}

    def _deny(self, reason: str, wait_seconds: float) -> PacerDecision:
        self._denials[reason] = self._denials.get(reason, 0) + 1
        return PacerDecision(granted=False, reason=reason, wait_seconds=max(0.0, float(wait_seconds)))


class _ChampionWindow:
    def __init__(self, pacer: IbPacer, family: str, symbol: str):
        self._pacer = pacer
        self._family = family
        self._symbol = symbol

    def __enter__(self):
        with self._pacer._lock:
            self._pacer._champion_in_flight += 1
        self._pacer.note_champion_request(self._family, self._symbol)
        return self

    def __exit__(self, exc_type, exc, tb):
        with self._pacer._lock:
            self._pacer._champion_in_flight = max(0, self._pacer._champion_in_flight - 1)
            self._pacer._last_champion_at = self._pacer._clock()
        return False


_PACER: IbPacer | None = None
_PACER_LOCK = threading.Lock()


def get_pacer() -> IbPacer:
    """The one process-wide arbiter."""
    global _PACER
    with _PACER_LOCK:
        if _PACER is None:
            _PACER = IbPacer()
        return _PACER


def reset_pacer(pacer: IbPacer | None = None) -> IbPacer:
    """Test hook: install a fresh arbiter."""
    global _PACER
    with _PACER_LOCK:
        _PACER = pacer or IbPacer()
        return _PACER


__all__ = [
    "CAPTURE_WINDOW_ALLOWANCE",
    "CLIENT_IDS_BY_ROLE",
    "CLIENT_ID_CAPTURE_STREAMER",
    "CLIENT_ID_MINI_PC_RANGE",
    "CLIENT_ID_NIGHTLY_BACKFILL",
    "DENY_BUDGET_EXHAUSTED",
    "DENY_CHAMPION_ACTIVE",
    "DENY_IDENTICAL_COOLDOWN",
    "DENY_PACING_BACKOFF",
    "PACING_ERROR_CODES",
    "PUBLISHED_REQUESTS_PER_WINDOW",
    "RETIRED_CLIENT_IDS",
    "ROLE_CAPTURE_STREAM",
    "ROLE_MINI_PC_BUNDLE",
    "ROLE_NIGHTLY_BACKFILL",
    "ClientIdError",
    "IbPacer",
    "PacerDecision",
    "PacerSnapshot",
    "assert_client_id",
    "get_pacer",
    "is_pacing_error",
    "reset_pacer",
]
