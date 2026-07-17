"""TC2000-style relative volume for intraday (5-minute) bars.

The trader's TC2000 script on a 5-minute chart:

    V / ((V78 + V156 + V234 + ... + V1170) / 15)

78 five-minute bars make one regular session, so ``V78`` is the same
time-of-day bar one session back; the denominator is the 15-session average
volume for *this* bar of the day. Readings over 1.00 mean the name is
trading more volume than it normally does at this exact point in the session.

Two variants:

- ``bar_rvol`` is the script verbatim - the latest completed bar against its
  same-slot baseline. Faithful, but a single 5-minute print is noisy for a
  scanner that samples every ~30 minutes.
- ``session_rvol`` is the cumulative equivalent - today's volume so far
  against the 15-session average *for the same number of bars into the day*.
  Same "over 1.00 = elevated" reading, robust to one quiet bar.

Sessions with a different bar count than today (half days, data gaps) still
contribute per-slot: each bar index is averaged over the sessions that have
that bar. Baselines built from fewer than ``min_sessions`` prior sessions
return None - better no reading than a fake one.
"""

from __future__ import annotations

from typing import Iterable, Sequence

RVOL_BASELINE_SESSIONS = 15
RVOL_MIN_BASELINE_SESSIONS = 5


def same_slot_baseline(
    prior_sessions: Sequence[Sequence[float]],
    bar_index: int,
    *,
    sessions: int = RVOL_BASELINE_SESSIONS,
    min_sessions: int = RVOL_MIN_BASELINE_SESSIONS,
) -> float | None:
    """Average volume at ``bar_index`` across the most recent prior sessions."""
    window = [list(session) for session in prior_sessions][-max(1, int(sessions)) :]
    values = []
    for session in window:
        if 0 <= bar_index < len(session):
            try:
                value = float(session[bar_index])
            except (TypeError, ValueError):
                continue
            if value >= 0:
                values.append(value)
    if len(values) < max(1, int(min_sessions)):
        return None
    return sum(values) / len(values)


def bar_rvol(
    today_volumes: Sequence[float],
    prior_sessions: Sequence[Sequence[float]],
    *,
    sessions: int = RVOL_BASELINE_SESSIONS,
    min_sessions: int = RVOL_MIN_BASELINE_SESSIONS,
) -> float | None:
    """The TC2000 script verbatim: latest bar vs its same-slot baseline."""
    if not today_volumes:
        return None
    index = len(today_volumes) - 1
    baseline = same_slot_baseline(
        prior_sessions, index, sessions=sessions, min_sessions=min_sessions
    )
    if not baseline:
        return None
    try:
        current = float(today_volumes[index])
    except (TypeError, ValueError):
        return None
    return current / baseline


def session_rvol(
    today_volumes: Sequence[float],
    prior_sessions: Sequence[Sequence[float]],
    *,
    sessions: int = RVOL_BASELINE_SESSIONS,
    min_sessions: int = RVOL_MIN_BASELINE_SESSIONS,
) -> float | None:
    """Cumulative session volume vs the same-depth 15-session baseline."""
    if not today_volumes:
        return None
    baseline_total = 0.0
    today_total = 0.0
    counted = 0
    for index, raw in enumerate(today_volumes):
        baseline = same_slot_baseline(
            prior_sessions, index, sessions=sessions, min_sessions=min_sessions
        )
        if baseline is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        baseline_total += baseline
        today_total += value
        counted += 1
    if counted == 0 or baseline_total <= 0:
        return None
    return today_total / baseline_total


def split_sessions(volumes_by_day: Iterable[tuple[object, float]]) -> list[list[float]]:
    """Group (session_key, volume) pairs, in order, into per-session lists.

    Callers hand in bars already sorted by time; the session key is anything
    stable per day (a date works). Keeps arrival order within each session.
    """
    sessions: list[list[float]] = []
    current_key: object = object()
    for key, volume in volumes_by_day:
        if key != current_key:
            sessions.append([])
            current_key = key
        try:
            sessions[-1].append(float(volume))
        except (TypeError, ValueError):
            sessions[-1].append(0.0)
    return sessions
