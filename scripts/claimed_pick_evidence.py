"""The claimed D1 picks, graded beside the tracker's own populations - D1C-B.

Trader, 2026-09-14:

    *"Reuse the existing outcome and evidence services. Measure manually
    claimed opportunities from the claim time forward, retaining the claimed
    setup and the measurements known then. Let me compare FAV, HC and My liked
    trades, and compare the setup types I claimed. Show sample sizes, pending
    and unmeasured results, and the existing uncertainty measures. Handle
    overlapping buckets explicitly. Repeated clicks must not create extra
    independent trades. Keep opportunity results separate from actual journal
    trade results. Use these results to show what has worked best over time. Do
    not automatically change ranking weights, detector rules or promotion
    status."*

**No second pipeline.** Both sides of this readout were already graded forward
before this module existed and neither is regraded here:

* the LIKE side is `ui/annotations/like_cohort.py` -> `human_focus_tracking`
  (`HORIZONS = (1, 3, 5, 10)`), written nightly by
  `ai_jobs.cohorts.run_like_cohort_grading`;
* the TRACKER side is `master_avwap_tier_outcomes.csv`, read through the ONE
  reader `swing_evidence.read_eligible_rows(..., POLICY_SCANROW_V1)`.

**TWO CLOCKS, NEVER POOLED.** The like cohort measures
:data:`evidence_stats.SWING_HORIZON_SESSIONS` EXCHANGE SESSIONS from the claim
day's close; the tracker measures the same number of SCAN ROWS - the symbol's
own fifth later scan row. Those are different units over the same horizon
number, so the two n's sit side by side, each labelled with its own clock, and
no cell anywhere is their sum. An overlap is NAMED (`also_fav` / `also_near`),
never merged into a union.

**HC has no forward record and says so in words.** `priority_bucket` in the live
tracker is {`favorite_setup`, `near_favorite_zone`, blank}: high conviction is an
overlay computed at feed-write time (`legacy._priority_is_high_conviction`) and
never stamped on an outcome row. A population with no rows is UNMEASURED, and
:data:`HC_NO_ROWS_NOTE` is what it prints - never 0%.

**Ground rule 10: no new statistic.** Every number here is a count or
`swing_headline.wilson_lower_bound` (z 1.96). No bootstrap, no blend, no average
of two cells. The reportable floor is `evidence_stats.MIN_REPORTABLE_N` (30) and
the leader line refuses to name a setup below it.

**Nothing moves.** This module reads four files and writes none. Nothing here
reaches a detector, a score, a ranking weight, an alert, a watchlist, Focus, the
review queue, `review_policy.json` or a promotion status. It imports nothing
from the journal: opportunity results and actual trade results are different
surfaces and the Journal plus the said-vs-did report
(`preference_trade_outcomes`) own the second one.

:func:`build_comparison` is PURE - every input is an iterable of mappings and it
opens no file. :func:`load_inputs` is the ONE file read, so a Qt panel can run it
on its worker. :func:`render_text` is the ONE renderer; the CLI prints exactly
what it returns.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evidence_stats import MIN_REPORTABLE_N, SWING_HORIZON_SESSIONS, lately_window
from project_paths import (
    CLAIMED_PICKS_FILE,
    LIKE_COHORT_OUTCOMES_FILE,
    LIKE_COHORT_PICKS_FILE,
    MASTER_AVWAP_TIER_OUTCOMES_FILE,
)
from swing_evidence import POLICY_SCANROW_V1, read_eligible_rows
from swing_headline import wilson_lower_bound

#: The cohort a like with no named setup lands in (`like_cohort.like_cohort_source`).
#: It is a real answer - "this chart was good and I decline to name it" - and a
#: real cohort, but it is not one of MY CLAIMED trades, so it is not read here.
UNCLAIMED_SOURCE = "like_unclaimed"

#: `like_cohort.LIKE_COHORT_PREFIX` plus its separator, stripped for display only.
SOURCE_PREFIX = "like_"

#: P9. A quick like (Alt+L) names no setup and is excluded, COUNTED ONCE in a
#: footnote rather than silently dropped. An absent `like_mode` reads `claimed`.
QUICK_LIKE_MODE = "quick"

#: The two clocks, in the words each side is measured in.
LIKE_CLOCK = (
    f"{SWING_HORIZON_SESSIONS} exchange sessions from the claim day's close "
    "(like cohort, h5)"
)
TRACKER_CLOCK = (
    f"{SWING_HORIZON_SESSIONS} scan rows - the symbol's own fifth later scan row "
    "(tracker, POLICY_SCANROW_V1)"
)

#: What an HC cell says while the tracker has never stamped `high_conviction`.
HC_NO_ROWS_NOTE = (
    "unmeasured: the tracker records favorite_setup / near_favorite_zone "
    "only (0 HC rows)"
)

#: The tracker's own bucket spellings.
BUCKET_FAV = "favorite_setup"
BUCKET_NEAR = "near_favorite_zone"
BUCKET_HC = "high_conviction"

#: Population keys, in the order every surface prints them.
POPULATION_KEYS = ("liked", "fav", "hc", "near")
POPULATION_LABELS = {
    "liked": "My liked trades",
    "fav": "FAV",
    "hc": "HC",
    "near": "Near",
}

#: The provenance of a liked row whose claim predates the D1C claim store.
PROVENANCE_ANNOTATION_ONLY = "annotation only (pre-D1C)"
PROVENANCE_CLAIMED = "claimed (claimed_picks.jsonl)"

#: `all` is a window, not the absence of one: `read_eligible_rows(end=)` only
#: moves the RIGHT edge, so a caller that wants every row must say so with an
#: explicit `window=`.
EARLIEST_WINDOW_START = "0001-01-01"

WINDOW_LATELY = "lately"
WINDOW_ALL = "all"
WINDOWS = (WINDOW_LATELY, WINDOW_ALL)


# ---------------------------------------------------------------------------
# What one cell is
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Population:
    """One graded population, in ITS OWN clock. Never pooled with another."""

    key: str
    label: str
    clock: str
    n: int = 0
    wins: int = 0
    win_rate: float | None = None
    bound: float | None = None
    #: The horizon has not arrived yet. On the tracker side this is 0 by
    #: construction (`POLICY_SCANROW_V1.immature_value` is empty - a v1 row
    #: cannot exist until the symbol's later scan row does).
    pending: int = 0
    #: `None` on the tracker side: an unmeasurable scan row is an EXCLUSION the
    #: read reports once (`tracker read excluded: ...`), not a per-bucket count.
    unmeasured: int | None = None
    #: Overlap, NAMED. `None` where the question does not apply.
    also_fav: int | None = None
    also_near: int | None = None
    dropped_duplicates: int | None = None
    note: str = ""


@dataclass(frozen=True)
class LikedRow:
    """One claimed like, with what was known when the trader claimed it."""

    session_date: str
    symbol: str
    side: str
    source: str
    status: str = "unmeasured"
    win: bool | None = None
    h5_return: float | None = None
    unmeasured_reason: str = ""
    claimed_setup_id: str = ""
    claim_at: str = ""
    known_at_claim: Mapping[str, Any] | None = None
    provenance: str = PROVENANCE_ANNOTATION_ONLY
    also_fav: bool = False
    also_near: bool = False
    also_hc: bool = False

    @property
    def setup_name(self) -> str:
        return display_setup(self.source)


@dataclass(frozen=True)
class SetupRow:
    """One claimed setup family's own record."""

    source: str
    label: str
    n: int = 0
    wins: int = 0
    win_rate: float | None = None
    bound: float | None = None
    pending: int = 0
    unmeasured: int = 0


@dataclass(frozen=True)
class WindowReport:
    """Everything one window says. `lately` and `all` are both built, always."""

    window: str
    window_dates: tuple[str, str]
    populations: dict[str, Population] = field(default_factory=dict)
    by_setup: tuple[SetupRow, ...] = ()
    liked_rows: tuple[LikedRow, ...] = ()
    leader: str = ""
    footnotes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Comparison:
    """The requested window's fields at the top, both windows underneath."""

    as_of: str
    window: str
    by_window: dict[str, WindowReport]
    populations: dict[str, Population]
    by_setup: tuple[SetupRow, ...]
    liked_rows: tuple[LikedRow, ...]
    leader: str
    footnotes: tuple[str, ...]
    window_dates: tuple[str, str]


# ---------------------------------------------------------------------------
# Small readers - a present-and-empty cell is never a zero
# ---------------------------------------------------------------------------


def display_setup(source: Any) -> str:
    """`like_avwap_breakout` -> `avwap_breakout`. Display only."""
    text = str(source or "").strip()
    return text[len(SOURCE_PREFIX):] if text.startswith(SOURCE_PREFIX) else text


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _side(value: Any) -> str:
    """`human_focus_tracking._pick_key`'s spelling, so the joins agree."""
    text = _text(value).upper()
    return "SHORT" if text.startswith("SHORT") else "LONG"


def _float_or_none(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes"}


def _matured(value: Any) -> set[str]:
    return {part.strip() for part in _text(value).split(",") if part.strip()}


def _session_of(row: Mapping[str, Any]) -> str:
    """`session_date` when the row carries one, else `trade_date`.

    Ground rule 7 put `session_date` on every like-cohort row; the rows written
    before it exist with `trade_date` only, and they are the same market
    session. Both are read so an old row is graded rather than dropped.
    """
    return (_text(row.get("session_date")) or _text(row.get("trade_date")))[:10]


def _as_of_iso(as_of: Any) -> str:
    if isinstance(as_of, date):
        return as_of.isoformat()
    text = _text(as_of)[:10]
    if text:
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            pass
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# The ONE file read
# ---------------------------------------------------------------------------


def _read_csv(path: Any) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        # A missing store is zero rows, never a raised reader: the trader opens
        # this tab mid-session and an absent file is a fact, not a failure.
        return []
    try:
        with target.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def _read_jsonl(path: Any) -> list[dict[str, Any]]:
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


def store_paths(
    *,
    picks_path: Any = None,
    outcomes_path: Any = None,
    tier_path: Any = None,
    claims_path: Any = None,
) -> dict[str, Path]:
    """The four stores, resolved at CALL time.

    The module constants are looked up in the function body rather than bound
    into a default, so a test (or a panel pointed at a temporary home folder)
    can patch them on this module the way `tracker_export_files` already allows.
    """
    return {
        "picks_path": Path(picks_path if picks_path is not None else LIKE_COHORT_PICKS_FILE),
        "outcomes_path": Path(
            outcomes_path if outcomes_path is not None else LIKE_COHORT_OUTCOMES_FILE
        ),
        "tier_path": Path(
            tier_path if tier_path is not None else MASTER_AVWAP_TIER_OUTCOMES_FILE
        ),
        "claims_path": Path(claims_path if claims_path is not None else CLAIMED_PICKS_FILE),
    }


def load_inputs(
    *,
    picks_path: Any = None,
    outcomes_path: Any = None,
    tier_path: Any = None,
    claims_path: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Read the four stores. The ONLY file read in this module.

    Returns exactly `build_comparison`'s four row arguments, so the CLI and the
    Setup Tracker's worker both call `build_comparison(**load_inputs(), ...)`
    and neither opens a file on a paint path or the Qt thread.
    """
    paths = store_paths(
        picks_path=picks_path,
        outcomes_path=outcomes_path,
        tier_path=tier_path,
        claims_path=claims_path,
    )
    return {
        "like_picks": _read_csv(paths["picks_path"]),
        "like_outcomes": _read_csv(paths["outcomes_path"]),
        "tier_rows": _read_csv(paths["tier_path"]),
        "claims": _read_jsonl(paths["claims_path"]),
    }


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def _rows(value: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in (value or ()) if isinstance(row, Mapping)]


def _claim_index(claims: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict]:
    """`(session, symbol, side)` -> the FIRST `claim` row for it.

    Repeated clicks must not create extra independent trades: the store is
    append-only and a key claimed three times is three rows and ONE thesis, so
    the first claim of that key on that session is the one the row carries.
    """
    index: dict[tuple[str, str, str], dict] = {}
    for row in _rows(claims):
        if _text(row.get("action")) not in {"", "claim"}:
            continue
        key = (
            _text(row.get("session_date"))[:10],
            _text(row.get("symbol")).upper(),
            _side(row.get("side")),
        )
        if not key[0] or not key[1]:
            continue
        index.setdefault(key, row)
    return index


def _outcome_index(
    outcomes: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str], dict]:
    index: dict[tuple[str, str, str], dict] = {}
    for row in _rows(outcomes):
        key = (_session_of(row), _text(row.get("symbol")).upper(), _side(row.get("side")))
        index.setdefault(key, row)
    return index


def _tracker_days(rows: Iterable[Mapping[str, Any]], bucket: str) -> set[tuple[str, str, str]]:
    """`(scan_date, symbol, side)` for one bucket's ELIGIBLE rows."""
    return {
        (
            _text(row.get("scan_date"))[:10],
            _text(row.get("symbol")).upper(),
            _side(row.get("side")),
        )
        for row in rows
        if _text(row.get("priority_bucket")).lower() == bucket
    }


def _dedupe_picks(
    like_picks: Iterable[Mapping[str, Any]],
) -> tuple[list[dict], int, int]:
    """(claimed picks, one per key), quick likes excluded, duplicates dropped.

    The order is P9's: a QUICK like is excluded first (it names no setup, so it
    is not one of "the setup types I claimed"), then `like_unclaimed`, then the
    same click again.
    """
    quick = 0
    duplicates = 0
    kept: dict[tuple[str, str, str], dict] = {}
    for row in _rows(like_picks):
        if _text(row.get("like_mode")).lower() == QUICK_LIKE_MODE:
            quick += 1
            continue
        if _text(row.get("source")).lower() == UNCLAIMED_SOURCE:
            continue
        session = _session_of(row)
        symbol = _text(row.get("symbol")).upper()
        if not session or not symbol:
            continue
        key = (session, symbol, _side(row.get("side")))
        if key in kept:
            duplicates += 1
            continue
        kept[key] = row
    ordered = sorted(kept.items(), key=lambda item: (item[0][0], item[0][2], item[0][1]))
    return [row for _key, row in ordered], quick, duplicates


def _grade_liked(
    picks: Sequence[Mapping[str, Any]],
    outcomes: Mapping[tuple[str, str, str], Mapping[str, Any]],
    claims: Mapping[tuple[str, str, str], Mapping[str, Any]],
    overlaps: Mapping[str, set[tuple[str, str, str]]],
    window: tuple[str, str],
) -> list[LikedRow]:
    first, last = window
    out: list[LikedRow] = []
    for row in picks:
        session = _session_of(row)
        if session and not (first <= session <= last):
            continue
        symbol = _text(row.get("symbol")).upper()
        side = _side(row.get("side"))
        key = (session, symbol, side)
        outcome = outcomes.get(key)

        status = "unmeasured"
        reason = "no outcome row"
        win: bool | None = None
        value: float | None = None
        if outcome is not None:
            matured = _matured(outcome.get("matured_horizons"))
            value = _float_or_none(outcome.get("h5_return"))
            if str(int(SWING_HORIZON_SESSIONS)) not in matured:
                status, reason = "pending", ""
            elif value is None:
                status, reason = "unmeasured", "h5_return is present and empty"
            else:
                status, reason = "measured", ""
                win = value > 0

        claim = claims.get(key)
        out.append(
            LikedRow(
                session_date=session,
                symbol=symbol,
                side=side,
                source=_text(row.get("source")),
                status=status,
                win=win,
                h5_return=value if status == "measured" else None,
                unmeasured_reason=reason,
                claimed_setup_id=_text(claim.get("claimed_setup_id")) if claim else "",
                claim_at=_text(claim.get("claim_at")) if claim else "",
                known_at_claim=(claim.get("known_at_claim") if claim else None),
                provenance=PROVENANCE_CLAIMED if claim else PROVENANCE_ANNOTATION_ONLY,
                also_fav=key in overlaps["fav"],
                also_near=key in overlaps["near"],
                also_hc=key in overlaps["hc"],
            )
        )
    return out


def _counts(rows: Sequence[LikedRow]) -> tuple[int, int, int, int]:
    measured = [row for row in rows if row.status == "measured"]
    return (
        len(measured),
        sum(1 for row in measured if row.win),
        sum(1 for row in rows if row.status == "pending"),
        sum(1 for row in rows if row.status == "unmeasured"),
    )


def _rate(wins: int, n: int) -> float | None:
    return (wins / n) if n > 0 else None


def _tracker_population(
    key: str, bucket: str, eligible: Sequence[Mapping[str, Any]], pending_rows: Sequence[Mapping[str, Any]]
) -> Population:
    rows = [row for row in eligible if _text(row.get("priority_bucket")).lower() == bucket]
    wins = sum(1 for row in rows if _is_true(row.get("win")))
    n = len(rows)
    pending = sum(
        1 for row in pending_rows if _text(row.get("priority_bucket")).lower() == bucket
    )
    return Population(
        key=key,
        label=POPULATION_LABELS[key],
        clock=TRACKER_CLOCK,
        n=n,
        wins=wins,
        win_rate=_rate(wins, n),
        bound=wilson_lower_bound(wins, n),
        pending=pending,
        # The tracker's unmeasurable rows are EXCLUSIONS, reported once by the
        # read itself; repeating them per bucket would invent a number.
        unmeasured=None,
        note=HC_NO_ROWS_NOTE if (bucket == BUCKET_HC and n == 0) else "",
    )


def _by_setup(rows: Sequence[LikedRow]) -> tuple[SetupRow, ...]:
    families: dict[str, list[LikedRow]] = {}
    for row in rows:
        families.setdefault(row.source, []).append(row)
    built: list[SetupRow] = []
    for source, family in families.items():
        n, wins, pending, unmeasured = _counts(family)
        built.append(
            SetupRow(
                source=source,
                label=display_setup(source),
                n=n,
                wins=wins,
                win_rate=_rate(wins, n),
                bound=wilson_lower_bound(wins, n),
                pending=pending,
                unmeasured=unmeasured,
            )
        )
    # Sorted by the BOUND - a 100% on two claims is not better than a 70% on
    # thirty - with every ungraded family last and ties broken by name.
    built.sort(key=lambda row: (row.bound is None, -(row.bound or 0.0), row.label))
    return tuple(built)


def _leader_line(by_setup: Sequence[SetupRow]) -> str:
    """Names a setup only at `MIN_REPORTABLE_N`; otherwise says the best n."""
    floor = int(MIN_REPORTABLE_N)
    eligible = [row for row in by_setup if row.n >= floor and row.bound is not None]
    if not eligible:
        best = max((row.n for row in by_setup), default=0)
        return f"no setup has n >= {floor} yet (best n was {best})"
    best_row = max(eligible, key=lambda row: (row.bound or 0.0, row.n, row.label))
    return (
        f"{best_row.label} leads my claims: {(best_row.win_rate or 0.0) * 100:.0f}% win "
        f"rate over n={best_row.n} (at least {(best_row.bound or 0.0) * 100:.0f}%), "
        f"observational and never a promotion."
    )


def _excluded_footnote(excluded: Mapping[str, int]) -> str:
    if not excluded:
        return ""
    ordered = sorted(excluded.items(), key=lambda item: (-item[1], item[0]))
    detail = ", ".join(f"{name} {count}" for name, count in ordered)
    return f"tracker read excluded: {detail}"


def _window_report(
    *,
    window: str,
    dates: tuple[str, str],
    picks: Sequence[Mapping[str, Any]],
    outcomes: Mapping[tuple[str, str, str], Mapping[str, Any]],
    claims: Mapping[tuple[str, str, str], Mapping[str, Any]],
    tier_rows: Sequence[Mapping[str, Any]],
    quick_likes: int,
    duplicates: int,
) -> WindowReport:
    read = read_eligible_rows(tier_rows, POLICY_SCANROW_V1, window=dates)
    overlaps = {
        "fav": _tracker_days(read.rows, BUCKET_FAV),
        "near": _tracker_days(read.rows, BUCKET_NEAR),
        "hc": _tracker_days(read.rows, BUCKET_HC),
    }
    liked_rows = _grade_liked(picks, outcomes, claims, overlaps, dates)
    n, wins, pending, unmeasured = _counts(liked_rows)

    populations = {
        "liked": Population(
            key="liked",
            label=POPULATION_LABELS["liked"],
            clock=LIKE_CLOCK,
            n=n,
            wins=wins,
            win_rate=_rate(wins, n),
            bound=wilson_lower_bound(wins, n),
            pending=pending,
            unmeasured=unmeasured,
            also_fav=sum(1 for row in liked_rows if row.also_fav),
            also_near=sum(1 for row in liked_rows if row.also_near),
            dropped_duplicates=duplicates,
        ),
        "fav": _tracker_population("fav", BUCKET_FAV, read.rows, read.pending),
        "hc": _tracker_population("hc", BUCKET_HC, read.rows, read.pending),
        "near": _tracker_population("near", BUCKET_NEAR, read.rows, read.pending),
    }
    by_setup = _by_setup(liked_rows)
    footnotes = [f"quick likes excluded: {quick_likes}"]
    excluded = _excluded_footnote(read.excluded)
    if excluded:
        footnotes.append(excluded)
    return WindowReport(
        window=window,
        window_dates=dates,
        populations=populations,
        by_setup=by_setup,
        liked_rows=tuple(liked_rows),
        leader=_leader_line(by_setup),
        footnotes=tuple(footnotes),
    )


def build_comparison(
    *,
    like_picks: Iterable[Mapping[str, Any]] = (),
    like_outcomes: Iterable[Mapping[str, Any]] = (),
    tier_rows: Iterable[Mapping[str, Any]] = (),
    claims: Iterable[Mapping[str, Any]] = (),
    as_of: Any = None,
    window: str = WINDOW_LATELY,
) -> Comparison:
    """Grade both sides over both windows. PURE - it opens no file.

    `window` selects which window's numbers sit at the top of the result;
    :attr:`Comparison.by_window` always carries BOTH, so a surface can show two
    column groups off one build.
    """
    asked = str(window or WINDOW_LATELY).strip().lower()
    if asked not in WINDOWS:
        asked = WINDOW_LATELY
    stamp = _as_of_iso(as_of)

    picks, quick_likes, duplicates = _dedupe_picks(like_picks)
    outcomes = _outcome_index(like_outcomes)
    claim_index = _claim_index(claims)
    tier = _rows(tier_rows)

    windows = {
        WINDOW_LATELY: tuple(lately_window(stamp)),
        # `read_eligible_rows(end=)` only moves the RIGHT edge of the lately
        # window, so "all" is an explicit wide window rather than an absent one.
        WINDOW_ALL: (EARLIEST_WINDOW_START, stamp),
    }
    by_window = {
        name: _window_report(
            window=name,
            dates=(dates[0], dates[1]),
            picks=picks,
            outcomes=outcomes,
            claims=claim_index,
            tier_rows=tier,
            quick_likes=quick_likes,
            duplicates=duplicates,
        )
        for name, dates in windows.items()
    }
    chosen = by_window[asked]
    return Comparison(
        as_of=stamp,
        window=asked,
        by_window=by_window,
        # The SAME object, never a copy: a surface that reordered or rebuilt the
        # top-level cells could disagree with the window they came from.
        populations=chosen.populations,
        by_setup=chosen.by_setup,
        liked_rows=chosen.liked_rows,
        leader=chosen.leader,
        footnotes=chosen.footnotes,
        window_dates=chosen.window_dates,
    )


# ---------------------------------------------------------------------------
# The ONE renderer
# ---------------------------------------------------------------------------


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _pending_text(population: Population) -> str:
    if population.key == "liked":
        return str(int(population.pending))
    # `POLICY_SCANROW_V1.immature_value` is empty: a v1 row does not exist until
    # the symbol's later scan row does, so the read never yields a pending one.
    return f"{int(population.pending)} (mature by construction)"


def _unmeasured_text(population: Population) -> str:
    return "-" if population.unmeasured is None else str(int(population.unmeasured))


def render_text(comparison: Comparison) -> str:
    """The whole report as text. The CLI prints exactly this."""
    first, last = comparison.window_dates
    span = (
        f"{first}..{last}"
        if comparison.window == WINDOW_LATELY
        else f"every row through {last}"
    )
    lines = [
        "My claims - the claimed D1 picks graded beside the tracker's own populations",
        f"as of {comparison.as_of} | window: {comparison.window} ({span})",
        "",
        "Two clocks, side by side, never pooled:",
        f"  My liked trades: {LIKE_CLOCK}",
        f"  FAV / HC / Near: {TRACKER_CLOCK}",
        "",
        "Populations",
    ]
    for key in POPULATION_KEYS:
        population = comparison.populations.get(key)
        if population is None:
            continue
        if population.note:
            lines.append(f"  {population.label:<16} {population.note}")
            continue
        lines.append(
            f"  {population.label:<16} {_percent(population.win_rate):>5} win rate  "
            f"n={population.n:<5} wins={population.wins:<5} "
            f"at least {_percent(population.bound)}  "
            f"pending {_pending_text(population)}  "
            f"unmeasured {_unmeasured_text(population)}"
        )
        if key == "liked":
            lines.append(
                f"  {'':<16} of which {int(population.also_fav or 0)} also FAV that day, "
                f"{int(population.also_near or 0)} also Near that day "
                "(named, never added)"
            )
            lines.append(
                f"  {'':<16} repeated clicks folded: "
                f"{int(population.dropped_duplicates or 0)}"
            )
    lines.extend(["", "By claimed setup (sorted by the Wilson lower bound)"])
    if not comparison.by_setup:
        lines.append("  no claimed like has been graded yet")
    for row in comparison.by_setup:
        if row.win_rate is None:
            # Not graded is not a rate of zero. A family whose claims are all
            # still pending has no answer yet, and printing one would invent it.
            lines.append(
                f"  {row.label:<28} not graded yet  n=0     "
                f"pending {row.pending}  unmeasured {row.unmeasured}"
            )
            continue
        lines.append(
            f"  {row.label:<28} {_percent(row.win_rate):>5} win rate  n={row.n:<5} "
            f"wins={row.wins:<5} at least {_percent(row.bound)}  "
            f"pending {row.pending}  unmeasured {row.unmeasured}"
        )
    lines.extend(["", comparison.leader, ""])
    lines.extend(comparison.footnotes)
    lines.append("")
    lines.append(
        "Opportunity results only. Actual trades live in the Journal and the "
        "said-vs-did report; nothing here scores, ranks, gates, alerts or promotes."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="claimed_pick_evidence",
        description=(
            "Grade the trader's claimed D1 picks beside FAV / HC / Near. Reads "
            "four stores and writes nothing."
        ),
    )
    parser.add_argument("--window", choices=list(WINDOWS), default=WINDOW_LATELY)
    parser.add_argument("--as-of", dest="as_of", default=None, help="YYYY-MM-DD")
    parser.add_argument("--picks-path", default=None)
    parser.add_argument("--outcomes-path", default=None)
    parser.add_argument("--tier-path", default=None)
    parser.add_argument("--claims-path", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    inputs = load_inputs(
        picks_path=args.picks_path,
        outcomes_path=args.outcomes_path,
        tier_path=args.tier_path,
        claims_path=args.claims_path,
    )
    comparison = build_comparison(**inputs, as_of=args.as_of, window=args.window)
    print(render_text(comparison))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI tests
    raise SystemExit(main())
