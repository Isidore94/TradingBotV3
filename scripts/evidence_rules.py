"""Reader-side evidence rules — R10 ground rule 5.

**History is never rewritten.** When a store turns out to contain rows that were
recorded honestly but measured wrongly, the rows stay exactly as they are and a
**versioned rule** here tells every reader what is wrong with them. Rules are
identified by name (`daily_volume_mixed_v1`), never edited in place, and never
renumbered: a changed definition is a NEW name, so a report written last month
and a report written today mean the same thing by the same word.

Two properties follow from that and bind every rule in this module:

* **It reads. It never writes.** Nothing here may edit a store, and nothing here
  may reach a detector, score, gate, alert, watchlist or Focus decision. A tag
  is something a rollup PRINTS.
* **Missing data is uncertainty, never confirmation** (plan.md §5). Every rule
  has three answers, not two, and the third — `unknown` — is reported beside n
  rather than folded into the clean bucket.

---

## `daily_volume_mixed_v1`

**What is wrong.** The durable daily-bar store is unit-mixed. IB returns
regular-session volume in **round lots** (`whatToShow="TRADES"`, `useRTH=1`,
`master_avwap_lib/legacy.py:15245-15256`); Yahoo returns the full consolidated
session in **shares**. The observed ratio is symbol-dependent — 1.0× on SPY,
56× on TSLA, 81× on AAPL, 162× on A, 188× on NVDA — so no constant converts one
into the other. AVWAP bands are volume-weighted, so a **splice** between the two
units re-weights every level computed across it: measured at a median ×0.0088
step on 2026-07-29 in 1,179 of 1,236 rewritten files, which froze pre-splice
anchored VWAPs near their 07-28 value and moved replayed targets on 49.6% of
mark-days. Stops did not move (stored at scan time, never replayed). Full
measurement: `docs/analysis/DAILY_BAR_VOLUME_CLIFF_2026-08-22.md`.

**How the tag is derived.** From the run manifests' own
`provider.daily_bars.success.<provider>` counters — evidence the scans already
wrote about themselves — and not from a hard-coded date list. A session where
any run reported a non-`yahoo` daily-bar success is `mixed`; a session whose
runs all reported `yahoo` only is `shares`; anything else is `unknown`.

**`mixed` dominates.** One proven IB run contaminates that session's store
whatever the other runs did — including the two concurrent desks of 2026-08-20,
where one used IB and one did not. **`unknown` beats `shares`**: a manifest we
cannot read may have been the IB one.

**The known limit, stated rather than hidden.** `run_manifest.DEFAULT_KEEP` is
90, so manifest coverage is a rolling window and everything older reads
`unknown` — and reads `unknown` *increasingly* as time passes. A rollup that
must stay reproducible should `freeze_verdicts()` alongside its output rather
than re-deriving them later. R10.V's backfill manifest is where the durable
answer will live.

The interim measure that stops the store getting more mixed is the
`daily_bars_source` pin (`master_avwap_lib.daily_bars_source_pin`); this rule
describes the rows written before it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

# ---------------------------------------------------------------------------
# rule names — versioned, never edited in place
# ---------------------------------------------------------------------------
RULE_DAILY_VOLUME_MIXED = "daily_volume_mixed_v1"
RULE_H1_BAR_START = "h1_bar_start_v1"
RULE_FABRICATED_ZERO = "fabricated_zero_v1"
RULE_DUPLICATE_ROW = "duplicate_row_v1"
RULE_RISK_BELOW_FLOOR = "risk_below_floor_v1"

# verdicts
VERDICT_MIXED = "mixed"
VERDICT_SHARES = "shares"
VERDICT_UNKNOWN = "unknown"

_DAILY_BAR_SUCCESS_PREFIX = "provider.daily_bars.success."
_SHARE_DENOMINATED_PROVIDERS = frozenset({"yahoo", "yfinance"})


@dataclass(frozen=True)
class RuleSpec:
    """What a rule means, for a reader who meets its name in a report."""

    name: str
    summary: str
    applies_to: str
    introduced: str
    precision: str


RULES: Mapping[str, RuleSpec] = {
    RULE_DAILY_VOLUME_MIXED: RuleSpec(
        name=RULE_DAILY_VOLUME_MIXED,
        summary=(
            "The durable daily-bar store mixes IB round-lot regular-session volume with "
            "Yahoo consolidated share volume. AVWAP is volume-weighted, so levels replayed "
            "across the splice moved; stops, stored at scan time, did not."
        ),
        applies_to=(
            "anything derived from D1 AVWAP levels on a tagged session - D1 alerts, "
            "setup-tracker marks and scenario outcomes, Focus adoptions"
        ),
        introduced="2026-08-22 (R10.0b)",
        precision=(
            "derived from run manifests, which are pruned to 90 runs; sessions outside "
            "that window are unknown, not clean"
        ),
    ),
    RULE_H1_BAR_START: RuleSpec(
        name=RULE_H1_BAR_START,
        summary=(
            "An `h1_`-family row whose `entry_time` minute is exactly 30 carries the "
            "BAR START, not the signal time. An H1 bar in PT starts at :30."
        ),
        applies_to="any entry-timing statistic over the intraday outcome store",
        introduced="2026-08-22 (R10.0 decision 5)",
        precision=(
            "conjunctive - family AND minute. 9,623 of 9,914 minute-30 rows are H1; "
            "291 of 6,054 non-H1 rows also land on minute 30, so the family half is "
            "load-bearing and the minute alone does not discriminate"
        ),
    ),
    RULE_FABRICATED_ZERO: RuleSpec(
        name=RULE_FABRICATED_ZERO,
        summary=(
            "A final whose `close_r` is exactly 0 and whose `eod_close` equals its "
            "entry price was never measured - the close was assigned from the entry "
            "rather than observed."
        ),
        applies_to="any R statistic over the intraday outcome store",
        introduced="2026-08-22 (R10.0 D2)",
        precision=(
            "1,164 of 1,164 zero finals match on 2026-07-24..08-21 and 0 of 5,743 "
            "non-zero finals do; 251 never advanced a bar and 563 are stop-hits "
            "recorded as 0R"
        ),
    ),
    RULE_DUPLICATE_ROW: RuleSpec(
        name=RULE_DUPLICATE_ROW,
        summary=(
            "The same `event_id` appears more than once for one `event_type`. The "
            "extra rows are counted, never deleted, and every count carries its window."
        ),
        applies_to="any count over the intraday outcome store",
        introduced="2026-08-22 (R10.0 D1d)",
        precision=(
            "window-dependent and therefore always stated: 742 extra `registered` over "
            "609 ids and 430 extra `final` on 2026-07-24..08-21; 394 / 345 / 300 on "
            "2026-08-07..08-21. Concurrency is NOT the main cause - 0 of 609 duplicated "
            "ids were written within 5 s of each other"
        ),
    ),
    RULE_RISK_BELOW_FLOOR: RuleSpec(
        name=RULE_RISK_BELOW_FLOOR,
        summary=(
            "Risk per share below 0.1% of entry. R is a ratio, so a penny stop turns "
            "an ordinary move into a three-figure R."
        ),
        applies_to="R statistics; the raw `risk_per_share` and `stop_price` are never edited",
        introduced="2026-08-22 (R10.0 decision 6, R9.3's floor)",
        precision="1,127 all-time finals qualify; all-time max |close_r| is 799.0",
    ),
}

# R9.3's floor, reconciled as the ONE analytic definition (R10.0 decision 6).
RISK_FLOOR_PCT_OF_ENTRY = 0.1
# An H1 bar in PT starts at :30, which is what makes the minute half of the rule
# work at all.
H1_BAR_START_MINUTE = 30
H1_FAMILY_PREFIX = "h1_"


@dataclass(frozen=True)
class EvidenceTag:
    """One rule's answer about one thing, in the form a rollup prints."""

    rule: str
    verdict: str
    reason: str

    @property
    def tagged(self) -> bool:
        """True when the rule fired. `unknown` does NOT count as tagged-clean."""
        return self.verdict == VERDICT_MIXED


@dataclass(frozen=True)
class DailyBarRun:
    """One scan's own account of where its daily bars came from."""

    run_id: str
    job_type: str
    started_at: datetime
    session_date: date
    yahoo: int
    non_yahoo: Mapping[str, int]

    @property
    def verdict(self) -> str:
        if self.non_yahoo:
            return VERDICT_MIXED
        if self.yahoo:
            return VERDICT_SHARES
        return VERDICT_UNKNOWN


def default_manifest_dir() -> Path:
    """The same directory the scans write to — one owner, one location."""
    try:
        from diagnostics.run_manifest import default_manifest_dir as _dir

        return _dir()
    except Exception:
        return Path.home() / ".tradingbotv3" / "run_manifests"


def _market_tz():
    try:
        from market_calendar import MARKET_TZ

        return MARKET_TZ
    except Exception:  # pragma: no cover - zoneinfo is stdlib on 3.12
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/New_York")


def _session_date(started_at: datetime) -> date:
    """Market-local calendar date. `astimezone`, never `replace(tzinfo=None)`."""
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    return started_at.astimezone(_market_tz()).date()


def scan_daily_bar_runs(manifest_dir: Path | str | None = None) -> tuple[DailyBarRun, ...]:
    """Every manifest that says anything about daily-bar providers, oldest first.

    A manifest that fetched no daily bars is not here at all — it votes on
    nothing. A manifest we cannot read is not here either; `unreadable_sessions`
    is how that absence is reported, because dropping it silently would turn a
    gap into a clean bill of health.
    """
    runs, _ = _scan(manifest_dir)
    return runs


def unreadable_sessions(manifest_dir: Path | str | None = None) -> frozenset[date]:
    """Sessions holding a manifest that could not be parsed."""
    _, unreadable = _scan(manifest_dir)
    return unreadable


def _scan(manifest_dir: Path | str | None) -> tuple[tuple[DailyBarRun, ...], frozenset[date]]:
    directory = Path(manifest_dir) if manifest_dir is not None else default_manifest_dir()
    try:
        entries = sorted(directory.glob("*.json"))
    except OSError:
        return (), frozenset()

    runs: list[DailyBarRun] = []
    unreadable: set[date] = set()
    for path in entries:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("manifest is not an object")
            started_raw = str(payload.get("started_at") or "")
            started_at = datetime.fromisoformat(started_raw)
        except Exception:
            unreadable.add(_unreadable_session_hint(path))
            continue

        counters = payload.get("counters")
        if not isinstance(counters, dict):
            counters = {}
        yahoo = 0
        others: dict[str, int] = {}
        for key, value in counters.items():
            if not str(key).startswith(_DAILY_BAR_SUCCESS_PREFIX):
                continue
            provider = str(key)[len(_DAILY_BAR_SUCCESS_PREFIX):].strip()
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue
            if provider.lower() in _SHARE_DENOMINATED_PROVIDERS:
                yahoo += count
            else:
                others[provider] = others.get(provider, 0) + count
        if not yahoo and not others:
            continue
        runs.append(
            DailyBarRun(
                run_id=str(payload.get("run_id") or path.stem),
                job_type=str(payload.get("job_type") or ""),
                started_at=started_at,
                session_date=_session_date(started_at),
                yahoo=yahoo,
                non_yahoo=dict(others),
            )
        )
    runs.sort(key=lambda run: run.started_at)
    return tuple(runs), frozenset(unreadable)


def _unreadable_session_hint(path: Path) -> date | None:
    """Best effort at which session an unparseable manifest belonged to.

    Run ids embed a UTC stamp (`master_scan-20260821T140030Z-...`). If even that
    is unreadable the session is `None`, which taints nothing in particular and
    is reported on its own.
    """
    stem = path.stem
    for chunk in stem.split("-"):
        if len(chunk) == 16 and chunk.endswith("Z") and chunk[8:9] == "T":
            try:
                stamp = datetime.strptime(chunk, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            return _session_date(stamp)
    return None


def daily_volume_session_verdicts(
    manifest_dir: Path | str | None = None,
) -> dict[date, str]:
    """`{session_date: verdict}` for every session the manifests cover.

    A session absent from this mapping is **unknown**, not clean — callers must
    go through `daily_volume_mixed_v1`, which defaults correctly.
    """
    runs, unreadable = _scan(manifest_dir)
    verdicts: dict[date, str] = {}
    for run in runs:
        current = verdicts.get(run.session_date)
        verdicts[run.session_date] = _dominant(current, run.verdict)
    for session in unreadable:
        if session is None:
            continue
        verdicts[session] = _dominant(verdicts.get(session), VERDICT_UNKNOWN)
    return dict(sorted(verdicts.items()))


def _dominant(current: str | None, incoming: str) -> str:
    """mixed > unknown > shares.

    `mixed` wins because one proven IB run contaminates the session whatever
    else ran. `unknown` beats `shares` because the run we could not read may
    have been the IB one — the same reason UNKNOWN never reads as a pass
    anywhere else in this system.
    """
    order = {VERDICT_SHARES: 0, VERDICT_UNKNOWN: 1, VERDICT_MIXED: 2}
    if current is None:
        return incoming
    return current if order[current] >= order[incoming] else incoming


def daily_volume_mixed_v1(
    session_date: date,
    verdicts: Mapping[date, str] | None = None,
    manifest_dir: Path | str | None = None,
) -> EvidenceTag:
    """Tag one market session. Never raises; an unmeasurable session is UNKNOWN.

    Pass `verdicts` (from `daily_volume_session_verdicts`) when tagging many
    rows — re-deriving per row would re-read the whole manifest directory.
    """
    if verdicts is None:
        verdicts = daily_volume_session_verdicts(manifest_dir)
    verdict = verdicts.get(session_date, VERDICT_UNKNOWN)
    if verdict == VERDICT_MIXED:
        reason = (
            "a scan on this session wrote IB (ibkr) round-lot daily volume into the "
            "durable store; AVWAP levels replayed across the splice moved"
        )
    elif verdict == VERDICT_SHARES:
        reason = "every daily-bar fetch on this session reported Yahoo share volume"
    else:
        reason = (
            "no readable run manifest covers this session, so the daily-bar unit is "
            "unmeasured - not clean"
        )
    return EvidenceTag(rule=RULE_DAILY_VOLUME_MIXED, verdict=verdict, reason=reason)


def daily_volume_tag_counts(
    session_dates: Iterable[date],
    verdicts: Mapping[date, str] | None = None,
    manifest_dir: Path | str | None = None,
) -> dict[str, int]:
    """Count rows by verdict, so a rollup can print them beside n."""
    if verdicts is None:
        verdicts = daily_volume_session_verdicts(manifest_dir)
    counts = {VERDICT_MIXED: 0, VERDICT_SHARES: 0, VERDICT_UNKNOWN: 0}
    for session in session_dates:
        counts[verdicts.get(session, VERDICT_UNKNOWN)] += 1
    return {key: value for key, value in counts.items() if value}


def format_tag_counts(counts: Mapping[str, int], rule: str = RULE_DAILY_VOLUME_MIXED) -> str:
    """`n=412 (daily_volume_mixed_v1: 118 mixed, 30 unknown)`.

    Silent when nothing is tagged and nothing is unmeasured — the caption exists
    to report a qualification, not to decorate every clean line.
    """
    total = sum(int(value) for value in counts.values())
    flagged = [
        f"{counts[key]} {key}"
        for key in (VERDICT_MIXED, VERDICT_UNKNOWN)
        if counts.get(key)
    ]
    if not flagged:
        return f"n={total}"
    return f"n={total} ({rule}: {', '.join(flagged)})"


def freeze_verdicts(verdicts: Mapping[date, str]) -> dict[str, str]:
    """JSON-serialisable form, for filing beside a rollup that must reproduce.

    Manifests are pruned; a verdict re-derived in six weeks will read `unknown`
    where today it reads `shares`. Freeze what a published number relied on.
    """
    return {session.isoformat(): verdict for session, verdict in sorted(verdicts.items())}


def thaw_verdicts(frozen: Mapping[str, str]) -> dict[date, str]:
    """Inverse of `freeze_verdicts`; unparseable keys are dropped, not guessed."""
    out: dict[date, str] = {}
    for key, value in frozen.items():
        try:
            session = date.fromisoformat(str(key))
        except ValueError:
            continue
        if value in (VERDICT_MIXED, VERDICT_SHARES, VERDICT_UNKNOWN):
            out[session] = value
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# R10.A rules over the intraday outcome store
# ---------------------------------------------------------------------------
def _as_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def family_from_event_id(event_id: str | None) -> str:
    """`AAPL_long_20260724_06_30_00_h1_blue_after_red` -> `h1_blue_after_red`.

    The outcome CSV does **not** carry a `family` column - it is derived from the
    id, which is `SYMBOL_side_YYYYMMDD_HH_MM_SS_<type>`. A caller that forgets
    this measures every H1 row as non-H1 and the rule silently tags nothing,
    which is exactly what happened the first time I validated it against the
    live store. Splitting on `_` is safe because symbols use `-` for class
    shares (`BRK-B`), never `_`.
    """
    parts = str(event_id or "").split("_")
    return "_".join(parts[6:]) if len(parts) > 6 else ""


def h1_bar_start_v1(
    family: str | None, entry_time: str | None, *, event_id: str | None = None
) -> EvidenceTag:
    """Does this row's `entry_time` carry a bar START rather than a signal time?

    Conjunctive on purpose. The family alone is not enough - 291 of 6,054 non-H1
    rows also land on minute 30 - and the minute alone is not enough either. A
    stamp we cannot read is UNKNOWN, never "fine".

    Pass `event_id` when the row has no `family` column; the store's rows do not.
    """
    name = str(family or "").strip().lower()
    if not name and event_id:
        name = family_from_event_id(event_id).strip().lower()
    minute = _entry_minute(entry_time)
    if minute is None:
        return EvidenceTag(
            rule=RULE_H1_BAR_START,
            verdict=VERDICT_UNKNOWN,
            reason="the entry stamp could not be read, so its minute is unmeasured",
        )
    if name.startswith(H1_FAMILY_PREFIX) and minute == H1_BAR_START_MINUTE:
        return EvidenceTag(
            rule=RULE_H1_BAR_START,
            verdict=VERDICT_MIXED,
            reason=(
                "an H1-family row stamped on the half hour: this is the bar start, not "
                "the signal time, so it must not enter an entry-timing statistic"
            ),
        )
    return EvidenceTag(
        rule=RULE_H1_BAR_START,
        verdict=VERDICT_SHARES,
        reason="not an H1-family row stamped on the half hour",
    )


def _entry_minute(entry_time: str | None) -> int | None:
    raw = str(entry_time or "").strip()
    if not raw or ":" not in raw:
        # A date with no time carries no minute. Reading it as minute 0 would
        # turn an unmeasured stamp into a measured one.
        return None
    for parser in (datetime.fromisoformat,):
        try:
            return parser(raw).minute
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(raw, fmt).minute
        except ValueError:
            continue
    return None


def fabricated_zero_v1(
    *, close_r, eod_close, entry_price, tolerance: float = 1e-9
) -> EvidenceTag:
    """A 0R final whose close was assigned from its entry rather than observed.

    The signature is exact on the measured window - 1,164 of 1,164 zero finals
    match and 0 of 5,743 non-zero finals do - so a genuine scratch, which closes
    somewhere other than exactly its entry, is not caught by it.
    """
    close = _as_float(close_r)
    eod = _as_float(eod_close)
    entry = _as_float(entry_price)
    if close is None or eod is None or entry is None:
        return EvidenceTag(
            rule=RULE_FABRICATED_ZERO,
            verdict=VERDICT_UNKNOWN,
            reason="close_r, eod_close or entry_price is missing, so this cannot be judged",
        )
    if abs(close) <= tolerance and abs(eod - entry) <= tolerance:
        return EvidenceTag(
            rule=RULE_FABRICATED_ZERO,
            verdict=VERDICT_MIXED,
            reason=(
                "a 0R final whose close equals its entry exactly - the close was never "
                "measured, it was assigned, so this row is not a scratch"
            ),
        )
    return EvidenceTag(
        rule=RULE_FABRICATED_ZERO,
        verdict=VERDICT_SHARES,
        reason="the close differs from the entry, so it was observed",
    )


def risk_below_floor_v1(*, risk_per_share, entry_price) -> EvidenceTag:
    """Risk under 0.1% of entry - R9.3's floor, and the only analytic one."""
    risk = _as_float(risk_per_share)
    entry = _as_float(entry_price)
    if risk is None or entry is None or entry <= 0:
        return EvidenceTag(
            rule=RULE_RISK_BELOW_FLOOR,
            verdict=VERDICT_UNKNOWN,
            reason="risk or entry is missing, so the floor cannot be applied",
        )
    floor = entry * RISK_FLOOR_PCT_OF_ENTRY / 100.0
    if abs(risk) < floor:
        return EvidenceTag(
            rule=RULE_RISK_BELOW_FLOOR,
            verdict=VERDICT_MIXED,
            reason=(
                f"risk {abs(risk):.4f} is under {RISK_FLOOR_PCT_OF_ENTRY}% of entry "
                f"({floor:.4f}); R is a ratio, so this row's R is an artifact of its stop"
            ),
        )
    return EvidenceTag(
        rule=RULE_RISK_BELOW_FLOOR,
        verdict=VERDICT_SHARES,
        reason="risk is at or above the floor",
    )


def duplicate_row_v1(rows: Iterable[Mapping], *, window: str) -> dict:
    """Count repeated `(event_id, event_type)` pairs. Nothing is deleted.

    `window` is REQUIRED and is echoed back, because every count from this store
    is window-dependent and the same allegation reproduced at 742 on one window
    and 394 on another. A number that travels without its window is not evidence.
    """
    if not str(window or "").strip():
        raise ValueError("duplicate_row_v1 needs the window its counts were taken over")
    seen: dict[tuple[str, str], int] = {}
    total = 0
    without_id = 0
    for row in rows:
        total += 1
        event_id = str((row or {}).get("event_id") or "").strip()
        event_type = str((row or {}).get("event_type") or "").strip()
        if not event_id:
            without_id += 1
            continue
        key = (event_id, event_type)
        seen[key] = seen.get(key, 0) + 1
    by_type: dict[str, dict[str, int]] = {}
    duplicate_ids: set[str] = set()
    for (event_id, event_type), count in seen.items():
        if count <= 1:
            continue
        bucket = by_type.setdefault(event_type, {"extra_rows": 0, "ids": 0})
        bucket["extra_rows"] += count - 1
        bucket["ids"] += 1
        duplicate_ids.add(event_id)
    return {
        "rule": RULE_DUPLICATE_ROW,
        "window": window,
        "rows": total,
        "rows_without_id": without_id,
        "duplicate_ids": len(duplicate_ids),
        "by_event_type": by_type,
    }


def describe(rule: str) -> str:
    """One paragraph about a rule, for a report footer."""
    spec = RULES.get(rule)
    if spec is None:
        return f"{rule}: unknown rule"
    return (
        f"{spec.name} - {spec.summary} Applies to: {spec.applies_to}. "
        f"Introduced {spec.introduced}. Precision: {spec.precision}."
    )


def _main(argv: list[str]) -> int:
    """`python scripts/evidence_rules.py` - what the manifests currently say."""
    verdicts = daily_volume_session_verdicts()
    if not verdicts:
        print("no readable run manifests; every session is unknown")
        return 0
    counts: dict[str, int] = {}
    for verdict in verdicts.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    print(describe(RULE_DAILY_VOLUME_MIXED))
    print()
    for session, verdict in verdicts.items():
        print(f"  {session.isoformat()}  {verdict}")
    print()
    print(f"sessions covered: {len(verdicts)} - " + ", ".join(
        f"{count} {name}" for name, count in sorted(counts.items())
    ))
    print("sessions outside the manifest window read unknown, not clean")
    return 0


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(_main(sys.argv[1:]))
