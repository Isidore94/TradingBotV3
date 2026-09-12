"""Packet WS-FC1 - a forming candle never reaches the daily-bar cache.

The per-symbol daily-bar cache (``%LOCALAPPDATA%\\TradingBotV3\\machine_cache\\
daily_bars``, one CSV per symbol) is written by the scan while the session is
still open. Yahoo hands back today's PARTIAL bar during the session, the writer
stored it verbatim, and nothing replaced it afterwards, so on 2026-09-12 **66 of
1,988 cache files ended in a candle that is not possible** - e.g. ``ADC
2026-09-11 O=71.870 H=71.805 L=70.970 C=71.230``, an open outside its own range
because the day had not yet traded up to it. Every reader of that file (the D1
indicators, the band history, the SMA floors, the session-horizon outcomes)
reads it as a real close, against ``plan.md`` sec 5: *state transitions use
completed bars only; a forming bar is a labelled preview.*

This module is the guard, and it lives here rather than in ``legacy.py`` so the
ask-first file's diff stays at the writer seam the trader's FC1 prompt named.

Two rules, in this order, both of them refusals:

1. **Completed session only.** A daily bar dated ``D`` may be stored once the
   exchange session for ``D`` has closed. The comparison is made in exchange
   time with ``astimezone`` - never ``replace(tzinfo=None)`` - which is
   ``scripts/completed_bars.py``'s rule stated for a session-length bar.
   ``market_calendar`` deliberately does not model early closes (its docstring
   says so): every session is judged against 16:00 ET, which is conservative in
   the only direction that matters - a half-day's bar is called complete at
   16:00 rather than at 13:00, so a forming bar is never called finished.
2. **A possible candle.** ``low <= open, close <= high``. Completion alone does
   not make a candle real: MCW 2026-06-08 and TERN 2026-05-15 are months old and
   still impossible.

A row that is BOTH is counted ONCE, as forming, because forming is the cause.
That is what makes ``kept + forming_dropped + invalid_dropped == fetched`` hold
exactly, which is the reconciliation the run manifest publishes under
``daily_bars_forming_dropped`` / ``daily_bars_invalid_dropped``.

``repair`` is the one-off for files already on disk::

    cd scripts
    python -m master_avwap_lib.daily_bar_cache repair                 # dry run
    python -m master_avwap_lib.daily_bar_cache repair --apply         # writes

It prints the home folder and the cache directory before it looks at anything,
refuses a target under :data:`PROTECTED_DATA_ROOT`, reads every cache file, and
for a file whose LAST row is forming or impossible refetches that session
through the desk's pinned Yahoo path and writes the file temp-and-rename. Only
the LAST row is ever touched: an interior oddity is a data question this tool
does not get to answer. Running it while the session is open replaces nothing
for today - the refetched bar would itself be forming - and says so in the
report.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from threading import Lock

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:  # `python -m master_avwap_lib.daily_bar_cache`
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_calendar  # noqa: E402
from project_paths import DAILY_BARS_CACHE_DIR  # noqa: E402

#: The live home folder. The repair never writes anything under it. A module
#: constant so the refusal can be proven against a scratch tree
#: (``tracker_execution_compare.py:56``, the 2026-09-05 scratch-script rule).
PROTECTED_DATA_ROOT = Path(r"C:\TradingBotData")

#: Run-manifest counter names. The gate greps the log line for both.
FORMING_DROPPED_COUNTER = "daily_bars_forming_dropped"
INVALID_DROPPED_COUNTER = "daily_bars_invalid_dropped"

#: The floor on how much history the repair asks the provider for. The window
#: is widened to REACH the bad session - MCW's bad row is 2026-06-08 and a
#: ten-day window answers "the provider has no bar for that session", which is a
#: measurement of the request, not of Yahoo.
REPAIR_REFETCH_DAYS = 10

#: Calendar days of slack added past the bad session so the window contains it.
REPAIR_REFETCH_BUFFER_DAYS = 5

_PRICE_COLUMNS = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# the clock
# ---------------------------------------------------------------------------
def market_now() -> datetime:
    """Now, AWARE. The one clock hook in this module, so a test can freeze it.

    Aware on purpose: the session close is judged by converting through the
    offset, so 13:05 Pacific is recognised as 16:05 in New York.
    """
    try:
        from market_session import get_market_local_now

        moment = get_market_local_now()
    except Exception:  # pragma: no cover - settings/zoneinfo trouble
        moment = datetime.now().astimezone()
    return moment if moment.tzinfo is not None else moment.astimezone()


def _aware(moment: datetime | None) -> datetime:
    if moment is None:
        return market_now()
    if moment.tzinfo is None:
        # A naive stamp is the trader's own wall clock; attach it rather than
        # pretending it already is exchange time.
        return moment.astimezone()
    return moment


# ---------------------------------------------------------------------------
# the two rules
# ---------------------------------------------------------------------------
def last_completed_session(now: datetime | None = None) -> date:
    """The newest session whose close is at or before ``now`` (exchange time).

    One calendar call per frame instead of one per row, which is what lets the
    filter sit on the scan's hot path. Falls back to a direct 16:00 ET
    comparison when the calendar refuses (a date outside its validated range).
    """
    moment = _aware(now).astimezone(market_calendar.MARKET_TZ)
    try:
        return market_calendar.last_completed_session(moment)
    except Exception:
        today = moment.date()
        if moment >= datetime.combine(today, time(16, 0), tzinfo=market_calendar.MARKET_TZ):
            return today
        return today - timedelta(days=1)


def session_is_complete(day: date | None, now: datetime | None = None) -> bool:
    """Has the exchange session dated ``day`` finished?"""
    if day is None:
        return False
    return day <= last_completed_session(now)


def candle_is_possible(row) -> bool:
    """``low <= open, close <= high``, the invariant the charts already draw by.

    A row with an unreadable price is NOT possible - missing data is
    uncertainty, never confirmation.
    """
    try:
        values = {name: float(row[name]) for name in _PRICE_COLUMNS}
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    if any(value != value for value in values.values()):  # NaN
        return False
    low, high = values["low"], values["high"]
    return low <= values["open"] <= high and low <= values["close"] <= high


# ---------------------------------------------------------------------------
# the counts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DropCounts:
    """What one write offered and what it was allowed to keep.

    ``kept + forming_dropped + invalid_dropped == fetched`` by construction.
    """

    fetched: int = 0
    kept: int = 0
    forming_dropped: int = 0
    invalid_dropped: int = 0

    @property
    def dropped(self) -> int:
        return int(self.forming_dropped) + int(self.invalid_dropped)


_TOTALS_LOCK = Lock()
_TOTALS = {"forming": 0, "invalid": 0}


def begin_run() -> None:
    """Open a fresh per-scan bucket (the ``provider_counters.begin_run`` shape).

    Process-wide rather than thread-local on purpose: the scan refreshes daily
    bars from worker threads, and a thread-local total would publish whatever
    the manifest thread happened to do itself.
    """
    with _TOTALS_LOCK:
        _TOTALS["forming"] = 0
        _TOTALS["invalid"] = 0


def run_totals() -> tuple[int, int]:
    with _TOTALS_LOCK:
        return int(_TOTALS["forming"]), int(_TOTALS["invalid"])


def record_drops(counts: DropCounts) -> None:
    with _TOTALS_LOCK:
        _TOTALS["forming"] += int(counts.forming_dropped)
        _TOTALS["invalid"] += int(counts.invalid_dropped)


def flush_to_manifest(recorder=None) -> tuple[int, int]:
    """One INFO line per scan naming both counters, and both onto the manifest.

    Always both, even at zero: "the guard ran and refused nothing" is the
    reading the gate needs, and a counter that only appears on a bad day cannot
    be checked on a good one.
    """
    forming, invalid = run_totals()
    logging.info(
        "daily-bar cache guard: %s=%d %s=%d (this scan)",
        FORMING_DROPPED_COUNTER,
        forming,
        INVALID_DROPPED_COUNTER,
        invalid,
    )
    if recorder is not None:
        try:
            recorder.set_counter(FORMING_DROPPED_COUNTER, forming)
            recorder.set_counter(INVALID_DROPPED_COUNTER, invalid)
        except Exception:  # pragma: no cover - diagnostics never break a scan
            logging.debug("daily-bar drop counters could not be recorded.", exc_info=True)
    return forming, invalid


# ---------------------------------------------------------------------------
# the filter the writer seam calls
# ---------------------------------------------------------------------------
def _row_date(stamp) -> date | None:
    try:
        value = pd.Timestamp(stamp)
    except (ValueError, TypeError):
        return None
    if value is None or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # pragma: no cover - exotic stamp
        return None
    return value.date()


def filter_writable_rows(
    frame: pd.DataFrame | None,
    *,
    symbol: str = "",
    now: datetime | None = None,
) -> tuple[pd.DataFrame, DropCounts]:
    """The rows of ``frame`` that may be stored, and what was refused.

    Never raises. A guard that can break the scan's cache write is a worse
    failure than the row it exists to catch: a cache that silently stops
    updating is invisible, while a row that slips through is counted, logged and
    repairable. So an unexpected internal error refuses NOTHING and says so at
    WARNING.
    """
    try:
        return _filter_writable_rows(frame, symbol=symbol, now=now)
    except Exception:  # pragma: no cover - the guard must not cost the write
        logging.warning(
            "%s: the daily-bar write guard failed; the frame was written unfiltered.",
            str(symbol or "?").strip().upper() or "?",
            exc_info=True,
        )
        rows = int(len(frame)) if isinstance(frame, pd.DataFrame) else 0
        return frame, DropCounts(fetched=rows, kept=rows)


def _filter_writable_rows(
    frame: pd.DataFrame | None,
    *,
    symbol: str = "",
    now: datetime | None = None,
) -> tuple[pd.DataFrame, DropCounts]:
    """The rule itself.

    The frame comes back in its original order with its ``attrs`` intact, so the
    caller's provenance stamp survives, and the frame OBJECT is returned
    unchanged when nothing was dropped - the common case must not cost a copy.
    """
    if frame is None or getattr(frame, "empty", True) or "datetime" not in frame.columns:
        empty = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        return empty, DropCounts()

    moment = _aware(now)
    cutoff = pd.Timestamp(last_completed_session(moment))
    name = str(symbol or "").strip().upper() or "?"

    # Vectorised: this runs once per symbol per scan over a thousand symbols, so
    # the common case (nothing dropped) must not walk the rows in Python. Only
    # the refused rows are iterated, for their DEBUG line.
    stamps = pd.to_datetime(frame["datetime"], errors="coerce")
    if stamps.dtype == object:
        # Mixed offsets in one column: normalise THROUGH utc, never by stripping.
        stamps = pd.to_datetime(stamps, errors="coerce", utc=True)
    if getattr(stamps.dt, "tz", None) is not None:
        # astimezone, then drop - never `replace(tzinfo=None)` on an aware stamp.
        stamps = stamps.dt.tz_convert(market_calendar.MARKET_TZ).dt.tz_localize(None)
    days = stamps.dt.normalize()
    undated = stamps.isna()

    prices = {
        column: pd.to_numeric(frame[column], errors="coerce")
        if column in frame.columns
        else pd.Series(float("nan"), index=frame.index)
        for column in _PRICE_COLUMNS
    }
    possible = (
        (prices["low"] <= prices["open"])
        & (prices["open"] <= prices["high"])
        & (prices["low"] <= prices["close"])
        & (prices["close"] <= prices["high"])
    )

    forming_mask = (~undated) & (days > cutoff)
    invalid_mask = undated | ((~undated) & (~forming_mask) & (~possible))
    keep_mask = ~(forming_mask | invalid_mask)

    counts = DropCounts(
        fetched=int(len(frame)),
        kept=int(keep_mask.sum()),
        forming_dropped=int(forming_mask.sum()),
        invalid_dropped=int(invalid_mask.sum()),
    )
    if not counts.dropped:
        return frame, counts

    # Positional on purpose: a merged frame can carry duplicate index labels,
    # and a label lookup would answer with a Series rather than a row.
    for position, dropped in enumerate(forming_mask.tolist()):
        if not dropped:
            continue
        logging.debug(
            "%s: daily-bar row %s dropped from the cache - the session has not closed "
            "(last completed session %s).",
            name,
            days.iat[position].date().isoformat(),
            cutoff.date().isoformat(),
        )
    for position, dropped in enumerate(invalid_mask.tolist()):
        if not dropped:
            continue
        stamp = days.iat[position]
        row = frame.iloc[position]
        logging.debug(
            "%s: daily-bar row %s dropped from the cache - impossible candle "
            "O=%s H=%s L=%s C=%s.",
            name,
            "(unreadable date)" if pd.isna(stamp) else stamp.date().isoformat(),
            row.get("open"),
            row.get("high"),
            row.get("low"),
            row.get("close"),
        )

    record_drops(counts)
    kept_frame = frame.loc[keep_mask.to_numpy()].copy()
    kept_frame.attrs.update(dict(frame.attrs))
    return kept_frame, counts


# ---------------------------------------------------------------------------
# the repair
# ---------------------------------------------------------------------------
def _is_under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return False
    return True


def _price_text(value) -> str:
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return "?"


def _row_text(row) -> str:
    return " ".join(
        f"{letter}={_price_text(row.get(name))}"
        for letter, name in (("O", "open"), ("H", "high"), ("L", "low"), ("C", "close"))
    )


@dataclass
class RepairFinding:
    """One cache file whose last row is not a bar that could have happened."""

    symbol: str
    path: Path
    day: date | None
    old_text: str
    new_text: str
    reason: str
    repaired: bool = False


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Temp file beside the target, then one rename. A half-written cache file
    is exactly the corruption this tool exists to remove."""
    temp = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temp, index=False)
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - the rename already moved it
            pass


def _repair_one_file(path: Path, *, apply: bool, now: datetime) -> RepairFinding | None:
    from master_avwap_lib import legacy

    try:
        frame = pd.read_csv(path, parse_dates=["datetime"])
    except Exception as exc:
        logging.debug("%s: repair could not read the cache file (%s).", path.name, exc)
        return None
    if frame is None or frame.empty:
        return None

    last = frame.iloc[-1]
    day = _row_date(last.get("datetime"))
    forming = day is not None and not session_is_complete(day, now)
    if day is not None and not forming and candle_is_possible(last):
        return None  # a healthy file is never refetched and never rewritten

    symbol = path.stem.upper()
    reason = "forming" if forming else "impossible candle"
    if day is None:
        reason = "unreadable session date"
    finding = RepairFinding(
        symbol=symbol,
        path=path,
        day=day,
        old_text=_row_text(last),
        new_text="(removed, no replacement available)",
        reason=reason,
    )

    replacement = None
    if day is not None:
        fetched = None
        span = (now.astimezone(market_calendar.MARKET_TZ).date() - day).days
        window = max(REPAIR_REFETCH_DAYS, span + REPAIR_REFETCH_BUFFER_DAYS)
        try:
            fetched = legacy._normalize_daily_bar_frame(
                legacy.fetch_daily_bars_from_yahoo(symbol, window)
            )
        except Exception as exc:
            finding.new_text = f"(refetch failed: {exc})"
        if fetched is not None and not fetched.empty:
            for _, candidate in fetched.iterrows():
                if _row_date(candidate.get("datetime")) != day:
                    continue
                if not session_is_complete(day, now):
                    finding.new_text = "(session still open - rerun after the close)"
                elif not candle_is_possible(candidate):
                    finding.new_text = "(the refetched bar is impossible too)"
                else:
                    replacement = candidate
                    finding.new_text = _row_text(candidate)
                break
            else:
                finding.new_text = "(the provider has no bar for that session)"

    if not apply:
        return finding

    kept = frame.iloc[:-1]
    if replacement is not None:
        kept = pd.concat([kept, replacement.to_frame().T], ignore_index=True)
    repaired = legacy._normalize_daily_bar_frame(kept)
    if repaired.empty:
        finding.new_text = "(refused: the repair would have emptied the file)"
        return finding
    _atomic_write_csv(path, repaired)
    finding.repaired = True
    return finding


def repair(
    cache_dir: Path | str | None = None,
    *,
    apply: bool = False,
    now: datetime | None = None,
    stream=None,
) -> int:
    """Read every cache file, report, and write only under ``apply``."""
    out = stream or sys.stdout
    import project_paths

    directory = Path(cache_dir) if cache_dir is not None else Path(DAILY_BARS_CACHE_DIR)
    print(f"home folder (project_paths.DATA_DIR): {project_paths.DATA_DIR}", file=out)
    print(f"daily-bar cache directory: {directory}", file=out)
    print(f"mode: {'APPLY (writes)' if apply else 'DRY RUN (writes nothing)'}", file=out)

    if _is_under(directory, PROTECTED_DATA_ROOT):
        print(
            f"REFUSED: {directory} is under the protected home folder "
            f"{PROTECTED_DATA_ROOT}; this tool only ever touches the machine cache.",
            file=out,
        )
        return 2
    if not directory.is_dir():
        print(f"REFUSED: {directory} is not a directory.", file=out)
        return 2

    moment = _aware(now)
    files = sorted(directory.glob("*.csv"))
    findings: list[RepairFinding] = []
    for path in files:
        finding = _repair_one_file(path, apply=apply, now=moment)
        if finding is not None:
            findings.append(finding)

    for finding in findings:
        day_text = finding.day.isoformat() if finding.day else "(no date)"
        print(
            f"{finding.symbol:<8} {day_text}  {finding.reason:<22} "
            f"old [{finding.old_text}] -> new [{finding.new_text}]"
            f"{'' if finding.repaired else '  (not written)'}",
            file=out,
        )
    written = sum(1 for finding in findings if finding.repaired)
    print(
        f"{len(files)} cache files read, {len(findings)} end in a forming or impossible "
        f"candle, {written} rewritten.",
        file=out,
    )
    if not apply and findings:
        print("Dry run: nothing was written. Re-run with --apply to write.", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m master_avwap_lib.daily_bar_cache",
        description="Guard and repair the per-symbol daily-bar cache (packet WS-FC1).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    repair_parser = subparsers.add_parser(
        "repair", help="remove a forming or impossible LAST row and refetch that session"
    )
    repair_parser.add_argument(
        "--apply", action="store_true", help="write the repaired files (default: dry run)"
    )
    repair_parser.add_argument(
        "--cache-dir", default=None, help="the daily-bar cache directory to read"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "repair":
        return repair(cache_dir=args.cache_dir, apply=bool(args.apply))
    parser.print_help()  # pragma: no cover - argparse requires a subcommand
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry
    # `python -m` re-executes this file as `__main__` while the package has
    # already imported it, so delegate to the package's copy rather than run a
    # second one with its own counters.
    from master_avwap_lib.daily_bar_cache import main as _package_main

    raise SystemExit(_package_main())
