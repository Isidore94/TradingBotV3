"""The day-trade watchlists are wiped after each session's close.

Trader, 2026-09-15: *"i want all names wiped at the end of the day. its a
daytrade watchlist not a permanent one."* `longs.txt` and `shorts.txt` are the
intraday (M5) lists BounceBot scans; nothing on them is meant to outlive the
session it was typed for. The swing lists (`swinglongs.txt`, `shortswings.txt`)
are permanent by contrast and this module never touches them.

The rule, stateless and idempotent - the file's own modification time is the
record:

* **A list is due for a reset when it holds names and was last written at or
  before the close of the LAST COMPLETED exchange session** (16:00 ET on
  `market_calendar.session_close`). A name typed AFTER the close - evening prep
  for tomorrow - is tomorrow's, survives, and is wiped after tomorrow's close.
* The reset writes an EMPTY list through `autopilot_core.write_watchlist_file`
  (atomic, designated-writer gated) and then appends one `remove` row per name
  to the WS-5D intent stream with the source `session_reset`, so the Watchlist
  tab's next load reconciles to an empty list instead of inventing
  `observed_external` removals. The file is written first; a failed append
  costs the evidence, never the wipe (plan.md sec 5).
* A refused write (this machine is not the designated writer) records nothing:
  a `remove` for a name still on the file would be an invention.
* Once wiped the file's mtime is after the close, so the same session is never
  wiped twice, and an empty list has nothing to do.

Who runs it: `AutopilotService._maybe_reset_daytrade_watchlists`, on every
30-second tick before the weekend short-circuit and before the morning open
scan, in every Auto mode (it is a day roll, not a scan starter, so quiet hours
do not apply - the same standing as the autolongs/autoshorts day-roll clear).
A desk that was closed at the close wipes on its first tick back; a desk that is
open wipes on the first tick after 13:00 Pacific. The CLI below is the trader's
own door and is a DRY RUN unless `--apply` is passed.

The plan.md sec 5 invariant "user-entered watchlist names are never
automatically removed" is amended narrowly for these two lists (decision
0020): the reset is a session boundary that removes EVERY name alike, never a
machine judgement about one name. `FocusPickStore` membership is untouched -
a Focus pick is still scanned through the fast lane, and its later
un-injection finds the name already gone and records nothing.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from market_calendar import last_completed_session, session_close
from watchlist_utils import read_watchlist_symbols

logger = logging.getLogger(__name__)

#: `local_settings.json` switch. Default ON - the trader asked for the wipe;
#: the switch exists so it can be turned off without a code change.
SETTING_KEY = "daytrade_watchlists_reset"
SETTING_DEFAULT = True

#: The WS-5D source every row this module writes carries.
INTENT_SOURCE = "session_reset"
INTENT_WRITER = "daytrade_watchlist_reset"

#: The two day-trade lists, by their WS-5D list names. The swing lists are
#: deliberately absent: a swing name is held across sessions by design.
RESET_LISTS: tuple[str, ...] = ("longs", "shorts")


@dataclass(frozen=True)
class ResetResult:
    """What happened to one list on one call."""

    list_name: str
    path: Path
    session: date | None
    removed: tuple[str, ...]
    written: bool
    recorded: int
    reason: str

    @property
    def wiped(self) -> bool:
        return self.written and bool(self.removed)


def written_at(path: Path | str) -> datetime | None:
    """The file's last write as an AWARE UTC instant, or ``None`` when absent."""
    try:
        stamp = Path(path).stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(stamp, tz=timezone.utc)


def reset_due(now: datetime, *, written_at: datetime | None) -> date | None:
    """The session whose close makes the list stale, or ``None``.

    ``None`` when there is no file, when the calendar cannot name a completed
    session, or when the file was written after the last completed session's
    close (that write is for the NEXT session). Naive ``now`` is the desk's
    local clock; a naive ``written_at`` is treated the same way.
    """
    if written_at is None:
        return None
    try:
        session = last_completed_session(now)
    except Exception:  # noqa: BLE001 - a calendar that cannot answer wipes nothing
        return None
    close = session_close(session)
    stamp = written_at if written_at.tzinfo is not None else written_at.astimezone()
    if stamp > close:
        return None
    return session


def default_paths() -> dict[str, Path]:
    """The production pair, resolved at call time (never bound at import)."""
    from project_paths import LONGS_FILE, SHORTS_FILE

    return {"longs": Path(LONGS_FILE), "shorts": Path(SHORTS_FILE)}


def _enabled(get_setting: Callable[[str, object], object] | None) -> bool:
    if get_setting is None:
        try:
            from project_paths import get_local_setting as get_setting
        except Exception:  # noqa: BLE001 - no settings means the default
            return SETTING_DEFAULT
    try:
        return bool(get_setting(SETTING_KEY, SETTING_DEFAULT))
    except Exception:  # noqa: BLE001
        return SETTING_DEFAULT


def apply_reset(
    now: datetime | None = None,
    *,
    paths: dict[str, Path] | None = None,
    get_setting: Callable[[str, object], object] | None = None,
    dry_run: bool = False,
) -> list[ResetResult]:
    """Wipe every due day-trade list. Never raises; one result per list.

    ``dry_run`` reports what WOULD be wiped and writes nothing anywhere.
    """
    moment = now or datetime.now()
    targets = paths if paths is not None else default_paths()
    results: list[ResetResult] = []
    if not _enabled(get_setting):
        for list_name, path in targets.items():
            results.append(ResetResult(list_name, Path(path), None, (), False, 0, "switched off"))
        return results
    for list_name, path in targets.items():
        path = Path(path)
        try:
            results.append(_reset_one(list_name, path, moment, dry_run=dry_run))
        except Exception:  # noqa: BLE001 - one list's failure never costs the other
            logger.exception("day-trade watchlist reset failed for %s", path)
            results.append(ResetResult(list_name, path, None, (), False, 0, "failed"))
    return results


def _reset_one(list_name: str, path: Path, now: datetime, *, dry_run: bool) -> ResetResult:
    stamp = written_at(path)
    if stamp is None:
        return ResetResult(list_name, path, None, (), False, 0, "no file")
    symbols = tuple(str(s).strip().upper() for s in read_watchlist_symbols(path) if str(s).strip())
    if not symbols:
        return ResetResult(list_name, path, None, (), False, 0, "already empty")
    session = reset_due(now, written_at=stamp)
    if session is None:
        return ResetResult(list_name, path, None, (), False, 0, "written after the last close")
    if dry_run:
        return ResetResult(list_name, path, session, symbols, False, 0, "dry run")
    from autopilot_core import write_watchlist_file

    if not write_watchlist_file(path, []):
        # The role refused; the names are still there, so a `remove` row for
        # them would describe something that did not happen.
        return ResetResult(list_name, path, session, symbols, False, 0, "write refused")
    recorded = _record(list_name, symbols, session, now)
    return ResetResult(list_name, path, session, symbols, True, recorded, "wiped")


def _record(list_name: str, symbols: Sequence[str], session: date, now: datetime) -> int:
    try:
        import watchlist_intent_events as intent

        return int(
            intent.record_changes(
                list_name=list_name,
                removed=symbols,
                source=INTENT_SOURCE,
                writer=INTENT_WRITER,
                reason=f"day-trade list reset after the {session.isoformat()} close",
                now=now,
            )
        )
    except Exception:  # noqa: BLE001 - evidence never costs the wipe it records
        logger.warning("day-trade watchlist reset: intent rows not recorded for %s", list_name, exc_info=True)
        return 0


def describe(results: Iterable[ResetResult]) -> str:
    """One log line: what was wiped, for which session, and what was not."""
    parts = []
    for result in results:
        if result.wiped:
            note = f"{len(result.removed)} name(s) after the {result.session} close"
            if result.recorded != len(result.removed):
                note += f" ({result.recorded} of {len(result.removed)} intent rows recorded)"
            parts.append(f"{result.path.name}: wiped {note}")
        else:
            parts.append(f"{result.path.name}: {result.reason}")
    return "; ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wipe the day-trade watchlists that outlived their session.")
    parser.add_argument("--apply", action="store_true", help="write; without it this is a dry run")
    args = parser.parse_args(argv)
    results = apply_reset(dry_run=not args.apply)
    for result in results:
        names = ", ".join(result.removed) if result.removed else "-"
        print(f"{result.list_name:7s} {result.reason:28s} session={result.session} names={names}")
    if not args.apply and any(r.reason == "dry run" for r in results):
        print("dry run: pass --apply to wipe")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
