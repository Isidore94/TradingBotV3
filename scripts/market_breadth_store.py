"""The append-only daily breadth record (WISHLIST P2-8 8a).

`indicators.breadth` decides the numbers; this file KEEPS them, one row per
`(session, universe, rule_version)`, beside `d1_environment.jsonl` and under
the same rules as `d1_environment_store`:

* Append-only. A key already on disk is never written again.
* A new rule is a new version, beside the old one.
* The evidence never costs the event: every write returns a bool, never raises.
* Display and grading only. Nothing here reaches a detector, score, alert,
  watchlist, Focus, the review queue or `review_policy.json`.

One owner writes it: :func:`record_last_session`, called by the nightly
`read_grades_mature` slot, plus the dry-by-default backfill CLI::

    python -m market_breadth_store backfill --since 2026-09-01
    python -m market_breadth_store backfill --since 2026-09-01 --apply

Bars come from the desk's own daily-bar cache (no download). A cached file
last written BEFORE a session's close holds a forming bar for that session, so
the session is `unknown` for that name - never a half-day close.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import project_paths
from indicators.breadth import RULE_VERSION, BreadthReading, compute_breadth, is_thin, known_names

logger = logging.getLogger(__name__)

#: The universe a row is computed over. Only one today.
UNIVERSE_ALL = "universe_all"

SOURCE_NIGHT = "night"
SOURCE_BACKFILL = "backfill"

#: Daily bars read per name. SMA50 needs 50; the rest is slack for gaps.
_TAIL_BARS = 80

#: `rows_by_session` cache keyed by (path, universe, version) -> (mtime, rows).
_ROW_CACHE: dict[tuple[str, str, str], tuple[float, dict[str, dict]]] = {}


def default_path() -> Path:
    """The store's path, read at CALL time so a test can redirect it."""
    return Path(project_paths.MARKET_BREADTH_FILE)


def _resolve(path: Any) -> Path:
    return Path(path) if path else default_path()


def _market_tz():
    from market_calendar import MARKET_TZ

    return MARKET_TZ


def read_rows(path: Any = None) -> list[dict]:
    """Every row on disk, oldest first. Uncached (writers use it for dedupe)."""
    rows: list[dict] = []
    try:
        with open(_resolve(path), "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except ValueError:
                    continue
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
    except OSError:
        return []
    return rows


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("session") or "")[:10],
        str(row.get("universe") or "").strip(),
        str(row.get("rule_version") or "").strip(),
    )


def append_breadth(
    reading: BreadthReading,
    *,
    universe: str = UNIVERSE_ALL,
    universe_as_of: str = "",
    source: str = SOURCE_NIGHT,
    path: Any = None,
    now: datetime | None = None,
) -> bool:
    """Write one reading. False when the key is already on disk or on error."""
    target = _resolve(path)
    key = (str(reading.session or "")[:10], str(universe or "").strip(), str(reading.rule_version))
    if not key[0] or not key[1]:
        return False
    # A thin reading is never stored: the store is append-only, so a thin row
    # would block the good row a later retry could write for the same key.
    if is_thin(reading.as_row()):
        return False
    try:
        if key in {row_key(row) for row in read_rows(target)}:
            return False
        moment = now or datetime.now()
        if moment.tzinfo is None:
            moment = moment.astimezone()
        row = {
            "session": key[0],
            "universe": key[1],
            **{k: v for k, v in reading.as_row().items() if k != "session"},
            "universe_as_of": str(universe_as_of or ""),
            "written_at": moment.astimezone(_market_tz()).isoformat(timespec="seconds"),
            "source": str(source or SOURCE_NIGHT),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return True
    except Exception:
        logger.warning("Breadth row not written for %s", key[0], exc_info=True)
        return False


def rows_by_session(
    universe: str = UNIVERSE_ALL, rule_version: str = RULE_VERSION, path: Any = None
) -> dict[str, dict]:
    """`{session: row}` for one universe and version. One read, mtime-cached.

    The FIRST row wins for a key written twice (the store is append-only).
    """
    target = _resolve(path)
    cache_key = (str(target), str(universe), str(rule_version))
    try:
        mtime = target.stat().st_mtime
    except OSError:
        _ROW_CACHE.pop(cache_key, None)
        return {}
    cached = _ROW_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    rows: dict[str, dict] = {}
    for row in read_rows(target):
        session, row_universe, version = row_key(row)
        if session and row_universe == universe and version == rule_version:
            rows.setdefault(session, row)
    _ROW_CACHE[cache_key] = (mtime, rows)
    return rows


def row_for_session(session: Any, **kwargs: Any) -> dict | None:
    """One session's stored row, or None when nobody recorded it."""
    stamp = str(session or "")[:10]
    if not stamp:
        return None
    row = rows_by_session(**kwargs).get(stamp)
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# loading (files the desk already has; never a provider)
# ---------------------------------------------------------------------------


def load_universe(path: Any = None) -> tuple[list[str], str]:
    """`(names, as_of)` from `universe_all.txt`; as_of is the file's mtime date."""
    target = Path(path) if path else Path(project_paths.UNIVERSE_ALL_FILE)
    names: list[str] = []
    try:
        with open(target, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                name = text.replace(",", " ").split()[0].strip().upper()
                if name:
                    names.append(name)
        as_of = datetime.fromtimestamp(target.stat().st_mtime).date().isoformat()
    except OSError:
        return [], ""
    return names, as_of


def _read_tail_bars(target: Path) -> list[dict]:
    import csv

    try:
        with open(target, "r", newline="", encoding="utf-8-sig") as handle:
            lines = handle.readlines()
    except OSError:
        return []
    if not lines:
        return []
    header = lines[0]
    body = lines[1:][-_TAIL_BARS:]
    bars: list[dict] = []
    for row in csv.DictReader([header, *body]):
        stamp = ""
        for key in ("datetime", "date", "dt", "timestamp"):
            if row.get(key):
                stamp = str(row[key])[:10]
                break
        if not stamp:
            continue
        try:
            bars.append({"dt": stamp, "close": float(row.get("close"))})
        except (TypeError, ValueError):
            continue
    return bars


def load_daily_bars(
    names: Sequence[str], *, session: str, directory: Any = None
) -> dict[str, list[dict]]:
    """The cached daily bars per name, cut so a forming session bar is dropped.

    A file whose last write was before `session`'s close cannot hold that
    session's final bar, so bars dated `session` or later are removed and the
    name reads unknown for it.
    """
    import market_early_close

    folder = Path(directory) if directory else Path(project_paths.DAILY_BARS_CACHE_DIR)
    try:
        close_at = market_early_close.session_close(date.fromisoformat(str(session)[:10]))
    except Exception:  # noqa: BLE001 - no close means nothing is known complete
        return {}
    out: dict[str, list[dict]] = {}
    for name in names:
        target = folder / f"{name}.csv"
        try:
            written = datetime.fromtimestamp(target.stat().st_mtime).astimezone()
        except OSError:
            continue
        bars = _read_tail_bars(target)
        if written < close_at:
            bars = [bar for bar in bars if bar["dt"] < str(session)[:10]]
        if bars:
            out[name] = bars
    return out


def compute_for_session(session: str, *, universe_path: Any = None, directory: Any = None):
    """`(reading, universe_as_of)` for one completed session, from local files."""
    import market_calendar

    day = date.fromisoformat(str(session)[:10])
    prior = market_calendar.previous_session(day).isoformat()
    names, as_of = load_universe(universe_path)
    bars = load_daily_bars(names, session=day.isoformat(), directory=directory)
    reading = compute_breadth(bars, universe=names, session=day.isoformat(), prior_session=prior)
    return reading, as_of


def record_last_session(now: datetime, *, path: Any = None, **kwargs: Any) -> dict[str, Any]:
    """Compute and append the last COMPLETED session's breadth. Never raises.

    Only the last completed session: the universe file is today's, so an older
    session would be measured over a list it did not have (that is backfill's
    job, labelled as such).
    """
    try:
        import market_calendar

        session = market_calendar.last_completed_session(now).isoformat()
        if session in rows_by_session(path=path):
            return {"session": session, "written": False, "reason": "already recorded"}
        reading, as_of = compute_for_session(session, **kwargs)
        if not reading.names_total:
            return {"session": session, "written": False, "reason": "no universe_all names"}
        if is_thin(reading.as_row()):
            known = known_names(reading.as_row())
            return {
                "session": session,
                "written": False,
                "reason": f"thin: {known} of {reading.names_total} names known; retry later",
            }
        written = append_breadth(
            reading, universe_as_of=as_of, source=SOURCE_NIGHT, path=path, now=now
        )
        return {"session": session, "written": written, "reason": "" if written else "not written"}
    except Exception as exc:  # noqa: BLE001 - evidence never costs the night
        logger.warning("Breadth for the last session was not recorded.", exc_info=True)
        return {"session": "", "written": False, "reason": f"{type(exc).__name__}: {exc}"}


def _sessions_since(since: str, now: datetime) -> list[str]:
    import market_calendar

    last = market_calendar.last_completed_session(now)
    cursor = date.fromisoformat(since)
    out: list[str] = []
    while cursor <= last:
        if market_calendar.is_session(cursor):
            out.append(cursor.isoformat())
        cursor = date.fromordinal(cursor.toordinal() + 1)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """`backfill`, DRY BY DEFAULT. Prints where it is pointed before anything."""
    parser = argparse.ArgumentParser(prog="market_breadth_store", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fill = sub.add_parser("backfill", help="record past sessions from the cached daily bars")
    fill.add_argument("--since", required=True, help="first session (YYYY-MM-DD)")
    fill.add_argument("--path", default="", help="store path (default: the named constant)")
    fill.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    target = _resolve(args.path)
    print(f"DATA_DIR: {project_paths.DATA_DIR}")
    print(f"store: {target}")
    print(f"universe: {project_paths.UNIVERSE_ALL_FILE}")
    print(f"daily bar cache: {Path(project_paths.DAILY_BARS_CACHE_DIR)}")
    print(f"rule: {RULE_VERSION}")
    print("mode: APPLY (writing)" if args.apply else "mode: DRY RUN (writing nothing)")
    now = datetime.now().astimezone()
    for session in _sessions_since(args.since, now):
        reading, as_of = compute_for_session(session)
        row = reading.as_row()
        print(
            f"{session}: A/D {row['advancers']}/{row['decliners']} "
            f"(unknown {row['ad_unknown']}), >SMA20 {row['pct_above_sma20']}%, "
            f">SMA50 {row['pct_above_sma50']}%"
        )
        if is_thin(row):
            print(f"  thin ({known_names(row)} of {row['names_total']} known) - never written")
            continue
        if args.apply:
            append_breadth(reading, universe_as_of=as_of, source=SOURCE_BACKFILL, path=target)
    if not args.apply:
        print("dry run - re-run with --apply to write these rows")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main(sys.argv[1:]))
