"""The append-only daily record of what kind of day the market had (WS-ENV).

`indicators.d1_environment` decides the label; this file is where the desk
KEEPS it, one row per `(session, benchmark, rule_version)`, so that a readout
six months from now is cut by the label the rule gave that day rather than by a
label recomputed out of today's bar file.

Why a store at all, when the rule is pure and could be re-run on demand: a
re-run answers "what would this rule say about that session NOW", which is a
different question the moment the bar file is repaired, re-adjusted, or
back-filled from a second provider. The trader's question is the first one, and
only a dated row answers it.

The rules this file lives under:

* **Append-only, and never a rewrite.** A `(session, benchmark, rule_version)`
  already on disk is never written again - `append_environment` returns False
  and touches nothing. A second READING of a session (a re-run, a repaired bar
  file) is not news; it is a second opinion about a day that already has one.
* **A new rule is a new VERSION, beside the old one.** The store is never
  back-filled under a different rule without a version bump, and the bump is
  exactly what lets both live in one file: `label_for_session` is asked for a
  version, and answers only from that version's rows.
* **Every benchmark keeps its own row.** SPY is the one the readouts cut on;
  QQQ and IWM are extra benchmarks, never pooled with it and never averaged.
* **`unknown` is an answer.** A warm-up session, an unreadable bar file - the
  row is written with `label = "unknown"` and its reason. A missing row and a
  measured `unknown` are different facts and `label_for_session` returns
  `"unknown"` for both, which is why the row carries the reason.
* **The evidence never costs the event.** Every write swallows its own failure
  and returns a bool; the scan that called it does not care (ground rule: an
  evidence store never costs the thing it records).
* **Shadow only.** Nothing here reaches a detector, a score, an alert, a
  watchlist, Focus, the review queue or `review_policy.json` (plan.md sec 5).
  The label LABELS a readout.

The backfill CLI is DRY BY DEFAULT and prints where it is pointed before it
does anything (2026-09-05 rule: a scratch run that resolved the live home
folder by accident overwrote the live tracker)::

    python -m d1_environment_store backfill --benchmark SPY --since 2026-01-01
    python -m d1_environment_store backfill --benchmark SPY --since 2026-01-01 --apply

It labels each past session POINT-IN-TIME - session i from bars 0..i and
nothing after - under the CURRENT rule version, with `source = "backfill"`, and
it never relabels a session already written.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import project_paths
from completed_bars import is_completed_bar
from indicators.d1_environment import (
    RULE_VERSION,
    D1Environment,
    classify_environment,
    session_of,
)

logger = logging.getLogger(__name__)

#: The benchmarks the desk labels. SPY is the one every readout cuts on; the
#: other two are extra rows, never pooled into it.
BENCHMARKS = ("SPY", "QQQ", "IWM")

#: One row's fields, in the order they are written. Named here so a reader can
#: check a row against the contract without reading the writer.
ROW_FIELDS = (
    "session",
    "benchmark",
    "label",
    "rule_version",
    "range_atr",
    "slope_atr",
    "sma20",
    "atr14",
    "bars_used",
    "reason",
    "bars_through",
    "written_at",
    "source",
)

SOURCE_SCAN = "scan"
SOURCE_BACKFILL = "backfill"

#: A daily bar is complete when the session that owns it is over.
_DAILY_BAR_MINUTES = 24 * 60

#: How many daily bars a backfill reads at most. The cache holds a few years;
#: the rule needs 34 plus however far back the caller asked.
_MAX_BACKFILL_BARS = 5000

#: `labels_by_session`'s one read, keyed by (path, mtime). A worker calls it on
#: every Results redraw and the file is three rows a day; re-reading it on an
#: unchanged mtime is work nobody asked for, and caching it PAST a change would
#: be a stale label on a live page.
_LABEL_CACHE: dict[tuple[str, str, str], tuple[float, dict[str, str]]] = {}


def default_path() -> Path:
    """The store's path, read at CALL time so a test can redirect it."""
    return Path(project_paths.D1_ENVIRONMENT_FILE)


def _resolve(path: Any) -> Path:
    return Path(path) if path else default_path()


def _market_tz():
    try:
        from market_calendar import MARKET_TZ

        return MARKET_TZ
    except Exception:  # pragma: no cover - zoneinfo is stdlib on 3.12
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/New_York")


def written_time(now: datetime | None = None) -> datetime:
    """The write instant, AWARE and market-local.

    A naive `now` is ATTACHED with `astimezone()` and then converted, never
    `replace(tzinfo=...)` - the seam that relabels a Pacific afternoon as an
    Eastern one (the adoption gate's rule, same reasoning).
    """
    moment = now or datetime.now()
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.astimezone(_market_tz())


def read_rows(path: Any = None) -> list[dict]:
    """Every row on disk, oldest first. A missing file is zero rows.

    Deliberately uncached: a writer asks this for the keys it must not
    duplicate, and a cached answer would let one scan write a session twice
    inside a filesystem's mtime granularity.
    """
    target = _resolve(path)
    rows: list[dict] = []
    try:
        with open(target, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except ValueError:
                    # One unreadable line is one lost row, never a raised read:
                    # every caller here is a surface the trader opens.
                    continue
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
    except OSError:
        return []
    return rows


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """The identity a row is written under, and never written under twice."""
    return (
        str(row.get("session") or "")[:10],
        str(row.get("benchmark") or "").strip().upper(),
        str(row.get("rule_version") or "").strip(),
    )


def append_environment(
    env: D1Environment,
    *,
    benchmark: str,
    bars_through: str,
    source: str = SOURCE_SCAN,
    path: Any = None,
    now: datetime | None = None,
) -> bool:
    """Write one reading. False when that key is already on disk (or on error).

    Never raises: the scan that calls this is the product, and this is evidence
    about it.
    """
    target = _resolve(path)
    name = str(benchmark or "").strip().upper()
    key = (str(getattr(env, "as_of_session", "") or "")[:10], name, str(env.rule_version))
    if not key[0] or not key[1]:
        return False
    try:
        existing = {row_key(row) for row in read_rows(target)}
        if key in existing:
            return False
        row = {
            "session": key[0],
            "benchmark": name,
            "label": str(env.label),
            "rule_version": str(env.rule_version),
            "range_atr": env.range_atr,
            "slope_atr": env.slope_atr,
            "sma20": env.sma20,
            "atr14": env.atr14,
            "bars_used": int(env.bars_used or 0),
            "reason": str(env.reason or ""),
            "bars_through": str(bars_through or "")[:10],
            "written_at": written_time(now).isoformat(timespec="seconds"),
            "source": str(source or SOURCE_SCAN),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({field: row[field] for field in ROW_FIELDS}) + "\n")
        return True
    except Exception:
        logger.warning("D1 environment row not written for %s %s", name, key[0], exc_info=True)
        return False


def labels_by_session(
    benchmark: str = "SPY",
    rule_version: str = RULE_VERSION,
    path: Any = None,
) -> dict[str, str]:
    """`{session: label}` for one benchmark and version. ONE read, mtime-cached.

    The FIRST row wins for a key that somehow appears twice: the store is
    append-only, so the older row is the one that was written when the session
    was the news.
    """
    target = _resolve(path)
    name = str(benchmark or "").strip().upper()
    version = str(rule_version or "").strip()
    cache_key = (str(target), name, version)
    try:
        mtime = target.stat().st_mtime
    except OSError:
        _LABEL_CACHE.pop(cache_key, None)
        return {}
    cached = _LABEL_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    labels: dict[str, str] = {}
    for row in read_rows(target):
        session, row_benchmark, row_version = row_key(row)
        if row_benchmark != name or row_version != version or not session:
            continue
        labels.setdefault(session, str(row.get("label") or "unknown"))
    _LABEL_CACHE[cache_key] = (mtime, labels)
    return labels


def label_for_session(
    session: Any,
    benchmark: str = "SPY",
    rule_version: str = RULE_VERSION,
    path: Any = None,
) -> str:
    """One session's label, or `"unknown"` when nobody labelled it.

    A session nobody measured and a session measured as `unknown` both read
    `unknown` here, deliberately: neither is a claim about the tape. The row's
    own `reason` tells the two apart for a reader that needs to know.
    """
    stamp = str(session or "")[:10]
    if not stamp:
        return "unknown"
    return labels_by_session(benchmark=benchmark, rule_version=rule_version, path=path).get(
        stamp, "unknown"
    )


# ---------------------------------------------------------------------------
# the backfill
# ---------------------------------------------------------------------------


def _cached_daily_bars(benchmark: str) -> list[dict]:
    """The desk's own cached daily bars for one benchmark, oldest first.

    `project_paths.DAILY_BARS_CACHE_DIR` is read at CALL time. The file is the
    scanner's cache, which is the same store the live hook's fetch writes - the
    backfill deliberately opens no provider and makes no network call.
    """
    import csv

    directory = Path(project_paths.DAILY_BARS_CACHE_DIR)
    target = directory / f"{str(benchmark or '').strip().upper()}.csv"
    bars: list[dict] = []
    try:
        with open(target, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                stamp = ""
                for key in ("datetime", "date", "dt", "timestamp"):
                    if row.get(key):
                        stamp = str(row[key])[:10]
                        break
                if not stamp:
                    continue
                try:
                    bars.append(
                        {
                            "dt": stamp,
                            "open": float(row.get("open")),
                            "high": float(row.get("high")),
                            "low": float(row.get("low")),
                            "close": float(row.get("close")),
                        }
                    )
                except (TypeError, ValueError):
                    continue
    except OSError:
        return []
    bars.sort(key=lambda bar: bar["dt"])
    return bars[-_MAX_BACKFILL_BARS:]


def completed_daily_bars(bars: Sequence[Any], *, now: datetime | None = None) -> list[Any]:
    """Every bar whose SESSION is over. A forming last bar is preview.

    One rule, `scripts/completed_bars.py`, at daily length: a daily bar is
    complete when `bar_start + 24h <= now`. A cached file that ends in a
    forming candle - the desk has had 100 of those at once - would otherwise
    label today from half a day.
    """
    moment = now or datetime.now()
    return [bar for bar in bars or () if is_completed_bar(bar, _DAILY_BAR_MINUTES, now=moment)]


def backfill_rows(
    benchmark: str,
    *,
    since: str = "",
    bars: Sequence[Any] | None = None,
    now: datetime | None = None,
) -> list[tuple[str, D1Environment]]:
    """`(session, reading)` for every completed session at or after `since`.

    POINT-IN-TIME: session i is classified from bars 0..i and nothing after. A
    backfill that classified every session from the whole series would print
    one label as many times as there are sessions, which is the failure the
    tester's fixture is built to catch.
    """
    series = list(bars) if bars is not None else _cached_daily_bars(benchmark)
    series = completed_daily_bars(series, now=now)
    floor = str(since or "")[:10]
    out: list[tuple[str, D1Environment]] = []
    for index in range(len(series)):
        session = session_of(series[index])
        if floor and session < floor:
            continue
        out.append((session, classify_environment(series[: index + 1])))
    return out


def _label_distribution(readings: Iterable[tuple[str, D1Environment]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _session, env in readings:
        counts[env.label] = counts.get(env.label, 0) + 1
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    """`backfill`, DRY BY DEFAULT. Prints where it is pointed before anything."""
    parser = argparse.ArgumentParser(prog="d1_environment_store", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fill = sub.add_parser("backfill", help="label past sessions from the cached daily bars")
    fill.add_argument("--benchmark", default="SPY")
    fill.add_argument("--since", default="", help="first session to label (YYYY-MM-DD)")
    fill.add_argument("--path", default="", help="store path (default: the named constant)")
    fill.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    target = _resolve(args.path)
    # The 2026-09-05 rule: say where you are pointed BEFORE you do anything.
    print(f"DATA_DIR: {project_paths.DATA_DIR}")
    print(f"store: {target}")
    print(f"daily bar cache: {Path(project_paths.DAILY_BARS_CACHE_DIR)}")
    print(f"rule: {RULE_VERSION}   benchmark: {args.benchmark}   since: {args.since or 'all'}")
    print("mode: APPLY (writing)" if args.apply else "mode: DRY RUN (writing nothing)")

    readings = backfill_rows(args.benchmark, since=args.since)
    if not readings:
        print("no completed cached daily bars for that benchmark - nothing to label")
        return 0

    distribution = _label_distribution(readings)
    print(f"sessions: {len(readings)}  ({readings[0][0]} .. {readings[-1][0]})")
    for label in sorted(distribution, key=lambda name: (-distribution[name], name)):
        print(f"  {label}: {distribution[label]}")

    written = skipped = 0
    for _session, env in readings:
        if args.apply:
            if append_environment(
                env,
                benchmark=args.benchmark,
                bars_through=env.as_of_session,
                source=SOURCE_BACKFILL,
                path=target,
            ):
                written += 1
            else:
                skipped += 1
    if args.apply:
        print(f"written: {written}   already present: {skipped}")
    else:
        print("dry run - re-run with --apply to write these rows")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main(sys.argv[1:]))
