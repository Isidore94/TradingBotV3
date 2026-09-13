"""WS-10A - last scan, latest input bar and shown report are THREE clocks.

The trader's report (WISHLIST 10A): *"the strongest Master AVWAP updates seem to
arrive in the final hour or at EOD, even when the app says it updated earlier"*
and *"a recent file timestamp proves neither fresh input bars nor good
discovery"*. A report file's mtime answers only "when was this file last
written". It cannot say whether the scan reached its whole universe, whether the
daily bars it read were yesterday's completed ones or today's forming preview,
or whether the report on screen is the one THIS scan produced or the last good
one kept because this scan died.

So every scan writes one small manifest with all three clocks:

* ``started_at`` / ``finished_at`` - when the scan ran;
* ``latest_input_bar_session`` (+ ``preview_bar_used``) - how fresh its INPUTS
  were, judged by the exchange session close, never by a date comparison;
* ``outputs`` - what it published, with the row count of each file.

Deliberately a NEW module rather than a seam in ``legacy.py``: this packet
carries no trader yes for the ask-first file, so the ``legacy.py`` diff is zero.
The writer is called by ``runner.run_master`` - the wrapper that owns both the
success and the failure branch - and, like every other evidence store, it may
never cost the scan the thing it records (plan.md sec 5): a manifest that cannot
be written is logged and swallowed.

Two clocks, spelled apart on purpose
------------------------------------
``freshness_line`` renders every stamp **in the offset the stamp itself
carries** and never re-converts it (the repo's "attach, never strip" rule), so a
manifest written at 12:31 New York prints 12:31 on a Pacific desk. The
checkpoint windows in :mod:`master_avwap_lib.scan_replay` are the opposite
question - where in the MARKET's day a scan landed - and convert to exchange
time for that reason alone.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import project_paths

#: The packet says "a 30-day cap". It is counted as the 30 most recent DISTINCT
#: dates that have copies, not the newest 30 files and not 30 calendar days: a
#: long weekend or a week off must not silently shorten the window the replay
#: can answer over, and a busy day writes six copies without evicting five other
#: days.
REPORT_COPY_RETENTION_SESSIONS = 30

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

_SEPARATOR = " · "


# ---------------------------------------------------------------------------
# the clock
# ---------------------------------------------------------------------------
def market_now() -> datetime:
    """Now, AWARE, market-local. The ONE clock hook, so a test can freeze it."""
    try:
        from market_session import get_market_local_now

        moment = get_market_local_now()
    except Exception:  # pragma: no cover - settings/zoneinfo trouble
        moment = datetime.now().astimezone()
    return moment if moment.tzinfo is not None else moment.astimezone()


# ---------------------------------------------------------------------------
# the paths
# ---------------------------------------------------------------------------
def manifest_path() -> Path:
    return Path(project_paths.MASTER_AVWAP_SCAN_MANIFEST_FILE)


def history_path() -> Path:
    return Path(project_paths.MASTER_AVWAP_SCAN_MANIFEST_HISTORY_FILE)


def scan_reports_dir() -> Path:
    """Where the dated copies of the priority report live.

    The desk keeps no historic snapshot of the priority report today, so the
    replay would have nothing to read. The smallest useful copy per scan is
    added here - run id, status, the two freshness fields, and one
    ``{symbol, side, bucket}`` row per published row. Machine-local, because it
    is diagnostic evidence about THIS machine's scans.
    """
    return Path(project_paths.get_diagnostics_dir()) / "scan_reports"


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def read_manifest(path: Path | str | None = None) -> dict[str, Any] | None:
    """The manifest, or ``None``.

    ``None`` when there is no file or it cannot be parsed - missing data is
    uncertainty, never a guess, and in particular never a fallback to a file
    mtime, which is the very signal the trader says proves nothing.
    """
    target = Path(path) if path is not None else manifest_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_stamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed


def _clock(value: Any) -> str:
    """``HH:MM`` in the offset the stamp CARRIES - never re-converted."""
    parsed = _parse_stamp(value)
    return parsed.strftime("%H:%M") if parsed is not None else "?"


def _session_label(value: Any) -> str:
    try:
        return date.fromisoformat(str(value)).strftime("%a %m-%d")
    except (TypeError, ValueError):
        return ""


def freshness_line(
    manifest: dict[str, Any] | None, *, report_mtime: datetime | None = None
) -> str:
    """The one-line strip: what ran, how fresh its inputs were, what is shown.

    Pure and cheap - no file is opened here, so the Qt thread may format it
    after a worker has done the reading.
    """
    shown = (
        f"shown: {_clock(report_mtime)} report" if report_mtime is not None else "no report"
    )
    if not isinstance(manifest, dict) or not manifest:
        return _SEPARATOR.join(["Scan: not recorded yet", shown])

    status = str(manifest.get("status") or "").strip().lower()
    finished = _clock(manifest.get("finished_at"))

    if status == STATUS_FAILED:
        # The failure does not blank the screen and does not claim the attempt
        # produced what is on it: the trader is told, in one line, which report
        # they are reading and that it is stale.
        if report_mtime is None:
            return _SEPARATOR.join([f"Scan FAILED {finished}", "no report"])
        return _SEPARATOR.join(
            [f"Scan FAILED {finished}", f"showing {_clock(report_mtime)} report (stale)"]
        )

    if status == STATUS_PARTIAL:
        # "Partial" with no number is not truthful either: the two counts are
        # what let the trader judge whether an absent name means anything.
        head = (
            f"Scan partial {finished} "
            f"({int(manifest.get('symbols_fetched') or 0)} of "
            f"{int(manifest.get('universe_size') or 0)} symbols)"
        )
    else:
        head = f"Scan {status or 'ok'} {finished}"

    parts = [head]
    preview = bool(manifest.get("preview_bar_used"))
    session = _session_label(manifest.get("latest_input_bar_session"))
    if session:
        parts.append(f"inputs through {session}" + ("" if preview else " (D1 complete)"))
    else:
        parts.append("inputs: no completed session recorded")
    if preview:
        parts.append("inputs: today preview")
    parts.append(shown)
    return _SEPARATOR.join(parts)


# ---------------------------------------------------------------------------
# the input-bar clock
# ---------------------------------------------------------------------------
def _frame_dates(frame) -> list[date]:
    try:
        import pandas as pd

        if frame is None or getattr(frame, "empty", True):
            return []
        if "datetime" not in getattr(frame, "columns", []):
            return []
        stamps = pd.to_datetime(frame["datetime"], errors="coerce").dropna()
        if stamps.empty:
            return []
        return [value.date() for value in stamps]
    except Exception:  # noqa: BLE001 - a manifest never costs the scan
        return []


def input_bar_freshness(
    frames: dict[str, Any] | None, *, now: datetime
) -> tuple[str | None, bool]:
    """``(newest COMPLETED session, a forming preview was in the inputs)``.

    The completeness question is asked ONCE, of the exchange calendar, through
    WS-FC1's rule (16:00 ET inclusive, ``astimezone``, no early-close model - so
    a half day waits until the regular close, which is conservative in the only
    direction that matters). A manifest that took ``max(last bar date)`` would
    report today on every intraday scan, which is exactly the false freshness
    this packet exists to end.
    """
    from . import daily_bar_cache

    if not frames:
        return None, False
    last_complete = daily_bar_cache.last_completed_session(now)
    newest_complete: date | None = None
    preview = False
    for frame in frames.values():
        dates = _frame_dates(frame)
        if not dates:
            continue
        newest = max(dates)
        if newest > last_complete:
            preview = True
            complete = [value for value in dates if value <= last_complete]
        else:
            complete = [newest]
        if complete:
            candidate = max(complete)
            if newest_complete is None or candidate > newest_complete:
                newest_complete = candidate
    return (newest_complete.isoformat() if newest_complete is not None else None), preview


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


def _outputs(run_result: dict[str, Any]) -> list[dict[str, Any]]:
    from . import legacy

    puts = len(run_result.get("theta_put_rows") or [])
    spreads = len(run_result.get("theta_pcs_rows") or [])
    return [
        {
            "name": "priority_setups",
            "path": str(project_paths.MASTER_AVWAP_PRIORITY_SETUPS_FILE),
            # The report holds what the scan PUBLISHED; a manifest that counted
            # the tracked subset instead would be wrong by the watch rows.
            "rows": len(run_result.get("priority_rows") or []),
        },
        {
            "name": "theta_puts",
            "path": str(legacy.THETA_PUTS_FILE),
            # One report, two families of row: the sold puts and the credit
            # spreads are written into the same file, so its row count is both.
            "rows": puts + spreads,
        },
        {
            "name": "d1_watchlist",
            "path": str(project_paths.MASTER_AVWAP_D1_WATCHLIST_FILE),
            "rows": int(run_result.get("d1_watchlist_symbol_count") or 0),
        },
    ]


def build_manifest(
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    run_result: dict[str, Any] | None = None,
    error: str = "",
    forming_dropped: int = 0,
    invalid_dropped: int = 0,
) -> dict[str, Any]:
    """The payload. Pure: it opens nothing and decides nothing about disk."""
    payload: dict[str, Any] = {
        "run_id": str(run_id or ""),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "daily_bars_forming_dropped": int(forming_dropped),
        "daily_bars_invalid_dropped": int(invalid_dropped),
    }
    if run_result is None:
        payload.update(
            {
                "status": STATUS_FAILED,
                "universe_size": 0,
                "symbols_fetched": 0,
                "daily_bar_source_counts": {},
                "latest_input_bar_session": None,
                "preview_bar_used": False,
                "outputs": [],
                "error": str(error or ""),
            }
        )
        return payload

    from . import legacy

    frames = run_result.get("daily_frames_by_symbol") or {}
    universe_size = int(run_result.get("universe_size") or 0)
    symbols_fetched = len(frames)
    sources: Counter = Counter()
    for frame in frames.values():
        name = legacy._get_daily_bar_source(frame)
        if name:
            sources[name] += 1
    session, preview = input_bar_freshness(frames, now=finished_at)
    payload.update(
        {
            # A scan that RETURNED having reached fewer names than it set out to
            # published a partial report; calling that "ok" is the untruthful
            # freshness the packet exists to end.
            "status": STATUS_PARTIAL if symbols_fetched < universe_size else STATUS_OK,
            "universe_size": universe_size,
            "symbols_fetched": symbols_fetched,
            "daily_bar_source_counts": dict(sources),
            "latest_input_bar_session": session,
            "preview_bar_used": bool(preview),
            "outputs": _outputs(run_result),
        }
    )
    return payload


def _report_rows(run_result: dict[str, Any] | None) -> list[dict[str, str]]:
    rows = []
    for row in (run_result or {}).get("priority_rows") or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "symbol": str(row.get("symbol") or ""),
                "side": str(row.get("side") or ""),
                "bucket": str(row.get("priority_bucket") or ""),
            }
        )
    return rows


def _prune_report_copies(folder: Path) -> None:
    """Keep the ``REPORT_COPY_RETENTION_SESSIONS`` most recent DISTINCT dates."""
    by_date: dict[str, list[Path]] = {}
    for child in folder.glob("*.json"):
        key = child.name.split("_", 1)[0]
        by_date.setdefault(key, []).append(child)
    surplus = max(0, len(by_date) - REPORT_COPY_RETENTION_SESSIONS)
    for key in sorted(by_date)[:surplus]:
        for child in by_date[key]:
            try:
                child.unlink()
            except OSError:  # pragma: no cover - a locked file is not a scan failure
                logging.debug("scan report copy %s could not be pruned.", child, exc_info=True)


def write_report_copy(
    manifest: dict[str, Any], run_result: dict[str, Any] | None
) -> Path | None:
    """One dated snapshot per PUBLISHING scan, capped.

    A failed scan published nothing, so it gets no copy: a snapshot listing zero
    rows would read to the replay as "the name was not in the report", which is
    a reconstruction, and the brief forbids exactly that.
    """
    if str(manifest.get("status")) == STATUS_FAILED:
        return None
    finished = _parse_stamp(manifest.get("finished_at"))
    if finished is None:
        return None
    folder = scan_reports_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{finished.date().isoformat()}_{finished.strftime('%H%M')}.json"
    payload = {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "finished_at": manifest.get("finished_at"),
        "latest_input_bar_session": manifest.get("latest_input_bar_session"),
        "preview_bar_used": bool(manifest.get("preview_bar_used")),
        "rows": _report_rows(run_result),
    }
    _atomic_write(path, json.dumps(payload, ensure_ascii=False))
    _prune_report_copies(folder)
    return path


def append_history(manifest: dict[str, Any]) -> None:
    """Exactly ONE line per scan, append-only: the replay walks this file."""
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")


def record_scan(
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    run_result: dict[str, Any] | None = None,
    error: str = "",
    forming_dropped: int = 0,
    invalid_dropped: int = 0,
) -> dict[str, Any] | None:
    """Write the manifest, the history line and the dated copy.

    Called by ``runner.run_master`` on BOTH paths. Never raises: an evidence
    store may not cost the thing it records (plan.md sec 5), and a scan that
    produced a good report must not be turned into a failure by a full disk.
    """
    try:
        manifest = build_manifest(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            run_result=run_result,
            error=error,
            forming_dropped=forming_dropped,
            invalid_dropped=invalid_dropped,
        )
    except Exception:  # noqa: BLE001
        logging.exception("Scan manifest could not be built (scan result unaffected).")
        return None

    def _write() -> None:
        _atomic_write(
            manifest_path(), json.dumps(manifest, ensure_ascii=False, indent=2)
        )

    for step, action in (
        ("write", _write),
        ("history", lambda: append_history(manifest)),
        ("copy", lambda: write_report_copy(manifest, run_result)),
    ):
        try:
            action()
        except Exception:  # noqa: BLE001
            logging.exception("Scan manifest %s step failed (scan result unaffected).", step)
    return manifest
