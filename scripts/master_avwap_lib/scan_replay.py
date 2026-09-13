"""WS-10A item 3 - when did a name actually become eligible, and when was it shown?

    cd scripts
    python -m master_avwap_lib.scan_replay --symbol NVDA --session 2026-09-10

The trader's second question in WISHLIST 10A is not about freshness but about
DISCOVERY: *"the strongest updates seem to arrive in the final hour or at
EOD"*. That is a claim about when a name entered the report, and it can only be
answered from what was RECORDED at the time. So this walks the scan-manifest
history and the dated report copies
(:func:`master_avwap_lib.scan_manifest.scan_reports_dir`) and prints, for one
symbol on one session, what each checkpoint of the day held.

**It reconstructs nothing.** A checkpoint with no recorded snapshot prints
``no recorded snapshot`` and stops there; re-running today's scanner over an
old session would answer with today's data and today's code, which is the
false certainty the brief forbids. Read-only: this opens files and writes
none, and prints the home folder it resolved first (the 2026-09-05 scratch-
script rule).

The checkpoint clock is the MARKET's
-------------------------------------
``scan_manifest.freshness_line`` renders stamps in the offset they carry,
because "when did the scan run" is a question about the desk. "Open / midday /
final hour / close" is a question about the exchange's day, so the stamps are
converted to exchange time here - otherwise a Pacific desk's 09:58 stamp (12:58
in New York, squarely midday) would be filed under the open. The four windows
partition the whole session, so every snapshot lands in exactly one.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import project_paths

from . import scan_manifest

#: In the order they are printed. They partition the session; nothing is
#: dropped between them.
CHECKPOINTS = ("open", "midday", "final hour", "close")

#: Exchange-time boundaries. `open` is everything before 11:00 (a pre-market
#: scan is still the scan the open was traded from), `close` everything from
#: 16:00 (the after-close slot, where the trader says the strong updates land).
_WINDOWS: dict[str, tuple[time | None, time | None]] = {
    "open": (None, time(11, 0)),
    "midday": (time(11, 0), time(15, 0)),
    "final hour": (time(15, 0), time(16, 0)),
    "close": (time(16, 0), None),
}

_SEPARATOR = " · "


def _exchange_time(stamp: datetime) -> datetime:
    from market_calendar import MARKET_TZ

    if stamp.tzinfo is None:
        # A naive stamp is attached, never stripped: it was written by the
        # desk's own clock, so it means the desk's own offset.
        stamp = stamp.astimezone()
    return stamp.astimezone(MARKET_TZ)


def _checkpoint_for(moment: datetime) -> str:
    clock = moment.time()
    for name in CHECKPOINTS:
        start, end = _WINDOWS[name]
        if (start is None or clock >= start) and (end is None or clock < end):
            return name
    return CHECKPOINTS[-1]


def load_snapshots(session: date, *, reports_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every dated report copy for ``session``, oldest first.

    Selected by the copy's own filename date (the date the scan finished in its
    own offset) and ordered by the stamp inside it, so a file whose name was
    written in one timezone and whose stamp is in another still sorts by when
    it actually happened.
    """
    folder = Path(reports_dir) if reports_dir is not None else scan_manifest.scan_reports_dir()
    snapshots: list[dict[str, Any]] = []
    try:
        candidates = sorted(folder.glob(f"{session.isoformat()}_*.json"))
    except OSError:
        return []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        stamp = scan_manifest._parse_stamp(payload.get("finished_at"))
        if stamp is None:
            continue
        payload["_stamp"] = stamp
        payload["_exchange"] = _exchange_time(stamp)
        payload["_checkpoint"] = _checkpoint_for(payload["_exchange"])
        payload["_path"] = str(path)
        snapshots.append(payload)
    snapshots.sort(key=lambda item: item["_exchange"])
    return snapshots


def _buckets_for(snapshot: dict[str, Any], symbol: str) -> list[str]:
    buckets = []
    for row in snapshot.get("rows") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol") or "").strip().upper() != symbol:
            continue
        bucket = str(row.get("bucket") or "").strip()
        if bucket and bucket not in buckets:
            buckets.append(bucket)
    return buckets


def _snapshot_line(name: str, snapshot: dict[str, Any] | None, symbol: str) -> str:
    if snapshot is None:
        return f"{name}{_SEPARATOR}no recorded snapshot"
    clock = snapshot["_exchange"].strftime("%H:%M")
    buckets = _buckets_for(snapshot, symbol)
    inputs = str(snapshot.get("latest_input_bar_session") or "not recorded")
    parts = [name, clock]
    if buckets:
        parts.append("in report")
        parts.append("/".join(buckets))
    else:
        parts.append("not in report")
    parts.append(f"inputs {inputs}")
    if snapshot.get("preview_bar_used"):
        parts.append("preview bar in inputs")
    return _SEPARATOR.join(parts)


def build_report(symbol: str, session: date, *, reports_dir: Path | None = None) -> list[str]:
    """The printable lines. Pure over what is on disk; no clock of its own."""
    symbol = str(symbol or "").strip().upper()
    snapshots = load_snapshots(session, reports_dir=reports_dir)
    lines = [f"{symbol}{_SEPARATOR}session {session.isoformat()}"]

    by_checkpoint: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        # The LAST snapshot inside a window is the one reported: what stood at
        # the end of that checkpoint is what the trader would have acted on.
        by_checkpoint[snapshot["_checkpoint"]] = snapshot
    for name in CHECKPOINTS:
        lines.append(_snapshot_line(name, by_checkpoint.get(name), symbol))

    first_eligible: dict[str, Any] | None = None
    first_published: dict[str, Any] | None = None
    for snapshot in snapshots:
        buckets = _buckets_for(snapshot, symbol)
        if not buckets:
            continue
        if first_eligible is None:
            first_eligible = snapshot
        if first_published is None and "favorite_setup" in buckets:
            first_published = snapshot
    if first_eligible is None or first_published is None:
        # Never a number from a day with nothing recorded: an unmeasured delay
        # is labelled, and the label is what the brief asks for.
        lines.append(
            _SEPARATOR.join(
                [
                    "first eligible: "
                    + (
                        f"{first_eligible['_exchange']:%H:%M}"
                        if first_eligible is not None
                        else "none recorded"
                    ),
                    "first published: "
                    + (
                        f"{first_published['_exchange']:%H:%M}"
                        if first_published is not None
                        else "none recorded"
                    ),
                    "delay unmeasured",
                ]
            )
        )
    else:
        minutes = int(
            (first_published["_exchange"] - first_eligible["_exchange"]).total_seconds() // 60
        )
        lines.append(
            _SEPARATOR.join(
                [
                    f"first eligible {first_eligible['_exchange']:%H:%M}"
                    f" ({first_eligible['_checkpoint']})",
                    f"first published {first_published['_exchange']:%H:%M}"
                    f" ({first_published['_checkpoint']})",
                    f"delay {minutes} min",
                ]
            )
        )
    lines.append(f"snapshots recorded: {len(snapshots)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m master_avwap_lib.scan_replay",
        description=(
            "Replay one session's recorded scan snapshots for one symbol. "
            "Read-only; it reconstructs nothing."
        ),
    )
    parser.add_argument("--symbol", required=True, help="the ticker to trace")
    parser.add_argument(
        "--session",
        default="",
        help="YYYY-MM-DD exchange session (default: the market-local date today)",
    )
    parser.add_argument(
        "--reports-dir",
        default="",
        help="override the dated-copy folder (diagnostics; read-only either way)",
    )
    args = parser.parse_args(argv)

    # The scratch-script rule (2026-09-05): say which home folder was resolved
    # before reading a single byte of it.
    print(f"DATA_DIR: {project_paths.DATA_DIR}")
    try:
        session = (
            date.fromisoformat(args.session.strip())
            if args.session.strip()
            else scan_manifest.market_now().date()
        )
    except ValueError:
        print(f"--session must be YYYY-MM-DD, saw {args.session!r}")
        return 2
    reports_dir = Path(args.reports_dir) if args.reports_dir.strip() else None
    print(f"scan report copies: {reports_dir or scan_manifest.scan_reports_dir()}")
    for line in build_report(args.symbol, session, reports_dir=reports_dir):
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
