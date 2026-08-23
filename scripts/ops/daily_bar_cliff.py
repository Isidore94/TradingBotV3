"""Measure the daily-bar volume cliff (R10.V steps 4 and 6).

One definition, used by the backfill's before/after manifest AND by the nightly
health check, so "0 files over 20x" means the same thing in both places. If the
two ever measured it differently, the gate would be unfalsifiable.

**The cliff.** IB returns regular-session daily volume in round lots and Yahoo
returns the consolidated session in shares, so a file that took its early
history from one and its later history from the other has a **step** in its
volume series - measured on the live store at a median x158, with the step at
2026-07-29 in 1,179 of the 1,236 files the 2026-08-21 scan rewrote. AVWAP is
volume-weighted, so a step re-weights every level computed across it.

**The measure is a ratio of medians across a candidate boundary**, not of two
adjacent bars: single-bar ratios are noisy (a holiday half-session, an
index-rebalance print) and would flag files that are fine. A window of ten bars
either side is wide enough to be stable and narrow enough to date the step.

Nothing here writes. It reads parquet and returns numbers.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

# A splice between shares and round lots is a factor of ~100 before any
# RTH-vs-consolidated difference, so 20x is far below the real signal and far
# above ordinary volume variation. It is the same threshold the read-only audit
# used, kept so the before/after numbers are comparable.
CLIFF_RATIO_THRESHOLD = 20.0
CLIFF_WINDOW = 10
# Below this many usable bars a file cannot be judged. Those files are reported
# as UNMEASURABLE - never as clean. The read-only audit found 221 of them.
CLIFF_MIN_BARS = CLIFF_WINDOW * 2


@dataclass(frozen=True)
class CliffReading:
    """What one file's volume series says about itself."""

    measurable: bool
    ratio: float | None = None
    date: str | None = None
    direction: str = ""          # "down" (shares -> lots) or "up"
    bars: int = 0
    reason: str = ""

    @property
    def is_cliff(self) -> bool:
        return bool(self.measurable and self.ratio is not None and self.ratio > CLIFF_RATIO_THRESHOLD)


@dataclass
class StoreCliffReport:
    """The whole store, counted by outcome. Unmeasurable is its own bucket."""

    files: int = 0
    measurable: int = 0
    unmeasurable: int = 0
    cliffed: int = 0
    ratios: list[float] = field(default_factory=list)
    worst: list[tuple[str, float, str]] = field(default_factory=list)

    @property
    def median_ratio(self) -> float | None:
        return statistics.median(self.ratios) if self.ratios else None

    def as_dict(self) -> dict:
        return {
            "files": self.files,
            "measurable": self.measurable,
            "unmeasurable": self.unmeasurable,
            "cliffed": self.cliffed,
            "median_cliff_ratio": self.median_ratio,
            "threshold": CLIFF_RATIO_THRESHOLD,
            "window": CLIFF_WINDOW,
            "worst": [
                {"symbol": symbol, "ratio": ratio, "date": date}
                for symbol, ratio, date in sorted(self.worst, key=lambda row: -row[1])[:20]
            ],
        }


def _usable(frame) -> tuple[list[float], list[str]]:
    """Volumes that carry a number, with their dates. Blanks are not zeros."""
    if frame is None or getattr(frame, "empty", True):
        return [], []
    if "volume" not in frame.columns or "datetime" not in frame.columns:
        return [], []
    import pandas as pd

    work = frame[["datetime", "volume"]].copy()
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce")
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work = work.dropna(subset=["datetime", "volume"])
    work = work[work["volume"] > 0].sort_values("datetime")
    return (
        [float(value) for value in work["volume"].tolist()],
        [stamp.date().isoformat() for stamp in work["datetime"].tolist()],
    )


def _boundary_date(
    volumes: list[float],
    dates: list[str],
    index: int,
    before: float,
    after: float,
    direction: str,
) -> str:
    """The bar the step actually happened on, not the window that noticed it.

    A rolling median crosses about half a window BEFORE the splice - the trailing
    window has already filled with post-splice bars - so the raw crossing index
    dates the step several sessions early, and this manifest is read by date.
    The two levels are known at the crossing, so their geometric midpoint
    separates them cleanly: the boundary is the first bar in the neighbourhood
    that lands on the far side of it.
    """
    midpoint = (before * after) ** 0.5
    start = max(0, index - CLIFF_WINDOW)
    stop = min(len(volumes), index + CLIFF_WINDOW + 1)
    for position in range(start, stop):
        value = volumes[position]
        crossed = value < midpoint if direction == "down" else value > midpoint
        if crossed:
            return dates[position]
    return dates[index]


def first_cliff(frame) -> CliffReading:
    """The earliest boundary where the volume level steps by more than the threshold.

    Returns the FIRST such boundary rather than the largest, because the date is
    what identifies the splice - the backfill's before/after table is read by
    date, and the largest step in a repaired file is meaningless.
    """
    volumes, dates = _usable(frame)
    if len(volumes) < CLIFF_MIN_BARS:
        return CliffReading(
            measurable=False,
            bars=len(volumes),
            reason=f"fewer than {CLIFF_MIN_BARS} bars carry a volume",
        )
    for index in range(CLIFF_WINDOW, len(volumes) - CLIFF_WINDOW + 1):
        before = statistics.median(volumes[index - CLIFF_WINDOW:index])
        after = statistics.median(volumes[index:index + CLIFF_WINDOW])
        if before <= 0 or after <= 0:
            continue
        if after and before / after > CLIFF_RATIO_THRESHOLD:
            return CliffReading(
                measurable=True,
                ratio=before / after,
                date=_boundary_date(volumes, dates, index, before, after, "down"),
                direction="down",
                bars=len(volumes),
            )
        if before and after / before > CLIFF_RATIO_THRESHOLD:
            return CliffReading(
                measurable=True,
                ratio=after / before,
                date=_boundary_date(volumes, dates, index, before, after, "up"),
                direction="up",
                bars=len(volumes),
            )
    return CliffReading(measurable=True, bars=len(volumes))


def scan_store(directory: Path | str, *, limit: int | None = None) -> StoreCliffReport:
    """Every `*.parquet` in the durable daily store, counted by outcome."""
    import pandas as pd

    report = StoreCliffReport()
    root = Path(directory)
    try:
        paths = sorted(root.glob("*.parquet"))
    except OSError:
        return report
    if limit is not None:
        paths = paths[:limit]
    for path in paths:
        report.files += 1
        try:
            frame = pd.read_parquet(path, columns=["datetime", "volume"])
        except Exception:
            # Unreadable is unmeasurable, never clean.
            report.unmeasurable += 1
            continue
        reading = first_cliff(frame)
        if not reading.measurable:
            report.unmeasurable += 1
            continue
        report.measurable += 1
        if reading.is_cliff:
            report.cliffed += 1
            report.ratios.append(reading.ratio or 0.0)
            report.worst.append((path.stem, reading.ratio or 0.0, reading.date or ""))
    return report


def _main(argv: list[str]) -> int:  # pragma: no cover - operator convenience
    import argparse
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_paths import MASTER_AVWAP_DAILY_BARS_DIR

    parser = argparse.ArgumentParser(description="Measure the daily-bar volume cliff (read-only)")
    parser.add_argument("--dir", type=Path, default=Path(MASTER_AVWAP_DAILY_BARS_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = scan_store(args.dir, limit=args.limit)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
        return 0
    print(f"files            {report.files}")
    print(f"measurable       {report.measurable}")
    print(f"unmeasurable     {report.unmeasurable}  (not clean - unmeasured)")
    print(f"cliffed >{CLIFF_RATIO_THRESHOLD:g}x     {report.cliffed}")
    if report.median_ratio:
        print(f"median ratio     {report.median_ratio:.1f}x")
    for symbol, ratio, date in sorted(report.worst, key=lambda row: -row[1])[:10]:
        print(f"  {symbol:<8} {ratio:>10.1f}x at {date}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(_main(sys.argv[1:]))
