"""Re-source the durable daily store's volume from Yahoo (R10.V step 4).

The store mixes two units - IB regular-session round lots and Yahoo consolidated
shares - and the observed ratio is symbol-dependent (SPY 1.0x, TSLA 56x, AAPL
81x, A 162x, NVDA 188x), so no constant converts one into the other. This
rewrites the volume column from **one** source instead.

**Prices are never touched.** Only `volume` and the two provenance columns are
written; open/high/low/close and the set of dates come out exactly as they went
in. That is deliberate: prices in this store are fine (the field-level diff put
genuine restatement at 361 differences on 136 symbol-dates, ten of them a SCCO
dividend), and rewriting them would put a second, unmeasured change inside the
one this packet is accountable for.

**Zero IB traffic.** yfinance only, batched, `auto_adjust=False` - the same call
shape the scanner and the Strength Board already use.

**A row Yahoo does not cover keeps what it had and is COUNTED**, never guessed
and never blanked: an unmeasured row is uncertainty, and deleting the volume of
a bar nobody can re-source would silently narrow every AVWAP anchored across it.

**Nothing runs without `--apply`.** The default is a dry run that measures and
writes a manifest, so the number of files about to change is known before any of
them do. `--apply` freezes a dated copy of the entire directory first
(`evidence_frozen/daily_bars_pre_backfill_<date>`), verifies it by size, and
refuses to proceed if the freeze is incomplete.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ops.daily_bar_cliff import first_cliff, scan_store  # noqa: E402

FROZEN_DIR_NAME = "evidence_frozen"
PRE_BACKFILL_PREFIX = "daily_bars_pre_backfill"
# Yahoo tolerates large ticker lists, but a failed batch costs every symbol in
# it, so the batch is small enough that one failure is cheap to retry.
DEFAULT_BATCH = 40
DEFAULT_PERIOD = "max"
# A file whose rows Yahoo barely covers is LEFT ALONE. Rewriting two rows of a
# 787-row history does not repair it - it manufactures a second boundary in a
# file that had one, which is the exact defect this packet exists to remove.
# Measured on the live dry run: EA, TMHC, JHG, SATS and AVNS each came back with
# a near-empty history and would have had 0-2 rows rewritten.
MIN_COVERAGE = 0.90


@dataclass
class FileOutcome:
    """One symbol's before/after, in the terms the exit gate is written in."""

    symbol: str
    rows: int = 0
    rows_rewritten: int = 0
    rows_left_unknown: int = 0
    cliff_before: str | None = None
    cliff_ratio_before: float | None = None
    cliff_after: str | None = None
    cliff_ratio_after: float | None = None
    measurable_before: bool = True
    measurable_after: bool = True
    status: str = "ok"
    note: str = ""


@dataclass
class BackfillReport:
    started_at: str = ""
    ended_at: str = ""
    applied: bool = False
    store_dir: str = ""
    frozen_copy: str = ""
    symbols_requested: int = 0
    symbols_downloaded: int = 0
    symbols_missing: list[str] = field(default_factory=list)
    files_seen: int = 0
    files_changed: int = 0
    files_failed: int = 0
    files_skipped_low_coverage: int = 0
    files_skipped_would_worsen: int = 0
    rows_rewritten: int = 0
    rows_left_unknown: int = 0
    unmeasurable_before: int = 0
    unmeasurable_after: int = 0
    cliffed_before: int = 0
    cliffed_after: int = 0
    outcomes: list[FileOutcome] = field(default_factory=list)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["outcomes"] = [asdict(item) if not isinstance(item, dict) else item
                               for item in self.outcomes]
        return payload


def default_store_dir() -> Path:
    from project_paths import MASTER_AVWAP_DAILY_BARS_DIR

    return Path(MASTER_AVWAP_DAILY_BARS_DIR)


def default_frozen_dir() -> Path:
    from project_paths import CACHE_DIR

    return Path(CACHE_DIR) / "evidence_snapshots" / FROZEN_DIR_NAME


def symbol_for_stem(stem: str) -> str:
    """`CON_.parquet` holds `CON`. Windows cannot name a file after a device."""
    from master_avwap_lib.legacy import _WINDOWS_RESERVED_FILENAME_STEMS

    name = str(stem or "").strip().upper()
    if name.endswith("_") and name[:-1] in _WINDOWS_RESERVED_FILENAME_STEMS:
        return name[:-1]
    return name


def _default_downloader(symbols: list[str], *, period: str):
    """The one network call. yfinance, batched, adjusted prices OFF."""
    import yfinance as yf

    return yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )


def _symbol_frame(payload, symbol: str):
    """Pull one symbol out of a batched download, however it came back."""
    import pandas as pd

    if payload is None:
        return None
    frame = payload
    try:
        if isinstance(payload.columns, pd.MultiIndex):
            if symbol not in payload.columns.get_level_values(0):
                return None
            frame = payload[symbol]
    except Exception:
        return None
    if frame is None or getattr(frame, "empty", True):
        return None
    return frame


def yahoo_volume_by_date(frame) -> dict[str, float]:
    """`{"YYYY-MM-DD": volume}` in SHARES, from a yfinance daily frame."""
    import pandas as pd

    if frame is None or getattr(frame, "empty", True):
        return {}
    work = frame.copy()
    work.columns = [str(column).strip().lower() for column in work.columns]
    if "volume" not in work.columns:
        return {}
    stamps = pd.to_datetime(work.index, errors="coerce")
    volumes = pd.to_numeric(work["volume"], errors="coerce")
    out: dict[str, float] = {}
    for stamp, volume in zip(stamps, volumes):
        if stamp is None or stamp != stamp or volume != volume:
            continue
        if float(volume) <= 0:
            continue
        out[stamp.date().isoformat()] = float(volume)
    return out


def rewrite_frame(frame, volume_by_date: dict[str, float]) -> tuple[object, int, int]:
    """Return `(frame, rewritten, left_unknown)` - prices and dates untouched.

    A row is rewritten when Yahoo has a volume for its session; it then reads
    `source=yahoo`, `volume_unit=shares`. A row Yahoo does not cover keeps its
    existing value and its existing provenance, and is counted.
    """
    import pandas as pd

    from master_avwap_lib.legacy import (
        DAILY_BAR_SOURCE_COLUMN,
        DAILY_BAR_SOURCE_YAHOO,
        DAILY_BAR_UNIT_COLUMN,
        DAILY_BAR_UNIT_SHARES,
    )

    work = frame.copy()
    stamps = pd.to_datetime(work["datetime"], errors="coerce")
    if DAILY_BAR_SOURCE_COLUMN not in work.columns:
        work[DAILY_BAR_SOURCE_COLUMN] = "unknown"
    if DAILY_BAR_UNIT_COLUMN not in work.columns:
        work[DAILY_BAR_UNIT_COLUMN] = "unknown"
    work[DAILY_BAR_SOURCE_COLUMN] = work[DAILY_BAR_SOURCE_COLUMN].astype("object")
    work[DAILY_BAR_UNIT_COLUMN] = work[DAILY_BAR_UNIT_COLUMN].astype("object")

    rewritten = 0
    left_unknown = 0
    for position, stamp in enumerate(stamps):
        if stamp is None or stamp != stamp:
            left_unknown += 1
            continue
        key = stamp.date().isoformat()
        existing_unit = str(work.iloc[position][DAILY_BAR_UNIT_COLUMN] or "").strip().lower()
        volume = volume_by_date.get(key)
        if volume is None:
            if existing_unit != DAILY_BAR_UNIT_SHARES:
                left_unknown += 1
            continue
        work.iat[position, work.columns.get_loc("volume")] = float(volume)
        work.iat[position, work.columns.get_loc(DAILY_BAR_SOURCE_COLUMN)] = DAILY_BAR_SOURCE_YAHOO
        work.iat[position, work.columns.get_loc(DAILY_BAR_UNIT_COLUMN)] = DAILY_BAR_UNIT_SHARES
        rewritten += 1
    return work, rewritten, left_unknown


def freeze_pre_backfill_copy(store_dir: Path, frozen_dir: Path, *, stamp: str) -> Path:
    """A dated copy of the WHOLE directory before anything is written.

    Verified by file count and total size, and the caller refuses to proceed on
    a mismatch: a backfill whose undo is incomplete is not a backfill, it is a
    one-way door.
    """
    target = Path(frozen_dir) / f"{PRE_BACKFILL_PREFIX}_{stamp}"
    if target.exists():
        raise FileExistsError(f"{target} already exists; refusing to overwrite a frozen copy")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(store_dir, target)
    sources = sorted(Path(store_dir).glob("*.parquet"))
    copies = sorted(target.glob("*.parquet"))
    if len(sources) != len(copies):
        raise RuntimeError(
            f"frozen copy is incomplete: {len(copies)} of {len(sources)} files"
        )
    source_bytes = sum(path.stat().st_size for path in sources)
    copy_bytes = sum(path.stat().st_size for path in copies)
    if source_bytes != copy_bytes:
        raise RuntimeError(
            f"frozen copy size mismatch: {copy_bytes} vs {source_bytes} bytes"
        )
    return target


def run_backfill(
    *,
    store_dir: Path | None = None,
    frozen_dir: Path | None = None,
    downloader=None,
    apply: bool = False,
    batch_size: int = DEFAULT_BATCH,
    period: str = DEFAULT_PERIOD,
    limit: int | None = None,
    now: datetime | None = None,
) -> BackfillReport:
    import pandas as pd

    from master_avwap_lib.legacy import _write_daily_bar_parquet

    moment = now or datetime.now(timezone.utc)
    store = Path(store_dir or default_store_dir())
    report = BackfillReport(
        started_at=moment.isoformat(timespec="seconds"),
        applied=bool(apply),
        store_dir=str(store),
    )
    paths = sorted(store.glob("*.parquet"))
    if limit is not None:
        paths = paths[:limit]
    report.files_seen = len(paths)
    if not paths:
        report.ended_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return report

    if apply:
        frozen = freeze_pre_backfill_copy(
            store, Path(frozen_dir or default_frozen_dir()), stamp=moment.date().isoformat()
        )
        report.frozen_copy = str(frozen)

    # `CON_.parquet` holds CON: the file is named for what Windows allows, the
    # download has to ask for what the market calls it.
    symbol_by_stem = {path.stem: symbol_for_stem(path.stem) for path in paths}
    symbols = sorted(set(symbol_by_stem.values()))
    report.symbols_requested = len(symbols)
    download = downloader or _default_downloader

    volumes: dict[str, dict[str, float]] = {}
    for start in range(0, len(symbols), max(1, batch_size)):
        chunk = symbols[start:start + max(1, batch_size)]
        try:
            payload = download(chunk, period=period)
        except Exception:
            logging.exception("batch download failed for %s", chunk[:3])
            continue
        for symbol in chunk:
            by_date = yahoo_volume_by_date(_symbol_frame(payload, symbol))
            if by_date:
                volumes[symbol] = by_date

    # A batched download silently drops the odd ticker - BK came back empty in a
    # batch and full on its own. One individual retry each, so "no data" means
    # Yahoo has none rather than that the batch was unlucky.
    for symbol in sorted(set(symbols) - set(volumes)):
        try:
            by_date = yahoo_volume_by_date(_symbol_frame(download([symbol], period=period), symbol))
        except Exception:
            logging.exception("retry download failed for %s", symbol)
            continue
        if by_date:
            volumes[symbol] = by_date

    report.symbols_downloaded = len(volumes)
    report.symbols_missing = sorted(set(symbols) - set(volumes))

    for path in paths:
        symbol = path.stem
        outcome = FileOutcome(symbol=symbol)
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            outcome.status = "unreadable"
            outcome.note = str(exc)[:160]
            report.files_failed += 1
            report.outcomes.append(outcome)
            continue
        outcome.rows = int(len(frame))
        before = first_cliff(frame)
        outcome.measurable_before = before.measurable
        outcome.cliff_before = before.date
        outcome.cliff_ratio_before = before.ratio
        if not before.measurable:
            report.unmeasurable_before += 1
        elif before.is_cliff:
            report.cliffed_before += 1

        by_date = volumes.get(symbol_by_stem.get(symbol, symbol))
        if not by_date:
            outcome.status = "no_yahoo_data"
            outcome.rows_left_unknown = outcome.rows
            report.rows_left_unknown += outcome.rows
            # An untouched file keeps whatever cliff it had, and that cliff is
            # still in the store - so it counts. Leaving it out made the first
            # live run's manifest report 44 where an independent scan of the
            # same store found 53, and a summary that disagrees with the thing
            # it summarises is worse than no summary.
            outcome.cliff_after = before.date
            outcome.cliff_ratio_after = before.ratio
            outcome.measurable_after = before.measurable
            if not before.measurable:
                report.unmeasurable_after += 1
            elif before.is_cliff:
                report.cliffed_after += 1
            report.outcomes.append(outcome)
            continue

        rewritten_frame, rewritten, left_unknown = rewrite_frame(frame, by_date)
        outcome.rows_rewritten = rewritten
        outcome.rows_left_unknown = left_unknown

        after = first_cliff(rewritten_frame)
        outcome.measurable_after = after.measurable
        outcome.cliff_after = after.date
        outcome.cliff_ratio_after = after.ratio

        # Two refusals, both learned from the live dry run rather than guessed.
        #
        # Coverage: a near-empty download would have rewritten 2 of EA's 787
        # rows. That is not a repair - it puts a fresh unit boundary inside a
        # file, which is the defect itself.
        #
        # Worsening: whatever the reason, a file this run would leave with a
        # cliff it did not have (or a bigger one) is left alone. A repair that
        # can make a file worse is not a repair.
        coverage = (rewritten / outcome.rows) if outcome.rows else 0.0
        worsens = bool(
            after.is_cliff
            and (
                not before.is_cliff
                or (after.ratio or 0.0) > (before.ratio or 0.0) * 1.01
            )
        )
        if rewritten and coverage < MIN_COVERAGE:
            outcome.status = "insufficient_coverage"
            outcome.note = f"yahoo covered {coverage:.1%} of {outcome.rows} rows"
            report.files_skipped_low_coverage += 1
            report.rows_left_unknown += outcome.rows
            outcome.rows_rewritten = 0
            outcome.rows_left_unknown = outcome.rows
            outcome.cliff_after = before.date
            outcome.cliff_ratio_after = before.ratio
            if before.is_cliff:
                report.cliffed_after += 1
            report.outcomes.append(outcome)
            continue
        if rewritten and worsens:
            outcome.status = "would_worsen"
            outcome.note = (
                f"cliff {before.ratio or 0:.0f}x -> {after.ratio or 0:.0f}x; left unchanged"
            )
            report.files_skipped_would_worsen += 1
            report.rows_left_unknown += outcome.rows
            outcome.rows_rewritten = 0
            outcome.rows_left_unknown = outcome.rows
            outcome.cliff_after = before.date
            outcome.cliff_ratio_after = before.ratio
            if before.is_cliff:
                report.cliffed_after += 1
            report.outcomes.append(outcome)
            continue

        report.rows_rewritten += rewritten
        report.rows_left_unknown += left_unknown
        if not after.measurable:
            report.unmeasurable_after += 1
        elif after.is_cliff:
            report.cliffed_after += 1

        if apply and rewritten:
            try:
                _write_daily_bar_parquet(path, rewritten_frame)
                report.files_changed += 1
            except Exception as exc:
                outcome.status = "write_failed"
                outcome.note = str(exc)[:160]
                report.files_failed += 1
        elif rewritten:
            report.files_changed += 1
            outcome.status = "would_change"
        report.outcomes.append(outcome)

    report.ended_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return report


def write_manifest(report: BackfillReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    return path


def _main(argv: list[str]) -> int:  # pragma: no cover - operator entry point
    parser = argparse.ArgumentParser(
        description="Re-source the durable daily store's volume from Yahoo (R10.V step 4)"
    )
    parser.add_argument("--apply", action="store_true", help="write; default is a dry run")
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--frozen-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.time()
    report = run_backfill(
        store_dir=args.dir,
        frozen_dir=args.frozen_dir,
        apply=args.apply,
        batch_size=args.batch_size,
        period=args.period,
        limit=args.limit,
    )
    # The same stamp the frozen copy uses (UTC). The first live run named the
    # copy 2026-08-23 and the manifest 2026-08-22, from the same run.
    stamp = (report.started_at or "")[:10] or datetime.now(timezone.utc).date().isoformat()
    manifest = args.manifest or (
        default_frozen_dir()
        / f"{PRE_BACKFILL_PREFIX}_manifest_{stamp}"
        f"{'' if args.apply else '_dryrun'}.json"
    )
    write_manifest(report, manifest)

    print(f"{'APPLIED' if args.apply else 'DRY RUN'} in {time.time() - started:.1f}s")
    print(f"  files seen          {report.files_seen}")
    print(f"  symbols downloaded  {report.symbols_downloaded} of {report.symbols_requested}")
    print(f"  files changed       {report.files_changed}")
    print(f"  rows rewritten      {report.rows_rewritten:,}")
    print(f"  rows left unknown   {report.rows_left_unknown:,}")
    print(f"  skipped (coverage)  {report.files_skipped_low_coverage}")
    print(f"  skipped (worsens)   {report.files_skipped_would_worsen}")
    print(f"  cliffed before      {report.cliffed_before}")
    print(f"  cliffed after       {report.cliffed_after}")
    print(f"  unmeasurable before {report.unmeasurable_before} / after {report.unmeasurable_after}")
    if report.files_failed:
        print(f"  FAILURES            {report.files_failed}")
    if report.frozen_copy:
        print(f"  frozen copy         {report.frozen_copy}")
    print(f"  manifest            {manifest}")
    if args.apply:
        live = scan_store(Path(args.dir or default_store_dir()))
        print(f"  live store now      {live.cliffed} cliffed / {live.measurable} measurable "
              f"/ {live.unmeasurable} unmeasurable")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
