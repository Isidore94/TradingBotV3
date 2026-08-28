#!/usr/bin/env python3
"""Champion vs challenger AVWAP bands, one row per bar since an anchor (Phase 0.10 B-1).

The eyeball test. Give it a symbol and an anchor date and it prints the two
formulas side by side for every session since that anchor, so the trader can
hover the same bar in OneOption / Option Stalker Pro and compare three numbers
without reading any code:

    .venv/Scripts/python.exe scripts/avwap_band_variant_fit.py OKTA 2026-05-29

Columns: the champion is ``master_avwap_lib.legacy.calc_anchored_vwap_bands``,
frozen by decision 0008 and called here, never edited. The variant is
``indicators.avwap_band_variants``, the OneOption formula replicated in B-0:
an anchored HLC/3 centre with a 20-close population Bollinger sigma as its
half-width (``docs/AVWAP_BAND_VARIANT_STUDY.md`` section 2b).

The champion publishes only its FINAL bar, so its per-bar column is produced by
calling it once per session on the frame truncated at that session. That is
quadratic and it does not matter: a year of sessions is a few hundred calls.

Offline. It reads the durable daily store through the same loader the playbook
study uses and makes no provider call. It writes nothing unless ``--csv`` is
passed, and then only into ``OUTPUT_DIR/reports/``.

An unmeasurable cell prints EMPTY, never 0.00 - a band that does not exist yet
and a band sitting exactly on its centre are different claims.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from indicators.avwap_band_variants import (  # noqa: E402
    FEATURE_VERSION,
    oneoption_avwap_band_series,
)

DEFAULT_LOOKBACK = 20

COLUMNS = (
    ("date", "date", 10),
    ("close", "close", 9),
    ("champion_centre", "champ ctr", 10),
    ("champion_sigma", "champ sig", 10),
    ("champion_upper_1", "champ +1", 10),
    ("champion_lower_1", "champ -1", 10),
    ("variant_centre", "var ctr", 10),
    ("variant_sigma", "var sig", 10),
    ("variant_upper_1", "var +1", 10),
    ("variant_lower_1", "var -1", 10),
)


def _finite(value):
    """Champion cells arrive as float('nan') when it has no volume to weigh."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def build_rows(frame, anchor_index: int, lookback: int = DEFAULT_LOOKBACK) -> list[dict]:
    """One dict per session from ``anchor_index`` to the end of ``frame``.

    ``frame`` is a daily-bar DataFrame with ``datetime``/``open``/``high``/
    ``low``/``close``/``volume`` columns, as the durable store hands it over.
    """
    from master_avwap_lib.legacy import calc_anchored_vwap_bands

    anchor = int(anchor_index)
    if anchor < 0 or anchor >= len(frame):
        raise IndexError(f"anchor_index {anchor_index} outside 0..{len(frame) - 1}")

    variant = oneoption_avwap_band_series(frame, anchor, lookback=lookback)

    rows: list[dict] = []
    for index in range(anchor, len(frame)):
        # The champion answers for the last bar of whatever frame it is given.
        champion_centre, champion_sigma, champion_bands = calc_anchored_vwap_bands(
            frame.iloc[: index + 1], anchor
        )
        stamp = frame["datetime"].iloc[index]
        rows.append(
            {
                "date": stamp.strftime("%Y-%m-%d"),
                "close": float(frame["close"].iloc[index]),
                "champion_centre": _finite(champion_centre),
                "champion_sigma": _finite(champion_sigma),
                "champion_upper_1": _finite(champion_bands.get("UPPER_1")),
                "champion_lower_1": _finite(champion_bands.get("LOWER_1")),
                "variant_centre": variant["centre"][index],
                "variant_sigma": variant["sigma"][index],
                "variant_upper_1": variant["upper_1"][index],
                "variant_lower_1": variant["lower_1"][index],
                "variant_formula_version": FEATURE_VERSION,
            }
        )
    return rows


def _cell(value, width: int) -> str:
    if value is None:
        return "".rjust(width)
    if isinstance(value, str):
        return value.ljust(width)
    return f"{value:,.2f}".rjust(width)


def render_table(rows: list[dict], *, symbol: str, anchor_date: str, lookback: int) -> str:
    header = " | ".join(label.rjust(width) for _key, label, width in COLUMNS)
    lines = [
        f"{symbol}  anchor {anchor_date}  lookback {lookback}  "
        f"champion=calc_anchored_vwap_bands  variant={FEATURE_VERSION}",
        "an empty cell is unmeasurable, not zero",
        "",
        header,
        "-" * len(header),
    ]
    for row in rows:
        lines.append(" | ".join(_cell(row.get(key), width) for key, _label, width in COLUMNS))
    return "\n".join(lines)


def _load_frame(symbol: str):
    """The durable D1 store, through the playbook study's own loader."""
    from setup_playbook_study import _load_daily_frame

    frame = _load_daily_frame(symbol)
    if frame is None:
        raise SystemExit(
            f"{symbol}: no usable durable daily bars "
            "(missing file, or fewer sessions than the study's minimum)"
        )
    return frame.reset_index(drop=True)


def _resolve_anchor(frame, anchor_date: str) -> int:
    import pandas as pd

    stamp = pd.Timestamp(anchor_date)
    matches = frame.index[frame["datetime"] == stamp]
    if len(matches):
        return int(matches[0])
    # Not a session. Take the first session at or after it rather than guessing
    # backwards: an anchor is a bar the trader points at, and moving it EARLIER
    # would silently include a session they did not mean.
    later = frame.index[frame["datetime"] >= stamp]
    if not len(later):
        raise SystemExit(f"{anchor_date} is after the last session in the store")
    resolved = int(later[0])
    print(
        f"note: {anchor_date} is not a session; anchoring on "
        f"{frame['datetime'].iloc[resolved].strftime('%Y-%m-%d')} instead"
    )
    return resolved


def _write_csv(rows: list[dict], symbol: str, anchor_date: str) -> Path:
    from project_paths import REPORTS_DIR

    path = Path(REPORTS_DIR) / f"avwap_band_variant_fit_{symbol}_{anchor_date}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key, _label, _width in COLUMNS] + ["variant_formula_version"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Champion vs OneOption-variant AVWAP bands per bar since an anchor"
    )
    parser.add_argument("symbol")
    parser.add_argument("anchor_date", help="YYYY-MM-DD")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument(
        "--csv",
        action="store_true",
        help="also write the table to OUTPUT_DIR/reports/ (the only write this script does)",
    )
    args = parser.parse_args(argv)

    symbol = args.symbol.upper().strip()
    frame = _load_frame(symbol)
    anchor = _resolve_anchor(frame, args.anchor_date)
    rows = build_rows(frame, anchor, lookback=args.lookback)
    anchor_date = frame["datetime"].iloc[anchor].strftime("%Y-%m-%d")
    print(render_table(rows, symbol=symbol, anchor_date=anchor_date, lookback=args.lookback))
    if args.csv:
        print(f"\nwrote {_write_csv(rows, symbol, anchor_date)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
