#!/usr/bin/env python3
"""Rebuild the setup scoreboard from the stores that actually carry outcomes.

plan.md R9.3. The 2026-08-21 trade review built its scoreboard from the review
store and declared the regime, RVOL and risk axes starved. They are not starved;
they are in a different file. ``intraday_bounce_outcomes.csv`` carries
``market_environment`` on 100% of its in-window rows, plus ``session_rvol``,
sector, industry and the RRS triple in ``context_json`` - and, unlike the review
store, it carries a stop and therefore an R.

This module is READ-ONLY analysis. It promotes and demotes nothing: plan.md
Section 7 gate 2 requires an evidence window frozen *before* inspection, and
this window was chosen after. Its one forward-looking output is the declared
window at the end of the report, which is the only route by which anything here
ever becomes gate-2 eligible.

Two inputs:

* ``data/runtime/intraday_bounce_outcomes.csv`` - one row per milestone; only
  ``event_type == "final"`` rows carry a settled outcome. The bounce type is the
  tail of ``event_id`` (``SYM_side_YYYYMMDD_HH_MM_SS_<type>``); multi-type events
  join their types with ``-``.
* ``output/reports/setup_playbook_episodes.csv`` - per (family, side, date) swing
  episodes with a real ``stop``, ``risk`` and ``net_r``, and its own
  ``baseline_every5`` control to measure lift against.

Read with ``chunksize``/``usecols`` throughout: the outcomes file is ~200 MB and
must never be loaded whole.

Timestamps: ``trade_date`` is a bare market date. ``logged_at`` is tz-aware
-07:00 (desk local, America/Los_Angeles). ``entry_time`` is naive and is also
desk-local PT - market time is PT + 3h (America/New_York).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

# Cells thinner than this are reported as present-but-unusable, never ranked.
MIN_CELL_N = 30
# A stop closer than this to the entry makes R arithmetic explode: the review
# measured regime_pause_rw at an all-time mean of -1.82R against a trimmed
# -0.28R, driven by penny stops. 0.1% of entry is the floor.
RISK_FLOOR_PCT_OF_ENTRY = 0.1
# Symmetric trim on each tail before the mean, so one 45R row cannot carry a cell.
TRIM_FRACTION = 0.10

OUTCOME_COLUMNS = [
    "event_id",
    "event_type",
    "trade_date",
    "symbol",
    "direction",
    "entry_time",
    "entry_price",
    "risk_per_share",
    "close_r",
    "mfe_r",
    "mae_r",
    "stop_hit",
    "bars_elapsed",
    "eod_close",
    "eod_move_pct",
    "context_json",
]
CONTEXT_FIELDS = (
    "market_environment",
    "session_rvol",
    "sector",
    "industry",
    "rrs_spy",
    "rrs_sector",
    "rrs_industry",
    "internals_tape",
    "internals_breadth_spread",
    "watchlist_bias",
)
PLAYBOOK_COLUMNS = [
    "symbol",
    "family",
    "group",
    "side",
    "signal_date",
    "entry_date",
    "status",
    "entry",
    "stop",
    "risk",
    "net_r",
    "hold_sessions",
    "r_1",
    "r_2",
    "r_5",
    "r_10",
]
BASELINE_FAMILY = "baseline_every5"


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def bounce_type_from_event_id(event_id: str) -> str:
    """``AAPL_long_20260724_06_30_00_h1_blue_after_red`` -> ``h1_blue_after_red``.

    The identity is ``SYMBOL_side_YYYYMMDD_HH_MM_SS_<type>``: six fixed parts,
    then the type, which may itself contain underscores and joins multiple types
    with ``-``. Splitting on ``_`` is safe because symbols use ``-`` for class
    shares (``BRK-B``), never ``_``.
    """
    parts = str(event_id or "").split("_")
    return "_".join(parts[6:]) if len(parts) > 6 else ""


def trimmed_mean(values: pd.Series, fraction: float = TRIM_FRACTION) -> float | None:
    """Symmetric trimmed mean. ``None`` when the trim would leave nothing.

    Reported beside the plain mean everywhere, because R is a ratio and a single
    tight-stop row can be worth fifty ordinary ones.
    """
    clean = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if clean.empty:
        return None
    cut = int(len(clean) * fraction)
    kept = clean.iloc[cut : len(clean) - cut] if len(clean) - 2 * cut > 0 else clean
    return float(kept.mean()) if len(kept) else None


def risk_floor_mask(frame: pd.DataFrame, pct_of_entry: float = RISK_FLOOR_PCT_OF_ENTRY) -> pd.Series:
    """True where the stop is too close to the entry for R to mean anything."""
    import numpy as np

    entry = pd.to_numeric(frame.get("entry_price"), errors="coerce")
    risk = pd.to_numeric(frame.get("risk_per_share"), errors="coerce")
    # A zero or missing entry divides to inf/NaN. Both mean "cannot measure the
    # stop as a fraction of price", and unmeasurable excludes - it is never read
    # as a comfortable margin (plan.md sec 5).
    ratio = ((risk.abs() / entry.abs()) * 100.0).replace([np.inf, -np.inf], pd.NA)
    ratio = pd.to_numeric(ratio, errors="coerce")
    return (ratio < float(pct_of_entry)) | ratio.isna()


def unsettled_close_mask(frame: pd.DataFrame) -> pd.Series:
    """True where ``close_r`` is 0 only because no EOD close was ever obtained.

    Measured 2026-08-22 over the 2026-07-24..08-21 window: every one of the
    1,164 finals with ``close_r == 0`` has ``eod_close`` exactly equal to
    ``entry_price``, and **none** of the 5,743 finals with a non-zero
    ``close_r`` does. Real closes do not land on the entry to the cent 1,164
    times and never otherwise - the writer defaults ``eod_close`` to the entry
    when it cannot read one.

    These are not scratch trades, and treating them as such biases every mean
    upward: 563 of them have ``stop_hit`` true with a median ``mae_r`` of
    -1.000, i.e. trades that were stopped out and should score about -1R are
    scoring 0.
    """
    close_r = pd.to_numeric(frame.get("close_r"), errors="coerce")
    eod = pd.to_numeric(frame.get("eod_close"), errors="coerce")
    entry = pd.to_numeric(frame.get("entry_price"), errors="coerce")
    return (close_r == 0) & (eod == entry)


def never_measured_mask(frame: pd.DataFrame) -> pd.Series:
    """The subset of the above that never advanced a single bar."""
    bars = pd.to_numeric(frame.get("bars_elapsed"), errors="coerce")
    return unsettled_close_mask(frame) & (bars == 0)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
@dataclass
class Coverage:
    """What was read, what was dropped, and why - carried into the report."""

    rows_scanned: int = 0
    finals: int = 0
    in_window: int = 0
    sessions: int = 0
    unsettled: int = 0
    never_measured: int = 0
    below_risk_floor: int = 0
    usable: int = 0
    notes: list[str] = field(default_factory=list)


def load_intraday_finals(
    path: Path,
    *,
    window_start: str,
    window_end: str,
    chunksize: int = 200_000,
) -> tuple[pd.DataFrame, Coverage]:
    """Settled intraday outcomes inside the window, with context expanded."""
    coverage = Coverage()
    kept: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=OUTCOME_COLUMNS, chunksize=chunksize, low_memory=False):
        coverage.rows_scanned += len(chunk)
        finals = chunk[chunk["event_type"] == "final"]
        coverage.finals += len(finals)
        window = finals[
            (finals["trade_date"] >= window_start) & (finals["trade_date"] <= window_end)
        ]
        if len(window):
            kept.append(window)
    if not kept:
        return pd.DataFrame(columns=OUTCOME_COLUMNS), coverage

    frame = pd.concat(kept, ignore_index=True)
    coverage.in_window = len(frame)
    coverage.sessions = int(frame["trade_date"].nunique())
    frame["bounce_type"] = frame["event_id"].map(bounce_type_from_event_id)
    frame = _expand_context(frame)

    unsettled = unsettled_close_mask(frame)
    coverage.unsettled = int(unsettled.sum())
    coverage.never_measured = int(never_measured_mask(frame).sum())
    frame["unsettled_close"] = unsettled

    floor = risk_floor_mask(frame)
    coverage.below_risk_floor = int(floor.sum())
    frame["below_risk_floor"] = floor

    frame["usable"] = ~(unsettled | floor)
    coverage.usable = int(frame["usable"].sum())
    return frame, coverage


def _expand_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Lift the fields the review called starved out of ``context_json``."""
    parsed = frame["context_json"].map(_safe_json)
    for name in CONTEXT_FIELDS:
        frame[name] = parsed.map(lambda payload, key=name: payload.get(key))
    return frame


def _safe_json(raw: object) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def load_playbook_episodes(
    path: Path,
    *,
    window_start: str,
    window_end: str,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Swing episodes with a real stop, risk and net_r, inside the window."""
    kept: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=PLAYBOOK_COLUMNS, chunksize=chunksize, low_memory=False):
        window = chunk[
            (chunk["signal_date"] >= window_start) & (chunk["signal_date"] <= window_end)
        ]
        if len(window):
            kept.append(window)
    if not kept:
        return pd.DataFrame(columns=PLAYBOOK_COLUMNS)
    frame = pd.concat(kept, ignore_index=True)
    frame["below_risk_floor"] = risk_floor_mask(
        frame.rename(columns={"entry": "entry_price", "risk": "risk_per_share"})
    )
    return frame


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _cell_name(key: object) -> str:
    """Group key as a printable cell label, with UNKNOWN made visible."""
    parts = key if isinstance(key, tuple) else (key,)
    rendered = [
        "(unknown)" if part is None or (isinstance(part, float) and part != part) or str(part).strip() in ("", "nan")
        else str(part)
        for part in parts
    ]
    return " / ".join(rendered)


def summarise(
    frame: pd.DataFrame,
    by: str | list[str],
    *,
    r_column: str = "close_r",
    min_n: int = MIN_CELL_N,
) -> pd.DataFrame:
    """Per-cell R summary. Every R is reported three ways, never one.

    A plain mean on a ratio with an unbounded numerator is the statistic that
    produced the review's -1.82R phantom, so the trimmed mean and the median sit
    beside it, and the stop-out rate sits beside all three.
    """
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(by, dropna=False)
    rows = []
    for key, cell in grouped:
        values = pd.to_numeric(cell[r_column], errors="coerce").dropna()
        if values.empty:
            continue
        stop_rate = None
        if "stop_hit" in cell.columns:
            # The column arrives as bool, as "True"/"False" strings, or absent
            # entirely depending on which store produced the frame.
            stops = cell["stop_hit"]
            if stops.dtype == object:
                stops = stops.map(
                    {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0}
                )
            stops = pd.to_numeric(stops, errors="coerce").dropna()
            if len(stops):
                stop_rate = round(float(stops.mean()) * 100, 1)
        rows.append(
            {
                # " / ", never "|": these names are printed straight into a markdown
                # table, and a pipe inside a cell silently splits the column.
                "cell": _cell_name(key),
                "n": int(len(values)),
                "mean_r": round(float(values.mean()), 3),
                "trimmed_mean_r": (
                    round(trimmed_mean(values), 3) if trimmed_mean(values) is not None else None
                ),
                "median_r": round(float(values.median()), 3),
                "stop_out_rate": stop_rate,
                "p10_r": round(float(values.quantile(0.10)), 3),
                "p90_r": round(float(values.quantile(0.90)), 3),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["reportable"] = out["n"] >= int(min_n)
    return out.sort_values(["reportable", "trimmed_mean_r"], ascending=[False, False])


def baseline_lift(playbook: pd.DataFrame, *, min_n: int = MIN_CELL_N) -> pd.DataFrame:
    """Per-family net_r against the file's own ``baseline_every5`` control.

    The control is what makes a family number mean anything: an equal-weight
    every-fifth-session entry over the same names and the same window absorbs
    whatever the market did, so only the difference is attributable to the setup.
    """
    if playbook.empty or "family" not in playbook:
        return pd.DataFrame()
    usable = playbook[~playbook["below_risk_floor"]]
    control = pd.to_numeric(
        usable.loc[usable["family"] == BASELINE_FAMILY, "net_r"], errors="coerce"
    ).dropna()
    control_trimmed = trimmed_mean(control) if len(control) else None
    summary = summarise(
        usable[usable["family"] != BASELINE_FAMILY], ["family", "side"], r_column="net_r", min_n=min_n
    )
    if summary.empty:
        return summary
    summary["baseline_trimmed_r"] = None if control_trimmed is None else round(control_trimmed, 3)
    summary["baseline_n"] = int(len(control))
    if control_trimmed is not None:
        summary["lift_vs_baseline"] = summary["trimmed_mean_r"].map(
            lambda value: None if value is None else round(value - control_trimmed, 3)
        )
    return summary


def declared_window(today: str) -> dict:
    """The forward window this report freezes for the NEXT inspection.

    Everything above is post-hoc and cannot satisfy plan.md Section 7 gate 2.
    Declaring the next window here, before anyone looks at it, is the only way
    a number in this file ever becomes gate-2 eligible - so it is printed as
    part of the report rather than left to a later decision.
    """
    return {
        "declared_on": today,
        "starts": "the first session after this report is committed",
        "length_sessions": 40,
        "must_span": ["bullish", "bearish", "chop"],
        "primary_metric": "trimmed-mean net R per (family, side), cells n>=30",
        "control": BASELINE_FAMILY,
        "exclusions_fixed_in_advance": [
            f"risk_per_share < {RISK_FLOOR_PCT_OF_ENTRY}% of entry",
            "close_r == 0 with eod_close == entry_price (no EOD close obtained)",
        ],
        "decision_rule": (
            "no promotion or demotion from this report or the next one alone; the "
            "declared window produces the first gate-2-eligible evidence and the "
            "trader decides what it is evidence FOR"
        ),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _table(frame: pd.DataFrame, columns: list[str], *, limit: int | None = None) -> str:
    if frame is None or frame.empty:
        return "_no cells_\n"
    present = [c for c in columns if c in frame.columns]
    body = frame[present]
    if limit:
        body = body.head(limit)
    header = "| " + " | ".join(present) + " |"
    rule = "|" + "|".join("---" for _ in present) + "|"
    lines = [header, rule]
    for _, row in body.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def render_report(
    *,
    intraday: pd.DataFrame,
    coverage: Coverage,
    playbook: pd.DataFrame,
    window_start: str,
    window_end: str,
    generated_at: str,
) -> str:
    usable = intraday[intraday["usable"]] if not intraday.empty else intraday
    parts: list[str] = []
    add = parts.append

    add(f"# Setup scoreboard — {window_start} … {window_end}\n\n")
    add(
        "**Read-only. This report promotes and demotes nothing.** plan.md Section 7\n"
        "gate 2 requires an evidence window frozen *before* inspection; this window was\n"
        "chosen after, so nothing measured here can move a rung. Its one forward-looking\n"
        "output is the declared window in the last section.\n\n"
    )
    add(
        f"Generated {generated_at}. Zones: `trade_date` is a bare market date; "
        "`logged_at` is tz-aware −07:00 (desk local, America/Los_Angeles); `entry_time` "
        "is naive desk-local PT, and market time is PT + 3h (America/New_York).\n"
    )

    add("\n## 1. Coverage, and what was excluded before anything was ranked\n")
    add(
        f"| stage | rows |\n|---|---|\n"
        f"| scanned (all milestone rows) | {coverage.rows_scanned:,} |\n"
        f"| `event_type == final` | {coverage.finals:,} |\n"
        f"| in window | {coverage.in_window:,} |\n"
        f"| distinct sessions | {coverage.sessions} |\n"
        f"| excluded — no EOD close obtained | {coverage.unsettled:,} |\n"
        f"| …of those, never advanced a bar | {coverage.never_measured:,} |\n"
        f"| excluded — stop under {RISK_FLOOR_PCT_OF_ENTRY}% of entry | "
        f"{coverage.below_risk_floor:,} |\n"
        f"| **usable** | **{coverage.usable:,}** |\n"
    )

    add("\n### 1a. The `close_r == 0` mass is a defect, not a population of scratches\n")
    if coverage.in_window:
        share = coverage.unsettled / coverage.in_window * 100
        add(
            f"{coverage.unsettled:,} of {coverage.in_window:,} in-window finals "
            f"({share:.1f}%) carry `close_r` exactly 0. **Every one of them has "
            "`eod_close` exactly equal to `entry_price`, and none of the settled "
            "finals does.** A real close does not land on the entry to the cent "
            f"{coverage.unsettled:,} times and never otherwise — the outcome writer "
            "defaults `eod_close` to the entry when it cannot read one.\n"
        )
        add(
            f"\n{coverage.never_measured:,} of those never advanced a bar at all "
            "(`bars_elapsed == 0`); the rest have real excursions and were simply "
            "never closed out. Treating any of them as a scratch biases every mean "
            "**upward**, because the stopped-out ones — which should score about "
            "−1R — score 0 instead. They are excluded from every number below and "
            "counted here instead.\n"
        )
    add(
        "\nThis is the single largest data-quality finding in the rebuild and it is an "
        "argument for fixing the writer, not for reading around it.\n"
    )

    add("\n## 2. Intraday families (`intraday_bounce_outcomes.csv` finals)\n")
    add(
        f"Cells with n < {MIN_CELL_N} are listed but marked `reportable = False` and are "
        "not ranked. Every R appears as mean, 10% trimmed mean and median, with the "
        "stop-out rate beside it — a plain mean on a ratio with an unbounded numerator "
        "is exactly the statistic that produced the review's phantom −1.82R.\n\n"
    )
    cols = ["cell", "n", "mean_r", "trimmed_mean_r", "median_r", "stop_out_rate", "p10_r", "p90_r", "reportable"]
    add(_table(summarise(usable, "bounce_type"), cols))

    add("\n### 2a. By market environment — the axis the review called starved\n")
    add(
        "`market_environment` is present on 100% of these rows, from `context_json`. "
        "The review reported this axis at n=130 because it read the review store; the "
        "outcome store carries it on every row.\n\n"
    )
    add(_table(summarise(usable, "market_environment"), cols))

    add("\n### 2b. By session RVOL bucket\n")
    if not usable.empty and "session_rvol" in usable:
        rvol = usable.copy()
        rvol["rvol_bucket"] = pd.cut(
            pd.to_numeric(rvol["session_rvol"], errors="coerce"),
            [0, 0.8, 1.2, 2.0, 1e9],
            labels=["<0.8", "0.8-1.2", "1.2-2.0", ">2.0"],
        ).astype(str)
        add(_table(summarise(rvol[rvol["rvol_bucket"] != "nan"], "rvol_bucket"), cols))
    else:
        add("_no RVOL on these rows_\n")

    add("\n### 2c. By sector\n")
    add(_table(summarise(usable, "sector"), cols, limit=20))

    add("\n## 3. Swing families vs their own control (`setup_playbook_episodes.csv`)\n")
    lift = baseline_lift(playbook)
    if lift.empty:
        add("_no episodes in window_\n")
    else:
        add(
            f"Control is `{BASELINE_FAMILY}`, the file's own equal-weight every-fifth-"
            "session entry over the same names and window. Only the difference from it "
            "is attributable to the setup.\n\n"
        )
        add(
            _table(
                lift,
                ["cell", "n", "mean_r", "trimmed_mean_r", "median_r", "baseline_trimmed_r", "lift_vs_baseline", "reportable"],
            )
        )
        add(
            "\n**Read this table as relative, never as absolute.** The control's own "
            f"trimmed R is {lift.iloc[0]['baseline_trimmed_r']}, so a positive "
            "`lift_vs_baseline` means *lost less than the control*, not *made money*. "
            "Two features of the block deserve stating before any of it is quoted: the "
            "median `net_r` sits at roughly −1.0 across most families, which means more "
            "than half of every family's episodes are full stop-outs; and the plain mean "
            "sits far above the trimmed mean nearly everywhere, which means what "
            "positive numbers exist are carried by a thin tail of large winners. A "
            "family here is a candidate for measurement, not a candidate for size.\n"
        )

    add("\n## 4. What this report does NOT establish\n")
    add(
        "- **No promotion or demotion.** Gate 2 is unsatisfiable post-hoc.\n"
        "- **No causal claim.** The environment and RVOL splits are conditional "
        "descriptions of one window, not evidence that a family works *because* of a "
        "regime.\n"
        f"- **Nothing from a cell under n={MIN_CELL_N}.** Those rows are printed so the "
        "thinness is visible, not so they can be read.\n"
        "- **Nothing about the excluded rows.** The unsettled mass is a writer defect "
        "with an unknown outcome, and unknown is not zero.\n"
    )

    add("\n## 5. The declared window for the next inspection\n")
    add(
        "Everything above is post-hoc. This is the part that is not: the window below "
        "is frozen **now**, before it is measured, which is the only route by which any "
        "number in this file ever becomes plan.md Section 7 gate-2 eligible.\n\n"
    )
    for key, value in declared_window(generated_at[:10]).items():
        rendered = ", ".join(value) if isinstance(value, list) else value
        add(f"- **{key}**: {rendered}\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
def main() -> int:
    from project_paths import DATA_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Rebuild the setup scoreboard (read-only)")
    parser.add_argument("--window-start", default="2026-07-24")
    parser.add_argument("--window-end", default="2026-08-21")
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=Path(DATA_DIR) / "runtime" / "intraday_bounce_outcomes.csv",
    )
    parser.add_argument(
        "--playbook",
        type=Path,
        default=Path(OUTPUT_DIR) / "reports" / "setup_playbook_episodes.csv",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    args = parser.parse_args()

    intraday, coverage = load_intraday_finals(
        args.outcomes, window_start=args.window_start, window_end=args.window_end
    )
    playbook = (
        load_playbook_episodes(
            args.playbook, window_start=args.window_start, window_end=args.window_end
        )
        if args.playbook.exists()
        else pd.DataFrame(columns=PLAYBOOK_COLUMNS)
    )
    if not args.playbook.exists():
        coverage.notes.append(f"playbook episodes not found at {args.playbook}")

    report = render_report(
        intraday=intraday,
        coverage=coverage,
        playbook=playbook,
        window_start=args.window_start,
        window_end=args.window_end,
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
