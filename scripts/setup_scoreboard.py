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
import os
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


_CLAIM_ENTRY = "entry_claim"


def _claim_kind(family: object) -> str:
    """What this family claims, via the one registry (R10.B).

    Read through `outcome_semantics` rather than restated, so the scoreboard
    and the health tile can never disagree about which rows are trades.
    """
    try:
        import outcome_semantics

        return outcome_semantics.claim_kind(str(family or ""))
    except Exception:  # pragma: no cover - the module ships beside this one
        return "unconfigured"


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
    """True where no settled ``close_r`` was ever obtained.

    Measured 2026-08-22 over the 2026-07-24..08-21 window: every one of the
    1,164 finals with ``close_r == 0`` has ``eod_close`` exactly equal to
    ``entry_price``, and **none** of the 5,743 finals with a non-zero
    ``close_r`` does. Real closes do not land on the entry to the cent 1,164
    times and never otherwise - the writer defaults ``eod_close`` to the entry
    when it cannot read one.

    New final rows preserve that uncertainty directly with a blank ``close_r``
    and ``eod_close``. Older rows used the zero/entry sentinel described above;
    both encodings are unresolved and must be excluded.

    These are not scratch trades, and treating them as such biases every mean
    upward: 563 of them have ``stop_hit`` true with a median ``mae_r`` of
    -1.000, i.e. trades that were stopped out and should score about -1R are
    scoring 0.
    """
    close_r = pd.to_numeric(frame.get("close_r"), errors="coerce")
    eod = pd.to_numeric(frame.get("eod_close"), errors="coerce")
    entry = pd.to_numeric(frame.get("entry_price"), errors="coerce")
    missing_close = close_r.isna() | close_r.isin([float("inf"), float("-inf")])
    legacy_sentinel = (close_r == 0) & (eod == entry)
    return missing_close | legacy_sentinel


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
    #: R10.C / R10.B. Rows whose family does not CLAIM an entry - an H1 colour
    #: mark on a closed bar, a regime-pause observation. They were being
    #: averaged as trades; they are excluded here and counted by kind, never
    #: silently dropped.
    not_entry_claim: int = 0
    by_claim_kind: dict = field(default_factory=dict)
    usable: int = 0
    usable_before_claim_split: int = 0
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

    # R10.C: what a row's family actually CLAIMS decides whether it may be
    # averaged as a trade at all (R10.B). This is applied AFTER the other two
    # exclusions so the before/after in section 1b is measured on the same
    # population the previous report ranked.
    frame["claim_kind"] = frame["bounce_type"].map(_claim_kind)
    entry_claim = frame["claim_kind"] == _CLAIM_ENTRY
    frame["not_entry_claim"] = ~entry_claim

    prior_usable = ~(unsettled | floor)
    coverage.usable_before_claim_split = int(prior_usable.sum())
    coverage.not_entry_claim = int((prior_usable & ~entry_claim).sum())
    coverage.by_claim_kind = {
        str(kind): int(count)
        for kind, count in frame.loc[prior_usable, "claim_kind"].value_counts().items()
    }

    frame["usable"] = prior_usable & entry_claim
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


def _summary_for(cell: "pd.DataFrame", values: "pd.Series") -> dict:
    """One cell through `evidence_stats`, carrying symbol and session identity.

    A mean over 200 rows that are one name on two sessions has a sample size of
    roughly one; only concentration and the session-block interval can say so,
    and both need the labels to travel with the values.
    """
    import evidence_stats

    index = values.index
    symbols = (
        cell.loc[index, "symbol"].astype(str).tolist() if "symbol" in cell.columns else []
    )
    sessions = (
        cell.loc[index, "trade_date"].astype(str).tolist()
        if "trade_date" in cell.columns
        else []
    )
    return evidence_stats.summarize(values.tolist(), symbols=symbols, sessions=sessions)


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
        _cell_frame = cell
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
        # R10.C: ground rule 10 lives in `evidence_stats`, once, and every
        # ground-rule-11 surface reads it from there - so the scoreboard, the
        # cohort CSVs and the review report cannot drift into three different
        # definitions of the same word.
        summary = _summary_for(cell, values)
        raw = summary["raw"]
        clipped = summary.get("clipped") or {}
        boot = summary.get("bootstrap") or {}
        by_symbol = (summary.get("concentration") or {}).get("by_symbol") or {}
        by_session = (summary.get("concentration") or {}).get("by_session") or {}
        rows.append(
            {
                # " / ", never "|": these names are printed straight into a markdown
                # table, and a pipe inside a cell silently splits the column.
                "cell": _cell_name(key),
                "n": int(len(values)),
                "symbols": summary["counts"]["symbols"],
                "sessions": summary["counts"]["sessions"],
                "mean_r": raw["mean"],
                "trimmed_mean_r": raw["trimmed_mean"],
                "median_r": raw["median"],
                "clipped_mean_r": clipped.get("mean"),
                "stop_out_rate": stop_rate,
                "p10_r": raw["p10"],
                "p90_r": raw["p90"],
                "ci_low": boot.get("low") if boot.get("measured") else None,
                "ci_high": boot.get("high") if boot.get("measured") else None,
                "top_symbol_share": by_symbol.get("top_share"),
                "top_session_share": by_session.get("top_share"),
                "evidence_label": summary["evidence_label"],
                "meets_n_floor": summary["meets_n_floor"],
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # `meets_n_floor` comes from `evidence_stats` and means exactly what it
    # says. The old `reportable` column meant the same arithmetic under a name
    # that claimed more than it measured - n >= 30 is NECESSARY, never
    # sufficient (ground rule 10), and a column called "reportable" invites a
    # reader to treat a cleared floor as permission.
    return out.sort_values(["meets_n_floor", "trimmed_mean_r"], ascending=[False, False])


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


def _claim_split_section(coverage: Coverage, before, after) -> str:
    """Section 1b: what the R10.B claim-kind split MOVED, side by side.

    The split removes rows the previous reports averaged - 147,713 annotation
    rows across the whole store - so figures the trader has already read will
    change. An unannounced move reads as a regression; an announced one reads
    as the fix working. So every family whose number moved is printed with its
    before, its after, the rows removed and the claim kind that removed them,
    rather than a list of names and an assurance.
    """
    import pandas as pd

    lines = ["\n### 1b. What the claim-kind split moved (R10.B), before and after\n"]
    lines.append(
        "Rows whose family does not CLAIM an entry - an H1 colour mark on a bar that "
        "had already closed, a regime-pause observation - were previously averaged as "
        "trades. They are excluded now. **Figures below that you have seen before will "
        "have moved, and this table is where they move in the open.**\n\n"
        "`was_n` and `rows_removed` both count rows carrying a MEASURABLE R, so the "
        "two columns subtract.\n\n"
    )
    if coverage.by_claim_kind:
        kinds = ", ".join(
            f"`{kind}` {count:,}" for kind, count in sorted(coverage.by_claim_kind.items())
        )
        lines.append(f"Settled, above-the-floor rows by claim kind: {kinds}.\n\n")
    if coverage.not_entry_claim == 0:
        lines.append(
            "No settled row in this window was excluded by the split, so nothing in "
            "this report moved because of it.\n"
        )
        return "".join(lines)

    if before is None or getattr(before, "empty", True):
        return "".join(lines)

    rows = []
    for family, cell in before.groupby("bounce_type", dropna=False):
        kept = cell[cell["usable"]] if "usable" in cell.columns else cell.iloc[0:0]
        was = pd.to_numeric(cell["close_r"], errors="coerce").dropna()
        now = pd.to_numeric(kept["close_r"], errors="coerce").dropna()
        if was.empty:
            continue
        # Counted on the SAME basis as `was_n` - rows carrying a measurable R.
        # Counting raw rows here instead produced `rows_removed` LARGER than
        # `was_n` on three families, which is arithmetic nobody can follow.
        removed = len(was) - len(now)
        if not removed:
            continue
        kinds = sorted({str(value) for value in cell["claim_kind"].tolist()})
        rows.append(
            {
                "family": str(family),
                "was_n": len(was),
                "was_mean_r": round(float(was.mean()), 3),
                "now_n": len(now),
                # A family removed ENTIRELY has no "after". That is the honest
                # answer and it is printed as one, never as a zero.
                "now_mean_r": round(float(now.mean()), 3) if len(now) else None,
                "rows_removed": removed,
                "claim_kind": " / ".join(kinds),
            }
        )
    if not rows:
        return "".join(lines)
    frame = pd.DataFrame(rows).sort_values("rows_removed", ascending=False)
    lines.append(
        _table(
            frame,
            ["family", "claim_kind", "was_n", "was_mean_r", "now_n", "now_mean_r", "rows_removed"],
            limit=25,
        )
    )
    lines.append(
        "\nA blank `now_mean_r` is a family that left this report entirely: every one "
        "of its rows was an annotation or an observation, so it never had a trade "
        "average to keep. That is not a family whose edge vanished - it is a family "
        "that never claimed one.\n"
    )
    return "".join(lines)


#: Where derived reports live (R10 trader decision: a runtime report store with
#: atomic last-good; `docs/analysis/` receives only deliberately frozen,
#: hand-committed audits).
REPORT_STORE_SUBDIR = "evidence_reports"
BUNDLE_SCHEMA = "setup_scoreboard_bundle_v1"


def report_store_dir():
    from project_paths import REPORTS_DIR

    return Path(REPORTS_DIR) / REPORT_STORE_SUBDIR


def publish_atomically(path: Path, content: str) -> Path:
    """Write `path` so a reader never sees a half-written report.

    Temp file beside the target, then replace. The previous report stays
    readable for the whole write and is only swapped at the end, so a failed
    publish costs the NEW report and never the last good one - the same rule
    the away report already keeps.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, target)
    return target


def build_bundle(
    *,
    intraday,
    coverage: Coverage,
    playbook,
    window_start: str,
    window_end: str,
    generated_at: str,
    ledger_rows: int | None = None,
) -> dict:
    """The machine-readable half of the report (R10.C).

    The Markdown is for a person; this is for the `evidence_report` slot, the
    opt-in `setup_performance` scope, and anything else that needs the numbers
    without re-parsing prose. Both come from the SAME computation, so they
    cannot disagree.
    """
    usable = intraday[intraday["usable"]] if not intraday.empty else intraday
    families = summarise(usable, "bounce_type") if not usable.empty else pd.DataFrame()
    sides = summarise(usable, ["bounce_type", "direction"]) if not usable.empty else pd.DataFrame()
    return {
        "schema": BUNDLE_SCHEMA,
        "generated_at": generated_at,
        "window": {"start": window_start, "end": window_end},
        "coverage": {
            "rows_scanned": coverage.rows_scanned,
            "finals": coverage.finals,
            "in_window": coverage.in_window,
            "sessions": coverage.sessions,
            "unsettled": coverage.unsettled,
            "never_measured": coverage.never_measured,
            "below_risk_floor": coverage.below_risk_floor,
            "usable_before_claim_split": coverage.usable_before_claim_split,
            "not_entry_claim": coverage.not_entry_claim,
            "by_claim_kind": dict(coverage.by_claim_kind or {}),
            "usable": coverage.usable,
            "notes": list(coverage.notes),
        },
        "ledger_rows_read": ledger_rows,
        "families": families.to_dict("records") if not families.empty else [],
        "family_by_side": sides.to_dict("records") if not sides.empty else [],
        "exit_policies": _exit_policy_rows(usable),
        "declared_window": declared_window(generated_at[:10]),
        "declared_window_note": (
            "reprinted from R9.3 §5 unchanged; NOT measured here and not "
            "re-declared - the sessions it names have not elapsed"
        ),
        "statistics_contract": {
            "module": "evidence_stats",
            "schema": evidence_stats_schema(),
            "n_floor": 30,
            "n_floor_note": "necessary, never sufficient",
            "clip": 4.0,
        },
    }


def evidence_stats_schema() -> str:
    import evidence_stats

    return evidence_stats.SUMMARY_SCHEMA


def _exit_policy_rows(usable) -> list[dict]:
    """Each frozen exit policy on its own, per family (R10.B path payloads).

    Reported side by side and never blended. A row whose path was not captured
    contributes to `paths_missing` rather than to a policy - a policy average
    over the trades that happened to carry a path is a different statistic
    wearing the same name.
    """
    if usable is None or getattr(usable, "empty", True):
        return []
    if "context_json" not in usable.columns:
        return []
    import evidence_stats

    per_family: dict = {}
    missing: dict = {}
    for _index, row in usable.iterrows():
        family = str(row.get("bounce_type") or "")
        context = _safe_json(row.get("context_json"))
        path = (context or {}).get("path") or {}
        policies = path.get("exit_policies") or {}
        if not policies:
            missing[family] = missing.get(family, 0) + 1
            continue
        for name, value in policies.items():
            if value.get("r") is None:
                continue
            per_family.setdefault(family, {}).setdefault(name, []).append(float(value["r"]))
    rows = []
    for family in sorted(set(per_family) | set(missing)):
        entry = {"family": family, "paths_missing": missing.get(family, 0)}
        for name, values in sorted(per_family.get(family, {}).items()):
            summary = evidence_stats.summarize(values, clip=None)
            entry[name] = {
                "n": summary["n"],
                "mean_r": summary["raw"]["mean"],
                "trimmed_mean_r": summary["raw"]["trimmed_mean"],
                "median_r": summary["raw"]["median"],
            }
        rows.append(entry)
    return rows


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
    # The population earlier reports ranked, kept so section 1b can show what
    # moved rather than merely asserting that something did.
    before = (
        intraday[~(intraday["unsettled_close"] | intraday["below_risk_floor"])]
        if not intraday.empty
        else intraday
    )
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
        f"| settled, above the floor (what earlier reports ranked) | "
        f"{coverage.usable_before_claim_split:,} |\n"
        f"| excluded — family does not CLAIM an entry (R10.B) | "
        f"{coverage.not_entry_claim:,} |\n"
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

    add(_claim_split_section(coverage, before, usable))

    add("\n## 2. Intraday families (`intraday_bounce_outcomes.csv` finals)\n")
    add(
        f"Cells with n < {MIN_CELL_N} are listed but marked `meets_n_floor = False` and are "
        "not ranked. Every R appears as mean, 10% trimmed mean and median, with the "
        "stop-out rate beside it — a plain mean on a ratio with an unbounded numerator "
        "is exactly the statistic that produced the review's phantom −1.82R.\n\n"
    )
    cols = [
        "cell", "n", "symbols", "sessions", "mean_r", "trimmed_mean_r", "median_r",
        "clipped_mean_r", "stop_out_rate", "p10_r", "p90_r", "ci_low", "ci_high",
        "top_symbol_share", "evidence_label", "meets_n_floor",
    ]
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
                ["cell", "n", "mean_r", "trimmed_mean_r", "median_r", "baseline_trimmed_r", "lift_vs_baseline", "meets_n_floor"],
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
        "**R10.C did not alter, re-declare, or measure this window early.** It is\n"
        "reprinted below exactly as R9.3 §5 declared it, and nothing in this report\n"
        "measures it: the 40 sessions it names have not elapsed, and a number taken\n"
        "from a window before it closes is not the evidence the window exists to\n"
        "produce.\n\n"
    )
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
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="R10.A evidence ledger directory to count alongside the CSV",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="also copy a dated audit into docs/analysis/ (a deliberate, hand-committed freeze)",
    )
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

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    # R10.C: the append-only ledger beside the CSV. Counted, not merged - the
    # CSV remains the authority during R10.A's canary, and a report that
    # silently preferred one source over the other would make the canary
    # unreadable.
    ledger_rows = None
    if args.ledger is not None:
        try:
            from evidence_ledger import intraday_outcome_ledger

            result = intraday_outcome_ledger(args.ledger).read()
            ledger_rows = len(result.rows)
            coverage.notes.append(
                f"evidence ledger: {result.coverage_note} (counted beside the CSV, "
                "not merged into it)"
            )
        except Exception as exc:  # noqa: BLE001
            coverage.notes.append(f"evidence ledger unreadable: {exc}")

    report = render_report(
        intraday=intraday,
        coverage=coverage,
        playbook=playbook,
        window_start=args.window_start,
        window_end=args.window_end,
        generated_at=generated_at,
    )
    bundle = build_bundle(
        intraday=intraday,
        coverage=coverage,
        playbook=playbook,
        window_start=args.window_start,
        window_end=args.window_end,
        generated_at=generated_at,
        ledger_rows=ledger_rows,
    )

    if args.out:
        publish_atomically(args.out, report)
        print(f"wrote {args.out}")
    else:
        # The runtime report store, with atomic last-good: a failed publish
        # costs the new report, never the previous one.
        store = report_store_dir()
        published = publish_atomically(store / "setup_scoreboard.md", report)
        publish_atomically(
            store / "setup_scoreboard.json",
            json.dumps(bundle, indent=1, sort_keys=True, default=str) + "\n",
        )
        print(f"wrote {published} and its bundle")
    if args.freeze:
        # `docs/analysis/` receives only deliberately frozen, hand-committed
        # audits (R10 trader decision), so this is opt-in and dated.
        from project_paths import ROOT_DIR

        stamp = generated_at[:10]
        frozen = Path(ROOT_DIR) / "docs" / "analysis" / f"SETUP_SCOREBOARD_{stamp}.md"
        publish_atomically(frozen, report)
        print(f"froze {frozen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
