"""Real-shaped fixtures for packet D1C-B (the claimed picks graded beside FAV/HC).

Not a test module - no `test_` prefix, so pytest does not collect it. Three
CSV headers and one JSONL row shape, copied in SHAPE from the live stores on
2026-09-14 (read-only `head` of `C:\\TradingBotData\\data\\runtime\\`):

* ``like_cohort_picks.csv``  - `ui/annotations/like_cohort.PICK_COLUMNS`
* ``like_cohort_outcomes.csv`` - `human_focus_tracking`'s outcome columns
* ``master_avwap_tier_outcomes.csv`` - the tracker's scan-row outcomes
* ``claimed_picks.jsonl`` - `scripts/claimed_picks.ROW_FIELDS`

The fixture is built so its TRUE ANSWERS are specific numbers, and the tests
assert those numbers rather than the shape:

    window="all"     liked n=4 wins=3 (0.75)   FAV n=5 wins=4 (0.80)
    window="lately"  liked n=3 wins=2 (2/3)    FAV n=4 wins=3 (0.75)
    pending=1  unmeasured=1  dropped_duplicates=1  quick likes excluded=1
    also_fav=1  also_near=1   HC rows: 0

An old row is modelled the way an old row really is - the key PRESENT and
EMPTY (`h5_return` blank with `matured_horizons` "1,3"), never absent - and
every number is read back through `csv.DictReader`, which turns a blank cell
into `""` rather than `None`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# The live headers, verbatim.
# ---------------------------------------------------------------------------

LIKE_PICK_COLUMNS = (
    "trade_date",
    "symbol",
    "side",
    "source",
    "snapshotted_at",
    "active_at_snapshot",
    "claimed_at_utc",
    "session_date",
    "like_mode",
    "surface",
)

LIKE_OUTCOME_COLUMNS = (
    "trade_date",
    "symbol",
    "side",
    "source",
    "entry_date",
    "entry_close",
    "h1_date",
    "h1_return",
    "h3_date",
    "h3_return",
    "h5_date",
    "h5_return",
    "h10_date",
    "h10_return",
    "matured_horizons",
    "fully_matured",
    "updated_at",
)

TIER_OUTCOME_COLUMNS = (
    "observation_id",
    "scan_row_id",
    "run_id",
    "run_timestamp",
    "run_date",
    "watchlist_label",
    "scan_date",
    "future_scan_date",
    "horizon_sessions",
    "tier",
    "tier_source",
    "symbol",
    "side",
    "priority_bucket",
    "priority_score",
    "setup_family",
    "favorite_zone",
    "entry_close",
    "future_close",
    "raw_return_pct",
    "side_return_pct",
    "win",
    "spy_forward_return_pct",
    "spy_relative_side_return_pct",
    "sessions_spanned",
    "stale_horizon",
    "positive_scan_factor_match_count",
    "positive_scan_factor_matches",
    "outcome_kind",
)

#: The `as_of` every reader test pins. `evidence_stats.lately_window` makes the
#: lately window 2026-08-14 .. 2026-09-11 (20 exchange sessions, measured).
AS_OF = "2026-09-11"
LATELY_FIRST = "2026-08-14"

#: Well outside any lately window this suite will ever compute - the row that
#: separates `all` from `lately`.
LONG_AGO = "2024-01-03"


def like_pick(
    trade_date: str,
    symbol: str,
    side: str,
    source: str,
    *,
    like_mode: str = "",
    surface: str = "chart_review",
) -> dict[str, str]:
    """One `like_cohort_picks.csv` row. A blank `like_mode` reads CLAIMED (P9)."""
    return {
        "trade_date": trade_date,
        "symbol": symbol,
        "side": side,
        "source": source,
        "snapshotted_at": f"{trade_date}T16:48:57-07:00",
        "active_at_snapshot": "1",
        "claimed_at_utc": f"{trade_date}T23:48:57+00:00",
        "session_date": trade_date,
        "like_mode": like_mode,
        "surface": surface,
    }


def like_outcome(
    trade_date: str,
    symbol: str,
    side: str,
    source: str,
    *,
    h5_return: str,
    matured_horizons: str,
    entry_close: str = "24.5800",
) -> dict[str, str]:
    """One `like_cohort_outcomes.csv` row.

    `h5_return` blank with `matured_horizons` "1,3" is the real shape of a row
    whose fifth session has not closed: the key is PRESENT and EMPTY.
    """
    matured = {part.strip() for part in matured_horizons.split(",") if part.strip()}
    return {
        "trade_date": trade_date,
        "symbol": symbol,
        "side": side,
        "source": source,
        "entry_date": trade_date,
        "entry_close": entry_close,
        "h1_date": "2026-08-21" if "1" in matured else "",
        "h1_return": "0.014737" if "1" in matured else "",
        "h3_date": "2026-08-25" if "3" in matured else "",
        "h3_return": "0.011930" if "3" in matured else "",
        "h5_date": "2026-08-27" if "5" in matured else "",
        "h5_return": h5_return,
        "h10_date": "2026-09-03" if "10" in matured else "",
        "h10_return": "0.035439" if "10" in matured else "",
        "matured_horizons": matured_horizons,
        "fully_matured": "1" if "10" in matured else "0",
        "updated_at": "2026-09-04T06:38:36-04:00",
    }


def tier_row(
    scan_date: str,
    symbol: str,
    side: str,
    bucket: str,
    *,
    win: str,
    horizon_sessions: str = "5",
    stale_horizon: str = "False",
    setup_family: str = "avwap_breakout",
    observation_id: str | None = None,
) -> dict[str, str]:
    """One `master_avwap_tier_outcomes.csv` row, in the live column order."""
    row_id = f"{symbol}:{scan_date}:2026-05-04-130212"
    return {
        "observation_id": observation_id or f"{row_id}:{horizon_sessions}",
        "scan_row_id": row_id,
        "run_id": "2026-05-04-130212",
        "run_timestamp": f"{scan_date}T13:02:12",
        "run_date": scan_date,
        "watchlist_label": "home folder watchlists + Master AVWAP swing watchlists",
        "scan_date": scan_date,
        "future_scan_date": "2026-09-10",
        "horizon_sessions": horizon_sessions,
        "tier": "S",
        "tier_source": "derived_from_bucket",
        "symbol": symbol,
        "side": side,
        "priority_bucket": bucket,
        "priority_score": "65.0",
        "setup_family": setup_family,
        "favorite_zone": "",
        "entry_close": "128.06",
        "future_close": "134.89",
        "raw_return_pct": "5.33",
        "side_return_pct": "5.33",
        "win": win,
        "spy_forward_return_pct": "0.61",
        "spy_relative_side_return_pct": "4.72",
        "sessions_spanned": "5",
        "stale_horizon": stale_horizon,
        "positive_scan_factor_match_count": "0",
        "positive_scan_factor_matches": "",
        "outcome_kind": "favorable_direction_scanrow_v1",
    }


def claim_row(
    symbol: str,
    side: str,
    claimed_setup_id: str,
    session_date: str,
    *,
    action: str = "claim",
    known_at_claim: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One `claimed_picks.jsonl` row - `scripts.claimed_picks.ROW_FIELDS`."""
    return {
        "schema": "claimed_pick_v1",
        "action": action,
        "symbol": symbol,
        "side": side,
        "horizon": "d1",
        "claimed_setup_id": claimed_setup_id,
        "claim_at": f"{session_date}T09:41:00-07:00",
        "claim_at_utc": f"{session_date}T16:41:00+00:00",
        "session_date": session_date,
        "source": "chart_review",
        "annotation_ref": f"{symbol}-{session_date}-like",
        "known_at_claim": known_at_claim or {"close": 24.58, "rrs": 1.4},
        "note": "",
    }


# ---------------------------------------------------------------------------
# The canonical fixture. Every number below is deliberate.
# ---------------------------------------------------------------------------

BREAKOUT = "like_avwap_breakout"
ONE_STDEV = "like_avwape_to_1stdev"


def like_picks() -> list[dict[str, str]]:
    return [
        # measured, win
        like_pick("2026-09-02", "AAA", "LONG", BREAKOUT),
        # measured, loss
        like_pick("2026-09-02", "BBB", "LONG", BREAKOUT),
        # measured, win
        like_pick("2026-09-03", "CCC", "LONG", BREAKOUT),
        # h5 has not matured -> pending
        like_pick("2026-09-03", "DDD", "SHORT", ONE_STDEV),
        # no outcome row at all -> unmeasured "no outcome row"
        like_pick("2026-09-04", "EEE", "LONG", ONE_STDEV),
        # an UNCLAIMED like - never in "My liked trades", outcome row and all
        like_pick("2026-09-04", "FFF", "LONG", "like_unclaimed"),
        # a QUICK like (P9) - excluded, and counted once in the footnote
        like_pick("2026-09-04", "GGG", "LONG", BREAKOUT, like_mode="quick"),
        # the same click again: one pick, never a second trade
        like_pick("2026-09-02", "AAA", "LONG", BREAKOUT),
        # measured, win, but OUTSIDE the lately window
        like_pick(LONG_AGO, "OLD", "LONG", BREAKOUT),
    ]


def like_outcomes() -> list[dict[str, str]]:
    matured = "1,3,5,10"
    return [
        like_outcome("2026-09-02", "AAA", "LONG", BREAKOUT, h5_return="0.052631", matured_horizons=matured),
        like_outcome("2026-09-02", "BBB", "LONG", BREAKOUT, h5_return="-0.028885", matured_horizons=matured),
        like_outcome("2026-09-03", "CCC", "LONG", BREAKOUT, h5_return="0.012000", matured_horizons=matured),
        # PRESENT and EMPTY - the fifth session has not closed yet.
        like_outcome("2026-09-03", "DDD", "SHORT", ONE_STDEV, h5_return="", matured_horizons="1,3"),
        # EEE deliberately has NO outcome row.
        like_outcome("2026-09-04", "FFF", "LONG", "like_unclaimed", h5_return="0.090000", matured_horizons=matured),
        like_outcome("2026-09-04", "GGG", "LONG", BREAKOUT, h5_return="0.080000", matured_horizons=matured),
        like_outcome(LONG_AGO, "OLD", "LONG", BREAKOUT, h5_return="0.031000", matured_horizons=matured),
    ]


def tier_rows() -> list[dict[str, str]]:
    return [
        # FAV, inside the lately window: 3 wins of 4.
        # AAA on 2026-09-02 is ALSO a liked pick that day -> also_fav == 1.
        tier_row("2026-09-02", "AAA", "LONG", "favorite_setup", win="True"),
        tier_row("2026-08-20", "FV1", "LONG", "favorite_setup", win="True"),
        tier_row("2026-08-21", "FV2", "SHORT", "favorite_setup", win="True"),
        tier_row("2026-08-24", "FV3", "LONG", "favorite_setup", win="False"),
        # FAV, outside the lately window -> only the `all` group grades it.
        tier_row(LONG_AGO, "FV4", "LONG", "favorite_setup", win="True"),
        # Excluded by the ONE policy, both favorite_setup so no bucket has to
        # guess which population lost a row:
        tier_row("2026-08-25", "FV5", "LONG", "favorite_setup", win="True", stale_horizon="True"),
        tier_row("2026-08-26", "FV6", "LONG", "favorite_setup", win="True", horizon_sessions="1"),
        # Near: 1 win of 2. CCC on 2026-09-03 is also a liked pick -> also_near == 1.
        tier_row("2026-09-03", "CCC", "LONG", "near_favorite_zone", win="True"),
        tier_row("2026-08-27", "NR1", "LONG", "near_favorite_zone", win="False"),
    ]


def high_conviction_rows() -> list[dict[str, str]]:
    """A tracker that DOES stamp HC: 1 win of 2, inside the lately window."""
    return [
        tier_row("2026-08-28", "HC1", "LONG", "high_conviction", win="True"),
        tier_row("2026-08-31", "HC2", "SHORT", "high_conviction", win="False"),
    ]


def claims() -> list[dict[str, Any]]:
    """AAA was claimed through D1C-A; BBB is an older annotation-only like."""
    return [
        claim_row("AAA", "LONG", "avwap_breakout", "2026-09-02"),
    ]


# ---------------------------------------------------------------------------
# Writing them the way the desk stores them.
# ---------------------------------------------------------------------------


def write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def write_store(
    root: Path,
    *,
    picks=None,
    outcomes=None,
    tiers=None,
    claim_rows=None,
) -> dict[str, Path]:
    """The four files, at the four names, under `root`."""
    root = Path(root)
    paths = {
        "picks_path": write_csv(
            root / "like_cohort_picks.csv",
            LIKE_PICK_COLUMNS,
            like_picks() if picks is None else picks,
        ),
        "outcomes_path": write_csv(
            root / "like_cohort_outcomes.csv",
            LIKE_OUTCOME_COLUMNS,
            like_outcomes() if outcomes is None else outcomes,
        ),
        "tier_path": write_csv(
            root / "master_avwap_tier_outcomes.csv",
            TIER_OUTCOME_COLUMNS,
            tier_rows() if tiers is None else tiers,
        ),
        "claims_path": write_jsonl(
            root / "claimed_picks.jsonl",
            claims() if claim_rows is None else claim_rows,
        ),
    }
    return paths
