#!/usr/bin/env python3
r"""The frozen before/after for the Setup Tracker's episode selection (ST4.5).

Packet ST4, 2026-09-06. `build_recent_tracker_setup_family_rows` picks which
observation of a thesis becomes the graded episode, and today's rule reads the
OUTCOME to make that pick (`closed_first_v1`: a closed rescan beats the earlier
open entry). `first_actionable_v2` fixes the pick before the outcome is known.
**This tool decides nothing.** It runs the same builder twice on the same COPY
of a tracker and writes the side-by-side numbers the trader's decision will be
made on. Nothing it reads or writes reaches a detector, a score, a rank, an
alert, a watchlist, Focus, the review queue or `review_policy.json`.

Usage, from `scripts/`::

    python tracker_selection_compare.py --tracker <COPY.json> --out <dir>

Safety, per the scratch-script rule (CLAUDE.md, incident 2026-09-05):

* it prints `project_paths.DATA_DIR` on every run;
* it REFUSES when that resolves under the live home folder, and it refuses a
  `--tracker` or `--out` under the live home folder or under the well-known
  `C:\TradingBotData`, so it can neither read the live tracker in place nor
  write beside it;
* both paths are explicit - there is no default that could resolve to a live
  store by accident;
* every output name carries a stamp and an existing file is NEVER overwritten;
  a second run in the same second writes a new `-2` sibling.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import project_paths  # noqa: E402
import swing_headline  # noqa: E402
from master_avwap_lib import legacy, selection_policy  # noqa: E402

#: The live home folder this CLI refuses, as `project_paths` resolves it now.
PROTECTED_DATA_ROOT = Path(str(project_paths.SHARED_HOME_DIR))

#: The desk's home folder by name, refused even when `TRADINGBOTV3_DATA_DIR`
#: has been pointed somewhere else - which is exactly the situation a scratch
#: run is in, and exactly when the old guard would have stopped guarding.
WELL_KNOWN_LIVE_HOME = Path("C:/TradingBotData")

PROTECTED_ROOTS = (PROTECTED_DATA_ROOT, WELL_KNOWN_LIVE_HOME)

README_TEXT = (
    "Packet ST4.5, 2026-09-06. This file is EVIDENCE, not a decision. "
    "`closed_first_v1` is what the desk runs today and it is unchanged and "
    "still the default everywhere; `first_actionable_v2` is a shadow "
    "challenger that fixes the graded episode before its outcome is known "
    "(earliest scan row per attempt, a new attempt only after the previous "
    "one's representative scenario closed, the representative exit template "
    "DECLARED as full_band2, and an open representative left pending instead "
    "of borrowing the mean of the alternate exit plans that did close). "
    "Nothing here promotes anything and no export switches to v2. "
    "Three scoring questions were decided on 2026-09-06 and are NOT reopened "
    "by this comparison: (a) swept-measured M5 day trades stay OUT of the "
    "eod-hold tier cells - the M5 side is a different file and is unchanged "
    "by ST4; (b) EXPIRED_UNMEASURED stays IN the champion's scoring "
    "population and is excluded from EXPORTS only, so these rows COUNT it and "
    "merely name it as `expired_unmeasured_in_population`; (c) a baseline "
    "scenario that is neither open nor closed is UNTRADEABLE and was already "
    "outside every count through the `tradeable` filter, named here as "
    "`untradeable`. A `*_in_population` token counts episodes deliberately "
    "KEPT and is never summed into n_excluded."
)

CSV_COLUMNS = (
    "side",
    "priority_bucket",
    "setup_family",
    "n_observations",
    "n_episodes_v1",
    "n_episodes_v2",
    "n_pending_v1",
    "n_pending_v2",
    "n_wins_v1",
    "n_wins_v2",
    "n_losses_v1",
    "n_losses_v2",
    "n_flats_v1",
    "n_flats_v2",
    "n_unmeasured_v1",
    "n_unmeasured_v2",
    "win_rate_unweighted_v1",
    "win_rate_unweighted_v2",
    "wilson_lower_v1",
    "wilson_lower_v2",
    "mean_r_v1",
    "mean_r_v2",
    "n_excluded_v1",
    "n_excluded_v2",
    "fully_excluded_groups_v1",
    "fully_excluded_groups_v2",
    "excluded_reasons_v1",
    "excluded_reasons_v2",
    "rank_v1",
    "rank_v2",
    "rank_move",
    "changed",
)


def _is_under(path: Path, root: Path) -> bool:
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
        root_resolved = Path(root).expanduser().resolve(strict=False)
    except OSError:
        return False
    return resolved == root_resolved or root_resolved in resolved.parents


def _refuse_live_path(label: str, path: Path) -> str | None:
    for root in PROTECTED_ROOTS:
        if _is_under(path, root):
            return f"{label} {path} is under the live store {root}; refusing."
    return None


def _cell_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("side") or ""),
        str(row.get("priority_bucket") or ""),
        str(row.get("setup_family") or "general"),
    )


def _int(row: dict, key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _cell_stats(row: dict | None) -> dict[str, object]:
    """The counts ST2 exports plus the ONE Wilson bound, per cell.

    `win_rate_unweighted` is `n_wins / (n_wins + n_losses)` - the counted rate,
    not the recency-weighted `win_rate_closed` the score reads, because a
    comparison of two populations has to compare the populations.
    """
    if row is None:
        return {
            "n_episodes": 0,
            "n_pending": 0,
            "fully_excluded_groups": 0,
            "n_wins": 0,
            "n_losses": 0,
            "n_flats": 0,
            "n_unmeasured": 0,
            "n_observations": 0,
            "n_excluded": 0,
            "excluded_reasons": "",
            "win_rate_unweighted": None,
            "wilson_lower": None,
            "mean_r": None,
        }
    wins = _int(row, "n_wins")
    losses = _int(row, "n_losses")
    graded = wins + losses
    return {
        "n_episodes": _int(row, "n_episodes"),
        "n_pending": _int(row, "n_pending"),
        "fully_excluded_groups": _int(row, "fully_excluded_groups"),
        "n_wins": wins,
        "n_losses": losses,
        "n_flats": _int(row, "n_flats"),
        "n_unmeasured": _int(row, "n_unmeasured"),
        "n_observations": _int(row, "n_observations"),
        "n_excluded": _int(row, "n_excluded"),
        "excluded_reasons": str(row.get("excluded_reasons") or ""),
        "win_rate_unweighted": (wins / graded) if graded else None,
        "wilson_lower": swing_headline.wilson_lower_bound(wins, graded) if graded else None,
        "mean_r": _float_or_none(row.get("representative_closed_r")),
    }


def _ranked(cells: dict[tuple[str, str, str], dict[str, object]]) -> dict[tuple[str, str, str], int]:
    """Rank by the Wilson lower bound, ungraded cells below every graded one."""
    ordered = sorted(
        cells.items(),
        key=lambda item: (
            0 if item[1].get("wilson_lower") is not None else 1,
            -(float(item[1].get("wilson_lower") or 0.0)),
            -int(item[1].get("n_episodes") or 0),
            item[0],
        ),
    )
    return {key: index + 1 for index, (key, _stats) in enumerate(ordered)}


def _stamped_paths(out_dir: Path) -> tuple[Path, Path, str]:
    base = datetime.now().strftime("%Y%m%dT%H%M%S")
    stamp = base
    suffix = 1
    while True:
        json_path = out_dir / f"selection_comparison_{stamp}.json"
        csv_path = out_dir / f"selection_comparison_{stamp}.csv"
        if not json_path.exists() and not csv_path.exists():
            return json_path, csv_path, stamp
        suffix += 1
        stamp = f"{base}-{suffix}"


def _build(setups: dict, *, policy: str, reference_day: date, lookback_days: int,
           as_of_session: str | None) -> list[dict]:
    return legacy.build_recent_tracker_setup_family_rows(
        setups,
        reference_date=reference_day,
        lookback_days=lookback_days,
        selection_policy=policy,
        as_of_session=as_of_session,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tracker_selection_compare",
        description="Frozen v1-vs-v2 episode-selection comparison (ST4.5). Decides nothing.",
    )
    parser.add_argument("--tracker", required=True, help="path to a COPY of a setup tracker JSON")
    parser.add_argument("--out", required=True, help="output directory, outside the live home")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=legacy.TRACKER_RECENT_FAMILY_LOOKBACK_DAYS,
        help="same lookback for both policies",
    )
    parser.add_argument(
        "--as-of",
        default="",
        help="replay cutoff; defaults to the copy's own data_session",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print(f"project_paths.DATA_DIR = {project_paths.DATA_DIR}")
    if _is_under(Path(str(project_paths.DATA_DIR)), WELL_KNOWN_LIVE_HOME):
        print(
            "REFUSING: project_paths.DATA_DIR resolves under the live home "
            f"{WELL_KNOWN_LIVE_HOME}. Set TRADINGBOTV3_DATA_DIR to a scratch "
            "directory before running this tool.",
            file=sys.stderr,
        )
        return 2

    tracker_path = Path(args.tracker).expanduser()
    out_dir = Path(args.out).expanduser()
    for label, path in (("--tracker", tracker_path), ("--out", out_dir)):
        refusal = _refuse_live_path(label, path)
        if refusal:
            print(f"REFUSING: {refusal}", file=sys.stderr)
            return 2

    if not tracker_path.is_file():
        print(f"REFUSING: --tracker {tracker_path} is not a file.", file=sys.stderr)
        return 2

    try:
        payload = json.loads(tracker_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - the message is the whole point
        print(f"REFUSING: could not read {tracker_path}: {exc}", file=sys.stderr)
        return 2
    setups = payload.get("setups") if isinstance(payload, dict) else None
    if not isinstance(setups, dict) or not setups:
        print(f"REFUSING: {tracker_path} carries no `setups` mapping.", file=sys.stderr)
        return 2

    # A SCORING SNAPSHOT IS NOT A TRACKER, and comparing on one produces a
    # confident lie. `master_avwap_tracker_scoring_snapshot.json` holds compact
    # projections: `_scoring_outcome_summary` and NO `scenarios`. v1 would
    # answer every setup straight out of that cache - a cache the `as_of`
    # cutoff never touched, so the replay would grade trades it could not see -
    # while v2 cannot evaluate a scenario-less record at all and zeroes. The
    # output would read "v2 is broken" when the input was simply the wrong file.
    compact_projections = sum(
        1
        for setup in setups.values()
        if isinstance(setup, dict)
        and "_scoring_outcome_summary" in setup
        and not isinstance(setup.get("scenarios"), dict)
    )
    if compact_projections:
        print(
            f"REFUSING: {tracker_path} holds {compact_projections} COMPACT scoring "
            "projections (a `_scoring_outcome_summary` and no `scenarios`) - that is "
            "master_avwap_tracker_scoring_snapshot.json, not a setup tracker, and a "
            "policy comparison on it is meaningless. Pass a COPY of "
            "master_avwap_setup_tracker.json or an extract of the SQLite mirror.",
            file=sys.stderr,
        )
        return 2

    data_session = str((payload.get("data_session") or "")).strip()
    as_of_session = str(args.as_of or "").strip() or data_session or None
    reference_day = None
    if as_of_session:
        try:
            reference_day = date.fromisoformat(as_of_session)
        except ValueError:
            reference_day = None
    if reference_day is None:
        reference_day = datetime.now().date()

    lookback_days = max(1, int(args.lookback_days))
    rows_v1 = _build(
        setups,
        policy=selection_policy.SELECTION_CLOSED_FIRST_V1,
        reference_day=reference_day,
        lookback_days=lookback_days,
        as_of_session=as_of_session,
    )
    rows_v2 = _build(
        setups,
        policy=selection_policy.SELECTION_FIRST_ACTIONABLE_V2,
        reference_day=reference_day,
        lookback_days=lookback_days,
        as_of_session=as_of_session,
    )

    by_key_v1 = {_cell_key(row): row for row in rows_v1}
    by_key_v2 = {_cell_key(row): row for row in rows_v2}
    keys = sorted(set(by_key_v1) | set(by_key_v2))
    stats_v1 = {key: _cell_stats(by_key_v1.get(key)) for key in keys}
    stats_v2 = {key: _cell_stats(by_key_v2.get(key)) for key in keys}
    rank_v1 = _ranked(stats_v1)
    rank_v2 = _ranked(stats_v2)

    cells: list[dict[str, object]] = []
    for key in keys:
        left = stats_v1[key]
        right = stats_v2[key]
        changed = any(
            left.get(field) != right.get(field)
            for field in ("n_episodes", "n_pending", "n_wins", "n_losses", "mean_r")
        )
        cells.append(
            {
                "side": key[0],
                "priority_bucket": key[1],
                "setup_family": key[2],
                "n_observations": max(
                    int(left.get("n_observations") or 0), int(right.get("n_observations") or 0)
                ),
                "v1": left,
                "v2": right,
                "rank_v1": rank_v1[key],
                "rank_v2": rank_v2[key],
                "rank_move": rank_v1[key] - rank_v2[key],
                "changed": bool(changed),
            }
        )

    def _totals(stats: dict) -> dict[str, object]:
        wins = sum(int(cell.get("n_wins") or 0) for cell in stats.values())
        losses = sum(int(cell.get("n_losses") or 0) for cell in stats.values())
        graded = wins + losses
        return {
            "cells": len(stats),
            "n_episodes": sum(int(cell.get("n_episodes") or 0) for cell in stats.values()),
            "n_pending": sum(int(cell.get("n_pending") or 0) for cell in stats.values()),
            "n_wins": wins,
            "n_losses": losses,
            "n_excluded": sum(int(cell.get("n_excluded") or 0) for cell in stats.values()),
            # Build-level, identical on every row: a (side, bucket, family)
            # whose every record was excluded produces no row at all, so this
            # is the only place it is visible.
            "fully_excluded_groups": max(
                (int(cell.get("fully_excluded_groups") or 0) for cell in stats.values()),
                default=0,
            ),
            "win_rate_unweighted": (wins / graded) if graded else None,
            "wilson_lower": swing_headline.wilson_lower_bound(wins, graded) if graded else None,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path, stamp = _stamped_paths(out_dir)
    report = {
        "README": README_TEXT,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stamp": stamp,
        "packet": "ST4.5",
        "tracker_copy": str(tracker_path),
        "tracker_data_session": data_session,
        "as_of_session": as_of_session or "",
        "reference_date": reference_day.isoformat(),
        "lookback_days": lookback_days,
        "setups_in_copy": len(setups),
        "policies": {
            "v1": selection_policy.SELECTION_CLOSED_FIRST_V1,
            "v2": selection_policy.SELECTION_FIRST_ACTIONABLE_V2,
            "default": selection_policy.DEFAULT_SELECTION_POLICY,
        },
        "reentry_rule_v2": selection_policy.REENTRY_RULE_V2,
        "wilson_z": swing_headline.WILSON_Z,
        "totals": {"v1": _totals(stats_v1), "v2": _totals(stats_v2)},
        "cells_changed": sum(1 for cell in cells if cell["changed"]),
        "rank_moves": sum(1 for cell in cells if cell["rank_move"]),
        "cells": cells,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for cell in cells:
            left = cell["v1"]
            right = cell["v2"]
            writer.writerow(
                [
                    cell["side"],
                    cell["priority_bucket"],
                    cell["setup_family"],
                    cell["n_observations"],
                    left["n_episodes"], right["n_episodes"],
                    left["n_pending"], right["n_pending"],
                    left["n_wins"], right["n_wins"],
                    left["n_losses"], right["n_losses"],
                    left["n_flats"], right["n_flats"],
                    left["n_unmeasured"], right["n_unmeasured"],
                    left["win_rate_unweighted"], right["win_rate_unweighted"],
                    left["wilson_lower"], right["wilson_lower"],
                    left["mean_r"], right["mean_r"],
                    left["n_excluded"], right["n_excluded"],
                    left["fully_excluded_groups"], right["fully_excluded_groups"],
                    left["excluded_reasons"], right["excluded_reasons"],
                    cell["rank_v1"], cell["rank_v2"], cell["rank_move"],
                    "True" if cell["changed"] else "False",
                ]
            )

    totals_v1 = report["totals"]["v1"]
    totals_v2 = report["totals"]["v2"]
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(
        f"cells {len(cells)}; changed {report['cells_changed']}; "
        f"rank moves {report['rank_moves']}"
    )
    print(
        "v1 episodes {n_episodes} pending {n_pending} wins {n_wins} losses {n_losses} "
        "excluded {n_excluded}".format(**totals_v1)
    )
    print(
        "v2 episodes {n_episodes} pending {n_pending} wins {n_wins} losses {n_losses} "
        "excluded {n_excluded}".format(**totals_v2)
    )
    print("This comparison decides nothing; the default policy is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
