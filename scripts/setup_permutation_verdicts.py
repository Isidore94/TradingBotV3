"""Weak-variant and promotion-candidate verdicts over the permutation report history (P12).

    python scripts/setup_permutation_verdicts.py --history-dir <dir> --out <path>/permutation_verdicts.json

Reads the newest reports in `permutation_report_history/` (one per Saturday)
and, per population x horizon x family x side x facet key:

- `weak_variant`: the key FAILED hold-out in each of the newest two reports -
  it was shown both times (a rejected key in `top_rejected`), measured on at
  least HOLDOUT_MIN_N hold-out episodes, and its hold-out win rate was below
  the family's hold-out baseline both times.
- `promotion_candidate`: the key PASSED hold-out in each of the newest two
  reports (it is in `keys` both times). This is P1-4 4e's entry condition only;
  anything beyond the chip is ask-first.
- otherwise no verdict (absent from one report, mixed, or too few episodes).

Each verdict cites the two report dates and the two hold-out numbers.

Rank and annotate only: nothing reads a verdict for a detector, a score, an
alert, Focus, the queue or `review_policy.json`, and no row is ever hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SCHEMA = "setup_permutation_verdicts_v1"
WEAK = "weak_variant"
CANDIDATE = "promotion_candidate"
#: Short chip text per verdict, shown beside the setup key.
CHIPS = {WEAK: "weak variant", CANDIDATE: "candidate"}
#: Fewer hold-out episodes than this is not a result (same floor as the search).
HOLDOUT_MIN_N = 10
#: Reports read: the newest pair decides; older ones only lengthen the streak.
LOOKBACK = 8
NOTE = (
    "Rank and annotate only. A verdict never hides a row and never feeds a detector, "
    "a score, an alert, Focus, the queue or review_policy.json."
)

KeyId = tuple[str, str, str, str]  # population, horizon, family name, key label


def _rate(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def outcomes(report: Mapping[str, Any]) -> dict[KeyId, dict[str, Any]]:
    """Every key the report shows, with `passed` True/False, or None when not a result."""
    out: dict[KeyId, dict[str, Any]] = {}
    for population, block in sorted(((report or {}).get("populations") or {}).items()):
        for horizon, hz in sorted(((block or {}).get("horizons") or {}).items()):
            for name, family in sorted(((hz or {}).get("families") or {}).items()):
                baseline = _rate((family.get("holdout_baseline") or {}).get("win_rate"))
                base = {"family": str(family.get("family") or ""), "side": str(family.get("side") or "")}
                for key in family.get("keys") or ():
                    hold = key.get("holdout") or {}
                    out[(population, str(horizon), name, str(key.get("label") or ""))] = {
                        **base, "facets": dict(key.get("facets") or {}), "passed": True,
                        "holdout_win_rate": _rate(hold.get("win_rate")), "holdout_n": int(hold.get("n") or 0),
                        "holdout_baseline": baseline,
                    }
                for key in family.get("top_rejected") or ():
                    ident = (population, str(horizon), name, str(key.get("label") or ""))
                    if ident in out:
                        continue
                    hold = key.get("holdout") or {}
                    rate, n = _rate(hold.get("win_rate")), int(hold.get("n") or 0)
                    failed = (rate is not None and baseline is not None and n >= HOLDOUT_MIN_N
                              and rate < baseline)
                    out[ident] = {
                        **base, "facets": dict(key.get("facets") or {}), "passed": False if failed else None,
                        "holdout_win_rate": rate, "holdout_n": n, "holdout_baseline": baseline,
                    }
    return out


def _pct(value: float | None) -> str:
    return "?" if value is None else f"{value * 100:.0f}%"


def citation_text(verdict: Mapping[str, Any]) -> str:
    """One line: which two reports, and the two hold-out numbers."""
    word = "failed" if verdict.get("verdict") == WEAK else "passed"
    parts = [
        f"{cite['report_date']} {_pct(cite.get('holdout_win_rate'))} on n={cite.get('holdout_n', 0)} "
        f"vs {_pct(cite.get('holdout_baseline'))} baseline"
        for cite in verdict.get("citations") or ()
    ]
    streak = int(verdict.get("streak") or 0)
    tail = f" ({streak} reports in a row)" if streak > 2 else ""
    return (
        f"{verdict.get('family_key', '')} {verdict.get('label', '')} (h{verdict.get('horizon', '')}): "
        f"{word} hold-out " + " and ".join(parts) + tail
    )


def build_verdicts(dated_reports: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Pure: `[(report_date, report), ...]` in any order -> the verdicts payload."""
    ordered = sorted(dated_reports, key=lambda item: item[0])[-LOOKBACK:]
    read = [(day, outcomes(report)) for day, report in ordered]
    verdicts: list[dict[str, Any]] = []
    if len(read) >= 2:
        (old_day, older), (new_day, newest) = read[-2], read[-1]
        for ident in sorted(newest):
            now, before = newest[ident], older.get(ident)
            if before is None or now["passed"] is None or before["passed"] is not now["passed"]:
                continue
            streak = 0
            for _day, table in reversed(read):
                entry = table.get(ident)
                if entry is None or entry["passed"] is not now["passed"]:
                    break
                streak += 1
            population, horizon, name, label = ident
            verdict = {
                "population": population, "horizon": horizon, "family_key": name,
                "family": now["family"], "side": now["side"], "label": label, "facets": now["facets"],
                "verdict": CANDIDATE if now["passed"] else WEAK, "streak": streak,
                "citations": [
                    {"report_date": day, "holdout_win_rate": entry["holdout_win_rate"],
                     "holdout_n": entry["holdout_n"], "holdout_baseline": entry["holdout_baseline"]}
                    for day, entry in ((old_day, before), (new_day, now))
                ],
            }
            verdict["citation"] = citation_text(verdict)
            verdicts.append(verdict)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reports_read": [day for day, _report in ordered],
        "note": NOTE,
        "verdicts": verdicts,
    }


def read_history(history_dir: Path, *, limit: int = LOOKBACK) -> list[tuple[str, dict[str, Any]]]:
    """The newest `limit` history reports as `(date, report)`; an unreadable file is skipped."""
    from setup_permutation_search import history_files

    out: list[tuple[str, dict[str, Any]]] = []
    for day, path in history_files(Path(history_dir))[-limit:]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            out.append((day.isoformat(), payload))
    return out


def publish(history_dir: Path, out: Path) -> dict[str, Any]:
    """Read the history, write the verdicts file atomically, return the payload."""
    from setup_permutation_search import write_report

    payload = build_verdicts(read_history(history_dir))
    write_report(payload, out)
    return payload


def index_by_verdict(payload: Mapping[str, Any] | None) -> dict[KeyId, dict[str, Any]]:
    """`{(population, horizon, family name, label): verdict}` for display joins."""
    out: dict[KeyId, dict[str, Any]] = {}
    for verdict in (payload or {}).get("verdicts") or ():
        if isinstance(verdict, Mapping):
            ident = (str(verdict.get("population") or ""), str(verdict.get("horizon") or ""),
                     str(verdict.get("family_key") or ""), str(verdict.get("label") or ""))
            out[ident] = dict(verdict)
    return out


def chips(verdicts: Iterable[Mapping[str, Any]]) -> list[str]:
    """Distinct chip texts, weak first, for the verdicts a row carries."""
    kinds = {str(v.get("verdict") or "") for v in verdicts or ()}
    return [CHIPS[kind] for kind in (WEAK, CANDIDATE) if kind in kinds]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = publish(args.history_dir, args.out)
    print(json.dumps({"out": str(args.out), "verdicts": len(payload["verdicts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
