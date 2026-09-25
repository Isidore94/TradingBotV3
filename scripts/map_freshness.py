"""How old the sector and industry ETF maps are (WISHLIST P2-8 8c).

Each map gets a SIDECAR stamp, `<map stem>.reviewed.json` beside it, holding
`{"reviewed_on": "YYYY-MM-DD"}`. The maps themselves are never rewritten, so
every reader of them is unchanged. No sidecar reads as "age unknown".

Health shows one row per map: healthy inside 90 days, degraded past it,
unknown with no stamp. Stamping is a trader action through the CLI, dry by
default::

    python -m map_freshness stamp sector
    python -m map_freshness stamp industry --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import project_paths

#: Past this many days a map is stale and Health warns.
MAX_AGE_DAYS = 90

#: Map name -> the `project_paths` constant that holds its path (read at call time).
MAPS = {"sector": "SECTOR_ETF_MAP_FILE", "industry": "INDUSTRY_ETF_MAP_FILE"}

_LABELS = {"sector": "Sector ETF map", "industry": "Industry ETF map"}


def map_path(name: str) -> Path:
    return Path(getattr(project_paths, MAPS[name]))


def stamp_path(name: str) -> Path:
    target = map_path(name)
    return target.with_name(f"{target.stem}.reviewed.json")


def reviewed_on(name: str) -> date | None:
    """The stamped review date, or None (no sidecar or an unreadable one)."""
    try:
        payload = json.loads(stamp_path(name).read_text(encoding="utf-8"))
        return date.fromisoformat(str(payload.get("reviewed_on") or "")[:10])
    except (OSError, ValueError, AttributeError):
        return None


def freshness(name: str, today: date) -> dict[str, Any]:
    """`{name, reviewed_on, age_days, status}`; status healthy / degraded / unknown."""
    stamped = reviewed_on(name)
    if stamped is None:
        return {"name": name, "reviewed_on": "", "age_days": None, "status": "unknown"}
    age = (today - stamped).days
    return {
        "name": name,
        "reviewed_on": stamped.isoformat(),
        "age_days": age,
        "status": "degraded" if age > MAX_AGE_DAYS else "healthy",
    }


def health_checks(now: datetime | None = None) -> list[dict[str, Any]]:
    """One Health row per map. Called on the Health audit worker."""
    today = (now or datetime.now()).date()
    rows = []
    for name in MAPS:
        state = freshness(name, today)
        if state["status"] == "unknown":
            summary = f"age unknown - no review stamp ({stamp_path(name).name})"
        elif state["status"] == "degraded":
            summary = f"{state['age_days']} days old (reviewed {state['reviewed_on']}) - past {MAX_AGE_DAYS} days, review it"
        else:
            summary = f"{state['age_days']} days old (reviewed {state['reviewed_on']})"
        rows.append({
            "id": f"map_freshness_{name}",
            "label": _LABELS[name],
            "status": state["status"],
            "summary": summary,
            "updated_at": state["reviewed_on"],
            "source": "map_freshness.py",
            "details": {"map": str(map_path(name)), "stamp": str(stamp_path(name))},
        })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """`stamp <map>`, DRY BY DEFAULT. Writes only the sidecar, never the map."""
    parser = argparse.ArgumentParser(prog="map_freshness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stamp = sub.add_parser("stamp", help="record that a map was reviewed")
    stamp.add_argument("map", choices=sorted(MAPS))
    stamp.add_argument("--date", default="", help="review date (default: today)")
    stamp.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    day = date.fromisoformat(args.date) if args.date else date.today()
    target = stamp_path(args.map)
    print(f"map: {map_path(args.map)}")
    print(f"stamp: {target}")
    print(f"reviewed_on: {day.isoformat()}")
    if not args.apply:
        print("dry run - re-run with --apply to write the stamp")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"reviewed_on": day.isoformat()}) + "\n", encoding="utf-8")
    print("written")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main(sys.argv[1:]))
