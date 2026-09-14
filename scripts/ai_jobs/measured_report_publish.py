"""Publish the measured report - packet WS-RP item 2.

One deterministic slot at the END of the deterministic stage (decision 0018's
order: deterministic slots, then the digest's narration, then the model-gated
ones). It calls no model, reaches nothing, and writes two files beside the
packs:

* ``measured_report_<session>.json`` - the whole report, every cell with its
  population, window, clock, version, state and the paths its number came from;
* ``measured_report_<session>.md`` - the readable sibling, the same pattern
  `synthesis.py` and `setup_research.py` already use for their packs.

**A rerun on matured data writes a NEW version and never rewrites the old one.**
The report id is a sha1 over the cells, so "the same evidence" is answerable
without diffing: an identical id means nothing was written at all, and a
different one means `_v2` is published BESIDE `_v1` and the record of what was
believed on the night survives its correction (the packs' D6 rule).

**A failure here never fails the night.** The report is a convenience over
records that are already on disk; the slot returns a failed status with its
reason and writes nothing.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ai_jobs import ledger

_log = logging.getLogger(__name__)

#: Every published report for one session, `_v2` upward after the first.
FILENAME_STEM = "measured_report"

_VERSION_RE = re.compile(r"^measured_report_(\d{4}-\d{2}-\d{2})(?:_v(\d+))?$")


def report_paths(root: Path, session_date: str, version: int) -> tuple[Path, Path]:
    """`(json, md)` for one version. Version 1 carries no suffix."""
    stem = f"{FILENAME_STEM}_{session_date}" + ("" if version <= 1 else f"_v{version}")
    return Path(root) / f"{stem}.json", Path(root) / f"{stem}.md"


def published_versions(root: Path, session_date: str) -> list[tuple[int, Path]]:
    """Every published version for a session, oldest first."""
    out: list[tuple[int, Path]] = []
    for path in sorted(Path(root).glob(f"{FILENAME_STEM}_{session_date}*.json")):
        match = _VERSION_RE.match(path.stem)
        if not match or match.group(1) != str(session_date):
            continue
        out.append((int(match.group(2) or 1), path))
    return sorted(out)


def latest_published(root: Path, session_date: str = "") -> dict[str, Any]:
    """The newest published report, as its payload. `{}` when there is none.

    With no session named, the newest session that has one - which is what the
    Daily Recap's Review tab asks for when it opens.
    """
    root = Path(root)
    if not root.exists():
        return {}
    sessions: list[str] = []
    if session_date:
        sessions = [str(session_date)[:10]]
    else:
        seen: set[str] = set()
        for path in root.glob(f"{FILENAME_STEM}_*.json"):
            match = _VERSION_RE.match(path.stem)
            if match:
                seen.add(match.group(1))
        sessions = sorted(seen, reverse=True)
    for day in sessions:
        versions = published_versions(root, day)
        if not versions:
            continue
        _version, path = versions[-1]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # noqa: PERF203 - a bad file is skipped
            _log.info("Measured report: %s could not be read (%s).", path, exc)
            continue
        if isinstance(payload, dict):
            payload["_path"] = str(path)
            return payload
    return {}


def narration_for(root: Path, report_id: str) -> str:
    """The local-AI review of THIS report id, when one was written.

    This packet adds no model call. It reads what the narration stage leaves
    beside the report - a `<stem>.narration.json` carrying the same `report_id`
    - and answers "" when nothing has been written, so the page can say so.
    """
    root = Path(root)
    if not root.exists() or not report_id:
        return ""
    for path in sorted(root.glob(f"{FILENAME_STEM}_*.narration.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("report_id") or "") != str(report_id):
            continue
        return str(payload.get("narration") or payload.get("text") or "")
    return ""


def default_warehouse() -> Any | None:
    """The warehouse seam over the research lake, or None when it is disabled.

    Session-scoped and partition-narrowed, per the warehouse contract: the read
    goes through `ResearchStore.read_rows` with the year partition and an
    `entry_at` range, never by materialising a month and filtering a list.
    """

    class _LakeOutcomes:
        def __init__(self) -> None:
            self._paths: tuple[str, ...] = ()

        @property
        def source_paths(self) -> tuple[str, ...]:
            return self._paths

        def read_outcomes(self, session_date: str, *, now: datetime | None = None):
            from research_warehouse.store import ResearchStore

            store = ResearchStore.open()
            if store is None:
                raise OSError("research_store_dir is not configured")
            import market_calendar

            day = date.fromisoformat(str(session_date)[:10])
            start = datetime.combine(
                day, datetime.min.time(), tzinfo=market_calendar.MARKET_TZ
            )
            end = market_calendar.session_close(day)
            rows: list[dict[str, Any]] = []
            paths: list[str] = []
            for year in (day.year, day.year + 1):
                partition = f"year={year}"
                directory = store.partition_dir("outcome_path", partition)
                if not directory.exists():
                    continue
                paths.append(str(directory))
                rows.extend(store.read_rows(
                    "outcome_path",
                    partition,
                    interval_start_range=(start, end),
                    time_column="entry_at",
                ))
            self._paths = tuple(paths)
            # The lake is append-only and a recomputed outcome is a NEW row, so
            # the reader takes the LATEST per (occurrence, recipe, definition)
            # - a superseded interim reading must not coexist with its
            # replacement in anyone's arithmetic (`outcomes.latest_outcomes`).
            latest: dict[tuple[str, str, str], dict[str, Any]] = {}
            for row in rows:
                key = (
                    str(row.get("occurrence_id") or ""),
                    str(row.get("recipe_id") or ""),
                    str(row.get("outcome_definition_id") or ""),
                )
                current = latest.get(key)
                if current is None or str(row.get("computed_at") or "") >= str(
                    current.get("computed_at") or ""
                ):
                    latest[key] = row
            return list(latest.values())

    return _LakeOutcomes()


def run_measured_report(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Path | None = None,
    sources: Any | None = None,
    warehouse: Any | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """The slot. Deterministic, no model, and it never costs the night."""
    import measured_report
    from ai_jobs import digest

    moment = now or datetime.now()
    day = str(session_date or date.today().isoformat())[:10]

    try:
        target = Path(root) if root is not None else digest._default_root()
    except Exception as exc:  # noqa: BLE001
        return {"status": ledger.STATUS_FAILED, "model": "",
                "reason": f"AI store unavailable: {exc}", "outputs": []}

    if warehouse is None:
        try:
            warehouse = default_warehouse()
        except Exception:  # noqa: BLE001 - no warehouse is uncertainty, not a failure
            warehouse = None

    try:
        report = measured_report.build_report(
            day, now=moment, sources=sources, warehouse=warehouse
        )
    except Exception as exc:  # noqa: BLE001 - a half-written store never fails the night
        _log.warning("Measured report: %s could not be built (%s).", day, exc)
        return {"status": ledger.STATUS_FAILED, "model": "",
                "reason": f"the measured report could not be built: {exc}",
                "outputs": []}

    versions = published_versions(target, day)
    if versions:
        _version, newest = versions[-1]
        try:
            existing = json.loads(newest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
        if str(existing.get("report_id") or "") == report.report_id:
            return {
                "status": ledger.STATUS_OK,
                "model": "",
                "report_id": report.report_id,
                "session_date": day,
                "reason": (
                    f"the evidence is unchanged since {newest.name}; nothing was "
                    "rewritten"
                ),
                "outputs": [str(newest)],
            }
        next_version = versions[-1][0] + 1
    else:
        next_version = 1

    json_path, md_path = report_paths(target, day, next_version)
    try:
        digest._publish(
            json_path,
            json.dumps(report.as_dict(), indent=1, sort_keys=True, default=str) + "\n",
        )
        digest._publish(md_path, measured_report.render_markdown(report))
    except OSError as exc:
        return {"status": ledger.STATUS_FAILED, "model": "",
                "reason": f"the measured report could not be published: {exc}",
                "outputs": []}

    index_note = ""
    try:
        digest.write_entry_index(target, as_of=day)
    except Exception as exc:  # noqa: BLE001 - the index never costs the report
        index_note = f"; the entry index was not refreshed ({exc})"
        _log.info("Measured report: the entry index was not refreshed (%s).", exc)

    measured = sum(
        1 for cell in report.cells() if cell.state == measured_report.STATE_MEASURED
    )
    return {
        "status": ledger.STATUS_OK,
        "model": "",
        "report_id": report.report_id,
        "session_date": day,
        "version": next_version,
        "reason": (
            f"{measured} of {len(report.cells())} cells measured; published "
            f"{json_path.name}{index_note}"
        ),
        "outputs": [str(json_path), str(md_path)],
    }


__all__ = [
    "FILENAME_STEM",
    "default_warehouse",
    "latest_published",
    "narration_for",
    "published_versions",
    "report_paths",
    "run_measured_report",
]
