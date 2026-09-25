"""The trader's trading plan (P1-7): one Markdown file with fixed headings.

The trader edits `TRADING_PLAN_FILE` in any editor. The desk shows it read-only
and the night's `plan_review` slot argues with it. Every content change is kept
as `TRADING_PLAN_HISTORY_DIR/<stamp>_<seq>_<hash8>.md` (append-only), found by content
hash whenever the plan is read. A failed snapshot is logged and retried on the
next read, because the newest snapshot still differs from the plan.

File reads and writes here: call them on a worker or at night, never on the Qt
thread. `parse_plan` is pure. Evidence only: nothing here detects, scores,
ranks, gates or alerts.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_log = logging.getLogger(__name__)

#: The fixed headings, in file order.
HEADINGS: tuple[str, ...] = (
    "Goals",
    "Rules",
    "Setups I trade",
    "Risk",
    "What I am testing",
    "Decisions",
)
DECISIONS = "Decisions"
TESTING = "What I am testing"

#: The prefix of the one line the recap rule loop owns under "What I am testing".
RECAP_RULE_PREFIX = "Recap rule"

TEMPLATE = (
    "# My trading plan\n"
    "\n"
    "<!-- Edit this file in any editor. The desk shows it and the night AI checks it\n"
    "against your numbers. One idea per line, starting with \"- \". -->\n"
    "\n"
    "## Goals\n"
    "\n"
    "## Rules\n"
    "\n"
    "## Setups I trade\n"
    "\n"
    "## Risk\n"
    "\n"
    "## What I am testing\n"
    "\n"
    "## Decisions\n"
    "\n"
    "<!-- Dated lines: - YYYY-MM-DD: what you decided -->\n"
)

_HEADING = re.compile(r"^\s{0,3}#{2,3}\s+(?P<name>.+?)\s*#*\s*$")
_DATED = re.compile(r"^(?P<day>\d{4}-\d{2}-\d{2})\s*[:\-]\s*(?P<text>.+)$")
_BULLET = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
#: The exact line the recap rule loop writes; nothing else under the heading is touched.
_RECAP_LINE = re.compile(r"^- Recap rule for \d{4}-\d{2}-\d{2}: \S")
#: How long a desk plan write waits for another desk plan write.
LOCK_TIMEOUT_SECONDS = 5.0


def slug(heading: str) -> str:
    """`Setups I trade` -> `setups_i_trade`."""
    return re.sub(r"[^a-z0-9]+", "_", str(heading or "").lower()).strip("_")


def _canonical(name: str) -> str:
    wanted = slug(name)
    for heading in HEADINGS:
        if slug(heading) == wanted:
            return heading
    return ""


def _strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def parse_plan(text: Any) -> dict[str, Any]:
    """The plan as data. Pure.

    Returns ``{"sections": {heading: [line text]}, "lines": [{id, section, n,
    text}], "decisions": [{day, text, dated}], "missing": [heading],
    "unknown_sections": [name]}``. A line id is ``plan:<section slug>:<n>``,
    n counting the section's non-empty lines from 1. Comments, blank lines
    and text before the first fixed heading carry no id.
    """
    body = _strip_comments(str(text or "")).replace("\r\n", "\n")
    sections: dict[str, list[str]] = {heading: [] for heading in HEADINGS}
    seen: set[str] = set()
    unknown: list[str] = []
    current = ""
    for raw in body.split("\n"):
        match = _HEADING.match(raw)
        if match:
            name = _canonical(match.group("name"))
            current = name
            if name:
                seen.add(name)
            else:
                unknown.append(match.group("name").strip())
            continue
        line = raw.strip()
        if not current or not line:
            continue
        sections[current].append(_BULLET.sub("", line).strip())
    lines: list[dict[str, Any]] = []
    for heading in HEADINGS:
        for index, words in enumerate(sections[heading], start=1):
            lines.append(
                {"id": f"plan:{slug(heading)}:{index}", "section": heading, "n": index, "text": words}
            )
    decisions = []
    for words in sections[DECISIONS]:
        match = _DATED.match(words)
        if match:
            decisions.append({"day": match.group("day"), "text": match.group("text").strip(), "dated": True})
        else:
            decisions.append({"day": "", "text": words, "dated": False})
    return {
        "sections": sections,
        "lines": lines,
        "decisions": decisions,
        "missing": [heading for heading in HEADINGS if heading not in seen],
        "unknown_sections": unknown,
    }


def content_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# where it lives (resolved at call time so tests can redirect it)
# ---------------------------------------------------------------------------
def plan_path() -> Path:
    import project_paths

    return Path(project_paths.TRADING_PLAN_FILE)


def history_dir() -> Path:
    import project_paths

    return Path(project_paths.TRADING_PLAN_HISTORY_DIR)


def _stamp(now: datetime | None) -> str:
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def snapshots(directory: Path | None = None) -> list[Path]:
    """Every snapshot, oldest first (the stamp sorts by time)."""
    folder = Path(directory) if directory is not None else history_dir()
    try:
        return sorted(path for path in folder.glob("*.md") if path.is_file())
    except OSError:
        return []


def snapshot_at(as_of: datetime, directory: Path | None = None) -> Path | None:
    """The newest snapshot taken at or before `as_of`, or None."""
    moment = as_of if as_of.tzinfo is not None else as_of.astimezone()
    cutoff = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    chosen = None
    for path in snapshots(directory):
        if path.name[:16] <= cutoff:
            chosen = path
    return chosen


def snapshot_if_changed(
    text: str, *, directory: Path | None = None, now: datetime | None = None
) -> Path | None:
    """Keep `text` in the history when it differs from the newest snapshot.

    Returns the new snapshot, or None when nothing changed or the write
    failed. Never raises: a failed write is logged, and the next read tries
    again because the newest snapshot still differs.
    """
    folder = Path(directory) if directory is not None else history_dir()
    digest = content_hash(text)
    existing = snapshots(folder)
    if existing:
        try:
            if content_hash(existing[-1].read_text(encoding="utf-8")) == digest:
                return None
        except OSError:
            _log.warning("Trading plan: the newest snapshot %s could not be read.", existing[-1])
    # The sequence keeps two snapshots taken in the same second in order.
    target = folder / f"{_stamp(now)}_{len(existing) + 1:05d}_{digest[:8]}.md"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except FileExistsError:
        return None
    except OSError as exc:
        _log.warning("Trading plan: the change was not snapshotted (%s); the next read retries.", exc)
        return None
    return target


def read_plan(
    *, create: bool = True, snapshot: bool = True, now: datetime | None = None, path: Path | None = None
) -> dict[str, Any]:
    """Read the plan (worker or night only).

    With `create`, a missing plan is written from :data:`TEMPLATE` first -
    and only when it is missing. Returns ``{"path", "exists", "created",
    "text", "parsed", "snapshot", "error"}``; an unreadable file is
    ``error`` text, never an empty plan.
    """
    target = Path(path) if path is not None else plan_path()
    created = False
    result: dict[str, Any] = {
        "path": str(target), "exists": False, "created": False, "text": "",
        "parsed": parse_plan(""), "snapshot": "", "error": "",
    }
    if not target.exists():
        if not create:
            return result
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8", newline="") as handle:
                handle.write(TEMPLATE)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            result["error"] = f"the plan template could not be written: {exc}"
            return result
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        result["error"] = f"the plan could not be read: {exc}"
        return result
    result.update(exists=True, created=created, text=text, parsed=parse_plan(text))
    if snapshot:
        kept = snapshot_if_changed(text, now=now, directory=_history_for(path))
        result["snapshot"] = str(kept) if kept else ""
    return result


def _history_for(path: Path | None) -> Path | None:
    """A plan at an explicit path keeps its history beside it (tests, tools)."""
    return None if path is None else Path(path).parent / "trading_plan_history"


# ---------------------------------------------------------------------------
# writing (the desk's two writers; each also snapshots)
# ---------------------------------------------------------------------------
class PlanWriteError(RuntimeError):
    """The plan file could not be written. Raised, never swallowed."""


def _section_bounds(lines: list[str], heading: str) -> tuple[int, int] | None:
    """(heading index, end index exclusive) of `heading` in the raw lines."""
    start = None
    for index, raw in enumerate(lines):
        match = _HEADING.match(raw)
        if not match:
            continue
        if start is not None:
            return start, index
        if _canonical(match.group("name")) == heading:
            start = index
    return (start, len(lines)) if start is not None else None


def _insert_at_end(lines: list[str], heading: str, new_line: str) -> list[str]:
    bounds = _section_bounds(lines, heading)
    if bounds is None:
        body = list(lines)
        while body and not body[-1].strip():
            body.pop()
        return body + ["", f"## {heading}", "", new_line, ""]
    start, end = bounds
    last = start
    for index in range(start + 1, end):
        if lines[index].strip():
            last = index
    position = last + 1
    out = list(lines[:position])
    if position == start + 1:
        out.append("")
    out.append(new_line)
    rest = list(lines[position:])
    if rest and rest[0].strip():
        out.append("")
    return out + rest


def _write_plan(target: Path, text: str) -> None:
    temporary = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise PlanWriteError(f"the trading plan was not saved: {exc}") from exc


def _edit(mutate, *, now: datetime | None, path: Path | None) -> dict[str, Any]:
    """Read-modify-write under ONE machine-wide lock for the plan file."""
    from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path

    target = Path(path) if path is not None else plan_path()
    try:
        with local_writer_lock(lock_key_for_path(target), timeout_seconds=LOCK_TIMEOUT_SECONDS):
            return _edit_locked(mutate, target, now=now, path=path)
    except LocalLockUnavailable as exc:
        raise PlanWriteError(f"the trading plan is busy; nothing was written: {exc}") from exc


def _edit_locked(mutate, target: Path, *, now: datetime | None, path: Path | None) -> dict[str, Any]:
    current = read_plan(create=True, snapshot=True, now=now, path=path)
    if current["error"]:
        raise PlanWriteError(current["error"])
    old = current["text"]
    newline = "\r\n" if "\r\n" in old else "\n"
    lines = old.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    new_lines = mutate(lines)
    if new_lines is None:
        return {"changed": False, "path": str(target), "snapshot": ""}
    text = newline.join(new_lines) + newline
    _write_plan(target, text)
    kept = snapshot_if_changed(text, now=now, directory=_history_for(path))
    return {"changed": True, "path": str(target), "snapshot": str(kept) if kept else ""}


def _day(now: datetime | None, day: Any = None) -> str:
    if isinstance(day, date):
        return day.isoformat()
    if str(day or "").strip():
        return str(day).strip()[:10]
    moment = now or datetime.now().astimezone()
    return moment.date().isoformat()


def append_decision(
    text: Any, *, now: datetime | None = None, day: Any = None, path: Path | None = None,
    unless_contains: str = "",
) -> dict[str, Any]:
    """Add ``- YYYY-MM-DD: text`` as the last line under Decisions. Raises PlanWriteError.

    With `unless_contains`, nothing is written when a Decisions line already
    holds that text (checked under the lock, so a repeat never doubles a line).
    """
    words = " ".join(str(text or "").split())
    if not words:
        raise PlanWriteError("a decision needs text")
    line = f"- {_day(now, day)}: {words}"
    marker = str(unless_contains or "").strip()

    def mutate(lines: list[str]) -> list[str] | None:
        if marker:
            bounds = _section_bounds(lines, DECISIONS)
            if bounds is not None and any(marker in lines[i] for i in range(bounds[0] + 1, bounds[1])):
                return None
        return _insert_at_end(lines, DECISIONS, line)

    return _edit(mutate, now=now, path=path)


def set_testing_rule(
    text: Any, *, for_day: Any, now: datetime | None = None, path: Path | None = None
) -> dict[str, Any]:
    """Put the recap's rule under "What I am testing" as ONE line it owns.

    The line reads ``- Recap rule for YYYY-MM-DD: text``; an earlier recap line
    is replaced, and every line the trader typed is left alone. The history
    keeps the old rule. Raises PlanWriteError.
    """
    words = " ".join(str(text or "").split())
    if not words:
        raise PlanWriteError("a rule needs text")
    line = f"- {RECAP_RULE_PREFIX} for {_day(now, for_day)}: {words}"

    def mutate(lines: list[str]) -> list[str] | None:
        bounds = _section_bounds(lines, TESTING)
        if bounds is not None:
            start, end = bounds
            for index in range(start + 1, end):
                if _RECAP_LINE.match(lines[index].strip()):
                    if lines[index] == line:
                        return None
                    out = list(lines)
                    out[index] = line
                    return out
        return _insert_at_end(lines, TESTING, line)

    return _edit(mutate, now=now, path=path)


def plan_lines(parsed: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The citable lines of a parsed plan."""
    return [dict(row) for row in (parsed or {}).get("lines") or () if isinstance(row, Mapping)]


__all__ = [
    "DECISIONS",
    "HEADINGS",
    "PlanWriteError",
    "RECAP_RULE_PREFIX",
    "TEMPLATE",
    "TESTING",
    "append_decision",
    "content_hash",
    "history_dir",
    "parse_plan",
    "plan_lines",
    "plan_path",
    "read_plan",
    "set_testing_rule",
    "slug",
    "snapshot_at",
    "snapshot_if_changed",
    "snapshots",
]
