"""Night pack (P15a): what the night wrote, read-only, so the day coach stands on it.

``night_pack(section, days)`` reads the night's citation-checked publications and turns
them into citable rows ``night:<kind>:<date>:<n>``. Every row keeps the artifact's own
source id in ``src`` and ends its text with ``(src: ...)``, so a citation can be traced back
to the night's evidence. Sections: ``day_review`` (the latest day review's "were you right"
verdicts), ``ideas`` (top improvement ideas), ``contrast`` (miss and prediction contrast
headlines), ``week`` (the week review), ``story`` (the market story), ``digest`` (the daily
digest narration). ``night:asof`` names each artifact's date so a stale read shows as stale.
A missing artifact is one ``night:<kind>:none`` row naming it, never silence.

Pure and Qt-free: files are read directly (JSON/JSONL), nothing is written. Also the shared
reader for the latest ticker brief per symbol (``latest_briefs``), used by ``pick_pack``.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "night_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "What the night wrote for you: the day review's verdicts on your reads, improvement ideas, "
            "miss and prediction contrasts (what you are missing), the week review, the market story and "
            "the daily digest. Use it for 'what did the night say', 'what am I missing', 'what did I get wrong'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string",
                            "enum": ["all", "day_review", "ideas", "contrast", "week", "story", "digest"],
                            "description": "One part, or all of it (default)."},
                "days": {"type": "integer", "description": "How many recent nights of dated reads (1-5, default 1)."},
            },
        },
    },
}

ET = ZoneInfo("America/New_York")
SECTIONS = ("day_review", "ideas", "contrast", "week", "story", "digest")
MAX_DAYS = 5
MAX_TEXT = 280
MAX_IDEAS = 5
MAX_CONTRAST_ROWS = 5
MAX_WEEK_ROWS = 6
MAX_STORY_ROWS = 5
MAX_DIGEST_LINES = 6
MAX_FACT_ROWS = 10
BRIEF_SESSIONS = 10
BRIEF_LINES = 3
BRIEF_STATUS = "briefed"
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_WEEK = re.compile(r"(\d{4}-W\d{2})")
#: The daily digest narration's statement lists, in the order they are read.
DIGEST_SECTIONS = ("what_is_working", "what_is_not_working", "lessons_for_tomorrow", "risk_notes", "best_candidates")
LABELS = {
    "day_review": "day review", "ideas": "ideas", "miss": "miss contrast", "prediction": "prediction contrast",
    "week": "week review", "story": "market story", "digest": "daily digest",
}


@dataclass(frozen=True)
class NightPaths:
    """Every place the pack reads; tests pass fixture paths, the app uses :func:`live_paths`. None = no store."""

    digests: Path | None = None  # ai_store digests/: daily digest, contrasts, mentor digests, coach brief
    briefs: Path | None = None  # ai_store briefs/: <year>/<session>/ticker_briefs_manifest.jsonl
    day_review: Path | None = None  # DAY_REVIEW_DIR: narration/<date>.json, week/<YYYY-Www>.json
    ideas: Path | None = None  # AI_IDEAS_FILE (append-only JSONL, folded by idea_id)
    ideas_state: Path | None = None  # AI_IDEAS_STATE_FILE (the trader's keeps and dismissals)
    story: Path | None = None  # MARKET_STORY_NARRATIONS_DIR: <date>.json


def live_paths() -> NightPaths:
    import project_paths as pp

    digests = briefs = None
    try:
        from ai_jobs import store as ai_store

        digests = ai_store.digests_dir(create=False)
        briefs = ai_store.briefs_dir(create=False)
    except Exception:  # noqa: BLE001 - no ai_store configured = those reads are "none on file"
        pass
    return NightPaths(
        digests=digests, briefs=briefs, day_review=Path(pp.DAY_REVIEW_DIR), ideas=Path(pp.AI_IDEAS_FILE),
        ideas_state=Path(pp.AI_IDEAS_STATE_FILE), story=Path(pp.MARKET_STORY_NARRATIONS_DIR),
    )


# ---------------------------------------------------------------- small helpers
def _clean(value: Any, cap: int = MAX_TEXT) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= cap else text[: cap - 3].rstrip() + "..."


def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def _today(now: datetime | None) -> date:
    return _now(now).astimezone(ET).date()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dated(paths: Iterable[Path], today: date) -> list[tuple[str, Path]]:
    """``(date, path)`` newest first, for files whose name carries a date on or before ``today``."""
    out: list[tuple[str, Path]] = []
    for path in paths:
        found = _DATE.search(path.name)
        if found and found.group(1) <= today.isoformat():
            out.append((found.group(1), path))
    return sorted(out, key=lambda item: (item[0], item[1].name), reverse=True)


def _glob(folder: Path | None, pattern: str) -> list[Path]:
    if folder is None:
        return []
    try:
        return list(Path(folder).glob(pattern))
    except OSError:
        return []


def _row(kind: str, day: str, n: int, text: str, src: str, **extra: Any) -> dict[str, Any]:
    body = _clean(text)
    return {"id": f"night:{kind}:{day}:{n}", "kind": "night", "section": kind, "date": day, "src": src,
            "text": f"{body} (src: {src})" if src else body, **extra}


def _none(kind: str, what: str) -> dict[str, Any]:
    return {"id": f"night:{kind}:none", "kind": "none", "section": kind, "date": "", "src": "",
            "text": f"No {LABELS.get(kind, kind)} on file ({what}); unknown, not 'nothing'"}


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "unknown"


def stable_ref(row_id: str, text: str) -> int:
    """A stable integer for one row's id and text (embedding ``ref_id``): a changed artifact is a new ref."""
    import hashlib

    digest = hashlib.sha256(f"{row_id}\n{text}".encode("utf-8")).hexdigest()
    return int(digest[:13], 16)


# ---------------------------------------------------------------- day review
def day_review_rows(paths: NightPaths, today: date, days: int = 1) -> tuple[list[dict[str, Any]], list[str]]:
    """The latest ``days`` day reviews: headline, then each "were you right" claim with its verdict."""
    files = _dated(_glob(paths.day_review / "narration" if paths.day_review else None, "????-??-??.json"), today)
    rows: list[dict[str, Any]] = []
    dates: list[str] = []
    for day, path in files[:days]:
        payload = _read_json(path)
        narration = payload.get("narration") if isinstance(payload, Mapping) else None
        if not isinstance(narration, Mapping):
            continue
        dates.append(day)
        graded = payload.get("graded") or {}
        head = f"Day review {day}: {_clean(narration.get('headline'), 160)}"
        if graded:
            head += f" (graded {graded.get('reads_graded', 0)} of {graded.get('reads_in_pack', 0)} reads)"
        rows.append(_row("day_review", day, 0, head, f"day_review:{day}"))
        for n, claim in enumerate(narration.get("were_you_right") or (), start=1):
            if not isinstance(claim, Mapping):
                continue
            src = "; ".join(x for x in (str(claim.get("source_id") or ""), str(claim.get("evidence_id") or "")) if x)
            rows.append(_row("day_review", day, n, f'You said "{_clean(claim.get("claim"), 170)}" -> '
                             f'{_clean(claim.get("verdict"), 60) or "unknown"}', src,
                             verdict=str(claim.get("verdict") or ""), source_id=str(claim.get("source_id") or ""),
                             evidence_id=str(claim.get("evidence_id") or "")))
    if not rows:
        rows.append(_none("day_review", "no day_review_narration file"))
    return rows, dates


# ---------------------------------------------------------------- ideas
def read_ideas(paths: NightPaths) -> list[dict[str, Any]]:
    """Every idea folded by ``idea_id`` (last row wins), minus the ones the trader dismissed."""
    if paths.ideas is None:
        return []
    try:
        text = Path(paths.ideas).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    folded: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        try:
            row = json.loads(line) if line.strip() else None
        except ValueError:
            continue
        if isinstance(row, Mapping) and str(row.get("idea_id") or ""):
            folded[str(row["idea_id"])] = dict(row)
    state = _read_json(paths.ideas_state) if paths.ideas_state else None
    dismissed = {key for key, value in (state or {}).items()
                 if isinstance(value, Mapping) and str(value.get("status") or "") == "dismissed"}
    return [row for key, row in folded.items() if key not in dismissed]


def idea_rows(paths: NightPaths, today: date, limit: int = MAX_IDEAS) -> tuple[list[dict[str, Any]], list[str]]:
    """Top ideas: newest night first, then most often seen; ``n`` is the idea's place in its night."""
    ideas = [row for row in read_ideas(paths) if str(row.get("session_date") or "")[:10] <= today.isoformat()
             and str(row.get("text") or "").strip()]
    ideas.sort(key=lambda row: (str(row.get("session_date") or ""), int(row.get("seen_count") or 0),
                                str(row.get("idea_id") or "")), reverse=True)
    top = ideas[:limit]
    if not top:
        return [_none("ideas", "no improvement_ideas row")], []
    by_day: dict[str, list[dict[str, Any]]] = {}
    for row in top:
        by_day.setdefault(str(row.get("session_date"))[:10], []).append(row)
    numbers = {str(row["idea_id"]): n for day_rows in by_day.values()
               for n, row in enumerate(sorted(day_rows, key=lambda r: str(r["idea_id"])), start=1)}
    rows = []
    for row in top:
        day = str(row.get("session_date"))[:10]
        evidence = [str(ref) for ref in row.get("evidence") or () if str(ref)]
        kind = str(row.get("kind") or "idea")
        measurable = str(row.get("measurable") or "")
        label = f"Idea ({kind}{', measured by ' + measurable if measurable else ''}, seen {row.get('seen_count', 1)}x)"
        rows.append(_row("ideas", day, numbers[str(row["idea_id"])], f"{label}: {_clean(row.get('text'), 220)}",
                         ", ".join(evidence[:3]) or str(row["idea_id"]), idea_id=str(row["idea_id"]),
                         evidence=evidence))
    return rows, sorted(by_day, reverse=True)


# ---------------------------------------------------------------- contrasts
def _feature_text(group: Mapping[str, Any]) -> str:
    features = [f for f in group.get("features") or () if isinstance(f, Mapping)]
    if not features:
        return ""
    top = features[0]
    try:
        auc = f"{float(top.get('auc')):.2f}"
    except (TypeError, ValueError):
        auc = "unknown"
    return (f"; top feature {top.get('feature')} AUC {auc} ({group.get('label_a', 'a')} n={top.get('n_a')} median "
            f"{top.get('median_a')} vs {group.get('label_b', 'b')} n={top.get('n_b')} median {top.get('median_b')})")


def miss_rows(paths: NightPaths, today: date) -> tuple[list[dict[str, Any]], list[str]]:
    """The latest miss contrast: its statement, then the leader groups and other reportable ones (<= 5 rows)."""
    files = _dated(_glob(paths.digests, "miss_contrast-????-??-??.json"), today)
    payload = _read_json(files[0][1]) if files else None
    if not isinstance(payload, Mapping):
        return [_none("miss", "no miss_contrast file")], []
    day = files[0][0]
    stem = f"miss_contrast-{day}"
    rows = [_row("miss", day, 0, f"Miss contrast {day} over {payload.get('window_sessions', '?')} sessions: "
                 f"{_clean(payload.get('statement'), 200)}", stem)]
    groups = {str(g.get("name")): g for g in payload.get("groups") or () if isinstance(g, Mapping) and g.get("name")}
    order = [name for name in payload.get("leaders") or () if name in groups]
    order += sorted((name for name, g in groups.items() if g.get("reportable") and name not in order),
                    key=lambda name: (-(groups[name].get("rate") or 0.0), name))
    for n, name in enumerate(order[: MAX_CONTRAST_ROWS - 1], start=1):
        group = groups[name]
        text = (f"Miss contrast '{name}': real-miss rate {_pct(group.get('rate'))} of {group.get('measured', '?')} "
                f"measured (n={group.get('n', '?')}, {group.get('misses', '?')} misses)" + _feature_text(group))
        rows.append(_row("miss", day, n, text, f"{stem}:group:{name}", group=name))
    return rows, [day]


def prediction_rows(paths: NightPaths, today: date) -> tuple[list[dict[str, Any]], list[str]]:
    """The latest prediction contrast: its statement, then right/wrong per horizon (<= 5 rows)."""
    files = _dated(_glob(paths.digests, "prediction_contrast-????-??-??.json"), today)
    payload = _read_json(files[0][1]) if files else None
    if not isinstance(payload, Mapping):
        return [_none("prediction", "no prediction_contrast file")], []
    day = files[0][0]
    stem = f"prediction_contrast-{day}"
    rows = [_row("prediction", day, 0, f"Prediction contrast {day}: {_clean(payload.get('statement'), 200)}", stem)]
    horizons = payload.get("horizons") or {}
    for n, key in enumerate(sorted(horizons)[: MAX_CONTRAST_ROWS - 1], start=1):
        h = horizons[key] if isinstance(horizons[key], Mapping) else {}
        try:
            lb = f"{float(h.get('rate_lb')):.2f}"
        except (TypeError, ValueError):
            lb = "unknown"
        text = (f"Your reads, {h.get('label') or key}: right {h.get('right', 0)}, wrong {h.get('wrong', 0)}, flat "
                f"{h.get('flat', 0)}, pending {h.get('pending', 0)}; right rate {_pct(h.get('rate'))} (LB {lb}, "
                f"n={h.get('n', 0)}{'' if h.get('meets_floor') else ', under the floor'})")
        rows.append(_row("prediction", day, n, text, f"{stem}:horizon:{key}", horizon=key))
    return rows, [day]


# ---------------------------------------------------------------- week
def week_id(day: date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def week_rows(paths: NightPaths, today: date) -> tuple[list[dict[str, Any]], list[str]]:
    """Mon-Wed: last week's review; Thu-Sun: this week's when written (else last week's). <= 6 rows."""
    folder = paths.day_review / "week" if paths.day_review else None
    this_week, last_week = week_id(today), week_id(today - timedelta(days=7))
    wanted = [last_week, this_week] if today.weekday() <= 2 else [this_week, last_week]
    payload, wid = None, ""
    for candidate in wanted:
        payload = _read_json(folder / f"{candidate}.json") if folder else None
        if isinstance(payload, Mapping):
            wid = candidate
            break
    if not wid:
        # An older review is still the newest the night wrote: shown with its own week id.
        older = sorted((p for p in _glob(folder, "????-W??.json") if p.stem <= this_week), reverse=True)
        payload = _read_json(older[0]) if older else None
        wid = older[0].stem if older and isinstance(payload, Mapping) else ""
    narration = payload.get("narration") if isinstance(payload, Mapping) else None
    if not wid or not isinstance(narration, Mapping):
        return [_none("week", "no week_review_narration file")], []
    src = f"week_review:{wid}"
    rows = [_row("week", wid, 0, f"Week review {wid}: {_clean(narration.get('headline'), 180)}", src)]
    right = narration.get("were_you_right") or {}
    if isinstance(right, Mapping):
        rows.append(_row("week", wid, 1, f"Week reads: right {right.get('right', 0)}, wrong {right.get('wrong', 0)}, "
                         f"unresolved {right.get('unresolved', 0)}", src))
    items: list[tuple[str, str]] = []
    for key, label in (("tendencies", "Tendency"), ("chased", "Chased"), ("examples", "Read")):
        source = right.get("examples") if key == "examples" and isinstance(right, Mapping) else narration.get(key)
        for item in source or ():
            if isinstance(item, Mapping) and str(item.get("text") or "").strip():
                n_text = f" (n={item['n']})" if item.get("n") is not None else ""
                items.append((f"{label}: {_clean(item.get('text'), 200)}{n_text}", str(item.get("source_id") or src)))
    for text in narration.get("next_week_watch") or ():
        if str(text or "").strip():
            items.append((f"Watch next week: {_clean(text, 200)}", src))
    for n, (text, item_src) in enumerate(items[: MAX_WEEK_ROWS - len(rows)], start=len(rows)):
        rows.append(_row("week", wid, n, text, item_src))
    return rows, [wid]


# ---------------------------------------------------------------- market story
def story_rows(paths: NightPaths, today: date, days: int = 1) -> tuple[list[dict[str, Any]], list[str]]:
    """The latest market story: its summary, then what changed (<= 5 rows a night)."""
    files = _dated(_glob(paths.story, "????-??-??.json"), today)
    rows: list[dict[str, Any]] = []
    dates: list[str] = []
    for day, path in files[:days]:
        payload = _read_json(path)
        narration = payload.get("narration") if isinstance(payload, Mapping) else None
        if not isinstance(narration, Mapping):
            continue
        dates.append(day)
        periods = payload.get("periods") or {}
        weekly = f"; rollup:weekly:{periods['weekly']}" if isinstance(periods, Mapping) and periods.get("weekly") else ""
        src = f"market_story:{day}"
        rows.append(_row("story", day, 0, f"Market story {day}: {_clean(narration.get('summary'), 240)}", src + weekly))
        for n, change in enumerate((narration.get("changes") or ())[: MAX_STORY_ROWS - 1], start=1):
            if str(change or "").strip():
                rows.append(_row("story", day, n, f"Changed: {_clean(change, 220)}", src))
    if not rows:
        rows.append(_none("story", "no market_story_narration file"))
    return rows, dates


# ---------------------------------------------------------------- daily digest
def digest_rows(paths: NightPaths, today: date, days: int = 1) -> tuple[list[dict[str, Any]], list[str]]:
    """The latest daily digest narration: its summary and statements, <= 6 lines a night."""
    files = _dated(_glob(paths.digests / "narration" if paths.digests else None, "*/????-??-??.json"), today)
    rows: list[dict[str, Any]] = []
    dates: list[str] = []
    for day, path in files[:days]:
        payload = _read_json(path)
        narration = payload.get("narration") if isinstance(payload, Mapping) else None
        if not isinstance(narration, Mapping):
            continue
        dates.append(day)
        lines: list[tuple[str, str]] = []
        summary = _clean(narration.get("executive_summary"), 240)
        if summary and not summary.lower().startswith("executive summary withheld"):
            lines.append((f"Digest {day}: {summary}", f"daily_digest:{day}"))
        for section in DIGEST_SECTIONS:
            for item in narration.get(section) or ():
                if not isinstance(item, Mapping) or not str(item.get("statement") or "").strip():
                    continue
                refs = [str(ref) for ref in item.get("evidence_refs") or () if str(ref)]
                metric = item.get("metric_ref") if isinstance(item.get("metric_ref"), Mapping) else {}
                src = ", ".join(refs) or str(metric.get("source_id") or "") or f"daily_digest:{day}"
                lines.append((f"{section.replace('_', ' ')}: {_clean(item.get('statement'), 220)}", src))
        for n, (text, src) in enumerate(lines[:MAX_DIGEST_LINES]):
            rows.append(_row("digest", day, n, text, src))
    if not rows:
        rows.append(_none("digest", "no daily_digest narration file"))
    return rows, dates


def _fact_value(node: Any) -> str:
    if not isinstance(node, Mapping) or "value" not in node:
        return ""
    value = node.get("value")
    shown = "unknown" if value is None else (f"{value:.4g}" if isinstance(value, float) else str(value))
    return f"{shown} (n={node.get('n', '?')})"


def digest_fact_rows(paths: NightPaths, session: str, limit: int = MAX_FACT_ROWS) -> tuple[list[dict[str, Any]], str]:
    """Headline values of the newest daily digest fact pack on or before ``session`` (<= 10 rows)."""
    try:
        today = date.fromisoformat(str(session)[:10])
    except ValueError:
        return [_none("digest_fact", "no session date")], ""
    files = _dated(_glob(paths.digests / "facts" if paths.digests else None, "*/????-??-??.json"), today)
    payload = _read_json(files[0][1]) if files else None
    if not isinstance(payload, Mapping):
        return [_none("digest_fact", "no daily_digest facts file")], ""
    day = files[0][0]
    picks: list[tuple[str, Any, str]] = []
    outcomes = (payload.get("outcomes") or {}).get("overall") or {}
    pointer = ((payload.get("outcomes") or {}).get("pointer") or {}).get("source_id") or "outcomes"
    for key in ("close_r", "mfe_r", "mae_r", "stop_exit_r", "last_measured_r"):
        picks.append((f"settled outcomes mean {key}", outcomes.get(key), str(pointer)))
    picks.append(("settled outcomes distinct symbols", outcomes.get("symbols"), str(pointer)))
    behaviour = payload.get("behaviour") or {}
    b_src = str((behaviour.get("pointer") or {}).get("source_id") or "behaviour")
    picks.append(("alerts reviewed", behaviour.get("reviewed"), b_src))
    picks.append(("median review dwell ms", behaviour.get("median_dwell_ms"), b_src))
    ops = payload.get("operations") or {}
    picks.append(("night job rows", ops.get("job_rows"), str((ops.get("pointer") or {}).get("source_id") or "ops")))
    rows: list[dict[str, Any]] = []
    for label, node, src in picks:
        shown = _fact_value(node)
        if shown and len(rows) < limit:
            rows.append(_row("digest_fact", day, len(rows), f"Daily digest {day}: {label} {shown}", src))
    names = payload.get("names") or {}
    for key in sorted(names):
        node = names[key]
        if isinstance(node, Mapping) and node.get("n") is not None and len(rows) < limit:
            rows.append(_row("digest_fact", day, len(rows), f"Daily digest {day}: {key.replace('_', ' ')} n={node['n']}",
                             str(node.get("source_id") or key)))
    return rows or [_none("digest_fact", f"daily digest {day} carried no headline value")], day


# ---------------------------------------------------------------- ticker briefs
_BRIEF_CACHE: dict[str, tuple[tuple[int, int], dict[str, dict[str, Any]]]] = {}


def _manifest(path: Path) -> dict[str, dict[str, Any]]:
    """``{SYM: newest briefed row}`` of one session manifest, cached by size and mtime."""
    try:
        stat = path.stat()
    except OSError:
        return {}
    sig = (stat.st_size, stat.st_mtime_ns)
    hit = _BRIEF_CACHE.get(str(path))
    if hit and hit[0] == sig:
        return hit[1]
    out: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return {}
    for line in lines:
        try:
            row = json.loads(line) if line.strip() else None
        except ValueError:
            continue
        if not isinstance(row, Mapping):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        summary = (row.get("result") or {}).get("summary") if isinstance(row.get("result"), Mapping) else None
        if sym and str(row.get("status") or "") == BRIEF_STATUS and isinstance(summary, Mapping):
            out[sym] = dict(row)
    _BRIEF_CACHE[str(path)] = (sig, out)
    return out


def brief_lines(summary: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(<= 3 lines, their evidence refs) from one brief: the summary, then the first statement of each list."""
    lines: list[str] = []
    refs: list[str] = []
    head = _clean(summary.get("executive_summary"), 200)
    if head:
        lines.append(head)
    for section in ("what_is_working", "what_is_not_working", "risk_notes", "lessons_for_tomorrow", "best_candidates"):
        for item in summary.get(section) or ():
            text = _clean(item.get("statement"), 200) if isinstance(item, Mapping) else ""
            if text and not text.startswith("[system]"):
                lines.append(text)
                refs.extend(str(r) for r in item.get("evidence_refs") or () if str(r) and str(r) not in refs)
                break
    return lines[:BRIEF_LINES], refs


def latest_briefs(root: Path | None, today: date, *, sessions: int = BRIEF_SESSIONS) -> dict[str, dict[str, Any]]:
    """``{SYM: {session, lines, evidence_refs}}``: each symbol's newest briefed row in the last ``sessions`` manifests."""
    if root is None:
        return {}
    # The session is the manifest's folder name: <year>/<session>/ticker_briefs_manifest.jsonl.
    manifests = _glob(Path(root), "????/????-??-??/ticker_briefs_manifest.jsonl")
    found = sorted(((p.parent.name, p) for p in manifests if _DATE.fullmatch(p.parent.name)
                    and p.parent.name <= today.isoformat()), reverse=True)[:sessions]
    out: dict[str, dict[str, Any]] = {}
    for session, path in found:
        for sym, row in _manifest(path).items():
            if sym in out:
                continue
            lines, refs = brief_lines(row["result"]["summary"])
            if lines:
                out[sym] = {"session": session, "lines": lines, "evidence_refs": refs}
    return out


def brief_rows(root: Path | None, today: date) -> list[dict[str, Any]]:
    """One ``brief:<SYM>:<session>`` row per symbol (the embed queue's ``brief`` kind)."""
    rows = []
    for sym, brief in sorted(latest_briefs(root, today).items()):
        refs = ", ".join(brief["evidence_refs"][:3])
        text = f"{sym} night brief ({brief['session']}): " + " | ".join(brief["lines"])
        rows.append({"id": f"brief:{sym}:{brief['session']}", "kind": "brief", "symbol": sym,
                     "date": brief["session"], "src": refs, "text": f"{text} (src: {refs})" if refs else text})
    return rows


# ---------------------------------------------------------------- build
_READERS = {
    "day_review": lambda p, t, d: [day_review_rows(p, t, d)],
    "ideas": lambda p, t, d: [idea_rows(p, t)],
    "contrast": lambda p, t, d: [miss_rows(p, t), prediction_rows(p, t)],
    "week": lambda p, t, d: [week_rows(p, t)],
    "story": lambda p, t, d: [story_rows(p, t, d)],
    "digest": lambda p, t, d: [digest_rows(p, t, d)],
}
_ASOF_KINDS = {"day_review": ("day_review",), "ideas": ("ideas",), "contrast": ("miss", "prediction"),
               "week": ("week",), "story": ("story",), "digest": ("digest",)}


def _age(stamp: str, today: date) -> str:
    if not _DATE.fullmatch(stamp):
        return ""  # a week id carries its own age
    try:
        days = (today - date.fromisoformat(stamp[:10])).days
    except ValueError:
        return ""
    return "" if days <= 1 else f", {days} days old"


def build(section: str = "all", days: Any = 1, *, now: datetime | None = None,
          paths: NightPaths | None = None) -> Pack:
    """Build the night pack. File reads: call it on a worker."""
    wanted = str(section or "all").strip().lower()
    if wanted != "all" and wanted not in SECTIONS:
        return make_pack(NAME, (), empty_text=f"night_pack sections are all, {', '.join(SECTIONS)}; not {section!r}")
    try:
        span = max(1, min(MAX_DAYS, int(days or 1)))
    except (TypeError, ValueError):
        span = 1
    today = _today(now)
    src = paths or live_paths()
    parts = SECTIONS if wanted == "all" else (wanted,)
    body: list[dict[str, Any]] = []
    asof: list[str] = []
    for part in parts:
        try:
            results = _READERS[part](src, today, span)
        except Exception as exc:  # noqa: BLE001 - one unreadable artifact never blanks the others
            results = [([{**_none(kind, f"unreadable: {type(exc).__name__}"), "kind": "unknown"}], [])
                       for kind in _ASOF_KINDS[part]]
        for kind, (rows, dates) in zip(_ASOF_KINDS[part], results, strict=True):
            body.extend(rows)
            label = LABELS[kind]
            asof.append(f"{label} {', '.join(dates)}{_age(dates[0], today)}" if dates else f"{label} none")
    head = {"id": "night:asof", "kind": "asof", "date": today.isoformat(),
            "text": f"Night reads as of market date {today.isoformat()}: " + "; ".join(asof)}
    return make_pack(NAME, [head, *body])


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc)  # Wed 2026-09-30, 07:00 PT


def write_fixture_world(root: Path | str) -> NightPaths:
    """One small file per kind under ``root`` (night of Tue 2026-09-29), plus an older one to prove "latest"."""
    base = Path(root)
    digests, briefs, review, story = base / "digests", base / "briefs", base / "day_review", base / "story"
    for folder in (digests / "narration" / "2026", digests / "facts" / "2026", review / "narration",
                   review / "week", story, briefs / "2026" / "2026-09-28", briefs / "2026" / "2026-09-29"):
        folder.mkdir(parents=True, exist_ok=True)

    def put(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    put(review / "narration" / "2026-09-28.json", {"schema": "day_review_narration_v1", "narration": {
        "headline": "Older day.", "were_you_right": []}})
    put(review / "narration" / "2026-09-29.json", {
        "schema": "day_review_narration_v1", "session_date": "2026-09-29",
        "graded": {"reads_graded": 2, "reads_in_pack": 3},
        "narration": {"headline": "Choppy session testing support.", "were_you_right": [
            {"claim": "we are compressed, an H- below holds us up", "source_id": "said:mj-2026-09-29-f3f8:prediction",
             "verdict": "right", "evidence_id": "read:rd-5fa9"},
            {"claim": "SPY tests the 50sma later this week", "source_id": "said:mj-2026-09-29-a81d:prediction",
             "verdict": "wrong", "evidence_id": "read:rd-03cd"},
        ]}})
    put(review / "week" / "2026-W39.json", {"schema": "week_review_narration_v1", "week_id": "2026-W39", "narration": {
        "headline": "Choppy week with a slight downward bias.",
        "were_you_right": {"right": 14, "wrong": 2, "unresolved": 14, "examples": []},
        "tendencies": [{"text": "You call bounces early in a bear channel", "source_id": "2026-09-24/read:rd-1dbc",
                        "n": 6}],
        "chased": [], "next_week_watch": ["The 50 SMA on SPY"]}})
    ideas = base / "ai_ideas.jsonl"
    ideas.write_text("\n".join(json.dumps(row) for row in (
        {"idea_id": "idea:2026-09-25:aaa", "session_date": "2026-09-25", "kind": "process", "seen_count": 1,
         "measurable": "", "text": "An old idea.", "evidence": ["2026-09-25/report_card:did_well"]},
        {"idea_id": "idea:2026-09-29:bbb", "session_date": "2026-09-29", "kind": "process", "seen_count": 2,
         "measurable": "report_card_did_well_rate", "text": "Liked names you did not trade ran well; ask why.",
         "evidence": ["2026-09-28/report_card:did_well"]},
        {"idea_id": "idea:2026-09-29:ccc", "session_date": "2026-09-29", "kind": "program", "seen_count": 1,
         "measurable": "", "text": "Measure the congruence checks.",
         "evidence": ["2026-09-28/report_card:congruence"]},
        {"idea_id": "idea:2026-09-29:ddd", "session_date": "2026-09-29", "kind": "process", "seen_count": 1,
         "measurable": "", "text": "Dismissed by the trader.", "evidence": ["2026-09-28/report_card:x"]},
    )) + "\n", encoding="utf-8")
    state = base / "ai_ideas_state.json"
    put(state, {"idea:2026-09-29:ddd": {"status": "dismissed", "at": "2026-09-30T06:00:00+00:00"}})
    put(digests / "miss_contrast-2026-09-28.json", {"schema": "miss_contrast_v1", "statement": "older",
                                                    "groups": [], "leaders": []})
    put(digests / "miss_contrast-2026-09-29.json", {
        "schema": "miss_contrast_v1", "session_date": "2026-09-29", "window_sessions": 20,
        "statement": "observational, not causal: top 2 of 3 group(s).", "leaders": ["incoming_trendline"],
        "groups": [
            {"name": "incoming_trendline", "reportable": True, "rate": 0.2111, "measured": 90, "n": 105, "misses": 19,
             "label_a": "real_run", "label_b": "dud",
             "features": [{"feature": "atr_pct", "auc": 0.66, "n_a": 19, "n_b": 41, "median_a": 3.1, "median_b": 2.2}]},
            {"name": "overhead_horizontal", "reportable": True, "rate": 0.4667, "measured": 30, "n": 41, "misses": 14,
             "features": []},
            {"name": "too_early", "reportable": False, "rate": None, "measured": 3, "n": 17},
        ]})
    put(digests / "prediction_contrast-2026-09-29.json", {
        "schema": "prediction_contrast_v1", "session_date": "2026-09-29",
        "statement": "observational, not causal: 41 clicked read(s), right against wrong.",
        "horizons": {"rest_of_day": {"label": "Rest of day", "right": 19, "wrong": 1, "flat": 5, "pending": 0,
                                     "rate": 0.76, "rate_lb": 0.5657, "n": 25, "meets_floor": False}}})
    put(digests / "narration" / "2026" / "2026-09-29.json", {
        "schema": "daily_digest_narration_v1", "session_date": "2026-09-29", "narration": {
            "executive_summary": "Executive summary withheld: it asserted a position without a source.",
            "what_is_working": [{"statement": "The D1 scan found several high-tier shorts.",
                                 "evidence_refs": ["scan.tier_list"]}],
            "what_is_not_working": [{"statement": "M5 alerts closed mixed, mean close_r -0.05.",
                                     "evidence_refs": ["outcomes.intraday_finals"]}],
            "lessons_for_tomorrow": [], "risk_notes": [], "best_candidates": []}})
    put(digests / "facts" / "2026" / "2026-09-29.json", {
        "schema": "daily_digest_facts_v3", "session_date": "2026-09-29",
        "outcomes": {"pointer": {"source_id": "outcomes.intraday_finals"}, "overall": {
            "close_r": {"value": -0.0511, "n": 57}, "mfe_r": {"value": 0.7789, "n": 63}}},
        "behaviour": {"pointer": {"source_id": "review.alert_review_events"}, "reviewed": {"value": 214.0, "n": 214}},
        "names": {"m5_alerts": {"n": 63, "source_id": "outcomes.intraday_finals"}}})
    put(story / "2026-09-29.json", {"schema": "market_story_narration_v1", "session_date": "2026-09-29",
                                    "periods": {"weekly": "2026-W40"}, "narration": {
        "summary": "The market is in a bear channel, lower highs, day 2.",
        "changes": ["SPY is 0.06% above its 20-day SMA.", "Monthly rollups are incomplete."]}})
    brief = {"executive_summary": "NVDA held its AVWAP into the close.",
             "what_is_working": [{"statement": "[system] coverage note", "evidence_refs": []},
                                 {"statement": "Higher low on D1 above the anchor.", "evidence_refs": ["scan.tier_list"]}],
             "risk_notes": [{"statement": "Peer AVGO reports tomorrow.", "evidence_refs": ["earnings.calendar"]}]}
    (briefs / "2026" / "2026-09-28" / "ticker_briefs_manifest.jsonl").write_text(json.dumps({
        "schema": "ai_ticker_brief_manifest_v2", "session_date": "2026-09-28", "symbol": "TSLA", "status": "briefed",
        "result": {"summary": {"executive_summary": "TSLA faded its gap.", "risk_notes": []}}}) + "\n",
        encoding="utf-8")
    (briefs / "2026" / "2026-09-29" / "ticker_briefs_manifest.jsonl").write_text("\n".join(json.dumps(row) for row in (
        {"schema": "ai_ticker_brief_manifest_v2", "session_date": "2026-09-29", "symbol": "NVDA", "status": "briefed",
         "result": {"summary": brief}},
        {"schema": "ai_ticker_brief_manifest_v2", "session_date": "2026-09-29", "symbol": "TSLA",
         "status": "membership_only"},
    )) + "\n", encoding="utf-8")
    return NightPaths(digests=digests, briefs=briefs, day_review=review, ideas=ideas, ideas_state=state, story=story)


def fixture() -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        return build("all", now=FIXTURE_NOW, paths=write_fixture_world(tmp))
