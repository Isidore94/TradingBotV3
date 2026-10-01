"""Recaps pack (P15b): the trader's previous day recaps and the issues that keep coming back, read-only.

``recaps_pack(days=10|"all", section="all"|"cards"|"words"|"issues")``. Per session, newest first:

- ``recap:<date>:card:<field>`` the Day Review report card lines (``did_well``, ``missed``,
  ``your_reads``, ``congruence``, ``process``) from that session's day review pack;
- the trader's own recap words (``recap_store``): ``recap:<date>:lesson:<n>`` (keep / stop / try / mood),
  ``:rule:<tag>``, ``:rule_check:<n>``, ``:clue:<tag>``, ``:answer:<n>`` (card answers), ``:env``;
- ``recap:<date>:verdict:<n>`` the day review's "were you right" verdicts, when there are any;
- ``recap:<date>:reads`` a summary of that session's graded reads.

Then the recurrence table ``recap:issues:<key>``, computed and never guessed: a theme is an issue
only when it shows in at least ISSUE_MIN_SESSIONS sessions of the window. Themes: the report card's
shared veto reason (``missed:<reason>``), the key words of lesson "stop" lines (``stop:<words>``),
negative card answers (``answer:<kind>_<option>``), rules not kept (``rule_broken:<tag>``), repeated
clue tags (``clue:<tag>``), environment verdicts that corrected the auto label (``env:corrected``) and
reads graded wrong (``wrong_reads``). Each row carries its session count, its sessions and the first
date. The table leads the pack (``recap:asof`` first), so a tight budget keeps it.

Pure and Qt-free: files are read directly; nothing is written.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "recaps_pack"
SECTIONS = ("cards", "words", "issues")
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The trader's previous day recaps: report card lines, his own lessons, rules, rule checks, clues and "
            "environment verdicts, the day review's verdicts and graded reads, plus a computed table of issues that "
            "recur in two or more sessions (what he keeps missing). Use it for 'what are my issues', 'what have I "
            "been doing wrong', 'what did yesterday's recap say', 'am I making the same mistake'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": ["integer", "string"],
                         "description": "How many recent sessions (1-60, default 10), or 'all'."},
                "section": {"type": "string", "enum": ["all", *SECTIONS],
                            "description": "cards, words (his own recap), issues (the recurrence table), or all."},
            },
        },
    },
}

ET = ZoneInfo("America/New_York")
DEFAULT_DAYS = 10
MAX_DAYS = 60
ALL_DAYS = 120
#: A theme is an issue only when it shows in at least this many sessions.
ISSUE_MIN_SESSIONS = 2
MAX_TEXT = 300
MAX_WORDS_PER_SESSION = 12
MAX_VERDICTS = 5
CARD_KEYS = ("did_well", "missed", "your_reads", "congruence", "process")
#: Card answers that name a problem (the Walk's miss / good-pass / call cards).
NEGATIVE_OPTIONS = frozenset({"real_miss", "misread", "lucky"})
_STOPWORDS = frozenset("""
a an the and or but if then so to of in on at by for with from into onto over under about after before again
too very just not no nor only own same than that this these those there their them they he she it its i im me my
we our you your was were be been being is are am do did does doing done have has had having will would should
could can may might must out up down off more most less much many some any each every all both few other such
what which who whom when where why how
""".split())
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_SHARED_REASON = re.compile(r"share the reason ([^.;]+)", re.IGNORECASE)
LABELS = {
    "missed": "Vetoes sharing the reason", "stop": "The same 'stop' lesson", "answer": "Card answer",
    "rule_broken": "Rule not kept", "clue": "Clue you marked", "env": "You corrected the auto environment",
    "wrong_reads": "Reads graded wrong",
}


@dataclass(frozen=True)
class RecapPaths:
    """Where the pack reads; tests pass fixture paths, the app uses :func:`live_paths`. None = no store."""

    events: Path | None = None  # DAY_RECAP_EVENTS_FILE (recap_store, the trader's own words)
    records: Path | None = None  # DAY_SESSION_RECORDS_DIR (<date>.json, never pruned: the session list)
    day_review: Path | None = None  # DAY_REVIEW_DIR (sessions/<d>/pack.json, narration/<d>.json, reads/<d>.jsonl)


def live_paths() -> RecapPaths:
    import project_paths as pp

    return RecapPaths(events=Path(pp.DAY_RECAP_EVENTS_FILE), records=Path(pp.DAY_SESSION_RECORDS_DIR),
                      day_review=Path(pp.DAY_REVIEW_DIR))


# ---------------------------------------------------------------- small helpers
def _clean(value: Any, cap: int = MAX_TEXT) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= cap else text[: cap - 3].rstrip() + "..."


def _today(now: datetime | None) -> date:
    moment = now or datetime.now(timezone.utc)
    return (moment if moment.tzinfo else moment.astimezone()).astimezone(ET).date()


def slug(value: Any) -> str:
    """A citable id part: lowercase letters, digits and underscores."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").lower())).strip("_") or "other"


def keywords(text: Any, limit: int = 4) -> str:
    """The theme key of free text: its ``limit`` most frequent content words (fixed stem), in text order.

    Ties keep the earlier word, so two different long lines get two keys and the same line one key."""
    words: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    for word in re.findall(r"[a-z]+", str(text or "").lower()):
        if len(word) < 3 or word in _STOPWORDS:
            continue
        # A crude, fixed stem so "chased an extended name" and "chasing extended names" share a key.
        if len(word) > 5 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 4 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        counts[word] += 1
        if word not in words:
            words.append(word)
    top = set(sorted(words, key=lambda w: (-counts[w], words.index(w)))[:limit])
    return "_".join(word for word in words if word in top)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _dates_in(folder: Path | None, pattern: str) -> set[str]:
    if folder is None:
        return set()
    try:
        names = [path.name for path in Path(folder).glob(pattern)]
    except OSError:
        return set()
    stems = (name.split(".")[0] for name in names)
    return {stem for stem in stems if _DATE.fullmatch(stem)}


# ---------------------------------------------------------------- reading
def recap_events(paths: RecapPaths) -> list[dict[str, Any]]:
    """The trader's recap rows, superseded ones folded away (the store's own rule)."""
    if paths.events is None:
        return []
    import recap_store

    return recap_store.effective(recap_store.load_records(path=Path(paths.events)))


def sessions(paths: RecapPaths, today: date, events: Iterable[Mapping[str, Any]] = ()) -> list[str]:
    """Every session with any recap material on or before ``today``, newest first."""
    found = _dates_in(paths.records, "????-??-??.json")
    if paths.day_review is not None:
        root = Path(paths.day_review)
        found |= _dates_in(root / "sessions", "????-??-??")
        found |= _dates_in(root / "narration", "????-??-??.json")
        found |= _dates_in(root / "reads", "????-??-??.jsonl")
    found |= {str(row.get("session_date") or "")[:10] for row in events if _DATE.fullmatch(
        str(row.get("session_date") or "")[:10])}
    return sorted((day for day in found if day <= today.isoformat()), reverse=True)


def card_lines(paths: RecapPaths, day: str) -> list[dict[str, Any]]:
    """The session's report card lines (the five that describe the trader's day), as the day review stored them."""
    if paths.day_review is None:
        return []
    payload = _read_json(Path(paths.day_review) / "sessions" / day / "pack.json")
    lines = ((payload or {}).get("report_card") or {}).get("lines") if isinstance(payload, Mapping) else None
    out = []
    for line in lines or ():
        if isinstance(line, Mapping) and line.get("key") in CARD_KEYS and _clean(line.get("text")):
            out.append(dict(line))
    return sorted(out, key=lambda line: CARD_KEYS.index(line["key"]))


def verdicts(paths: RecapPaths, day: str) -> list[dict[str, Any]]:
    if paths.day_review is None:
        return []
    payload = _read_json(Path(paths.day_review) / "narration" / f"{day}.json")
    narration = payload.get("narration") if isinstance(payload, Mapping) else None
    claims = narration.get("were_you_right") if isinstance(narration, Mapping) else None
    return [dict(claim) for claim in claims or () if isinstance(claim, Mapping)]


def graded_reads(paths: RecapPaths, day: str) -> dict[str, int]:
    """Counts of the session's graded reads by verdict (right / wrong / flat / pending / unmeasured)."""
    if paths.day_review is None:
        return {}
    rows = _jsonl(Path(paths.day_review) / "reads" / f"{day}.jsonl")
    gone = {str(row.get("supersedes") or "") for row in rows if row.get("supersedes")}
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if str(row.get("grade_id") or "") in gone:
            continue
        verdict = str(row.get("verdict") or "unknown").split(":")[0].split(" ")[0] or "unknown"
        counts[verdict] += 1
    return dict(counts)


# ---------------------------------------------------------------- rows per session
def _rule_tags(events: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    return {str(row.get("id")): str(row.get("tag") or "") for row in events if row.get("kind") == "rule"}


def session_rows(paths: RecapPaths, day: str, events: list[dict[str, Any]], part: str) -> list[dict[str, Any]]:
    """One session's rows: ``cards`` (report card, verdicts, reads) or ``words`` (the trader's recap)."""
    rows: list[dict[str, Any]] = []

    def put(suffix: str, text: str, section: str, **extra: Any) -> None:
        rows.append({"id": f"recap:{day}:{suffix}", "kind": "recap", "section": section, "date": day,
                     "text": f"{day} {_clean(text)}", **extra})

    if part == "cards":
        for line in card_lines(paths, day):
            put(f"card:{line['key']}", f"report card {line['key'].replace('_', ' ')}: {line['text']}", "cards",
                field=line["key"])
        for n, claim in enumerate(verdicts(paths, day)[:MAX_VERDICTS], start=1):
            put(f"verdict:{n}", f'day review: you said "{_clean(claim.get("claim"), 160)}" -> '
                f'{_clean(claim.get("verdict"), 40) or "unknown"}', "cards", verdict=str(claim.get("verdict") or ""))
        reads = graded_reads(paths, day)
        if reads:
            parts = ", ".join(f"{k} {reads[k]}" for k in sorted(reads, key=lambda k: (-reads[k], k)))
            put("reads", f"graded reads: {parts}", "cards", counts=reads)
        return rows
    mine = [row for row in events if str(row.get("session_date") or "")[:10] == day]
    tags = _rule_tags(events)
    seen: dict[str, int] = defaultdict(int)

    def unique(base: str) -> str:
        seen[base] += 1
        return base if seen[base] == 1 else f"{base}-{seen[base]}"

    counter: dict[str, int] = defaultdict(int)
    for row in mine:
        kind = row.get("kind")
        if kind == "lesson":
            counter["lesson"] += 1
            bits = [f"{label} {row[key]!s}" for key, label in (("keep", "keep:"), ("stop", "stop:"), ("try", "try:"))
                    if _clean(row.get(key))]
            if row.get("mood") is not None:
                bits.append(f"mood {row['mood']}")
            put(f"lesson:{counter['lesson']}", "lesson: " + "; ".join(bits), "words")
        elif kind == "rule":
            put(unique(f"rule:{slug(row.get('tag') or 'untagged')}"), f"rule for next session ({row.get('tag') or 'no tag'}):"
                f" {row.get('text')}", "words", tag=str(row.get("tag") or ""))
        elif kind == "rule_check":
            counter["rule_check"] += 1
            tag = tags.get(str(row.get("rule_id") or ""), "")
            put(f"rule_check:{counter['rule_check']}", f"kept the rule{' (' + tag + ')' if tag else ''}? "
                f"{row.get('answer')}" + (f": {row.get('text')}" if _clean(row.get("text")) else ""), "words",
                answer=str(row.get("answer") or ""), tag=tag)
        elif kind == "clue":
            put(unique(f"clue:{slug(row.get('clue_tag'))}"), f"clue {row.get('clue_tag')} on {row.get('symbol')} "
                f"{row.get('timeframe')} @ {row.get('price')}" + (f": {row.get('text')}" if _clean(row.get("text")) else ""),
                "words", tag=str(row.get("clue_tag") or ""))
        elif kind == "environment_verdict":
            verdict = str(row.get("verdict") or "")
            auto = row.get("auto_label") or "unknown"
            said = f"agreed with the auto label {auto}" if verdict == "agree" else (
                f"corrected the auto label {auto} to {verdict}")
            put(unique("env"), f"environment: you {said}"
                + (f": {row.get('text')}" if _clean(row.get("text")) else ""), "words",
                auto=str(row.get("auto_label") or ""), verdict=verdict)
        elif kind == "card_answer":
            counter["answer"] += 1
            subject = row.get("subject") or {}
            name = " ".join(str(subject.get(k) or "") for k in ("symbol", "side")).strip()
            put(f"answer:{counter['answer']}", f"{row.get('card_kind')} card{' ' + name if name else ''}: "
                f"{row.get('option')}" + (f": {row.get('text')}" if _clean(row.get("text")) else ""), "words",
                card_kind=str(row.get("card_kind") or ""), option=str(row.get("option") or ""))
    return rows[:MAX_WORDS_PER_SESSION]


# ---------------------------------------------------------------- the recurrence table
def recurrence(paths: RecapPaths, days: list[str], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``[{key, label, sessions (newest first), count, first, detail}]`` for themes in >= ISSUE_MIN_SESSIONS sessions."""
    hits: dict[str, set[str]] = defaultdict(set)
    detail: dict[str, list[str]] = defaultdict(list)
    window = set(days)
    tags = _rule_tags(events)

    def hit(key: str, day: str, what: str = "") -> None:
        hits[key].add(day)
        if what and what not in detail[key]:
            detail[key].append(what)

    for day in days:
        for line in card_lines(paths, day):
            found = _SHARED_REASON.search(str(line.get("text") or "")) if line["key"] == "missed" else None
            if found:
                reason = _clean(found.group(1), 60)
                hit(f"missed:{slug(reason)}", day, reason)
        if graded_reads(paths, day).get("wrong"):
            hit("wrong_reads", day)
    for row in events:
        day = str(row.get("session_date") or "")[:10]
        if day not in window:
            continue
        kind = row.get("kind")
        if kind == "lesson" and _clean(row.get("stop")):
            # The lesson's tag when it carries one; else the key words of its stop line.
            key = slug(row.get("tag")) if _clean(row.get("tag")) else keywords(row.get("stop"))
            if key:
                hit(f"stop:{key}", day, _clean(row.get("stop"), 80))
        elif kind == "card_answer" and str(row.get("option") or "") in NEGATIVE_OPTIONS:
            hit(f"answer:{slug(row.get('card_kind'))}_{slug(row.get('option'))}", day,
                str((row.get("subject") or {}).get("symbol") or ""))
        elif kind == "rule_check" and str(row.get("answer") or "") in ("no", "partly"):
            tag = tags.get(str(row.get("rule_id") or ""), "") or "untagged"
            hit(f"rule_broken:{slug(tag)}", day, str(row.get("answer")))
        elif kind == "clue" and row.get("clue_tag"):
            hit(f"clue:{slug(row.get('clue_tag'))}", day, str(row.get("symbol") or ""))
        elif kind == "environment_verdict" and str(row.get("verdict") or "") not in ("", "agree"):
            hit("env:corrected", day, f"{row.get('auto_label') or 'unknown'} -> {row.get('verdict')}")
    out = []
    for key, found in hits.items():
        if len(found) < ISSUE_MIN_SESSIONS:
            continue
        ordered = sorted(found, reverse=True)
        out.append({"key": key, "label": LABELS.get(key.split(":")[0], key), "sessions": ordered,
                    "count": len(ordered), "first": ordered[-1], "detail": detail[key][:4]})
    out.sort(key=lambda item: (-item["count"], item["first"], item["key"]))
    return out


def issue_row(item: Mapping[str, Any]) -> dict[str, Any]:
    # A stop theme's key is stemmed words: the trader's own lines name it instead.
    name = item["key"].split(":", 1)[1].replace("_", " ") if ":" in item["key"] and not item["key"].startswith(
        "stop:") else ""
    what = f"{item['label']}{' ' + repr(name) if name else ''}"
    examples = [d for d in item["detail"] if d and d.replace(" ", "_").lower() != slug(name)]
    extra = f"; e.g. {', '.join(examples)}" if examples else ""
    return {"id": f"recap:issues:{item['key']}", "kind": "issue", "section": "issues", "key": item["key"],
            "count": item["count"], "sessions": list(item["sessions"]), "first": item["first"],
            "text": _clean(f"{what}: {item['count']} sessions ({', '.join(item['sessions'])}), first {item['first']}"
                           f"{extra}")}


def issue_rows(paths: RecapPaths | None = None, *, today: date | None = None,
               days: int = DEFAULT_DAYS) -> list[dict[str, Any]]:
    """The recurrence table over the last ``days`` sessions on or before ``today`` (for the night's issues)."""
    src = paths or live_paths()
    events = recap_events(src)
    window = sessions(src, today or _today(None), events)[:max(1, int(days))]
    return [issue_row(item) for item in recurrence(src, window, events)]


# ---------------------------------------------------------------- build
def _span(days: Any) -> int:
    if str(days).strip().lower() == "all":
        return ALL_DAYS
    try:
        return max(1, min(MAX_DAYS, int(days)))
    except (TypeError, ValueError):
        return DEFAULT_DAYS


def build(days: Any = DEFAULT_DAYS, section: str = "all", *, now: datetime | None = None,
          paths: RecapPaths | None = None) -> Pack:
    """Build the recaps pack. File reads: call it on a worker."""
    wanted = str(section or "all").strip().lower()
    if wanted != "all" and wanted not in SECTIONS:
        return make_pack(NAME, (), empty_text=f"recaps_pack sections are all, {', '.join(SECTIONS)}; not {section!r}")
    src = paths or live_paths()
    today = _today(now)
    events = recap_events(src)
    window = sessions(src, today, events)[:_span(days)]
    if not window:
        return make_pack(NAME, [{"id": "recap:none", "kind": "none", "text": "No day recaps on file yet; unknown, "
                                                                            "not 'nothing to fix'."}])
    issues = [issue_row(item) for item in recurrence(src, window, events)]
    for index, row in enumerate(issues, start=1):
        # P16: the table is already ordered by sessions count; the rank says so in the data.
        row["rank"] = index
        row["text"] = f"#{index} of {len(issues)} by sessions: {row['text']}"
    asof = {"id": "recap:asof", "kind": "asof", "date": today.isoformat(),
            "text": f"Recaps for {len(window)} session(s), {window[-1]} to {window[0]}; {len(issues)} recurring "
                    f"issue(s) (a theme counts at {ISSUE_MIN_SESSIONS}+ sessions)."}
    body: list[dict[str, Any]] = []
    if wanted in ("all", "issues"):
        body += issues or [{"id": "recap:issues:none", "kind": "none", "section": "issues",
                            "text": f"No theme repeats in {ISSUE_MIN_SESSIONS}+ of these sessions."}]
    for day in window if wanted != "issues" else ():
        for part in ("cards", "words"):
            if wanted in ("all", part):
                body += session_rows(src, day, events, part)
    return make_pack(NAME, [asof, *body])


def card_markdown(pack: Pack) -> str:
    """The /recaps card: the issues table, then each session's rows, every row with its id."""
    if not pack.rows:
        return f"**Recaps**: {pack.empty_text or 'nothing to show'}"
    lines = ["**Day recaps**", ""]
    lines += [f"- [{row['id']}] {row['text']}" for row in pack.rows]
    return "\n".join(lines)


def issues_markdown(rows: Iterable[Mapping[str, Any]]) -> str:
    """The recap half of /issues: one line per recurring theme, most sessions first."""
    rows = [row for row in rows if row.get("kind") == "issue"]
    if not rows:
        return ""
    lines = ["**Recurring in your day recaps** (computed: 2+ sessions; observations, never rules)", ""]
    lines += [f"- [{row['id']}] {row['text']}" for row in rows]
    return "\n".join(lines)


def embed_rows(pack: Pack) -> list[tuple[int, str]]:
    """``(ref_id, text)`` per session row; a changed row is a new ref (the recurrence table is not embedded)."""
    from mentor_packs.night_pack import stable_ref

    return [(stable_ref(str(row["id"]), str(row["text"])), f"[{row['id']}] {row['text']}")
            for row in pack.rows if row.get("kind") == "recap"]


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc)  # Wed 2026-09-30, 07:00 PT


def write_fixture_world(root: Path | str) -> RecapPaths:
    """Three sessions (Fri 09-25, Mon 09-28, Tue 09-29): report cards, the trader's recap rows, a verdict, reads."""
    base = Path(root)
    events, records, review = base / "day_recap_events.jsonl", base / "records", base / "day_review"
    records.mkdir(parents=True, exist_ok=True)
    days = ("2026-09-25", "2026-09-28", "2026-09-29")
    reasons = {"2026-09-25": "compressed", "2026-09-28": "compressed", "2026-09-29": "overhead"}
    for day in days:
        (records / f"{day}.json").write_text(json.dumps({"schema": "day_session_record_v1", "session_date": day}),
                                             encoding="utf-8")
        folder = review / "sessions" / day
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "pack.json").write_text(json.dumps({"schema": "day_review_pack_v1", "report_card": {"lines": [
            {"key": "did_well", "text": f"Did well: You liked 4 you did not trade ({day})."},
            {"key": "missed", "text": f"Missed: You vetoed 30. 2 were real misses; 12 share the reason "
                                      f"{reasons[day]}."},
            {"key": "how_fresh", "text": "How fresh: machine status, never a recap line."},
        ]}}), encoding="utf-8")
    (review / "narration").mkdir(parents=True, exist_ok=True)
    (review / "narration" / "2026-09-29.json").write_text(json.dumps({"narration": {"were_you_right": [
        {"claim": "SPY tests the 50sma later this week", "verdict": "wrong"}]}}), encoding="utf-8")
    (review / "reads").mkdir(parents=True, exist_ok=True)
    for day, verdicts_ in (("2026-09-28", ("wrong", "right")), ("2026-09-29", ("wrong", "pending 2026-10-05"))):
        (review / "reads" / f"{day}.jsonl").write_text("\n".join(json.dumps(
            {"grade_id": f"g-{day}-{n}", "verdict": v, "supersedes": ""}) for n, v in enumerate(verdicts_)) + "\n",
            encoding="utf-8")

    def ev(n: int, day: str, kind: str, **fields: Any) -> dict[str, Any]:
        return {"schema": "day_recap_event_v1", "id": f"rc-{n}", "kind": kind, "session_date": day,
                "recorded_at": f"{day}T17:00:00-07:00", "supersedes": fields.pop("supersedes", ""), **fields}

    rows = [
        ev(1, "2026-09-25", "rule", text="Wait for the 5-min close", tag="wait_for_confirmation"),
        ev(2, "2026-09-25", "lesson", keep="sizing", stop="Chasing extended names", **{"try": ""}, mood=2),
        ev(3, "2026-09-28", "rule_check", answer="no", rule_id="rc-1", text="jumped in early"),
        ev(4, "2026-09-28", "lesson", keep="", stop="chased an extended name", **{"try": "alerts"}, mood=None),
        ev(5, "2026-09-28", "clue", symbol="NVDA", timeframe="M5", bar_time="2026-09-28T10:05:00-04:00", price=120.5,
           clue_tag="volume_dry_up", text=""),
        ev(6, "2026-09-29", "clue", symbol="AMD", timeframe="M5", bar_time="2026-09-29T10:15:00-04:00", price=150.0,
           clue_tag="volume_dry_up", text="it dried up before the drop"),
        ev(7, "2026-09-29", "rule_check", answer="partly", rule_id="rc-1", text=""),
        ev(8, "2026-09-29", "card_answer", card_id="miss:ALL", card_kind="miss", subject={"symbol": "ALL"},
           option="real_miss", text="I passed and it ran"),
        ev(9, "2026-09-29", "environment_verdict", auto_label="neutral_chop", verdict="bearish_weak", clue_ids=[],
           text=""),
        ev(10, "2026-09-29", "lesson", keep="", stop="old words", **{"try": ""}, mood=None),
        ev(11, "2026-09-29", "lesson", keep="patience", stop="", **{"try": ""}, mood=3, supersedes="rc-10"),
    ]
    events.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return RecapPaths(events=events, records=records, day_review=review)


def fixture() -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        return build(10, now=FIXTURE_NOW, paths=write_fixture_world(tmp))
