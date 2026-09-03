"""What the trader keeps writing that no code says — P10 A4. Deterministic.

P10 gives every screen a note box, so the desk is about to start collecting the
one thing it has never had much of: the trader's own words about a specific chart
on a specific day. The veto vocabulary is nine codes and the pass vocabulary a
handful more; the sentences will not fit inside them, and that is the point.

**This slot adds no code and proposes none.** It lists the notes written that day
beside the vocabulary as it currently stands, so a human — or the opt-in
`trader_judgement` scope — can see which words keep coming back without one.
Reading the list is a person's job; a machine that coined a code from a frequency
count would be inventing the trader's categories for them, and a vocabulary code
is permanent and never reused.

**No model is called.** Nothing here reaches a detector, a score, an alert, a
watchlist, Focus, the review queue or `review_policy.json`. It reads two files
and writes one Markdown page.

The counting is deliberately crude — lowercase word frequency, minus a stop list
and minus every word the vocabulary already uses. A cleverer measure would be a
model's opinion wearing arithmetic's clothes, and this page exists to be argued
with, not believed.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

#: Words that carry no signal about a setup. Short, and deliberately not a real
#: stop-word corpus: a longer list starts making editorial decisions about which
#: of the trader's words matter, which is the thing this module must not do.
_STOP_WORDS = frozenset(
    """
    a an the and or but if then than so as at by for from in into of on to with
    is was are were be been being it its this that these those there here
    i me my we our you your he she they them his her their
    not no nor too very just really quite still yet also
    do does did done have has had having will would can could should may might
    up down out off over under again more most other some such only own same
    """.split()
)

#: Notes shorter than this contribute their words but are also printed whole:
#: "too extended" is the entire thought, and a frequency table loses it.
SHORT_NOTE_WORDS = 6

#: How many uncoded words the page names. A ranked list with a floor, not a
#: threshold on the count - a word used twice on a quiet day is as interesting as
#: one used ten times on a busy one, and the reader can see the counts.
TOP_WORDS = 25


def _tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z][a-z'-]{1,}", str(text or "").lower())]


def vocabulary_words() -> set[str]:
    """Every word the shipped vocabularies already use, codes and labels both.

    Both files, because a code the trader never sees (`too_extended`) and the
    label they do see ("Too extended from the anchor") are the same category, and
    a word that appears in either is already spoken for.
    """
    words: set[str] = set()
    for loader in ("load_veto_vocabulary", "load_pass_vocabulary"):
        try:
            from ui.annotations import vocabulary as vocab_module

            vocabulary = getattr(vocab_module, loader)()
        except Exception:
            continue
        for reason in getattr(vocabulary, "reasons", ()):
            words.update(_tokens(str(getattr(reason, "code", "")).replace("_", " ")))
            words.update(_tokens(getattr(reason, "label", "")))
    return words


def collect_notes(rows: list[dict[str, Any]], *, session_date: str = "") -> list[dict]:
    """Every row that carries a note, in file order.

    A note row and a click row are BOTH counted when both carry text - they are
    two statements, and the superseding note is the one the trader typed rather
    than one the desk offered them.
    """
    wanted = str(session_date or "").strip()
    notes = []
    for row in rows:
        if not hasattr(row, "get"):
            continue
        text = str(row.get("note") or "").strip()
        if not text:
            continue
        if wanted and str(row.get("session_date") or "").strip() != wanted:
            continue
        notes.append(
            {
                "symbol": str(row.get("symbol") or ""),
                "event_type": str(row.get("event_type") or ""),
                "surface": str(row.get("surface") or ""),
                "reason_code": str(row.get("reason_code") or ""),
                "note": text,
                "supersedes": str(row.get("supersedes") or ""),
            }
        )
    return notes


def uncoded_word_counts(notes: list[dict]) -> list[tuple[str, int]]:
    """Ranked words that appear in notes and in no vocabulary entry."""
    spoken_for = vocabulary_words()
    counter: Counter[str] = Counter()
    for note in notes:
        for word in _tokens(note.get("note", "")):
            if word in _STOP_WORDS or word in spoken_for:
                continue
            counter[word] += 1
    return counter.most_common(TOP_WORDS)


def render_markdown(
    notes: list[dict],
    counts: list[tuple[str, int]],
    *,
    session_date: str,
    generated_at: str,
) -> str:
    lines = [
        f"# Note vocabulary audit - {session_date}",
        "",
        f"Generated {generated_at}. Deterministic: no model was called.",
        "",
        "**This page proposes nothing.** It lists what the trader wrote beside the",
        "vocabulary that exists, so a person can see which words keep recurring",
        "without a code. No code is ever added by machine - a vocabulary code is",
        "permanent and never reused, and coining one from a frequency count would",
        "be inventing the trader's categories for them.",
        "",
    ]
    if not notes:
        lines += [
            "## No notes today",
            "",
            "Not an error and not an empty vocabulary: the trader wrote no free",
            "text on this session. A day with no notes is a real day.",
            "",
        ]
        return "\n".join(lines) + "\n"

    lines += [f"## The notes ({len(notes)})", ""]
    for note in notes:
        code = f" `{note['reason_code']}`" if note["reason_code"] else " (uncoded)"
        screen = f" · {note['surface']}" if note["surface"] else ""
        lines.append(f"- **{note['symbol']}** {note['event_type']}{code}{screen}")
        lines.append(f"  > {note['note']}")
    lines.append("")

    short = [note for note in notes if len(_tokens(note["note"])) <= SHORT_NOTE_WORDS]
    if short:
        lines += [
            f"## Whole short notes ({len(short)})",
            "",
            "Printed entire because a frequency table loses them: \"too extended\"",
            "is not two words, it is the whole thought.",
            "",
        ]
        for note in short:
            lines.append(f"- {note['symbol']}: {note['note']}")
        lines.append("")

    lines += ["## Words with no code", ""]
    if not counts:
        lines += [
            "Every word the trader used today is already in a vocabulary. That is",
            "a result, not a blank.",
            "",
        ]
    else:
        lines += ["| word | times |", "|---|---|"]
        lines += [f"| {word} | {count} |" for word, count in counts]
        lines.append("")
    return "\n".join(lines) + "\n"


def run_note_vocabulary_audit(
    *,
    session_date: str = "",
    now: datetime | None = None,
    annotations_path: Path | None = None,
    output_dir: Path | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """The nightly slot. Always writes a page, even on a day with no notes.

    A missing page and a page saying "no notes" are different facts, and the
    second is the one that is true.
    """
    from project_paths import REPORTS_DIR, TRADER_ANNOTATIONS_FILE
    from ui.annotations.store import load_annotations

    moment = now or datetime.now()
    stamp = str(session_date or "").strip() or moment.date().isoformat()
    source = Path(annotations_path) if annotations_path else TRADER_ANNOTATIONS_FILE
    target_dir = Path(output_dir) if output_dir else Path(REPORTS_DIR)

    try:
        rows = load_annotations(source)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "model": "", "reason": f"annotations unreadable: {exc}", "outputs": []}

    notes = collect_notes(rows, session_date=stamp)
    counts = uncoded_word_counts(notes)
    text = render_markdown(
        notes,
        counts,
        session_date=stamp,
        generated_at=moment.isoformat(timespec="seconds"),
    )
    target = target_dir / f"note_vocabulary_audit_{stamp}.md"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        import os

        os.replace(tmp, target)
    except OSError as exc:
        return {"status": "failed", "model": "", "reason": f"could not write the audit: {exc}", "outputs": []}

    return {
        "status": "ok",
        "model": "",
        "reason": f"{len(notes)} note(s); {len(counts)} word(s) with no code",
        "outputs": [str(target)],
    }
