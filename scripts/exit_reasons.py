r"""Why a trade was closed - a closed, versioned picklist (TJ-9E item 2).

The trader, 2026-09-21: *"trade exits should ask for 'why did you exit, what
emotions did you have, what technicals were you observing' ideally we can just
write it out and the AI fills this stuff in overnight."* This module owns the
first of those three - the WHY - as a closed list of codes a night may choose
from and a later reader can still interpret.

It is a straight copy of :mod:`trader_state_tags` in shape, for the reason that
module already records: `ui.annotations.vocabulary` serves the capture rail and
is veto-shaped (a unique single-character ``hotkey``, a boolean
``note_required``), and this is a picklist with neither. A family owns its own
loader.

Three rules, and each one exists because the alternative silently rewrites the
past:

1. **The declared ``vocab_version`` must equal the number in the FILENAME.** A
   file that lies about its version stamps rows with a version that never
   described them, so it is REFUSED rather than read.
2. **A code is a permanent identifier.** It is never renamed and never reused
   for a different meaning; a v2 ships beside v1 and a row stamped v1 stays
   interpretable against exactly the list that produced it.
3. **Fail closed.** A missing or malformed vocabulary is a packaging defect,
   not a runtime condition to paper over: the alternative is a nightly job
   writing codes no later reader will recognise.

An exit reason is something the desk REPORTS. Nothing here reaches a detector,
a score, an alert, a watchlist, Focus, the review queue or the review policy
file, and **no outcome, R statistic or verdict may select, rank or pre-fill
one** - which is why this module imports nothing that could hand it a result.
It is a picklist loader and nothing else.

Import-light by design (no Qt, no pandas, no network): the 09:00 card reads it
to offer the list and the nightly slot reads it to bound a reply.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

#: One family owns its own version series, as a filename prefix.
VOCABULARY_FAMILY = "exit_reasons"

#: A code is a column name, a filename fragment and, one day, a context-feature
#: suffix. The house's character rule, restated rather than imported private.
_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{2,47}$")

_FILE_RE = re.compile(rf"^{re.escape(VOCABULARY_FAMILY)}_v(\d+)\.json$")


class ExitReasonError(RuntimeError):
    """The exit-reason vocabulary is missing, unreadable, or breaks its contract."""


def _vocabulary_dir(directory: Any = None) -> Path:
    if directory is not None:
        return Path(directory)
    # ONE answer to "where do vocabularies live", already resolved file-relative
    # so a frozen run finds it under `sys._MEIPASS`.
    from ui.annotations.vocabulary import VOCABULARY_DIR

    return Path(VOCABULARY_DIR)


def _shipped(folder: Path) -> list[tuple[int, Path]]:
    try:
        found = [
            (int(match.group(1)), path)
            for path in folder.glob(f"{VOCABULARY_FAMILY}_v*.json")
            if (match := _FILE_RE.fullmatch(path.name))
        ]
    except OSError as exc:
        raise ExitReasonError(f"the exit-reason folder is unreadable: {exc}") from exc
    return sorted(found, key=lambda pair: pair[0])


def load_vocabulary(version: int | None = None, *, directory: Any = None) -> dict[str, Any]:
    """One version of the picklist, validated. Fail-closed.

    With no `version` this is the NEWEST file present; with one it is exactly
    that file, so a row stamped v1 can always be read back against the list that
    produced it.
    """
    folder = _vocabulary_dir(directory)
    found = _shipped(folder)
    if not found:
        raise ExitReasonError(f"no {VOCABULARY_FAMILY}_v*.json vocabulary under {folder}")
    if version is None:
        number, path = found[-1]
    else:
        wanted = int(version)
        match = [pair for pair in found if pair[0] == wanted]
        if not match:
            raise ExitReasonError(
                f"no {VOCABULARY_FAMILY}_v{wanted}.json under {folder}; "
                f"the versions present are {[pair[0] for pair in found]}"
            )
        number, path = match[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExitReasonError(f"the exit-reason vocabulary at {path} is unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ExitReasonError(f"the exit-reason vocabulary at {path} is not a JSON object")
    declared = payload.get("vocab_version")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared != number:
        raise ExitReasonError(
            f"{path.name} declares vocab_version {declared!r}, which its own "
            f"filename says is {number}"
        )
    if str(payload.get("vocabulary_id") or "") != VOCABULARY_FAMILY:
        raise ExitReasonError(
            f"{path.name} declares vocabulary_id {payload.get('vocabulary_id')!r}, "
            f"not {VOCABULARY_FAMILY!r}"
        )
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in payload.get("reasons") or ():
        if not isinstance(row, Mapping):
            raise ExitReasonError(f"{path.name} holds a reason that is not an object")
        code = str(row.get("code") or "")
        if not _CODE_RE.fullmatch(code):
            raise ExitReasonError(f"{path.name}: code {code!r} must match {_CODE_RE.pattern}")
        if code in seen:
            raise ExitReasonError(f"{path.name}: code {code!r} appears twice")
        label = str(row.get("label") or "").strip()
        if not label:
            raise ExitReasonError(f"{path.name}: code {code!r} has no label")
        seen.add(code)
        entries.append(
            {"code": code, "label": label, "hint": str(row.get("hint") or "").strip()}
        )
    if not entries:
        raise ExitReasonError(f"{path.name} holds no reasons")
    return {
        "vocabulary_id": VOCABULARY_FAMILY,
        "vocab_version": int(number),
        "description": str(payload.get("description") or ""),
        "entries": entries,
        "path": str(path),
    }


def codes(version: int | None = None, *, directory: Any = None) -> tuple[str, ...]:
    """Every code in one version, in the file's own order."""
    book = load_vocabulary(version, directory=directory)
    return tuple(str(entry["code"]) for entry in book["entries"])


def label_for(code: str, version: int | None = None, *, directory: Any = None) -> str:
    """One code's label, or ``""`` for a code this version never held."""
    wanted = str(code or "").strip()
    for entry in load_vocabulary(version, directory=directory)["entries"]:
        if str(entry["code"]) == wanted:
            return str(entry["label"])
    return ""


__all__ = [
    "VOCABULARY_FAMILY",
    "ExitReasonError",
    "codes",
    "label_for",
    "load_vocabulary",
]
