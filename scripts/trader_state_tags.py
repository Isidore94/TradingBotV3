r"""The trader's own state chips - a closed, versioned picklist (TJ-7 change 1).

`plan.md` §12.4 "TJ-7" change 1: *"`state_tags` (<=2 from
`ui/annotations/vocabularies/state_tags_v1.json`: calm, focused, rushed, fomo,
tilted, bored, tired, confident - versioned like the veto vocabulary, codes
never reused)"*.

Two shipped precedents decide the shape. `ui.annotations.vocabulary` serves the
capture rail and is veto-shaped (a unique single-character `hotkey`, a boolean
`note_required`); `ai_jobs.observation_tags` keeps its own loader over a `tags`
list for exactly that reason. This list is a trader-facing picklist with no
hotkey and no note gate, so it takes the second shape and owns its own loader
rather than widening a rail the desk calls on every click.

Three rules, and each one exists because the alternative silently rewrites the
past:

1. **The declared `vocab_version` must equal the number in the FILENAME.** A
   file that lies about its version stamps rows with a version that never
   described them, so it is REFUSED rather than read.
2. **A code is a permanent identifier.** It is never renamed and never reused
   for a different meaning; a v2 ships beside v1 and a row stamped v1 stays
   interpretable against exactly the list that produced it.
3. **The cap of two lives HERE**, with the vocabulary, so the journal writer and
   the two strips read the number from one place instead of keeping three
   copies that can drift apart.

A state tag is something the desk REPORTS. Nothing here reaches a detector, a
score, an alert, a watchlist, Focus, the review queue or the review policy file,
and no outcome, R statistic or verdict may select, rank or pre-fill one - which
is why this module imports nothing that could hand it a result.

Import-light by design (no Qt, no pandas, no network): the capture path calls it
on every click and offline analysis imports it headless.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

#: One family owns its own version series, as a filename prefix.
VOCABULARY_FAMILY = "state_tags"

#: The cap `plan.md` names, and its ONE owner. Two chips is a picklist; five is
#: a form, and a form is not a two-click strip.
MAX_STATE_TAGS = 2

#: A code is a column name, a filename fragment and, one day, a context-feature
#: suffix. The house's character rule, restated rather than imported private.
_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{2,47}$")

_FILE_RE = re.compile(rf"^{re.escape(VOCABULARY_FAMILY)}_v(\d+)\.json$")


class StateTagError(RuntimeError):
    """The state-tag vocabulary is missing, unreadable, or breaks its contract."""


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
        raise StateTagError(f"the state-tag folder is unreadable: {exc}") from exc
    return sorted(found, key=lambda pair: pair[0])


def load_vocabulary(version: int | None = None, *, directory: Any = None) -> dict[str, Any]:
    """One version of the picklist, validated. Fail-closed.

    With no `version` this is the NEWEST file present; with one it is exactly
    that file, so a row stamped v1 can always be read back against the list that
    produced it. A missing, malformed or mis-versioned vocabulary is a packaging
    defect, not a runtime condition to paper over: the alternative is a strip
    that writes codes no later reader will recognise.
    """
    folder = _vocabulary_dir(directory)
    found = _shipped(folder)
    if not found:
        raise StateTagError(f"no {VOCABULARY_FAMILY}_v*.json vocabulary under {folder}")
    if version is None:
        number, path = found[-1]
    else:
        wanted = int(version)
        match = [pair for pair in found if pair[0] == wanted]
        if not match:
            raise StateTagError(
                f"no {VOCABULARY_FAMILY}_v{wanted}.json under {folder}; "
                f"the versions present are {[pair[0] for pair in found]}"
            )
        number, path = match[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StateTagError(f"the state-tag vocabulary at {path} is unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise StateTagError(f"the state-tag vocabulary at {path} is not a JSON object")
    declared = payload.get("vocab_version")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared != number:
        raise StateTagError(
            f"{path.name} declares vocab_version {declared!r}, which its own "
            f"filename says is {number}"
        )
    if str(payload.get("vocabulary_id") or "") != VOCABULARY_FAMILY:
        raise StateTagError(
            f"{path.name} declares vocabulary_id {payload.get('vocabulary_id')!r}, "
            f"not {VOCABULARY_FAMILY!r}"
        )
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in payload.get("tags") or ():
        if not isinstance(row, Mapping):
            raise StateTagError(f"{path.name} holds a tag that is not an object")
        code = str(row.get("code") or "")
        if not _CODE_RE.fullmatch(code):
            raise StateTagError(f"{path.name}: code {code!r} must match {_CODE_RE.pattern}")
        if code in seen:
            raise StateTagError(f"{path.name}: code {code!r} appears twice")
        label = str(row.get("label") or "").strip()
        if not label:
            raise StateTagError(f"{path.name}: code {code!r} has no label")
        seen.add(code)
        entries.append(
            {"code": code, "label": label, "hint": str(row.get("hint") or "").strip()}
        )
    if not entries:
        raise StateTagError(f"{path.name} holds no tags")
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
