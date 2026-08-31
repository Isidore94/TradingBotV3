"""Versioned picklists for trader annotations, loaded from bundled JSON.

Why a JSON asset and not a Python constant: a vocabulary is data the trader
and the analysis side both have to agree on months from now, and every row
written stamps the ``vocab_version`` it used. Keeping the list in a file
per version means a later vocabulary is a new file - ``veto_reasons_v2.json``
next to ``veto_reasons_v1.json`` - and rows stamped ``1`` stay interpretable
against exactly the list that produced them. Editing a shipped version in
place is the one thing that breaks that, so :func:`load_vocabulary`
validates the invariants a version file must hold rather than trusting it.

Two FAMILIES ship, and they are separate files on purpose. ``veto_reasons``
is why the trader will not take the chart in front of them at all;
``pass_reasons`` (2026-08-31) is why they liked a name for a day trade and
still passed on it - "one issue" rather than a rejection. Folding the second
into the first would have re-stamped a shared ``vocab_version`` onto every
veto cohort already accruing forward returns, for two lists that answer
different questions. A family owns its own version series.

Fail-closed on purpose. A missing or malformed vocabulary is a packaging
defect, not a runtime condition to paper over: the alternative is a capture
rail that silently writes reason codes no analysis will recognise. The panel
catches :class:`VocabularyError` and disables the veto action with the reason
shown, so the failure is visible on the desk instead of in the data.

Import-light by design (no Qt, no pandas): the capture rail calls this on
every click and offline analysis imports it headless.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

VOCABULARY_DIR = Path(__file__).resolve().parent / "vocabularies"

#: A reason code is a permanent identifier. Restricting the character set now
#: keeps codes usable as column names, filename fragments and cohort-source
#: suffixes later without escaping (see annotations.veto_cohort).
_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{2,47}$")
#: Which picklists ship. A family is a filename prefix and a series of
#: versions that only ever grows: ``<family>_v<N>.json``.
VETO_FAMILY = "veto_reasons"
PASS_FAMILY = "pass_reasons"
FAMILIES = (VETO_FAMILY, PASS_FAMILY)

_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")

#: Cohort sources are built as ``veto_<code>``; a code that collided with the
#: existing focus prefixes would land veto rows in a focus cohort.
_RESERVED_CODE_PREFIXES = ("focus_", "veto_")


class VocabularyError(RuntimeError):
    """A vocabulary file is missing, unreadable, or violates its contract."""


@dataclass(frozen=True)
class VetoReason:
    """One selectable reason. ``note_required`` gates the capture action."""

    code: str
    label: str
    hotkey: str
    note_required: bool
    hint: str

    def accepts(self, note: str) -> bool:
        """Whether ``note`` satisfies this reason's note requirement."""
        return bool(str(note or "").strip()) if self.note_required else True


@dataclass(frozen=True)
class VetoVocabulary:
    """One immutable version of one family's picklist.

    Named for the veto because it shipped first and rows, tests and analysis
    all import it by that name; ``vocabulary_id`` is what actually says which
    family a loaded list belongs to.
    """

    vocabulary_id: str
    vocab_version: int
    description: str
    reasons: tuple[VetoReason, ...]

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(reason.code for reason in self.reasons)

    def reason(self, code: str) -> VetoReason | None:
        wanted = str(code or "").strip().lower()
        for candidate in self.reasons:
            if candidate.code == wanted:
                return candidate
        return None

    def by_hotkey(self, key: str) -> VetoReason | None:
        wanted = str(key or "").strip()
        for candidate in self.reasons:
            if candidate.hotkey == wanted:
                return candidate
        return None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VocabularyError(message)


def _version_file_re(family: str):
    _require(
        bool(_FAMILY_RE.fullmatch(str(family or ""))),
        f"vocabulary family {family!r} must match {_FAMILY_RE.pattern}",
    )
    return re.compile(rf"^{re.escape(family)}_v(\d+)\.json$")


def available_versions(
    family: str = VETO_FAMILY, directory: Path | None = None
) -> tuple[int, ...]:
    """Every version of ``family`` present, ascending."""
    pattern = _version_file_re(family)
    target = Path(directory) if directory is not None else VOCABULARY_DIR
    try:
        entries = list(target.iterdir())
    except OSError:
        return ()
    versions = []
    for entry in entries:
        match = pattern.fullmatch(entry.name)
        if match and entry.is_file():
            versions.append(int(match.group(1)))
    return tuple(sorted(versions))


def available_veto_versions(directory: Path | None = None) -> tuple[int, ...]:
    """Every veto vocabulary version present, ascending."""
    return available_versions(VETO_FAMILY, directory)


def available_pass_versions(directory: Path | None = None) -> tuple[int, ...]:
    """Every day-trade pass-reason version present, ascending."""
    return available_versions(PASS_FAMILY, directory)


def _parse(
    payload: object, *, family: str, version: int, origin: str
) -> VetoVocabulary:
    _require(isinstance(payload, dict), f"{origin}: top level is not an object")
    assert isinstance(payload, dict)  # narrowed by _require
    declared_id = payload.get("vocabulary_id")
    _require(
        declared_id == family,
        f"{origin}: declares vocabulary_id {declared_id!r} but is named {family}",
    )
    declared = payload.get("vocab_version")
    _require(
        isinstance(declared, int) and declared == version,
        f"{origin}: declares vocab_version {declared!r} but is named v{version}",
    )
    raw_reasons = payload.get("reasons")
    _require(
        isinstance(raw_reasons, list) and bool(raw_reasons),
        f"{origin}: 'reasons' must be a non-empty list",
    )
    assert isinstance(raw_reasons, list)

    reasons: list[VetoReason] = []
    seen_codes: set[str] = set()
    seen_hotkeys: set[str] = set()
    for index, entry in enumerate(raw_reasons):
        where = f"{origin}: reasons[{index}]"
        _require(isinstance(entry, dict), f"{where} is not an object")
        assert isinstance(entry, dict)
        code = str(entry.get("code") or "").strip()
        _require(
            bool(_CODE_RE.fullmatch(code)),
            f"{where}: code {code!r} must match {_CODE_RE.pattern}",
        )
        _require(
            not code.startswith(_RESERVED_CODE_PREFIXES),
            f"{where}: code {code!r} uses a reserved cohort prefix",
        )
        _require(code not in seen_codes, f"{where}: duplicate code {code!r}")
        label = str(entry.get("label") or "").strip()
        _require(bool(label), f"{where}: label is required")
        hotkey = str(entry.get("hotkey") or "").strip()
        _require(len(hotkey) == 1, f"{where}: hotkey must be exactly one character")
        _require(hotkey not in seen_hotkeys, f"{where}: duplicate hotkey {hotkey!r}")
        note_required = entry.get("note_required")
        _require(
            isinstance(note_required, bool),
            f"{where}: note_required must be a boolean",
        )
        seen_codes.add(code)
        seen_hotkeys.add(hotkey)
        reasons.append(
            VetoReason(
                code=code,
                label=label,
                hotkey=hotkey,
                note_required=note_required,
                hint=str(entry.get("hint") or "").strip(),
            )
        )

    return VetoVocabulary(
        vocabulary_id=family,
        vocab_version=version,
        description=str(payload.get("description") or "").strip(),
        reasons=tuple(reasons),
    )


@lru_cache(maxsize=16)
def _load_cached(path_text: str, family: str, version: int) -> VetoVocabulary:
    path = Path(path_text)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VocabularyError(f"{path.name}: cannot be read ({exc})") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VocabularyError(f"{path.name}: is not valid JSON ({exc})") from exc
    return _parse(payload, family=family, version=version, origin=path.name)


def load_vocabulary(
    family: str = VETO_FAMILY,
    version: int | None = None,
    *,
    directory: Path | None = None,
) -> VetoVocabulary:
    """One family's picklist, defaulting to the newest version present.

    Pass ``version`` to read an older list back - that is what makes a row
    stamped ``vocab_version: 1`` interpretable after v2 ships.
    """
    target = Path(directory) if directory is not None else VOCABULARY_DIR
    versions = available_versions(family, target)
    if not versions:
        raise VocabularyError(
            f"no {family}_v*.json under {target} - the vocabulary asset is "
            "missing from this build (packaging/tradingbotv3.spec mirrors every "
            "non-.py file under scripts/ui)"
        )
    wanted = versions[-1] if version is None else int(version)
    if wanted not in versions:
        raise VocabularyError(
            f"{family} vocabulary v{wanted} not present; have {list(versions)}"
        )
    return _load_cached(str(target / f"{family}_v{wanted}.json"), family, wanted)


def load_veto_vocabulary(
    version: int | None = None,
    *,
    directory: Path | None = None,
) -> VetoVocabulary:
    """The veto picklist - why this chart is not for today at all."""
    return load_vocabulary(VETO_FAMILY, version, directory=directory)


def load_pass_vocabulary(
    version: int | None = None,
    *,
    directory: Path | None = None,
) -> VetoVocabulary:
    """The day-trade pass picklist - the ONE issue that cost a liked name.

    Trader, 2026-08-31: "many times I really like this stock for a daytrade
    but it has this one issue" and they pass. That is a different judgement
    from a veto, and it gets its own family for the reason the module
    docstring gives.
    """
    return load_vocabulary(PASS_FAMILY, version, directory=directory)


def clear_vocabulary_cache() -> None:
    """Drop the parsed-vocabulary cache (tests write temporary versions)."""
    _load_cached.cache_clear()
