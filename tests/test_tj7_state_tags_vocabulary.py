r"""TJ-7 change 1 - the state tags are a VERSIONED, closed vocabulary. RED.

`plan.md` §12.4 "TJ-7" change 1: *"`state_tags` (<=2 from
`ui/annotations/vocabularies/state_tags_v1.json`: calm, focused, rushed, fomo,
tilted, bored, tired, confident - **versioned like the veto vocabulary, codes
never reused**). ... Never asserted by literal version in a test."*

The two shipped precedents are `ui/annotations/vocabulary.py` (veto and pass
reasons: a family is a filename prefix, the declared version must match the
FILENAME, fail-closed on anything else) and `ai_jobs/observation_tags.py` (the
same rules with a `tags` list and its own loader). TJ-7's list is a trader-
facing picklist with no hotkey and no note gate, so it takes the second shape.

WHAT IS PINNED
--------------
* the eight codes, by name and by count;
* the declared version matches its own filename, and a file that lies is
  REFUSED rather than read (a vocabulary nobody can trust silently writes codes
  no analysis will recognise);
* a code is never renamed and never reused for a different meaning ACROSS
  versions - asserted over every `state_tags_v*.json` present, so it goes on
  holding the day a v2 ships;
* the cap of two lives WITH the vocabulary, so the journal has one place to
  read it from;
* the module is import-light: the capture path calls it on every click.

NO TEST HERE ASSERTS A LITERAL `vocab_version`. The version is read off the
filename and compared with the file's own declaration.

RED FOR: `scripts/trader_state_tags.py` does not exist (ModuleNotFoundError),
and `scripts/ui/annotations/vocabularies/state_tags_v1.json` is not shipped.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj7_support as fx  # noqa: E402

VOCABULARY_DIR = ROOT_DIR / "scripts" / "ui" / "annotations" / "vocabularies"
_VERSION_RE = re.compile(r"^state_tags_v(\d+)\.json$")


def _shipped_files() -> list[Path]:
    return sorted(
        path for path in VOCABULARY_DIR.glob("state_tags_v*.json")
        if _VERSION_RE.fullmatch(path.name)
    )


def test_the_eight_state_tags_ship_as_a_versioned_file():
    """The codes `plan.md` names, all eight, and nothing else."""
    import trader_state_tags

    book = trader_state_tags.load_vocabulary()
    codes = tuple(entry["code"] for entry in book["entries"])

    assert book["vocabulary_id"] == trader_state_tags.VOCABULARY_FAMILY == "state_tags"
    assert codes == fx.STATE_TAG_CODES, "eight codes, in plan.md's order"
    assert len(set(codes)) == 8
    assert all(str(entry["label"]).strip() for entry in book["entries"])


def test_the_declared_version_matches_the_filename_it_ships_under():
    """Read off the FILE, never written here (CLAUDE.md's literal-version rule)."""
    import trader_state_tags

    files = _shipped_files()
    assert files, "no state_tags_v*.json is shipped"
    newest = max(files, key=lambda path: int(_VERSION_RE.fullmatch(path.name).group(1)))
    declared = json.loads(newest.read_text(encoding="utf-8"))["vocab_version"]

    assert declared == int(_VERSION_RE.fullmatch(newest.name).group(1))
    assert trader_state_tags.load_vocabulary()["vocab_version"] == declared


def test_a_file_that_lies_about_its_version_is_refused_not_read(tmp_path):
    """Fail-closed. The alternative is codes stamped with a version that never
    described them."""
    import trader_state_tags

    (tmp_path / "state_tags_v1.json").write_text(
        json.dumps(
            {
                "vocabulary_id": "state_tags",
                "vocab_version": 2,
                "tags": [{"code": "calm", "label": "Calm"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(trader_state_tags.StateTagError):
        trader_state_tags.load_vocabulary(directory=tmp_path)


def test_a_missing_vocabulary_is_refused_never_defaulted(tmp_path):
    import trader_state_tags

    with pytest.raises(trader_state_tags.StateTagError):
        trader_state_tags.load_vocabulary(directory=tmp_path)


def test_a_duplicate_code_inside_one_version_is_refused(tmp_path):
    import trader_state_tags

    (tmp_path / "state_tags_v1.json").write_text(
        json.dumps(
            {
                "vocabulary_id": "state_tags",
                "vocab_version": 1,
                "tags": [
                    {"code": "calm", "label": "Calm"},
                    {"code": "calm", "label": "Also calm"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(trader_state_tags.StateTagError):
        trader_state_tags.load_vocabulary(directory=tmp_path)


def test_a_code_is_never_renamed_or_reused_across_versions():
    """Rows already written carry the code. A v2 that gave `fomo` a new meaning
    would silently rewrite the past - so a code shared by two versions must
    carry the same label in both."""
    import trader_state_tags

    labels: dict[str, tuple[int, str]] = {}
    for path in _shipped_files():
        version = int(_VERSION_RE.fullmatch(path.name).group(1))
        book = trader_state_tags.load_vocabulary(version=version)
        assert book["vocab_version"] == version
        for entry in book["entries"]:
            code = entry["code"]
            label = str(entry["label"]).strip()
            if code in labels:
                first_version, first_label = labels[code]
                assert label == first_label, (
                    f"{code!r} means {label!r} in v{version} and {first_label!r} "
                    f"in v{first_version}"
                )
            else:
                labels[code] = (version, label)


def test_an_older_version_can_still_be_read_back():
    """That is the whole point of a version series: a row stamped v1 stays
    interpretable against exactly the list that produced it."""
    import trader_state_tags

    newest = trader_state_tags.load_vocabulary()
    again = trader_state_tags.load_vocabulary(version=newest["vocab_version"])
    assert again == newest

    with pytest.raises(trader_state_tags.StateTagError):
        trader_state_tags.load_vocabulary(version=999)


def test_the_cap_of_two_lives_with_the_vocabulary():
    """`plan.md`: `state_tags` (<=2 ...). ONE owner of the number."""
    import trader_state_tags

    assert trader_state_tags.MAX_STATE_TAGS == 2


def test_the_codes_helper_answers_the_shipped_list():
    import trader_state_tags

    assert tuple(trader_state_tags.codes()) == fx.STATE_TAG_CODES


def test_the_vocabulary_module_is_import_light():
    """No Qt, no pandas, no network: the capture path calls this on a click and
    offline analysis imports it headless (`ui/annotations/vocabulary.py`'s own
    rule, kept)."""
    import ast

    source = (ROOT_DIR / "scripts" / "trader_state_tags.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    heavy = {"PySide6", "PyQt5", "pandas", "numpy", "requests", "urllib", "httpx",
             "yfinance", "openai", "socket"}
    assert not {name.split(".")[0] for name in imported} & heavy, sorted(imported)
