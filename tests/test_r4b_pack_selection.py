"""R4 Part B item B1 - the newest fact pack is the newest one, not the oldest.

`_superseding` in `ai_jobs/setup_research.py` writes `<date>.json` first and
appends an ordinal on every re-run: `<date>.1.json`, `<date>.2.json`. Both
Weekend Prep readers picked `sorted(root.rglob("*.json"))[-1]`, and in ASCII
`"2026-09-01.1.json" < "2026-09-01.json"` because `.` (0x2E) sorts below `1`
(0x31). So the last name in the sorted list is the FIRST pack written for the
day - the one every re-run superseded.

Measured on the live store on 2026-09-03, with three packs for 2026-09-01:

    2026-09-01.json    gate.eligible_policy_cells = 47, no `eligible_policies`
    2026-09-01.1.json  gate.eligible_policy_cells = 33, 33 `eligible_policies`
    2026-09-01.2.json  gate.eligible_policy_cells = 33, 33 `eligible_policies`

The reader took the first, whose older shape carries no `eligible_policies` list,
so `weekend_verdict.research_line` fell to its "no cell has cleared the evidence
floor yet" branch while the current pack had 33 that had.

Offline: every pack here is written into `tmp_path`. No live store is read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _pack(eligible: int, *, new_shape: bool) -> dict:
    cells = [
        {
            "family": f"FAM{i}",
            "side": "LONG",
            "recipe_id": "r1",
            "stats": {"eligible": True, "n": 40, "clipped": {"trimmed_mean": 0.10 + i}},
        }
        for i in range(eligible)
    ]
    pack = {
        "gate": {"eligible_policy_cells": eligible, "met": bool(eligible)},
        "policies": cells + [
            {
                "family": "THIN",
                "side": "LONG",
                "recipe_id": "r9",
                "stats": {"eligible": False, "n": 3, "clipped": {"trimmed_mean": 9.0}},
            }
        ],
    }
    if new_shape:
        pack["eligible_policies"] = cells
    return pack


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / "2026" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------


def test_the_triple_resolves_to_the_last_pack_written_not_the_first(tmp_path):
    """`.json`, `.1.json`, `.2.json` - the packet's exact triple."""
    from ai_jobs.setup_research import latest_pack_path

    _write(tmp_path, "2026-09-01.json", _pack(47, new_shape=False))
    _write(tmp_path, "2026-09-01.1.json", _pack(33, new_shape=True))
    newest = _write(tmp_path, "2026-09-01.2.json", _pack(33, new_shape=True))

    assert latest_pack_path(tmp_path) == newest, (
        "sorted()[-1] picks 2026-09-01.json, which every re-run superseded"
    )


def test_a_double_digit_ordinal_still_sorts_after_a_single_digit(tmp_path):
    """`.10` is a later run than `.9`, and a string sort says otherwise."""
    from ai_jobs.setup_research import latest_pack_path

    _write(tmp_path, "2026-09-01.json", _pack(1, new_shape=True))
    _write(tmp_path, "2026-09-01.9.json", _pack(2, new_shape=True))
    newest = _write(tmp_path, "2026-09-01.10.json", _pack(3, new_shape=True))

    assert latest_pack_path(tmp_path) == newest


def test_the_newest_date_beats_a_supersession_on_an_older_one(tmp_path):
    """A re-run of yesterday never outranks today's first pack."""
    from ai_jobs.setup_research import latest_pack_path

    _write(tmp_path, "2026-09-01.3.json", _pack(9, new_shape=True))
    newest = _write(tmp_path, "2026-09-02.json", _pack(1, new_shape=True))

    assert latest_pack_path(tmp_path) == newest


def test_a_narration_is_never_mistaken_for_a_pack(tmp_path):
    from ai_jobs.setup_research import latest_pack_path

    newest = _write(tmp_path, "2026-09-01.2.json", _pack(33, new_shape=True))
    _write(tmp_path, "2026-09-01.2.narration.json", {"narration": {}})

    assert latest_pack_path(tmp_path) == newest


def test_an_empty_root_is_none_rather_than_an_exception(tmp_path):
    from ai_jobs.setup_research import latest_pack_path

    assert latest_pack_path(tmp_path) is None
    assert latest_pack_path(tmp_path / "does-not-exist") is None


# ---------------------------------------------------------------------------
# The two Weekend Prep readers
# ---------------------------------------------------------------------------


@pytest.fixture
def packed(tmp_path, monkeypatch):
    """A retros dir holding the packet's triple, bound for both readers."""
    from ai_jobs import store as ai_store

    root = tmp_path / "retros"
    _write(root / "setup_research", "2026-09-01.json", _pack(47, new_shape=False))
    _write(root / "setup_research", "2026-09-01.1.json", _pack(33, new_shape=True))
    _write(root / "setup_research", "2026-09-01.2.json", _pack(33, new_shape=True))
    monkeypatch.setattr(ai_store, "retros_dir", lambda **_kwargs: root)
    return root


def test_the_research_reader_returns_the_superseding_pack(packed):
    pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
    from ui.panels import weekend_prep_panel

    pack = weekend_prep_panel._read_research_pack()
    assert pack.get("gate", {}).get("eligible_policy_cells") == 33
    assert len(pack.get("eligible_policies") or ()) == 33


def test_the_after_like_reader_returns_the_superseding_pack(packed):
    pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
    from ui.panels import weekend_prep_panel

    pack = weekend_prep_panel._read_after_like_block()
    assert pack.get("gate", {}).get("eligible_policy_cells") == 33


# ---------------------------------------------------------------------------
# The fallback to the older shape
# ---------------------------------------------------------------------------


def test_an_old_shape_pack_reports_its_eligible_cells_instead_of_nothing():
    """A pack with `policies` and no `eligible_policies` still says something.

    Every pack before 2026-09-01 carries the older shape, and eligibility lives
    at `cell["stats"]["eligible"]`. Reading only the newer top-level list turns a
    measured 9-cell night into "no cell has cleared the evidence floor yet",
    which is a different fact and the wrong one.
    """
    from weekend_verdict import research_line

    line = research_line(_pack(9, new_shape=False))
    assert line.measured is True
    assert "9 eligible cell(s)" in line.text
    # Best cell by trimmed mean, and never the ineligible one that scores 9.0.
    assert "THIN" not in line.text


def test_a_pack_with_no_eligible_cell_still_says_so():
    from weekend_verdict import research_line

    line = research_line(_pack(0, new_shape=False))
    assert line.measured is False
    assert "no cell has cleared" in line.text
