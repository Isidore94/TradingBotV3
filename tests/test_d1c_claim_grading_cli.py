"""Packet D1C-B, item 3 - `python -m claimed_pick_evidence`.

The same report as text, for the trader and (later) for the AI evidence
package. The gate the lead will number reads: *"the numbers match
`python -m claimed_pick_evidence` byte for byte"*, so there is exactly ONE
renderer and the CLI prints what it returns.

The child runs in a SEPARATE interpreter against a scratch
`TRADINGBOTV3_DATA_DIR`, which is also the proof that nothing here depends on
this process's state - and the scratch-store rule from the working agreement
(2026-09-05): the env var is set BEFORE the child imports anything under
`scripts/`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tests import d1c_claim_grading_fixtures as fx  # noqa: E402

HC_SENTENCE = (
    "unmeasured: the tracker records favorite_setup / near_favorite_zone only (0 HC rows)"
)


def _store(root: Path) -> dict[str, Path]:
    """The four files where `project_paths` puts them under a data root."""
    runtime = root / "data" / "runtime"
    fx.write_csv(runtime / "like_cohort_picks.csv", fx.LIKE_PICK_COLUMNS, fx.like_picks())
    fx.write_csv(
        runtime / "like_cohort_outcomes.csv", fx.LIKE_OUTCOME_COLUMNS, fx.like_outcomes()
    )
    fx.write_csv(
        runtime / "master_avwap_tier_outcomes.csv", fx.TIER_OUTCOME_COLUMNS, fx.tier_rows()
    )
    fx.write_jsonl(root / "claimed_picks.jsonl", fx.claims())
    return {
        "picks_path": runtime / "like_cohort_picks.csv",
        "outcomes_path": runtime / "like_cohort_outcomes.csv",
        "tier_path": runtime / "master_avwap_tier_outcomes.csv",
        "claims_path": root / "claimed_picks.jsonl",
    }


def _run(root: Path, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["TRADINGBOTV3_DATA_DIR"] = str(root)
    environment["LOCALAPPDATA"] = str(tmp_path / "localappdata")
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "scripts"), str(ROOT)])
    (tmp_path / "localappdata").mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, "-m", "claimed_pick_evidence", *args],
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
        cwd=str(tmp_path),
    )


@pytest.fixture
def data_root(tmp_path) -> Path:
    root = tmp_path / "scratch_data"
    _store(root)
    return root


# ---------------------------------------------------------------------------
# It runs, and it prints the report
# ---------------------------------------------------------------------------


def test_the_cli_prints_the_leader_line_and_the_quick_like_footnote(data_root, tmp_path):
    done = _run(data_root, tmp_path, "--window", "all", "--as-of", fx.AS_OF)
    assert done.returncode == 0, done.stderr[-3000:]

    assert "no setup has n >= 30 yet (best n was 4)" in done.stdout
    assert "quick likes excluded: 1" in done.stdout


def test_the_cli_names_all_four_populations_and_the_hc_sentence(data_root, tmp_path):
    done = _run(data_root, tmp_path, "--window", "all", "--as-of", fx.AS_OF)
    assert done.returncode == 0, done.stderr[-3000:]

    for label in ("My liked trades", "FAV", "HC", "Near"):
        assert label in done.stdout, label
    assert HC_SENTENCE in done.stdout


def test_the_cli_prints_the_true_win_rates_and_not_a_pooled_one(data_root, tmp_path):
    """liked 3/4 = 75%, FAV 4/5 = 80%, Near 1/2 = 50%. A pooled 7/9 is 78%."""
    done = _run(data_root, tmp_path, "--window", "all", "--as-of", fx.AS_OF)
    assert done.returncode == 0, done.stderr[-3000:]

    for percent in (75, 80, 50):
        assert re.search(rf"\b{percent}(\.\d+)?\s*%", done.stdout), (percent, done.stdout)
    assert not re.search(r"\b78(\.\d+)?\s*%", done.stdout), "the two clocks were pooled"


def test_the_window_flag_changes_the_window_and_the_leader(data_root, tmp_path):
    lately = _run(data_root, tmp_path, "--window", "lately", "--as-of", fx.AS_OF)
    assert lately.returncode == 0, lately.stderr[-3000:]
    assert "no setup has n >= 30 yet (best n was 3)" in lately.stdout
    assert fx.LATELY_FIRST in lately.stdout, "the lately window's own dates are not printed"


# ---------------------------------------------------------------------------
# ONE renderer - the gate's "byte for byte"
# ---------------------------------------------------------------------------


def test_the_cli_prints_exactly_what_the_one_renderer_returns(data_root, tmp_path):
    import claimed_pick_evidence

    inputs = claimed_pick_evidence.load_inputs(**_store(data_root))
    comparison = claimed_pick_evidence.build_comparison(
        **inputs, as_of=date.fromisoformat(fx.AS_OF), window="all"
    )
    rendered = claimed_pick_evidence.render_text(comparison)

    done = _run(data_root, tmp_path, "--window", "all", "--as-of", fx.AS_OF)
    assert done.returncode == 0, done.stderr[-3000:]
    assert rendered.strip() and rendered.strip() in done.stdout


# ---------------------------------------------------------------------------
# Item 4: nothing moves
# ---------------------------------------------------------------------------


def test_the_cli_writes_nothing_into_the_data_root(data_root, tmp_path):
    """Item 4. FILES only: importing `project_paths` makes `logs/` and
    `output/` on any data root (measured), and an empty directory is not an
    evidence write. A new or rewritten FILE is."""

    def _files() -> dict[str, int]:
        return {
            str(path): path.stat().st_mtime_ns
            for path in data_root.rglob("*")
            if path.is_file()
        }

    before = _files()
    assert before, "the fixture wrote nothing"

    done = _run(data_root, tmp_path, "--window", "all", "--as-of", fx.AS_OF)
    assert done.returncode == 0, done.stderr[-3000:]

    assert _files() == before
