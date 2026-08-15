"""Settings ▸ Testing Plan is a viewer, and an honest one (packet R2 follow-up).

The markdown file is the single source of truth. These tests pin the two
things that make the tab trustworthy: it renders the REAL file rather than a
copy that could drift, and a missing file says so instead of showing whatever
was loaded last. A stale runbook read as current would have the trader
checking for log lines the build no longer prints.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PLAN_PATH = ROOT_DIR / "docs" / "DESK_TESTING_PLAN.md"


def _view(path=None):
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - PySide6 is on the desk
        pytest.skip("PySide6 is not installed")
    QApplication.instance() or QApplication([])
    from ui.widgets.testing_plan_view import TestingPlanView

    return TestingPlanView(path)


# ---------------------------------------------------------------------------
# The document itself
# ---------------------------------------------------------------------------


def test_the_plan_exists_and_covers_the_owed_proofs():
    """It restates CURRENT_CHECKPOINT's proofs, so it has to mention them all."""
    assert PLAN_PATH.is_file()
    text = PLAN_PATH.read_text(encoding="utf-8")
    for topic in (
        "quiet",          # quiet-hours boot and daytime halves
        "EVENING",        # the evening stop
        "AWAY",           # away discipline
        "SPY",            # the wake alarm
        "Focus gate",     # eviction + adoption re-check
        "Strength Board", # first board session
        "20\u201340",     # the expected names-per-side
        "RVOL",           # the deferred column the trader must judge
    ):
        assert topic in text, f"the plan never mentions {topic!r}"


def test_the_plan_says_it_must_be_updated_with_the_checkpoint():
    """Without this line the two drift apart silently."""
    text = PLAN_PATH.read_text(encoding="utf-8")
    assert "CURRENT_CHECKPOINT.md" in text
    assert "same pass" in text


def test_the_plan_is_classified_in_the_docs_index():
    index = (ROOT_DIR / "docs" / "README.md").read_text(encoding="utf-8")
    assert "DESK_TESTING_PLAN.md" in index


def test_the_merge_step_tells_the_trader_to_ask_rather_than_run_git():
    """Trader instruction: the merge is 'tell the AI to do this part'."""
    text = PLAN_PATH.read_text(encoding="utf-8")
    assert "tell the AI" in text
    assert "git merge" not in text and "git checkout" not in text


# ---------------------------------------------------------------------------
# The viewer
# ---------------------------------------------------------------------------


def test_the_tab_renders_the_real_file():
    view = _view()
    assert view.path == PLAN_PATH
    rendered = view.viewer.toPlainText()
    assert "Desk testing plan" in rendered
    assert len(rendered) > 2000, "the whole document, not a stub"
    assert "last changed" in view.status_label.text()


def test_a_missing_file_is_honest_and_shows_no_stale_copy(tmp_path):
    view = _view(tmp_path / "nope.md")
    assert view.reload() is False
    rendered = view.viewer.toPlainText()
    assert "Plan file not found" in rendered
    assert "Desk testing plan" not in rendered, "no cached content may survive"
    assert "not found" in view.status_label.text()


def test_an_empty_file_is_treated_as_missing(tmp_path):
    """Rendering blank would look like a rendering bug, not a content problem."""
    empty = tmp_path / "empty.md"
    empty.write_text("   \n", encoding="utf-8")
    view = _view(empty)
    assert view.reload() is False
    assert "Plan file not found" in view.viewer.toPlainText()


def test_refresh_picks_up_a_change_without_restarting_the_desk(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("# First\n\noriginal text\n", encoding="utf-8")
    view = _view(plan)
    assert "original text" in view.viewer.toPlainText()

    plan.write_text("# Second\n\nreplaced text\n", encoding="utf-8")
    assert view.reload() is True
    rendered = view.viewer.toPlainText()
    assert "replaced text" in rendered and "original text" not in rendered


def test_a_recovered_file_replaces_the_not_found_message(tmp_path):
    plan = tmp_path / "plan.md"
    view = _view(plan)
    assert "Plan file not found" in view.viewer.toPlainText()

    plan.write_text("# Back\n\nthe plan is here again\n", encoding="utf-8")
    assert view.reload() is True
    assert "the plan is here again" in view.viewer.toPlainText()


def test_the_viewer_owns_no_timer_and_writes_nothing(tmp_path):
    """Display only: it must not be able to affect the desk it reports on."""
    from PySide6.QtCore import QTimer

    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\nbody\n", encoding="utf-8")
    before = sorted(p.name for p in tmp_path.iterdir())

    view = _view(plan)
    view.reload()

    assert not view.findChildren(QTimer), "the viewer must own no timer"
    assert sorted(p.name for p in tmp_path.iterdir()) == before, "it wrote something"


def test_the_settings_page_carries_the_tab():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - PySide6 is on the desk
        pytest.skip("PySide6 is not installed")
    QApplication.instance() or QApplication([])
    from ui.panels.settings_panel import SettingsPanel
    from ui.state import UiState

    panel = SettingsPanel(UiState())
    labels = [
        panel.settings_tabs.tabText(index)
        for index in range(panel.settings_tabs.count())
    ]
    assert "Testing Plan" in labels


# ---------------------------------------------------------------------------
# Packaging: this asset lives outside scripts/, where nothing else guards it
# ---------------------------------------------------------------------------


def test_the_spec_bundles_the_plan_explicitly():
    """The spec's package-asset sweep only mirrors files inside
    FIRST_PARTY_PACKAGES, and the drift test only walks scripts/. This asset is
    in docs/, so an explicit datas rule is the ONLY thing shipping it - and this
    test is the only thing guarding that rule."""
    spec_text = (ROOT_DIR / "packaging" / "tradingbotv3.spec").read_text(encoding="utf-8")
    assert "DESK_TESTING_PLAN.md" in spec_text
    assert 'raise SystemExit' in spec_text


def test_the_selftest_loads_the_plan():
    """So a frozen build that lost the datas rule fails loudly at selftest
    rather than quietly on the trader's desk."""
    sys.path.insert(0, str(SCRIPTS_DIR)) if str(SCRIPTS_DIR) not in sys.path else None
    import selftest

    names = [name for name, _check in selftest.ASSET_CHECKS]
    assert "docs/DESK_TESTING_PLAN.md" in names


def test_the_frozen_root_is_used_when_frozen(monkeypatch, tmp_path):
    """A frozen run has no scripts/ tree, so the source-relative walk up would
    resolve to a path that does not exist."""
    from ui.widgets import testing_plan_view

    bundled = tmp_path / "docs" / "DESK_TESTING_PLAN.md"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("# Bundled\n", encoding="utf-8")

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert testing_plan_view.resolve_testing_plan_path() == bundled

    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert testing_plan_view.resolve_testing_plan_path() == PLAN_PATH


def test_the_plan_carries_the_provocations(monkeypatch):
    """R2.1 item 7: the adversarial checks, each targeting something that has
    already gone wrong or is one mistake from going wrong."""
    text = PLAN_PATH.read_text(encoding="utf-8")
    for topic, why in (
        ("steal one of your picks", "the ownership-collision blocker"),
        ("gone stale in the queue", "the near-45-minute stale-verdict flip"),
        ("focus_auto_picks.json", "sidecar delete/corrupt/restart"),
        ("five-second mode-cache race", "the DESK->AWAY beep window"),
        ("Break the phone push", "ntfy hang/reject/backoff"),
        ("failed data chunk", "a failed Yahoo chunk"),
        ("by symbol list, not by feel", "TC2000 comparison by exported sets"),
    ):
        assert topic in text, f"the plan never covers {why}"


def test_the_tc2000_comparison_asks_for_lists_not_impressions():
    """"The character looked off" points at nothing; a list of misses points
    at a specific filter."""
    text = PLAN_PATH.read_text(encoding="utf-8")
    assert "both symbol lists" in text.lower() or "both lists" in text.lower()
    assert "which names does TC2000 have that the board" in text
