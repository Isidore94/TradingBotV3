"""TJ-13A item 1, the last door: the page button is night-only too.

The packet's item 1 says *"No path - scheduled, ``--force``, a page button -
starts LOCAL INFERENCE outside it"*. The first two were closed by the runner and
the window. This is the third: ``AiSummaryPanel.generate_summary`` starts a
daemon thread that calls ``request_ai_summary`` with no window check and no
market-session check at all, and the "Local (on this desk)" provider is offered
whenever ``ai_local_endpoint_url`` is set. A click at 14:00 on a Saturday put a
model load on the desk the trader was using.

Two halves, and the second matters as much as the first:

* a LOCAL run outside the night window is refused, with the window's own reason
  on the status line and NO thread started;
* a CLOUD run is untouched. The rule is about this desk's own hardware - a
  metered API call competes with nothing here - and quietly gating OpenAI
  behind the night window would be a different decision nobody made.

NO MODEL IS CALLED HERE: the local branch never reaches a request, and the
cloud branch is stopped at the credential check before any transport.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

PACIFIC = ZoneInfo("America/Los_Angeles")
SATURDAY_AFTERNOON = datetime(2026, 9, 19, 14, 0, tzinfo=PACIFIC)


def _overrides(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for source_id in ("daily.auto_report", "daily.market_prep", "daily.master_events"):
        path = tmp_path / (source_id.replace(".", "_") + ".txt")
        path.write_text(f"Evidence from {source_id}\n", encoding="utf-8")
        paths[source_id] = path
    return paths


def _panel(tmp_path):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ai_credentials import AiCredentialVault, MemoryCredentialBackend
    from ui.panels.ai_summary_panel import AiSummaryPanel

    panel = AiSummaryPanel(
        credential_vault=AiCredentialVault(MemoryCredentialBackend(), environ={}),
        source_overrides=_overrides(tmp_path),
        output_dir=tmp_path / "exports",
    )
    for scope, checkbox in panel.scope_inputs.items():
        checkbox.setChecked(scope == "daily_report")
    return panel


def test_a_local_run_by_day_is_refused_and_starts_no_thread(tmp_path, monkeypatch):
    """The click the rule exists for."""
    panel = _panel(tmp_path)

    from ui.panels import ai_summary_panel as module

    monkeypatch.setattr(module, "request_ai_summary", _must_not_be_called)
    monkeypatch.setattr(panel, "_provider", lambda: "local")
    monkeypatch.setattr(
        module.window,
        "launch_allowed",
        lambda now=None, **kwargs: (
            False,
            "outside the off-hours window (01:00-09:00 ET); now is 17:00 ET",
        ),
        raising=False,
    )

    panel.generate_summary()

    assert panel._run_thread is None, "no thread may be started"
    text = panel.status_label.text()
    assert "night" in text.lower()
    assert "outside the off-hours window" in text
    # The buttons stay usable: nothing was disabled for a run that never began.
    assert panel.generate_button.isEnabled()


def test_the_night_window_lets_a_local_run_past_at_night(tmp_path, monkeypatch):
    """The other half. Closing the door must not brick the button at night.

    It asserts the GATE and only the gate: at night the run gets PAST it, so
    the refusal sentence is not what the trader is shown. Where it stops after
    that is the panel's own business and is pinned by the next test.
    """
    panel = _panel(tmp_path)

    from ui.panels import ai_summary_panel as module

    monkeypatch.setattr(module, "request_ai_summary", _must_not_be_called)
    monkeypatch.setattr(panel, "_provider", lambda: "local")
    monkeypatch.setattr(
        module.window, "launch_allowed", lambda now=None, **kwargs: (True, "window open"),
        raising=False,
    )

    panel.generate_summary()

    assert "night only" not in panel.status_label.text().lower()
    assert panel._run_thread is None


def test_the_panels_local_path_already_stops_at_the_credential_vault(tmp_path, monkeypatch):
    """RECORDED, not repaired: the page button cannot reach a local model today.

    The packet's item 1 lists "a page button" as a third door onto local
    inference. At the code level it is not one: `AiCredentialVault` knows
    `openai` and `anthropic` only (`ai_credentials.PROVIDER_ENV_KEYS`), so
    `generate_summary`'s credential lookup raises "unsupported AI provider:
    local" and returns before any thread is created - even though the combo box
    offers "Local (on this desk)" whenever `ai_local_endpoint_url` is set.

    So the window gate added above is defence in depth rather than the repair of
    a live hole, and it is deliberately NOT accompanied by a fix to the
    credential path: making the button able to call a local model is adding a
    capability, which is a trader decision and not this packet's.

    This test exists so that whoever does enable it finds the window gate
    already in front of them, and finds this note.
    """
    panel = _panel(tmp_path)

    from ui.panels import ai_summary_panel as module

    monkeypatch.setattr(module, "request_ai_summary", _must_not_be_called)
    monkeypatch.setattr(panel, "_provider", lambda: "local")
    monkeypatch.setattr(
        module.window, "launch_allowed", lambda now=None, **kwargs: (True, "window open"),
        raising=False,
    )

    panel.generate_summary()

    assert "unsupported AI provider: local" in panel.status_label.text()
    assert panel._run_thread is None


def test_a_cloud_run_is_not_gated_by_the_night_window(tmp_path, monkeypatch):
    """Guard: the rule is about THIS DESK's hardware.

    An OpenAI call competes with nothing on this machine. It is stopped here by
    the panel's own missing-key check, which is exactly where an ungated cloud
    run stops today - and never by the window.
    """
    panel = _panel(tmp_path)

    from ui.panels import ai_summary_panel as module

    monkeypatch.setattr(panel, "_provider", lambda: "openai")
    monkeypatch.setattr(
        module.window,
        "launch_allowed",
        _window_must_not_be_consulted,
        raising=False,
    )

    panel.generate_summary()

    assert "No API key" in panel.status_label.text()


def _must_not_be_called(**kwargs):  # pragma: no cover - must never run
    raise AssertionError("a local model call must not start by day")


def _window_must_not_be_consulted(now=None, **kwargs):  # pragma: no cover
    raise AssertionError("a cloud run must not consult the off-hours window")
