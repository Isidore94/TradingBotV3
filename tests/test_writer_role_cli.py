"""Operator CLI for the machine-local writer role (plan.md sec 12 item 4).

The user switches the designated writer per day -- mini-PC on away days, desktop
on home days -- so the switch is a routine manual step, not a one-time install.
That makes two things worth testing beyond the role logic itself:

* the write must PRESERVE the rest of local_settings.json.  That file also holds
  API keys and broker tokens; a setter that rewrote it wholesale would turn a
  routine role switch into a credential-loss event.
* forgetting the switch must be DETECTABLE.  ``main`` exits non-zero when this
  machine may not publish, so a preflight can catch it before the session rather
  than the user discovering it as a missing phone report hours later.
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import project_paths  # noqa: E402
import writer_role  # noqa: E402


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Redirect machine-local settings and clear any ambient role env vars."""
    settings_dir = tmp_path / "local_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_file = settings_dir / "local_settings.json"
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_DIR", settings_dir, raising=False)
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_FILE", settings_file, raising=False)
    for key in writer_role.ENV_WRITER_KEYS + writer_role.ENV_ROLE_KEYS:
        monkeypatch.delenv(key, raising=False)
    return settings_file


def _write(settings_file: Path, payload: dict) -> None:
    settings_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_designate_self_makes_this_machine_the_writer(settings, capsys):
    exit_code = writer_role.main(["--designate-self"])

    assert exit_code == 0
    resolved = writer_role.resolve_writer_role()
    assert resolved.role == writer_role.ROLE_DESIGNATED
    assert resolved.may_publish is True
    assert resolved.designated_writer == writer_role.local_machine_name()
    assert "designated_writer" in capsys.readouterr().out


def test_secondary_names_the_other_machine_and_refuses_to_publish(settings, capsys):
    exit_code = writer_role.main(["--secondary", "MINI-PC-7"])

    resolved = writer_role.resolve_writer_role()
    assert resolved.role == writer_role.ROLE_SECONDARY
    assert resolved.may_publish is False
    assert resolved.designated_writer == "MINI-PC-7"
    # Non-zero so a preflight catches a forgotten switch before the session.
    assert exit_code == 1
    assert "MINI-PC-7" in capsys.readouterr().out


def test_status_is_the_default_and_never_writes(settings, capsys):
    _write(settings, {"gui_mode": "full"})

    exit_code = writer_role.main([])

    assert exit_code == 1  # unconfigured cannot publish
    assert json.loads(settings.read_text(encoding="utf-8")) == {"gui_mode": "full"}
    assert "unconfigured" in capsys.readouterr().out


def test_switching_roles_preserves_every_other_setting(settings):
    """The credential-safety property: a role switch must not touch secrets."""
    original = {
        "gui_mode": "full",
        "shared_data_dir": r"G:\My Drive\Trading\TradingBot",
        "market_prep_openai_api_key": "sk-proj-EXAMPLE-DO-NOT-LOSE-ME",
        "journal_questrade_refresh_token": "refresh-token-EXAMPLE",
        "qt_theme": "dark",
        "qt_alert_min_tier": "B",
    }
    _write(settings, original)

    writer_role.main(["--designate-self"])
    writer_role.main(["--secondary", "MINI-PC-7"])
    writer_role.main(["--designate-self"])

    after = json.loads(settings.read_text(encoding="utf-8"))
    for key, value in original.items():
        assert after[key] == value, f"role switch clobbered {key!r}"
    assert after["writer_role"] == writer_role.ROLE_DESIGNATED


def test_an_unknown_role_is_rejected_rather_than_guessed(settings):
    with pytest.raises(ValueError):
        writer_role.set_role("maybe_the_writer", "SOME-MACHINE")
    # Nothing was written on the rejected path.
    assert not settings.exists() or "writer_role" not in json.loads(
        settings.read_text(encoding="utf-8")
    )


def test_secondary_requires_naming_the_writer(settings):
    with pytest.raises(ValueError):
        writer_role.set_role(writer_role.ROLE_SECONDARY, "   ")


def test_set_role_returns_the_freshly_resolved_role(settings):
    """The setter re-reads rather than reporting what it hoped it wrote."""
    resolved = writer_role.set_role(writer_role.ROLE_SECONDARY, "MINI-PC-7")

    assert resolved.role == writer_role.ROLE_SECONDARY
    assert resolved.designated_writer == "MINI-PC-7"
    assert resolved.source, "the resolved role should name where it was configured"
