"""Startup must not wait on the Windows DirectWrite font service."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ui import theme  # noqa: E402


@pytest.mark.parametrize("existing", [None, ""])
def test_windows_default_avoids_directwrite(monkeypatch, existing):
    monkeypatch.setattr(theme, "sys", SimpleNamespace(platform="win32"))
    if existing is None:
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    else:
        monkeypatch.setenv("QT_QPA_PLATFORM", existing)

    theme.configure_platform()

    import os

    assert os.environ["QT_QPA_PLATFORM"] == "windows:fontengine=gdi"


@pytest.mark.parametrize("existing", ["offscreen", "minimal", "windows", "windows:fontengine=directwrite"])
def test_explicit_platform_is_preserved(monkeypatch, existing):
    monkeypatch.setattr(theme, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("QT_QPA_PLATFORM", existing)

    theme.configure_platform()

    import os

    assert os.environ["QT_QPA_PLATFORM"] == existing


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_other_systems_keep_their_platform(monkeypatch, platform):
    monkeypatch.setattr(theme, "sys", SimpleNamespace(platform=platform))
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)

    theme.configure_platform()

    import os

    assert "QT_QPA_PLATFORM" not in os.environ


@pytest.mark.qt
def test_main_configures_platform_before_qapplication(monkeypatch):
    from ui import app

    configured = []
    monkeypatch.setattr(theme, "configure_platform", lambda: configured.append(True))
    monkeypatch.setattr(app, "install_qt_message_rate_limit", lambda: None)

    class ReachedApplication(Exception):
        pass

    class Application:
        @staticmethod
        def setAttribute(*args):
            pass

        def __init__(self, *args):
            assert configured == [True]
            raise ReachedApplication

    monkeypatch.setattr(app, "QApplication", Application)
    with pytest.raises(ReachedApplication):
        app.main([])
