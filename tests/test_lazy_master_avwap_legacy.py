"""P2-11e: desk boot no longer imports master_avwap_lib.legacy.

Each check runs in a fresh interpreter (scratch stores), because this process
has long since loaded every module.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"


def _run(code: str) -> str:
    scratch = Path(tempfile.mkdtemp(prefix="lazy-legacy-"))
    env = dict(os.environ)
    env.update(
        {
            "TRADINGBOTV3_DATA_DIR": str(scratch / "data"),
            "LOCALAPPDATA": str(scratch / "local"),
            "TRADINGBOT_DIAGNOSTICS_DIR": str(scratch / "diag"),
            "QT_QPA_PLATFORM": "offscreen",
        }
    )
    prelude = f"import sys\nsys.path.insert(0, {str(SCRIPTS_DIR)!r})\n"
    result = subprocess.run(
        [sys.executable, "-c", prelude + code],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


def test_importing_the_desk_does_not_load_legacy_but_still_configures_logging():
    out = _run(
        "import logging\n"
        "import ui.app\n"
        "print('legacy', 'master_avwap_lib.legacy' in sys.modules)\n"
        "root = logging.getLogger()\n"
        "print('level', logging.getLevelName(root.level))\n"
        "print('files', sorted(type(h).__name__ for h in root.handlers))\n"
    )
    assert "legacy False" in out
    assert "level INFO" in out
    assert "SafeRotatingFileHandler" in out


def test_the_package_still_resolves_legacy_names_on_first_use():
    out = _run(
        "import master_avwap_lib\n"
        "print('before', 'master_avwap_lib.legacy' in sys.modules)\n"
        "from master_avwap_lib import legacy\n"
        "print('same', master_avwap_lib.load_tickers is legacy.load_tickers)\n"
        "print('logging', legacy.configure_logging is master_avwap_lib.app_logging.configure_logging)\n"
    )
    assert "before False" in out
    assert "same True" in out
    assert "logging True" in out
