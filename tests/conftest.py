"""Suite-wide isolation for machine-local runtime diagnostics.

Tests exercise live composition paths, including the D1 Greatness hook.  They
must never append synthetic events to the running application's evidence.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile


_TEST_SHARED_DIR = tempfile.mkdtemp(prefix="tradingbotv3-pytest-shared-")
_TEST_DIAGNOSTICS_DIR = tempfile.mkdtemp(prefix="tradingbotv3-pytest-diagnostics-")
_TEST_LOCAL_APPDATA = tempfile.mkdtemp(prefix="tradingbotv3-pytest-localappdata-")
os.environ["LOCALAPPDATA"] = _TEST_LOCAL_APPDATA
os.environ["TRADINGBOTV3_DATA_DIR"] = _TEST_SHARED_DIR
os.environ["TRADINGBOT_DIAGNOSTICS_DIR"] = _TEST_DIAGNOSTICS_DIR
os.environ["TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE"] = "1"
atexit.register(shutil.rmtree, _TEST_SHARED_DIR, ignore_errors=True)
atexit.register(shutil.rmtree, _TEST_DIAGNOSTICS_DIR, ignore_errors=True)
atexit.register(shutil.rmtree, _TEST_LOCAL_APPDATA, ignore_errors=True)
