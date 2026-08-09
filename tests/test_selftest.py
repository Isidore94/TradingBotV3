"""``launch_gui.py --selftest`` is the post-build check, so it has to be right.

Two things are worth pinning: that a healthy tree passes every check, and -
more importantly - that a BROKEN build actually fails it. A selftest that
returns 0 no matter what would be worse than none, because the trader would
stop doing the click-through it replaced.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import selftest  # noqa: E402


def test_the_selftest_passes_on_this_tree():
    stream = io.StringIO()
    assert selftest.run_selftest(stream=stream) == 0
    assert "selftest OK" in stream.getvalue()


def test_a_missing_engine_fails_the_selftest():
    """The whole point: an import the bundle dropped must be caught here."""
    stream = io.StringIO()
    code = selftest.run_selftest(
        modules=("this_module_does_not_exist",), checks=(), stream=stream
    )
    assert code == 1
    assert "selftest FAILED" in stream.getvalue()
    assert "this_module_does_not_exist" in stream.getvalue()


def test_a_missing_asset_fails_the_selftest():
    def broken() -> None:
        raise RuntimeError("theme.qss is not in the bundle")

    stream = io.StringIO()
    code = selftest.run_selftest(
        modules=(), checks=(("stylesheet", broken),), stream=stream
    )
    assert code == 1
    assert "theme.qss is not in the bundle" in stream.getvalue()


def test_every_failure_is_reported_not_just_the_first():
    """A four-minute rebuild per discovered problem is not an acceptable loop."""

    def broken() -> None:
        raise RuntimeError("nope")

    stream = io.StringIO()
    selftest.run_selftest(
        modules=("no_such_module_a", "no_such_module_b"),
        checks=(("one", broken), ("two", broken)),
        stream=stream,
    )
    output = stream.getvalue()
    assert "4 of 4 checks" in output
    for name in ("no_such_module_a", "no_such_module_b", "check one", "check two"):
        assert name in output


def test_the_engine_list_covers_the_lazily_imported_families():
    """A shrinking list is the quiet way this check stops being worth running."""
    names = set(selftest.LAZY_ENGINE_MODULES)
    for expected in (
        "master_avwap",          # the scan subprocess entry
        "bounce_bot",            # the intraday detector
        "market_prep",           # pre-session services
        "chart_snapshot",        # the chart path
        "chart_levels",          # A4 paint lines
        "ai_jobs",               # the local AI batch runner
        "ui.annotations.store",  # capture
        "ui.app",                # the desk
    ):
        assert expected in names


def test_the_selftest_reads_no_files_outside_the_installation(tmp_path, monkeypatch):
    """It must run on a machine with no home folder and no network.

    Enforced by pointing the shared store at an empty directory: every check
    still has to pass, because none of them may depend on the trader's data.
    """
    monkeypatch.setenv("TRADINGBOTV3_DATA_DIR", str(tmp_path))
    stream = io.StringIO()
    assert selftest.run_selftest(stream=stream) == 0


def test_launch_gui_routes_selftest_before_the_gui():
    """--selftest must not build a window, a crash log, or an app object."""
    source = (ROOT / "launch_gui.py").read_text(encoding="utf-8")
    body = source.split("def main()", 1)[1]
    selftest_at = body.index("--selftest")
    assert selftest_at < body.index("_enable_crash_log()")
    assert selftest_at < body.index("from ui import app")
