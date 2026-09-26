r"""Long setups (p9): the scan publishes the long-setups file and changes nothing else.

The trader, 2026-09-26: "let's build a scan to find these strong names on pullbacks.
Give them their own setup." Every EXISTING scan output must stay byte-identical, so the
same synthetic leader-pullback symbol goes through the real `runner._run_master_impl`
twice, in a child process with a scratch home (the P1-4 4a idiom,
`test_setup_permutation_scan_parity.py`): once as shipped, once with the long-setups
hook replaced by a no-op. The priority row, the ai_state entry, every
`d1_features_history.csv` / `d1_features.csv` column and the text reports must match;
only the new long-setups files may differ.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
for entry in (SCRIPTS_DIR, TESTS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import test_setup_permutation_scan_parity as base  # noqa: E402

#: A rise to a 52-week high, then a 10-session pullback on light volume (a leader pullback).
_SHAPE = r'''
if BARS_SHAPE == "leader_pullback":
    peak_index = SESSION_COUNT - 11
    for index in range(peak_index + 1, SESSION_COUNT):
        step = index - peak_index
        close = float(frame.at[peak_index, "close"]) * (1.0 - 0.012 * step)
        frame.loc[index, ["open", "high", "low", "close"]] = [close + 0.3, close + 0.8, close - 0.6, close]
        frame.loc[index, "volume"] = 600_000
'''

_DISABLE = '''
# p9 long setups: only the long-setups hook is switched off (every other hook stays on).
if os.environ.get("LONG_SETUPS_HOOK") == "off" and hasattr(runner, "publish_long_setups"):
    runner.publish_long_setups = lambda *args, **kwargs: None
'''

_DUMP = r'''
def _read(path):
    path = Path(path)
    return path.read_text(encoding="utf-8-sig") if path.is_file() else None

with Path(project_paths.D1_FEATURES_FILE).open("r", encoding="utf-8-sig", newline="") as handle:
    features = list(csv.DictReader(handle))
long_path = getattr(project_paths, "LONG_SETUPS_FILE", None)
print("LONG-RESULT::" + json.dumps({
    "features": features,
    "priority_report": _read(project_paths.MASTER_AVWAP_PRIORITY_SETUPS_FILE),
    "output_report": _read(project_paths.MASTER_AVWAP_OUTPUT_FILE) if hasattr(project_paths, "MASTER_AVWAP_OUTPUT_FILE") else None,
    "long_setups": json.loads(_read(long_path)) if long_path and _read(long_path) else None,
}, default=str))
'''


def _child_source() -> str:
    source = base.CHILD_SOURCE
    anchor = 'frame["volume"] = 1_000_000\n'
    assert anchor in source
    source = source.replace(anchor, anchor + _SHAPE, 1)
    disable_anchor = "\n\ndef _sessions(count):"
    assert disable_anchor in source
    source = source.replace(disable_anchor, "\n" + _DISABLE + disable_anchor, 1)
    return source + _DUMP


def _run(tmp_path: Path, mode: str) -> dict:
    import os
    import subprocess

    scratch = tmp_path / f"long-{mode}"
    for name in ("home", "localappdata", "diag"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    child = scratch / "child.py"
    child.write_text(_child_source(), encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["LONG_SETUPS_HOOK"] = mode
    completed = subprocess.run(
        [sys.executable, str(child), str(scratch), str(SCRIPTS_DIR), "on", "300", "", "leader_pullback"],
        capture_output=True, text=True, timeout=900, env=environment, cwd=str(SCRIPTS_DIR),
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    out: dict = {}
    for marker in (base.RESULT_MARKER, "LONG-RESULT::"):
        line = next((text[len(marker):] for text in reversed(completed.stdout.splitlines())
                     if text.startswith(marker)), "")
        assert line, completed.stdout[-4000:]
        out.update(json.loads(line))
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("long-setups-parity")
    return _run(folder, "on"), _run(folder, "off")


_CLOCK = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?|\d{2}:\d{2}:\d{2}|\d{8}-\d{6}|\d{4}-\d{2}-\d{2}-\d{6}")


def _strip(value):
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items()
                if key not in base.VOLATILE_KEYS and "timestamp" not in key}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def _text(value):
    return _CLOCK.sub("<clock>", value) if isinstance(value, str) else value


def test_every_existing_scan_output_is_identical_with_and_without_the_long_setups(runs):
    on, off = runs
    assert on["priority_row"] and off["priority_row"]
    assert _strip(on["priority_row"]) == _strip(off["priority_row"])
    assert _strip(on["ai_state_entry"]) == _strip(off["ai_state_entry"])
    assert on["history"] and len(on["history"]) == len(off["history"])
    for on_row, off_row in zip(on["history"], off["history"], strict=True):
        assert list(on_row) == list(off_row), "no column added to the feature history"
        assert _strip(on_row) == _strip(off_row)
    assert [list(row) for row in on["features"]] == [list(row) for row in off["features"]]
    assert _strip(on["features"]) == _strip(off["features"])
    assert on["priority_report"] and off["priority_report"]
    assert _text(on["priority_report"]) == _text(off["priority_report"])
    assert _text(on["output_report"]) == _text(off["output_report"])


def test_the_scan_publishes_the_long_setups_file(runs):
    on, off = runs
    assert off["long_setups"] is None, "the hook off writes no long-setups file"
    published = on["long_setups"]
    assert published is not None, "the scan did not publish the long-setups file"
    rows = [row for row in published["rows"] if row["symbol"] == "PKEY"]
    assert [row["setup"] for row in rows] == ["leader_pullback"]
    row = rows[0]
    assert row["entry_limit"] < row["close"]
    assert row["stop"] < row["entry_limit"] < row["target"]
    # SPY is the same pulled-back frame here: under its 20-day, so the market is not working.
    assert row["promoted"] is False
    assert row["status"] == "waiting for the market"
