"""Plan.md 22.3: strict side parsing with visible legacy coercions."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from diagnostics import ManifestRecorder, clear_active_recorder, set_active_recorder  # noqa: E402
from side_types import Side, coerce_side_legacy, parse_side  # noqa: E402


def test_parse_side_is_strict():
    assert parse_side("LONG") is Side.LONG
    assert parse_side("short") is Side.SHORT
    assert parse_side(" buy ") is Side.LONG
    assert parse_side("SELL") is Side.SHORT
    for garbage in ("", None, "LNOG", "SHRT", "0", "yes"):
        assert parse_side(garbage) is Side.UNKNOWN, garbage


def test_legacy_coercion_counts_into_run_manifest():
    recorder = ManifestRecorder(job_type="master_scan")
    set_active_recorder(recorder)
    try:
        assert coerce_side_legacy("SHORT") == "SHORT"
        assert coerce_side_legacy("LNOG") == "LONG"      # typo -> counted
        assert coerce_side_legacy("") == "LONG"          # empty -> counted separately
    finally:
        clear_active_recorder()
    assert recorder.counters.get("side_coercions_invalid") == 1
    assert recorder.counters.get("side_coercions_empty") == 1


def test_master_normalize_side_keeps_valid_behavior():
    from master_avwap_lib.legacy import normalize_side

    assert normalize_side("SHORT") == "SHORT"
    assert normalize_side("short") == "SHORT"
    assert normalize_side("LONG") == "LONG"
    assert normalize_side("") == "LONG"       # legacy default, now visible
    assert normalize_side("junk") == "LONG"   # legacy default, now visible
