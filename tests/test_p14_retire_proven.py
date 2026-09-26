"""P14: retire PROVEN, raise the M5 bar (trader 2026-09-26, "retire PROVEN").

The `[X-TIER] PROVEN` stamp is gone; the Alert Center tier-gate bypass and the
first-30 exemption follow the setup grade: A and up while any day-trade cell has
it, B and up while none does (F1). The text change itself is pinned on 280 real
alerts in `test_s9_p14_tier_golden.py`.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
for _path in (SCRIPTS_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import load_fixture_contract  # noqa: E402
from test_p8_p9_alert_filter import _make_panel, env  # noqa: E402,F401

ET = ZoneInfo("America/New_York")


def _lookup(grades: dict) -> dict:
    import setup_grades

    return setup_grades.daytrade_lookup(
        {
            "daytrade": [
                {"key": setup_grades.daytrade_key(kind, "LONG"), "bounce_type": kind, "side": "LONG", "grade": grade}
                for kind, grade in grades.items()
            ]
        }
    )


# --------------------------------------------------------------------------- the bar
def test_the_bar_is_a_while_any_a_exists_else_b():
    import alert_show_filter as f
    import setup_grades as g

    assert f.bypass_grades({}) == frozenset()
    assert f.bypass_grades(None) == frozenset()
    assert f.bypass_grades(_lookup({"x": "B", "y": "C"})) == {g.PROVEN, g.A, g.B}
    assert f.bypass_grades(_lookup({"x": "A", "y": "B"})) == {g.PROVEN, g.A}
    assert f.bypass_grades(_lookup({"x": "PROVEN", "y": "B"})) == {g.PROVEN, g.A}


def test_todays_grades_have_no_a_so_the_bar_is_b_and_one_cell_clears_it():
    """Scratch read of the live grades (2026-09-25), frozen in the golden."""
    import alert_show_filter as f
    import setup_grades as g

    golden = load_fixture_contract("s9_p14_tier_golden_v1")
    lookup = g.daytrade_lookup({"daytrade": golden["raw"]["daytrade_grades"]})
    grades = sorted((cell["grade"], key) for key, cell in lookup.items())
    assert not [key for grade, key in grades if grade in (g.PROVEN, g.A)]
    assert [key for grade, key in grades if grade == g.B] == ["regime_pause_rs|LONG"]
    assert f.bypass_grades(lookup) == {g.PROVEN, g.A, g.B}


def test_first30_exempts_the_bypass_grades_only():
    import alert_show_filter as f
    import setup_grades as g

    when = datetime(2026, 9, 28, 9, 40, tzinfo=ET)
    common = dict(best=None, symbol="AAA", side="LONG", privileged=False, first30=True, when=when)
    # Default (no grades read yet by the caller): A and up.
    assert f.hide_reason(f.ALL, grade=g.A, **common) == ""
    assert f.hide_reason(f.ALL, grade=g.B, **common) == f.REASON_FIRST30
    # No A anywhere: B stands in.
    b_bar = frozenset({g.PROVEN, g.A, g.B})
    assert f.hide_reason(f.ALL, grade=g.B, bypass=b_bar, **common) == ""
    assert f.hide_reason(f.ALL, grade=g.C, bypass=b_bar, **common) == f.REASON_FIRST30
    assert f.hides(f.ALL, grade=g.B, bypass=b_bar, **common) is False


# --------------------------------------------------------------------------- the grade text
def test_grade_text_reads_the_best_graded_type():
    from bounce_bot_lib import learning

    lookup = _lookup({"regime_pause_rs": "B", "vwap": "C"})
    assert learning.daytrade_grade_text(lookup, direction="long", bounce_types=["regime_pause_rs", "vwap"]) == "1:1 B"
    assert learning.daytrade_grade_text(lookup, direction="long", bounce_types=["other"]) == "1:1 NEW"
    assert learning.daytrade_grade_text({}, direction="long", bounce_types=["vwap"]) == "unknown"
    lookup["vwap|LONG"] = dict(lookup["vwap|LONG"], grade_2r="D")
    assert learning.daytrade_grade_text(lookup, direction="long", bounce_types=["vwap"]) == "1:1 C · 2R D"


def test_grade_lookup_is_read_from_the_published_file(tmp_path):
    import json

    from bounce_bot_lib import learning

    path = tmp_path / "setup_grades_latest.json"
    assert learning.load_daytrade_grade_lookup(path) is None
    path.write_text(
        json.dumps({"daytrade": [{"key": "vwap|LONG", "bounce_type": "vwap", "side": "LONG", "grade": "B"}]}),
        encoding="utf-8",
    )
    assert learning.load_daytrade_grade_lookup(path)["vwap|LONG"]["grade"] == "B"


def test_the_alert_path_prints_the_published_grade(monkeypatch):
    from bounce_bot_lib import learning, legacy

    monkeypatch.setattr(learning, "load_daytrade_grade_lookup", lambda path=None: _lookup({"vwap": "B"}))
    text = legacy.BounceBot._daytrade_grade_suffix(legacy.BounceBot, "long", {"vwap": 1.0})
    assert text == "1:1 B"


# --------------------------------------------------------------------------- the panel
def _alert(symbol, kind, tier):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="11:31:00",
        symbol=symbol,
        side="LONG",
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text=f"[{tier}-TIER] {symbol}: Bounce confirmed",
        payload={"feedback": {"bounce_types": kind}},
        received_at=datetime(2026, 9, 28, 11, 31, tzinfo=ET),
    )


@pytest.fixture
def s_only_panel(env, tmp_path, monkeypatch):  # noqa: F811
    import alert_show_filter

    made = _make_panel(tmp_path, monkeypatch)
    monkeypatch.setattr(made, "_min_tier_mode", lambda: "S")
    made.show_filter_input.setCurrentIndex(made.show_filter_input.findData(alert_show_filter.ALL))
    made.first30_input.setChecked(False)
    yield made
    made.deleteLater()


def _grades_payload(grades: dict) -> dict:
    return {"daytrade": list(_lookup(grades).values())}


def _feed_symbols(panel) -> set[str]:
    return {key[0] for key in panel._feed_row_registry()}


def test_a_b_graded_row_passes_the_s_only_gate_while_no_a_exists(s_only_panel):
    s_only_panel.set_setup_grades(_grades_payload({"beetype": "B", "ceetype": "C"}))
    s_only_panel.add_alert(_alert("BEE", "beetype", "C"))
    s_only_panel.add_alert(_alert("CEE", "ceetype", "C"))
    stamped = _alert("OLD", "ceetype", "C")
    stamped.raw_text = "[C-TIER] PROVEN OLD: Bounce confirmed"
    s_only_panel.add_alert(stamped)
    assert _feed_symbols(s_only_panel) == {"BEE"}  # the old stamp alone no longer passes


def test_once_an_a_exists_a_b_row_obeys_the_gate_again(s_only_panel):
    s_only_panel.set_setup_grades(_grades_payload({"beetype": "B"}))
    s_only_panel.add_alert(_alert("BEE", "beetype", "C"))
    assert _feed_symbols(s_only_panel) == {"BEE"}
    s_only_panel.set_setup_grades(_grades_payload({"beetype": "B", "aytype": "A"}))
    assert _feed_symbols(s_only_panel) == set()
    s_only_panel.add_alert(_alert("AAY", "aytype", "C"))
    assert _feed_symbols(s_only_panel) == {"AAY"}


def test_no_grades_loaded_means_no_bypass(s_only_panel):
    s_only_panel.add_alert(_alert("BEE", "beetype", "C"))
    s_only_panel.add_alert(_alert("ESS", "beetype", "S"))
    assert _feed_symbols(s_only_panel) == {"ESS"}
