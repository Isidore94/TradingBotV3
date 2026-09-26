"""S7 golden: live alert output and the existing M5 engines, frozen before the shadow engines.

The tapes are `regime_pause_sweep_v1` (two sessions, twelve names and SPY per side)
and this fixture's own seeded tapes (three full sessions, six names; the existing
engines are silent on the short regime tapes and speak on these). The frozen output is the regime-pause sweep (flagged names, measures and the alert
lines it emits) and every event the three existing `m5_signal_engines` engines
(LRSI cross, confluence, ORB) produce on every tape, both sides. The canonical
bytes must stay identical with the S7 shadow engines on.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from datetime import date, datetime
from pathlib import Path

from conftest import _canonical_json, load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = ROOT_DIR / "tests"
for _path in (SCRIPTS_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

FIXTURE_NAME = "s7_shadow_parity_v1"
SOURCE_FIXTURE = "regime_pause_sweep_v1"


def _plain(value):
    if dataclasses.is_dataclass(value):
        return {key: _plain(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _tapes(source):
    """{side: {name: IB bars}} for every case and SPY, from the source fixture."""
    from test_regime_pause_sweep_golden import _to_ib

    out = {}
    for side in ("long", "short"):
        tapes = {name: _to_ib(rows) for name, rows in source["cases"][side].items()}
        tapes["SPY"] = _to_ib(source["spy"][side])
        out[side] = tapes
    return out


#: After the last bar of the seeded tapes (2026-08-21 12:55 local start): every bar complete.
SEEDED_NOW = datetime(2026, 8, 21, 13, 0)


def seeded_tapes(fixture=None) -> dict[str, list[dict]]:
    """This fixture's seeded tapes as chart-dict bars (naive market-local ``dt``)."""
    fixture = fixture if fixture is not None else load_fixture_contract(FIXTURE_NAME)
    return {
        name: [dict(row, dt=datetime.fromisoformat(row["dt"])) for row in rows]
        for name, rows in fixture["tapes"].items()
    }


def all_tapes() -> list[tuple[str, str, list, datetime]]:
    """(group, name, bars, now) for every tape the golden covers."""
    from test_regime_pause_sweep_golden import MEASURED_AT

    out = []
    for tape_side, tapes in _tapes(load_fixture_contract(SOURCE_FIXTURE)).items():
        out.extend((tape_side, name, bars, MEASURED_AT) for name, bars in sorted(tapes.items()))
    out.extend(("seeded", name, bars, SEEDED_NOW) for name, bars in sorted(seeded_tapes().items()))
    return out


def _engine_outputs() -> dict:
    """Every event of the three existing engines on every tape, both sides."""
    from m5_signal_engines import confluence_events, lrsi_cross_events, orb_events

    out = {}
    for group, name, bars, now in all_tapes():
        for side in ("long", "short"):
            out[f"{group}/{name}/{side}"] = {
                "lrsi": _plain(lrsi_cross_events(bars, symbol=name, side=side, now=now)),
                "confluence": _plain(confluence_events(bars, symbol=name, side=side, now=now)),
                "orb": _plain(orb_events(bars, symbol=name, side=side, now=now)),
            }
    return out


def actual_outputs() -> dict:
    """The live sweep output and the existing engines' events on the frozen tapes."""
    from test_regime_pause_sweep_golden import _actual

    source = load_fixture_contract(SOURCE_FIXTURE)
    return {"regime_pause_sweep": _plain(_actual(source)), "engines": _engine_outputs()}


def _assert_byte_identical(actual: dict) -> None:
    fixture = load_fixture_contract(FIXTURE_NAME)
    source = load_fixture_contract(SOURCE_FIXTURE)
    assert fixture["source_sha256"] == source["raw_input_sha256"], "the source tapes changed"
    expected = fixture["expected"]
    actual_bytes = _canonical_json(actual)
    assert hashlib.sha256(actual_bytes).hexdigest() == expected["canonical_sha256"]
    assert actual_bytes == _canonical_json(expected["outputs"])


def test_live_sweep_and_existing_engines_are_byte_identical_to_the_frozen_golden():
    _assert_byte_identical(actual_outputs())


def test_the_golden_is_not_empty():
    # A frozen golden of nothing proves nothing: the tapes must make the engines and sweep speak.
    outputs = load_fixture_contract(FIXTURE_NAME)["expected"]["outputs"]
    assert outputs["regime_pause_sweep"]["long"]["flagged"]
    assert any(entry["lrsi"] for entry in outputs["engines"].values())
    assert any(entry["orb"] for entry in outputs["engines"].values())


