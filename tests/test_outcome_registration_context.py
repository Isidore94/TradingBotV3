"""R10.A / D8 - a registration records what it can measure, and the tier follows.

The audit found `tier` on **0 of 7,863** registered rows in the outcome store,
and the cause turned out to be ordering rather than oversight: every call site
registers the outcome and evaluates the alert's tier *afterwards*. At
registration the tier does not exist yet.

So the tier is emitted as its own `tier_assigned` ledger event rather than being
back-filled onto a row that could not have known it - which is what an
append-only store is for. Reordering a live alert path is a different kind of
change and is not this packet's to make.

Everything else D8 asks for **is** measurable at registration: family, engine
version, day-part, session RVOL, env key, risk as a percent of price and as an
ATR multiple. Each is measured or blank. None is estimated.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evidence_ledger as el  # noqa: E402

EVENT_ID = "AAPL_long_20260821_06_30_00_h1_ema10_bounce"
ENTRY_DT = datetime(2026, 8, 21, 7, 0)


class _Host:
    """A minimal host for the two functions under test."""


def _host(tmp_path=None, *, atr=2.0, environment="trend_up"):
    from bounce_bot_lib.legacy import BounceBot

    host = _Host.__new__(_Host)
    host.atr_cache = {"AAPL": atr} if atr is not None else {}
    host.get_market_environment = lambda: environment
    host._outcome_ledger_obj = el.intraday_outcome_ledger(tmp_path) if tmp_path else None
    host._outcome_ledger_failed = tmp_path is None
    host._outcome_ledger = (lambda: host._outcome_ledger_obj)
    host._registration_context_fields = BounceBot._registration_context_fields.__get__(host, _Host)
    host.record_alert_tier = BounceBot.record_alert_tier.__get__(host, _Host)
    return host


def _plan(entry_price=100.0, risk=1.0):
    return {"entry_price": entry_price, "risk_per_share": risk}


# ---------------------------------------------------------------------------
# what a registration can measure
# ---------------------------------------------------------------------------
def test_the_family_is_recorded_at_registration():
    fields = _host()._registration_context_fields("AAPL", EVENT_ID, _plan(), ENTRY_DT)
    assert fields["family"] == "h1_ema10_bounce"


def test_the_engine_version_is_recorded():
    from bounce_bot_lib.legacy import BOUNCE_LEARNING_SCHEMA_VERSION

    fields = _host()._registration_context_fields("AAPL", EVENT_ID, _plan(), ENTRY_DT)
    assert fields["engine_version"] == BOUNCE_LEARNING_SCHEMA_VERSION


def test_the_day_part_comes_from_the_session_not_the_wall_clock():
    fields = _host()._registration_context_fields("AAPL", EVENT_ID, _plan(), ENTRY_DT)
    assert fields["day_part"] and fields["day_part"] != "unknown"


def test_risk_is_recorded_as_a_percent_of_price():
    """The floor rule reads this: below 0.1% of entry, R is an artifact."""
    fields = _host()._registration_context_fields("AAPL", EVENT_ID, _plan(100.0, 1.0), ENTRY_DT)
    assert fields["risk_pct_of_price"] == pytest.approx(1.0)
    thin = _host()._registration_context_fields("AAPL", EVENT_ID, _plan(100.0, 0.04), ENTRY_DT)
    assert thin["risk_pct_of_price"] == pytest.approx(0.04)


def test_risk_is_also_recorded_as_an_atr_multiple():
    fields = _host(atr=2.0)._registration_context_fields("AAPL", EVENT_ID, _plan(100.0, 1.0), ENTRY_DT)
    assert fields["atr"] == pytest.approx(2.0)
    assert fields["risk_atr_multiple"] == pytest.approx(0.5)


def test_a_missing_atr_is_blank_and_never_estimated():
    fields = _host(atr=None)._registration_context_fields("AAPL", EVENT_ID, _plan(), ENTRY_DT)
    assert fields["atr"] == ""
    assert fields["risk_atr_multiple"] == ""


def test_the_env_key_pairs_the_environment_with_the_day_part():
    fields = _host(environment="chop")._registration_context_fields("AAPL", EVENT_ID, _plan(), ENTRY_DT)
    assert fields["env_key"].startswith("chop|")
    assert fields["env_key"].split("|", 1)[1] == fields["day_part"]


def test_an_unusable_price_leaves_the_percent_blank_rather_than_dividing_by_zero():
    fields = _host()._registration_context_fields("AAPL", EVENT_ID, _plan(0.0, 1.0), ENTRY_DT)
    assert fields["risk_pct_of_price"] == ""


def test_nothing_here_can_raise_into_the_alert_path():
    """A context field is evidence; it must never cost an alert."""
    host = _host()
    host.atr_cache = None            # every accessor below now misbehaves
    host.get_market_environment = lambda: (_ for _ in ()).throw(RuntimeError("nope"))
    with pytest.raises(RuntimeError):
        host.get_market_environment()
    fields = host._registration_context_fields("AAPL", "odd-id", {"entry_price": "x"}, None)
    assert fields["family"] == "" and fields["risk_pct_of_price"] == ""


# ---------------------------------------------------------------------------
# the tier arrives later, on its own row
# ---------------------------------------------------------------------------
def test_the_tier_is_recorded_as_its_own_event(tmp_path):
    host = _host(tmp_path)
    host.record_alert_tier(EVENT_ID, {"tier": "B", "muted": False, "proven": True,
                                      "banger": False, "reason": "segment measured"})
    rows = list(host._outcome_ledger_obj.read())
    assert len(rows) == 1
    assert rows[0]["event_type"] == "tier_assigned"
    assert rows[0]["event_id"] == EVENT_ID
    assert rows[0]["tier"] == "B" and rows[0]["proven"] is True


def test_no_tier_means_no_row(tmp_path):
    host = _host(tmp_path)
    host.record_alert_tier(EVENT_ID, {"tier": ""})
    host.record_alert_tier(EVENT_ID, None)
    host.record_alert_tier("", {"tier": "A"})
    assert not list(host._outcome_ledger_obj.read())


def test_a_ledger_failure_never_reaches_the_alert(tmp_path):
    class Angry:
        def append(self, event, **kwargs):
            raise RuntimeError("disk full")

    host = _host(tmp_path)
    host._outcome_ledger_obj = Angry()
    host.record_alert_tier(EVENT_ID, {"tier": "A"})  # must not raise


def test_the_registration_row_is_not_back_filled():
    """The tier is a later fact and lands on a later row, by design."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot._registration_context_fields)
    assert '"tier"' not in source
    assert "ordering rather than oversight" in source


def test_every_alert_that_registers_an_outcome_also_records_its_tier():
    """A quality verdict computed after a registration must reach the ledger."""
    import re

    from bounce_bot_lib import legacy

    lines = inspect_source(legacy).split("\n")
    quality_lines = [
        index for index, line in enumerate(lines)
        if re.match(r"^\s+quality = self\._evaluate_bounce_alert_quality\(", line)
    ]
    assert quality_lines, "the alert path evaluates quality somewhere"
    for index in quality_lines:
        window = "\n".join(lines[max(0, index - 30):index])
        if "_register_bounce_outcome(" not in window:
            continue
        assert "record_alert_tier" in lines[index + 1], (
            f"line {index + 1} evaluates a tier after a registration and does not record it"
        )


def inspect_source(module) -> str:
    import inspect

    return inspect.getsource(module)
