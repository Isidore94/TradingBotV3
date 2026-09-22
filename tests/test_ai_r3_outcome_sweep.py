"""AI-R3: the overnight sweep uses the canonical BounceBot finalizer safely."""

from __future__ import annotations

import threading
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_outcome_sweep_factory_has_no_constructor_or_broker_side_effects(monkeypatch, tmp_path):
    """The after-close job must use the existing finalizer without starting a scanner.

    A missing checkpoint is an honest empty state.  The factory is deliberately
    driven rather than a hand-built object, because an ``__init__`` call here
    would create IB clients and maintenance threads before the job could decide
    whether there is anything to finalize.
    """
    from bounce_bot_lib import legacy

    checkpoint = tmp_path / "pending_bounce_outcomes.json"
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOME_STATE_JSON", checkpoint)

    def constructor_called(*_args, **_kwargs):  # pragma: no cover - the assertion is the guard
        raise AssertionError("the outcome sweep factory called BounceBot.__init__")

    monkeypatch.setattr(legacy.BounceBot, "__init__", constructor_called)

    bot = legacy.BounceBot.for_outcome_sweep()

    assert isinstance(bot, legacy.BounceBot)
    assert bot.pending_bounce_outcomes == {}
    assert bot._finalized_outcome_ids() == {}
    assert bot._finalizing_outcome_ids() == {}
    assert isinstance(bot._pending_lock, type(threading.RLock()))
    assert not checkpoint.exists(), "constructing an empty sweep wrote a checkpoint"


def test_outcome_sweep_factory_refuses_a_bad_checkpoint_without_quarantining_or_rewriting_it(
    monkeypatch, tmp_path
):
    """A scanner may recover a checkpoint; the finalizer must not guess from one."""
    from bounce_bot_lib import legacy

    checkpoint = tmp_path / "pending_bounce_outcomes.json"
    checkpoint.write_text("{not json", encoding="utf-8")
    before = checkpoint.read_bytes()
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOME_STATE_JSON", checkpoint)
    monkeypatch.setattr(
        legacy.BounceBot,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail("the factory called BounceBot.__init__"),
    )

    with pytest.raises(ValueError):
        legacy.BounceBot.for_outcome_sweep()

    assert checkpoint.read_bytes() == before
    assert list(tmp_path.glob("*.corrupt-*.json")) == []
