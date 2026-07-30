"""Champion-invariance guard (plan.md sec 5 + sec 12 item 7).

The legacy SPY pause detector and the D1 wick/level trigger alerts are the
CHAMPIONS.  Both shadow challengers are invoked from *inside* those champion
code paths:

* ``scripts/bounce_bot_lib/legacy.py`` -> ``check_regime_pause_setups`` calls
  ``market_state_bridge.record_spy_shadow`` right after
  ``_detect_spy_pause_start``.
* ``scripts/bounce_bot_lib/legacy.py`` ->
  ``emit_master_avwap_intraday_trigger_flags`` calls
  ``greatness_shadow.record_d1_shadow`` before the trigger rows are built.

"Shadow evidence collection is observationally inert for the champion" has
been a prose invariant with no executable proof.  This module is that proof.
It fails if a future change lets shadow state leak into a live decision.

Four guarantees are asserted:

1. A raising SPY shadow leaves the champion's return value, pause state and
   recorded pause timestamp untouched, and the exception never propagates.
2. A raising Greatness shadow leaves the D1 trigger rows and every emitted
   alert byte-identical, and the exception never propagates.
3. Shadow engines absent (module unimportable) vs. fully enabled and really
   writing evidence to disk produce identical champion output.
4. No champion branch READS a shadow return value.  Proven two ways:
   * behaviourally - the shadow entry point is swapped for one returning a
     poison sentinel whose every interaction raises a ``BaseException``
     subclass.  ``BaseException`` is deliberate: the champion wraps the hook
     in ``except Exception``, so an ``Exception`` would be swallowed and the
     probe would prove nothing, whereas a ``BaseException`` escapes the
     instant any champion code touches the value.
   * structurally - an AST assertion over ``legacy.py`` that every shadow
     call is a bare expression statement (return value discarded) living
     inside a ``try``/``except Exception``, and that the shadow names are
     never bound to anything.  The AST check is what makes the guard
     airtight: a discarded expression statement cannot be read at all, so
     no clever runtime probe is needed to cover every branch.

Fast, deterministic, network-free: no IB connection is made (the bot objects
are built with ``__new__`` / attribute stubs, exactly as the neighbouring
``tests/test_bounce_feedback.py`` and ``tests/test_bounce_learning.py`` do).
"""

import ast
import json
import sys
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bounce_bot  # noqa: E402
import bounce_bot_lib.legacy as legacy  # noqa: E402
import greatness_shadow  # noqa: E402
import market_state_bridge  # noqa: E402

LEGACY_SOURCE_PATH = ROOT_DIR / "scripts" / "bounce_bot_lib" / "legacy.py"
SHADOW_ENTRY_POINTS = ("record_spy_shadow", "record_d1_shadow")


# ---------------------------------------------------------------------------
# Poison sentinel: any interaction other than "discard me" raises.
# ---------------------------------------------------------------------------
class ShadowValueTouched(BaseException):
    """Raised when champion code reads a shadow return value.

    Subclasses BaseException on purpose so the champion's ``except Exception``
    around the shadow hook cannot swallow the evidence.
    """


def _touched(*_args, **_kwargs):
    raise ShadowValueTouched(
        "champion code read a shadow return value (plan.md sec 5 violation)"
    )


class PoisonShadowResult:
    """Return value that explodes on every access a champion could make."""

    __getattr__ = _touched
    __bool__ = _touched
    __len__ = _touched
    __iter__ = _touched
    __contains__ = _touched
    __getitem__ = _touched
    __call__ = _touched
    __eq__ = _touched
    __ne__ = _touched
    __lt__ = _touched
    __gt__ = _touched
    __hash__ = _touched
    __repr__ = _touched
    __str__ = _touched
    __format__ = _touched
    __int__ = _touched
    __float__ = _touched
    __add__ = _touched
    __radd__ = _touched


# ---------------------------------------------------------------------------
# Fingerprinting: turn champion output into a stable comparable string.
# ---------------------------------------------------------------------------
# Wall-clock stamps written by the champion itself (HH:MM strings); they would
# flake across a minute boundary between two runs of the same scenario and say
# nothing about shadow influence.
_VOLATILE_KEYS = {"first_seen", "last_seen"}


def _normalize(value, *, key=""):
    if key in _VOLATILE_KEYS:
        return "<clock>"
    if isinstance(value, dict):
        return {
            str(name): _normalize(item, key=str(name))
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(json.dumps(_normalize(item), sort_keys=True) for item in value)
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "dt") and hasattr(value, "close"):  # IbBar
        return {
            "dt": value.dt.isoformat(),
            "open": repr(value.open),
            "high": repr(value.high),
            "low": repr(value.low),
            "close": repr(value.close),
        }
    return f"{type(value).__name__}:{value!r}"


def _fingerprint(payload):
    return json.dumps(_normalize(payload), sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Shadow-mode installers.
# ---------------------------------------------------------------------------
def _install_shadow_mode(monkeypatch, mode, module, attribute, tmp_path=None):
    """Configure how the shadow hook behaves for one champion run.

    Returns a list that records every invocation, so a run can assert the hook
    was genuinely reached rather than silently skipped.
    """
    invocations = []
    if mode == "absent":
        # The champion imports the shadow module lazily inside the try block;
        # a None entry in sys.modules makes that import raise ImportError.
        monkeypatch.setitem(sys.modules, module.__name__, None)
    elif mode == "raises":
        def _boom(*args, **kwargs):
            invocations.append((args, kwargs))
            raise RuntimeError("shadow engine exploded")

        monkeypatch.setattr(module, attribute, _boom)
    elif mode == "poison":
        def _poison(*args, **kwargs):
            invocations.append((args, kwargs))
            return PoisonShadowResult()

        monkeypatch.setattr(module, attribute, _poison)
    elif mode == "enabled":
        assert tmp_path is not None
        if module is market_state_bridge:
            log = tmp_path / "spy_state_shadow.jsonl"
            monkeypatch.setattr(market_state_bridge, "shadow_log_path", lambda: log)
            market_state_bridge.reset_shadow_dedupe()
        else:
            monkeypatch.setattr(greatness_shadow, "_diag_dir", lambda: tmp_path)
            monkeypatch.setattr(greatness_shadow, "_board", None)
    else:  # pragma: no cover - guards a typo in the test itself
        raise AssertionError(f"unknown shadow mode {mode!r}")
    return invocations


# ===========================================================================
# Champion 1 - legacy SPY pause detection.
# ===========================================================================
def _bar(dt, open_, high, low, close):
    return legacy.IbBar(dt=dt, open=open_, high=high, low=low, close=close)


SESSION_START = datetime(2026, 7, 2, 9, 30)


def _downtrend(base, *, candles, step, start, green_last=False):
    bars = []
    price = base
    for index in range(candles):
        open_ = price
        close = price + step
        if green_last and index == candles - 1:
            close = open_ + abs(step)
        bars.append(
            _bar(
                start + timedelta(minutes=5 * index),
                open_,
                max(open_, close) + 0.05,
                min(open_, close) - 0.05,
                close,
            )
        )
        price = close
    return bars


def _spy_pause_session():
    """SPY sells off all day then prints one green pause candle."""
    # A prior-session bar so `_spy_session_bars` yields a real prev_close and
    # the enabled shadow engine actually has usable input.
    prior = [_bar(datetime(2026, 7, 1, 15, 55), 100.9, 101.0, 100.8, 100.9)]
    return prior + _downtrend(
        100.0, candles=12, step=-0.08, start=SESSION_START, green_last=True
    )


def _spy_champion_stub():
    """Real champion methods on a bare stub - no IB, no GUI, no network."""

    class Stub:
        pass

    for name in (
        "_spy_session_bars",
        "_detect_spy_pause_start",
        "check_regime_pause_setups",
        "_sweep_regime_pause_bangers",
        "_regime_pause_day_alerted",
        "_regime_pause_observation_store",
        "_record_regime_pause_observation",
        "get_market_environment",
    ):
        setattr(Stub, name, getattr(legacy.BounceBot, name))
    Stub._window_change_pct = legacy.BounceBot.__dict__["_window_change_pct"]

    spy = _spy_pause_session()
    # Sells off ten times harder than SPY and keeps making lows into the pause.
    weak = _downtrend(140.0, candles=12, step=-1.6, start=SESSION_START)
    # Bounces with SPY on the pause candle: must NOT be flagged.
    bouncer = _downtrend(
        50.0, candles=12, step=-0.3, start=SESSION_START, green_last=True
    )

    stub = Stub()
    stub.market_environment = "bearish_strong"
    stub.market_environment_lock = threading.Lock()
    stub._regime_pause_state = None
    stub._regime_pause_observations = None
    stub.longs = []
    stub.shorts = ["AAOI", "BNCR"]
    stub.emitted = []
    stub.summaries = []
    stub.get_cached_5m_bars = lambda symbol: (
        spy if symbol == "SPY" else {"AAOI": weak, "BNCR": bouncer}.get(symbol, [])
    )
    stub._record_regime_pause_banger = stub.emitted.append
    stub._emit_regime_pause_summary = (
        lambda side, spy_window, hits, state: stub.summaries.append(
            (side, repr(spy_window), [hit["symbol"] for hit in hits])
        )
    )
    stub._save_regime_pause_observations = lambda: None
    return stub


def _run_spy_champion(monkeypatch, mode, tmp_path=None):
    """Run the champion once under `mode`; return its full observable output."""
    with monkeypatch.context() as patcher:
        invocations = _install_shadow_mode(
            patcher, mode, market_state_bridge, "record_spy_shadow", tmp_path
        )
        stub = _spy_champion_stub()
        flagged = stub.check_regime_pause_setups()
    if mode in {"raises", "poison"}:
        assert invocations, "the SPY shadow hook was never reached"
    return {
        "returned": flagged,
        "pause_state": stub._regime_pause_state,
        "observations": stub._regime_pause_observations,
        "recorded_bangers": stub.emitted,
        "summaries": stub.summaries,
    }


def test_spy_champion_scenario_is_a_real_exercise():
    """Guard against a vacuous suite: the fixture must actually fire."""
    result = _run_spy_champion(pytest.MonkeyPatch(), "absent")
    assert [hit["symbol"] for hit in result["returned"]] == ["AAOI"]
    assert result["pause_state"]["start_dt"] is not None
    assert result["recorded_bangers"]
    assert result["observations"]["sides"]["short"]["AAOI"]["pause_count"] == 1


def test_spy_champion_unchanged_when_shadow_raises(monkeypatch):
    baseline = _run_spy_champion(monkeypatch, "absent")
    raised = _run_spy_champion(monkeypatch, "raises")

    assert _fingerprint(raised) == _fingerprint(baseline)
    # The pause timestamp specifically - the champion's core state.
    assert raised["pause_state"]["start_dt"] == baseline["pause_state"]["start_dt"]
    assert raised["pause_state"]["side"] == baseline["pause_state"]["side"]
    assert raised["pause_state"]["date"] == baseline["pause_state"]["date"]


def test_spy_champion_never_reads_the_shadow_return_value(monkeypatch):
    baseline = _run_spy_champion(monkeypatch, "absent")
    # ShadowValueTouched is a BaseException: it escapes `except Exception`.
    poisoned = _run_spy_champion(monkeypatch, "poison")

    assert _fingerprint(poisoned) == _fingerprint(baseline)


def test_spy_champion_identical_with_shadow_enabled_and_writing(monkeypatch, tmp_path):
    baseline = _run_spy_champion(monkeypatch, "absent")
    enabled = _run_spy_champion(monkeypatch, "enabled", tmp_path=tmp_path)

    assert _fingerprint(enabled) == _fingerprint(baseline)

    # Non-vacuity: the shadow really ran and really persisted evidence.
    log = tmp_path / "spy_state_shadow.jsonl"
    assert log.exists(), "shadow engine never wrote - the comparison is vacuous"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[0]["schema"] == market_state_bridge.SHADOW_SCHEMA
    assert rows[0]["legacy_paused"] is True


# ===========================================================================
# Champion 2 - D1 master-AVWAP intraday trigger alerts.
# ===========================================================================
D1_TRIGGER_LEVELS = [
    {
        "trigger_id": "first_dev_break:UPPER_1:102.0000",
        "label": "UPPER_1",
        "level": 102.0,
        "action": "break_above",
        "event_type": "first_dev_break",
        "alert_label": "1st-dev break",
        "reason": "Armed from AVWAPE-to-UPPER_1 zone.",
        "source": "favorite_zone",
        "armed_price": 101.0,
        "setup_family": "first_dev_break",
    }
]


def _d1_frame():
    return pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-05-06 09:35:00"),
                "open": 101.1,
                "high": 101.8,
                "low": 100.9,
                "close": 101.6,
                "volume": 1000,
                "time": "20260506  09:35:00",
            },
            {
                "datetime": pd.Timestamp("2026-05-06 09:40:00"),
                "open": 101.7,
                "high": 102.4,
                "low": 101.5,
                "close": 102.3,
                "volume": 1200,
                "time": "20260506  09:40:00",
            },
        ]
    )


def _d1_champion_bot():
    bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)
    bot.master_avwap_d1_watchlist = {
        "AAPL": {
            "symbol": "AAPL",
            "side": "LONG",
            "direction": "long",
            "active_current_scan": True,
            "priority_score": 240,
            "watchlist_run_date": "2026-05-05",
            "trigger_levels": [dict(level) for level in D1_TRIGGER_LEVELS],
        }
    }
    bot.emitted_master_avwap_d1_flags = set()
    bot.gui_alerts = []
    bot.symbol_logs = []
    bot.gui_callback = lambda message, tag: bot.gui_alerts.append((message, tag))
    bot.log_symbol = lambda symbol, message: bot.symbol_logs.append((symbol, message))

    # Capture the champion's trigger ROWS, not just the alerts they produce.
    real_finder = bounce_bot.BounceBot._find_master_avwap_intraday_trigger_events
    bot.trigger_rows = []

    def _capturing_finder(symbol, today_df):
        rows = real_finder(bot, symbol, today_df)
        bot.trigger_rows.append(rows)
        return rows

    bot._find_master_avwap_intraday_trigger_events = _capturing_finder
    return bot


def _run_d1_champion(monkeypatch, mode, tmp_path=None):
    frame = _d1_frame()
    frame_before = frame.copy(deep=True)
    with monkeypatch.context() as patcher:
        invocations = _install_shadow_mode(
            patcher, mode, greatness_shadow, "record_d1_shadow", tmp_path
        )
        bot = _d1_champion_bot()
        emitted = bot.emit_master_avwap_intraday_trigger_flags("AAPL", frame)
    if mode in {"raises", "poison"}:
        assert invocations, "the Greatness shadow hook was never reached"
    # The champion's own input must survive the shadow pass untouched.
    assert frame.equals(frame_before), "shadow hook mutated the champion's bars"
    return {
        "emitted_count": emitted,
        "trigger_rows": bot.trigger_rows,
        "gui_alerts": bot.gui_alerts,
        "symbol_logs": bot.symbol_logs,
        "flag_keys": bot.emitted_master_avwap_d1_flags,
    }


def test_d1_champion_scenario_is_a_real_exercise():
    result = _run_d1_champion(pytest.MonkeyPatch(), "absent")
    assert result["emitted_count"] == 1
    assert result["trigger_rows"] and result["trigger_rows"][0]
    message, tag = result["gui_alerts"][0]
    assert "MASTER_AVWAP_D1_FLAG: AAPL" in message
    assert tag == "d1_flag_long"


def test_d1_champion_unchanged_when_shadow_raises(monkeypatch):
    baseline = _run_d1_champion(monkeypatch, "absent")
    raised = _run_d1_champion(monkeypatch, "raises")

    assert _fingerprint(raised["trigger_rows"]) == _fingerprint(baseline["trigger_rows"])
    assert _fingerprint(raised) == _fingerprint(baseline)


def test_d1_champion_never_reads_the_shadow_return_value(monkeypatch):
    baseline = _run_d1_champion(monkeypatch, "absent")
    poisoned = _run_d1_champion(monkeypatch, "poison")

    assert _fingerprint(poisoned) == _fingerprint(baseline)


def test_d1_champion_identical_with_shadow_enabled_and_writing(monkeypatch, tmp_path):
    baseline = _run_d1_champion(monkeypatch, "absent")
    enabled = _run_d1_champion(monkeypatch, "enabled", tmp_path=tmp_path)

    assert _fingerprint(enabled["trigger_rows"]) == _fingerprint(baseline["trigger_rows"])
    assert _fingerprint(enabled) == _fingerprint(baseline)

    # Non-vacuity: the Greatness board really ran against these bars.
    store = tmp_path / "greatness_candidates.json"
    assert store.exists(), "greatness shadow never ran - the comparison is vacuous"
    payload = json.loads(store.read_text(encoding="utf-8"))
    assert payload["coverage"]["evaluations"] >= 1
    assert payload["coverage"]["bars_consumed"] >= 1
    assert payload["coverage"]["errors"] == 0


# ===========================================================================
# Structural guard - the shadow return value is unreadable by construction.
# ===========================================================================
def _legacy_tree():
    return ast.parse(LEGACY_SOURCE_PATH.read_text(encoding="utf-8"))


def _shadow_calls(tree):
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name in SHADOW_ENTRY_POINTS:
            calls.append((name, node))
    return calls


def test_shadow_hooks_are_wired_into_the_champion_paths():
    """If a hook disappears the behavioural guards above go vacuous."""
    found = {name for name, _ in _shadow_calls(_legacy_tree())}
    assert found == set(SHADOW_ENTRY_POINTS), (
        "champion call sites changed; re-check the invariance guards"
    )


def test_shadow_call_results_are_discarded_expression_statements():
    """A bare expression statement's value cannot be read by any branch."""
    tree = _legacy_tree()
    discarded = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    }
    calls = _shadow_calls(tree)
    assert calls
    for name, call in calls:
        assert id(call) in discarded, (
            f"{name}() result is consumed at legacy.py line {call.lineno}; "
            "the champion must never read shadow output (plan.md sec 5)"
        )


def test_shadow_entry_points_are_never_bound_to_a_name():
    """No aliasing/storing the hook so its result can be read elsewhere."""
    tree = _legacy_tree()
    call_funcs = {id(call.func) for _, call in _shadow_calls(tree)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in SHADOW_ENTRY_POINTS:
            assert id(node) in call_funcs, (
                f"shadow entry point {node.id} referenced as a value at "
                f"legacy.py line {node.lineno}"
            )
        if isinstance(node, ast.alias) and node.name in SHADOW_ENTRY_POINTS:
            assert node.asname is None, "shadow entry points must not be aliased"


def test_shadow_calls_are_wrapped_in_except_exception():
    """Structural half of guards 1 and 2: the hook can never break the champion."""
    tree = _legacy_tree()
    protected = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_exception = any(
            handler.type is None
            or (isinstance(handler.type, ast.Name) and handler.type.id == "Exception")
            for handler in node.handlers
        )
        if not catches_exception:
            continue
        for statement in node.body:
            protected.update(id(child) for child in ast.walk(statement))
    for name, call in _shadow_calls(tree):
        assert id(call) in protected, (
            f"{name}() at legacy.py line {call.lineno} is not inside a "
            "try/except Exception; a shadow failure could break the champion"
        )
