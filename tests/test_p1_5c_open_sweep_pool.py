"""P1-5 5c: the open sweep reads all of universe_all.txt; Focus and typed names first.

The IB budget is unchanged: the sweep is yfinance, and only the top
AUTOPILOT_WATCHLIST_CAP names per side by gap and RS reach the bot's files,
beside every name the trader typed (merge_autopilot_watchlist keeps them).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import autopilot_core as core  # noqa: E402


def _write(path: Path, symbols) -> Path:
    path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    return path


def _universe(tmp_path, monkeypatch, *, longs, shorts, all_names):
    monkeypatch.setattr(core, "UNIVERSE_LONGS_FILE", _write(tmp_path / "ul.txt", longs))
    monkeypatch.setattr(core, "UNIVERSE_SHORTS_FILE", _write(tmp_path / "us.txt", shorts))
    monkeypatch.setattr(core, "UNIVERSE_ALL_FILE", _write(tmp_path / "ua.txt", all_names))


def test_every_universe_all_name_is_swept_when_the_lists_overflow_the_old_cap(tmp_path, monkeypatch):
    """Live 2026-09-24: 1,163 long/short names + 1,467 in universe_all; the old 1,200 cap cut 267."""
    longs = [f"L{i:04d}" for i in range(500)]
    shorts = [f"S{i:04d}" for i in range(700)]
    extra = [f"A{i:04d}" for i in range(300)]
    _universe(tmp_path, monkeypatch, longs=longs, shorts=shorts, all_names=longs + shorts + extra)
    pool = core.load_universe_pool()
    assert set(extra) <= set(pool)
    assert len(pool) == 1500 and len(set(pool)) == 1500


def test_focus_and_typed_names_lead_and_are_never_cut(tmp_path, monkeypatch):
    all_names = [f"A{i:04d}" for i in range(50)]
    _universe(tmp_path, monkeypatch, longs=[], shorts=[], all_names=all_names)
    pool = core.load_universe_pool(max_symbols=10, priority=["myfocus", "TYPED", "A0003", "TYPED"])
    assert pool[:3] == ["MYFOCUS", "TYPED", "A0003"]
    assert len(pool) == 10
    # Even a cap smaller than the priority list keeps every priority name.
    tight = core.load_universe_pool(max_symbols=1, priority=["ONE", "TWO"])
    assert tight == ["ONE", "TWO"]


def test_universe_all_order_leads_the_pool_after_the_priority_names(tmp_path, monkeypatch):
    _universe(tmp_path, monkeypatch, longs=["ZLONG"], shorts=["ZSHORT"], all_names=["AAA", "BBB"])
    assert core.load_universe_pool() == ["AAA", "BBB", "ZLONG", "ZSHORT"]


def test_the_ib_budget_cap_is_unchanged():
    assert core.AUTOPILOT_WATCHLIST_CAP == 40


def test_the_service_hands_focus_and_typed_names_to_the_pool(monkeypatch):
    from ui.services import autopilot_service as svc_mod
    from ui.services.autopilot_service import AutopilotService

    service = AutopilotService.__new__(AutopilotService)
    service._read_watchlists = lambda: (["TYPEDL"], ["TYPEDS"])
    monkeypatch.setattr(
        svc_mod, "read_watchlist_symbols", lambda path: ["FOCUSL"] if "long" in str(path).lower() else ["FOCUSS"]
    )
    names = service._open_sweep_priority_names()
    assert names[:2] == ["TYPEDL", "TYPEDS"]
    assert {"FOCUSL", "FOCUSS"} <= set(names)
