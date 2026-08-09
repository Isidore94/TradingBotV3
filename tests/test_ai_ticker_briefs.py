"""Phase 1 per-ticker briefs and the small Drive morning publication."""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = ZoneInfo("America/New_York")
OVERNIGHT = datetime(2026, 8, 12, 2, 0, tzinfo=ET)
SESSION = "2026-08-11"


def _base_evidence() -> dict:
    return {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": "2026-08-12T02:00:00-04:00",
        "trade_date": "2026-08-12",
        "session_date": SESSION,
        "selected_scopes": ["setup_trackers"],
        "scope_labels": ["Setup trackers"],
        "sources": [
            {
                "source_id": "setups.rows",
                "label": "Setup rows",
                "path": "/evidence/setups.json",
                "status": "available",
                "as_of": "2026-08-11T16:00:00-04:00",
                "content": [
                    {"symbol": "NVDA", "setup_id": "nvda-1", "state": "ready"},
                    {"symbol": "MSFT", "setup_id": "msft-1", "state": "watch"},
                ],
            }
        ],
        "coverage": {
            "requested_session": SESSION,
            "excluded": [],
            "stale": [],
            "truncated": [],
            "journal_import_health": {"lag_days": 0},
            "counts": {"requested": 1, "usable": 1},
        },
        "safety_contract": {
            "purpose": "advisory only",
            "forbidden_effects": ["scanner scores", "watchlists", "alerts", "bot state"],
        },
    }


def _model_result(symbol: str, source_id: str) -> dict:
    import ai_summary

    summary = {section: [] for section in ai_summary.AI_SUMMARY_SECTIONS}
    summary["executive_summary"] = f"{symbol} has one supported setup row."
    summary["best_candidates"] = [
        {
            "statement": f"Review {symbol}'s ready setup.",
            "evidence_refs": [source_id],
            "confidence": "high",
        }
    ]
    return {
        "schema_version": "ai_summary_result_v1",
        "status": "validated",
        "provider": "local",
        "model": "gemma3:12b",
        "response_id": f"mock-{symbol}",
        "generated_at": "2026-08-12T02:01:00-04:00",
        "duration_seconds": 0.01,
        "summary": summary,
    }


def test_ticker_briefs_use_medium_tier_and_evidence_pointers_without_network(
    tmp_path, monkeypatch
):
    import ai_summary
    from ai_jobs import briefs, window

    focus = tmp_path / "focus_longs.txt"
    longs = tmp_path / "longs.txt"
    focus.write_text("NVDA\n", encoding="utf-8")
    longs.write_text("MSFT\nNVDA\n", encoding="utf-8")
    watchlists = {"focus_longs": focus, "longs": longs}
    morning = tmp_path / "drive" / briefs.MORNING_BRIEF_FILENAME
    calls: list[dict] = []

    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")
    monkeypatch.setattr(window, "in_offhours_window", lambda now=None: True)
    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier: f"{tier}-model")
    monkeypatch.setattr(ai_summary, "build_evidence_package", lambda *a, **k: _base_evidence())

    def endpoint_mock(**kwargs):
        evidence = kwargs["evidence"]
        symbol = evidence["brief_symbol"]
        calls.append(kwargs)
        assert evidence["brief_request"].startswith(f"Produce an advisory brief for {symbol}")
        setup = next(row for row in evidence["sources"] if row["source_id"] == "setups.rows")
        assert setup["content"] == [{"symbol": symbol, "setup_id": f"{symbol.lower()}-1", "state": "ready" if symbol == "NVDA" else "watch"}]
        assert setup["evidence_pointer"] == {
            "source_id": "setups.rows",
            "path": "/evidence/setups.json",
            "as_of": "2026-08-11T16:00:00-04:00",
        }
        return _model_result(symbol, "setups.rows")

    monkeypatch.setattr(ai_summary, "request_ai_summary", endpoint_mock)
    outcome = briefs.run_ticker_briefs(
        session_date=SESSION,
        now=OVERNIGHT,
        watchlist_paths=watchlists,
        output_root=tmp_path / "ai_store" / "briefs",
        morning_path=morning,
    )

    assert outcome["status"] == "ok"
    assert outcome["model"] == "medium-model"
    assert outcome["tokens"]["ticker_calls"] == 2
    assert [call["provider"] for call in calls] == ["local", "local"]
    assert [call["model"] for call in calls] == ["medium-model", "medium-model"]
    text = morning.read_text(encoding="utf-8")
    assert "ADVISORY ONLY" in text
    assert "## NVDA  [focus_longs, longs]" in text
    assert "Review NVDA's ready setup. [setups.rows]" in text
    assert "## MSFT  [longs]" in text
    assert morning.stat().st_size <= briefs.MAX_MORNING_BRIEF_BYTES

    evidence_files = sorted((tmp_path / "ai_store" / "briefs").rglob("*_evidence.json"))
    assert len(evidence_files) == 2
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in evidence_files]
    assert {payload["brief_symbol"] for payload in payloads} == {"NVDA", "MSFT"}
    assert all(payload["safety_contract"]["forbidden_effects"] for payload in payloads)


def test_market_hours_direct_invocation_refuses_before_any_endpoint_call(
    tmp_path, monkeypatch
):
    import ai_summary
    from ai_jobs import briefs, window

    watch = tmp_path / "focus.txt"
    watch.write_text("NVDA\n", encoding="utf-8")
    called: list[bool] = []
    monkeypatch.setattr(
        window,
        "market_session_block",
        lambda now=None: "market session is live (09:30-16:00 ET)",
    )
    monkeypatch.setattr(ai_summary, "request_ai_summary", lambda **kwargs: called.append(True))

    with pytest.raises(RuntimeError, match="refused.*market session is live"):
        briefs.run_ticker_briefs(
            session_date=SESSION,
            now=datetime(2026, 8, 11, 11, 0, tzinfo=ET),
            watchlist_paths={"focus": watch},
            output_root=tmp_path / "briefs",
            morning_path=tmp_path / "morning.txt",
        )
    assert called == []
    assert not (tmp_path / "morning.txt").exists()


def _offhours(monkeypatch):
    from ai_jobs import window

    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")
    monkeypatch.setattr(window, "in_offhours_window", lambda now=None: True)


def test_unreadable_watchlists_refuse_to_publish_and_keep_the_last_morning_file(
    tmp_path, monkeypatch
):
    """plan.md sec 5: missing data is uncertainty, never confirmation.

    A Drive folder that did not mount makes every watchlist unreadable. That
    must not become a published morning file claiming the trader watches
    nothing, and it must not destroy the last verified brief.
    """
    import ai_summary
    from ai_jobs import briefs

    _offhours(monkeypatch)
    called: list[bool] = []
    monkeypatch.setattr(ai_summary, "request_ai_summary", lambda **kwargs: called.append(True))
    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)

    morning = tmp_path / "drive" / briefs.MORNING_BRIEF_FILENAME
    morning.parent.mkdir(parents=True, exist_ok=True)
    morning.write_text("LAST VERIFIED BRIEF\n", encoding="utf-8")
    before = morning.read_bytes()

    missing = tmp_path / "gone"  # never created: the mount is not there
    outcome = briefs.run_ticker_briefs(
        session_date=SESSION,
        now=OVERNIGHT,
        watchlist_paths={
            "focus_longs": missing / "focus_longs.txt",
            "longs": missing / "longs.txt",
        },
        output_root=tmp_path / "ai_store" / "briefs",
        morning_path=morning,
    )

    assert outcome["status"] == "skipped"
    assert outcome["outputs"] == []
    assert "refused to publish" in outcome["reason"]
    assert "focus_longs" in outcome["reason"] and "longs" in outcome["reason"]
    assert called == []
    assert morning.read_bytes() == before
    assert list(morning.parent.glob(".*.tmp")) == []


def test_one_unreadable_source_still_refuses_an_empty_morning_file(tmp_path, monkeypatch):
    """A partial read cannot certify emptiness - the missing list is exactly
    where the names would have been."""
    import ai_summary
    from ai_jobs import briefs

    _offhours(monkeypatch)
    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)

    readable = tmp_path / "longs.txt"
    readable.write_text("# nothing today\n", encoding="utf-8")
    morning = tmp_path / "ai_morning_brief.txt"
    morning.write_text("LAST VERIFIED BRIEF\n", encoding="utf-8")

    outcome = briefs.run_ticker_briefs(
        session_date=SESSION,
        now=OVERNIGHT,
        watchlist_paths={"longs": readable, "focus_longs": tmp_path / "gone.txt"},
        output_root=tmp_path / "briefs",
        morning_path=morning,
    )

    assert outcome["status"] == "skipped"
    assert morning.read_text(encoding="utf-8") == "LAST VERIFIED BRIEF\n"


def test_readable_but_empty_watchlists_publish_an_honest_empty_morning_file(
    tmp_path, monkeypatch
):
    """Every source read fine and held no ticker. That IS a finding, so the
    morning file says so rather than leaving yesterday's brief in place."""
    import ai_summary
    from ai_jobs import briefs

    _offhours(monkeypatch)
    called: list[bool] = []
    monkeypatch.setattr(ai_summary, "request_ai_summary", lambda **kwargs: called.append(True))
    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)

    focus = tmp_path / "focus_longs.txt"
    longs = tmp_path / "longs.txt"
    focus.write_text("", encoding="utf-8")
    longs.write_text("# cleared out after the close\n\n", encoding="utf-8")
    morning = tmp_path / "drive" / briefs.MORNING_BRIEF_FILENAME
    morning.parent.mkdir(parents=True, exist_ok=True)
    morning.write_text("YESTERDAY\n", encoding="utf-8")

    outcome = briefs.run_ticker_briefs(
        session_date=SESSION,
        now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus, "longs": longs},
        output_root=tmp_path / "ai_store" / "briefs",
        morning_path=morning,
    )

    assert outcome["status"] == "ok"
    assert outcome["reason"] == f"no Focus/watchlist tickers for {SESSION}"
    assert outcome["outputs"] == [str(morning)]
    assert called == []
    text = morning.read_text(encoding="utf-8")
    assert "ADVISORY ONLY" in text
    assert f"Session reviewed: {SESSION}" in text
    assert "YESTERDAY" not in text


def test_load_brief_symbols_separates_read_sources_from_unreadable_ones(tmp_path):
    from ai_jobs import briefs

    good = tmp_path / "longs.txt"
    good.write_text("NVDA\n$msft\n", encoding="utf-8")
    result = briefs.load_brief_symbols(
        {"longs": good, "swing_longs": tmp_path / "absent.txt"}
    )

    assert result.symbols == ["NVDA", "MSFT"]
    assert result.read == ["longs"]
    assert result.unreadable == ["swing_longs"]
    assert result.is_trustworthy_empty is False
    assert result.memberships["NVDA"] == [{"list": "longs", "path": str(good)}]

    empty = tmp_path / "shorts.txt"
    empty.write_text("\n", encoding="utf-8")
    clean = briefs.load_brief_symbols({"shorts": empty})
    assert clean.symbols == [] and clean.unreadable == []
    assert clean.is_trustworthy_empty is True

    # No source at all is not evidence of emptiness either.
    assert briefs.load_brief_symbols({}).is_trustworthy_empty is False


def test_morning_atomic_publish_failure_keeps_the_previous_verified_file(tmp_path):
    from ai_jobs.briefs import atomic_publish_morning_file

    target = tmp_path / "ai_morning_brief.txt"
    target.write_text("LAST VERIFIED\n", encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("Drive unavailable")

    with pytest.raises(OSError, match="Drive unavailable"):
        atomic_publish_morning_file(
            "NEW BUT UNPUBLISHED\n", path=target, replace=fail_replace
        )
    assert target.read_text(encoding="utf-8") == "LAST VERIFIED\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_ai_jobs_stay_out_of_detector_scoring_and_alert_modules():
    forbidden_dependencies = (
        "autopilot_core",
        "bounce_bot",
        "master_avwap",
        "technical_integrity",
        "price_alert",
        "d1_level_feed",
    )
    for path in (SCRIPTS_DIR / "ai_jobs").glob("*.py"):
        imports = _imported_modules(path)
        assert not any(
            name.startswith(prefix)
            for name in imports
            for prefix in forbidden_dependencies
        ), f"{path.name} crossed into live decision code: {sorted(imports)}"

    live_modules = (
        SCRIPTS_DIR / "autopilot_core.py",
        SCRIPTS_DIR / "bounce_bot_lib" / "legacy.py",
        SCRIPTS_DIR / "master_avwap_lib" / "legacy.py",
        SCRIPTS_DIR / "technical_integrity.py",
        SCRIPTS_DIR / "d1_level_feed.py",
    )
    for path in live_modules:
        assert not any(
            name == "ai_jobs" or name.startswith("ai_jobs.")
            for name in _imported_modules(path)
        ), f"{path.name} must not consume advisory AI output"
