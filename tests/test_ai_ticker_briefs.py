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


def test_ledger_tokens_carry_real_usage_and_stay_silent_when_unreported(
    tmp_path, monkeypatch
):
    """A call count cannot answer "how close to the context ceiling did it run?".

    That question went unanswerable on 2026-08-10, when six nights of truncated
    prompts left no trace but a parse error. Usage is summed across the night's
    calls, and omitted entirely when nothing reported it -- a zero would claim
    the batch was free rather than admit the number is unknown.
    """
    import ai_summary
    from ai_jobs import briefs, window

    focus = tmp_path / "focus_longs.txt"
    focus.write_text("NVDA\nMSFT\n", encoding="utf-8")
    morning = tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME

    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")
    monkeypatch.setattr(window, "in_offhours_window", lambda now=None: True)
    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier: f"{tier}-model")
    monkeypatch.setattr(ai_summary, "build_evidence_package", lambda *a, **k: _base_evidence())

    def _run(usage_per_call, label):
        def endpoint_mock(**kwargs):
            result = _model_result(kwargs["evidence"]["brief_symbol"], "setups.rows")
            if usage_per_call is not None:
                result["usage"] = dict(usage_per_call)
            return result

        monkeypatch.setattr(ai_summary, "request_ai_summary", endpoint_mock)
        # A distinct store per run: two firings against the same store would
        # (correctly) resume from the manifest and make no calls at all.
        return briefs.run_ticker_briefs(
            session_date=SESSION,
            now=OVERNIGHT,
            watchlist_paths={"focus_longs": focus},
            output_root=tmp_path / label / "briefs",
            morning_path=morning,
        )

    reported = _run({"prompt_tokens": 4000, "completion_tokens": 250}, "first")
    assert reported["tokens"]["ticker_calls"] == 2
    assert reported["tokens"]["prompt_tokens"] == 8000
    assert reported["tokens"]["completion_tokens"] == 500

    silent = _run(None, "second")
    assert silent["tokens"] == {
        "ticker_calls": 2,
        "tickers_resolved": 2,
        "tickers_reused": 0,
        "tickers_failed": 0,
    }


# ---------------------------------------------------------------------------
# TB-0: project first, budget second
# ---------------------------------------------------------------------------
def _crowded_base(symbol: str = "MRVL", rows: int = 400) -> dict:
    """A base package shaped like the real one: one huge per-symbol table.

    The live `setups.current_tracker` was 95,806 chars on 2026-08-10. The
    target symbol's row sits near the *front*, which is where the budget's
    keep-the-most-recent-rows rule drops it first.
    """
    table = [
        {"symbol": f"SYM{index:03d}", "state": "watch", "note": "x" * 180}
        for index in range(rows)
    ]
    table.insert(3, {"symbol": symbol, "state": "ready", "note": "the row that matters"})
    base = _base_evidence()
    base["sources"] = [
        {
            "source_id": "setups.current_tracker",
            "label": "Master AVWAP tracker",
            "path": "/evidence/tracker.json",
            "status": "available",
            "as_of": "2026-08-11T16:00:00-04:00",
            "notices": [],
            "content": table,
        }
    ]
    return base


def _content_chars(evidence: dict) -> int:
    return sum(
        len(json.dumps(source.get("content"), sort_keys=True, default=str))
        for source in evidence["sources"]
    )


def test_budgeting_the_base_first_is_what_sheared_the_symbol_out(tmp_path):
    """The defect TB-0 repairs, stated as an executable fact.

    Rationing the *base* to the local ceiling drops the front rows of a large
    table. Every symbol living in those rows is then invisible to projection,
    which is how all 95 briefs on 2026-08-10/11 came back content-free.
    """
    import ai_summary
    from ai_jobs import briefs

    base = _crowded_base()
    starved, _excluded = ai_summary.ration_projected_sources(
        base["sources"], total=ai_summary.DEFAULT_LOCAL_EVIDENCE_BUDGET_CHARS
    )
    starved_base = dict(base, sources=starved)

    old_way = briefs.build_ticker_evidence(starved_base, "MRVL", [{"list": "longs"}])
    assert briefs.is_membership_only(old_way), (
        "budget-then-project leaves the symbol with nothing but its own membership"
    )

    new_way = briefs.build_ticker_evidence(
        base,
        "MRVL",
        [{"list": "longs"}],
        budget_chars=ai_summary.DEFAULT_LOCAL_EVIDENCE_BUDGET_CHARS,
    )
    assert not briefs.is_membership_only(new_way)
    tracker = next(
        row for row in new_way["sources"] if row["source_id"] == "setups.current_tracker"
    )
    assert {"symbol": "MRVL", "state": "ready", "note": "the row that matters"} in tracker[
        "content"
    ]


def test_every_per_symbol_package_is_still_rationed_to_the_local_budget(
    tmp_path, monkeypatch
):
    """Projection may not become a way around the local context window."""
    import ai_summary
    from ai_jobs import briefs, window

    focus = tmp_path / "focus_longs.txt"
    focus.write_text("MRVL\n", encoding="utf-8")
    seen: dict = {}
    packages: list[dict] = []

    def _capture(scopes, *, session_date=None, **kwargs):
        seen["budget_chars"] = kwargs.get("budget_chars")
        # Every row mentions the symbol, so the projection stays enormous and
        # the per-symbol budget is the only thing standing between it and the
        # model.
        base = _crowded_base()
        base["sources"][0]["content"] = [
            {"symbol": "MRVL", "state": f"row-{index}", "note": "y" * 180}
            for index in range(400)
        ]
        return base

    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")
    monkeypatch.setattr(window, "in_offhours_window", lambda now=None: True)
    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier: f"{tier}-model")
    monkeypatch.setattr(ai_summary, "build_evidence_package", _capture)

    def endpoint_mock(**kwargs):
        packages.append(kwargs["evidence"])
        return _model_result("MRVL", "setups.current_tracker")

    monkeypatch.setattr(ai_summary, "request_ai_summary", endpoint_mock)
    briefs.run_ticker_briefs(
        session_date=SESSION,
        now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus},
        output_root=tmp_path / "ai_store" / "briefs",
        morning_path=tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME,
    )

    # The base asks for the cloud ceiling so the rows survive to projection...
    assert seen["budget_chars"] == ai_summary.MAX_TOTAL_EVIDENCE_CHARS
    # ...and the package actually sent still fits the local one.
    assert len(packages) == 1
    assert _content_chars(packages[0]) <= ai_summary.DEFAULT_LOCAL_EVIDENCE_BUDGET_CHARS
    # Rationed honestly, in the packager's own vocabulary.
    coverage = packages[0]["coverage"]
    assert coverage["counts"]["truncated"] >= 1
    assert any("showing most recent" in notice for row in coverage["truncated"] for notice in row["notices"])


def test_the_daily_summary_keeps_its_own_local_budget(tmp_path, monkeypatch):
    """TB-0 is scoped to the ticker path; the daily brief is untouched."""
    import ai_summary
    from ai_jobs import briefs

    seen: dict = {}

    def _capture(scopes, *, session_date=None, **kwargs):
        seen["budget_chars"] = kwargs.get("budget_chars")
        return {"package_id": "p", "sources": [], "coverage": {"counts": {"requested": 0}}}

    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier: f"{tier}-model")
    monkeypatch.setattr(ai_summary, "build_evidence_package", _capture)
    monkeypatch.setattr(ai_summary, "has_usable_sources", lambda evidence: False)
    monkeypatch.setattr(
        ai_summary, "degraded_result", lambda evidence, *, reason, model="": {"summary": {}}
    )
    monkeypatch.setattr(
        ai_summary, "export_ai_summary", lambda result, evidence, *, output_dir: {}
    )
    monkeypatch.setattr(briefs, "_summary_dir", lambda session: tmp_path)

    briefs.run_daily_summary(session_date=SESSION)

    assert seen["budget_chars"] == ai_summary.evidence_budget_for("local", tier="medium")
    assert seen["budget_chars"] == ai_summary.DEFAULT_LOCAL_EVIDENCE_BUDGET_CHARS


def test_a_projection_never_mutates_the_base_it_came_from(tmp_path):
    """Two symbols share one base list; banners must not accumulate on it."""
    import ai_summary
    from ai_jobs import briefs

    base = _crowded_base()
    before = json.dumps(base, sort_keys=True, default=str)
    for symbol in ("MRVL", "SYM001"):
        briefs.build_ticker_evidence(
            base,
            symbol,
            [{"list": "longs"}],
            budget_chars=ai_summary.DEFAULT_LOCAL_EVIDENCE_BUDGET_CHARS,
        )
    assert json.dumps(base, sort_keys=True, default=str) == before


# ---------------------------------------------------------------------------
# TB-1: per-ticker isolation, honest partial publication
# ---------------------------------------------------------------------------
def _briefs_env(monkeypatch):
    import ai_summary
    from ai_jobs import window

    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")
    monkeypatch.setattr(window, "in_offhours_window", lambda now=None: True)
    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier: f"{tier}-model")
    monkeypatch.setattr(ai_summary, "build_evidence_package", lambda *a, **k: _base_evidence())
    return ai_summary


def test_one_failing_ticker_costs_its_own_brief_and_nothing_else(tmp_path, monkeypatch):
    """94 good briefs used to be thrown away by the 95th failure."""
    from ai_jobs import briefs

    ai_summary = _briefs_env(monkeypatch)
    focus = tmp_path / "focus_longs.txt"
    focus.write_text("NVDA\nMSFT\n", encoding="utf-8")
    morning = tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME
    attempts: list[str] = []

    def endpoint_mock(**kwargs):
        symbol = kwargs["evidence"]["brief_symbol"]
        attempts.append(symbol)
        if symbol == "MSFT":
            raise ValueError("summary cited evidence that does not exist")
        return _model_result(symbol, "setups.rows")

    monkeypatch.setattr(ai_summary, "request_ai_summary", endpoint_mock)
    outcome = briefs.run_ticker_briefs(
        session_date=SESSION,
        now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus},
        output_root=tmp_path / "ai_store" / "briefs",
        morning_path=morning,
    )

    # The daily summary's single fed-back-error retry now applies per symbol.
    assert attempts == ["NVDA", "MSFT", "MSFT"]
    assert outcome["status"] == "degraded_no_narrative"
    assert outcome["tokens"]["tickers_failed"] == 1

    text = morning.read_text(encoding="utf-8")
    header, _, body = text.partition("## ")
    assert "Analyzed 1 of 2. Membership-only 0. Failed 1." in header
    assert "Failed: MSFT (summary cited evidence that does not exist)" in header
    assert "NVDA" in body and "## MSFT" not in text


def test_a_window_closing_mid_batch_publishes_what_completed(tmp_path, monkeypatch):
    """The gate is unchanged; what happens after it fires is not."""
    from ai_jobs import briefs, window

    ai_summary = _briefs_env(monkeypatch)
    focus = tmp_path / "focus_longs.txt"
    focus.write_text("NVDA\nMSFT\n", encoding="utf-8")
    morning = tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME
    calls: list[str] = []

    def endpoint_mock(**kwargs):
        calls.append(kwargs["evidence"]["brief_symbol"])
        # The window closes while the first brief is being generated.
        monkeypatch.setattr(window, "in_offhours_window", lambda now=None: False)
        return _model_result(kwargs["evidence"]["brief_symbol"], "setups.rows")

    monkeypatch.setattr(ai_summary, "request_ai_summary", endpoint_mock)
    outcome = briefs.run_ticker_briefs(
        session_date=SESSION,
        now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus},
        output_root=tmp_path / "ai_store" / "briefs",
        morning_path=morning,
    )

    assert calls == ["NVDA"], "no further inference after the window closed"
    assert outcome["status"] == "degraded_no_narrative"
    text = morning.read_text(encoding="utf-8")
    assert "Analyzed 1 of 2. Membership-only 0. Failed 0." in text
    assert "Stopped early" in text and "off-hours window closed" in text
    assert "## NVDA" in text


def test_the_market_session_remains_an_unconditional_stop_mid_batch(tmp_path, monkeypatch):
    """Plan sec 2 is a hard rule: the session stops the job outright."""
    from ai_jobs import briefs, window

    ai_summary = _briefs_env(monkeypatch)
    focus = tmp_path / "focus_longs.txt"
    focus.write_text("NVDA\nMSFT\n", encoding="utf-8")
    morning = tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME
    morning.parent.mkdir(parents=True, exist_ok=True)
    morning.write_text("LAST VERIFIED\n", encoding="utf-8")

    def endpoint_mock(**kwargs):
        monkeypatch.setattr(
            window, "market_session_block", lambda now=None: "market session is live"
        )
        return _model_result(kwargs["evidence"]["brief_symbol"], "setups.rows")

    monkeypatch.setattr(ai_summary, "request_ai_summary", endpoint_mock)
    with pytest.raises(RuntimeError, match="refused.*market session is live"):
        briefs.run_ticker_briefs(
            session_date=SESSION,
            now=OVERNIGHT,
            watchlist_paths={"focus_longs": focus},
            output_root=tmp_path / "ai_store" / "briefs",
            morning_path=morning,
        )

    # Nothing was published, and the completed brief is safe in the manifest
    # for the next legitimate firing to re-render.
    assert morning.read_text(encoding="utf-8") == "LAST VERIFIED\n"
    manifest = briefs.brief_manifest_path(tmp_path / "ai_store" / "briefs", SESSION)
    assert set(briefs.read_brief_manifest(manifest)) == {"NVDA"}


# ---------------------------------------------------------------------------
# TB-2: membership-only symbols never reach the model
# ---------------------------------------------------------------------------
def test_a_symbol_with_no_evidence_is_answered_without_a_model_call(tmp_path, monkeypatch):
    from ai_jobs import briefs

    ai_summary = _briefs_env(monkeypatch)
    focus = tmp_path / "focus_longs.txt"
    swings = tmp_path / "swing_longs.txt"
    focus.write_text("NVDA\n", encoding="utf-8")
    swings.write_text("TSLA\n", encoding="utf-8")  # in no report, tracker, or journal
    morning = tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME
    calls: list[str] = []

    def endpoint_mock(**kwargs):
        calls.append(kwargs["evidence"]["brief_symbol"])
        return _model_result(kwargs["evidence"]["brief_symbol"], "setups.rows")

    monkeypatch.setattr(ai_summary, "request_ai_summary", endpoint_mock)
    outcome = briefs.run_ticker_briefs(
        session_date=SESSION,
        now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus, "swing_longs": swings},
        output_root=tmp_path / "ai_store" / "briefs",
        morning_path=morning,
    )

    assert calls == ["NVDA"]
    # Skipped, but resolved: the night is complete and the job is ok.
    assert outcome["status"] == "ok"
    assert outcome["tokens"]["ticker_calls"] == 1
    text = morning.read_text(encoding="utf-8")
    # Q3.3: TSLA was never analysed - it got no model call and no evidence
    # beyond a list it is on. One total counted it as a brief; three counts
    # cannot.
    assert "Analyzed 1 of 2. Membership-only 1. Failed 0." in text
    assert "## TSLA  [swing_longs]" in text
    assert "no session evidence beyond membership in swing_longs" in text
    # No artifact set for a symbol nothing was said about.
    assert not list((tmp_path / "ai_store" / "briefs").rglob("*TSLA*"))


# ---------------------------------------------------------------------------
# TB-3: resumable completion keyed by (session, symbol, evidence hash)
# ---------------------------------------------------------------------------
def test_a_retry_regenerates_only_what_changed(tmp_path, monkeypatch):
    from ai_jobs import briefs

    ai_summary = _briefs_env(monkeypatch)
    focus = tmp_path / "focus_longs.txt"
    focus.write_text("NVDA\nMSFT\n", encoding="utf-8")
    morning = tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME
    root = tmp_path / "ai_store" / "briefs"
    calls: list[str] = []
    fail_msft = {"on": True}

    def endpoint_mock(**kwargs):
        symbol = kwargs["evidence"]["brief_symbol"]
        calls.append(symbol)
        if symbol == "MSFT" and fail_msft["on"]:
            raise RuntimeError("local AI endpoint is unreachable")
        return _model_result(symbol, "setups.rows")

    monkeypatch.setattr(ai_summary, "request_ai_summary", endpoint_mock)
    first = briefs.run_ticker_briefs(
        session_date=SESSION, now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus}, output_root=root, morning_path=morning,
    )
    assert first["status"] == "degraded_no_narrative"
    assert calls == ["NVDA", "MSFT", "MSFT"]

    # Second firing: NVDA's evidence is unchanged, so it is not regenerated.
    calls.clear()
    fail_msft["on"] = False
    second = briefs.run_ticker_briefs(
        session_date=SESSION, now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus}, output_root=root, morning_path=morning,
    )
    assert calls == ["MSFT"], "only the unresolved symbol is retried"
    assert second["status"] == "ok"
    assert second["tokens"] == {
        "ticker_calls": 1,
        "tickers_resolved": 2,
        "tickers_reused": 1,
        "tickers_failed": 0,
    }
    text = morning.read_text(encoding="utf-8")
    assert "Analyzed 2 of 2. Membership-only 0. Failed 0." in text and "Failed:" not in text
    # One artifact set per symbol, not one per attempt.
    assert len(list((root / SESSION[:4] / SESSION / "tickers" / "NVDA").glob("*_manifest.json"))) == 1

    # Third firing with changed evidence: the stale brief is regenerated.
    calls.clear()
    changed = _base_evidence()
    changed["sources"][0]["content"] = [
        {"symbol": "NVDA", "setup_id": "nvda-1", "state": "triggered"},
        {"symbol": "MSFT", "setup_id": "msft-1", "state": "watch"},
    ]
    monkeypatch.setattr(ai_summary, "build_evidence_package", lambda *a, **k: changed)
    briefs.run_ticker_briefs(
        session_date=SESSION, now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus}, output_root=root, morning_path=morning,
    )
    assert calls == ["NVDA"], "a changed evidence hash means a stale brief"


def test_an_unreadable_manifest_regenerates_rather_than_refusing(tmp_path, monkeypatch):
    from ai_jobs import briefs

    ai_summary = _briefs_env(monkeypatch)
    focus = tmp_path / "focus_longs.txt"
    focus.write_text("NVDA\n", encoding="utf-8")
    root = tmp_path / "ai_store" / "briefs"
    manifest = briefs.brief_manifest_path(root, SESSION)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{ this is not json\n", encoding="utf-8")

    monkeypatch.setattr(
        ai_summary,
        "request_ai_summary",
        lambda **kwargs: _model_result(kwargs["evidence"]["brief_symbol"], "setups.rows"),
    )
    outcome = briefs.run_ticker_briefs(
        session_date=SESSION, now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus}, output_root=root,
        morning_path=tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME,
    )
    assert outcome["status"] == "ok"
    assert outcome["tokens"]["ticker_calls"] == 1


# ---------------------------------------------------------------------------
# TB-3 repair: resume on the evidence, not on when it was read
# ---------------------------------------------------------------------------
def test_the_resume_key_ignores_read_stamps_that_move_every_firing(tmp_path):
    """The live 2026-08-11 defect: identical evidence, a different hash.

    ``evidence_hash`` covers ``generated_at`` and every source's ``as_of``, and
    ``run_ticker_briefs`` builds its base without passing ``now``. Two firings
    over the same session therefore produced two hashes for the same evidence,
    so the resume never matched: a second runner instance re-briefed the first
    25 symbols from the top and left 25 duplicate artifact sets on the DAS.
    """
    from ai_jobs import briefs

    memberships = [{"list": "longs", "path": "/home/longs.txt"}]
    first = _base_evidence()
    second = _base_evidence()
    second["generated_at"] = "2026-08-12T04:30:00-04:00"  # a later firing
    second["sources"][0]["as_of"] = "2026-08-11T16:05:00-04:00"  # re-read stamp

    a = briefs.build_ticker_evidence(first, "NVDA", memberships)
    b = briefs.build_ticker_evidence(second, "NVDA", memberships)

    assert a["evidence_hash"] != b["evidence_hash"], "package identity still moves"
    assert a["resume_key"] == b["resume_key"], "the evidence itself did not change"

    changed = _base_evidence()
    changed["sources"][0]["content"] = [{"symbol": "NVDA", "setup_id": "nvda-1", "state": "triggered"}]
    assert briefs.build_ticker_evidence(changed, "NVDA", memberships)["resume_key"] != a["resume_key"]


def test_a_manifest_row_without_a_resume_key_is_regenerated_not_reused(tmp_path, monkeypatch):
    """A v1 row costs a regeneration. It must never cost a wrong skip."""
    from ai_jobs import briefs

    ai_summary = _briefs_env(monkeypatch)
    focus = tmp_path / "focus_longs.txt"
    focus.write_text("NVDA\n", encoding="utf-8")
    root = tmp_path / "ai_store" / "briefs"
    manifest = briefs.brief_manifest_path(root, SESSION)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "ai_ticker_brief_manifest_v1",
                "session_date": SESSION,
                "symbol": "NVDA",
                "status": briefs.BRIEF_STATUS_BRIEFED,
                "evidence_hash": "whatever-the-old-scheme-produced",
                "memberships": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[str] = []
    monkeypatch.setattr(
        ai_summary,
        "request_ai_summary",
        lambda **kw: (calls.append(kw["evidence"]["brief_symbol"]),
                      _model_result(kw["evidence"]["brief_symbol"], "setups.rows"))[1],
    )
    briefs.run_ticker_briefs(
        session_date=SESSION, now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus}, output_root=root,
        morning_path=tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME,
    )
    assert calls == ["NVDA"]


# ---------------------------------------------------------------------------
# TB-5: a roster line is not evidence about the symbol
# ---------------------------------------------------------------------------
def test_roster_lines_are_dropped_and_real_rows_are_kept():
    """96.2% of the 2026-08-11 payload was ticker name-dumps."""
    from ai_jobs import briefs

    roster = "SE, P, OUST, ONTO, BCRX, JBL, DCH, CHYM, ARES, ENTG, MDB, BEP, AMPL"
    tv_paste = "TV paste: SE,P,OUST,ONTO,BCRX,JBL,DCH,CHYM,ARES,ENTG,MDB"
    events = "  LONG: A, AAPL, ABCL, ABNB, ACAD, ACGL, ADI, MDB, MDLZ, MET, MFC"
    copy_text = '      "copy_text": "AA, ABCL, ABNB, ABSI, AEHR, MDB, MGNI, MPC, NEM",'
    scan_line = "MDB    LONG  rs=+12.18  1d=+5.4%   5d=+15.6%  ind=IGV ind_rs=+11.14  family=general"
    tier_row = (
        "RBRK 2026-08-04->2026-08-11 (+19.68%); MDB 2026-08-04->2026-08-11 (+15.56%); "
        "RGEN 2026-08-04->2026-08-11 (+10.01%); A 2026-08-04->2026-08-11 (+7.39%); "
        "CHRW 2026-08-04->2026-08-11 (+6.24%); OI 2026-08-04->2026-08-11 (+5.67%)"
    )

    for line in (roster, tv_paste, events, copy_text):
        assert briefs.is_roster_line(line, "MDB"), line
    # The tier row carries eight tickers and is pure signal: a ticker-count
    # threshold would have discarded exactly the rows worth keeping.
    assert not briefs.is_roster_line(tier_row, "MDB")
    assert not briefs.is_roster_line(scan_line, "MDB")

    source = "\n".join([roster, tv_paste, events, copy_text, scan_line, tier_row])
    projected = briefs._extract_ticker_content(source, "MDB")
    assert projected == f"{scan_line}\n{tier_row}"


def test_a_bare_name_in_a_list_does_not_defeat_the_membership_only_skip():
    """Auto Pilot's longs array said "MDB" and nothing else."""
    from ai_jobs import briefs

    assert briefs.is_bare_membership_line('   "MDB",', "MDB")
    assert briefs.is_bare_membership_line("MDB", "MDB")
    assert not briefs.is_bare_membership_line("MDB LONG rs=+12.18", "MDB")

    auto_state = {"autopilot_written": {"longs": ["SE", "MDB", "ABCL"]}}
    assert briefs._extract_ticker_content(auto_state, "MDB") is None

    package = briefs.build_ticker_evidence(
        {
            "session_date": SESSION,
            "generated_at": "2026-08-12T02:00:00-04:00",
            "sources": [
                {"source_id": "market.auto_state", "label": "Auto Pilot state",
                 "status": "available", "content": auto_state},
            ],
            "coverage": {"excluded": []},
        },
        "MDB",
        [{"list": "longs", "path": "/home/longs.txt"}],
    )
    assert briefs.is_membership_only(package), "membership wearing a second hat is still membership"


# ---------------------------------------------------------------------------
# Crash-safe publication: a killed run still leaves the briefs it finished
# ---------------------------------------------------------------------------
def test_a_hard_kill_mid_batch_still_leaves_the_finished_briefs_published(
    tmp_path, monkeypatch
):
    """2026-08-11: 126 briefs on the DAS, yesterday's file in the home folder.

    The desk entered Modern Standby at 01:39 and the process died at symbol 101
    of 182. Publication happened only after the loop, so nothing was published
    at all. KeyboardInterrupt stands in for that kill: it is deliberately not
    one of the exceptions the per-symbol handler catches.
    """
    from ai_jobs import briefs

    ai_summary = _briefs_env(monkeypatch)
    focus = tmp_path / "focus_longs.txt"
    focus.write_text("NVDA\nMSFT\n", encoding="utf-8")
    morning = tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME

    def endpoint_mock(**kwargs):
        symbol = kwargs["evidence"]["brief_symbol"]
        if symbol == "MSFT":
            raise KeyboardInterrupt("the machine went to sleep")
        return _model_result(symbol, "setups.rows")

    monkeypatch.setattr(ai_summary, "request_ai_summary", endpoint_mock)
    with pytest.raises(KeyboardInterrupt):
        briefs.run_ticker_briefs(
            session_date=SESSION, now=OVERNIGHT,
            watchlist_paths={"focus_longs": focus},
            output_root=tmp_path / "ai_store" / "briefs",
            morning_path=morning,
        )

    text = morning.read_text(encoding="utf-8")
    assert "## NVDA" in text, "the brief that finished before the kill is published"
    assert "Analyzed 1 of 2. Membership-only 0. Failed 0." in text
    assert briefs.INCOMPLETE_RUN_NOTE in text, "and it says it was still running"


def test_a_completed_run_does_not_claim_to_be_still_running(tmp_path, monkeypatch):
    from ai_jobs import briefs

    ai_summary = _briefs_env(monkeypatch)
    focus = tmp_path / "focus_longs.txt"
    focus.write_text("NVDA\n", encoding="utf-8")
    morning = tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME
    monkeypatch.setattr(
        ai_summary,
        "request_ai_summary",
        lambda **kw: _model_result(kw["evidence"]["brief_symbol"], "setups.rows"),
    )
    briefs.run_ticker_briefs(
        session_date=SESSION, now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus},
        output_root=tmp_path / "ai_store" / "briefs",
        morning_path=morning,
    )
    text = morning.read_text(encoding="utf-8")
    assert "Analyzed 1 of 1. Membership-only 0. Failed 0." in text
    assert briefs.INCOMPLETE_RUN_NOTE not in text


def test_an_interim_publish_failure_never_costs_the_batch(tmp_path, monkeypatch):
    """Publishing is the cheap part; inference is not. A publish fault waits."""
    from ai_jobs import briefs

    ai_summary = _briefs_env(monkeypatch)
    focus = tmp_path / "focus_longs.txt"
    focus.write_text("NVDA\nMSFT\n", encoding="utf-8")
    calls: list[str] = []
    real_publish = briefs.atomic_publish_morning_file
    state = {"fail": True}

    def flaky_publish(content, *, path=None, replace=None):
        if state["fail"]:
            state["fail"] = False
            raise OSError("the home folder blinked")
        return real_publish(content, path=path)

    monkeypatch.setattr(briefs, "atomic_publish_morning_file", flaky_publish)
    monkeypatch.setattr(
        ai_summary,
        "request_ai_summary",
        lambda **kw: (calls.append(kw["evidence"]["brief_symbol"]),
                      _model_result(kw["evidence"]["brief_symbol"], "setups.rows"))[1],
    )
    outcome = briefs.run_ticker_briefs(
        session_date=SESSION, now=OVERNIGHT,
        watchlist_paths={"focus_longs": focus},
        output_root=tmp_path / "ai_store" / "briefs",
        morning_path=tmp_path / "home" / briefs.MORNING_BRIEF_FILENAME,
    )
    assert calls == ["NVDA", "MSFT"], "both symbols were still briefed"
    assert outcome["status"] == "ok"
