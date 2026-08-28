import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _valid_summary(ref: str) -> dict:
    # Model sections only: data_quality is machine-owned and a model that
    # returns it is rejected (Sol 5.6 verification review, item 4).
    sections = {
        name: []
        for name in (
            "what_is_working",
            "what_is_not_working",
            "best_candidates",
            "lessons_for_tomorrow",
            "risk_notes",
        )
    }
    sections["what_is_working"] = [
        {"statement": "Swing rows are shown first.", "evidence_refs": [ref], "confidence": "high"}
    ]
    return {"executive_summary": "The selected evidence supports one measured finding.", **sections}


def _daily_overrides(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for source_id in ("daily.auto_report", "daily.market_prep", "daily.master_events"):
        path = tmp_path / (source_id.replace(".", "_") + ".txt")
        path.write_text(f"Evidence from {source_id}\n", encoding="utf-8")
        paths[source_id] = path
    return paths


def test_credential_vault_prefers_environment_and_never_exposes_value():
    from ai_credentials import AiCredentialVault, MemoryCredentialBackend

    backend = MemoryCredentialBackend()
    vault = AiCredentialVault(backend, environ={})
    vault.save("openai", "saved-secret")
    assert vault.resolve("openai") == ("saved-secret", "saved key store")
    assert "saved-secret" not in vault.status("openai")

    env_vault = AiCredentialVault(backend, environ={"OPENAI_API_KEY": "env-secret"})
    assert env_vault.resolve("openai") == ("env-secret", "environment (OPENAI_API_KEY)")
    env_vault.delete("openai")
    assert backend.values == {}


def test_evidence_package_is_explicit_bounded_and_source_addressable(tmp_path):
    from ai_summary import build_evidence_package

    evidence = build_evidence_package(
        ["daily_report"],
        source_overrides=_daily_overrides(tmp_path),
        now=datetime(2026, 7, 14, 17, 0, tzinfo=timezone.utc),
    )

    assert evidence["schema_version"] == "ai_evidence_package_v2"
    assert evidence["selected_scopes"] == ["daily_report"]
    assert evidence["source_count"] == 3
    assert {row["source_id"] for row in evidence["sources"]} == {
        "daily.auto_report",
        "daily.market_prep",
        "daily.master_events",
    }
    assert len(evidence["evidence_hash"]) == 64
    assert "orders" in evidence["safety_contract"]["forbidden_effects"]

    with pytest.raises(ValueError):
        build_evidence_package([])
    with pytest.raises(ValueError):
        build_evidence_package(["made_up_scope"])


def test_market_condition_evidence_includes_advisory_industry_m5_snapshot(tmp_path):
    from ai_summary import build_evidence_package

    intraday = tmp_path / "industry_intraday_rs_snapshot.json"
    intraday.write_text(
        json.dumps(
            {
                "schema": "industry_intraday_rs_snapshot_v1",
                "advisory_only": True,
                "production_score_effect": "none",
                # A real snapshot: an empty industries list would (correctly)
                # classify the document as carrying no records at all, which
                # tests/test_ai_evidence_coverage.py covers separately.
                "industries": [{"industry": "Semiconductors", "rs": 1.4}],
            }
        ),
        encoding="utf-8",
    )
    evidence = build_evidence_package(
        ["market_conditions"],
        source_overrides={"market.industry_intraday_rs": intraday},
    )
    by_id = {source["source_id"]: source for source in evidence["sources"]}
    assert "market.industry_intraday_rs" in by_id
    assert by_id["market.industry_intraday_rs"]["status"] == "available"
    assert by_id["market.industry_intraday_rs"]["content"]["advisory_only"] is True


def test_validation_rejects_hallucinated_evidence_reference(tmp_path):
    from ai_summary import build_evidence_package, validate_ai_summary

    evidence = build_evidence_package(["daily_report"], source_overrides=_daily_overrides(tmp_path))
    valid = validate_ai_summary(_valid_summary("daily.auto_report"), evidence)
    assert valid["what_is_working"][0]["confidence"] == "high"

    # A hallucinated reference is still never published. Where it is the ONLY
    # citation in the whole document, nothing survives and the document raises:
    # a summary supported by nothing is not a degraded summary.
    bad = _valid_summary("not.a.real.source")
    with pytest.raises(ValueError, match="every citing statement was unsupported"):
        validate_ai_summary(bad, evidence)


def _two_finding_summary(first_refs: list[str], second_refs: list[str]) -> dict:
    sections = {
        name: []
        for name in (
            "what_is_working",
            "what_is_not_working",
            "best_candidates",
            "lessons_for_tomorrow",
            "risk_notes",
        )
    }
    sections["what_is_working"] = [
        {"statement": "First finding.", "evidence_refs": first_refs, "confidence": "high"},
        {"statement": "Second finding.", "evidence_refs": second_refs, "confidence": "medium"},
    ]
    return {"executive_summary": "Two findings.", **sections}


def test_one_bad_citation_costs_its_row_and_not_the_document(tmp_path):
    """Trader decision 2026-08-28.

    Before this, a single unsupported ref raised and threw away every supported
    statement beside it. With two model attempts and a three-attempt session cap,
    one predictable 12B slip cost a whole night - the daily digest lost
    2026-08-25, -26 and -27 that way while every store and the model were healthy.
    """
    from ai_summary import build_evidence_package, validate_ai_summary

    evidence = build_evidence_package(["daily_report"], source_overrides=_daily_overrides(tmp_path))
    drops: list[dict] = []
    result = validate_ai_summary(
        _two_finding_summary(["daily.auto_report"], ["not.a.real.source"]),
        evidence,
        dropped=drops,
    )

    kept = result["what_is_working"]
    assert [row["statement"] for row in kept] == ["First finding."]
    assert kept[0]["evidence_refs"] == ["daily.auto_report"]
    assert len(drops) == 1
    assert drops[0]["struck_refs"] == ["not.a.real.source"]
    assert drops[0]["row_dropped"] is True


def test_a_row_keeps_its_good_citations_when_only_one_is_bad(tmp_path):
    from ai_summary import build_evidence_package, validate_ai_summary

    evidence = build_evidence_package(["daily_report"], source_overrides=_daily_overrides(tmp_path))
    drops: list[dict] = []
    result = validate_ai_summary(
        _two_finding_summary(
            ["daily.auto_report", "not.a.real.source"], ["daily.market_prep"]
        ),
        evidence,
        dropped=drops,
    )

    # The statement stands on the evidence that IS there; only the bad ref goes.
    assert [row["statement"] for row in result["what_is_working"]] == [
        "First finding.",
        "Second finding.",
    ]
    assert result["what_is_working"][0]["evidence_refs"] == ["daily.auto_report"]
    assert drops[0]["row_dropped"] is False


def test_a_dropped_citation_is_disclosed_in_the_published_document(tmp_path):
    """A quietly thinner document reads exactly like a thin evidence night."""
    from ai_summary import (
        build_evidence_package,
        merge_coverage_into_summary,
        validate_ai_summary,
    )

    evidence = build_evidence_package(["daily_report"], source_overrides=_daily_overrides(tmp_path))
    drops: list[dict] = []
    summary = validate_ai_summary(
        _two_finding_summary(["daily.auto_report"], ["not.a.real.source"]),
        evidence,
        dropped=drops,
    )
    merged = merge_coverage_into_summary(summary, evidence, citation_drops=drops)
    disclosure = [
        row["statement"] for row in merged["data_quality"] if "cited evidence" in row["statement"]
    ]
    assert len(disclosure) == 1
    assert "not.a.real.source" in disclosure[0]
    assert disclosure[0].startswith("[system]")


def test_citable_aliases_extend_the_set_but_never_create_one(tmp_path):
    """The digest's fact pack prints the stores its cells came from."""
    from ai_summary import usable_source_ids

    package = {
        "sources": [{"source_id": "digest.facts", "status": "available"}],
        "citable_aliases": ["outcomes.intraday_finals"],
    }
    assert usable_source_ids(package) == {"digest.facts", "outcomes.intraday_finals"}

    # With nothing usable in the package, an alias cannot conjure citability -
    # otherwise a package with no evidence would still accept a citation.
    assert usable_source_ids({"sources": [], "citable_aliases": ["outcomes.intraday_finals"]}) == set()


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_provider_requests_use_current_structured_output_contracts(provider, tmp_path):
    from ai_summary import build_evidence_package, request_ai_summary

    evidence = build_evidence_package(["daily_report"], source_overrides=_daily_overrides(tmp_path))
    summary_text = json.dumps(_valid_summary("daily.auto_report"))
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if provider == "openai":
            return _Response(
                {
                    "id": "resp-openai",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": summary_text}]}
                    ],
                }
            )
        return _Response({"id": "msg-anthropic", "content": [{"type": "text", "text": summary_text}]})

    result = request_ai_summary(
        provider=provider,
        model="test-model",
        api_key="super-secret",
        evidence=evidence,
        post=fake_post,
    )

    assert result["status"] == "validated"
    url, kwargs = calls[0]
    assert "super-secret" not in json.dumps(kwargs["json"])
    if provider == "openai":
        assert url.endswith("/v1/responses")
        assert kwargs["json"]["store"] is False
        assert kwargs["json"]["text"]["format"]["strict"] is True
        assert kwargs["headers"]["Authorization"] == "Bearer super-secret"
    else:
        assert url.endswith("/v1/messages")
        assert kwargs["json"]["output_config"]["format"]["type"] == "json_schema"
        assert kwargs["headers"]["anthropic-version"] == "2023-06-01"
        assert kwargs["headers"]["x-api-key"] == "super-secret"


def test_validated_export_contains_manifest_and_no_secret(tmp_path):
    from ai_summary import build_evidence_package, export_ai_summary

    evidence = build_evidence_package(["daily_report"], source_overrides=_daily_overrides(tmp_path))
    result = {
        "schema_version": "ai_summary_result_v1",
        "status": "validated",
        "provider": "openai",
        "model": "test-model",
        "response_id": "r1",
        "generated_at": "2026-07-14T12:00:00",
        "evidence_package_id": evidence["package_id"],
        "evidence_hash": evidence["evidence_hash"],
        "summary": _valid_summary("daily.auto_report"),
    }
    paths = export_ai_summary(result, evidence, output_dir=tmp_path / "exports")
    assert all(path.exists() for path in paths.values())
    assert "validated_export_only" in paths["manifest"].read_text(encoding="utf-8")
    assert "super-secret" not in "".join(path.read_text(encoding="utf-8") for path in paths.values())


def test_ai_summary_panel_previews_exact_scope_without_network(tmp_path):
    try:
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ai_credentials import AiCredentialVault, MemoryCredentialBackend
        from ui.panels.ai_summary_panel import AiSummaryPanel
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    panel = AiSummaryPanel(
        credential_vault=AiCredentialVault(MemoryCredentialBackend(), environ={}),
        source_overrides=_daily_overrides(tmp_path),
        output_dir=tmp_path / "exports",
    )
    for scope, checkbox in panel.scope_inputs.items():
        checkbox.setChecked(scope == "daily_report")
    panel.build_preview()
    preview = panel.evidence_view.toPlainText()
    assert '"selected_scopes": [' in preview
    assert '"daily_report"' in preview
    assert "Nothing has been sent" in panel.status_label.text()
