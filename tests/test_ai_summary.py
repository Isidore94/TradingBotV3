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
    sections = {
        name: []
        for name in (
            "what_is_working",
            "what_is_not_working",
            "best_candidates",
            "lessons_for_tomorrow",
            "data_quality",
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
    assert vault.resolve("openai") == ("saved-secret", "Windows Credential Manager")
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

    assert evidence["schema_version"] == "ai_evidence_package_v1"
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


def test_validation_rejects_hallucinated_evidence_reference(tmp_path):
    from ai_summary import build_evidence_package, validate_ai_summary

    evidence = build_evidence_package(["daily_report"], source_overrides=_daily_overrides(tmp_path))
    valid = validate_ai_summary(_valid_summary("daily.auto_report"), evidence)
    assert valid["what_is_working"][0]["confidence"] == "high"

    bad = _valid_summary("not.a.real.source")
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_ai_summary(bad, evidence)


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
