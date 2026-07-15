from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ai_credentials import AiCredentialVault
from ai_summary import (
    DEFAULT_MODELS,
    SCOPE_LABELS,
    build_evidence_package,
    export_ai_summary,
    render_ai_summary_markdown,
    request_ai_summary,
)
from project_paths import (
    AI_SUMMARY_EXPORT_DIR,
    get_local_setting,
    open_path_in_file_manager,
    save_local_setting,
)
from ui.widgets.section_header import SectionHeader


class AiSummaryPanel(QFrame):
    """Explicit, export-only provider workspace for evidence-grounded review."""

    statusChanged = Signal(str)
    _runFinished = Signal(object)

    def __init__(
        self,
        bounce_service=None,
        parent=None,
        *,
        credential_vault: AiCredentialVault | None = None,
        source_overrides: Mapping[str, Path] | None = None,
        journal_store=None,
        output_dir: Path = AI_SUMMARY_EXPORT_DIR,
        post=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.bounce_service = bounce_service
        self.credential_vault = credential_vault or AiCredentialVault()
        self.source_overrides = dict(source_overrides or {})
        self.journal_store = journal_store
        self.output_dir = Path(output_dir)
        self._post = post
        self._run_thread: threading.Thread | None = None
        self._last_export: Path | None = None
        self._last_evidence: dict[str, Any] | None = None

        self.provider_input = QComboBox()
        self.provider_input.addItem("ChatGPT / OpenAI", "openai")
        self.provider_input.addItem("Claude / Anthropic", "anthropic")
        saved_provider = str(get_local_setting("qt_ai_summary_provider", "openai") or "openai")
        self.provider_input.setCurrentIndex(max(0, self.provider_input.findData(saved_provider)))

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Provider model ID")
        self._load_model_for_provider()

        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setPlaceholderText("Paste API key for this run (never displayed or exported)")
        self.key_status = QLabel("")
        self.key_status.setObjectName("MutedLabel")
        self.save_key_button = QPushButton("Save in Windows Credential Manager")
        self.delete_key_button = QPushButton("Delete saved key")

        saved_scopes = get_local_setting(
            "qt_ai_summary_scopes",
            ["daily_report", "market_conditions", "setup_trackers", "journal_review"],
        )
        selected = set(saved_scopes if isinstance(saved_scopes, list) else [])
        self.scope_inputs: dict[str, QCheckBox] = {}
        for scope, label in SCOPE_LABELS.items():
            checkbox = QCheckBox(label)
            checkbox.setChecked(scope in selected)
            if scope == "journal_review":
                checkbox.setToolTip("Includes trade results, tags, notes, and structured review events.")
            self.scope_inputs[scope] = checkbox

        self.preview_button = QPushButton("Build Evidence Preview")
        self.generate_button = QPushButton("Generate Advisory Summary")
        self.generate_button.setObjectName("PrimaryButton")
        self.open_export_button = QPushButton("Open Last Export")
        self.open_export_button.setEnabled(False)

        self.evidence_view = QPlainTextEdit()
        self.evidence_view.setReadOnly(True)
        self.evidence_view.setPlaceholderText(
            "Build a preview to inspect the exact bounded evidence package before anything is sent."
        )
        self.summary_view = QTextBrowser()
        self.summary_view.setOpenExternalLinks(False)
        self.summary_view.setMarkdown(
            "# A.I. Summary\n\nSelect evidence scopes, preview them, then explicitly generate a review. "
            "No model output can change the bot."
        )
        self.tabs = QTabWidget()
        self.tabs.addTab(self.summary_view, "Validated Summary")
        self.tabs.addTab(self.evidence_view, "Exact Evidence Preview")

        self.status_label = QLabel("No external request has been made.")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)

        self._build_layout()
        self._wire()
        self._refresh_key_status()

    def _build_layout(self) -> None:
        header = SectionHeader(
            "A.I. Summary",
            "Attach ChatGPT or Claude to selected evidence for an advisory daily/research review.",
        )
        header.add_action(self.open_export_button)

        safety = QLabel(
            "ADVISORY + EXPORT ONLY · The provider receives only checked scopes after you click Generate. "
            "Output cannot change scanner scores, watchlists, alerts, bot state, or place orders."
        )
        safety.setObjectName("MutedLabel")
        safety.setWordWrap(True)

        provider_row = QHBoxLayout()
        provider_row.setContentsMargins(0, 0, 0, 0)
        provider_row.setSpacing(8)
        provider_row.addWidget(QLabel("Provider"))
        provider_row.addWidget(self.provider_input)
        provider_row.addWidget(QLabel("Model"))
        provider_row.addWidget(self.model_input, 1)

        key_row = QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        key_row.setSpacing(8)
        key_row.addWidget(self.key_input, 1)
        key_row.addWidget(self.save_key_button)
        key_row.addWidget(self.delete_key_button)
        key_row.addWidget(self.key_status)

        scopes = QFrame()
        scopes.setObjectName("InfoDrawer")
        scope_layout = QVBoxLayout(scopes)
        scope_layout.setContentsMargins(10, 8, 10, 8)
        scope_layout.setSpacing(4)
        scope_title = QLabel("What should the model review?")
        scope_title.setStyleSheet("font-weight: 700;")
        scope_layout.addWidget(scope_title)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for checkbox in self.scope_inputs.values():
            row.addWidget(checkbox)
        row.addStretch(1)
        scope_layout.addLayout(row)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        actions.addWidget(self.preview_button)
        actions.addWidget(self.generate_button)
        actions.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addWidget(safety)
        layout.addLayout(provider_row)
        layout.addLayout(key_row)
        layout.addWidget(scopes)
        layout.addLayout(actions)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status_label)

    def _wire(self) -> None:
        self.provider_input.currentIndexChanged.connect(self._on_provider_changed)
        self.model_input.editingFinished.connect(self._save_nonsecret_settings)
        for checkbox in self.scope_inputs.values():
            checkbox.toggled.connect(self._save_nonsecret_settings)
        self.save_key_button.clicked.connect(self._save_key)
        self.delete_key_button.clicked.connect(self._delete_key)
        self.preview_button.clicked.connect(self.build_preview)
        self.generate_button.clicked.connect(self.generate_summary)
        self.open_export_button.clicked.connect(self._open_last_export)
        self._runFinished.connect(self._on_run_finished)

    def _provider(self) -> str:
        return str(self.provider_input.currentData() or "openai")

    def _selected_scopes(self) -> list[str]:
        return [scope for scope, checkbox in self.scope_inputs.items() if checkbox.isChecked()]

    def _load_model_for_provider(self) -> None:
        provider = self._provider()
        model = str(
            get_local_setting(f"qt_ai_summary_model_{provider}", DEFAULT_MODELS[provider])
            or DEFAULT_MODELS[provider]
        )
        self.model_input.setText(model)

    def _save_nonsecret_settings(self, *_args) -> None:
        provider = self._provider()
        save_local_setting("qt_ai_summary_provider", provider)
        save_local_setting(f"qt_ai_summary_model_{provider}", self.model_input.text().strip())
        save_local_setting("qt_ai_summary_scopes", self._selected_scopes())

    def _on_provider_changed(self, *_args) -> None:
        self._load_model_for_provider()
        self.key_input.clear()
        self._save_nonsecret_settings()
        self._refresh_key_status()

    def _refresh_key_status(self) -> None:
        try:
            text = self.credential_vault.status(self._provider())
        except Exception as exc:
            text = f"Credential store unavailable: {exc}"
        self.key_status.setText(text)

    def _save_key(self) -> None:
        try:
            self.credential_vault.save(self._provider(), self.key_input.text())
        except Exception as exc:
            QMessageBox.warning(self, "Could Not Save Key", str(exc))
            return
        self.key_input.clear()
        self._refresh_key_status()
        self._set_status("API key saved securely; its value is not stored in app settings.")

    def _delete_key(self) -> None:
        try:
            self.credential_vault.delete(self._provider())
        except Exception as exc:
            QMessageBox.warning(self, "Could Not Delete Key", str(exc))
            return
        self.key_input.clear()
        self._refresh_key_status()
        self._set_status("Saved provider key deleted. Environment variables, if present, still take priority.")

    def _live_market_context(self) -> dict[str, Any]:
        service = self.bounce_service
        bot = service.current_bot() if service is not None else None
        if bot is None:
            return {"status": "BounceBot not connected"}
        context: dict[str, Any] = {"status": "read-only live snapshot"}
        for label, method_name in (
            ("auto_regime", "get_auto_regime_reading"),
            ("entry_assist", "entry_assist_state"),
            ("entry_board", "entry_assist_board_snapshot"),
        ):
            try:
                context[label] = getattr(bot, method_name)()
            except Exception as exc:
                context[label] = {"error": str(exc)}
        return context

    def _build_evidence(self) -> dict[str, Any]:
        return build_evidence_package(
            self._selected_scopes(),
            live_context=self._live_market_context(),
            source_overrides=self.source_overrides,
            journal_store=self.journal_store,
        )

    def build_preview(self) -> None:
        try:
            evidence = self._build_evidence()
        except Exception as exc:
            self._set_status(f"Evidence preview failed: {exc}")
            return
        self._last_evidence = evidence
        self.evidence_view.setPlainText(json.dumps(evidence, indent=2, sort_keys=True, default=str))
        self.tabs.setCurrentWidget(self.evidence_view)
        available = sum(1 for item in evidence["sources"] if item.get("status") == "available")
        self._set_status(
            f"Evidence preview ready: package {evidence['package_id']}, {available}/{len(evidence['sources'])} "
            "sources available. Nothing has been sent."
        )

    def generate_summary(self) -> None:
        if self._run_thread is not None and self._run_thread.is_alive():
            return
        typed_key = self.key_input.text().strip()
        try:
            saved_key, key_source = self.credential_vault.resolve(self._provider())
        except Exception as exc:
            self._set_status(f"Credential lookup failed: {exc}")
            return
        api_key = typed_key or saved_key
        if not api_key:
            self._set_status("No API key. Paste one for this run, save it securely, or set the provider environment variable.")
            return
        try:
            evidence = self._build_evidence()
        except Exception as exc:
            self._set_status(f"Could not build evidence: {exc}")
            return
        self._last_evidence = evidence
        provider = self._provider()
        model = self.model_input.text().strip() or DEFAULT_MODELS[provider]
        self._save_nonsecret_settings()
        self.generate_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        source_text = "one-run pasted key" if typed_key else key_source
        self._set_status(
            f"Sending package {evidence['package_id']} to {provider}/{model} using {source_text}…"
        )
        self._run_thread = threading.Thread(
            target=self._run_worker,
            args=(provider, model, api_key, evidence),
            name="ai-summary-request",
            daemon=True,
        )
        self._run_thread.start()

    def _run_worker(self, provider: str, model: str, api_key: str, evidence: dict[str, Any]) -> None:
        try:
            kwargs = {
                "provider": provider,
                "model": model,
                "api_key": api_key,
                "evidence": evidence,
            }
            if self._post is not None:
                kwargs["post"] = self._post
            result = request_ai_summary(**kwargs)
            paths = export_ai_summary(result, evidence, output_dir=self.output_dir)
            payload: dict[str, Any] = {"ok": True, "result": result, "evidence": evidence, "paths": paths}
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._runFinished.emit(payload)

    def _on_run_finished(self, payload: object) -> None:
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        data = payload if isinstance(payload, dict) else {}
        if not data.get("ok"):
            self._set_status(f"A.I. summary failed validation or request: {data.get('error') or 'unknown error'}")
            return
        result = data["result"]
        evidence = data["evidence"]
        paths = data["paths"]
        self.summary_view.setMarkdown(render_ai_summary_markdown(result, evidence))
        self.evidence_view.setPlainText(json.dumps(evidence, indent=2, sort_keys=True, default=str))
        self.tabs.setCurrentWidget(self.summary_view)
        self._last_export = Path(paths["markdown"])
        self.open_export_button.setEnabled(True)
        self.key_input.clear()
        self._set_status(
            f"Validated advisory summary exported: {self._last_export.name}. No bot state was changed."
        )

    def _open_last_export(self) -> None:
        if self._last_export is not None:
            open_path_in_file_manager(self._last_export.parent)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"A.I. Summary: {message}")

    def shutdown(self) -> None:
        # Requests are daemonized and have a bounded timeout; closing the app
        # never blocks or lets a late response mutate another panel.
        pass
