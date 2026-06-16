from __future__ import annotations

from datetime import date
from typing import Any

from journal_importers import (
    IBKR_CLIENT_ID_SETTING,
    IBKR_DEFAULT_CLIENT_ID,
    IBKR_ENABLED_SETTING,
    IBKR_HOST_SETTING,
    IBKR_PORT_SETTING,
    QuestradeImporter,
    import_ibkr_executions,
    resolve_ibkr_client_id,
)
from journal_store import JournalStore
from project_paths import get_local_setting


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def run_journal_import_for_date(
    target_date: date,
    *,
    trigger: str = "manual",
    store: JournalStore | None = None,
    include_questrade: bool = True,
    include_ibkr: bool | None = None,
    ibkr_host: str | None = None,
    ibkr_port: int | None = None,
    ibkr_client_id: int | None = None,
) -> dict[str, Any]:
    """Import broker executions and rebuild grouped journal trades."""

    journal_store = store or JournalStore()
    messages: list[str] = []
    source_results: list[dict[str, Any]] = []
    total_imported = 0
    had_errors = False

    qt_importer = QuestradeImporter()
    if include_questrade and (
        qt_importer.refresh_token or (qt_importer.access_token and qt_importer.api_server)
    ):
        run_id = journal_store.start_import_run("QUESTRADE")
        try:
            executions, accounts = qt_importer.import_executions_for_date(target_date)
            journal_store.upsert_accounts("QUESTRADE", accounts)
            count = journal_store.upsert_executions(executions)
            journal_store.finish_import_run(
                run_id,
                status="OK",
                imported_executions=count,
                message=f"Trigger={trigger}",
            )
            total_imported += count
            messages.append(f"Questrade {count}")
            source_results.append({"source": "QUESTRADE", "status": "OK", "executions": count})
        except Exception as exc:
            had_errors = True
            journal_store.finish_import_run(
                run_id,
                status="FAILED",
                imported_executions=0,
                message=str(exc),
            )
            messages.append(f"Questrade failed: {exc}")
            source_results.append({"source": "QUESTRADE", "status": "FAILED", "message": str(exc)})
    else:
        reason = "disabled" if not include_questrade else "no token"
        messages.append(f"Questrade skipped: {reason}.")
        source_results.append({"source": "QUESTRADE", "status": "SKIPPED", "message": reason})

    if include_ibkr is None:
        include_ibkr = _coerce_bool(get_local_setting(IBKR_ENABLED_SETTING, False), default=False)

    if include_ibkr:
        run_id = journal_store.start_import_run("IBKR")
        try:
            resolved_host = str(ibkr_host or get_local_setting(IBKR_HOST_SETTING, "127.0.0.1") or "127.0.0.1")
            resolved_port = _coerce_int(
                ibkr_port if ibkr_port is not None else get_local_setting(IBKR_PORT_SETTING, 7496),
                7496,
            )
            resolved_client_id = resolve_ibkr_client_id(
                ibkr_client_id
                if ibkr_client_id is not None
                else get_local_setting(IBKR_CLIENT_ID_SETTING, IBKR_DEFAULT_CLIENT_ID)
            )
            executions = import_ibkr_executions(
                host=resolved_host,
                port=resolved_port,
                client_id=resolved_client_id,
            )
            accounts = [
                {
                    "account_number": item.account_number,
                    "account_label": item.account_label,
                    "currency": item.currency,
                }
                for item in executions
            ]
            journal_store.upsert_accounts("IBKR", accounts)
            count = journal_store.upsert_executions(executions)
            journal_store.finish_import_run(
                run_id,
                status="OK",
                imported_executions=count,
                message=f"Trigger={trigger}",
            )
            total_imported += count
            messages.append(f"IBKR {count}")
            source_results.append({"source": "IBKR", "status": "OK", "executions": count})
        except Exception as exc:
            had_errors = True
            journal_store.finish_import_run(
                run_id,
                status="FAILED",
                imported_executions=0,
                message=str(exc),
            )
            messages.append(f"IBKR failed: {exc}")
            source_results.append({"source": "IBKR", "status": "FAILED", "message": str(exc)})
    else:
        messages.append("IBKR skipped: disabled.")
        source_results.append({"source": "IBKR", "status": "SKIPPED", "message": "disabled"})

    trade_count = None
    try:
        trade_count = journal_store.rebuild_trades()
        messages.append(f"rebuilt {trade_count} grouped trades")
    except Exception as exc:
        had_errors = True
        messages.append(f"rebuild failed: {exc}")

    return {
        "status": "FAILED" if had_errors else "OK",
        "target_date": target_date.isoformat(),
        "trigger": trigger,
        "total_imported": total_imported,
        "trade_count": trade_count,
        "messages": messages,
        "source_results": source_results,
    }
