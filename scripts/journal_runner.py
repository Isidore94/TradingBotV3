from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from journal_importers import (
    IBKR_CLIENT_ID_SETTING,
    IBKR_DEFAULT_CLIENT_ID,
    IBKR_ENABLED_SETTING,
    IBKR_FLEX_QUERY_ID_SETTING,
    IBKR_FLEX_TOKEN_SETTING,
    IBKR_HOST_SETTING,
    IBKR_PORT_SETTING,
    QuestradeImporter,
    import_ibkr_executions,
    import_ibkr_flex_executions,
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


def run_journal_backfill(
    *,
    days: int = 365,
    store: JournalStore | None = None,
    include_questrade: bool = True,
    include_ibkr_flex: bool | None = None,
) -> dict[str, Any]:
    """Pull the COMPLETE trade list: Questrade executions across the whole date
    range (chunked to its 31-day API limit) and the IBKR Flex Query statement
    (the socket API only ever returns the current session's fills). Existing
    executions dedupe on execution_uid, so re-running is safe."""

    journal_store = store or JournalStore()
    messages: list[str] = []
    total_imported = 0
    had_errors = False
    end_date = date.today()
    start_date = end_date - timedelta(days=max(1, int(days)))

    qt_importer = QuestradeImporter()
    if include_questrade and (
        qt_importer.refresh_token or (qt_importer.access_token and qt_importer.api_server)
    ):
        run_id = journal_store.start_import_run("QUESTRADE_BACKFILL")
        try:
            executions, accounts = qt_importer.import_executions_for_range(start_date, end_date)
            journal_store.upsert_accounts("QUESTRADE", accounts)
            count = journal_store.upsert_executions(executions)
            journal_store.finish_import_run(
                run_id, status="OK", imported_executions=count, message=f"{start_date}..{end_date}"
            )
            total_imported += count
            messages.append(f"Questrade backfill {count}")
        except Exception as exc:
            had_errors = True
            journal_store.finish_import_run(run_id, status="FAILED", imported_executions=0, message=str(exc))
            messages.append(f"Questrade backfill failed: {exc}")
    else:
        messages.append("Questrade backfill skipped (no token).")

    if include_ibkr_flex is None:
        include_ibkr_flex = bool(
            str(get_local_setting(IBKR_FLEX_TOKEN_SETTING, "") or "").strip()
            and str(get_local_setting(IBKR_FLEX_QUERY_ID_SETTING, "") or "").strip()
        )
    if include_ibkr_flex:
        run_id = journal_store.start_import_run("IBKR_FLEX")
        try:
            executions = import_ibkr_flex_executions()
            accounts = [
                {"account_number": item.account_number, "account_label": item.account_label, "currency": item.currency}
                for item in executions
            ]
            journal_store.upsert_accounts("IBKR", accounts)
            count = journal_store.upsert_executions(executions)
            journal_store.finish_import_run(run_id, status="OK", imported_executions=count, message="flex")
            total_imported += count
            messages.append(f"IBKR flex {count}")
        except Exception as exc:
            had_errors = True
            journal_store.finish_import_run(run_id, status="FAILED", imported_executions=0, message=str(exc))
            messages.append(f"IBKR flex failed: {exc}")
    else:
        messages.append(
            "IBKR flex skipped (set the journal_ibkr_flex_token / journal_ibkr_flex_query_id "
            "local settings for complete IBKR history)."
        )

    trade_count = None
    try:
        trade_count = journal_store.rebuild_trades()
        messages.append(f"rebuilt {trade_count} grouped trades")
    except Exception as exc:
        had_errors = True
        messages.append(f"rebuild failed: {exc}")

    return {
        "status": "FAILED" if had_errors else "OK",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_imported": total_imported,
        "trade_count": trade_count,
        "messages": messages,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Journal import runner")
    parser.add_argument("--backfill-days", type=int, default=0, help="backfill this many days (0 = today only)")
    args = parser.parse_args()
    if args.backfill_days > 0:
        result = run_journal_backfill(days=args.backfill_days)
    else:
        result = run_journal_import_for_date(date.today(), trigger="cli")
    print("; ".join(result.get("messages") or []) or result.get("status"))
    return 0 if result.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
