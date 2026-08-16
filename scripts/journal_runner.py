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
    flex_cash_transactions,
    flex_open_positions,
    flex_option_eae_executions,
    import_ibkr_executions,
    import_ibkr_flex_executions,
    normalize_questrade_activity,
    questrade_trade_activity_dates,
    resolve_ibkr_client_id,
)
import journal_coverage
import journal_fx
import journal_reconcile
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


def _import_questrade_activities(
    journal_store: JournalStore,
    importer: Any,
    chunk: dict[str, Any],
    *,
    messages_out: list[str],
) -> set[date]:
    """Pull one chunk's activities alongside its executions (A7).

    Activities are fees, dividends, interest and FX, plus the independent
    completeness cross-check. A broker that answers executions but not
    activities has not proved the chunk complete, so callers keep its coverage
    failed while preserving any executions already returned.
    """
    try:
        activities = importer.get_activities(chunk["account_number"], chunk["start"], chunk["end"])
    except Exception as exc:  # noqa: BLE001 - additive; never fails the chunk
        messages_out.append(f"Questrade activities {chunk['account_number']} skipped: {exc}")
        raise RuntimeError(
            f"Questrade activities cross-check unavailable for {chunk['account_number']}: {exc}"
        ) from exc
    rows = []
    for raw in activities:
        row = normalize_questrade_activity(raw, chunk["account"])
        if row is not None:
            rows.append(row)
    if rows:
        journal_store.upsert_cash_transactions(rows)
        messages_out.append(f"Questrade cash rows {len(rows)}")

    # The cross-check: a day the activities endpoint calls a trading day, on
    # which executions returned nothing, means the journal is missing trades
    # there. Saying so is the point - the executions endpoint stays
    # authoritative and nothing is imported from here.
    traded = questrade_trade_activity_dates(activities)
    imported = {
        str(
            item.get("trade_date") if isinstance(item, dict) else getattr(item, "trade_date", "")
        )[:10]
        for item in chunk.get("executions") or []
    }
    missing = sorted(day for day in traded if day.isoformat() not in imported)
    if missing:
        messages_out.append(
            f"Questrade {chunk['account_number']}: activities report trades on "
            f"{', '.join(day.isoformat() for day in missing)} that executions did not return"
        )
        for day in missing:
            journal_coverage.mark_coverage(
                journal_store,
                broker="QUESTRADE",
                account_number=chunk["account_number"],
                day=day,
                status=journal_coverage.FAILED,
                source="QT_API",
                message="activities report trades the executions endpoint did not return",
            )
    return set(missing)


def run_journal_backfill(
    *,
    days: int = 365,
    store: JournalStore | None = None,
    include_questrade: bool = True,
    include_ibkr_flex: bool | None = None,
    rebuild: bool = True,
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
        # A5: persist and mark coverage per (account, chunk). One failing chunk
        # used to discard every execution the whole pull had already fetched.
        chunk_failures = 0
        chunk_count = 0
        try:
            for chunk in qt_importer.iter_execution_chunks(start_date, end_date):
                chunk_count += 1
                account_number = chunk["account_number"]
                run_id = journal_store.start_import_run(
                    "QUESTRADE_BACKFILL",
                    account_number=account_number,
                    trigger="backfill",
                    coverage_start=chunk["start"],
                    coverage_end=chunk["end"],
                )
                if "error" in chunk:
                    chunk_failures += 1
                    had_errors = True
                    journal_store.finish_import_run(
                        run_id, status="FAILED", imported_executions=0, message=chunk["error"]
                    )
                    journal_coverage.mark_range(
                        journal_store,
                        broker="QUESTRADE",
                        account_number=account_number,
                        start=chunk["start"],
                        end=chunk["end"],
                        status=journal_coverage.FAILED,
                        source="QT_API",
                        import_run_id=run_id,
                        message=chunk["error"],
                    )
                    messages.append(
                        f"Questrade {account_number} {chunk['start']}..{chunk['end']} failed: {chunk['error']}"
                    )
                    continue
                journal_store.upsert_accounts("QUESTRADE", [chunk["account"]])
                count = journal_store.upsert_executions(chunk["executions"])
                quarantined = list(chunk.get("quarantined") or [])
                chunk_status = "FAILED" if quarantined else "OK"
                coverage_status = journal_coverage.FAILED if quarantined else journal_coverage.COVERED
                detail = (
                    f"{len(quarantined)} unreadable row(s) quarantined"
                    if quarantined else f"{count} execution(s)"
                )
                journal_store.finish_import_run(
                    run_id, status=chunk_status, imported_executions=count, message=detail
                )
                journal_coverage.mark_range(
                    journal_store, broker="QUESTRADE", account_number=account_number,
                    start=chunk["start"], end=chunk["end"], status=coverage_status,
                    source="QT_API", import_run_id=run_id, message=detail,
                )
                if quarantined:
                    chunk_failures += 1
                    had_errors = True
                # After the chunk is marked, never before: the cross-check below
                # can downgrade a day to FAILED, and marking COVERED afterwards
                # would paint over exactly the disagreement it just found.
                try:
                    _import_questrade_activities(
                        journal_store, qt_importer, chunk, messages_out=messages
                    )
                except RuntimeError as exc:
                    journal_store.finish_import_run(
                        run_id, status="PARTIAL", imported_executions=count, message=str(exc)
                    )
                    journal_coverage.mark_range(
                        journal_store, broker="QUESTRADE", account_number=account_number,
                        start=chunk["start"], end=chunk["end"], status=journal_coverage.FAILED,
                        source="QT_API", import_run_id=run_id, message=str(exc),
                    )
                total_imported += count
            if qt_importer.quarantined:
                messages.append(f"Questrade quarantined {len(qt_importer.quarantined)} unreadable row(s)")
            messages.append(
                f"Questrade backfill {total_imported} over {chunk_count} chunk(s)"
                + (f", {chunk_failures} failed" if chunk_failures else "")
            )
        except Exception as exc:
            # Only reachable before any chunk exists - account discovery itself
            # failing. No day is marked, because none was attempted.
            had_errors = True
            messages.append(f"Questrade backfill failed: {exc}")
    else:
        messages.append("Questrade backfill skipped (no token).")

    if include_ibkr_flex is None:
        include_ibkr_flex = bool(
            str(get_local_setting(IBKR_FLEX_TOKEN_SETTING, "") or "").strip()
            and str(get_local_setting(IBKR_FLEX_QUERY_ID_SETTING, "") or "").strip()
        )
    if include_ibkr_flex:
        run_id = journal_store.start_import_run(
            "IBKR_FLEX", trigger="backfill", coverage_start=start_date, coverage_end=end_date
        )
        try:
            statement = import_ibkr_flex_executions(with_metadata=True)
            # Option expiries, exercises and assignments are fills too. Without
            # them an option that expired worthless has no closing execution and
            # the position stays open forever (A7).
            eae = flex_option_eae_executions(statement.get("option_eae") or [])
            executions = list(statement["executions"]) + eae
            cash_rows = flex_cash_transactions(statement.get("cash_transactions") or [])
            if cash_rows:
                journal_store.upsert_cash_transactions(cash_rows)
                messages.append(f"IBKR cash rows {len(cash_rows)}")
            if eae:
                messages.append(f"IBKR option expiries/assignments {len(eae)}")
            accounts = [
                {"account_number": item.account_number, "account_label": item.account_label, "currency": item.currency}
                for item in executions
            ]
            journal_store.upsert_accounts("IBKR", accounts)
            count = journal_store.upsert_executions(executions)
            quarantined = list(statement.get("quarantined") or [])
            journal_store.finish_import_run(
                run_id, status=("FAILED" if quarantined else "OK"), imported_executions=count,
                message=(f"flex; {len(quarantined)} unreadable row(s) quarantined" if quarantined else "flex"),
            )
            if quarantined:
                had_errors = True
                messages.append(f"IBKR Flex quarantined {len(quarantined)} unreadable row(s)")
            total_imported += count
            messages.append(f"IBKR flex {count}")

            # I2: coverage comes from the statement's own declared span, not
            # from the range this function was asked for. A Flex query set to
            # "last 365 days" does not prove anything about day 366, and the
            # 365-day service cap means the two often disagree.
            span_start = statement.get("from_date")
            span_end = statement.get("to_date")
            if span_start and span_end:
                for account_number in sorted(set(statement.get("accounts") or [])):
                    journal_coverage.mark_range(
                        journal_store, broker="IBKR", account_number=account_number,
                        start=span_start, end=span_end,
                        status=(journal_coverage.FAILED if quarantined else journal_coverage.COVERED),
                        source="IBKR_FLEX", import_run_id=run_id,
                        message=(f"{len(quarantined)} unreadable row(s) quarantined"
                                 if quarantined else "flex statement span"),
                    )
            else:
                had_errors = True
                messages.append("IBKR Flex coverage not marked: statement declared no span")
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
    # `rebuild=False` is for the nightly path, which heals its gaps first and
    # then rebuilds once. Assembling here as well would build the journal twice
    # a night, and the first of the two from a set of executions already known
    # to have holes in it.
    if rebuild:
        try:
            needed = sorted(
                set(journal_fx.rates_needed_for_trades(journal_store))
                | set(journal_fx.rates_needed_for_executions(journal_store))
            )
            rates = journal_fx.ensure_rates(journal_store, needed)
            if rates["errors"] or rates["unavailable"]:
                messages.append(
                    f"FX: {len(rates['unavailable'])} unavailable, {len(rates['errors'])} error(s)"
                )
            trade_count = journal_store.rebuild_trades()
            journal_store.book_cad_values()
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


#: How many days back the nightly ranged pull re-reads. Seven, not one: a
#: broker can amend or late-report a fill for days after the fact, and a nightly
#: job that only ever looked at yesterday would never see the amendment.
NIGHTLY_LOOKBACK_DAYS = 7


def run_nightly_journal_import(*, store: JournalStore | None = None, trigger: str = "nightly") -> dict[str, Any]:
    """The overnight journal pull: import, heal, convert, rebuild, reconcile.

    Spec §6, and the promotion of the queued P3.3 slice. The order is the order:
    a rebuild before the self-heal would assemble a journal with known holes in
    it, and a reconciliation before the rebuild would compare against trades
    that the night's imports had already invalidated.

    **A night with no executions is `ok`.** A quiet market is not a failure, and
    a job that reported one would teach the trader to ignore it.

    No new timer, no new thread, no new ntfy sender (I8). This runs inside the
    existing `ai_jobs` runner slot and its only outputs are database rows and
    the ledger entry the runner writes.
    """
    journal_store = store or JournalStore()
    messages: list[str] = []
    had_errors = False
    end_date = date.today()
    start_date = end_date - timedelta(days=NIGHTLY_LOOKBACK_DAYS)

    backfill = run_journal_backfill(days=NIGHTLY_LOOKBACK_DAYS, store=journal_store, rebuild=False)
    messages.extend(backfill.get("messages") or [])
    had_errors = had_errors or backfill.get("status") == "FAILED"

    # Self-heal before the rebuild: assembling a journal that is known to have
    # holes in it produces trades that are wrong in a way no later step can see.
    try:
        healed = journal_coverage.self_heal(
            journal_store,
            lambda broker, account, day: _fetch_one_day(journal_store, broker, account, day),
            today=end_date,
        )
        if healed["repaired"] or healed["failed"] or healed["exhausted"]:
            messages.append(
                f"self-heal repaired {len(healed['repaired'])}, failed {len(healed['failed'])}, "
                f"exhausted {len(healed['exhausted'])}"
            )
    except Exception as exc:  # noqa: BLE001 - one bad night is not a broken journal
        had_errors = True
        messages.append(f"self-heal failed: {exc}")

    try:
        rates = journal_fx.ensure_rates(
            journal_store,
            sorted(
                set(journal_fx.rates_needed_for_trades(journal_store))
                | set(journal_fx.rates_needed_for_executions(journal_store))
            ),
        )
        if rates["booked"] or rates["errors"] or rates["unavailable"]:
            messages.append(
                f"fx booked {rates['booked']}"
                + (f", carried back {rates['carried_back']}" if rates["carried_back"] else "")
                + (f", {len(rates['unavailable'])} unavailable" if rates["unavailable"] else "")
                + (f", {len(rates['errors'])} error(s)" if rates["errors"] else "")
            )
    except Exception as exc:  # noqa: BLE001 - unconverted is an honest state
        messages.append(f"fx booking failed: {exc}")

    trade_count = None
    try:
        trade_count = journal_store.rebuild_trades()
        journal_store.book_cad_values()
        messages.append(f"rebuilt {trade_count} trades")
        rekey = journal_store.last_rekey
        if rekey.get("ambiguous") or rekey.get("orphaned"):
            messages.append(
                f"re-key needs review: {len(rekey.get('ambiguous') or [])} ambiguous, "
                f"{len(rekey.get('orphaned') or [])} orphaned"
            )
    except Exception as exc:  # noqa: BLE001
        had_errors = True
        messages.append(f"rebuild failed: {exc}")

    try:
        positions, reachable = _broker_positions(journal_store, messages)
        if reachable:
            report = journal_reconcile.reconcile(
                journal_store, positions, brokers=reachable, trigger=trigger
            )
            messages.append(
                f"reconciled {report['positions_checked']} position(s), "
                f"{len(report['mismatched'])} mismatch(es)"
            )
        else:
            messages.append("reconcile skipped: no broker position source was reachable")
    except Exception as exc:  # noqa: BLE001
        had_errors = True
        messages.append(f"reconcile failed: {exc}")

    return {
        "status": "FAILED" if had_errors else "OK",
        "ok": not had_errors,
        "trigger": trigger,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "trade_count": trade_count,
        "messages": messages,
    }


def _fetch_one_day(journal_store: JournalStore, broker: str, account_number: str, day: date) -> int:
    """Import a single (broker, account, day) for the self-heal callback.

    IBKR is deliberately unsupported here: the socket sees only the current TWS
    session, and a Flex pull is a whole-statement operation rather than a
    per-day one. Raising says so, which marks the day FAILED with a readable
    reason instead of pretending the gap was repaired.
    """
    if str(broker).upper() != "QUESTRADE":
        raise RuntimeError(f"{broker} has no per-day fetch; its coverage comes from the Flex statement")
    importer = QuestradeImporter()
    total = 0
    found_account = False
    for chunk in importer.iter_execution_chunks(day, day):
        if chunk["account_number"] != account_number:
            continue
        found_account = True
        if "error" in chunk:
            raise RuntimeError(chunk["error"])
        if chunk.get("quarantined"):
            raise RuntimeError(f"{len(chunk['quarantined'])} unreadable execution row(s) quarantined")
        journal_store.upsert_accounts("QUESTRADE", [chunk["account"]])
        total += journal_store.upsert_executions(chunk["executions"])
        missing = _import_questrade_activities(
            journal_store, importer, chunk, messages_out=[]
        )
        if day in missing:
            raise RuntimeError("activities report trades the executions endpoint did not return")
    if not found_account:
        raise RuntimeError(f"Questrade did not return account {account_number}")
    return total


def _broker_positions(
    journal_store: JournalStore, messages: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Current positions from every configured broker, for reconciliation.

    A broker that cannot be reached contributes nothing and says so. It is not
    scoped out of the comparison silently, because "the broker holds nothing"
    and "we could not ask" must never look the same - the first is a mismatch
    worth flagging and the second is not evidence at all.
    """
    positions: list[dict[str, Any]] = []
    reachable: list[str] = []

    importer = QuestradeImporter()
    if importer.refresh_token or (importer.access_token and importer.api_server):
        try:
            for account in importer.get_accounts():
                number = str(account.get("number") or account.get("accountNumber") or "").strip()
                if number:
                    positions.extend(importer.get_positions(number))
            reachable.append("QUESTRADE")
        except Exception as exc:  # noqa: BLE001
            messages.append(f"Questrade positions unavailable: {exc}")

    try:
        statement = import_ibkr_flex_executions(with_metadata=True)
        positions.extend(flex_open_positions(statement.get("open_positions") or []))
        reachable.append("IBKR")
    except Exception as exc:  # noqa: BLE001
        messages.append(f"IBKR positions unavailable: {exc}")

    return positions, reachable


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Journal import runner")
    parser.add_argument("--backfill-days", type=int, default=0, help="backfill this many days (0 = today only)")
    parser.add_argument("--nightly", action="store_true", help="run the nightly slot's work once")
    args = parser.parse_args()
    if args.nightly:
        result = run_nightly_journal_import(trigger="cli")
    elif args.backfill_days > 0:
        result = run_journal_backfill(days=args.backfill_days)
    else:
        result = run_journal_import_for_date(date.today(), trigger="cli")
    print("; ".join(result.get("messages") or []) or result.get("status"))
    return 0 if result.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
