from __future__ import annotations

import logging
import threading
import time
from copy import deepcopy

from . import legacy as _legacy

# Scanner orchestration is extracted while helper functions continue to migrate.
globals().update(
    {
        name: value
        for name, value in vars(_legacy).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def _log_phase_duration(label: str, since: float) -> float:
    """Log wall-clock seconds elapsed for a run_master phase; returns a fresh mark."""
    now = time.perf_counter()
    logging.info("[run_master timing] %-26s %6.1fs", label, now - since)
    return now


def run_master_with_shared_watchlists():
    return run_master(use_shared_watchlists=True)


_theta_enrichment_state_lock = threading.Lock()
_latest_theta_enrichment_run_id = ""


def _mark_latest_theta_enrichment_run(run_id: str) -> None:
    global _latest_theta_enrichment_run_id
    with _theta_enrichment_state_lock:
        _latest_theta_enrichment_run_id = str(run_id or "")


# The playbook study (hypothesis-family backfill + AI digest) refreshes weekly,
# piggybacking on the after-close tracker window so the digest the AI reads is
# never more than a week stale. A full run is ~2-3 minutes over the durable
# daily-bar store and needs no IBKR session.
PLAYBOOK_STUDY_REFRESH_DAYS = 7
PLAYBOOK_STUDY_BACKFILL_SESSIONS = 60


def refresh_playbook_study_if_stale(max_age_days: int = PLAYBOOK_STUDY_REFRESH_DAYS) -> bool:
    from datetime import datetime as _dt

    from setup_playbook_study import PLAYBOOK_AI_DIGEST_JSON, run_playbook_study

    try:
        age_days = (_dt.now() - _dt.fromtimestamp(PLAYBOOK_AI_DIGEST_JSON.stat().st_mtime)).days
        if age_days < max_age_days:
            return False
    except OSError:
        pass  # no digest yet -> build the first one
    logging.info("Refreshing setup playbook study (digest missing or stale)...")
    result = run_playbook_study(days=PLAYBOOK_STUDY_BACKFILL_SESSIONS, write_outputs=True)
    logging.info(
        "Playbook study refreshed: %s episodes, %s actionable combo(s) in the AI digest.",
        len(result["episodes"]),
        len(result["digest"]["actionable"]),
    )
    return True


def _theta_enrichment_run_is_latest(run_id: str) -> bool:
    with _theta_enrichment_state_lock:
        return str(run_id or "") == _latest_theta_enrichment_run_id


def _write_master_avwap_output_file(sorted_events: list, range_buckets: dict) -> None:
    output_buffer = io.StringIO()
    f = output_buffer
    priority_text = PRIORITY_SETUPS_FILE.read_text(encoding="utf-8").strip()
    upgrade_alerts_text = (
        MASTER_AVWAP_D1_UPGRADE_ALERTS_REPORT_FILE.read_text(encoding="utf-8").strip()
        if MASTER_AVWAP_D1_UPGRADE_ALERTS_REPORT_FILE.exists()
        else ""
    )
    theta_text = THETA_PUTS_FILE.read_text(encoding="utf-8").strip()
    if priority_text:
        f.write(priority_text)
        f.write("\n\n")
    if upgrade_alerts_text:
        f.write(upgrade_alerts_text)
        f.write("\n\n")
    if theta_text:
        f.write("MASTER AVWAP THETA PLAYS\n")
        f.write("=" * 80)
        f.write("\n")
        f.write(theta_text)
        f.write("\n\n")
    for s, d, lbl, side in sorted_events:
        f.write(f"{s},{d},{lbl},{side}\n")

    def _write_range_line(label, tickers):
        items = ", ".join(sorted(set(tickers))) if tickers else "None"
        f.write(f"{label}: {items}\n")

    f.write("\nPrice ranges (current anchors):\n")
    _write_range_line(
        "Longs between AVWAP and UPPER_1",
        range_buckets["long_avwap_to_upper_1"],
    )
    _write_range_line(
        "Longs between UPPER_1 and UPPER_2",
        range_buckets["long_upper_1_to_upper_2"],
    )
    _write_range_line(
        "Shorts between AVWAP and LOWER_1",
        range_buckets["short_avwap_to_lower_1"],
    )
    _write_range_line(
        "Shorts between LOWER_1 and LOWER_2",
        range_buckets["short_lower_1_to_lower_2"],
    )
    f.write(f"\nRun completed at {datetime.now().strftime('%H:%M:%S')}\n")
    _write_text_atomic(OUTPUT_FILE, output_buffer.getvalue().rstrip() + "\n")


def _copy_range_buckets_for_background(range_buckets: dict) -> dict:
    return {key: list(value or []) for key, value in (range_buckets or {}).items()}


def _refresh_ai_state_theta_candidate(ai_state: dict, row: dict, key: str) -> None:
    symbol = str(row.get("symbol") or "").strip().upper()
    if not symbol:
        return
    symbols = ai_state.get("symbols") if isinstance(ai_state, dict) else None
    if not isinstance(symbols, dict):
        return
    entry = symbols.get(symbol)
    if isinstance(entry, dict):
        entry[key] = row


def _run_deferred_theta_enrichment(
    *,
    run_id: str,
    theta_put_rows: list[dict],
    theta_pcs_rows: list[dict],
    priority_rows: list[dict],
    ai_state: dict,
    sorted_events: list,
    range_buckets: dict,
    reference_date,
    favorite_watchlist_context: dict,
) -> None:
    theta_ib = None
    theta_ib_owned = False
    sold_put_rows_all = list(theta_put_rows or [])
    pcs_rows_all = list(theta_pcs_rows or [])
    sold_put_rows = list(sold_put_rows_all)
    pcs_rows = list(pcs_rows_all)
    try:
        logging.info(
            "Deferred theta option premium enrichment starting for run %s: %d sold-put row(s), %d PCS row(s).",
            run_id,
            len(sold_put_rows),
            len(pcs_rows),
        )
        theta_ib, theta_ib_owned = ensure_theta_option_data_client(None)
        try:
            enrich_theta_rows_with_ib_option_premiums(theta_ib, sold_put_rows, pcs_rows, reference_date)
        finally:
            if theta_ib_owned:
                disconnect_daily_data_client(theta_ib)
                theta_ib = None

        for row in sold_put_rows_all:
            _refresh_ai_state_theta_candidate(ai_state, row, "theta_put_candidate")
        for row in pcs_rows_all:
            _refresh_ai_state_theta_candidate(ai_state, row, "theta_pcs_candidate")

        if not _theta_enrichment_run_is_latest(run_id):
            logging.info(
                "Deferred theta enrichment for run %s finished but skipped file writes because a newer scan is active.",
                run_id,
            )
            return

        write_theta_put_report(THETA_PUTS_FILE, sold_put_rows, pcs_rows)
        save_json(AI_STATE_FILE, ai_state)

        if favorite_watchlist_context.get("allowed"):
            write_favorite_zone_watchlist_outputs(
                focus_path=MASTER_AVWAP_FOCUS_FILE,
                d1_watchlist_path=MASTER_AVWAP_D1_WATCHLIST_FILE,
                priority_rows=priority_rows,
                theta_put_rows=sold_put_rows,
                theta_pcs_rows=pcs_rows,
                ai_state=ai_state,
                now=favorite_watchlist_context.get("reference_time"),
                window_start=favorite_watchlist_context.get("window_start"),
                window_end=favorite_watchlist_context.get("window_end"),
            )

        _write_master_avwap_output_file(sorted_events, range_buckets)
        logging.info(
            "Deferred theta option premium enrichment finished for run %s: %d sold-put row(s), %d PCS row(s).",
            run_id,
            len(sold_put_rows),
            len(pcs_rows),
        )
    except Exception:
        logging.exception("Deferred theta option premium enrichment failed for run %s.", run_id)
    finally:
        if theta_ib is not None and theta_ib_owned:
            disconnect_daily_data_client(theta_ib)


def _schedule_deferred_theta_enrichment(
    *,
    run_id: str,
    theta_put_rows: list[dict],
    theta_pcs_rows: list[dict],
    priority_rows: list[dict],
    ai_state: dict,
    sorted_events: list,
    range_buckets: dict,
    reference_date,
    favorite_watchlist_context: dict,
) -> bool:
    if not (theta_put_rows or theta_pcs_rows):
        logging.info("No theta rows found; no deferred option premium enrichment needed.")
        return False

    worker = threading.Thread(
        target=_run_deferred_theta_enrichment,
        kwargs={
            "run_id": run_id,
            "theta_put_rows": deepcopy(theta_put_rows),
            "theta_pcs_rows": deepcopy(theta_pcs_rows),
            "priority_rows": deepcopy(priority_rows),
            "ai_state": deepcopy(ai_state),
            "sorted_events": list(sorted_events or []),
            "range_buckets": _copy_range_buckets_for_background(range_buckets),
            "reference_date": reference_date,
            "favorite_watchlist_context": dict(favorite_watchlist_context or {}),
        },
        name=f"theta-option-enrichment-{run_id}",
        daemon=False,
    )
    worker.start()
    logging.info(
        "Theta option strike/price enrichment deferred in background for run %s (%d sold-put row(s), %d PCS row(s)).",
        run_id,
        len(theta_put_rows or []),
        len(theta_pcs_rows or []),
    )
    return True

# ============================================================================
# CACHE HELPERS
# ============================================================================

def run_master(
    longs_path: Path | None = None,
    shorts_path: Path | None = None,
    use_shared_watchlists: bool = False,
    update_setup_tracker: bool | None = None,
    require_ib_for_setup_tracker: bool = False,
):
    _run_t0 = time.perf_counter()
    _phase_t = _run_t0
    reset_ibkr_historical_failure_circuit()
    long_paths, short_paths, watchlist_label = resolve_master_scan_watchlist_paths(
        longs_path=longs_path,
        shorts_path=shorts_path,
        use_shared_watchlists=use_shared_watchlists,
    )
    logging.info(
        f"Running Master AVWAP scan using {watchlist_label}: "
        f"{_format_watchlist_path_group(long_paths)} | {_format_watchlist_path_group(short_paths)}"
    )

    optional_paths = {SWING_LONGS_FILE, SWING_SHORTS_FILE, UNIVERSE_LONGS_FILE, UNIVERSE_SHORTS_FILE}
    longs = load_tickers_from_paths(long_paths, optional_paths=optional_paths)
    shorts = load_tickers_from_paths(short_paths, optional_paths=optional_paths)
    longs, shorts, d1_watchlist_added = append_master_avwap_d1_watchlist_symbols(longs, shorts)
    symbols = sorted(set(longs + shorts))
    if d1_watchlist_added:
        logging.info("Master AVWAP scan added %s symbol(s) from the D1 development watchlist.", d1_watchlist_added)

    if not symbols:
        logging.warning(
            f"No symbols found in {watchlist_label}. Skipping Master AVWAP scan."
        )
        write_theta_put_report(THETA_PUTS_FILE, [])
        return {
            "watchlist_label": watchlist_label,
            "tracked_rows": [],
            "theta_put_rows": [],
            "theta_pcs_rows": [],
            "ai_state": {},
            "feature_rows_by_symbol": {},
            "daily_frames_by_symbol": {},
            "d1_watchlist_scan_symbols_added": 0,
            "setup_tracker_updated": False,
            "setup_tracker_allowed": False,
            "setup_tracker_skip_reason": "No symbols were available for the scan.",
        }

    curr_cache = load_json(CURRENT_CACHE_FILE, default={})
    prev_cache = load_json(PREV_CACHE_FILE, default={})
    history = load_history()

    earnings_data, latest_release_map = load_scan_earnings_context(symbols)
    upcoming_earnings_map = collect_upcoming_earnings_dates(symbols)
    today_iso = datetime.now().date().isoformat()
    earnings_cache_updated = False

    logging.info(f"Refreshing earnings anchors for {len(symbols)} symbols…")
    refreshed_curr = 0
    refreshed_prev = 0
    missing_anchors = []

    for sym in symbols:
        dates = earnings_data.get(sym, [])

        if not dates:
            missing_anchors.append(sym)
            continue

        current_anchor = pick_current_earnings_anchor(dates)
        if current_anchor:
            curr_iso = current_anchor.isoformat()
            if curr_cache.get(sym) != curr_iso:
                curr_cache[sym] = curr_iso
                refreshed_curr += 1
                logging.info(f"{sym}: current anchor -> {current_anchor}")

        previous_anchor = pick_previous_earnings_anchor(dates)
        if previous_anchor:
            prev_iso = previous_anchor.isoformat()
            if prev_cache.get(sym) != prev_iso:
                prev_cache[sym] = prev_iso
                refreshed_prev += 1
                logging.info(f"{sym}: previous anchor -> {previous_anchor}")
        elif sym in prev_cache:
            prev_cache.pop(sym, None)
            refreshed_prev += 1
            logging.info(f"{sym}: previous anchor cleared; no older earnings date available.")

    if earnings_cache_updated:
        earnings_cache = load_earnings_date_cache()
        symbol_cache = earnings_cache.setdefault("symbols", {})
        for sym, dates in earnings_data.items():
            if not dates:
                continue
            entry = symbol_cache.setdefault(sym, _normalize_earnings_cache_entry({}))
            entry["dates"] = _merge_earnings_dates(entry.get("dates", []), dates)
            entry["last_yf_refresh_on"] = today_iso
        save_earnings_date_cache(earnings_cache)

    if missing_anchors:
        logging.warning(
            "No earnings data found for: " + ", ".join(sorted(missing_anchors))
        )
    logging.info(
        f"Earnings anchors refreshed (current: {refreshed_curr}, previous: {refreshed_prev})."
    )

    ib = connect_daily_data_client(client_id=1003, startup_wait=1.5)

    today_run = datetime.now().date()
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    run_id = f"{today_run.isoformat()}-{datetime.now().strftime('%H%M%S')}"
    _mark_latest_theta_enrichment_run(run_id)
    scoring_config_metadata = get_scoring_config_metadata()
    market_regime_snapshot = build_market_regime_snapshot(ib, today_run)
    events_for_output = []
    csv_rows = []
    feature_rows = []
    feature_rows_by_symbol = {}
    daily_frames_by_symbol = {}
    priority_rows = []
    htf_trend_study_rows = []
    hv_level_study_rows = []
    phase6_study_rows = []
    relative_avwap_study_rows = []
    first_dev_breakout_study_rows = []
    second_dev_breakout_study_rows = []
    theta_put_rows = []
    theta_pcs_rows = []
    positions = {
        "current": {lvl: [] for lvl in POSITION_LEVELS},
        "previous": {lvl: [] for lvl in POSITION_LEVELS},
    }
    range_buckets = {
        "long_avwap_to_upper_1": [],
        "long_upper_1_to_upper_2": [],
        "short_avwap_to_lower_1": [],
        "short_lower_1_to_lower_2": [],
    }
    market_prep_range_buckets = {
        "long_upper_2_to_upper_3_2_sessions": [],
    }
    stdev_range_hits = {"long": [], "short": []}
    stdev_cross_hits = {"long": [], "short": []}
    ai_state = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "run_date": today_run.isoformat(),
        "scoring_config": scoring_config_metadata,
        "market_regime": market_regime_snapshot,
        "symbols": {}
    }

    for sym in symbols:
        side = "LONG" if sym in longs else "SHORT"
        curr_iso = curr_cache.get(sym)
        prev_iso = prev_cache.get(sym)

        if not curr_iso and not prev_iso:
            logging.warning(f"{sym}: no earnings anchors available.")
            continue

        # Determine days needed for a single daily fetch
        days_needed = ATR_LENGTH + 5
        anchor_dates = []
        if curr_iso:
            anchor_dates.append(datetime.fromisoformat(curr_iso).date())
        if prev_iso:
            anchor_dates.append(datetime.fromisoformat(prev_iso).date())

        if anchor_dates:
            max_span = max((today_run - d).days for d in anchor_dates)
            days_needed = max(days_needed, max_span + 5)

        df = fetch_daily_bars(ib, sym, days_needed)
        if df.empty:
            logging.warning(f"{sym}: no daily bars returned.")
            continue
        daily_bar_source = _get_daily_bar_source(df)
        daily_frames_by_symbol[sym] = df.copy()

        last_trade_date = df["datetime"].iloc[-1].date()
        dstr = df["datetime"].iloc[-1].strftime("%m/%d")
        next_earnings_summary = _next_earnings_window_summary(
            last_trade_date,
            upcoming_earnings_map.get(sym, []),
        )

        logging.info(f"-> Processing {sym} ({side}) with {len(df)} daily bars; last date {last_trade_date}")

        symbol_events_today = []
        symbol_multi_day = []
        current_anchor_meta = None
        prev_anchor_meta = None
        symbol_signal_info = {}
        skip_current_events = False

        recent_earnings_dates = earnings_data.get(sym, [])
        if recent_earnings_dates:
            try:
                last_earnings_date = datetime.fromisoformat(recent_earnings_dates[0]).date()
                sessions_since_last_earnings = sessions_since_date(df, last_earnings_date)
                if (
                    sessions_since_last_earnings is not None
                    and sessions_since_last_earnings <= RECENT_EARNINGS_SESSION_BLOCK
                ):
                    skip_current_events = True
                    logging.info(
                        f"{sym}: skipping CURRENT AVWAPE events; last earnings {last_earnings_date} "
                        f"was {sessions_since_last_earnings} session(s) ago (<= {RECENT_EARNINGS_SESSION_BLOCK})."
                    )
            except ValueError:
                logging.warning(f"{sym}: invalid recent earnings date format: {recent_earnings_dates[0]}")

        def add_signal(event_name, anchor_type, anchor_date, avwap_value, stdev_value, band_value):
            symbol_events_today.append(event_name)
            if event_name not in symbol_signal_info:
                symbol_signal_info[event_name] = {
                    "run_date": today_run.isoformat(),
                    "symbol": sym,
                    "trade_date": last_trade_date.isoformat(),
                    "side": side,
                    "anchor_type": anchor_type,
                    "anchor_date": anchor_date,
                    "signal_type": event_name,
                    "avwap_price": _to_float(avwap_value),
                    "band_price": _to_float(band_value),
                    "stdev": _to_float(stdev_value),
                    "priority_bucket": "",
                    "is_favorite_setup": False,
                    "is_near_favorite_zone": False,
                    "favorite_zone": "",
                    "favorite_signals": "",
                    "favorite_context_signals": "",
                }

        # Current earnings AVWAP
        if curr_iso:
            curr_date = datetime.fromisoformat(curr_iso).date()
            idxs = df.index[df["datetime"].dt.date == curr_date]
            if not idxs.empty:
                anchor_idx = int(idxs[0])
                vwap_c, sd_c, bands_c = calc_anchored_vwap_bands(df, anchor_idx)
                if pd.notna(vwap_c) and bands_c:
                    current_anchor_meta = {
                        "date": curr_iso,
                        "vwap": float(vwap_c),
                        "stdev": float(sd_c),
                        "bands": {k: float(v) for k, v in bands_c.items()}
                    }
                    if not skip_current_events:
                        primary_cross = select_primary_cross_signal(
                            df,
                            side,
                            "",
                            vwap_c,
                            bands_c,
                        )
                        if primary_cross:
                            lbl, lvl = primary_cross
                            add_signal(lbl, "CURRENT", curr_iso, vwap_c, sd_c, lvl)

                        # bounces (current)
                        if side == "LONG":
                            bounce_tests = [
                                ("BOUNCE_VWAP", vwap_c),
                                ("BOUNCE_LOWER_1", bands_c["LOWER_1"]),
                                ("BOUNCE_LOWER_2", bands_c["LOWER_2"]),
                                ("BOUNCE_LOWER_3", bands_c["LOWER_3"]),
                                ("BOUNCE_UPPER_1", bands_c["UPPER_1"]),
                                ("BOUNCE_UPPER_2", bands_c["UPPER_2"]),
                                ("BOUNCE_UPPER_3", bands_c["UPPER_3"]),
                            ]
                            for lbl, lvl in bounce_tests:
                                if bounce_up_at_level(df, lvl):
                                    add_signal(lbl, "CURRENT", curr_iso, vwap_c, sd_c, lvl)
                        else:
                            bounce_tests = [
                                ("BOUNCE_VWAP", vwap_c),
                                ("BOUNCE_UPPER_1", bands_c["UPPER_1"]),
                                ("BOUNCE_UPPER_2", bands_c["UPPER_2"]),
                                ("BOUNCE_UPPER_3", bands_c["UPPER_3"]),
                                ("BOUNCE_LOWER_1", bands_c["LOWER_1"]),
                                ("BOUNCE_LOWER_2", bands_c["LOWER_2"]),
                                ("BOUNCE_LOWER_3", bands_c["LOWER_3"]),
                            ]
                            for lbl, lvl in bounce_tests:
                                if bounce_down_at_level(df, lvl):
                                    add_signal(lbl, "CURRENT", curr_iso, vwap_c, sd_c, lvl)

                else:
                    logging.warning(f"{sym}: invalid current AVWAP / bands.")
            else:
                logging.warning(f"{sym}: no candle on current earnings date {curr_date}.")

        # Previous earnings AVWAP
        if prev_iso:
            prev_date = datetime.fromisoformat(prev_iso).date()
            idxs = df.index[df["datetime"].dt.date == prev_date]
            if not idxs.empty:
                anchor_idx = int(idxs[0])
                vwap_p, sd_p, bands_p = calc_anchored_vwap_bands(df, anchor_idx)
                if pd.notna(vwap_p) and bands_p:
                    prev_anchor_meta = {
                        "date": prev_iso,
                        "vwap": float(vwap_p),
                        "stdev": float(sd_p),
                        "bands": {k: float(v) for k, v in bands_p.items()}
                    }

                    # previous bounces
                    if side == "LONG":
                        prev_bounce_tests = [
                            ("PREV_BOUNCE_VWAP", vwap_p),
                            ("PREV_BOUNCE_LOWER_1", bands_p.get("LOWER_1")),
                            ("PREV_BOUNCE_LOWER_2", bands_p.get("LOWER_2")),
                            ("PREV_BOUNCE_LOWER_3", bands_p.get("LOWER_3")),
                            ("PREV_BOUNCE_UPPER_1", bands_p.get("UPPER_1")),
                            ("PREV_BOUNCE_UPPER_2", bands_p.get("UPPER_2")),
                            ("PREV_BOUNCE_UPPER_3", bands_p.get("UPPER_3")),
                        ]
                        for lbl, lvl in prev_bounce_tests:
                            if bounce_up_at_level(df, lvl):
                                add_signal(lbl, "PREVIOUS", prev_iso, vwap_p, sd_p, lvl)
                    else:
                        prev_bounce_tests = [
                            ("PREV_BOUNCE_VWAP", vwap_p),
                            ("PREV_BOUNCE_UPPER_1", bands_p.get("UPPER_1")),
                            ("PREV_BOUNCE_UPPER_2", bands_p.get("UPPER_2")),
                            ("PREV_BOUNCE_UPPER_3", bands_p.get("UPPER_3")),
                            ("PREV_BOUNCE_LOWER_1", bands_p.get("LOWER_1")),
                            ("PREV_BOUNCE_LOWER_2", bands_p.get("LOWER_2")),
                            ("PREV_BOUNCE_LOWER_3", bands_p.get("LOWER_3")),
                        ]
                        for lbl, lvl in prev_bounce_tests:
                            if bounce_down_at_level(df, lvl):
                                add_signal(lbl, "PREVIOUS", prev_iso, vwap_p, sd_p, lvl)

                    # previous crosses
                    primary_prev_cross = select_primary_cross_signal(
                        df,
                        side,
                        "PREV_",
                        vwap_p,
                        bands_p,
                    )
                    if primary_prev_cross:
                        lbl, lvl = primary_prev_cross
                        add_signal(lbl, "PREVIOUS", prev_iso, vwap_p, sd_p, lvl)
                else:
                    logging.warning(f"{sym}: invalid previous AVWAP / bands.")
            else:
                logging.warning(f"{sym}: no candle on previous earnings date {prev_date}.")

        # prepare daily OHLC slice for AI (recent window only)
        # use last ~60 days of df
        df_recent = df.tail(60).copy()
        daily_ohlc = []
        for _, row in df_recent.iterrows():
            if pd.isna(row["datetime"]):
                continue
            daily_ohlc.append({
                "date": row["datetime"].date().isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

        last_row = get_last_daily_row_for_date(daily_ohlc, last_trade_date)
        last_close = float(last_row["close"]) if last_row else None
        last_volume = float(last_row["volume"]) if last_row else None

        atr20 = compute_atr_from_ohlc(daily_ohlc, last_trade_date)
        trend_label = compute_trend_label_20d(daily_ohlc, last_trade_date)
        previous_day_range_summary = assess_previous_day_range_break(
            daily_ohlc,
            last_trade_date,
            last_close,
            side,
        )

        current_vwap = current_anchor_meta.get("vwap") if current_anchor_meta else None
        current_upper_1 = (
            current_anchor_meta.get("bands", {}).get("UPPER_1")
            if current_anchor_meta else None
        )
        current_upper_2 = (
            current_anchor_meta.get("bands", {}).get("UPPER_2")
            if current_anchor_meta else None
        )
        current_upper_3 = (
            current_anchor_meta.get("bands", {}).get("UPPER_3")
            if current_anchor_meta else None
        )
        current_lower_1 = (
            current_anchor_meta.get("bands", {}).get("LOWER_1")
            if current_anchor_meta else None
        )
        current_lower_2 = (
            current_anchor_meta.get("bands", {}).get("LOWER_2")
            if current_anchor_meta else None
        )
        current_lower_3 = (
            current_anchor_meta.get("bands", {}).get("LOWER_3")
            if current_anchor_meta else None
        )

        indicator_frame = compute_indicator_frame(df)
        indicator_row = None
        if not indicator_frame.empty:
            eligible_indicator_rows = indicator_frame[indicator_frame["trade_date"] <= last_trade_date.isoformat()]
            if not eligible_indicator_rows.empty:
                indicator_row = eligible_indicator_rows.iloc[-1]

        latest_release_context = _build_latest_earnings_release_context(
            df,
            latest_release_map.get(sym),
        )
        latest_known_earnings_context = _build_latest_known_earnings_context(
            df,
            latest_release_map.get(sym),
            reference_date=today_run,
        )
        post_earnings_summary = analyze_post_earnings_setups(
            df,
            side,
            latest_release_context,
        )
        if post_earnings_summary.get("break_signal"):
            post_anchor_meta = post_earnings_summary.get("anchor_meta") if isinstance(post_earnings_summary.get("anchor_meta"), dict) else {}
            add_signal(
                POST_EARNINGS_BREAK_SIGNAL,
                "POST_EARNINGS",
                post_earnings_summary.get("anchor_date", ""),
                post_anchor_meta.get("vwap"),
                post_anchor_meta.get("stdev"),
                post_earnings_summary.get("monitor_level"),
            )
            if post_earnings_summary.get("break_close"):
                add_signal(
                    POST_EARNINGS_CLOSE_CONFIRM_SIGNAL,
                    "POST_EARNINGS",
                    post_earnings_summary.get("anchor_date", ""),
                    post_anchor_meta.get("vwap"),
                    post_anchor_meta.get("stdev"),
                    post_earnings_summary.get("monitor_level"),
                )
        if post_earnings_summary.get("bounce_signal"):
            post_anchor_meta = post_earnings_summary.get("anchor_meta") if isinstance(post_earnings_summary.get("anchor_meta"), dict) else {}
            add_signal(
                POST_EARNINGS_BOUNCE_SIGNAL,
                "POST_EARNINGS",
                post_earnings_summary.get("anchor_date", ""),
                post_anchor_meta.get("vwap"),
                post_anchor_meta.get("stdev"),
                post_anchor_meta.get("vwap"),
            )

        stdev_blocked_by_recent_earnings = False
        if recent_earnings_dates:
            try:
                stdev_last_earnings = datetime.fromisoformat(recent_earnings_dates[0]).date()
                stdev_sessions_since = sessions_since_date(df, stdev_last_earnings)
                stdev_blocked_by_recent_earnings = (
                    stdev_sessions_since is not None
                    and stdev_sessions_since <= STDEV_RECENT_EARNINGS_BLOCK
                )
            except ValueError:
                stdev_blocked_by_recent_earnings = False

        extreme_move_summary = analyze_extreme_move_retest_setup(
            df=df,
            daily_rows=daily_ohlc,
            last_trade_date=last_trade_date,
            side=side,
            current_anchor_meta=current_anchor_meta,
            indicator_frame=indicator_frame,
            atr20=atr20,
            blocked_by_recent_earnings=stdev_blocked_by_recent_earnings,
        )
        if extreme_move_summary.get("favorite_signal"):
            add_signal(
                "EXTREME_MOVE_RETEST",
                "CURRENT",
                current_anchor_meta.get("date") if current_anchor_meta else "",
                current_vwap,
                current_anchor_meta.get("stdev") if current_anchor_meta else None,
                extreme_move_summary.get("retest_level_value"),
            )

        mid_earnings_summary = analyze_mid_earnings_ema_retest_setup(
            df,
            side,
            latest_release_context,
            indicator_frame,
        )
        if mid_earnings_summary.get("favorite_signal"):
            mid_anchor_meta = mid_earnings_summary.get("anchor_meta") if isinstance(mid_earnings_summary.get("anchor_meta"), dict) else {}
            mid_signal_levels = {
                MID_EARNINGS_EMA15_RETEST_SIGNAL: _coerce_float(indicator_row.get("ema_15")) if indicator_row is not None else None,
                MID_EARNINGS_EMA21_RETEST_SIGNAL: _coerce_float(indicator_row.get("ema_21")) if indicator_row is not None else None,
                MID_EARNINGS_FIRST_DEV_RETEST_SIGNAL: (
                    mid_anchor_meta.get("bands", {}).get("UPPER_1" if side == "LONG" else "LOWER_1")
                    if mid_anchor_meta
                    else None
                ),
                MID_EARNINGS_EMA21_CONFLUENCE_SIGNAL: _coerce_float(indicator_row.get("ema_21")) if indicator_row is not None else None,
                MID_EARNINGS_FIRST_DEV_CONFLUENCE_SIGNAL: (
                    mid_anchor_meta.get("bands", {}).get("UPPER_1" if side == "LONG" else "LOWER_1")
                    if mid_anchor_meta
                    else None
                ),
                MID_EARNINGS_EMA8_CONFLUENCE_SIGNAL: _coerce_float(indicator_row.get("ema_8")) if indicator_row is not None else None,
            }
            for signal_name in mid_earnings_summary.get("events", []):
                add_signal(
                    signal_name,
                    "MID_EARNINGS",
                    mid_earnings_summary.get("anchor_date", ""),
                    mid_anchor_meta.get("vwap"),
                    mid_anchor_meta.get("stdev"),
                    mid_signal_levels.get(signal_name),
                )

        top_pattern_summary = analyze_top_pattern_setup(
            df,
            side=side,
            current_anchor_meta=current_anchor_meta,
            indicator_frame=indicator_frame,
            atr20=atr20,
        )
        if top_pattern_summary.get("favorite_signal") and top_pattern_summary.get("signal"):
            add_signal(
                top_pattern_summary.get("signal"),
                "TOP",
                current_anchor_meta.get("date") if current_anchor_meta else "",
                current_vwap,
                current_anchor_meta.get("stdev") if current_anchor_meta else None,
                top_pattern_summary.get("entry_level"),
            )

        # dedupe and sort events for consistency
        symbol_events_today = sorted(set(symbol_events_today))

        # multi-day pattern detection
        prev_entries = history.get(sym, [])
        prev_events = prev_entries[-1]["events"] if prev_entries else []
        md_patterns = compute_multi_day_patterns(sym, side, symbol_events_today, prev_events)
        symbol_multi_day = md_patterns

        # include multi-day patterns as events as well
        full_event_list = symbol_events_today + symbol_multi_day

        # record in history
        entry = {
            "date": today_run.isoformat(),
            "side": side,
            "events": full_event_list
        }
        history.setdefault(sym, []).append(entry)

        # append to output lines
        for lbl in full_event_list:
            events_for_output.append((sym, dstr, lbl, side))

        has_bounce_event_today = any(_bounce_signal_level(event_name) for event_name in symbol_events_today)
        current_band_context = get_band_context(last_close, current_anchor_meta, side)
        previous_band_context = get_band_context(last_close, prev_anchor_meta, side)

        def _distance(level):
            if last_close is None or level is None:
                return None
            return last_close - level

        def _pct(level):
            if last_close is None or level is None or level == 0:
                return None
            return (last_close - level) / level * 100

        dist_vwap = _distance(current_vwap)
        pct_vwap = _pct(current_vwap)
        dist_upper_1 = _distance(current_upper_1)
        pct_upper_1 = _pct(current_upper_1)
        dist_lower_1 = _distance(current_lower_1)
        pct_lower_1 = _pct(current_lower_1)

        def _between(level_a, level_b):
            if last_close is None or level_a is None or level_b is None:
                return False
            low, high = sorted([level_a, level_b])
            return low <= last_close <= high

        if side == "LONG":
            if _between(current_vwap, current_upper_1):
                range_buckets["long_avwap_to_upper_1"].append(sym)
            if _between(current_upper_1, current_upper_2):
                range_buckets["long_upper_1_to_upper_2"].append(sym)
            if (
                current_upper_2 is not None
                and current_upper_3 is not None
                and closes_between_bands(df, current_upper_2, current_upper_3, 2)
            ):
                market_prep_range_buckets["long_upper_2_to_upper_3_2_sessions"].append(sym)
        else:
            if _between(current_lower_1, current_vwap):
                range_buckets["short_avwap_to_lower_1"].append(sym)
            if _between(current_lower_2, current_lower_1):
                range_buckets["short_lower_1_to_lower_2"].append(sym)

        favorite_zone = None
        if side == "LONG" and _between(current_vwap, current_upper_1):
            favorite_zone = "AVWAPE to UPPER_1"
        elif side == "SHORT" and _between(current_lower_1, current_vwap):
            favorite_zone = "LOWER_1 to AVWAPE"

        breakout_long_5d, breakout_short_5d = compute_five_day_breakout_flags(daily_ohlc, last_trade_date)
        recent_band_extension_days = count_recent_band_extension_days(
            daily_ohlc,
            last_trade_date,
            current_lower_1 if side == "LONG" else current_upper_1,
            side,
        )
        recent_second_band_test_days = count_recent_band_test_days(
            daily_ohlc,
            last_trade_date,
            current_upper_2 if side == "LONG" else current_lower_2,
            side,
        )
        second_band_penalty = compute_recent_second_band_penalty(recent_second_band_test_days)
        first_dev_quality = assess_first_dev_break_quality(
            daily_ohlc,
            last_trade_date,
            current_upper_1 if side == "LONG" else current_lower_1,
            side,
            atr20=atr20,
        )
        extension_note = ""
        if recent_second_band_test_days > 0:
            second_band_label = "UPPER_2" if side == "LONG" else "LOWER_2"
            extension_note = (
                f"Recent {second_band_label} tests={recent_second_band_test_days} "
                f"(-{second_band_penalty})"
            )
        breakout_5d = breakout_long_5d if side == "LONG" else breakout_short_5d
        retest_summary = analyze_avwap_retest_behavior(
            daily_ohlc,
            last_trade_date,
            current_vwap,
            side,
            current_upper_1=current_upper_1,
            current_lower_1=current_lower_1,
            atr20=atr20,
        )
        retest_followthrough = bool(retest_summary["retest_followthrough"])
        compression_summary = evaluate_anchor_compression(
            df,
            current_anchor_meta.get("date") if current_anchor_meta else None,
            current_anchor_meta.get("stdev") if current_anchor_meta else None,
            atr20,
            last_trade_date,
        )
        entry_bar = pd.Series(last_row or {"open": last_close, "high": last_close, "low": last_close, "close": last_close})
        entry_feature_snapshot = build_tracker_feature_snapshot(
            normalize_side(side),
            entry_bar,
            indicator_row,
            current_anchor_meta,
            prev_anchor_meta,
            compression_summary,
        )
        trend_ma_summary = assess_trend_ma_alignment(
            side,
            last_close,
            entry_feature_snapshot.get("ema15"),
            entry_feature_snapshot.get("sma20"),
        )
        bouncebot_focus_context = build_bouncebot_focus_context(
            sym,
            side,
            trade_date=last_trade_date.isoformat(),
        )
        if current_anchor_meta:
            if not stdev_blocked_by_recent_earnings:
                if side == "LONG":
                    if (
                        current_upper_2 is not None
                        and current_upper_3 is not None
                        and last_close is not None
                        and last_close >= current_upper_2
                        and closes_between_bands(df, current_upper_2, current_upper_3, 4)
                    ):
                        stdev_range_hits["long"].append(sym)
                    if current_upper_2 is not None and cross_up_through_level(df, current_upper_2):
                        stdev_cross_hits["long"].append(sym)
                    if current_upper_2 is not None and bounce_up_at_level(df, current_upper_2):
                        stdev_cross_hits["long"].append(f"{sym} (bounce)")
                else:
                    if (
                        current_lower_2 is not None
                        and current_lower_3 is not None
                        and last_close is not None
                        and last_close <= current_lower_2
                        and closes_between_bands(df, current_lower_3, current_lower_2, 4)
                    ):
                        stdev_range_hits["short"].append(sym)
                    if current_lower_2 is not None and cross_down_through_level(df, current_lower_2):
                        stdev_cross_hits["short"].append(sym)
                    if current_lower_2 is not None and bounce_down_at_level(df, current_lower_2):
                        stdev_cross_hits["short"].append(f"{sym} (bounce)")

        current_position = current_band_context["active_level"]
        if current_position:
            positions["current"].setdefault(current_position, []).append(sym)

        previous_position = previous_band_context["active_level"]
        if previous_position:
            positions["previous"].setdefault(previous_position, []).append(sym)

        symbol_entry = {
            "side": side,
            "last_trade_date": last_trade_date.isoformat(),
            "current_anchor": current_anchor_meta,
            "previous_anchor": prev_anchor_meta,
            "events_today": symbol_events_today,
            "multi_day_patterns": symbol_multi_day,
            "events_all_for_day": full_event_list,
            "daily_ohlc": daily_ohlc,
            "daily_bar_source": daily_bar_source,
            "market_regime_label": market_regime_snapshot.get("label", ""),
            "market_regime_score": market_regime_snapshot.get("directional_score"),
            "last_close": last_close,
            "last_volume": last_volume,
            "atr20": atr20,
            "current_active_level": current_band_context["active_level"],
            "current_nearby_bands": list(current_band_context["nearby_levels"]),
            "current_band_zone": current_band_context["zone"],
            "previous_active_level": previous_band_context["active_level"],
            "previous_nearby_bands": list(previous_band_context["nearby_levels"]),
            "previous_band_zone": previous_band_context["zone"],
            "distance_from_current_vwap": dist_vwap,
            "pct_from_current_vwap": pct_vwap,
            "distance_from_current_upper_1": dist_upper_1,
            "pct_from_current_upper_1": pct_upper_1,
            "distance_from_current_lower_1": dist_lower_1,
            "pct_from_current_lower_1": pct_lower_1,
            "trend_20d": trend_label,
            "previous_day_date": previous_day_range_summary["previous_day_date"],
            "previous_day_high": previous_day_range_summary["previous_day_high"],
            "previous_day_low": previous_day_range_summary["previous_day_low"],
            "previous_day_range_break": bool(previous_day_range_summary["previous_day_range_break"]),
            "previous_day_range_break_bonus": int(previous_day_range_summary["previous_day_range_break_bonus"] or 0),
            "previous_day_range_note": previous_day_range_summary["previous_day_range_note"],
            "trend_ma_alignment": bool(trend_ma_summary["trend_ma_alignment"]),
            "trend_ma_alignment_bonus": int(trend_ma_summary["trend_ma_alignment_bonus"] or 0),
            "trend_ma_alignment_note": trend_ma_summary["trend_ma_alignment_note"],
            "has_bounce_event_today": has_bounce_event_today,
            "favorite_zone": favorite_zone,
            "recent_band_extension_days": recent_band_extension_days,
            "recent_second_band_test_days": recent_second_band_test_days,
            "second_band_penalty": second_band_penalty,
            "first_dev_break_bonus": int(first_dev_quality.get("fresh_break_bonus", 0) or 0),
            "first_dev_chop_penalty": int(first_dev_quality.get("chop_penalty", 0) or 0),
            "first_dev_note": first_dev_quality.get("note", ""),
            "breakout_5d": breakout_5d,
            "retest_followthrough": retest_followthrough,
            "retest_reference_level": retest_summary["retest_reference_level"],
            "retest_note": retest_summary["retest_note"],
            "extreme_move_watch": bool(extreme_move_summary.get("watch")),
            "extreme_move_favorite_ready": bool(extreme_move_summary.get("favorite_signal")),
            "extreme_move_retest_level": extreme_move_summary.get("retest_level", ""),
            "extreme_move_band_crosses": int(extreme_move_summary.get("band_crosses", 0) or 0),
            "extreme_move_band_width_atr": _coerce_float(extreme_move_summary.get("band_width_atr")),
            "extreme_move_displacement_date": extreme_move_summary.get("displacement_date", ""),
            "extreme_move_note": extreme_move_summary.get("note", ""),
            "extension_note": extension_note,
            "compression_flag": bool(compression_summary.get("is_compressed")),
            "compression_penalty": int(compression_summary.get("compression_penalty", 0) or 0),
            "compression_note": compression_summary.get("compression_note", ""),
            "latest_release_earnings_date": latest_release_context.get("earnings_date", "")
            or latest_known_earnings_context.get("earnings_date", ""),
            "latest_release_gap_date": latest_release_context.get("gap_date", ""),
            "latest_release_anchor_date": latest_release_context.get("anchor_date", ""),
            "latest_release_gap_atr_multiple": _coerce_float(latest_release_context.get("gap_atr_multiple")),
            "latest_release_sessions_since_gap": latest_release_context.get("sessions_since_gap"),
            "latest_known_earnings_date": latest_known_earnings_context.get("earnings_date", ""),
            "latest_known_earnings_calendar_days_since": latest_known_earnings_context.get("calendar_days_since_earnings"),
            "latest_known_earnings_sessions_since": latest_known_earnings_context.get("sessions_since_earnings"),
            "latest_known_earnings_in_post_window": bool(latest_known_earnings_context.get("in_post_earnings_window")),
            "next_earnings_date": next_earnings_summary.get("next_earnings_date", ""),
            "days_to_next_earnings": next_earnings_summary.get("days_to_next_earnings"),
            "pre_earnings_setup_blocked": bool(next_earnings_summary.get("pre_earnings_setup_blocked")),
            "pre_earnings_setup_block_reason": next_earnings_summary.get("pre_earnings_setup_block_reason", ""),
            "post_earnings_active": bool(post_earnings_summary.get("active")),
            "post_earnings_monitor_level": _coerce_float(post_earnings_summary.get("monitor_level")),
            "post_earnings_monitor_level_label": post_earnings_summary.get("monitor_level_label", ""),
            "post_earnings_break_intraday": bool(post_earnings_summary.get("break_intraday")),
            "post_earnings_break_close": bool(post_earnings_summary.get("break_close")),
            "post_earnings_sessions_since_gap": post_earnings_summary.get("sessions_since_gap"),
            "post_earnings_bounce_date": post_earnings_summary.get("bounce_date", ""),
            "post_earnings_bounce_age_sessions": post_earnings_summary.get("bounce_age_sessions"),
            "post_earnings_note": post_earnings_summary.get("note", ""),
            "post_earnings_anchor": post_earnings_summary.get("anchor_meta"),
            "mid_earnings_watch": bool(mid_earnings_summary.get("watch")),
            "mid_earnings_active_second_stdev_hold": bool(mid_earnings_summary.get("active_second_stdev_hold")),
            "mid_earnings_sessions_since_gap": mid_earnings_summary.get("sessions_since_gap"),
            "mid_earnings_zone_streak_days": int(mid_earnings_summary.get("zone_streak_days", 0) or 0),
            "mid_earnings_zone_start_date": mid_earnings_summary.get("zone_start_date", ""),
            "mid_earnings_zone_end_date": mid_earnings_summary.get("zone_end_date", ""),
            "mid_earnings_primary_trigger_level": mid_earnings_summary.get("primary_trigger_level", ""),
            "mid_earnings_retest_date": mid_earnings_summary.get("retest_date", ""),
            "mid_earnings_sessions_after_zone": mid_earnings_summary.get("sessions_after_zone"),
            "mid_earnings_trigger_levels": list(mid_earnings_summary.get("trigger_levels") or []),
            "mid_earnings_ema15_trigger": bool(mid_earnings_summary.get("ema15_trigger")),
            "mid_earnings_ema21_trigger": bool(mid_earnings_summary.get("ema21_trigger")),
            "mid_earnings_first_dev_trigger": bool(mid_earnings_summary.get("first_dev_trigger")),
            "mid_earnings_ema8_confluence": bool(mid_earnings_summary.get("ema8_confluence")),
            "mid_earnings_ema21_confluence": bool(mid_earnings_summary.get("ema21_confluence")),
            "mid_earnings_first_dev_confluence": bool(mid_earnings_summary.get("first_dev_confluence")),
            "mid_earnings_note": mid_earnings_summary.get("note", ""),
            "top_pattern_watch": bool(top_pattern_summary.get("watch")),
            "top_pattern_entry": bool(top_pattern_summary.get("favorite_signal")),
            "top_pattern_signal": top_pattern_summary.get("signal", ""),
            "top_pattern_entry_trigger": top_pattern_summary.get("entry_trigger", ""),
            "top_pattern_entry_triggers": list(top_pattern_summary.get("entry_triggers") or []),
            "top_pattern_entry_level": _coerce_float(top_pattern_summary.get("entry_level")),
            "top_pattern_score_bonus": int(top_pattern_summary.get("score_bonus", 0) or 0),
            "top_pattern_note": top_pattern_summary.get("note", ""),
            "top_pattern_weekly_return_13w_pct": _coerce_float(top_pattern_summary.get("weekly_return_13w_pct")),
            "top_pattern_weekly_return_26w_pct": _coerce_float(top_pattern_summary.get("weekly_return_26w_pct")),
            "top_pattern_weekly_pullback_from_52w_high_pct": _coerce_float(
                top_pattern_summary.get("weekly_pullback_from_52w_high_pct")
            ),
            "top_pattern_weekly_ema15_hold": bool(top_pattern_summary.get("weekly_ema15_hold")),
            "top_pattern_weekly_ema15_hold_ratio": _coerce_float(
                top_pattern_summary.get("weekly_ema15_hold_ratio")
            ),
            "top_pattern_weekly_above_sma100": bool(top_pattern_summary.get("weekly_above_sma100")),
            "top_pattern_weekly_sma50_retest_recent": bool(
                top_pattern_summary.get("weekly_sma50_retest_recent")
            ),
            "top_pattern_daily_sma50_reclaim": bool(top_pattern_summary.get("daily_sma50_reclaim")),
            "top_pattern_daily_sma50_bounce": bool(top_pattern_summary.get("daily_sma50_bounce")),
            "top_pattern_weekly_sma50_entry": bool(top_pattern_summary.get("weekly_sma50_entry")),
            "top_pattern_avwape_retest": bool(top_pattern_summary.get("avwape_retest")),
            "entry_feature_snapshot": entry_feature_snapshot,
            **bouncebot_focus_context,
        }

        theta_candidate = evaluate_theta_put_candidate(
            symbol=sym,
            side=side,
            df=df,
            last_trade_date=last_trade_date,
            last_close=last_close,
            atr20=atr20,
            current_anchor_meta=current_anchor_meta,
            previous_anchor_meta=prev_anchor_meta,
            indicator_row=indicator_row,
            compression_summary=compression_summary,
            recent_earnings_dates=recent_earnings_dates,
            upcoming_earnings_dates=upcoming_earnings_map.get(sym, []),
        )
        if theta_candidate:
            theta_put_rows.append(theta_candidate)
            symbol_entry["theta_put_candidate"] = theta_candidate

        theta_pcs_candidate = evaluate_theta_pcs_candidate(
            symbol=sym,
            side=side,
            df=df,
            last_trade_date=last_trade_date,
            last_close=last_close,
            atr20=atr20,
            current_anchor_meta=current_anchor_meta,
            previous_anchor_meta=prev_anchor_meta,
            indicator_row=indicator_row,
            compression_summary=compression_summary,
            recent_earnings_dates=recent_earnings_dates,
            upcoming_earnings_dates=upcoming_earnings_map.get(sym, []),
        )
        if theta_pcs_candidate:
            theta_pcs_rows.append(theta_pcs_candidate)
            symbol_entry["theta_pcs_candidate"] = theta_pcs_candidate

        priority_summary = build_priority_setup_summary(
            symbol=sym,
            side=side,
            events_today=symbol_events_today,
            all_events=full_event_list,
            trend_label=trend_label,
            favorite_zone=favorite_zone,
            recent_band_extension_days=recent_band_extension_days,
            recent_second_band_test_days=recent_second_band_test_days,
            second_band_penalty=second_band_penalty,
            first_dev_break_bonus=first_dev_quality.get("fresh_break_bonus", 0),
            first_dev_chop_penalty=first_dev_quality.get("chop_penalty", 0),
            first_dev_note=first_dev_quality.get("note", ""),
            breakout_5d=breakout_5d,
            retest_followthrough=retest_followthrough,
            retest_reference_level=retest_summary["retest_reference_level"],
            retest_note=retest_summary["retest_note"],
            extreme_move_watch=bool(extreme_move_summary.get("watch")),
            extreme_move_favorite_ready=bool(extreme_move_summary.get("favorite_signal")),
            extreme_move_retest_level=extreme_move_summary.get("retest_level", ""),
            extreme_move_band_crosses=int(extreme_move_summary.get("band_crosses", 0) or 0),
            extreme_move_band_width_atr=_coerce_float(extreme_move_summary.get("band_width_atr")),
            extreme_move_displacement_date=extreme_move_summary.get("displacement_date", ""),
            extreme_move_note=extreme_move_summary.get("note", ""),
            previous_day_range_break=bool(previous_day_range_summary["previous_day_range_break"]),
            previous_day_range_break_bonus=int(previous_day_range_summary["previous_day_range_break_bonus"] or 0),
            previous_day_range_note=previous_day_range_summary["previous_day_range_note"],
            trend_ma_alignment=bool(trend_ma_summary["trend_ma_alignment"]),
            trend_ma_alignment_bonus=int(trend_ma_summary["trend_ma_alignment_bonus"] or 0),
            trend_ma_alignment_note=trend_ma_summary["trend_ma_alignment_note"],
            extension_note=extension_note,
            mid_earnings_active_second_stdev_hold=bool(mid_earnings_summary.get("active_second_stdev_hold")),
            mid_earnings_primary_trigger_level=mid_earnings_summary.get("primary_trigger_level", ""),
            top_pattern_watch=bool(top_pattern_summary.get("watch")),
            top_pattern_entry=bool(top_pattern_summary.get("favorite_signal")),
            top_pattern_signal=top_pattern_summary.get("signal", ""),
            top_pattern_score_bonus=int(top_pattern_summary.get("score_bonus", 0) or 0),
            top_pattern_note=top_pattern_summary.get("note", ""),
            top_pattern_entry_trigger=top_pattern_summary.get("entry_trigger", ""),
        )
        priority_summary["post_earnings_active"] = bool(post_earnings_summary.get("active"))
        priority_summary["latest_release_earnings_date"] = (
            latest_release_context.get("earnings_date", "")
            or latest_known_earnings_context.get("earnings_date", "")
        )
        priority_summary["latest_release_gap_date"] = latest_release_context.get("gap_date", "")
        priority_summary["latest_release_sessions_since_gap"] = latest_release_context.get("sessions_since_gap")
        priority_summary["latest_known_earnings_date"] = latest_known_earnings_context.get("earnings_date", "")
        priority_summary["latest_known_earnings_calendar_days_since"] = latest_known_earnings_context.get("calendar_days_since_earnings")
        priority_summary["latest_known_earnings_sessions_since"] = latest_known_earnings_context.get("sessions_since_earnings")
        priority_summary["latest_known_earnings_in_post_window"] = bool(
            latest_known_earnings_context.get("in_post_earnings_window")
        )
        priority_summary["post_earnings_break_intraday"] = bool(post_earnings_summary.get("break_intraday"))
        priority_summary["post_earnings_break_close"] = bool(post_earnings_summary.get("break_close"))
        priority_summary["post_earnings_sessions_since_gap"] = post_earnings_summary.get("sessions_since_gap")
        priority_summary["post_earnings_gap_atr_multiple"] = _coerce_float(post_earnings_summary.get("gap_atr_multiple"))
        priority_summary["next_earnings_date"] = next_earnings_summary.get("next_earnings_date", "")
        priority_summary["days_to_next_earnings"] = next_earnings_summary.get("days_to_next_earnings")
        priority_summary["pre_earnings_setup_blocked"] = bool(next_earnings_summary.get("pre_earnings_setup_blocked"))
        priority_summary["pre_earnings_setup_block_reason"] = next_earnings_summary.get("pre_earnings_setup_block_reason", "")
        priority_summary["post_earnings_anchor_date"] = post_earnings_summary.get("anchor_date", "")
        priority_summary["post_earnings_gap_date"] = post_earnings_summary.get("gap_date", "")
        priority_summary["post_earnings_bounce_date"] = post_earnings_summary.get("bounce_date", "")
        priority_summary["post_earnings_bounce_age_sessions"] = post_earnings_summary.get("bounce_age_sessions")
        priority_summary["post_earnings_monitor_level"] = _coerce_float(post_earnings_summary.get("monitor_level"))
        priority_summary["post_earnings_note"] = post_earnings_summary.get("note", "")
        priority_summary["mid_earnings_watch"] = bool(mid_earnings_summary.get("watch"))
        priority_summary["mid_earnings_active_second_stdev_hold"] = bool(mid_earnings_summary.get("active_second_stdev_hold"))
        priority_summary["mid_earnings_sessions_since_gap"] = mid_earnings_summary.get("sessions_since_gap")
        priority_summary["mid_earnings_zone_streak_days"] = int(mid_earnings_summary.get("zone_streak_days", 0) or 0)
        priority_summary["mid_earnings_zone_start_date"] = mid_earnings_summary.get("zone_start_date", "")
        priority_summary["mid_earnings_zone_end_date"] = mid_earnings_summary.get("zone_end_date", "")
        priority_summary["mid_earnings_primary_trigger_level"] = mid_earnings_summary.get("primary_trigger_level", "")
        priority_summary["mid_earnings_retest_date"] = mid_earnings_summary.get("retest_date", "")
        priority_summary["mid_earnings_sessions_after_zone"] = mid_earnings_summary.get("sessions_after_zone")
        priority_summary["mid_earnings_trigger_levels"] = list(mid_earnings_summary.get("trigger_levels") or [])
        priority_summary["mid_earnings_ema15_trigger"] = bool(mid_earnings_summary.get("ema15_trigger"))
        priority_summary["mid_earnings_ema21_trigger"] = bool(mid_earnings_summary.get("ema21_trigger"))
        priority_summary["mid_earnings_first_dev_trigger"] = bool(mid_earnings_summary.get("first_dev_trigger"))
        priority_summary["mid_earnings_ema8_confluence"] = bool(mid_earnings_summary.get("ema8_confluence"))
        priority_summary["mid_earnings_ema21_confluence"] = bool(mid_earnings_summary.get("ema21_confluence"))
        priority_summary["mid_earnings_first_dev_confluence"] = bool(mid_earnings_summary.get("first_dev_confluence"))
        priority_summary["mid_earnings_note"] = mid_earnings_summary.get("note", "")
        priority_summary["top_pattern_entry_triggers"] = list(top_pattern_summary.get("entry_triggers") or [])
        priority_summary["top_pattern_entry_level"] = _coerce_float(top_pattern_summary.get("entry_level"))
        priority_summary["top_pattern_weekly_return_13w_pct"] = _coerce_float(
            top_pattern_summary.get("weekly_return_13w_pct")
        )
        priority_summary["top_pattern_weekly_return_26w_pct"] = _coerce_float(
            top_pattern_summary.get("weekly_return_26w_pct")
        )
        priority_summary["top_pattern_weekly_pullback_from_52w_high_pct"] = _coerce_float(
            top_pattern_summary.get("weekly_pullback_from_52w_high_pct")
        )
        priority_summary["top_pattern_weekly_ema15_hold"] = bool(top_pattern_summary.get("weekly_ema15_hold"))
        priority_summary["top_pattern_weekly_ema15_hold_ratio"] = _coerce_float(
            top_pattern_summary.get("weekly_ema15_hold_ratio")
        )
        priority_summary["top_pattern_weekly_above_sma100"] = bool(top_pattern_summary.get("weekly_above_sma100"))
        priority_summary["top_pattern_weekly_sma50_retest_recent"] = bool(
            top_pattern_summary.get("weekly_sma50_retest_recent")
        )
        priority_summary["top_pattern_daily_sma50_reclaim"] = bool(top_pattern_summary.get("daily_sma50_reclaim"))
        priority_summary["top_pattern_daily_sma50_bounce"] = bool(top_pattern_summary.get("daily_sma50_bounce"))
        priority_summary["top_pattern_weekly_sma50_entry"] = bool(top_pattern_summary.get("weekly_sma50_entry"))
        priority_summary["top_pattern_avwape_retest"] = bool(top_pattern_summary.get("avwape_retest"))
        effective_compression_penalty, effective_compression_note = _effective_compression_penalty(
            compression_summary,
            priority_summary,
            side,
        )
        priority_summary["score"] = float(priority_summary["score"] - effective_compression_penalty)
        priority_summary["current_active_level"] = current_band_context["active_level"]
        priority_summary["current_band_zone"] = current_band_context["zone"]
        priority_summary["compression_flag"] = bool(compression_summary.get("is_compressed"))
        priority_summary["compression_penalty"] = int(effective_compression_penalty or 0)
        priority_summary["compression_note"] = effective_compression_note
        symbol_entry["compression_penalty"] = int(effective_compression_penalty or 0)
        symbol_entry["compression_note"] = effective_compression_note
        if isinstance(entry_feature_snapshot, dict):
            entry_feature_snapshot["compression_penalty"] = int(effective_compression_penalty or 0)
            entry_feature_snapshot["compression_note"] = effective_compression_note
        symbol_entry["priority_score"] = priority_summary["score"]
        symbol_entry["favorite_signals"] = priority_summary["favorite_signals"]
        symbol_entry["favorite_context_signals"] = priority_summary["context_signals"]
        symbol_entry["setup_family"] = priority_summary.get("setup_family", "")
        symbol_entry["setup_tags"] = list(priority_summary.get("setup_tags") or [])
        priority_rows.append(priority_summary)

        clean_favorite_zone_setup = (
            not priority_summary["compression_flag"]
            and priority_summary.get("current_active_level") == "VWAP"
            and (
                (
                    side == "LONG"
                    and favorite_zone == "AVWAPE to UPPER_1"
                    and priority_summary.get("current_band_zone") == "VWAP to UPPER_1"
                )
                or (
                    side == "SHORT"
                    and favorite_zone == "LOWER_1 to AVWAPE"
                    and priority_summary.get("current_band_zone") == "VWAP to LOWER_1"
                )
            )
        )
        priority_bucket = ""
        is_favorite_setup = False
        is_near_favorite_zone = False
        if priority_summary["has_favorite_signal"] or clean_favorite_zone_setup:
            priority_bucket = "favorite_setup"
            is_favorite_setup = True
        elif favorite_zone or priority_summary.get("extreme_move_watch"):
            priority_bucket = "near_favorite_zone"
            is_near_favorite_zone = True

        for record in symbol_signal_info.values():
            record["priority_bucket"] = priority_bucket
            record["is_favorite_setup"] = is_favorite_setup
            record["is_near_favorite_zone"] = is_near_favorite_zone
            record["favorite_zone"] = favorite_zone or ""
            record["favorite_signals"] = ";".join(priority_summary["favorite_signals"])
            record["favorite_context_signals"] = ";".join(priority_summary["context_signals"])

        for lbl in symbol_events_today:
            record = symbol_signal_info.get(lbl)
            if record:
                csv_rows.append(record)

        ai_state["symbols"][sym] = symbol_entry

        feature_row = {
            "symbol": sym,
            "side": side,
            "last_trade_date": last_trade_date.isoformat(),
            "daily_bar_source": daily_bar_source,
            "market_regime_label": market_regime_snapshot.get("label", ""),
            "market_regime_score": market_regime_snapshot.get("directional_score"),
            "spy_one_day_return_pct": (market_regime_snapshot.get("benchmarks", {}).get("SPY", {}) or {}).get("one_day_return_pct"),
            "spy_five_day_return_pct": (market_regime_snapshot.get("benchmarks", {}).get("SPY", {}) or {}).get("five_day_return_pct"),
            "spy_above_sma20": (market_regime_snapshot.get("benchmarks", {}).get("SPY", {}) or {}).get("above_sma20"),
            "spy_above_sma50": (market_regime_snapshot.get("benchmarks", {}).get("SPY", {}) or {}).get("above_sma50"),
            "last_close": last_close,
            "last_volume": last_volume,
            "atr20": atr20,
            "current_active_level": current_band_context["active_level"],
            "current_nearby_bands": ";".join(current_band_context["nearby_levels"]),
            "current_band_zone": current_band_context["zone"],
            "previous_active_level": previous_band_context["active_level"],
            "previous_nearby_bands": ";".join(previous_band_context["nearby_levels"]),
            "previous_band_zone": previous_band_context["zone"],
            "current_anchor_date": current_anchor_meta.get("date") if current_anchor_meta else None,
            "current_anchor_vwap": current_vwap,
            "current_anchor_stdev": current_anchor_meta.get("stdev") if current_anchor_meta else None,
            "distance_from_current_vwap": dist_vwap,
            "pct_from_current_vwap": pct_vwap,
            "distance_from_current_upper_1": dist_upper_1,
            "pct_from_current_upper_1": pct_upper_1,
            "distance_from_current_lower_1": dist_lower_1,
            "pct_from_current_lower_1": pct_lower_1,
            "trend_20d": trend_label,
            "previous_day_date": previous_day_range_summary["previous_day_date"],
            "previous_day_high": previous_day_range_summary["previous_day_high"],
            "previous_day_low": previous_day_range_summary["previous_day_low"],
            "previous_day_range_break": bool(previous_day_range_summary["previous_day_range_break"]),
            "previous_day_range_break_bonus": int(previous_day_range_summary["previous_day_range_break_bonus"] or 0),
            "previous_day_range_note": previous_day_range_summary["previous_day_range_note"],
            "trend_ma_alignment": bool(trend_ma_summary["trend_ma_alignment"]),
            "trend_ma_alignment_bonus": int(trend_ma_summary["trend_ma_alignment_bonus"] or 0),
            "trend_ma_alignment_note": trend_ma_summary["trend_ma_alignment_note"],
            "has_bounce_event_today": has_bounce_event_today,
            "favorite_zone": favorite_zone,
            "recent_band_extension_days": recent_band_extension_days,
            "recent_second_band_test_days": recent_second_band_test_days,
            "second_band_penalty": second_band_penalty,
            "breakout_5d": breakout_5d,
            "retest_followthrough": retest_followthrough,
            "retest_reference_level": retest_summary["retest_reference_level"],
            "retest_note": retest_summary["retest_note"],
            "extreme_move_watch": bool(extreme_move_summary.get("watch")),
            "extreme_move_favorite_ready": bool(extreme_move_summary.get("favorite_signal")),
            "extreme_move_retest_level": extreme_move_summary.get("retest_level", ""),
            "extreme_move_band_crosses": int(extreme_move_summary.get("band_crosses", 0) or 0),
            "extreme_move_band_width_atr": _coerce_float(extreme_move_summary.get("band_width_atr")),
            "extreme_move_displacement_date": extreme_move_summary.get("displacement_date", ""),
            "extreme_move_note": extreme_move_summary.get("note", ""),
            "extension_note": extension_note,
            "compression_flag": bool(compression_summary.get("is_compressed")),
            "compression_penalty": int(effective_compression_penalty or 0),
            "compression_note": effective_compression_note,
            "setup_family": priority_summary.get("setup_family", ""),
            "setup_tags": ";".join(priority_summary.get("setup_tags") or []),
            "latest_release_earnings_date": (
                latest_release_context.get("earnings_date", "")
                or latest_known_earnings_context.get("earnings_date", "")
            ),
            "latest_release_gap_date": latest_release_context.get("gap_date", ""),
            "latest_release_sessions_since_gap": latest_release_context.get("sessions_since_gap"),
            "latest_known_earnings_date": latest_known_earnings_context.get("earnings_date", ""),
            "latest_known_earnings_calendar_days_since": latest_known_earnings_context.get("calendar_days_since_earnings"),
            "latest_known_earnings_sessions_since": latest_known_earnings_context.get("sessions_since_earnings"),
            "latest_known_earnings_in_post_window": bool(latest_known_earnings_context.get("in_post_earnings_window")),
            "next_earnings_date": priority_summary.get("next_earnings_date", ""),
            "days_to_next_earnings": priority_summary.get("days_to_next_earnings"),
            "pre_earnings_setup_blocked": bool(priority_summary.get("pre_earnings_setup_blocked")),
            "pre_earnings_setup_block_reason": priority_summary.get("pre_earnings_setup_block_reason", ""),
            "post_earnings_hard_rule_blocked": bool(priority_summary.get("post_earnings_hard_rule_blocked")),
            "post_earnings_hard_rule_reason": priority_summary.get("post_earnings_hard_rule_reason", ""),
            "post_earnings_active": bool(post_earnings_summary.get("active")),
            "post_earnings_monitor_level": _coerce_float(post_earnings_summary.get("monitor_level")),
            "post_earnings_break_intraday": bool(post_earnings_summary.get("break_intraday")),
            "post_earnings_break_close": bool(post_earnings_summary.get("break_close")),
            "post_earnings_sessions_since_gap": post_earnings_summary.get("sessions_since_gap"),
            "post_earnings_gap_atr_multiple": _coerce_float(post_earnings_summary.get("gap_atr_multiple")),
            "post_earnings_bounce_date": post_earnings_summary.get("bounce_date", ""),
            "post_earnings_bounce_age_sessions": post_earnings_summary.get("bounce_age_sessions"),
            "post_earnings_note": post_earnings_summary.get("note", ""),
            "mid_earnings_watch": bool(mid_earnings_summary.get("watch")),
            "mid_earnings_active_second_stdev_hold": bool(mid_earnings_summary.get("active_second_stdev_hold")),
            "mid_earnings_sessions_since_gap": mid_earnings_summary.get("sessions_since_gap"),
            "mid_earnings_zone_streak_days": int(mid_earnings_summary.get("zone_streak_days", 0) or 0),
            "mid_earnings_zone_start_date": mid_earnings_summary.get("zone_start_date", ""),
            "mid_earnings_zone_end_date": mid_earnings_summary.get("zone_end_date", ""),
            "mid_earnings_primary_trigger_level": mid_earnings_summary.get("primary_trigger_level", ""),
            "mid_earnings_retest_date": mid_earnings_summary.get("retest_date", ""),
            "mid_earnings_sessions_after_zone": mid_earnings_summary.get("sessions_after_zone"),
            "mid_earnings_trigger_levels": ";".join(mid_earnings_summary.get("trigger_levels") or []),
            "mid_earnings_ema15_trigger": bool(mid_earnings_summary.get("ema15_trigger")),
            "mid_earnings_ema21_trigger": bool(mid_earnings_summary.get("ema21_trigger")),
            "mid_earnings_first_dev_trigger": bool(mid_earnings_summary.get("first_dev_trigger")),
            "mid_earnings_ema8_confluence": bool(mid_earnings_summary.get("ema8_confluence")),
            "mid_earnings_ema21_confluence": bool(mid_earnings_summary.get("ema21_confluence")),
            "mid_earnings_first_dev_confluence": bool(mid_earnings_summary.get("first_dev_confluence")),
            "mid_earnings_note": mid_earnings_summary.get("note", ""),
            "top_pattern_watch": bool(top_pattern_summary.get("watch")),
            "top_pattern_entry": bool(top_pattern_summary.get("favorite_signal")),
            "top_pattern_signal": top_pattern_summary.get("signal", ""),
            "top_pattern_entry_trigger": top_pattern_summary.get("entry_trigger", ""),
            "top_pattern_entry_triggers": ";".join(top_pattern_summary.get("entry_triggers") or []),
            "top_pattern_entry_level": _coerce_float(top_pattern_summary.get("entry_level")),
            "top_pattern_score_bonus": int(top_pattern_summary.get("score_bonus", 0) or 0),
            "top_pattern_note": top_pattern_summary.get("note", ""),
            "top_pattern_weekly_return_13w_pct": _coerce_float(top_pattern_summary.get("weekly_return_13w_pct")),
            "top_pattern_weekly_return_26w_pct": _coerce_float(top_pattern_summary.get("weekly_return_26w_pct")),
            "top_pattern_weekly_pullback_from_52w_high_pct": _coerce_float(
                top_pattern_summary.get("weekly_pullback_from_52w_high_pct")
            ),
            "top_pattern_weekly_ema15_hold": bool(top_pattern_summary.get("weekly_ema15_hold")),
            "top_pattern_weekly_ema15_hold_ratio": _coerce_float(
                top_pattern_summary.get("weekly_ema15_hold_ratio")
            ),
            "top_pattern_weekly_above_sma100": bool(top_pattern_summary.get("weekly_above_sma100")),
            "top_pattern_weekly_sma50_retest_recent": bool(
                top_pattern_summary.get("weekly_sma50_retest_recent")
            ),
            "top_pattern_daily_sma50_reclaim": bool(top_pattern_summary.get("daily_sma50_reclaim")),
            "top_pattern_daily_sma50_bounce": bool(top_pattern_summary.get("daily_sma50_bounce")),
            "top_pattern_weekly_sma50_entry": bool(top_pattern_summary.get("weekly_sma50_entry")),
            "top_pattern_avwape_retest": bool(top_pattern_summary.get("avwape_retest")),
            "bouncebot_relevant_focus_hit_today": bool(bouncebot_focus_context.get("bouncebot_relevant_focus_hit_today")),
            "bouncebot_relevant_focus_hit_count": int(bouncebot_focus_context.get("bouncebot_relevant_focus_hit_count", 0) or 0),
            "bouncebot_relevant_focus_max_score": _coerce_float(bouncebot_focus_context.get("bouncebot_relevant_focus_max_score")),
            "bouncebot_bullish_weak_long_seen_today": bool(bouncebot_focus_context.get("bouncebot_bullish_weak_long_seen_today")),
            "bouncebot_bearish_weak_short_seen_today": bool(bouncebot_focus_context.get("bouncebot_bearish_weak_short_seen_today")),
            "priority_score": priority_summary["score"],
            "priority_bucket": "",
            "is_favorite_setup": False,
            "is_near_favorite_zone": False,
            "clean_first_zone_score_bonus": 0,
            "clean_first_zone_score_note": "",
            "recent_tracker_score_delta": 0,
            "recent_tracker_score_note": "",
            "setup_type_score_delta": 0,
            "setup_type_score_note": "",
            "tracker_guardrail_score_delta": 0.0,
            "tracker_guardrail_score_note": "",
            "watch_only": False,
            "short_confirmation_reasons": [],
            "market_regime_score_delta": 0,
            "market_regime_score_note": "",
            "rejection_score_cap": None,
            "rejection_score_cap_delta": 0.0,
            "rejection_score_cap_note": "",
            "favorite_signals": ";".join(priority_summary["favorite_signals"]),
            "favorite_context_signals": ";".join(priority_summary["context_signals"]),
            "events_today": ";".join(symbol_events_today),
        }
        symbol_entry["feature_row"] = feature_row
        feature_rows.append(feature_row)
        feature_rows_by_symbol[sym] = feature_row

        logging.info(
            f"{sym}: events_today={symbol_events_today}, "
            f"multi_day={symbol_multi_day}"
        )

    spy_benchmark = (market_regime_snapshot.get("benchmarks", {}) or {}).get("SPY", {}) or {}
    sides_by_symbol = {
        sym: ("LONG" if sym in longs else "SHORT") for sym in daily_frames_by_symbol
    }
    industry_context_by_symbol = build_symbol_industry_contexts(daily_frames_by_symbol.keys())
    industry_etfs = sorted(
        {
            str(context.get("industry_etf") or "").strip().upper()
            for context in industry_context_by_symbol.values()
            if isinstance(context, dict)
            and str(context.get("industry_etf") or "").strip().upper()
            and str(context.get("industry_etf") or "").strip().upper() != "SPY"
        }
    )
    industry_daily_frames_by_etf = {}
    for industry_etf in industry_etfs:
        etf_df = fetch_daily_bars(ib, industry_etf, MARKET_PREP_INDUSTRY_LOOKBACK_DAYS)
        if etf_df is not None and not etf_df.empty:
            industry_daily_frames_by_etf[industry_etf] = etf_df.copy()
    # Convert each daily frame to rows + ATR20 once and share across the
    # universe-strength and HV-level enrichers (avoids re-running the per-symbol
    # frame->rows + ATR conversion two extra times across the pipeline).
    daily_rows_cache = build_daily_rows_cache(daily_frames_by_symbol)
    industry_rows_by_etf = build_daily_rows_cache(industry_daily_frames_by_etf)
    industry_rows_by_etf = {etf: entry["rows"] for etf, entry in industry_rows_by_etf.items()}
    universe_strength_rows = build_universe_strength_rows(
        daily_frames_by_symbol,
        spy_benchmark,
        sides_by_symbol=sides_by_symbol,
        industry_context_by_symbol=industry_context_by_symbol,
        industry_daily_frames_by_etf=industry_daily_frames_by_etf,
        daily_rows_cache=daily_rows_cache,
        industry_rows_by_etf=industry_rows_by_etf,
    )
    industry_strength_rows = build_industry_strength_rows(
        industry_daily_frames_by_etf,
        spy_benchmark,
        industry_context_by_symbol=industry_context_by_symbol,
    )
    enrich_priority_rows_with_industry_relative_strength(
        priority_rows,
        universe_strength_rows,
        industry_strength_rows,
        ai_state=ai_state,
        feature_rows_by_symbol=feature_rows_by_symbol,
    )
    _phase_t = _log_phase_duration("prep+fetch+priority", _phase_t)
    htf_trend_study_rows = enrich_priority_rows_with_htf_trend_context(
        priority_rows,
        ib=ib,
        ai_state=ai_state,
        feature_rows_by_symbol=feature_rows_by_symbol,
    )
    hv_level_stores_by_symbol: dict[str, dict] = {}
    hv_level_study_rows = enrich_priority_rows_with_hv_levels(
        priority_rows,
        daily_frames_by_symbol,
        earnings_dates_by_symbol=earnings_data,
        ai_state=ai_state,
        feature_rows_by_symbol=feature_rows_by_symbol,
        stores_out=hv_level_stores_by_symbol,
        daily_rows_cache=daily_rows_cache,
        ib=ib,
    )

    refine_priority_rows_with_directional_filters(
        priority_rows,
        ai_state,
        ib,
        daily_frames_by_symbol=daily_frames_by_symbol,
    )
    phase6_study_rows = enrich_priority_rows_with_phase6_studies(
        priority_rows,
        daily_frames_by_symbol,
        earnings_dates_by_symbol=earnings_data,
        ai_state=ai_state,
        feature_rows_by_symbol=feature_rows_by_symbol,
        level_stores_by_symbol=hv_level_stores_by_symbol,
    )
    relative_avwap_study_rows = enrich_priority_rows_with_relative_avwap_studies(
        priority_rows,
        daily_frames_by_symbol,
        ai_state=ai_state,
        feature_rows_by_symbol=feature_rows_by_symbol,
    )
    first_dev_breakout_study_rows = enrich_priority_rows_with_first_dev_breakouts(
        priority_rows,
        daily_frames_by_symbol,
        ai_state=ai_state,
        feature_rows_by_symbol=feature_rows_by_symbol,
    )
    second_dev_breakout_study_rows = enrich_priority_rows_with_second_dev_breakouts(
        priority_rows,
        daily_frames_by_symbol,
        ai_state=ai_state,
        feature_rows_by_symbol=feature_rows_by_symbol,
    )
    weekly_ema8_hold_study_rows = enrich_priority_rows_with_weekly_ema8_hold(
        priority_rows,
        daily_frames_by_symbol,
        ai_state=ai_state,
        feature_rows_by_symbol=feature_rows_by_symbol,
    )
    playbook_study_rows = enrich_priority_rows_with_playbook_studies(
        priority_rows,
        daily_frames_by_symbol,
        ai_state=ai_state,
        feature_rows_by_symbol=feature_rows_by_symbol,
    )
    study_rows = [
        *htf_trend_study_rows,
        *hv_level_study_rows,
        *phase6_study_rows,
        *relative_avwap_study_rows,
        *first_dev_breakout_study_rows,
        *second_dev_breakout_study_rows,
        *weekly_ema8_hold_study_rows,
        *playbook_study_rows,
    ]
    apply_pre_earnings_priority_blocks(priority_rows, ai_state, feature_rows_by_symbol)
    apply_post_earnings_hard_rule_blocks(priority_rows, ai_state, feature_rows_by_symbol)
    apply_final_priority_buckets(priority_rows, ai_state, csv_rows, feature_rows_by_symbol)
    apply_clean_first_zone_score_bonus(priority_rows, ai_state, feature_rows_by_symbol)
    _phase_t = _log_phase_duration("studies+enrichment", _phase_t)
    # The tracker payload lives on a cloud-synced drive and is large (hundreds of MB);
    # load it once and thread it through every consumer this scan instead of letting
    # each re-parse the file. update_setup_tracker_from_scan mutates + saves this same
    # object below, so calibrate_expected_r_prior_anchors then reads the post-update
    # state in memory with no extra reload.
    tracker_payload = load_setup_tracker_payload()
    recent_family_rows = apply_recent_tracker_setup_family_adjustments(
        priority_rows,
        ai_state,
        feature_rows_by_symbol,
        tracker_payload=tracker_payload,
        reference_date=today_run,
    )
    apply_tracker_setup_type_adjustments(
        priority_rows,
        ai_state,
        feature_rows_by_symbol,
        tracker_payload=tracker_payload,
    )
    # BounceBot's intraday SPY-pause defiance observations (today only) ride
    # in as capped swing-row evidence before the guardrails/caps run.
    regime_pause_boosted = apply_regime_pause_evidence_adjustments(
        priority_rows,
        ai_state,
        feature_rows_by_symbol,
        reference_date=today_run,
    )
    if regime_pause_boosted:
        logging.info(
            "Regime-pause defiance evidence boosted %d swing row(s).", regime_pause_boosted
        )
    apply_tracker_scoring_guardrails(priority_rows, ai_state, feature_rows_by_symbol)
    apply_market_regime_score_adjustments(priority_rows, ai_state, feature_rows_by_symbol)
    apply_priority_rejection_score_caps(priority_rows, ai_state, feature_rows_by_symbol)
    apply_final_priority_buckets(priority_rows, ai_state, csv_rows, feature_rows_by_symbol)
    attach_setup_candidate_payloads(priority_rows, ai_state, feature_rows_by_symbol)
    # D1 Focus Alerts: diff this scan's final buckets vs the persisted state and keep
    # only genuine upgrades into Favorite / High Conviction (see GUI_REDESIGN_PLAN.md).
    from master_avwap_bucket_state import record_scan_bucket_upgrades

    bucket_upgrades = record_scan_bucket_upgrades(priority_rows, today_run)
    apply_expected_r_ranking(
        priority_rows,
        ai_state,
        feature_rows_by_symbol,
        recent_family_rows=recent_family_rows,
    )
    # Study rows are pre-ranking shallow copies; refresh their score/ExpR so
    # capped originals don't leave stale stacked scores on the study clones.
    sync_study_row_ranking_fields(study_rows, priority_rows)
    d1_upgrade_alert_payload = write_master_avwap_d1_upgrade_alert_outputs(
        alerts_path=MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE,
        report_path=MASTER_AVWAP_D1_UPGRADE_ALERTS_REPORT_FILE,
        priority_rows=priority_rows,
        ai_state=ai_state,
        bucket_upgrades=bucket_upgrades,
    )
    try:
        from human_focus_tracking import (
            load_human_focus_map_for_date,
            mark_human_focus_rows,
            update_human_focus_tracking,
        )

        human_focus_market_date = get_market_session_window().market_date
        human_focus_tracking_result = update_human_focus_tracking(
            market_date=human_focus_market_date,
            daily_frames_by_symbol=daily_frames_by_symbol,
        )
        human_focus_marked_count = mark_human_focus_rows(
            priority_rows,
            feature_rows_by_symbol,
            focus_map=load_human_focus_map_for_date(human_focus_market_date),
        )
    except Exception:
        logging.exception("Human focus snapshot/performance tracking failed.")
        human_focus_tracking_result = {
            "snapshot": {"snapshotted": False, "reason": "error"},
            "outcomes": {"outcome_rows": 0, "updated_outcomes": 0, "performance_rows": 0},
        }
        human_focus_marked_count = 0
    existing_d1_watchlist_payload = load_json(MASTER_AVWAP_D1_WATCHLIST_FILE, default={})
    if not isinstance(existing_d1_watchlist_payload, dict):
        existing_d1_watchlist_payload = {}
    logging.info(
        "Priority scoring complete; theta option strike/price enrichment will run after ranking outputs are written "
        "(%d sold-put row(s), %d PCS row(s)).",
        len(theta_put_rows),
        len(theta_pcs_rows),
    )
    tracked_rows = [
        row
        for row in priority_rows
        if row.get("priority_bucket") in {"favorite_setup", "near_favorite_zone"}
        and not row.get("ranking_blocked")
    ]
    run_result: dict[str, object] = {
        "watchlist_label": watchlist_label,
        "tracked_rows": tracked_rows,
        "bucket_upgrades": bucket_upgrades,
        "theta_put_rows": theta_put_rows,
        "theta_pcs_rows": theta_pcs_rows,
        "ai_state": ai_state,
        "feature_rows_by_symbol": feature_rows_by_symbol,
        "daily_frames_by_symbol": daily_frames_by_symbol,
        "universe_strength_rows": universe_strength_rows,
        "industry_strength_rows": industry_strength_rows,
        "htf_trend_study_rows": htf_trend_study_rows,
        "htf_trend_study_count": len(htf_trend_study_rows),
        "hv_level_study_rows": hv_level_study_rows,
        "hv_level_study_count": len(hv_level_study_rows),
        "phase6_study_rows": phase6_study_rows,
        "phase6_study_count": len(phase6_study_rows),
        "relative_avwap_study_rows": relative_avwap_study_rows,
        "relative_avwap_study_count": len(relative_avwap_study_rows),
        "first_dev_breakout_study_rows": first_dev_breakout_study_rows,
        "first_dev_breakout_study_count": len(first_dev_breakout_study_rows),
        "second_dev_breakout_study_rows": second_dev_breakout_study_rows,
        "second_dev_breakout_study_count": len(second_dev_breakout_study_rows),
        "weekly_ema8_hold_study_rows": weekly_ema8_hold_study_rows,
        "weekly_ema8_hold_study_count": len(weekly_ema8_hold_study_rows),
        "playbook_study_rows": playbook_study_rows,
        "playbook_study_count": len(playbook_study_rows),
        "d1_watchlist_scan_symbols_added": d1_watchlist_added,
        "human_focus_tracking": human_focus_tracking_result,
        "human_focus_marked_setup_count": human_focus_marked_count,
        "setup_tracker_updated": False,
        "study_setups_tracked": 0,
        "setup_tracker_allowed": False,
        "setup_tracker_skip_reason": "",
        "theta_enrichment_pending": False,
        "theta_enrichment_mode": "deferred",
        "d1_upgrade_alert_count": len(d1_upgrade_alert_payload.get("alerts", [])),
        "d1_watchlist_symbol_count": len(existing_d1_watchlist_payload.get("symbols", {})),
    }
    setup_tracker_allowed = (
        bool(update_setup_tracker)
        if update_setup_tracker is not None
        else should_update_setup_tracker_now()
    )
    setup_tracker_skip_reason = ""
    run_result["setup_tracker_allowed"] = bool(setup_tracker_allowed)

    if setup_tracker_allowed and require_ib_for_setup_tracker:
        if not is_daily_data_client_connected(ib):
            setup_tracker_allowed = False
            setup_tracker_skip_reason = (
                "Setup tracker refresh skipped for this mini-PC run because the IBKR daily-bar "
                "client was unavailable."
            )
        else:
            tracked_symbols = sorted(
                {
                    str(row.get("symbol", "")).strip().upper()
                    for row in tracked_rows
                    if str(row.get("symbol", "")).strip()
                }
            )
            non_ib_symbols = [
                symbol
                for symbol in tracked_symbols
                if _get_daily_bar_source(daily_frames_by_symbol.get(symbol)) != DAILY_BAR_SOURCE_IBKR
            ]
            if non_ib_symbols:
                sources = sorted(
                    {
                        _get_daily_bar_source(daily_frames_by_symbol.get(symbol)) or "unknown"
                        for symbol in non_ib_symbols
                    }
                )
                setup_tracker_allowed = False
                setup_tracker_skip_reason = (
                    "Setup tracker refresh skipped for this mini-PC run because tracked setups "
                    f"used non-IBKR daily data (symbols={', '.join(non_ib_symbols)}; "
                    f"sources={', '.join(sources)})."
                )
    run_result["setup_tracker_allowed"] = bool(setup_tracker_allowed)
    run_result["setup_tracker_skip_reason"] = setup_tracker_skip_reason

    _phase_t = _log_phase_duration("tracker scoring+ranking", _phase_t)
    if setup_tracker_allowed:
        control_rows = select_tracker_control_rows(
            priority_rows,
            tracked_rows,
            scan_date=today_run.isoformat() if hasattr(today_run, "isoformat") else str(today_run or ""),
        )
        update_setup_tracker_from_scan(
            tracked_rows,
            ai_state,
            feature_rows_by_symbol,
            daily_frames_by_symbol,
            ib,
            control_rows=control_rows,
            study_rows=study_rows,
            tracker_payload=tracker_payload,
        )
        run_result["setup_tracker_updated"] = True
        run_result["control_setups_tracked"] = len(control_rows)
        run_result["study_setups_tracked"] = len(study_rows)
        logging.info(
            "Setup tracker updated for %s tracked symbol(s); %s control/holdout setup(s); %s study setup(s).",
            len(tracked_rows),
            len(control_rows),
            len(study_rows),
        )
        # Re-fit the Expected-R prior anchors to the freshly-updated closed
        # outcomes so the next scan's headline ranking is grounded in this
        # trader's own realized R (no-op until enough closed history exists).
        # tracker_payload is the just-updated in-memory tracker, so no reload.
        calibrate_expected_r_prior_anchors(tracker_payload=tracker_payload, persist=True)
        try:
            refresh_playbook_study_if_stale()
        except Exception as exc:  # never let the study block the scan pipeline
            logging.warning("Playbook study refresh failed (non-fatal): %s", exc)
        try:
            # BounceBot's alert-time learning state rides the same after-close
            # window so day-trade tiers/mutes stay current even on days the
            # bounce bot itself is not restarted.
            from bounce_bot_lib.learning import refresh_bounce_learning_if_stale

            if refresh_bounce_learning_if_stale():
                logging.info("Bounce learning state refreshed after close.")
        except Exception as exc:
            logging.warning("Bounce learning refresh failed (non-fatal): %s", exc)
    else:
        if setup_tracker_skip_reason:
            logging.info(setup_tracker_skip_reason)
        elif update_setup_tracker is None:
            window_start, window_end = get_setup_tracker_update_window_labels()
            logging.info(
                "Setup tracker refresh skipped for this run because local time is before the final-hour/after-close update window (starts %s; close %s).",
                window_start,
                window_end,
            )
        else:
            logging.info(
                "Setup tracker refresh skipped for this run; final scheduled slot will refresh stored setups."
            )

    _phase_t = _log_phase_duration("tracker update+calibrate", _phase_t)
    disconnect_daily_data_client(ib)

    if csv_rows:
        df_signals = pd.DataFrame(csv_rows)
        df_signals = df_signals.reindex(columns=AVWAP_CSV_COLUMNS)
        df_signals.sort_values(["run_date", "trade_date", "symbol", "signal_type"], inplace=True)
        new_signal_count = len(df_signals)

        if AVWAP_SIGNALS_FILE.exists() and AVWAP_SIGNALS_FILE.stat().st_size > 0:
            existing_signals = pd.read_csv(AVWAP_SIGNALS_FILE)
            existing_signals = existing_signals.reindex(columns=AVWAP_CSV_COLUMNS)
            df_signals = pd.concat([existing_signals, df_signals], ignore_index=True)
            df_signals.sort_values(["run_date", "trade_date", "symbol", "signal_type"], inplace=True)

        df_signals.to_csv(
            AVWAP_SIGNALS_FILE,
            index=False,
        )
        logging.info(
            f"Appended {new_signal_count} AVWAP signals to {AVWAP_SIGNALS_FILE} "
            f"({len(df_signals)} total rows)."
        )
    else:
        logging.info(
            f"No AVWAP signals generated for {today_run.isoformat()}; nothing appended."
        )

    # trim history to last N days
    trim_history(history)
    write_priority_setup_report(PRIORITY_SETUPS_FILE, priority_rows)
    write_theta_put_report(THETA_PUTS_FILE, theta_put_rows, theta_pcs_rows)
    favorite_watchlist_reference = datetime.now()
    favorite_watchlist_result = write_favorite_zone_watchlist_outputs(
        focus_path=MASTER_AVWAP_FOCUS_FILE,
        d1_watchlist_path=MASTER_AVWAP_D1_WATCHLIST_FILE,
        priority_rows=priority_rows,
        theta_put_rows=theta_put_rows,
        theta_pcs_rows=theta_pcs_rows,
        ai_state=ai_state,
        study_rows=study_rows,
        now=favorite_watchlist_reference,
    )
    run_result["favorite_zone_watchlists_updated"] = bool(favorite_watchlist_result.get("updated"))
    run_result["favorite_zone_watchlists_allowed"] = bool(favorite_watchlist_result.get("allowed"))
    run_result["favorite_zone_watchlists_skip_reason"] = favorite_watchlist_result.get("skip_reason", "")
    updated_d1_watchlist = favorite_watchlist_result.get("d1_watchlist")
    if isinstance(updated_d1_watchlist, dict):
        run_result["d1_watchlist_symbol_count"] = len(updated_d1_watchlist.get("symbols", {}))
    favorite_watchlist_context = {
        "allowed": bool(favorite_watchlist_result.get("allowed")),
        "reference_time": favorite_watchlist_reference,
        "window_start": favorite_watchlist_result.get("window_start"),
        "window_end": favorite_watchlist_result.get("window_end"),
    }

    # write human-readable events file (grouped for easier scanning)
    sorted_events = sort_events_for_output(events_for_output)
    _write_master_avwap_output_file(sorted_events, range_buckets)

    # write grouped ticker lists for easy copy/paste into TradingView/TC2000
    event_buckets = {}
    for sym, _, lbl, side in sorted_events:
        event_buckets.setdefault(lbl, {"LONG": [], "SHORT": []})[side].append(sym)

    def _fmt_items(values):
        return ", ".join(sorted(set(values))) if values else "None"

    event_buffer = io.StringIO()
    f = event_buffer
    f.write("AVWAP crosses and bounces by event type\n")
    f.write(f"Priority setups report: {PRIORITY_SETUPS_FILE.name}\n\n")
    for lbl in sorted(event_buckets.keys(), key=event_label_sort_key):
        for side in ("LONG", "SHORT"):
            tickers = sorted(set(event_buckets[lbl][side]))
            if not tickers:
                continue
            display_label = format_signal_label(lbl)
            f.write(f"{display_label}, {side.capitalize()}: {', '.join(tickers)}\n")

    f.write("\nPrice ranges (current anchors)\n")
    range_labels = [
        ("Longs between AVWAP and UPPER_1", "long_avwap_to_upper_1"),
        ("Longs between UPPER_1 and UPPER_2", "long_upper_1_to_upper_2"),
        ("Shorts between AVWAP and LOWER_1", "short_avwap_to_lower_1"),
        ("Shorts between LOWER_1 and LOWER_2", "short_lower_1_to_lower_2"),
    ]
    for label, key in range_labels:
        f.write(f"{label}: {_fmt_items(range_buckets[key])}\n")
    _write_text_atomic(EVENT_TICKERS_FILE, event_buffer.getvalue().rstrip() + "\n")

    write_stdev_range_report(
        STDEV_RANGE_FILE,
        stdev_range_hits,
        stdev_cross_hits,
        priority_rows=priority_rows,
    )
    write_tradingview_report(
        TRADINGVIEW_REPORT_FILE,
        priority_rows,
        event_buckets,
        range_buckets,
        stdev_range_hits,
        stdev_cross_hits,
    )
    market_prep_payload = build_market_prep_payload(
        range_buckets=range_buckets,
        market_prep_range_buckets=market_prep_range_buckets,
        priority_rows=priority_rows,
        latest_release_map=latest_release_map,
        reference_date=today_run,
        daily_frames_by_symbol=daily_frames_by_symbol,
        spy_benchmark=spy_benchmark,
        sides_by_symbol=sides_by_symbol,
        universe_strength_rows=universe_strength_rows,
        industry_context_by_symbol=industry_context_by_symbol,
        industry_daily_frames_by_etf=industry_daily_frames_by_etf,
        industry_strength_rows=industry_strength_rows,
    )
    write_market_prep_files(market_prep_payload)
    run_result["market_prep_payload"] = market_prep_payload

    feature_columns = [
        "symbol",
        "side",
        "last_trade_date",
        "daily_bar_source",
        "market_regime_label",
        "market_regime_score",
        "spy_one_day_return_pct",
        "spy_five_day_return_pct",
        "spy_above_sma20",
        "spy_above_sma50",
        "sector",
        "industry",
        "sector_etf",
        "industry_etf",
        "industry_one_day_return_pct",
        "industry_five_day_return_pct",
        "industry_13w_return_pct",
        "rs_vs_industry",
        "rs_vs_industry_1d",
        "rs_vs_industry_5d",
        "industry_rs_consistent",
        "return_13w_vs_industry_pct",
        "industry_relative_strength_score",
        "industry_daily_relative_strength_score",
        "industry_relative_strength_bonus",
        "industry_relative_strength_note",
        "last_close",
        "last_volume",
        "atr20",
        "previous_day_date",
        "previous_day_high",
        "previous_day_low",
        "current_active_level",
        "current_nearby_bands",
        "current_band_zone",
        "previous_active_level",
        "previous_nearby_bands",
        "previous_band_zone",
        "current_anchor_date",
        "current_anchor_vwap",
        "current_anchor_stdev",
        "distance_from_current_vwap",
        "pct_from_current_vwap",
        "distance_from_current_upper_1",
        "pct_from_current_upper_1",
        "distance_from_current_lower_1",
        "pct_from_current_lower_1",
        "trend_20d",
        "htf_trend_1h",
        "htf_trend_4h",
        "htf_trend_aligned",
        "htf_retest_confirmed",
        "htf_retest_sma",
        "htf_retest_timeframes",
        "htf_retest_age_bars",
        "htf_intraday_bar_count",
        "htf_4h_bar_count",
        "htf_trend_score_bonus",
        "htf_trend_note",
        "hv_level_nearby_count",
        "hv_level_blocking_count",
        "hv_level_break_today",
        "hv_level_nearest_price",
        "hv_level_nearest_bucket",
        "hv_level_nearest_distance_atr",
        "hv_level_nearby_summary",
        "hv_level_blocking_summary",
        "hv_level_note",
        "cloud_level_nearby_count",
        "cloud_level_nearest_price",
        "cloud_level_nearest_distance_atr",
        "cloud_level_effective_range",
        "cloud_level_nearby_summary",
        "cloud_level_note",
        "compression_break_today",
        "compression_break_direction",
        "compression_break_level",
        "compression_break_distance_atr",
        "compression_break_prior_bars",
        "compression_break_note",
        "previous_day_range_break",
        "previous_day_range_break_bonus",
        "previous_day_range_note",
        "trend_ma_alignment",
        "trend_ma_alignment_bonus",
        "trend_ma_alignment_note",
        "has_bounce_event_today",
        "favorite_zone",
        "setup_family",
        "setup_tags",
        "latest_release_gap_date",
        "latest_release_sessions_since_gap",
        "next_earnings_date",
        "days_to_next_earnings",
        "pre_earnings_setup_blocked",
        "pre_earnings_setup_block_reason",
        "post_earnings_hard_rule_blocked",
        "post_earnings_hard_rule_reason",
        "recent_band_extension_days",
        "recent_second_band_test_days",
        "second_band_penalty",
        "breakout_5d",
        "retest_followthrough",
        "retest_reference_level",
        "retest_note",
        "extreme_move_watch",
        "extreme_move_favorite_ready",
        "extreme_move_retest_level",
        "extreme_move_band_crosses",
        "extreme_move_band_width_atr",
        "extreme_move_displacement_date",
        "extreme_move_note",
        "extension_note",
        "compression_flag",
        "compression_penalty",
        "compression_note",
        "post_earnings_active",
        "post_earnings_monitor_level",
        "post_earnings_break_intraday",
        "post_earnings_break_close",
        "post_earnings_sessions_since_gap",
        "post_earnings_gap_atr_multiple",
        "post_earnings_bounce_date",
        "post_earnings_bounce_age_sessions",
        "post_earnings_note",
        "mid_earnings_watch",
        "mid_earnings_active_second_stdev_hold",
        "mid_earnings_sessions_since_gap",
        "mid_earnings_zone_streak_days",
        "mid_earnings_zone_start_date",
        "mid_earnings_zone_end_date",
        "mid_earnings_primary_trigger_level",
        "mid_earnings_retest_date",
        "mid_earnings_sessions_after_zone",
        "mid_earnings_trigger_levels",
        "mid_earnings_ema15_trigger",
        "mid_earnings_ema21_trigger",
        "mid_earnings_first_dev_trigger",
        "mid_earnings_ema8_confluence",
        "mid_earnings_ema21_confluence",
        "mid_earnings_first_dev_confluence",
        "mid_earnings_note",
        "top_pattern_watch",
        "top_pattern_entry",
        "top_pattern_signal",
        "top_pattern_entry_trigger",
        "top_pattern_entry_triggers",
        "top_pattern_entry_level",
        "top_pattern_score_bonus",
        "top_pattern_note",
        "top_pattern_weekly_return_13w_pct",
        "top_pattern_weekly_return_26w_pct",
        "top_pattern_weekly_pullback_from_52w_high_pct",
        "top_pattern_weekly_ema15_hold",
        "top_pattern_weekly_ema15_hold_ratio",
        "top_pattern_weekly_above_sma100",
        "top_pattern_weekly_sma50_retest_recent",
        "top_pattern_daily_sma50_reclaim",
        "top_pattern_daily_sma50_bounce",
        "top_pattern_weekly_sma50_entry",
        "top_pattern_avwape_retest",
        "priority_score",
        "priority_bucket",
        "is_favorite_setup",
        "is_near_favorite_zone",
        "clean_first_zone_score_bonus",
        "clean_first_zone_score_note",
        "recent_tracker_score_delta",
        "recent_tracker_score_note",
        "setup_type_score_delta",
        "setup_type_score_note",
        "tracker_guardrail_score_delta",
        "tracker_guardrail_score_note",
        "watch_only",
        "short_confirmation_reasons",
        "market_regime_score_delta",
        "market_regime_score_note",
        "rejection_score_cap",
        "rejection_score_cap_delta",
        "rejection_score_cap_note",
        "favorite_signals",
        "favorite_context_signals",
        "events_today",
        "candidate_rejection_reasons",
        "setup_candidate_json",
    ]

    df_features = pd.DataFrame(feature_rows, columns=feature_columns)
    df_features.to_csv(D1_FEATURES_FILE, index=False)
    append_d1_feature_history(
        df_features,
        {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "run_date": today_run.isoformat(),
            "watchlist_label": watchlist_label,
            "scoring_config_hash": scoring_config_metadata.get("hash", ""),
            "scoring_config_updated_at": scoring_config_metadata.get("updated_at", ""),
        },
    )
    try:
        scan_factor_result = export_scan_factor_views()
        run_result["scan_factor_observation_count"] = int(scan_factor_result.get("observation_count", 0) or 0)
        run_result["scan_factor_leaderboard_count"] = int(scan_factor_result.get("leaderboard_count", 0) or 0)
        logging.info(
            "Scan factor tracker exported %s observation row(s) and %s leaderboard row(s).",
            run_result["scan_factor_observation_count"],
            run_result["scan_factor_leaderboard_count"],
        )
    except Exception:
        run_result["scan_factor_observation_count"] = 0
        run_result["scan_factor_leaderboard_count"] = 0
        logging.exception("Scan factor tracker export failed.")
    try:
        tier_tracker_result = export_bot_tier_tracker_views()
        run_result["tier_pick_count"] = int(tier_tracker_result.get("tier_pick_count", 0) or 0)
        run_result["tier_outcome_count"] = int(tier_tracker_result.get("tier_outcome_count", 0) or 0)
        run_result["tier_catch_rate_count"] = int(tier_tracker_result.get("tier_catch_rate_count", 0) or 0)
        logging.info(
            "Bot tier tracker exported %s current pick(s), %s outcome row(s), and %s catch-rate row(s).",
            run_result["tier_pick_count"],
            run_result["tier_outcome_count"],
            run_result["tier_catch_rate_count"],
        )
    except Exception:
        run_result["tier_pick_count"] = 0
        run_result["tier_outcome_count"] = 0
        run_result["tier_catch_rate_count"] = 0
        logging.exception("Bot tier tracker export failed.")

    positions_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "levels": POSITION_LEVELS,
        "current": {
            lvl: sorted(set(positions["current"].get(lvl, [])))
            for lvl in POSITION_LEVELS
        },
        "previous": {
            lvl: sorted(set(positions["previous"].get(lvl, [])))
            for lvl in POSITION_LEVELS
        },
    }

    with open(MASTER_POSITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(positions_payload, f, indent=2)

    save_json(CURRENT_CACHE_FILE, curr_cache)
    save_json(PREV_CACHE_FILE, prev_cache)
    save_history(history)
    save_json(AI_STATE_FILE, ai_state)

    theta_enrichment_pending = _schedule_deferred_theta_enrichment(
        run_id=run_id,
        theta_put_rows=theta_put_rows,
        theta_pcs_rows=theta_pcs_rows,
        priority_rows=priority_rows,
        ai_state=ai_state,
        sorted_events=sorted_events,
        range_buckets=range_buckets,
        reference_date=today_run,
        favorite_watchlist_context=favorite_watchlist_context,
    )
    run_result["theta_enrichment_pending"] = theta_enrichment_pending
    if not theta_enrichment_pending:
        run_result["theta_enrichment_mode"] = "not_needed"

    _phase_t = _log_phase_duration("output writes", _phase_t)
    _log_phase_duration("TOTAL (theta enrichment deferred)", _run_t0)
    logging.info(
        f"Master AVWAP run complete. "
        f"Events: {OUTPUT_FILE}, AI state: {AI_STATE_FILE}, history: {HISTORY_FILE}"
    )
    return run_result


def launch_gui():
    from .gui import launch_gui as _launch_gui

    return _launch_gui()


def main():
    from .gui import main as _main

    return _main()


run_anchor_watchlist_scan = _legacy.run_anchor_watchlist_scan

__all__ = [
    "run_master",
    "run_master_with_shared_watchlists",
    "run_anchor_watchlist_scan",
    "launch_gui",
    "main",
]
