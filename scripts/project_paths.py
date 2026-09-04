from __future__ import annotations

import contextlib
import filecmp
import logging.handlers
import os
import threading
import re
import shutil
import json
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that tolerates a locked target file on rollover.

    On Windows a log file can be held open by another process - historically a
    sync client, today an antivirus scanner or a tailing editor - so the
    ``os.rename`` in ``doRollover`` raises PermissionError (WinError 32). The
    stock handler then re-raises that on every subsequent record, flooding the
    console with rollover tracebacks. This keeps writing to the current file and
    backs off, retrying the rotation later.
    """

    _ROLLOVER_BACKOFF_SECONDS = 60.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rollover_blocked_until = 0.0

    def shouldRollover(self, record):  # noqa: N802 (logging override naming)
        if time.monotonic() < self._rollover_blocked_until:
            return False
        return super().shouldRollover(record)

    def doRollover(self):  # noqa: N802
        try:
            super().doRollover()
            self._rollover_blocked_until = 0.0
        except OSError:
            # Target file is locked (commonly Drive/OneDrive sync). Keep logging
            # to the current file and retry rotation after a cool-off rather than
            # raising, which would spam a rollover traceback on every record.
            self._rollover_blocked_until = time.monotonic() + self._ROLLOVER_BACKOFF_SECONDS
            if self.stream is None and not self.delay:
                try:
                    self.stream = self._open()
                except OSError:
                    pass


ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_DATA_DIR = ROOT_DIR / "data"
REPO_OUTPUT_DIR = ROOT_DIR / "output"
REPO_LOG_DIR = ROOT_DIR / "logs"


def _adopt_legacy_windows_shaped_dir(legacy: Path, preferred: Path) -> Path:
    """Migrate a POSIX machine off the Windows-shaped ``~/AppData`` fallback.

    Early macOS/Linux runs (before the platform-native branch existed) wrote
    everything to ``~/AppData/Local/TradingBotV3`` — including user-authored
    watchlists and the local_settings.json that names the shared home folder.
    Losing either is a data-loss bug, so the move is deliberately cautious:

    - copy → byte-verify EVERY file first; sources are deleted only after the
      whole set verified, so a failure at any point leaves the legacy store
      complete and authoritative;
    - a file that exists on both sides with different bytes is preserved next
      to the preferred copy as ``<name>.from-appdata`` — nothing is ever
      overwritten;
    - the emptied legacy directory is left in place (an empty husk), not
      deleted;
    - idempotent: leftovers verify as identical on the next launch and are
      cleaned up then.

    Returns the directory the app should use. Windows never reaches this
    code — LOCALAPPDATA is set there and wins in the caller.
    """
    preferred_existed = preferred.exists()
    fallback = preferred if preferred_existed else legacy  # the pre-migration selection rule
    try:
        files = sorted(path for path in legacy.rglob("*") if path.is_file())
    except OSError:
        return fallback
    if not files:
        return preferred

    verified_sources: list[Path] = []
    try:
        for source in files:
            destination = preferred / source.relative_to(legacy)
            if destination.exists() and not filecmp.cmp(source, destination, shallow=False):
                destination = destination.with_name(destination.name + ".from-appdata")
                if destination.exists() and not filecmp.cmp(source, destination, shallow=False):
                    return fallback  # even the conflict slot is taken by different bytes
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            if not filecmp.cmp(source, destination, shallow=False):
                return fallback
            verified_sources.append(source)
    except OSError:
        return fallback

    # Every file has a verified copy under `preferred`: it is now the
    # authoritative store regardless of how cleanup below fares.
    for source in verified_sources:
        with contextlib.suppress(OSError):
            source.unlink()
    for directory in sorted((path for path in legacy.rglob("*") if path.is_dir()), reverse=True):
        with contextlib.suppress(OSError):
            directory.rmdir()
    return preferred


def _default_local_settings_dir() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "TradingBotV3"

    if sys.platform == "darwin":
        preferred = Path.home() / "Library" / "Application Support" / "TradingBotV3"
    else:
        preferred = Path.home() / ".local" / "share" / "TradingBotV3"

    legacy = Path.home() / "AppData" / "Local" / "TradingBotV3"
    if legacy.exists():
        return _adopt_legacy_windows_shaped_dir(legacy, preferred)
    return preferred


LOCAL_SETTINGS_DIR = _default_local_settings_dir()
LOCAL_SETTINGS_FILE = LOCAL_SETTINGS_DIR / "local_settings.json"


#: (mtime_ns, size, payload) for the settings file as last read. Keyed on the
#: file's own stamp rather than a timeout, so an edit is picked up on the very
#: next call and an unchanged file is never parsed twice.
_local_settings_cache: tuple[int, int, dict] | None = None
_local_settings_lock = threading.Lock()


def _load_local_settings() -> dict:
    """The machine-local settings, parsed at most once per change.

    Read on **every** ``get_local_setting`` call - there are around a hundred
    call sites, including the market-timezone lookup that the completed-bar rule
    and both gates make per symbol. Uncached, each one cost an ``exists()``, a
    ``read_text()`` and a ``json.loads()``, and the GUI-thread stall log for
    2026-08-21 caught the main thread inside this function 97 times.

    A copy is returned, never the cached dict: callers such as
    ``save_local_setting`` mutate what they get back, and handing out the cache
    itself would let one caller's edit appear in every later read without ever
    reaching disk.
    """
    global _local_settings_cache
    try:
        stat = LOCAL_SETTINGS_FILE.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        with _local_settings_lock:
            _local_settings_cache = None
        return {}
    with _local_settings_lock:
        cached = _local_settings_cache
        if cached is not None and (cached[0], cached[1]) == stamp:
            return dict(cached[2])
    try:
        payload = json.loads(LOCAL_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    with _local_settings_lock:
        _local_settings_cache = (stamp[0], stamp[1], dict(payload))
    return payload


def invalidate_local_settings_cache() -> None:
    """Drop the parsed copy. Called after every write in this module.

    A same-millisecond write can land on an unchanged mtime on some filesystems,
    so writers say so explicitly rather than trusting the stamp to move.
    """
    global _local_settings_cache
    with _local_settings_lock:
        _local_settings_cache = None


def _resolve_persistent_data_dir() -> tuple[Path, str]:
    env_value = os.environ.get("TRADINGBOTV3_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser(), "environment"

    settings = _load_local_settings()
    config_value = settings.get("shared_data_dir")
    if isinstance(config_value, str) and config_value.strip():
        return Path(config_value).expanduser(), "local_config"

    # No cloud-drive auto-discovery (decision 0015, amended by packet R1).
    # This used to probe ~/Google Drive, ~/My Drive, $GOOGLE_DRIVE and the macOS
    # CloudStorage accounts and adopt the first writable one as the operational
    # store. Cloud sync is no part of this system, so an unset `shared_data_dir`
    # has to land somewhere plainly local rather than quietly inside whatever
    # sync folder happens to be mounted.
    return LOCAL_SETTINGS_DIR, "default_local"


PERSISTENT_DATA_DIR, PERSISTENT_DATA_DIR_SOURCE = _resolve_persistent_data_dir()


def _unmounted_shared_anchor(path: Path) -> Path | None:
    """Return the mount point startup must wait on, or None when none is missing.

    Generic on Windows: the configured store's drive letter or UNC root. A
    plain local home folder - what the desk actually uses - always has its
    anchor present, so this costs nothing there; it earns its place only if the
    store is ever pointed at a network or removable path.

    macOS keeps the File Provider mount check under ``~/Library/CloudStorage``,
    which decision 0015 explicitly blessed retaining until a macOS run is
    actually needed.

    Either way the point is the same: mkdir onto a missing mount would fork the
    store into a plain local folder that silently shadows the real one.
    """
    anchor = Path(path.anchor) if path.anchor else None
    if anchor is not None and str(anchor) not in ("", ".") and not anchor.exists():
        return anchor

    cloud_root = Path.home() / "Library" / "CloudStorage"
    try:
        relative = path.relative_to(cloud_root)
    except ValueError:
        return None
    if relative.parts:
        mount = cloud_root / relative.parts[0]
        if not mount.exists():
            return mount
    return None


def _wait_for_shared_store(path: Path, source: str) -> None:
    """Bounded wait for the shared store's mount point to appear.

    A network or removable store can come up late at boot, so an auto-started
    GUI can race it and die in ``_ensure_directories`` with a bare mkdir
    traceback. Waiting (default 120s, TRADINGBOTV3_DRIVE_WAIT_SECONDS to
    change, 0 = fail fast) rides out the normal delay; if the mount never
    appears the failure is a clear, actionable message instead.

    A local store returns immediately - its anchor is always present - so this
    is a no-op on the desk.

    Deliberately NO silent local fallback: the shared store carries the
    tracker, watchlists and outcome history, and quietly writing them to a
    different folder would fork that state.
    """
    anchor = _unmounted_shared_anchor(path)
    if anchor is None:
        return
    try:
        wait_seconds = float(os.environ.get("TRADINGBOTV3_DRIVE_WAIT_SECONDS", "120"))
    except (TypeError, ValueError):
        wait_seconds = 120.0
    if wait_seconds > 0:
        print(
            f"[TradingBotV3] Shared data store mount {anchor} is not available yet "
            f"(store: {path}, configured via {source}). Waiting up to "
            f"{int(wait_seconds)}s for it to appear...",
            file=sys.stderr,
            flush=True,
        )
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            time.sleep(2.0)
            if anchor.exists():
                print(f"[TradingBotV3] Mount {anchor} is available - continuing startup.", file=sys.stderr, flush=True)
                return
    raise RuntimeError(
        f"Shared data store mount {anchor} is unavailable, so the store at {path} is unreachable "
        f"(configured via {source}). Bring the mount up and relaunch - or point the app elsewhere via "
        "the TRADINGBOTV3_DATA_DIR environment variable or 'shared_data_dir' in local_settings.json. "
        "A silent local fallback is refused on purpose: it would fork the tracker/watchlist state. "
        "TRADINGBOTV3_DRIVE_WAIT_SECONDS adjusts the wait (0 = fail fast)."
    )


_wait_for_shared_store(PERSISTENT_DATA_DIR, PERSISTENT_DATA_DIR_SOURCE)

SHARED_HOME_DIR = PERSISTENT_DATA_DIR
DATA_DIR = PERSISTENT_DATA_DIR / "data"
OUTPUT_DIR = PERSISTENT_DATA_DIR / "output"
LOG_DIR = PERSISTENT_DATA_DIR / "logs"

LOCAL_MACHINE_CACHE_DIR = LOCAL_SETTINGS_DIR / "machine_cache"
CACHE_DIR = LOCAL_MACHINE_CACHE_DIR
DIAGNOSTICS_DIR_OVERRIDE_ENV = "TRADINGBOT_DIAGNOSTICS_DIR"


def get_diagnostics_dir() -> Path:
    """Machine-local diagnostics root, overridable for hermetic test runs."""
    override = str(os.environ.get(DIAGNOSTICS_DIR_OVERRIDE_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    return CACHE_DIR.parent / "diagnostics"
# Diagnostic app logs are per-machine and rotate (rename) frequently, which fights
# Kept on local disk rather than the shared store: caches are replaceable and
# would only bloat the operational folder and the hourly cold push to the DAS.
LOCAL_LOG_DIR = LOCAL_SETTINGS_DIR / "logs"
RUNTIME_DATA_DIR = DATA_DIR / "runtime"
REPORTS_DIR = OUTPUT_DIR / "reports"
AI_SUMMARY_EXPORT_DIR = REPORTS_DIR / "ai_summaries"
PERSISTENT_RUNTIME_DATA_DIR = RUNTIME_DATA_DIR

LONGS_FILE = PERSISTENT_DATA_DIR / "longs.txt"
SHORTS_FILE = PERSISTENT_DATA_DIR / "shorts.txt"
# The bot's own morning picks (gap + RS/RW open scan). Written every session
# regardless of Auto Pilot mode; BounceBot scans them like longs/shorts.txt so
# the bot's picks build their own outcome history separate from the trader's.
AUTO_LONGS_FILE = PERSISTENT_DATA_DIR / "autolongs.txt"
AUTO_SHORTS_FILE = PERSISTENT_DATA_DIR / "autoshorts.txt"
SWING_LONGS_FILE = PERSISTENT_DATA_DIR / "swinglongs.txt"
SWING_SHORTS_FILE = PERSISTENT_DATA_DIR / "shortswings.txt"

# Self-built scan universe (universe_builder.py). The master scan always folds
# these in alongside longs.txt / shorts.txt, which stay reserved for the
# trader's intraday M5 RS/RW dumps.
UNIVERSE_ALL_FILE = PERSISTENT_DATA_DIR / "universe_all.txt"
UNIVERSE_LONGS_FILE = PERSISTENT_DATA_DIR / "universe_longs.txt"
UNIVERSE_SHORTS_FILE = PERSISTENT_DATA_DIR / "universe_shorts.txt"

# Trader-curated daily Focus Picks (shared home, synced across machines) and the
# runtime files for human-pick membership tracking + the human-vs-bot cohort.
# See plan.md, Milestone 8 (Human focus lists).
FOCUS_LONGS_FILE = PERSISTENT_DATA_DIR / "focus_longs.txt"
FOCUS_SHORTS_FILE = PERSISTENT_DATA_DIR / "focus_shorts.txt"
FOCUS_PICK_MEMBERSHIP_FILE = RUNTIME_DATA_DIR / "focus_pick_membership.json"
# Which longs.txt/shorts.txt entries the universe auto-populator owns (so
# rotation/cuts never delete a name the trader typed), plus the day's
# VWAP-cut blacklist so a cut name is not re-added the same session.
AUTO_POPULATE_MEMBERSHIP_FILE = RUNTIME_DATA_DIR / "auto_watchlist_membership.json"
# DESK-mode approval queue: auto-populate candidates proposed while the trader
# is at the desk. The Alert Center charts each one; Approve appends it to the
# auto-owned watchlist slice, Pass records the decision for the day. AWAY mode
# never writes this file - its picks apply directly.
AUTO_POPULATE_PENDING_FILE = RUNTIME_DATA_DIR / "auto_populate_pending.json"
# Day-scoped registry of D1 interest flags already raised on Focus picks
# ("SYM|kind" strings, same file format as the ignored-symbols store) so each
# event flags a Focus name at most once per session.
FOCUS_D1_FLAGS_FILE = RUNTIME_DATA_DIR / "focus_d1_flags.json"
# The day's first directional regime read (bearish_*/bullish_*). Discovery
# keeps hunting the opening side after the live label decays to neutral
# (2026-07-17: bearish_strong open -> neutral by noon shut off RW shorts).
AUTO_OPENING_ENV_FILE = RUNTIME_DATA_DIR / "auto_opening_environment.json"
# Append-only JSONL log of the trader's pick verdicts: star likes (with origin
# alert timeframe/surface), X dislikes (with the typed reason), unfavorites.
# Lives in the shared home so it syncs across machines and can be handed to an
# AI for review ("why did I like/hate these picks -> tune the scans").
PICK_FEEDBACK_FILE = PERSISTENT_DATA_DIR / "pick_feedback.jsonl"
# Legacy single-writer JSONL plus the partitioned store used by current
# builds.  New review decisions go to one file per stable machine-local
# installation so two PCs sharing the folder never append to the same file;
# readers merge the directory with the legacy file.
ALERT_REVIEW_EVENTS_FILE = PERSISTENT_DATA_DIR / "alert_review_events.jsonl"
ALERT_REVIEW_EVENTS_DIR = PERSISTENT_DATA_DIR / "alert_review_events"
# Append-only JSONL of the trader's Chart Review decisions: vetoes with a
# reason from the versioned picklist, likes with a claimed setup, hypothetical
# stops, and freeform notes. Same storage class as the two logs above - small,
# human-relevant, shared home so it syncs and can be read by an AI - and the
# desk GUI is its sole writer. Analysis-only evidence: nothing in the running
# system reads this file to mute, score, gate, or alert (plan.md sec 5).
TRADER_ANNOTATIONS_FILE = PERSISTENT_DATA_DIR / "trader_annotations.jsonl"
# Append-only JSONL of the trader's hand-vetted swing picks - "today's best
# swing targets", typed at the end of a session into the strip under the M5
# alert list. One row per action: an add carries (session_date, symbol, side,
# event_at, origin="trader"), a removal appends a RETRACTION row and nothing is
# ever rewritten. Same storage class as the two logs above: small, trader-
# authored, shared home. The live list is derived by replaying a session's rows
# in file order. Evidence only - it is written beside the Focus write-through,
# never in front of it, and nothing in the running system reads this file to
# detect, score, rank, gate or alert (plan.md sec 5).
SWING_FAVORITES_FILE = PERSISTENT_DATA_DIR / "swing_favorites.jsonl"
# Aggregated revealed-preference state derived from the review-events log by
# scripts/review_learning.py: per-segment take rates, taken-vs-passed
# outcomes, blind spots / leaks, watch conversion. Rebuilt when stale.
REVIEW_PREFERENCE_STATE_FILE = PERSISTENT_DATA_DIR / "review_preference_state.json"
REVIEW_LEARNING_REPORT_FILE = OUTPUT_DIR / "review_learning_report.txt"
# AI-authored review policy: segment-level priority deltas, annotations, and
# watch presets decided by an AI (Fable/Sol) reviewing the scoreboard docs
# (see AGENTS.md "Review-learning loop"). The Alert Center reads it to order
# the review queue and annotate charts - advisory only, never suppression.
# The draft is a mechanical starting point the AI curates before promoting.
REVIEW_POLICY_FILE = PERSISTENT_DATA_DIR / "review_policy.json"
REVIEW_POLICY_DRAFT_FILE = PERSISTENT_DATA_DIR / "review_policy_draft.json"
# Alert Center symbols removed from review for the current trading day. This
# suppresses only the visual surface; it never removes a watchlist entry or
# changes a scanner.
ALERT_CENTER_IGNORED_SYMBOLS_FILE = PERSISTENT_DATA_DIR / "alert_center_ignored_symbols.txt"
# Symbols "parked" out of the review CHART queue for the day: the trader armed
# a D1 alert on the chart and then hit Skip - decision made, the armed alert
# does the watching. Feed items still record; Focus names and armed-watch hits
# still occupy the chart.
ALERT_REVIEW_PARKED_SYMBOLS_FILE = PERSISTENT_DATA_DIR / "alert_center_parked_symbols.json"
# Armed visual-chart watches (New HOD/LOD, VWAP/σ-band bounces): trading-day
# scoped so a GUI restart keeps them armed; a new session starts clean.
ALERT_CHART_WATCHES_FILE = PERSISTENT_DATA_DIR / "alert_chart_watches.json"
# Persistent D1 candle-level alerts: armed by clicking a D1 chart candle and
# kept ACROSS sessions until the level flags (the symbol need not be in any
# scan - evaluation waits for whatever price evidence appears).
D1_LEVEL_WATCHES_FILE = PERSISTENT_DATA_DIR / "d1_level_watches.json"
# Persistent D1 EVENT watches (15EMA reject, new 5d/20d extreme, SMA break):
# armed from the dock's D1 row, kept across sessions until they fire. Their
# reference levels are re-derived from the daily store on every poll.
D1_EVENT_WATCHES_FILE = PERSISTENT_DATA_DIR / "d1_event_watches.json"
# Persistent ANY-BOUNCE watches (R5 section 4): one armed request per
# symbol and side covering a SET of levels - D1 bands, current and prior
# AVWAP, daily and session EMAs, the H1 15 EMA. Its own store beside the
# others because it carries a set rather than one kind, and because one
# component must own one file.
ANY_BOUNCE_WATCHES_FILE = PERSISTENT_DATA_DIR / "any_bounce_watches.json"
# Append-only trader annotations for the BounceBot market-environment control.
# The bot's automatic SPY read remains separate; each manual selection records
# that contemporaneous read so later research can learn where/why the trader
# disagreed without treating the annotation as ground truth.
MARKET_ENVIRONMENT_ANNOTATIONS_FILE = PERSISTENT_DATA_DIR / "market_environment_annotations.jsonl"
HUMAN_FOCUS_SNAPSHOT_STATE_FILE = RUNTIME_DATA_DIR / "human_focus_snapshot_state.json"
HUMAN_FOCUS_DAILY_PICKS_FILE = RUNTIME_DATA_DIR / "human_focus_daily_picks.csv"
HUMAN_FOCUS_OUTCOMES_FILE = RUNTIME_DATA_DIR / "human_focus_outcomes.csv"
HUMAN_FOCUS_PERFORMANCE_FILE = RUNTIME_DATA_DIR / "human_focus_performance.csv"
# Forward tracking for vetoed names, in the human-focus column schema but in
# its OWN files. Deliberately not the human-focus picks CSV: that file is keyed
# (trade_date, symbol, side) with no source, so a veto row for a name that is
# also a focus pick that day would collide with - and suppress - the focus row.
# Separate files let the same outcome machinery grade both without either
# cohort touching the other. Written only by ui.annotations.veto_cohort.
VETO_COHORT_PICKS_FILE = RUNTIME_DATA_DIR / "veto_cohort_picks.csv"
VETO_COHORT_OUTCOMES_FILE = RUNTIME_DATA_DIR / "veto_cohort_outcomes.csv"
VETO_COHORT_PERFORMANCE_FILE = RUNTIME_DATA_DIR / "veto_cohort_performance.csv"
# R10.F: the LIKE mirror of the veto trio. Named here beside it so every reader
# addresses these files by CONSTANT - AI-P1 found the Focus Pick Review step had
# been rendering an empty table for six days because it composed a path under
# PERSISTENT_DATA_DIR instead, and that is the spelling that cannot drift from
# where the writers put them.
LIKE_COHORT_PICKS_FILE = RUNTIME_DATA_DIR / "like_cohort_picks.csv"
LIKE_COHORT_OUTCOMES_FILE = RUNTIME_DATA_DIR / "like_cohort_outcomes.csv"
LIKE_COHORT_PERFORMANCE_FILE = RUNTIME_DATA_DIR / "like_cohort_performance.csv"
# P5: the two remaining verdicts, each its own trio. A PASS is "I like this
# name but not this setup" and a REJECTION is "not today" / "I dislike this";
# neither is a veto and neither is a like, and pooling either with those would
# average opposite judgements into one unreadable number.
PASS_COHORT_PICKS_FILE = RUNTIME_DATA_DIR / "pass_cohort_picks.csv"
PASS_COHORT_OUTCOMES_FILE = RUNTIME_DATA_DIR / "pass_cohort_outcomes.csv"
PASS_COHORT_PERFORMANCE_FILE = RUNTIME_DATA_DIR / "pass_cohort_performance.csv"
REJECTION_COHORT_PICKS_FILE = RUNTIME_DATA_DIR / "rejection_cohort_picks.csv"
REJECTION_COHORT_OUTCOMES_FILE = RUNTIME_DATA_DIR / "rejection_cohort_outcomes.csv"
REJECTION_COHORT_PERFORMANCE_FILE = RUNTIME_DATA_DIR / "rejection_cohort_performance.csv"
MASTER_AVWAP_BUCKET_STATE_FILE = RUNTIME_DATA_DIR / "master_avwap_bucket_state.json"

SECTOR_ETF_MAP_FILE = DATA_DIR / "sector_etf_map.json"
INDUSTRY_ETF_MAP_FILE = DATA_DIR / "industry_etf_map.json"
SYMBOL_CLASSIFICATION_CACHE_FILE = DATA_DIR / "symbol_classification.csv"
EARNINGS_ANCHORS_FILE = DATA_DIR / "earnings_avwap_anchors.csv"
EARNINGS_ANCHOR_CANDIDATES_FILE = RUNTIME_DATA_DIR / "earnings_anchor_candidates.csv"
EARNINGS_CALENDAR_HISTORY_FILE = DATA_DIR / "earnings_calendar_history.json"
MASTER_AVWAP_LEVELS_DIR = DATA_DIR / "levels"
# Durable, shared daily-bar history in the home folder. The local machine cache under
# CACHE_DIR is a fast L1; this is the L2 that survives cache wipes / fresh
# machines so cold starts only fetch the delta instead of full price history.
MASTER_AVWAP_DAILY_BARS_DIR = DATA_DIR / "daily_bars"
# Durable, shared intraday (H1) bar history in the home folder. H4 is resampled from
# H1, so only H1 is stored. Same L1/L2 split as the daily store.
MASTER_AVWAP_INTRADAY_BARS_DIR = DATA_DIR / "intraday_bars"

EARNINGS_CACHE_FILE = CACHE_DIR / "earnings_cache.json"
THETA_OPTION_CHAIN_CACHE_FILE = CACHE_DIR / "theta_option_chain_cache.json"
PREV_EARNINGS_CACHE_FILE = CACHE_DIR / "prev_earnings_cache.json"
EARNINGS_DATES_CACHE_FILE = CACHE_DIR / "earnings_dates_cache.json"
EARNINGS_CALENDAR_CACHE_FILE = CACHE_DIR / "earnings_calendar_rows.json"
YAHOO_SYMBOL_META_CACHE_FILE = CACHE_DIR / "yahoo_symbol_metadata.json"
DAILY_BARS_CACHE_DIR = CACHE_DIR / "daily_bars"
INTRADAY_BARS_CACHE_DIR = CACHE_DIR / "intraday_bars"
# Chart-only feather mirror of the shared daily store, owned solely by
# ui.services.bar_cache. Deliberately NOT the scanner's DAILY_BARS_CACHE_DIR
# (CSV, different owner): the chart must be able to warm from local disk
# without a shared mutable export gaining a second writer.
CHART_BAR_CACHE_DIR = CACHE_DIR / "chart_bars"

AVWAP_SIGNALS_FILE = RUNTIME_DATA_DIR / "avwap_signals.csv"
# Compact, current-session projection of ``avwap_signals.csv`` for GUI health.
# The scanner is the sole writer; readers validate the source signature before
# trusting it and fall back to the historical CSV when an older build omitted it.
MASTER_AVWAP_ACTIVE_EVENTS_FILE = RUNTIME_DATA_DIR / "master_avwap_active_events.json"
D1_FEATURES_FILE = RUNTIME_DATA_DIR / "d1_features.csv"
D1_FEATURES_HISTORY_FILE = RUNTIME_DATA_DIR / "d1_features_history.csv"
INTRADAY_BOUNCES_FILE = RUNTIME_DATA_DIR / "intraday_bounces.csv"
INTRADAY_BOUNCE_CANDIDATES_FILE = RUNTIME_DATA_DIR / "intraday_bounce_candidates.csv"
INTRADAY_BOUNCE_OUTCOMES_FILE = RUNTIME_DATA_DIR / "intraday_bounce_outcomes.csv"
INTRADAY_BOUNCE_OUTCOME_STATE_FILE = RUNTIME_DATA_DIR / "intraday_bounce_outcome_state.json"
INTRADAY_BOUNCE_FEEDBACK_FILE = RUNTIME_DATA_DIR / "intraday_bounce_feedback.csv"
MASTER_AVWAP_AI_STATE_FILE = RUNTIME_DATA_DIR / "master_avwap_ai_state.json"
MASTER_AVWAP_HISTORY_FILE = RUNTIME_DATA_DIR / "master_avwap_history.json"
MASTER_POSITIONS_FILE = RUNTIME_DATA_DIR / "master_positions.json"
PREVIOUS_GAP_UPS_FILE = RUNTIME_DATA_DIR / "previous_gap_ups.csv"
ANCHOR_AVWAP_SIGNALS_FILE = RUNTIME_DATA_DIR / "master_anchor_avwap_signals.csv"
MASTER_AVWAP_FOCUS_FILE = RUNTIME_DATA_DIR / "master_avwap_focus.json"
MASTER_AVWAP_D1_WATCHLIST_FILE = RUNTIME_DATA_DIR / "master_avwap_d1_watchlist.json"
MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE = RUNTIME_DATA_DIR / "master_avwap_d1_upgrade_alerts.json"
# Per-symbol D1 band-zone "arms" for every scanned symbol: the M5 bounce/break
# rubric levels the bounce bot watches to fire D1 Focus alerts (decision-support).
MASTER_AVWAP_D1_ZONE_ARMS_FILE = RUNTIME_DATA_DIR / "master_avwap_d1_zone_arms.json"
# BounceBot's intraday SPY-pause defiance observations (day-scoped). The
# hourly master scan reads these back as swing-row evidence, so the file lives
# on the shared store where both processes (and both machines) can see it.
REGIME_PAUSE_OBSERVATIONS_FILE = RUNTIME_DATA_DIR / "regime_pause_observations.json"
MASTER_AVWAP_SETUP_TRACKER_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_tracker.json"
# F3 step 1 (2026-09-04): the SQLite mirror of the tracker, one row per record, written
# AFTER the JSON save and never read by the scanner yet (scripts/tracker_store.py).
MASTER_AVWAP_SETUP_TRACKER_DB = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_tracker.sqlite"
MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_tracker_scoring_snapshot.json"
MASTER_AVWAP_SETUP_SCENARIOS_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_scenarios.csv"
MASTER_AVWAP_SETUP_DAILY_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_daily.csv"
MASTER_AVWAP_SETUP_STATS_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_stats.csv"
MASTER_AVWAP_SETUP_ATTRIBUTES_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_attributes.csv"
MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_attribute_leaderboard.csv"
MASTER_AVWAP_SCAN_FACTOR_OBSERVATIONS_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_scan_factor_observations.csv"
MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_scan_factor_leaderboard.csv"
MASTER_AVWAP_TIER_LIST_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_tier_list.csv"
MASTER_AVWAP_TIER_OUTCOMES_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_tier_outcomes.csv"
MASTER_AVWAP_TIER_PERFORMANCE_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_tier_performance.csv"
MASTER_AVWAP_TIER_CATCH_RATE_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_tier_catch_rate.csv"
MASTER_AVWAP_SCORING_CONFIG_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_scoring_config.json"
MASTER_AVWAP_SCORING_RECOMMENDATIONS_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_scoring_recommendations.json"
MASTER_AVWAP_SCORING_TUNER_REPORT_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_scoring_tuner_report.txt"
MASTER_AVWAP_USER_FAVORITES_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_user_favorites.csv"
JOURNAL_DB_FILE = PERSISTENT_RUNTIME_DATA_DIR / "trade_journal.sqlite3"
#: Weekend Prep stepper progress (R8). Compact operational state, so it
#: belongs in the shared home folder beside the journal rather than in a
#: per-machine cache - a routine half-finished on Saturday is still
#: half-finished on Sunday.
WEEKEND_PREP_STATE_FILE = PERSISTENT_RUNTIME_DATA_DIR / "weekend_prep_state.json"
JOURNAL_FX_CACHE_FILE = PERSISTENT_RUNTIME_DATA_DIR / "trade_journal_fx_rates.json"
JOURNAL_EXPORT_DIR = OUTPUT_DIR / "journal"

MASTER_AVWAP_REPORT_FILE = REPORTS_DIR / "master_avwap_events.txt"
MASTER_AVWAP_EVENT_TICKERS_FILE = REPORTS_DIR / "master_avwap_event_tickers.txt"
MASTER_AVWAP_PRIORITY_SETUPS_FILE = REPORTS_DIR / "master_avwap_priority_setups.txt"
MASTER_AVWAP_UNMAPPED_CLASSIFICATIONS_FILE = REPORTS_DIR / "master_avwap_unmapped_classifications.csv"
MASTER_AVWAP_STDEV_REPORT_FILE = REPORTS_DIR / "master_avwap_stdev2_3.txt"
MASTER_ANCHOR_AVWAP_REPORT_FILE = REPORTS_DIR / "master_anchor_avwap_events.txt"
MASTER_AVWAP_TRADINGVIEW_REPORT_FILE = REPORTS_DIR / "master_avwap_tradingview.txt"
MASTER_AVWAP_D1_UPGRADE_ALERTS_REPORT_FILE = REPORTS_DIR / "master_avwap_d1_upgrade_alerts.txt"
MASTER_AVWAP_MARKET_PREP_FILE = RUNTIME_DATA_DIR / "master_avwap_market_prep.json"
INDUSTRY_BOARD_STATE_FILE = RUNTIME_DATA_DIR / "industry_board_snapshot.json"
INDUSTRY_INTRADAY_RS_STATE_FILE = RUNTIME_DATA_DIR / "industry_intraday_rs_snapshot.json"
MASTER_AVWAP_MARKET_PREP_REPORT_FILE = REPORTS_DIR / "master_avwap_market_prep.txt"
EARNINGS_ANCHOR_CANDIDATES_REPORT_FILE = REPORTS_DIR / "earnings_anchor_candidates.txt"

# Auto Pilot. The report is the away-from-desk digest and lives at the top of
# the home folder; scheduler state + activity log are per-machine, so two
# machines could never fight over them.
AUTOPILOT_REPORT_FILE = SHARED_HOME_DIR / "autopilot_today.txt"
AUTOPILOT_STATE_FILE = LOCAL_MACHINE_CACHE_DIR / "autopilot_state.json"
AUTOPILOT_LOG_FILE = LOCAL_LOG_DIR / "autopilot.log"
# Auto-pick evidence trail: every self-built watchlist symbol (with its gap/RS
# numbers) plus the per-day scorecard joining picks against day-trade outcomes,
# so the gap/RS thresholds get tuned from data.
AUTOPILOT_PICKS_FILE = RUNTIME_DATA_DIR / "autopilot_picks.csv"
AUTOPILOT_SCORECARD_FILE = RUNTIME_DATA_DIR / "autopilot_pick_scorecard.csv"

# Auto Evening mode (sleep-in support). The price-alert watchlist (tickers +
# above/below levels, edited in Research -> Price Alerts) lives on the shared
# store so it syncs like the other watchlists; the trigger history feeds the
# morning briefing's "overnight alerts" section. Briefing state is per-machine
# (only the machine running Evening mode takes the strength checks); the
# rendered briefing sits next to autopilot_today.txt so it is one tap in the
# Drive mobile app.
PRICE_ALERTS_FILE = SHARED_HOME_DIR / "price_alerts.json"
PRICE_ALERT_TRIGGERS_FILE = RUNTIME_DATA_DIR / "price_alert_triggers.csv"
EVENING_BRIEFING_STATE_FILE = LOCAL_MACHINE_CACHE_DIR / "evening_briefing_state.json"
EVENING_BRIEFING_FILE = SHARED_HOME_DIR / "evening_briefing.txt"

BOUNCE_LOG_FILE = LOG_DIR / "bouncers.txt"
# Rotating diagnostic log lives on local disk (see LOCAL_LOG_DIR) so rotation never
# collides with cloud-sync file locks; data-style logs (bouncers, RRS CSVs) stay on
# the shared store.
APP_LOG_FILE = LOCAL_LOG_DIR / "trading_bot.log"
APP_LOG_BACKUP_COUNT = 1
TRADING_BOT_LOG_FILE = APP_LOG_FILE
MASTER_AVWAP_LOG_FILE = APP_LOG_FILE
RRS_STRENGTH_LOG_FILE = LOG_DIR / "rrs_strength_extremes.csv"
RRS_GROUP_STRENGTH_LOG_FILE = LOG_DIR / "rrs_group_strength_extremes.csv"
RRS_ENVIRONMENT_FOCUS_HISTORY_FILE = RUNTIME_DATA_DIR / "rrs_environment_focus_history.json"


def get_tracker_storage_details() -> dict[str, str]:
    source_labels = {
        "environment": "Environment variable",
        "local_config": "Saved local setting",
        "default_local": "Default local storage",
    }
    return {
        "data_dir": str(PERSISTENT_DATA_DIR),
        "shared_root_dir": str(SHARED_HOME_DIR),
        "mutable_data_dir": str(DATA_DIR),
        "logs_dir": str(LOG_DIR),
        "app_log_dir": str(LOCAL_LOG_DIR),
        "output_dir": str(OUTPUT_DIR),
        "runtime_dir": str(PERSISTENT_RUNTIME_DATA_DIR),
        "local_cache_dir": str(LOCAL_MACHINE_CACHE_DIR),
        "source": PERSISTENT_DATA_DIR_SOURCE,
        "source_label": source_labels.get(PERSISTENT_DATA_DIR_SOURCE, PERSISTENT_DATA_DIR_SOURCE),
        "settings_file": str(LOCAL_SETTINGS_FILE),
    }


def get_shared_watchlist_paths() -> tuple[Path, Path]:
    return (LONGS_FILE, SHORTS_FILE)


def get_master_avwap_watchlist_paths() -> tuple[Path, Path, Path, Path]:
    return (LONGS_FILE, SHORTS_FILE, SWING_LONGS_FILE, SWING_SHORTS_FILE)


def get_shared_watchlist_details() -> dict[str, str]:
    longs_path, shorts_path = get_shared_watchlist_paths()
    return {
        "longs_path": str(longs_path),
        "shorts_path": str(shorts_path),
        "longs_exists": "yes" if longs_path.exists() else "no",
        "shorts_exists": "yes" if shorts_path.exists() else "no",
    }


def get_master_avwap_watchlist_details() -> dict[str, str]:
    longs_path, shorts_path, swing_longs_path, swing_shorts_path = get_master_avwap_watchlist_paths()
    return {
        "longs_path": str(longs_path),
        "shorts_path": str(shorts_path),
        "swing_longs_path": str(swing_longs_path),
        "swing_shorts_path": str(swing_shorts_path),
        "longs_exists": "yes" if longs_path.exists() else "no",
        "shorts_exists": "yes" if shorts_path.exists() else "no",
        "swing_longs_exists": "yes" if swing_longs_path.exists() else "no",
        "swing_shorts_exists": "yes" if swing_shorts_path.exists() else "no",
    }


def save_tracker_storage_dir(path: str) -> Path:
    target = Path(path).expanduser()
    LOCAL_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = _load_local_settings()
    payload["shared_data_dir"] = str(target)
    LOCAL_SETTINGS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    invalidate_local_settings_cache()
    return target


def clear_tracker_storage_dir() -> None:
    if not LOCAL_SETTINGS_FILE.exists():
        return
    payload = _load_local_settings()
    payload.pop("shared_data_dir", None)
    if payload:
        LOCAL_SETTINGS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        invalidate_local_settings_cache()
        return
    LOCAL_SETTINGS_FILE.unlink(missing_ok=True)
    invalidate_local_settings_cache()


def get_local_setting(key: str, default=None):
    payload = _load_local_settings()
    return payload.get(key, default)


def save_local_setting(key: str, value) -> None:
    save_local_settings({key: value})


def save_local_settings(values: dict) -> None:
    """Write several settings in ONE read-modify-write.

    Four separate `save_local_setting` calls are four full read-modify-write
    cycles over one JSON file, and each gap is a window in which another process
    can read the file, edit its own key and write back - dropping whatever was
    saved in between. The Questrade rotation saves four related keys, one of
    which is a single-use token: losing it costs the trader the whole credential
    chain (2026-08-25).

    The write goes through a temp file and `os.replace`, so a crash or a full
    disk cannot leave a half-written settings file behind. This file holds every
    machine-local secret; truncating it is not an acceptable failure mode.
    """
    if not values:
        return
    LOCAL_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = _load_local_settings()
    payload.update(values)
    tmp = LOCAL_SETTINGS_FILE.with_name(LOCAL_SETTINGS_FILE.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, LOCAL_SETTINGS_FILE)
    invalidate_local_settings_cache()


def open_path_in_file_manager(path: Path) -> None:
    target = Path(path).expanduser()
    if sys.platform == "win32":
        os.startfile(str(target))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


def _ensure_directories() -> None:
    for path in (
        CACHE_DIR,
        RUNTIME_DATA_DIR,
        REPORTS_DIR,
        PERSISTENT_DATA_DIR,
        PERSISTENT_RUNTIME_DATA_DIR,
        DATA_DIR,
        OUTPUT_DIR,
        LOG_DIR,
        LOCAL_LOG_DIR,
        JOURNAL_EXPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _migrate_legacy_file(legacy_path: Path, new_path: Path) -> None:
    if not legacy_path.exists() or new_path.exists():
        return
    try:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_path), str(new_path))
    except OSError:
        # Another process or AV scan can briefly lock files; don't block app startup on legacy migration.
        return


def _append_legacy_text_file(source_path: Path, destination_path: Path) -> None:
    if not source_path.exists() or source_path.is_dir():
        return
    try:
        if source_path.resolve() == destination_path.resolve():
            return
    except Exception:
        pass

    try:
        content = source_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        content = ""

    if content.strip():
        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            had_content = destination_path.exists() and destination_path.stat().st_size > 0
            with destination_path.open("a", encoding="utf-8") as destination_file:
                if had_content:
                    destination_file.write("\n")
                destination_file.write(
                    f"=== migrated from {source_path} at {datetime.now().isoformat(timespec='seconds')} ===\n"
                )
                destination_file.write(content.rstrip())
                destination_file.write("\n")
        except OSError:
            return

    try:
        source_path.unlink(missing_ok=True)
    except OSError:
        return


def _consolidate_log_variants(destination_path: Path, base_name: str, search_dir: Path, keep_backups: int) -> None:
    direct_path = search_dir / base_name
    _append_legacy_text_file(direct_path, destination_path)

    for rotated_path in sorted(search_dir.glob(f"{base_name}.*")):
        suffix = rotated_path.name[len(base_name) + 1 :]
        if not suffix.isdigit():
            continue
        if search_dir == destination_path.parent and base_name == destination_path.name:
            if int(suffix) <= max(0, int(keep_backups)):
                continue
        _append_legacy_text_file(rotated_path, destination_path)


def _consolidate_legacy_logs() -> None:
    legacy_log_dir = LOCAL_SETTINGS_DIR / "logs"

    for source_path in (
        ROOT_DIR / "trading_bot.log",
        ROOT_DIR / "scripts" / "trading_bot.log",
        ROOT_DIR / "bouncers.txt",
    ):
        destination = BOUNCE_LOG_FILE if source_path.name == "bouncers.txt" else APP_LOG_FILE
        _append_legacy_text_file(source_path, destination)

    for search_dir in (REPO_LOG_DIR, legacy_log_dir):
        # Skip both the shared logs dir and the now-active local log dir so their
        # live rotation backups are not folded back into the active log.
        if search_dir in (LOG_DIR, LOCAL_LOG_DIR):
            continue
        _consolidate_log_variants(APP_LOG_FILE, "trading_bot.log", search_dir, keep_backups=0)
        _consolidate_log_variants(APP_LOG_FILE, "master_avwap.log", search_dir, keep_backups=0)
        _consolidate_log_variants(BOUNCE_LOG_FILE, "bouncers.txt", search_dir, keep_backups=0)

    _consolidate_log_variants(APP_LOG_FILE, "master_avwap.log", LOG_DIR, keep_backups=0)
    _consolidate_log_variants(APP_LOG_FILE, APP_LOG_FILE.name, LOG_DIR, keep_backups=APP_LOG_BACKUP_COUNT)


# --- orphaned atomic-write staging files ----------------------------------
# Every atomic writer here stages through tempfile in the target's own
# directory and unlinks the staging file in a ``finally``. That cleanup cannot
# run when the process is killed mid-write, and it used to fail outright on
# Windows when a cloud-sync client held the staging file open (WinError 32) —
# which is how ~2.3GB of ``intraday_bounce_candidates<8>.csv`` accumulated in
# the runtime dir alongside 19MB of ``.earnings_calendar_history.json.<8>.tmp``.
# The writers now swallow that unlink failure instead of masking the real
# error, and this sweep reclaims whatever still leaks, so the growth is
# self-healing regardless of cause.
#
# OWNERSHIP RULE (plan.md sec 5: one component owns each mutable export).
# The sweep deletes only files it can name an owner for. SHARED_HOME_DIR is a
# cloud-synced folder that other programs also write into, so "any old file of
# roughly this shape" is not a licence to delete — every candidate has to be the
# staging file of a canonical target this module itself defines. An earlier
# draft that swept every ``.<anything>.<8>.tmp`` in that directory was deleting
# files this project could not prove it owned.

# tempfile's random component: exactly 8 chars from ascii_letters + digits + "_".
_TEMP_TOKEN = "[A-Za-z0-9_]{8}"
# A staging file younger than this may belong to a write that is still running,
# so it is never touched. Compaction of a 300MB CSV takes seconds.
STALE_TEMP_MIN_AGE_SECONDS = 6 * 3600


def _owned_staging_targets(directories: Iterable[Path]) -> list[Path]:
    """Every *file* constant this module defines directly inside ``directories``.

    Deriving the list from the module's own globals means a new path constant is
    covered the day it is added, and a file this module cannot name is never a
    sweep candidate at all.

    Suffixless constants are excluded because they are directories, not staging
    targets: ``DATA_DIR``, ``LOG_DIR``, ``ALERT_REVIEW_EVENTS_DIR`` and the bar
    stores all sit directly inside the swept directories, and no writer ever
    stages ``.daily_bars.<token>.tmp``. Including them would hand the sweep a
    pattern matching a temp file this project could not have written — the very
    thing the ownership rule above exists to prevent.
    """
    parents = {Path(directory) for directory in directories}
    owned = {
        value
        for name, value in globals().items()
        if name.isupper()
        and isinstance(value, Path)
        and value.suffix
        and value.parent in parents
    }
    return sorted(owned)


def sweep_stale_atomic_write_temps(
    dotted_for: Iterable[Path] = (),
    *,
    staged_for: Iterable[Path] = (),
    min_age_seconds: float = STALE_TEMP_MIN_AGE_SECONDS,
    now: float | None = None,
) -> list[Path]:
    """Delete orphaned staging files of OWNED targets; return what was removed.

    Both arguments name *canonical targets*, never directories, and both match
    by that target's exact name — nothing is deleted on shape alone.

    ``dotted_for`` covers writers that stage as ``.<target.name>.<token>.tmp``
    beside the target (tempfile with ``prefix=f".{target.name}."``,
    ``suffix=".tmp"``: ``review_learning``, ``review_policy``,
    ``earnings_history``, the industry writers). ``staged_for`` covers those
    that stage as ``<stem><token><suffix>`` (the bounce candidate CSV
    compaction). In both shapes the 8-char token is mandatory, so a real file
    can never match: ``foo.csv`` is not ``foo<8 chars>.csv``, and
    ``.foo.csv.tmp`` is not ``.foo.csv.<8 chars>.tmp``. A temp file whose owner
    this module cannot name is never touched, regardless of age. Only the owned
    targets' own directories are scanned, non-recursively.

    Never raises: housekeeping must not break startup.
    """
    cutoff = (time.time() if now is None else now) - max(0.0, float(min_age_seconds))
    removed: list[Path] = []

    targeted: dict[Path, list[re.Pattern[str]]] = {}
    for target in dotted_for:
        target = Path(target)
        pattern = re.compile(rf"^\.{re.escape(target.name)}\.{_TEMP_TOKEN}\.tmp$")
        targeted.setdefault(target.parent, []).append(pattern)
    for target in staged_for:
        target = Path(target)
        pattern = re.compile(rf"^{re.escape(target.stem)}{_TEMP_TOKEN}{re.escape(target.suffix)}$")
        targeted.setdefault(target.parent, []).append(pattern)

    for directory, patterns in targeted.items():
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not any(pattern.match(entry.name) for pattern in patterns):
                continue
            try:
                if not entry.is_file() or entry.stat().st_mtime > cutoff:
                    continue
                entry.unlink()
            except OSError:
                # Still locked, or already gone. Next startup tries again.
                continue
            removed.append(entry)

    if removed:
        logging.info(
            "Removed %d orphaned atomic-write staging file(s): %s",
            len(removed),
            ", ".join(sorted(path.name for path in removed)[:5]),
        )
    return removed


def migrate_legacy_layout() -> None:
    _ensure_directories()

    legacy_moves = [
        (ROOT_DIR / "longs.txt", LONGS_FILE),
        (ROOT_DIR / "shorts.txt", SHORTS_FILE),
        (REPO_DATA_DIR / "sector_etf_map.json", SECTOR_ETF_MAP_FILE),
        (REPO_DATA_DIR / "industry_etf_map.json", INDUSTRY_ETF_MAP_FILE),
        (REPO_DATA_DIR / "symbol_classification.csv", SYMBOL_CLASSIFICATION_CACHE_FILE),
        (REPO_DATA_DIR / "earnings_avwap_anchors.csv", EARNINGS_ANCHORS_FILE),
        (REPO_DATA_DIR / "levels", MASTER_AVWAP_LEVELS_DIR),
        (REPO_DATA_DIR / "earnings_cache.json", EARNINGS_CACHE_FILE),
        (REPO_DATA_DIR / "prev_earnings_cache.json", PREV_EARNINGS_CACHE_FILE),
        (REPO_DATA_DIR / "earnings_dates_cache.json", EARNINGS_DATES_CACHE_FILE),
        (REPO_DATA_DIR / "earnings_calendar_rows.json", EARNINGS_CALENDAR_CACHE_FILE),
        (REPO_DATA_DIR / "yahoo_symbol_metadata.json", YAHOO_SYMBOL_META_CACHE_FILE),
        (REPO_DATA_DIR / "avwap_signals.csv", AVWAP_SIGNALS_FILE),
        (REPO_DATA_DIR / "d1_features.csv", D1_FEATURES_FILE),
        (REPO_DATA_DIR / "d1_features_history.csv", D1_FEATURES_HISTORY_FILE),
        (REPO_DATA_DIR / "intraday_bounces.csv", INTRADAY_BOUNCES_FILE),
        (REPO_DATA_DIR / "intraday_bounce_candidates.csv", INTRADAY_BOUNCE_CANDIDATES_FILE),
        (REPO_DATA_DIR / "intraday_bounce_outcomes.csv", INTRADAY_BOUNCE_OUTCOMES_FILE),
        (REPO_DATA_DIR / "intraday_bounce_outcome_state.json", INTRADAY_BOUNCE_OUTCOME_STATE_FILE),
        (REPO_DATA_DIR / "master_avwap_ai_state.json", MASTER_AVWAP_AI_STATE_FILE),
        (REPO_DATA_DIR / "master_avwap_history.json", MASTER_AVWAP_HISTORY_FILE),
        (REPO_DATA_DIR / "runtime" / "earnings_anchor_candidates.csv", EARNINGS_ANCHOR_CANDIDATES_FILE),
        (REPO_DATA_DIR / "runtime" / "avwap_signals.csv", AVWAP_SIGNALS_FILE),
        (REPO_DATA_DIR / "runtime" / "d1_features.csv", D1_FEATURES_FILE),
        (REPO_DATA_DIR / "runtime" / "d1_features_history.csv", D1_FEATURES_HISTORY_FILE),
        (REPO_DATA_DIR / "runtime" / "intraday_bounces.csv", INTRADAY_BOUNCES_FILE),
        (REPO_DATA_DIR / "runtime" / "intraday_bounce_candidates.csv", INTRADAY_BOUNCE_CANDIDATES_FILE),
        (REPO_DATA_DIR / "runtime" / "intraday_bounce_outcomes.csv", INTRADAY_BOUNCE_OUTCOMES_FILE),
        (REPO_DATA_DIR / "runtime" / "intraday_bounce_outcome_state.json", INTRADAY_BOUNCE_OUTCOME_STATE_FILE),
        (REPO_DATA_DIR / "runtime" / "rrs_environment_focus_history.json", RRS_ENVIRONMENT_FOCUS_HISTORY_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_ai_state.json", MASTER_AVWAP_AI_STATE_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_history.json", MASTER_AVWAP_HISTORY_FILE),
        (REPO_DATA_DIR / "runtime" / "master_positions.json", MASTER_POSITIONS_FILE),
        (REPO_DATA_DIR / "runtime" / "previous_gap_ups.csv", PREVIOUS_GAP_UPS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_anchor_avwap_signals.csv", ANCHOR_AVWAP_SIGNALS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_focus.json", MASTER_AVWAP_FOCUS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_d1_watchlist.json", MASTER_AVWAP_D1_WATCHLIST_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_d1_upgrade_alerts.json", MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_d1_zone_arms.json", MASTER_AVWAP_D1_ZONE_ARMS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_market_prep.json", MASTER_AVWAP_MARKET_PREP_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_tracker.json", MASTER_AVWAP_SETUP_TRACKER_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_scenarios.csv", MASTER_AVWAP_SETUP_SCENARIOS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_daily.csv", MASTER_AVWAP_SETUP_DAILY_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_stats.csv", MASTER_AVWAP_SETUP_STATS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_attributes.csv", MASTER_AVWAP_SETUP_ATTRIBUTES_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_attribute_leaderboard.csv", MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_scan_factor_observations.csv", MASTER_AVWAP_SCAN_FACTOR_OBSERVATIONS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_scan_factor_leaderboard.csv", MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_tier_list.csv", MASTER_AVWAP_TIER_LIST_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_tier_outcomes.csv", MASTER_AVWAP_TIER_OUTCOMES_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_tier_performance.csv", MASTER_AVWAP_TIER_PERFORMANCE_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_tier_catch_rate.csv", MASTER_AVWAP_TIER_CATCH_RATE_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_scoring_config.json", MASTER_AVWAP_SCORING_CONFIG_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_scoring_recommendations.json", MASTER_AVWAP_SCORING_RECOMMENDATIONS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_scoring_tuner_report.txt", MASTER_AVWAP_SCORING_TUNER_REPORT_FILE),
        (REPO_DATA_DIR / "cache" / "earnings_cache.json", EARNINGS_CACHE_FILE),
        (REPO_DATA_DIR / "cache" / "prev_earnings_cache.json", PREV_EARNINGS_CACHE_FILE),
        (REPO_DATA_DIR / "cache" / "earnings_dates_cache.json", EARNINGS_DATES_CACHE_FILE),
        (REPO_DATA_DIR / "cache" / "earnings_calendar_rows.json", EARNINGS_CALENDAR_CACHE_FILE),
        (REPO_DATA_DIR / "cache" / "yahoo_symbol_metadata.json", YAHOO_SYMBOL_META_CACHE_FILE),
        (REPO_DATA_DIR / "cache" / "daily_bars", DAILY_BARS_CACHE_DIR),
        (REPO_OUTPUT_DIR / "master_positions.json", MASTER_POSITIONS_FILE),
        (REPO_OUTPUT_DIR / "previous_gap_ups.csv", PREVIOUS_GAP_UPS_FILE),
        (REPO_OUTPUT_DIR / "master_anchor_avwap_signals.csv", ANCHOR_AVWAP_SIGNALS_FILE),
        (REPO_OUTPUT_DIR / "master_avwap_events.txt", MASTER_AVWAP_REPORT_FILE),
        (REPO_OUTPUT_DIR / "master_avwap_event_tickers.txt", MASTER_AVWAP_EVENT_TICKERS_FILE),
        (REPO_OUTPUT_DIR / "master_avwap_priority_setups.txt", MASTER_AVWAP_PRIORITY_SETUPS_FILE),
        (REPO_OUTPUT_DIR / "master_avwap_stdev2_3.txt", MASTER_AVWAP_STDEV_REPORT_FILE),
        (REPO_OUTPUT_DIR / "master_anchor_avwap_events.txt", MASTER_ANCHOR_AVWAP_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_events.txt", MASTER_AVWAP_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_event_tickers.txt", MASTER_AVWAP_EVENT_TICKERS_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_priority_setups.txt", MASTER_AVWAP_PRIORITY_SETUPS_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_stdev2_3.txt", MASTER_AVWAP_STDEV_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_anchor_avwap_events.txt", MASTER_ANCHOR_AVWAP_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_tradingview.txt", MASTER_AVWAP_TRADINGVIEW_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_d1_upgrade_alerts.txt", MASTER_AVWAP_D1_UPGRADE_ALERTS_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_market_prep.txt", MASTER_AVWAP_MARKET_PREP_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "earnings_anchor_candidates.txt", EARNINGS_ANCHOR_CANDIDATES_REPORT_FILE),
        (REPO_LOG_DIR / "bouncers.txt", BOUNCE_LOG_FILE),
        (REPO_LOG_DIR / "trading_bot.log", TRADING_BOT_LOG_FILE),
        (REPO_LOG_DIR / "master_avwap.log", MASTER_AVWAP_LOG_FILE),
        (REPO_LOG_DIR / "rrs_strength_extremes.csv", RRS_STRENGTH_LOG_FILE),
        (REPO_LOG_DIR / "rrs_group_strength_extremes.csv", RRS_GROUP_STRENGTH_LOG_FILE),
        (DATA_DIR / "cache" / "earnings_cache.json", EARNINGS_CACHE_FILE),
        (DATA_DIR / "cache" / "prev_earnings_cache.json", PREV_EARNINGS_CACHE_FILE),
        (DATA_DIR / "cache" / "earnings_dates_cache.json", EARNINGS_DATES_CACHE_FILE),
        (DATA_DIR / "cache" / "earnings_calendar_rows.json", EARNINGS_CALENDAR_CACHE_FILE),
        (DATA_DIR / "cache" / "yahoo_symbol_metadata.json", YAHOO_SYMBOL_META_CACHE_FILE),
        (DATA_DIR / "cache" / "daily_bars", DAILY_BARS_CACHE_DIR),
        (LOCAL_SETTINGS_DIR / "data" / "sector_etf_map.json", SECTOR_ETF_MAP_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "industry_etf_map.json", INDUSTRY_ETF_MAP_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "symbol_classification.csv", SYMBOL_CLASSIFICATION_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "earnings_avwap_anchors.csv", EARNINGS_ANCHORS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "levels", MASTER_AVWAP_LEVELS_DIR),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "earnings_cache.json", EARNINGS_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "prev_earnings_cache.json", PREV_EARNINGS_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "earnings_dates_cache.json", EARNINGS_DATES_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "earnings_calendar_rows.json", EARNINGS_CALENDAR_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "yahoo_symbol_metadata.json", YAHOO_SYMBOL_META_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "daily_bars", DAILY_BARS_CACHE_DIR),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "earnings_anchor_candidates.csv", EARNINGS_ANCHOR_CANDIDATES_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "avwap_signals.csv", AVWAP_SIGNALS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "d1_features.csv", D1_FEATURES_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "d1_features_history.csv", D1_FEATURES_HISTORY_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "intraday_bounces.csv", INTRADAY_BOUNCES_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "intraday_bounce_candidates.csv", INTRADAY_BOUNCE_CANDIDATES_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "intraday_bounce_outcomes.csv", INTRADAY_BOUNCE_OUTCOMES_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "intraday_bounce_outcome_state.json", INTRADAY_BOUNCE_OUTCOME_STATE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "rrs_environment_focus_history.json", RRS_ENVIRONMENT_FOCUS_HISTORY_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_avwap_ai_state.json", MASTER_AVWAP_AI_STATE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_avwap_history.json", MASTER_AVWAP_HISTORY_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_positions.json", MASTER_POSITIONS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "previous_gap_ups.csv", PREVIOUS_GAP_UPS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_anchor_avwap_signals.csv", ANCHOR_AVWAP_SIGNALS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_avwap_focus.json", MASTER_AVWAP_FOCUS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_avwap_d1_watchlist.json", MASTER_AVWAP_D1_WATCHLIST_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_avwap_d1_upgrade_alerts.json", MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_avwap_d1_zone_arms.json", MASTER_AVWAP_D1_ZONE_ARMS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_avwap_market_prep.json", MASTER_AVWAP_MARKET_PREP_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_tracker.json", MASTER_AVWAP_SETUP_TRACKER_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_scenarios.csv", MASTER_AVWAP_SETUP_SCENARIOS_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_daily.csv", MASTER_AVWAP_SETUP_DAILY_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_stats.csv", MASTER_AVWAP_SETUP_STATS_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_attributes.csv", MASTER_AVWAP_SETUP_ATTRIBUTES_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_attribute_leaderboard.csv", MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_scan_factor_observations.csv", MASTER_AVWAP_SCAN_FACTOR_OBSERVATIONS_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_scan_factor_leaderboard.csv", MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_tier_list.csv", MASTER_AVWAP_TIER_LIST_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_tier_outcomes.csv", MASTER_AVWAP_TIER_OUTCOMES_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_tier_performance.csv", MASTER_AVWAP_TIER_PERFORMANCE_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_tier_catch_rate.csv", MASTER_AVWAP_TIER_CATCH_RATE_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_scoring_config.json", MASTER_AVWAP_SCORING_CONFIG_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_scoring_recommendations.json", MASTER_AVWAP_SCORING_RECOMMENDATIONS_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_scoring_tuner_report.txt", MASTER_AVWAP_SCORING_TUNER_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_events.txt", MASTER_AVWAP_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_event_tickers.txt", MASTER_AVWAP_EVENT_TICKERS_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_priority_setups.txt", MASTER_AVWAP_PRIORITY_SETUPS_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_stdev2_3.txt", MASTER_AVWAP_STDEV_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_anchor_avwap_events.txt", MASTER_ANCHOR_AVWAP_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_tradingview.txt", MASTER_AVWAP_TRADINGVIEW_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_d1_upgrade_alerts.txt", MASTER_AVWAP_D1_UPGRADE_ALERTS_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_market_prep.txt", MASTER_AVWAP_MARKET_PREP_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "earnings_anchor_candidates.txt", EARNINGS_ANCHOR_CANDIDATES_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "logs" / "bouncers.txt", BOUNCE_LOG_FILE),
        (LOCAL_SETTINGS_DIR / "logs" / "trading_bot.log", TRADING_BOT_LOG_FILE),
        (LOCAL_SETTINGS_DIR / "logs" / "master_avwap.log", MASTER_AVWAP_LOG_FILE),
        (LOCAL_SETTINGS_DIR / "logs" / "rrs_strength_extremes.csv", RRS_STRENGTH_LOG_FILE),
        (LOCAL_SETTINGS_DIR / "logs" / "rrs_group_strength_extremes.csv", RRS_GROUP_STRENGTH_LOG_FILE),
    ]
    for legacy_path, new_path in legacy_moves:
        _migrate_legacy_file(legacy_path, new_path)
    _consolidate_legacy_logs()
    try:
        sweep_stale_atomic_write_temps(
            # Only this module's own file constants sitting directly in the
            # three shallow directories that carry staging files: the shared
            # home (watchlists, review artifacts), DATA_DIR (earnings history)
            # and RUNTIME_DATA_DIR (the big candidate-CSV compaction). A new
            # file constant is covered the day it is added, and nothing this
            # module cannot name is a candidate — the shared home is a
            # cloud-synced folder other programs write into too. Only those
            # files' own directories are scanned; the bar stores under
            # DATA_DIR are never descended into.
            dotted_for=_owned_staging_targets((SHARED_HOME_DIR, DATA_DIR, RUNTIME_DATA_DIR)),
            staged_for=(INTRADAY_BOUNCE_CANDIDATES_FILE,),
        )
    except Exception:  # housekeeping must never break startup
        logging.debug("Stale atomic-write temp sweep failed.", exc_info=True)


migrate_legacy_layout()
