"""Greatness Monitor shadow runner (plan.md sections 7.3 and 9, shadow-first).

Runs the confirmation engine beside the existing one-wick D1 trigger alerts:
each time the bot evaluates a D1 watchlist symbol, the same intraday bars
feed a persistent DevelopmentCandidate. Stages/attempts survive restarts via
an atomic JSON store; every typed transition appends to a JSONL log. Live
alerts are untouched - this accumulates the evidence for replacing the
single-cross rule with staged confirmation.

What an auditor needs from these rows (plan.md sec 4 "shadow logs must be
auditable, not merely present", sec 7.3):

* **candidate identity** - ``symbol|side|setup_family|session|plan_version|
  plan_revision|plan_hash``.  plan.md:549 says outright that the old
  ``symbol|session_date`` key "is insufficient if a side or plan changes during
  the session".  The lineage prefix (everything before the plan version) is the
  stable thread through a session; the revision orders plan changes inside it.
* **plan revisions** - when the D1 scan re-arms different levels mid-session the
  board emits a ``PLAN_REVISED`` row carrying the old *and* new plan version,
  revision and hash, the level diff, and exactly what lifecycle progress was
  carried across.  Progress migration is the plan.md sec 5 invariant "rescans
  and GUI refreshes must not erase valid lifecycle progress".
* **data health** - every row carries the freshness / gap / provider state of
  the bars that drove it, so a silent feed outage can never be read back as
  "the setup simply never confirmed" (plan.md sec 5: missing data is
  uncertainty, never silent confirmation).
* **run and machine identity** - the run id comes from the existing run-manifest
  mechanism (``diagnostics.run_manifest``) rather than a private scheme, so a
  shadow row joins to the manifest of the scan that produced it.

"No event" vs "not evaluated" (plan.md:303-305) is answered by *per-candidate
evaluation counters* held in the status store and flushed into one daily
crash-safe per-session summary artifact, not by a heartbeat row per candidate
per bar: the live log is already ~14.5 MB and a per-bar heartbeat would bury
the transitions it exists to record.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import socket
import threading
from datetime import datetime, timedelta
from pathlib import Path

from diagnostics.artifact_io import append_jsonl_rows, atomic_write_json
from diagnostics.shadow_session_rollup import (
    GREATNESS_ENGINE,
    RETENTION_POLICY as SESSION_RETENTION_POLICY,
    finalize_session,
)
from greatness_monitor import (
    DevelopmentCandidate,
    ENGINE_VERSION,
    GreatnessEngine,
    candidate_from_d1_trigger_levels,
)
from market_state import M5Bar
from market_session import get_market_local_timezone, normalize_market_local_datetime

_BAR_MINUTES = 5
#: v4 adds candidate lineage + plan revision identity, the ``data_health``
#: block, run/machine identity, and the daily ``SESSION_SUMMARY`` row.  v3's
#: fields are all still written.
SHADOW_SCHEMA = "greatness_shadow_v4"
#: v2, v3 and v4 rows are readable side by side, so an existing evidence file
#: keeps accumulating instead of being rotated away mid-programme (plan.md sec
#: 6.1 / sec 7: shadow evidence must not be broken up or deleted).  Only
#: genuinely pre-v2 rows trigger an archive.
COMPATIBLE_SHADOW_SCHEMAS = frozenset(
    {"greatness_shadow_v2", "greatness_shadow_v3", SHADOW_SCHEMA}
)
STORE_SCHEMA = "greatness_store_v4"
#: Sub-schemas of the blocks embedded in a row, versioned independently so a
#: reader can tell "this field is absent" from "this field is older".
DATA_HEALTH_SCHEMA = "greatness_data_health_v1"
PLAN_REVISION_SCHEMA = "greatness_plan_revision_v1"
RUN_IDENTITY_SCHEMA = "greatness_run_identity_v1"

#: Board-emitted row kinds (the engine's own kinds live in ``EventType``).
EVENT_PLAN_REVISED = "PLAN_REVISED"
EVENT_SESSION_SUMMARY = "SESSION_SUMMARY"

# --- data-health thresholds -------------------------------------------------
#: The newest completed bar may be at most this many bar intervals old before an
#: evaluation counts as stale.  Two intervals tolerates one late-arriving bar.
_STALE_AFTER_BARS = 2.0
#: Two consecutive completed bars further apart than this are a feed gap.
_GAP_TOLERANCE_MINUTES = _BAR_MINUTES
#: Clock skew beyond this (bar stamped ahead of the evaluation) is reported.
_CLOCK_SKEW_TOLERANCE_SECONDS = 90.0

HEALTH_OK = "OK"
HEALTH_STALE = "STALE"
HEALTH_GAPPED = "GAPPED"
HEALTH_NO_COMPLETE_BAR = "NO_COMPLETE_BAR"
HEALTH_NO_DATA = "NO_DATA"
#: Worst-first severity; the reported ``status`` is the worst reason present.
_HEALTH_SEVERITY = (
    HEALTH_NO_DATA,
    HEALTH_NO_COMPLETE_BAR,
    HEALTH_GAPPED,
    HEALTH_STALE,
    HEALTH_OK,
)

#: Declared, and actually enforced, retention (plan.md sec 4).
RETENTION_POLICY = {
    "events": "current session/config scope; atomically rotated before reset",
    "session_evidence": dict(SESSION_RETENTION_POLICY),
    "store": "current session only; prior counters are atomically summarized before reset",
    "pre_v2_events": "archived beside the log on first write, never deleted",
}

_lock = threading.Lock()
_board: "GreatnessBoard | None" = None


def _diag_dir() -> Path:
    try:
        from project_paths import get_diagnostics_dir

        return get_diagnostics_dir()
    except Exception:
        return Path.home() / ".tradingbotv3"


def _json_hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_hash(candidate: DevelopmentCandidate) -> str:
    """Hash of the *decision* content of a plan.

    Deliberately excludes provenance: candidate ids are built from this hash and
    must stay stable across restarts and schema versions, so descriptive
    metadata may never move them.
    """
    plan = candidate.plan
    return _json_hash(
        {
            "side_sign": plan.side_sign,
            "invalidation": plan.invalidation,
            "obstacle": plan.obstacle,
            "target": plan.target,
            "version": plan.version,
            "rearm": {
                "max_attempts": plan.rearm.max_attempts,
                "min_reset_bars": plan.rearm.min_reset_bars,
            },
            "steps": [
                {
                    "label": step.label,
                    "level": step.level,
                    "condition": step.condition.value,
                    "required_bars": step.required_bars,
                    "mandatory": step.mandatory,
                }
                for step in plan.steps
            ],
        }
    )


def _config_hash(engine: GreatnessEngine) -> str:
    config = engine.config
    return _json_hash(
        {
            "version": config.version,
            "near_trigger_pct": config.near_trigger_pct,
            "touch_tolerance_pct": config.touch_tolerance_pct,
        }
    )


def _candidate_id(candidate: DevelopmentCandidate) -> str:
    family = str(candidate.setup_family or "general").replace("|", "/")
    return "|".join(
        (
            candidate.symbol.upper(),
            candidate.side.upper(),
            family,
            candidate.session_date,
            candidate.plan.version,
            _plan_hash(candidate)[:12],
        )
    )


class GreatnessBoard:
    """Manages persistent candidates and feeds them completed bars once."""

    def __init__(self, store_path: Path | None = None, events_path: Path | None = None) -> None:
        self.store_path = store_path or (_diag_dir() / "greatness_candidates.json")
        self.events_path = events_path or (_diag_dir() / "greatness_shadow.jsonl")
        self.engine = GreatnessEngine()
        self.machine = socket.gethostname()
        self.local_timezone, self.timezone_name = get_market_local_timezone()
        self.config_hash = _config_hash(self.engine)
        self.candidates: dict[str, DevelopmentCandidate] = {}
        self.last_bar_ts: dict[str, str] = {}
        self.coverage = self._empty_coverage("")
        self._load()

    # ------------------------------------------------------------------
    def update(
        self,
        symbol: str,
        side: str,
        trigger_levels: list[dict],
        bars: list[M5Bar],
        *,
        session_date: str,
        evaluated_at: datetime | None = None,
    ) -> list:
        evaluated_at = normalize_market_local_datetime(
            evaluated_at, local_timezone=self.local_timezone
        )
        self._activate_session(session_date, evaluated_at=evaluated_at)
        proposed = candidate_from_d1_trigger_levels(
            symbol, side, trigger_levels, session_date=session_date
        )
        if proposed is None:
            self._bump("candidates_skipped_no_plan")
            self._save()
            return []
        key = _candidate_id(proposed)
        candidate = self.candidates.get(key)
        if candidate is None:
            revisions = [
                (candidate_id, existing)
                for candidate_id, existing in self.candidates.items()
                if existing.symbol.upper() == proposed.symbol.upper()
                and existing.session_date == session_date
            ]
            if revisions:
                self._append_rows(
                    [
                        self._audit_row(
                            event="PLAN_REVISED",
                            candidate=proposed,
                            candidate_id=key,
                            evaluated_at=evaluated_at,
                            bar=None,
                            extra={"previous_candidate_ids": [item[0] for item in revisions]},
                        )
                    ]
                )
                self._bump("plan_revisions")
            candidate = proposed
            self.candidates[key] = candidate
            self._bump("candidates_created")
            provenance = candidate.provenance or {}
            if provenance.get("primary_trigger_id"):
                self._bump("candidates_with_trigger_provenance")
            else:
                self._bump("candidates_without_trigger_provenance")
            missing = int(provenance.get("steps_missing_trigger_id") or 0)
            if missing:
                self._bump("steps_missing_trigger_id", missing)
        self._bump("evaluations")
        self.coverage["last_evaluation_at"] = evaluated_at.isoformat(timespec="seconds")
        self.coverage["last_candidate_id"] = key
        last_seen = self.last_bar_ts.get(key, "")
        events = []
        for bar in bars:
            stamp = bar.ts.isoformat(timespec="seconds")
            self._bump("bars_seen")
            if not bar.complete:
                self._bump("bars_skipped_incomplete")
                continue
            if stamp <= last_seen:
                self._bump("bars_skipped_duplicate")
                continue
            bar_events = self.engine.on_bar(candidate, bar)
            self._bump("bars_consumed")
            if bar_events:
                self._append_events(
                    bar_events,
                    candidate=candidate,
                    candidate_id=key,
                    bar=bar,
                    evaluated_at=evaluated_at,
                )
                self._bump("events_emitted", len(bar_events))
                events.extend(bar_events)
            last_seen = stamp
        self.last_bar_ts[key] = last_seen
        self.coverage["last_complete_bar_at"] = last_seen
        self._save()
        return events

    # ------------------------------------------------------------------
    def _empty_coverage(self, session_date: str) -> dict:
        return {
            "session_date": str(session_date or ""),
            "engine_version": ENGINE_VERSION,
            "config_hash": self.config_hash,
            "machine": self.machine,
            "timezone": self.timezone_name,
            "evaluations": 0,
            "candidates_created": 0,
            "candidates_skipped_no_plan": 0,
            "bars_seen": 0,
            "bars_consumed": 0,
            "bars_skipped_incomplete": 0,
            "bars_skipped_duplicate": 0,
            "events_emitted": 0,
            "plan_revisions": 0,
            # provenance quality of the D1 triggers behind today's candidates
            "candidates_with_trigger_provenance": 0,
            "candidates_without_trigger_provenance": 0,
            "steps_missing_trigger_id": 0,
            "errors": 0,
            "last_error": "",
            "last_error_at": "",
            "last_evaluation_at": "",
            "last_complete_bar_at": "",
            "last_candidate_id": "",
        }

    def _activate_session(self, session_date: str, *, evaluated_at: datetime) -> None:
        previous_session = str(self.coverage.get("session_date") or "")
        previous_config = str(self.coverage.get("config_hash") or self.config_hash)
        if previous_session and (
            previous_session != session_date or previous_config != self.config_hash
        ):
            finalize_session(
                engine=GREATNESS_ENGINE,
                log_path=self.events_path,
                coverage=dict(self.coverage),
                finalized_at=evaluated_at,
                reason=(
                    "session_rollover"
                    if previous_session != session_date
                    else "configuration_changed"
                ),
                engine_version=str(
                    self.coverage.get("engine_version") or ENGINE_VERSION
                ),
                machine=str(self.coverage.get("machine") or self.machine),
                timezone=str(self.coverage.get("timezone") or self.timezone_name),
                configuration=previous_config,
            )
        if previous_session != session_date or previous_config != self.config_hash:
            self.coverage = self._empty_coverage(session_date)
        stale_keys = [
            key
            for key, candidate in self.candidates.items()
            if candidate.session_date != session_date
        ]
        for key in stale_keys:
            self.candidates.pop(key, None)
            self.last_bar_ts.pop(key, None)

    def _bump(self, name: str, amount: int = 1) -> None:
        self.coverage[name] = int(self.coverage.get(name, 0) or 0) + int(amount)

    def record_error(self, error: Exception | str, *, evaluated_at: datetime | None = None) -> None:
        moment = normalize_market_local_datetime(
            evaluated_at, local_timezone=self.local_timezone
        )
        self._bump("errors")
        self.coverage["last_error"] = str(error)[:500]
        self.coverage["last_error_at"] = moment.isoformat(timespec="seconds")
        self._save()

    def _base_row(
        self,
        *,
        candidate: DevelopmentCandidate,
        candidate_id: str,
        evaluated_at: datetime,
        bar: M5Bar | None,
    ) -> dict:
        plan_hash = _plan_hash(candidate)
        provenance = dict(candidate.provenance or {})
        # v2 wrote this fabricated string and called it provenance; it is kept
        # for one schema version so any external reader of the old value has a
        # migration window, and is dropped in v4.
        legacy_trigger_id = f"d1:{candidate.symbol.upper()}:{plan_hash[:12]}"
        real_trigger_id = str(provenance.get("primary_trigger_id") or "")
        return {
            "schema": SHADOW_SCHEMA,
            "engine_version": ENGINE_VERSION,
            "config_hash": self.config_hash,
            "machine": self.machine,
            "timezone": self.timezone_name,
            "session_date": candidate.session_date,
            "candidate_id": candidate_id,
            "source_trigger_id": real_trigger_id or legacy_trigger_id,
            # never let a fallback masquerade as sourced provenance
            "source_trigger_id_synthesized": not real_trigger_id,
            "source_trigger_id_legacy": legacy_trigger_id,
            "source_trigger_ids": list(provenance.get("trigger_ids") or []),
            "source_provenance": provenance,
            "symbol": candidate.symbol.upper(),
            "side": candidate.side.upper(),
            "setup_family": candidate.setup_family,
            "plan_version": candidate.plan.version,
            "plan_hash": plan_hash,
            "evaluated_at": evaluated_at.isoformat(timespec="seconds"),
            "bar_ts": bar.ts.isoformat(timespec="seconds") if bar is not None else "",
            "bar": (
                {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "complete": bar.complete,
                }
                if bar is not None
                else {}
            ),
        }

    @staticmethod
    def _step_trigger(candidate: DevelopmentCandidate, step_label: str) -> dict:
        """Provenance of the step a transition belongs to (empty when none)."""
        label = str(step_label or "")
        if not label:
            return {}
        for step in candidate.plan.steps:
            if step.label == label:
                return dict(step.provenance or {})
        return {}

    def _audit_row(
        self,
        *,
        event: str,
        candidate: DevelopmentCandidate,
        candidate_id: str,
        evaluated_at: datetime,
        bar: M5Bar | None,
        extra: dict | None = None,
    ) -> dict:
        row = self._base_row(
            candidate=candidate,
            candidate_id=candidate_id,
            evaluated_at=evaluated_at,
            bar=bar,
        )
        row.update(
            {
                "ts": bar.ts.isoformat(timespec="seconds") if bar is not None else evaluated_at.isoformat(timespec="seconds"),
                "event": str(event),
                "step": "",
                "step_trigger": {},
                "price": bar.close if bar is not None else None,
                "attempt": candidate.attempts,
                "stage": candidate.stage.value,
            }
        )
        row.update(extra or {})
        return row

    def _append_events(
        self,
        events,
        *,
        candidate: DevelopmentCandidate,
        candidate_id: str,
        bar: M5Bar,
        evaluated_at: datetime,
    ) -> None:
        rows = []
        for event in events:
            row = self._base_row(
                candidate=candidate,
                candidate_id=candidate_id,
                evaluated_at=evaluated_at,
                bar=bar,
            )
            row.update(
                {
                    "ts": event.ts.isoformat(timespec="seconds"),
                    "event": event.event.value,
                    "step": event.step_label,
                    "step_trigger": self._step_trigger(candidate, event.step_label),
                    "price": event.price,
                    "attempt": event.attempt,
                    "stage": event.stage.value,
                }
            )
            rows.append(row)
        self._append_rows(rows)

    def _append_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        try:
            self._rotate_legacy_events_if_needed()
            append_jsonl_rows(self.events_path, rows)
        except OSError as exc:
            self._bump("errors")
            self.coverage["last_error"] = f"event append failed: {exc}"[:500]
            self.coverage["last_error_at"] = normalize_market_local_datetime().isoformat(
                timespec="seconds"
            )
            logging.warning("Greatness shadow event append failed.", exc_info=True)

    def _rotate_legacy_events_if_needed(self) -> None:
        if not self.events_path.exists() or self.events_path.stat().st_size == 0:
            return
        try:
            first = next(
                (line for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()),
                "",
            )
            payload = json.loads(first) if first else {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("schema") in COMPATIBLE_SHADOW_SCHEMAS:
            return
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        archive = self.events_path.with_name(
            f"{self.events_path.stem}.legacy-{stamp}{self.events_path.suffix}"
        )
        counter = 1
        while archive.exists():
            archive = self.events_path.with_name(
                f"{self.events_path.stem}.legacy-{stamp}-{counter}{self.events_path.suffix}"
            )
            counter += 1
        os.replace(self.events_path, archive)
        logging.info("Archived legacy Greatness shadow evidence to %s", archive)

    def _save(self) -> None:
        try:
            payload = {
                "schema": STORE_SCHEMA,
                "engine_version": ENGINE_VERSION,
                "config_hash": self.config_hash,
                "machine": self.machine,
                "timezone": self.timezone_name,
                "updated_at": normalize_market_local_datetime().isoformat(timespec="seconds"),
                "coverage": self.coverage,
                "candidates": {k: c.to_dict() for k, c in self.candidates.items()},
                "last_bar_ts": self.last_bar_ts,
            }
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            # Shared primitive: same-dir temp + os.replace with guaranteed temp
            # cleanup on every failure path (the hand-rolled mkstemp copy this
            # replaces leaked a .tmp file whenever the write raised).
            atomic_write_json(self.store_path, payload, indent=None)
        except OSError:
            logging.warning("Greatness store save failed.", exc_info=True)

    def _load(self) -> None:
        try:
            if not self.store_path.exists():
                return
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
            old_last_bar_ts = dict(payload.get("last_bar_ts") or {})
            for old_key, raw in (payload.get("candidates") or {}).items():
                candidate = DevelopmentCandidate.from_dict(raw)
                new_key = _candidate_id(candidate)
                self.candidates[new_key] = candidate
                self.last_bar_ts[new_key] = str(old_last_bar_ts.get(old_key) or "")
            coverage = payload.get("coverage")
            if isinstance(coverage, dict):
                self.coverage.update(coverage)
                self.coverage.setdefault(
                    "config_hash", str(payload.get("config_hash") or self.config_hash)
                )
                self.coverage.setdefault("engine_version", ENGINE_VERSION)
                self.coverage.setdefault("machine", self.machine)
                self.coverage.setdefault("timezone", self.timezone_name)
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            logging.warning("Greatness store load failed; starting fresh.", exc_info=True)


def shadow_board() -> GreatnessBoard:
    global _board
    with _lock:
        if _board is None:
            _board = GreatnessBoard()
        return _board


def _bars_from_frame(today_df, *, now: datetime | None = None) -> list[M5Bar]:
    local_timezone, _ = get_market_local_timezone()
    moment = normalize_market_local_datetime(now, local_timezone=local_timezone)
    bars: list[M5Bar] = []
    rows = today_df.dropna(subset=["datetime"]).sort_values("datetime")
    total = len(rows)
    for index, (_, row) in enumerate(rows.iterrows()):
        start = normalize_market_local_datetime(
            row["datetime"].to_pydatetime(), local_timezone=local_timezone
        )
        complete = True
        if index == total - 1:
            complete = (moment - start) >= timedelta(minutes=_BAR_MINUTES)
        try:
            bars.append(
                M5Bar(
                    ts=start + timedelta(minutes=_BAR_MINUTES),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0) or 0.0),
                    complete=complete,
                )
            )
        except (TypeError, ValueError, KeyError):
            continue
    return bars


def record_d1_shadow(bot, symbol: str, today_df, *, now: datetime | None = None) -> list:
    """Fail-safe hook called from the live D1 trigger path. Never raises."""
    try:
        symbol = str(symbol or "").strip().upper()
        watch_entry = (
            getattr(bot, "master_avwap_d1_upgrade_alerts", {}).get(symbol)
            or getattr(bot, "master_avwap_d1_watchlist", {}).get(symbol)
            or {}
        )
        trigger_levels = watch_entry.get("trigger_levels") or []
        if not trigger_levels or today_df is None or getattr(today_df, "empty", True):
            return []
        working = today_df
        if "datetime" not in working.columns:
            return []
        side = str(watch_entry.get("side") or "LONG")
        moment = normalize_market_local_datetime(now)
        session_date = moment.date().isoformat()
        return shadow_board().update(
            symbol,
            side,
            trigger_levels,
            _bars_from_frame(working, now=now),
            session_date=session_date,
            evaluated_at=moment,
        )
    except Exception as exc:
        try:
            shadow_board().record_error(exc, evaluated_at=now)
        except Exception:
            pass
        logging.warning("Greatness D1 shadow failed (live alerts unaffected).", exc_info=True)
        return []
