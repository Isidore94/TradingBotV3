"""Append-only completed-M5 recorder for NYSE breadth data.

``$VOLD`` is not a portable IBKR contract name. The live adapter therefore
qualifies an ordered set of candidates and verifies that historical M5 data is
actually available before activating one. The recorder persists the exact
contract returned by IBKR on every row; a fallback is never mislabeled as true
up-volume minus down-volume.

This module owns persistence and completed-bar discipline only. Broker calls
remain in the BounceBot adapter so the recorder stays unit-testable.
"""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from diagnostics.artifact_io import (
    CAPTURE_MODE_BACKFILL,
    CAPTURE_MODE_LIVE,
    append_jsonl_rows,
    atomic_write_json,
    read_jsonl,
)
from market_session import (
    get_market_session_window,
    normalize_market_local_datetime,
)
from project_paths import get_diagnostics_dir, get_local_setting


FEATURE_VERSION = "vold_session_recorder_v1"
EVENT_SCHEMA = "vold_session_event_v1"
STATE_SCHEMA = "vold_session_state_v1"
BAR_MINUTES = 5
BREADTH_BACKFILL_SETTING_KEY = "breadth_backfill"


def breadth_backfill_enabled() -> bool:
    """Default-on: a marked, append-only recovery path (durability sec 2.4)."""
    return bool(get_local_setting(BREADTH_BACKFILL_SETTING_KEY, True))


@dataclass(frozen=True)
class BreadthContractCandidate:
    symbol: str
    sec_type: str
    exchange: str
    currency: str
    proxy_kind: str
    semantic_description: str


CONTRACT_CANDIDATES: tuple[BreadthContractCandidate, ...] = (
    BreadthContractCandidate(
        "VOLD",
        "IND",
        "NYSE",
        "USD",
        "exact_vold",
        "NYSE up-volume minus down-volume",
    ),
    BreadthContractCandidate(
        "VOLD-NYSE",
        "IND",
        "NYSE",
        "USD",
        "exact_vold",
        "NYSE up-volume minus down-volume",
    ),
    BreadthContractCandidate(
        "VOL-NYSE",
        "IND",
        "NYSE",
        "USD",
        "nyse_total_volume_proxy",
        "NYSE total-volume index; not directional volume breadth",
    ),
    BreadthContractCandidate(
        "AD-NYSE",
        "IND",
        "NYSE",
        "USD",
        "nyse_advance_decline_proxy",
        "NYSE advancing issues minus declining issues; not volume breadth",
    ),
    BreadthContractCandidate(
        "TICK-NYSE",
        "IND",
        "NYSE",
        "USD",
        "nyse_tick_proxy",
        "NYSE upticking issues minus downticking issues; not volume breadth",
    ),
)


def vold_ledger_path() -> Path:
    return get_diagnostics_dir() / "vold_m5.jsonl"


def vold_state_path() -> Path:
    return get_diagnostics_dir() / "vold_recorder_state.json"


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:
            return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for pattern in ("%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def _records(rows: Any) -> list[Mapping[str, Any]]:
    if rows is None:
        return []
    if hasattr(rows, "to_dict"):
        try:
            converted = rows.to_dict("records")
        except Exception:
            return []
        return [row for row in converted if isinstance(row, Mapping)]
    return [row for row in rows if isinstance(row, Mapping)]


def completed_breadth_bars(
    rows: Any,
    *,
    now: datetime | None = None,
    session_date: date | None = None,
) -> list[dict[str, Any]]:
    """Normalize valid rows and exclude a still-forming five-minute bar.

    ``session_date`` selects a specific session instead of the newest one in
    ``rows``. Live polling leaves it unset; the gap fill sets it so a
    multi-day historical fetch can repair *yesterday's* ledger.
    """

    moment = normalize_market_local_datetime(now)
    output: list[dict[str, Any]] = []
    for row in _records(rows):
        raw_start = _parse_datetime(row.get("datetime") or row.get("time") or row.get("date"))
        if raw_start is None:
            continue
        local_start = normalize_market_local_datetime(raw_start)
        local_end = local_start + timedelta(minutes=BAR_MINUTES)
        if local_end > moment:
            continue
        try:
            open_value = float(row["open"])
            high_value = float(row["high"])
            low_value = float(row["low"])
            close_value = float(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (open_value, high_value, low_value, close_value)):
            continue
        if low_value > high_value:
            continue
        if not low_value <= min(open_value, close_value) <= high_value:
            continue
        if not low_value <= max(open_value, close_value) <= high_value:
            continue
        output.append(
            {
                "_start_local": local_start,
                "_end_local": local_end,
                "bar_start": local_start.isoformat(timespec="seconds"),
                "bar_end": local_end.isoformat(timespec="seconds"),
                "open": open_value,
                "high": high_value,
                "low": low_value,
                "close": close_value,
            }
        )
    output.sort(key=lambda row: row["_start_local"])
    if not output:
        return []
    target_session = session_date or output[-1]["_start_local"].date()
    return [row for row in output if row["_start_local"].date() == target_session]


def contract_metadata(
    contract: Any,
    *,
    candidate: BreadthContractCandidate,
    long_name: str = "",
    valid_exchanges: str = "",
) -> dict[str, Any]:
    """Serialize the exact qualified IBKR contract without importing ibapi."""

    def field(name: str, default: Any = "") -> Any:
        if isinstance(contract, Mapping):
            return contract.get(name, default)
        return getattr(contract, name, default)

    return {
        "con_id": int(field("conId", field("con_id", 0)) or 0),
        "symbol": str(field("symbol") or ""),
        "local_symbol": str(field("localSymbol", field("local_symbol", "")) or ""),
        "sec_type": str(field("secType", field("sec_type", "")) or ""),
        "exchange": str(field("exchange") or ""),
        "primary_exchange": str(
            field("primaryExchange", field("primary_exchange", "")) or ""
        ),
        "currency": str(field("currency") or ""),
        "trading_class": str(field("tradingClass", field("trading_class", "")) or ""),
        "long_name": str(long_name or ""),
        "valid_exchanges": str(valid_exchanges or ""),
        "requested_candidate": asdict(candidate),
        "proxy_kind": candidate.proxy_kind,
        "semantic_description": candidate.semantic_description,
        "is_exact_vold": candidate.proxy_kind == "exact_vold",
    }


def _wall_clock(now: datetime | None = None) -> str:
    return normalize_market_local_datetime(now).isoformat(timespec="seconds")


def _contiguous_runs(
    bar_ends: list[datetime],
) -> list[tuple[datetime, datetime, int]]:
    """Group sorted bar ends into ``(first, last, count)`` runs of adjacent
    five-minute slots, so one outage becomes one gap row rather than dozens."""
    runs: list[tuple[datetime, datetime, int]] = []
    for value in bar_ends:
        if runs and value - runs[-1][1] == timedelta(minutes=BAR_MINUTES):
            first, _last, count = runs[-1]
            runs[-1] = (first, value, count + 1)
        else:
            runs.append((value, value, 1))
    return runs


class VoldSessionRecorder:
    """Crash-recoverable, append-only recorder for one qualified breadth feed."""

    def __init__(
        self,
        *,
        ledger_path: Path | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.ledger_path = Path(ledger_path or vold_ledger_path())
        self.state_path = Path(state_path or vold_state_path())
        self._lock = threading.RLock()
        self.session_date = ""
        self.contract: dict[str, Any] = {}
        self.seen_bar_ends: set[str] = set()
        self.seen_gap_keys: set[str] = set()
        self.contract_markers: set[str] = set()
        self.backfill_markers: set[str] = set()
        self.last_error = ""
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            import json

            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, Mapping) or payload.get("schema") != STATE_SCHEMA:
            return
        self.session_date = str(payload.get("session_date") or "")
        self.contract = (
            dict(payload.get("contract") or {})
            if isinstance(payload.get("contract"), Mapping)
            else {}
        )
        self.seen_bar_ends = {
            str(value) for value in payload.get("seen_bar_ends") or [] if str(value)
        }
        self.seen_gap_keys = {
            str(value) for value in payload.get("seen_gap_keys") or [] if str(value)
        }
        self.contract_markers = {
            str(value) for value in payload.get("contract_markers") or [] if str(value)
        }
        self.backfill_markers = {
            str(value) for value in payload.get("backfill_markers") or [] if str(value)
        }
        self.last_error = str(payload.get("last_error") or "")
        self._recover_from_ledger()

    def _recover_from_ledger(self) -> None:
        if not self.session_date:
            return
        for row in read_jsonl(self.ledger_path):
            if str(row.get("session_date") or "") != self.session_date:
                continue
            event_type = str(row.get("event_type") or "")
            if event_type == "breadth_bar":
                self.seen_bar_ends.add(str(row.get("bar_end") or ""))
            elif event_type == "data_gap":
                self.seen_gap_keys.add(str(row.get("gap_key") or ""))
            elif event_type == "contract_verified":
                self.contract_markers.add(str(row.get("contract_marker") or ""))

    def _ensure_session(self, session_date: str) -> None:
        if self.session_date == session_date:
            return
        self.session_date = session_date
        self.seen_bar_ends = set()
        self.seen_gap_keys = set()
        self.contract_markers = set()
        self.last_error = ""
        self._recover_from_ledger()

    def _save_state(self, *, now: datetime | None = None) -> None:
        atomic_write_json(
            self.state_path,
            {
                "schema": STATE_SCHEMA,
                "feature_version": FEATURE_VERSION,
                "code_version": FEATURE_VERSION,
                "session_date": self.session_date,
                "contract": self.contract,
                "seen_bar_ends": sorted(self.seen_bar_ends),
                "seen_gap_keys": sorted(self.seen_gap_keys),
                "contract_markers": sorted(self.contract_markers),
                # Not reset per session: the marker names the session it
                # covers, so an earlier session's completed repair survives the
                # rollover that would otherwise let it run again tomorrow.
                "backfill_markers": sorted(self.backfill_markers),
                "last_error": self.last_error,
                "written_at": _wall_clock(now),
            },
        )

    def activate_contract(
        self,
        metadata: Mapping[str, Any],
        *,
        as_of: datetime | str | None = None,
        now: datetime | None = None,
    ) -> None:
        moment = normalize_market_local_datetime(now)
        session = get_market_session_window(moment)
        with self._lock:
            self._ensure_session(session.market_date.isoformat())
            self.contract = dict(metadata)
            marker = (
                f"{self.session_date}|{self.contract.get('con_id')}|"
                f"{self.contract.get('symbol')}|{self.contract.get('proxy_kind')}"
            )
            if marker not in self.contract_markers:
                as_of_text = (
                    as_of.isoformat(timespec="seconds")
                    if isinstance(as_of, datetime)
                    else str(as_of or moment.isoformat(timespec="seconds"))
                )
                append_jsonl_rows(
                    self.ledger_path,
                    (
                        {
                            "schema": EVENT_SCHEMA,
                            "feature_version": FEATURE_VERSION,
                            "code_version": FEATURE_VERSION,
                            "event_type": "contract_verified",
                            "session_date": self.session_date,
                            "contract_marker": marker,
                            "target_metric": "$VOLD",
                            "contract": self.contract,
                            "as_of": as_of_text,
                            "written_at": _wall_clock(now),
                        },
                    ),
                    fsync=True,
                )
                self.contract_markers.add(marker)
            self.last_error = ""
            self._save_state(now=now)

    def observe(
        self,
        rows: Any,
        *,
        now: datetime | None = None,
    ) -> int:
        bars = completed_breadth_bars(rows, now=now)
        if not bars:
            return 0
        moment = normalize_market_local_datetime(now)
        session = get_market_session_window(bars[-1]["_start_local"])
        with self._lock:
            self._ensure_session(session.market_date.isoformat())
            return self._ingest_bars(
                bars,
                session,
                capture_mode=CAPTURE_MODE_LIVE,
                now=moment,
            )

    def _ingest_bars(
        self,
        bars: list[dict[str, Any]],
        session: Any,
        *,
        capture_mode: str,
        now: datetime,
        record_internal_gaps: bool = True,
    ) -> int:
        """Append unseen completed bars. Caller holds the lock and has already
        established the session; dedupe by bar end is what makes this safe to
        run over the same provider response any number of times."""
        moment = now
        new_rows: list[dict[str, Any]] = []
        prior_end: datetime | None = None
        for bar in bars:
            if not session.open_local <= bar["_start_local"] < session.close_local:
                continue
            if record_internal_gaps and prior_end is not None and bar["_start_local"] > prior_end:
                missing = int(
                    (bar["_start_local"] - prior_end).total_seconds()
                    // (BAR_MINUTES * 60)
                )
                if missing > 0:
                    gap_key = (
                        f"internal|{prior_end.isoformat(timespec='seconds')}|"
                        f"{bar['_start_local'].isoformat(timespec='seconds')}"
                    )
                    if gap_key not in self.seen_gap_keys:
                        new_rows.append(
                            self._gap_row(
                                gap_key=gap_key,
                                as_of=bar["bar_start"],
                                reason="missing_completed_bars_inside_provider_response",
                                missing_bar_count=missing,
                                gap_start=prior_end.isoformat(timespec="seconds"),
                                gap_end=bar["_start_local"].isoformat(timespec="seconds"),
                                now=moment,
                                capture_mode=capture_mode,
                            )
                        )
                        self.seen_gap_keys.add(gap_key)
            prior_end = bar["_end_local"]
            if bar["bar_end"] in self.seen_bar_ends:
                continue
            new_rows.append(
                {
                    "schema": EVENT_SCHEMA,
                    "feature_version": FEATURE_VERSION,
                    "code_version": FEATURE_VERSION,
                    "event_type": "breadth_bar",
                    "session_date": self.session_date,
                    "target_metric": "$VOLD",
                    "contract": self.contract,
                    "capture_mode": capture_mode,
                    "bar_start": bar["bar_start"],
                    "bar_end": bar["bar_end"],
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "as_of": bar["bar_end"],
                    "written_at": _wall_clock(moment),
                }
            )
            self.seen_bar_ends.add(bar["bar_end"])
        if not new_rows:
            return 0
        append_jsonl_rows(self.ledger_path, new_rows, fsync=True)
        self.last_error = ""
        self._save_state(now=moment)
        return sum(row["event_type"] == "breadth_bar" for row in new_rows)

    def _gap_row(
        self,
        *,
        gap_key: str,
        as_of: str,
        reason: str,
        missing_bar_count: int,
        gap_start: str,
        gap_end: str,
        now: datetime,
        capture_mode: str = CAPTURE_MODE_LIVE,
    ) -> dict[str, Any]:
        return {
            "schema": EVENT_SCHEMA,
            "feature_version": FEATURE_VERSION,
            "code_version": FEATURE_VERSION,
            "event_type": "data_gap",
            "session_date": self.session_date,
            "target_metric": "$VOLD",
            "contract": self.contract,
            "capture_mode": capture_mode,
            "gap_key": gap_key,
            "reason": str(reason or "provider_data_unavailable"),
            "missing_bar_count": int(max(1, missing_bar_count)),
            "gap_start": gap_start,
            "gap_end": gap_end,
            "data_gap": True,
            "as_of": as_of,
            "written_at": _wall_clock(now),
        }

    def backfill_trigger(self, *, now: datetime | None = None) -> str:
        """Which gap fill is due right now, or "" for none.

        The close (the session's rows are final) and startup on a later day
        (the close was missed entirely) - the same two moments the follow-up
        chain sweeper uses. Each fires once per session; the marker is
        persisted so a restart cannot re-spend IB requests on a session that
        already got its honest answer.
        """
        if not self.session_date:
            return ""
        moment = normalize_market_local_datetime(now)
        window = get_market_session_window(moment)
        if self.session_date != window.market_date.isoformat():
            trigger = "startup_after_missed_close"
        elif moment >= window.close_local:
            trigger = "close_of_day"
        else:
            return ""
        if f"{self.session_date}|{trigger}" in self.backfill_markers:
            return ""
        return trigger

    def _session_window_for(self, session_date: date, reference: datetime):
        """Session window for an arbitrary market date, using midday so no
        boundary rounding can land the probe outside the session."""
        probe = datetime(
            session_date.year,
            session_date.month,
            session_date.day,
            12,
            0,
            tzinfo=reference.tzinfo,
        )
        return get_market_session_window(normalize_market_local_datetime(probe))

    def _gap_covered_bar_ends(self) -> set[str]:
        """Bar ends this session's ledger already marks as an explicit gap.

        Expanded from the recorded [gap_start, gap_end) ranges so the gap fill
        never writes a second marker over evidence the live path already
        recorded honestly.
        """
        covered: set[str] = set()
        for row in read_jsonl(self.ledger_path):
            if str(row.get("session_date") or "") != self.session_date:
                continue
            if str(row.get("event_type") or "") != "data_gap":
                continue
            start = _parse_datetime(row.get("gap_start"))
            end = _parse_datetime(row.get("gap_end"))
            if start is None or end is None:
                continue
            cursor = normalize_market_local_datetime(start) + timedelta(minutes=BAR_MINUTES)
            stop = normalize_market_local_datetime(end)
            while cursor <= stop:
                covered.add(cursor.isoformat(timespec="seconds"))
                cursor += timedelta(minutes=BAR_MINUTES)
        return covered

    def backfill_session_bars(
        self,
        rows: Any,
        *,
        session_date: str | date,
        now: datetime | None = None,
        trigger: str = "",
    ) -> dict[str, Any]:
        """Fill a session's missing completed M5 bars (durability sec 2.4).

        A completed breadth bar is a pure function of provider history, so an
        outage's missing rows are recoverable (Tier B). Recovered rows carry
        ``capture_mode: "backfill"`` and the contract provenance actually in
        use; bar ends stay unique, so this is safe to run repeatedly and can
        never rewrite or displace a row the live poller already wrote.

        Slots that remain missing afterwards - the provider genuinely has no
        data, or the desk was down and no poll ever recorded them - get one
        explicit ``data_gap`` row per contiguous run, but only where the ledger
        does not already carry a marker covering them. Missing data stays
        uncertainty; it is never quietly filled in.
        """
        summary: dict[str, Any] = {
            "ran": False,
            "trigger": str(trigger or ""),
            "session_date": "",
            "bars_added": 0,
            "gap_rows_added": 0,
            "still_missing": 0,
            "reason": "",
        }
        if not breadth_backfill_enabled():
            summary["reason"] = f"disabled by {BREADTH_BACKFILL_SETTING_KEY} setting"
            return summary

        target = (
            session_date
            if isinstance(session_date, date)
            else date.fromisoformat(str(session_date))
        )
        moment = normalize_market_local_datetime(now)
        session = self._session_window_for(target, moment)
        bars = completed_breadth_bars(rows, now=moment, session_date=target)

        with self._lock:
            # Roll onto the session being repaired: _ensure_session rebuilds the
            # seen-bar/gap sets from the append-only ledger, so nothing is lost
            # and the next live observe rolls back the same way.
            self._ensure_session(target.isoformat())
            summary["session_date"] = self.session_date
            summary["bars_added"] = self._ingest_bars(
                bars,
                session,
                capture_mode=CAPTURE_MODE_BACKFILL,
                now=moment,
                # The whole-session pass below is the authoritative gap
                # accounting for a repair; per-response internal markers would
                # double-count the same slots.
                record_internal_gaps=False,
            )

            covered = self._gap_covered_bar_ends()
            complete_through = min(moment, session.close_local)
            missing: list[datetime] = []
            slot_start = session.open_local
            while slot_start + timedelta(minutes=BAR_MINUTES) <= complete_through:
                slot_end = slot_start + timedelta(minutes=BAR_MINUTES)
                stamp = slot_end.isoformat(timespec="seconds")
                if stamp not in self.seen_bar_ends and stamp not in covered:
                    missing.append(slot_end)
                slot_start = slot_end
            summary["still_missing"] = len(missing)

            gap_rows: list[dict[str, Any]] = []
            for run_start, run_end, count in _contiguous_runs(missing):
                gap_start = (run_start - timedelta(minutes=BAR_MINUTES)).isoformat(
                    timespec="seconds"
                )
                gap_key = f"backfill|{gap_start}|{run_end.isoformat(timespec='seconds')}"
                if gap_key in self.seen_gap_keys:
                    continue
                gap_rows.append(
                    self._gap_row(
                        gap_key=gap_key,
                        as_of=run_end.isoformat(timespec="seconds"),
                        reason="no_completed_bars_available_for_backfill",
                        missing_bar_count=count,
                        gap_start=gap_start,
                        gap_end=run_end.isoformat(timespec="seconds"),
                        now=moment,
                        capture_mode=CAPTURE_MODE_BACKFILL,
                    )
                )
                self.seen_gap_keys.add(gap_key)
            if gap_rows:
                append_jsonl_rows(self.ledger_path, gap_rows, fsync=True)
            if trigger:
                self.backfill_markers.add(f"{self.session_date}|{trigger}")
            self._save_state(now=moment)
            summary["gap_rows_added"] = len(gap_rows)
            summary["ran"] = True
            summary["reason"] = (
                f"added {summary['bars_added']} backfilled bar(s) and "
                f"{len(gap_rows)} gap row(s) for {self.session_date}"
            )
            return summary

    def record_data_gap(
        self,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> bool:
        moment = normalize_market_local_datetime(now)
        session = get_market_session_window(moment)
        if moment < session.open_local:
            return False
        expected_end = min(moment, session.close_local).replace(second=0, microsecond=0)
        expected_end -= timedelta(minutes=expected_end.minute % BAR_MINUTES)
        if expected_end <= session.open_local:
            return False
        expected_start = expected_end - timedelta(minutes=BAR_MINUTES)
        with self._lock:
            self._ensure_session(session.market_date.isoformat())
            gap_key = f"poll|{expected_end.isoformat(timespec='seconds')}"
            if gap_key in self.seen_gap_keys or expected_end.isoformat(timespec="seconds") in self.seen_bar_ends:
                return False
            row = self._gap_row(
                gap_key=gap_key,
                as_of=expected_end.isoformat(timespec="seconds"),
                reason=reason,
                missing_bar_count=1,
                gap_start=expected_start.isoformat(timespec="seconds"),
                gap_end=expected_end.isoformat(timespec="seconds"),
                now=moment,
            )
            append_jsonl_rows(self.ledger_path, (row,), fsync=True)
            self.seen_gap_keys.add(gap_key)
            self.last_error = str(reason or "")
            self._save_state(now=moment)
            return True

    def record_unavailable(
        self,
        attempts: Iterable[Mapping[str, Any]],
        *,
        reason: str,
        now: datetime | None = None,
    ) -> bool:
        moment = normalize_market_local_datetime(now)
        session = get_market_session_window(moment)
        with self._lock:
            self._ensure_session(session.market_date.isoformat())
            marker = f"unavailable|{self.session_date}"
            if marker in self.contract_markers:
                return False
            row = {
                "schema": EVENT_SCHEMA,
                "feature_version": FEATURE_VERSION,
                "code_version": FEATURE_VERSION,
                "event_type": "recorder_unavailable",
                "session_date": self.session_date,
                "target_metric": "$VOLD",
                "attempts": [dict(item) for item in attempts],
                "reason": str(reason or ""),
                "data_gap": True,
                "as_of": moment.isoformat(timespec="seconds"),
                "written_at": _wall_clock(moment),
            }
            append_jsonl_rows(self.ledger_path, (row,), fsync=True)
            self.contract_markers.add(marker)
            self.last_error = str(reason or "")
            self._save_state(now=moment)
            return True
