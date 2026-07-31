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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from diagnostics.artifact_io import append_jsonl_rows, atomic_write_json, read_jsonl
from market_session import (
    get_market_session_window,
    normalize_market_local_datetime,
)
from project_paths import get_diagnostics_dir


FEATURE_VERSION = "vold_session_recorder_v1"
EVENT_SCHEMA = "vold_session_event_v1"
STATE_SCHEMA = "vold_session_state_v1"
BAR_MINUTES = 5


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
) -> list[dict[str, Any]]:
    """Normalize valid rows and exclude a still-forming five-minute bar."""

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
    latest_session = output[-1]["_start_local"].date()
    return [row for row in output if row["_start_local"].date() == latest_session]


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
            new_rows: list[dict[str, Any]] = []
            prior_end: datetime | None = None
            for bar in bars:
                if not session.open_local <= bar["_start_local"] < session.close_local:
                    continue
                if prior_end is not None and bar["_start_local"] > prior_end:
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
    ) -> dict[str, Any]:
        return {
            "schema": EVENT_SCHEMA,
            "feature_version": FEATURE_VERSION,
            "code_version": FEATURE_VERSION,
            "event_type": "data_gap",
            "session_date": self.session_date,
            "target_metric": "$VOLD",
            "contract": self.contract,
            "gap_key": gap_key,
            "reason": str(reason or "provider_data_unavailable"),
            "missing_bar_count": int(max(1, missing_bar_count)),
            "gap_start": gap_start,
            "gap_end": gap_end,
            "data_gap": True,
            "as_of": as_of,
            "written_at": _wall_clock(now),
        }

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
