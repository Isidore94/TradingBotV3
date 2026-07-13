from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from project_paths import (
    MASTER_AVWAP_FOCUS_FILE,
    MASTER_AVWAP_PRIORITY_SETUPS_FILE,
)
from master_avwap_lib.setup_tagging import derive_setup_tag_payload
from ui.models.setup import SetupRow


# The "Ranked by Expected-R (blended)" section of the priority report is the
# machine-readable feed (flush-left, raw internal bucket names, e.g.
#   AVB LONG ExpR=+0.12R score=112 WR=60% PF=3.2 n=19
#       family=mid earnings 1st-dev retest bucket=near_favorite_zone
# The optional trailing "R" on ExpR is stripped; family may contain spaces so it
# is captured lazily up to the " bucket=" delimiter.
RANKED_LINE_RE = re.compile(
    r"^(?P<symbol>[A-Z][A-Z0-9.\-]*)\s+"
    r"(?P<side>LONG|SHORT)\s+"
    r"(?:ExpR=(?P<expected_r>[+-]?\d+(?:\.\d+)?)R?\s+)?"
    r"score=(?P<score>-?\d+(?:\.\d+)?)\s+"
    r"(?:WR=\S+\s+)?(?:PF=\S+\s+)?(?:n=\S+\s+)?"
    r"family=(?P<family>.+?)\s+"
    r"bucket=(?P<bucket>\S+)\s*$"
)

# The "Overall score rankings" section carries the band zone (used as key level)
# but uses padded display labels with spaces, so values are delimited by the
# following ``token=`` rather than whitespace.
ZONE_LINE_RE = re.compile(
    r"^\s*\d+\.\s+(?P<symbol>[A-Z][A-Z0-9.\-]*)\s+(?:LONG|SHORT)\s+"
    r".*?\bzone=(?P<zone>.+?)\s+trend="
)


def rows_from_run_result(run_result: dict[str, Any] | None) -> list[SetupRow]:
    if not isinstance(run_result, dict):
        return []

    theta_by_symbol = _theta_by_symbol(run_result)
    rows: list[SetupRow] = []
    seen: set[tuple[str, str, str]] = set()

    # Desk-worthy sources: tracked rows plus study families that are real
    # entry patterns with docs and a trade plan (playbook discoveries, weekly
    # 8EMA basket, dev breakouts). Context-overlay studies (HV/HTF/relative/
    # phase6) live in the tracker only.
    seen_symbol_side: set[tuple[str, str]] = set()
    for source_key in (
        "tracked_rows",
        "weekly_ema8_hold_study_rows",
        "playbook_study_rows",
        "first_dev_breakout_study_rows",
        "second_dev_breakout_study_rows",
    ):
        is_study_source = source_key.endswith("_study_rows")
        for raw in _iter_dicts(run_result.get(source_key)):
            row = setup_row_from_mapping(raw, theta_by_symbol=theta_by_symbol, source=source_key)
            identity = (row.symbol, row.side, row.bucket)
            if not row.symbol or identity in seen:
                continue
            # A study clone adds nothing when the symbol/side is already on the
            # board with its real bucket - skip it instead of duplicating.
            if is_study_source and (row.symbol, row.side) in seen_symbol_side:
                continue
            rows.append(row)
            seen.add(identity)
            seen_symbol_side.add((row.symbol, row.side))

    rows.sort(key=_sort_key)
    return rows


def _rows_from_focus_payload(payload: Any) -> list[SetupRow]:
    if not isinstance(payload, dict):
        return []

    rows: list[SetupRow] = []
    seen: set[tuple[str, str, str]] = set()
    seen_symbol_side: set[tuple[str, str]] = set()
    for key in (
        "high_conviction",
        "favorites",
        "near_favorite_zones",
        "post_earnings_plays",
        "sma_breakout_tracking",
        "stdev_retest_tracking",
        "study_setups",
    ):
        is_study_source = key == "study_setups"
        for raw in _iter_dicts(payload.get(key)):
            row = setup_row_from_mapping(raw, source=f"focus:{key}")
            identity = (row.symbol, row.side, row.bucket)
            if not row.symbol or identity in seen:
                continue
            if is_study_source and (row.symbol, row.side) in seen_symbol_side:
                continue
            rows.append(row)
            seen.add(identity)
            seen_symbol_side.add((row.symbol, row.side))

    rows.sort(key=_sort_key)
    return rows


def load_setup_rows_from_focus(path: Path = MASTER_AVWAP_FOCUS_FILE) -> list[SetupRow]:
    return _rows_from_focus_payload(_read_json(path))


def load_setup_rows_from_priority_report(path: Path = MASTER_AVWAP_PRIORITY_SETUPS_FILE) -> list[SetupRow]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    zone_by_symbol: dict[str, str] = {}
    for line in lines:
        zone_match = ZONE_LINE_RE.match(line)
        if zone_match:
            symbol = zone_match.group("symbol").upper()
            zone_by_symbol.setdefault(symbol, zone_match.group("zone").strip())

    rows: list[SetupRow] = []
    seen: set[tuple[str, str, str]] = set()
    for line in lines:
        match = RANKED_LINE_RE.match(line)
        if not match:
            continue
        data = match.groupdict()
        symbol = (data.get("symbol") or "").upper()
        family = (data.get("family") or "").strip()
        row = setup_row_from_mapping(
            {
                "report_line": line,
                "symbol": symbol,
                "side": data.get("side") or "",
                "score": data.get("score"),
                "priority_bucket": (data.get("bucket") or "").strip(),
                "setup_family": family,
                "current_band_zone": zone_by_symbol.get(symbol, ""),
                "expected_r": data.get("expected_r"),
                **{k: v for k, v in data.items() if v},
            },
            source="priority_report",
        )
        identity = (row.symbol, row.side, row.bucket)
        if row.symbol and identity not in seen:
            rows.append(row)
            seen.add(identity)

    rows.sort(key=_sort_key)
    return rows


_PRIORITY_GENERATED_RE = re.compile(r"Generated at\s+(\d{4}-\d{2}-\d{2})")

# Mirror the scanner's daily-bar recency tolerance (a 2-weekday gap is still
# "recent"). Anything beyond that is flagged as stale in the UI.
_STALE_WEEKDAY_GAP = 2


def _payload_data_date(payload: Any) -> str | None:
    """Best-effort scan date (YYYY-MM-DD) embedded in the focus feed payload."""
    if not isinstance(payload, dict):
        return None
    for key in ("run_date", "generated_at"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value[:10]
    return None


def read_priority_report_date(path: Path = MASTER_AVWAP_PRIORITY_SETUPS_FILE) -> str | None:
    """Parse the 'Generated at YYYY-MM-DD ...' header from the priority report."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(10):
                line = handle.readline()
                if not line:
                    break
                match = _PRIORITY_GENERATED_RE.search(line)
                if match:
                    return match.group(1)
    except OSError:
        return None
    return None


def _weekday_gap(start: date, end: date) -> int:
    if start >= end:
        return 0
    cursor = start
    count = 0
    while cursor < end:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            count += 1
    return count


def _is_stale(data_date: str | None, *, today: date | None = None) -> bool:
    if not data_date:
        return False
    try:
        parsed = date.fromisoformat(data_date[:10])
    except ValueError:
        return False
    return _weekday_gap(parsed, today or date.today()) > _STALE_WEEKDAY_GAP


def load_latest_setup_rows_with_meta() -> dict[str, Any]:
    """Return the freshest available setup rows plus the date/source behind them.

    The focus feed is the richer source but it is only rewritten in the
    final-hour/after-close window, so a pre-market scan leaves it stale while a
    fresh priority report exists. Prefer the focus feed only when it is at least
    as new as the priority report; otherwise fall back to the fresh report so
    the panel never silently shows days-old setups.
    """
    focus_payload = _read_json(MASTER_AVWAP_FOCUS_FILE)
    focus_rows = _rows_from_focus_payload(focus_payload)
    focus_date = _payload_data_date(focus_payload)
    priority_date = read_priority_report_date()

    focus_is_current = bool(focus_rows) and (
        priority_date is None or focus_date is None or focus_date >= priority_date
    )
    if focus_is_current:
        return {
            "rows": focus_rows,
            "data_date": focus_date,
            "source": "focus",
            "is_stale": _is_stale(focus_date),
        }

    priority_rows = load_setup_rows_from_priority_report()
    if priority_rows:
        return {
            "rows": priority_rows,
            "data_date": priority_date,
            "source": "priority_report",
            "is_stale": _is_stale(priority_date),
        }

    # No fresher report available; show whatever the focus feed still holds.
    return {
        "rows": focus_rows,
        "data_date": focus_date,
        "source": "focus" if focus_rows else "none",
        "is_stale": _is_stale(focus_date),
    }


def load_latest_setup_rows() -> list[SetupRow]:
    return load_latest_setup_rows_with_meta()["rows"]


def setup_row_from_mapping(
    raw: dict[str, Any],
    *,
    theta_by_symbol: dict[str, str] | None = None,
    source: str = "",
) -> SetupRow:
    symbol = str(raw.get("symbol") or "").strip().upper()
    side = _normalize_side(raw.get("side"))
    bucket = str(raw.get("priority_bucket") or raw.get("bucket") or "").strip()
    if not bucket and source.endswith("_study_rows"):
        bucket = "study"

    score = _float_or_none(raw.get("priority_score", raw.get("score")))
    normalized_raw = dict(raw)
    tag_payload = derive_setup_tag_payload(normalized_raw)
    normalized_raw.update(tag_payload)
    tags = tag_payload["setup_tags"]

    supports = _int_or_none(raw.get("support_count"))
    if supports is None:
        nearby = _int_or_none(raw.get("hv_level_nearby_count")) or 0
        blocking = _int_or_none(raw.get("hv_level_blocking_count")) or 0
        supports = nearby + blocking if nearby or blocking else None

    theta_by_symbol = theta_by_symbol or {}
    theta = theta_by_symbol.get(symbol, "")

    return SetupRow(
        symbol=symbol,
        side=side,
        score=score,
        bucket=bucket,
        setup_tags=tags,
        key_level=_key_level_text(raw),
        supports=supports,
        hv_summary=_hv_summary(raw),
        theta=theta,
        expected_r=_float_or_none(raw.get("expected_r")),
        expected_r_rank=_float_or_none(raw.get("expected_r_rank_score")),
        days_to_earnings=_int_or_none(raw.get("days_to_next_earnings")),
        last_trade_date=str(raw.get("last_trade_date") or raw.get("scan_date") or ""),
        source=source,
        raw=normalized_raw,
    )


def copy_symbols(rows: Iterable[SetupRow], kind: str) -> str:
    normalized = kind.strip().lower()
    selected: list[SetupRow] = []
    for row in rows:
        bucket = row.bucket.strip().lower()
        if normalized == "longs" and row.side != "LONG":
            continue
        if normalized == "shorts" and row.side != "SHORT":
            continue
        if normalized == "favorites" and bucket not in {"favorite_setup", "high_conviction"}:
            continue
        if normalized == "active" and bucket not in {"favorite_setup", "near_favorite_zone", "high_conviction"}:
            continue
        selected.append(row)

    if normalized == "ranked":
        # Preserve the table's current (rank) order; only de-duplicate.
        seen: set[str] = set()
        ordered: list[str] = []
        for row in selected:
            if row.symbol and row.symbol not in seen:
                seen.add(row.symbol)
                ordered.append(row.symbol)
        return ", ".join(ordered)

    symbols = sorted({row.symbol for row in selected if row.symbol})
    return ", ".join(symbols)


def _theta_by_symbol(run_result: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for source_key, label in (("theta_put_rows", "Put"), ("theta_pcs_rows", "PCS")):
        for row in _iter_dicts(run_result.get(source_key)):
            symbol = str(row.get("symbol") or "").strip().upper()
            if not symbol or symbol in lookup:
                continue
            lookup[symbol] = _theta_summary(row, label)
    return lookup


def _theta_summary(row: dict[str, Any], label: str) -> str:
    option = row.get("best_option") if isinstance(row.get("best_option"), dict) else {}
    strike = option.get("strike", option.get("short_strike", row.get("sell_strike", "")))
    credit = option.get("credit", row.get("recommended_credit", ""))
    dte = option.get("market_days", row.get("market_days", ""))
    parts = [label]
    if strike not in (None, ""):
        parts.append(f"{_format_number(strike)} strike")
    if credit not in (None, ""):
        parts.append(f"@ {_format_number(credit)}")
    if dte not in (None, ""):
        parts.append(f"{dte}d")
    return " ".join(parts)


def _key_level_text(row: dict[str, Any]) -> str:
    for key in (
        "current_band_zone",
        "favorite_zone",
        "retest_reference_level",
        "top_pattern_entry_level",
        "mid_earnings_primary_trigger_level",
        "sma_breakout_retest_level",
        "post_earnings_monitor_level",
        "hv_level_nearest_price",
    ):
        value = row.get(key)
        if value not in (None, ""):
            if isinstance(value, (float, int)):
                return _format_number(value)
            return str(value)
    return ""


def _hv_summary(row: dict[str, Any]) -> str:
    nearby = _int_or_none(row.get("hv_level_nearby_count")) or 0
    blocking = _int_or_none(row.get("hv_level_blocking_count")) or 0
    if not nearby and not blocking and not row.get("hv_level_break_today"):
        return ""
    parts: list[str] = []
    if nearby:
        parts.append(f"{nearby} near")
    if blocking:
        parts.append(f"{blocking} block")
    if row.get("hv_level_break_today"):
        parts.append("break")
    return ", ".join(parts)


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item


def _listish(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        separators = ";" if ";" in value else ","
        return [part.strip() for part in value.split(separators) if part.strip()]
    return []


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _normalize_side(value: Any) -> str:
    side = str(value or "").strip().upper()
    if side in {"LONG", "SHORT"}:
        return side
    return side


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_number(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return str(value)
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.2f}"


def _sort_key(row: SetupRow) -> tuple[int, float, float, str]:
    bucket_rank = {
        "high_conviction": 0,
        "favorite_setup": 1,
        "near_favorite_zone": 2,
        "post_earnings_play": 3,
        "sma_breakout_tracking": 4,
        "stdev_retest_tracking": 5,
        "study_playbook": 6,
        "study": 7,
    }.get(row.bucket.strip().lower(), 8)
    # Expected-R is the ranking spine (2026-07-02): the outcome-calibrated
    # estimate orders rows; raw score only breaks ties or orders rows that have
    # no estimate yet - it can no longer put stacked-signal junk on top.
    primary = row.expected_r_rank if row.expected_r_rank is not None else row.expected_r
    return (
        bucket_rank,
        -(primary if primary is not None else -999999.0),
        -(row.score if row.score is not None else -999999.0),
        row.symbol,
    )
