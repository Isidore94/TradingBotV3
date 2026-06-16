from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from project_paths import (
    AVWAP_SIGNALS_FILE,
    INTRADAY_BOUNCES_FILE,
    MASTER_AVWAP_FOCUS_FILE,
    MASTER_AVWAP_SETUP_TRACKER_FILE,
)


DEFAULT_SWING_LOOKBACK_CALENDAR_DAYS = 16


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value or "").strip()
    if not text:
        return None
    for candidate in (text[:10], text):
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).date()
        except ValueError:
            continue
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for fmt in ("%Y%m%d  %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"LONG", "BUY", "BOT", "BTO", "COVER"}:
        return "LONG"
    if text in {"SHORT", "SELL", "SLD", "STO", "SSHORT"}:
        return "SHORT"
    return text


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _priority_tag(row: dict[str, Any]) -> str:
    family = str(row.get("setup_family") or row.get("family") or row.get("setup") or "").strip()
    bucket = str(row.get("priority_bucket") or row.get("bucket") or "").strip()
    zone = str(row.get("favorite_zone") or "").strip()
    parts = [part for part in (family, bucket, zone) if part]
    return " | ".join(parts) if parts else "bot_context"


def _date_distance_score(trade_date: date, context_date: date, lookback_days: int) -> float | None:
    delta_days = (trade_date - context_date).days
    if delta_days < 0 or delta_days > lookback_days:
        return None
    if delta_days == 0:
        return 0.28
    return max(0.04, 0.22 * (1.0 - (delta_days / max(1, lookback_days))))


class AutoTagger:
    """Suggest journal setup tags from existing bot outputs without importing scanner code."""

    def __init__(
        self,
        *,
        setup_tracker_path: Path = MASTER_AVWAP_SETUP_TRACKER_FILE,
        focus_path: Path = MASTER_AVWAP_FOCUS_FILE,
        avwap_signals_path: Path = AVWAP_SIGNALS_FILE,
        intraday_bounces_path: Path = INTRADAY_BOUNCES_FILE,
        lookback_calendar_days: int = DEFAULT_SWING_LOOKBACK_CALENDAR_DAYS,
    ) -> None:
        self.setup_tracker_path = Path(setup_tracker_path)
        self.focus_path = Path(focus_path)
        self.avwap_signals_path = Path(avwap_signals_path)
        self.intraday_bounces_path = Path(intraday_bounces_path)
        self.lookback_calendar_days = int(lookback_calendar_days)
        self._context_rows: list[dict[str, Any]] | None = None

    def load_context_rows(self) -> list[dict[str, Any]]:
        if self._context_rows is not None:
            return self._context_rows
        rows: list[dict[str, Any]] = []
        rows.extend(self._load_tracker_rows())
        rows.extend(self._load_focus_rows())
        rows.extend(self._load_avwap_signal_rows())
        rows.extend(self._load_intraday_bounce_rows())
        self._context_rows = rows
        return rows

    def _load_tracker_rows(self) -> list[dict[str, Any]]:
        payload = _load_json(self.setup_tracker_path)
        if not isinstance(payload, dict):
            return []
        setups = payload.get("setups")
        if not isinstance(setups, dict):
            return []
        rows = []
        for setup in setups.values():
            if not isinstance(setup, dict):
                continue
            rows.append(
                {
                    "source": "setup_tracker",
                    "symbol": _normalize_symbol(setup.get("symbol")),
                    "side": _normalize_side(setup.get("side")),
                    "date": _parse_date(setup.get("scan_date") or setup.get("entry_trade_date")),
                    "setup_family": setup.get("setup_family") or "general",
                    "priority_bucket": setup.get("priority_bucket") or "",
                    "favorite_zone": setup.get("favorite_zone") or "",
                    "priority_score": _coerce_float(setup.get("priority_score")),
                    "retest": setup.get("retest_reference_level") or setup.get("mid_earnings_primary_trigger_level") or "",
                    "compression": bool(setup.get("compression_flag")),
                }
            )
        return rows

    def _load_focus_rows(self) -> list[dict[str, Any]]:
        payload = _load_json(self.focus_path)
        if not isinstance(payload, dict):
            return []
        rows = []
        updated_date = _parse_date(payload.get("updated_at") or payload.get("scan_date") or datetime.now())

        def add_entry(entry: Any, source: str, bucket: str = "") -> None:
            if not isinstance(entry, dict):
                return
            rows.append(
                {
                    "source": source,
                    "symbol": _normalize_symbol(entry.get("symbol")),
                    "side": _normalize_side(entry.get("side")),
                    "date": _parse_date(entry.get("scan_date") or entry.get("last_trade_date")) or updated_date,
                    "setup_family": entry.get("setup_family") or entry.get("family") or "focus",
                    "priority_bucket": entry.get("priority_bucket") or bucket,
                    "favorite_zone": entry.get("favorite_zone") or "",
                    "priority_score": _coerce_float(entry.get("priority_score") or entry.get("score")),
                    "retest": entry.get("retest_reference_level") or "",
                    "compression": bool(entry.get("compression_flag")),
                }
            )

        for entry in payload.get("favorites") or []:
            add_entry(entry, "focus_favorite", "favorite_setup")
        for entry in payload.get("near_favorite_zones") or []:
            add_entry(entry, "focus_near_zone", "near_favorite_zone")
        symbols = payload.get("symbols")
        if isinstance(symbols, dict):
            for entry in symbols.values():
                add_entry(entry, "focus_symbol")
        return rows

    def _load_avwap_signal_rows(self) -> list[dict[str, Any]]:
        rows = []
        for raw in _read_csv_rows(self.avwap_signals_path):
            rows.append(
                {
                    "source": "avwap_signal",
                    "symbol": _normalize_symbol(raw.get("symbol")),
                    "side": _normalize_side(raw.get("side")),
                    "date": _parse_date(raw.get("scan_date") or raw.get("trade_date") or raw.get("last_trade_date")),
                    "setup_family": raw.get("setup_family") or raw.get("family") or "avwap_signal",
                    "priority_bucket": raw.get("priority_bucket") or "",
                    "favorite_zone": raw.get("favorite_zone") or "",
                    "priority_score": _coerce_float(raw.get("priority_score") or raw.get("score")),
                    "retest": raw.get("retest_reference_level") or "",
                    "compression": str(raw.get("compression_flag") or "").lower() in {"1", "true", "yes"},
                }
            )
        return rows

    def _load_intraday_bounce_rows(self) -> list[dict[str, Any]]:
        rows = []
        for raw in _read_csv_rows(self.intraday_bounces_path):
            bounce_time = _parse_datetime(
                raw.get("time") or raw.get("timestamp") or raw.get("bounce_time") or raw.get("trade_date")
            )
            rows.append(
                {
                    "source": "intraday_bounce",
                    "symbol": _normalize_symbol(raw.get("symbol") or raw.get("ticker")),
                    "side": _normalize_side(raw.get("direction") or raw.get("side") or raw.get("watchlist_bias")),
                    "date": bounce_time.date() if bounce_time else _parse_date(raw.get("trade_date")),
                    "setup_family": raw.get("bounce_type") or raw.get("setup_family") or "intraday_bounce",
                    "priority_bucket": "intraday",
                    "favorite_zone": raw.get("level") or raw.get("levels") or "",
                    "priority_score": _coerce_float(raw.get("score")),
                    "retest": raw.get("level") or "",
                    "compression": False,
                }
            )
        return rows

    def suggest_for_trade(
        self,
        trade: dict[str, Any],
        corrections: list[dict[str, Any]] | None = None,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        symbol = _normalize_symbol(trade.get("symbol"))
        direction = _normalize_side(trade.get("direction"))
        trade_date = _parse_date(trade.get("opened_at") or trade.get("trade_date") or trade.get("closed_at"))
        if not symbol or trade_date is None:
            return []

        candidates: dict[str, dict[str, Any]] = {}
        for row in self.load_context_rows():
            if _normalize_symbol(row.get("symbol")) != symbol:
                continue
            context_date = row.get("date")
            if not isinstance(context_date, date):
                continue
            date_score = _date_distance_score(trade_date, context_date, self.lookback_calendar_days)
            if date_score is None:
                continue

            row_side = _normalize_side(row.get("side"))
            side_score = 0.16 if not row_side or not direction or row_side == direction else -0.10
            source = str(row.get("source") or "bot_context")
            source_score = {
                "setup_tracker": 0.28,
                "focus_favorite": 0.24,
                "focus_near_zone": 0.20,
                "focus_symbol": 0.12,
                "avwap_signal": 0.18,
                "intraday_bounce": 0.18,
            }.get(source, 0.08)
            score_value = _coerce_float(row.get("priority_score"))
            priority_score = min(0.14, max(0.0, (score_value or 0.0) / 1000.0))
            bucket_bonus = 0.08 if str(row.get("priority_bucket") or "") in {"favorite_setup", "near_favorite_zone"} else 0.0
            confidence = max(0.01, min(0.98, source_score + date_score + side_score + priority_score + bucket_bonus))
            tag = _priority_tag(row)
            current = candidates.get(tag)
            rationale = (
                f"{source}; {symbol}; context {context_date.isoformat()}; "
                f"{row.get('setup_family') or 'setup'}"
            )
            if current is None or confidence > float(current.get("confidence", 0.0) or 0.0):
                candidates[tag] = {
                    "tag": tag,
                    "confidence": confidence,
                    "source": source,
                    "rationale": rationale,
                }

        for correction in corrections or []:
            if _normalize_symbol(correction.get("symbol")) != symbol:
                continue
            tag = str(correction.get("setup_tag") or "").strip()
            if not tag:
                continue
            boost = _coerce_float(correction.get("confidence_boost")) or 0.12
            current = candidates.get(tag)
            if current:
                current["confidence"] = min(0.99, float(current["confidence"]) + boost)
                current["rationale"] = f"{current['rationale']}; manual correction boost"
            else:
                candidates[tag] = {
                    "tag": tag,
                    "confidence": min(0.80, 0.40 + boost),
                    "source": "manual_correction",
                    "rationale": "Historical manual correction for this symbol.",
                }

        ordered = sorted(
            candidates.values(),
            key=lambda item: (-float(item.get("confidence", 0.0) or 0.0), str(item.get("tag") or "")),
        )
        return ordered[: max(1, int(limit))]


def calendar_pnl_by_day(trades: list[dict[str, Any]], *, pnl_key: str = "net_pnl") -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for trade in trades:
        if str(trade.get("status") or "").upper() != "CLOSED":
            continue
        trade_day = _parse_date(trade.get("closed_at") or trade.get("trade_date") or trade.get("opened_at"))
        if trade_day is None:
            continue
        pnl = _coerce_float(trade.get(pnl_key))
        if pnl is None:
            continue
        totals[trade_day.isoformat()] += pnl
    return dict(totals)


def _first_setup_tag(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "untagged"
    for separator in (";", ",", "|"):
        if separator in text:
            first = text.split(separator, 1)[0].strip()
            return first or "untagged"
    return text


def _summary_for_rows(rows: list[dict[str, Any]], pnl_key: str = "net_pnl") -> dict[str, Any]:
    closed = [row for row in rows if str(row.get("status") or "").upper() == "CLOSED"]
    pnl_values = [_coerce_float(row.get(pnl_key)) or 0.0 for row in closed]
    wins = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    gross_win = sum(wins)
    gross_loss = sum(losses)
    profit_factor = (gross_win / abs(gross_loss)) if gross_loss < 0 else None
    return {
        "trades": len(rows),
        "closed": len(closed),
        "open": len(rows) - len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "profit_factor": profit_factor,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "net_pnl": sum(pnl_values),
        "avg_win": (gross_win / len(wins)) if wins else None,
        "avg_loss": (gross_loss / len(losses)) if losses else None,
    }


def build_analytics_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"overall": _summary_for_rows(trades), "groups": {}}
    group_specs = {
        "setup": lambda row: _first_setup_tag(row.get("setup_tags") or row.get("auto_tag_summary")),
        "account": lambda row: str(row.get("account_label") or row.get("account_number") or "unknown"),
        "broker": lambda row: str(row.get("broker") or "unknown"),
        "symbol": lambda row: str(row.get("symbol") or "unknown"),
        "direction": lambda row: str(row.get("direction") or "unknown"),
        "mid_term_regime": lambda row: str(row.get("mid_term_regime") or "unset"),
        "short_term_regime": lambda row: str(row.get("short_term_regime") or "unset"),
        "intraday_regime": lambda row: str(row.get("intraday_regime") or "unset"),
    }
    for group_name, key_fn in group_specs.items():
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in trades:
            buckets[key_fn(row)].append(row)
        rows = []
        for label, bucket_rows in buckets.items():
            item = _summary_for_rows(bucket_rows)
            item["label"] = label
            rows.append(item)
        rows.sort(key=lambda item: (-int(item.get("closed", 0)), -abs(float(item.get("net_pnl", 0.0) or 0.0)), str(item["label"])))
        summary["groups"][group_name] = rows
    return summary


def _fmt_money(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:,.2f}"


def _fmt_pct(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric * 100.0:.1f}%"


def _fmt_ratio(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.2f}"


def build_analytics_text(trades: list[dict[str, Any]]) -> str:
    summary = build_analytics_summary(trades)
    overall = summary["overall"]
    lines = [
        "Journal Analytics",
        "",
        (
            f"Closed={overall['closed']} Open={overall['open']} WR={_fmt_pct(overall['win_rate'])} "
            f"PF={_fmt_ratio(overall['profit_factor'])} Net={_fmt_money(overall['net_pnl'])} "
            f"GrossWin={_fmt_money(overall['gross_win'])} GrossLoss={_fmt_money(overall['gross_loss'])}"
        ),
        "",
    ]
    for group_name, rows in summary["groups"].items():
        lines.append(group_name.replace("_", " ").title())
        if not rows:
            lines.append("  None")
        for row in rows[:25]:
            lines.append(
                "  "
                f"{row['label']}: closed={row['closed']} WR={_fmt_pct(row['win_rate'])} "
                f"PF={_fmt_ratio(row['profit_factor'])} net={_fmt_money(row['net_pnl'])} "
                f"avgW={_fmt_money(row['avg_win'])} avgL={_fmt_money(row['avg_loss'])}"
            )
        lines.append("")
    return "\n".join(lines).strip()
