from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date, datetime
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


#: Column written by :func:`apply_manual_usd_estimate`. Named "estimated" on
#: purpose: it must never be mistaken for a booked value in a log or a CSV.
USD_ESTIMATE_KEY = "net_pnl_usd_estimated"

#: Column booked by ``JournalStore.book_currency_values`` from the stored BoC
#: observation for each trade's OWN session (2026-08-24). Preferred over the
#: manual estimate wherever every selected row carries it - one is a
#: measurement, the other is one rate applied to a year.
USD_BOOKED_KEY = "net_pnl_usd"


def apply_manual_usd_estimate(
    trades: list[dict[str, Any]], rate: float | None = None
) -> tuple[float, list[dict[str, Any]]] | None:
    """Annotate rows with an estimated USD P&L. Returns (rate, unconverted).

    ``None`` when no manual rate is set, which leaves every existing refusal
    exactly as it was. A USD-native row passes its own value through untouched;
    anything else divides the BOOKED CAD value by the entered rate, so the
    estimate inherits the booked path's honesty about what it could not convert.
    """
    if rate is None:
        from journal_fx import manual_usd_rate

        stored = manual_usd_rate()
        if not stored:
            return None
        rate = float(stored["rate_cad_per_usd"])
    if not rate:
        return None

    unconverted: list[dict[str, Any]] = []
    for row in trades:
        if str(row.get("currency") or "").upper() == "USD":
            row[USD_ESTIMATE_KEY] = row.get("net_pnl")
            continue
        cad = row.get("net_pnl_cad")
        if cad is None:
            row[USD_ESTIMATE_KEY] = None
            unconverted.append(row)
            continue
        row[USD_ESTIMATE_KEY] = float(cad) / rate
    return float(rate), unconverted


def resolve_pnl_key(
    trades: list[dict[str, Any]], currency_mode: str | None = None
) -> tuple[str, str]:
    """Which P&L column may be summed, and what to tell the reader.

    Root cause B8. ``_summary_for_rows`` defaulted to ``net_pnl``, which is the
    trade's **native** currency, and then added a USD win to a CAD loss as if
    they were the same number. For a Canadian trader filing Canadian tax that is
    not a rounding error, it is a wrong total.

    Three honest outcomes, and no fourth:

    * one currency across the whole selection - sum ``net_pnl``, it means
      something;
    * mixed currencies and every trade converted - sum ``net_pnl_cad``;
    * mixed currencies with anything unconverted - **refuse**. The caller gets
      ``("", reason)`` and shows the reason instead of a number, because a total
      that silently omits the unconverted rows is worse than no total.
    """
    closed = [row for row in trades if str(row.get("status") or "").upper() == "CLOSED"]
    mode = str(currency_mode or "").strip().upper()
    currencies = {str(row.get("currency") or "").upper() for row in closed if row.get("currency")}
    if mode == "CAD":
        unconverted = [row for row in closed if row.get("net_pnl_cad") is None]
        if unconverted:
            missing = sorted({str(row.get("currency") or "?").upper() for row in unconverted})
            return "", (
                f"{len(unconverted)} of {len(closed)} trades have no booked FX rate "
                f"({', '.join(missing)}); CAD totals are not shown"
            )
        return "net_pnl_cad", "converted to CAD at each trade's booked rate"
    if mode == "USD":
        non_usd = [row for row in closed if str(row.get("currency") or "").upper() != "USD"]
        if not non_usd:
            return "net_pnl", ""
        # True conversion first (2026-08-24). Every row carries a USD value
        # booked at import from the BoC observation for its own session, so this
        # is a measurement rather than an approximation - and it is preferred
        # over the manual rate whenever it can answer for the WHOLE selection.
        # Partially booked is not good enough: summing booked rows and estimated
        # rows in one total would produce a number that is neither.
        unbooked = [row for row in closed if row.get(USD_BOOKED_KEY) is None]
        if not unbooked:
            return USD_BOOKED_KEY, (
                "converted to USD at each trade's booked Bank of Canada rate for "
                "its own session"
            )
        # A manually entered display rate is the ONLY way a mixed selection
        # gets a USD total, and it is an estimate, not a booked figure. It
        # converts from the booked CAD value, so a row the booked path could
        # not convert stays unconvertible here too - a manual rate buys an
        # approximation, never a missing observation.
        estimate = apply_manual_usd_estimate(closed)
        if estimate is not None:
            rate, unconverted = estimate
            if unconverted:
                missing = sorted({str(row.get("currency") or "?").upper() for row in unconverted})
                return "", (
                    f"{len(unconverted)} of {len(closed)} trades have no booked FX rate "
                    f"({', '.join(missing)}); USD totals are not shown"
                )
            return USD_ESTIMATE_KEY, (
                f"ESTIMATE - non-USD trades converted at a manually entered "
                f"{rate:.4f} CAD/USD, not each trade's booked rate. Not a tax figure."
            )
        missing = sorted({str(row.get("currency") or "?").upper() for row in unbooked})
        return "", (
            f"{len(unbooked)} of {len(closed)} trades have no booked USD rate for "
            f"their session ({', '.join(missing)}); USD totals are not shown. Enter "
            f"a USD/CAD rate in the Journal header for an estimate."
        )
    # Native mode (and legacy callers with no explicit mode) can add values only
    # when the selection has one currency. Legacy mixed selections retain the
    # tax-grade CAD fallback used by non-UI reports.
    if mode == "NATIVE" and len(currencies) > 1:
        return "", "multiple native currencies selected; Native totals are not shown"
    if len(currencies) <= 1:
        return "net_pnl", ""
    unconverted = [row for row in closed if row.get("net_pnl_cad") is None]
    if unconverted:
        missing = sorted({str(row.get("currency") or "?").upper() for row in unconverted})
        return "", (
            f"{len(unconverted)} of {len(trades)} trades have no booked FX rate "
            f"({', '.join(missing)}); totals across currencies are not shown"
        )
    return "net_pnl_cad", "converted to CAD at each trade's booked rate"


def split_tags(value: Any) -> list[str]:
    """Split one stored tag string into its tags.

    The first separator present wins, in the order ``;`` ``,`` ``|``, rather
    than splitting on all three. That matters because ``_priority_tag`` builds
    a setup tag as ``"family | bucket | zone"`` -- pipes are INSIDE a tag, and
    only a string with no ``;`` or ``,`` at all is treated as pipe-separated.

    Named and exported because the store, the tag list and the rename tool all
    need this exact rule; a second copy anywhere would eventually disagree
    about what one tag is.
    """
    text = str(value or "").strip()
    if not text:
        return []
    for separator in (";", ",", "|"):
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def _tags_for_row(row: dict[str, Any], field: str = "setup_tags") -> list[str]:
    """Every setup tag on a trade, not just the first one.

    ``_first_setup_tag`` kept only the leading tag, so a trade tagged
    "avwap-reclaim; earnings-gap" counted entirely towards the first and not at
    all towards the second - which quietly understated every setup that tends to
    be named second.
    """
    return split_tags(row.get(field))


def build_analytics_summary(
    trades: list[dict[str, Any]], currency_mode: str | None = None
) -> dict[str, Any]:
    pnl_key, pnl_note = resolve_pnl_key(trades, currency_mode)
    summary = {
        "overall": _summary_for_rows(trades, pnl_key or "net_pnl"),
        "groups": {},
        "pnl_key": pnl_key,
        "pnl_note": pnl_note,
        "currencies": sorted({str(row.get("currency") or "").upper() for row in trades if row.get("currency")}),
    }
    if not pnl_key:
        # Mixed currencies with unconverted rows: the per-group totals would be
        # as meaningless as the overall one, so say why and stop.
        summary["overall"] = {**summary["overall"], "net_pnl": None, "gross_win": None, "gross_loss": None}
    group_specs = {
        "my setups": lambda row: _tags_for_row(row, "setup_tags") or ["untagged"],
        "auto tags": lambda row: _tags_for_row(row, "auto_tag_summary") or ["untagged"],
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
            keys = key_fn(row)
            if not isinstance(keys, list):
                keys = [keys]
            for key in dict.fromkeys(keys):
                buckets[str(key)].append(row)
        rows = []
        for label, bucket_rows in buckets.items():
            # The same column the overall total used. A per-group breakdown that
            # summed native P&L under a CAD headline would disagree with the
            # number above it, which is B8 back again one row down.
            item = _summary_for_rows(bucket_rows, pnl_key or "net_pnl")
            if not pnl_key:
                item = {**item, "net_pnl": None, "gross_win": None, "gross_loss": None}
            item["label"] = label
            rows.append(item)
        rows.sort(
            key=lambda item: (
                -int(item.get("closed", 0)),
                -abs(float(item.get("net_pnl") or 0.0)),
                str(item["label"]),
            )
        )
        summary["groups"][group_name] = rows
    summary["nonexclusive_groups"] = ["my setups", "auto tags"]
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
