from __future__ import annotations

import csv
import json
import logging
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


#: Projected context rows, keyed by source path, holding the file stamp they
#: were built from. Bounded to one entry per source file by construction (four
#: of them), and each entry is the SMALL projection, never the parsed blob.
#:
#: Why it exists: `master_avwap_setup_tracker.json` measured 1.08 GB on
#: 2026-08-31 and `json.loads` of it runs behind the Corrections dialog's OK
#: button. Two retags in a row - accept a correction, add an execution - parsed
#: it twice for byte-identical input.
_CONTEXT_ROW_CACHE: dict[str, tuple[tuple[int, int], list[dict[str, Any]]]] = {}


def _file_stamp(path: Path) -> tuple[int, int] | None:
    """(mtime_ns, size), or None when the file cannot be stamped.

    None means "do not cache this" - an unreadable or missing source is not a
    fact worth remembering, and a later appearance must be picked up.
    """
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _cached_context_rows(path: Path, builder) -> list[dict[str, Any]]:
    """`builder()`'s projected rows, reused while the file has not changed."""
    key = str(Path(path))
    stamp = _file_stamp(path)
    if stamp is not None:
        cached = _CONTEXT_ROW_CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
    rows = builder()
    if stamp is not None:
        _CONTEXT_ROW_CACHE[key] = (stamp, rows)
    return rows


def clear_context_row_cache() -> None:
    """Forget every cached projection. For tests and for a forced re-read."""
    _CONTEXT_ROW_CACHE.clear()


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


#: ``auto_tag_candidates.source`` prefix for P6's EXACT-ID lane. The stored
#: value is ``trader_capture:<kind>`` - veto, like_claim, pass, or a take-class
#: review action - and it ranks ABOVE every fuzzy source because it is the
#: trader's own statement about that name on that day.
#:
#: Defined HERE and re-exported by `journal_store`, not the other way round:
#: `journal_store` already imports from this module, so the dependency runs one
#: way only.
TRADER_CAPTURE_SOURCE = "trader_capture"


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
        self._capture_rows: list[dict[str, Any]] | None = None

    def load_capture_rows(self) -> list[dict[str, Any]]:
        """The trader's OWN statements about a name, with their event ids.

        A separate list from `load_context_rows` because it is a different kind
        of evidence. Those are scanner rows a trade fell near; these are things
        the trader typed about that symbol on that day - a veto, a like+claim, a
        pass, or a chart they took action on. Matched by exact event id rather
        than by a symbol landing inside a 16-day window, which is why they rank
        above every fuzzy source.

        Read-only over two append-only stores. Any failure yields NOTHING
        rather than raising: the auto-tagger runs behind an OK button, and a
        suggestion source that cannot be read must cost its own suggestions and
        never the pane.
        """
        if self._capture_rows is not None:
            return self._capture_rows
        rows: list[dict[str, Any]] = []
        rows.extend(self._load_annotation_capture_rows())
        rows.extend(self._load_review_capture_rows())
        self._capture_rows = rows
        return rows

    def _load_annotation_capture_rows(self) -> list[dict[str, Any]]:
        """Vetoes, like+claims and passes from `trader_annotations.jsonl`.

        The tag each contributes is what the trader actually said:

        * a **like_claim** contributes its `claimed_setup_id` - they named the
          setup, so that IS the tag;
        * a **veto** contributes ``vetoed:<code>`` and a **pass**
          ``passed:<code>``, prefixed so a rejection can never be mistaken for
          an endorsement in a Tags column.

        The reason codes are carried verbatim; nothing here interprets or
        pools them.
        """
        try:
            from project_paths import TRADER_ANNOTATIONS_FILE
            from ui.annotations.store import (
                EVENT_LIKE_CLAIM,
                EVENT_PASS,
                EVENT_VETO,
                load_annotations,
            )

            annotations = load_annotations(
                Path(TRADER_ANNOTATIONS_FILE),
                event_types=(EVENT_VETO, EVENT_LIKE_CLAIM, EVENT_PASS),
            )
        except Exception:  # noqa: BLE001 - a suggestion source is never fatal
            logging.debug("Trader annotations unavailable to the auto-tagger.", exc_info=True)
            return []

        rows: list[dict[str, Any]] = []
        for annotation in annotations:
            symbol = _normalize_symbol(annotation.get("symbol"))
            session = _parse_date(annotation.get("session_date"))
            if not symbol or session is None:
                continue
            kind = str(annotation.get("event_type") or "")
            if kind == "like_claim":
                tag = str(annotation.get("claimed_setup_id") or "").strip() or "liked"
            elif kind == "veto":
                code = str(annotation.get("reason_code") or "").strip()
                tag = f"vetoed:{code}" if code else "vetoed"
            elif kind == "pass":
                codes = [
                    str(code or "").strip()
                    for code in (annotation.get("reason_codes") or [])
                    if str(code or "").strip()
                ]
                # ALL of them, in VOCABULARY order (R2). A pass is
                # multi-select and `codes[0]` threw the rest away, so a pass for
                # "extended from VWAP AND thin liquidity" reached the tagger as
                # the first reason alone - which is a different statement from
                # the one the trader made. The annotation writes its codes in
                # vocabulary order already (never click order), so preserving
                # the list preserves that too.
                tag = f"passed:{','.join(codes)}" if codes else "passed"
            else:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "date": session,
                    "side": _normalize_side(annotation.get("side")),
                    "kind": kind,
                    "tag": tag,
                    "event_id": str(annotation.get("event_id") or ""),
                    "detail": str(annotation.get("note") or ""),
                }
            )
        return rows

    def _load_review_capture_rows(self) -> list[dict[str, Any]]:
        """TAKE-class review events - the charts the trader acted on.

        The take set is `review_learning`'s, read rather than restated: it is
        the one place that decides what counts as the trader saying yes to a
        chart, and a second copy here would drift from it.
        """
        try:
            from review_events import load_review_events
            from review_learning import TAKE_ACTIONS, TOGGLE_TAKE_ACTIONS

            events = load_review_events()
        except Exception:  # noqa: BLE001
            logging.debug("Review events unavailable to the auto-tagger.", exc_info=True)
            return []

        rows: list[dict[str, Any]] = []
        for event in events:
            action = str(event.get("action") or "")
            if action in TOGGLE_TAKE_ACTIONS:
                detail = event.get("detail")
                if not (isinstance(detail, dict) and detail.get("on")):
                    continue
            elif action not in TAKE_ACTIONS:
                continue
            symbol = _normalize_symbol(event.get("symbol"))
            session = _parse_date(event.get("trade_date"))
            if not symbol or session is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "date": session,
                    "side": _normalize_side(event.get("side")),
                    "kind": f"review:{action}",
                    # A CHART HOUSEKEEPING ACTION IS A LINK, NOT A TAG (R1).
                    #
                    # `add_focus`, `arm_level`, `arm_watch` and the toggles say
                    # the trader did something WITH the chart. They say nothing
                    # about which setup it was, and 676 of 730 live rows carry
                    # no `bounce_types` at all - so this minted `took:add_focus`
                    # and, ranked first as a capture candidate, spent the
                    # four-slot summary on it. Measured on eight live trades:
                    # EYPT and SMPL lost `avwape_to_1stdev` from their Tags
                    # column to a housekeeping click.
                    #
                    # The row is still stored - it carries a `context_row_id`
                    # worth following - but it contributes NO tag text, so it
                    # can never evict a real setup match from the summary. Only
                    # a like_claim, a veto and a pass name a setup.
                    "tag": "",
                    "link_only": True,
                    # The alert's own id when it has one - only 54 of 730 take
                    # rows do - and otherwise the row's natural identity, its
                    # timestamp, PREFIXED so a reader knows which store to open.
                    # An empty pointer would look exactly like a fuzzy
                    # candidate, which is the one thing this lane is not.
                    "event_id": (
                        str(event.get("event_id") or "")
                        or f"review_event:{str(event.get('ts') or '').strip()}"
                    ),
                    "detail": str(event.get("tier") or ""),
                }
            )
        return rows

    def load_context_rows(self) -> list[dict[str, Any]]:
        """The scanner-output rows the tagger matches trades against.

        Cached per SOURCE FILE, not just per tagger: every one of these is a
        pure projection of a file, so two taggers built minutes apart over
        unchanged files must not parse them twice. The tracker file alone
        measured 1.08 GB on 2026-08-31 and this runs behind an OK button.
        """
        if self._context_rows is not None:
            return self._context_rows
        rows: list[dict[str, Any]] = []
        rows.extend(_cached_context_rows(self.setup_tracker_path, self._load_tracker_rows))
        rows.extend(_cached_context_rows(self.focus_path, self._load_focus_rows))
        rows.extend(_cached_context_rows(self.avwap_signals_path, self._load_avwap_signal_rows))
        rows.extend(
            _cached_context_rows(self.intraday_bounces_path, self._load_intraday_bounce_rows)
        )
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
        # The parsed blob is 1.08 GB and the projection above is a few MB.
        # Dropping the references here rather than at the return statement
        # means the tagging that follows never runs alongside both.
        del setups
        del payload
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

        # ---------------------------------------------------------- P6 -----
        # The EXACT-ID lane first: what the trader themselves said about this
        # symbol while the trade was open. Matched on the trade's OWN WINDOW -
        # open date to close date, not a 16-day neighbourhood - because an
        # event id is only worth carrying when the statement and the trade
        # really are about the same episode.
        #
        # A capture candidate carries `context_row_id`, which every surface
        # renders beside its confidence. It is a POINTER for a reader, never a
        # canonical link: plan.md P5.3/P5.4 own the canonical opportunity id.
        opened = trade_date
        closed = _parse_date(trade.get("closed_at")) or opened
        window_start, window_end = (opened, closed) if opened <= closed else (closed, opened)
        for row in self.load_capture_rows():
            if row.get("symbol") != symbol:
                continue
            said_on = row.get("date")
            if not isinstance(said_on, date):
                continue
            if not (window_start <= said_on <= window_end):
                continue
            row_side = row.get("side") or ""
            if row_side and direction and row_side != direction:
                # A long statement about a short trade is a different claim.
                continue
            link_only = bool(row.get("link_only"))
            tag = str(row.get("tag") or "").strip()
            if not tag and not link_only:
                continue
            if link_only:
                # A pointer, under a name that cannot be mistaken for a setup.
                # It is excluded from `auto_tag_summary` by the store, so it
                # occupies no slot in the Tags column.
                tag = f"{LINK_TAG_PREFIX}{row.get('kind')}"
            # A stated judgement inside the trade's own window is the strongest
            # thing this tagger has, and it is still a SUGGESTION: the trader
            # accepts or ignores it, and nothing here writes trade_annotations.
            confidence = 0.95 if row_side and direction else 0.90
            current = candidates.get(tag)
            if current is not None and float(current.get("confidence", 0.0) or 0.0) >= confidence:
                continue
            detail = str(row.get("detail") or "").strip()
            candidates[tag] = {
                "tag": tag,
                "confidence": confidence,
                "source": f"{TRADER_CAPTURE_SOURCE}:{row.get('kind')}",
                "context_row_id": str(row.get("event_id") or ""),
                # Read by `refresh_auto_tags`: a link is stored as a candidate
                # and kept out of the summary (R1).
                "link_only": link_only,
                "rationale": (
                    f"you said this on {said_on.isoformat()} ({row.get('kind')})"
                    + (f": {detail}" if detail else "")
                    + "; inside this trade's own window"
                ),
            }

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
            if str(current.get("source") or "").startswith(f"{TRADER_CAPTURE_SOURCE}:") if current else False:
                # A fuzzy match never displaces the trader's own statement,
                # whatever its computed confidence.
                continue
            if current is None or confidence > float(current.get("confidence", 0.0) or 0.0):
                candidates[tag] = {
                    "tag": tag,
                    "confidence": confidence,
                    "source": source,
                    "rationale": rationale,
                    "context_row_id": "",
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
                    "context_row_id": "",
                }

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                # The capture lane leads: the trader's own statement about this
                # name on this day outranks anything inferred about it.
                0 if str(item.get("source") or "").startswith(f"{TRADER_CAPTURE_SOURCE}:") else 1,
                -float(item.get("confidence", 0.0) or 0.0),
                str(item.get("tag") or ""),
            ),
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


#: The three tag lanes (P6a), named here rather than imported from
#: ``journal_store`` so this module keeps its one-way dependency: the store
#: imports the analytics helpers, not the other way round.
TAG_STATUS_CONFIRMED = "confirmed"
TAG_STATUS_PROVISIONAL = "provisional"


#: The prefix every link-only candidate's tag carries. A LINK records that the
#: trader did something WITH the chart - added it to Focus, armed a level - and
#: says nothing about which setup it was. It is stored, it renders, it carries a
#: `context_row_id` worth following, and it is NEVER a tag.
LINK_TAG_PREFIX = "link:"


def is_link_candidate(candidate: Any) -> bool:
    """ONE predicate for "this is a pointer, not a tag" (R2).

    R1 kept links out of `auto_tag_summary` and three other seams still let them
    through: the bulk tagger's lane filter, its `max(confidence)` pick, the
    Accept-all button, and `tag_confidence`. Each had its own idea of what a
    link was - or no idea at all - so the rule held in one place and leaked in
    four.

    Both spellings are accepted deliberately. `link_only` is what the tagger
    sets in memory; the PREFIX is what survives a round trip through
    `auto_tag_candidates`, which stores a tag and a source but no flag. A reader
    that only knew the flag would be right until the row came back from the
    database.
    """
    if isinstance(candidate, str):
        return candidate.startswith(LINK_TAG_PREFIX)
    if not hasattr(candidate, "get"):
        return False
    if candidate.get("link_only"):
        return True
    return str(candidate.get("tag") or "").startswith(LINK_TAG_PREFIX)


def _confirmed_setup_tags(row: dict[str, Any]) -> list[str]:
    """The tags on this trade that the TRADER stands behind (P6a).

    A provisional tag lives in the same column, so grouping on ``setup_tags``
    alone would fold 100+ machine guesses into "my setups" - the one group in
    the journal that is supposed to answer what the trader themself said this
    trade was. A row with no annotation at all reports ``confirmed`` and has no
    tags, so it lands in ``untagged`` exactly as before.
    """
    if str(row.get("tag_status") or TAG_STATUS_CONFIRMED) != TAG_STATUS_CONFIRMED:
        return []
    return _tags_for_row(row, "setup_tags")


def _provisional_setup_tags(row: dict[str, Any]) -> list[str]:
    """The tags a machine applied and nobody has reviewed yet (P6a)."""
    if str(row.get("tag_status") or TAG_STATUS_CONFIRMED) != TAG_STATUS_PROVISIONAL:
        return []
    return _tags_for_row(row, "setup_tags")


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
        # CONFIRMED only. The provisional lane is its own group below and the
        # two are never blended: a per-setup win rate that mixes what the trader
        # said with what a machine guessed is not a statement about either.
        "my setups": lambda row: _confirmed_setup_tags(row) or ["untagged"],
        # No "untagged" fallback here on purpose: a trade with no provisional
        # tag belongs in no bucket of this group at all, and a catch-all bucket
        # holding every other trade would be the biggest bar on the chart while
        # meaning nothing.
        "provisional setups": _provisional_setup_tags,
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
    summary["group_notes"] = _empty_dimension_notes(trades, summary["groups"])
    summary["nonexclusive_groups"] = ["my setups", "provisional setups", "auto tags"]
    #: Groups whose buckets are machine-applied and awaiting review. The chart
    #: says so out loud - a bar chart of "provisional setups" beside one of "my
    #: setups" is otherwise two answers to the same question with nothing to
    #: separate them.
    summary["provisional_groups"] = ["provisional setups"]
    return summary


#: Below this share of closed trades, a confirmed-tag dimension is not a
#: breakdown of the trader's setups - it is a breakdown of the handful they
#: happened to tag. Live on 2026-09-01: ONE confirmed tag across 193 trades.
CONFIRMED_TAG_COVERAGE_FLOOR = 0.10


def _empty_dimension_notes(
    trades: list[dict[str, Any]], groups: dict[str, list[dict[str, Any]]]
) -> dict[str, str]:
    """One sentence per group whose coverage is too thin to read as a chart.

    "My setups" renders beside a full "auto tags" chart, so two charts of the
    same width sit side by side while one of them rests on a single trade. The
    reader is not told; they see a bar and read it as a finding.

    THE GROUP IS NEVER HIDDEN. Hiding it would replace a visible thin answer
    with an invisible one, and the whole point is that the trader can see how
    little they have tagged - that is the prompt to tag more. The note is
    PREPENDED to the group's own label, using the same refusal-message
    mechanism `resolve_pnl_key` already uses to explain a total it will not
    compute.

    Coverage is measured against CLOSED trades, which is the denominator every
    number in these groups is computed over.
    """
    closed = [row for row in trades if str(row.get("status") or "").upper() == "CLOSED"]
    if not closed:
        return {}
    notes: dict[str, str] = {}
    for group_name in ("my setups",):
        # COUNTED OVER TRADES, NOT OVER BUCKETS (R1).
        #
        # "My setups" is NON-EXCLUSIVE: a trade carrying three tags appears in
        # three buckets, so summing each bucket's `closed` counted it three
        # times. Live, 24 tagged trades of 156 measured as 40% coverage and the
        # note therefore never appeared - the one honesty this note exists to
        # provide was suppressed by its own arithmetic.
        #
        # And it ignored `tag_status` (P6a), so a machine-applied PROVISIONAL
        # tag counted as the trader's. Coverage of confirmed tags means exactly
        # that: distinct closed trades whose tags the trader stands behind.
        tagged = sum(
            1
            for row in closed
            if _confirmed_setup_tags(row)
        )
        share = tagged / len(closed)
        if share >= CONFIRMED_TAG_COVERAGE_FLOOR:
            continue
        notes[group_name] = (
            f"ONLY {tagged} OF {len(closed)} CLOSED TRADES CARRY A CONFIRMED TAG "
            f"({share * 100:.0f}%). This is a breakdown of those few, not of your "
            "setups - read it as a prompt to tag more, never as a ranking. The "
            "auto-tag chart beside it covers every trade and is the one to read "
            "until this catches up."
        )
    return notes


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
