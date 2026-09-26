"""Setup permutation key on M5 alerts (WISHLIST P1-4 4a): a shadow sidecar.

The M5 outcome writer (`bounce_bot_lib.legacy`) calls `submit(row)` once per
outcome row, after the row is on disk. The first row of each ``event_id`` is
queued; one daemon worker looks up the PREVIOUS session's last D1 scan row for
the name and side (the backfill's rule, `setup_permutation_backfill`) and
appends one jsonl record to `project_paths.M5_SETUP_KEY_STAMPS_FILE`.

Shadow only: nothing here reads or changes the outcome row, and nothing reads
the stamp for an alert, a grade or a score. `submit` never raises and never
waits; a full queue or a failed lookup loses the stamp, never the row. With no
scan row for the name, or no stamp record at all, the key is unknown.

A scan run on the alert's own day (before the alert) can write rows dated the
previous session (the pre-market and after-close scans use the last completed
bar); those rows count, as they do in the backfill, because they were on disk
before the alert and hold only completed bars.

Memory: only the previous session's lines are kept (raw), only the columns the
rule needs are split out, and only each representative row is parsed whole,
one at a time. Review events are streamed for that session only.

Owner: this module is the only writer of the sidecar (append-only).

P11: each record also carries an M5 key (`setup_permutations.m5_facets_for`)
over what the alert carried then: its registered row (entry time, session
RVOL, bounce type), the bot's cached completed M5 bars up to the alert bar
(VWAP distance in M5 ATR14; `register_bar_source`), and the SPY market-state
shadow log at that bar. Never a bar after the alert bar; a missing input is
unknown. The D1 part and the M5 part stand alone: either can be unknown.

S6: the same cached bars also give the structure inputs (`structure_inputs`:
EMA 8/21, previous day's range, first-30-minute range, 12-bar squeeze), and
SPY's D1 environment label for the session before the alert is read from
`d1_environment_store`.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import weakref
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA = "m5_setup_key_stamp_v1"
STATUS_STAMPED = "stamped"
STATUS_NO_SCAN_ROW = "no_scan_row"
STATUS_FAILED = "lookup_failed"
STAMP_FIELDS = ("permutation_key", "permutation_label", "permutation_rule_version")
#: Rows waiting for the worker; past this the stamp is dropped, never the caller delayed.
QUEUE_MAX = 5000
#: How far back from the end of the D1 history the previous session may lie.
MAX_TAIL_BYTES = 256 << 20
#: A failed load is not retried for this long (a persistent failure never re-reads per alert).
FAILURE_BACKOFF_SECONDS = 300.0
#: The columns the backfill's representative rule reads (`representatives_of`, `scan_row_id`).
_RULE_COLUMNS = ("symbol", "side", "last_trade_date", "run_date", "last_close", "run_timestamp", "run_id")
#: A registered row's context_json past this size is not copied (the M5 part then reads no RVOL).
MAX_CONTEXT_CHARS = 64_000
#: Set to "0" to switch the hook off (tests, a bad day).
ENABLED_ENV = "TRADINGBOTV3_M5_SETUP_KEY_STAMP"

_lock = threading.Lock()
_queue: "queue.Queue[dict]" = queue.Queue(maxsize=QUEUE_MAX)
_worker: threading.Thread | None = None
_seen: set[str] = set()
_logged_reasons: set[str] = set()


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _side(direction: Any) -> str:
    text = _text(direction).upper()
    return text if text in {"LONG", "SHORT"} else ""


def enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "1").strip() != "0"


def _log_once(reason: str) -> None:
    """One warning per distinct reason per process: a broken lookup never floods the log."""
    with _lock:
        if reason in _logged_reasons:
            return
        _logged_reasons.add(reason)
    logging.warning("M5 setup key stamp skipped: %s", reason)


# --- the lookup (worker thread only)


class ScanKeyLookup:
    """Keys for one previous session, cached until the D1 history file changes.

    ``history_path`` is the append-ordered `d1_features_history.csv`; only its
    tail back to the previous session is read. ``context_loader(session)``
    gives the `SessionContext` the backfill would use for that session.
    """

    def __init__(
        self,
        history_path: Path | None = None,
        *,
        context_loader: Callable[[str], Any] | None = None,
        max_tail_bytes: int = MAX_TAIL_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._history_path = history_path
        self._context_loader = context_loader
        self._max_tail_bytes = max_tail_bytes
        self._clock = clock
        self._cache: dict[str, tuple[Any, dict[tuple[str, str], dict]]] = {}
        self._failure: tuple[str, float, Exception] | None = None
        self.loads = 0

    def history_path(self) -> Path:
        if self._history_path is not None:
            return Path(self._history_path)
        from project_paths import D1_FEATURES_HISTORY_FILE

        return Path(D1_FEATURES_HISTORY_FILE)

    def _context(self, session: str):
        if self._context_loader is not None:
            return self._context_loader(session)
        import setup_permutation_context as spc

        return spc.SessionContext.load(session, stream_review_events=True)

    def keys_for_session(self, session: str) -> dict[tuple[str, str], dict]:
        """``{(SYMBOL, SIDE): stamp}`` for every representative row of ``session``. Raises on failure."""

        path = self.history_path()
        stat = path.stat()
        signature = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = self._cache.get(session)
        if cached is not None and cached[0] == signature:
            return cached[1]
        failure = self._failure
        if failure is not None and failure[0] == session and self._clock() - failure[1] < FAILURE_BACKOFF_SECONDS:
            raise failure[2]
        try:
            keys = self._load(path, session)
        except FileNotFoundError:
            raise
        except Exception as exc:
            self._failure = (session, self._clock(), exc)
            raise
        self._failure = None
        self._cache = {session: (signature, keys)}  # one session at a time
        self.loads += 1
        return keys

    def _load(self, path: Path, session: str) -> dict[tuple[str, str], dict]:
        import csv
        import io

        import setup_permutation_backfill as bf
        import setup_permutation_context as spc

        fieldnames, lines = spc._tail_lines_for_session(
            path, session, date_columns=("last_trade_date", "run_date"),
            stamp_columns=("run_timestamp", "run_id"), max_bytes=self._max_tail_bytes,
        )
        wanted = [(name, fieldnames.index(name)) for name in _RULE_COLUMNS if name in fieldnames]

        def parse(raw: bytes) -> list[str]:
            return next(csv.reader(io.StringIO(raw.decode("utf-8", errors="replace"))), [])

        def slim(values: list[str]) -> dict[str, str]:
            return {name: values[index] if index < len(values) else "" for name, index in wanted}

        representatives = bf.representatives_of(slim(parse(raw)) for raw in lines)
        context = self._context(session)
        keys: dict[tuple[str, str], dict] = {}
        for index, (symbol, side, rep_session) in representatives.items():
            if rep_session != session:
                continue
            row = dict(zip(fieldnames, parse(lines[index]), strict=False))
            key = bf.key_scan_row(row, symbol, side, session, context)
            keys[(symbol, side)] = {
                "permutation_key": key.compact_key,
                "permutation_label": key.label,
                "permutation_rule_version": key.permutation_rule_version,
                "d1_family": key.family,
                "facets": key.as_dict(),
                "scan_row_id": bf.scan_row_id(row),
            }
        return keys

    def stamp(self, symbol: str, side: str, trade_date: str) -> dict:
        """The sidecar fields for one alert. Never raises: a failure is a blank stamp with its reason."""
        import setup_permutation_backfill as bf

        previous = bf.previous_session_text(trade_date)
        out: dict[str, Any] = {"d1_session": previous, **{name: "" for name in STAMP_FIELDS}}
        if not previous or not symbol or not side:
            out.update(status=STATUS_NO_SCAN_ROW, reason="no previous session, symbol or side")
            return out
        try:
            found = self.keys_for_session(previous).get((symbol, side))
        except FileNotFoundError:
            out.update(status=STATUS_NO_SCAN_ROW, reason="no D1 history file")
            return out
        except Exception as exc:  # noqa: BLE001 - a lookup failure costs the stamp, never the row
            reason = f"{type(exc).__name__}: {exc}"
            _log_once(reason)
            out.update(status=STATUS_FAILED, reason=reason)
            return out
        if found is None:
            out.update(status=STATUS_NO_SCAN_ROW, reason="no scan row for the name and side")
            return out
        out.update(found)
        out["status"] = STATUS_STAMPED
        return out


# --- P11: the M5 part (worker thread only)

M5_BAR_MINUTES = 5
M5_ATR_LENGTH = 14
#: Only this much of the SPY shadow log's tail is read (it only grows on state changes).
SPY_LOG_MAX_BYTES = 8 << 20
_SPY_STATE_SCHEMA_PREFIX = "spy_state_shadow"
_EXCHANGE_TZ = "America/New_York"

_bar_source: Callable[[], Any] | None = None


def register_bar_source(bot: Any) -> None:
    """Point the M5 part at a live bot's cached bars (`m5_chart_bars`; cache only, never IB).

    Held by weak reference, so a retired bot is never kept alive; ``None`` clears it.
    """
    global _bar_source
    if bot is None:
        _bar_source = None
        return
    try:
        _bar_source = weakref.ref(bot)
    except TypeError:
        _bar_source = lambda bot=bot: bot  # noqa: E731 - an object without weakref support


def _cached_bars(symbol: str) -> list | None:
    source = _bar_source
    bot = source() if source is not None else None
    reader = getattr(bot, "m5_chart_bars", None) if bot is not None else None
    if reader is None:
        return None
    try:
        return list(reader(symbol, max_sessions=2) or [])
    except Exception as exc:  # noqa: BLE001 - a bar read costs the facet, never the record
        _log_once(f"M5 bars: {type(exc).__name__}: {exc}")
        return None


def _local_tz() -> tzinfo:
    """The zone the bot's naive bar and entry stamps are written in."""
    from market_session import get_market_local_timezone

    return get_market_local_timezone()[0]


def _spy_log_path() -> Path:
    from market_state_bridge import shadow_log_path

    return Path(shadow_log_path())


def _naive(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo is None else None
    text = _text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is None else None


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number or number in (float("inf"), float("-inf")) else number


def _bounce_type(row: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    from setup_scoreboard import bounce_type_from_event_id

    return bounce_type_from_event_id(_text(row.get("event_id"))) or _text(context.get("family"))


def alert_bar_close(row: Mapping[str, Any]) -> datetime | None:
    """The alert bar's CLOSE, naive in the bot's local zone.

    The bot writes an M5 alert's ``entry_time`` as the bar START (``current_candle["time"]``),
    so its close is five minutes later; only the H1 families (``h1_`` prefix) stamp the close.
    """
    from evidence_rules import H1_FAMILY_PREFIX

    entry = _naive(row.get("entry_time"))
    if entry is None:
        return None
    if _bounce_type(row, {}).lower().startswith(H1_FAMILY_PREFIX):
        return entry
    return entry + timedelta(minutes=M5_BAR_MINUTES)


def alert_bar_complete(row: Mapping[str, Any], close: datetime | None, tz: tzinfo) -> bool | None:
    """Had the alert bar closed when the row was logged? None when ``logged_at`` cannot say."""
    if close is None:
        return None
    try:
        logged = datetime.fromisoformat(_text(row.get("logged_at")))
    except ValueError:
        return None
    if logged.tzinfo is None:
        return None
    return close.replace(tzinfo=tz) <= logged


def alert_inputs(row: Mapping[str, Any], tz: tzinfo | None = None) -> dict[str, Any]:
    """The M5 facet inputs one registered outcome row carried: bar close, session RVOL, bounce type.

    ``alert_bar_close`` (see `alert_bar_close`) is written in exchange time
    (America/New_York); ``alert_bar_complete`` says whether that bar had closed
    by ``logged_at``. The bounce type is the event id's, the same one the M5
    population groups by.
    """
    from zoneinfo import ZoneInfo

    zone = tz if tz is not None else _local_tz()
    close = alert_bar_close(row)
    context: Any = {}
    raw = row.get("context_json")
    if isinstance(raw, str) and raw.strip():
        try:
            context = json.loads(raw)
        except ValueError:
            context = {}
    if not isinstance(context, Mapping):
        context = {}
    return {
        "alert_bar_close": (
            close.replace(tzinfo=zone).astimezone(ZoneInfo(_EXCHANGE_TZ)).isoformat() if close is not None else ""
        ),
        "alert_bar_complete": alert_bar_complete(row, close, zone),
        "session_rvol": _float(context.get("session_rvol")),
        "bounce_type": _bounce_type(row, context),
        "alert_date": _text(row.get("trade_date"))[:10],
    }


def _bar_fields(bar: Any) -> tuple[datetime, float, float, float, float] | None:
    get = bar.get if isinstance(bar, Mapping) else lambda name, default=None: getattr(bar, name, default)
    when = _naive(get("dt"))
    values = [_float(get(name)) for name in ("high", "low", "close", "volume")]
    if when is None or any(value is None for value in values[:3]):
        return None
    high, low, close, volume = values
    return when, high, low, close, volume or 0.0


def vwap_distance_atr(bars: Any, bar_close: datetime, tz: tzinfo) -> float | None:
    """(alert-bar close - session VWAP) / M5 ATR14 at the alert bar; None when the bars cannot say.

    ``bars`` are naive local-time M5 bars stamped at their START; ``bar_close``
    is the alert bar's CLOSE in the same zone (`alert_bar_close`). Only bars that
    closed at or before it are read, and the last of them must be the alert bar.
    """
    from zoneinfo import ZoneInfo

    step = timedelta(minutes=M5_BAR_MINUTES)
    kept = []
    for bar in bars or ():
        fields = _bar_fields(bar)
        if fields is not None and fields[0] + step <= bar_close:
            kept.append(fields)
    kept.sort(key=lambda item: item[0])
    if not kept or kept[-1][0] + step != bar_close or len(kept) < M5_ATR_LENGTH + 1:
        return None
    trs = [
        max(high - low, abs(high - kept[index - 1][3]), abs(low - kept[index - 1][3]))
        for index, (_when, high, low, _close, _volume) in enumerate(kept) if index > 0
    ]
    atr = sum(trs[-M5_ATR_LENGTH:]) / M5_ATR_LENGTH
    exchange = ZoneInfo(_EXCHANGE_TZ)
    session_day = bar_close.date()
    volume_sum = weighted = 0.0
    for when, high, low, close, volume in kept:
        start = when.replace(tzinfo=tz).astimezone(exchange)
        minutes = start.hour * 60 + start.minute
        if when.date() != session_day or not 9 * 60 + 30 <= minutes < 16 * 60:
            continue  # the regular session only
        volume_sum += volume
        weighted += (high + low + close) / 3.0 * volume
    if atr <= 0 or volume_sum <= 0:
        return None
    return round((kept[-1][3] - weighted / volume_sum) / atr, 6)


# --- S6: M5 structure inputs over the same cached bars (regular session, completed bars only)

M5_EMA_FAST = 8
M5_EMA_SLOW = 21
#: Regular-session M5 bars in a full session (09:30-16:00 ET); a short or holed day is unknown.
M5_SESSION_BARS = 78
M5_OPEN_RANGE_BARS = 6
M5_COMPRESSION_BARS = 12
M5_COMPRESSION_ATR_BARS = 20
_REGULAR_OPEN_MINUTES = 9 * 60 + 30
_REGULAR_CLOSE_MINUTES = 16 * 60

STRUCTURE_INPUT_FIELDS = (
    "alert_price", "m5_ema8", "m5_ema21", "prev_day_high", "prev_day_low", "open_range_high",
    "open_range_low", "m5_range12_atr20", "m5_range12_break",
)


def _ema_last(values: list[float], length: int) -> float | None:
    """EMA of ``values`` at the last value, seeded with the SMA of the first ``length``; None if under 2x."""
    if len(values) < 2 * length:
        return None
    ema = sum(values[:length]) / length
    alpha = 2.0 / (length + 1)
    for value in values[length:]:
        ema += alpha * (value - ema)
    return ema


def structure_inputs(bars: Any, bar_close: datetime, tz: tzinfo) -> dict[str, Any]:
    """The S6 M5 structure inputs at the alert bar; None for any input the bars cannot say.

    ``bars`` are naive local-time M5 bars stamped at their START and ``bar_close`` is the
    alert bar's CLOSE in the same zone. Only bars that closed at or before it are read,
    the last of them must be the alert bar, and it must be a regular-session bar. Only
    regular-session bars (09:30-16:00 ET) count. EMA 8/21 over those bars (2x warm-up),
    the previous full session's high/low, the first-30-minute range (only when the alert
    bar starts at or after 10:00 ET), and the 12 bars before the alert bar (same session)
    as a range in ATR20 plus where the alert bar closed against that range.
    """
    from zoneinfo import ZoneInfo

    out: dict[str, Any] = dict.fromkeys(STRUCTURE_INPUT_FIELDS)
    step = timedelta(minutes=M5_BAR_MINUTES)
    exchange = ZoneInfo(_EXCHANGE_TZ)
    kept = []
    for bar in bars or ():
        fields = _bar_fields(bar)
        if fields is not None and fields[0] + step <= bar_close:
            kept.append(fields)
    kept.sort(key=lambda item: item[0])
    if not kept or kept[-1][0] + step != bar_close:
        return out
    regular = []  # (et date, minutes after midnight ET, high, low, close)
    for when, high, low, close, _volume in kept:
        start = when.replace(tzinfo=tz).astimezone(exchange)
        minutes = start.hour * 60 + start.minute
        if _REGULAR_OPEN_MINUTES <= minutes < _REGULAR_CLOSE_MINUTES:
            regular.append((start.date(), minutes, high, low, close))
    alert_start = (bar_close - step).replace(tzinfo=tz).astimezone(exchange)
    alert_minutes = alert_start.hour * 60 + alert_start.minute
    if not regular or regular[-1][:2] != (alert_start.date(), alert_minutes):
        return out  # the alert bar is not a regular-session bar
    session_day = alert_start.date()
    price = regular[-1][4]
    out["alert_price"] = price

    closes = [close for _day, _minutes, _high, _low, close in regular]
    fast, slow = _ema_last(closes, M5_EMA_FAST), _ema_last(closes, M5_EMA_SLOW)
    if fast is not None and slow is not None:
        out["m5_ema8"], out["m5_ema21"] = round(fast, 6), round(slow, 6)

    earlier_days = sorted({day for day, *_rest in regular if day < session_day})
    if earlier_days:
        previous = [bar for bar in regular if bar[0] == earlier_days[-1]]
        minutes = [bar[1] for bar in previous]
        full_day = list(range(_REGULAR_OPEN_MINUTES, _REGULAR_CLOSE_MINUTES, M5_BAR_MINUTES))
        if len(previous) == M5_SESSION_BARS and minutes == full_day:
            out["prev_day_high"] = max(bar[2] for bar in previous)
            out["prev_day_low"] = min(bar[3] for bar in previous)

    today = [bar for bar in regular if bar[0] == session_day]
    opening = [bar for bar in today if bar[1] < _REGULAR_OPEN_MINUTES + M5_OPEN_RANGE_BARS * M5_BAR_MINUTES]
    opening_minutes = [bar[1] for bar in opening]
    if alert_minutes >= 10 * 60 and opening_minutes == [
        _REGULAR_OPEN_MINUTES + index * M5_BAR_MINUTES for index in range(M5_OPEN_RANGE_BARS)
    ]:
        out["open_range_high"] = max(bar[2] for bar in opening)
        out["open_range_low"] = min(bar[3] for bar in opening)

    before = regular[:-1]
    box = before[-M5_COMPRESSION_BARS:]
    if (len(box) == M5_COMPRESSION_BARS and all(bar[0] == session_day for bar in box)
            and len(before) >= M5_COMPRESSION_ATR_BARS + 1):
        trs = [
            max(high - low, abs(high - before[index - 1][4]), abs(low - before[index - 1][4]))
            for index, (_day, _minutes, high, low, _close) in enumerate(before) if index > 0
        ]
        atr = sum(trs[-M5_COMPRESSION_ATR_BARS:]) / M5_COMPRESSION_ATR_BARS
        box_high = max(bar[2] for bar in box)
        box_low = min(bar[3] for bar in box)
        if atr > 0:
            out["m5_range12_atr20"] = round((box_high - box_low) / atr, 6)
            out["m5_range12_break"] = "up" if price > box_high else "down" if price < box_low else "inside"
    return out


class SpyStateReader:
    """The SPY market-state engine's recorded state at a bar, from its shadow log (cached by size/mtime)."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._signature: tuple | None = None
        self._rows: list[tuple[str, datetime, str, int, bool]] = []

    def _load(self) -> None:
        path = Path(self._path) if self._path is not None else _spy_log_path()
        try:
            stat = path.stat()
        except OSError:
            self._signature, self._rows = None, []
            return
        signature = (str(path), stat.st_mtime_ns, stat.st_size)
        if signature == self._signature:
            return
        rows = []
        with path.open("rb") as handle:
            if stat.st_size > SPY_LOG_MAX_BYTES:
                handle.seek(stat.st_size - SPY_LOG_MAX_BYTES)
                handle.readline()  # a partial first line
            for raw in handle:
                try:
                    record = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(record, dict) or not str(record.get("schema") or "").startswith(
                    _SPY_STATE_SCHEMA_PREFIX
                ):
                    continue
                try:
                    bar = datetime.fromisoformat(_text(record.get("bar_ts")))
                except ValueError:
                    continue
                if bar.tzinfo is None:
                    continue
                try:
                    sign = int(record.get("side_sign") or 0)
                except (TypeError, ValueError):
                    sign = 0
                rows.append((_text(record.get("session_date"))[:10], bar, _text(record.get("state")).upper(),
                             sign, bool(record.get("usable"))))
        self._signature, self._rows = signature, rows

    def state_at(self, alert_close: datetime) -> tuple[str, int] | None:
        """``(state, side_sign)`` of the last USABLE row at or before ``alert_close`` that session, or None."""
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 - a log read costs the facet, never the record
            _log_once(f"SPY state log: {type(exc).__name__}: {exc}")
            return None
        from zoneinfo import ZoneInfo

        session = alert_close.astimezone(ZoneInfo(_EXCHANGE_TZ)).date().isoformat()
        found = None
        for day, bar, state, sign, usable in self._rows:
            if usable and day == session and bar <= alert_close and (found is None or bar >= found[0]):
                found = (bar, state, sign)
        if found is None:
            return None
        return found[1], found[2]


_spy_reader: SpyStateReader | None = None


def d1_environment_before(trade_date: Any, path: Any = None) -> str:
    """SPY's D1 environment label for the session before ``trade_date`` (known before the open)."""
    import d1_environment_store
    import setup_permutation_backfill as bf

    previous = bf.previous_session_text(_text(trade_date)[:10])
    return d1_environment_store.label_for_session(previous, path=path) if previous else "unknown"


def m5_part(row: Mapping[str, Any], *, symbol: str, side: str) -> dict[str, Any]:
    """The M5 key fields for one registered outcome row. Never raises: a failed input is unknown."""
    global _spy_reader
    import setup_permutations as sp

    inputs: dict[str, Any] = {}
    try:
        zone = _local_tz()
        inputs = alert_inputs(row, zone)
        close = alert_bar_close(row)
        if close is not None:
            # A bar still forming when the alert was logged is never measured.
            if inputs.get("alert_bar_complete") is True:
                bars = _cached_bars(symbol)
                if bars:
                    inputs["vwap_dist_atr"] = vwap_distance_atr(bars, close, zone)
                    inputs.update(structure_inputs(bars, close, zone))
            if _spy_reader is None:
                _spy_reader = SpyStateReader()
            spy = _spy_reader.state_at(close.replace(tzinfo=zone))
            if spy is not None:
                inputs["spy_state"], inputs["spy_side_sign"] = spy
    except Exception as exc:  # noqa: BLE001 - the M5 part never costs the record
        _log_once(f"M5 part: {type(exc).__name__}: {exc}")
    try:
        inputs["d1_environment"] = d1_environment_before(row.get("trade_date"))
    except Exception as exc:  # noqa: BLE001 - the M5 part never costs the record
        _log_once(f"D1 environment: {type(exc).__name__}: {exc}")
    key = sp.m5_facets_for(inputs, side)
    return {
        "m5_key": key.compact_key,
        "m5_label": key.label,
        "m5_rule_version": key.permutation_rule_version,
        "m5_facets": key.as_dict(),
        "m5_inputs": {name: inputs.get(name) for name in sp.M5_INPUT_FIELDS},
    }


def record_for(row: Mapping[str, Any], lookup: ScanKeyLookup) -> dict | None:
    """The sidecar record for one outcome row, or None when the row has no event id."""
    event_id = _text(row.get("event_id"))
    if not event_id:
        return None
    symbol = _text(row.get("symbol")).upper()
    side = _side(row.get("direction"))
    trade_date = _text(row.get("trade_date"))[:10]
    return {
        "schema": SCHEMA,
        "event_id": event_id,
        "symbol": symbol,
        "side": side,
        "trade_date": trade_date,
        **lookup.stamp(symbol, side, trade_date),
        **m5_part(row, symbol=symbol, side=side),
        "stamped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def append_record(record: Mapping[str, Any], path: Path | None = None) -> None:
    from project_paths import M5_SETUP_KEY_STAMPS_FILE

    target = Path(path) if path is not None else Path(M5_SETUP_KEY_STAMPS_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n").encode("ascii")
    # Unbuffered: the whole line goes down in one write() call, never a partial line.
    with target.open("ab", buffering=0) as handle:
        handle.write(line)


def read_stamps(path: Path) -> dict[str, dict]:
    """``{event_id: record}``, the FIRST record per event (the one made at alert time)."""
    out: dict[str, dict] = {}
    target = Path(path)
    if not target.is_file():
        return out
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("schema") == SCHEMA:
                out.setdefault(_text(record.get("event_id")), record)
    return out


# --- the hook and its one worker


_lookup: ScanKeyLookup | None = None


def _run() -> None:
    global _lookup
    while True:
        item = _queue.get()
        try:
            if _lookup is None:
                _lookup = ScanKeyLookup()
            record = record_for(item, _lookup)
            if record is not None:
                append_record(record)
        except Exception as exc:  # noqa: BLE001 - the worker outlives any one bad row
            _log_once(f"{type(exc).__name__}: {exc}")
        finally:
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(target=_run, name="m5-setup-key-stamp", daemon=True)
        _worker.start()


def _market_today() -> str:
    from market_calendar import MARKET_TZ

    return datetime.now(MARKET_TZ).date().isoformat()


def submit(row: Mapping[str, Any]) -> bool:
    """Queue the first outcome row of today's event for stamping. Never raises, never waits.

    Only seven plain strings are copied off ``row``; the row itself is not kept.
    """
    try:
        if not enabled():
            return False
        event_id = _text(row.get("event_id"))
        if not event_id or _text(row.get("trade_date"))[:10] != _market_today():
            return False  # an old session's row (the startup sweep) is left to the backfill
        with _lock:
            if event_id in _seen:
                return False
            _seen.add(event_id)
        context = row.get("context_json")
        item = {
            "event_id": event_id,
            "symbol": _text(row.get("symbol")),
            "direction": _text(row.get("direction")),
            "trade_date": _text(row.get("trade_date")),
            "entry_time": _text(row.get("entry_time")),
            "logged_at": _text(row.get("logged_at")),
            "context_json": context if isinstance(context, str) and len(context) <= MAX_CONTEXT_CHARS else "",
        }
        _ensure_worker()
        _queue.put_nowait(item)
        return True
    except queue.Full:
        _log_once("queue full")
        return False
    except Exception as exc:  # noqa: BLE001 - the hook never costs the caller
        _log_once(f"{type(exc).__name__}: {exc}")
        return False


def drain(timeout: float = 10.0) -> bool:
    """Wait until the queue is empty (tests and CLIs only; never from the bot loop)."""
    deadline = time.monotonic() + timeout
    while _queue.unfinished_tasks:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def reset_for_tests(lookup: ScanKeyLookup | None = None) -> None:
    global _lookup, _spy_reader
    drain(5.0)
    _spy_reader = None
    with _lock:
        _seen.clear()
        _logged_reasons.clear()
    _lookup = lookup


__all__ = [
    "SCHEMA",
    "ScanKeyLookup",
    "SpyStateReader",
    "alert_inputs",
    "append_record",
    "drain",
    "read_stamps",
    "record_for",
    "register_bar_source",
    "submit",
    "vwap_distance_atr",
]
