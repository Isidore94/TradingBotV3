r"""Does the compression measure agree with the trader's eye? (packet PCT-3, item 3)

Trader, 2026-09-15: *"we need a way to measure for compression as the most COMMON
veto I have is a compression veto. we need to fine tune this so we stop getting
so many compressed picks."* Live counts 2026-08-20..09-15: 598 coded vetoes, of
which `compressed` (v3) plus `support_resistance_cluttered` (v1) are 214 - 36 %.
The scan already measures compression on every priority row and already docks
the score for it, so either the measure and the eye agree and something else is
wrong, or they do not agree and the threshold is wrong. **This report answers
that question and changes nothing.**

It is a read-only report and it is the whole of its own authority: it proposes
no threshold, writes to no store the desk reads, touches no detector, no score,
no alert, no watchlist, no Focus list, no review queue and no `review_policy.json`.
The penalty decision is a SEPARATE ask, after the trader has read this.

    cd scripts
    python -m compression_calibration --since 2026-08-20            # scratch home
    python -m compression_calibration --since 2026-08-20 --live     # the real one

Every veto is counted and named
-------------------------------

The first review of this report found it quietly dropping 127 of 213 vetoes: it
joined a veto to a tracker record whose `scan_date` equalled the veto's session,
so a veto on a setup the trader had been watching for a week matched nothing and
vanished without a word. A report that silently discards 60 % of its own
evidence is worse than no report. So every veto now lands in exactly one of
three buckets, and the count of all three is printed:

``joined``
    a tracker record was ACTIVE on that session (see below) for that symbol.
``untracked``
    no record was active - the trader vetoed something the tracker was not
    following. It still gets the four bar-derived measures and still counts on
    the vetoed side; only the anchor measures are unavailable.
``pending``
    the session is not complete yet, so no measure may be taken on it at all
    (`plan.md` sec 5: completed bars only).

A session's population
----------------------

For a session S, the population is every tracker record **active on S**: entered
on or before S and not yet closed or expired by it, read off the record's own
dates (`entry_trade_date` / `scan_date` for the entry, `last_replayed_session`
for the last session it was still being carried). That is the set the trader was
looking at. It is NOT "the rows the scan printed that morning" - the earlier
version said "shown", which was wrong twice over, and the header now prints the
definition it actually used.

Seven measures, no verdict
--------------------------

Per row, at the session's completed close:

===================================  ======================================
`compression_stdev_atr_ratio`        the scan's own three anchor ratios,
`compression_range_atr_ratio`        read off the record when it carries them
`compression_close_range_atr_ratio`  and RECOMPUTED from the bar cache when it
                                     does not (`summarize_anchor_compression`,
                                     the same function, never a second copy)
`range10_atr14`                      the `d1_environment` rule, per symbol
`range20_atr14`                      the same over twenty sessions
`bollinger20_width_pct`              Bollinger(20, 2) relative width, as a
                                     percentile over the last 120 sessions
`atr14_atr50`                        is this name quiet against its own history?
===================================  ======================================

Each prints `n = vetoed / rest`, the two medians and a rank-sum **AUC** - the
probability that a randomly chosen VETOED row scores HIGHER on that measure than
a randomly chosen rest row, ties counted as half. 0.5 is "this measure cannot
tell them apart"; 0.0 and 1.0 are perfect separation in one direction or the
other. A LOW AUC on every measure is a finding, not a failure.

Point-in-time (`plan.md` sec 5)
-------------------------------

No bar dated after the session is ever read: the cached frame is cut at the
session date before a single measure is computed, and that includes the
recomputed anchor. A measure that cannot be taken is `None` and is left out of
its own n - the unmeasured is shown, never assumed.

Read-only, and it says so first
-------------------------------

`project_paths.DATA_DIR` is read at CALL time (the `d1_environment_store`
idiom). If it resolves under the live home folder the run REFUSES before it
opens anything, names the folder and names `--live`. Every store below is opened
for reading only; the one file written is the CSV, under `--out` or
`%LOCALAPPDATA%\TradingBotV3\diagnostics\`.

The tracker JSON is 1.26 GB live. `read_text` on it peaked at 3.79 GB of RSS,
and `json.loads` on it is one of the three causes of the 10 GB desk on
2026-08-27, so :func:`iter_tracker_records` STREAMS it: a bounded sliding window
fed by a buffered reader, with `json.JSONDecoder.raw_decode` parsing one record
at a time out of that window and the window discarded behind it. Peak memory is
a few megabytes above the largest single record. The SQLite mirror would be
cheaper still and is deliberately not read: decision 0017 fences every reader out
of it until gate #57.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import project_paths
from indicators.atr import wilder_atr

#: The two codes this report pools, by name. `veto_cohort` does not pool them
#: (they are different definitions, not a rename); see the DESK_INTERNALS entry.
COMPRESSION_VETO_CODES = frozenset({"compressed", "support_resistance_cluttered"})

#: The tracker sections that hold per-(symbol, session) scan rows.
TRACKER_RECORD_SECTIONS = ("setups", "control_setups", "study_setups")

#: One printed block per key, in this order. The first three are the scan's own
#: anchor ratios; the rest are the candidates the packet named.
MEASURE_KEYS = (
    "compression_stdev_atr_ratio",
    "compression_range_atr_ratio",
    "compression_close_range_atr_ratio",
    "range10_atr14",
    "range20_atr14",
    "bollinger20_width_pct",
    "atr14_atr50",
)

#: The measures that come from the anchored box rather than a fixed window.
ANCHOR_MEASURE_KEYS = MEASURE_KEYS[:3]
#: The measures a row can always have, anchor or no anchor.
FIXED_WINDOW_MEASURE_KEYS = MEASURE_KEYS[3:]

#: Where a row's anchor measures came from. Printed per row in the CSV.
ANCHOR_FROM_RECORD = "record"
ANCHOR_RECOMPUTED = "recomputed"
ANCHOR_UNKNOWN = "anchor unknown"

#: What a row IS, in the population. Printed per row in the CSV.
ROLE_JOINED = "joined"
ROLE_UNTRACKED = "untracked"

#: How a veto ended up. Every veto gets exactly one of these.
OUTCOME_JOINED = "joined"
OUTCOME_UNTRACKED = "untracked"
OUTCOME_PENDING = "pending"

#: Window lengths, stated once.
RANGE10_SESSIONS = 10
RANGE20_SESSIONS = 20
ATR_SHORT_SESSIONS = 14
ATR_LONG_SESSIONS = 50
BOLLINGER_SESSIONS = 20
BOLLINGER_STDEVS = 2.0
BOLLINGER_PERCENTILE_SESSIONS = 120

#: Folder names that mean "this is the trader's live store, not a scratch one".
#: Spaces are removed before the compare so the DAS (`\\MINI-PC\Trading Bot
#: Data`) is caught by the same rule as `C:\TradingBotData`.
_LIVE_FOLDER_NAMES = frozenset({"tradingbotdata"})

#: The streaming reader's read size. Peak memory is roughly this plus the
#: largest single tracker record.
TRACKER_CHUNK_BYTES = 1 << 20


# ---------------------------------------------------------------------------
# the refusal
# ---------------------------------------------------------------------------


def _folder_is_live(path: Path | str | None) -> bool:
    if path is None:
        return False
    for part in Path(str(path)).parts:
        if part.replace(" ", "").replace("_", "").lower() in _LIVE_FOLDER_NAMES:
            return True
    return False


def live_home_folder() -> str:
    """The live path this run would read, or `""` when it is a scratch home.

    `project_paths.DATA_DIR` and `PERSISTENT_DATA_DIR` are read at CALL time so
    a test can point them somewhere and be believed - the same idiom
    `d1_environment_store._cached_daily_bars` uses.
    """
    for attribute in ("DATA_DIR", "PERSISTENT_DATA_DIR"):
        candidate = getattr(project_paths, attribute, None)
        if _folder_is_live(candidate):
            return str(candidate)
    return ""


# ---------------------------------------------------------------------------
# the vetoes
# ---------------------------------------------------------------------------


def _session_of(row: dict) -> str:
    for key in ("session_date", "scan_date", "trade_date"):
        value = str(row.get(key) or "").strip()[:10]
        if value:
            return value
    return ""


def _side_of(row: dict) -> str:
    value = str(row.get("side") or "").strip().upper()
    return value if value in {"LONG", "SHORT"} else ""


def read_compression_vetoes(path: Path | str, *, since: str) -> dict[tuple[str, str, str], str]:
    """`(symbol, session_date, side) -> reason_code` for every compression veto.

    A row with a blank or absent `reason_code` is skipped: it IS a veto and it
    is NOT a compression veto, and counting it as one would move `n` without
    moving the evidence.
    """
    found: dict[tuple[str, str, str], str] = {}
    try:
        handle = open(path, "r", encoding="utf-8-sig")
    except OSError:
        return found
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            code = str(row.get("reason_code") or "").strip().lower()
            if code not in COMPRESSION_VETO_CODES:
                continue
            session = _session_of(row)
            symbol = str(row.get("symbol") or "").strip().upper()
            if not session or not symbol or session < since:
                continue
            found[(symbol, session, _side_of(row))] = code
    return found


def session_is_complete(session: str) -> bool:
    """Has the exchange session for `session` closed? Unknown reads as NOT.

    Missing data is uncertainty, never confirmation - a session nobody can date
    is pending, not measured.
    """
    try:
        from master_avwap_lib import daily_bar_cache

        return bool(daily_bar_cache.session_is_complete(date.fromisoformat(session[:10])))
    except Exception:  # noqa: BLE001 - an unreadable calendar is uncertainty
        return False


# ---------------------------------------------------------------------------
# the tracker, streamed
# ---------------------------------------------------------------------------


class _JsonWindow:
    """A bounded sliding window over a JSON text stream.

    `raw_decode` needs the whole value it is parsing to be in one string, but
    it does NOT need the rest of the file. So the window grows until the value
    at hand parses and is discarded as soon as it has, which is what keeps the
    1.26 GB tracker's peak a few megabytes rather than 3.79 GB.
    """

    def __init__(self, handle, *, chunk_size: int = TRACKER_CHUNK_BYTES):
        self._handle = handle
        self._chunk = max(4096, int(chunk_size))
        self._buffer = ""
        self._base = 0
        self._eof = False
        self._decoder = json.JSONDecoder()

    def _read_more(self) -> bool:
        if self._eof:
            return False
        data = self._handle.read(self._chunk)
        if not data:
            self._eof = True
            return False
        self._buffer += data
        return True

    def _trim(self, absolute: int) -> None:
        cut = absolute - self._base
        if cut > 0:
            self._buffer = self._buffer[cut:]
            self._base = absolute

    def peek(self, absolute: int) -> str:
        """The character at `absolute`, or `""` at end of file."""
        self._trim(absolute)
        while not self._buffer and self._read_more():
            pass
        return self._buffer[0] if self._buffer else ""

    def skip_whitespace(self, absolute: int) -> int:
        while True:
            char = self.peek(absolute)
            if char and char in " \t\r\n":
                absolute += 1
                continue
            return absolute

    def decode(self, absolute: int) -> tuple[Any, int]:
        """Parse exactly one JSON value starting at `absolute`."""
        self._trim(absolute)
        while True:
            if self._buffer:
                try:
                    value, end = self._decoder.raw_decode(self._buffer, 0)
                    return value, self._base + end
                except ValueError:
                    pass
            if not self._read_more():
                raise ValueError(f"unterminated JSON value at offset {absolute}")

    def skip_value(self, absolute: int) -> int:
        """Walk PAST one JSON value without building it.

        Used for the tracker's non-record sections, which can be large and
        which this report never reads: decoding them would put their whole
        object graph in memory for nothing.
        """
        char = self.peek(absolute)
        if not char:
            raise ValueError("unterminated JSON value")
        if char in "{[":
            depth = 0
            in_string = False
            escaped = False
            while True:
                char = self.peek(absolute)
                if not char:
                    raise ValueError("unterminated JSON container")
                absolute += 1
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char in "{[":
                    depth += 1
                elif char in "}]":
                    depth -= 1
                    if depth == 0:
                        return absolute
        if char == '"':
            escaped = False
            absolute += 1
            while True:
                char = self.peek(absolute)
                if not char:
                    raise ValueError("unterminated JSON string")
                absolute += 1
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    return absolute
        while True:
            char = self.peek(absolute)
            if not char or char in ",}] \t\r\n":
                return absolute
            absolute += 1


def iter_tracker_records(path: Path | str, *, chunk_size: int = TRACKER_CHUNK_BYTES) -> Iterator[dict]:
    """Every tracker record, ONE at a time, streamed.

    See the module docstring: the live file is 1.26 GB, and neither `json.load`
    nor `read_text` on it is allowed here.
    """
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError:
        return
    with handle:
        window = _JsonWindow(handle, chunk_size=chunk_size)
        index = window.skip_whitespace(0)
        if window.peek(index) != "{":
            return
        index = window.skip_whitespace(index + 1)
        try:
            while window.peek(index) not in ("}", ""):
                key, index = window.decode(index)
                index = window.skip_whitespace(index)
                if window.peek(index) != ":":
                    return
                index = window.skip_whitespace(index + 1)
                if key in TRACKER_RECORD_SECTIONS and window.peek(index) == "{":
                    index = window.skip_whitespace(index + 1)
                    while window.peek(index) not in ("}", ""):
                        _record_key, index = window.decode(index)
                        index = window.skip_whitespace(index)
                        if window.peek(index) != ":":
                            return
                        value, index = window.decode(window.skip_whitespace(index + 1))
                        if isinstance(value, dict):
                            yield value
                        index = window.skip_whitespace(index)
                        if window.peek(index) == ",":
                            index = window.skip_whitespace(index + 1)
                    index += 1
                else:
                    index = window.skip_value(index)
                index = window.skip_whitespace(index)
                if window.peek(index) == ",":
                    index = window.skip_whitespace(index + 1)
        except ValueError:
            return


def record_active_window(record: dict) -> tuple[str, str]:
    """The first and last session a tracker record was being carried.

    Entry is the record's own `entry_trade_date` (or `scan_date`); the last
    session is `last_replayed_session`, which the tracker's replay advances
    every session a setup is still open and stops advancing the moment it
    closes or expires. A record that was never replayed was live for its entry
    session alone, which is the honest answer rather than a guess.
    """
    entry = str(record.get("entry_trade_date") or record.get("scan_date") or "").strip()[:10]
    last = str(record.get("last_replayed_session") or "").strip()[:10] or entry
    if entry and last < entry:
        last = entry
    return entry, last


# ---------------------------------------------------------------------------
# the bars
# ---------------------------------------------------------------------------


def read_cached_daily_bars(symbol: str) -> list[dict]:
    """The desk's cached daily bars for `symbol`, oldest first, WHOLE.

    `project_paths.DAILY_BARS_CACHE_DIR` is read at CALL time. Nothing here
    opens a provider or makes a network call. The point-in-time cut is made by
    :func:`bars_through`, once per (symbol, session), so one symbol's file is
    read once no matter how many sessions it appears in.
    """
    directory = Path(project_paths.DAILY_BARS_CACHE_DIR)
    target = directory / f"{str(symbol or '').strip().upper()}.csv"
    bars: list[dict] = []
    try:
        handle = open(target, newline="", encoding="utf-8-sig")
    except OSError:
        return bars
    with handle:
        for row in csv.DictReader(handle):
            stamp = ""
            for key in ("datetime", "date", "dt", "timestamp"):
                if row.get(key):
                    stamp = str(row[key])[:10]
                    break
            if not stamp:
                continue
            try:
                bar = {
                    "dt": stamp,
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            for optional in ("open", "volume"):
                try:
                    bar[optional] = float(row[optional])
                except (KeyError, TypeError, ValueError):
                    bar[optional] = None
            bars.append(bar)
    bars.sort(key=lambda bar: bar["dt"])
    return bars


def bars_through(bars: Sequence[dict], session: str) -> list[dict]:
    """Point-in-time: the bars up to AND INCLUDING `session`, never past it."""
    return [bar for bar in bars if bar["dt"] <= session]


# ---------------------------------------------------------------------------
# the measures
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _range_over_atr(bars: Sequence[dict], sessions: int) -> float | None:
    if len(bars) < sessions:
        return None
    atr = wilder_atr(bars, ATR_SHORT_SESSIONS)
    if not atr:
        return None
    window = bars[-sessions:]
    return (max(bar["high"] for bar in window) - min(bar["low"] for bar in window)) / atr


def _bollinger_width(bars: Sequence[dict], end: int) -> float | None:
    """Relative Bollinger(20, 2) width at `bars[end]`, or `None`."""
    if end + 1 < BOLLINGER_SESSIONS:
        return None
    closes = [bar["close"] for bar in bars[end + 1 - BOLLINGER_SESSIONS : end + 1]]
    mean = statistics.fmean(closes)
    if mean <= 0:
        return None
    return (2.0 * BOLLINGER_STDEVS * statistics.pstdev(closes)) / mean


def _bollinger_width_percentile(bars: Sequence[dict]) -> float | None:
    """Today's width as a percentile of the last 120 sessions' widths.

    100 means "the widest this name has been in that window"; a low number is
    the compression this report is looking for.
    """
    widths: list[float] = []
    last = len(bars) - 1
    for index in range(max(0, last - BOLLINGER_PERCENTILE_SESSIONS + 1), last + 1):
        width = _bollinger_width(bars, index)
        if width is not None:
            widths.append(width)
    if len(widths) < BOLLINGER_PERCENTILE_SESSIONS:
        return None
    current = widths[-1]
    at_or_below = sum(1 for width in widths if width <= current)
    return 100.0 * at_or_below / len(widths)


def _atr_ratio(bars: Sequence[dict]) -> float | None:
    short = wilder_atr(bars, ATR_SHORT_SESSIONS)
    long = wilder_atr(bars, ATR_LONG_SESSIONS)
    if not short or not long:
        return None
    return short / long


def fixed_window_measures(bars: Sequence[dict]) -> dict[str, float | None]:
    """The four measures that need only a price history, never an anchor."""
    return {
        "range10_atr14": _range_over_atr(bars, RANGE10_SESSIONS),
        "range20_atr14": _range_over_atr(bars, RANGE20_SESSIONS),
        "bollinger20_width_pct": _bollinger_width_percentile(bars),
        "atr14_atr50": _atr_ratio(bars),
    }


def _anchor_measures_from_record(record: dict) -> dict[str, float | None] | None:
    """The scan's own compression numbers off the tracker record, or `None`.

    Top level first, then the record's `compression_summary`, then its entry
    feature snapshot - three places one scan writes the same reading. `None`
    means the record predates PCT-3 item 1 and carries no measure at all, which
    is the caller's cue to recompute rather than to print a zero.
    """
    sources = [record]
    for key in ("compression_summary", "entry_feature_snapshot"):
        nested = record.get(key)
        if isinstance(nested, dict):
            sources.append(nested)
    measures: dict[str, float | None] = {}
    for key in ANCHOR_MEASURE_KEYS:
        measures[key] = next(
            (value for value in (_number(source.get(key)) for source in sources) if value is not None),
            None,
        )
    if all(value is None for value in measures.values()):
        return None
    return measures


def recompute_anchor_measures(anchor_date: str, bars: Sequence[dict]) -> dict[str, float | None] | None:
    """`summarize_anchor_compression` on the bars, at the anchor the record names.

    The rule lives ONCE, in `legacy.py`; this only supplies its three inputs -
    the anchored slice, that anchor's running-deviation sigma (the champion's
    `calc_anchored_vwap_bands`, never a second sigma) and the scan's own ATR-20
    at the session's close. `None` when any of the three cannot be had.

    `legacy` is imported here rather than at module scope because it is a large
    import and a run whose records all carry the measure never needs it.
    """
    anchor = str(anchor_date or "").strip()[:10]
    if not anchor or not bars:
        return None
    anchor_index = next((index for index, bar in enumerate(bars) if bar["dt"] >= anchor), None)
    if anchor_index is None or anchor_index >= len(bars):
        return None
    try:
        import pandas as pd

        from master_avwap_lib import legacy
    except Exception:  # noqa: BLE001 - a missing engine is an unmeasured row
        return None
    if any(bar.get("open") is None or bar.get("volume") is None for bar in bars):
        # `calc_anchored_vwap_bands` weights by volume and prices by OHLC/4; a
        # cache row missing either is unmeasurable, not zero.
        return None
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime([bar["dt"] for bar in bars]),
            "open": [bar["open"] for bar in bars],
            "high": [bar["high"] for bar in bars],
            "low": [bar["low"] for bar in bars],
            "close": [bar["close"] for bar in bars],
            "volume": [bar["volume"] for bar in bars],
        }
    )
    try:
        _vwap, stdev, _bands = legacy.calc_anchored_vwap_bands(frame, anchor_index)
    except Exception:  # noqa: BLE001
        return None
    if stdev is None or stdev != stdev:
        return None
    daily_rows = [
        {"date": bar["dt"], "high": bar["high"], "low": bar["low"], "close": bar["close"]}
        for bar in bars
    ]
    atr20 = legacy.compute_atr_from_ohlc(daily_rows, date.fromisoformat(bars[-1]["dt"]))
    if not atr20:
        return None
    summary = legacy.summarize_anchor_compression(frame.iloc[anchor_index:], stdev, atr20)
    measures = {key: _number(summary.get(key)) for key in ANCHOR_MEASURE_KEYS}
    if all(value is None for value in measures.values()):
        return None
    return measures


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == value and value != 0
    return str(value or "").strip().lower() in {"true", "t", "yes", "y", "1"}


def _record_flag(record: dict) -> bool:
    if "compression_flag" in record:
        return _flag(record.get("compression_flag"))
    summary = record.get("compression_summary")
    if isinstance(summary, dict):
        return _flag(summary.get("is_compressed") or summary.get("compression_flag"))
    return False


def auc(vetoed: Sequence[float], rest: Sequence[float]) -> float | None:
    """P(a vetoed row scores higher than a rest row), ties counted as half.

    The rank-sum statistic, computed from the pairs directly - the populations
    here are a few hundred rows a day, so the O(n*m) form is honest and needs no
    tie-correction argument.
    """
    if not vetoed or not rest:
        return None
    wins = 0.0
    for left in vetoed:
        for right in rest:
            if left > right:
                wins += 1.0
            elif left == right:
                wins += 0.5
    return wins / float(len(vetoed) * len(rest))


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def build_rows(*, since: str) -> tuple[list[dict], dict]:
    """Every population row, plus the accounting for every veto.

    Returns `(rows, summary)`. `summary` names what happened to each veto and
    which sessions were excluded and why, because a number this report does not
    print is a number it has silently thrown away.
    """
    vetoes = read_compression_vetoes(project_paths.TRADER_ANNOTATIONS_FILE, since=since)
    summary = {
        "veto_count": len(vetoes),
        "joined": 0,
        "untracked": 0,
        "pending": 0,
        "sessions_measured": [],
        "sessions_pending": [],
    }
    if not vetoes:
        return [], summary

    all_sessions = sorted({session for _symbol, session, _side in vetoes})
    measured_sessions = {session for session in all_sessions if session_is_complete(session)}
    pending_sessions = [session for session in all_sessions if session not in measured_sessions]
    summary["sessions_measured"] = sorted(measured_sessions)
    summary["sessions_pending"] = pending_sessions
    summary["pending"] = sum(
        1 for _symbol, session, _side in vetoes if session not in measured_sessions
    )

    # Pass 1: stream the tracker once and keep only what each measured session
    # needs. A record is kept per (symbol, session) it was ACTIVE on.
    population: dict[str, dict[tuple[str, str], dict]] = {}
    for record in iter_tracker_records(project_paths.MASTER_AVWAP_SETUP_TRACKER_FILE):
        symbol = str(record.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        entry, last = record_active_window(record)
        if not entry:
            continue
        side = _side_of(record)
        for session in measured_sessions:
            if not (entry <= session <= last):
                continue
            slot = population.setdefault(symbol, {})
            slot.setdefault(
                (session, side),
                {
                    "anchor_date": str(record.get("anchor_date") or "").strip()[:10],
                    "compression_flag": _record_flag(record),
                    "compression_score": record.get("compression_score"),
                    "anchor_measures": _anchor_measures_from_record(record),
                },
            )

    # Every veto on a measured session that found no active record.
    untracked: dict[str, set[tuple[str, str]]] = {}
    for (symbol, session, side), _code in vetoes.items():
        if session not in measured_sessions:
            continue
        slot = population.get(symbol) or {}
        if (session, side) in slot or (session, "") in slot:
            continue
        untracked.setdefault(symbol, set()).add((session, side))

    # Pass 2: one symbol at a time, so one bar file is read once and dropped.
    rows: list[dict] = []
    symbols = sorted(set(population) | set(untracked))
    for symbol in symbols:
        bars = read_cached_daily_bars(symbol)
        for (session, side), entry in sorted((population.get(symbol) or {}).items()):
            rows.append(
                _build_row(
                    symbol=symbol,
                    session=session,
                    side=side,
                    role=ROLE_JOINED,
                    entry=entry,
                    bars=bars,
                    vetoes=vetoes,
                )
            )
        for session, side in sorted(untracked.get(symbol) or ()):
            rows.append(
                _build_row(
                    symbol=symbol,
                    session=session,
                    side=side,
                    role=ROLE_UNTRACKED,
                    entry=None,
                    bars=bars,
                    vetoes=vetoes,
                )
            )

    summary["joined"] = sum(1 for row in rows if row["compressed_veto"] and row["population_role"] == ROLE_JOINED)
    summary["untracked"] = sum(
        1 for row in rows if row["compressed_veto"] and row["population_role"] == ROLE_UNTRACKED
    )
    rows.sort(key=lambda row: (row["session_date"], row["symbol"], row["side"]))
    return rows, summary


def _build_row(
    *,
    symbol: str,
    session: str,
    side: str,
    role: str,
    entry: dict | None,
    bars: Sequence[dict],
    vetoes: dict[tuple[str, str, str], str],
) -> dict:
    window = bars_through(bars, session)
    code = vetoes.get((symbol, session, side)) or vetoes.get((symbol, session, ""))
    row = {
        "symbol": symbol,
        "session_date": session,
        "side": side,
        "population_role": role,
        "compressed_veto": bool(code),
        "reason_code": code or "",
        "compression_flag": bool((entry or {}).get("compression_flag")),
        "compression_score": (entry or {}).get("compression_score", ""),
    }

    anchor_measures = (entry or {}).get("anchor_measures")
    anchor_source = ANCHOR_FROM_RECORD if anchor_measures else ""
    if anchor_measures is None and entry is not None:
        anchor_measures = recompute_anchor_measures(entry.get("anchor_date") or "", window)
        anchor_source = ANCHOR_RECOMPUTED if anchor_measures else ANCHOR_UNKNOWN
    if anchor_measures is None:
        anchor_measures = {key: None for key in ANCHOR_MEASURE_KEYS}
        anchor_source = anchor_source or ANCHOR_UNKNOWN
    row["anchor_source"] = anchor_source
    row.update(anchor_measures)
    row.update(fixed_window_measures(window))
    return row


def _format(value: float | None) -> str:
    return "nan" if value is None else f"{value:.4f}"


def render_report(rows: Sequence[dict], summary: dict, *, since: str) -> str:
    """The printed report. The accounting first, then one block per measure."""
    vetoed = [row for row in rows if row["compressed_veto"]]
    rest = [row for row in rows if not row["compressed_veto"]]
    measured = summary.get("sessions_measured") or []
    pending = summary.get("sessions_pending") or []

    lines = [
        "compression calibration - read-only, proposes nothing",
        (
            f"window asked for: since {since}. Sessions carrying a compression veto and"
            f" MEASURED: {len(measured)}"
            + (f" ({measured[0]} .. {measured[-1]})" if measured else "")
        ),
    ]
    if pending:
        lines.append(
            "sessions EXCLUDED, not yet complete (completed bars only): " + ", ".join(pending)
        )
    else:
        lines.append("sessions excluded: none")
    lines.extend(
        [
            (
                f"{summary.get('veto_count', 0)} vetoes:"
                f" {summary.get('joined', 0)} joined,"
                f" {summary.get('untracked', 0)} untracked,"
                f" {summary.get('pending', 0)} pending"
            ),
            (
                "population per session: every tracker record ACTIVE on it"
                " (entered on or before it, still carried by it) - not the rows the scan"
                " happened to print that morning"
            ),
            f"rows: {len(rows)} - {len(vetoed)} vetoed, {len(rest)} rest",
            f"codes pooled by name: {', '.join(sorted(COMPRESSION_VETO_CODES))}",
            "auc = P(a vetoed row reads HIGHER than a rest row), ties half; 0.5 tells them apart not at all",
            "",
        ]
    )

    for key in MEASURE_KEYS:
        left = [value for value in (row.get(key) for row in vetoed) if value is not None]
        right = [value for value in (row.get(key) for row in rest) if value is not None]
        counts = f"n = {len(left)} / {len(right)}"
        if (len(left), len(right)) != (len(vetoed), len(rest)):
            counts += f"   measured (of {len(vetoed)} / {len(rest)} rows)"
        lines.extend(
            [
                key,
                f"  {counts}",
                "  median = "
                f"{_format(statistics.median(left) if left else None)} / "
                f"{_format(statistics.median(right) if right else None)}",
                f"  auc = {_format(auc(left, right))}",
                "",
            ]
        )

    unknown = sum(1 for row in rows if row.get("anchor_source") == ANCHOR_UNKNOWN)
    recomputed = sum(1 for row in rows if row.get("anchor_source") == ANCHOR_RECOMPUTED)
    lines.append(
        f"anchor measures: {recomputed} recomputed from the bar cache, {unknown} row(s) anchor unknown"
    )
    hits = sum(1 for row in vetoed if row["compression_flag"])
    rate = (hits / len(vetoed)) if vetoed else None
    lines.append(
        f"compression_flag hit rate = {_format(rate)}"
        f"   ({hits} of {len(vetoed)} vetoed row(s) carried today's flag)"
    )
    misses = sum(1 for row in rest if row["compression_flag"])
    lines.append(
        f"  and it was set on {misses} of {len(rest)} row(s) the trader did NOT veto for compression"
    )
    return "\n".join(lines)


CSV_COLUMNS = (
    "symbol",
    "session_date",
    "side",
    "compressed_veto",
    "population_role",
    "reason_code",
    "anchor_source",
    "compression_flag",
    "compression_score",
    *MEASURE_KEYS,
)


def write_csv(rows: Sequence[dict], out_dir: Path, *, stamp: date | None = None) -> Path:
    """One row per population row. Temp-and-rename, so a half-written file never lands."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"compression_calibration_{(stamp or date.today()).isoformat()}.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
    temporary.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compression_calibration",
        description="Compare candidate compression measures against the trader's compression vetoes.",
    )
    parser.add_argument("--since", required=True, help="first session date to read (YYYY-MM-DD)")
    parser.add_argument("--out", default=None, help="where the CSV goes (default: the diagnostics folder)")
    parser.add_argument(
        "--live",
        action="store_true",
        help="allow reading the trader's live stores (this report never writes to them)",
    )
    args = parser.parse_args(argv)

    try:
        since = date.fromisoformat(str(args.since)[:10]).isoformat()
    except ValueError:
        print(f"--since must be a date (YYYY-MM-DD), not {args.since!r}", file=sys.stderr)
        return 2

    live = live_home_folder()
    if live and not args.live:
        print(
            "Refusing to read the trader's live stores without --live.\n"
            f"  project_paths.DATA_DIR resolves under {live}\n"
            "  This run would read, READ-ONLY: the veto file and the setup tracker under that\n"
            "  folder, and the daily-bar cache under %LOCALAPPDATA%\\TradingBotV3. It writes\n"
            "  only its own CSV, and never to either store.\n"
            "  Re-run with --live to read them, or point TRADINGBOTV3_DATA_DIR at a scratch\n"
            "  copy first.",
            file=sys.stderr,
        )
        return 2

    rows, summary = build_rows(since=since)
    if not rows:
        print(
            f"No measured session since {since} carries a compression veto"
            f" ({summary.get('veto_count', 0)} veto(es) read,"
            f" {summary.get('pending', 0)} on sessions that are not complete yet)."
        )
        return 1

    print(render_report(rows, summary, since=since))
    out_dir = Path(args.out) if args.out else project_paths.get_diagnostics_dir()
    written = write_csv(rows, out_dir)
    print(f"\nwrote {written}")
    print(f"read at {datetime.now().isoformat(timespec='seconds')}; no store was modified")
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
