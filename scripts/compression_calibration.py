r"""Does the compression measure agree with the trader's eye? (packet PCT-3, item 3)

Trader, 2026-09-15: *"we need a way to measure for compression as the most COMMON
veto I have is a compression veto. we need to fine tune this so we stop getting
so many compressed picks."* Live counts 2026-08-20..09-15: 598 coded vetoes, of
which `compressed` (v3) plus `support_resistance_cluttered` (v1) are 214 - 36 %.
The scan already measures compression on every priority row and already docks
the score for it, so either the measure and the eye agree and something else is
wrong, or they do not agree and the threshold is wrong. **This report answers
that question and changes nothing.**

It is a read-only report, and it is the whole of its own authority: it proposes
no threshold, writes to no store the desk reads, touches no detector, no score,
no alert, no watchlist, no Focus list, no review queue and no `review_policy.json`.
The penalty decision is a SEPARATE ask, after the trader has read this.

    cd scripts
    python -m compression_calibration --since 2026-08-20            # scratch home
    python -m compression_calibration --since 2026-08-20 --live     # the real one

What it joins
-------------

1. Every **coded compression veto** in `trader_annotations.jsonl` since
   `--since`: `reason_code` in :data:`COMPRESSION_VETO_CODES`. The two codes are
   pooled HERE and by name. They do NOT pool in
   `veto_cohort.canonical_veto_cohort` - measured on this branch,
   `veto_v1_support_resistance_cluttered` stays itself, because v2 introduced
   `compressed` as a NEW definition and not a rename - so "189 + 25 = 214" is a
   sum this report makes for itself. A veto with a blank `reason_code` (136 of
   the live rows) is a veto and is NOT a compression veto: it lands in "rest".
2. That session's **shown population**, out of the setup tracker's own records.
   Only sessions carrying at least one compression veto are read, because the
   comparison is within a day: what did the trader refuse, against what else was
   on the screen beside it.
3. That symbol's **cached daily bars up to and including the session**, for the
   three measures the tracker row does not carry.

Seven measures, no verdict
--------------------------

Per row, at the session's completed close:

===============================  ==========================================
`compression_stdev_atr_ratio`    the scan's own three anchor ratios and its
`compression_range_atr_ratio`    0-3 `compression_score`, read off the
`compression_close_range_atr_ratio`  tracker record (PCT-3 item 1 puts them there)
`range10_atr14`                  the `d1_environment` rule, per symbol
`range20_atr14`                  the same over twenty sessions
`bollinger20_width_pct`          Bollinger(20, 2) relative width, as a
                                 percentile over the last 120 sessions
`atr14_atr50`                    is this name quiet against its own history?
===============================  ==========================================

Each prints `n = vetoed / rest`, the two medians and a rank-sum **AUC** - the
probability that a randomly chosen VETOED row scores HIGHER on that measure than
a randomly chosen rest row, ties counted as half. 0.5 is "this measure cannot
tell them apart"; 0.0 and 1.0 are perfect separation in one direction or the
other. A LOW AUC on every measure is a finding, not a failure.

Point-in-time (`plan.md` sec 5)
-------------------------------

No bar dated after the session is ever read: the cached frame is cut at the
session date before a single measure is computed. A measure that cannot be taken
is `None` and is left out of its own n - the unmeasured is shown, never assumed.

Read-only, and it says so first
-------------------------------

`project_paths.DATA_DIR` is read at CALL time (the `d1_environment_store`
idiom). If it resolves under the live home folder the run REFUSES before it
opens anything, names the folder and names `--live`. Every store below is opened
for reading only; the one file written is the CSV, under `--out` or
`%LOCALAPPDATA%\TradingBotV3\diagnostics\`.

The tracker JSON is 1.1 GB live, and `json.loads` on it is one of the three
causes of the 10 GB desk on 2026-08-27. :func:`iter_tracker_records` therefore
walks it one record at a time with `json.JSONDecoder.raw_decode`, holding the
file's text and ONE record rather than the whole object graph. The SQLite mirror
would be cheaper still and is deliberately not read: decision 0017 fences every
reader out of it until gate #57.
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
#: (they are different definitions); see the module docstring.
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

#: The measures taken from the tracker record rather than from bars.
_ANCHOR_MEASURE_KEYS = MEASURE_KEYS[:3]

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
# the three reads
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


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def iter_tracker_records(path: Path | str) -> Iterator[dict]:
    """Every tracker record, ONE at a time, without building the whole payload.

    The live file is 1.1 GB. `json.load` on it costs about ten gigabytes of
    dicts - the 2026-08-27 desk incident - so the top-level object is walked
    with the stdlib decoder's own `raw_decode`, which parses exactly one value
    from a given offset. Only the file's text and one record are ever live.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return
    decoder = json.JSONDecoder()
    index = _skip_whitespace(text, 0)
    if index >= len(text) or text[index] != "{":
        return
    index = _skip_whitespace(text, index + 1)
    while index < len(text) and text[index] != "}":
        try:
            key, index = decoder.raw_decode(text, index)
        except ValueError:
            return
        index = _skip_whitespace(text, index)
        if index >= len(text) or text[index] != ":":
            return
        index = _skip_whitespace(text, index + 1)
        if key in TRACKER_RECORD_SECTIONS and index < len(text) and text[index] == "{":
            index = _skip_whitespace(text, index + 1)
            while index < len(text) and text[index] != "}":
                try:
                    _record_key, index = decoder.raw_decode(text, index)
                    index = _skip_whitespace(text, index)
                    if index >= len(text) or text[index] != ":":
                        return
                    value, index = decoder.raw_decode(text, _skip_whitespace(text, index + 1))
                except ValueError:
                    return
                if isinstance(value, dict):
                    yield value
                index = _skip_whitespace(text, index)
                if index < len(text) and text[index] == ",":
                    index = _skip_whitespace(text, index + 1)
            index += 1
        else:
            try:
                _value, index = decoder.raw_decode(text, index)
            except ValueError:
                return
        index = _skip_whitespace(text, index)
        if index < len(text) and text[index] == ",":
            index = _skip_whitespace(text, index + 1)


def read_cached_daily_bars(symbol: str, *, through: str) -> list[dict]:
    """The desk's cached daily bars for `symbol`, oldest first, cut at `through`.

    `project_paths.DAILY_BARS_CACHE_DIR` is read at CALL time. Nothing here
    opens a provider or makes a network call, and **no bar dated after the
    session is ever returned** - the cut happens before any measure is taken.
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
            if not stamp or stamp > through:
                continue
            try:
                bars.append(
                    {
                        "dt": stamp,
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    bars.sort(key=lambda bar: bar["dt"])
    return bars


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


def _anchor_measures(record: dict) -> dict[str, float | None]:
    """The scan's own compression numbers off the tracker record (PCT-3 item 1).

    Top level first, then the record's `compression_summary`, then its entry
    feature snapshot - three places one scan writes the same reading, and a
    record written before item 1 landed simply has nothing to say.
    """
    sources = [record]
    for key in ("compression_summary", "entry_feature_snapshot"):
        nested = record.get(key)
        if isinstance(nested, dict):
            sources.append(nested)
    measures: dict[str, float | None] = {}
    for key in _ANCHOR_MEASURE_KEYS:
        measures[key] = next(
            (value for value in (_number(source.get(key)) for source in sources) if value is not None),
            None,
        )
    return measures


def _bar_measures(symbol: str, session: str) -> dict[str, float | None]:
    bars = read_cached_daily_bars(symbol, through=session)
    return {
        "range10_atr14": _range_over_atr(bars, RANGE10_SESSIONS),
        "range20_atr14": _range_over_atr(bars, RANGE20_SESSIONS),
        "bollinger20_width_pct": _bollinger_width_percentile(bars),
        "atr14_atr50": _atr_ratio(bars),
    }


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
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


def build_rows(*, since: str) -> list[dict]:
    """One row per setup shown on a session that carried a compression veto."""
    vetoes = read_compression_vetoes(project_paths.TRADER_ANNOTATIONS_FILE, since=since)
    sessions = {session for _symbol, session, _side in vetoes}
    if not sessions:
        return []

    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for record in iter_tracker_records(project_paths.MASTER_AVWAP_SETUP_TRACKER_FILE):
        session = _session_of(record)
        if session not in sessions:
            continue
        symbol = str(record.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        side = _side_of(record)
        identity = (symbol, session, side)
        if identity in seen:
            continue
        seen.add(identity)
        code = vetoes.get(identity) or vetoes.get((symbol, session, ""))
        row = {
            "symbol": symbol,
            "session_date": session,
            "side": side,
            "compressed_veto": bool(code),
            "reason_code": code or "",
            "compression_flag": _record_flag(record),
            "compression_score": record.get("compression_score", ""),
        }
        row.update(_anchor_measures(record))
        row.update(_bar_measures(symbol, session))
        rows.append(row)
    rows.sort(key=lambda row: (row["session_date"], row["symbol"], row["side"]))
    return rows


def _format(value: float | None) -> str:
    return "nan" if value is None else f"{value:.4f}"


def render_report(rows: Sequence[dict], *, since: str) -> str:
    """The printed report. One block per measure, then the flag's hit rate."""
    vetoed = [row for row in rows if row["compressed_veto"]]
    rest = [row for row in rows if not row["compressed_veto"]]
    sessions = sorted({row["session_date"] for row in rows})
    lines = [
        "compression calibration - read-only, proposes nothing",
        (
            f"window: {since} .. {sessions[-1] if sessions else since}"
            f" ({len(sessions)} session(s) carrying a compression veto)"
        ),
        f"joined: {len(rows)} shown row(s) - {len(vetoed)} vetoed, {len(rest)} rest",
        f"codes pooled by name: {', '.join(sorted(COMPRESSION_VETO_CODES))}",
        "auc = P(a vetoed row reads HIGHER than a rest row), ties half; 0.5 tells them apart not at all",
        "",
    ]
    for key in MEASURE_KEYS:
        left = [value for value in (row.get(key) for row in vetoed) if value is not None]
        right = [value for value in (row.get(key) for row in rest) if value is not None]
        counts = f"n = {len(left)} / {len(right)}"
        if (len(left), len(right)) != (len(vetoed), len(rest)):
            counts += f"   measured (of {len(vetoed)} / {len(rest)} joined)"
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
    "reason_code",
    "compression_flag",
    "compression_score",
    *MEASURE_KEYS,
)


def write_csv(rows: Sequence[dict], out_dir: Path, *, stamp: date | None = None) -> Path:
    """One row per shown row. Temp-and-rename, so a half-written file never lands."""
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
        help="allow reading the trader's live home folder (this report never writes to it)",
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
            "Refusing to read the live home folder without --live.\n"
            f"  project_paths.DATA_DIR resolves under {live}\n"
            "  Re-run with --live to read it (this report is read-only either way),\n"
            "  or point TRADINGBOTV3_DATA_DIR at a scratch copy first.",
            file=sys.stderr,
        )
        return 2

    rows = build_rows(since=since)
    if not rows:
        print(
            f"No session since {since} carries both a compression veto and a tracker population - "
            "nothing to compare."
        )
        return 1

    print(render_report(rows, since=since))
    out_dir = Path(args.out) if args.out else project_paths.get_diagnostics_dir()
    written = write_csv(rows, out_dir)
    print(f"\nwrote {written}")
    print(f"read at {datetime.now().isoformat(timespec='seconds')}; no store was modified")
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
