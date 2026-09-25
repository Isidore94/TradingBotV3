"""Options chase helper: the out-of-the-money option to chase a high-volume Movers pop.

Decision support only: it never places an order. Given one Pop row (symbol, side,
last, atr, rvol, move15) and an option chain snapshot, it picks the nearest weekly
expiry with at least MIN_SESSIONS_TO_EXPIRY sessions to go, the OTM strike in the
pop's direction with |delta| nearest TARGET_DELTA (DELTA_MIN..DELTA_MAX accepted),
and reports bid, ask, mid, spread % of mid, IV and IV vs the 20-session realized
vol. It refuses, with a reason, on a wide spread, no delta in range, an empty chain
or RVOL under MIN_RVOL. Every missing input is "unknown", never a guess.

A chain snapshot is a plain mapping::

    {"expiries": ["2026-10-02", ...], "strikes": [25.0, 27.0, ...],
     "quotes": [{"expiry": "2026-10-02", "strike": 27.0, "right": "C",
                 "bid": 0.85, "ask": 0.95, "delta": 0.26, "iv": 0.62}, ...]}

The log (`OPTIONS_CHASE_LOG_FILE`) is evidence only: flag rows, then outcome rows
at +30 and +60 minutes and at the close. `python scripts/options_chase.py --summary`
prints counts and the median option mid change where known. Pure except
`append_records`, `load_records`, `read_daily_closes` and the CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import statistics
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

#: A pop is chased only at this relative volume or more.
MIN_RVOL = 2.0
#: |delta| aimed for, and the band accepted around it.
TARGET_DELTA = 0.25
DELTA_MIN = 0.15
DELTA_MAX = 0.35
#: A spread wider than this share of the mid (percent) is refused.
MAX_SPREAD_PCT = 15.0
#: The expiry must have at least this many sessions after today (expiry day included).
MIN_SESSIONS_TO_EXPIRY = 2
#: Realized vol: close-to-close log returns over this many sessions, annualised.
HV_SESSIONS = 20
SESSIONS_PER_YEAR = 252
#: Quote plan: at most this many strikes are quoted per symbol.
QUOTE_STRIKES = 8
#: Quote plan window in standard deviations out of the money (|delta| ~0.40..~0.10).
PLAN_Z_MIN = 0.25
PLAN_Z_MAX = 1.3
#: Quote plan vol when the name's realized vol is unknown, and its floor otherwise.
#: Only chooses WHICH strikes to quote; the picked delta is always IB's.
PLAN_FALLBACK_VOL = 0.6
PLAN_MIN_VOL = 0.3
#: Outcome horizons after a flag (minutes), and the close row.
OUTCOME_MINUTES = (30, 60)
CLOSE_LABEL = "close"
SESSION_CLOSE = time(16, 0)

STATUS_CANDIDATE = "candidate"
STATUS_REFUSED = "refused"
STATUS_NO_DATA = "no_data"


# ---------------------------------------------------------------- small helpers
def _num(value: Any) -> float | None:
    """A finite float, else None (unknown)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip().replace("-", "")
    if len(text) < 8:
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def right_for(side: str) -> str:
    """Calls chase a long pop, puts a short one."""
    return "P" if str(side or "").lower() == "short" else "C"


def _sessions_after(today: date, expiry: date) -> int | None:
    """Sessions after today up to and including expiry; None when the calendar can't say."""
    try:
        from market_calendar import trading_days_between

        return int(trading_days_between(today, expiry))
    except Exception:
        return None


# ---------------------------------------------------------------- expiry and vol
def weekly_expiries(expiries: Iterable[Any]) -> list[date]:
    """The last listed expiry of each calendar week (Friday, or Thursday on a holiday)."""
    by_week: dict[tuple[int, int], date] = {}
    for value in expiries or ():
        day = _day(value)
        if day is None:
            continue
        key = tuple(day.isocalendar()[:2])
        if key not in by_week or day > by_week[key]:
            by_week[key] = day
    return sorted(by_week.values())


def pick_expiry(
    expiries: Iterable[Any],
    today: date,
    *,
    sessions_after: Callable[[date, date], int | None] = _sessions_after,
) -> tuple[date | None, int | None, str]:
    """(expiry, sessions to go, reason): nearest weekly with MIN_SESSIONS_TO_EXPIRY+ to go, else the next."""
    weeklies = [d for d in weekly_expiries(expiries) if d > today]
    if not weeklies:
        return None, None, "no weekly expiry after today"
    for expiry in weeklies:
        sessions = sessions_after(today, expiry)
        if sessions is None:
            return None, None, "session calendar unknown"
        if sessions >= MIN_SESSIONS_TO_EXPIRY:
            return expiry, sessions, ""
    return None, None, f"no weekly expiry with {MIN_SESSIONS_TO_EXPIRY}+ sessions to go"


def realized_vol(closes: Sequence[Any], *, sessions: int = HV_SESSIONS) -> float | None:
    """Annualised close-to-close vol over the last `sessions` returns; None when too short."""
    values = [_num(c) for c in closes or ()]
    values = [v for v in values if v is not None and v > 0]
    if len(values) < sessions + 1:
        return None
    window = values[-(sessions + 1):]
    returns = [math.log(b / a) for a, b in zip(window[:-1], window[1:], strict=True)]
    if len(returns) < 2:
        return None
    return statistics.stdev(returns) * math.sqrt(SESSIONS_PER_YEAR)


def read_daily_closes(path: Path, *, before: date) -> list[float]:
    """Closes from one cached daily-bar CSV, oldest first, sessions before `before` only."""
    rows: list[tuple[date, float]] = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                day = None
                for key in ("datetime", "date", "dt", "timestamp"):
                    if row.get(key):
                        day = _day(str(row[key])[:10])
                        break
                close = _num(row.get("close"))
                if day is not None and close is not None and day < before:
                    rows.append((day, close))
    except OSError:
        return []
    rows.sort(key=lambda item: item[0])
    return [close for _day_, close in rows]


# ---------------------------------------------------------------- quote plan
def quote_plan(
    last: Any,
    side: str,
    strikes: Iterable[Any],
    *,
    sessions: int | None,
    hv: float | None,
    max_strikes: int = QUOTE_STRIKES,
) -> list[float]:
    """Which OTM strikes to quote: up to `max_strikes` spread over the ~0.40..0.10 delta
    window estimated from realized vol; the nearest OTM strikes when the window is empty."""
    price = _num(last)
    if price is None or price <= 0:
        return []
    call = right_for(side) == "C"
    otm = sorted(
        {s for s in (_num(x) for x in strikes or ()) if s is not None and (s > price if call else s < price)},
        key=lambda s: abs(s - price),
    )
    if not otm:
        return []
    vol = max(hv, PLAN_MIN_VOL) if hv else PLAN_FALLBACK_VOL
    sigma = price * vol * math.sqrt(max(1, sessions or 5) / SESSIONS_PER_YEAR)
    low, high = PLAN_Z_MIN * sigma, PLAN_Z_MAX * sigma
    window = [s for s in otm if low <= abs(s - price) <= high]
    if len(window) > max_strikes:
        step = (len(window) - 1) / (max_strikes - 1)
        window = [window[round(i * step)] for i in range(max_strikes)]
    if not window:
        target = 0.674 * sigma
        window = sorted(otm, key=lambda s: abs(abs(s - price) - target))[:3]
    return sorted(window)


# ---------------------------------------------------------------- the picker
def _base(pop: Mapping[str, Any]) -> dict[str, Any]:
    side = "short" if str(pop.get("side") or pop.get("_side") or "").lower() == "short" else "long"
    return {
        "symbol": str(pop.get("symbol") or "").strip().upper(),
        "side": side,
        "right": right_for(side),
        "last": _num(pop.get("last")),
        "atr": _num(pop.get("atr")),
        "rvol": _num(pop.get("rvol")),
        "move15": _num(pop.get("move15") if "move15" in pop else pop.get("move15_pct")),
        "expiry": None, "sessions_to_expiry": None, "strike": None, "delta": None,
        "bid": None, "ask": None, "mid": None, "spread_pct": None,
        "iv": None, "hv": None, "iv_vs_hv": None, "reason": "",
    }


def refuse(result: dict[str, Any], reason: str, *, status: str = STATUS_REFUSED) -> dict[str, Any]:
    result["status"] = status
    result["reason"] = reason
    return result


def no_data(pop: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """The row for a name the service could not get option data for."""
    return refuse(_base(pop), reason, status=STATUS_NO_DATA)


def pick_candidate(
    pop: Mapping[str, Any],
    chain: Mapping[str, Any] | None,
    *,
    today: date,
    hv: float | None = None,
    sessions_after: Callable[[date, date], int | None] = _sessions_after,
) -> dict[str, Any]:
    """The chase contract for one Pop row, or a refusal with its reason."""
    result = _base(pop)
    result["hv"] = _num(hv)
    rvol = result["rvol"]
    if rvol is None:
        return refuse(result, "RVOL unknown")
    if rvol < MIN_RVOL:
        return refuse(result, f"RVOL {rvol:.1f} < {MIN_RVOL:g}")
    last = result["last"]
    if last is None or last <= 0:
        return refuse(result, "last price unknown")
    chain = chain or {}
    quotes = [q for q in (chain.get("quotes") or ()) if isinstance(q, Mapping)]
    if not chain.get("expiries") or not quotes:
        return refuse(result, "empty chain")
    expiry, sessions, why = pick_expiry(chain.get("expiries") or (), today, sessions_after=sessions_after)
    if expiry is None:
        return refuse(result, why)
    result["expiry"], result["sessions_to_expiry"] = expiry.isoformat(), sessions
    call = result["right"] == "C"
    at_expiry = []
    for quote in quotes:
        strike = _num(quote.get("strike"))
        if (_day(quote.get("expiry")) != expiry or strike is None
                or str(quote.get("right") or "").upper()[:1] != result["right"]
                or not (strike > last if call else strike < last)):
            continue
        at_expiry.append((strike, quote))
    if not at_expiry:
        return refuse(result, "no OTM quote at that expiry")
    with_delta = [(s, q, abs(d)) for s, q in at_expiry if (d := _num(q.get("delta"))) is not None]
    if not with_delta:
        return refuse(result, "no delta quoted")
    in_band = [item for item in with_delta if DELTA_MIN <= item[2] <= DELTA_MAX]
    if not in_band:
        return refuse(result, f"no delta in {DELTA_MIN:.2f}-{DELTA_MAX:.2f}")
    strike, quote, delta = min(in_band, key=lambda i: (abs(i[2] - TARGET_DELTA), abs(i[0] - last)))
    result["strike"], result["delta"] = strike, delta
    bid, ask, iv = _num(quote.get("bid")), _num(quote.get("ask")), _num(quote.get("iv"))
    result["bid"] = bid if bid is not None and bid >= 0 else None
    result["ask"] = ask if ask is not None and ask > 0 else None
    result["iv"] = iv if iv is not None and iv > 0 else None
    if result["iv"] is not None and result["hv"]:
        result["iv_vs_hv"] = result["iv"] / result["hv"]
    if result["bid"] is None or result["ask"] is None or result["ask"] < result["bid"]:
        return refuse(result, "no two-sided quote")
    mid = (result["bid"] + result["ask"]) / 2.0
    result["mid"] = mid
    result["spread_pct"] = (result["ask"] - result["bid"]) / mid * 100.0 if mid > 0 else None
    if result["spread_pct"] is None:
        return refuse(result, "spread unknown")
    if result["spread_pct"] > MAX_SPREAD_PCT:
        return refuse(result, f"spread {result['spread_pct']:.0f}% of mid > {MAX_SPREAD_PCT:g}%")
    result["status"] = STATUS_CANDIDATE
    return result


# ---------------------------------------------------------------- display text
def _strike_text(strike: float | None) -> str:
    if strike is None:
        return "?"
    return f"{strike:g}"


def _expiry_text(expiry: Any) -> str:
    day = _day(expiry)
    return day.strftime("%m/%d") if day else "?"


def _price(value: float | None) -> str:
    return "?" if value is None else f"{value:.2f}"


def contract_text(result: Mapping[str, Any]) -> str:
    """'27C 10/03 · 0.85x0.95 · 11% · IV 62 (HV 41)'; unknown parts read '?' / '—'."""
    head = f"{_strike_text(result.get('strike'))}{result.get('right') or '?'} {_expiry_text(result.get('expiry'))}"
    spread = result.get("spread_pct")
    iv, hv = result.get("iv"), result.get("hv")
    iv_text = "IV —" if iv is None else f"IV {iv * 100:.0f}"
    hv_text = "HV —" if hv is None else f"HV {hv * 100:.0f}"
    return (f"{head} · {_price(result.get('bid'))}x{_price(result.get('ask'))} · "
            f"{'—' if spread is None else f'{spread:.0f}%'} · {iv_text} ({hv_text})")


def cell_text(result: Mapping[str, Any] | None) -> str:
    """The board's Opt cell."""
    if not result:
        return "—"
    status = result.get("status")
    if status == STATUS_CANDIDATE:
        return contract_text(result)
    if status == STATUS_NO_DATA:
        return f"no option data ({result.get('reason') or 'unknown'})"
    return f"no chase ({result.get('reason') or 'unknown'})"


def detail_text(result: Mapping[str, Any] | None) -> str:
    """The Opt cell's hover: every field, and the refusal reason."""
    if not result:
        return "Options chase: not checked (RVOL under 2, or outside the top names this tick)."
    lines = [f"Options chase {result.get('symbol') or ''} {result.get('side') or ''}".rstrip()]
    status = result.get("status")
    if status == STATUS_CANDIDATE:
        lines.append("candidate: " + contract_text(result))
    elif status == STATUS_NO_DATA:
        lines.append(f"no option data: {result.get('reason') or 'unknown'}")
    else:
        lines.append(f"refused: {result.get('reason') or 'unknown'}")
    if result.get("strike") is not None:
        lines.append("contract: " + contract_text(result))

    def show(value, fmt):
        return "unknown" if value is None else fmt.format(value)

    lines.append(
        f"expiry {result.get('expiry') or 'unknown'} ({show(result.get('sessions_to_expiry'), '{} sessions')})"
        f" · delta {show(result.get('delta'), '{:.2f}')} · mid {show(result.get('mid'), '{:.2f}')}"
    )
    lines.append(
        f"IV {show(result.get('iv'), '{:.0%}')} · HV20 {show(result.get('hv'), '{:.0%}')}"
        f" · IV/HV {show(result.get('iv_vs_hv'), '{:.2f}x')}"
    )
    lines.append(
        f"last {show(result.get('last'), '{:.2f}')} · ATR {show(result.get('atr'), '{:.2f}')}"
        f" · RVOL {show(result.get('rvol'), '{:.1f}x')} · 15m {show(result.get('move15'), '{:+.2f}%')}"
    )
    if result.get("as_of"):
        lines.append(f"quoted {result['as_of']}")
    lines.append("Decision support only: nothing is ever ordered. Click copies the text.")
    return "\n".join(lines)


# ---------------------------------------------------------------- log rows
def _clock(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.astimezone()


def flag_record(result: Mapping[str, Any], *, now: datetime) -> dict[str, Any]:
    """One flag row: the candidate or refusal as shown, with the option mid at flag."""
    moment = _clock(now)
    session = moment.astimezone(NY_TZ).date().isoformat()
    stamp = moment.isoformat(timespec="seconds")
    row = {k: result.get(k) for k in (
        "symbol", "side", "status", "reason", "right", "expiry", "sessions_to_expiry", "strike",
        "delta", "bid", "ask", "mid", "spread_pct", "iv", "hv", "iv_vs_hv", "last", "atr",
        "rvol", "move15", "as_of")}
    row.update({"kind": "flag", "session": session, "flagged_at": stamp,
                "flag_id": f"{session}|{row['symbol']}|{row['side']}|{stamp}"})
    return row


def outcome_record(flag: Mapping[str, Any], label: str, *, now: datetime, last: float | None,
                   option_mid: float | None) -> dict[str, Any]:
    """The underlying move in ATRs since the flag (signed, and in the chase direction)
    and, when a fresh quote is cached, the option mid then."""
    start, atr = _num(flag.get("last")), _num(flag.get("atr"))
    price = _num(last)
    move = (price - start) / atr if None not in (price, start, atr) and atr > 0 else None
    side_move = None if move is None else (-move if flag.get("side") == "short" else move)
    mid0, mid = _num(flag.get("mid")), _num(option_mid)
    change = (mid - mid0) / mid0 * 100.0 if mid is not None and mid0 else None
    return {
        "kind": "outcome", "flag_id": flag.get("flag_id"), "session": flag.get("session"),
        "symbol": flag.get("symbol"), "side": flag.get("side"), "status": flag.get("status"),
        "horizon": label, "at": _clock(now).isoformat(timespec="seconds"), "last": price,
        "move_atr": move, "move_atr_chase": side_move, "option_mid": mid,
        "option_mid_change_pct": change,
    }


def _flag_key(result: Mapping[str, Any]) -> tuple:
    """Same status, same contract and same kind of reason (numbers ignored) = same answer."""
    reason = re.sub(r"[0-9.]+", "#", str(result.get("reason") or ""))
    return (result.get("status"), reason, result.get("expiry"), result.get("strike"),
            result.get("right"))


class ChaseOutcomeTracker:
    """Decides which results are logged and owes each flag its +30/+60/close rows.

    A symbol/side is logged again only when its answer changes (another contract, or
    another refusal reason), so a pop that stays on the board is one flag, not one per
    tick. Pending outcomes live in memory: a desk restart loses them (evidence only)."""

    def __init__(self) -> None:
        self._session: str = ""
        self._last_key: dict[str, tuple] = {}
        self._pending: list[dict[str, Any]] = []

    def _roll(self, now: datetime) -> str:
        session = _clock(now).astimezone(NY_TZ).date().isoformat()
        if session != self._session:
            self._session, self._last_key, self._pending = session, {}, []
        return session

    def flag(self, results: Iterable[Mapping[str, Any]], *, now: datetime) -> list[dict[str, Any]]:
        """Flag rows for results whose answer is new for their symbol/side this session."""
        self._roll(now)
        rows = []
        for result in results:
            if result.get("status") not in (STATUS_CANDIDATE, STATUS_REFUSED):
                continue
            name = f"{result.get('symbol')}|{result.get('side')}"
            key = _flag_key(result)
            if self._last_key.get(name) == key:
                continue
            self._last_key[name] = key
            row = flag_record(result, now=now)
            rows.append(row)
            self._pending.append({"flag": row, "left": [*(f"+{m}m" for m in OUTCOME_MINUTES), CLOSE_LABEL]})
        return rows

    def observe(
        self,
        prices: Mapping[str, float],
        *,
        now: datetime,
        option_mid: Callable[[Mapping[str, Any]], float | None] = lambda _flag: None,
    ) -> list[dict[str, Any]]:
        """Outcome rows that fell due by `now`, priced from `prices` (last completed close)."""
        self._roll(now)
        moment = _clock(now)
        ny = moment.astimezone(NY_TZ)
        closed = ny.time() >= SESSION_CLOSE
        rows: list[dict[str, Any]] = []
        keep = []
        for item in self._pending:
            flag = item["flag"]
            flagged = datetime.fromisoformat(flag["flagged_at"])
            symbol = flag.get("symbol")
            due = []
            for label in item["left"]:
                if label == CLOSE_LABEL:
                    if closed:
                        due.append(label)
                elif moment >= flagged + timedelta(minutes=int(label[1:-1])) or closed:
                    due.append(label)
            if due and symbol in prices:
                mid = option_mid(flag) if flag.get("status") == STATUS_CANDIDATE else None
                for label in due:
                    if label != CLOSE_LABEL and closed and moment < flagged + timedelta(minutes=int(label[1:-1])):
                        continue  # the close came first: that horizon is never measured
                    rows.append(outcome_record(flag, label, now=now, last=prices.get(symbol), option_mid=mid))
                item["left"] = [label for label in item["left"] if label not in due]
            if item["left"] and not closed:
                keep.append(item)
        self._pending = keep
        return rows


def append_records(path: Path, records: Iterable[Mapping[str, Any]]) -> bool:
    """Append rows. A failed write loses these rows and logs a warning; never raises."""
    rows = list(records)
    if not rows:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return True
    except Exception as exc:
        logging.warning("Options chase log write failed (%s rows lost): %s", len(rows), exc)
        return False


def load_records(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except FileNotFoundError:
        return []
    return out


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Flag counts (candidates, refusals by reason) and, per horizon, the outcome count,
    how many had an option mid, the median mid change % and the median chase-direction ATR move."""
    flags = candidates = 0
    reasons: dict[str, int] = {}
    horizons: dict[str, dict[str, list[float] | int]] = {}
    for row in records:
        if row.get("kind") == "flag":
            flags += 1
            if row.get("status") == STATUS_CANDIDATE:
                candidates += 1
            else:
                reason = str(row.get("reason") or "unknown").split(" ")[0]
                reasons[reason] = reasons.get(reason, 0) + 1
        elif row.get("kind") == "outcome":
            slot = horizons.setdefault(str(row.get("horizon")), {"n": 0, "mid": [], "atr": []})
            slot["n"] += 1  # type: ignore[operator]
            change = _num(row.get("option_mid_change_pct"))
            if change is not None:
                slot["mid"].append(change)  # type: ignore[union-attr]
            move = _num(row.get("move_atr_chase"))
            if move is not None:
                slot["atr"].append(move)  # type: ignore[union-attr]
    out_h = {}
    for label, slot in horizons.items():
        mids, atrs = slot["mid"], slot["atr"]
        out_h[label] = {
            "outcomes": slot["n"],
            "with_option_mid": len(mids),  # type: ignore[arg-type]
            "median_option_mid_change_pct": statistics.median(mids) if mids else None,
            "median_move_atr_chase": statistics.median(atrs) if atrs else None,
        }
    return {"flags": flags, "candidates": candidates, "refused": flags - candidates,
            "refusals_by_reason": dict(sorted(reasons.items())), "horizons": out_h}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Options chase log summary")
    parser.add_argument("--summary", action="store_true", help="print the summary as JSON")
    parser.add_argument("--path", help="log file (default: project_paths.OPTIONS_CHASE_LOG_FILE)")
    args = parser.parse_args(argv)
    if not args.summary:
        parser.print_help()
        return 2
    if args.path:
        path = Path(args.path)
    else:
        from project_paths import OPTIONS_CHASE_LOG_FILE

        path = Path(OPTIONS_CHASE_LOG_FILE)
    print(json.dumps(summarize(load_records(path)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
