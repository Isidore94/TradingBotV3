"""The warehouse build job (plan sec 8.4, Phase 8).

No new process, no daemon, no service: one CLI invoked post-scan or at EOD,
registered in the existing job ledger, holding a **single-flight lock** so a
manual run during a scheduled build refuses with a clear message instead of
two writers racing.

    python -m scripts.research_warehouse.cli build
    python -m scripts.research_warehouse.cli status
    python -m scripts.research_warehouse.cli restore-check --target <dir>

Every command is a no-op with a clear message when ``research_store_dir`` is
unset. The build job is resumable by construction: each step is idempotent, so
an interrupted run (sleep, wake, TWS restart, power loss) simply repeats the
steps that did not finish and rewrites nothing that did.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:  # package import
    from . import (
        backup as backup_mod,
        config,
        exchange_calendar as xcal,
        features,
        market_bias_context,
        occurrences,
        outcome_coverage,
        outcomes,
        queries,
        schemas,
        tracker_adapter,
        trial_ledger,
    )
    from . import after_like, like_links
    from .aggregate import build_derived_bars, build_trading_sessions, build_weekly_bars
    from .ingest_existing import ingest_daily_bars, run_bronze_ingest, run_daily_snapshots
    from .manifest import utc_now
    from .spool import seal_spool
    from .store import LakeIntegrityError, ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import backup as backup_mod  # type: ignore
    import config  # type: ignore
    import exchange_calendar as xcal  # type: ignore
    import features  # type: ignore
    import market_bias_context  # type: ignore
    import occurrences  # type: ignore
    import outcomes  # type: ignore
    import queries  # type: ignore
    import schemas  # type: ignore
    import tracker_adapter  # type: ignore
    from aggregate import build_derived_bars, build_trading_sessions, build_weekly_bars  # type: ignore
    from ingest_existing import ingest_daily_bars, run_bronze_ingest, run_daily_snapshots  # type: ignore
    from manifest import utc_now  # type: ignore
    from spool import seal_spool  # type: ignore
    from store import LakeIntegrityError, ResearchStore  # type: ignore
    import outcome_coverage  # type: ignore
    import trial_ledger  # type: ignore
    import after_like  # type: ignore
    import like_links  # type: ignore

LOCK_NAME = "research_build.lock"
JOB_TYPE = "research_warehouse_build"
OUTCOME_BUCKETS = 32
OUTCOME_BUCKET_MIN_SYMBOLS = 64


class SingleFlightError(RuntimeError):
    """Another build already holds the lock."""


def _outcome_bucket(day: date, stamp: datetime) -> int:
    """A same-slot retry is stable; consecutive days cover every bucket."""
    return (day.toordinal() + int(stamp.hour)) % OUTCOME_BUCKETS


def _lock_path() -> Path:
    return Path(config.research_spool_dir()) / LOCK_NAME


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) on Windows is TerminateProcess, not a liveness probe:
        # probing the lock holder would kill the running build. Query instead.
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        ERROR_ACCESS_DENIED = 5
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Access denied means the pid exists but is not ours: treat as alive.
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


@contextmanager
def single_flight(lock_path: Path | None = None):
    """One build at a time. A dead holder's lock is reclaimed, not obeyed."""
    path = Path(lock_path) if lock_path is not None else _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = {}
        try:
            holder = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            holder = {}
        pid = int(holder.get("pid") or 0)
        # A live holder refuses even when it is this same process: a build
        # must never nest, or two passes write the lake concurrently.
        if pid and _process_alive(pid):
            raise SingleFlightError(
                f"a research warehouse build is already running (pid {pid}, started "
                f"{holder.get('started_at', 'unknown')}). Wait for it, or stop it first."
            )
        # The holder is gone (crash, power loss): reclaim rather than wedge.
        path.unlink(missing_ok=True)
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(handle, json.dumps({"pid": os.getpid(), "started_at": utc_now().isoformat()}).encode("utf-8"))
        os.close(handle)
        yield path
    finally:
        path.unlink(missing_ok=True)


def _record_job(state: str, detail: dict | None = None) -> None:
    """Register the run in the existing job ledger; never fatal."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from job_ledger import JobLedger  # type: ignore

        ledger = JobLedger()
        recorder = getattr(ledger, "record_event", None) or getattr(ledger, "append", None)
        if callable(recorder):
            recorder({"job_type": JOB_TYPE, "state": state, **(detail or {})})
    except Exception:
        pass  # telemetry must never break a build


@dataclass
class BuildReport:
    status: str = "OK"  # OK | DISABLED | REFUSED
    steps: dict = field(default_factory=dict)
    message: str = ""


#: Bronze payloads that are JSON *text*, whatever the source file was. A CSV
#: row is wrapped as ``CSV_ROW`` but its payload is a JSON object of the
#: header-to-value mapping, so it decodes the same way.
_JSON_PAYLOAD_FORMATS = {
    schemas.BRONZE_FORMAT_JSON,
    schemas.BRONZE_FORMAT_JSONL,
    schemas.BRONZE_FORMAT_CSV_ROW,
}


def _bronze_payloads(store: ResearchStore, dataset: str) -> list[dict]:
    """Decode a bronze wrap's verbatim payloads back into dicts."""
    rows = []
    try:
        table = store.read_table(dataset, columns=["payload", "payload_format"])
    except Exception:
        return rows
    for payload, fmt in zip(
        table.column("payload").to_pylist(), table.column("payload_format").to_pylist()
    ):
        if str(fmt or "").upper() not in _JSON_PAYLOAD_FORMATS:
            continue
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, dict):
            rows.append(decoded)
    return rows


def anchors_from_bronze(store: ResearchStore) -> list[dict]:
    """Earnings anchors for ``anchor_instance``, from the bronze CSV wrap.

    The trader's ``earnings_avwap_anchors.csv`` carries one current anchor per
    ticker (`ticker`, `anchor_date`, ...), and bronze keeps every version of
    that file it has ever seen. LD-09 scopes the slice to *current and
    previous* earnings, so per ticker the newest distinct ``anchor_date``
    becomes ``EARNINGS_CURRENT`` and the one before it ``EARNINGS_PREVIOUS``;
    older ones are history, not slice anchors. Nothing is invented - a ticker
    bronze has only ever seen once simply has no previous anchor.
    """
    by_symbol: dict[str, set] = {}
    for payload in _bronze_payloads(store, "bronze_earnings_avwap_anchors"):
        symbol = str(payload.get("ticker") or payload.get("symbol") or "").strip().upper()
        raw = str(payload.get("anchor_date") or "").strip()
        if not symbol or not raw:
            continue
        try:
            day = date.fromisoformat(raw[:10])
        except ValueError:
            continue
        by_symbol.setdefault(symbol, set()).add(day)

    anchors = []
    for symbol, days in sorted(by_symbol.items()):
        ordered = sorted(days, reverse=True)
        for anchor_type, day in zip(
            (features.ANCHOR_TYPE_CURRENT, features.ANCHOR_TYPE_PREVIOUS), ordered
        ):
            anchors.append(
                {
                    "symbol": symbol,
                    "anchor_type": anchor_type,
                    "anchor_bar_date": day,
                    "source": "earnings_avwap_anchors.csv",
                }
            )
    return anchors


def cohort_for(store: ResearchStore, day: date) -> list[str]:
    """The session's captured cohort, from its own membership snapshot.

    Read from ``universe_membership_daily`` rather than from today's watchlist
    files: LD-05 makes that snapshot the point-in-time truth about who was a
    member, and a rebuild months later must see the same cohort.
    """
    symbols = set()
    for row in store.read_table(
        "universe_membership_daily", f"year={day.year}", columns=["session_date", "symbol"]
    ).to_pylist():
        session_day = row.get("session_date")
        if isinstance(session_day, datetime):
            session_day = session_day.date()
        if session_day == day:
            value = str(row.get("symbol") or "").strip().upper()
            if value:
                symbols.add(value)
    return sorted(symbols)


def anchor_dates_by_symbol(store: ResearchStore, day: date) -> dict:
    """Current-earnings anchor per symbol for the daily AVWAP block, STAMPED.

    Returns ``{symbol: features.AnchorChoice}``: the newest anchor bar on or
    before ``day`` (an anchor that had not happened yet is not knowable and is
    still excluded), plus whether the lake KNEW it then - the row's own
    ``system_from``, read market-local (sec 6.2: "an anchor is available once
    observed, not retroactively at its bar").

    Q2.1/BD-99: the 2026-09-04 bridge back-fills ~2,200 anchors whose bars are
    months old and whose knowledge stamp is that night. Without this
    distinction a snapshot rebuilt for an August session would present them as
    something the desk knew that day. Where one bar date carries several rows
    the EARLIEST ``system_from`` wins - the first time it became knowable.
    """
    chosen: dict[str, features.AnchorChoice] = {}
    known_from: dict[str, datetime | None] = {}
    for year in (day.year, day.year - 1):
        for row in store.read_table("anchor_instance", f"year={year}").to_pylist():
            if str(row.get("anchor_type") or "") != features.ANCHOR_TYPE_CURRENT:
                continue
            bar_date = row.get("anchor_bar_date")
            if isinstance(bar_date, datetime):
                bar_date = bar_date.date()
            if bar_date is None or bar_date > day:
                continue  # an anchor that had not happened yet is not knowable
            symbol = str(row.get("symbol") or "")
            if not symbol:
                continue
            current = chosen.get(symbol)
            stamp = row.get("system_from")
            if current is not None:
                if bar_date < current.anchor_bar_date:
                    continue
                if bar_date == current.anchor_bar_date and not _earlier_stamp(stamp, known_from.get(symbol)):
                    continue
            chosen[symbol] = features.AnchorChoice(bar_date, _anchor_knowledge_for(stamp, day))
            known_from[symbol] = stamp
    return chosen


def _earlier_stamp(candidate, current) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    return candidate < current


def _anchor_knowledge_for(system_from, day: date) -> str:
    """``observed`` only when the row's knowledge stamp lands on or before the
    session, market-local. A missing stamp establishes nothing, so it reads as
    ``reconstructed`` - uncertainty is never confirmation."""
    if not isinstance(system_from, datetime):
        return features.ANCHOR_KNOWLEDGE_RECONSTRUCTED
    stamped = system_from if system_from.tzinfo else system_from.replace(tzinfo=timezone.utc)
    local_day = stamped.astimezone(xcal.EXCHANGE_TZ).date()
    return (
        features.ANCHOR_KNOWLEDGE_OBSERVED
        if local_day <= day
        else features.ANCHOR_KNOWLEDGE_RECONSTRUCTED
    )


def _run_backups(store: ResearchStore, stamp: datetime) -> dict:
    """Class A/B copies, but only to destinations the trader configured."""
    steps: dict = {}
    class_a = config.backup_class_a_dirs()
    if class_a:
        steps["class_a"] = vars(backup_mod.backup_class_a(store, class_a, now=stamp))
    else:
        steps["class_a"] = {
            "status": "NO_TARGET",
            "message": (
                f"no Class-A backup target: set {config.BACKUP_CLASS_A_SETTING} in "
                f"local_settings.json (or {config.BACKUP_CLASS_A_ENV})."
            ),
        }
    class_b = config.backup_class_b_dir()
    if class_b is not None:
        steps["class_b"] = vars(backup_mod.backup_class_b(store, class_b, now=stamp))
    else:
        steps["class_b"] = {
            "status": "NO_TARGET",
            "message": (
                f"no Class-B backup target: set {config.BACKUP_CLASS_B_SETTING} in "
                f"local_settings.json (or {config.BACKUP_CLASS_B_ENV}). It must be a "
                "second physical disk, never the lake's own server."
            ),
        }
    return steps


def _m5_partitions_for(known: dict, day: date) -> list[str]:
    """Month partitions holding the M5 bars these occurrences can be simulated on.

    ``known`` spans two years of occurrences and BD-53 re-simulates every
    non-terminal one on every build, but the M5 read was the *build day's*
    month alone. An intraday occurrence triggered in any earlier month was
    therefore re-simulated against an empty archive every night, and drew
    conclusions from that absence rather than from its own session (BD-69).
    The trigger's previous, own, and following months are read. The previous
    month supplies ATR warm-up; the following month supplies the 18-session
    path (and a winter session's ETH tail, BD-66).
    """
    months = {f"month={day:%Y-%m}"}
    for row in known.values():
        trigger = row.get("trigger_at")
        if not isinstance(trigger, datetime):
            continue
        entry = trigger.date()
        months.add(f"month={entry - timedelta(days=31):%Y-%m}")
        months.add(f"month={entry:%Y-%m}")
        months.add(f"month={entry + timedelta(days=31):%Y-%m}")
    return sorted(months)


def _run_outcomes(
    store: ResearchStore,
    day: date,
    stamp: datetime,
    run_id: str,
    *,
    bucket: int | None = None,
    force: bool = False,
) -> dict:
    """Simulate outcomes for occurrences already in the lake.

    Occurrence *ingestion* is still blocked on the BD-44 detector adapter, so
    this step is a clean no-op until something writes ``setup_occurrence``.
    That is a declared gap, not a silent one.

    ``bucket`` overrides the (day, hour)-derived symbol bucket. The nightly
    build never passes it; ``recompute-outcomes`` (BD-98) walks all 32 with it
    so a month computed over polluted bars can be re-simulated in full.
    """
    known = {}
    for year in (day.year, day.year - 1):
        known.update(occurrences.latest_occurrences(store, year))
    if not known:
        return {
            "status": "NO_OCCURRENCES",
            "message": "no setup_occurrence rows yet; the detector adapter is BD-44.",
        }

    all_symbols = sorted({str(row.get("symbol") or "") for row in known.values() if row.get("symbol")})
    bucket = _outcome_bucket(day, stamp) if bucket is None else int(bucket) % OUTCOME_BUCKETS
    symbols = (
        set(all_symbols)
        if len(all_symbols) <= OUTCOME_BUCKET_MIN_SYMBOLS
        else {
            symbol for symbol in all_symbols
            if int(hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:8], 16) % OUTCOME_BUCKETS == bucket
        }
    )
    selected = [row for row in known.values() if str(row.get("symbol") or "") in symbols]
    if not selected:
        return {
            "status": "NOTHING_IN_BUCKET",
            "bucket": bucket,
            "bucket_count": OUTCOME_BUCKETS,
            "symbols": 0,
        }
    _partitions, d1_by_symbol = features.daily_history_window(store, day)
    spy_d1 = list(d1_by_symbol.get("SPY") or [])
    d1_by_symbol = {symbol: rows for symbol, rows in d1_by_symbol.items() if symbol in symbols}

    m5_by_symbol: dict[str, list] = {}
    wanted_symbols = sorted(symbols | {"SPY"})
    for partition in _m5_partitions_for(
        {str(row.get("occurrence_id")): row for row in selected}, day
    ):
        # SYMBOL only, in Arrow - deliberately no date narrowing. The outcome
        # walk runs FORWARD over a horizon that can cross sessions, which is
        # why `_m5_partitions_for` already widens to the trigger's month and
        # the next one (BD-66/BD-69); narrowing to a day here would silently
        # re-simulate against a truncated future. The symbol predicate is
        # exactly the `symbol in symbols` test it replaces, so the walk sees
        # the same bars - it just never materialises everyone else's first.
        for row in store.read_rows(
            "bar_m5", partition, symbols=wanted_symbols
        ):
            symbol = str(row.get("symbol") or "")
            if symbol in symbols or symbol == "SPY":
                m5_by_symbol.setdefault(symbol, []).append(row)

    primary = outcomes.build_outcomes(
        store,
        selected,
        d1_by_symbol=d1_by_symbol,
        m5_by_symbol=m5_by_symbol,
        bands_by_occurrence=_bands_by_occurrence(
            store, {str(row.get("occurrence_id")): row for row in selected}
        ),
        # The M5-close grid plus the Phase 0.12 B3 higher-timeframe LRSI
        # study, plus the Phase 0.13 P8 entry-timing grid. All three are read
        # off the SAME occurrences and the SAME canonical M5 bars already
        # materialised above, so a study adds simulation work and not a second
        # data pass - which is what keeps them inside `setup_research`'s
        # reserve. Shadow only: 16 + 12 diagnostic recipes that reach no
        # detector, score, alert, Focus list or review queue.
        #
        # P8's twelve are cheap despite the grid's width: nine of them share one
        # exit machine with the m5close rows and the other three cost one
        # derived series each, memoised per occurrence.
        recipes=(
            tuple(outcomes.M5_CLOSE_RECIPES)
            + tuple(outcomes.HTF_LRSI_RECIPES)
            + tuple(outcomes.SETUP_ENTRY_TIMING_RECIPES)
        ),
        as_of=stamp,
        now=stamp,
        run_id=run_id,
        job_id="m5_close_recipe_outcomes",
        force=force,
    )
    after_like_step = _run_after_like_pass(
        store, m5_by_symbol, stamp=stamp, run_id=run_id
    )
    slice_rows = [
        row for row in selected
        if str(row.get("canonical_setup_id") or "") in occurrences.SLICE_SETUPS
    ]
    slice_known = {str(row.get("occurrence_id")): row for row in slice_rows}
    legacy_slice = outcomes.build_outcomes(
        store,
        slice_rows,
        d1_by_symbol=d1_by_symbol,
        m5_by_symbol=m5_by_symbol,
        bands_by_occurrence=_bands_by_occurrence(store, slice_known),
        # The challenger's family, for `swing_house_variant_v1` (M4.2). A second
        # read of the same already-resolved snapshot rows; the twin walks these
        # or, where the challenger could not be measured, nothing.
        variant_bands_by_occurrence=_bands_by_occurrence(
            store, slice_known, prefix=VARIANT_BAND_PREFIX
        ),
        as_of=stamp,
        now=stamp,
        run_id=run_id,
        force=force,
    ) if slice_rows else None
    context = market_bias_context.record_context(
        store,
        selected,
        spy_m5=list(m5_by_symbol.get("SPY") or []),
        spy_d1=spy_d1,
        now=stamp,
        run_id=run_id,
    )
    return {
        "status": primary.status,
        "bucket": bucket,
        "bucket_count": OUTCOME_BUCKETS,
        "symbols": len(symbols),
        "occurrences": len(selected),
        "m5_close": vars(primary),
        "legacy_slice": vars(legacy_slice) if legacy_slice is not None else None,
        "market_context": vars(context),
        "after_like": after_like_step,
    }


#: How far back the after-like pass looks for likes each night. Bounded, and
#: bounded generously: a like's day-5 cell cannot be measured until five sessions
#: have passed, so a window shorter than that would grade the early offsets of a
#: like and then never come back for the late ones. Thirty calendar days covers
#: five sessions plus a holiday week with room to spare, and the rows are
#: idempotent by grain so re-simulating an older like costs time and changes
#: nothing.
AFTER_LIKE_LOOKBACK_DAYS = 30


def _run_after_like_pass(store, m5_by_symbol, *, stamp, run_id: str) -> dict:
    """P10 Part C: what the trader's likes did, entered N sessions later.

    Reads the likes from the annotation log, links each to a warehouse
    occurrence (P10 B2), and simulates the twenty registered cells over the M5
    bars THIS BUILD HAS ALREADY MATERIALISED - so the pass costs simulation time
    and not a second read of the lake.

    Never allowed to cost the build. Every failure here is swallowed and
    reported as a step status, exactly like the coverage and trial-ledger lines:
    an evidence store never costs the build that feeds it.
    """
    from datetime import timedelta

    try:
        from project_paths import TRADER_ANNOTATIONS_FILE
        from ui.annotations.store import EVENT_LIKE_CLAIM, load_annotations

        cutoff = (stamp - timedelta(days=AFTER_LIKE_LOOKBACK_DAYS)).date().isoformat()
        likes = [
            row
            for row in load_annotations(
                TRADER_ANNOTATIONS_FILE, event_types=(EVENT_LIKE_CLAIM,)
            )
            if str(row.get("session_date") or "") >= cutoff
        ]
        if not likes:
            return {"status": "ok", "likes": 0, "episodes": 0, "rows": 0}

        links = {
            link.event_id: link
            for link in like_links.link_likes(store, likes)
        }
        wanted = sorted(
            {link.occurrence_id for link in links.values() if link.occurrence_id}
        )
        occurrences_by_id = {
            str(row.get("occurrence_id")): row
            for row in (
                store.read_rows("setup_occurrence", occurrence_ids=wanted)
                if wanted
                else []
            )
        }
        # THE LINK DATASET, PUBLISHED (R4 A4).
        #
        # `link_rows_for_bronze` had no production caller, while the ERD, the
        # CHANGELOG and gate 42 all say `bronze_like_occurrence_link` is written
        # nightly - and BD-92 makes it the ONLY way to recover the setup family
        # behind an after-like outcome row, because those rows are keyed by the
        # like episode. The claims were true of the code that existed and false
        # of the code that ran; this makes them true.
        #
        # Written BEFORE the outcomes below, so a night that fails in simulation
        # still leaves the join it was asked for. The record hash is over the
        # payload, so an unchanged lake re-writes nothing. Month-keyed through
        # the shared bronze record's `partition_ts`, and never allowed to cost
        # the pass: a failed link publish is reported, never raised.
        link_rows = like_links.link_rows_for_bronze(
            list(links.values()), observed_at=stamp, run_id=run_id
        )
        links_published = 0
        link_status = "ok"
        if link_rows:
            try:
                link_rows = _unwritten_link_rows(store, link_rows)
                if link_rows:
                    store.publish(
                        schemas.bronze_dataset_name(like_links.ARTIFACT),
                        link_rows,
                        job_id="after_like_links",
                    )
                links_published = len(link_rows)
            except Exception as exc:  # noqa: BLE001
                link_status = f"skipped: {exc}"

        result = after_like.run_after_like(
            likes,
            links,
            occurrences_by_id,
            m5_by_symbol,
            as_of=stamp,
            computed_at=stamp,
            run_id=run_id,
        )
        published = 0
        if result.rows:
            outcome = store.publish(
                "outcome_path", result.rows, job_id="after_like_outcomes"
            )
            published = len(result.rows)
        return {
            "status": "ok",
            "likes": result.likes_seen,
            "episodes": result.episodes_graded,
            "rows": published,
            "link_rows": links_published,
            "link_status": link_status,
            "excluded": dict(result.excluded_by_reason),
            "basis": like_links.basis_counts(list(links.values())),
            "publish": vars(outcome) if result.rows else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": str(exc)}


def _unwritten_link_rows(store, rows: list[dict]) -> list[dict]:
    """The link rows this month's partition does not already hold.

    `publish` appends a file; it does not merge, so a nightly pass that re-linked
    the same lookback would leave one copy of every like per night and any count
    over the dataset would be the number of nights rather than the number of
    likes. The record hash is over the payload, so an unchanged like hashes the
    same every night and this is an exact identity rather than a heuristic.

    Read narrowed to the row's OWN month partition (BD-74) - a month-keyed read
    of the whole dataset is the cost that put the desk at 10 GB, and this
    dataset will grow for as long as the trader keeps liking things.
    """
    spec = schemas.dataset_spec(schemas.bronze_dataset_name(like_links.ARTIFACT))
    by_partition: dict[str, list[dict]] = {}
    for row in rows:
        by_partition.setdefault(store.partition_of(spec, row), []).append(row)

    keep: list[dict] = []
    for partition, partition_rows in by_partition.items():
        try:
            existing = {
                str(value)
                for value in store.read_table(
                    spec.name, partition, columns=["record_hash"]
                )
                .column("record_hash")
                .to_pylist()
            }
        except Exception:  # noqa: BLE001 - an unreadable partition is not a reason to lose the row
            existing = set()
        keep.extend(
            row for row in partition_rows if str(row.get("record_hash")) not in existing
        )
    return keep


#: The champion's band columns (decision 0008's running-deviation sigma).
CHAMPION_BAND_PREFIX = "avwape_"
#: The challenger's (M4.1, `indicators.avwap_band_variants`). Two prefixes, one
#: reader: the families are never merged into one set of levels.
VARIANT_BAND_PREFIX = "avwap_variant_"


def _keep_newer_snapshot(current: dict | None, candidate: dict) -> dict:
    """Which of two ``feature_snapshot_daily`` rows for one (symbol, session) wins.

    The dataset identity is (symbol, session_date, feature_set_version), so
    since the M4.1 bump to ``tier1_v2`` a session can legitimately hold a row of
    each shape - the old one written before the bump, the new one after. A
    reader that took whichever landed last would be reading file order. The
    newest ``computed_at`` wins; a tie keeps what is already held, so the read
    is stable rather than arbitrary.
    """
    if current is None:
        return candidate
    return candidate if _computed_at(candidate) > _computed_at(current) else current


def _computed_at(row: dict) -> datetime:
    value = row.get("computed_at")
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def _bands_by_occurrence(
    store: ResearchStore, known: dict, *, prefix: str = CHAMPION_BAND_PREFIX
) -> dict:
    """AVWAP bands pinned to each occurrence's own trigger session.

    The review's point-in-time note: bands computed later than the trigger
    would be look-ahead, so they are read from the ``feature_snapshot_daily``
    row for the trigger session, never from today's.

    ``prefix`` selects the band FAMILY (M4.2): the champion's ``avwape_*`` or
    the challenger's ``avwap_variant_*``. Two calls, two maps, never one merged
    set of levels - a row whose family could not be measured gets no bands at
    all rather than the other formula's numbers.
    """
    wanted = {}
    for identity, row in known.items():
        trigger = row.get("trigger_at")
        if isinstance(trigger, datetime):
            wanted[identity] = (str(row.get("symbol") or ""), trigger.date())

    if not wanted:
        return {}
    snapshots: dict[tuple[str, date], dict] = {}
    for year in {day.year for _symbol, day in wanted.values()}:
        for row in store.read_table("feature_snapshot_daily", f"year={year}").to_pylist():
            session_day = row.get("session_date")
            if isinstance(session_day, datetime):
                session_day = session_day.date()
            key = (str(row.get("symbol") or ""), session_day)
            snapshots[key] = _keep_newer_snapshot(snapshots.get(key), row)

    bands = {}
    for identity, key in wanted.items():
        snapshot = snapshots.get(key)
        if snapshot is None:
            continue
        resolved = _bands_from_snapshot(snapshot, prefix=prefix)
        if any(value is not None for value in resolved.values()):
            bands[identity] = resolved
    return bands


def run_build(
    store: ResearchStore | None = None,
    *,
    session_date: date | None = None,
    now: datetime | None = None,
    run_id: str = "",
    lock_path: Path | None = None,
) -> BuildReport:
    """Run the full EOD step list. Every step is idempotent; a re-run is a no-op.

    The order is a dependency order, not a preference (BD-61): reconcile and
    seal first so the lake is consistent and the session's spooled M5 bars are
    in it; bronze next because the D1 wrap, the universe snapshots and the
    anchors all read wrapped artifacts; then ``bar_d1``, because sessions,
    aggregates and every feature snapshot read it; then sessions and the
    derived/weekly frames the intraday features join to; then anchors, because
    a daily snapshot's AVWAP block needs its ``anchor_instance``; then the
    feature snapshots; then outcomes; then backups, which should copy the
    night's work rather than yesterday's; then retirement last, so nothing is
    swept before it has been backed up.
    """
    report = BuildReport()
    target = store if store is not None else ResearchStore.open()
    if target is None:
        report.status = "DISABLED"
        report.message = "research_store_dir is not configured; the warehouse is a no-op."
        return report
    stamp = now or utc_now()
    day = session_date or stamp.date()
    try:
        with single_flight(lock_path):
            _record_job("RUNNING", {"run_id": run_id})
            report.steps["reconcile"] = vars(target.reconcile(job_id=run_id or "build"))
            report.steps["spool"] = vars(seal_spool(target))
            report.steps["bronze"] = [vars(item) for item in run_bronze_ingest(target, run_id=run_id, now=stamp)]
            report.steps["snapshots"] = [
                vars(item) for item in run_daily_snapshots(target, session_date=day, run_id=run_id, now=stamp)
            ]
            cohort = cohort_for(target, day)
            report.steps["bar_d1"] = vars(
                ingest_daily_bars(target, cohort, as_of=day, run_id=run_id, now=stamp)
            )
            report.steps["sessions"] = vars(build_trading_sessions(target, day, day, now=stamp, run_id=run_id))
            report.steps["derived"] = vars(build_derived_bars(target, [day], as_of=stamp, now=stamp, run_id=run_id))
            report.steps["weekly"] = vars(build_weekly_bars(target, [day], as_of=stamp, now=stamp, run_id=run_id))
            report.steps["anchors"] = vars(
                features.build_anchor_instances(
                    target, anchors_from_bronze(target), now=stamp, run_id=run_id
                )
            )
            report.steps["features_daily"] = vars(
                features.build_daily_snapshots(
                    target,
                    day,
                    anchors_by_symbol=anchor_dates_by_symbol(target, day),
                    now=stamp,
                    run_id=run_id,
                )
            )
            report.steps["features_intraday"] = vars(
                features.build_intraday_snapshots(target, day, now=stamp, run_id=run_id)
            )
            report.steps["occurrences"] = vars(
                tracker_adapter.record_tracker_occurrences(
                    target, run_id=run_id, now=stamp
                )
            )
            # THE TRIAL LEDGER, WRITTEN BY THE BUILD, AND WRITTEN FIRST (R4 A2).
            #
            # Gate 37 asks for a ledger row after one overnight run, and nothing
            # in production wrote one - the module existed and only tests called
            # it, so the declarations that are supposed to predate every outcome
            # would have been written after them, by hand, whenever somebody
            # remembered.
            #
            # It now sits ABOVE `_run_outcomes`, which is the whole point of a
            # trial ledger: "an append-only row per registered grid BEFORE any
            # outcome is inspected". Below the outcomes step it was written after
            # the after-like grid had already been simulated and published, so
            # the declaration that is supposed to predate the evidence followed
            # it by one step every night.
            #
            # Idempotent by construction: `register` refuses a trial_id the
            # ledger already carries, so every firing after the first writes
            # nothing. Never allowed to cost the build.
            try:
                report.steps["trial_ledger"] = {
                    "registered": trial_ledger.backfill(target.root)
                }
            except Exception:  # noqa: BLE001
                # Swallowed like the coverage line below: an evidence store
                # never costs the build that feeds it.
                report.steps["trial_ledger"] = {"registered": [], "status": "skipped"}
            report.steps["outcomes"] = _run_outcomes(target, day, stamp, run_id)
            # One line per firing naming the symbol bucket it covered, so a
            # fact pack can tell "not measured yet" from "measured and flat".
            # Never allowed to cost the build: it returns False and logs.
            outcome_coverage.record_firing(
                target.root,
                report.steps["outcomes"],
                run_id=run_id,
                now=stamp,
            )
            report.steps["backups"] = _run_backups(target, stamp)
            report.steps["retired"] = vars(target.collect_retired(now=stamp))
            _record_job("COMPLETED", {"run_id": run_id})
    except SingleFlightError as exc:
        report.status = "REFUSED"
        report.message = str(exc)
        _record_job("SKIPPED", {"reason": "single_flight"})
    return report


# ---------------------------------------------------------------------------
# The backfill jobs (plan sec 5.1-5.2, LD-02/LD-03)
# ---------------------------------------------------------------------------
#: Overnight window: 20:00-02:00 ET minus the ~23:45 TWS restart is ~5.5 usable
#: hours (sec 5.1). The default budget is the part of it a single invocation may
#: spend *waiting* on the pacer; the job stops cleanly when it runs out and the
#: next night resumes from the gaps it recorded.
NIGHTLY_TIME_BUDGET_SECONDS = 4 * 3600
#: The Saturday full-universe sweep is ~4.3 h at the published floor (sec 5.2).
WEEKLY_TIME_BUDGET_SECONDS = 5 * 3600

JOB_NIGHTLY = "nightly"
JOB_WEEKLY = "weekly"
JOB_SEED = "seed"


def _capture_fetcher(pacer=None):
    """The real IB capture fetcher, or ``None`` with a reason.

    Built here rather than inside :mod:`backfill` so the job logic stays
    provider-agnostic and offline-testable (BD-15): the socket lives in exactly
    one adapter. This is the BD-25 path - it has no offline coverage by
    construction, so a failure to build it is reported, never raised.
    """
    try:
        from . import ib_capture
    except ImportError:  # pragma: no cover - scripts/ on sys.path
        import ib_capture  # type: ignore
    spec = ib_capture.backfill_connection_spec()
    transport = ib_capture.build_ib_transport(spec)
    return ib_capture.IbCaptureFetcher(transport, spec=spec, pacer=pacer)


def run_backfill_job(
    store: ResearchStore | None = None,
    *,
    job: str = JOB_NIGHTLY,
    session_date: date | None = None,
    cohort=None,
    fetcher=None,
    time_budget_seconds: float | None = None,
    max_requests: int | None = None,
    now: datetime | None = None,
    run_id: str = "",
    lock_path: Path | None = None,
) -> dict:
    """Run one nightly / weekly / seed backfill pass.

    Holds the **same** single-flight lock as the EOD build: both write the lake,
    and LD-01 allows exactly one writer. A backfill that lands while a build is
    running refuses with a clear message instead of racing it.

    The cohort is the session's own ``universe_membership_daily`` snapshot for
    the nightly job (LD-05 point-in-time membership, the same source the D1 wrap
    uses) and the full captured universe for the weekly sweep.
    """
    target = store if store is not None else ResearchStore.open()
    if target is None:
        return {
            "status": "DISABLED",
            "message": "research_store_dir is not configured; the warehouse is a no-op.",
        }
    stamp = now or utc_now()
    day = session_date or (stamp.date() - timedelta(days=1))

    try:
        with single_flight(lock_path):
            _record_job("RUNNING", {"run_id": run_id, "job": f"backfill_{job}"})
            if job == JOB_SEED:
                report = _run_seed(target, cohort, fetcher=fetcher, now=stamp, run_id=run_id)
            else:
                report = _run_ib_backfill(
                    target,
                    job=job,
                    day=day,
                    cohort=cohort,
                    fetcher=fetcher,
                    time_budget_seconds=time_budget_seconds,
                    max_requests=max_requests,
                    stamp=stamp,
                    run_id=run_id,
                )
            _record_job("COMPLETED", {"run_id": run_id, "job": f"backfill_{job}"})
            return report
    except SingleFlightError as exc:
        _record_job("SKIPPED", {"reason": "single_flight", "job": f"backfill_{job}"})
        return {"status": "REFUSED", "message": str(exc)}


def _run_ib_backfill(
    store: ResearchStore,
    *,
    job: str,
    day: date,
    cohort,
    fetcher,
    time_budget_seconds: float | None,
    max_requests: int | None,
    stamp: datetime,
    run_id: str,
) -> dict:
    try:
        from . import backfill as backfill_mod
    except ImportError:  # pragma: no cover - scripts/ on sys.path
        import backfill as backfill_mod  # type: ignore

    symbols = list(cohort) if cohort is not None else _backfill_cohort(store, job, day)
    if not symbols:
        return {
            "status": "NO_COHORT",
            "message": (
                "no cohort for this session: universe_membership_daily has no rows for "
                f"{day.isoformat()}. Run the EOD build for that session first."
            ),
        }

    if fetcher is None:
        try:
            fetcher = _capture_fetcher()
        except Exception as exc:  # no TWS, no ibapi, bad client id
            return {
                "status": "NO_PROVIDER",
                "message": f"capture transport unavailable: {exc}",
            }

    if job == JOB_WEEKLY:
        report = backfill_mod.run_weekly_universe_sweep(
            store,
            symbols,
            fetcher=fetcher,
            week_ending=day,
            time_budget_seconds=(
                WEEKLY_TIME_BUDGET_SECONDS if time_budget_seconds is None else time_budget_seconds
            ),
            max_requests=max_requests,
            now=stamp,
            run_id=run_id or "weekly_universe_sweep",
        )
    else:
        report = backfill_mod.run_nightly_backfill(
            store,
            symbols,
            fetcher=fetcher,
            session_date=day,
            time_budget_seconds=(
                NIGHTLY_TIME_BUDGET_SECONDS if time_budget_seconds is None else time_budget_seconds
            ),
            max_requests=max_requests,
            now=stamp,
            run_id=run_id or "nightly_backfill",
        )
    payload = vars(report)
    payload["cohort"] = len(symbols)
    payload["session_date"] = day.isoformat()
    return payload


def _run_seed(store: ResearchStore, cohort, *, fetcher, now: datetime, run_id: str) -> dict:
    try:
        from . import backfill as backfill_mod
    except ImportError:  # pragma: no cover - scripts/ on sys.path
        import backfill as backfill_mod  # type: ignore

    symbols = list(cohort) if cohort is not None else _captured_universe(store)
    if not symbols:
        return {"status": "NO_COHORT", "message": "no universe to seed."}
    if fetcher is None:
        return {
            "status": "NO_PROVIDER",
            "message": (
                "the yfinance seed needs an explicit fetcher; it is a one-time "
                "trickle and is never started implicitly (R11)."
            ),
        }
    report = backfill_mod.run_yahoo_seed(
        store,
        symbols,
        fetcher=fetcher,
        spool_dir=config.research_spool_dir(),
        now=now,
        run_id=run_id or "yahoo_m5_seed",
    )
    return vars(report)


def _backfill_cohort(store: ResearchStore, job: str, day: date) -> list[str]:
    """Who this job covers: the session's cohort, or everything already captured."""
    if job == JOB_WEEKLY:
        return _captured_universe(store)
    return cohort_for(store, day)


def _captured_universe(store: ResearchStore) -> list[str]:
    """Every symbol the lake has ever recorded membership for (sec 5.2)."""
    symbols = set()
    for row in store.read_table("universe_membership_daily", columns=["symbol"]).to_pylist():
        value = str(row.get("symbol") or "").strip().upper()
        if value:
            symbols.add(value)
    return sorted(symbols)


def run_status(store: ResearchStore | None = None) -> dict:
    """What the lake holds, straight from the ledger. Reads no bar data."""
    target = store if store is not None else ResearchStore.open()
    if target is None:
        return {"enabled": False, "message": "research_store_dir is not configured."}
    return {
        "enabled": True,
        "root": str(target.root),
        "health": target.health_counts(),
        "datasets": queries.dataset_inventory(target),
    }


def run_dedupe(
    store: ResearchStore | None,
    *,
    dataset: str = "bar_m5",
    partitions=None,
    apply: bool = False,
    job_id: str = "dedupe",
    lock_path: Path | None = None,
) -> dict:
    """Count (and with ``apply``, drop) rows repeated at the dataset grain.

    A dry run by default: the lake is read, nothing is written, and the report
    says what ``--apply`` would retire. With ``apply`` each partition is
    rewritten through ``ResearchStore.dedupe_partition`` - one COMPACT manifest
    line each, inputs retired never deleted - so the change is reversible by
    repointing the manifest, exactly like a compaction (BD-96).
    """
    if store is None:
        return {"status": "DISABLED", "message": "research_store_dir is not configured."}
    wanted = list(partitions or [])
    if not wanted:
        wanted = sorted({entry.partition for entry in store.manifest.resolve(dataset=dataset).entries})
    report: dict = {"status": "OK", "dataset": dataset, "applied": bool(apply), "partitions": []}

    def _one(partition: str):
        try:
            result = (
                store.dedupe_partition(dataset, partition, job_id=job_id)
                if apply
                else store.duplicate_rows(dataset, partition)
            )
        except Exception as exc:
            report["status"] = "ERROR"
            report["partitions"].append({"partition": partition, "error": str(exc)})
            return
        report["partitions"].append(
            {
                "partition": partition,
                "rows_before": result.rows_before,
                "rows_after": result.rows_after,
                "rows_dropped": result.rows_dropped,
                "rewritten": bool(result.entry is not None),
            }
        )

    if apply:
        # A rewrite shares the build's single-flight lock so it can never
        # interleave with a seal of the same partition. A dry run reads only.
        try:
            with single_flight(lock_path):
                for partition in wanted:
                    _one(partition)
        except SingleFlightError as exc:
            return {"status": "REFUSED", "dataset": dataset, "applied": True, "reason": str(exc)}
    else:
        for partition in wanted:
            _one(partition)
    report["rows_dropped"] = sum(int(item.get("rows_dropped") or 0) for item in report["partitions"])
    return report


# ---------------------------------------------------------------------------
# Band coverage (Q2.3) - read-only
# ---------------------------------------------------------------------------
#: The D1 recipes a coverage report can speak about: the default set
#: ``build_outcomes`` simulates for a swing occurrence. Each is asked for its
#: OWN required bands (``outcomes.required_band_numbers``), never a shared list.
BAND_COVERAGE_RECIPES = (
    outcomes.SWING_HOUSE_V1,
    # The band challenger's twin (M4.2). Asked for its OWN family's columns:
    # the same required band NUMBERS, read from `avwap_variant_*`.
    outcomes.SWING_HOUSE_VARIANT_V1,
    outcomes.CONTROL_FIXED_1R2R_V1,
    outcomes.CONTROL_TIME_ONLY_V1,
)
#: An occurrence with no ``outcome_path`` row for this recipe is NAMED rather
#: than dropped: "not simulated" and "simulated and flat" are different facts.
STATE_NOT_SIMULATED = "NOT_SIMULATED"
#: An occurrence whose trigger session has no ``feature_snapshot_daily`` row at
#: all. Distinct from a row that HAS one and used no anchor (``none``) and from
#: one written before the column existed (``legacy``).
KNOWLEDGE_NO_SNAPSHOT = "no_snapshot"


def _month_bounds(month: str) -> tuple[date, date]:
    first = date.fromisoformat(f"{month}-01")
    last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return first, last


def _empty_coverage_bucket() -> dict:
    return {
        "occurrences": 0,
        "required_bands_present": 0,
        "plain_no_target": 0,
        "geometry_valid": 0,
        "geometry_checked": 0,
        "null_bands": 0,
        "by_result_state": {},
    }


def _bands_from_snapshot(snapshot: dict, *, prefix: str = CHAMPION_BAND_PREFIX) -> dict:
    return {
        band.upper(): snapshot.get(f"{prefix}{band}")
        for band in ("upper_1", "upper_2", "upper_3", "lower_1", "lower_2", "lower_3")
    }


def _band_prefix_for(recipe) -> str:
    """Which snapshot columns this recipe's band family lives in (M4.2)."""
    family = str(getattr(recipe, "band_family", outcomes.BAND_FAMILY_CHAMPION) or "")
    return VARIANT_BAND_PREFIX if family == outcomes.BAND_FAMILY_VARIANT else CHAMPION_BAND_PREFIX


def run_band_coverage(
    store: ResearchStore | None,
    *,
    month: str,
    recipe_id: str | None = None,
) -> dict:
    """How much of a month's swing evidence actually had bands to walk (Q2.3).

    READ-ONLY: it resolves files, reads them Arrow-narrowed, and writes no row,
    no manifest line and no file. Per recipe and per anchor-knowledge bucket it
    reports the occurrences, how many carried every band the RECIPE ITSELF
    requires, how many fell to the no-target path, how many had valid geometry
    for their side, how many had no band at all, and the ``result_state``
    spread. The 2026-09-04 investigation had to INFER 942 of 947 no-target
    losses; this is that number, read.
    """
    if store is None:
        return {"status": "DISABLED", "message": "research_store_dir is not configured."}
    try:
        first, last = _month_bounds(month)
    except ValueError:
        return {"status": "ERROR", "message": f"--month must be YYYY-MM, got {month!r}"}
    start = datetime(first.year, first.month, first.day, tzinfo=timezone.utc)
    end = datetime(last.year, last.month, last.day, tzinfo=timezone.utc) + timedelta(days=1)

    selected = [
        recipe
        for recipe in BAND_COVERAGE_RECIPES
        if not recipe_id or recipe.recipe_id == recipe_id
    ]
    if not recipe_id and not selected:  # pragma: no cover - defensive
        return {"status": "ERROR", "message": "no recipe selected"}
    if recipe_id and not selected:
        return {
            "status": "ERROR",
            "message": f"unknown recipe {recipe_id!r}; known: "
            + ", ".join(item.recipe_id for item in BAND_COVERAGE_RECIPES),
        }

    # Session-scoped and narrowed Arrow-side on the trigger stamp (BD-91), the
    # same +/- one year span `latest_occurrences` reads for the boundary case.
    latest: dict[str, dict] = {}
    for year in (first.year - 1, first.year, first.year + 1):
        for row in store.read_rows(
            "setup_occurrence",
            f"year={year}",
            interval_start_range=(start, end),
            time_column="trigger_at",
        ):
            identity = str(row.get("occurrence_id") or "")
            current = latest.get(identity)
            if current is None or occurrences._revision_number(
                row.get("revision_id")
            ) > occurrences._revision_number(current.get("revision_id")):
                latest[identity] = row

    symbols = sorted({str(row.get("symbol") or "") for row in latest.values()})
    snapshots: dict[tuple[str, date], dict] = {}
    if symbols:
        for year in {first.year, last.year}:
            for row in store.read_rows(
                "feature_snapshot_daily", f"year={year}", symbols=symbols
            ):
                day = row.get("session_date")
                if isinstance(day, datetime):
                    day = day.date()
                key = (str(row.get("symbol") or ""), day)
                snapshots[key] = _keep_newer_snapshot(snapshots.get(key), row)

    report: dict = {
        "status": "OK",
        "month": month,
        "occurrences_in_month": len(latest),
        "recipes": {},
    }
    for recipe in selected:
        stored = outcomes.latest_outcomes(
            store, sorted(latest) or None, recipe_ids=[recipe.recipe_id]
        )
        required = list(outcomes.required_band_numbers(recipe))
        buckets: dict[str, dict] = {}
        totals = _empty_coverage_bucket()
        for identity, occurrence in sorted(latest.items()):
            trigger = occurrence.get("trigger_at")
            trigger_day = trigger.date() if isinstance(trigger, datetime) else trigger
            snapshot = snapshots.get((str(occurrence.get("symbol") or ""), trigger_day))
            if snapshot is None:
                knowledge = KNOWLEDGE_NO_SNAPSHOT
                bands = {}
            else:
                knowledge = features.anchor_knowledge_bucket(snapshot.get("anchor_knowledge"))
                bands = _bands_from_snapshot(snapshot, prefix=_band_prefix_for(recipe))
            geometry = outcomes.swing_geometry(occurrence, recipe, bands)
            levels = geometry["bands"]
            state = stored.get(
                (identity, recipe.recipe_id, outcomes.outcome_definition_for(recipe))
            )
            state_name = str(state.get("result_state")) if state else STATE_NOT_SIMULATED

            bucket = buckets.setdefault(knowledge, _empty_coverage_bucket())
            for target in (bucket, totals):
                target["occurrences"] += 1
                if required and all(levels.get(number) is not None for number in required):
                    target["required_bands_present"] += 1
                if geometry["path_kind"] == outcomes.PATH_KIND_PLAIN_NO_TARGET:
                    target["plain_no_target"] += 1
                if geometry["valid"] is not None:
                    target["geometry_checked"] += 1
                    if geometry["valid"]:
                        target["geometry_valid"] += 1
                if all(value is None for value in levels.values()):
                    target["null_bands"] += 1
                target["by_result_state"][state_name] = (
                    target["by_result_state"].get(state_name, 0) + 1
                )
        if not required:
            # A recipe whose target is an R multiple needs no band, so "every
            # required band present" is not a fact about it. Counting every
            # occurrence made the live table read
            # `control_fixed_1r2r_v1 n=2437 bands=2437 null=2431`, which is a
            # contradiction rather than a measurement; the honest value is "not
            # applicable", and `null_bands` still says what the lake holds.
            for block in (totals, *buckets.values()):
                block["required_bands_present"] = None
        report["recipes"][recipe.recipe_id] = {
            "required_bands": required,
            "by_knowledge": buckets,
            "totals": totals,
        }
    return report


def format_band_coverage(report: dict) -> str:
    """One line per (recipe, knowledge bucket). The report is the evidence; the
    table is how a human reads it."""
    if report.get("status") != "OK":
        return f"{report.get('status')}: {report.get('message', '')}".strip()
    lines = [
        f"band coverage {report['month']} - {report['occurrences_in_month']} occurrence(s)",
        f"{'recipe':<24}{'knowledge':<14}{'n':>6}{'bands':>7}{'noTgt':>7}{'geom':>7}{'null':>6}  states",
    ]
    for recipe_id, block in sorted(report["recipes"].items()):
        required = block["required_bands"]
        lines.append(
            f"  {recipe_id} - required bands: "
            + (",".join(str(number) for number in required) if required else "none")
        )
        rows = list(sorted(block["by_knowledge"].items())) + [("TOTAL", block["totals"])]
        for knowledge, bucket in rows:
            states = ", ".join(
                f"{name}={count}" for name, count in sorted(bucket["by_result_state"].items())
            )
            present = bucket["required_bands_present"]
            present_cell = "n/a" if present is None else str(present)
            lines.append(
                f"{recipe_id:<24}{knowledge:<14}{bucket['occurrences']:>6}"
                f"{present_cell:>7}{bucket['plain_no_target']:>7}"
                f"{bucket['geometry_valid']:>7}{bucket['null_bands']:>6}  {states}"
            )
    return "\n".join(lines)


def _swing_headline():
    """`swing_headline`, imported lazily - it is a `scripts/` root module.

    ONE Wilson: `swing_headline.WILSON_Z` is the z for every trader-facing win
    rate, and this table reaches for it rather than defining a second one.
    `master_avwap_lib/expected_r.py`'s z=1.28 is a parameter inside a fenced
    scoring file and is deliberately not used here.
    """
    scripts_dir = str(Path(__file__).resolve().parents[1])
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import swing_headline  # type: ignore

    return swing_headline


def _empty_compare_cell() -> dict:
    return {
        "n": 0,
        "resolved": 0,
        "targeted": 0,
        "stopped": 0,
        "open": 0,
        "other": 0,
        "net_r_sum": 0.0,
        "net_r_n": 0,
    }


def _finish_compare_cell(cell: dict) -> dict:
    """Turn the running counts into the numbers the table prints.

    ``resolved`` is TARGETED + STOPPED, and the win rate is over RESOLVED - an
    OPEN row has not answered the question and must not sit in either the
    numerator or the denominator. The lower bound is `swing_headline`'s Wilson,
    which is the ONE Wilson for every trader-facing win rate.
    """
    wilson = _swing_headline().wilson_lower_bound
    resolved = cell["targeted"] + cell["stopped"]
    cell["resolved"] = resolved
    cell["win_rate"] = (cell["targeted"] / resolved) if resolved else None
    cell["win_rate_lb"] = wilson(cell["targeted"], resolved) if resolved else None
    cell["mean_net_r"] = (cell["net_r_sum"] / cell["net_r_n"]) if cell["net_r_n"] else None
    return cell


def run_band_coverage_compare(
    store: ResearchStore | None,
    *,
    month: str,
    recipe_ids,
) -> dict:
    """Two recipes, ONE table, on the SAME occurrence ids (M4.3).

    Built for the band challenger - `swing_house_v1` against
    `swing_house_variant_v1` - but it is a general pairing: any two D1 recipes
    `band-coverage` knows about can be read side by side.

    **Pairing is the whole point.** An occurrence that has an outcome row under
    one recipe and not the other is counted on a ``not_paired`` line and is in
    NEITHER recipe's numbers. Reading each recipe over whatever rows it happens
    to have would measure coverage and report it as edge: the challenger's bands
    are missing on a different population than the champion's, so the recipe
    with fewer rows can look better simply by having skipped the losses.

    READ-ONLY. It resolves files, reads them Arrow-narrowed, and writes no row,
    no manifest line and no file.
    """
    if store is None:
        return {"status": "DISABLED", "message": "research_store_dir is not configured."}
    wanted = [str(item) for item in (recipe_ids or ())]
    if len(wanted) != 2:
        return {"status": "ERROR", "message": "--compare takes exactly two recipe ids"}
    known = {recipe.recipe_id: recipe for recipe in BAND_COVERAGE_RECIPES}
    unknown = [item for item in wanted if item not in known]
    if unknown:
        return {
            "status": "ERROR",
            "message": f"unknown recipe {', '.join(repr(item) for item in unknown)}; known: "
            + ", ".join(known),
        }

    singles = {
        recipe_id: run_band_coverage(store, month=month, recipe_id=recipe_id)
        for recipe_id in wanted
    }
    for report in singles.values():
        if report.get("status") != "OK":
            return report

    try:
        first, last = _month_bounds(month)
    except ValueError:
        return {"status": "ERROR", "message": f"--month must be YYYY-MM, got {month!r}"}
    start = datetime(first.year, first.month, first.day, tzinfo=timezone.utc)
    end = datetime(last.year, last.month, last.day, tzinfo=timezone.utc) + timedelta(days=1)

    latest: dict[str, dict] = {}
    for year in (first.year - 1, first.year, first.year + 1):
        for row in store.read_rows(
            "setup_occurrence",
            f"year={year}",
            interval_start_range=(start, end),
            time_column="trigger_at",
        ):
            identity = str(row.get("occurrence_id") or "")
            current = latest.get(identity)
            if current is None or occurrences._revision_number(
                row.get("revision_id")
            ) > occurrences._revision_number(current.get("revision_id")):
                latest[identity] = row

    symbols = sorted({str(row.get("symbol") or "") for row in latest.values()})
    snapshots: dict[tuple[str, date], dict] = {}
    if symbols:
        for year in {first.year, last.year}:
            for row in store.read_rows(
                "feature_snapshot_daily", f"year={year}", symbols=symbols
            ):
                day = row.get("session_date")
                if isinstance(day, datetime):
                    day = day.date()
                key = (str(row.get("symbol") or ""), day)
                snapshots[key] = _keep_newer_snapshot(snapshots.get(key), row)

    stored = {
        recipe_id: outcomes.latest_outcomes(
            store, sorted(latest) or None, recipe_ids=[recipe_id]
        )
        for recipe_id in wanted
    }

    report: dict = {
        "status": "OK",
        "month": month,
        "recipes": list(wanted),
        "wilson_z": _swing_headline().WILSON_Z,
        "occurrences_in_month": len(latest),
        "paired": 0,
        "not_paired": {
            "total": 0,
            "missing_both": 0,
            **{f"missing_{recipe_id}": 0 for recipe_id in wanted},
        },
        "by_knowledge": {},
        "totals": {"n": 0, "recipes": {item: _empty_compare_cell() for item in wanted}},
    }

    for identity, occurrence in sorted(latest.items()):
        rows = {}
        for recipe_id in wanted:
            recipe = known[recipe_id]
            rows[recipe_id] = stored[recipe_id].get(
                (identity, recipe_id, outcomes.outcome_definition_for(recipe))
            )
        missing = [recipe_id for recipe_id in wanted if rows[recipe_id] is None]
        if missing:
            report["not_paired"]["total"] += 1
            if len(missing) == len(wanted):
                report["not_paired"]["missing_both"] += 1
            for recipe_id in missing:
                report["not_paired"][f"missing_{recipe_id}"] += 1
            continue

        trigger = occurrence.get("trigger_at")
        trigger_day = trigger.date() if isinstance(trigger, datetime) else trigger
        snapshot = snapshots.get((str(occurrence.get("symbol") or ""), trigger_day))
        knowledge = (
            KNOWLEDGE_NO_SNAPSHOT
            if snapshot is None
            else features.anchor_knowledge_bucket(snapshot.get("anchor_knowledge"))
        )
        bucket = report["by_knowledge"].setdefault(
            knowledge,
            {"n": 0, "recipes": {item: _empty_compare_cell() for item in wanted}},
        )
        report["paired"] += 1
        bucket["n"] += 1
        report["totals"]["n"] += 1
        for recipe_id in wanted:
            state = str(rows[recipe_id].get("result_state") or "")
            net = rows[recipe_id].get("net_r")
            for cell in (bucket["recipes"][recipe_id], report["totals"]["recipes"][recipe_id]):
                cell["n"] += 1
                if state == outcomes.STATE_TARGETED:
                    cell["targeted"] += 1
                elif state == outcomes.STATE_STOPPED:
                    cell["stopped"] += 1
                elif state == outcomes.STATE_OPEN:
                    cell["open"] += 1
                else:
                    cell["other"] += 1
                if net is not None:
                    cell["net_r_sum"] += float(net)
                    cell["net_r_n"] += 1

    for block in (report["totals"], *report["by_knowledge"].values()):
        for cell in block["recipes"].values():
            _finish_compare_cell(cell)
    return report


def _compare_cell_text(cell: dict) -> str:
    rate = "-" if cell["win_rate"] is None else f"{cell['win_rate'] * 100:.0f}%"
    bound = "" if cell["win_rate_lb"] is None else f" (>={cell['win_rate_lb'] * 100:.0f}%)"
    mean = "-" if cell["mean_net_r"] is None else f"{cell['mean_net_r']:+.2f}R"
    return (
        f"{cell['n']:>4}{cell['resolved']:>5}{cell['targeted']:>5}{cell['stopped']:>5}"
        f"{rate + bound:>13}{mean:>8}"
    )


def format_band_coverage_compare(report: dict) -> str:
    """The two recipes as adjacent column groups, one row per knowledge bucket."""
    if report.get("status") != "OK":
        return f"{report.get('status')}: {report.get('message', '')}".strip()
    left, right = report["recipes"]
    header_group = f"{'n':>4}{'resl':>5}{'TGT':>5}{'STOP':>5}{'win (lower)':>13}{'meanR':>8}"
    lines = [
        f"band compare {report['month']} - {report['occurrences_in_month']} occurrence(s), "
        f"{report['paired']} paired",
        f"win rate is over RESOLVED (TARGETED + STOPPED); lower bound is Wilson z="
        f"{report['wilson_z']:.4f}",
        f"{'knowledge':<14}{left:^40}|{right:^40}",
        f"{'':<14}{header_group:<40}|{header_group:<40}",
    ]
    rows = list(sorted(report["by_knowledge"].items())) + [("TOTAL", report["totals"])]
    for knowledge, block in rows:
        lines.append(
            f"{knowledge:<14}{_compare_cell_text(block['recipes'][left]):<40}"
            f"|{_compare_cell_text(block['recipes'][right]):<40}"
        )
    missing = report["not_paired"]
    lines.append(
        f"not_paired {missing['total']}  (missing {left}: {missing[f'missing_{left}']}, "
        f"missing {right}: {missing[f'missing_{right}']}, missing both: {missing['missing_both']})"
        "  - counted here, never in either recipe's numbers"
    )
    return "\n".join(lines)


#: What a month rebuild recomputes, in dependency order. Both are DERIVED from
#: ``bar_m5`` and both were computed from the duplicated rows (BD-96): the
#: aggregator counted every twin as a constituent (volume x N, quality PARTIAL)
#: and the intraday features windowed over the doubled list. ``bar_m5`` itself
#: is repaired by ``dedupe``; ``bar_d1`` and the daily features never read it.
REBUILD_DATASETS = ("bar_derived", "feature_snapshot_intraday")


def run_rebuild_month(
    store: ResearchStore | None,
    *,
    month: str,
    apply: bool = False,
    job_id: str = "rebuild_month",
    now: datetime | None = None,
    lock_path: Path | None = None,
) -> dict:
    """Retire a month's derived partitions and recompute them session by session.

    A dry run (the default) lists the partitions that would be retired and the
    sessions that would be rebuilt, and writes nothing. With ``apply`` it takes
    the build's single-flight lock, appends one RETIRE line per partition, then
    for every exchange session of the month runs the derived-bar, weekly-bar and
    intraday-feature steps exactly as the nightly build does. Nothing is deleted:
    the retired files move to ``_retired/`` on the next GC and stay restorable.
    Run it AFTER ``dedupe --apply`` on ``bar_m5``, never before.
    """
    if store is None:
        return {"status": "DISABLED", "message": "research_store_dir is not configured."}
    try:
        first = date.fromisoformat(f"{month}-01")
    except ValueError:
        return {"status": "ERROR", "message": f"--month must be YYYY-MM, got {month!r}"}
    stamp = now or utc_now()
    last_day = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    try:
        from . import exchange_calendar as xcal
    except ImportError:  # pragma: no cover - scripts/ directly on sys.path
        import exchange_calendar as xcal  # type: ignore
    sessions = [
        session.session_date if hasattr(session, "session_date") else session
        for session in xcal.sessions_between(first, min(last_day, stamp.date()))
    ]
    live = store.manifest.resolve()
    partitions = sorted(
        {
            (entry.dataset, entry.partition)
            for entry in live.entries
            if entry.dataset in REBUILD_DATASETS and entry.partition.endswith(f"month={month}")
        }
    )
    report: dict = {
        "status": "OK",
        "month": month,
        "applied": bool(apply),
        "partitions": [f"{dataset}/{partition}" for dataset, partition in partitions],
        "sessions": [day.isoformat() for day in sessions],
        "retired_files": 0,
        "steps": {},
    }
    if not apply:
        return report
    try:
        with single_flight(lock_path):
            for dataset, partition in partitions:
                report["retired_files"] += len(
                    store.retire_partition(dataset, partition, job_id=job_id, reason=f"rebuild {month} (BD-96)")
                )
            for day in sessions:
                steps: dict = {}
                steps["derived"] = vars(build_derived_bars(store, [day], as_of=stamp, now=stamp, run_id=job_id))
                steps["weekly"] = vars(build_weekly_bars(store, [day], as_of=stamp, now=stamp, run_id=job_id))
                steps["features_intraday"] = vars(
                    features.build_intraday_snapshots(store, day, now=stamp, run_id=job_id)
                )
                report["steps"][day.isoformat()] = steps
            report["retired"] = vars(store.collect_retired(now=stamp))
    except SingleFlightError as exc:
        return {"status": "REFUSED", "month": month, "applied": True, "reason": str(exc)}
    return report


#: The daily-feature rebuild is its OWN command rather than a third entry in
#: ``REBUILD_DATASETS``: ``feature_snapshot_daily`` is partitioned by YEAR, so
#: ``rebuild-month``'s "retire the month's partitions" mechanic would retire
#: every other month of that year with it. Same shape, different key.
REBUILD_DAILY_DATASET = "feature_snapshot_daily"


def run_rebuild_daily_features(
    store: ResearchStore | None,
    *,
    start: date,
    end: date,
    apply: bool = False,
    job_id: str = "rebuild_daily_features",
    now: datetime | None = None,
    lock_path: Path | None = None,
) -> dict:
    """Recompute ``feature_snapshot_daily`` for past sessions, WITH their anchors.

    The nightly build writes daily features for ONE day, so every session that
    ran before the 2026-09-04 earnings-anchor bridge carries null AVWAP bands
    and no amount of re-simulating outcomes can fix that: ``_bands_by_occurrence``
    reads the trigger session's own feature row and correctly finds nothing.
    This walks the exchange sessions in ``[start, end]`` and rebuilds them with
    ``anchor_dates_by_symbol``'s stamped choice, so every rebuilt row SAYS
    whether its anchor was knowable then (Q2.1/BD-99). Expect ``reconstructed``
    to dominate, and expect it to dominate completely once the bridge has run:
    the choice keeps the NEWEST anchor bar on or before the session **regardless
    of knowledge**, so a bridged anchor with a newer bar displaces the 14
    hand-imported ones rather than losing to them. `observed` rows appear only
    for a symbol whose newest qualifying anchor bar was already in the lake
    before that session. The report is the SPLIT, not a target for one bucket.

    A dry run by default: it lists the sessions and writes nothing. With
    ``apply`` it takes the build's single-flight lock and, per affected YEAR
    partition, retires the partition (one RETIRE line, files kept and
    restorable, exactly as BD-97's month rebuild does), republishes verbatim
    every row OUTSIDE the range - the partition is year-keyed, so a January row
    must survive an August rebuild - and then recomputes each session. A second
    run therefore supersedes rather than duplicating.

    Cost note: the carry materialises the year partition's out-of-range rows
    once. That is the price of a year-keyed partition and is why this is a
    maintenance command under the lock, never a step in the nightly build.
    """
    if store is None:
        return {"status": "DISABLED", "message": "research_store_dir is not configured."}
    if end < start:
        return {"status": "ERROR", "message": f"--to {end} is before --from {start}"}
    stamp = now or utc_now()
    sessions = [
        session.session_date if hasattr(session, "session_date") else session
        for session in xcal.sessions_between(start, end)
    ]
    partitions = sorted({f"year={day.year}" for day in sessions})
    report: dict = {
        "status": "OK",
        "from": start.isoformat(),
        "to": end.isoformat(),
        "applied": bool(apply),
        "sessions": [day.isoformat() for day in sessions],
        "partitions": partitions,
        "retired_files": 0,
        "carried_rows": 0,
        "steps": {},
    }
    if not apply or not sessions:
        return report

    try:
        import pyarrow as pa
        import pyarrow.compute as pc

        with single_flight(lock_path):
            for partition in partitions:
                table = store.read_table(REBUILD_DAILY_DATASET, partition)
                survivors = None
                if table.num_rows:
                    outside = pc.or_(
                        pc.less(table.column("session_date"), pa.scalar(start, pa.date32())),
                        pc.greater(table.column("session_date"), pa.scalar(end, pa.date32())),
                    )
                    survivors = table.filter(outside)
                retired = store.retire_partition(
                    REBUILD_DAILY_DATASET,
                    partition,
                    job_id=job_id,
                    reason=f"rebuild daily features {start}..{end} (BD-100)",
                )
                report["retired_files"] += len(retired)
                if survivors is not None and survivors.num_rows:
                    # Carried verbatim: a row outside the range keeps its own
                    # values, including a NULL `anchor_knowledge` where it
                    # predates the column. The rebuild never relabels history.
                    published = store.publish(REBUILD_DAILY_DATASET, survivors, job_id=job_id)
                    # A carried row was LIVE a moment ago and its old file is
                    # already retired. If the republish is short by even one row
                    # or quarantines any, that data is out of the live set and
                    # the run must say so loudly - a count that cannot fail is
                    # not evidence. The retired files are still on disk, so the
                    # repair is repointing the manifest.
                    if (
                        published.rows_published != survivors.num_rows
                        or published.rows_quarantined
                    ):
                        raise LakeIntegrityError(
                            f"{REBUILD_DAILY_DATASET}/{partition}: carried "
                            f"{published.rows_published} of {survivors.num_rows} out-of-range "
                            f"row(s), {published.rows_quarantined} quarantined; the retired "
                            "files are still on disk - repoint the manifest before re-running."
                        )
                    report["carried_rows"] += published.rows_published
            for day in sessions:
                report["steps"][day.isoformat()] = vars(
                    features.build_daily_snapshots(
                        store,
                        day,
                        anchors_by_symbol=anchor_dates_by_symbol(store, day),
                        now=stamp,
                        run_id=job_id,
                        job_id=job_id,
                    )
                )
            report["retired"] = vars(store.collect_retired(now=stamp))
    except SingleFlightError as exc:
        return {"status": "REFUSED", "applied": True, "reason": str(exc)}
    return report


def run_recompute_outcomes(
    store: ResearchStore | None,
    *,
    buckets=None,
    apply: bool = False,
    time_budget_minutes: float | None = None,
    session_date: date | None = None,
    now: datetime | None = None,
    job_id: str = "outcomes_recompute",
    lock_path: Path | None = None,
) -> dict:
    """Re-simulate every outcome bucket, terminal rows included (BD-98).

    The nightly build covers ONE of the 32 symbol buckets per firing and never
    re-simulates a terminal row. Both are right for a lake whose inputs were
    right; after the duplicated M5 bars of 2026-08/09 (BD-96/97) the stored
    outcomes were computed over doubled series, and the only repair is to walk
    every bucket with ``force``. A re-simulation that reproduces the stored
    result writes nothing; a changed result supersedes it, so the current view
    is repaired in place and the old row stays in the ledger.

    The lock is taken PER BUCKET, so a scheduled build slots in between two
    buckets instead of being refused for hours; ``time_budget_minutes`` stops
    starting new buckets once spent, and the per-bucket coverage lines
    (``outcome_coverage.record_firing``) say which buckets a later run still
    owes. Dry run by default: the plan and nothing else.
    """
    if store is None:
        return {"status": "DISABLED", "message": "research_store_dir is not configured."}
    stamp = now or utc_now()
    day = session_date or stamp.date()
    wanted = [int(b) % OUTCOME_BUCKETS for b in (buckets if buckets is not None else range(OUTCOME_BUCKETS))]
    report: dict = {
        "status": "OK",
        "applied": bool(apply),
        "session_date": day.isoformat(),
        "buckets_planned": wanted,
        "buckets_done": [],
        "buckets_skipped": [],
        "steps": {},
    }
    if not apply:
        return report
    import time as _time

    started = _time.monotonic()
    budget = None if time_budget_minutes is None else float(time_budget_minutes) * 60.0
    for bucket in wanted:
        # The first bucket always starts; the budget decides whether the NEXT one does.
        if budget is not None and report["buckets_done"] and (_time.monotonic() - started) >= budget:
            report["buckets_skipped"].append(bucket)
            report["status"] = "BUDGET_EXHAUSTED"
            continue
        try:
            with single_flight(lock_path):
                run_id = f"{job_id}-b{bucket:02d}"
                step = _run_outcomes(store, day, stamp, run_id, bucket=bucket, force=True)
                outcome_coverage.record_firing(store.root, step, run_id=run_id, now=stamp)
        except SingleFlightError as exc:
            report["buckets_skipped"].append(bucket)
            report["steps"][str(bucket)] = {"status": "REFUSED", "reason": str(exc)}
            continue
        except Exception as exc:  # noqa: BLE001 - one bucket must not end the walk
            report["buckets_skipped"].append(bucket)
            report["steps"][str(bucket)] = {"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}
            continue
        report["buckets_done"].append(bucket)
        report["steps"][str(bucket)] = {
            "status": step.get("status"),
            "symbols": step.get("symbols"),
            "occurrences": step.get("occurrences"),
            "m5_close_rows": (step.get("m5_close") or {}).get("rows"),
            "m5_close_skipped": (step.get("m5_close") or {}).get("skipped"),
        }
    report["seconds"] = round(_time.monotonic() - started, 1)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research_warehouse", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="seal the spool, wrap bronze, snapshot, derive")
    build.add_argument("--session-date", default="")
    build.add_argument("--run-id", default="")
    backfill_cmd = sub.add_parser(
        "backfill", help="net-new provider capture: nightly ETH, weekly sweep, or the one-time seed"
    )
    backfill_cmd.add_argument("--job", choices=(JOB_NIGHTLY, JOB_WEEKLY, JOB_SEED), default=JOB_NIGHTLY)
    backfill_cmd.add_argument("--session-date", default="", help="defaults to yesterday")
    backfill_cmd.add_argument("--run-id", default="")
    backfill_cmd.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="wall clock this run may spend waiting on the pacer (0 never waits)",
    )
    backfill_cmd.add_argument("--max-requests", type=int, default=None)
    sub.add_parser("status", help="lake inventory and health counters")
    restore = sub.add_parser("restore-check", help="restore one partition to a new root and verify it")
    restore.add_argument("--target", required=True)
    restore.add_argument("--dataset", default="bar_m5")
    restore.add_argument("--partition", default="")
    dedupe = sub.add_parser(
        "dedupe",
        help="drop rows repeated at the dataset grain (BD-96); a DRY RUN unless --apply is given",
    )
    dedupe.add_argument("--dataset", default="bar_m5")
    dedupe.add_argument("--partition", action="append", default=[], help="repeatable; default: every live partition")
    dedupe.add_argument("--apply", action="store_true", help="rewrite the partitions; without it only counts are printed")
    rebuild = sub.add_parser(
        "rebuild-month",
        help="retire a month's bar_derived + feature_snapshot_intraday partitions and recompute them (BD-96); DRY RUN unless --apply",
    )
    rebuild.add_argument("--month", required=True, help="YYYY-MM")
    rebuild.add_argument("--apply", action="store_true", help="retire and recompute; without it only the plan is printed")
    rebuild_daily = sub.add_parser(
        "rebuild-daily-features",
        help="recompute feature_snapshot_daily for a past date range WITH its anchors (BD-100); DRY RUN unless --apply",
    )
    rebuild_daily.add_argument("--from", dest="start", required=True, help="YYYY-MM-DD")
    rebuild_daily.add_argument("--to", dest="end", required=True, help="YYYY-MM-DD")
    rebuild_daily.add_argument(
        "--apply", action="store_true", help="retire and recompute; without it only the plan is printed"
    )
    coverage = sub.add_parser(
        "band-coverage",
        help="read-only: how much of a month's swing evidence had the bands its recipe needs (Q2.3)",
    )
    coverage.add_argument("--month", required=True, help="YYYY-MM")
    coverage.add_argument("--recipe", default="", help="one recipe id; default every D1 recipe")
    coverage.add_argument(
        "--compare",
        nargs=2,
        metavar=("RECIPE_A", "RECIPE_B"),
        default=None,
        help=(
            "two recipe ids side by side on the SAME occurrence ids (M4.3), "
            "e.g. --compare swing_house_v1 swing_house_variant_v1"
        ),
    )
    coverage.add_argument("--json", action="store_true", help="print the report object instead of the table")
    recompute = sub.add_parser(
        "recompute-outcomes",
        help="re-simulate every outcome bucket with force (BD-98); DRY RUN unless --apply",
    )
    recompute.add_argument("--buckets", default="", help="comma list or a-b range; default all 32")
    recompute.add_argument("--time-budget-minutes", type=float, default=None)
    recompute.add_argument("--session-date", default="")
    recompute.add_argument("--apply", action="store_true")

    args = parser.parse_args(argv)
    store = ResearchStore.open()
    if args.command == "status":
        print(json.dumps(run_status(store), indent=2, default=str))
        return 0
    if args.command == "dedupe":
        report = run_dedupe(store, dataset=args.dataset, partitions=args.partition or None, apply=bool(args.apply))
        print(json.dumps(report, indent=2, default=str))
        return 0
    if args.command == "rebuild-month":
        report = run_rebuild_month(store, month=args.month, apply=bool(args.apply))
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("status") in {"OK", "DISABLED"} else 1
    if args.command == "rebuild-daily-features":
        report = run_rebuild_daily_features(
            store,
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
            apply=bool(args.apply),
        )
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("status") in {"OK", "DISABLED"} else 1
    if args.command == "band-coverage":
        if args.compare:
            report = run_band_coverage_compare(
                store, month=args.month, recipe_ids=tuple(args.compare)
            )
            formatted = format_band_coverage_compare(report)
        else:
            report = run_band_coverage(store, month=args.month, recipe_id=args.recipe or None)
            formatted = format_band_coverage(report)
        print(json.dumps(report, indent=2, default=str) if args.json else formatted)
        return 0 if report.get("status") in {"OK", "DISABLED"} else 1
    if args.command == "recompute-outcomes":
        buckets = None
        if args.buckets:
            buckets = []
            for piece in str(args.buckets).split(","):
                piece = piece.strip()
                if "-" in piece:
                    low, high = piece.split("-", 1)
                    buckets.extend(range(int(low), int(high) + 1))
                elif piece:
                    buckets.append(int(piece))
        report = run_recompute_outcomes(
            store,
            buckets=buckets,
            apply=bool(args.apply),
            time_budget_minutes=args.time_budget_minutes,
            session_date=date.fromisoformat(args.session_date) if args.session_date else None,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("status") in {"OK", "DISABLED", "BUDGET_EXHAUSTED"} else 1
    if args.command == "build":
        day = date.fromisoformat(args.session_date) if args.session_date else None
        report = run_build(store, session_date=day, run_id=args.run_id)
        print(json.dumps({"status": report.status, "message": report.message, "steps": report.steps}, indent=2, default=str))
        return 0 if report.status in {"OK", "DISABLED"} else 1
    if args.command == "backfill":
        day = date.fromisoformat(args.session_date) if args.session_date else None
        report = run_backfill_job(
            store,
            job=args.job,
            session_date=day,
            time_budget_seconds=args.time_budget_seconds,
            max_requests=args.max_requests,
            run_id=args.run_id,
        )
        print(json.dumps(report, indent=2, default=str))
        # A missing cohort or provider is a condition to report, not a crash;
        # only a refused (racing) run is a non-zero exit.
        return 1 if report.get("status") == "REFUSED" else 0
    report = backup_mod.restore_check(
        store, args.target, dataset=args.dataset, partition=args.partition or None
    )
    print(json.dumps(vars(report), indent=2, default=str))
    return 0 if report.passed or report.status == "DISABLED" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
