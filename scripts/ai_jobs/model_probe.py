"""Measure a local model tier on THIS desk, once, and write the number down.

`plan.md` §12.4 TJ-13 item 7, decision 0021 answers 19-20, decision 0018's
2026-09-19 amendment.

The week story is meant to be written by the large local model
(`ai_local_model_large`, `hf.co/bartowski/google_gemma-3-27b-it-GGUF:Q3_K_M` on
this desk). In 476 live ledger rows that model has **never run**: every row
names a 12B. So the week slot's `reserve_minutes` - the number that decides
whether a slot may start at all - cannot be written down by anybody. It has to
be MEASURED on this box, and the measurement has to be a durable record rather
than a number in a commit message. That is this module, and
:mod:`ai_jobs.provider` is what reads it back.

**A probe is a MODEL LOAD, and it is treated as one.**

* It is night-only, seven days a week, and ``force`` does not buy the clock
  (TJ-13A item 1; trader, 2026-09-19). A 27B load at 14:00 on a Saturday is
  precisely the thing that rule is about.
* It refuses while the nightly runner holds the machine-local AI lock. The
  runner takes that lock for the whole night's slate, so a probe that ignored
  it would load a 27B beside a working 12B on a 32 GB box. Where
  ``ai_jobs.runner`` runs UNGUARDED when the machine offers no exclusion
  primitive at all, this refuses: the runner's unguarded fallback protects a
  night of cheap deterministic work, and what is at stake here is a second
  model load that the box may not survive. Uncertainty is not confirmation.
* It reads a COPY of one week's fact packs (`evidence_stats.WEEK_SESSIONS`),
  never the live store, and deletes the copy afterwards.
* It writes ONE ledger row, ``manual_test``, which publishes a real record and
  never counts as session coverage - a Saturday probe must not tell the next
  firing that Friday was served.

What it records, and what those four numbers mean:

``load_seconds``
    How long this desk needed before the model answered at all. A probe is ONE
    call, so it cannot separate the weight load from the prompt evaluation and
    the generation - and it does not pretend to. On the ``single_call_end_to_end``
    basis this is the WHOLE call, which is an upper bound on the load; when the
    server reports its own timings (llama.cpp's ``timings`` block, Ollama's
    native durations) the split is exact and the basis says so.
``tokens_per_second``
    Generated tokens over that same span: a lower bound on throughput on the
    end-to-end basis.
``peak_memory_mb``
    The peak memory in use **on the machine** while the model answered, beside
    a running desk - the number that says whether this box swaps. Not a delta:
    the model lives in the inference server's process, not in ours.
``context_tokens_accepted``
    What the SERVER reported it ingested (``usage.prompt_tokens``), never the
    configured ``ai_local_context_tokens``. A local server silently truncates
    and answers from what survived, and learning what the 27B really takes is
    half the reason for measuring it.

A reserve derived from an upper-bound load and a lower-bound rate is
conservative by construction, which is the direction a reserve should err in.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests

from ai_jobs import ledger, store, window
from swallowed import note_swallowed

_log = logging.getLogger(__name__)

#: This measurement's own ledger job name. Its own row kind, so no reader can
#: mistake a probe for one of the night's slots.
PROBE_JOB = "model_probe"

#: The tiers that exist. A typo must not quietly measure the 12B and file the
#: number under the 27B.
PROBE_TIERS = ("large", "medium")

#: Row field carrying the measurement. Read by :func:`latest_measurement`.
MEASUREMENT_FIELD = "model_probe"

#: What one probe reserves of the window before it is allowed to start. The
#: probe exists BECAUSE nobody knows what the large model costs, so this cannot
#: itself be derived; it is a floor generous enough for a cold 27B load plus one
#: bounded answer, and the window is eight hours wide.
PROBE_RESERVE_MINUTES = 45.0

#: Read timeout for the probe's one call, in seconds. The same 900 s every other
#: local session-scale call uses; ``ai_summary`` clamps it to its own cap.
PROBE_TIMEOUT_SECONDS = 900

#: How often the memory sampler looks, in seconds, while the call is in flight.
MEMORY_SAMPLE_SECONDS = 1.0

#: A single call cannot split the load from the work, so both numbers are
#: reported end to end and labelled with this.
BASIS_END_TO_END = "single_call_end_to_end"
#: ...unless the server reported its own timings, in which case the split is the
#: server's own.
BASIS_SERVER_TIMINGS = "server_reported_timings"

#: Multiplier applied to the derived reserve. A reserve is a refusal threshold,
#: not an estimate: being 25% pessimistic costs one late-window skip, being
#: optimistic runs a 27B into the opening bell.
RESERVE_SAFETY_FACTOR = 1.25


# ---------------------------------------------------------------------------
# the machine-local guard
# ---------------------------------------------------------------------------


def runner_lock():
    """The AI-jobs machine lock, taken without waiting.

    The DEFAULT guard for :func:`run_model_probe` (lead decision, 2026-09-20):
    a bare call - a REPL, a later slot, a script - must not be able to load a
    27B beside a running nightly job just by leaving an argument out. It was a
    caller-supplied parameter for one day, which meant the most dangerous call
    in this package was the one that named nothing.

    It is resolved through this module's own attribute at CALL time (see
    :data:`USE_RUNNER_LOCK`), so a harness hands the probe a free lock by
    patching ``model_probe.runner_lock`` rather than by passing an argument at
    every call site.
    """
    from local_writer_lock import local_writer_lock

    from ai_jobs.runner import RUNNER_LOCK_KEY

    return local_writer_lock(RUNNER_LOCK_KEY, timeout_seconds=0.0)


#: "Take the machine lock." A sentinel rather than ``lock=runner_lock``,
#: because a default argument binds the function object at import and could
#: then never be replaced; this resolves :func:`runner_lock` through the module
#: at call time. ``lock=None`` means the same thing - there is deliberately no
#: value of ``lock`` that means "load a model with no guard at all".
USE_RUNNER_LOCK = object()


# ---------------------------------------------------------------------------
# what the machine has in use, right now
# ---------------------------------------------------------------------------


def _memory_in_use_mb() -> tuple[float, str]:
    """Memory in use on this machine in MB, and how it was measured.

    ``psutil`` is not a dependency of ``requirements-core.txt`` and this module
    has to stay importable headlessly, so each platform is asked in its own
    words. A platform that cannot answer says ``unmeasured`` rather than 0.
    """
    import sys

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                used = int(status.ullTotalPhys) - int(status.ullAvailPhys)
                if used > 0:
                    return used / (1024.0 * 1024.0), "system_in_use"
        except Exception as swallowed_exc:  # a measurement must never be the thing that fails
            note_swallowed("Windows memory status reading unavailable", swallowed_exc, quiet=True)
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            values: dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    values[key.strip()] = int(parts[0])  # kB
            total = values.get("MemTotal", 0)
            available = values.get("MemAvailable", values.get("MemFree", 0))
            if total and total > available:
                return (total - available) / 1024.0, "system_in_use"
    except Exception as swallowed_exc:
        note_swallowed("/proc/meminfo memory reading unavailable", swallowed_exc, quiet=True)
    try:
        import resource

        peak = int(getattr(resource.getrusage(resource.RUSAGE_SELF), "ru_maxrss", 0) or 0)
        if peak > 0:
            # Linux reports kB, macOS bytes.
            mb = peak / 1024.0 if sys.platform != "darwin" else peak / (1024.0 * 1024.0)
            if mb > 0:
                return mb, "probe_process_rss"
    except Exception as exc:
        note_swallowed("process RSS memory reading unavailable", exc, quiet=True)
    return 0.0, "unmeasured"


class _MemoryPeak:
    """Samples memory in use while one call is in flight.

    A plain before/after pair would miss the load entirely: the weights come in
    and the peak is gone again by the time the answer arrives.
    """

    def __init__(self, interval: float = MEMORY_SAMPLE_SECONDS) -> None:
        self._interval = max(0.05, float(interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mb, self.basis = _memory_in_use_mb()
        self.baseline_mb = self.peak_mb

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            value, basis = _memory_in_use_mb()
            if value > self.peak_mb:
                self.peak_mb = value
                self.basis = basis

    def __enter__(self) -> "_MemoryPeak":
        self._thread = threading.Thread(
            target=self._run, name="model-probe-memory", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)
        value, basis = _memory_in_use_mb()
        if value > self.peak_mb:
            self.peak_mb = value
            self.basis = basis
        return False


# ---------------------------------------------------------------------------
# the week of packs the probe reads, on a copy
# ---------------------------------------------------------------------------


def _week_sessions() -> int:
    from evidence_stats import WEEK_SESSIONS

    return int(WEEK_SESSIONS)


def probe_evidence_package(
    packs: Sequence[Mapping[str, Any]], *, not_sent: int = 0
) -> dict[str, Any]:
    """One evidence package holding this week's packs and nothing else.

    The same SHAPE `ai_jobs.digest.narration_evidence_package` uses, so the
    shared validator applies unchanged: the model may cite only source ids that
    are in front of it. Bounded the way every other local package is bounded
    (TJ-13A) - what did not fit is COUNTED and said, never silently dropped.
    """
    import hashlib

    sources: list[dict[str, Any]] = []
    aliases: set[str] = set()
    for pack in packs:
        day = str(pack.get("session_date") or "")
        encoded = json.dumps(pack, sort_keys=True, default=str).encode("utf-8")
        sources.append(
            {
                "source_id": f"probe.facts.{day}",
                "label": f"Deterministic fact pack for {day}",
                "status": "available",
                "observed_at": pack.get("generated_at"),
                "content_through": day,
                "content_through_basis": "the session the pack describes",
                "session_date": day,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "truncated": False,
                "content": dict(pack),
            }
        )
        aliases.add(f"probe.facts.{day}")
    package: dict[str, Any] = {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_date": str(packs[-1].get("session_date") or "") if packs else "",
        "selected_scopes": ["model_probe"],
        "scope_labels": ["One week of deterministic daily fact packs"],
        "source_count": len(sources),
        "sources": sources,
        "citable_aliases": sorted(aliases),
        "coverage": {
            "counts": {
                "requested": len(sources) + max(0, int(not_sent)),
                "usable": len(sources),
                "stale": 0,
                "truncated": 0,
                "not_sent": max(0, int(not_sent)),
            },
            "note": (
                f"This package carries {len(sources)} fact pack(s); "
                f"{max(0, int(not_sent))} pack(s) outside this week or outside the "
                "evidence budget were not sent. It exists to MEASURE this desk's "
                "large local model, not to publish anything: nothing written from "
                "it reaches a detector, a score, an alert, a watchlist, Focus or "
                "the review queue."
            ),
        },
        "safety_contract": {
            "purpose": "timing measurement over already-complete fact packs",
            "forbidden_effects": [
                "scanner scores",
                "watchlists",
                "alerts",
                "bot state",
                "orders",
            ],
        },
        "scope_caveats": [
            "Every figure is one session's DISCOVERY. Do not describe it as a "
            "trend, a confirmation, or evidence about a setup.",
            "close_r and mfe_r/mae_r are result and opportunity. Never combine them.",
        ],
    }
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["evidence_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package["package_id"] = package["evidence_hash"][:16]
    return package


def _bounded_week_package(
    packs: Sequence[Mapping[str, Any]], *, not_sent: int, budget: int
) -> tuple[dict[str, Any], int]:
    """The package, cut to ``budget`` by dropping the OLDEST packs first."""
    kept = list(packs)
    dropped = int(not_sent)
    package = probe_evidence_package(kept, not_sent=dropped)
    if budget <= 0:
        return package, dropped
    while len(kept) > 1 and len(json.dumps(package, default=str)) > budget:
        kept = kept[1:]
        dropped += 1
        package = probe_evidence_package(kept, not_sent=dropped)
        _log.warning(
            "model probe: evidence trimmed to fit %s chars; %s pack(s) carried",
            budget,
            len(kept),
        )
    return package, dropped


def _copy_week_of_packs(packs_root: Path, destination: Path) -> tuple[list[dict], int]:
    """Copy the newest `WEEK_SESSIONS` fact packs and read them FROM the copy.

    The live store is read exactly once, to list and copy. Everything after
    this point touches the copy, which is deleted when the probe finishes.
    """
    from ai_jobs import digest

    entries = digest.read_fact_pack_files(Path(packs_root))
    if not entries:
        return [], 0
    week = _week_sessions()
    selected = entries[-week:]
    facts = Path(destination) / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    for path, _pack in selected:
        shutil.copy2(path, facts / path.name)
    packs = digest.read_fact_packs(Path(destination))
    return packs, max(0, len(entries) - len(selected))


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------


def _session_date(now: datetime | None) -> str:
    """The session this probe's row belongs to, or "" when unanswerable."""
    from market_calendar import SessionCalendarError

    from ai_jobs import runner

    try:
        return runner.session_date_for(now)
    except SessionCalendarError as exc:
        _log.warning("model probe: the session calendar cannot answer (%s)", exc)
        return ""


def _readable(value: float, digits: int = 3) -> float:
    """Round a measured number for reading, without rounding it away.

    A span under a millisecond is a harness rather than a model load, and
    rounding one to ``0.0`` would make a measurement that happened read exactly
    like one that never did.
    """
    number = float(value)
    if number <= 0 or number >= 0.001:
        return round(number, digits)
    return float(f"{number:.3e}")


def _server_timings(body: Mapping[str, Any]) -> tuple[float, float] | None:
    """``(load_seconds, tokens_per_second)`` when the server reported its own.

    llama.cpp's OpenAI-compatible server returns a ``timings`` block and Ollama
    returns native nanosecond durations on some builds. Neither is guaranteed,
    which is why the end-to-end basis exists.
    """
    timings = body.get("timings") if isinstance(body, Mapping) else None
    if isinstance(timings, Mapping):
        prompt_ms = timings.get("prompt_ms")
        predicted_ms = timings.get("predicted_ms")
        rate = timings.get("predicted_per_second")
        if all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in (prompt_ms, predicted_ms, rate)) and float(rate) > 0:
            return (float(prompt_ms) + float(predicted_ms)) / 1000.0, float(rate)
    load = body.get("load_duration") if isinstance(body, Mapping) else None
    eval_count = body.get("eval_count") if isinstance(body, Mapping) else None
    eval_duration = body.get("eval_duration") if isinstance(body, Mapping) else None
    if (
        isinstance(load, (int, float))
        and isinstance(eval_count, (int, float))
        and isinstance(eval_duration, (int, float))
        and float(eval_duration) > 0
    ):
        seconds = float(eval_duration) / 1_000_000_000.0
        return float(load) / 1_000_000_000.0, float(eval_count) / seconds
    return None


def _record(
    *,
    status: str,
    session_date: str,
    model: str,
    reason: str,
    ledger_path,
    error: str = "",
    measurement: Mapping[str, Any] | None = None,
    started_at: datetime | None = None,
    tokens: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One ledger row and the outcome the caller reads. Never raises."""
    extra = {MEASUREMENT_FIELD: dict(measurement)} if measurement else None
    try:
        ledger.record(
            job=PROBE_JOB,
            status=status,
            session_date=session_date,
            model=model,
            reason=reason,
            error=error,
            started_at=started_at,
            tokens=tokens or {},
            extra=extra,
            path=ledger_path,
        )
    except Exception as exc:  # an evidence store never costs the event
        _log.error("model probe: could not write its ledger row (%s)", exc)
    return {
        "status": status,
        "reason": reason,
        "measurement": dict(measurement) if measurement else None,
    }


def run_model_probe(
    *,
    tier: str = "large",
    packs_root: Path | None = None,
    now: datetime | None = None,
    force: bool = False,
    post: Callable[..., Any] = requests.post,
    ledger_path: Path | None = None,
    lock: Callable[[], Any] | None = USE_RUNNER_LOCK,
) -> dict[str, Any]:
    """Measure one local tier on this desk and write the measurement down.

    Returns ``{"status", "reason", "measurement"}``. ``measurement`` is ``None``
    for every refusal and every failure - there is no such thing as a partial
    measurement here.

    ``force`` re-spends the "already measured for this session" check and
    NOTHING else. It does not buy the clock (this is a model load) and it does
    not beat a held lock.

    ``lock`` is a zero-argument callable returning a context manager. It
    defaults to the machine's own AI-jobs lock; ``None`` means the same thing.
    A harness that wants a free lock patches :func:`runner_lock`.
    """
    if tier not in PROBE_TIERS:
        raise ValueError(
            f"unknown model tier {tier!r}; this desk configures {', '.join(PROBE_TIERS)}"
        )

    import ai_summary

    model = ai_summary.local_model(tier)
    session_date = _session_date(now)

    # 1. The clock. A probe LOADS the model, so --force never reaches this.
    allowed, reason = window.launch_allowed(now, reserve_minutes=PROBE_RESERVE_MINUTES)
    if not allowed:
        return _record(
            status=ledger.STATUS_SKIPPED,
            session_date=session_date,
            model=model,
            reason=(
                f"{reason}. A probe LOADS the model, so --force does not buy the "
                "clock: local inference is night-only, seven days a week"
            ),
            ledger_path=ledger_path,
        )

    # 2. One measurement a night is enough, unless it is asked for again.
    if not force and session_date:
        existing = _measurement_rows(tier=tier, ledger_path=ledger_path)
        if any(str(row.get("session_date") or "") == session_date for row in existing):
            return _record(
                status=ledger.STATUS_SKIPPED,
                session_date=session_date,
                model=model,
                reason=(
                    f"{model} was already measured for session {session_date}; "
                    "pass --force to measure it again rather than loading it twice"
                ),
                ledger_path=ledger_path,
            )

    # 3. The machine lock. Never a second model load beside a working one.
    from local_writer_lock import LocalLockUnavailable

    guard = runner_lock if lock is USE_RUNNER_LOCK or lock is None else lock
    with ExitStack() as stack:
        if guard is not None:
            try:
                stack.enter_context(guard())
            except LocalLockUnavailable as exc:
                return _record(
                    status=ledger.STATUS_SKIPPED,
                    session_date=session_date,
                    model=model,
                    reason=(
                        f"the machine-local AI lock is not free ({exc}); refusing to "
                        f"load {model} beside a job that may already be running"
                    ),
                    ledger_path=ledger_path,
                )
        return _measure(
            tier=tier,
            model=model,
            session_date=session_date,
            packs_root=packs_root,
            post=post,
            ledger_path=ledger_path,
        )


def _measure(
    *,
    tier: str,
    model: str,
    session_date: str,
    packs_root: Path | None,
    post: Callable[..., Any],
    ledger_path: Path | None,
) -> dict[str, Any]:
    """The measurement itself, always inside whatever guard the caller took."""
    import ai_summary

    if packs_root is None:
        try:
            packs_root = store.digests_dir(create=False)
        except Exception as exc:
            return _record(
                status=ledger.STATUS_SKIPPED,
                session_date=session_date,
                model=model,
                reason=f"the AI store's digests folder is unreachable ({exc})",
                ledger_path=ledger_path,
            )

    started_at = datetime.now().astimezone()
    workspace = Path(tempfile.mkdtemp(prefix="tbv3-model-probe-"))
    try:
        packs, outside_week = _copy_week_of_packs(Path(packs_root), workspace)
        if not packs:
            return _record(
                status=ledger.STATUS_SKIPPED,
                session_date=session_date,
                model=model,
                reason=(
                    f"no fact packs under {packs_root}; there is nothing to measure "
                    "the model against"
                ),
                ledger_path=ledger_path,
                started_at=started_at,
            )
        package, not_sent = _bounded_week_package(
            packs,
            not_sent=outside_week,
            budget=ai_summary.local_evidence_budget_chars(),
        )
        sent = len(package.get("sources") or [])

        body: dict[str, Any] = {}

        def _recording_post(url, **kwargs):
            response = post(url, **kwargs)
            try:
                payload = response.json()
            except Exception:  # a body we cannot read is not a failed call
                payload = None
            if isinstance(payload, Mapping):
                body.clear()
                body.update(payload)
            return response

        with _MemoryPeak() as memory:
            clock = time.perf_counter()
            try:
                result = ai_summary.request_ai_summary(
                    provider="local",
                    model=model,
                    api_key="",
                    evidence=package,
                    timeout_seconds=PROBE_TIMEOUT_SECONDS,
                    post=_recording_post,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - clock
                unreachable = ai_summary.is_endpoint_unreachable(exc)
                return _record(
                    status=ledger.STATUS_FAILED,
                    session_date=session_date,
                    model=model,
                    reason=(
                        f"{model} did not answer after {elapsed:.1f}s: {exc}"
                        if unreachable
                        else f"{model} could not be measured: {exc}"
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                    ledger_path=ledger_path,
                    started_at=started_at,
                )
            elapsed = max(time.perf_counter() - clock, 1e-6)

        usage = dict(result.get("usage") or {})
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        reported = _server_timings(body)
        if reported is not None:
            load_seconds, tokens_per_second = reported
            load_seconds = max(load_seconds, 1e-6)
            basis = BASIS_SERVER_TIMINGS
        else:
            # One call cannot split the load from the work. The whole call is
            # the load's upper bound and the end-to-end rate its lower bound.
            load_seconds = elapsed
            tokens_per_second = completion_tokens / elapsed if completion_tokens else 0.0
            basis = BASIS_END_TO_END

        measurement = {
            "model": model,
            "tier": tier,
            "packs": sent,
            "load_seconds": _readable(load_seconds),
            "tokens_per_second": _readable(tokens_per_second),
            "peak_memory_mb": round(float(memory.peak_mb), 1),
            "context_tokens_accepted": prompt_tokens,
            # provenance: what these four numbers are, and what they are not
            "basis": basis,
            "peak_memory_basis": memory.basis,
            "baseline_memory_mb": round(float(memory.baseline_mb), 1),
            "elapsed_seconds": _readable(elapsed),
            "completion_tokens": completion_tokens,
            "context_tokens_configured": int(ai_summary.local_context_tokens()),
            "packs_not_sent": int(not_sent),
            "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        throughput_note = (
            ""
            if tokens_per_second > 0
            else "; the server reported no completion tokens, so throughput is unmeasured"
        )
        return _record(
            status=ledger.STATUS_MANUAL,
            session_date=session_date,
            model=model,
            reason=(
                f"measured {model} on a copy of {sent} pack(s): "
                f"{measurement['load_seconds']}s to answer, "
                f"{measurement['tokens_per_second']} tok/s, "
                f"{measurement['peak_memory_mb']} MB peak in use, "
                f"{prompt_tokens} prompt token(s) accepted by the server "
                f"({basis}){throughput_note}"
            ),
            ledger_path=ledger_path,
            measurement=measurement,
            started_at=started_at,
            tokens=usage,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# ---------------------------------------------------------------------------
# reading the measurement back
# ---------------------------------------------------------------------------


def _measurement_rows(*, tier: str, ledger_path: Path | None) -> list[dict[str, Any]]:
    """Every probe row that actually carries a measurement for ``tier``."""
    try:
        target = Path(ledger_path) if ledger_path is not None else ledger.ledger_path(create=False)
    except Exception:
        return []
    try:
        rows = ledger._read_rows(target)
    except Exception:
        return []
    found: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("job") or "") != PROBE_JOB:
            continue
        if str(row.get("status") or "") != ledger.STATUS_MANUAL:
            continue
        payload = row.get(MEASUREMENT_FIELD)
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get("tier") or "") != str(tier):
            continue
        found.append(dict(row))
    return found


def latest_measurement(
    *, tier: str = "large", ledger_path: Path | None = None
) -> dict[str, Any] | None:
    """The newest recorded measurement for ``tier``, or None.

    The ledger is append-only, so two probes leave two rows and the desk the
    week story will run on is the one measured LAST. Nothing is averaged: an
    older number describes a machine that has since changed.
    """
    rows = _measurement_rows(tier=tier, ledger_path=ledger_path)
    if not rows:
        return None
    return dict(rows[-1][MEASUREMENT_FIELD])


def reserve_minutes_from_probe(
    *, tier: str = "large", ledger_path: Path | None = None
) -> float | None:
    """Minutes a slot on ``tier`` should reserve, DERIVED from the measurement.

    ``None`` means NEVER MEASURED, and it is never a default. "How long does a
    model nobody has ever run take" has one honest answer, and a slot handed a
    guessed reserve would reserve the window against a number from nowhere.

    The arithmetic is the load plus one bounded answer at the measured rate,
    with :data:`RESERVE_SAFETY_FACTOR` on top. The load is INSIDE that sum, so
    the reserve always covers it - the model has to BE there before it writes a
    word. (An explicit `max(minutes, load/60)` floor stood here until the
    2026-09-20 review pointed out that a positive generation term and a factor
    above 1.0 make it unreachable; a branch no input can take is not a
    safeguard, it is a claim nobody can check.)
    """
    measurement = latest_measurement(tier=tier, ledger_path=ledger_path)
    if not measurement:
        return None
    try:
        load_seconds = float(measurement.get("load_seconds") or 0.0)
        tokens_per_second = float(measurement.get("tokens_per_second") or 0.0)
    except (TypeError, ValueError):
        return None
    if load_seconds <= 0 or tokens_per_second <= 0:
        # A probe that could not measure throughput answers nothing about how
        # long a document takes. Unmeasured, not zero.
        return None

    import ai_summary

    generation_seconds = ai_summary.LOCAL_MAP_GENERATION_TOKENS / tokens_per_second
    minutes = (load_seconds + generation_seconds) * RESERVE_SAFETY_FACTOR / 60.0
    return round(minutes, 1)


#: Statuses whose `duration_seconds` is a real run of the slot.
_RAN_STATUSES = frozenset(
    {ledger.STATUS_OK, ledger.STATUS_DEGRADED, ledger.STATUS_FAILED, ledger.STATUS_MANUAL}
)


def measured_slot_minutes(
    *,
    ledger_path: Path | None = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
    sample: int = 5,
) -> dict[str, float]:
    """Median minutes of each job's last ``sample`` real runs, from the ledger (P1-3 3a).

    A job with no recorded run is absent: unmeasured, never zero.
    """
    if rows is None:
        try:
            target = (
                Path(ledger_path) if ledger_path is not None else ledger.ledger_path(create=False)
            )
            rows = ledger._read_rows(target)
        except Exception:  # noqa: BLE001 - no ledger means nothing measured
            return {}
    durations: dict[str, list[float]] = {}
    for row in rows:
        if str(row.get("status") or "") not in _RAN_STATUSES:
            continue
        try:
            seconds = float(row.get("duration_seconds") or 0.0)
        except (TypeError, ValueError):
            continue
        if seconds <= 0:
            continue
        durations.setdefault(str(row.get("job") or ""), []).append(seconds)
    out: dict[str, float] = {}
    for job, values in durations.items():
        recent = sorted(values[-max(1, int(sample)) :])
        middle = len(recent) // 2
        median = (
            recent[middle] if len(recent) % 2 else (recent[middle - 1] + recent[middle]) / 2.0
        )
        out[job] = round(median / 60.0, 2)
    return out


def describe_measurement(
    *, tier: str = "large", ledger_path: Path | None = None
) -> str:
    """One plain line for an operator: what was measured, or that nothing was."""
    measurement = latest_measurement(tier=tier, ledger_path=ledger_path)
    if not measurement:
        return (
            f"the {tier} local model has never been measured on this desk; "
            f"run `python scripts/run_ai_jobs.py --probe-model {tier}` inside the "
            "night window, with no AI job running"
        )
    reserve = reserve_minutes_from_probe(tier=tier, ledger_path=ledger_path)
    return (
        f"{measurement.get('model')}: {measurement.get('load_seconds')}s to answer, "
        f"{measurement.get('tokens_per_second')} tok/s, "
        f"{measurement.get('peak_memory_mb')} MB peak in use, "
        f"{measurement.get('context_tokens_accepted')} prompt token(s) accepted, "
        f"measured {measurement.get('measured_at')} "
        f"({measurement.get('basis')}) -> reserve "
        + (f"{reserve} min" if reserve is not None else "unmeasured")
    )
