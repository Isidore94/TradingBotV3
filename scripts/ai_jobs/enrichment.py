"""Journal enrichment — LOCAL-AI Phase 3 machinery, RUNS GATED (packet W6).

Authorized 2026-08-24 (`docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md`
§2) ahead of its phase gate, on the R10.I scaffolding pattern: built now, and
refusing to run until the gate it exists behind has passed.

**The gate is Phase 2's**: ten clean digest sessions. It is read from
`ai_jobs.digest.digest_gate_state` rather than restated, so there is one
definition of "ten clean digest sessions" in the repo. Enriching a journal from
a layer whose own facts have never been audited would be building on unverified
ground, and the phase order exists to stop exactly that. **Below the gate, no
model is called and nothing is written.**

**Advisory fields only, and structurally so.** R7's invariant I7 names what
belongs to the trader — tags, notes, reviews, planned stop/risk, tax status —
and no machine path writes any of it. This pass writes ONE table,
`ai_trade_enrichment`, through the `JournalStore` API, and that table is
append-only: a re-run adds a row rather than rewriting what an earlier night
believed. The trader's `trade_annotations` row is never opened.

**The vocabulary decides, the model proposes.** Tags come from
`docs/SETUPS_MAJOR.md` and `docs/SETUPS_TEST.md`; anything the model returns
that is not in that list is DROPPED and counted, because an invented family name
becomes a bucket nobody can compare against anything.

Nothing here reaches a detector, a score, an alert, a watchlist, Focus, the
review queue or `review_policy.json`.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_log = logging.getLogger(__name__)

ENRICHMENT_SCHEMA = "ai_trade_enrichment_v1"

GATE_NOT_MET_PREFIX = "ENRICHMENT GATE NOT MET."

#: Trades enriched in one night. A cap rather than the whole journal, because
#: this is a nightly pass over what is NEW and a backfill is a separate,
#: deliberate act.
MAX_TRADES_PER_NIGHT = 25

#: Bullet shape in the setup documents: ``- **Name** — description``.
_FAMILY_BULLET = re.compile(r"^\s*-\s+\*\*(?P<name>[^*]+)\*\*")


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def gate_state(digest_root: Path | None = None) -> dict[str, Any]:
    """Phase 2's counter, read rather than restated.

    An absent or unreadable digest store reads as UNMET - the conservative
    direction, and the only one that cannot turn scaffolding into a phase
    transition by accident.
    """
    from ai_jobs import digest

    try:
        root = Path(digest_root) if digest_root is not None else _digest_root()
    except Exception as exc:  # noqa: BLE001
        _log.debug("Enrichment gate: no digest store (%s).", exc)
        return digest.digest_gate_state(Path("this-path-does-not-exist"))
    return digest.digest_gate_state(root)


def _digest_root() -> Path:
    from ai_jobs import store

    return store.digests_dir(create=False)


# ---------------------------------------------------------------------------
# the vocabulary
# ---------------------------------------------------------------------------


def setup_vocabulary(paths: Sequence[Path] | None = None) -> tuple[str, ...]:
    """Family names as the setup documents state them, slugified.

    Read from the documents rather than duplicated in code for the reason AI-P5
    established elsewhere: a list written in two places drifts, and the copy
    nobody edits becomes a machine-written falsehood shipped as data.
    """
    if paths is None:
        docs = Path(__file__).resolve().parents[2] / "docs"
        paths = [docs / "SETUPS_MAJOR.md", docs / "SETUPS_TEST.md"]
    names: list[str] = []
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            match = _FAMILY_BULLET.match(line)
            if not match:
                continue
            slug = slugify(match.group("name"))
            if slug and slug not in names:
                names.append(slug)
    return tuple(names)


def slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower())
    return cleaned.strip("_")


def filter_tags(tags: Iterable[Any], *, vocabulary: Sequence[str]) -> tuple[list[str], list[str]]:
    """(kept, dropped). The vocabulary decides; the model only proposes."""
    allowed = {slugify(name) for name in vocabulary}
    kept: list[str] = []
    dropped: list[str] = []
    for tag in tags or ():
        slug = slugify(tag)
        if not slug:
            continue
        (kept if slug in allowed else dropped).append(slug)
    return kept, dropped


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------


def run_journal_enrichment(
    *,
    session_date: str = "",
    now: datetime | None = None,
    digest_root: Path | None = None,
    store: Any = None,
    review_rows: Sequence[Mapping[str, Any]] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Enrich the session's new journal rows. Refuses below the digest gate."""
    from ai_jobs import ledger as job_ledger

    moment = _now(now)
    day = str(session_date or moment.date().isoformat())
    state = gate_state(digest_root)
    if not state["window_met"]:
        # No model, no write, and a reason that says why in words. A pass that
        # ran anyway would produce advisory rows nobody could later separate
        # from ones written on audited ground.
        return {
            "status": job_ledger.STATUS_OK,
            "model": "",
            "reason": (
                f"{GATE_NOT_MET_PREFIX} {state['sessions_collected']} of "
                f"{state['sessions_required']} clean digest session(s); no model was "
                "called and nothing was written. Phase 3 waits for Phase 2's evidence "
                "to be audited, which is what the phase order is for."
            ),
            "outputs": [],
        }

    try:
        journal = store if store is not None else _journal_store()
    except Exception as exc:  # noqa: BLE001
        return {"status": job_ledger.STATUS_FAILED, "model": "",
                "reason": f"journal store unavailable: {exc}", "outputs": []}

    try:
        trades = _trades_for_session(journal, day)
    except Exception as exc:  # noqa: BLE001
        return {"status": job_ledger.STATUS_FAILED, "model": "",
                "reason": f"journal trades unreadable: {exc}", "outputs": []}
    if not trades:
        return {"status": job_ledger.STATUS_OK, "model": "",
                "reason": f"no journal trades for {day}; nothing to enrich", "outputs": []}

    vocabulary = setup_vocabulary()
    evidence_rows = list(review_rows) if review_rows is not None else _review_rows(day)
    model = ""
    enriched = 0
    failures: list[str] = []
    dropped_total = 0

    for trade in trades[:MAX_TRADES_PER_NIGHT]:
        try:
            result = _enrich_one(
                trade=trade, vocabulary=vocabulary, review_rows=evidence_rows,
                session_date=day,
            )
        except Exception as exc:  # noqa: BLE001 - one trade's failure is its own
            failures.append(f"{trade.get('trade_id')}: {type(exc).__name__}: {exc}")
            continue
        if result is None:
            continue
        model = model or str(result.get("model") or "")
        dropped_total += len(result.get("dropped_tags") or ())
        try:
            journal.save_ai_enrichment(
                trade_id=str(trade.get("trade_id")),
                session_date=day,
                summary=str(result.get("summary") or ""),
                tags=list(result.get("tags") or ()),
                evidence=list(result.get("evidence") or ()),
                model=str(result.get("model") or ""),
                now=moment.isoformat(timespec="seconds"),
            )
            enriched += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{trade.get('trade_id')}: save failed: {exc}")

    reason = (
        f"enriched {enriched} of {len(trades)} trade(s) for {day}"
        + (f"; {dropped_total} proposed tag(s) outside the vocabulary were dropped"
           if dropped_total else "")
        + (f"; {len(failures)} failure(s): " + "; ".join(failures[:3]) if failures else "")
    )
    if failures and not enriched:
        return {"status": job_ledger.STATUS_DEGRADED, "model": model,
                "reason": reason, "outputs": []}
    return {"status": job_ledger.STATUS_OK, "model": model, "reason": reason, "outputs": []}


def _journal_store():
    from journal_store import JournalStore
    from project_paths import JOURNAL_DB_FILE

    return JournalStore(Path(JOURNAL_DB_FILE))


def _trades_for_session(store: Any, day: str) -> list[dict[str, Any]]:
    """Closed trades on this session that carry no enrichment for it yet."""
    rows = store.list_trades(trade_date=day)
    fresh = []
    for row in rows:
        existing = store.list_ai_enrichment(str(row.get("trade_id")))
        if any(str(item.get("session_date")) == day for item in existing):
            continue
        fresh.append(dict(row))
    return fresh


def _review_rows(day: str) -> list[dict[str, Any]]:
    try:
        from review_events import load_review_events

        return [
            row for row in load_review_events()
            if str(row.get("trade_date") or "")[:10] == day
        ]
    except Exception:  # noqa: BLE001 - enrichment never fails over its context
        return []


def _evidence_links(trade: Mapping[str, Any], review_rows: Sequence[Mapping[str, Any]]):
    """Review decisions on this trade's symbol and side, as pointers.

    Deliberately a JOIN, not a judgement: what fired and what the trader did
    about it, named so a later reader can drill to the raw record.
    """
    symbol = str(trade.get("symbol") or "").upper()
    side = str(trade.get("direction") or "").upper()
    links = []
    for row in review_rows:
        if str(row.get("symbol") or "").upper() != symbol:
            continue
        if side and str(row.get("side") or "").upper() not in ("", side):
            continue
        links.append({
            "source_id": "review.alert_review_events",
            "selector": (
                f"trade_date={row.get('trade_date')}&symbol={symbol}"
                f"&review_record_id={row.get('review_record_id')}"
            ),
            "action": str(row.get("action") or ""),
        })
    return links[:5]


def _enrich_one(
    *,
    trade: Mapping[str, Any],
    vocabulary: Sequence[str],
    review_rows: Sequence[Mapping[str, Any]],
    session_date: str,
) -> dict[str, Any] | None:
    """One trade's advisory summary and tags. Medium tier; raises on failure."""
    import ai_summary

    if not ai_summary.local_provider_enabled():
        raise RuntimeError("local AI provider is not configured (ai_local_endpoint_url unset)")

    evidence = _evidence_package(
        trade=trade, vocabulary=vocabulary,
        links=_evidence_links(trade, review_rows), session_date=session_date,
    )
    result = ai_summary.request_ai_summary(
        provider="local",
        model=ai_summary.local_model("medium"),
        api_key="",
        evidence=evidence,
        timeout_seconds=900,
    )
    summary = result.get("summary") or {}
    proposed = _proposed_tags(summary)
    kept, dropped = filter_tags(proposed, vocabulary=vocabulary)
    return {
        "summary": _summary_text(summary),
        "tags": kept,
        "dropped_tags": dropped,
        "evidence": _evidence_links(trade, review_rows),
        "model": result.get("model", ""),
    }


def _proposed_tags(summary: Mapping[str, Any]) -> list[str]:
    for key in ("tags", "setups", "families"):
        value = summary.get(key)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        if isinstance(value, str) and value.strip():
            return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
    return []


def _summary_text(summary: Mapping[str, Any]) -> str:
    for key in ("headline", "summary", "what_worked", "lessons"):
        value = summary.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:2000]
        if isinstance(value, (list, tuple)) and value:
            return "; ".join(str(item) for item in value)[:2000]
    return ""


def _evidence_package(*, trade, vocabulary, links, session_date) -> dict[str, Any]:
    """One trade, its review evidence, and the closed vocabulary. Nothing else."""
    import hashlib

    content = {
        "trade": {
            key: trade.get(key)
            for key in (
                "trade_id", "symbol", "direction", "status", "trade_date",
                "opened_at", "closed_at", "net_pnl", "net_pnl_cad", "currency",
                "average_entry_price", "average_exit_price", "quantity_closed",
            )
        },
        "review_evidence": list(links),
        "allowed_setup_families": list(vocabulary),
        "instructions": (
            "Summarize this trade in one or two plain sentences and choose zero "
            "or more setup families from allowed_setup_families. Never invent a "
            "family name; an empty list is a valid answer. Do not give advice, "
            "and do not restate numbers you were not given."
        ),
    }
    encoded = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
    package = {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": _now().isoformat(timespec="seconds"),
        "session_date": session_date,
        "selected_scopes": ["journal_enrichment"],
        "scope_labels": ["One journal trade and its review evidence"],
        "source_count": 1,
        "sources": [{
            "source_id": "journal.trade",
            "label": f"Journal trade {trade.get('trade_id')}",
            "status": "available",
            "observed_at": _now().isoformat(timespec="seconds"),
            "content_through": session_date,
            "content_through_basis": "the session this trade closed on",
            "session_date": session_date,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "truncated": False,
            "content": content,
        }],
        "coverage": {"counts": {"requested": 1, "usable": 1, "stale": 0, "truncated": 0}},
        "safety_contract": {
            "purpose": "advisory journal enrichment; the trader's own tags and notes are untouched",
            "forbidden_effects": ["scanner scores", "watchlists", "alerts", "bot state", "orders"],
        },
        "scope_caveats": [
            "Tags outside allowed_setup_families are discarded by code before "
            "anything is stored.",
            "This is ADVISORY. The trader's tags, notes and planned risk are "
            "never read from or written by this pass.",
        ],
    }
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["evidence_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package["package_id"] = package["evidence_hash"][:16]
    return package
