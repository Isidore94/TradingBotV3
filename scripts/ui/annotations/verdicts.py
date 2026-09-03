"""ONE writer for every like and every dislike, from every screen — P10 A1.

Trader, 2026-09-02: *"the veto and like+claim tabs are just quicker ways to make
a note for a stock"*, and — decisively — a star in Master AVWAP setups and a like
in chart review are **the same thing**: one bucket, graded together, and the
screen it came from is a column.

Before this, a like or a dislike could be written three different ways and only
one of them was graded:

* **Master AVWAP ★ / ✕** wrote a review event (`favorite` / `dislike`) plus, for
  the ✕, a `pick_feedback` row. The review event reaches the scoreboard and
  **no graded cohort at all**.
* **"Not today"** in chart review wrote a `pick_feedback` verdict with the
  hardcoded free-text reason `"not today"`. P5 grades it, as
  `focus__m5_not_today`.
* **The capture rail's like** wrote a `trader_annotations` `like_claim` row,
  which `like_cohort` grades.

So the trader's star on a D1 setup — the most considered judgement they make all
day — left no forward record, while the same opinion expressed two panels away
did. This module makes every one of them write the annotation row as well.

**Nothing existing changes meaning.** The review event, the `pick_feedback` row
and the Focus removal all still happen exactly as they did; several surfaces and
both the review scoreboard and the Focus store depend on them. The annotation row
is an ADDITION, and it is written FIRST, on the click, before any dialog opens —
a verdict must be on disk before anything can go wrong with the note that
explains it.

**A like still carries zero privileges** (plan.md P3.1). Nothing here reaches a
detector, a score, an alert, a watchlist, a Focus list, the review queue or
`review_policy.json`. It records; it does not act.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from project_paths import TRADER_ANNOTATIONS_FILE
from ui.annotations.store import (
    EVENT_LIKE_CLAIM,
    EVENT_VETO,
    LIKE_MODE_QUICK,
    SURFACE_CHART_REVIEW,
    SURFACE_FOCUS_PANEL,
    SURFACE_M5_ALERT_BAR,
    SURFACE_MASTER_AVWAP,
    SURFACE_RAIL,
    SURFACES,
    AnnotationError,
    record_annotation,
    record_annotation_with_bars,
)

__all__ = [
    "SURFACE_CHART_REVIEW",
    "SURFACE_FOCUS_PANEL",
    "SURFACE_M5_ALERT_BAR",
    "SURFACE_MASTER_AVWAP",
    "SURFACE_RAIL",
    "SURFACES",
    "record_dislike",
    "record_like",
    "record_not_today",
    "record_note_on",
    "scan_context_from_row",
]


def scan_context_from_row(row: Any) -> dict[str, Any]:
    """The scanner row under the click, as far as it can be read from it — B1.

    Reads only what the object already carries. It never looks a symbol up, and
    it never computes: a capture click costs one write, and a field that had to
    be fetched would either block the click or be wrong when the fetch failed.

    `canonical_setup_id` comes from P7's registry, which **RAISES on a name it
    does not know** — deliberately, so an unmapped family is discovered rather
    than silently filed under GENERAL. Here that refusal must not cost the
    verdict, so an unknown family leaves the field absent and the row still
    records what the trader thought. The registry's complaint belongs in the
    registry's own tests, not in the trader's click.

    Absent is a real answer. A bare symbol lookup has no row at all, and a row
    that reports "" for a field it never had would be indistinguishable from one
    where the scanner genuinely found nothing.
    """
    if row is None:
        return {}
    raw = getattr(row, "raw", None)
    raw = raw if isinstance(raw, dict) else {}

    def _first(*names: str) -> Any:
        for name in names:
            value = getattr(row, name, None)
            if value is None or str(value).strip() == "":
                value = raw.get(name)
            if value is not None and str(value).strip() != "":
                return value
        return None

    context: dict[str, Any] = {}
    for key, names in (
        ("scan_date", ("scan_date", "session_date", "as_of")),
        ("tracker_setup_id", ("setup_id", "tracker_setup_id")),
        ("priority_bucket", ("priority_bucket", "bucket")),
        ("score", ("score",)),
        ("expected_r", ("expected_r",)),
    ):
        value = _first(*names)
        if value is not None:
            context[key] = value

    family = _first("setup_family", "master_avwap_setup_family")
    if family:
        try:
            import setup_registry

            # `find` returns None for an unknown name where `resolve` raises;
            # either way the verdict is never at risk.
            entry = setup_registry.find(str(family))
            if entry:
                context["canonical_setup_id"] = setup_registry.canonical_setup_id(
                    str(family)
                )
        except Exception:
            # An unknown family, or a registry that will not load, costs the
            # canonical id and never the verdict.
            pass
    return context


def _validated_surface(surface: str) -> str:
    screen = str(surface or "").strip().lower()
    if screen not in SURFACES:
        raise AnnotationError(f"unknown surface {surface!r}; expected one of {SURFACES}")
    return screen


def record_like(
    *,
    symbol: Any,
    side: Any = "",
    surface: str,
    session_date: Any = None,
    like_mode: str = LIKE_MODE_QUICK,
    claimed_setup_id: str = "",
    note: Any = "",
    scan_context: Any = None,
    m5_bars: Any = (),
    timeframe: str = "",
    path: Path = TRADER_ANNOTATIONS_FILE,
    **extra: Any,
) -> dict[str, Any] | None:
    """One like, from one screen. Returns the written row, or None.

    `**extra` passes the rail's own fields through untouched - `last_price`,
    `ref_level_id`, `ref_level_family`. They are the CHART's context rather than
    the screen's, only the rail has them, and `build_annotation` already
    validates every one; naming them here would be a second list to keep in step
    with that one.

    `like_mode` defaults to QUICK because every surface added by P10 is a
    one-click verb: a star, a heart, a keystroke. Only the rail's Alt+K path
    passes CLAIMED, and only it supplies a `claimed_setup_id` — the store
    refuses the two in the wrong combination, which is what keeps a later split
    by mode meaningful.

    Bars ride along when the caller has them, through the writer a day-trade
    pass already uses, so an M5 like references the chart the trader was looking
    at (P9). A surface with no bars passes none and the row is written anyway.
    """
    return record_annotation_with_bars(
        EVENT_LIKE_CLAIM,
        symbol=symbol,
        side=side,
        session_date=session_date,
        like_mode=like_mode,
        claimed_setup_id=claimed_setup_id,
        note=note,
        surface=_validated_surface(surface),
        scan_context=scan_context,
        timeframe=timeframe,
        m5_bars=m5_bars,
        path=path,
        **extra,
    )


def record_dislike(
    *,
    symbol: Any,
    side: Any = "",
    surface: str,
    session_date: Any = None,
    reason_code: str = "",
    note: Any = "",
    scan_context: Any = None,
    vocabulary: Any = None,
    timeframe: str = "",
    path: Path = TRADER_ANNOTATIONS_FILE,
    **extra: Any,
) -> dict[str, Any] | None:
    """One dislike, from one screen. A veto row, coded or not.

    `reason_code` is OPTIONAL here and required nowhere: the Master AVWAP ✕
    already asks for a code from the versioned picklist and passes it, while
    "Not today" has never had one and the trader asked for a note box instead of
    a picklist. An uncoded row carries no `vocab_version` either and grades under
    `veto_uncoded` — see the store for why a version stamp without a code would
    be actively wrong.
    """
    return record_annotation(
        EVENT_VETO,
        symbol=symbol,
        side=side,
        session_date=session_date,
        reason_code=reason_code,
        note=note,
        vocabulary=vocabulary,
        surface=_validated_surface(surface),
        scan_context=scan_context,
        timeframe=timeframe,
        path=path,
        **extra,
    )


def record_not_today(
    *,
    symbol: Any,
    side: Any = "",
    session_date: Any = None,
    timeframe: str = "",
    path: Path = TRADER_ANNOTATIONS_FILE,
) -> dict[str, Any] | None:
    """"Not today" on a review chart: an UNCODED veto from `chart_review`.

    Its own name because the surface and the codelessness travel together and a
    caller should not be able to get one without the other. The verdict is a
    veto - the chart in front of the trader is not for today, which is exactly
    what a veto says - and it carries no code because this button has never had
    a picklist and the trader asked for a note box instead of one.

    The `pick_feedback` row this accompanies is written by the Focus service and
    is untouched: P5 grades it as `focus__m5_not_today`. Two records of one
    click, in two files, answering two different questions - which is already
    true of every other verdict on the desk.
    """
    return record_dislike(
        symbol=symbol,
        side=side,
        surface=SURFACE_CHART_REVIEW,
        session_date=session_date,
        timeframe=timeframe,
        path=path,
    )


def record_note_on(
    row: Any,
    note: Any,
    *,
    path: Path = TRADER_ANNOTATIONS_FILE,
) -> dict[str, Any] | None:
    """The note the trader typed after the click — a SECOND row, never an edit.

    The click row is already on disk and is never touched. This one repeats the
    identity (symbol, side, session, surface) so it stands alone if anything ever
    reads it without the join, and points at the click through `supersedes`.

    An empty note writes NOTHING and returns None. Escape on the dialog, or OK on
    an empty box, leaves exactly the click — which is the trader's own rule:
    *"sometimes I may not want to write a note but the fact I clicked like should
    be processed by the bot eventually."*
    """
    text = str(note or "").strip()
    if not text or not hasattr(row, "get"):
        return None
    event_id = str(row.get("event_id") or "").strip()
    if not event_id:
        return None
    kind = str(row.get("event_type") or "").strip().lower()
    fields: dict[str, Any] = {
        "symbol": row.get("symbol"),
        "side": row.get("side") or "",
        "session_date": row.get("session_date"),
        "note": text,
        "supersedes": event_id,
        "surface": row.get("surface") or "",
        "timeframe": row.get("timeframe") or "",
        "path": path,
    }
    if kind == EVENT_LIKE_CLAIM:
        fields["like_mode"] = row.get("like_mode") or LIKE_MODE_QUICK
        if row.get("claimed_setup_id"):
            fields["claimed_setup_id"] = row.get("claimed_setup_id")
    elif kind == EVENT_VETO:
        # Carried so the pair grades in ONE cohort. A note row that dropped the
        # code would land in `veto_uncoded` while its own click row sat in a
        # coded cohort, and the two would disagree about the same decision.
        if row.get("reason_code"):
            fields["reason_code"] = row.get("reason_code")
    return record_annotation(kind, **fields)
