"""TJ-13A item 4, second half: a projection says whether IT was cut.

MEASURED, read-only, over all 53 published evidence packages of the 2026-09-18
night (``\\\\MINI-PC\\Trading Bot Data\\ai_store\\briefs\\2026\\2026-09-18\\tickers``):

    packages          53, 6,476-9,675 chars each (per-item budget 22,000)
    flagged truncated 11 source ids, EVERY occurrence, max content 825 chars
                      (``MAX_TICKER_SOURCE_CHARS`` is 16,000)

    market.master_prep_state  18 of 18   max  107 chars
    setups.tier_performance   13 of 13   max  668
    daily.market_prep         11 of 11   max   96
    setups.short_horizon       8 of 8    max  250
    setups.current_tiers       6 of 6    max  674
    daily.auto_report          6 of 6    max  825
    setups.playbooks           6 of 6    max  245

So NEITHER budget constant was the cut, and the packet's premise ("fix the
truncation only if it is a budget constant") is refuted by its own numbers. The
flag was INHERITED: ``build_ticker_evidence`` does ``source = dict(raw)`` from
the SESSION-level package and then replaces ``content`` with the symbol
projection, so ``truncated`` rode along from the session budget onto a
projection that was never itself cut. The word "truncated" appears 63 times in
the live ``ai_morning_brief.txt``, and the model hedges accordingly -
"truncated, limiting the scope of the analysis" - about a 107-character source
it received whole.

Missing data is uncertainty, never confirmation; but a source that arrived WHOLE
is not missing data, and telling the model it is degraded is its own kind of
lying. The flag now describes THIS package.

NO MODEL IS CALLED HERE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import briefs  # noqa: E402

SESSION = "2026-09-18"


def _base(source_id, content, *, truncated):
    """A nightly package with ONE source, flagged the way the session flagged it."""
    return {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": "2026-09-19T02:00:00-04:00",
        "session_date": SESSION,
        "sources": [
            {
                "source_id": source_id,
                "label": source_id,
                "status": "available",
                "truncated": truncated,
                "content": content,
            }
        ],
        "coverage": {"counts": {}, "excluded": []},
    }


def _projected(package, source_id):
    return next(
        source
        for source in package["sources"]
        if source.get("source_id") == source_id
    )


def test_a_projection_that_was_not_cut_is_not_called_truncated():
    """The live defect, in miniature.

    The session had to truncate `market.master_prep_state` to fit the session
    budget. NVDA's slice of it is 40 characters and arrives whole, so nothing
    about THIS package is truncated and the model must not be told otherwise.
    """
    base = _base(
        "market.master_prep_state",
        "NVDA is above its anchor.\nAMD is below.\nTSLA is flat.",
        truncated=True,
    )

    package = briefs.build_ticker_evidence(base, "NVDA", [{"list": "focus_longs"}])
    source = _projected(package, "market.master_prep_state")

    assert source["truncated"] is False
    assert "NVDA" in source["content"]
    # ...and the coverage block agrees, because it counts the same field.
    assert package["coverage"]["counts"]["truncated"] == 0
    assert package["coverage"]["truncated"] == []


def test_a_projection_that_WAS_cut_still_says_so():
    """The other half, and it is the one that must not be lost.

    A symbol that really does have more rows than the per-source ceiling holds
    is genuinely reading part of its evidence, and `MAX_TICKER_SOURCE_CHARS` is
    where that happens. Nothing here may turn that into silence.
    """
    line = "NVDA held the anchored band on the retest and closed above it.\n"
    repeats = (briefs.MAX_TICKER_SOURCE_CHARS // len(line)) + 50
    base = _base("setups.tier_performance", line * repeats, truncated=False)

    package = briefs.build_ticker_evidence(base, "NVDA", [{"list": "focus_longs"}])
    source = _projected(package, "setups.tier_performance")

    assert source["truncated"] is True
    assert len(source["content"]) == briefs.MAX_TICKER_SOURCE_CHARS
    assert package["coverage"]["counts"]["truncated"] == 1
    assert [row["source_id"] for row in package["coverage"]["truncated"]] == [
        "setups.tier_performance"
    ]


def test_an_oversized_single_record_is_still_truncated():
    """The mapping branch of the projection has its own cut.

    `_extract_ticker_content` replaces a symbol record that will not fit with a
    `truncated_record` string, which is a cut by any other name.
    """
    base = _base(
        "setups.playbooks",
        {"symbol": "NVDA", "notes": "x" * (briefs.MAX_TICKER_SOURCE_CHARS + 2_000)},
        truncated=False,
    )

    package = briefs.build_ticker_evidence(base, "NVDA", [{"list": "focus_longs"}])
    source = _projected(package, "setups.playbooks")

    assert "truncated_record" in source["content"]
    assert source["truncated"] is True


def test_the_membership_source_is_never_truncated():
    """Guard: the one source this code WRITES cannot have been cut."""
    base = _base("daily.market_prep", "NVDA led the tape.", truncated=True)

    package = briefs.build_ticker_evidence(base, "NVDA", [{"list": "focus_longs"}])
    membership = _projected(package, briefs.MEMBERSHIP_SOURCE_ID)

    assert not membership.get("truncated")


def test_the_live_shape_stops_claiming_eleven_truncated_sources():
    """The measurement at the top of this file, replayed as a fixture.

    Eleven session-truncated sources, each of which projects to well under a
    hundred characters for this symbol. Before the repair every one of them
    reached the model flagged `truncated`; the brief said the word 63 times.
    """
    source_ids = [
        "market.master_prep_state",
        "setups.tier_performance",
        "daily.market_prep",
        "setups.short_horizon",
        "setups.current_tiers",
        "daily.auto_report",
        "setups.playbooks",
        "setups.scan_factors",
        "daily.master_events",
        "setups.type_stats",
        "setups.recent_type_stats",
    ]
    base = {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": "2026-09-19T02:00:00-04:00",
        "session_date": SESSION,
        "sources": [
            {
                "source_id": source_id,
                "label": source_id,
                "status": "available",
                "truncated": True,
                "content": f"NVDA row in {source_id}.\nAMD row in {source_id}.",
            }
            for source_id in source_ids
        ],
        "coverage": {"counts": {}, "excluded": []},
    }

    package = briefs.build_ticker_evidence(base, "NVDA", [{"list": "focus_longs"}])

    assert package["coverage"]["counts"]["truncated"] == 0
    encoded = json.dumps(package, sort_keys=True, default=str)
    assert '"truncated": true' not in encoded.lower()
