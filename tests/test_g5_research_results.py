"""Packet G5 - Research > Results, the PURE module (`scripts/research_results.py`).

Written by the TESTER on `claude/g5-research-results`, off `origin/main`
`7e018c99` (ST1-ST7 landed), and proven RED there before any of it existed. The
builder makes these pass; it may ADD tests and may not weaken, skip or delete
one.

Trader, 2026-09-06 (decision 0016 answer 7, AMENDED that day): *"Research gains
a **Results** landing page the trader may read - Bot setups / My trades and
Swing / Day trading kept as four separate populations, never pooled. It is the
FULL readout; the Desk's 'what is working lately' line remains the primary
surface, and both read ONE evidence snapshot."* Decision 3 of that day: Results
opens on **Bot setups x Swing x Recent 20 sessions**.

===========================================================================
THE API THIS FILE PINS  (`scripts/research_results.py`)
===========================================================================

Frozen dataclasses, all of them::

    ResultsRow(cell, line, reason, eligible, values, display)
        `cell` is the snapshot's own `working_lately.EvidenceCell` for a bot
        row and `None` for a "My trades" bucket row. `values` carries the RAW
        numbers - the SAME OBJECTS the cell holds, never re-derived - and
        `display` the strings a table shows.

    ResultsBands(stronger, weaker, not_enough)   # tuples of ResultsRow

    ResultsSection(key, title, kind, sentence, verdict_line, bands, rows,
                   studies, stats, analytics)

    ResultsView(population, horizon, window, window_label, window_start,
                window_end, sections, freshness_line)

    band_cells(cells) -> ResultsBands
        The banding rule, alone and testable. **Raises `ValueError` on a
        mixed-kind cell list**, exactly as `working_lately.pool_cells` refuses
        across the axes: the refusal IS the mechanism.

    build_results_view(*, population, horizon, window, snapshot,
                       journal_trades, as_of, currency_mode) -> ResultsView
        `population` in {"bot", "mine"}, `horizon` in {"swing", "day"},
        `window` in {"recent", "all", (start, end)}.

The layout lane's one rule, from the packet: **the page may compute a VIEW
(ordering, banding, labels) from existing statistics and may never compute a
new statistic, threshold or eligibility rule.** No number in the view is
derived here; `rate * n` never appears.

===========================================================================
THE FIXTURE
===========================================================================

One real `working_lately.EvidenceSnapshot.to_payload()` dict - built from real
`EvidenceCell`s through the real `select_cell_leader` - carrying cells of all
three `SNAPSHOT_KINDS`, both namespaces, and eligible / below-floor /
concentrated / unmeasured cells. Cells are handed to the snapshot in SCRAMBLED
order so an implementation that trusts the input order cannot pass.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


AS_OF = date(2026, 9, 4)
SESSION = AS_OF.isoformat()

#: Deliberately above CPython's small-int cache so a re-derived count
#: (`n_wins + n_losses`) is a DIFFERENT object and `is` can tell them apart.
BIG = 613


# ---------------------------------------------------------------------------
# the fixture snapshot
# ---------------------------------------------------------------------------


def _cell(**kwargs):
    """One `EvidenceCell` with the boring fields filled in."""
    from working_lately import EvidenceCell

    base = dict(
        outcome_kind="trade_r_representative_exit",
        outcome_version="recent_types_v2",
        knowledge_basis="entry_scan_row_close_to_representative_exit",
        horizon="30d lookback, representative exit",
        window_sessions=20,
        latest_measured_session=SESSION,
        n_pending=0,
        n_excluded=0,
        n_symbols=17,
        n_sessions=12,
        top_symbol_share=0.21,
        top_session_share=0.18,
        statistic_name="win rate (closed, unweighted)",
        uncertainty_kind="wilson lower bound",
        namespace="live",
        n_floor=30,
        meets_floor=True,
    )
    base.update(kwargs)
    return EvidenceCell(**base)


#: (family, side, statistic, uncertainty_low). Six ELIGIBLE swing_trade_r cells,
#: so "top three by lower bound" and "three lowest statistic" PARTITION them and
#: the never-both claim is decidable.
ELIGIBLE_SWING_R = (
    ("alpha", "LONG", 0.72, 0.61),
    ("beta", "LONG", 0.68, 0.57),
    ("gamma", "SHORT", 0.64, 0.53),
    ("delta", "LONG", 0.56, 0.45),
    ("epsilon", "SHORT", 0.52, 0.41),
    ("zeta", "SHORT", 0.48, 0.37),
)

STRONGER_EXPECTED = ["LONG alpha", "LONG beta", "SHORT gamma"]
WEAKER_EXPECTED = ["SHORT zeta", "SHORT epsilon", "LONG delta"]


def _swing_trade_r_cells():
    cells = [
        _cell(
            kind="swing_trade_r",
            side=side,
            family=family,
            n_eligible=BIG + index,
            n_graded=BIG + index,
            statistic=stat,
            uncertainty_low=low,
        )
        for index, (family, side, stat, low) in enumerate(ELIGIBLE_SWING_R)
    ]
    cells.append(
        # BELOW THE FLOOR. Its statistic is the best on the page, which is the
        # whole point: eleven graded outcomes are not evidence.
        _cell(
            kind="swing_trade_r",
            side="LONG",
            family="thin",
            n_eligible=11,
            n_graded=11,
            statistic=0.91,
            uncertainty_low=0.80,
            meets_floor=False,
        )
    )
    cells.append(
        # CONCENTRATED: four fifths of the sample is one name.
        _cell(
            kind="swing_trade_r",
            side="SHORT",
            family="narrow",
            n_eligible=BIG,
            n_graded=BIG,
            statistic=0.88,
            uncertainty_low=0.79,
            n_symbols=3,
            top_symbol_share=0.81,
            top_session_share=0.22,
        )
    )
    cells.append(
        # UNMEASURED: the group exists, the statistic was never taken.
        _cell(
            kind="swing_trade_r",
            side="LONG",
            family="unmeasured",
            n_eligible=0,
            n_graded=0,
            n_pending=44,
            statistic=None,
            uncertainty_low=None,
        )
    )
    cells.append(
        # A STUDY. Highest bound on the page; an unpromoted idea may never lead.
        _cell(
            kind="swing_trade_r",
            side="LONG",
            family="studyfam",
            n_eligible=BIG,
            n_graded=BIG,
            statistic=0.95,
            uncertainty_low=0.90,
            namespace="study",
        )
    )
    return cells


def _swing_favorable_cells():
    shared = dict(
        kind="swing_favorable",
        outcome_kind="favorable_direction",
        outcome_version="favorable_direction_session_v2",
        knowledge_basis="scan row close to target session close",
        horizon="5 sessions",
        statistic_name="favorable direction (percent)",
        uncertainty_kind="wilson lower bound (percent)",
    )
    return [
        _cell(
            side="LONG",
            family="fav_a",
            n_eligible=BIG,
            n_graded=BIG,
            statistic=58.0,
            uncertainty_low=49.0,
            **shared,
        ),
        _cell(
            side="SHORT",
            family="fav_b",
            n_eligible=BIG,
            n_graded=BIG,
            statistic=41.0,
            uncertainty_low=33.0,
            **shared,
        ),
        _cell(
            side="LONG",
            family="fav_study",
            n_eligible=BIG,
            n_graded=BIG,
            statistic=99.0,
            uncertainty_low=97.0,
            namespace="study",
            **shared,
        ),
    ]


def _daytrade_cells():
    from working_lately import HELD_RUN_STATISTIC_NAME

    shared = dict(
        kind="daytrade_held_run",
        outcome_kind="held_run",
        outcome_version="held_run_v1",
        knowledge_basis="m5 alert to 30-minute hold",
        horizon="30 minutes held, session MFE",
        statistic_name=HELD_RUN_STATISTIC_NAME,
        uncertainty_kind="bootstrap 5th percentile",
        n_floor=20,
    )
    return [
        _cell(
            side="LONG",
            family="vwap_reclaim",
            n_eligible=BIG,
            n_graded=0,
            statistic=0.44,
            uncertainty_low=0.31,
            **shared,
        ),
        _cell(
            side="SHORT",
            family="ema_15",
            n_eligible=BIG,
            n_graded=0,
            statistic=0.28,
            uncertainty_low=0.19,
            **shared,
        ),
        _cell(
            side="LONG",
            family="day_thin",
            n_eligible=7,
            n_graded=0,
            statistic=0.90,
            uncertainty_low=0.71,
            meets_floor=False,
            **shared,
        ),
        _cell(
            side="SHORT",
            family="day_study",
            n_eligible=BIG,
            n_graded=0,
            statistic=0.99,
            uncertainty_low=0.95,
            namespace="study",
            **shared,
        ),
    ]


@pytest.fixture(scope="module")
def snapshot_payload():
    """A real `EvidenceSnapshot.to_payload()`, cells handed over SCRAMBLED."""
    import working_lately
    from working_lately import EvidenceSnapshot, SNAPSHOT_KINDS, select_cell_leader

    cells = _swing_trade_r_cells() + _swing_favorable_cells() + _daytrade_cells()
    # Nothing about the page may depend on the order the cells arrive in.
    scrambled = [cells[index] for index in range(len(cells) - 1, -1, -1)]

    verdicts = {
        kind: select_cell_leader(
            scrambled,
            kind=kind,
            last_completed_session=AS_OF,
            previous=None,
            source_rows=len([cell for cell in scrambled if cell.kind == kind]),
        )
        for kind in SNAPSHOT_KINDS
    }
    snapshot = EvidenceSnapshot(
        snapshot_id="g5fixture0000000000000000000000000000000",
        as_of=SESSION,
        built_at="2026-09-04T17:31:00-04:00",
        cells=tuple(scrambled),
        verdicts=verdicts,
        sources={
            "swing_trade_r": {
                "path": "master_avwap_setup_type_recent_stats.csv",
                "mtime": "2026-09-04T17:20:11-04:00",
                "rows": 10,
                "rows_by_session": {SESSION: 10},
            },
            "swing_favorable": {
                "path": "swing_favorable_direction.csv",
                "mtime": "2026-09-04T17:20:44-04:00",
                "rows": 3,
                "rows_by_session": {SESSION: 3},
            },
            "daytrade_held_run": {
                "path": "held_run_episodes.jsonl",
                "mtime": "2026-09-04T17:21:02-04:00",
                "rows": 4,
                "rows_by_session": {SESSION: 4},
            },
        },
    )
    payload = snapshot.to_payload()
    assert payload["schema"] == working_lately.SNAPSHOT_SCHEMA
    return payload


# ---------------------------------------------------------------------------
# the fixture journal trades
# ---------------------------------------------------------------------------


def _trade(
    trade_id,
    *,
    opened_at,
    closed_at,
    net_pnl,
    commission=0.0,
    fees=0.0,
    setup_tags="",
    tag_status="confirmed",
    planned_risk=None,
    symbol="AAA",
):
    """A `ui.models.journal.JournalTrade` built the way `load_trades` builds one."""
    from ui.models.journal import JournalTrade

    row = {
        "trade_id": trade_id,
        "trade_date": str(closed_at)[:10],
        "symbol": symbol,
        "direction": "LONG",
        "status": "CLOSED",
        "quantity_closed": 100,
        "average_entry_price": 10.0,
        "average_exit_price": 11.0,
        "net_pnl": net_pnl,
        "commission": commission,
        "fees": fees,
        "currency": "USD",
        "account_label": "MAIN",
        "account_number": "U1",
        "broker": "IBKR",
        # An old row has the key PRESENT and EMPTY, never absent.
        "setup_tags": setup_tags,
        "display_tags": setup_tags,
        "auto_tag_summary": "",
        "tag_status": tag_status,
        "notes": "",
        "opened_at": opened_at,
        "closed_at": closed_at,
        "planned_risk": planned_risk,
        "instrument_kind": "STOCK",
    }
    return JournalTrade.from_mapping(row)


@pytest.fixture(scope="module")
def journal_trades():
    return [
        # Opened and closed inside ONE exchange session, with real clock times.
        _trade(
            "T-DAY",
            opened_at="2026-09-02T09:45:00",
            closed_at="2026-09-02T14:10:00",
            net_pnl=120.0,
            commission=3.0,
            fees=2.0,
            setup_tags="avwap-reclaim",
            tag_status="confirmed",
        ),
        # Same day, LOSING, PROVISIONAL tag, and the only one carrying R.
        _trade(
            "T-DAY2",
            symbol="BBB",
            opened_at="2026-09-04T09:31:00",
            closed_at="2026-09-04T15:55:00",
            net_pnl=-40.0,
            commission=2.0,
            fees=1.0,
            setup_tags="machine-guess",
            tag_status="provisional",
            planned_risk=80.0,
        ),
        # Across sessions.
        _trade(
            "T-SWING",
            symbol="CCC",
            opened_at="2026-09-01T10:05:00",
            closed_at="2026-09-04T11:30:00",
            net_pnl=250.0,
            commission=4.0,
            fees=0.0,
        ),
        # A BROKER FILE row: midnight market-local at both ends is "the time is
        # not known" (`journal_trade_shape.is_date_only`), never a day trade.
        _trade(
            "T-DATEONLY",
            symbol="DDD",
            opened_at="2026-09-03T00:00:00",
            closed_at="2026-09-03T00:00:00",
            net_pnl=15.0,
            commission=1.0,
            fees=0.0,
        ),
    ]


# ---------------------------------------------------------------------------
# helpers over the view
# ---------------------------------------------------------------------------


def _view(population, horizon, snapshot, trades, window="recent"):
    import research_results

    return research_results.build_results_view(
        population=population,
        horizon=horizon,
        window=window,
        snapshot=snapshot,
        journal_trades=trades,
        as_of=AS_OF,
        currency_mode="native",
    )


def _cell_names(view):
    """Every EvidenceCell identity anywhere in a view - bands, table, studies."""
    out = set()
    for section in view.sections:
        streams = list(section.rows) + list(section.studies)
        streams += list(section.bands.stronger)
        streams += list(section.bands.weaker)
        streams += list(section.bands.not_enough)
        for row in streams:
            if row.cell is not None:
                out.add((row.cell.kind, row.cell.name))
    return out


def _band_names(section, band):
    return [row.cell.name for row in getattr(section.bands, band) if row.cell is not None]


def _code_lines(source: str) -> list[str]:
    """The module's EXECUTABLE lines - `#` comments and docstrings removed.

    A source-level guard that convicted a module for describing its own rule in
    a docstring would push the sentence out of the file, which is the opposite
    of what it is for.
    """
    out: list[str] = []
    fence = ""
    for raw in source.splitlines():
        line = raw
        if fence:
            if fence in line:
                line = line.split(fence, 1)[1]
                fence = ""
            else:
                continue
        while True:
            opener = min(
                (pos for pos in (line.find('"""'), line.find("'''")) if pos >= 0),
                default=-1,
            )
            if opener < 0:
                break
            mark = line[opener : opener + 3]
            rest = line[opener + 3 :]
            if mark in rest:
                line = line[:opener] + rest.split(mark, 1)[1]
                continue
            line = line[:opener]
            fence = mark
            break
        line = line.split("#", 1)[0]
        if line.strip():
            out.append(line)
    return out


def _section(view, key):
    for section in view.sections:
        if section.key == key:
            return section
    raise AssertionError(
        f"no section {key!r} in {[section.key for section in view.sections]}"
    )


# ===========================================================================
# 1
# ===========================================================================


def test_the_four_selections_never_share_a_cell_and_a_mixed_kind_list_raises(
    snapshot_payload, journal_trades
):
    """Item G5.3.1. Four populations, never pooled - and the refusal is real.

    Decision 0016 as amended: *"Bot setups / My trades and Swing / Day trading
    kept as four separate populations, never pooled."* A cell that appeared
    under two selections would be that pooling wearing two labels.
    """
    import research_results
    from working_lately import cells_from_payload

    bot_swing = _view("bot", "swing", snapshot_payload, journal_trades)
    bot_day = _view("bot", "day", snapshot_payload, journal_trades)
    mine_swing = _view("mine", "swing", snapshot_payload, journal_trades)
    mine_day = _view("mine", "day", snapshot_payload, journal_trades)

    swing_names = _cell_names(bot_swing)
    day_names = _cell_names(bot_day)
    assert swing_names, "bot/swing showed no cells at all"
    assert day_names, "bot/day showed no cells at all"
    assert swing_names.isdisjoint(day_names), (
        f"a cell appears under BOTH bot selections: {sorted(swing_names & day_names)}"
    )

    assert {kind for kind, _ in swing_names} == {"swing_trade_r", "swing_favorable"}
    assert {kind for kind, _ in day_names} == {"daytrade_held_run"}

    # The two swing kinds are two labelled SECTIONS and are never one list.
    assert [section.kind for section in bot_swing.sections] == [
        "swing_trade_r",
        "swing_favorable",
    ], "bot/swing must show the two swing kinds as two sections, in this order"

    # "My trades" is a different population and carries no snapshot cell at all.
    assert not _cell_names(mine_swing), "a snapshot cell leaked into My trades / Swing"
    assert not _cell_names(mine_day), "a snapshot cell leaked into My trades / Day"

    # The banding helper refuses a mixed-kind list the way `pool_cells` does.
    mixed = cells_from_payload(snapshot_payload)
    assert len({cell.kind for cell in mixed}) > 1
    with pytest.raises(ValueError):
        research_results.band_cells(mixed)

    # A study never enters a band, under any selection.
    for view in (bot_swing, bot_day):
        for section in view.sections:
            for band in ("stronger", "weaker", "not_enough"):
                for row in getattr(section.bands, band):
                    assert row.cell is None or row.cell.namespace != "study", (
                        f"study cell {row.cell.name!r} entered the {band} band"
                    )
    study_names = {
        row.cell.name
        for view in (bot_swing, bot_day)
        for section in view.sections
        for row in section.studies
        if row.cell is not None
    }
    assert {"LONG studyfam", "LONG fav_study", "SHORT day_study"} <= study_names, (
        "the study cells are not listed under their own label"
    )


# ===========================================================================
# 2
# ===========================================================================


def test_an_eligible_cell_lands_in_one_band_and_an_ineligible_one_names_its_reason(
    snapshot_payload, journal_trades
):
    """Item G5.3.2. Stronger, Weaker, and "Not enough evidence" with the WHY.

    Stronger is the eligible cells by `uncertainty_low` DESCENDING, top three;
    Weaker the three with the lowest `statistic`; the fixture's six eligible
    cells partition exactly, so "never both" is decidable rather than lucky.
    The cells were handed to the snapshot in reverse order, so an
    implementation that trusts the input order fails here.
    """
    view = _view("bot", "swing", snapshot_payload, journal_trades)
    section = _section(view, "swing_trade_r")

    stronger = _band_names(section, "stronger")
    weaker = _band_names(section, "weaker")

    assert stronger == STRONGER_EXPECTED, (
        f"Stronger lately must be the top three by uncertainty_low descending, got {stronger}"
    )
    assert weaker == WEAKER_EXPECTED, (
        f"Weaker lately must be the three lowest statistic, got {weaker}"
    )
    assert not set(stronger) & set(weaker), (
        f"a cell is in BOTH bands: {sorted(set(stronger) & set(weaker))}"
    )
    assert set(stronger) | set(weaker) == {
        f"{side} {family}" for family, side, _stat, _low in ELIGIBLE_SWING_R
    }, "the six eligible cells must partition across the two bands"

    not_enough = {
        row.cell.name: row.reason
        for row in section.bands.not_enough
        if row.cell is not None
    }
    assert "LONG thin" in not_enough, "a below-floor cell is not in Not enough evidence"
    thin_reason = not_enough["LONG thin"]
    assert "11" in thin_reason and "30" in thin_reason, (
        f"the below-floor reason must carry n_graded and n_floor, got {thin_reason!r}"
    )

    assert "SHORT narrow" in not_enough, "a concentrated cell is not in Not enough evidence"
    narrow_reason = not_enough["SHORT narrow"]
    assert "0.81" in narrow_reason, (
        f"the concentrated reason must name the share, got {narrow_reason!r}"
    )

    assert "LONG unmeasured" in not_enough
    assert not_enough["LONG unmeasured"].strip(), "an unmeasured cell got a blank reason"

    # Every ineligible cell is accounted for, and no eligible one is here.
    assert set(not_enough) == {"LONG thin", "SHORT narrow", "LONG unmeasured"}

    # The kind's verdict is printed above the bands VERBATIM.
    verdict = snapshot_payload["verdicts"]["swing_trade_r"]
    assert verdict["reason"] and verdict["reason"] in section.verdict_line, (
        f"the verdict reason is not printed verbatim: {section.verdict_line!r}"
    )
    assert verdict["state"] in section.verdict_line


# ===========================================================================
# 3
# ===========================================================================


def test_no_number_in_the_view_is_re_derived_from_the_cell(
    snapshot_payload, journal_trades
):
    """Item G5.3.3. The layout lane's whole rule, asserted by IDENTITY.

    A copied number is the SAME OBJECT; `n_wins + n_losses`, `round(rate, 4)` or
    `rate * n` all build a new one. The counts in the fixture sit above CPython's
    small-int cache on purpose, so `is` can tell a copy from a re-derivation.

    The second half is the source-level guard the packet asks for: no `round(`
    on a rate, and no `rate * n` anywhere in the module.
    """
    import research_results

    checked = 0
    for horizon in ("swing", "day"):
        view = _view("bot", horizon, snapshot_payload, journal_trades)
        for section in view.sections:
            rows = list(section.rows) + list(section.studies)
            for band in ("stronger", "weaker", "not_enough"):
                rows += list(getattr(section.bands, band))
            for row in rows:
                cell = row.cell
                if cell is None:
                    continue
                for field in (
                    "statistic",
                    "uncertainty_low",
                    "n_eligible",
                    "n_graded",
                    "n_pending",
                    "n_excluded",
                    "n_symbols",
                    "n_sessions",
                    "n_floor",
                    "top_symbol_share",
                    "top_session_share",
                ):
                    assert field in row.values, (
                        f"row {cell.name!r} does not carry the raw {field}"
                    )
                    assert row.values[field] is getattr(cell, field), (
                        f"{cell.name!r}.{field} was RE-DERIVED, not copied: "
                        f"{row.values[field]!r} is not the cell's {getattr(cell, field)!r}"
                    )
                    checked += 1
    assert checked, "no bot row was checked - the view produced nothing"

    # The rehydrated cells are the fixture's cells, field for field.
    by_name = {
        (cell.kind, cell.name): cell
        for cell in _swing_trade_r_cells() + _swing_favorable_cells() + _daytrade_cells()
    }
    seen = 0
    for horizon in ("swing", "day"):
        for section in _view("bot", horizon, snapshot_payload, journal_trades).sections:
            for row in list(section.rows) + list(section.studies):
                if row.cell is None:
                    continue
                assert row.cell == by_name[(row.cell.kind, row.cell.name)], (
                    f"{row.cell.name!r} came back changed from the snapshot"
                )
                seen += 1
    assert seen

    # ---- the source-level guard -------------------------------------------
    # Prose is not code: comments and docstrings are stripped so a module that
    # DESCRIBES the rule ("never round a rate") is not convicted by it.
    code = _code_lines(Path(research_results.__file__).read_text(encoding="utf-8"))
    assert code, "the module has no executable lines at all"

    offenders = [
        line
        for line in code
        if re.search(r"\bround\s*\(", line)
        and re.search(r"\b(statistic|uncertainty\w*|rate|win_rate|bound|share)\b", line)
    ]
    assert not offenders, (
        "the module rounds a rate - a view may not re-compute a statistic: " + repr(offenders)
    )
    products = [
        line
        for line in code
        if re.search(
            r"\b(statistic|uncertainty_low|win_rate|rate)\b\s*\*|"
            r"\*\s*\b(n_graded|n_eligible|n_wins|n_losses)\b",
            line,
        )
    ]
    assert not products, (
        "`rate * n` appears in the module - a weighted rate never becomes an "
        "integer count: " + repr(products)
    )


# ===========================================================================
# 4
# ===========================================================================


def test_my_trades_split_by_holding_period_with_r_only_where_planned_risk_is_present(
    snapshot_payload, journal_trades
):
    """Item G5.3.4. Four buckets' worth of honesty over four real trades.

    `day` = opened and closed in the same exchange session with REAL times;
    `swing` = across sessions; `unknown timing` = either end is
    `journal_trade_shape.is_date_only` (a broker file is authoritative for money
    and blind to time), shown under BOTH horizons and assigned to neither.
    """
    mine_day = _view("mine", "day", snapshot_payload, journal_trades)
    mine_swing = _view("mine", "swing", snapshot_payload, journal_trades)

    def _ids(view, key):
        section = _section(view, key)
        return sorted(str(tid) for tid in section.stats["trade_ids"])

    assert _ids(mine_day, "day") == ["T-DAY", "T-DAY2"]
    assert _ids(mine_swing, "swing") == ["T-SWING"]

    # The unknown-timing bucket is its own labelled section under BOTH horizons.
    assert _ids(mine_day, "unknown_timing") == ["T-DATEONLY"]
    assert _ids(mine_swing, "unknown_timing") == ["T-DATEONLY"]
    assert "T-DATEONLY" not in _ids(mine_day, "day")
    assert "T-DATEONLY" not in _ids(mine_swing, "swing")

    # Neither horizon borrows the other's trades.
    assert [section.key for section in mine_day.sections] == ["day", "unknown_timing"]
    assert [section.key for section in mine_swing.sections] == ["swing", "unknown_timing"]

    day = _section(mine_day, "day")
    stats = day.stats
    assert stats["trades"] == 2
    assert stats["wins"] == 1 and stats["losses"] == 1 and stats["flats"] == 0
    # 120.00 + -40.00, and the fees stated beside it rather than folded away.
    assert stats["net_pnl"] == pytest.approx(80.0)
    assert stats["fees"] == pytest.approx(8.0), (
        "the bucket must report the fees it charged (3+2 and 2+1), not drop them"
    )
    # R only where the trader's own planned_risk is present. One of two.
    assert stats["n_with_r"] == 1, (
        "R must be counted only on trades carrying planned_risk, never on all of them"
    )
    assert stats["n_with_r"] != stats["trades"]

    # Confirmed-tag coverage: one of two, and the provisional one named apart.
    assert stats["confirmed_tagged"] == 1
    assert stats["provisional_tagged"] == 1
    assert stats["total"] == 2

    named = [
        row.values["label"]
        for row in day.rows
        if row.values.get("label") not in (None, "", "untagged")
    ]
    assert named == ["avwap-reclaim"], (
        f"one confirmed tag must yield exactly one named row, got {named}"
    )
    assert "machine-guess" not in named, "a PROVISIONAL tag entered the confirmed table"
    assert not day.sentence.startswith("no confirmed tags yet")

    # The swing bucket has no confirmed tag at all: a sentence, not a leaderboard.
    swing = _section(mine_swing, "swing")
    assert swing.stats["confirmed_tagged"] == 0
    assert swing.sentence == (
        "no confirmed tags yet - nothing here names a setup"
    ), f"zero confirmed tags must read as the sentence, got {swing.sentence!r}"
    assert not [
        row
        for row in swing.rows
        if row.values.get("label") not in (None, "", "untagged")
    ], "a bucket with no confirmed tag still produced a leaderboard row"


# ===========================================================================
# 5
# ===========================================================================


def test_the_recent_window_prints_its_exact_sessions_and_all_history_prints_none(
    snapshot_payload, journal_trades
):
    """Item G5.3.5. "Lately" is ONE number in SESSIONS, walked on the calendar.

    The expected dates are computed here by calling `evidence_stats.lately_window`
    itself, so this can never drift into a calendar-day window; the session count
    is read from `evidence_stats.LATELY_SESSIONS` rather than typed.
    """
    import evidence_stats

    first, last = evidence_stats.lately_window(AS_OF)
    assert first != last

    recent = _view("bot", "swing", snapshot_payload, journal_trades, window="recent")
    assert recent.window == "recent"
    assert recent.window_start == first, f"{recent.window_start!r} != {first!r}"
    assert recent.window_end == last, f"{recent.window_end!r} != {last!r}"
    assert first in recent.window_label and last in recent.window_label, (
        f"the exact session dates are not printed: {recent.window_label!r}"
    )
    assert f"{evidence_stats.LATELY_SESSIONS} sessions" in recent.window_label, (
        f"the window must be stated in SESSIONS: {recent.window_label!r}"
    )

    everything = _view("bot", "swing", snapshot_payload, journal_trades, window="all")
    assert everything.window == "all"
    assert everything.window_start == "" and everything.window_end == ""
    assert re.search(r"\d{4}-\d{2}-\d{2}", everything.window_label) is None, (
        f"All history must print no dates at all: {everything.window_label!r}"
    )

    custom = _view(
        "bot",
        "swing",
        snapshot_payload,
        journal_trades,
        window=("2026-08-03", "2026-09-04"),
    )
    assert custom.window == "custom"
    assert custom.window_start == "2026-08-03" and custom.window_end == "2026-09-04"
    assert "2026-08-03" in custom.window_label and "2026-09-04" in custom.window_label

    # The freshness line names the reading and where every number came from.
    assert snapshot_payload["snapshot_id"][:8] in recent.freshness_line
    assert snapshot_payload["as_of"] in recent.freshness_line
    assert snapshot_payload["built_at"] in recent.freshness_line
    for source in snapshot_payload["sources"].values():
        assert str(source["mtime"]) in recent.freshness_line, (
            f"the freshness line omits a source's mtime: {recent.freshness_line!r}"
        )

    # For My trades it is the journal's newest closed_at instead.
    mine = _view("mine", "day", snapshot_payload, journal_trades, window="recent")
    assert "2026-09-04T15:55:00" in mine.freshness_line, (
        f"My trades' freshness must name the newest closed_at: {mine.freshness_line!r}"
    )
