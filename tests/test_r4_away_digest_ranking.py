"""R4 A11 - the digest ranks by the tracker's record, not by the bucket.

Decision 0016 answer 8, in the trader's words: *"the best pick is often in the
near bucket, not the favourite bucket, so the cream is not being sent."*

V1 was supposed to build this and did not: `_swing_bucket_priority` still ranked
high-conviction, then favorite, then everything else, with insertion order inside
each - and the near cap was applied in that order, so a near pick could be
dropped by POSITION alone while a worse favorite was printed above it.

What changed is the ORDER and the LABEL. The bucket is still printed on every
row, the cap is still three, the header still leads, and the section still holds
the same number of lines for the same picks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import autopilot_core as core  # noqa: E402

#: A tracker record where the NEAR family is the one that has been working.
RECORDS = {
    # 33 of 40. Lower bound ~0.68.
    "avwape_retest": {"wins": 33, "losses": 7},
    # 56 of 90 - the trader's own example. Raw 62%, lower bound ~0.52.
    "first_dev_hold": {"wins": 56, "losses": 34},
    # Three of three. Raw 100%, lower bound ~0.44 - BELOW the 90-sample 62%,
    # which is the whole reason the sort reads the bound and not the rate.
    "thin_thing": {"wins": 3, "losses": 0},
}


def _payload(picks, **extra):
    payload = {
        "generated_at": "2026-09-02 10:00:00",
        "enabled": True,
        "auto_mode": "AWAY",
        "ib_status": "connected",
        "regime": "trend_up",
        "longs": [],
        "shorts": [],
        "swing_picks": picks,
        "swing_data_current": True,
        "swing_data_line": "Swing data: current session 2026-09-02 (scan)",
        "alerts": [],
    }
    payload.update(extra)
    return payload


def _pick(symbol, *, bucket, family, expected_r, side="LONG"):
    return {
        "symbol": symbol,
        "side": side,
        "bucket": bucket,
        "family": family,
        "expected_r": expected_r,
    }


def _swing_section(text: str) -> list[str]:
    body = text.split("== BEST SWING TRADES ==", 1)[1]
    lines = []
    for line in body.splitlines():
        if line.startswith("=="):
            break
        if line.strip():
            lines.append(line)
    return lines


def test_a_near_pick_with_the_better_record_outranks_a_favorite():
    """The exact case answer 8 describes, as a number."""
    text = core.render_away_report(
        _payload(
            [
                _pick("FAVE", bucket="Favorite", family="first_dev_hold", expected_r=0.9),
                _pick("NEAR", bucket="Near Favorite Zone", family="avwape_retest", expected_r=0.4),
            ],
            swing_family_records=RECORDS,
        )
    )

    assert text.index("NEAR (LONG)") < text.index("FAVE (LONG)")
    # The bucket is still PRINTED - what changed is that it is not ranked on.
    assert "NEAR (LONG) | Near Favorite Zone" in text
    assert "FAVE (LONG) | Favorite" in text


def test_the_ranking_is_the_lower_bound_and_never_the_raw_rate():
    """100% on three is not a better record than 62% on ninety."""
    text = core.render_away_report(
        _payload(
            [
                _pick("THIN", bucket="Favorite", family="thin_thing", expected_r=0.1),
                _pick("SOLID", bucket="Favorite", family="first_dev_hold", expected_r=0.1),
            ],
            swing_family_records=RECORDS,
        )
    )

    assert text.index("SOLID (LONG)") < text.index("THIN (LONG)")


def test_an_ungraded_family_sorts_below_every_graded_one():
    """"Not measured" is not "measured badly", so it is not ranked at zero."""
    text = core.render_away_report(
        _payload(
            [
                _pick("UNKNOWN", bucket="High Conviction", family="never_graded", expected_r=5.0),
                _pick("WEAK", bucket="Near Favorite Zone", family="first_dev_hold", expected_r=0.1),
            ],
            swing_family_records=RECORDS,
        )
    )

    assert text.index("WEAK (LONG)") < text.index("UNKNOWN (LONG)")


def test_expected_r_breaks_every_tie_including_the_ungraded_group():
    text = core.render_away_report(
        _payload(
            [
                _pick("LOW", bucket="Favorite", family="never_graded", expected_r=0.2),
                _pick("HIGH", bucket="Favorite", family="never_graded", expected_r=1.8),
            ],
            swing_family_records=RECORDS,
        )
    )

    assert text.index("HIGH (LONG)") < text.index("LOW (LONG)")


def test_the_near_cap_hides_the_weakest_near_rows_and_never_the_best_one():
    """The cap is applied AFTER the ranking. That is the whole of A11's second half."""
    picks = [
        _pick("NBEST", bucket="Near Favorite Zone", family="avwape_retest", expected_r=0.4),
        *[
            _pick(f"NWEAK{index}", bucket="Near Favorite Zone", family="first_dev_hold", expected_r=0.05 - index * 0.01)
            for index in range(5)
        ],
        _pick("FAVE", bucket="Favorite", family="first_dev_hold", expected_r=0.9),
    ]
    text = core.render_away_report(_payload(picks, swing_family_records=RECORDS))

    assert "NBEST (LONG)" in text, "the best near pick was dropped by the cap"
    assert text.index("NBEST (LONG)") < text.index("FAVE (LONG)")
    shown = [name for name in ("NBEST", *(f"NWEAK{i}" for i in range(5))) if f"{name} (LONG)" in text]
    assert len(shown) == core.AWAY_REPORT_MAX_NEAR_ROWS
    assert "more near-favorite rows hidden" in text
    assert "applied AFTER the win-rate ranking" in text


def test_the_section_keeps_its_shape_the_header_and_the_count():
    """A11 reorders and relabels. It must not add, drop or renumber a row."""
    picks = [
        _pick("A", bucket="Favorite", family="first_dev_hold", expected_r=0.9),
        _pick("B", bucket="Near Favorite Zone", family="avwape_retest", expected_r=0.4),
        _pick("C", bucket="High Conviction", family="thin_thing", expected_r=1.2),
    ]
    ranked = _swing_section(core.render_away_report(_payload(picks, swing_family_records=RECORDS)))
    unranked = _swing_section(core.render_away_report(_payload(picks)))

    assert len(ranked) == len(unranked)
    assert ranked[0].startswith("1. ") and ranked[1].startswith("2. ")
    # B (near, 0.68) then A (favorite, 0.52) then C (high conviction, 0.44 on
    # three) - every bucket order the old rule would have produced, inverted by
    # the record, which is exactly what answer 8 asked for.
    assert [line.split(". ", 1)[1].split(" ")[0] for line in ranked[:3]] == ["B", "A", "C"]
    # The TV paste follows the same order and holds the same names.
    paste = next(line for line in ranked if line.startswith("TV paste: "))
    assert paste.split(": ", 1)[1].split(",") == ["B", "A", "C"]


def test_an_unreadable_tracker_file_degrades_to_expected_r_and_never_raises():
    assert core.swing_family_records(Path("does-not-exist.csv")) == {}

    text = core.render_away_report(
        _payload(
            [
                _pick("LOW", bucket="High Conviction", family="first_dev_hold", expected_r=0.2),
                _pick("HIGH", bucket="Near Favorite Zone", family="avwape_retest", expected_r=1.5),
            ]
        )
    )
    assert text.index("HIGH (LONG)") < text.index("LOW (LONG)")


def test_the_records_are_counted_from_the_trackers_own_win_column(tmp_path):
    """The tracker decides what a win IS; nothing here re-derives one.

    R4 fix round 1: every row carries `horizon_sessions` now, and one row at a
    horizon that is NOT the declared one rides along - the fixture that could
    not see the pooling defect was the one with a single row per family.
    """
    path = tmp_path / "tier_outcomes.csv"
    horizon = core.SWING_DIGEST_HORIZON_SESSIONS
    path.write_text(
        "scan_date,setup_family,horizon_sessions,win,side_return_pct\n"
        f"2026-09-01,AVWAPE Retest,{horizon},True,1.2\n"
        f"2026-09-01,avwape_retest,{horizon},False,-0.4\n"
        f"2026-09-01,avwape_retest,{horizon},,0.0\n"
        "2026-09-01,avwape_retest,1,True,2.0\n"
        "2026-09-01,avwape_retest,10,True,2.0\n"
        f"2026-09-01,other,{horizon},True,0.3\n",
        encoding="utf-8",
    )

    records = core.swing_family_records(path, window=("2026-08-01", "2026-09-30"))

    assert records["avwape_retest"] == {"wins": 1, "losses": 1}, (
        "an unreadable verdict is UNMEASURED and belongs in neither count, and "
        "another horizon's look at the same pick is not a second decision"
    )
    assert records["other"] == {"wins": 1, "losses": 0}


def test_a_row_outside_the_lately_window_is_not_counted(tmp_path):
    """"Lately" is one number and it is counted in trading sessions."""
    path = tmp_path / "tier_outcomes.csv"
    horizon = core.SWING_DIGEST_HORIZON_SESSIONS
    path.write_text(
        "scan_date,setup_family,horizon_sessions,win\n"
        f"2026-01-05,avwape_retest,{horizon},True\n"
        f"2026-09-01,avwape_retest,{horizon},True\n",
        encoding="utf-8",
    )

    records = core.swing_family_records(path, window=("2026-08-01", "2026-09-30"))
    assert records["avwape_retest"] == {"wins": 1, "losses": 0}


def test_no_new_ntfy_sender_rides_in_with_the_ranking():
    """AWAY stays the only routine pusher; A11 changed an ORDER, not a channel."""
    source = (ROOT / "scripts" / "autopilot_core.py").read_text(encoding="utf-8")
    block = source.split("def swing_family_records(", 1)[1].split("\ndef swing_push_due", 1)[0]
    for forbidden in ("ntfy", "requests.post", "urlopen"):
        assert forbidden not in block, forbidden
