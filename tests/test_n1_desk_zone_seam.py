"""N1, added by the builder - what the `desk_zone()` seam itself has to be right about.

`tests/test_n1_sidecar_aware_read.py` pins the seam's CONTRACT: it exists in both
modules, it is called through the module global, and a naive desk bar comes back
aware. It pins the zone with `monkeypatch` on purpose, so that those tests say the
same thing on every machine.

Which leaves two questions nobody has asked yet, and both of them decide whether
the live nightly slot reads the right bars:

1. **Whose zone is it?** A trader who has stated their desk zone in
   `local_settings.json` has stated it, and the sidecar reader must not quietly
   prefer whatever the operating system is set to.
2. **Is it right in January?** This is the trap. On Windows the local zone has no
   IANA key, so `market_session.get_market_local_timezone()` falls back to
   `datetime.now().astimezone().tzinfo` - a FIXED offset frozen at the instant it
   was asked. Attaching July's -07:00 to a January bar is an hour wrong, and an
   hour is twelve M5 bars of the wrong window. The fallback here resolves the
   offset per moment instead, which is what `datetime.astimezone()` does for a
   naive value and the only DST-correct answer available without a new
   dependency.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

JANUARY = datetime(2026, 1, 15, 6, 30)
JULY = datetime(2026, 7, 15, 6, 30)


def test_the_configured_desk_zone_wins_over_the_machine(monkeypatch):
    """A stated zone is an answer; the machine's setting is a guess about it."""
    from market_session import LOCAL_MARKET_TIMEZONE_ENV
    from ui.annotations.pass_bars import desk_zone

    monkeypatch.setenv(LOCAL_MARKET_TIMEZONE_ENV, "America/New_York")
    zone = desk_zone()
    assert getattr(zone, "key", None) == "America/New_York"
    assert JANUARY.replace(tzinfo=zone).utcoffset() == timedelta(hours=-5)
    assert JULY.replace(tzinfo=zone).utcoffset() == timedelta(hours=-4)


def test_the_fallback_zone_resolves_dst_per_moment_not_per_process(monkeypatch):
    """The Windows fallback must not freeze one offset over the whole year.

    Asserted against the platform's own answer for each date rather than against
    a literal, so this holds on a desk in any zone - including one with no DST at
    all, where the two answers are legitimately equal.
    """
    import market_session

    from ui.annotations import pass_bars

    # What Windows actually hands back: a bare fixed offset with no IANA key.
    frozen = datetime.now().astimezone().tzinfo
    assert getattr(frozen, "key", None) is None, "a Windows local zone has no key"
    monkeypatch.setattr(
        market_session,
        "get_market_local_timezone",
        lambda *args, **kwargs: (frozen, "Frozen Standard Time"),
    )

    zone = pass_bars.desk_zone()
    assert zone is pass_bars._PLATFORM_LOCAL, (
        "a keyless fixed offset carries no DST rules and must not be attached"
    )
    for moment in (JANUARY, JULY):
        assert moment.replace(tzinfo=zone).utcoffset() == moment.astimezone().utcoffset(), (
            f"the offset for {moment.isoformat()} is resolved from that moment"
        )
        assert moment.replace(tzinfo=zone).tzname() == moment.astimezone().tzname()

    # And the same instant survives a round trip through it.
    instant = datetime(2026, 1, 15, 14, 30, tzinfo=timezone.utc)
    assert instant.astimezone(zone) == instant, "fromutc keeps the instant"
    assert instant.astimezone(zone).tzinfo is zone


def test_a_freshly_written_sidecar_reads_back_as_the_same_instant(tmp_path, monkeypatch):
    """The writer and the reader agree - which is the whole point of naming a zone.

    Before N1 the writer emitted a naive stamp and the reader guessed. Here the
    desk's own bar cache shape (naive `dt`) goes in, and what comes out of
    `_bar_moment` is the instant the trader was actually looking at.
    """
    from ui.annotations import pass_bars, sidecar_completion as sc

    pacific = ZoneInfo("America/Los_Angeles")
    monkeypatch.setattr(pass_bars, "desk_zone", lambda: pacific)
    monkeypatch.setattr(sc, "desk_zone", lambda: pacific)

    bars = [
        {
            "dt": datetime(2026, 1, 15, 6, 30) + timedelta(minutes=5 * step),
            "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1, "volume": 1000,
        }
        for step in range(4)
    ]
    log = tmp_path / "trader_annotations.jsonl"
    fields = pass_bars.write_pass_bars("evt-n1", bars, symbol="SHW", annotations_path=log)
    stored = pass_bars.read_pass_bars(
        {"m5_bars_ref": fields["m5_bars_ref"]}, annotations_path=log
    )

    assert fields["m5_first_bar"] == "2026-01-15T06:30:00-08:00", (
        "January is -08:00 on a Pacific desk, not the -07:00 of the day it was asked"
    )
    read_back = [sc._bar_moment(bar) for bar in stored["bars"]]
    assert read_back[0] == datetime(2026, 1, 15, 6, 30, tzinfo=pacific)
    assert read_back[-1] == datetime(2026, 1, 15, 6, 45, tzinfo=pacific)
    assert all(moment.tzinfo is not None for moment in read_back)

    # And the close derived from them is the exchange's, not the desk's.
    close = sc._session_close(read_back[-1])
    assert close.astimezone(ZoneInfo("America/New_York")).hour == 16
    assert close.astimezone(pacific).hour == 13


def test_a_naive_bound_handed_to_the_lake_is_attached_not_passed_on(monkeypatch):
    """`_lake_bars` is the last gate before Arrow, so it closes the hole itself.

    `complete_sidecar` hands aware bounds. A future caller that does not is
    asking for a read, and answering with the three-night mystery instead would
    be the same defect wearing a different caller.
    """
    from research_warehouse.store import ResearchStore
    from ui.annotations import sidecar_completion as sc

    pacific = ZoneInfo("America/Los_Angeles")
    monkeypatch.setattr(sc, "desk_zone", lambda: pacific)
    seen: dict[str, object] = {}

    class _Recording:
        def read_rows(self, dataset, **kwargs):
            seen["dataset"] = dataset
            seen["range"] = kwargs.get("interval_start_range")
            seen["symbols"] = kwargs.get("symbols")
            return []

    monkeypatch.setattr(
        ResearchStore, "open", classmethod(lambda cls, *a, **k: _Recording())
    )
    rows, reason = sc._lake_bars(
        "SHW", datetime(2026, 9, 1, 8, 0), datetime(2026, 9, 1, 13, 0)
    )

    assert (rows, reason) == ([], "")
    assert seen["dataset"] == "bar_m5"
    assert seen["symbols"] == ["SHW"]
    start, end = seen["range"]
    assert start.tzinfo is not None and end.tzinfo is not None
    assert start == datetime(2026, 9, 1, 15, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 1, 20, 0, tzinfo=timezone.utc)
