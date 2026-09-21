"""TJ-12 follow-up - the pack's `report_card` hook is actually FILLED.

The docs pass, 2026-09-20: `day_review_pack.build_pack(..., report_card=None)`
accepts a card and the ONLY caller that ever passed one was
`tests/test_tj12_pack_hook.py`. `DayReviewService.build_pack_for` did not, so
in production the pack's `report_card` section was still TJ-4's empty hook and
the overnight day story could never cite a card line.

**`How fresh` is excluded from the pack** (lead, 2026-09-20). It describes the
MACHINE's night, not the trader's day, and its text moves every time the ledger
gains a row - inside the hashed `body` that would move `inputs_hash` and make
`day_review_narration` re-narrate the same session night after night. The other
five lines go in: they are facts OF THE DAY, and when a D1 horizon matures and
a line really moves, the hash SHOULD move.

The exclusion lives at ONE seam - `day_report_card.pack_card` over
`day_report_card.PACK_LINE_KEYS` - and NOT inside `build_pack`, which still
carries whatever it is handed: `test_tj12_pack_hook.py` pins that a full card
handed to `build_pack` keeps all six lines, and that pin stays true.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj12_support as fx  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
EVENING = datetime(2026, 9, 18, 13, 20, tzinfo=PACIFIC)
SESSION = fx.SESSION


class _Journal:
    def entries_about(self, _session):
        return []

    def daily_story(self, _session):
        return None

    def theses_for(self, _session):
        return []


@pytest.fixture()
def service(monkeypatch):
    """A real `DayReviewService` whose STORE reads are stubbed out.

    Only the pack seam is under test here; the regime, label and internals
    readers each own their own guard and are proven elsewhere.
    """
    import day_review_pack
    from ui.services.day_review_service import DayReviewService

    built = DayReviewService(journal_service=_Journal())
    monkeypatch.setattr(built, "_regime_shifts", lambda *_a, **_k: [])
    monkeypatch.setattr(built, "_d1_label_for", lambda *_a, **_k: "")
    monkeypatch.setattr(built, "_internals_marks", lambda *_a, **_k: [])
    # Nothing under test needs the file, and a test that wrote one would be
    # reaching into a store to prove an arithmetic fact.
    monkeypatch.setattr(day_review_pack, "write_pack", lambda pack, **_k: Path("x"))
    return built


def _ledger(tmp_path: Path, *, extra: int = 0) -> Path:
    """A ledger, optionally with rows the night added AFTER the first build."""
    import ai_jobs.ledger as ledger

    rows = list(fx.ledger_rows())
    for index in range(extra):
        moment = datetime(2026, 9, 19, 3, index, tzinfo=PACIFIC)
        rows.append(
            {
                "schema": ledger.LEDGER_SCHEMA,
                "job": f"late_slot_{index}",
                "status": ledger.STATUS_OK,
                "session_date": SESSION,
                "model": "",
                "started_at": moment.isoformat(timespec="seconds"),
                "finished_at": (moment + timedelta(minutes=1)).isoformat(timespec="seconds"),
                "duration_seconds": 60.0,
                "reason": "",
                "outputs": [],
                "tokens": {},
                "error": "",
            }
        )
    return fx.write_ledger_file(
        tmp_path / f"ai_store{extra}" / "logs" / "ai_job_ledger.jsonl", rows
    )


#: Which scratch roots already hold the session's graded clicks. The read
#: ledger is APPEND-ONLY, so writing it twice inside one test would change
#: `your_reads` as well as the AI ledger - and a hash test whose fixture moved
#: two things at once proves nothing about either.
_PREPARED: set[str] = set()


def _inputs(tmp_path: Path, *, extra: int = 0) -> dict:
    key = str(tmp_path)
    if key not in _PREPARED:
        fx.one_session_of_clicks(tmp_path)
        _PREPARED.add(key)
    return fx.day_inputs(tmp_path, ledger_path=_ledger(tmp_path, extra=extra))


def _payload(tmp_path: Path, *, extra: int = 0, card=None) -> dict:
    """A Day Review payload shaped exactly as `read_day` hands one back."""
    import day_report_card
    from ui.services.day_review_service import empty_payload

    built = card or day_report_card.build(_inputs(tmp_path, extra=extra))
    payload = empty_payload(SESSION)
    payload["report_card"] = {
        "session": built.session,
        "lines": [dict(line) for line in built.lines],
    }
    return payload


def _keys(pack) -> list[str]:
    return [line["key"] for line in (pack.get("report_card") or {}).get("lines", ())]


# ---------------------------------------------------------------------------
# the wiring
# ---------------------------------------------------------------------------
def test_the_pack_the_service_writes_carries_the_card(service, tmp_path):
    import day_report_card

    pack = service.build_pack_for(SESSION, payload=_payload(tmp_path), now=EVENING)

    assert pack is not None
    assert _keys(pack) == list(day_report_card.PACK_LINE_KEYS)
    assert pack["report_card"]["session"] == SESSION


def test_every_card_line_in_the_pack_is_citable(service, tmp_path):
    """A narration may quote a card line, so each one needs its own id."""
    import day_review_pack

    pack = service.build_pack_for(SESSION, payload=_payload(tmp_path), now=EVENING)

    ids = [line["source_id"] for line in pack["report_card"]["lines"]]
    assert all(ids) and len(set(ids)) == len(ids)
    allowed = day_review_pack.allowed_source_ids(pack)
    for source_id in ids:
        assert source_id in allowed, source_id


def test_how_fresh_is_the_one_line_the_pack_does_not_carry(service, tmp_path):
    """It describes the MACHINE's night, not the trader's day."""
    import day_report_card

    pack = service.build_pack_for(SESSION, payload=_payload(tmp_path), now=EVENING)

    assert "how_fresh" not in _keys(pack)
    assert tuple(day_report_card.PACK_LINE_KEYS) == tuple(
        key for key in day_report_card.LINE_KEYS if key != "how_fresh"
    )
    body = str(pack["report_card"])
    assert "overnight slot" not in body, body[:200]


def test_a_night_that_gained_ledger_rows_does_not_move_the_hash(service, tmp_path):
    """The reason `How fresh` is excluded, as arithmetic.

    Same session, same day, a ledger that grew between the two builds: the
    story the night already wrote is still a story about THIS day, and a hash
    that moved would buy a model call to re-narrate it.
    """
    first = _payload(tmp_path, extra=0)
    later = _payload(tmp_path, extra=3)

    fresh_first = next(
        line for line in first["report_card"]["lines"] if line["key"] == "how_fresh"
    )
    fresh_later = next(
        line for line in later["report_card"]["lines"] if line["key"] == "how_fresh"
    )
    assert fresh_first["text"] != fresh_later["text"], "the fixture must really move"

    before = service.build_pack_for(SESSION, payload=first, now=EVENING)
    after = service.build_pack_for(SESSION, payload=later, now=EVENING + timedelta(hours=9))

    assert before["inputs_hash"] == after["inputs_hash"]


def test_a_card_line_that_really_moves_still_moves_the_hash(service, tmp_path):
    """A matured D1 horizon SHOULD earn a new narration."""
    payload = _payload(tmp_path)
    before = service.build_pack_for(SESSION, payload=payload, now=EVENING)

    moved = _payload(tmp_path)
    for line in moved["report_card"]["lines"]:
        if line["key"] == "did_well":
            line["text"] = line["text"] + " And one more real run."
    after = service.build_pack_for(SESSION, payload=moved, now=EVENING)

    assert before["inputs_hash"] != after["inputs_hash"]


def test_a_line_whose_owner_raised_goes_in_as_what_it_is(service, tmp_path, monkeypatch):
    """Never dropped silently, and never a zero: the night has to be able to
    say "the desk could not read this" rather than narrate around a hole."""
    import day_report_card

    def _boom(*_args, **_kwargs):
        raise ValueError("the congruence store is torn")

    monkeypatch.setattr(day_report_card, "congruence_line", _boom)
    pack = service.build_pack_for(SESSION, payload=_payload(tmp_path), now=EVENING)

    assert _keys(pack) == list(day_report_card.PACK_LINE_KEYS)
    broken = next(
        line for line in pack["report_card"]["lines"] if line["key"] == "congruence"
    )
    assert broken["measured_ok"] is False
    assert "could not be read" in broken["text"].lower(), broken["text"]
    assert broken["source_id"]


def test_a_day_with_no_card_still_writes_its_pack(service, tmp_path):
    """The card's own build raised, so the payload's hook is empty. The pack is
    still the night's evidence for everything else."""
    from ui.services.day_review_service import empty_payload

    payload = empty_payload(SESSION)
    pack = service.build_pack_for(SESSION, payload=payload, now=EVENING)

    assert pack is not None
    assert pack["report_card"] == {}


def test_pack_card_is_the_one_seam_and_build_pack_still_carries_what_it_is_given(tmp_path):
    """`build_pack` is NOT where the exclusion lives.

    `tests/test_tj12_pack_hook.py` pins that a full card handed to `build_pack`
    keeps all six lines; the filter is `pack_card`'s, upstream, so both truths
    hold at once.
    """
    import day_report_card
    import day_review_pack

    fx.one_session_of_clicks(tmp_path)
    card = day_report_card.build(fx.day_inputs(tmp_path))

    whole = day_review_pack.build_pack(SESSION, report_card=card, now=EVENING)
    assert [line["key"] for line in whole["report_card"]["lines"]] == list(
        day_report_card.LINE_KEYS
    )

    trimmed = day_report_card.pack_card(card)
    assert [line["key"] for line in trimmed.lines] == list(day_report_card.PACK_LINE_KEYS)
    assert trimmed.session == card.session
    assert day_report_card.pack_card(None) is None
    assert day_report_card.pack_card({}) is None
