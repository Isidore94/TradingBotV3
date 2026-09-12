"""Packet WS-SX - the setups table's ★ and ✕ reflect the day's decisions.

Trader, 2026-09-10 (WISHLIST item 8): the star is filled for anything in Focus
**and** anything liked today; the X is painted BRIGHT RED when the trader has
vetoed, disliked, passed or skipped that symbol today. A symbol both liked and
vetoed shows BOTH marks - the two columns are independent facts. Presentation
only: nothing is hidden, re-ordered, muted or written.

WHAT THESE TESTS PIN (the seam names the builder has to match)
-------------------------------------------------------------
`pick_feedback.decisions_today(...)`, keyword-compatible with its sibling
`reviewed_symbols_today` so the two can share one cached, mtime-keyed read::

    decisions_today(
        market_date=None,
        pick_feedback_path=PICK_FEEDBACK_FILE,
        review_events_path=ALERT_REVIEW_EVENTS_FILE,
        annotations_path=TRADER_ANNOTATIONS_FILE,
    ) -> DayDecisions

    DayDecisions.liked      # {SYMBOL: [(kind, ts), ...]}
    DayDecisions.rejected   # {SYMBOL: [(kind, ts), ...]}
    DayDecisions.for_symbol(symbol)  # the per-symbol view, .liked / .rejected,
                                     # the same (kind, ts) pairs, never None

Kinds, fixed here because the tooltip text is built from them:

* liked: `quick` (a `like_claim` annotation with `like_mode: "quick"`),
  `claimed` (a `like_claim` annotation with `like_mode: "claimed"` **or with no
  `like_mode` key at all** - an absent mode reads `claimed`), `like`
  (`pick_feedback.jsonl` `verdict: "like"`);
* rejected: `veto`, `dislike`, `not_today`, `pass`, `m5_click_away`,
  `remove_today` - exactly the packet's list.
* `unfavorite` is in NEITHER map. It is never a rejection (CLAUDE.md, P5).

`SetupTableDelegate.set_decision_lookup(fn)` where `fn(symbol)` returns that
per-symbol view.

TOOLTIPS - which surface. The packet lets the builder choose `helpEvent` or the
model's `ToolTipRole`; `_tooltip_text` below reads BOTH (Qt's default
`QStyledItemDelegate.helpEvent` routes `ToolTipRole` through
`QToolTip.showText`, so the spy sees either) and asserts on whichever answers.

PAINT - how it is asserted. The two cells are rendered offscreen into a QImage
and the PIXELS are sampled for the exact theme token, because "bright red" is a
claim about what the trader sees, not about which helper was called.

Nothing here may be weakened by the builder; only added to.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

from PySide6.QtCore import QEvent, QPoint, QRect, Qt  # noqa: E402
from PySide6.QtGui import QColor, QHelpEvent, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QStyle,
    QStyleOptionViewItem,
    QTableView,
    QToolTip,
)

import pick_feedback  # noqa: E402
from ui import theme  # noqa: E402
from ui.models.setup import SetupRow  # noqa: E402
from ui.models.setup_table_model import SetupTableModel  # noqa: E402
from ui.widgets.setup_delegate import SetupTableDelegate  # noqa: E402


CELL = QRect(0, 0, 34, 40)  # the ★/✕ columns at the setups row height


# ---------------------------------------------------------------------------
# Qt plumbing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture(autouse=True)
def _cold_caches():
    """The ledger readers memoize on (paths, mtime, size). Start and end cold.

    `review_events.load_review_events` keys its own module cache on the source
    paths too, and every fixture here writes to a fresh `tmp_path`, so the two
    caches can never answer for each other's files.
    """
    pick_feedback.clear_reviewed_today_cache()
    yield
    pick_feedback.clear_reviewed_today_cache()


def _spin(predicate, timeout_ms: int = 2000) -> bool:
    application = QApplication.instance() or QApplication([])
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    application.processEvents()
    return bool(predicate())


# ---------------------------------------------------------------------------
# Real-shaped ledgers
# ---------------------------------------------------------------------------


def _today() -> str:
    return pick_feedback._trade_date_text()


def _yesterday() -> str:
    return (date.fromisoformat(_today()) - timedelta(days=1)).isoformat()


def _at(hh: int, mm: int, *, day: str | None = None) -> datetime:
    return datetime.combine(date.fromisoformat(day or _today()), clock_time(hh, mm, 0))


def _pick_row(symbol: str, verdict: str, *, hh: int, mm: int, day: str | None = None, **over):
    """One `pick_feedback.jsonl` row in the shape `record_pick_feedback` writes.

    Every key is PRESENT; the ones the trader left blank are EMPTY STRINGS, not
    absent - that is what an old row on disk actually looks like.
    """
    row = {
        "ts": _at(hh, mm, day=day).isoformat(timespec="seconds"),
        "trade_date": day or _today(),
        "symbol": symbol,
        "side": "LONG",
        "verdict": verdict,
        "category": "",
        "origin": "",
        "reason": "",
        "context": "",
    }
    row.update(over)
    return row


def _write_pick_feedback(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _event_row(symbol: str, action: str, *, hh: int, mm: int, day: str | None = None, detail=None):
    """One `alert_review_events.jsonl` row in `record_review_event`'s shape."""
    from review_events import REVIEW_EVENTS_SCHEMA

    row = {
        "schema": REVIEW_EVENTS_SCHEMA,
        "review_record_id": f"{symbol}{action}{hh:02d}{mm:02d}",
        "ts": _at(hh, mm, day=day).isoformat(timespec="microseconds"),
        "trade_date": day or _today(),
        "installation_id": "ws-sx-test",
        "machine": "ws-sx",
        "pid": 4242,
        "action": action,
        "symbol": symbol,
        "side": "",  # present and empty, as a board-sourced row really is
    }
    if detail is not None:
        row["detail"] = detail
    return row


def _write_review_events(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def _annotate(path: Path, event_type: str, symbol: str, *, hh: int, mm: int, day=None, **fields):
    from ui.annotations.store import record_annotation

    row = record_annotation(
        event_type,
        path=path,
        symbol=symbol,
        session_date=day or _today(),
        created_at=_at(hh, mm, day=day),
        **fields,
    )
    assert row is not None, f"the {event_type} fixture did not write"
    return row


def _drop_key(path: Path, symbol: str, key: str) -> None:
    """Rewrite one fixture row without `key` - a row written before the field
    existed has NO key at all (a pre-P9 `like_claim` has no `like_mode`)."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("symbol") == symbol:
            row.pop(key, None)
        lines.append(json.dumps(row, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _veto_code_and_note():
    """A live veto code, never a literal: the vocabulary is versioned."""
    from ui.annotations.vocabulary import load_veto_vocabulary

    vocab = load_veto_vocabulary()
    return vocab.reasons[0].code, "ws-sx fixture note"


def _pass_codes():
    from ui.annotations.vocabulary import load_pass_vocabulary

    return [load_pass_vocabulary().reasons[0].code]


def _ledgers(tmp_path):
    return {
        "pick_feedback_path": tmp_path / "pick_feedback.jsonl",
        "review_events_path": tmp_path / "alert_review_events.jsonl",
        "annotations_path": tmp_path / "trader_annotations.jsonl",
    }


def _kinds(mapping, symbol):
    return sorted(str(kind) for kind, _ts in mapping.get(symbol, ()))


def _stamp(mapping, symbol, kind):
    for entry_kind, ts in mapping.get(symbol, ()):
        if str(entry_kind) == kind:
            return str(ts)
    raise AssertionError(f"{symbol} carries no {kind!r} entry: {mapping.get(symbol)!r}")


# ---------------------------------------------------------------------------
# Item 1 - decisions_today
# ---------------------------------------------------------------------------


def test_a_quick_like_and_a_claimed_like_are_both_liked_today(tmp_path):
    """Three ways to like a name today, three kinds, one `liked` map.

    The middle row is the one that breaks a naive reader: a `like_claim`
    written before P9 has NO `like_mode` key, and absence reads `claimed`.
    """
    paths = _ledgers(tmp_path)
    annotations = paths["annotations_path"]
    _annotate(annotations, "like_claim", "QUIKCK", hh=9, mm=41, like_mode="quick")
    _annotate(
        annotations,
        "like_claim",
        "OLDROW",
        hh=10,
        mm=5,
        like_mode="claimed",
        claimed_setup_id="avwap_reclaim",
    )
    _drop_key(annotations, "OLDROW", "like_mode")
    _write_pick_feedback(
        paths["pick_feedback_path"],
        [_pick_row("STARRD", "like", hh=11, mm=17, origin="setups")],
    )
    _write_review_events(paths["review_events_path"], [])

    decisions = pick_feedback.decisions_today(market_date=_today(), **paths)

    assert set(decisions.liked) == {"QUIKCK", "OLDROW", "STARRD"}
    assert _kinds(decisions.liked, "QUIKCK") == ["quick"]
    assert _kinds(decisions.liked, "OLDROW") == ["claimed"], (
        "a like row with no `like_mode` key is a CLAIMED like, not an unknown one"
    )
    assert _kinds(decisions.liked, "STARRD") == ["like"]
    assert decisions.rejected == {} or set(decisions.rejected) == set()
    assert "09:41" in _stamp(decisions.liked, "QUIKCK", "quick"), (
        "the tooltip has to be able to say WHEN, so the pair carries the row's own time"
    )


def test_an_unfavorite_is_neither_liked_nor_rejected(tmp_path):
    """`unfavorite` is never a rejection (CLAUDE.md, P5) and is not a like.

    `not_today` beside it IS a rejection, so a reader that lumps the two
    verdicts together fails here rather than painting a red X on a name the
    trader merely took out of Focus.
    """
    paths = _ledgers(tmp_path)
    _write_pick_feedback(
        paths["pick_feedback_path"],
        [
            _pick_row("UNFAVD", "unfavorite", hh=9, mm=50),
            _pick_row("NOTTDY", "not_today", hh=9, mm=51, category="m5"),
        ],
    )
    _write_review_events(
        paths["review_events_path"],
        [_event_row("UNSTAR", "favorite", hh=9, mm=52, detail={"on": False, "origin": "setups"})],
    )
    paths["annotations_path"].write_text("", encoding="utf-8")

    decisions = pick_feedback.decisions_today(market_date=_today(), **paths)

    assert "UNFAVD" not in decisions.rejected
    assert "UNFAVD" not in decisions.liked
    assert "UNSTAR" not in decisions.rejected, (
        "taking a name out of Focus is not a rejection of it"
    )
    assert _kinds(decisions.rejected, "NOTTDY") == ["not_today"]


def test_the_whole_reject_family_lands_in_rejected_with_its_kind(tmp_path):
    """Every reject-family verdict the packet names, each keeping its own kind.

    The M5 click-away is the one with no verb of its own: it is an `action:
    "skip"` review event whose `detail.reason` is `clicked_away_from_m5_alert`
    (`alert_center_panel.py:2399`). A click away IS a pass (trader 2026-09-01).
    """
    paths = _ledgers(tmp_path)
    code, note = _veto_code_and_note()
    _annotate(paths["annotations_path"], "veto", "VETOED", hh=10, mm=12, reason_code=code, note=note)
    _annotate(paths["annotations_path"], "veto", "UNCODD", hh=10, mm=13)  # uncoded "not today"
    _annotate(
        paths["annotations_path"],
        "pass",
        "PASSED",
        hh=11,
        mm=3,
        reason_codes=_pass_codes(),
        note="one issue",
    )
    _write_pick_feedback(
        paths["pick_feedback_path"],
        [
            _pick_row("DISLKD", "dislike", hh=9, mm=30, reason="too extended"),
            _pick_row("NOTTDY", "not_today", hh=9, mm=31),
        ],
    )
    _write_review_events(
        paths["review_events_path"],
        [
            _event_row("RMVTDY", "remove_today", hh=12, mm=1),
            _event_row(
                "CLKAWY",
                "skip",
                hh=13,
                mm=44,
                detail={"reason": "clicked_away_from_m5_alert"},
            ),
        ],
    )

    decisions = pick_feedback.decisions_today(market_date=_today(), **paths)

    assert _kinds(decisions.rejected, "VETOED") == ["veto"]
    assert _kinds(decisions.rejected, "UNCODD") == ["veto"]
    assert _kinds(decisions.rejected, "PASSED") == ["pass"]
    assert _kinds(decisions.rejected, "DISLKD") == ["dislike"]
    assert _kinds(decisions.rejected, "NOTTDY") == ["not_today"]
    assert _kinds(decisions.rejected, "RMVTDY") == ["remove_today"]
    assert _kinds(decisions.rejected, "CLKAWY") == ["m5_click_away"], (
        "a click away from an M5 alert IS a pass and it is the only `skip` that is"
    )
    assert "11:03" in _stamp(decisions.rejected, "PASSED", "pass")
    assert set(decisions.liked) == set()


def test_yesterdays_decisions_are_not_todays(tmp_path):
    """The marks reset on the day roll: only rows on the target date count."""
    paths = _ledgers(tmp_path)
    code, note = _veto_code_and_note()
    _annotate(
        paths["annotations_path"],
        "veto",
        "YSTRDY",
        hh=10,
        mm=12,
        day=_yesterday(),
        reason_code=code,
        note=note,
    )
    _annotate(paths["annotations_path"], "like_claim", "TODAYL", hh=9, mm=41, like_mode="quick")
    _write_pick_feedback(
        paths["pick_feedback_path"],
        [_pick_row("YSTRDD", "dislike", hh=14, mm=0, day=_yesterday())],
    )
    _write_review_events(
        paths["review_events_path"],
        [_event_row("YSTRDR", "remove_today", hh=15, mm=0, day=_yesterday())],
    )

    decisions = pick_feedback.decisions_today(market_date=_today(), **paths)

    assert set(decisions.liked) == {"TODAYL"}
    assert set(decisions.rejected) == set()

    yesterdays = pick_feedback.decisions_today(market_date=_yesterday(), **paths)
    assert set(yesterdays.rejected) == {"YSTRDY", "YSTRDD", "YSTRDR"}
    assert set(yesterdays.liked) == set()


def test_decisions_today_and_reviewed_symbols_today_share_one_read(tmp_path, monkeypatch):
    """One cached, mtime-keyed read answers BOTH questions (packet item 1).

    The setups table repaints its whole viewport through the delegate; a second
    parse of three ledgers per repaint is the 2026-08-31 stall all over again.
    So after `decisions_today` has answered, `reviewed_symbols_today` for the
    same day and the same files parses NOTHING.
    """
    paths = _ledgers(tmp_path)
    code, note = _veto_code_and_note()
    _annotate(paths["annotations_path"], "veto", "VETOED", hh=10, mm=12, reason_code=code, note=note)
    _annotate(paths["annotations_path"], "like_claim", "QUIKCK", hh=9, mm=41, like_mode="quick")
    _write_pick_feedback(
        paths["pick_feedback_path"], [_pick_row("DISLKD", "dislike", hh=9, mm=30)]
    )
    _write_review_events(
        paths["review_events_path"], [_event_row("RMVTDY", "remove_today", hh=12, mm=1)]
    )

    import review_events
    from ui.annotations import store as annotation_store

    parses: list[str] = []

    def _count(name, real):
        def wrapped(*args, **kwargs):
            parses.append(name)
            return real(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        pick_feedback, "load_pick_feedback", _count("pick", pick_feedback.load_pick_feedback)
    )
    monkeypatch.setattr(
        review_events, "load_review_events", _count("events", review_events.load_review_events)
    )
    monkeypatch.setattr(
        annotation_store,
        "load_annotations",
        _count("annotations", annotation_store.load_annotations),
    )

    decisions = pick_feedback.decisions_today(market_date=_today(), **paths)
    assert set(decisions.rejected) == {"VETOED", "DISLKD", "RMVTDY"}
    first_pass = list(parses)
    assert first_pass, "the first call has to actually read the ledgers"

    parses.clear()
    reviewed = pick_feedback.reviewed_symbols_today(market_date=_today(), **paths)
    assert reviewed == {"VETOED", "QUIKCK", "DISLKD", "RMVTDY"}
    assert parses == [], (
        "the second answer came from a second parse of the same unchanged files; "
        f"re-read {parses}"
    )


# ---------------------------------------------------------------------------
# Item 2 - the theme token and the two marks
# ---------------------------------------------------------------------------


def test_the_reject_today_colour_is_a_bright_red_token_in_both_themes():
    """`reject_today` is a real token in BOTH `THEMES` dicts, not a fallback.

    `theme.color()` answers an unknown name with `neutral`, so a missing token
    would paint the X grey and nothing would raise.
    """
    for name in ("dark", "light"):
        assert "reject_today" in theme.THEMES[name], (
            f"`reject_today` is missing from the {name} theme, so theme.color() "
            "falls back to `neutral` and the X paints grey"
        )
        colour = QColor(theme.color("reject_today", name))
        assert colour.isValid()
        assert colour.red() >= 150, f"{name}: a bright red needs red, got {colour.name()}"
        assert colour.red() > 2 * colour.green() and colour.red() > 2 * colour.blue(), (
            f"{name}: {colour.name()} is not red-dominant"
        )
        assert colour.rgb() != QColor(theme.color("neutral", name)).rgb()


def _rows():
    return [
        SetupRow(
            symbol="ALPHAA",
            side="LONG",
            score=71.2,
            bucket="favorite_setup",
            key_level="$100.00 2nd dev",
            last_trade_date="2026-09-11",
            raw={"setup_family": "avwap_reclaim"},
        ),
        SetupRow(
            symbol="BRAVOO",
            side="SHORT",
            score=64.0,
            bucket="near_favorite_zone",
            key_level="$40.00 1st dev",
            last_trade_date="2026-09-11",
            raw={"setup_family": "avwap_breakdown"},
        ),
        SetupRow(
            symbol="CHARLI",
            side="LONG",
            score=58.5,
            bucket="favorite_setup",
            key_level="$12.00 base",
            last_trade_date="2026-09-11",
            raw={"setup_family": "avwap_reclaim"},
        ),
    ]


def _column(key: str) -> int:
    for index, (name, _label) in enumerate(SetupTableModel.COLUMNS):
        if name == key:
            return index
    raise AssertionError(f"{key} is not a setups column")


def _table(app, rows=None):
    model = SetupTableModel(rows if rows is not None else _rows())
    view = QTableView()
    view.setModel(model)
    delegate = SetupTableDelegate(view)
    view.setItemDelegate(delegate)
    view.resize(900, 300)
    return view, model, delegate


def _option(view):
    option = QStyleOptionViewItem()
    option.initFrom(view)
    option.font = view.font()
    option.rect = QRect(CELL)
    option.state = QStyle.StateFlag.State_Enabled
    return option


def _render(delegate, view, model, row: int, key: str) -> QImage:
    image = QImage(CELL.width(), CELL.height(), QImage.Format.Format_ARGB32)
    image.fill(QColor("#000000"))
    painter = QPainter(image)
    try:
        delegate.paint(painter, _option(view), model.index(row, _column(key)))
    finally:
        painter.end()
    return image


def _has_exact(image: QImage, colour: QColor) -> int:
    """How many pixels are EXACTLY this colour (a solid pen, not a dimmed one)."""
    wanted = QColor(colour).rgb()
    return sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if QColor(image.pixel(x, y)).rgb() == wanted
    )


class _Decisions:
    """The per-symbol view the delegate lookup hands back, built from the REAL
    `decisions_today` result so the test cannot invent a shape."""

    def __init__(self, decisions):
        self._decisions = decisions

    def __call__(self, symbol):
        return self._decisions.for_symbol(symbol)


def _decide(tmp_path, *, liked=(), vetoed=()):
    paths = _ledgers(tmp_path)
    paths["annotations_path"].write_text("", encoding="utf-8")
    _write_pick_feedback(paths["pick_feedback_path"], [])
    _write_review_events(paths["review_events_path"], [])
    for symbol in liked:
        _annotate(paths["annotations_path"], "like_claim", symbol, hh=9, mm=41, like_mode="quick")
    code, note = _veto_code_and_note()
    for symbol in vetoed:
        _annotate(
            paths["annotations_path"], "veto", symbol, hh=10, mm=12, reason_code=code, note=note
        )
    pick_feedback.clear_reviewed_today_cache()
    return pick_feedback.decisions_today(market_date=_today(), **paths)


def test_a_liked_symbol_that_is_not_in_focus_paints_the_filled_star(app, tmp_path):
    """Liked today with no Focus membership paints the SAME filled gold ★ a
    Focus pick does - one boolean, `in Focus OR liked today`."""
    view, model, delegate = _table(app)
    hollow = _render(delegate, view, model, 1, "favorite")

    delegate.set_focus_lookup(lambda symbol: symbol == "BRAVOO")
    focused = _render(delegate, view, model, 1, "favorite")
    assert focused != hollow, "the fixture proves nothing if Focus does not change the star"
    gold = _has_exact(focused, QColor(theme.color("favorite")))
    assert gold > 0, "the Focus star is the gold `favorite` token"

    delegate.set_focus_lookup(lambda symbol: False)
    delegate.set_decision_lookup(_Decisions(_decide(tmp_path, liked=["BRAVOO"])))
    liked = _render(delegate, view, model, 1, "favorite")

    assert liked == focused, (
        "a liked-but-not-Focus row must paint the same filled gold star as a Focus row"
    )
    assert _has_exact(liked, QColor(theme.color("favorite"))) == gold


def test_a_rejected_symbol_paints_the_x_in_the_reject_today_colour(app, tmp_path):
    """The ✕ of a vetoed name is BRIGHT RED - the solid `reject_today` token,
    not today's dimmed `short`."""
    view, model, delegate = _table(app)
    dim = _render(delegate, view, model, 0, "dislike")
    assert _has_exact(dim, QColor(theme.color("reject_today"))) == 0, (
        "an undecided row must not already be wearing the reject colour"
    )

    delegate.set_decision_lookup(_Decisions(_decide(tmp_path, vetoed=["ALPHAA"])))
    red = _render(delegate, view, model, 0, "dislike")

    assert red != dim
    assert _has_exact(red, QColor(theme.color("reject_today"))) > 0, (
        "the X for a rejected symbol is painted with the solid `reject_today` "
        "pen; a dimmed one is what an undecided row gets"
    )
    # And the row next to it, with no decision, is untouched.
    assert _render(delegate, view, model, 1, "dislike") == _render(
        delegate, view, model, 1, "dislike"
    )
    assert _has_exact(_render(delegate, view, model, 1, "dislike"), QColor(theme.color("reject_today"))) == 0


def test_a_symbol_liked_and_vetoed_today_shows_both_marks(app, tmp_path):
    """Two independent facts, two marks (the lead's answer to the packet's open
    question). Neither column may cancel the other."""
    view, model, delegate = _table(app)
    delegate.set_focus_lookup(lambda symbol: False)
    delegate.set_decision_lookup(
        _Decisions(_decide(tmp_path, liked=["CHARLI"], vetoed=["CHARLI"]))
    )

    star = _render(delegate, view, model, 2, "favorite")
    cross = _render(delegate, view, model, 2, "dislike")

    assert _has_exact(star, QColor(theme.color("favorite"))) > 0, "the like still fills the star"
    assert _has_exact(cross, QColor(theme.color("reject_today"))) > 0, "the veto still reddens the X"


def test_a_symbol_with_no_decision_renders_the_row_exactly_as_today(app, tmp_path):
    """The golden. G2b's row styling is the regression to avoid: a no-decision
    row renders byte-identically with and without the decision lookup, and the
    background, the favorite tint, the accent stripe and the hairline separator
    are all still there."""
    view, model, delegate = _table(app)
    delegate.set_focus_lookup(lambda symbol: False)
    before = {key: _render(delegate, view, model, 1, key) for key in ("favorite", "dislike")}

    delegate.set_decision_lookup(_Decisions(_decide(tmp_path, liked=["ALPHAA"], vetoed=["CHARLI"])))
    after = {key: _render(delegate, view, model, 1, key) for key in ("favorite", "dislike")}

    assert after["favorite"] == before["favorite"], "BRAVOO decided nothing today"
    assert after["dislike"] == before["dislike"]

    # The row furniture, sampled where no glyph can reach.
    panel = QColor(theme.color("bg_panel")).rgb()
    elevated = QColor(theme.color("bg_elevated")).rgb()
    favorite_row = _render(delegate, view, model, 0, "favorite")  # row 0, favorite_setup
    plain_row = _render(delegate, view, model, 1, "favorite")  # row 1, near_favorite_zone

    assert QColor(plain_row.pixel(CELL.width() - 1, 1)).rgb() == elevated, (
        "row 1 is the alternating elevated base"
    )
    tinted = QColor(favorite_row.pixel(CELL.width() - 1, 1)).rgb()
    assert tinted != panel, "the favorite tint is gone from the favorite-bucket row"
    stripe = QColor(favorite_row.pixel(3, CELL.height() // 2))
    assert stripe.rgb() != tinted, "the favorite accent stripe is gone from column 0"
    separator = QColor(favorite_row.pixel(CELL.width() - 1, CELL.height() - 1)).rgb()
    assert separator != tinted, "the hairline row separator is gone"


# ---------------------------------------------------------------------------
# Item 3 - tooltips
# ---------------------------------------------------------------------------


def _tooltip_text(delegate, view, model, row: int, key: str) -> str:
    """Whichever surface answers: `helpEvent` (spied at `QToolTip.showText`) or
    the model's `ToolTipRole`. Both are legal per the packet."""
    index = model.index(row, _column(key))
    shown: list[str] = []
    original = QToolTip.showText

    def spy(*args, **kwargs):
        for arg in args:
            if isinstance(arg, str):
                shown.append(arg)
                break
        return None

    QToolTip.showText = staticmethod(spy)  # type: ignore[assignment]
    try:
        event = QHelpEvent(QEvent.Type.ToolTip, QPoint(10, 10), view.mapToGlobal(QPoint(10, 10)))
        delegate.helpEvent(event, view, _option(view), index)
    finally:
        QToolTip.showText = original  # type: ignore[assignment]
    role = index.data(Qt.ItemDataRole.ToolTipRole)
    return "\n".join([text for text in shown if text] + ([str(role)] if role else []))


def test_the_tooltips_name_the_decision_and_its_time(app, tmp_path):
    """"In Focus" / "Liked today (quick, 09:41)" / "Vetoed today 10:12" - the
    mark says THAT, the tooltip says WHICH and WHEN."""
    view, model, delegate = _table(app)
    view.show()
    app.processEvents()
    try:
        delegate.set_focus_lookup(lambda symbol: symbol == "ALPHAA")
        delegate.set_decision_lookup(
            _Decisions(_decide(tmp_path, liked=["BRAVOO", "CHARLI"], vetoed=["CHARLI"]))
        )

        focus_tip = _tooltip_text(delegate, view, model, 0, "favorite").lower()
        assert "in focus" in focus_tip, (
            "the Focus star says why it is filled - and it says it in the "
            f"packet's words, `In Focus`: {focus_tip!r}"
        )

        liked_tip = _tooltip_text(delegate, view, model, 1, "favorite").lower()
        assert "liked" in liked_tip, f"a liked star says so: {liked_tip!r}"
        assert "quick" in liked_tip, "a quick like and a claimed like are named apart"
        assert "09:41" in liked_tip, "the tooltip says when"

        both_tip = _tooltip_text(delegate, view, model, 2, "favorite").lower()
        assert "liked" in both_tip

        veto_tip = _tooltip_text(delegate, view, model, 2, "dislike").lower()
        assert "veto" in veto_tip, f"the red X says which decision: {veto_tip!r}"
        assert "10:12" in veto_tip, "and when"
    finally:
        view.deleteLater()


# ---------------------------------------------------------------------------
# Items 4 and 5 - the panel: one repaint per burst, no file read on paint,
# nothing hidden or re-ordered
# ---------------------------------------------------------------------------


def _focus_service(tmp_path):
    from focus_picks import FocusPickStore
    from ui.services.focus_service import FocusService

    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "membership.json",
    )
    return FocusService(store)


def test_a_burst_of_three_capture_events_repaints_once(app, tmp_path, monkeypatch):
    """Three capture verbs inside one event-loop slot = ONE reaction.

    The 2026-08-31 rule: a burst of one signal is ONE reaction, coalesced at the
    LISTENER through `ui.timer_utils.SignalCoalescer`. The setups viewport is a
    full pass through `SetupTableDelegate` for every visible cell - the single
    hottest stack in that day's stall log.
    """
    from ui.panels import master_avwap_panel
    from ui.timer_utils import SignalCoalescer

    built: list = []

    class Spy(SignalCoalescer):
        def __init__(self, target, *args, **kwargs):
            self.requests = 0
            self.fires = 0
            super().__init__(self._run, *args, **kwargs)
            self._real_target = target
            built.append(self)

        def _run(self):
            self.fires += 1
            self._real_target()

        def request(self):
            self.requests += 1
            super().request()

    monkeypatch.setattr(master_avwap_panel, "SignalCoalescer", Spy)

    panel = master_avwap_panel.MasterAvwapPanel(
        focus_service=_focus_service(tmp_path),
        review_events_path=tmp_path / "alert_review_events.jsonl",
    )
    try:
        panel.set_rows(_rows())
        app.processEvents()
        for coalescer in built:
            coalescer.flush()
        app.processEvents()
        baseline_fires = sum(coalescer.fires for coalescer in built)

        for row in panel.model.rows():
            panel._record_review_event("favorite", row, {"on": True, "origin": "setups"})
        app.processEvents()

        requested = sum(coalescer.requests for coalescer in built)
        assert requested >= 3, (
            "the three capture verbs never asked for a refresh, so the marks are "
            f"stale until something else repaints ({requested} requests)"
        )
        assert sum(coalescer.fires for coalescer in built) == baseline_fires, (
            "a reaction fired while the burst was still arriving"
        )

        for coalescer in built:
            coalescer.flush()
        _spin(lambda: sum(c.fires for c in built) > baseline_fires, 1000)
        assert sum(coalescer.fires for coalescer in built) == baseline_fires + 1, (
            "three capture events, one repaint"
        )
    finally:
        panel.deleteLater()
        app.processEvents()


@pytest.fixture
def panel_with_decisions(app, tmp_path):
    """A real panel whose decisions come from the real default annotations
    ledger (pytest's temp home folder). The file is restored afterwards."""
    from project_paths import TRADER_ANNOTATIONS_FILE
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    annotations = Path(TRADER_ANNOTATIONS_FILE)
    annotations.parent.mkdir(parents=True, exist_ok=True)
    backup = annotations.read_bytes() if annotations.exists() else None
    panel = None
    try:
        annotations.write_text("", encoding="utf-8")
        _annotate(annotations, "like_claim", "BRAVOO", hh=9, mm=41, like_mode="quick")
        code, note = _veto_code_and_note()
        _annotate(annotations, "veto", "CHARLI", hh=10, mm=12, reason_code=code, note=note)
        pick_feedback.clear_reviewed_today_cache()

        panel = MasterAvwapPanel(
            focus_service=_focus_service(tmp_path),
            review_events_path=tmp_path / "alert_review_events.jsonl",
        )
        panel.set_rows(_rows())
        app.processEvents()
        panel.flush_pending_refresh()
        _spin(lambda: False, 350)  # let a 200 ms coalesced window close on its own
        yield panel
    finally:
        if panel is not None:
            panel.deleteLater()
            app.processEvents()
        if backup is None:
            annotations.unlink(missing_ok=True)
        else:
            annotations.write_bytes(backup)
        pick_feedback.clear_reviewed_today_cache()


def test_paint_reads_no_ledger_file(panel_with_decisions, monkeypatch, app):
    """A repaint answers from the cached read, never from the disk.

    The three ledger loaders are made to raise for the duration of the paint.
    The marks still have to be right, which is what stops this from passing for
    the wrong reason (a delegate that reads nothing because it knows nothing).
    """
    panel = panel_with_decisions
    view, model, delegate = panel.table, panel.model, panel.delegate

    import review_events
    from ui.annotations import store as annotation_store

    def forbidden(*args, **kwargs):
        raise AssertionError("a ledger was parsed inside paint")

    monkeypatch.setattr(pick_feedback, "load_pick_feedback", forbidden)
    monkeypatch.setattr(review_events, "load_review_events", forbidden)
    monkeypatch.setattr(annotation_store, "load_annotations", forbidden)

    star = _render(delegate, view, model, 1, "favorite")
    cross = _render(delegate, view, model, 2, "dislike")

    assert _has_exact(star, QColor(theme.color("favorite"))) > 0, (
        "BRAVOO was liked today, so its star is filled - from the cached read"
    )
    assert _has_exact(cross, QColor(theme.color("reject_today"))) > 0, (
        "CHARLI was vetoed today, so its X is bright red - from the cached read"
    )


def test_a_decision_marks_a_row_and_moves_nothing(panel_with_decisions, app):
    """Presentation only (item 5): the same rows, in the same order, with the
    same filter - one of them simply wearing a red X."""
    panel = panel_with_decisions
    proxy = panel.proxy
    visible = [
        proxy.index(row, _column("symbol")).data(Qt.ItemDataRole.DisplayRole)
        for row in range(proxy.rowCount())
    ]
    assert visible == ["ALPHAA", "BRAVOO", "CHARLI"], (
        f"a decision hid or re-ordered a row: {visible}"
    )
    cross = _render(panel.delegate, panel.table, panel.model, 2, "dislike")
    assert _has_exact(cross, QColor(theme.color("reject_today"))) > 0, (
        "the vetoed row is marked, not moved"
    )
