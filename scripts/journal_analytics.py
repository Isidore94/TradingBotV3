from __future__ import annotations

import csv
import json
import logging
import math
import re
from collections import defaultdict
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from project_paths import (
    AVWAP_SIGNALS_FILE,
    INTRADAY_BOUNCES_FILE,
    MASTER_AVWAP_FOCUS_FILE,
    MASTER_AVWAP_SETUP_TRACKER_FILE,
)


DEFAULT_SWING_LOOKBACK_CALENDAR_DAYS = 16


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value or "").strip()
    if not text:
        return None
    for candidate in (text[:10], text):
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).date()
        except ValueError:
            continue
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for fmt in ("%Y%m%d  %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_side(value: Any) -> str:
    text = str(value or "").strip().upper()
    from journal_identity import BUY_SIDE_WORDS, SELL_SIDE_WORDS

    if text == "LONG" or text in BUY_SIDE_WORDS:
        return "LONG"
    if text in SELL_SIDE_WORDS:
        return "SHORT"
    return text


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


#: Projected context rows, keyed by source path, holding the file stamp they
#: were built from. Bounded to one entry per source file by construction (four
#: of them), and each entry is the SMALL projection, never the parsed blob.
#:
#: Why it exists: `master_avwap_setup_tracker.json` measured 1.08 GB on
#: 2026-08-31 and `json.loads` of it runs behind the Corrections dialog's OK
#: button. Two retags in a row - accept a correction, add an execution - parsed
#: it twice for byte-identical input.
_CONTEXT_ROW_CACHE: dict[str, tuple[tuple[int, int], list[dict[str, Any]]]] = {}


def _file_stamp(path: Path) -> tuple[int, int] | None:
    """(mtime_ns, size), or None when the file cannot be stamped.

    None means "do not cache this" - an unreadable or missing source is not a
    fact worth remembering, and a later appearance must be picked up.
    """
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _cached_context_rows(path: Path, builder) -> list[dict[str, Any]]:
    """`builder()`'s projected rows, reused while the file has not changed."""
    key = str(Path(path))
    stamp = _file_stamp(path)
    if stamp is not None:
        cached = _CONTEXT_ROW_CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
    rows = builder()
    if stamp is not None:
        _CONTEXT_ROW_CACHE[key] = (stamp, rows)
    return rows


def clear_context_row_cache() -> None:
    """Forget every cached projection. For tests and for a forced re-read."""
    _CONTEXT_ROW_CACHE.clear()


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _priority_tag(row: dict[str, Any]) -> str:
    family = str(row.get("setup_family") or row.get("family") or row.get("setup") or "").strip()
    bucket = str(row.get("priority_bucket") or row.get("bucket") or "").strip()
    zone = str(row.get("favorite_zone") or "").strip()
    parts = [part for part in (family, bucket, zone) if part]
    return " | ".join(parts) if parts else "bot_context"


def _date_distance_score(trade_date: date, context_date: date, lookback_days: int) -> float | None:
    delta_days = (trade_date - context_date).days
    if delta_days < 0 or delta_days > lookback_days:
        return None
    if delta_days == 0:
        return 0.28
    return max(0.04, 0.22 * (1.0 - (delta_days / max(1, lookback_days))))


#: ``auto_tag_candidates.source`` prefix for P6's EXACT-ID lane. The stored
#: value is ``trader_capture:<kind>`` - veto, like_claim, pass, or a take-class
#: review action - and it ranks ABOVE every fuzzy source because it is the
#: trader's own statement about that name on that day.
#:
#: Defined HERE and re-exported by `journal_store`, not the other way round:
#: `journal_store` already imports from this module, so the dependency runs one
#: way only.
TRADER_CAPTURE_SOURCE = "trader_capture"

#: ``auto_tag_candidates.source`` prefix for WS-10E's Market Journal lane. The
#: stored value is ``trader_note:market_journal``, and it sits BETWEEN the
#: capture lane and the fuzzy scanner lane: prose the trader typed about a name
#: while the trade was open is weaker than a structured claim carrying an event
#: id, and stronger than a scanner row that merely fell near the same day.
TRADER_NOTE_SOURCE = "trader_note"

#: The ``match_basis`` a note-lane candidate carries: ``note:<entry_id>``. The
#: same shape the packet found already living in ``context_row_id``, kept as one
#: constant so the reader and the writer cannot disagree about the prefix.
NOTE_MATCH_BASIS_PREFIX = "note:"

#: What a note-lane candidate is worth. Below the capture lane's 0.90/0.95 -
#: a written sentence is not a structured claim - and above the scanner lane,
#: whose observed ceiling is 0.80 (P6a's own histogram: tracker + same day +
#: same side is 0.72). Both numbers clear ``journal_bulk_tag``'s 0.70, so a
#: setup the trader NAMED reaches the provisional writer under the same
#: threshold every other lane is measured against.
#:
#: The ORDER, though, is by LANE and never by these numbers - see
#: ``suggest_for_trade`` and ``JournalStore.list_auto_tag_candidates``.
NOTE_LANE_CONFIDENCE = 0.88
NOTE_LANE_CONFIDENCE_NO_SIDE = 0.84

#: How far before the open a note still counts: ONE trading session. A thesis
#: typed the afternoon before the fill is about the trade; the same words a week
#: earlier are about the ticker.
NOTE_WINDOW_MARGIN_SESSIONS = 1

#: Words that make a note's own side explicit. A note that states the OPPOSITE
#: side of the trade never matches it (the packet's rule: never a match on
#: ticker alone), and a note that states neither is side-silent and may match
#: either - silence is not a contradiction.
_LONG_WORDS = ("long", "longs", "longed", "bought", "buying", "reclaim", "reclaimed")
_SHORT_WORDS = ("short", "shorts", "shorted", "shorting", "sold")

_LONG_PATTERN = re.compile(r"\b(" + "|".join(_LONG_WORDS) + r")\b", re.IGNORECASE)
_SHORT_PATTERN = re.compile(r"\b(" + "|".join(_SHORT_WORDS) + r")\b", re.IGNORECASE)

#: Compiled ``(token_count, slug, pattern)`` triples, longest phrase first.
#: Built once from :data:`setup_docs.SETUP_DOCS` - the encyclopedia the desk
#: already keeps - so the vocabulary this lane recognises cannot drift from the
#: one every other surface names a setup by.
_SETUP_CLAIM_PATTERNS: list[tuple[int, str, re.Pattern[str]]] | None = None


def _lane_rank(source: Any) -> int:
    """Which lane a candidate belongs to. Lower leads.

    ONE ordering, used by ``suggest_for_trade`` and mirrored by
    ``JournalStore.list_auto_tag_candidates``' SQL. The shape lane is ranked
    last by the store rather than here because a shape tag is appended after
    this function has already ordered the setup lanes.
    """
    text = str(source or "")
    if text.startswith(f"{TRADER_CAPTURE_SOURCE}:"):
        return 0
    if text.startswith(f"{TRADER_NOTE_SOURCE}:"):
        return 1
    return 2


def slugify_setup(value: Any) -> str:
    """One spelling of a setup name: lowercase, non-alphanumerics collapsed."""
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _setup_claim_patterns() -> list[tuple[int, str, re.Pattern[str]]]:
    """Every phrase that NAMES a setup, as a whole-token regular expression.

    Two spellings per family - the registry key (``avwap_breakout``) and the
    encyclopedia's own label (``AVWAP Breakout``) - joined by a bounded run of
    non-alphanumerics so ``AVWAP Breakout``, ``avwap-breakout`` and
    ``avwap_breakout`` are one phrase and ``avwap ... breakout`` a page apart is
    not.

    A single-token phrase is DROPPED. ``general`` is a family key and an
    ordinary English word, and a lane that matched it would tag a trade because
    the trader wrote "general weakness". Every real family name is two tokens or
    more, so the rule costs nothing and closes the whole class.

    A vocabulary that cannot be read yields an empty list: this lane then
    matches nothing, which is the direction that invents no tags.
    """
    global _SETUP_CLAIM_PATTERNS
    if _SETUP_CLAIM_PATTERNS is not None:
        return _SETUP_CLAIM_PATTERNS
    try:
        from setup_docs import SETUP_DOCS

        families = dict(SETUP_DOCS)
    except Exception:  # noqa: BLE001 - a vocabulary source is never fatal
        logging.debug("Setup vocabulary unavailable to the auto-tagger.", exc_info=True)
        families = {}
    patterns: list[tuple[int, str, re.Pattern[str]]] = []
    seen: set[tuple[str, str]] = set()
    for key, entry in families.items():
        slug = slugify_setup(key)
        if not slug:
            continue
        label = str((entry or {}).get("label") or "")
        # A parenthetical is a qualifier the trader never types - "(Favorite)",
        # "(study)" - and leaving it in would build a phrase nothing matches.
        label = re.sub(r"\([^)]*\)", " ", label)
        for phrase in (str(key), label):
            tokens = [part for part in re.split(r"[^a-z0-9]+", phrase.lower()) if part]
            if len(tokens) < 2:
                continue
            expression = r"\b" + r"[^a-z0-9]{1,4}".join(
                re.escape(token) for token in tokens
            ) + r"\b"
            if (slug, expression) in seen:
                continue
            seen.add((slug, expression))
            patterns.append((len(tokens), slug, re.compile(expression, re.IGNORECASE)))
    patterns.sort(key=lambda item: (-item[0], item[1], item[2].pattern))
    _SETUP_CLAIM_PATTERNS = patterns
    return patterns


def clear_setup_claim_patterns() -> None:
    """Forget the compiled vocabulary. For tests and a forced re-read."""
    global _SETUP_CLAIM_PATTERNS
    _SETUP_CLAIM_PATTERNS = None


def setup_claims_in_text(text: Any) -> list[tuple[str, str]]:
    """``(slug, span)`` for every setup this text NAMES, longest phrase first.

    The span is the matched words taken verbatim out of the text, never a
    paraphrase and never the whole sentence: it is what a reader is shown when
    they ask why a tag says what it says.
    """
    body = str(text or "")
    if not body.strip():
        return []
    found: list[tuple[str, str]] = []
    claimed: set[str] = set()
    for _size, slug, pattern in _setup_claim_patterns():
        if slug in claimed:
            continue
        match = pattern.search(body)
        if match is None:
            continue
        claimed.add(slug)
        found.append((slug, match.group(0)))
    return found


def stated_side_in_text(text: Any) -> str:
    """``LONG``, ``SHORT`` or ``""`` - the side the writer's own words claim.

    Both families of words present means the sentence is about both sides
    (``"short covered, went long"``), and that is not a claim about one: it
    answers ``""``, which matches either trade rather than refusing both.
    """
    body = str(text or "")
    says_long = bool(_LONG_PATTERN.search(body))
    says_short = bool(_SHORT_PATTERN.search(body))
    if says_long == says_short:
        return ""
    return "LONG" if says_long else "SHORT"


def _market_moment(value: Any) -> datetime | None:
    """`value` as an aware market-local datetime, or None."""
    moment = _parse_datetime(value)
    if moment is None:
        return None
    try:
        from market_calendar import MARKET_TZ
    except Exception:  # pragma: no cover - zoneinfo is stdlib on 3.12
        from zoneinfo import ZoneInfo

        MARKET_TZ = ZoneInfo("America/New_York")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=MARKET_TZ)
    return moment.astimezone(MARKET_TZ)


def decode_note_lane(value: Any) -> dict[str, Any]:
    """The stored note-lane verdict, as a mapping. ``{}`` when there is none."""
    if isinstance(value, dict):
        return dict(value)
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except ValueError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def format_note_lane_line(value: Any) -> str:
    """The one sentence every surface prints for the note lane (WS-10E item 3).

    Three shapes and no fourth: the claim it found and where it read it, the
    refusal to reach when the window held notes that named nothing, and the
    honest blank when the trade's own clock cannot establish a window at all.
    """
    data = decode_note_lane(value)
    if not data:
        return ""
    verdict = str(data.get("verdict") or "")
    if verdict == "unmeasured":
        reason = str(data.get("reason") or "").strip() or "no window"
        return f"note lane: unmeasured ({reason})"
    if verdict == "claim":
        return (
            f"note lane: {data.get('tag')} from note {data.get('entry_id')} "
            f"\"{data.get('span')}\""
        )
    count = int(data.get("candidates") or 0)
    return f"note lane: no explicit claim in {count} candidate note(s)"


def note_lane_tag(value: Any) -> str:
    """The setup this trade's note lane claimed, or ``""``.

    What Weekend Prep's Tag Week compares a provisional tag against, so the
    trader can see which proposals came out of their own words.
    """
    data = decode_note_lane(value)
    if str(data.get("verdict") or "") != "claim":
        return ""
    return str(data.get("tag") or "").strip()


class AutoTagger:
    """Suggest journal setup tags from existing bot outputs without importing scanner code."""

    def __init__(
        self,
        *,
        setup_tracker_path: Path = MASTER_AVWAP_SETUP_TRACKER_FILE,
        focus_path: Path = MASTER_AVWAP_FOCUS_FILE,
        avwap_signals_path: Path = AVWAP_SIGNALS_FILE,
        intraday_bounces_path: Path = INTRADAY_BOUNCES_FILE,
        lookback_calendar_days: int = DEFAULT_SWING_LOOKBACK_CALENDAR_DAYS,
        evidence: Any = None,
    ) -> None:
        self.setup_tracker_path = Path(setup_tracker_path)
        self.focus_path = Path(focus_path)
        self.avwap_signals_path = Path(avwap_signals_path)
        self.intraday_bounces_path = Path(intraday_bounces_path)
        self.lookback_calendar_days = int(lookback_calendar_days)
        self._context_rows: list[dict[str, Any]] | None = None
        self._capture_rows: list[dict[str, Any]] | None = None
        self._note_rows: list[dict[str, Any]] | None = None
        # `journal_setup_evidence.EvidenceIndex`; loaded from the live logs on first use.
        self._evidence = evidence

    def load_evidence(self):
        """The pre-entry evidence index (alerts, armed watches, claims, Focus). Never raises."""
        if self._evidence is None:
            try:
                from journal_setup_evidence import EvidenceIndex

                self._evidence = EvidenceIndex.load()
            except Exception:  # noqa: BLE001 - a suggestion source is never fatal
                logging.debug("Evidence lane unavailable to the auto-tagger.", exc_info=True)
                from journal_setup_evidence import EvidenceIndex

                self._evidence = EvidenceIndex()
        return self._evidence

    def load_capture_rows(self) -> list[dict[str, Any]]:
        """The trader's OWN statements about a name, with their event ids.

        A separate list from `load_context_rows` because it is a different kind
        of evidence. Those are scanner rows a trade fell near; these are things
        the trader typed about that symbol on that day - a veto, a like+claim, a
        pass, or a chart they took action on. Matched by exact event id rather
        than by a symbol landing inside a 16-day window, which is why they rank
        above every fuzzy source.

        Read-only over two append-only stores. Any failure yields NOTHING
        rather than raising: the auto-tagger runs behind an OK button, and a
        suggestion source that cannot be read must cost its own suggestions and
        never the pane.
        """
        if self._capture_rows is not None:
            return self._capture_rows
        rows: list[dict[str, Any]] = []
        rows.extend(self._load_annotation_capture_rows())
        rows.extend(self._load_review_capture_rows())
        self._capture_rows = rows
        return rows

    def _load_annotation_capture_rows(self) -> list[dict[str, Any]]:
        """Vetoes, like+claims and passes from `trader_annotations.jsonl`.

        The tag each contributes is what the trader actually said:

        * a **like_claim** contributes its `claimed_setup_id` - they named the
          setup, so that IS the tag;
        * a **veto** contributes ``vetoed:<code>`` and a **pass**
          ``passed:<code>``, prefixed so a rejection can never be mistaken for
          an endorsement in a Tags column.

        The reason codes are carried verbatim; nothing here interprets or
        pools them.
        """
        try:
            from project_paths import TRADER_ANNOTATIONS_FILE
            from ui.annotations.store import (
                EVENT_LIKE_CLAIM,
                EVENT_PASS,
                EVENT_VETO,
                load_annotations,
            )

            annotations = load_annotations(
                Path(TRADER_ANNOTATIONS_FILE),
                event_types=(EVENT_VETO, EVENT_LIKE_CLAIM, EVENT_PASS),
            )
        except Exception:  # noqa: BLE001 - a suggestion source is never fatal
            logging.debug("Trader annotations unavailable to the auto-tagger.", exc_info=True)
            return []

        rows: list[dict[str, Any]] = []
        for annotation in annotations:
            symbol = _normalize_symbol(annotation.get("symbol"))
            session = _parse_date(annotation.get("session_date"))
            if not symbol or session is None:
                continue
            kind = str(annotation.get("event_type") or "")
            link_only = False
            if kind == "like_claim":
                claimed = str(annotation.get("claimed_setup_id") or "").strip()
                # A QUICK like (P9) says "something about this was good" and
                # names no setup, so it contributes a LINK - a pointer with an
                # event id and NO tag text - exactly as a chart housekeeping
                # action does (R2). Reading "liked" as a setup name would put a
                # word in the Tags column that means nothing about the setup and
                # would outrank the scanner match beneath it, which is the bug
                # R2 spent a packet removing.
                tag = claimed or ""
                link_only = not claimed
            elif kind == "veto":
                code = str(annotation.get("reason_code") or "").strip()
                tag = f"{VETO_TAG_WORD}:{code}" if code else VETO_TAG_WORD
            elif kind == "pass":
                codes = [
                    str(code or "").strip()
                    for code in (annotation.get("reason_codes") or [])
                    if str(code or "").strip()
                ]
                # ALL of them, in VOCABULARY order (R2). A pass is
                # multi-select and `codes[0]` threw the rest away, so a pass for
                # "extended from VWAP AND thin liquidity" reached the tagger as
                # the first reason alone - which is a different statement from
                # the one the trader made. The annotation writes its codes in
                # vocabulary order already (never click order), so preserving
                # the list preserves that too.
                tag = f"{PASS_TAG_WORD}:{','.join(codes)}" if codes else PASS_TAG_WORD
            else:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "date": session,
                    "side": _normalize_side(annotation.get("side")),
                    "kind": kind,
                    "tag": tag,
                    "link_only": link_only,
                    "event_id": str(annotation.get("event_id") or ""),
                    "detail": str(annotation.get("note") or ""),
                }
            )
        return rows

    def _load_review_capture_rows(self) -> list[dict[str, Any]]:
        """TAKE-class review events - the charts the trader acted on.

        The take set is `review_learning`'s, read rather than restated: it is
        the one place that decides what counts as the trader saying yes to a
        chart, and a second copy here would drift from it.
        """
        try:
            from review_events import load_review_events
            from review_learning import TAKE_ACTIONS, TOGGLE_TAKE_ACTIONS

            events = load_review_events()
        except Exception:  # noqa: BLE001
            logging.debug("Review events unavailable to the auto-tagger.", exc_info=True)
            return []

        rows: list[dict[str, Any]] = []
        for event in events:
            action = str(event.get("action") or "")
            if action in TOGGLE_TAKE_ACTIONS:
                detail = event.get("detail")
                if not (isinstance(detail, dict) and detail.get("on")):
                    continue
            elif action not in TAKE_ACTIONS:
                continue
            symbol = _normalize_symbol(event.get("symbol"))
            session = _parse_date(event.get("trade_date"))
            if not symbol or session is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "date": session,
                    "side": _normalize_side(event.get("side")),
                    "kind": f"review:{action}",
                    # A CHART HOUSEKEEPING ACTION IS A LINK, NOT A TAG (R1).
                    #
                    # `add_focus`, `arm_level`, `arm_watch` and the toggles say
                    # the trader did something WITH the chart. They say nothing
                    # about which setup it was, and 676 of 730 live rows carry
                    # no `bounce_types` at all - so this minted `took:add_focus`
                    # and, ranked first as a capture candidate, spent the
                    # four-slot summary on it. Measured on eight live trades:
                    # EYPT and SMPL lost `avwape_to_1stdev` from their Tags
                    # column to a housekeeping click.
                    #
                    # The row is still stored - it carries a `context_row_id`
                    # worth following - but it contributes NO tag text, so it
                    # can never evict a real setup match from the summary. Only
                    # a like_claim, a veto and a pass name a setup.
                    "tag": "",
                    "link_only": True,
                    # The alert's own id when it has one - only 54 of 730 take
                    # rows do - and otherwise the row's natural identity, its
                    # timestamp, PREFIXED so a reader knows which store to open.
                    # An empty pointer would look exactly like a fuzzy
                    # candidate, which is the one thing this lane is not.
                    "event_id": (
                        str(event.get("event_id") or "")
                        or f"review_event:{str(event.get('ts') or '').strip()}"
                    ),
                    "detail": str(event.get("tier") or ""),
                }
            )
        return rows

    # ------------------------------------------------------------ WS-10E ---
    # The Market Journal lane. Everything below reads FOUR things about an
    # entry - its id, its actual write time, the symbols it is about, and its
    # own words - and nothing whatsoever about what any trade did.

    def load_market_note_rows(self) -> list[dict[str, Any]]:
        """The trader's written entries, projected to what this lane may use.

        Cached on the instance like ``load_capture_rows``: ``refresh_auto_tags``
        walks every trade in the journal, and re-reading the ledger 204 times
        for a store that held 43 entries on 2026-09-12 would be a file read per
        trade for one answer.

        A SUPERSEDED entry is dropped. The Market Journal corrects by appending
        an entry that names the one it replaces, so reading both would let a
        sentence the trader has already retracted go on naming a setup.

        Read-only, and any failure yields NOTHING rather than raising - this
        runs behind an OK button, and a source that cannot be read must cost its
        own suggestions and never the pane.
        """
        if self._note_rows is not None:
            return self._note_rows
        try:
            import market_journal
            from evidence_ledger import EvidenceLedger

            ledger = EvidenceLedger(
                stream=market_journal.STREAM,
                schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
            )
            entries = list(ledger.read(event_types=("entry",)).rows)
        except Exception:  # noqa: BLE001 - a suggestion source is never fatal
            logging.debug("Market Journal unavailable to the auto-tagger.", exc_info=True)
            self._note_rows = []
            return self._note_rows

        replaced = {
            str(entry.get("supersedes") or "").strip()
            for entry in entries
            if str(entry.get("supersedes") or "").strip()
        }
        rows: list[dict[str, Any]] = []
        for entry in entries:
            entry_id = str(entry.get("entry_id") or "").strip()
            if not entry_id or entry_id in replaced:
                continue
            # `created_at` is when the entry was actually WRITTEN. Phase 0.31
            # preserves the subject in `session_date` and stores the write day
            # separately; this intraday window still needs the exact write time.
            written_at = _market_moment(entry.get("created_at"))
            body = str(entry.get("text") or "")
            symbols = {
                _normalize_symbol(item)
                for item in (entry.get("symbols") or ())
                if _normalize_symbol(item)
            }
            if written_at is None or not body.strip() or not symbols:
                continue
            rows.append(
                {
                    "entry_id": entry_id,
                    "written_at": written_at,
                    "symbols": symbols,
                    "text": body,
                    "side": stated_side_in_text(body),
                    "claims": setup_claims_in_text(body),
                }
            )
        rows.sort(key=lambda row: (row["written_at"], row["entry_id"]))
        self._note_rows = rows
        return rows

    def note_window_for(self, trade: dict[str, Any]) -> tuple[datetime, datetime] | None:
        """The trade's OWN window, or ``None`` when its clock cannot say.

        Open to close, widened by one trading session BEFORE the open so a
        thesis typed the afternoon before the fill still belongs to the trade.

        ``None`` for a date-only broker fill. The statement importers stamp a
        fill at midnight market-local precisely so ``is_date_only`` can
        recognise it, and midnight is not a time a fill happens at - so the
        trade has no intraday window and uncertainty here emits nothing.
        """
        from journal_trade_shape import is_date_only

        opened = _market_moment(trade.get("opened_at") or trade.get("trade_date"))
        if opened is None:
            return None
        closed = _market_moment(trade.get("closed_at")) or opened
        if is_date_only(opened) or is_date_only(closed):
            return None
        first, last = (opened, closed) if opened <= closed else (closed, opened)
        try:
            from market_calendar import previous_session

            start_day = first.date()
            for _step in range(max(0, NOTE_WINDOW_MARGIN_SESSIONS)):
                start_day = previous_session(start_day)
        except Exception:  # noqa: BLE001 - a calendar refusal falls back NARROWER
            logging.debug("Note window margin unavailable; using the open.", exc_info=True)
            start_day = first.date()
        start = datetime.combine(start_day, time(0, 0), tzinfo=first.tzinfo)
        return (start, last)

    def note_lane_report(self, trade: dict[str, Any]) -> dict[str, Any]:
        """What this lane saw for one trade, and what it concluded.

        Three verdicts and no fourth: ``claim`` (a note inside the window named
        a setup this desk knows), ``no_claim`` (notes were there and named
        none - said out loud, because a silent lane and an empty window look
        identical to a reader), and ``unmeasured`` (the trade's own clock
        cannot establish a window).

        The ``notes`` list travels to the advisory package so a model can cite
        an entry by id; every other reader stores and prints the rest.
        """
        symbol = _normalize_symbol(trade.get("symbol"))
        direction = _normalize_side(trade.get("direction"))
        window = self.note_window_for(trade)
        if not symbol or window is None:
            return {
                "verdict": "unmeasured",
                "reason": "date-only fill" if symbol else "no symbol",
                "candidates": 0,
                "tag": "",
                "entry_id": "",
                "span": "",
                "notes": [],
            }
        start, end = window
        seen: list[dict[str, Any]] = []
        for row in self.load_market_note_rows():
            if symbol not in row["symbols"]:
                continue
            written_at = row["written_at"]
            if not (start <= written_at <= end):
                continue
            side = str(row.get("side") or "")
            if side and direction and side != direction:
                # The words say the other side. A thesis about a short is not
                # evidence about a long, and matching on the ticker alone is
                # exactly what the packet forbids.
                continue
            seen.append(row)
        # ONE pick, not two. The claim this verdict names is the same candidate
        # `journal_bulk_tag` would write - `max` on confidence, first wins on a
        # tie, exactly as the writer does it. A second rule here would let the
        # Journal print one setup while the Tags column carried another, and Tag
        # Week's mark compares the two.
        best: dict[str, Any] = {}
        claims = self.note_lane_candidates(trade)
        if claims:
            top = max(claims, key=lambda item: float(item.get("confidence") or 0.0))
            best = {
                "tag": str(top.get("tag") or ""),
                "entry_id": str(top.get("match_basis") or "")[len(NOTE_MATCH_BASIS_PREFIX):],
                "span": str(top.get("span") or ""),
            }
        payload = {
            "verdict": "claim" if best else "no_claim",
            "reason": "",
            "candidates": len(seen),
            "tag": best.get("tag", ""),
            "entry_id": best.get("entry_id", ""),
            "span": best.get("span", ""),
            "notes": [
                {
                    "note_id": row["entry_id"],
                    "written_at": row["written_at"].isoformat(timespec="minutes"),
                    "text": row["text"],
                    "side_words": row["side"] or "none stated",
                }
                for row in seen
            ],
        }
        return payload

    def note_lane_candidates(self, trade: dict[str, Any]) -> list[dict[str, Any]]:
        """The lane's suggestions for one trade, in the tagger's own shape.

        At most one per note that names a setup: the trader wrote a sentence
        about a name, and the sentence's claim is the candidate. Each carries
        ``match_basis = note:<entry_id>`` and the quoted ``span`` it matched, so
        the record answers "why does this say avwap_breakout?" without
        re-deriving anything.
        """
        symbol = _normalize_symbol(trade.get("symbol"))
        direction = _normalize_side(trade.get("direction"))
        window = self.note_window_for(trade)
        if not symbol or window is None:
            return []
        start, end = window
        found: list[dict[str, Any]] = []
        for row in self.load_market_note_rows():
            if symbol not in row["symbols"]:
                continue
            written_at = row["written_at"]
            if not (start <= written_at <= end):
                continue
            side = str(row.get("side") or "")
            if side and direction and side != direction:
                continue
            if not row["claims"]:
                continue
            slug, span = row["claims"][0]
            basis = f"{NOTE_MATCH_BASIS_PREFIX}{row['entry_id']}"
            found.append(
                {
                    "tag": slug,
                    "confidence": (
                        NOTE_LANE_CONFIDENCE if side else NOTE_LANE_CONFIDENCE_NO_SIDE
                    ),
                    "source": f"{TRADER_NOTE_SOURCE}:market_journal",
                    "context_row_id": basis,
                    "match_basis": basis,
                    "span": span,
                    "link_only": False,
                    "rationale": (
                        f"you wrote this on {written_at.date().isoformat()} "
                        f"({basis}): \"{span}\"; inside this trade's own window"
                    ),
                }
            )
        return found

    def load_context_rows(self) -> list[dict[str, Any]]:
        """The scanner-output rows the tagger matches trades against.

        Cached per SOURCE FILE, not just per tagger: every one of these is a
        pure projection of a file, so two taggers built minutes apart over
        unchanged files must not parse them twice. The tracker file alone
        measured 1.08 GB on 2026-08-31 and this runs behind an OK button.
        """
        if self._context_rows is not None:
            return self._context_rows
        rows: list[dict[str, Any]] = []
        rows.extend(_cached_context_rows(self.setup_tracker_path, self._load_tracker_rows))
        rows.extend(_cached_context_rows(self.focus_path, self._load_focus_rows))
        rows.extend(_cached_context_rows(self.avwap_signals_path, self._load_avwap_signal_rows))
        rows.extend(
            _cached_context_rows(self.intraday_bounces_path, self._load_intraday_bounce_rows)
        )
        self._context_rows = rows
        return rows

    def _load_tracker_rows(self) -> list[dict[str, Any]]:
        payload = _load_json(self.setup_tracker_path)
        if not isinstance(payload, dict):
            return []
        setups = payload.get("setups")
        if not isinstance(setups, dict):
            return []
        rows = []
        for setup in setups.values():
            if not isinstance(setup, dict):
                continue
            rows.append(
                {
                    "source": "setup_tracker",
                    "symbol": _normalize_symbol(setup.get("symbol")),
                    "side": _normalize_side(setup.get("side")),
                    "date": _parse_date(setup.get("scan_date") or setup.get("entry_trade_date")),
                    "setup_family": setup.get("setup_family") or "general",
                    "priority_bucket": setup.get("priority_bucket") or "",
                    "favorite_zone": setup.get("favorite_zone") or "",
                    "priority_score": _coerce_float(setup.get("priority_score")),
                    "retest": setup.get("retest_reference_level") or setup.get("mid_earnings_primary_trigger_level") or "",
                    "compression": bool(setup.get("compression_flag")),
                }
            )
        # The parsed blob is 1.08 GB and the projection above is a few MB.
        # Dropping the references here rather than at the return statement
        # means the tagging that follows never runs alongside both.
        del setups
        del payload
        return rows

    def _load_focus_rows(self) -> list[dict[str, Any]]:
        payload = _load_json(self.focus_path)
        if not isinstance(payload, dict):
            return []
        rows = []
        updated_date = _parse_date(payload.get("updated_at") or payload.get("scan_date") or datetime.now())

        def add_entry(entry: Any, source: str, bucket: str = "") -> None:
            if not isinstance(entry, dict):
                return
            rows.append(
                {
                    "source": source,
                    "symbol": _normalize_symbol(entry.get("symbol")),
                    "side": _normalize_side(entry.get("side")),
                    "date": _parse_date(entry.get("scan_date") or entry.get("last_trade_date")) or updated_date,
                    "setup_family": entry.get("setup_family") or entry.get("family") or "focus",
                    "priority_bucket": entry.get("priority_bucket") or bucket,
                    "favorite_zone": entry.get("favorite_zone") or "",
                    "priority_score": _coerce_float(entry.get("priority_score") or entry.get("score")),
                    "retest": entry.get("retest_reference_level") or "",
                    "compression": bool(entry.get("compression_flag")),
                }
            )

        for entry in payload.get("favorites") or []:
            add_entry(entry, "focus_favorite", "favorite_setup")
        for entry in payload.get("near_favorite_zones") or []:
            add_entry(entry, "focus_near_zone", "near_favorite_zone")
        symbols = payload.get("symbols")
        if isinstance(symbols, dict):
            for entry in symbols.values():
                add_entry(entry, "focus_symbol")
        return rows

    def _load_avwap_signal_rows(self) -> list[dict[str, Any]]:
        rows = []
        for raw in _read_csv_rows(self.avwap_signals_path):
            rows.append(
                {
                    "source": "avwap_signal",
                    "symbol": _normalize_symbol(raw.get("symbol")),
                    "side": _normalize_side(raw.get("side")),
                    "date": _parse_date(raw.get("scan_date") or raw.get("trade_date") or raw.get("last_trade_date")),
                    "setup_family": raw.get("setup_family") or raw.get("family") or "avwap_signal",
                    "priority_bucket": raw.get("priority_bucket") or "",
                    "favorite_zone": raw.get("favorite_zone") or "",
                    "priority_score": _coerce_float(raw.get("priority_score") or raw.get("score")),
                    "retest": raw.get("retest_reference_level") or "",
                    "compression": str(raw.get("compression_flag") or "").lower() in {"1", "true", "yes"},
                }
            )
        return rows

    def _load_intraday_bounce_rows(self) -> list[dict[str, Any]]:
        rows = []
        for raw in _read_csv_rows(self.intraday_bounces_path):
            bounce_time = _parse_datetime(
                raw.get("time") or raw.get("timestamp") or raw.get("bounce_time") or raw.get("trade_date")
            )
            rows.append(
                {
                    "source": "intraday_bounce",
                    "symbol": _normalize_symbol(raw.get("symbol") or raw.get("ticker")),
                    "side": _normalize_side(raw.get("direction") or raw.get("side") or raw.get("watchlist_bias")),
                    "date": bounce_time.date() if bounce_time else _parse_date(raw.get("trade_date")),
                    "setup_family": raw.get("bounce_type") or raw.get("setup_family") or "intraday_bounce",
                    "priority_bucket": "intraday",
                    "favorite_zone": raw.get("level") or raw.get("levels") or "",
                    "priority_score": _coerce_float(raw.get("score")),
                    "retest": raw.get("level") or "",
                    "compression": False,
                }
            )
        return rows

    def suggest_for_trade(
        self,
        trade: dict[str, Any],
        corrections: list[dict[str, Any]] | None = None,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        symbol = _normalize_symbol(trade.get("symbol"))
        direction = _normalize_side(trade.get("direction"))
        trade_date = _parse_date(trade.get("opened_at") or trade.get("trade_date") or trade.get("closed_at"))
        if not symbol or trade_date is None:
            return []

        candidates: dict[str, dict[str, Any]] = {}

        # ---------------------------------------------------------- P6 -----
        # The EXACT-ID lane first: what the trader themselves said about this
        # symbol while the trade was open. Matched on the trade's OWN WINDOW -
        # open date to close date, not a 16-day neighbourhood - because an
        # event id is only worth carrying when the statement and the trade
        # really are about the same episode.
        #
        # A capture candidate carries `context_row_id`, which every surface
        # renders beside its confidence. It is a POINTER for a reader, never a
        # canonical link: plan.md P5.3/P5.4 own the canonical opportunity id.
        opened = trade_date
        closed = _parse_date(trade.get("closed_at")) or opened
        window_start, window_end = (opened, closed) if opened <= closed else (closed, opened)
        for row in self.load_capture_rows():
            if row.get("symbol") != symbol:
                continue
            said_on = row.get("date")
            if not isinstance(said_on, date):
                continue
            if not (window_start <= said_on <= window_end):
                continue
            row_side = row.get("side") or ""
            if row_side and direction and row_side != direction:
                # A long statement about a short trade is a different claim.
                continue
            link_only = bool(row.get("link_only"))
            tag = str(row.get("tag") or "").strip()
            if not tag and not link_only:
                continue
            if link_only:
                # A pointer, under a name that cannot be mistaken for a setup.
                # It is excluded from `auto_tag_summary` by the store, so it
                # occupies no slot in the Tags column.
                tag = f"{LINK_TAG_PREFIX}{row.get('kind')}"
            # A stated judgement inside the trade's own window is the strongest
            # thing this tagger has, and it is still a SUGGESTION: the trader
            # accepts or ignores it, and nothing here writes trade_annotations.
            confidence = 0.95 if row_side and direction else 0.90
            current = candidates.get(tag)
            if current is not None and float(current.get("confidence", 0.0) or 0.0) >= confidence:
                continue
            detail = str(row.get("detail") or "").strip()
            candidates[tag] = {
                "tag": tag,
                "confidence": confidence,
                "source": f"{TRADER_CAPTURE_SOURCE}:{row.get('kind')}",
                "context_row_id": str(row.get("event_id") or ""),
                # Read by `refresh_auto_tags`: a link is stored as a candidate
                # and kept out of the summary (R1).
                "link_only": link_only,
                "rationale": (
                    f"you said this on {said_on.isoformat()} ({row.get('kind')})"
                    + (f": {detail}" if detail else "")
                    + "; inside this trade's own window"
                ),
            }

        # -------------------------------------------------------- WS-10E ---
        # The Market Journal lane, between the capture lane and the scanner's.
        # A tag the trader WROTE never displaces a claim they STRUCTURED, so a
        # slot the capture lane already owns is left alone.
        for note in self.note_lane_candidates(trade):
            tag = str(note.get("tag") or "").strip()
            if not tag:
                continue
            current = candidates.get(tag)
            if current is not None and str(current.get("source") or "").startswith(
                f"{TRADER_CAPTURE_SOURCE}:"
            ):
                continue
            if current is not None and float(current.get("confidence", 0.0) or 0.0) >= float(
                note.get("confidence", 0.0) or 0.0
            ):
                continue
            candidates[tag] = dict(note)

        # The scanner lane matches an option on its UNDERLYING, and on the side
        # the option takes on it (a long put is short the underlying).
        from journal_setup_evidence import underlying_view

        scan_symbol, scan_side = underlying_view(trade)
        scan_side = scan_side or direction
        for row in self.load_context_rows():
            if _normalize_symbol(row.get("symbol")) != scan_symbol:
                continue
            context_date = row.get("date")
            if not isinstance(context_date, date):
                continue
            date_score = _date_distance_score(trade_date, context_date, self.lookback_calendar_days)
            if date_score is None:
                continue

            row_side = _normalize_side(row.get("side"))
            side_score = 0.16 if not row_side or not scan_side or row_side == scan_side else -0.10
            source = str(row.get("source") or "bot_context")
            source_score = {
                "setup_tracker": 0.28,
                "focus_favorite": 0.24,
                "focus_near_zone": 0.20,
                "focus_symbol": 0.12,
                "avwap_signal": 0.18,
                "intraday_bounce": 0.18,
            }.get(source, 0.08)
            score_value = _coerce_float(row.get("priority_score"))
            priority_score = min(0.14, max(0.0, (score_value or 0.0) / 1000.0))
            bucket_bonus = 0.08 if str(row.get("priority_bucket") or "") in {"favorite_setup", "near_favorite_zone"} else 0.0
            confidence = max(0.01, min(0.98, source_score + date_score + side_score + priority_score + bucket_bonus))
            tag = _priority_tag(row)
            current = candidates.get(tag)
            rationale = (
                f"{source}; {symbol}; context {context_date.isoformat()}; "
                f"{row.get('setup_family') or 'setup'}"
            )
            if current is not None and str(current.get("source") or "").startswith(
                (f"{TRADER_CAPTURE_SOURCE}:", f"{TRADER_NOTE_SOURCE}:")
            ):
                # A fuzzy match never displaces the trader's own statement -
                # structured (the capture lane) or written (WS-10E's note
                # lane) - whatever its computed confidence.
                continue
            if current is None or confidence > float(current.get("confidence", 0.0) or 0.0):
                candidates[tag] = {
                    "tag": tag,
                    "confidence": confidence,
                    "source": source,
                    "rationale": rationale,
                    "context_row_id": "",
                }

        # Evidence lane (same rank as the scanner): alerts that fired, watches
        # armed, cards liked and setups claimed on this name BEFORE the entry.
        for item in self.load_evidence().candidates_for(trade):
            tag = str(item.get("tag") or "").strip()
            if not tag:
                continue
            current = candidates.get(tag)
            if current is not None and (
                str(current.get("source") or "").startswith(
                    (f"{TRADER_CAPTURE_SOURCE}:", f"{TRADER_NOTE_SOURCE}:")
                )
                or float(current.get("confidence", 0.0) or 0.0) >= float(item["confidence"])
            ):
                continue
            candidates[tag] = dict(item)

        for correction in corrections or []:
            if _normalize_symbol(correction.get("symbol")) != symbol:
                continue
            tag = str(correction.get("setup_tag") or "").strip()
            if not tag:
                continue
            boost = _coerce_float(correction.get("confidence_boost")) or 0.12
            current = candidates.get(tag)
            if current:
                current["confidence"] = min(0.99, float(current["confidence"]) + boost)
                current["rationale"] = f"{current['rationale']}; manual correction boost"
            else:
                candidates[tag] = {
                    "tag": tag,
                    "confidence": min(0.80, 0.40 + boost),
                    "source": "manual_correction",
                    "rationale": "Historical manual correction for this symbol.",
                    "context_row_id": "",
                }

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                # BY LANE, never by confidence. The capture lane leads: the
                # trader's own structured statement about this name on this day
                # outranks anything inferred about it. WS-10E's note lane is
                # second - prose they typed inside the trade's own window is
                # weaker than a claim carrying an event id and stronger than a
                # scanner row that merely fell near the same date.
                _lane_rank(item.get("source")),
                -float(item.get("confidence", 0.0) or 0.0),
                str(item.get("tag") or ""),
            ),
        )
        return ordered[: max(1, int(limit))]


#: Account tax statuses that cannot hold a stock short (TFSA, RRSP and kin).
REGISTERED_TAX_STATUSES = frozenset({"TAX_FREE", "TAX_DEFERRED"})
#: Account labels that name a registered account when no tax status is stored.
REGISTERED_ACCOUNT_WORDS = ("TFSA", "RRSP", "RRIF", "FHSA", "RESP", "LIRA", "LIF", "RDSP")


def _is_option_row(row: dict[str, Any]) -> bool:
    from journal_identity import normalize_security_type

    security_type = normalize_security_type(row.get("security_type"))
    if security_type in {"OPT", "FOP", "WAR"}:
        return True
    if security_type != "UNKNOWN":
        return False
    from journal_importers import classify_questrade_security_type

    return classify_questrade_security_type({"symbol": row.get("symbol")}) == "OPT"


def is_registered_account(row: dict[str, Any]) -> bool:
    """True for a TFSA/RRSP-style account: stored tax status first, label second."""
    status = str(row.get("account_tax_status") or "").strip().upper()
    if status:
        return status in REGISTERED_TAX_STATUSES
    words = f"{row.get('account_label') or ''} {row.get('account_type') or ''}".upper()
    return any(word in words for word in REGISTERED_ACCOUNT_WORDS)


def is_registered_stock_short(row: dict[str, Any]) -> bool:
    """A stock SHORT in a registered account: impossible, so its buy is missing.

    Writing an option there is allowed, so options are never flagged.
    """
    if str(row.get("direction") or "").upper() != "SHORT":
        return False
    if not is_registered_account(row):
        return False
    return not _is_option_row(row)


def has_invented_entry(row: dict[str, Any]) -> bool:
    """True when the trade's entry was made up rather than imported.

    Either the rebuild stood the closing fill in for a missing opening fill (a
    ``SYNTHETIC_OPEN`` leg), or it is a stock short a registered account cannot
    hold. The trade stays in the journal; it is left out of P&L totals.
    """
    if row.get("entry_invented") or row.get("synthetic_entry"):
        return True
    return is_registered_stock_short(row)


def counts_in_pnl(row: dict[str, Any]) -> bool:
    """A CLOSED trade whose entry is real: the only kind P&L totals may add up."""
    return str(row.get("status") or "").upper() == "CLOSED" and not has_invented_entry(row)


def not_counted_summary(trades: list[dict[str, Any]], pnl_key: str = "net_pnl") -> dict[str, Any]:
    """The CLOSED trades left out of totals for a made-up entry, and one line saying so."""
    left_out = [
        row for row in trades
        if str(row.get("status") or "").upper() == "CLOSED" and has_invented_entry(row)
    ]
    pnl = sum(_coerce_float(row.get(pnl_key)) or 0.0 for row in left_out)
    count = len(left_out)
    line = ""
    if count:
        noun = "trade needs" if count == 1 else "trades need"
        line = f"{count} {noun} missing fills - not counted (${pnl:,.2f})"
    return {"trades": count, "net_pnl": pnl, "line": line}


def calendar_pnl_by_day(trades: list[dict[str, Any]], *, pnl_key: str = "net_pnl") -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for trade in trades:
        if not counts_in_pnl(trade):
            continue
        trade_day = _parse_date(trade.get("closed_at") or trade.get("trade_date") or trade.get("opened_at"))
        if trade_day is None:
            continue
        pnl = _coerce_float(trade.get(pnl_key))
        if pnl is None:
            continue
        totals[trade_day.isoformat()] += pnl
    return dict(totals)


# ---------------------------------------------------------------------------
# Derived per-trade fields. Pure functions of the trade's own stamps, computed
# on read and never stored. Unmeasurable is "unknown" (or None for minutes).
# ---------------------------------------------------------------------------

UNKNOWN_FIELD = "unknown"
OPEN_FIELD = "open"
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
#: `journal_trade_shape.hold_bucket` names that mean the trade was flat the same session.
DAY_HOLD_BUCKETS = frozenset({"scalp", "day_trade"})


def trade_time_of_day(trade: dict[str, Any]) -> str:
    """Session bucket of the first fill (`opening_drive`, `midday`, ...), ET clock.

    Same buckets as the `trade_shape:entry_time` tag; `unknown` for a date-only fill.
    """
    from journal_trade_shape import session_bucket

    return session_bucket(trade.get("opened_at")) or UNKNOWN_FIELD


def trade_weekday(trade: dict[str, Any]) -> str:
    """`Mon`..`Fri` of the first fill in market-local time, or `unknown`."""
    from journal_trade_shape import _coerce_datetime

    moment = _coerce_datetime(trade.get("opened_at"))
    return WEEKDAY_NAMES[moment.weekday()] if moment is not None else UNKNOWN_FIELD


def trade_hold_minutes(trade: dict[str, Any]) -> float | None:
    """Minutes from first fill to close, or None (open, unparseable or date-only)."""
    from journal_trade_shape import _coerce_datetime, is_date_only

    opened = _coerce_datetime(trade.get("opened_at"))
    closed = _coerce_datetime(trade.get("closed_at"))
    if opened is None or closed is None or closed < opened:
        return None
    if is_date_only(opened) or is_date_only(closed):
        return None
    return round((closed - opened).total_seconds() / 60.0, 1)


def trade_hold_bucket(trade: dict[str, Any]) -> str:
    """`scalp` / `day_trade` / `overnight` / `swing` / `position`, `open`, or `unknown`."""
    if str(trade.get("status") or "").upper() not in {"", "CLOSED"}:
        return OPEN_FIELD
    from journal_trade_shape import hold_bucket

    found = hold_bucket(trade.get("opened_at"), trade.get("closed_at"))
    return found[0] if found else UNKNOWN_FIELD


def trade_horizon(trade: dict[str, Any]) -> str:
    """`day` (flat the same session) / `swing` (held overnight or longer), `open`, or `unknown`."""
    bucket = trade_hold_bucket(trade)
    if bucket in (OPEN_FIELD, UNKNOWN_FIELD):
        return bucket
    return "day" if bucket in DAY_HOLD_BUCKETS else "swing"


def derived_trade_fields(trade: dict[str, Any]) -> dict[str, Any]:
    """All derived fields for one trade: time_of_day, weekday, hold_minutes, hold_bucket, horizon."""
    return {
        "time_of_day": trade_time_of_day(trade),
        "weekday": trade_weekday(trade),
        "hold_minutes": trade_hold_minutes(trade),
        "hold_bucket": trade_hold_bucket(trade),
        "horizon": trade_horizon(trade),
    }


#: Group name -> key function, for a UI that groups trades by a derived field.
DERIVED_GROUPS = {
    "time of day": trade_time_of_day,
    "weekday": trade_weekday,
    "hold": trade_hold_bucket,
    "day vs swing": trade_horizon,
}


def derived_group_summary(
    trades: list[dict[str, Any]], group: str, *, pnl_key: str = "net_pnl"
) -> list[dict[str, Any]]:
    """One `_summary_for_rows` row per bucket of a `DERIVED_GROUPS` field, most trades first."""
    key_fn = DERIVED_GROUPS[group]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        buckets[str(key_fn(trade))].append(trade)
    rows = []
    for label, bucket_rows in buckets.items():
        item = _summary_for_rows(bucket_rows, pnl_key)
        item["label"] = label
        rows.append(item)
    rows.sort(key=lambda item: (-int(item.get("closed", 0)), str(item["label"])))
    return rows


def _summary_for_rows(rows: list[dict[str, Any]], pnl_key: str = "net_pnl") -> dict[str, Any]:
    closed = [row for row in rows if counts_in_pnl(row)]
    all_closed = sum(1 for row in rows if str(row.get("status") or "").upper() == "CLOSED")
    pnl_values = [_coerce_float(row.get(pnl_key)) or 0.0 for row in closed]
    wins = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    gross_win = sum(wins)
    gross_loss = sum(losses)
    profit_factor = (gross_win / abs(gross_loss)) if gross_loss < 0 else None
    return {
        "trades": len(rows),
        "closed": len(closed),
        "open": len(rows) - all_closed,
        "not_counted": all_closed - len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "profit_factor": profit_factor,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "net_pnl": sum(pnl_values),
        "avg_win": (gross_win / len(wins)) if wins else None,
        "avg_loss": (gross_loss / len(losses)) if losses else None,
    }


#: Column written by :func:`apply_manual_usd_estimate`. Named "estimated" on
#: purpose: it must never be mistaken for a booked value in a log or a CSV.
USD_ESTIMATE_KEY = "net_pnl_usd_estimated"

#: Column booked by ``JournalStore.book_currency_values`` from the stored BoC
#: observation for each trade's OWN session (2026-08-24). Preferred over the
#: manual estimate wherever every selected row carries it - one is a
#: measurement, the other is one rate applied to a year.
USD_BOOKED_KEY = "net_pnl_usd"


def apply_manual_usd_estimate(
    trades: list[dict[str, Any]], rate: float | None = None
) -> tuple[float, list[dict[str, Any]]] | None:
    """Annotate rows with an estimated USD P&L. Returns (rate, unconverted).

    ``None`` when no manual rate is set, which leaves every existing refusal
    exactly as it was. A USD-native row passes its own value through untouched;
    anything else divides the BOOKED CAD value by the entered rate, so the
    estimate inherits the booked path's honesty about what it could not convert.
    """
    if rate is None:
        from journal_fx import manual_usd_rate

        stored = manual_usd_rate()
        if not stored:
            return None
        rate = float(stored["rate_cad_per_usd"])
    if not rate:
        return None

    unconverted: list[dict[str, Any]] = []
    for row in trades:
        if str(row.get("currency") or "").upper() == "USD":
            row[USD_ESTIMATE_KEY] = row.get("net_pnl")
            continue
        cad = row.get("net_pnl_cad")
        if cad is None:
            row[USD_ESTIMATE_KEY] = None
            unconverted.append(row)
            continue
        row[USD_ESTIMATE_KEY] = float(cad) / rate
    return float(rate), unconverted


def resolve_pnl_key(
    trades: list[dict[str, Any]], currency_mode: str | None = None
) -> tuple[str, str]:
    """Which P&L column may be summed, and what to tell the reader.

    Root cause B8. ``_summary_for_rows`` defaulted to ``net_pnl``, which is the
    trade's **native** currency, and then added a USD win to a CAD loss as if
    they were the same number. For a Canadian trader filing Canadian tax that is
    not a rounding error, it is a wrong total.

    Three honest outcomes, and no fourth:

    * one currency across the whole selection - sum ``net_pnl``, it means
      something;
    * mixed currencies and every trade converted - sum ``net_pnl_cad``;
    * mixed currencies with anything unconverted - **refuse**. The caller gets
      ``("", reason)`` and shows the reason instead of a number, because a total
      that silently omits the unconverted rows is worse than no total.
    """
    closed = [row for row in trades if counts_in_pnl(row)]
    mode = str(currency_mode or "").strip().upper()
    currencies = {str(row.get("currency") or "").upper() for row in closed if row.get("currency")}
    if mode == "CAD":
        unconverted = [row for row in closed if row.get("net_pnl_cad") is None]
        if unconverted:
            missing = sorted({str(row.get("currency") or "?").upper() for row in unconverted})
            return "", (
                f"{len(unconverted)} of {len(closed)} trades have no booked FX rate "
                f"({', '.join(missing)}); CAD totals are not shown"
            )
        return "net_pnl_cad", "converted to CAD at each trade's booked rate"
    if mode == "USD":
        non_usd = [row for row in closed if str(row.get("currency") or "").upper() != "USD"]
        if not non_usd:
            return "net_pnl", ""
        # True conversion first (2026-08-24). Every row carries a USD value
        # booked at import from the BoC observation for its own session, so this
        # is a measurement rather than an approximation - and it is preferred
        # over the manual rate whenever it can answer for the WHOLE selection.
        # Partially booked is not good enough: summing booked rows and estimated
        # rows in one total would produce a number that is neither.
        unbooked = [row for row in closed if row.get(USD_BOOKED_KEY) is None]
        if not unbooked:
            return USD_BOOKED_KEY, (
                "converted to USD at each trade's booked Bank of Canada rate for "
                "its own session"
            )
        # A manually entered display rate is the ONLY way a mixed selection
        # gets a USD total, and it is an estimate, not a booked figure. It
        # converts from the booked CAD value, so a row the booked path could
        # not convert stays unconvertible here too - a manual rate buys an
        # approximation, never a missing observation.
        estimate = apply_manual_usd_estimate(closed)
        if estimate is not None:
            rate, unconverted = estimate
            if unconverted:
                missing = sorted({str(row.get("currency") or "?").upper() for row in unconverted})
                return "", (
                    f"{len(unconverted)} of {len(closed)} trades have no booked FX rate "
                    f"({', '.join(missing)}); USD totals are not shown"
                )
            return USD_ESTIMATE_KEY, (
                f"ESTIMATE - non-USD trades converted at a manually entered "
                f"{rate:.4f} CAD/USD, not each trade's booked rate. Not a tax figure."
            )
        missing = sorted({str(row.get("currency") or "?").upper() for row in unbooked})
        return "", (
            f"{len(unbooked)} of {len(closed)} trades have no booked USD rate for "
            f"their session ({', '.join(missing)}); USD totals are not shown. Enter "
            f"a USD/CAD rate in the Journal header for an estimate."
        )
    # Native mode (and legacy callers with no explicit mode) can add values only
    # when the selection has one currency. Legacy mixed selections retain the
    # tax-grade CAD fallback used by non-UI reports.
    if mode == "NATIVE" and len(currencies) > 1:
        return "", "multiple native currencies selected; Native totals are not shown"
    if len(currencies) <= 1:
        return "net_pnl", ""
    unconverted = [row for row in closed if row.get("net_pnl_cad") is None]
    if unconverted:
        missing = sorted({str(row.get("currency") or "?").upper() for row in unconverted})
        return "", (
            f"{len(unconverted)} of {len(trades)} trades have no booked FX rate "
            f"({', '.join(missing)}); totals across currencies are not shown"
        )
    return "net_pnl_cad", "converted to CAD at each trade's booked rate"


def split_tags(value: Any) -> list[str]:
    """Split one stored tag string into its tags.

    The first separator present wins, in the order ``;`` ``,`` ``|``, rather
    than splitting on all three. That matters because ``_priority_tag`` builds
    a setup tag as ``"family | bucket | zone"`` -- pipes are INSIDE a tag, and
    only a string with no ``;`` or ``,`` at all is treated as pipe-separated.

    Named and exported because the store, the tag list and the rename tool all
    need this exact rule; a second copy anywhere would eventually disagree
    about what one tag is.
    """
    text = str(value or "").strip()
    if not text:
        return []
    for separator in (";", ",", "|"):
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


#: The three tag lanes (P6a), named here rather than imported from
#: ``journal_store`` so this module keeps its one-way dependency: the store
#: imports the analytics helpers, not the other way round.
TAG_STATUS_CONFIRMED = "confirmed"
TAG_STATUS_PROVISIONAL = "provisional"


#: The prefix every link-only candidate's tag carries. A LINK records that the
#: trader did something WITH the chart - added it to Focus, armed a level - and
#: says nothing about which setup it was. It is stored, it renders, it carries a
#: `context_row_id` worth following, and it is NEVER a tag.
LINK_TAG_PREFIX = "link:"

#: The two words a REJECTION's tag starts with, and the `:` that carries its
#: reason code. `_load_annotation_capture_rows` writes `vetoed:<code>` and
#: `passed:<c1>,<c2>` - and the bare word when there is no code - *"prefixed so
#: a rejection can never be mistaken for an endorsement in a Tags column"*.
#:
#: Named here, beside `LINK_TAG_PREFIX`, because TJ-9 gave that sentence a
#: second reader: the Trade Mentor's one-click setup confirm, which offered
#: `vetoed:too_extended_from_base` as a SETUP on a live trade and would have
#: written it `confirmed`, where "My setups" counts it. A second list spelled
#: out over there is the kind of copy that drifts.
VETO_TAG_WORD = "vetoed"
PASS_TAG_WORD = "passed"
REJECTION_TAG_WORDS = (VETO_TAG_WORD, PASS_TAG_WORD)


def is_rejection_tag(tag: Any) -> bool:
    """Is this tag a REJECTION rather than a setup? The ONE predicate.

    True for ``vetoed``, ``vetoed:<code>``, ``passed`` and ``passed:<codes>``.
    A rejection is a statement about why the trader stayed OUT, and no reader
    may ever count one as a setup they were in.
    """
    text = str(tag or "").strip().lower()
    if not text:
        return False
    return text.split(":", 1)[0].strip() in REJECTION_TAG_WORDS


def is_link_candidate(candidate: Any) -> bool:
    """ONE predicate for "this is a pointer, not a tag" (R2).

    R1 kept links out of `auto_tag_summary` and three other seams still let them
    through: the bulk tagger's lane filter, its `max(confidence)` pick, the
    Accept-all button, and `tag_confidence`. Each had its own idea of what a
    link was - or no idea at all - so the rule held in one place and leaked in
    four.

    Both spellings are accepted deliberately. `link_only` is what the tagger
    sets in memory; the PREFIX is what survives a round trip through
    `auto_tag_candidates`, which stores a tag and a source but no flag. A reader
    that only knew the flag would be right until the row came back from the
    database.
    """
    if isinstance(candidate, str):
        return candidate.startswith(LINK_TAG_PREFIX)
    if not hasattr(candidate, "get"):
        return False
    if candidate.get("link_only"):
        return True
    return str(candidate.get("tag") or "").startswith(LINK_TAG_PREFIX)


def _confirmed_setup_tags(row: dict[str, Any]) -> list[str]:
    """The tags on this trade that the TRADER stands behind (P6a).

    A provisional tag lives in the same column, so grouping on ``setup_tags``
    alone would fold 100+ machine guesses into "my setups" - the one group in
    the journal that is supposed to answer what the trader themself said this
    trade was. A row with no annotation at all reports ``confirmed`` and has no
    tags, so it lands in ``untagged`` exactly as before.
    """
    if str(row.get("tag_status") or TAG_STATUS_CONFIRMED) != TAG_STATUS_CONFIRMED:
        return []
    return _tags_for_row(row, "setup_tags")


def _provisional_setup_tags(row: dict[str, Any]) -> list[str]:
    """The tags a machine applied and nobody has reviewed yet (P6a)."""
    if str(row.get("tag_status") or TAG_STATUS_CONFIRMED) != TAG_STATUS_PROVISIONAL:
        return []
    return _tags_for_row(row, "setup_tags")


def _tags_for_row(row: dict[str, Any], field: str = "setup_tags") -> list[str]:
    """Every setup tag on a trade, not just the first one.

    ``_first_setup_tag`` kept only the leading tag, so a trade tagged
    "avwap-reclaim; earnings-gap" counted entirely towards the first and not at
    all towards the second - which quietly understated every setup that tends to
    be named second.
    """
    return split_tags(row.get(field))


def build_analytics_summary(
    trades: list[dict[str, Any]], currency_mode: str | None = None
) -> dict[str, Any]:
    pnl_key, pnl_note = resolve_pnl_key(trades, currency_mode)
    summary = {
        "overall": _summary_for_rows(trades, pnl_key or "net_pnl"),
        "groups": {},
        "pnl_key": pnl_key,
        "pnl_note": pnl_note,
        "currencies": sorted({str(row.get("currency") or "").upper() for row in trades if row.get("currency")}),
        # Native money: a left-out trade has no converted value worth trusting.
        "not_counted": not_counted_summary(trades),
    }
    if not pnl_key:
        # Mixed currencies with unconverted rows: the per-group totals would be
        # as meaningless as the overall one, so say why and stop.
        summary["overall"] = {**summary["overall"], "net_pnl": None, "gross_win": None, "gross_loss": None}
    group_specs = {
        # CONFIRMED only. The provisional lane is its own group below and the
        # two are never blended: a per-setup win rate that mixes what the trader
        # said with what a machine guessed is not a statement about either.
        "my setups": lambda row: _confirmed_setup_tags(row) or ["untagged"],
        # No "untagged" fallback here on purpose: a trade with no provisional
        # tag belongs in no bucket of this group at all, and a catch-all bucket
        # holding every other trade would be the biggest bar on the chart while
        # meaning nothing.
        "provisional setups": _provisional_setup_tags,
        "auto tags": lambda row: _tags_for_row(row, "auto_tag_summary") or ["untagged"],
        "account": lambda row: str(row.get("account_label") or row.get("account_number") or "unknown"),
        "broker": lambda row: str(row.get("broker") or "unknown"),
        "symbol": lambda row: str(row.get("symbol") or "unknown"),
        "direction": lambda row: str(row.get("direction") or "unknown"),
        "mid_term_regime": lambda row: str(row.get("mid_term_regime") or "unset"),
        "short_term_regime": lambda row: str(row.get("short_term_regime") or "unset"),
        "intraday_regime": lambda row: str(row.get("intraday_regime") or "unset"),
    }
    for group_name, key_fn in group_specs.items():
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in trades:
            keys = key_fn(row)
            if not isinstance(keys, list):
                keys = [keys]
            for key in dict.fromkeys(keys):
                buckets[str(key)].append(row)
        rows = []
        for label, bucket_rows in buckets.items():
            # The same column the overall total used. A per-group breakdown that
            # summed native P&L under a CAD headline would disagree with the
            # number above it, which is B8 back again one row down.
            item = _summary_for_rows(bucket_rows, pnl_key or "net_pnl")
            if not pnl_key:
                item = {**item, "net_pnl": None, "gross_win": None, "gross_loss": None}
            item["label"] = label
            rows.append(item)
        rows.sort(
            key=lambda item: (
                -int(item.get("closed", 0)),
                -abs(float(item.get("net_pnl") or 0.0)),
                str(item["label"]),
            )
        )
        summary["groups"][group_name] = rows
    summary["group_notes"] = _empty_dimension_notes(trades, summary["groups"])
    summary["nonexclusive_groups"] = ["my setups", "provisional setups", "auto tags"]
    #: Groups whose buckets are machine-applied and awaiting review. The chart
    #: says so out loud - a bar chart of "provisional setups" beside one of "my
    #: setups" is otherwise two answers to the same question with nothing to
    #: separate them.
    summary["provisional_groups"] = ["provisional setups"]
    # ST5.4: the four never-pooled populations, the coverage line and the
    # refusal to name a best setup, computed over the SAME rows this summary
    # already holds. Additive - every existing key is untouched - and it costs
    # one pass in memory, so the tab that renders it opens no second query.
    summary["personal_evidence"] = personal_evidence_summary(trades)
    return summary


#: The three populations that PARTITION every trade, by STATUS (ST5.4, fixed
#: after the reviewer's reproduction 2026-09-06). What finished, what is half
#: out, and what is still on.
PERSONAL_EVIDENCE_STATUS_POPULATIONS = (
    "complete",
    "partly_closed",
    "open_exposure",
)

#: The blocks a reader iterates: the three status populations plus the
#: CROSS-CUTTING `uncertain` label. `uncertain` is NOT a fourth bucket - its
#: members are already counted in one of the three - so these four do not sum
#: to the whole and are not meant to.
PERSONAL_EVIDENCE_POPULATIONS = PERSONAL_EVIDENCE_STATUS_POPULATIONS + ("uncertain",)


def _population_of(trade: dict[str, Any], exposure=None) -> str:
    """Which STATUS population this trade belongs to. Exactly one, always.

    **Status decides, and uncertainty is a LABEL on top of it.** The first cut
    of this function checked ``exposure.is_uncertain`` FIRST, which made
    "uncertain" a fourth bucket that ate the other three. Reproduced on a copy
    of the live journal 2026-09-06: `uncertain` came out **n=120** holding 84
    CLOSED trades, ALL 7 CLOSED_PARTIAL and 29 of the 32 OPEN ones - so
    `partly_closed` read **n=0** while seven exist, `open_exposure` read n=3
    with a notional of 7,726 against roughly 9% of the real open exposure, and
    one pooled P&L figure summed realized results together with OPEN positions'
    unrealized marks and counted those marks as WINNERS.

    That is the exact defect this whole summary exists to prevent, one level
    down. So: CLOSED is complete, CLOSED_PARTIAL is partly closed, everything
    else is open exposure (an unrecognised status is open, never a result), and
    whether the instrument or the structure can be named is carried beside each
    of them as ``n_uncertain`` plus the cross-cutting ``uncertain`` block.

    ``exposure`` is accepted and ignored so every existing caller still works.
    """
    status = str(trade.get("status") or "").upper()
    if status == "CLOSED":
        return "complete"
    if status == "CLOSED_PARTIAL":
        return "partly_closed"
    return "open_exposure"


def _cad_pnl(trade: dict[str, Any]) -> float | None:
    for key in ("net_pnl_cad", "net_pnl"):
        value = _coerce_float(trade.get(key))
        if value is not None:
            return value
    return None


def _notional(trade: dict[str, Any]) -> float | None:
    quantity = _coerce_float(trade.get("quantity_opened"))
    price = _coerce_float(trade.get("average_entry_price"))
    if quantity is None or price is None:
        return None
    return abs(quantity) * abs(price)


def _bias_cell(rows: list[dict[str, Any]], *, with_pnl: bool) -> dict[str, Any]:
    """One cell: how many, how many won, how much - or nothing, said as nothing.

    ``with_pnl`` is False for open exposure, and there it means BOTH halves are
    absent: ``net_pnl`` is ``None`` and ``winners`` is ``None``. An open
    position's mark is not a result and counting it as a win is the blocker this
    cell was rebuilt for. An EMPTY bucket also reports ``None`` rather than
    ``0.0`` - a blank cell says "nothing here", where a net of 0.00 says
    "measured, and it came to nothing".
    """
    pnls = [_cad_pnl(row) for row in rows]
    measured = [value for value in pnls if value is not None]
    return {
        "n": len(rows),
        "winners": (sum(1 for value in measured if value > 0) if with_pnl else None),
        "net_pnl": (sum(measured) if with_pnl and measured else None),
        "trade_ids": [str(row.get("trade_id") or "") for row in rows],
    }


def personal_evidence_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Four populations, a market-bias split, and what the journal cannot say.

    ST5.4, from the trader's own words: *"Separate complete trades, partly
    closed trades, open exposure, and instrument/strategy uncertainty in
    summaries"* and *"Do not tell me a personal setup is best when there are no
    confirmed tags."*

    * **Never pooled.** The four populations partition the input - every trade
      lands in exactly one, and their counts sum to the whole.
    * **An open position has NO result.** ``open_exposure["net_pnl"]`` is
      ``None``, not zero: an unrealized number read as a result is the defect
      this separation exists to prevent. Its size travels as ``notional``.
    * **The bias split is `journal_exposure`'s**, so a sold put counts as
      ``bullish_or_neutral`` and a bought put as ``bearish``. ``unknown`` is its
      own bucket and is printed, never dropped.
    * **No "best" without confirmed tags at the floor.** ``best_setup`` is
      ``None`` until the trader's own confirmed tags reach
      ``evidence_stats.MIN_REPORTABLE_N``, and the headline says so with the
      provisional count beside it. A provisional tag is a machine's guess and
      has never been an answer.

    Pure and read-only: no store is opened, and nothing here computes a
    ``planned_risk`` from an outcome - that number is the trader's own and
    ``JournalStore.save_risk_fields`` is its only writer.
    """
    from evidence_stats import MIN_REPORTABLE_N
    from journal_exposure import BIAS_UNKNOWN, DIRECTIONAL_BIASES, classify_all
    from swing_headline import wilson_lower_bound

    # A CLOSED trade with a made-up entry is not a result; it is left out here too.
    rows = [
        row for row in trades
        if isinstance(row, dict)
        and not (str(row.get("status") or "").upper() == "CLOSED" and has_invented_entry(row))
    ]
    exposures = classify_all(rows)

    def _exposure_of(row):
        return exposures.get(str(row.get("trade_id") or ""))

    buckets: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PERSONAL_EVIDENCE_STATUS_POPULATIONS
    }
    for row in rows:
        buckets[_population_of(row)].append(row)

    summary: dict[str, Any] = {}
    for name, bucket in buckets.items():
        # OPEN exposure has no result at all: no net, no winners, no USD total.
        # The other two are measured populations.
        with_pnl = name != "open_exposure"
        by_bias: dict[str, list[dict[str, Any]]] = {BIAS_UNKNOWN: []}
        structures: dict[str, int] = {}
        uncertain_rows: list[dict[str, Any]] = []
        for row in bucket:
            exposure = _exposure_of(row)
            bias = exposure.market_bias if exposure is not None else BIAS_UNKNOWN
            by_bias.setdefault(bias, []).append(row)
            key = exposure.structure if exposure is not None else "unknown"
            structures[key] = structures.get(key, 0) + 1
            if exposure is None or exposure.is_uncertain:
                uncertain_rows.append(row)
        cell = _bias_cell(bucket, with_pnl=with_pnl)
        usd = [_coerce_float(row.get("net_pnl_usd")) for row in bucket]
        cell.update(
            {
                # Both currencies, and USD only when EVERY row in the bucket
                # booked one - a partial sum under a currency heading is the
                # defect the journal's own `resolve_pnl_key` already refuses.
                "net_pnl_usd": (
                    sum(value for value in usd if value is not None)
                    if with_pnl and bucket and all(value is not None for value in usd)
                    else None
                ),
                "by_market_bias": {
                    bias: _bias_cell(bias_rows, with_pnl=with_pnl)
                    for bias, bias_rows in by_bias.items()
                },
                "by_structure": structures,
                # The cross-cutting label, counted where it applies. A reader of
                # `complete` sees at once how much of that population rests on
                # an instrument or a structure the store cannot name.
                "n_uncertain": len(uncertain_rows),
                "uncertain_trade_ids": [
                    str(row.get("trade_id") or "") for row in uncertain_rows
                ],
            }
        )
        if name == "open_exposure":
            notionals = [_notional(row) for row in bucket]
            cell["notional"] = (
                sum(value for value in notionals if value is not None)
                if any(value is not None for value in notionals)
                else None
            )
            cell["notional_unmeasured"] = sum(1 for value in notionals if value is None)
        summary[name] = cell

    # ------------------------------------------------------------------
    # `uncertain` - a LABEL across the three, never a bucket beside them
    # ------------------------------------------------------------------
    uncertain_all = [
        row
        for row in rows
        if (_exposure_of(row) is None or _exposure_of(row).is_uncertain)
    ]
    uncertain_by_bias: dict[str, list[dict[str, Any]]] = {BIAS_UNKNOWN: []}
    uncertain_structures: dict[str, int] = {}
    for row in uncertain_all:
        exposure = _exposure_of(row)
        bias = exposure.market_bias if exposure is not None else BIAS_UNKNOWN
        uncertain_by_bias.setdefault(bias, []).append(row)
        key = exposure.structure if exposure is not None else "unknown"
        uncertain_structures[key] = uncertain_structures.get(key, 0) + 1
    # NO POOLED MONEY HERE. Its members span three statuses, and one figure over
    # a closed result and an open mark is exactly what the reviewer measured.
    # The money is already reported, once, inside whichever status population
    # owns the row.
    uncertain_cell = _bias_cell(uncertain_all, with_pnl=False)
    uncertain_cell.update(
        {
            "net_pnl_usd": None,
            "cross_cutting": True,
            "by_market_bias": {
                bias: _bias_cell(bias_rows, with_pnl=False)
                for bias, bias_rows in uncertain_by_bias.items()
            },
            "by_structure": uncertain_structures,
            "by_population": {
                name: summary[name]["n_uncertain"]
                for name in PERSONAL_EVIDENCE_STATUS_POPULATIONS
            },
            # Every member, named with the status it is ALSO counted under, so
            # nobody has to guess which population a listed trade came from.
            "members": [
                {
                    "trade_id": str(row.get("trade_id") or ""),
                    "symbol": str(row.get("symbol") or ""),
                    "status": str(row.get("status") or "").upper(),
                    "population": _population_of(row),
                    "instrument": (
                        _exposure_of(row).instrument if _exposure_of(row) else "UNKNOWN"
                    ),
                    "structure": (
                        _exposure_of(row).structure if _exposure_of(row) else "unknown"
                    ),
                    "market_bias": (
                        _exposure_of(row).market_bias if _exposure_of(row) else BIAS_UNKNOWN
                    ),
                }
                for row in uncertain_all
            ],
            "note": (
                "Counted inside the population its status puts it in; listed here "
                "because the instrument or the structure cannot be named. No money "
                "is pooled across the three."
            ),
        }
    )
    summary["uncertain"] = uncertain_cell

    # ------------------------------------------------------------------
    # coverage - ONE denominator for both tag lanes
    # ------------------------------------------------------------------
    closed = [row for row in rows if str(row.get("status") or "").upper() == "CLOSED"]
    partly = [row for row in rows if str(row.get("status") or "").upper() == "CLOSED_PARTIAL"]
    # REVIEWABLE = closed OR partly closed. The first cut counted confirmed tags
    # over CLOSED only and provisional over EVERY row, so the live journal's one
    # confirmed tag - on a CLOSED_PARTIAL trade, EAT 2026-08-21 - fell out of the
    # numerator while its 26 provisional siblings stayed in, and the headline
    # said "No confirmed setup tags" about a journal that holds one. A tag on a
    # half-closed trade is still the trader's answer.
    reviewable = closed + partly
    confirmed_rows = [row for row in reviewable if _confirmed_setup_tags(row)]
    provisional_rows = [row for row in reviewable if _provisional_setup_tags(row)]
    risk_rows = [row for row in reviewable if _coerce_float(row.get("planned_risk")) is not None]
    total = len(reviewable)
    coverage = {
        "confirmed": len(confirmed_rows),
        "closed": len(closed),
        "partly_closed": len(partly),
        "reviewable": total,
        "provisional": len(provisional_rows),
        "planned_risk": len(risk_rows),
        "line": (
            f"Confirmed tags: {len(confirmed_rows)} of {total} closed or partly "
            f"closed trades. Provisional awaiting review: {len(provisional_rows)}. "
            f"Planned risk recorded: {len(risk_rows)} of {total}."
        ),
    }

    best, headline = _best_confirmed_setup(
        confirmed_rows, coverage, floor=MIN_REPORTABLE_N, wilson=wilson_lower_bound
    )
    summary["coverage"] = coverage
    summary["best_setup"] = best
    summary["headline"] = headline
    summary["populations"] = list(PERSONAL_EVIDENCE_POPULATIONS)
    summary["status_populations"] = list(PERSONAL_EVIDENCE_STATUS_POPULATIONS)
    summary["directional_biases"] = list(DIRECTIONAL_BIASES)
    return summary


def _best_confirmed_setup(
    confirmed_rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    *,
    floor: int,
    wilson,
) -> tuple[dict[str, Any] | None, str]:
    """The best CONFIRMED setup, or the refusal that stands in for it.

    Two refusals, one sentence each, and the phrase *"no personal setup can be
    called best"* survives in both because that is the claim being refused.
    Above the floor the winner is chosen the way decision 0016 requires of every
    trader-facing surface: **win rate first, ranked on the Wilson LOWER BOUND**
    (`swing_headline`'s one z), with the raw rate and n beside it.
    """
    confirmed = int(coverage["confirmed"])
    provisional = int(coverage["provisional"])
    if confirmed <= 0:
        return None, (
            f"No confirmed setup tags ({provisional} provisional awaiting review) - "
            "no personal setup can be called best."
        )
    if confirmed < int(floor):
        # NAMES THE COUNT IT HAS. "No confirmed setup tags" is reserved for a
        # true zero: the live journal holds ONE (on a CLOSED_PARTIAL trade, EAT
        # 2026-08-21) and saying it holds none is a false statement about the
        # trader's own work, not a conservative one.
        tags = "tag" if confirmed == 1 else "tags"
        return None, (
            f"{confirmed} confirmed setup {tags} - under the n={int(floor)} floor "
            f"({provisional} provisional awaiting review) - "
            "no personal setup can be called best."
        )

    per_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in confirmed_rows:
        for tag in _confirmed_setup_tags(row):
            per_tag[tag].append(row)
    ranked: list[dict[str, Any]] = []
    for tag, tag_rows in per_tag.items():
        pnls = [_cad_pnl(row) for row in tag_rows]
        measured = [value for value in pnls if value is not None]
        if len(measured) < int(floor):
            continue
        wins = sum(1 for value in measured if value > 0)
        ranked.append(
            {
                "label": tag,
                "n": len(measured),
                "wins": wins,
                "win_rate": wins / len(measured),
                "win_rate_lower_bound": wilson(wins, len(measured)),
                "net_pnl": sum(measured),
            }
        )
    if not ranked:
        return None, (
            f"{confirmed} confirmed setup tag(s), but no single setup reaches "
            f"{int(floor)} closed trades - no personal setup can be called best."
        )
    ranked.sort(key=lambda row: (-(row["win_rate_lower_bound"] or 0.0), row["label"]))
    best = ranked[0]
    return best, (
        f"Best confirmed setup: {best['label']} - win rate {best['win_rate'] * 100:.0f}% "
        f"on n={best['n']} (Wilson lower bound "
        f"{(best['win_rate_lower_bound'] or 0.0) * 100:.0f}%)."
    )


#: Below this share of closed trades, a confirmed-tag dimension is not a
#: breakdown of the trader's setups - it is a breakdown of the handful they
#: happened to tag. Live on 2026-09-01: ONE confirmed tag across 193 trades.
CONFIRMED_TAG_COVERAGE_FLOOR = 0.10


def _empty_dimension_notes(
    trades: list[dict[str, Any]], groups: dict[str, list[dict[str, Any]]]
) -> dict[str, str]:
    """One sentence per group whose coverage is too thin to read as a chart.

    "My setups" renders beside a full "auto tags" chart, so two charts of the
    same width sit side by side while one of them rests on a single trade. The
    reader is not told; they see a bar and read it as a finding.

    THE GROUP IS NEVER HIDDEN. Hiding it would replace a visible thin answer
    with an invisible one, and the whole point is that the trader can see how
    little they have tagged - that is the prompt to tag more. The note is
    PREPENDED to the group's own label, using the same refusal-message
    mechanism `resolve_pnl_key` already uses to explain a total it will not
    compute.

    Coverage is measured against CLOSED trades, which is the denominator every
    number in these groups is computed over.
    """
    closed = [row for row in trades if counts_in_pnl(row)]
    if not closed:
        return {}
    notes: dict[str, str] = {}
    for group_name in ("my setups",):
        # COUNTED OVER TRADES, NOT OVER BUCKETS (R1).
        #
        # "My setups" is NON-EXCLUSIVE: a trade carrying three tags appears in
        # three buckets, so summing each bucket's `closed` counted it three
        # times. Live, 24 tagged trades of 156 measured as 40% coverage and the
        # note therefore never appeared - the one honesty this note exists to
        # provide was suppressed by its own arithmetic.
        #
        # And it ignored `tag_status` (P6a), so a machine-applied PROVISIONAL
        # tag counted as the trader's. Coverage of confirmed tags means exactly
        # that: distinct closed trades whose tags the trader stands behind.
        tagged = sum(
            1
            for row in closed
            if _confirmed_setup_tags(row)
        )
        share = tagged / len(closed)
        if share >= CONFIRMED_TAG_COVERAGE_FLOOR:
            continue
        notes[group_name] = (
            f"ONLY {tagged} OF {len(closed)} CLOSED TRADES CARRY A CONFIRMED TAG "
            f"({share * 100:.0f}%). This is a breakdown of those few, not of your "
            "setups - read it as a prompt to tag more, never as a ranking. The "
            "auto-tag chart beside it covers every trade and is the one to read "
            "until this catches up."
        )
    return notes


def _fmt_money(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:,.2f}"


def _fmt_pct(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric * 100.0:.1f}%"


def _fmt_ratio(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.2f}"


def build_analytics_text(trades: list[dict[str, Any]]) -> str:
    summary = build_analytics_summary(trades)
    overall = summary["overall"]
    lines = [
        "Journal Analytics",
        "",
        (
            f"Closed={overall['closed']} Open={overall['open']} WR={_fmt_pct(overall['win_rate'])} "
            f"PF={_fmt_ratio(overall['profit_factor'])} Net={_fmt_money(overall['net_pnl'])} "
            f"GrossWin={_fmt_money(overall['gross_win'])} GrossLoss={_fmt_money(overall['gross_loss'])}"
        ),
        "",
    ]
    for group_name, rows in summary["groups"].items():
        lines.append(group_name.replace("_", " ").title())
        if not rows:
            lines.append("  None")
        for row in rows[:25]:
            lines.append(
                "  "
                f"{row['label']}: closed={row['closed']} WR={_fmt_pct(row['win_rate'])} "
                f"PF={_fmt_ratio(row['profit_factor'])} net={_fmt_money(row['net_pnl'])} "
                f"avgW={_fmt_money(row['avg_win'])} avgL={_fmt_money(row['avg_loss'])}"
            )
        lines.append("")
    return "\n".join(lines).strip()


# -- Journal page stats (journal UI overhaul 2026-09-23) ---------------------
#
# Pure functions over trade rows the page has already loaded. Each takes the
# P&L column `resolve_pnl_key` chose, so every number agrees with the headline.

WEEKDAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

#: Hold-time buckets as (label, upper bound in minutes). The first four are
#: same-day holds; the last two are for trades held past the entry day.
HOLD_TIME_BUCKETS = (
    ("under 5 min", 5.0),
    ("5-30 min", 30.0),
    ("30 min-2 h", 120.0),
    ("2 h+ same day", None),
    ("overnight, 1-5 days", 5 * 24 * 60.0),
    ("over 5 days", None),
)

#: A closed trade whose P&L is within this of zero is breakeven, not a win or loss.
BREAKEVEN_EPSILON = 0.005


def _is_closed(row: dict[str, Any]) -> bool:
    # Closed with a real entry: a made-up entry never reaches a stat.
    return counts_in_pnl(row)


def close_order_key(row: dict[str, Any]) -> tuple[str, str]:
    """Sort key putting closed trades in the order they actually closed."""
    moment = _market_moment(row.get("closed_at") or row.get("trade_date") or row.get("opened_at"))
    stamp = moment.isoformat() if moment is not None else str(row.get("trade_date") or "")
    return (stamp, str(row.get("trade_id") or ""))


def trade_r_multiple(row: dict[str, Any]) -> float | None:
    """`net_pnl_cad / |planned_risk|`, the journal's one R, or None."""
    risk = _coerce_float(row.get("planned_risk"))
    pnl = _coerce_float(row.get("net_pnl_cad"))
    if risk is None or pnl is None or abs(risk) < 1e-9:
        return None
    return pnl / abs(risk)


def trade_performance_stats(
    trades: list[dict[str, Any]], pnl_key: str = "net_pnl"
) -> dict[str, Any]:
    """The stat-card numbers for closed trades, ordered by close time.

    A closed trade with no value in ``pnl_key`` is counted in ``unpriced`` and
    left out of every figure - missing is unknown, never zero.
    """
    closed = sorted((row for row in trades if _is_closed(row)), key=close_order_key)
    values: list[float] = []
    unpriced = 0
    for row in closed:
        value = _coerce_float(row.get(pnl_key)) if pnl_key else None
        if value is None:
            unpriced += 1
            continue
        values.append(value)
    wins = [value for value in values if value > BREAKEVEN_EPSILON]
    losses = [value for value in values if value < -BREAKEVEN_EPSILON]
    gross_win = sum(wins)
    gross_loss = sum(losses)
    count = len(values)
    net = sum(values)

    peak = 0.0
    running = 0.0
    max_drawdown = 0.0
    win_streak = loss_streak = best_win_streak = best_loss_streak = 0
    for value in values:
        running += value
        peak = max(peak, running)
        max_drawdown = min(max_drawdown, running - peak)
        if value > BREAKEVEN_EPSILON:
            win_streak, loss_streak = win_streak + 1, 0
        elif value < -BREAKEVEN_EPSILON:
            win_streak, loss_streak = 0, loss_streak + 1
        else:
            win_streak = loss_streak = 0
        best_win_streak = max(best_win_streak, win_streak)
        best_loss_streak = max(best_loss_streak, loss_streak)

    r_values = [r for r in (trade_r_multiple(row) for row in closed) if r is not None]
    avg_win = (gross_win / len(wins)) if wins else None
    avg_loss = (gross_loss / len(losses)) if losses else None
    return {
        "trades": len(trades),
        "closed": count,
        "unpriced": unpriced,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": count - len(wins) - len(losses),
        "win_rate": (len(wins) / count) if count else None,
        "net_pnl": net if count else None,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": (gross_win / abs(gross_loss)) if gross_loss < 0 else None,
        "expectancy": (net / count) if count else None,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": (avg_win / abs(avg_loss)) if avg_win is not None and avg_loss else None,
        "largest_win": max(wins) if wins else None,
        "largest_loss": min(losses) if losses else None,
        "max_drawdown": max_drawdown if count else None,
        "max_win_streak": best_win_streak,
        "max_loss_streak": best_loss_streak,
        "current_streak": win_streak if win_streak else -loss_streak,
        "avg_r": (sum(r_values) / len(r_values)) if r_values else None,
        "r_trades": len(r_values),
    }


def direction_split_stats(
    trades: list[dict[str, Any]], pnl_key: str = "net_pnl"
) -> dict[str, dict[str, Any]]:
    """`trade_performance_stats` for longs and shorts, side by side."""
    split: dict[str, list[dict[str, Any]]] = {"LONG": [], "SHORT": []}
    for row in trades:
        side = _normalize_side(row.get("direction"))
        if side in split:
            split[side].append(row)
    return {side: trade_performance_stats(rows, pnl_key) for side, rows in split.items()}


def group_expectancy(row: dict[str, Any]) -> float | None:
    """Net per closed trade for one `_summary_for_rows` bucket, or None."""
    net = _coerce_float(row.get("net_pnl"))
    closed = int(row.get("closed") or 0)
    if net is None or closed <= 0:
        return None
    return net / closed


def hold_time_bucket(row: dict[str, Any]) -> str | None:
    """Which `HOLD_TIME_BUCKETS` label a closed trade falls in, or None."""
    if not _is_closed(row):
        return None
    opened = _market_moment(row.get("opened_at"))
    closed = _market_moment(row.get("closed_at"))
    if opened is None or closed is None:
        return None
    minutes = max(0.0, (closed - opened).total_seconds() / 60.0)
    if closed.date() == opened.date():
        for label, bound in HOLD_TIME_BUCKETS[:4]:
            if bound is None or minutes < bound:
                return label
    overnight_label, overnight_bound = HOLD_TIME_BUCKETS[4]
    return overnight_label if minutes <= overnight_bound else HOLD_TIME_BUCKETS[5][0]


def entry_hour_label(moment: datetime) -> str:
    """One-hour entry bucket in market time, e.g. ``09:00-10:00 ET``."""
    return f"{moment.hour:02d}:00-{(moment.hour + 1) % 24:02d}:00 ET"


def time_breakdown_groups(
    trades: list[dict[str, Any]], pnl_key: str = "net_pnl"
) -> dict[str, list[dict[str, Any]]]:
    """By weekday and hour of ENTRY, and by hold time, in natural order.

    Rows have the same shape as `build_analytics_summary` group rows, plus
    ``expectancy``. A trade with no readable timestamp is in no bucket.
    """
    weekday: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hour: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trades:
        moment = _market_moment(row.get("opened_at"))
        if moment is not None:
            weekday[WEEKDAY_LABELS[moment.weekday()]].append(row)
            hour[entry_hour_label(moment)].append(row)
        bucket = hold_time_bucket(row)
        if bucket is not None:
            hold[bucket].append(row)

    def rows_for(buckets: dict[str, list[dict[str, Any]]], order: list[str]) -> list[dict[str, Any]]:
        out = []
        for label in order:
            if label not in buckets:
                continue
            item = _summary_for_rows(buckets[label], pnl_key or "net_pnl")
            if not pnl_key:
                item = {**item, "net_pnl": None, "gross_win": None, "gross_loss": None}
            item["label"] = label
            item["expectancy"] = group_expectancy(item)
            out.append(item)
        return out

    return {
        "weekday (entry)": rows_for(weekday, list(WEEKDAY_LABELS)),
        "hour of entry": rows_for(hour, sorted(hour)),
        "hold time": rows_for(hold, [label for label, _bound in HOLD_TIME_BUCKETS]),
    }


def calendar_day_stats(
    trades: list[dict[str, Any]], *, pnl_key: str = "net_pnl"
) -> dict[str, dict[str, Any]]:
    """Per-day net, trade count, wins and losses; days as `calendar_pnl_by_day`."""
    days: dict[str, dict[str, Any]] = {}
    for trade in trades:
        if not _is_closed(trade):
            continue
        trade_day = _parse_date(trade.get("closed_at") or trade.get("trade_date") or trade.get("opened_at"))
        if trade_day is None:
            continue
        pnl = _coerce_float(trade.get(pnl_key))
        if pnl is None:
            continue
        entry = days.setdefault(
            trade_day.isoformat(), {"net": 0.0, "trades": 0, "wins": 0, "losses": 0}
        )
        entry["net"] += pnl
        entry["trades"] += 1
        if pnl > BREAKEVEN_EPSILON:
            entry["wins"] += 1
        elif pnl < -BREAKEVEN_EPSILON:
            entry["losses"] += 1
    return days


def pnl_currency_label(
    currency_mode: str | None, pnl_key: str, currencies: list[str] | None = None
) -> str:
    """What currency the page's totals are in, in a few words."""
    if not pnl_key:
        return "no total (mixed currencies)"
    if pnl_key == "net_pnl_cad":
        return "CAD"
    if pnl_key == USD_BOOKED_KEY:
        return "USD"
    if pnl_key == USD_ESTIMATE_KEY:
        return "USD (estimate)"
    found = {str(code).upper() for code in (currencies or []) if code}
    if len(found) == 1:
        return next(iter(found))
    mode = str(currency_mode or "").strip().upper()
    return mode if mode in {"CAD", "USD"} else "native currency"
