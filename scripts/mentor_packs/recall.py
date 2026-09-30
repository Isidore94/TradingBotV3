"""Recall pack: what was said before, found by embedding search over the mentor chat store.

The app installs a searcher once the brain answers (``set_searcher``) over turns, night
digests and profile notes. With the brain down it falls back to a plain substring search
(``set_fallback``); with neither the pack says recall is off. Pure: the embedding call and
the store are injected. Row ids are ``mem:<kind>:<id>``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from mentor_packs.registry import Pack, make_pack

NAME = "recall"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": "Search earlier mentor conversations, night digests and profile notes for a topic, e.g. 'NVDA last week'.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look for."}},
            "required": ["query"],
        },
    },
}
DEFAULT_K = 5
OFF_TEXT = "recall is off (the brain is not up)"

Searcher = Callable[[str, int], list[Mapping[str, Any]]]
_searcher: Searcher | None = None


_fallback: Searcher | None = None
_drop_logged = False
_log = logging.getLogger(__name__)
FALLBACK_EMPTY = "nothing found (plain text search: the brain is off)"


def set_searcher(searcher: Searcher | None) -> None:
    global _searcher
    _searcher = searcher


def set_fallback(fallback: Searcher | None) -> None:
    """The substring search used while no embedding searcher is installed."""
    global _fallback
    _fallback = fallback


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0  # vectors from two different embedding models never compare
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def top_k(query_vec: Sequence[float], rows: Iterable[Mapping[str, Any]], k: int = DEFAULT_K) -> list[dict[str, Any]]:
    """Rows (each with ``vector``) ranked by cosine to the query, best first."""
    scored = [{**dict(row), "score": cosine(query_vec, row.get("vector") or ())} for row in rows]
    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored[: max(0, int(k))]


def make_searcher(
    embed: Callable[[list[str]], list[list[float]]],
    rows: Callable[[], Iterable[Mapping[str, Any]]],
) -> Searcher:
    """A searcher from an embed call and a row source (``kind``, ``ref_id``, ``text``, ``vector``)."""

    def search(query: str, k: int) -> list[Mapping[str, Any]]:
        vectors = embed([query])
        if not vectors:
            return []
        return top_k(vectors[0], rows(), k)

    return search


def _search_or_fall_back(search: Searcher, query: str, k: int) -> tuple[list[Mapping[str, Any]], bool] | None:
    """(hits, fell_back): an embed searcher that raises hands this call to the substring search.

    One warning per drop; None when there is no fallback to hand to.
    """
    global _drop_logged
    try:
        hits = list(search(query, k))
    except Exception as exc:  # noqa: BLE001 - the brain can drop mid-day; recall never raises
        if not _drop_logged:
            _log.warning("recall: the embedding search failed (%s); using plain text search", exc)
            _drop_logged = True
        if _fallback is None or search is _fallback:
            return None
        try:
            return list(_fallback(query, k)), True
        except Exception:  # noqa: BLE001
            _log.debug("recall: the substring search failed too.", exc_info=True)
            return None
    if search is not _fallback:
        _drop_logged = False  # the brain is back: the next drop logs again
    return hits, search is _fallback


def build(query: str = "", k: int = DEFAULT_K, *, searcher: Searcher | None = None) -> Pack:
    search = searcher or _searcher
    empty = "nothing found"
    if search is None and _fallback is not None:
        search, empty = _fallback, FALLBACK_EMPTY
    if not str(query or "").strip():
        return make_pack(NAME, (), empty_text="recall needs a query")
    if search is None:
        return make_pack(NAME, (), empty_text=OFF_TEXT)
    found = _search_or_fall_back(search, str(query), int(k))
    if found is None:
        return make_pack(NAME, (), empty_text=OFF_TEXT)
    hits, fell_back = found
    if fell_back:
        empty = FALLBACK_EMPTY
    rows = [
        {
            "id": f"mem:{hit.get('kind', 'turn')}:{hit.get('ref_id')}",
            "kind": "memory",
            "text": str(hit.get("text") or "")[:400],
            "score": round(float(hit.get("score") or 0.0), 3),
        }
        for hit in hits
    ]
    return make_pack(NAME, rows, empty_text=empty)


def fixture() -> Pack:
    stored = [
        {"kind": "turn", "ref_id": 1, "text": "NVDA short worked below the AVWAP", "vector": [1.0, 0.0]},
        {"kind": "note", "ref_id": 2, "text": "I stop trading after two losses", "vector": [0.0, 1.0]},
    ]
    searcher = make_searcher(lambda texts: [[1.0, 0.1] for _ in texts], lambda: stored)
    return build("NVDA", searcher=searcher)
