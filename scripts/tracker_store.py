"""The setup tracker as a record store (assessment packet F3, step 1).

The Master AVWAP setup tracker is one JSON document: 11,334 setups, 3,992 study
setups and 396 control setups at ~76 KB each - 1.15 GB on 2026-09-03 - rewritten
WHOLE once a day, plus a 1.13 GB rolling ``.bak``. Every reader that wants one
symbol's episodes loads all of them (the 2026-08-31 journal freeze was exactly
that), and every save is a 1.15 GB atomic replace.

This module is the first, deliberately shadow-only step away from that shape:

* ``save_payload`` writes the same payload into a SQLite file beside the JSON,
  one row per tracker RECORD (section, key, JSON blob, content hash), and only
  the records whose content changed are rewritten. The small sections
  (``daily_watchlists``, ``stats``, ``setup_type_stats``, ``attribute_registry``)
  and the header fields ride in a ``sections`` table.
* ``load_payload`` rebuilds the exact dict the JSON loader would hand back, and
  ``load_records`` answers "these symbols / this section" without loading the
  rest - the read shape every reader will move to.
* ``verify`` compares the SQLite view against the JSON file and reports every
  difference, so parity is measured on the live desk before any reader changes.

**What this step does NOT do.** The JSON file stays authoritative: the scanner
still loads from it and still writes it first; the SQLite write happens after,
behind ``tracker_storage_shadow`` (a local setting, default ON), and a failure
there is a warning, never a failed save. No reader is moved. No detector,
scoring or tracker logic changes - the payload is copied, not interpreted.
Moving the readers and retiring the JSON is F3 step 2, gated on ``verify``
reporting zero differences across a week of live saves.

**Readers (P0-2 2d, decision 0017).** ``load_fresh_payload`` and
``load_fresh_projection`` serve the store only when its source stamp matches
the JSON file on disk (path, size, mtime); otherwise they return the reason and
the caller reads the JSON. The JSON is still written first and stays the truth.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
#: The three sections whose values are one record per setup key.
RECORD_SECTIONS = ("setups", "control_setups", "study_setups")
#: Everything else in the payload is small and stored as one JSON value each.
#: `saved_at` / `saved_by` joined on 2026-09-05 (packet M3.2). The mirror
#: FOLLOWS the JSON (decision 0017): a header key the JSON carries and the
#: mirror does not is a parity difference `verify` would report forever, which
#: is gate #57's whole measurement.
HEADER_FIELDS = ("schema_version", "updated_at", "data_session", "saved_at", "saved_by")
SECTION_FIELDS = ("daily_watchlists", "stats", "setup_type_stats", "attribute_registry")
SHADOW_SETTING = "tracker_storage_shadow"
#: Mirror format 2 (P0-2 2d): record text is encoded exactly as the JSON save
#: encodes it (insertion order, the saver's ``default``), each section's key order
#: is kept in ``meta``, and the mirror is stamped with the JSON file it copies.
#: Readers use the store only when that stamp matches the file on disk.
MIRROR_FORMAT = 2
ORDER_META_PREFIX = "order:"
SOURCE_META_KEYS = ("mirror_format", "source_path", "source_size", "source_mtime_ns")


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _encode(value: Any, json_default=None) -> str:
    """The JSON save's own encoding: insertion order, compact, the saver's ``default``."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=json_default or str)


def _normalized_source_path(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _source_stamp(path: Path | str) -> dict[str, str] | None:
    """The JSON file's identity (path, size, mtime_ns); None when it cannot be stat'ed."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return {
        "mirror_format": str(MIRROR_FORMAT),
        "source_path": _normalized_source_path(path),
        "source_size": str(int(stat.st_size)),
        "source_mtime_ns": str(int(stat.st_mtime_ns)),
    }


def file_stamp(path: Path | str) -> dict[str, str] | None:
    """The stamp a mirror of ``path`` carries: take it when payload and file are known to match."""
    return _source_stamp(path)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def shadow_enabled() -> bool:
    """Whether the scanner mirrors each save into SQLite. Default ON."""
    try:
        from project_paths import get_local_setting

        raw = get_local_setting(SHADOW_SETTING, True)
    except Exception:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() not in ("0", "false", "no", "off")
    return bool(raw)


@dataclass
class SaveReport:
    path: str = ""
    records_seen: int = 0
    records_written: int = 0
    records_deleted: int = 0
    sections_written: int = 0
    seconds: float = 0.0


@dataclass
class VerifyReport:
    ok: bool = True
    records_json: int = 0
    records_db: int = 0
    missing_in_db: list[str] = field(default_factory=list)
    extra_in_db: list[str] = field(default_factory=list)
    differing: list[str] = field(default_factory=list)
    header_differences: list[str] = field(default_factory=list)

    @property
    def differences(self) -> int:
        return len(self.missing_in_db) + len(self.extra_in_db) + len(self.differing) + len(self.header_differences)


class TrackerStore:
    """SQLite record store for the tracker payload. One file, WAL mode."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    # -- connection ---------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sections (name TEXT PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS records ("
            " section TEXT NOT NULL, key TEXT NOT NULL, symbol TEXT NOT NULL DEFAULT '',"
            " scan_date TEXT NOT NULL DEFAULT '', payload TEXT NOT NULL, digest TEXT NOT NULL,"
            " written_at TEXT NOT NULL, PRIMARY KEY (section, key))"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS records_symbol ON records (section, symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS records_scan_date ON records (section, scan_date)")
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
        )
        return conn

    # -- writing ------------------------------------------------------------
    def save_payload(
        self,
        payload: dict,
        *,
        now: datetime | None = None,
        source_path: Path | str | None = None,
        source_stamp: dict | None = None,
        json_default=None,
    ) -> SaveReport:
        """Mirror ``payload`` into the store, rewriting only what changed.

        ``source_stamp`` is ``file_stamp(source_path)`` taken when ``payload``
        and the file were known to match (right after the save, or before the
        read). The mirror is stamped only if the file still has that stamp at
        commit; otherwise, or without both arguments, the stamp is cleared and
        readers use the JSON.
        """
        started = datetime.now(timezone.utc)
        stamp = (now or started).isoformat(timespec="seconds")
        report = SaveReport(path=str(self.path))
        conn = self._connect()
        try:
            with conn:
                for name in HEADER_FIELDS:
                    conn.execute(
                        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (name, _encode(payload.get(name), json_default)),
                    )
                for name in SECTION_FIELDS:
                    text = _encode(payload.get(name), json_default)
                    digest = _digest(text)
                    row = conn.execute("SELECT digest FROM sections WHERE name = ?", (name,)).fetchone()
                    if row is None or row[0] != digest:
                        conn.execute(
                            "INSERT INTO sections (name, payload, digest) VALUES (?, ?, ?)"
                            " ON CONFLICT(name) DO UPDATE SET payload=excluded.payload, digest=excluded.digest",
                            (name, text, digest),
                        )
                        report.sections_written += 1
                for section in RECORD_SECTIONS:
                    records = payload.get(section)
                    records = records if isinstance(records, dict) else {}
                    known = dict(
                        conn.execute("SELECT key, digest FROM records WHERE section = ?", (section,)).fetchall()
                    )
                    seen: set[str] = set()
                    for key, value in records.items():
                        key = str(key)
                        seen.add(key)
                        report.records_seen += 1
                        text = _encode(value, json_default)
                        digest = _digest(text)
                        if known.get(key) == digest:
                            continue
                        symbol = str(value.get("symbol") or "") if isinstance(value, dict) else ""
                        scan_date = str(value.get("scan_date") or "") if isinstance(value, dict) else ""
                        conn.execute(
                            "INSERT INTO records (section, key, symbol, scan_date, payload, digest, written_at)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?)"
                            " ON CONFLICT(section, key) DO UPDATE SET symbol=excluded.symbol,"
                            " scan_date=excluded.scan_date, payload=excluded.payload, digest=excluded.digest,"
                            " written_at=excluded.written_at",
                            (section, key, symbol, scan_date, text, digest, stamp),
                        )
                        report.records_written += 1
                    gone = [key for key in known if key not in seen]
                    if gone:
                        conn.executemany(
                            "DELETE FROM records WHERE section = ? AND key = ?", [(section, key) for key in gone]
                        )
                        report.records_deleted += len(gone)
                    order_text = _encode([str(key) for key in records])
                    order_key = ORDER_META_PREFIX + section
                    row = conn.execute("SELECT value FROM meta WHERE key = ?", (order_key,)).fetchone()
                    if row is None or row[0] != order_text:
                        conn.execute(
                            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                            (order_key, order_text),
                        )
                stamp_meta = None
                if source_path is not None and source_stamp is not None:
                    stamp_meta = _source_stamp(source_path)
                    if stamp_meta != source_stamp:
                        logging.warning(
                            "Setup tracker mirror left unstamped: %s changed between its save and the mirror "
                            "(expected %s, found %s); readers will use the JSON.",
                            source_path, source_stamp, stamp_meta,
                        )
                        stamp_meta = None
                if stamp_meta is None:
                    conn.executemany("DELETE FROM meta WHERE key = ?", [(key,) for key in SOURCE_META_KEYS])
                else:
                    conn.executemany(
                        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        list(stamp_meta.items()),
                    )
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('mirrored_at', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (stamp,),
                )
        finally:
            conn.close()
        report.seconds = (datetime.now(timezone.utc) - started).total_seconds()
        return report

    # -- reading ------------------------------------------------------------
    def load_payload(self) -> dict | None:
        """The whole payload, shaped exactly like the JSON loader's input."""
        if not self.path.exists():
            return None
        conn = self._connect()
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            payload: dict = {}
            for name in HEADER_FIELDS:
                payload[name] = json.loads(meta[name]) if name in meta else None
            for name, text, _digest in conn.execute("SELECT name, payload, digest FROM sections"):
                payload[name] = json.loads(text)
            for section in RECORD_SECTIONS:
                payload[section] = _ordered_section(conn, meta, section)
        finally:
            conn.close()
        return payload

    def load_records(
        self, section: str = "setups", *, symbols: Iterable[str] | None = None, scan_dates: Iterable[str] | None = None
    ) -> dict[str, dict]:
        """One section, narrowed by symbol and/or scan date, without the rest."""
        if not self.path.exists():
            return {}
        clauses = ["section = ?"]
        params: list[Any] = [section]
        wanted_symbols = sorted({str(s).strip().upper() for s in (symbols or []) if str(s).strip()})
        if wanted_symbols:
            clauses.append(f"symbol IN ({','.join('?' * len(wanted_symbols))})")
            params.extend(wanted_symbols)
        wanted_dates = sorted({str(d) for d in (scan_dates or []) if str(d)})
        if wanted_dates:
            clauses.append(f"scan_date IN ({','.join('?' * len(wanted_dates))})")
            params.extend(wanted_dates)
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT key, payload FROM records WHERE {' AND '.join(clauses)} ORDER BY rowid", params
            ).fetchall()
        finally:
            conn.close()
        return {key: json.loads(text) for key, text in rows}

    def counts(self) -> dict[str, int]:
        if not self.path.exists():
            return {}
        conn = self._connect()
        try:
            return dict(conn.execute("SELECT section, COUNT(*) FROM records GROUP BY section").fetchall())
        finally:
            conn.close()

    # -- parity -------------------------------------------------------------
    def verify(self, payload: dict) -> VerifyReport:
        """Every difference between ``payload`` (the JSON truth) and this store."""
        report = VerifyReport()
        mirrored = self.load_payload()
        if mirrored is None:
            report.ok = False
            report.header_differences.append("store file missing")
            return report
        for name in HEADER_FIELDS + SECTION_FIELDS:
            if _dumps(payload.get(name)) != _dumps(mirrored.get(name)):
                report.header_differences.append(name)
        for section in RECORD_SECTIONS:
            truth = payload.get(section) if isinstance(payload.get(section), dict) else {}
            mine = mirrored.get(section) or {}
            report.records_json += len(truth)
            report.records_db += len(mine)
            for key, value in truth.items():
                key = str(key)
                if key not in mine:
                    report.missing_in_db.append(f"{section}:{key}")
                elif _dumps(value) != _dumps(mine[key]):
                    report.differing.append(f"{section}:{key}")
            for key in mine:
                if key not in truth:
                    report.extra_in_db.append(f"{section}:{key}")
        report.ok = report.differences == 0
        return report


def _ordered_section(conn: sqlite3.Connection, meta: dict, section: str) -> dict:
    """One record section as a dict, in the JSON's key order when the mirror kept it."""
    rows = {
        key: json.loads(text)
        for key, text in conn.execute(
            "SELECT key, payload FROM records WHERE section = ? ORDER BY rowid", (section,)
        )
    }
    return _apply_order(rows, meta.get(ORDER_META_PREFIX + section))


def _apply_order(rows: dict, order_text: str | None) -> dict:
    if not order_text:
        return rows
    try:
        order = json.loads(order_text)
    except ValueError:
        return rows
    ordered = {key: rows.pop(key) for key in order if key in rows}
    ordered.update(rows)
    return ordered


def _open_reader(path: Path) -> sqlite3.Connection:
    """A connection that never creates tables, holding one read snapshot."""
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.execute("BEGIN")
    return conn


def _staleness_reason(meta: dict, json_path: Path) -> str:
    """Empty when the store mirrors exactly the JSON file on disk, else why not."""
    if str(meta.get("mirror_format") or "") != str(MIRROR_FORMAT):
        return "store carries no format-2 source stamp (older mirror, or the last mirror had no source file)"
    current = _source_stamp(json_path)
    if current is None:
        return f"tracker JSON {json_path} is missing"
    if meta.get("source_path") != current["source_path"]:
        return f"store mirrors {meta.get('source_path')}, not {current['source_path']}"
    if (meta.get("source_size"), meta.get("source_mtime_ns")) != (
        current["source_size"],
        current["source_mtime_ns"],
    ):
        return "tracker JSON changed after the last mirror (size or mtime differ)"
    return ""


def load_fresh_payload(json_path: Path | str, db_path: Path | str | None = None) -> tuple[dict | None, str]:
    """``(payload, "")`` from the store when it mirrors ``json_path`` exactly, else ``(None, reason)``.

    The payload is the dict the JSON file holds (same keys, values and key
    order); callers normalize it as they normalize the JSON. Never raises.
    """
    json_path = Path(json_path)
    store_path = Path(db_path) if db_path is not None else default_store_path()
    if not store_path.exists():
        return None, f"no SQLite store at {store_path}"
    try:
        conn = _open_reader(store_path)
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            reason = _staleness_reason(meta, json_path)
            if reason:
                return None, reason
            payload: dict = {}
            for name in HEADER_FIELDS:
                if name in meta:
                    payload[name] = json.loads(meta[name])
            for name, text in conn.execute("SELECT name, payload FROM sections"):
                payload[name] = json.loads(text)
            # ~18k json.loads calls build millions of containers; cyclic GC passes
            # over them cost ~5 s on the live tracker and free nothing.
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                for section in RECORD_SECTIONS:
                    payload[section] = _ordered_section(conn, meta, section)
            finally:
                if gc_was_enabled:
                    gc.enable()
        finally:
            conn.close()
    except Exception as exc:
        return None, f"store unreadable: {type(exc).__name__}: {exc}"
    return payload, ""


def load_fresh_projection(
    json_path: Path | str,
    fields: Iterable[str],
    *,
    section: str = "setups",
    sections: Iterable[str] | None = None,
    db_path: Path | str | None = None,
) -> tuple[list[dict] | None, str]:
    """A few top-level fields of every dict record in ``section``, in the JSON's order.

    ``sections`` reads several sections, one after another, from one read
    snapshot. SQLite extracts the fields, so no record is parsed whole in
    Python. A field the record lacks is absent from its row, so ``row.get``
    reads ``None`` as it would on the JSON dict. ``(None, reason)`` unless the
    store mirrors ``json_path`` exactly. Never raises.
    """
    wanted_sections = [str(name) for name in sections] if sections is not None else [section]
    names = [str(name) for name in fields]
    json_path = Path(json_path)
    store_path = Path(db_path) if db_path is not None else default_store_path()
    if not store_path.exists():
        return None, f"no SQLite store at {store_path}"
    columns = ", ".join(["json_type(payload)"] + ["payload -> ?"] * len(names))
    paths = ['$."' + name.replace('"', '""') + '"' for name in names]
    try:
        conn = _open_reader(store_path)
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            reason = _staleness_reason(meta, json_path)
            if reason:
                return None, reason
            projected: list[dict] = []
            for name_of_section in wanted_sections:
                by_key: dict[str, dict] = {}
                for row in conn.execute(
                    f"SELECT key, {columns} FROM records WHERE section = ? ORDER BY rowid",
                    [*paths, name_of_section],
                ):
                    if row[1] != "object":
                        continue
                    by_key[row[0]] = {
                        name: json.loads(text) for name, text in zip(names, row[2:], strict=False) if text is not None
                    }
                projected.extend(_apply_order(by_key, meta.get(ORDER_META_PREFIX + name_of_section)).values())
        finally:
            conn.close()
    except Exception as exc:
        return None, f"store unreadable: {type(exc).__name__}: {exc}"
    return projected, ""


def default_store_path() -> Path:
    from project_paths import MASTER_AVWAP_SETUP_TRACKER_DB

    return Path(MASTER_AVWAP_SETUP_TRACKER_DB)


def mirror_payload(
    payload: dict,
    *,
    path: Path | str | None = None,
    source_path: Path | str | None = None,
    source_stamp: dict | None = None,
    json_default=None,
) -> SaveReport | None:
    """The scanner's hook: mirror after the JSON save. Never raises."""
    if not shadow_enabled():
        return None
    try:
        store = TrackerStore(path or default_store_path())
        report = store.save_payload(
            payload, source_path=source_path, source_stamp=source_stamp, json_default=json_default
        )
        logging.info(
            "Setup tracker mirrored to %s: %d records seen, %d written, %d deleted, %d sections, %.1fs",
            report.path, report.records_seen, report.records_written, report.records_deleted,
            report.sections_written, report.seconds,
        )
        return report
    except Exception:
        logging.warning("Setup tracker SQLite mirror failed; the JSON tracker is untouched.", exc_info=True)
        return None


def write_state_path() -> Path:
    from project_paths import SETUP_TRACKER_WRITE_STATE_FILE

    return Path(SETUP_TRACKER_WRITE_STATE_FILE)


def read_write_state(path: Path | str | None = None) -> dict:
    """The tracker write stamp ({last_written_at, last_failed_at, last_error, last_result}); {} if unknown."""
    try:
        payload = json.loads(Path(path or write_state_path()).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_state(update: dict, path: Path | str | None) -> None:
    target = Path(path or write_state_path())
    state = read_write_state(target)
    state.update(update)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(target)


def record_write_success(*, path: Path | str | None = None, now: datetime | None = None) -> None:
    """Stamp a good tracker write. Never raises."""
    try:
        stamp = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
        _write_state({"last_written_at": stamp, "last_result": "ok"}, path)
    except Exception:
        logging.warning("Setup tracker write stamp not saved.", exc_info=True)


def record_write_failure(
    error: str,
    *,
    slot: str = "",
    path: Path | str | None = None,
    ledger_path: Path | None = None,
    now: datetime | None = None,
) -> None:
    """Stamp a failed tracker write and append a keyless ``setup_tracker_write_failed`` ledger row. Never raises."""
    stamp = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    try:
        _write_state({"last_failed_at": stamp, "last_error": str(error)[:500], "last_result": "failed"}, path)
    except Exception:
        logging.warning("Setup tracker failure stamp not saved.", exc_info=True)
    try:
        from job_ledger import append_keyless_event

        append_keyless_event(
            "setup_tracker_write_failed",
            {"ts": stamp, "error": str(error)[:500], "slot": str(slot or "")},
            path=ledger_path,
        )
    except Exception:
        logging.warning("setup_tracker_write_failed ledger row not written.", exc_info=True)


def _short_stamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return text


def tracker_last_written_line(state: dict | None = None) -> str:
    """Health line: ``tracker last written <stamp>``."""
    state = read_write_state() if state is None else state
    return f"tracker last written {_short_stamp(state.get('last_written_at'))}"


def tracker_write_failure_line(state: dict | None = None) -> str:
    """Digest line when the last tracker write failed; "" otherwise."""
    state = read_write_state() if state is None else state
    if str(state.get("last_result") or "") != "failed":
        return ""
    return (
        f"setup tracker write failed {_short_stamp(state.get('last_failed_at'))}; "
        f"last good {_short_stamp(state.get('last_written_at'))}"
    )


#: The only stale tracker copies ``--prune-copies`` may delete (decided 2026-09-24; the
#: three 2026-09-05 leftovers beside the damaged SQLite added for P0-2).
PRUNABLE_COPY_NAMES = (
    "master_avwap_setup_tracker.json.bak",
    "master_avwap_setup_tracker.sqlite.damaged-20260905T200233",
    "master_avwap_setup_tracker.sqlite-shm.damaged-20260905T200233",
    "master_avwap_setup_tracker.sqlite-wal.damaged-20260905T200233",
    "master_avwap_setup_tracker_digests.json.damaged-20260905T200233",
)


def prune_copies(
    directory: Path | str,
    names: Iterable[str] = PRUNABLE_COPY_NAMES,
    *,
    delete: bool = False,
    ledger_path: Path | None = None,
) -> tuple[int, dict]:
    """List (and with ``delete`` remove) the stale tracker copies in ``directory``.

    Returns ``(exit_code, report)``. Any name outside ``PRUNABLE_COPY_NAMES``, or a
    target that is not a plain file, refuses the whole run: exit 2, nothing deleted,
    no ledger row. A listing or deletion writes one keyless ``tracker_prune_copies``
    ledger row.
    """
    root = Path(directory)
    requested = [str(name) for name in names]
    report: dict[str, Any] = {"dir": str(root), "delete": bool(delete), "candidates": [], "refused": []}
    for name in requested:
        target = root / name
        if name not in PRUNABLE_COPY_NAMES or Path(name).name != name:
            report["refused"].append({"name": name, "reason": "not one of the prunable copies"})
        elif target.is_symlink() or (target.exists() and not target.is_file()):
            report["refused"].append({"name": name, "reason": "not a plain file"})
    if report["refused"]:
        return 2, report
    for name in requested:
        target = root / name
        exists = target.is_file()
        report["candidates"].append(
            {"name": name, "exists": exists, "bytes": target.stat().st_size if exists else 0, "deleted": False}
        )
    exit_code = 0
    if delete:
        for item in report["candidates"]:
            if not item["exists"]:
                continue
            try:
                (root / item["name"]).unlink()
                item["deleted"] = True
            except OSError as exc:
                item["error"] = str(exc)
                exit_code = 1
    report["total_bytes"] = sum(int(item["bytes"]) for item in report["candidates"])
    try:
        from job_ledger import append_keyless_event

        append_keyless_event(
            "tracker_prune_copies",
            {
                "dir": str(root),
                "names": [item["name"] for item in report["candidates"] if item["exists"]],
                "bytes": report["total_bytes"],
                "deleted": bool(delete) and exit_code == 0,
                "deleted_names": [item["name"] for item in report["candidates"] if item["deleted"]],
            },
            path=ledger_path,
        )
    except Exception:
        logging.warning("tracker_prune_copies ledger row not written.", exc_info=True)
    return exit_code, report


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Setup tracker SQLite mirror: verify parity or mirror once.")
    parser.add_argument("command", nargs="?", choices=("verify", "mirror", "counts"))
    parser.add_argument("--json", default="", help="tracker JSON path (default: the desk's)")
    parser.add_argument("--db", default="", help="SQLite path (default: beside the JSON)")
    parser.add_argument(
        "--prune-copies",
        action="store_true",
        help="list the stale tracker copies (.bak and the .damaged-20260905T200233 leftovers); delete only with --yes",
    )
    parser.add_argument("--yes", action="store_true", help="with --prune-copies: actually delete")
    parser.add_argument("--dry-run", action="store_true", help="with --prune-copies: list only (the default)")
    parser.add_argument("--tracker-dir", default="", help="with --prune-copies: directory (default: the tracker's)")
    parser.add_argument("--name", action="append", default=None, help="with --prune-copies: one copy name")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from project_paths import MASTER_AVWAP_SETUP_TRACKER_FILE

    if args.prune_copies:
        directory = Path(args.tracker_dir) if args.tracker_dir else Path(MASTER_AVWAP_SETUP_TRACKER_FILE).parent
        code, report = prune_copies(
            directory,
            args.name or PRUNABLE_COPY_NAMES,
            delete=bool(args.yes) and not args.dry_run,
        )
        print(json.dumps(report, indent=2))
        return code
    if not args.command:
        parser.error("a command (verify, mirror, counts) or --prune-copies is required")

    json_path = Path(args.json) if args.json else Path(MASTER_AVWAP_SETUP_TRACKER_FILE)
    store = TrackerStore(Path(args.db) if args.db else default_store_path())
    if args.command == "counts":
        print(json.dumps({"path": str(store.path), "records": store.counts()}, indent=2))
        return 0
    stamp_before_read = file_stamp(json_path)  # a rewrite during the read or mirror leaves it unstamped
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if args.command == "mirror":
        report = store.save_payload(payload, source_path=json_path, source_stamp=stamp_before_read)
        print(json.dumps(report.__dict__, indent=2))
        return 0
    report = store.verify(payload)
    print(
        json.dumps(
            {
                "ok": report.ok,
                "records_json": report.records_json,
                "records_db": report.records_db,
                "differences": report.differences,
                "missing_in_db": report.missing_in_db[:20],
                "extra_in_db": report.extra_in_db[:20],
                "differing": report.differing[:20],
                "header_differences": report.header_differences,
            },
            indent=2,
        )
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
