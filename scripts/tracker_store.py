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
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
#: The three sections whose values are one record per setup key.
RECORD_SECTIONS = ("setups", "control_setups", "study_setups")
#: Everything else in the payload is small and stored as one JSON value each.
HEADER_FIELDS = ("schema_version", "updated_at", "data_session")
SECTION_FIELDS = ("daily_watchlists", "stats", "setup_type_stats", "attribute_registry")
SHADOW_SETTING = "tracker_storage_shadow"


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


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
    def save_payload(self, payload: dict, *, now: datetime | None = None) -> SaveReport:
        """Mirror ``payload`` into the store, rewriting only what changed."""
        started = datetime.now(timezone.utc)
        stamp = (now or started).isoformat(timespec="seconds")
        report = SaveReport(path=str(self.path))
        conn = self._connect()
        try:
            with conn:
                for name in HEADER_FIELDS:
                    conn.execute(
                        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (name, _dumps(payload.get(name))),
                    )
                for name in SECTION_FIELDS:
                    text = _dumps(payload.get(name))
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
                        text = _dumps(value)
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
            for name, text, _digest_ in conn.execute("SELECT name, payload, digest FROM sections"):
                payload[name] = json.loads(text)
            for section in RECORD_SECTIONS:
                payload[section] = {
                    key: json.loads(text)
                    for key, text in conn.execute(
                        "SELECT key, payload FROM records WHERE section = ? ORDER BY rowid", (section,)
                    )
                }
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


def default_store_path() -> Path:
    from project_paths import MASTER_AVWAP_SETUP_TRACKER_DB

    return Path(MASTER_AVWAP_SETUP_TRACKER_DB)


def mirror_payload(payload: dict, *, path: Path | str | None = None) -> SaveReport | None:
    """The scanner's hook: mirror after the JSON save. Never raises."""
    if not shadow_enabled():
        return None
    try:
        store = TrackerStore(path or default_store_path())
        report = store.save_payload(payload)
        logging.info(
            "Setup tracker mirrored to %s: %d records seen, %d written, %d deleted, %d sections, %.1fs",
            report.path, report.records_seen, report.records_written, report.records_deleted,
            report.sections_written, report.seconds,
        )
        return report
    except Exception:
        logging.warning("Setup tracker SQLite mirror failed; the JSON tracker is untouched.", exc_info=True)
        return None


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Setup tracker SQLite mirror: verify parity or mirror once.")
    parser.add_argument("command", choices=("verify", "mirror", "counts"))
    parser.add_argument("--json", default="", help="tracker JSON path (default: the desk's)")
    parser.add_argument("--db", default="", help="SQLite path (default: beside the JSON)")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from project_paths import MASTER_AVWAP_SETUP_TRACKER_FILE

    json_path = Path(args.json) if args.json else Path(MASTER_AVWAP_SETUP_TRACKER_FILE)
    store = TrackerStore(Path(args.db) if args.db else default_store_path())
    if args.command == "counts":
        print(json.dumps({"path": str(store.path), "records": store.counts()}, indent=2))
        return 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if args.command == "mirror":
        report = store.save_payload(payload)
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
