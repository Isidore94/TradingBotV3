"""R10.A: the dated evidence snapshot.

The cold push deliberately excludes hot state, which is the 3.5 GB of
`data\\runtime`, the 529 MB diagnostics tree and the home-root evidence files -
i.e. everything the Evidence Plane program exists to protect. This snapshots
them, dated, without moving them (decision 0015: hot files stay on local SSD).

Every test here works on a temp tree. Nothing touches the live store.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ops import evidence_snapshot as snap  # noqa: E402


def _scope(tmp_path: Path):
    src = tmp_path / "home"
    (src / "data" / "runtime").mkdir(parents=True)
    (src / "data" / "runtime" / "outcomes.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (src / "trader_annotations.jsonl").write_text('{"x":1}\n', encoding="utf-8")
    # A subdirectory of the home root that "files" mode must NOT descend into:
    # it is the cold push's scope, not this one.
    (src / "output").mkdir()
    (src / "output" / "report.txt").write_text("cold", encoding="utf-8")
    return src, [
        snap.ScopeItem("data-runtime", src / "data" / "runtime", "tree"),
        snap.ScopeItem("home-root", src, "files"),
    ]


# ---------------------------------------------------------------------------
# what it copies, and what it deliberately does not
# ---------------------------------------------------------------------------
def test_it_stages_the_scope_and_writes_a_manifest(tmp_path):
    _, scope = _scope(tmp_path)
    result = snap.build_snapshot(tmp_path / "stage", scope=scope, snapshot_date="2026-08-22")
    assert (result.staging / "manifest.json").exists()
    m = json.loads((result.staging / "manifest.json").read_text(encoding="utf-8"))
    assert m["schema"] == snap.MANIFEST_SCHEMA
    assert m["snapshot_date"] == "2026-08-22"
    assert m["files"] == 2
    assert all(e["sha256"] for e in m["entries"] if not e.get("skipped"))


def test_home_root_takes_files_but_never_descends(tmp_path):
    """`output/` belongs to the cold push. Two jobs, two scopes."""
    _, scope = _scope(tmp_path)
    result = snap.build_snapshot(tmp_path / "stage", scope=scope, snapshot_date="2026-08-22")
    staged = {r.relative for r in result.copied}
    assert "trader_annotations.jsonl" in staged
    assert not any("report.txt" in name for name in staged)


def test_it_copies_and_never_moves(tmp_path):
    """Decision 0015: hot files stay on the local SSD."""
    src, scope = _scope(tmp_path)
    snap.build_snapshot(tmp_path / "stage", scope=scope, snapshot_date="2026-08-22")
    assert (src / "data" / "runtime" / "outcomes.csv").exists()
    assert (src / "trader_annotations.jsonl").exists()


# ---------------------------------------------------------------------------
# copy-while-hot
# ---------------------------------------------------------------------------
def test_a_big_file_is_compressed(tmp_path):
    src = tmp_path / "home"
    src.mkdir()
    big = src / "tracker.json"
    big.write_text("{}" + " " * 5000, encoding="utf-8")
    scope = [snap.ScopeItem("home-root", src, "files")]
    result = snap.build_snapshot(
        tmp_path / "stage", scope=scope, snapshot_date="2026-08-22",
        compress_min_bytes=1000, stability_min_bytes=10**9,
    )
    record = result.copied[0]
    assert record.compressed and record.method == "gzip"
    stored = result.staging / "home-root" / "tracker.json.gz"
    assert stored.exists()
    assert record.stored_bytes < record.source_bytes, "compression must actually shrink it"
    with gzip.open(stored, "rt", encoding="utf-8") as handle:
        assert handle.read().startswith("{}")


def test_an_unstable_big_file_is_skipped_with_a_reason_not_silently(tmp_path):
    """A ~1 GB atomic replace caught mid-write restores to nothing, and a torn
    JSON is indistinguishable from a good one until the day you need it."""
    src = tmp_path / "home"
    src.mkdir()
    target = src / "tracker.json"
    target.write_text("x" * 4000, encoding="utf-8")

    def _grow(_seconds):
        target.write_text("x" * 9000, encoding="utf-8")

    scope = [snap.ScopeItem("home-root", src, "files")]
    result = snap.build_snapshot(
        tmp_path / "stage", scope=scope, snapshot_date="2026-08-22",
        stability_min_bytes=1000, sleep=_grow,
    )
    assert result.copied == []
    assert [r.skipped for r in result.skipped] == ["unstable_during_snapshot"]
    m = json.loads((result.staging / "manifest.json").read_text(encoding="utf-8"))
    assert m["skipped"] == 1
    assert m["skipped_by_reason"] == {"unstable_during_snapshot": 1}


def test_a_stable_big_file_is_taken(tmp_path):
    src = tmp_path / "home"
    src.mkdir()
    (src / "tracker.json").write_text("x" * 4000, encoding="utf-8")
    scope = [snap.ScopeItem("home-root", src, "files")]
    result = snap.build_snapshot(
        tmp_path / "stage", scope=scope, snapshot_date="2026-08-22",
        stability_min_bytes=1000, compress_min_bytes=10**9, sleep=lambda _s: None,
    )
    assert len(result.copied) == 1 and not result.skipped


def test_a_live_sqlite_is_copied_through_the_backup_api(tmp_path):
    """A byte copy can catch an open database mid-transaction, and the journal
    is written to every night by the AI runner."""
    src = tmp_path / "home"
    src.mkdir()
    db = src / "trade_journal.sqlite3"
    con = sqlite3.connect(db)
    con.execute("create table t (a int)")
    con.execute("insert into t values (7)")
    con.commit()
    try:
        scope = [snap.ScopeItem("home-root", src, "files")]
        result = snap.build_snapshot(tmp_path / "stage", scope=scope, snapshot_date="2026-08-22")
    finally:
        con.close()
    record = next(r for r in result.copied if r.relative.endswith(".sqlite3"))
    assert record.method == "sqlite_backup_api"
    copied = sqlite3.connect(result.staging / "home-root" / "trade_journal.sqlite3")
    try:
        assert copied.execute("select a from t").fetchone()[0] == 7
    finally:
        copied.close()


def test_one_unreadable_file_never_costs_the_whole_snapshot(tmp_path):
    src = tmp_path / "home"
    src.mkdir()
    (src / "good.txt").write_text("ok", encoding="utf-8")
    (src / "bad.txt").write_text("x", encoding="utf-8")
    scope = [snap.ScopeItem("home-root", src, "files")]
    real = snap._copy_one

    def _explode(source, target, *, compress):
        if source.name == "bad.txt":
            raise OSError("locked")
        return real(source, target, compress=compress)

    snap._copy_one, saved = _explode, real
    try:
        result = snap.build_snapshot(tmp_path / "stage", scope=scope, snapshot_date="2026-08-22")
    finally:
        snap._copy_one = saved
    assert [r.relative for r in result.copied] == ["good.txt"]
    assert result.skipped[0].skipped.startswith("copy_failed")


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------
def test_retention_keeps_seven_daily_four_weekly_twelve_monthly():
    from datetime import date, timedelta

    start = date(2026, 8, 22)
    dates = [(start - timedelta(days=i)).isoformat() for i in range(400)]
    pruned = set(snap.snapshots_to_prune(dates))
    kept = [d for d in dates if d not in pruned]
    # The seven most recent are always kept.
    assert dates[:7] == sorted(dates[:7], reverse=True)
    for day in dates[:7]:
        assert day not in pruned
    # And the total kept is bounded, not the whole 400.
    assert len(kept) <= snap.KEEP_DAILY + snap.KEEP_WEEKLY + snap.KEEP_MONTHLY
    assert len(pruned) > 350


def test_retention_is_a_pure_function_of_the_dates():
    dates = ["2026-08-22", "2026-08-21", "2026-08-20"]
    assert snap.snapshots_to_prune(dates) == snap.snapshots_to_prune(list(reversed(dates)))
    assert snap.snapshots_to_prune(dates) == []


def test_pruning_never_touches_the_frozen_directory(tmp_path):
    root = tmp_path / "snaps"
    (root / snap.FROZEN_DIR_NAME).mkdir(parents=True)
    for i in range(20):
        from datetime import date, timedelta
        (root / (date(2026, 8, 22) - timedelta(days=i)).isoformat()).mkdir()
    snap.prune(root)
    assert (root / snap.FROZEN_DIR_NAME).exists(), "frozen audits are permanent"


# ---------------------------------------------------------------------------
# verify and restore
# ---------------------------------------------------------------------------
def test_verify_confirms_a_good_snapshot_and_catches_a_corrupted_one(tmp_path):
    _, scope = _scope(tmp_path)
    result = snap.build_snapshot(tmp_path / "stage", scope=scope, snapshot_date="2026-08-22")
    assert snap.verify(result.staging)["ok"] is True
    (result.staging / "home-root" / "trader_annotations.jsonl").write_text("tampered", encoding="utf-8")
    bad = snap.verify(result.staging)
    assert bad["ok"] is False and bad["mismatched"] == 1


def test_restore_round_trips_into_a_scratch_directory(tmp_path):
    src, scope = _scope(tmp_path)
    result = snap.build_snapshot(
        tmp_path / "stage", scope=scope, snapshot_date="2026-08-22", compress_min_bytes=1
    )
    scratch = tmp_path / "scratch"
    assert snap.restore(result.staging, scratch, dry_run=True)["would_restore"] == 2
    assert not scratch.exists(), "a dry run writes nothing"
    snap.restore(result.staging, scratch, dry_run=False)
    assert (scratch / "data-runtime" / "outcomes.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_restore_refuses_to_write_into_the_live_store(tmp_path):
    """A drill that overwrites live state is how a drill becomes an incident."""
    from project_paths import PERSISTENT_DATA_DIR

    _, scope = _scope(tmp_path)
    result = snap.build_snapshot(tmp_path / "stage", scope=scope, snapshot_date="2026-08-22")
    with pytest.raises(ValueError, match="live store"):
        snap.restore(result.staging, Path(PERSISTENT_DATA_DIR), dry_run=False)
    with pytest.raises(ValueError, match="live store"):
        snap.restore(result.staging, Path(PERSISTENT_DATA_DIR) / "data" / "runtime", dry_run=False)


# ---------------------------------------------------------------------------
# health tile
# ---------------------------------------------------------------------------
def test_health_reports_the_latest_snapshot_and_survives_an_empty_root(tmp_path):
    assert snap.health(tmp_path / "nothing")["files"] == 0
    _, scope = _scope(tmp_path)
    snap.build_snapshot(tmp_path / "stage", scope=scope, snapshot_date="2026-08-22")
    h = snap.health(tmp_path / "stage", das_root=tmp_path / "no-such-share")
    assert h["last_snapshot_date"] == "2026-08-22"
    assert h["files"] == 2 and h["source_bytes"] > 0
    assert h["das_reachable"] is False


def test_the_module_reaches_no_live_decision_surface():
    source = (SCRIPTS_DIR / "ops" / "evidence_snapshot.py").read_text(encoding="utf-8")
    for forbidden in ("review_policy", "focus_service", "record_review_event", "ibapi", "add_alert"):
        assert forbidden not in source, forbidden


# ---------------------------------------------------------------------------
# the shipped scripts
# ---------------------------------------------------------------------------
def test_the_ops_scripts_are_versioned_here_and_installed_there():
    r"""`push_cold_to_das.ps1` lived ONLY in C:\TradingBotData\_tools until R10.A.

    The script that protects the evidence was itself unversioned, untested and
    unreviewable. The repo copy is now the source of truth; `_tools` holds an
    installed copy. If they drift, the installed one is running code nobody
    reviewed - so this compares them byte for byte when both exist.
    """
    installed_root = Path("C:/TradingBotData/_tools")

    def _content(path: Path) -> str:
        # Content, not bytes: git normalises line endings on checkout, so a
        # byte comparison would fail on a clean clone for a reason that has
        # nothing to do with anyone editing the running script.
        return path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip()

    for name in ("push_cold_to_das.ps1", "snapshot_to_das.ps1", "restore_from_das.ps1"):
        repo = SCRIPTS_DIR / "ops" / name
        assert repo.exists(), f"{name} must be versioned in the repo"
        installed = installed_root / name
        if not installed.exists():
            continue  # a fresh checkout, or a machine without the home folder
        assert _content(repo) == _content(installed), (
            f"{name} differs between the repo and _tools; the installed copy is "
            "running code nobody reviewed"
        )


def test_the_two_backup_jobs_declare_that_they_are_not_each_other():
    """Two jobs, two scopes. The next reader must not merge them."""
    cold = (SCRIPTS_DIR / "ops" / "push_cold_to_das.ps1").read_text(encoding="utf-8")
    snapshot = (SCRIPTS_DIR / "ops" / "snapshot_to_das.ps1").read_text(encoding="utf-8")
    for text in (cold, snapshot):
        assert "TWO JOBS, TWO SCOPES" in text
    assert "snapshot_to_das.ps1" in cold, "the cold push must point at the snapshot"
    assert "push_cold_to_das.ps1" in snapshot, "and the snapshot back at the cold push"


def test_the_cold_push_carries_the_ledger_directory_and_no_hot_state():
    """R10's month-segmented ledgers are append-only, which is the cold push's
    shape. Everything else in data\runtime is hot and belongs to the snapshot."""
    cold = (SCRIPTS_DIR / "ops" / "push_cold_to_das.ps1").read_text(encoding="utf-8")
    assert r"data\runtime\evidence_ledgers" in cold
    for hot in ("master_avwap_setup_tracker", "intraday_bounce_outcomes", "trade_journal"):
        assert hot not in cold, f"{hot} is hot state and must not be in the cold push"


def test_a_real_restore_records_itself_and_a_dry_run_does_not(tmp_path):
    """A backup nobody has restored is a hypothesis - so the drill is recorded.

    A dry run must NOT count: it proved nothing about the bytes.
    """
    staging = tmp_path / "stage"
    _, scope = _scope(tmp_path)
    result = snap.build_snapshot(staging, scope=scope, snapshot_date="2026-08-22")
    marker = staging / "last_restore_test.json"

    assert snap.health(staging)["last_restore_test"] == ""
    snap.record_restore_test(staging, snapshot_date="2026-08-22", restored=2)
    assert marker.exists()
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    assert recorded["files_restored"] == 2 and recorded["snapshot_date"] == "2026-08-22"
    assert snap.health(staging)["last_restore_test"].startswith("20")
    assert snap.restore(result.staging, tmp_path / "scratch", dry_run=True)["dry_run"] is True


# ---------------------------------------------------------------------------
# R10.A §2: the .bak exclusion and source_sha256
# ---------------------------------------------------------------------------
def test_the_rotated_tracker_bak_is_excluded_with_a_reason(tmp_path):
    """Trader, 2026-08-22: exclude it.

    The tracker rotates its .bak on every save, so once the snapshot runs
    nightly, day N's main IS day N+1's .bak - 133 MB a night of the same bytes
    under a different name. Excluded by an explicit rule carrying a reason, never
    by a silent skip, and the on-disk .bak is never deleted: the tracker reads it
    when the main is corrupt.
    """
    src = tmp_path / "home"
    src.mkdir()
    (src / "master_avwap_setup_tracker.json").write_text("{}", encoding="utf-8")
    (src / "master_avwap_setup_tracker.json.bak").write_text("{}", encoding="utf-8")
    (src / "keep_me.json").write_text("{}", encoding="utf-8")
    scope = [snap.ScopeItem("data-runtime", src, "files")]
    result = snap.build_snapshot(tmp_path / "stage", scope=scope, snapshot_date="2026-08-22")

    copied = {r.relative for r in result.copied}
    assert "master_avwap_setup_tracker.json" in copied
    assert "keep_me.json" in copied
    assert "master_avwap_setup_tracker.json.bak" not in copied

    excluded = [r for r in result.skipped if r.relative.endswith(".bak")]
    assert len(excluded) == 1
    assert excluded[0].skipped == "excluded_rotated_duplicate"
    m = json.loads((result.staging / "manifest.json").read_text(encoding="utf-8"))
    assert m["skipped_by_reason"] == {"excluded_rotated_duplicate": 1}
    # And the source file is untouched - the tracker still needs it on disk.
    assert (src / "master_avwap_setup_tracker.json.bak").exists()


def test_the_exclusion_can_be_overridden_for_a_deliberate_freeze(tmp_path):
    """§0's frozen pair is the one-time exception, so the rule takes a switch."""
    src = tmp_path / "home"
    src.mkdir()
    (src / "master_avwap_setup_tracker.json.bak").write_text("{}", encoding="utf-8")
    scope = [snap.ScopeItem("data-runtime", src, "files")]
    result = snap.build_snapshot(
        tmp_path / "stage", scope=scope, snapshot_date="2026-08-22", exclude_rotated=False
    )
    assert [r.relative for r in result.copied] == ["master_avwap_setup_tracker.json.bak"]


def test_the_manifest_carries_the_source_hash_beside_the_stored_hash(tmp_path):
    """The stored hash proves the archive; only the source hash proves the
    CONTENT survived compression. `verify()` stays on stored bytes so it is cheap."""
    src = tmp_path / "home"
    src.mkdir()
    (src / "big.json").write_text("{}" + " " * 5000, encoding="utf-8")
    scope = [snap.ScopeItem("home-root", src, "files")]
    result = snap.build_snapshot(
        tmp_path / "stage", scope=scope, snapshot_date="2026-08-22", compress_min_bytes=1000
    )
    record = result.copied[0]
    assert record.compressed
    assert record.source_sha256 and record.sha256
    assert record.source_sha256 != record.sha256, "compressed bytes differ from source bytes"

    import hashlib
    expected = hashlib.sha256((src / "big.json").read_bytes()).hexdigest()
    assert record.source_sha256 == expected

    entry = json.loads((result.staging / "manifest.json").read_text(encoding="utf-8"))["entries"][0]
    assert entry["source_sha256"] == expected
    # Round-tripping the archive reproduces the source hash.
    stored = result.staging / "home-root" / "big.json.gz"
    with gzip.open(stored, "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == expected


def test_the_scheduled_task_is_versioned_and_runs_outside_market_hours():
    """The task XML lives in the repo like the scripts it launches.

    Timing is a real constraint, not a preference: the snapshot must land after
    the close and before the AI runner's 22:00 window, and never inside the
    06:00-14:00 band where the launch task fires every 15 minutes and the desk
    is scanning.
    """
    import re
    import xml.etree.ElementTree as ET

    task = SCRIPTS_DIR / "ops" / "TradingBotV3 - Evidence snapshot.xml"
    assert task.exists(), "the task definition must be versioned beside the scripts"
    text = task.read_text(encoding="utf-8-sig")
    assert "snapshot_to_das.ps1" in text

    root = ET.fromstring(text)
    ns = {"t": root.tag.split("}")[0].strip("{")}
    starts = [e.text for e in root.findall(".//t:CalendarTrigger/t:StartBoundary", ns)]
    assert starts, "the task must carry a daily trigger"
    hour, minute = (int(x) for x in re.search(r"T(\d{2}):(\d{2})", starts[0]).groups())
    minutes = hour * 60 + minute
    assert minutes > 13 * 60, "must run after the 13:00 PT close"
    assert minutes < 22 * 60, "must run before the AI runner's 22:00 window opens"
    assert not (6 * 60 <= minutes < 14 * 60), "must never run inside market hours"
