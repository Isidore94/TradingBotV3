# 0017 — The operational stores move to record stores, tracker first, shadow first

Date: 2026-09-04

Amends nothing in decisions 0001–0016; 0005 (plain-file home-folder storage)
and 0015 (no cloud sync, the DAS is the durable tier) still bind. This record
narrows 0005: a plain file is still the unit the trader can copy, back up and
restore, but the unit of WRITE and READ inside it stops being "the whole thing".

## Context

The assessment of 2026-09-03 ("Where the Desk's Time Goes", packet F3) measured
the operational tier as it stood:

| Store | Size | Written | Read |
|---|---|---|---|
| `master_avwap_setup_tracker.json` (+ `.bak`) | 1.15 GB (+1.13) | once a day, whole, 1.15 GB atomic replace | whole, by every reader that wants one symbol |
| `master_avwap_setup_attributes.csv` | 615 MB | daily, whole | never by the desk |
| `d1_features_history.csv` | 592 MB | appended per scan | whole |
| `intraday_bounce_outcomes.csv` / `candidates.csv` | 279 / 309 MB | continuously | whole, to answer a question about today |

The tracker is 11,334 setups, 3,992 study setups and 396 control setups at
~76 KB each (`scenarios` 26 KB, `feature_row` 5 KB, `entry_attributes` 3 KB).
The 2026-08-31 journal freeze was a reader loading all of it to answer one
click; the 2026-08-31 health-audit stall was a reader parsing 269 MB of
outcomes every 15 seconds. Both were fixed one reader at a time; the shape
that produces them was not.

## Decision

1. **The tracker becomes a record store: SQLite, one row per tracker record**
   (`scripts/tracker_store.py`, `master_avwap_setup_tracker.sqlite` beside the
   JSON, WAL mode). A save rewrites only the records whose content hash changed;
   a read can narrow by section, symbol and scan date without loading the rest.
2. **Step 1 is shadow only, and it is what landed on 2026-09-04.** The scanner
   still loads the JSON and still writes it first; the mirror runs after the
   JSON save behind the `tracker_storage_shadow` local setting (default ON), and
   a mirror failure is a warning, never a failed save. No reader changes. The
   payload is copied, not interpreted: no detector, scoring or tracker logic
   moves, so decision 0009's golden-fixture rule is not engaged by this step.
3. **Step 2 - readers move, then the JSON retires - is gated on parity**:
   `python scripts/tracker_store.py verify` must report zero differences on at
   least five consecutive live saves. Readers move one at a time, narrowest read
   first (the Setup Tracker panel's per-symbol views, the journal's `took` join,
   `held_run_score`'s `d1_setup_present`), each behind its own fail-before-fix
   test. The scanner's own load is last, and it keeps the JSON `.bak` recovery
   rule (an empty or corrupt store never wipes the Expected-R history).
4. **The CSV stores follow the same path only after the tracker's step 2 is
   live**: monthly Parquet for the append-only histories (the research lake
   already proves the shape), SQLite for the ones readers query. Each is its own
   packet with its own fixtures.

## Rationale

A whole-document JSON file couples every reader's memory to the file's size and
every writer's latency to the file's size, and the file grows with every scan.
A record store breaks both couplings without changing what is stored, which is
why the first step can be shadow-only and measured rather than argued.

## Consequences

- `scripts/tracker_store.py` and `MASTER_AVWAP_SETUP_TRACKER_DB` exist;
  `save_setup_tracker_payload` calls `mirror_payload` after `save_json`.
- The first live mirror happens at the next tracker save (the 13:00 PT close
  slot on a desk day); gate #57 reads its parity.
- The `.bak` rotation, the empty-payload refusal and the scoring snapshot are
  untouched.
- **Not decided here**: the SQLite schema for the CSV stores, and whether the
  frozen exe bundles a different SQLite than the venv's (it is stdlib; the
  selftest will say).
