# 0015 — No cloud sync; the DAS file server is the storage tier

Date: 2026-08-10

Supersedes the cloud-sync premise of decision 0005, and amends the Drive-based
rationale in decisions 0006 and 0014. Those records stay in place unedited: the
mechanisms they justify (writer-lease fencing, the lake path guard) still exist
in the code, and deleting their rationale would leave that code unexplained.

## Context
The build was designed around a cloud-synced "home folder" — Google Drive or
OneDrive — as the shared operational store. Two facts have since changed:

1. **There is no cloud drive anymore.** The trader removed it entirely.
2. **The desk is a single do-everything machine** (the always-on 8845HS
   mini-PC), backed by a DAS file server at `\\MINI-PC\Trading Bot Data` that
   can be expanded to ~100 TB.

Multi-device sharing through a sync client was the original reason for the
cloud folder. With one machine and a file server, that reason is gone, and
the sync client's costs — quota ceilings, sync latency, DriveFS wedges, and
the absence of atomic test-and-set across machines — bought nothing.

## Decision
1. **No cloud sync anywhere in the system.** Neither Drive nor OneDrive is
   part of the storage design, the setup instructions, or the failure modes.
2. **`C:\TradingBotData` keeps its path and its role**, now as a plain local
   folder on the desk's SSD: compact operational state — watchlists, reports,
   JSONL/CSV evidence, tracker state. It remains the "shared home folder" in
   code and configuration; only the "cloud-synced" description was wrong.
3. **The DAS at `\\MINI-PC\Trading Bot Data` is the durable storage tier.** It
   holds the research lake (`research_lake/`), the AI store (`ai_store/`), and
   the cold subtrees pushed hourly by `_tools/push_cold_to_das.ps1`.
4. **Local-first, then DAS.** Large or high-churn writes land on local disk and
   move to the DAS afterwards, so a file-server outage degrades throughput
   rather than correctness. The research warehouse's machine-local spool
   (`%LOCALAPPDATA%\TradingBotV3\research_spool`) is the built-in instance of
   this pattern.
5. **The lake still may not live inside `C:\TradingBotData`.** The guard in
   `research_warehouse/config.py` stays exactly as written. Its *reason* is now
   storage-class separation and the cold-push scope, not sync quota: the home
   folder is for compact operational state, and the push script mirrors its
   subtrees wholesale.
6. **Writer-lease fencing stays.** See Consequences.

## Rationale
Removing the sync client removes a whole class of failure the design spent real
complexity defending against, and the DAS answers the only requirement the cloud
folder was actually meeting — durable capacity — with far more headroom.

Keeping `C:\TradingBotData` where it is makes this a documentation correction
rather than a migration: no path changes, no data movement, no code behavior
change, and nothing to re-point during a live testing week.

## Consequences
- **Writer-lease fencing (0006) is retained even though the race it was built
  for cannot occur with one machine.** It is a cheap, already-tested guard, and
  it earned its place on 2026-08-10: it correctly refused to publish
  `autopilot_today.txt` when no designated writer was configured after the old
  desktop was retired, preserving the last verified report instead of writing a
  report from an unconfigured machine. The single-machine topology is a fact
  about today, not an invariant — a second machine must not silently become a
  writer if one is ever added.
- **The "never in Drive" backup rule (0014 sec 8.5) needs restating, not
  deleting.** Class B was "a second physical disk, never Drive." The DAS *is*
  the second physical disk relative to the desk SSD, so it satisfies Class B —
  but a lake and its only backup on the same server is not a backup. An
  off-server Class A destination is still owed.
- **Off-site copy is now an open question.** Cloud sync was providing incidental
  off-site redundancy for the small Class A set. Nothing replaces it yet. This
  is a real gap, tracked as such rather than assumed solved.
- macOS support keeps its CloudStorage mount-discovery code, which is now dead
  on this desk. It is harmless and stays until a macOS run is actually needed.
  **Amended 2026-08-15 — see below: "discovery" was not in fact harmless.**

## Amendment — 2026-08-15 (packet R1, `plan.md` Phase 0.5)

Building R1 surfaced a distinction this record had collapsed. "CloudStorage
mount-discovery code" was two mechanisms, not one, and only the second was
harmless:

1. **Store selection** — `_default_google_drive_shared_dir()` probed
   `$GOOGLE_DRIVE`, `~/My Drive`, `~/Google Drive`, and every
   `~/Library/CloudStorage/GoogleDrive-*` account, and returned the first
   writable one as `PERSISTENT_DATA_DIR` with source `google_drive_default`.
   That ran **at import**, on Windows as well as macOS. It was inert on this
   desk only because `shared_data_dir` happens to be set; had that setting ever
   been lost — exactly what happened to three other settings when the old
   desktop was retired — the app would have silently adopted a sync folder as
   its operational store. That directly contradicts Decision 1 ("no cloud sync
   anywhere in the system") and was never the harmless part.

   **Removed.** With no configured store, the fallback is now
   `LOCAL_SETTINGS_DIR` (source `default_local`). Two tests in
   `tests/test_project_paths.py` were inverted to assert a mounted Drive and a
   CloudStorage account are ignored.

2. **Mount presence** — `_unmounted_shared_anchor()` / the bounded startup wait
   check that the *configured* store's mount exists before `mkdir` runs. This
   is what the original consequence meant to keep, and it is genuinely useful:
   creating the store on a missing mount forks it into a plain local folder
   that silently shadows the real one.

   **Kept**, including the macOS CloudStorage branch. Renamed
   `_wait_for_shared_drive` → `_wait_for_shared_store`, and its messages no
   longer instruct the operator to start GoogleDriveFS. A local home folder
   returns immediately, so it costs the desk nothing; it earns its place only
   if the store is ever pointed at a network or removable path.

No path moved: the desk still resolves `C:\TradingBotData` from `local_config`.

## Status
Accepted. Supersedes the cloud-sync premise of 0005; amends 0006 and 0014.
Amended 2026-08-15 (Consequences, cloud-drive discovery removed).
