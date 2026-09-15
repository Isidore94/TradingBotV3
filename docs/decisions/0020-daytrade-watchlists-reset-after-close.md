# 0020 — The day-trade watchlists are wiped after each session's close

Date: 2026-09-15

Amends plan.md section 5's "user-entered watchlist names are never automatically
removed" for exactly two files, `longs.txt` and `shorts.txt`. Decisions 0001-0019
are untouched.

## Context

The trader, 2026-09-15: *"i want all names wiped at the end of the day. its a
daytrade watchlist not a permanent one."* `longs.txt` / `shorts.txt` are the
intraday (M5) lists BounceBot scans; nothing on them is meant for tomorrow. Until
now they carried over indefinitely and only the machine's own slice moved.

The invariant was written against a machine JUDGEMENT about one name: an
auto-populate rotation or a cross-machine writer dropping a name the trader typed.
A session boundary that empties the whole list is a different act.

## Decision

- `longs.txt` and `shorts.txt` are emptied WHOLE after each exchange session's
  close, every name alike (`scripts/daytrade_watchlist_reset.py`). A list is due
  when it holds names and was last written at or before the last completed
  session's close; a name typed after the close is the next session's and stays.
- The file is written first (atomic, designated-writer gated) and one WS-5D
  `remove` row per name follows with the source `session_reset`; a refused write
  records nothing.
- The swing lists (`swinglongs.txt`, `shortswings.txt`) keep the invariant
  unchanged. No writer may remove one user-entered name by its own judgement on
  any list; the reset never picks a name.
- The `local_settings` switch `daytrade_watchlists_reset` (default ON) turns the
  reset off.

## Consequences

- A day-trade name the trader wants back tomorrow is typed again, or typed after
  the close.
- The M5 Focus picks are NOT reset with the lists (lead decision; they fade on
  their own ten-session clock and are scanned through the fast lane regardless).
  The trader may overrule.
- Live proof is gate #123 in `CURRENT_CHECKPOINT.md`. Long form:
  `docs/DESK_INTERNALS.md` "DTR".
