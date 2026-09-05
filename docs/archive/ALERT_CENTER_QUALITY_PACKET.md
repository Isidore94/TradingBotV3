# Alert Center quality packet (P1.6)

Armed by the trader: **2026-08-14** (WISHLIST promotion + chat clarifications the same
day). Roadmap slot: `plan.md` P1.6. This document is the build contract; `plan.md`
owns order and gates. Origin: WISHLIST.md "Trader-entered ideas — 2026-08-14" plus
the trader's spoken clarifications recorded in section 2.

**Builder notice — ask-first fence.** `scripts/ui/panels/alert_center_panel.py` and
`scripts/ui/widgets/alert_chart_review.py` host alert code. Per the file-scoped
ask-first rule (checkpoint review 2026-08-08), show the trader the intended edit plan
for those files BEFORE editing them, even though every change here is display/verb
side. No detector, scoring, or evidence-capture behavior changes in this packet.

## 1. Why this packet exists

Trader outcomes, in the trader's own framing:

1. "If I dislike an auto picked stock I can take it off quickly and not see it
   again." — one click removes an auto-adopted pick from M5 Focus and from today's
   feed.
2. "If I like a stock I can add it to m5 focus picks. Then I get flagged on
   pullbacks." — liking from the Alert screen must be obvious and labeled; the
   existing ★ semantics are already correct.
3. "I don't want to be constantly seeing the same stocks over and over ... less spam
   and more quality ... I basically don't want to see the same ticker over and over
   again. It def finds bangers though." — repetition control in the feed, not weaker
   detection.

## 2. Trader clarification 2026-08-14 — armed alerts survive "Not today"

> "'Not today' should still trigger on the alerts I set."

This is a **general rule for every "✕ Not today" path**, not just the new auto-pick
decline: dismissing a name for the day must never disarm anything the trader armed —
chart watches, D1 event watches, armed D1 level alerts, and price alerts. Only the
explicit disarm toggles (arm dock, armed list) remove them.

Two places violate this today and must change:

- `_ignore_alert_symbol` (`alert_center_panel.py`) deletes the symbol's
  `_chart_watches`. Stop doing that.
- `add_alert` returns early for any symbol in `_ignored_symbols`. Exempt user-armed
  hits: alerts with `CHART_WATCH_TAG` (includes armed D1 level fires and D1 event
  watch hits routed through `_chart_watch_alert`) must pass the ignored-symbol
  filter, land in the feed, and sound as they do now. `PriceAlertService` delivery
  is a separate path and is untouched.

What correctly still lapses on a Focus removal: the *automatic* D1 interest flags a
name gets only because it is a Focus pick (`FOCUS_D1_EVENT_TAG` sources). Those are
Focus-derived, not trader-armed; they follow Focus membership.

## 3. Packet A — "✕ Not today" declines an auto-adopted M5 Focus pick

### Current defect

Since 2026-08-05, staged auto-populate picks land straight in M5 Focus
(`_adopt_auto_pick_into_focus`), injecting `longs.txt`/`shorts.txt` lines. But "✕
Not today" on that name from the live feed only hides it from the Alert Center: it
stays in M5 Focus, stays injected, keeps scanning, keeps skipping the tier gate and
sounding. Only the Focus board's Review ▶ actually prunes it. One verb, three
meanings, and the auto-pick meaning is the lie.

### A1. Persist auto/manual provenance in the Focus store

- Extend `FocusPickStore` membership entries (`scripts/focus_picks.py`) with
  `origin: "auto" | "manual"` and the market date of the add. Missing field on an
  existing entry reads as `"manual"` (backward compatible; no migration).
- `_adopt_auto_pick_into_focus` writes `origin="auto"`. Every trader-initiated add
  (★ like, snapshot-dialog buttons, Focus board add/paste, chart-review Add verb)
  writes `origin="manual"`.
- **Upgrade rule:** a trader-initiated add/like of a symbol that already sits in m5
  Focus with `origin="auto"` upgrades the entry to `"manual"`. Once the trader has
  claimed a name, "Not today" no longer deletes it from Focus (falls back to the
  scanner-alert behavior in A2 step 4).
- Provenance must survive a desk restart; the in-memory `_auto_picks_enqueued` set
  is not a substitute. The store remains the single writer of its files.

### A2. Decline flow — what one "✕ Not today" click does

On "✕ Not today" (feed-row ✕ or chart-review verb) for symbol S:

1. If S is in **m5** Focus with `origin="auto"` and today's date:
   `store.remove(S, side, "m5")` for each auto-origin m5 side entry — never
   `remove_everywhere`. This un-injects the watchlist line, so BounceBot stops
   alerting S. Swing entries and manual entries are untouchable by this path.
2. Remove S from the feed, D1 feed, chart queue, and review guidance, and add S to
   `_ignored_symbols` for today — subject to the section 2 rule: armed watches stay
   armed and their hits still surface.
3. Record the decline so no auto-populate cycle re-proposes or re-adopts S today
   (write the decline into the auto-populate decision log alongside the existing
   `resolve_auto_populate_pick` records; the existing already-resolved state plus
   the persisted decline must hold across a desk restart).
4. If S is not an auto-origin m5 pick, behavior is today's, minus the section 2
   violations: Focus-walkthrough dismiss keeps its `remove_everywhere`; auto-pick
   *proposal* decline (fallback queue) keeps its verdict path; ordinary scanner
   alert keeps feed-ignore-only.

### A3. Logging — a decline is not a trader dislike

- Do **not** write a `pick_feedback.jsonl` "unfavorite"/dislike for an auto-pick
  decline. The machine picked it; the trader declining it is a verdict on the auto
  pick, not on the stock.
- Record a distinct review event `auto_pick_decline` (symbol, side, dwell,
  queue_len, the pick's staged reason/score) so the learning loop can measure which
  auto picks the trader rejects. This is additive evidence only.

## 4. Packet B — labeled Like from the Alert screen

Semantics are already correct (`_toggle_favorite`: files into M5/Swing Focus by
alert type via `favorite_category_for_alert`, logs a like, Focus names skip the tier
gate and sound). This packet is discoverability and parity only:

- B1. Promote the edge ★ glyph on feed rows (`alert_feed_item.py`) to a labeled
  button sized like the ✕ — e.g. "★ Like → M5 Focus" / "★ Like → Swing Focus" per
  the alert's category; lit state keeps the unfavorite affordance.
- B2. Add the same labeled Like (and existing ✕ Dislike) to the chart-review verb
  area on the Trading Desk so the Alert screen has full parity with Chart Review's
  labeled buttons. The unified three-verb row layout rule (2026-07-31: same buttons,
  same spots, no shifting layouts) still governs — additions go in the dock/edge,
  not by reordering the three verbs.
- B3. No semantic changes: same `FocusService.add` call, same origin/context logging,
  and Packet A's `origin="manual"` stamp.

## 5. Packet C — repetition and open-burst control (display-only)

Main **Alerts feed only**; the D1 Focus feed is already curated and is out of scope.
This is the early display-side slice of the P5.1 typed/deduplicated ladder and is
superseded when P5.1's challenger passes its manifest. All of it is feed
presentation: detection, scoring, every JSONL/evidence capture, review events, and
the AWAY push policy are byte-identical with the feature on or off.

- C1. **One live row per symbol+side per day.** A repeat updates the existing row in
  place — latest trigger text, a repeat-count badge ("×4"), original first-seen
  time retained. The row re-floats to the top and re-sounds **only on escalation**.
- C2. **Escalation** = any of: tier strictly above the best tier shown for that
  symbol+side today; first BANGER stamp; first PROVEN stamp. Focus-privileged names
  and user-armed hits keep today's always-surface/always-sound behavior and are
  never collapsed silently into a stale row. Non-escalating repeats tick the badge
  with no sound and no re-float.
- C3. **Open-window digest.** For the first N minutes of the session (default 30,
  user-configurable in Settings, 0 disables), non-breakthrough alerts roll into one
  ranked digest row per scan cycle instead of individual rows. **Breakthrough list
  (always immediate, never digested):** BANGER, PROVEN, Focus-privileged names,
  user-armed hits (chart watches / D1 levels / D1 event watches), entry-assist
  output, and ready D1 alerts. At window end, collapsed C1 rows resume normally;
  digest contents remain reachable (expandable row or history), not discarded.
- C4. This is a deliberate, user-owned display preference with a visible setting.
  It is not AI suppression and adds no suppression field anywhere;
  `review_policy.json` is untouched.

## 6. Invariants checklist (plan.md sec 5)

- No detector/scoring behavior change → no golden fixtures triggered; capture,
  review, and warehouse evidence unchanged.
- User-entered names never auto-removed: Packet A deletes only machine-added
  (`origin="auto"`) m5 entries, on an explicit trader click.
- Single writer per file: provenance lives in the store's own membership sidecar;
  `alert_center_panel` still writes Focus only through the store/service.
- `review_policy.json` untouched; no suppression field anywhere.
- Completed-bars, σ-formula, point-in-time rules: not in scope, not touched.

## 7. Touch points

| File | Change |
|---|---|
| `scripts/focus_picks.py` | Membership `origin` + date; upgrade rule (A1) |
| `scripts/ui/panels/alert_center_panel.py` | Decline flow, ignored-filter exemption, watch preservation, C1–C3 feed behavior (**ask first**) |
| `scripts/ui/widgets/alert_chart_review.py` | Verb wiring for decline + Like parity (**ask first**) |
| `scripts/ui/widgets/alert_feed_item.py` | Labeled Like button, repeat badge |
| `scripts/ui/panels/settings_panel.py` (or equivalent) | Digest window setting |
| `tests/test_focus_picks.py`, `tests/test_qt_alert_center.py` | Section 8 |

## 8. Test plan (additions, all offline/deterministic)

1. Store: auto-origin add persists across store reload; missing origin reads
   manual; trader add upgrades auto → manual; scoped m5 auto removal never touches
   swing or manual entries and un-injects only Focus-owned watchlist lines.
2. Decline: "Not today" on an auto-adopted pick removes the m5 entry + watchlist
   line, ignores the symbol for today, and survives restart without re-adoption;
   on a manual/liked name it does not remove Focus entries.
3. Armed survival (the section 2 rule): armed chart watch + D1 event watch + armed
   level survive a "Not today" on the same symbol; a subsequent watch hit passes
   the ignored-symbol filter, renders, and sounds.
4. Feedback hygiene: decline writes `auto_pick_decline` review event and no
   pick-feedback unfavorite; ★ like still writes a like.
5. Feed: repeat same symbol+side collapses with badge, no re-sound; tier upgrade /
   first BANGER / first PROVEN re-floats and re-sounds; Focus and chart-watch
   alerts never collapse silently.
6. Digest: inside the window non-breakthrough alerts group per cycle; every
   breakthrough class lands immediately; window=0 disables; boundary behavior at
   window end.

## 9. Exit gate

Full Windows desk suite green (pytest's own exit code), smoke 7/7, no packaging
trigger hit (no new dependency/asset/package — expected), one live desk session in
which the trader confirms: a declined auto pick stayed gone all day while an armed
alert on the same symbol still fired; the Like button filed a pick correctly; the
open window produced digest rows with bangers breaking through. Do not tune C2/C3
thresholds from that single session.

## 10. Open trader-confirm items

1. Digest window default: 30 minutes from session open — confirm or change.
2. Like reason prompt: B3 ships zero-friction (no prompt). Add an optional,
   Enter-to-skip "why" prompt to feed the learning loop, or leave likes unprompted?
3. C2 escalation definition (tier upgrade / first BANGER / first PROVEN) — confirm
   this is the full list; anything else that should re-sound?

## Scheduling

Phase 0 (validate + merge the testing week) bars new feature work; this packet is
first in Phase 1 by trader direction (2026-08-14, "opus will code it out"). Build it
as the first post-P0 packet unless the trader explicitly directs an earlier start on
a branch that does not disturb the armed testing-week desk task.
