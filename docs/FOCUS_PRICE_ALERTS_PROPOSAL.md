# Focus price alerts + phone push — design proposal (for trader review)

Status: **PROPOSED 2026-08-03**, trader-directed (plan.md sec 12 item 7b);
trader decisions recorded 2026-08-03 (see "Trader decisions" below). Not yet
implemented. Nothing here changes a detector, a score, or a champion path —
this is a presentation + relay packet on top of machinery that already ships.

Trader requirements, as given:

- Pushes **originate from the main PC only**.
- Delivery channel is the **ntfy app** (phone; the Apple Watch mirrors it).
- **Basic price alerts only** — no setup logic, no detector coupling.
- The **Focus tab** gains a section: type in tickers, then up to **two levels
  each — one cross-up and one cross-down**.
- When one fires it goes to the phone **and** shows on the **main desk and
  every satellite**, unmissably.

## What already exists (do not re-implement)

This packet is deliberately small because four of the five pieces landed with
Evening mode and Desk Link:

| Piece | Where | State |
| --- | --- | --- |
| ntfy channel (topic/server/token, priorities, fail-quiet POST) | `scripts/push_notify.py` | Done |
| Alert store + cross evaluation + trigger log | `scripts/price_alerts.py` | Done |
| Background poller (60 s, extended hours, urgent-in-EVENING) | `scripts/ui/services/price_alert_service.py` | Done |
| Entry table + ntfy settings UI | `scripts/ui/panels/price_alerts_panel.py` (Research ▸ Price Alerts) | Done |
| Generic relay envelope for new surfaces | `desk_link` `TYPE_DESK_STREAM` + `publish_stream` | Done |

The store already has exactly the shape the trader asked for: one row per
symbol with an optional `above` and an optional `below` level, each with its
own armed flag, firing **once per arm** so a level chopping around a price
cannot spam the phone. No schema change is needed.

## Gaps this packet closes

1. **G1 — Entry is in the wrong place.** It lives in Research ▸ Price Alerts;
   the trader wants it on the Focus tab, where the names being watched are.
2. **G2 — A fired alert never leaves the main.** `triggered` is echoed into
   the local alert stream (`app.py::_on_price_alert`) and pushed to the phone;
   satellites learn nothing.
3. **G3 — The mirror satellite window ignores `desk_stream` entirely**
   (`ui/satellite.py::_on_message` handles popups, snapshots, and lease
   traffic only), so it needs a handler before it can show anything.
4. **G4 — "Main PC only" uses the wrong authority today.** A satellite desk
   builds the Research panel, so it constructs a `PriceAlertService` with a
   live 60 s timer. `shared_write_refusal()` currently stops the entire
   monitoring path before quote fetch or push, but it answers *which machine
   may write shared files*, not *which process is the Desk Link engine*. Keep
   that writer gate as defense in depth and add the explicit engine rule.
5. **G5 — Editing from a satellite would write shared state.** The Focus board
   must not let a satellite write `price_alerts.json` directly; that is what
   the Tier 2 intent channel is for.
6. **G6 — Nothing is unmissable.** A fired alert is one more red row in the
   alert stream.

## Design decisions

- **One store, two views.** The Focus section is a view onto the same
  `price_alerts.json` the Research page edits — never a second store. Research
  ▸ Price Alerts stays as the Evening-mode/advanced view (notes, re-arm all,
  ntfy configuration, test push).
- **One service per process.** Ownership of `PriceAlertService` moves up to
  `TradingDeskPanel` so the Focus board and the Research page share one poll
  timer instead of racing two.
- **One-shot per arm, confirmed.** "Once it fires it's done" — the level
  disarms itself and stays disarmed until the trader re-arms it. No overnight
  or next-session auto-re-arm.
- **Always urgent, confirmed.** Every price alert pushes at ntfy `urgent`
  priority, not just in EVENING mode, so it breaks through the phone's Focus
  and sleep modes whenever it fires.
- **A price alert is not an engine signal.** It is a last-price crossing, not
  a completed-bar state transition, and must never feed a detector, a score,
  or a state machine. It notifies; that is all.
- **Focus membership and alert rows stay independent.** Removing a Focus pick
  never deletes an alert row (plan.md sec 5: trader-entered names are never
  auto-removed). The symbol input may *offer* current Focus names for
  convenience.

## Phases

### Phase A — Focus tab entry section (main PC)

- New `scripts/ui/widgets/price_alert_board.py`, embedded in
  `FocusPicksPanel` beneath the Swing and M5 sections.
- Columns: Symbol · Cross up · Cross down · ▲ armed · ▼ armed · Last trigger.
  Buttons: Add, Remove selected, Re-arm.
- Reads and writes through the shared service's existing store passthrough
  (`entries()` / `save_entries()`), so validation and normalization stay in
  `price_alerts.py`.
- Tests: row round-trip through the store; a row with one side left blank arms
  only the filled side; removing a Focus pick leaves its alert row intact.

### Phase B — enforce main-PC origin

- `PriceAlertService` takes an explicit engine flag from `MainWindow`
  (`not satellite_desk`). A non-engine process never polls, never fetches, and
  never pushes — the panel's status line says so in words.
- The existing designated-writer gate stays as the second layer, for two
  machines that both believe they are the engine.
- Tests: a satellite-desk `MainWindow` builds the board but the service
  reports "not the engine machine" and no quote fetch is attempted.

### Phase C — relay the fire

- On trigger the main calls
  `desk_link_service.publish_stream("price_alert", payload)`; payload is the
  trigger dict plus the formatted message and the priority used. No protocol
  change — `desk_stream` is the generic envelope, and an older satellite skips
  an unknown stream rather than erroring.
- `DeskLinkFeedService` handles the new stream and re-emits it as
  `priceAlertReceived`, so a satellite desk renders it exactly like the main.
- `ui/satellite.py` gains a `TYPE_DESK_STREAM` branch (G3) so the mirror
  window shows the alert in its feed and beeps.
- Today's fired triggers ride along in the sticky state snapshot, so a
  satellite that reconnects after a Wi-Fi blip still sees what it missed —
  the same "never silently lose one" contract as the missed-popup replay.
- Tests: end-to-end over the real server/client pair (the
  `test_desk_link_feed.py` pattern) — fire on the main, assert both satellite
  kinds render it; assert an unknown stream is still skipped.

### Phase D — unmissable presentation

- A shared `PriceAlertToast`: persistent until dismissed, red accent, audible,
  stacking up to a small cap. Same gentle-raise contract as the alert popup
  (`WA_ShowWithoutActivating` + `WindowDoesNotAcceptFocus`) — an alert must
  never steal the keyboard mid-order-entry.
- Used identically by the main desk, the satellite desk, and the mirror
  window, alongside the existing alert-stream row.
- **Every** satellite beeps, whether or not it holds the control lease — a
  price alert is information, not a decision prompt, so there is no reason to
  silence the machines that are only watching.
- The phone push is the one delivery that must never be skipped: it fires
  before the relay fan-out, and a relay failure never suppresses it.
- Tests: the toast never activates its window; dismissal is clean; the cap
  holds under a burst.

### Phase E — satellite-initiated edits (only after A–D prove out live)

- New intent actions `price_alert_set` / `price_alert_rearm` over the Tier 2
  channel: journaled, acked, idempotent, applied on the main, which then
  re-publishes the snapshot.
- Until then the Focus board is **read-only on satellites**, with a hint
  pointing at the take-control button (G5).

## Invariants respected (plan.md sec 5)

- Decision-support only; no order execution, ever.
- Trader-entered rows are never auto-removed — the engine only flips armed
  flags and appends history.
- No detector or scoring behavior changes, so no golden fixtures are required
  for this packet.
- Shared-file writes stay on the designated writer machine.
- Champion paths (legacy SPY pause, D1 wick alerts) are untouched; price
  alerts never influence them.

## Verification

Per phase: new unit/Qt tests as listed, plus the standing gates —
`pytest tests/ -q` fully green and `scripts/smoke_check.py` 7/7.

Live check (one session, per plan.md sec 6 discipline): set a level a few
cents away on a liquid name, confirm in order — phone buzzes, main desk shows
the toast, connected satellite shows the toast — then confirm the level
disarms and does not re-fire on the next poll.

## Trader decisions (2026-08-03)

1. **Overnight re-arm — no.** Once a level fires it is done; it stays
   disarmed until re-armed by hand. This is already the engine's behavior, so
   Phase A only has to keep it true in the new surface.
2. **Priority — always urgent.** Drops the EVENING-vs-daytime split in
   `PriceAlertService._notify()`; every price alert goes out at `urgent`.
3. **Sound — every satellite beeps**, control lease or not. And the phone push
   is non-negotiable: it happens first and independently of the relay.
4. **Cap — not specified; assumed soft.** No hard limit and no auto-removal
   (sec 5). The board warns above ~25 armed symbols, which is roughly where
   one batched 1-minute quote poll starts to get slow. Say the word for a
   different number or a hard cap.

### Consequence of decision 2, worth knowing

The Focus board and Research ▸ Price Alerts share one store, and a row carries
no marker for which surface created it — so "always urgent" applies to *every*
price alert, including the Evening-mode position levels that previously pushed
at `high` during the day. That is the simplest reading of the decision and
probably the intent ("DEF do the phone").

If daytime urgents turn out to be too loud, the fallback is one added field:
tag rows created on the Focus board with an `origin`, and scope urgent to
those. `normalize_price_alert()` already drops unknown keys and defaults
missing ones, so adding the field later is backward compatible with every
store file written before it — no migration.
