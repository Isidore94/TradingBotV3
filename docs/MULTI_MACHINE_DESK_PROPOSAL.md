# Multi-machine desk — design proposal (for trader review)

Status: **PROPOSAL — not scheduled.** Nothing here is implemented. If approved,
this enters plan.md Section 12 as an ordered milestone; until then it binds
nothing. It deliberately reuses the invariants and single-writer machinery the
repo already has.

## Goal

Run the full desk on the main PC ("Engine-Main") and let a second machine
(macBook, second PC — a "satellite") sign in over the local network to:

- receive **alert chart popups live** (critical-path requirement),
- see all data/outputs the main sees,
- take actions (remove-for-day, add/remove Focus picks, Alert Center
  approve/pass/park, arm watches) that apply back on the main.

Non-goals: order execution (never — plan.md sec 5), internet-exposed remote
access, more than one satellite in control at a time, satellites running
their own scanners or TWS sessions.

## Roles: engine vs. control

Two concepts, deliberately separate:

- **Engine** — always the main PC. It alone talks to TWS, runs the scanners
  and detectors, fires alerts, and writes every shared file. This never moves.
  (Also the only IBKR consumer, so the single-login TWS constraint and the
  IB pacing budget stay solved by construction.)
- **Control** — who is allowed to *decide*: act on the review queue, edit
  watchlists/Focus, arm/disarm watches. Exactly one machine holds control at
  any moment. Default holder is the main.

Trader-chosen semantics: **when a satellite signs in it takes full control;
the main becomes a relay** (its decision surfaces lock, engines keep
running); when the satellite signs out the main resumes primary.

## Control lease (the one amendment to "sign in / sign out")

An explicit sign-out cannot be the only way control returns: a sleeping
laptop, dead Wi-Fi, or a crashed satellite must not leave the desk headless.
Control is therefore a **lease with a heartbeat**:

- Satellite signs in → main grants the control lease and locks its own
  decision UI (banner: "Controlled by <machine> — desk is relaying").
- Satellite renews the lease every ~5 s over the socket.
- Missed renewals for a grace window (30–60 s, configurable) → main
  **reclaims control automatically**, unlocks its UI, and posts a visible
  notice + phone push ("Satellite lease expired — main resumed control").
- Sign-out is the graceful path: immediate reclaim, no grace window.
- One lease exists; a second satellite connecting is view-only until the
  first releases.

The lease governs *decision rights only*. Engines, alert evaluation, file
publishing, and completed-bar state transitions are untouched by who holds
control, so every plan.md sec 5 invariant survives handoffs.

## Transport: a small server on the main

- A LAN-only WebSocket server inside the main's GUI process (or a sibling
  thread owned by one component, per the one-owner-per-thread invariant).
- Pairing: first connection requires a short pairing code shown on the main;
  thereafter a stored per-machine token. Bind to the LAN interface only —
  never the public internet; no inbound cloud hop.
- The Drive shared store stays the durable record. The socket is a live
  mirror + command channel, not a second source of truth: a satellite that
  reconnects resyncs from the main's state snapshot, and everything the main
  applies lands in the same shared files as today.

## Message types (sketch)

| Direction | Message | Contents |
|---|---|---|
| main → sat | `alert_popup` | Self-contained chart payload: bars (M5/D1), AVWAP bands + σ levels, armed levels, annotations, review-policy context — everything `alert_chart_review` needs to render without TWS |
| main → sat | `state_snapshot` / `state_delta` | Watchlists, Focus, review queue, tracker/report surfaces |
| main → sat | `lease_grant` / `lease_revoke` | Control state changes |
| sat → main | `intent` | One decision: {action, symbol, surface, client_seq, ts+tz} |
| sat → main | `heartbeat` | Lease renewal |

Intents are also appended to the satellite's own per-machine JSONL (same
partitioned pattern as `alert_review_events/`) before sending, so a socket
drop never loses a decision — the main's applier dedupes by (machine,
client_seq) when the file syncs via Drive.

## Alert chart popups on the satellite

The popup is the critical path, so it is payload-driven: the main serializes
the exact inputs its own popup uses and the satellite renders them with the
**same widget code** (`scripts/ui/widgets/alert_chart_review.py` /
`candle_chart.py`). One rendering path, two feeds — a change to the popup on
Windows is automatically the change on the mac. Bar payloads on a LAN are
tens of KB; latency will be dominated by the detector, not the wire.

Live-updating charts on the satellite (forming-bar preview after the popup
opens) are a later tier: same channel, incremental bar frames for symbols the
satellite has open. Forming bars remain preview-only everywhere.

## Failure modes

| Failure | Behavior |
|---|---|
| Satellite crash / sleep / Wi-Fi drop | Lease expires → main reclaims control, notifies. Unsent intents replay from the satellite's JSONL on reconnect (deduped). |
| Main crash | Desk is down — same as today. Satellite shows "engine offline"; nothing pretends to be live (missing data is uncertainty, never confirmation). |
| Drive down, LAN up | Live flow unaffected; publishes queue exactly as they do today (failed publish never destroys the last verified report). |
| LAN down, Drive up | Degrades to today's behavior: satellite sees synced files; intents land via the JSONL path with sync latency. |

## Rollout tiers (each independently shippable)

1. **Relay + popups (view-only).** Server, pairing, state snapshot, live
   alert popups on the satellite. No control changes; main stays primary.
   Proves the critical path first.
2. **Control lease + intents.** Sign-in takes control, main locks to relay,
   heartbeat/reclaim, intent applier on the main.
3. **Live chart streaming.** Incremental bars for open satellite charts.

## Open questions for the trader

1. Grace window before the main auto-reclaims control: 30 s? 60 s?
2. While a satellite holds control, is the main's UI fully locked, or should
   an emergency "take back control now" button on the main override?
3. Should satellite sign-in require the main to confirm (a click on the main
   PC) the first time each day, or is the stored pairing token enough?
4. Tier 1 alone useful enough to ship first? (Recommended: yes.)
