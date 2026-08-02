# Multi-machine desk — design proposal (for trader review)

Status: **APPROVED 2026-08-02 — Tiers 1 and 2 implemented** (plan.md sec 12
item 7a, "Desk Link"); live two-machine validation pending; Tier 3 (live
chart streaming) not started. Trader decisions on the open questions:

- The main can **take back control at any time** — an immediate override,
  no grace period, always available while a satellite holds the lease.
- **No first-connection confirmation** on the main (single-user system): a
  satellite that presents the stored link token connects directly.
- **Tier 1 ships first** (relay + live alert popups, view-only satellite).
  Tiers 2-3 wait for Tier 1 to prove out live.

The design deliberately reuses the invariants and single-writer machinery the
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

- A LAN-only TCP socket server (newline-delimited JSON messages — our own
  client on both ends, so no WebSocket/browser framing is needed and the
  stack stays stdlib-only) inside the main's GUI process, its accept/writer
  threads owned by one Desk Link component per the one-owner-per-thread
  invariant. A slow or stuck satellite gets dropped, never blocks the desk.
- Pairing: the main generates a link token once and stores it in its local
  settings; the satellite is configured with main's host + token one time.
  No per-connection confirmation (trader decision — single-user system).
  Bind to the LAN interface only — never the public internet; no inbound
  cloud hop.
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

## Running Tier 1 (implemented)

On the **main PC**: Settings page → **Desk Link** section. Turn on "Serve
satellites from this machine" (applies immediately, no relaunch), adjust
the port if needed (default 47600), and hit **Copy token** for the
satellite's first launch. **Regenerate token** revokes the old one and
disconnects satellites. The section shows live serving status and which
satellites are connected. Allow the port through the OS firewall for your
private network only. (Everything persists in the machine-local settings,
so it survives relaunches; nothing Desk Link touches the shared store.)

On the **satellite** (same repo, same setup, no TWS needed): double-click
`TradingBotV3_Satellite.command` (macOS) or `TradingBotV3_Satellite.cmd`
(Windows). The first launch opens a **connect dialog** — type the main
PC's name/IP, keep the port, paste the token — and everything is
remembered, so every later launch just connects. The
"Connect / change main desk…" button reopens the dialog any time (new
IP, regenerated token). CLI equivalents still work:
`scripts/gui.py --ui qt --satellite [HOST[:PORT]] [--link-token T]`.

The satellite shows link status, a desk mirror line (watchlists/Focus), a
rolling alert feed, and pops the same chart popup the main renders —
gated by the alert-sound rule, not by the main's sound checkbox, so a
muted main still pops the satellite. Alert relay activates only while a
satellite is connected and never blocks the desk.

Reliability details:

- **Missed popups replay.** The main buffers recent popups (50, ≤15 min
  old); a reconnecting satellite presents the last one it saw and gets
  everything newer, marked "⟲ missed" (one beep per replay burst). A
  Wi-Fi blip cannot silently swallow an alert. A fresh satellite session
  starts from now — it is not sprayed with history.
- **Instant mirror.** A satellite's own applied action republishes the
  desk snapshot immediately instead of waiting out the 60 s timer.
- **Phone push on auto-reclaim.** If the controlling satellite dies and
  the main reclaims control on its own, a push goes out through the
  existing ntfy channel (when configured) so the trader knows their
  actions stopped applying. A deliberate take-back does not page.

## Using control (Tier 2, implemented)

- Satellite: **Take control** (top of the window). On grant, the popups'
  action buttons go live — *Remove for day*, *Focus long/short*,
  *Unfocus* — and every action applies on the main through the exact code
  path a local click takes (same review events, same focus feedback with
  ``origin=desk_link``). **Release control** hands it back.
- Main, while a satellite holds control: a banner appears
  ("CONTROLLED BY <machine> — this desk is relaying"), the page stack and
  status bar lock, engines/scans/TWS keep running, and **Take back
  control** on the banner reclaims immediately — always available.
- The lease dies with the connection: pings renew it implicitly and the
  server's 30 s idle timeout is the grace window, so a sleeping laptop or
  dead Wi-Fi returns control to the main automatically.
- Delivery: every satellite decision is journaled locally
  (``desk_link_intent_journal.jsonl``) before it is sent, acked by the
  main, and resent on the next grant if unacked. The three actions are
  idempotent, so at-least-once delivery is safe. A non-controller's
  intent is refused, never applied.

## Rollout tiers (each independently shippable)

1. **Relay + popups (view-only).** Server, pairing, state snapshot, live
   alert popups on the satellite. No control changes; main stays primary.
   Proves the critical path first.
2. **Control lease + intents.** Sign-in takes control, main locks to relay,
   heartbeat/reclaim, intent applier on the main.
3. **Live chart streaming.** Incremental bars for open satellite charts.

## Resolved questions (trader, 2026-08-02)

1. Grace window before auto-reclaim: default **45 s** (configurable) — and
   moot in most cases because of (2).
2. The main keeps a **"Take back control" button at all times** while a
   satellite holds the lease; it reclaims immediately.
3. **No first-connection confirmation** — stored link token is enough.
4. **Tier 1 ships first.**
