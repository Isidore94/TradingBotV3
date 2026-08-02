# Multi-machine desk — design proposal (for trader review)

Status: **APPROVED 2026-08-02 — Tier 1 in progress** (plan.md sec 12, "Desk
Link"). Trader decisions on the open questions:

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

On the **main PC**:

1. Add `"desk_link_enabled": true` to `local_settings.json`
   (`%LOCALAPPDATA%\TradingBotV3` on Windows, `~/Library/Application
   Support/TradingBotV3` on macOS). Optional: `"desk_link_port"` (default
   47600).
2. Relaunch the desk. The first start generates `"desk_link_token"` in the
   same file — copy it for the satellite. Allow the port through the OS
   firewall for your private network only.

On the **satellite** (same repo, same setup, no TWS needed):

    .venv/bin/python scripts/gui.py --ui qt --satellite <main-hostname-or-ip> --link-token <token>

(`.venv\Scripts\python.exe` on Windows; the token is saved locally after
the first run, so later launches only need `--satellite <host>`.)

The satellite shows link status, a desk mirror line (watchlists/Focus), a
rolling alert feed, and pops the same chart popup the main renders —
gated by the alert-sound rule, not by the main's sound checkbox, so a
muted main still pops the satellite. Alert relay activates only while a
satellite is connected and never blocks the desk.

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
