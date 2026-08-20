# Auto EVENING mode - sleep-in runbook

Document role: **active single-main operator runbook**. Last reconciled 2026-08-15
(packet R1: EVENING stops scanning after its early block, and gains the SPY
wake alarm). Current live-validation location: `plan.md` P0.3.

Purpose: get home at 23:30, sleep past the 06:30 open, and still wake to a
phone that guards your positions and a desk that already did the morning's
work.

## What EVENING does (vs DESK / AWAY)

- **Same discovery, zero recommendations.** Trading logic is identical to
  DESK: auto-populate picks stage silently in the Alert Center for chart
  approval. Nothing self-applies and nothing pings the desk while the mode is
  on. Flip the mode off (to DESK or OFF) when you sit down — that flip is also
  what adopts the staged picks into M5 Focus.
- **Early Master AVWAP run.** The swing scan gets an extra slot at open+30
  (07:00 on a normal session) instead of waiting for the usual open+60, so
  the best-D1 ranking is current when you arrive.
- **Then it stops (new 2026-08-14, packet R1).** After the early slot, the
  strength checks and the briefing, EVENING runs no ordinary hourly swing slot
  and no open watchlist self-build for the rest of the day. The Auto Pilot log
  names each refused slot once — `Evening mode: swing slot(s) 09:00 not run` —
  so a quiet log is the mode working, not a broken scheduler. Flip to DESK to
  resume the hourly schedule.
- **SPY ±1% wake alarm (new 2026-08-14, packet R1).** If SPY is a full percent
  from yesterday's close, the phone gets an urgent push and repeats every five
  minutes until you flip out of EVENING. It reads the champion cached SPY bars;
  missing bars mean silence, never a false alarm. Machine-local kill switch:
  `push_evening_spy_alarm` (default on); threshold override:
  `push_evening_spy_alarm_pct`. This is the second deliberate exception to the
  AWAY-only push rule.
- **Strength persistence checks at 07:00 / 07:15 / 07:30.** Every staged
  intraday pick is snapshotted against its HOD/LOD. A name still pressing its
  extreme at 07:30 is HELD; one that slipped is FADED and explicitly not
  recommended.
- **Morning briefing.** After the 07:30 check the briefing finalizes: market
  environment, best D1 swing setups per side (by expected R), held vs faded
  intraday picks, and any overnight price alerts. It is written to
  `evening_briefing.txt` next to `autopilot_today.txt` in the shared home folder,
  folded into the hourly phone report (EVENING publishes hourly from 07:00
  like AWAY), and echoed in the Alert Center. It is **no longer pushed to the
  phone**: since 2026-08-11 AWAY is the only mode that pushes, and EVENING ends
  with the trader walking to this screen. The file and the desk announcement are
  unchanged.
- **Price alerts push at urgent priority** (see below) so a level cross can
  wake you. They are the always-on phone channel: they fire in every Auto mode,
  including OFF and EVENING. Together with the SPY wake alarm above, they are
  the only two exceptions to the AWAY-only push rule.

## Night-before checklist

1. Leave the GUI running and IB/TWS logged in (same as an Away day).
2. Click the Auto Mode button in the status bar until it reads
   `Auto: EVENING` (cycle: OFF -> DESK -> AWAY -> EVENING -> OFF).
3. Focus -> Phone Price Alerts (or Research -> Price Alerts for the advanced
   view): enter each position, and SPY if you want an index tripwire, with an
   *Alert Above* and/or *Alert Below* level armed.
4. Hit **Test Push** and confirm the phone (and watch) buzz.
5. If you are relying on being woken: run the **Sleep breakthrough checklist**
   below at least once, and re-run it after any iOS update.

## Sleep breakthrough checklist (2026-08-20)

**Urgent priority alone does not override iOS Sleep Focus.** ntfy has no Apple
critical-alert entitlement - that is a per-app entitlement Apple grants, and
without it no notification of any priority can break a Focus mode on its own.
The ntfy priority header only decides how the *app* treats the message once
iOS has already let it through. So the desk's side of this is already at
maximum and cannot be improved by changing any code:

| Sender | Priority | Verified |
|---|---|---|
| Focus/Research price alerts (`price_alert_service._notify`) | `urgent` | code, 2026-08-20 |
| EVENING SPY +/-1% wake alarm (`AutopilotService._maybe_push_spy_alarm`) | `urgent` | code, 2026-08-20 |

Everything that decides whether you actually wake up is on the phone:

1. **iOS Settings > Focus > Sleep > Allowed Notifications > Apps** - add
   **ntfy**. Without this, Sleep Focus silences it no matter what the desk
   sends. *(to be confirmed on the desk/phone)*
2. **iOS Settings > Notifications > ntfy** - Alerts on, **Sounds on**, and the
   delivery style must NOT be *Deliver Quietly* (check the notification's own
   `...` menu too; one swipe sets Deliver Quietly permanently and it is easy to
   do by accident). *(to be confirmed on the desk/phone)*
3. **In the ntfy app, per subscribed topic** - open the topic, and make sure
   its notification setting is not muted and not set to a silent sound.
   Some ntfy versions expose "Time Sensitive"/priority-to-channel mapping
   here; enable the loudest option available. *(to be confirmed on the
   desk/phone)*
4. **Verify it, do not assume it.** Turn Sleep Focus ON on the phone, put it
   down, then on the desk open Focus -> Phone Price Alerts (or Research ->
   Price Alerts) and click **Test wake alert (urgent)**. That button sends one
   push at exactly the priority the two senders above use - the ordinary
   **Test Push** goes out at `high`, which proves nothing about the overnight
   path. The message on the phone says what should have happened, so a silent
   phone and a sounding phone are told apart without walking back to the desk.
5. If it did not sound: work back up this list (1 -> 3), then re-run step 4.
   If it still will not break through, iOS is refusing it and the remaining
   options are outside this app - the phone's own alarm clock, or an ntfy
   build carrying an Apple critical-alert entitlement.

The wake test is a **test**, not a sender: nothing schedules it, only that
button calls it, and it does not change the phone push policy (AWAY is still
the only Auto mode that pushes routine output; the price alerts and the SPY
wake alarm remain the two deliberate exceptions).

## Price alerts - one-time phone setup

The push channel is [ntfy](https://ntfy.sh): outbound HTTPS only, so no
router ports need opening and it works from any network. Self-hosting later
is just pointing the *ntfy server* field at your own instance.

1. Install the **ntfy** app on the iPhone.
2. Pick a long, randomly generated topic name (it is effectively a password
   unless access control is enabled), e.g. `tbv3-k7q9x2m8-v4n6p1cz`, and
   subscribe to it in the app. Do not put a name, account number, or other
   personal information in the topic.
3. Enter the same topic in Research -> Price Alerts. Server stays
   `https://ntfy.sh` unless self-hosting; token only if the topic is
   protected.
4. In iPhone Settings, allow ntfy notifications and sounds. Allow ntfy through
   the Focus modes you rely on, or enable Time Sensitive/Critical delivery when
   that option is available in the installed ntfy/iOS version. Do not assume
   the priority header alone overrides an iPhone Focus configuration - it does
   not, and the Sleep breakthrough checklist above is how you find out before
   the night you need it.
5. The Apple Watch mirrors iPhone notifications automatically - no extra
   setup.

Alert semantics: each side (above/below) fires **once per arm**, then
disarms so a chopping price can't spam you; re-arm from the panel (or
**Re-arm All** for the next night). Levels are checked about once a minute,
pre/post market included (01:00-17:00 local). Every triggered crossing pushes
at urgent priority, lands in the Alert Center, and is listed in the morning
briefing. Monitoring runs in every mode by default - untick "Monitor in every
mode" to restrict it to EVENING. Only the main-desk alert engine monitors; the
designated shared-store writer check remains a second guard against duplicate
processes. Satellite and mini-PC roles are retired.

Protected-topic option: create/reserve the topic and an access token in ntfy,
enter that token in the TradingBot token field, and authenticate the phone's
subscription to the same protected topic. TradingBot sends the token as a
Bearer credential over HTTPS. Self-hosting is optional and requires the phone
to subscribe to that server too; simply changing the desktop URL is not enough.

## Morning arrival

- Slept to 07:30+? The briefing is final - read it in `evening_briefing.txt`,
  the phone report, or the Alert Center line, approve/pass the staged picks
  off their charts, then click the mode off EVENING.
- **Flipping to DESK is what restarts the day.** It resumes the ordinary hourly
  swing slots, adopts the staged picks into M5 Focus, and stops the SPY wake
  alarm. Until you flip, EVENING stays in its post-briefing quiet.
- At the desk before 07:00? Nothing is lost: the checks and early scan run on
  schedule anyway; switch to DESK whenever you want live flow back.
